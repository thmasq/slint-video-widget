use anyhow::{Context, Result, anyhow};
use crossbeam_channel::{Receiver, Sender, bounded};
use ffmpeg_sys_next::SwsFlags::SWS_BILINEAR;
use ffmpeg_sys_next::{
    AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX, AV_ERROR_MAX_STRING_SIZE, AV_NOPTS_VALUE, AVBufferRef,
    AVCodec, AVCodecContext, AVCodecParameters, AVERROR_EOF, AVFormatContext, AVFrame,
    AVHWDeviceType, AVMediaType, AVPacket, AVPixelFormat, AVSEEK_FLAG_BACKWARD, av_buffer_ref,
    av_buffer_unref, av_frame_alloc, av_frame_copy_props, av_frame_free, av_frame_unref,
    av_hwdevice_ctx_create, av_hwframe_transfer_data, av_packet_alloc, av_packet_free,
    av_packet_unref, av_read_frame, av_seek_frame, av_strerror, avcodec_alloc_context3,
    avcodec_find_decoder, avcodec_flush_buffers, avcodec_free_context, avcodec_get_hw_config,
    avcodec_open2, avcodec_parameters_to_context, avcodec_receive_frame, avcodec_send_packet,
    avformat_close_input, avformat_find_stream_info, avformat_open_input, sws_freeContext,
    sws_getContext, sws_scale,
};
use slint::{Image, Rgba8Pixel, SharedPixelBuffer};
use std::env;
use std::ffi::{CStr, CString};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

slint::include_modules!();

const AVERROR_EAGAIN: i32 = -11;

/// Player control commands
enum PlayerCommand {
    Play,
    Pause,
    Seek(u64),      // Seek to frame number
    SetVolume(i32), // 0 to 100
    Stop,
}

/// Player state shared between threads
struct PlayerState {
    is_playing: AtomicBool,
    is_muted: AtomicBool,
    volume: AtomicU64,                    // 0-100
    seek_target: Arc<Mutex<Option<u64>>>, // Target frame number
}

impl PlayerState {
    fn new() -> Self {
        PlayerState {
            is_playing: AtomicBool::new(false),
            is_muted: AtomicBool::new(false),
            volume: AtomicU64::new(100),
            seek_target: Arc::new(Mutex::new(None)),
        }
    }
}

/// Represents a decoded video frame ready for display
struct DecodedFrame {
    /// RGB pixel data
    data: Vec<u8>,
    /// Frame width
    width: u32,
    /// Frame height
    height: u32,
    /// Frame number
    frame_number: u64,
}

/// Video metadata for time conversions
#[derive(Clone)]
struct VideoMetadata {
    fps_num: i32,
    fps_den: i32,
    total_frames: u64,
    duration_ms: u64,
}

impl VideoMetadata {
    /// Convert frame number to milliseconds
    fn frame_to_ms(&self, frame: u64) -> u64 {
        if self.fps_num == 0 {
            return 0;
        }
        // ms = frame * 1000 * fps_den / fps_num
        frame
            .saturating_mul(1000)
            .saturating_mul(self.fps_den as u64)
            / (self.fps_num as u64)
    }

    /// Convert milliseconds to frame number
    fn ms_to_frame(&self, ms: u64) -> u64 {
        if self.fps_den == 0 {
            return 0;
        }
        // frame = ms * fps_num / (1000 * fps_den)
        ms.saturating_mul(self.fps_num as u64) / (1000u64.saturating_mul(self.fps_den as u64))
    }

    /// Get frames per second as integer (rounded)
    fn get_fps_rounded(&self) -> u64 {
        if self.fps_den == 0 {
            return 25;
        }
        ((self.fps_num as u64 * 1000) / (self.fps_den as u64) + 500) / 1000
    }
}

/// RAII wrapper for `FFmpeg` format context
struct FormatContext {
    ctx: *mut AVFormatContext,
}

unsafe impl Send for FormatContext {}

impl FormatContext {
    fn open(path: &str) -> Result<Self> {
        unsafe {
            let mut ctx: *mut AVFormatContext = ptr::null_mut();
            let c_path = CString::new(path)?;

            let ret = avformat_open_input(
                &raw mut ctx,
                c_path.as_ptr(),
                ptr::null_mut(),
                ptr::null_mut(),
            );
            if ret < 0 {
                return Err(anyhow!("Failed to open input file: {}", av_err2str(ret)));
            }

            let ret = avformat_find_stream_info(ctx, ptr::null_mut());
            if ret < 0 {
                avformat_close_input(&raw mut ctx);
                return Err(anyhow!("Failed to find stream info: {}", av_err2str(ret)));
            }

            Ok(FormatContext { ctx })
        }
    }

    fn as_ptr(&self) -> *mut AVFormatContext {
        self.ctx
    }
}

impl Drop for FormatContext {
    fn drop(&mut self) {
        unsafe {
            avformat_close_input(&raw mut self.ctx);
        }
    }
}

/// RAII wrapper for `FFmpeg` codec context
struct CodecContext {
    ctx: *mut AVCodecContext,
}

unsafe impl Send for CodecContext {}

impl CodecContext {
    fn as_ptr(&self) -> *mut AVCodecContext {
        self.ctx
    }
}

impl Drop for CodecContext {
    fn drop(&mut self) {
        unsafe {
            avcodec_free_context(&raw mut self.ctx);
        }
    }
}

/// RAII wrapper for `AVFrame`
struct Frame {
    frame: *mut AVFrame,
}

impl Frame {
    fn new() -> Result<Self> {
        unsafe {
            let frame = av_frame_alloc();
            if frame.is_null() {
                return Err(anyhow!("Failed to allocate AVFrame"));
            }
            Ok(Frame { frame })
        }
    }

    fn as_ptr(&self) -> *mut AVFrame {
        self.frame
    }
}

impl Drop for Frame {
    fn drop(&mut self) {
        unsafe {
            av_frame_free(&raw mut self.frame);
        }
    }
}

/// RAII wrapper for `AVPacket`
struct Packet {
    packet: *mut AVPacket,
}

impl Packet {
    fn new() -> Result<Self> {
        unsafe {
            let packet = av_packet_alloc();
            if packet.is_null() {
                return Err(anyhow!("Failed to allocate AVPacket"));
            }
            Ok(Packet { packet })
        }
    }

    fn as_ptr(&self) -> *mut AVPacket {
        self.packet
    }

    fn unref(&self) {
        unsafe {
            av_packet_unref(self.packet);
        }
    }
}

impl Drop for Packet {
    fn drop(&mut self) {
        unsafe {
            av_packet_free(&raw mut self.packet);
        }
    }
}

/// RAII wrapper for `SwsContext` (software scaler)
struct SwsContext {
    ctx: *mut ffmpeg_sys_next::SwsContext,
}

unsafe impl Send for SwsContext {}

impl SwsContext {
    fn as_ptr(&self) -> *mut ffmpeg_sys_next::SwsContext {
        self.ctx
    }
}

impl Drop for SwsContext {
    fn drop(&mut self) {
        unsafe {
            sws_freeContext(self.ctx);
        }
    }
}

/// Hardware device context wrapper
struct HwDeviceContext {
    ctx: *mut AVBufferRef,
}

unsafe impl Send for HwDeviceContext {}

impl Drop for HwDeviceContext {
    fn drop(&mut self) {
        unsafe {
            av_buffer_unref(&raw mut self.ctx);
        }
    }
}

/// Video decoder that handles both hardware and software decoding
#[allow(dead_code)]
struct VideoDecoder {
    format_ctx: FormatContext,
    codec_ctx: CodecContext,
    sws_ctx: Option<SwsContext>,
    hw_device_ctx: Option<HwDeviceContext>,
    video_stream_idx: usize,
    time_base: f64,
    metadata: VideoMetadata,
    using_hw: bool,
    hw_pix_fmt: AVPixelFormat,
    current_frame: u64,
}

impl VideoDecoder {
    /// Create a new video decoder, attempting hardware acceleration
    fn new(path: &str) -> Result<Self> {
        unsafe {
            let format_ctx = FormatContext::open(path)?;
            let fmt_ctx_ref = &*format_ctx.as_ptr();

            // Find video stream
            let video_stream_idx = (0..fmt_ctx_ref.nb_streams)
                .find(|&i| {
                    let stream = *fmt_ctx_ref.streams.add(i as usize);
                    (*(*stream).codecpar).codec_type == AVMediaType::AVMEDIA_TYPE_VIDEO
                })
                .ok_or_else(|| anyhow!("No video stream found"))?
                as usize;

            let stream = *fmt_ctx_ref.streams.add(video_stream_idx);
            let codecpar = (*stream).codecpar;

            // Calculate time base for PTS conversion
            let time_base = f64::from((*stream).time_base.num) / f64::from((*stream).time_base.den);

            // Get video metadata
            let avg_frame_rate = (*stream).avg_frame_rate;
            let fps_num = avg_frame_rate.num;
            let fps_den = if avg_frame_rate.den > 0 {
                avg_frame_rate.den
            } else {
                1
            };

            // Calculate duration and total frames
            let duration_seconds = if (*stream).duration != AV_NOPTS_VALUE {
                (*stream).duration as f64 * time_base
            } else if fmt_ctx_ref.duration != AV_NOPTS_VALUE {
                fmt_ctx_ref.duration as f64 / 1_000_000.0
            } else {
                0.0
            };

            let duration_ms = (duration_seconds * 1000.0) as u64;
            let fps = if fps_den > 0 {
                fps_num as f64 / fps_den as f64
            } else {
                25.0
            };
            let total_frames = (duration_seconds * fps) as u64;

            let metadata = VideoMetadata {
                fps_num,
                fps_den,
                total_frames,
                duration_ms,
            };

            println!("Video FPS: {}/{} ({:.2} fps)", fps_num, fps_den, fps);
            println!(
                "Video Duration: {} ms ({:.2} seconds)",
                duration_ms, duration_seconds
            );
            println!("Total Frames: {}", total_frames);

            // Find decoder
            let codec = avcodec_find_decoder((*codecpar).codec_id);
            if codec.is_null() {
                return Err(anyhow!("Codec not found"));
            }

            // Attempt hardware decoding setup
            let (codec_ctx, hw_device_ctx, using_hw, hw_pix_fmt) =
                Self::try_hardware_decoder(codec, codecpar)?;

            Ok(VideoDecoder {
                format_ctx,
                codec_ctx,
                sws_ctx: None,
                hw_device_ctx,
                video_stream_idx,
                time_base,
                metadata,
                using_hw,
                hw_pix_fmt,
                current_frame: 0,
            })
        }
    }

    /// Attempt to create a hardware-accelerated decoder
    unsafe fn try_hardware_decoder(
        codec: *const AVCodec,
        codecpar: *mut AVCodecParameters,
    ) -> Result<(CodecContext, Option<HwDeviceContext>, bool, AVPixelFormat)> {
        unsafe {
            // Try to find hardware config
            let mut hw_type = AVHWDeviceType::AV_HWDEVICE_TYPE_NONE;
            let mut hw_pix_fmt = AVPixelFormat::AV_PIX_FMT_NONE;
            let mut i = 0;

            // Prefer specific hardware types based on platform
            #[cfg(target_os = "linux")]
            let preferred_types = vec![AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI];

            #[cfg(target_os = "windows")]
            let preferred_types = vec![
                AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA,
                AVHWDeviceType::AV_HWDEVICE_TYPE_DXVA2,
                AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA,
            ];

            #[cfg(not(any(target_os = "linux", target_os = "windows")))]
            let preferred_types: Vec<AVHWDeviceType> = vec![];

            loop {
                let config = avcodec_get_hw_config(codec, i);
                if config.is_null() {
                    break;
                }

                if ((*config).methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX as i32) != 0 {
                    let device_type = (*config).device_type;
                    if preferred_types.contains(&device_type) {
                        hw_type = device_type;
                        hw_pix_fmt = (*config).pix_fmt;
                        println!(
                            "Found hardware config: {:?} with pixel format: {}",
                            hw_type, hw_pix_fmt as i32
                        );
                        break;
                    }
                }
                i += 1;
            }

            // Try to create hardware device context
            if hw_type != AVHWDeviceType::AV_HWDEVICE_TYPE_NONE {
                let mut hw_device_ctx: *mut AVBufferRef = ptr::null_mut();
                let ret = av_hwdevice_ctx_create(
                    &raw mut hw_device_ctx,
                    hw_type,
                    ptr::null(),
                    ptr::null_mut(),
                    0,
                );

                if ret >= 0 {
                    let mut codec_ctx = avcodec_alloc_context3(codec);
                    if codec_ctx.is_null() {
                        av_buffer_unref(&raw mut hw_device_ctx);
                        return Err(anyhow!("Failed to allocate codec context"));
                    }

                    let ret = avcodec_parameters_to_context(codec_ctx, codecpar);
                    if ret < 0 {
                        avcodec_free_context(&raw mut codec_ctx);
                        av_buffer_unref(&raw mut hw_device_ctx);
                        return Err(anyhow!("Failed to copy codec parameters"));
                    }

                    (*codec_ctx).hw_device_ctx = av_buffer_ref(hw_device_ctx);

                    if hw_type == AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI {
                        extern "C" fn get_hw_format(
                            _ctx: *mut AVCodecContext,
                            pix_fmts: *const AVPixelFormat,
                        ) -> AVPixelFormat {
                            let mut p = pix_fmts;
                            unsafe {
                                while *p != AVPixelFormat::AV_PIX_FMT_NONE {
                                    if *p == AVPixelFormat::AV_PIX_FMT_VAAPI {
                                        return *p;
                                    }
                                    p = p.add(1);
                                }
                            }
                            println!("Failed to find VAAPI pixel format!");
                            AVPixelFormat::AV_PIX_FMT_NONE
                        }

                        (*codec_ctx).get_format = Some(get_hw_format);
                    }

                    let ret = avcodec_open2(codec_ctx, codec, ptr::null_mut());
                    if ret >= 0 {
                        println!(
                            "Hardware decoding enabled ({})",
                            match hw_type {
                                AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI => "VAAPI",
                                AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA => "CUDA",
                                AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA => "D3D11VA",
                                AVHWDeviceType::AV_HWDEVICE_TYPE_DXVA2 => "DXVA2",
                                _ => "Unknown",
                            }
                        );
                        return Ok((
                            CodecContext { ctx: codec_ctx },
                            Some(HwDeviceContext { ctx: hw_device_ctx }),
                            true,
                            hw_pix_fmt,
                        ));
                    }

                    avcodec_free_context(&raw mut codec_ctx);
                    av_buffer_unref(&raw mut hw_device_ctx);
                    println!("Hardware decoder failed to open, falling back to software");
                }
            }

            // Fall back to software decoding
            println!("Using software decoding");
            let mut codec_ctx = avcodec_alloc_context3(codec);
            if codec_ctx.is_null() {
                return Err(anyhow!("Failed to allocate codec context"));
            }

            let ret = avcodec_parameters_to_context(codec_ctx, codecpar);
            if ret < 0 {
                avcodec_free_context(&raw mut codec_ctx);
                return Err(anyhow!("Failed to copy codec parameters"));
            }

            let ret = avcodec_open2(codec_ctx, codec, ptr::null_mut());
            if ret < 0 {
                avcodec_free_context(&raw mut codec_ctx);
                return Err(anyhow!("Failed to open codec: {}", av_err2str(ret)));
            }

            Ok((
                CodecContext { ctx: codec_ctx },
                None,
                false,
                AVPixelFormat::AV_PIX_FMT_NONE,
            ))
        }
    }

    /// Seek to a specific frame
    fn seek(&mut self, target_frame: u64) -> Result<()> {
        unsafe {
            // Convert frame number to timestamp
            let target_seconds = self.metadata.frame_to_ms(target_frame) as f64 / 1000.0;
            let seek_target = (target_seconds / self.time_base) as i64;

            let ret = av_seek_frame(
                self.format_ctx.as_ptr(),
                self.video_stream_idx as i32,
                seek_target,
                AVSEEK_FLAG_BACKWARD,
            );

            if ret < 0 {
                return Err(anyhow!("Failed to seek: {}", av_err2str(ret)));
            }

            avcodec_flush_buffers(self.codec_ctx.as_ptr());
            self.current_frame = target_frame;

            Ok(())
        }
    }

    /// Decode the video and send frames to the UI thread
    fn decode_video(
        mut self,
        sender: &Sender<DecodedFrame>,
        command_receiver: &Receiver<PlayerCommand>,
        state: Arc<PlayerState>,
    ) -> Result<()> {
        unsafe {
            let packet = Packet::new()?;
            let frame = Frame::new()?;
            let hw_frame = if self.using_hw {
                Some(Frame::new()?)
            } else {
                None
            };

            let fmt_ctx = self.format_ctx.as_ptr();
            let mut is_paused = false;

            // Main decoding loop
            loop {
                // Check for commands
                while let Ok(cmd) = command_receiver.try_recv() {
                    match cmd {
                        PlayerCommand::Play => {
                            is_paused = false;
                            state.is_playing.store(true, Ordering::Relaxed);
                        }
                        PlayerCommand::Pause => {
                            is_paused = true;
                            state.is_playing.store(false, Ordering::Relaxed);
                        }
                        PlayerCommand::Seek(frame_num) => {
                            if let Err(e) = self.seek(frame_num) {
                                eprintln!("Seek failed: {}", e);
                            }
                        }
                        PlayerCommand::SetVolume(vol) => {
                            state.volume.store(vol as u64, Ordering::Relaxed);
                        }
                        PlayerCommand::Stop => {
                            return Ok(());
                        }
                    }
                }

                if is_paused {
                    thread::sleep(Duration::from_millis(50));
                    continue;
                }

                // Check if a seek is requested
                if let Some(seek_frame) = *state.seek_target.lock().unwrap() {
                    *state.seek_target.lock().unwrap() = None;
                    if let Err(e) = self.seek(seek_frame) {
                        eprintln!("Seek failed: {}", e);
                    }
                    continue;
                }

                let ret = av_read_frame(fmt_ctx, packet.as_ptr());
                if ret < 0 {
                    if ret == AVERROR_EOF {
                        println!("End of file reached");
                        break;
                    }
                    return Err(anyhow!("Error reading frame: {}", av_err2str(ret)));
                }

                // Only process video packets
                if (*packet.as_ptr()).stream_index == self.video_stream_idx as i32 {
                    let ret = avcodec_send_packet(self.codec_ctx.as_ptr(), packet.as_ptr());
                    if ret < 0 && ret != AVERROR_EAGAIN {
                        eprintln!("Error sending packet: {}", av_err2str(ret));
                        packet.unref();
                        continue;
                    }

                    loop {
                        let decode_frame = if self.using_hw {
                            hw_frame.as_ref().unwrap().as_ptr()
                        } else {
                            frame.as_ptr()
                        };

                        let ret = avcodec_receive_frame(self.codec_ctx.as_ptr(), decode_frame);
                        if ret == AVERROR_EAGAIN || ret == AVERROR_EOF {
                            break;
                        }
                        if ret < 0 {
                            return Err(anyhow!("Error decoding frame: {}", av_err2str(ret)));
                        }

                        // Get the actual frame to convert
                        let cpu_frame = if self.using_hw {
                            let ret = av_hwframe_transfer_data(frame.as_ptr(), decode_frame, 0);
                            if ret < 0 {
                                eprintln!("Failed to transfer frame from GPU: {}", av_err2str(ret));
                                av_frame_unref(decode_frame);
                                continue;
                            }

                            av_frame_copy_props(frame.as_ptr(), decode_frame);
                            frame.as_ptr()
                        } else {
                            decode_frame
                        };

                        // Convert to RGB and send to UI
                        match self.convert_frame_to_rgb(cpu_frame) {
                            Ok(decoded) => {
                                if sender.send(decoded).is_err() {
                                    println!("UI closed, exiting decoder");
                                    av_frame_unref(decode_frame);
                                    if self.using_hw {
                                        av_frame_unref(frame.as_ptr());
                                    }
                                    return Ok(());
                                }
                            }
                            Err(e) => {
                                eprintln!("Failed to convert frame {}: {}", self.current_frame, e);
                            }
                        }

                        av_frame_unref(decode_frame);
                        if self.using_hw {
                            av_frame_unref(frame.as_ptr());
                        }

                        self.current_frame += 1;
                    }
                }

                packet.unref();
            }

            println!("Flushing decoder...");
            // Flush decoder
            avcodec_send_packet(self.codec_ctx.as_ptr(), ptr::null());
            loop {
                let decode_frame = if self.using_hw {
                    hw_frame.as_ref().unwrap().as_ptr()
                } else {
                    frame.as_ptr()
                };

                let ret = avcodec_receive_frame(self.codec_ctx.as_ptr(), decode_frame);
                if ret == AVERROR_EOF || ret == AVERROR_EAGAIN {
                    break;
                }
                if ret >= 0 {
                    let cpu_frame = if self.using_hw {
                        let ret = av_hwframe_transfer_data(frame.as_ptr(), decode_frame, 0);
                        if ret < 0 {
                            av_frame_unref(decode_frame);
                            continue;
                        }
                        av_frame_copy_props(frame.as_ptr(), decode_frame);
                        frame.as_ptr()
                    } else {
                        decode_frame
                    };

                    if let Ok(decoded) = self.convert_frame_to_rgb(cpu_frame) {
                        let _ = sender.send(decoded);
                    }

                    av_frame_unref(decode_frame);
                    if self.using_hw {
                        av_frame_unref(frame.as_ptr());
                    }

                    self.current_frame += 1;
                }
            }

            println!(
                "Decoder finished. Total frames processed: {}",
                self.current_frame
            );
            Ok(())
        }
    }

    /// Convert an `AVFrame` to RGB format
    unsafe fn convert_frame_to_rgb(&mut self, frame: *mut AVFrame) -> Result<DecodedFrame> {
        unsafe {
            let frame_ref = &*frame;
            let width = frame_ref.width as u32;
            let height = frame_ref.height as u32;

            if width == 0 || height == 0 {
                return Err(anyhow!("Invalid frame dimensions: {width}x{height}"));
            }

            // Initialize SwsContext if needed
            if self.sws_ctx.is_none() {
                println!(
                    "Creating SwsContext for format {} -> RGBA",
                    frame_ref.format
                );

                let sws = sws_getContext(
                    frame_ref.width,
                    frame_ref.height,
                    std::mem::transmute(frame_ref.format),
                    frame_ref.width,
                    frame_ref.height,
                    AVPixelFormat::AV_PIX_FMT_RGBA,
                    SWS_BILINEAR as i32,
                    ptr::null_mut(),
                    ptr::null_mut(),
                    ptr::null(),
                );

                if sws.is_null() {
                    return Err(anyhow!(
                        "Failed to create SwsContext for format {}",
                        frame_ref.format
                    ));
                }

                self.sws_ctx = Some(SwsContext { ctx: sws });
            }

            // Allocate RGB buffer
            let rgb_size = (width * height * 4) as usize;
            let mut rgb_data = vec![0u8; rgb_size];

            // Convert frame to RGBA
            let rgb_linesize = (width * 4) as i32;
            let rgb_data_ptr = rgb_data.as_mut_ptr();
            let ret = sws_scale(
                self.sws_ctx.as_ref().unwrap().as_ptr(),
                frame_ref.data.as_ptr().cast::<*const u8>(),
                frame_ref.linesize.as_ptr(),
                0,
                frame_ref.height,
                &raw const rgb_data_ptr,
                &raw const rgb_linesize,
            );

            if ret < 0 {
                return Err(anyhow!("Failed to convert frame: {}", av_err2str(ret)));
            }

            Ok(DecodedFrame {
                data: rgb_data,
                width,
                height,
                frame_number: self.current_frame,
            })
        }
    }

    fn get_metadata(&self) -> VideoMetadata {
        self.metadata.clone()
    }
}

/// Convert `FFmpeg` error code to string
unsafe fn av_err2str(errnum: i32) -> String {
    unsafe {
        let mut buf = [0i8; AV_ERROR_MAX_STRING_SIZE];
        av_strerror(errnum, buf.as_mut_ptr(), AV_ERROR_MAX_STRING_SIZE);
        CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned()
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <video_file>", args[0]);
        std::process::exit(1);
    }

    let video_path = &args[1];

    // Create the video decoder
    let decoder = VideoDecoder::new(video_path).context("Failed to create video decoder")?;
    let metadata = decoder.get_metadata();

    // Create shared state
    let state = Arc::new(PlayerState::new());

    // Create channels
    let (frame_sender, frame_receiver): (Sender<DecodedFrame>, Receiver<DecodedFrame>) =
        bounded(30);
    let (command_sender, command_receiver): (Sender<PlayerCommand>, Receiver<PlayerCommand>) =
        bounded(10);

    // Clone state for decoder thread
    let decoder_state = state.clone();

    // Spawn decoder thread
    thread::spawn(move || {
        if let Err(e) = decoder.decode_video(&frame_sender, &command_receiver, decoder_state) {
            eprintln!("Decoder error: {e}");
        }
    });

    // Create and run Slint UI
    let ui = VideoPlayer::new()?;

    // Set initial UI state and start playing
    ui.set_video_duration_ms(metadata.duration_ms as i32);
    ui.set_current_time_ms(0);
    ui.set_is_playing(true);
    ui.set_volume(100);
    state.is_playing.store(true, Ordering::Relaxed);

    // Setup callbacks
    let cmd_sender = command_sender.clone();
    let state_clone = state.clone();
    ui.on_play_pause_clicked(move || {
        let is_playing = state_clone.is_playing.load(Ordering::Relaxed);
        if is_playing {
            let _ = cmd_sender.send(PlayerCommand::Pause);
        } else {
            let _ = cmd_sender.send(PlayerCommand::Play);
        }
    });

    let state_clone = state.clone();
    let metadata_clone = metadata.clone();
    ui.on_seek_changed(move |position_ms| {
        let frame_num = metadata_clone.ms_to_frame(position_ms as u64);
        *state_clone.seek_target.lock().unwrap() = Some(frame_num);
    });

    let cmd_sender = command_sender.clone();
    let ui_handle_vol = ui.as_weak();
    ui.on_volume_changed(move |volume| {
        let _ = cmd_sender.send(PlayerCommand::SetVolume(volume));
        let _ = ui_handle_vol.upgrade_in_event_loop(move |handle| {
            handle.set_volume(volume);
        });
    });

    let ui_handle_mute = ui.as_weak();
    let state_clone = state.clone();
    ui.on_mute_clicked(move || {
        let is_muted = state_clone.is_muted.load(Ordering::Relaxed);
        state_clone.is_muted.store(!is_muted, Ordering::Relaxed);
        let _ = ui_handle_mute.upgrade_in_event_loop(move |handle| {
            handle.set_is_muted(!is_muted);
        });
    });

    let ui_handle_fs = ui.as_weak();
    ui.on_fullscreen_clicked(move || {
        let _ = ui_handle_fs.upgrade_in_event_loop(|handle| {
            let is_fullscreen = handle.get_is_fullscreen();
            handle.set_is_fullscreen(!is_fullscreen);
        });
    });

    // Spawn frame update thread with FPS-based timing
    let ui_handle = ui.as_weak();
    let display_state = state.clone();
    let fps = metadata.get_fps_rounded();
    let frame_interval = Duration::from_millis(1000 / fps);

    thread::spawn(move || {
        while let Ok(frame) = frame_receiver.recv() {
            let frame_start = Instant::now();
            let is_playing = display_state.is_playing.load(Ordering::Relaxed);

            // Capture frame data for moving into the closure
            let data = frame.data;
            let width = frame.width;
            let height = frame.height;
            let frame_ms = metadata.frame_to_ms(frame.frame_number);

            // Create the image inside the event loop
            let _ = ui_handle.upgrade_in_event_loop(move |handle| {
                let pixel_buffer =
                    SharedPixelBuffer::<Rgba8Pixel>::clone_from_slice(&data, width, height);
                let image = Image::from_rgba8(pixel_buffer);
                handle.set_video_frame(image);
                handle.set_current_time_ms(frame_ms as i32);
                handle.set_is_playing(is_playing);
            });

            // Enforce frame timing
            if is_playing {
                let elapsed = frame_start.elapsed();
                if elapsed < frame_interval {
                    thread::sleep(frame_interval - elapsed);
                }
            } else {
                thread::sleep(Duration::from_millis(10));
            }
        }
    });

    ui.run()?;

    // Send stop command to decoder thread when UI closes
    let _ = command_sender.send(PlayerCommand::Stop);

    Ok(())
}
