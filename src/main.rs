use anyhow::{Context, Result, anyhow};
use crossbeam_channel::{Receiver, Sender, bounded};
use ffmpeg_sys_next::SwsFlags::SWS_BILINEAR;
use ffmpeg_sys_next::{
    AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX, AV_ERROR_MAX_STRING_SIZE, AV_NOPTS_VALUE, AVBufferRef,
    AVCodec, AVCodecContext, AVCodecParameters, AVERROR_EOF, AVFormatContext, AVFrame,
    AVHWDeviceType, AVMediaType, AVPacket, AVPixelFormat, av_buffer_ref, av_buffer_unref,
    av_frame_alloc, av_frame_copy_props, av_frame_free, av_frame_unref, av_hwdevice_ctx_create,
    av_hwframe_transfer_data, av_packet_alloc, av_packet_free, av_packet_unref, av_read_frame,
    av_strerror, avcodec_alloc_context3, avcodec_find_decoder, avcodec_free_context,
    avcodec_get_hw_config, avcodec_open2, avcodec_parameters_to_context, avcodec_receive_frame,
    avcodec_send_packet, avformat_close_input, avformat_find_stream_info, avformat_open_input,
    sws_freeContext, sws_getContext, sws_scale,
};
use slint::{Image, Rgba8Pixel, SharedPixelBuffer};
use std::env;
use std::ffi::{CStr, CString};
use std::ptr;
use std::thread;
use std::time::{Duration, Instant};

slint::include_modules!();

const AVERROR_EAGAIN: i32 = -11;

/// Represents a decoded video frame ready for display
struct DecodedFrame {
    /// RGB pixel data
    data: Vec<u8>,
    /// Frame width
    width: u32,
    /// Frame height
    height: u32,
    /// Presentation timestamp in seconds
    pts: f64,
}

/// RAII wrapper for `FFmpeg` format context
struct FormatContext {
    ctx: *mut AVFormatContext,
}

// SAFETY: FFmpeg contexts are safe to send between threads when not shared.
// Each thread owns its context exclusively.
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

// SAFETY: Codec contexts are safe to send between threads when owned exclusively
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

// SAFETY: SwsContext is safe to send between threads when owned exclusively
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

// SAFETY: Hardware device contexts use reference counting and are safe to send
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
    using_hw: bool,
    hw_pix_fmt: AVPixelFormat,
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
                using_hw,
                hw_pix_fmt,
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
            let preferred_types = vec![
                AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI,
                // AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA,
            ];

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
                    // Hardware context created successfully
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

                    // Set the hardware pixel format callback for VAAPI
                    if hw_type == AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI {
                        extern "C" fn get_hw_format(
                            _ctx: *mut AVCodecContext,
                            pix_fmts: *const AVPixelFormat,
                        ) -> AVPixelFormat {
                            let mut p = pix_fmts;
                            unsafe {
                                while *p != AVPixelFormat::AV_PIX_FMT_NONE {
                                    // For VAAPI, we typically want VAAPI format
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

                    // Hardware decoder failed to open, clean up
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

    /// Decode the video and send frames to the UI thread
    fn decode_video(mut self, sender: &Sender<DecodedFrame>) -> Result<()> {
        unsafe {
            let packet = Packet::new()?;
            let frame = Frame::new()?;
            let hw_frame = if self.using_hw {
                Some(Frame::new()?)
            } else {
                None
            };

            let fmt_ctx = self.format_ctx.as_ptr();
            let mut frame_count = 0u64;

            // Main decoding loop
            loop {
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

                        frame_count += 1;

                        // Get the actual frame to convert
                        let cpu_frame = if self.using_hw {
                            // Transfer from GPU to CPU
                            let ret = av_hwframe_transfer_data(frame.as_ptr(), decode_frame, 0);
                            if ret < 0 {
                                eprintln!("Failed to transfer frame from GPU: {}", av_err2str(ret));
                                av_frame_unref(decode_frame);
                                continue;
                            }

                            // Copy properties (pts, etc.)
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
                                eprintln!("Failed to convert frame {frame_count}: {e}");
                            }
                        }

                        // Unref frames
                        av_frame_unref(decode_frame);
                        if self.using_hw {
                            av_frame_unref(frame.as_ptr());
                        }
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
                }
            }

            println!("Decoder finished. Total frames processed: {frame_count}");
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

            // Initialize SwsContext if needed (or reinitialize if format changed)
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

            // Calculate presentation timestamp
            let pts = if frame_ref.pts == AV_NOPTS_VALUE {
                0.0
            } else {
                frame_ref.pts as f64 * self.time_base
            };

            Ok(DecodedFrame {
                data: rgb_data,
                width,
                height,
                pts,
            })
        }
    }

    /// Get video frame rate
    fn get_fps(&self) -> f64 {
        unsafe {
            let stream = *(*self.format_ctx.as_ptr())
                .streams
                .add(self.video_stream_idx);
            let avg_frame_rate = (*stream).avg_frame_rate;

            if avg_frame_rate.den > 0 {
                f64::from(avg_frame_rate.num) / f64::from(avg_frame_rate.den)
            } else {
                25.0 // Default fallback
            }
        }
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

    let fps = decoder.get_fps();
    println!("Video FPS: {fps:.2}");

    // Create channel for frame communication
    let (frame_sender, frame_receiver): (Sender<DecodedFrame>, Receiver<DecodedFrame>) =
        bounded(30);

    // Spawn decoder thread
    thread::spawn(move || {
        if let Err(e) = decoder.decode_video(&frame_sender) {
            eprintln!("Decoder error: {e}");
        }
    });

    // Create and run Slint UI
    let ui = VideoPlayer::new()?;

    // Spawn frame update thread
    let ui_handle = ui.as_weak();
    thread::spawn(move || {
        let mut start_time: Option<Instant> = None;
        let mut first_pts: Option<f64> = None;

        while let Ok(frame) = frame_receiver.recv() {
            // Initialize timing on first frame
            if start_time.is_none() {
                start_time = Some(Instant::now());
                first_pts = Some(frame.pts);
            }

            // Calculate target display time
            let elapsed = start_time.unwrap().elapsed().as_secs_f64();
            let target_time = frame.pts - first_pts.unwrap();

            // Sleep if we're ahead of schedule
            if target_time > elapsed {
                let sleep_duration = Duration::from_secs_f64(target_time - elapsed);
                thread::sleep(sleep_duration);
            }

            // Capture frame data for moving into the closure
            let data = frame.data;
            let width = frame.width;
            let height = frame.height;

            // Create the image inside the event loop to avoid Send issues
            let _ = ui_handle.upgrade_in_event_loop(move |handle| {
                let pixel_buffer =
                    SharedPixelBuffer::<Rgba8Pixel>::clone_from_slice(&data, width, height);
                let image = Image::from_rgba8(pixel_buffer);
                handle.set_video_frame(image);
            });
        }
    });

    ui.run()?;
    Ok(())
}
