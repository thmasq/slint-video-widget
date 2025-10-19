use anyhow::{Context, Result, anyhow};
use glib::prelude::*;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use slint::{Image, Rgba8Pixel, SharedPixelBuffer};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::{env, thread};

slint::include_modules!();

/// Represents a decoded video frame ready for display
struct DecodedFrame {
    data: Vec<u8>,
    width: u32,
    height: u32,
    position_ms: i64,
}

/// Video player state
struct VideoPlayer {
    pipeline: gst::Pipeline,
    appsink: gst_app::AppSink,
    width: i32,
    height: i32,
    duration_ms: i64,
    framerate: f64,
    is_playing: Arc<AtomicBool>,
    is_eos: Arc<AtomicBool>,
    volume_element: gst::Element,
}

impl VideoPlayer {
    fn new(path: &str) -> Result<Self> {
        // Initialize GStreamer
        gst::init()?;

        // Create the pipeline
        // Using decodebin for automatic codec selection and videoconvert for format conversion
        let pipeline_str = format!(
            "filesrc location=\"{}\" ! decodebin name=d ! videoconvert ! videoscale ! \
             appsink name=sink caps=video/x-raw,format=RGBA,pixel-aspect-ratio=1/1 \
             d. ! audioconvert ! volume name=audio-volume ! autoaudiosink",
            path
        );

        let pipeline = gst::parse::launch(&pipeline_str)?
            .downcast::<gst::Pipeline>()
            .map_err(|_| anyhow!("Failed to create pipeline"))?;

        // Get the appsink element
        let appsink = pipeline
            .by_name("sink")
            .ok_or_else(|| anyhow!("Failed to get appsink"))?
            .downcast::<gst_app::AppSink>()
            .map_err(|_| anyhow!("Failed to cast to AppSink"))?;

        let volume_element = pipeline
            .by_name("audio-volume")
            .ok_or_else(|| anyhow!("Failed to get volume element"))?;

        // Configure appsink
        appsink.set_property("drop", true);
        appsink.set_property("max-buffers", 2u32);

        // Start playing to get stream info
        pipeline
            .set_state(gst::State::Paused)
            .map_err(|_| anyhow!("Failed to set pipeline to Paused"))?;

        // Wait for state change
        pipeline
            .state(gst::ClockTime::from_seconds(5))
            .0
            .map_err(|_| anyhow!("Failed to get pipeline state"))?;

        // Get video information from the caps
        let pad = appsink
            .static_pad("sink")
            .ok_or_else(|| anyhow!("Failed to get sink pad"))?;
        let caps = pad
            .current_caps()
            .ok_or_else(|| anyhow!("Failed to get caps"))?;
        let s = caps
            .structure(0)
            .ok_or_else(|| anyhow!("Failed to get structure"))?;

        let width = s.get::<i32>("width")?;
        let height = s.get::<i32>("height")?;
        let framerate = s.get::<gst::Fraction>("framerate")?;
        let fps = framerate.numer() as f64 / framerate.denom() as f64;

        // Get duration
        let duration = pipeline
            .query_duration::<gst::ClockTime>()
            .map(|d| d.mseconds())
            .unwrap_or(0);

        println!("Video info:");
        println!("  Resolution: {}x{}", width, height);
        println!("  FPS: {:.2}", fps);
        println!("  Duration: {} ms", duration);

        Ok(VideoPlayer {
            pipeline,
            appsink,
            width,
            height,
            duration_ms: duration as i64,
            framerate: fps,
            is_playing: Arc::new(AtomicBool::new(false)),
            is_eos: Arc::new(AtomicBool::new(false)),
            volume_element,
        })
    }

    fn play(&self) {
        self.pipeline
            .set_state(gst::State::Playing)
            .expect("Unable to set pipeline to Playing");
        self.is_playing.store(true, Ordering::Relaxed);
    }

    fn pause(&self) {
        self.pipeline
            .set_state(gst::State::Paused)
            .expect("Unable to set pipeline to Paused");
        self.is_playing.store(false, Ordering::Relaxed);
    }

    fn seek(&self, position_ms: i64) -> Result<()> {
        self.pipeline
            .seek_simple(
                gst::SeekFlags::FLUSH | gst::SeekFlags::KEY_UNIT,
                gst::ClockTime::from_mseconds(position_ms as u64),
            )
            .map_err(|_| anyhow!("Seek failed"))?;
        Ok(())
    }

    fn get_position(&self) -> i64 {
        self.pipeline
            .query_position::<gst::ClockTime>()
            .map(|pos| pos.mseconds() as i64)
            .unwrap_or(0)
    }

    fn set_volume(&self, volume: f64) {
        self.volume_element.set_property("volume", volume);
    }

    fn set_muted(&self, muted: bool) {
        self.volume_element.set_property("mute", muted);
    }

    /// Pull a frame from the appsink
    fn pull_frame(&self) -> Option<DecodedFrame> {
        let sample = if self.is_playing.load(Ordering::Relaxed) {
            self.appsink
                .try_pull_sample(gst::ClockTime::from_mseconds(16))
        } else {
            self.appsink
                .try_pull_preroll(gst::ClockTime::from_mseconds(16))
        }?;

        let buffer = sample.buffer()?;
        let map = buffer.map_readable().ok()?;

        // Get frame info
        let caps = sample.caps()?;
        let s = caps.structure(0)?;
        let width = s.get::<i32>("width").ok()? as u32;
        let height = s.get::<i32>("height").ok()? as u32;

        // Get position
        let position_ms = buffer.pts().map(|pts| pts.mseconds() as i64).unwrap_or(0);

        // Copy RGBA data
        let data = map.as_slice().to_vec();

        Some(DecodedFrame {
            data,
            width,
            height,
            position_ms,
        })
    }

    fn check_bus(&self) -> bool {
        if let Some(bus) = self.pipeline.bus() {
            while let Some(msg) =
                bus.pop_filtered(&[gst::MessageType::Error, gst::MessageType::Eos])
            {
                match msg.view() {
                    gst::MessageView::Error(err) => {
                        eprintln!(
                            "Error from {:?}: {} ({:?})",
                            err.src().map(|s| s.path_string()),
                            err.error(),
                            err.debug()
                        );
                        return false;
                    }
                    gst::MessageView::Eos(..) => {
                        println!("End of stream");
                        self.is_eos.store(true, Ordering::Relaxed);
                        self.pause();
                    }
                    _ => {}
                }
            }
        }
        true
    }
}

impl Drop for VideoPlayer {
    fn drop(&mut self) {
        let _ = self.pipeline.set_state(gst::State::Null);
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <video_file>", args[0]);
        std::process::exit(1);
    }

    let video_path = &args[1];

    // Create video player
    let player = Arc::new(Mutex::new(
        VideoPlayer::new(video_path).context("Failed to create video player")?,
    ));

    // Get initial metadata
    let (width, height, duration_ms, framerate) = {
        let p = player.lock().unwrap();
        (p.width, p.height, p.duration_ms, p.framerate)
    };

    // Create Slint UI
    let ui = VideoPlayerUI::new()?;

    // Set initial state
    ui.set_video_duration_ms(duration_ms as i32);
    ui.set_current_time_ms(0);
    ui.set_is_playing(false);
    ui.set_volume(100);
    ui.set_is_muted(false);
    ui.set_is_fullscreen(false);

    // Play/Pause callback
    let player_clone = player.clone();
    ui.on_play_pause_clicked(move || {
        let p = player_clone.lock().unwrap();
        if p.is_eos.load(Ordering::Relaxed) {
            if let Err(e) = p.seek(0) {
                eprintln!("Seek failed: {}", e);
            }
            p.is_eos.store(false, Ordering::Relaxed);
            p.play();
        } else if p.is_playing.load(Ordering::Relaxed) {
            p.pause();
        } else {
            p.play();
        }
    });

    // Seek callback
    let player_clone = player.clone();
    ui.on_seek_changed(move |position_ms| {
        let p = player_clone.lock().unwrap();
        if let Err(e) = p.seek(position_ms as i64) {
            eprintln!("Seek failed: {}", e);
        }
        if p.is_eos.load(Ordering::Relaxed) {
            p.is_eos.store(false, Ordering::Relaxed);
        }
    });

    // Volume callback
    let player_clone = player.clone();
    let ui_weak = ui.as_weak();
    ui.on_volume_changed(move |volume| {
        let p = player_clone.lock().unwrap();
        p.set_volume(volume as f64 / 100.0);
        let _ = ui_weak.upgrade_in_event_loop(move |handle| {
            handle.set_volume(volume);
        });
    });

    // Mute callback
    let player_clone = player.clone();
    let ui_weak = ui.as_weak();
    ui.on_mute_clicked(move || {
        let p = player_clone.lock().unwrap();
        let is_muted = ui_weak.upgrade().map(|h| h.get_is_muted()).unwrap_or(false);
        p.set_muted(!is_muted);
        let _ = ui_weak.upgrade_in_event_loop(move |handle| {
            handle.set_is_muted(!is_muted);
        });
    });

    // Fullscreen callback
    let ui_weak = ui.as_weak();
    ui.on_fullscreen_clicked(move || {
        let _ = ui_weak.upgrade_in_event_loop(|handle| {
            let is_fullscreen = handle.get_is_fullscreen();
            handle.set_is_fullscreen(!is_fullscreen);
        });
    });

    // Start playback
    {
        let p = player.lock().unwrap();
        p.play();
    }

    // Frame update thread
    let ui_weak = ui.as_weak();
    let player_clone = player.clone();
    let frame_interval = Duration::from_millis(1000 / framerate.round() as u64);

    thread::spawn(move || {
        loop {
            let frame_start = Instant::now();

            // Pull frame
            let (frame_opt, is_playing, should_continue) = {
                let p = player_clone.lock().unwrap();
                let frame = p.pull_frame();
                let is_playing = p.is_playing.load(Ordering::Relaxed);
                let should_continue = p.check_bus();
                (frame, is_playing, should_continue)
            };

            if !should_continue {
                break;
            }

            // Update UI with new frame
            if let Some(frame) = frame_opt {
                let _ = ui_weak.upgrade_in_event_loop(move |handle| {
                    let pixel_buffer = SharedPixelBuffer::<Rgba8Pixel>::clone_from_slice(
                        &frame.data,
                        frame.width,
                        frame.height,
                    );
                    let image = Image::from_rgba8(pixel_buffer);
                    handle.set_video_frame(image);
                    handle.set_current_time_ms(frame.position_ms as i32);
                    handle.set_is_playing(is_playing);
                });
            }

            // Update position even if no new frame
            let position_ms = {
                let p = player_clone.lock().unwrap();
                p.get_position()
            };

            let _ = ui_weak.upgrade_in_event_loop(move |handle| {
                handle.set_current_time_ms(position_ms as i32);
            });

            // Frame timing
            let elapsed = frame_start.elapsed();
            if elapsed < frame_interval {
                thread::sleep(frame_interval - elapsed);
            }
        }

        println!("Frame update thread finished");
    });

    // Run UI
    ui.run()?;

    Ok(())
}
