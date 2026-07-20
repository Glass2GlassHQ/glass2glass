//! M740: on-screen macOS present. H.264 decoded by VideoToolbox in `cv-output`
//! mode (retained IOSurface-backed `CVPixelBuffer`s, no CPU copy), presented in
//! a real window by [`MetalVideoSink`]: the app owns the window (winit) and a
//! `CAMetalLayer` hosted in its `NSView`, and hands that layer to the sink via
//! [`MetalVideoSink::with_layer`]. The on-screen sibling of the headless
//! `m736_metal_sink` test, and the macOS analog of `vulkan_video_on_screen`.
//!
//! Window + event loop ownership belongs to the application (an element cannot
//! own `NSApplication` / the main thread), which is why this is an example, not
//! a self-checking test.
//!
//! Run (macOS only):
//!
//! ```sh
//! cargo run --release -p g2g-plugins --features vtdecode,metal-sink \
//!     --example metal_video_on_screen              # the bundled 640x480 clip
//! cargo run --release -p g2g-plugins --features vtdecode,metal-sink \
//!     --example metal_video_on_screen -- my.h264   # an Annex-B H.264 stream
//! ```
//!
//! The bundled clip is two GOPs of animated content, so it visibly moves; it
//! loops. Close the window (or Esc) to quit.

#[cfg(not(all(target_os = "macos", feature = "vtdecode", feature = "metal-sink")))]
fn main() {
    eprintln!("this example needs macOS and --features vtdecode,metal-sink");
}

#[cfg(all(target_os = "macos", feature = "vtdecode", feature = "metal-sink"))]
fn main() {
    demo::run();
}

#[cfg(all(target_os = "macos", feature = "vtdecode", feature = "metal-sink"))]
mod demo {
    use core::future::Future;
    use core::pin::Pin;
    use core::ptr::NonNull;
    use std::time::{Duration, Instant};

    use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
    use g2g_core::memory::{MemoryDomain, OwnedCvPixelBuffer, SystemSlice};
    use g2g_core::runtime::block_on;
    use g2g_core::{
        AsyncElement, Caps, Dim, G2gError, OutputSink, PushOutcome, Rate, RawVideoFormat,
        VideoCodec,
    };
    use g2g_plugins::h264parse::H264Parse;
    use g2g_plugins::metalvideosink::MetalVideoSink;
    use g2g_plugins::vtdecode::VtDecode;

    use objc2::msg_send;
    use objc2::runtime::AnyObject;
    use objc2_quartz_core::CAMetalLayer;

    use winit::application::ApplicationHandler;
    use winit::event::{ElementState, KeyEvent, WindowEvent};
    use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
    use winit::keyboard::{Key, NamedKey};
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use winit::window::{Window, WindowId};

    /// The 640x480 baseline clip the tests use, embedded so the demo runs with
    /// no arguments (two GOPs of IDR + P frames, so it exercises the DPB and
    /// moves).
    const BUNDLED_CLIP: &[u8] = include_bytes!("../tests/fixtures/h264_640x480.h264");

    /// How long each decoded frame stays on screen before advancing. The clip
    /// is authored at 30 fps; 66 ms (~15 fps) makes the animation easy to
    /// follow while it loops.
    const FRAME_INTERVAL: Duration = Duration::from_millis(66);

    const FRAME_DURATION_NS: u64 = 33_333_333;

    /// Collects an element's output packets (the harness the tests use).
    #[derive(Default)]
    struct Collect {
        packets: Vec<PipelinePacket>,
    }

    impl OutputSink for Collect {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async move {
                self.packets.push(packet);
                Ok(PushOutcome::Accepted)
            })
        }
    }

    /// Discards output (the present sink is the pipeline tail).
    struct NullSink;

    impl OutputSink for NullSink {
        fn push<'a>(
            &'a mut self,
            _packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async move { Ok(PushOutcome::Accepted) })
        }
    }

    /// Parse + decode the whole clip up front (cv-output, zero-copy), returning
    /// the retained pixel buffers and the stream geometry from the SPS.
    async fn decode_clip(clip: &[u8]) -> (Vec<OwnedCvPixelBuffer>, u32, u32) {
        let caps = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        let mut parse = H264Parse::reframing();
        parse.configure_pipeline(&caps).expect("configure parser");
        let mut sink = Collect::default();
        for (i, chunk) in clip.chunks(4096).enumerate() {
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(chunk.to_vec().into_boxed_slice())),
                FrameTiming::default(),
                i as u64,
            );
            parse
                .process(PipelinePacket::DataFrame(frame), &mut sink)
                .await
                .expect("parse chunk");
        }
        parse
            .process(PipelinePacket::Eos, &mut sink)
            .await
            .expect("Eos");

        // Geometry comes from the parser's CapsChanged (parsed out of the SPS).
        let (w, h) = sink
            .packets
            .iter()
            .find_map(|p| match p {
                PipelinePacket::CapsChanged(Caps::CompressedVideo {
                    width: Dim::Fixed(w),
                    height: Dim::Fixed(h),
                    ..
                }) => Some((*w, *h)),
                _ => None,
            })
            .expect("stream has an SPS with fixed geometry");
        let mut aus: Vec<Frame> = sink
            .packets
            .into_iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some(f),
                _ => None,
            })
            .collect();
        for (i, au) in aus.iter_mut().enumerate() {
            au.timing.pts_ns = i as u64 * FRAME_DURATION_NS;
        }

        let mut dec = VtDecode::h264().with_cv_output();
        let dec_caps = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Fixed(30 << 16),
        };
        let narrowed = dec.intercept_caps(&dec_caps).expect("intercept H.264");
        dec.configure_pipeline(&narrowed).expect("decoder session");
        let mut sink = Collect::default();
        for au in aus {
            dec.process(PipelinePacket::DataFrame(au), &mut sink)
                .await
                .expect("decode");
        }
        dec.process(PipelinePacket::Eos, &mut sink)
            .await
            .expect("drain");

        let bufs = sink
            .packets
            .iter()
            .filter_map(|p| match p {
                // Clone is a refcount bump: the buffers stay decoded once and
                // are re-presented every loop of the clip.
                PipelinePacket::DataFrame(f) => match &f.domain {
                    MemoryDomain::CvPixelBuffer(b) => Some(b.clone()),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        (bufs, w, h)
    }

    struct App {
        frames: Vec<OwnedCvPixelBuffer>,
        width: u32,
        height: u32,
        // Created once the event loop resumes and a window exists.
        window: Option<Window>,
        sink: Option<MetalVideoSink>,
        idx: usize,
        last_advance: Instant,
    }

    impl App {
        /// Present the current pixel buffer, advancing once `FRAME_INTERVAL`
        /// has elapsed so playback runs at a watchable rate and loops.
        fn present_current(&mut self) {
            let Some(sink) = self.sink.as_mut() else {
                return;
            };
            let frame = Frame::new(
                MemoryDomain::CvPixelBuffer(self.frames[self.idx].clone()),
                FrameTiming::default(),
                self.idx as u64,
            );
            block_on(sink.process(PipelinePacket::DataFrame(frame), &mut NullSink))
                .expect("present");

            // A periodic heartbeat so the live present is visible in the terminal.
            let n = sink.presented();
            if n % 120 == 0 {
                eprintln!(
                    "  presented {n} frames (looping the {}-frame clip)",
                    self.frames.len()
                );
            }

            if self.last_advance.elapsed() >= FRAME_INTERVAL {
                self.idx = (self.idx + 1) % self.frames.len();
                self.last_advance = Instant::now();
            }
        }
    }

    impl ApplicationHandler for App {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if self.window.is_some() {
                return;
            }
            let attrs = Window::default_attributes()
                .with_title("g2g: VideoToolbox -> MetalVideoSink")
                .with_inner_size(winit::dpi::PhysicalSize::new(self.width, self.height));
            let window = event_loop.create_window(attrs).expect("create window");

            // Host a CAMetalLayer in the window's NSView (layer-hosting: set
            // the layer before wantsLayer) and hand it to the sink; AppKit
            // resizes the hosted layer with the view.
            let RawWindowHandle::AppKit(handle) =
                window.window_handle().expect("window handle").as_raw()
            else {
                panic!("not an AppKit window");
            };
            let layer = CAMetalLayer::layer();
            // SAFETY: ns_view is the live NSView of the window we just created,
            // and we are on the main thread (winit delivers resumed there).
            unsafe {
                let view: &AnyObject = handle.ns_view.cast::<AnyObject>().as_ref();
                let _: () = msg_send![view, setLayer: &*layer];
                let _: () = msg_send![view, setWantsLayer: true];
            }
            // SAFETY: the layer is a valid CAMetalLayer the view now hosts; the
            // app does not mutate it while the sink presents.
            let mut sink = unsafe { MetalVideoSink::new().with_layer(NonNull::from(&*layer)) };
            let caps = Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(self.width),
                height: Dim::Fixed(self.height),
                framerate: Rate::Fixed(30 << 16),
            };
            sink.configure_pipeline(&caps).expect("sink configure");
            self.sink = Some(sink);
            window.request_redraw();
            self.window = Some(window);
        }

        fn window_event(
            &mut self,
            event_loop: &ActiveEventLoop,
            _id: WindowId,
            event: WindowEvent,
        ) {
            match event {
                WindowEvent::CloseRequested
                | WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            logical_key: Key::Named(NamedKey::Escape),
                            state: ElementState::Pressed,
                            ..
                        },
                    ..
                } => {
                    event_loop.exit();
                }
                WindowEvent::RedrawRequested => {
                    self.present_current();
                    if let Some(window) = self.window.as_ref() {
                        window.request_redraw();
                    }
                }
                _ => {}
            }
        }
    }

    pub fn run() {
        if !MetalVideoSink::device_available() {
            eprintln!("no Metal device; nothing to present on.");
            return;
        }
        let clip_path = std::env::args().nth(1);
        let clip: Vec<u8> = match &clip_path {
            Some(p) => std::fs::read(p).unwrap_or_else(|e| panic!("read {p}: {e}")),
            None => BUNDLED_CLIP.to_vec(),
        };
        println!(
            "decoding {} ({} bytes)",
            clip_path.as_deref().unwrap_or("<bundled 640x480 clip>"),
            clip.len()
        );
        let (frames, width, height) = block_on(decode_clip(&clip));
        println!(
            "decoded {} frames at {width}x{height} (VideoToolbox cv-output, zero-copy)",
            frames.len()
        );
        if frames.is_empty() {
            eprintln!("stream produced no frames; nothing to show.");
            return;
        }

        let event_loop = EventLoop::new().expect("create event loop");
        event_loop.set_control_flow(ControlFlow::Poll);
        let mut app = App {
            frames,
            width,
            height,
            window: None,
            sink: None,
            idx: 0,
            last_advance: Instant::now(),
        };
        if let Err(e) = event_loop.run_app(&mut app) {
            eprintln!("event loop ended: {e}");
        }
        let shown = app.sink.as_ref().map(|s| s.presented()).unwrap_or(0);
        println!("presented {shown} frames; bye.");
    }
}
