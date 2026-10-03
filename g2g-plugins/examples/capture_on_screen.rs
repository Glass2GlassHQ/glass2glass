//! Live macOS capture to an on-screen window: `AvfVideoSrc` (camera) or
//! `ScreenCaptureSrc` (display) in `cv-output` mode, presented by
//! [`MetalVideoSink`] with no CPU copy. The live-capture sibling of
//! `metal_video_on_screen`: the source runs on its own thread and hands each
//! retained `CVPixelBuffer` to the window's event loop, which presents the
//! newest one.
//!
//! Run (macOS only; camera / screen recording are TCC permission-gated):
//!
//! ```sh
//! cargo run --release -p g2g-plugins --features avfoundation,screencapture,metal-sink \
//!     --example capture_on_screen              # the default camera
//! cargo run --release -p g2g-plugins --features avfoundation,screencapture,metal-sink \
//!     --example capture_on_screen -- screen    # the main display
//! ```
//!
//! Close the window (or Esc) to quit.

fn main() {
    demo::run();
}

#[cfg(all(
    target_os = "macos",
    feature = "avfoundation",
    feature = "screencapture",
    feature = "metal-sink"
))]
mod demo {
    use core::ptr::NonNull;
    use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
    use std::time::Instant;

    use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
    use g2g_core::memory::{MemoryDomain, OwnedCvPixelBuffer};
    use g2g_core::runtime::{block_on, SourceLoop};
    use g2g_core::{AsyncElement, Caps, Dim, G2gError, OutputSink, PushOutcome};
    use g2g_plugins::avf::AvfVideoSrc;
    use g2g_plugins::metalvideosink::MetalVideoSink;
    use g2g_plugins::sck::ScreenCaptureSrc;
    use objc2::msg_send;
    use objc2::runtime::AnyObject;
    use objc2_quartz_core::CAMetalLayer;
    use winit::application::ApplicationHandler;
    use winit::event::{ElementState, KeyEvent, WindowEvent};
    use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
    use winit::keyboard::{Key, NamedKey};
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use winit::window::{Window, WindowId};

    /// Hands captured pixel buffers to the window thread, dropping a frame
    /// when the window falls behind (live capture never blocks on present).
    struct ChannelSink {
        tx: SyncSender<OwnedCvPixelBuffer>,
        dropped: u64,
    }

    impl OutputSink for ChannelSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");
            if let PipelinePacket::DataFrame(f) = packet {
                let MemoryDomain::CvPixelBuffer(buf) = f.domain else {
                    panic!("cv-output emits CvPixelBuffer frames");
                };
                match self.tx.try_send(buf) {
                    Ok(()) => {}
                    Err(TrySendError::Full(_)) => self.dropped += 1,
                    // The window closed: stop capturing.
                    Err(TrySendError::Disconnected(_)) => {
                        return core::task::Poll::Ready(Err(G2gError::Shutdown));
                    }
                }
            }
            core::task::Poll::Ready(Ok(PushOutcome::Accepted))
        }
    }

    /// Discards output (the present sink is the pipeline tail).
    struct NullSink;

    impl OutputSink for NullSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            packet_slot.take();
            core::task::Poll::Ready(Ok(PushOutcome::Accepted))
        }
    }

    /// Negotiate and run `src` on a capture thread; returns its caps (or the
    /// probe error) once configured, while frames flow into `tx`.
    fn spawn_capture<S>(mut src: S, tx: SyncSender<OwnedCvPixelBuffer>) -> Result<Caps, G2gError>
    where
        S: SourceLoop + Send + 'static,
    {
        let (caps_tx, caps_rx) = sync_channel(1);
        std::thread::spawn(move || {
            let caps = block_on(src.intercept_caps())
                .and_then(|caps| src.configure_pipeline(&caps).map(|_| caps));
            let ok = caps.is_ok();
            caps_tx.send(caps).ok();
            if !ok {
                return;
            }
            let mut out = ChannelSink { tx, dropped: 0 };
            let result = block_on(src.run(&mut out));
            eprintln!(
                "capture ended: {result:?}, {} frames dropped behind the window",
                out.dropped
            );
        });
        caps_rx.recv().expect("capture thread reports its caps")
    }

    struct App {
        rx: Receiver<OwnedCvPixelBuffer>,
        caps: Caps,
        width: u32,
        height: u32,
        title: &'static str,
        window: Option<Window>,
        sink: Option<MetalVideoSink>,
        received: u64,
        started: Instant,
    }

    impl App {
        /// Present the newest captured buffer, if one arrived since the last
        /// redraw.
        fn present_latest(&mut self) {
            let Some(sink) = self.sink.as_mut() else {
                return;
            };
            let mut latest = None;
            while let Ok(buf) = self.rx.try_recv() {
                self.received += 1;
                latest = Some(buf);
            }
            let Some(buf) = latest else {
                return;
            };
            let frame = Frame::new(
                MemoryDomain::CvPixelBuffer(buf),
                FrameTiming::default(),
                self.received,
            );
            block_on(sink.process(PipelinePacket::DataFrame(frame), &mut NullSink))
                .expect("present");
            let n = sink.presented();
            if n % 120 == 0 {
                let secs = self.started.elapsed().as_secs_f64();
                eprintln!(
                    "  presented {n} frames, received {} ({:.1} fps)",
                    self.received,
                    self.received as f64 / secs
                );
            }
        }
    }

    impl ApplicationHandler for App {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if self.window.is_some() {
                return;
            }
            let attrs = Window::default_attributes()
                .with_title(self.title)
                .with_inner_size(winit::dpi::PhysicalSize::new(self.width, self.height));
            let window = event_loop.create_window(attrs).expect("create window");
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
            sink.configure_pipeline(&self.caps).expect("sink configure");
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
                    self.present_latest();
                    if let Some(window) = self.window.as_ref() {
                        window.request_redraw();
                    }
                }
                _ => {}
            }
        }
    }

    pub(crate) fn run() {
        if !MetalVideoSink::device_available() {
            eprintln!("no Metal device; nothing to present on.");
            return;
        }
        let screen = std::env::args().nth(1).as_deref() == Some("screen");
        // A small queue: the window presents the newest frame and the rest
        // are dropped, so latency stays at about one frame.
        let (tx, rx) = sync_channel(2);
        let (caps, title) = if screen {
            let src = ScreenCaptureSrc::new(u64::MAX).with_cv_output();
            (
                spawn_capture(src, tx),
                "g2g: ScreenCaptureSrc -> MetalVideoSink",
            )
        } else {
            let src = AvfVideoSrc::new(u64::MAX).with_cv_output();
            (spawn_capture(src, tx), "g2g: AvfVideoSrc -> MetalVideoSink")
        };
        let caps = match caps {
            Ok(caps) => caps,
            Err(e) => {
                eprintln!("capture unavailable (no device or permission): {e:?}");
                return;
            }
        };
        let Caps::RawVideo {
            width: Dim::Fixed(width),
            height: Dim::Fixed(height),
            ..
        } = caps
        else {
            panic!("capture caps must be fixed raw video, got {caps:?}");
        };
        println!("capturing {width}x{height} (cv-output, zero-copy)");

        let event_loop = EventLoop::new().expect("create event loop");
        event_loop.set_control_flow(ControlFlow::Poll);
        let mut app = App {
            rx,
            caps,
            width,
            height,
            title,
            window: None,
            sink: None,
            received: 0,
            started: Instant::now(),
        };
        if let Err(e) = event_loop.run_app(&mut app) {
            eprintln!("event loop ended: {e}");
        }
        let shown = app.sink.as_ref().map(|s| s.presented()).unwrap_or(0);
        println!(
            "presented {shown} of {} captured frames; bye.",
            app.received
        );
    }
}

#[cfg(not(all(
    target_os = "macos",
    feature = "avfoundation",
    feature = "screencapture",
    feature = "metal-sink"
)))]
mod demo {
    pub(crate) fn run() {
        eprintln!("needs macOS with the avfoundation, screencapture and metal-sink features");
    }
}
