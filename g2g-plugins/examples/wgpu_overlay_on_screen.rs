//! Turnkey windowed runner for [`WgpuSink`] (M978): a synthetic source, synthetic
//! analytics metadata, the Vello GPU overlay and the sink presenting into a real
//! winit window. No media file, no codec, no capture device: run it and watch the
//! metadata overlay path draw live.
//!
//! `VideoTestSrc` runs on its own thread and pushes frames through a bounded
//! link, so the source is paced by the display rather than free-running. Each
//! redraw takes the next frame, attaches an [`AnalyticsMeta`] whose boxes sweep
//! across the picture (the stand-in for a detector on a second branch), and runs
//! `VelloAnalyticsOverlay -> WgpuSink`: the overlay strokes the boxes into a
//! `wgpu::Texture` and the sink blits that texture onto the window's swapchain,
//! with no GPU->CPU readback.
//!
//! Window + event loop ownership belongs to the application (a wgpu surface is
//! built from a window handle and driven by the app's event loop), which is why
//! this is an example, not a self-checking test. The headless sibling that does
//! assert is the `m214_gpu_fanout` test.
//!
//! Run (needs a GPU and a display):
//!
//! ```sh
//! cargo run --release -p g2g-plugins --features vello-overlay,wgpu-sink \
//!     --example wgpu_overlay_on_screen
//! ```
//!
//! Close the window (or Esc) to quit.

use core::future::Future;
use core::pin::Pin;
use std::sync::Arc;

use g2g_core::runtime::{block_on, link, LinkReceiver, SenderSink, SourceLoop};
use g2g_core::{
    AnalyticsMeta, AsyncElement, BBox, Caps, Dim, G2gError, Interlace, ObjectDetection, OutputSink,
    PipelinePacket, PushOutcome, Rate, RawVideoFormat,
};
use g2g_plugins::gpu::GpuContext;
use g2g_plugins::vellooverlay::VelloAnalyticsOverlay;
use g2g_plugins::videotestsrc::{Pattern, VideoTestSrc};
use g2g_plugins::wgpusink::WgpuSink;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

const WIDTH: u32 = 960;
const HEIGHT: u32 = 540;
const FRAMERATE: u32 = 60;

/// Frames the source may run ahead of the display. Two keeps the picture on
/// screen close to the one being generated.
const LINK_CAPACITY: usize = 2;

/// The generated picture: RGBA8 in system memory, what the overlay draws on.
fn video_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Fixed(FRAMERATE << 16),
        interlace: Interlace::Any,
    }
}

fn main() {
    println!("videotestsrc -> analytics overlay (Vello) -> WgpuSink; Esc or close to quit.");

    // The source runs on its own thread: its `run` loop only returns at EOS, and
    // the window needs the main thread. The link's capacity applies the
    // backpressure that paces it to the display.
    let (sender, frames) = link(LINK_CAPACITY);
    std::thread::spawn(move || {
        let mut source =
            VideoTestSrc::new(WIDTH, HEIGHT, FRAMERATE, u64::MAX).with_pattern(Pattern::Ball);
        source
            .configure_pipeline(&video_caps())
            .expect("source configure");
        let mut out = SenderSink::new(sender);
        // Ends with a send error once the window is gone and the receiver drops.
        let _ = block_on(source.run(&mut out));
    });

    let event_loop = EventLoop::new().expect("create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        frames,
        window: None,
        present: None,
        seq: 0,
    };
    if let Err(e) = event_loop.run_app(&mut app) {
        // A compositor dropping the connection (e.g. window closed abruptly) is
        // not a pipeline failure; report it without aborting.
        eprintln!("event loop ended: {e}");
    }
    let shown = app.present.as_ref().map(|p| p.sink.presented_count());
    println!("presented {} frames; bye.", shown.unwrap_or(0));
}

/// The overlay and the sink, both on the window's GPU context so the overlay's
/// texture presents with no copy.
struct Present {
    overlay: VelloAnalyticsOverlay,
    sink: WgpuSink,
}

struct App {
    frames: LinkReceiver,
    // Created once the event loop resumes and a window exists.
    window: Option<Arc<Window>>,
    present: Option<Present>,
    /// Frames presented so far, which animates the synthetic detections.
    seq: u64,
}

impl App {
    /// Build the overlay -> sink pair once a real (non-zero) window size is known.
    /// A no-op if already built or the size is still zero.
    fn ensure_present(&mut self, width: u32, height: u32) {
        if self.present.is_some() || width == 0 || height == 0 {
            return;
        }
        let Some(window) = self.window.clone() else {
            return;
        };

        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window.clone())
            .expect("create surface for the window");
        let ctx = block_on(GpuContext::for_surface(instance, &surface))
            .expect("no GPU on this host can present to the window");
        let config = surface_config(&surface, &ctx.adapter, width, height);
        surface.configure(&ctx.device, &config);
        println!("presenting on {}", ctx.adapter.get_info().name);

        let mut overlay = VelloAnalyticsOverlay::new()
            .with_context(ctx.clone())
            .with_thickness(4.0);
        overlay
            .configure_pipeline(&video_caps())
            .expect("overlay configure");
        let mut sink = WgpuSink::with_surface(ctx, surface, config);
        sink.configure_pipeline(&video_caps())
            .expect("sink configure");

        self.present = Some(Present { overlay, sink });
        window.request_redraw();
    }

    /// Draw the next generated frame: attach the synthetic detections, then run
    /// it through the overlay into the sink. Does nothing if the source has not
    /// produced the next frame yet (the next redraw picks it up).
    fn present_next(&mut self) {
        let Some(present) = self.present.as_mut() else {
            return;
        };
        let mut frame = loop {
            match self.frames.try_recv() {
                Some(PipelinePacket::DataFrame(frame)) => break frame,
                // The source emits nothing else, but a control packet must not
                // be mistaken for "no frame yet".
                Some(_) => continue,
                None => return,
            }
        };

        let mut analytics = AnalyticsMeta::new();
        for detection in sweeping_detections(self.seq) {
            analytics.add_detection(detection);
        }
        frame.meta.attach(analytics);
        self.seq += 1;

        block_on(present.overlay.process(
            PipelinePacket::DataFrame(frame),
            &mut PresentSink(&mut present.sink),
        ))
        .expect("overlay and present");
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("g2g: analytics overlay -> WgpuSink")
            .with_inner_size(winit::dpi::PhysicalSize::new(WIDTH, HEIGHT));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        let size = window.inner_size();
        self.window = Some(window);
        // Build now if the compositor already gave us a real size, else wait for
        // the first Resized. Only ONE surface is ever created per window: creating
        // a second on the same Wayland surface is a protocol error.
        self.ensure_present(size.width, size.height);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
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
            WindowEvent::Resized(new) => {
                // Build once, the first time we learn a non-zero size (Wayland
                // often reports 0x0 at creation), then keep the swapchain on the
                // window's size (a no-op for the build-time size).
                self.ensure_present(new.width, new.height);
                if let Some(present) = self.present.as_mut() {
                    present.sink.resize(new.width, new.height);
                }
            }
            WindowEvent::RedrawRequested => {
                self.present_next();
                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

/// Two detections sweeping over the frame, the metadata a detector branch would
/// attach. Normalized `[0, 1]` boxes, so they track any frame size; the labels
/// differ so the overlay draws them in two palette colours.
fn sweeping_detections(seq: u64) -> [ObjectDetection; 2] {
    let t = seq as f32 / FRAMERATE as f32;
    let sweep = |phase: f32, span: f32, size: f32| 0.5 + span * (t + phase).sin() - size / 2.0;
    [
        ObjectDetection {
            bbox: BBox {
                x: sweep(0.0, 0.34, 0.24),
                y: sweep(1.7, 0.26, 0.30),
                w: 0.24,
                h: 0.30,
            },
            label: 0,
            confidence: 0.93,
        },
        ObjectDetection {
            bbox: BBox {
                x: sweep(3.1, 0.30, 0.16),
                y: sweep(0.6, 0.30, 0.16),
                w: 0.16,
                h: 0.16,
            },
            label: 2,
            confidence: 0.71,
        },
    ]
}

/// A surface config at `width` x `height`, preferring a plain (non-sRGB) format
/// so the overlay's RGBA presents without a regamma.
fn surface_config(
    surface: &wgpu::Surface<'static>,
    adapter: &wgpu::Adapter,
    width: u32,
    height: u32,
) -> wgpu::SurfaceConfiguration {
    let caps = surface.get_capabilities(adapter);
    let format = caps
        .formats
        .iter()
        .copied()
        .find(|f| !f.is_srgb())
        .unwrap_or(caps.formats[0]);
    wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: width.max(1),
        height: height.max(1),
        present_mode: wgpu::PresentMode::AutoVsync,
        alpha_mode: caps.alpha_modes[0],
        view_formats: Vec::new(),
        desired_maximum_frame_latency: 2,
    }
}

/// Feeds the overlay's GPU frames into the sink.
struct PresentSink<'s>(&'s mut WgpuSink);

impl OutputSink for PresentSink<'_> {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            self.0.process(packet, &mut NullSink).await?;
            Ok(PushOutcome::Accepted)
        })
    }
}

/// A discarding sink for the terminal `WgpuSink` (it forwards nothing).
struct NullSink;
impl OutputSink for NullSink {
    fn push<'a>(
        &'a mut self,
        _packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async { Ok(PushOutcome::Accepted) })
    }
}
