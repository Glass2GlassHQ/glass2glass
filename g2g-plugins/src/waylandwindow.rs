//! Shared Wayland window worker for the display sinks that own their window.
//!
//! Wayland (and the graphics API drawn on it) is single-thread-affine, so each
//! such sink runs its window on a dedicated worker thread. Everything that thread
//! does apart from getting one frame's pixels onto the surface is the same for
//! every sink: connect to the compositor, map an `xdg_toplevel`, build the
//! renderer over that surface, then loop on a `calloop` channel drawing each
//! handed-over frame and presenting it. That is [`run_window`]; the sink supplies
//! a [`WindowRenderer`] for the graphics half (EGL + GL ES for
//! [`crate::glwindow`], a `wgpu::Surface` for [`crate::wgpupresent`]).
//!
//! The first `xdg` configure signals the sink's readiness handshake, and a frame
//! that arrives before the surface is mappable is held and drawn then.

use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use alloc::boxed::Box;
use alloc::string::String;

use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState},
    delegate_compositor, delegate_output, delegate_registry, delegate_xdg_shell,
    delegate_xdg_window,
    output::{OutputHandler, OutputState},
    reexports::calloop::{
        channel::{Channel, Event as ChanEvent},
        EventLoop,
    },
    reexports::calloop_wayland_source::WaylandSource,
    reexports::client::{
        globals::registry_queue_init,
        protocol::{wl_output, wl_surface},
        Connection, QueueHandle,
    },
    registry::{ProvidesRegistryState, RegistryState},
    registry_handlers,
    shell::{
        xdg::{
            window::{Window, WindowConfigure, WindowDecorations, WindowHandler},
            XdgShell,
        },
        WaylandSurface,
    },
};

use crate::worker_ready::Handshake;
use g2g_core::log::Target;
use g2g_core::metrics::{monotonic_ns, LatencyHistogram};
#[cfg(feature = "wgpu-present")]
use g2g_core::Caps;
use g2g_core::G2gError;

pub(crate) enum WorkerCmd<F> {
    Frame {
        frame: F,
        arrival_ns: u64,
        ack: tokio::sync::oneshot::Sender<()>,
    },
    /// New negotiated caps for the renderer, plus the size to ask the compositor
    /// for, so a mid-stream format / geometry change costs a reconfigure instead
    /// of tearing the window and its GPU device down and building them again.
    /// Only the wgpu sink follows caps in place; the GL sinks respawn.
    #[cfg(feature = "wgpu-present")]
    Reconfigure {
        caps: Caps,
        width: u32,
        height: u32,
    },
    Shutdown,
}

/// The per-sink half of the worker: how one frame's pixels reach the window's
/// surface. The window, the event loop, the readiness handshake and the pacing
/// ack are shared.
pub(crate) trait WindowRenderer: 'static {
    /// Payload the sink hands over the channel (a byte vector, a GPU frame).
    type Frame: Send + 'static;

    /// Draw `frame` and present it to the surface.
    fn present(&mut self, frame: &Self::Frame) -> Result<(), G2gError>;

    /// Take new negotiated caps mid-stream. Default: nothing to do, for a
    /// renderer whose sink respawns the worker on a caps change instead.
    #[cfg(feature = "wgpu-present")]
    fn reconfigure(&mut self, _caps: &Caps) -> Result<(), G2gError> {
        Ok(())
    }

    /// Follow the compositor's configured window size. Default: ignore it and
    /// keep drawing at the size the renderer was built for, letting the
    /// compositor letterbox.
    fn resize(&mut self, _width: u32, _height: u32) {}
}

/// Window geometry + identity the worker opens with.
pub(crate) struct WindowParams {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) title: String,
    pub(crate) app_id: String,
    /// Map the window fullscreen rather than as a normal toplevel.
    pub(crate) fullscreen: bool,
    /// Log category the worker's error lines go under (the sink's element name).
    pub(crate) log_tag: &'static str,
}

/// The handles the sink shares with its worker: the frame channel plus the
/// counters and readiness handshake the sink reads.
pub(crate) struct WorkerChannels<F> {
    pub(crate) rx: Channel<WorkerCmd<F>>,
    pub(crate) presented: Arc<AtomicU64>,
    pub(crate) latency: Arc<LatencyHistogram>,
    pub(crate) ready: Arc<Handshake>,
}

/// Open the Wayland window, build the sink's renderer over its surface, and run
/// the present loop until the sink sends `Shutdown` (or the compositor closes the
/// window). `build_renderer` is called once the surface exists, on this thread,
/// because a graphics context is bound to the thread that made it. Runs on the
/// sink's worker thread and returns when the loop exits.
pub(crate) fn run_window<R: WindowRenderer>(
    build_renderer: impl FnOnce(
        &Connection,
        &wl_surface::WlSurface,
        u32,
        u32,
    ) -> Result<R, Box<dyn std::error::Error>>,
    params: WindowParams,
    channels: WorkerChannels<R::Frame>,
) -> Result<(), Box<dyn std::error::Error>> {
    let WindowParams {
        width,
        height,
        title,
        app_id,
        fullscreen,
        log_tag,
    } = params;
    let conn = Connection::connect_to_env()?;
    let (globals, event_queue) = registry_queue_init(&conn)?;
    let qh = event_queue.handle();

    let mut event_loop: EventLoop<WindowWorker<R>> = EventLoop::try_new()?;
    let loop_handle = event_loop.handle();
    WaylandSource::new(conn.clone(), event_queue).insert(loop_handle.clone())?;

    let compositor = CompositorState::bind(&globals, &qh)?;
    let xdg_shell = XdgShell::bind(&globals, &qh)?;

    let surface = compositor.create_surface(&qh);
    let window = xdg_shell.create_window(surface, WindowDecorations::RequestServer, &qh);
    window.set_title(&title);
    window.set_app_id(&app_id);
    window.set_min_size(Some((width, height)));
    if fullscreen {
        window.set_fullscreen(None);
    }
    window.commit();

    let renderer = build_renderer(&conn, window.wl_surface(), width, height)?;

    let mut state = WindowWorker {
        registry_state: RegistryState::new(&globals),
        output_state: OutputState::new(&globals, &qh),
        renderer,
        window,
        qh: qh.clone(),
        log_tag,
        configured: false,
        exit: false,
        ready: Some(channels.ready),
        presented: channels.presented,
        latency: channels.latency,
        pending: None,
    };

    loop_handle.insert_source(
        channels.rx,
        |event, _, state: &mut WindowWorker<R>| match event {
            ChanEvent::Msg(WorkerCmd::Frame {
                frame,
                arrival_ns,
                ack,
            }) => {
                if state.configured {
                    state.draw(frame, arrival_ns, ack);
                } else {
                    state.pending = Some((frame, arrival_ns, ack));
                }
            }
            #[cfg(feature = "wgpu-present")]
            ChanEvent::Msg(WorkerCmd::Reconfigure {
                caps,
                width,
                height,
            }) => state.reconfigure(&caps, width, height),
            ChanEvent::Msg(WorkerCmd::Shutdown) | ChanEvent::Closed => {
                state.exit = true;
            }
        },
    )?;

    while !state.exit {
        event_loop.dispatch(Some(Duration::from_millis(100)), &mut state)?;
    }
    Ok(())
}

// =================================================================
// Worker state + SCTK handlers
// =================================================================

struct WindowWorker<R: WindowRenderer> {
    registry_state: RegistryState,
    output_state: OutputState,
    /// Declared before `window`: the renderer's surface is built over the
    /// `wl_surface` and has to be torn down before it.
    renderer: R,
    window: Window,
    qh: QueueHandle<WindowWorker<R>>,
    log_tag: &'static str,
    configured: bool,
    exit: bool,
    ready: Option<Arc<Handshake>>,
    presented: Arc<AtomicU64>,
    latency: Arc<LatencyHistogram>,
    /// Frame that arrived before the surface was mappable.
    pending: Option<(R::Frame, u64, tokio::sync::oneshot::Sender<()>)>,
}

impl<R: WindowRenderer> WindowWorker<R> {
    /// Draw one frame and present it. Signals `ack` after the present returns
    /// (compositor-paced backpressure).
    fn draw(&mut self, frame: R::Frame, arrival_ns: u64, ack: tokio::sync::oneshot::Sender<()>) {
        let drawn = self.draw_inner(&frame);
        // Let the frame go before acking. The ack is what lets the pipeline move
        // on, and at end of stream moving on means tearing the producers down: a
        // GPU frame whose memory belongs to one of them (a decoder's CUDA
        // context) must be released while that producer is still alive.
        drop(frame);
        if let Err(e) = drawn {
            g2g_core::g2g_error!(Target::category(self.log_tag), "cannot draw frame: {e:?}");
            // Release the producer so a transient GPU error doesn't deadlock
            // the pipeline; the frame just didn't paint.
            let _ = ack.send(());
            return;
        }
        self.presented.fetch_add(1, Ordering::Relaxed);
        if arrival_ns != 0 {
            let now = monotonic_ns();
            if now >= arrival_ns {
                self.latency.record(now - arrival_ns);
            }
        }
        let _ = ack.send(());
    }

    #[cfg(feature = "wgpu-present")]
    /// Point the renderer at the new caps and ask the compositor for the size
    /// that goes with them. A renderer that cannot take them leaves the window
    /// showing a stale layout, so the worker exits instead and the pipeline fails
    /// on the next frame.
    fn reconfigure(&mut self, caps: &Caps, width: u32, height: u32) {
        if let Err(e) = self.renderer.reconfigure(caps) {
            g2g_core::g2g_error!(
                Target::category(self.log_tag),
                "cannot apply the new caps to the window: {e:?}"
            );
            self.exit = true;
            return;
        }
        self.window.set_min_size(Some((width, height)));
        self.window.commit();
    }

    fn draw_inner(&mut self, frame: &R::Frame) -> Result<(), G2gError> {
        // Subscribe to the next frame callback (compositor pacing) before the
        // present commits the surface.
        let surface = self.window.wl_surface().clone();
        surface.frame(&self.qh, surface.clone());
        self.renderer.present(frame)
    }
}

impl<R: WindowRenderer> CompositorHandler for WindowWorker<R> {
    fn scale_factor_changed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: i32,
    ) {
    }
    fn transform_changed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: wl_output::Transform,
    ) {
    }
    fn frame(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_surface::WlSurface, _: u32) {
        // Compositor released the buffer; pacing is handled by the per-frame
        // ack in `draw`, so nothing extra is needed here.
    }
    fn surface_enter(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: &wl_output::WlOutput,
    ) {
    }
    fn surface_leave(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: &wl_output::WlOutput,
    ) {
    }
}

impl<R: WindowRenderer> WindowHandler for WindowWorker<R> {
    fn request_close(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &Window) {
        self.exit = true;
    }

    fn configure(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &Window,
        configure: WindowConfigure,
        _serial: u32,
    ) {
        if let (Some(width), Some(height)) = configure.new_size {
            self.renderer.resize(width.get(), height.get());
        }
        let was_first = !self.configured;
        self.configured = true;
        if was_first {
            if let Some(ready) = self.ready.take() {
                ready.notify();
            }
            if let Some((frame, arrival_ns, ack)) = self.pending.take() {
                self.draw(frame, arrival_ns, ack);
            }
        }
    }
}

impl<R: WindowRenderer> OutputHandler for WindowWorker<R> {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }
    fn new_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
    fn update_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
    fn output_destroyed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
}

impl<R: WindowRenderer> ProvidesRegistryState for WindowWorker<R> {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    registry_handlers![OutputState,];
}

delegate_compositor!(@<R: WindowRenderer> WindowWorker<R>);
delegate_output!(@<R: WindowRenderer> WindowWorker<R>);
delegate_xdg_shell!(@<R: WindowRenderer> WindowWorker<R>);
delegate_xdg_window!(@<R: WindowRenderer> WindowWorker<R>);
delegate_registry!(@<R: WindowRenderer> WindowWorker<R>);
