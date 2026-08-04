//! Shared Wayland + EGL worker for the GL display sinks.
//!
//! GL and Wayland are both single-thread-affine, so each GL sink runs its window
//! on a dedicated worker thread. Everything that thread does apart from getting
//! the pixels into the textures is the same for every sink: connect to the
//! compositor, map an `xdg_toplevel`, bring up an EGL display + GL ES 3 context
//! on a `wl_egl_window`, build the [`GlState`], then loop on a `calloop` channel
//! drawing each handed-over frame and `eglSwapBuffers`-ing it. That is
//! [`run_gl_window`]; the sink supplies a [`FramePresenter`] for the one
//! per-sink step (a CUDA device->texture copy for [`crate::cudaglsink`], a
//! `glTexSubImage2D` for [`crate::glsink`]).
//!
//! The first `xdg` configure signals the sink's readiness handshake, and a frame
//! that arrives before the surface is mappable is held and drawn then.

use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use alloc::boxed::Box;
use alloc::string::String;

use khronos_egl as egl;
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
        Connection, Proxy, QueueHandle,
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
use wayland_egl::WlEglSurface;

use crate::glnv12::{GlMode, GlState};
use crate::worker_ready::Handshake;
use g2g_core::metrics::{monotonic_ns, LatencyHistogram};
use g2g_core::{G2gError, HardwareError};

/// Worker-thread command. `Frame` carries the sink's frame payload plus the
/// source-side `arrival_ns` for latency and a one-shot `ack` the worker signals
/// once the frame is presented.
pub(crate) enum WorkerCmd<F> {
    Frame {
        frame: F,
        arrival_ns: u64,
        ack: tokio::sync::oneshot::Sender<()>,
    },
    Shutdown,
}

/// The per-sink half of the worker: which pixel layout to build the GL state
/// for, and how to get one frame's pixels into it. The draw itself (program,
/// textures, quad) and the present are shared.
pub(crate) trait FramePresenter: 'static {
    /// Payload the sink hands over the channel (a CUDA buffer, a byte vector).
    type Frame: Send + 'static;

    /// Pixel layout the [`GlState`] is built for, from the negotiated caps.
    fn mode(&self) -> GlMode;

    /// Upload `frame` into `gl`'s textures and draw it. The worker presents.
    fn present(&mut self, gl: &mut GlState, frame: &Self::Frame) -> Result<(), G2gError>;
}

/// Window geometry + identity the worker opens with.
pub(crate) struct WindowParams {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) title: String,
    pub(crate) app_id: String,
    /// Prefix for the worker's error lines (the sink's element name).
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

/// Open the Wayland window + EGL context, build the GL state, and run the
/// present loop until the sink sends `Shutdown` (or the compositor closes the
/// window). Runs on the sink's worker thread and returns when the loop exits.
pub(crate) fn run_gl_window<P: FramePresenter>(
    presenter: P,
    params: WindowParams,
    channels: WorkerChannels<P::Frame>,
) -> Result<(), Box<dyn std::error::Error>> {
    let WindowParams {
        width,
        height,
        title,
        app_id,
        log_tag,
    } = params;
    let conn = Connection::connect_to_env()?;
    let (globals, event_queue) = registry_queue_init(&conn)?;
    let qh = event_queue.handle();

    let mut event_loop: EventLoop<WindowWorker<P>> = EventLoop::try_new()?;
    let loop_handle = event_loop.handle();
    WaylandSource::new(conn.clone(), event_queue).insert(loop_handle.clone())?;

    let compositor = CompositorState::bind(&globals, &qh)?;
    let xdg_shell = XdgShell::bind(&globals, &qh)?;

    let surface = compositor.create_surface(&qh);
    let window = xdg_shell.create_window(surface, WindowDecorations::RequestServer, &qh);
    window.set_title(&title);
    window.set_app_id(&app_id);
    window.set_min_size(Some((width, height)));
    window.commit();

    let (egl_window, gl) = EglWindow::new(&conn, window.wl_surface(), width, height)?;
    // SAFETY: `gl` wraps the GL ES 3 context `EglWindow::new` made current on
    // this thread.
    let gl_state = unsafe { GlState::build(gl, width, height, presenter.mode()) }?;

    let mut state = WindowWorker {
        registry_state: RegistryState::new(&globals),
        output_state: OutputState::new(&globals, &qh),
        window,
        qh: qh.clone(),
        egl: egl_window,
        gl: gl_state,
        presenter,
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
        |event, _, state: &mut WindowWorker<P>| match event {
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
// EGL on a Wayland surface
// =================================================================

/// EGL display + context + window surface over a `wl_surface`. Kept alive for
/// the worker's lifetime: the `WlEglSurface` must outlive the EGL surface, which
/// must outlive the `wl_surface`.
struct EglWindow {
    egl: egl::Instance<egl::Static>,
    display: egl::Display,
    surface: egl::Surface,
    context: egl::Context,
    _wl_egl: WlEglSurface,
}

impl EglWindow {
    /// Bring up EGL on `wl_surface` at `width` x `height`, make the GL ES 3
    /// context current on this thread, and load the GL entry points.
    fn new(
        conn: &Connection,
        wl_surface: &wl_surface::WlSurface,
        width: u32,
        height: u32,
    ) -> Result<(Self, glow::Context), Box<dyn std::error::Error>> {
        let egl = egl::Instance::new(egl::Static);

        // The wl_display raw pointer on wayland-client 0.31. The display is a
        // special global; `backend().display_ptr()` returns the libwayland
        // `*mut wl_display` EGL wants as its native display handle.
        let display_ptr = conn.backend().display_ptr() as *mut core::ffi::c_void;
        // SAFETY: `display_ptr` is the live connection's libwayland `*mut wl_display`,
        // valid for the worker thread's lifetime (the `Connection` outlives the EGL
        // display via the caller's `conn`); `get_display` only records the handle.
        let display = unsafe { egl.get_display(display_ptr) }.ok_or("eglGetDisplay failed")?;
        egl.initialize(display)?;
        egl.bind_api(egl::OPENGL_ES_API)?;

        let config_attribs = [
            egl::SURFACE_TYPE,
            egl::WINDOW_BIT,
            egl::RENDERABLE_TYPE,
            egl::OPENGL_ES3_BIT, // GLES 3 for R8/RG8 single/two-channel textures
            egl::RED_SIZE,
            8,
            egl::GREEN_SIZE,
            8,
            egl::BLUE_SIZE,
            8,
            egl::NONE,
        ];
        let config = egl
            .choose_first_config(display, &config_attribs)?
            .ok_or("no matching EGL config")?;

        let context_attribs = [egl::CONTEXT_MAJOR_VERSION, 3, egl::NONE];
        let context = egl.create_context(display, config, None, &context_attribs)?;

        // wl_egl_window from the SCTK surface; EGL window surface on top of it.
        let wl_egl = WlEglSurface::new(wl_surface.id(), width as i32, height as i32)?;
        // SAFETY: `wl_egl.ptr()` is a live `wl_egl_window` for this display/config;
        // `wl_egl` is moved into the returned struct, so it outlives the surface.
        let surface = unsafe {
            egl.create_window_surface(
                display,
                config,
                wl_egl.ptr() as *mut core::ffi::c_void,
                None,
            )
        }?;
        egl.make_current(display, Some(surface), Some(surface), Some(context))?;

        // glow loads GL ES entry points through eglGetProcAddress.
        // SAFETY: `egl.get_proc_address` resolves GL ES symbols against the context
        // just made current; glow only invokes the returned pointers as the GL
        // entry points whose names it passed.
        let gl = unsafe {
            glow::Context::from_loader_function(|s| match egl.get_proc_address(s) {
                Some(p) => p as *const core::ffi::c_void,
                None => core::ptr::null(),
            })
        };

        Ok((
            Self {
                egl,
                display,
                surface,
                context,
                _wl_egl: wl_egl,
            },
            gl,
        ))
    }

    fn swap(&self) -> Result<(), G2gError> {
        self.egl
            .swap_buffers(self.display, self.surface)
            .map_err(|_| G2gError::Hardware(HardwareError::Other))
    }
}

impl Drop for EglWindow {
    fn drop(&mut self) {
        // Tear down the EGL display / context / surface the worker created. A
        // mid-stream resolution change respawns the worker, so without this each
        // resize leaks an EGL context + surface. Best-effort; the worker is
        // exiting. Releasing the current context first, then destroying the
        // surface here (this Drop runs before the `_wl_egl` field drops its
        // backing `wl_egl_window`) keeps the required outlives ordering.
        let _ = self.egl.make_current(self.display, None, None, None);
        let _ = self.egl.destroy_surface(self.display, self.surface);
        let _ = self.egl.destroy_context(self.display, self.context);
        let _ = self.egl.terminate(self.display);
    }
}

// =================================================================
// Worker state + SCTK handlers
// =================================================================

struct WindowWorker<P: FramePresenter> {
    registry_state: RegistryState,
    output_state: OutputState,
    window: Window,
    qh: QueueHandle<WindowWorker<P>>,
    egl: EglWindow,
    gl: GlState,
    presenter: P,
    log_tag: &'static str,
    configured: bool,
    exit: bool,
    ready: Option<Arc<Handshake>>,
    presented: Arc<AtomicU64>,
    latency: Arc<LatencyHistogram>,
    /// Frame that arrived before the surface was mappable.
    pending: Option<(P::Frame, u64, tokio::sync::oneshot::Sender<()>)>,
}

impl<P: FramePresenter> WindowWorker<P> {
    /// Upload + draw one frame and present it. Signals `ack` after
    /// `eglSwapBuffers` returns (compositor-paced backpressure).
    fn draw(&mut self, frame: P::Frame, arrival_ns: u64, ack: tokio::sync::oneshot::Sender<()>) {
        if let Err(e) = self.draw_inner(&frame) {
            std::eprintln!("{} draw error: {e:?}", self.log_tag);
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

    fn draw_inner(&mut self, frame: &P::Frame) -> Result<(), G2gError> {
        self.presenter.present(&mut self.gl, frame)?;
        // Subscribe to the next frame callback (compositor pacing) and present.
        let surface = self.window.wl_surface().clone();
        surface.frame(&self.qh, surface.clone());
        self.egl.swap()
    }
}

impl<P: FramePresenter> CompositorHandler for WindowWorker<P> {
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

impl<P: FramePresenter> WindowHandler for WindowWorker<P> {
    fn request_close(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &Window) {
        self.exit = true;
    }

    fn configure(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &Window,
        _configure: WindowConfigure,
        _serial: u32,
    ) {
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

impl<P: FramePresenter> OutputHandler for WindowWorker<P> {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }
    fn new_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
    fn update_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
    fn output_destroyed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
}

impl<P: FramePresenter> ProvidesRegistryState for WindowWorker<P> {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    registry_handlers![OutputState,];
}

delegate_compositor!(@<P: FramePresenter> WindowWorker<P>);
delegate_output!(@<P: FramePresenter> WindowWorker<P>);
delegate_xdg_shell!(@<P: FramePresenter> WindowWorker<P>);
delegate_xdg_window!(@<P: FramePresenter> WindowWorker<P>);
delegate_registry!(@<P: FramePresenter> WindowWorker<P>);
