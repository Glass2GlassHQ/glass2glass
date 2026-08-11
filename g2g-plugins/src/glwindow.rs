//! EGL + GL ES renderer for the GL display sinks, over the shared Wayland window
//! worker ([`crate::waylandwindow`]).
//!
//! Brings up an EGL display + GL ES 3 context on a `wl_egl_window` over the
//! worker's `wl_surface`, builds the [`GlState`] for the negotiated layout, and
//! presents each frame with `eglSwapBuffers`. The sink supplies a
//! [`FramePresenter`] for the one per-sink step (a CUDA device->texture copy for
//! [`crate::cudaglsink`], a `glTexSubImage2D` for [`crate::glsink`]).

use alloc::boxed::Box;

use khronos_egl as egl;
use smithay_client_toolkit::reexports::client::{protocol::wl_surface, Connection, Proxy};
use wayland_egl::WlEglSurface;

use crate::glnv12::{GlMode, GlState};
use crate::waylandwindow::{run_window, WindowParams, WindowRenderer, WorkerChannels};
use g2g_core::{G2gError, HardwareError};

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

/// Open the GL window and run the shared present loop until the sink sends
/// `Shutdown` (or the compositor closes the window). Runs on the sink's worker
/// thread and returns when the loop exits.
pub(crate) fn run_gl_window<P: FramePresenter>(
    presenter: P,
    params: WindowParams,
    channels: WorkerChannels<P::Frame>,
) -> Result<(), Box<dyn std::error::Error>> {
    run_window(
        |conn, wl_surface, width, height| {
            let (egl, gl) = EglWindow::new(conn, wl_surface, width, height)?;
            // SAFETY: `gl` wraps the GL ES 3 context `EglWindow::new` made
            // current on this thread.
            let state = unsafe { GlState::build(gl, width, height, presenter.mode()) }?;
            Ok(GlRenderer {
                gl: state,
                egl,
                presenter,
            })
        },
        params,
        channels,
    )
}

/// The GL half of the worker: the sink's upload + draw, then `eglSwapBuffers`.
/// `gl` is declared before `egl` so the GL state is dropped while its context is
/// still alive.
struct GlRenderer<P: FramePresenter> {
    gl: GlState,
    egl: EglWindow,
    presenter: P,
}

impl<P: FramePresenter> WindowRenderer for GlRenderer<P> {
    type Frame = P::Frame;

    fn present(&mut self, frame: &Self::Frame) -> Result<(), G2gError> {
        self.presenter.present(&mut self.gl, frame)?;
        self.egl.swap()
    }
}

// =================================================================
// EGL on a Wayland surface
// =================================================================

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
