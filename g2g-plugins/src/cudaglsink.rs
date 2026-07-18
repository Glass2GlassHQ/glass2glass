//! CUDA-GL display sink (C3 Phase 3, step 2): the zero-copy-ish payoff.
//!
//! Keeps `Backend::NvdecCuda` decoded NV12 resident on the GPU and presents it
//! without a PCIe round-trip or CPU colour convert. Per frame: CUDA copies the
//! two NV12 planes device->`cudaArray` into two registered GL textures
//! (`CudaGlInterop`), then a fragment shader converts NV12->RGB on the GPU and
//! presents via `eglSwapBuffers` (DESIGN-C3-cuda.md §3.2, Appendix A). Not
//! literally zero-copy (one device->device copy into the texture), but it
//! removes the device->host copy `CudaDownload` pays and the CPU convert
//! `WaylandSink` pays.
//!
//! ## Pipeline shape
//!
//! ```text
//! RtspSrc ─► H264Parse ─► FfmpegH264Dec(NvdecCuda) ─► CudaGlSink
//!                                                          │
//!                                                          └─► EGL/GL window
//! ```
//!
//! ## Threading
//!
//! GL and Wayland are both single-thread-affine, so (like [`WaylandSink`]) all
//! of it lives on a dedicated worker thread spun up at `configure_pipeline`.
//! The sink struct holds only `Send` handles (a `calloop` channel sender plus
//! shared atomics), so the runner can move it between executor tasks. The
//! decoded `OwnedCudaBuffer` is `Send` (its keep-alive owner is), so the frame
//! crosses to the worker and the device memory stays pinned until the worker
//! drops it after upload.
//!
//! ## Verification status
//!
//! `cuda-gl` + Linux + NVIDIA-gated. Validated on the RTX 3060 host (M252):
//! compiles + lints clean, and the `cudagl_smoke` on-display e2e presents real
//! `NvdecCuda` frames through the CUDA-GL path (60 frames, glass-to-glass
//! p50 ~8 ms on a GNOME Wayland session). The off-host draft needed only two
//! adjustments at first compile, the `khronos-egl` 6 `get_display` now being
//! `unsafe`, and importing `alloc::string::ToString`; the crate-API spots that
//! were flagged `// VERIFY:` (the `wayland-client` 0.31 raw-pointer accessors,
//! glow 0.17's `tex_image_2d` pixel-source parameter, the `eglGetProcAddress`
//! cast) all held. On a hybrid iGPU+NVIDIA host the GL context must be forced
//! onto the NVIDIA GPU (`__NV_PRIME_RENDER_OFFLOAD` / `__EGL_VENDOR_LIBRARY_FILENAMES`)
//! or `cuGraphicsGLRegisterImage` fails. The `cudagl_vs_wayland` A/B benchmark
//! (M253) measured this device-resident path at **10.7x lower present latency**
//! than the `NvdecCuvid -> WaylandSink` baseline at 1080p (p50 ~8 ms vs ~90 ms):
//! the GPU NV12->RGB convert replaces the baseline's per-frame CPU convert +
//! `wl_shm` upload.
//!
//! ## Constraints (v1)
//!
//! - NV12 in CUDA device memory only (`MemoryDomain::Cuda`); a system-memory
//!   frame is rejected loud (use `CudaDownload` + `WaylandSink` for that).
//! - No scaling: the window opens at the video dimensions; the compositor
//!   letterboxes/clips if it resizes us.
//! - BT.601 limited range (Appendix A shader); BT.709 awaits colour metadata.

use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
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
        channel::{channel, Channel, Event as ChanEvent, Sender as CalloopSender},
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

use crate::worker_ready::Handshake;
use g2g_core::memory::OwnedCudaBuffer;
use g2g_core::metrics::{monotonic_ns, LatencyHistogram, LatencySnapshot};
use g2g_core::{
    AllocationParams, AsyncElement, Caps, CapsConstraint, CapsSet, ClockCandidate, ClockPriority,
    ConfigureOutcome, Dim, Frame, G2gError, HardwareError, MemoryDomain, OutputSink, PipelineClock,
    PipelinePacket, Rate, RawVideoFormat,
};

use crate::cuda::nv12_byte_size;
use crate::glnv12::GlState;

/// Device-buffer pool headroom the sink asks the producer to keep resident:
/// the frame in flight on the GL thread plus the one the runner link holds, so
/// the decoder's hwframe pool does not starve under live pacing.
const CUDA_POOL_HEADROOM: usize = 3;

/// GPU upload alignment the sink requests (256 bytes is the common CUDA / NVENC
/// surface alignment).
const CUDA_ALIGN: usize = 256;

/// Worker-thread command. `Frame` carries the decoded CUDA buffer (still
/// device-resident) plus the source-side `arrival_ns` for latency and a
/// one-shot `ack` the worker signals once the frame is presented.
enum WorkerCmd {
    Frame {
        buf: OwnedCudaBuffer,
        arrival_ns: u64,
        ack: tokio::sync::oneshot::Sender<()>,
    },
    Shutdown,
}

/// Sink-side handle set. Only `Send + Sync` state lives here so the
/// multi-thread runner can move the sink between tasks.
pub struct CudaGlSink {
    title: String,
    app_id: String,
    cmd_tx: Option<CalloopSender<WorkerCmd>>,
    worker: Option<JoinHandle<()>>,
    width: u32,
    height: u32,
    frames_presented: Arc<AtomicU64>,
    latency: Arc<LatencyHistogram>,
}

impl core::fmt::Debug for CudaGlSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CudaGlSink")
            .field("title", &self.title)
            .field("width", &self.width)
            .field("height", &self.height)
            .field(
                "frames_presented",
                &self.frames_presented.load(Ordering::Relaxed),
            )
            .finish()
    }
}

impl Default for CudaGlSink {
    fn default() -> Self {
        Self::new()
    }
}

impl CudaGlSink {
    pub fn new() -> Self {
        Self {
            title: String::from("glass2glass"),
            app_id: String::from("io.glass2glass.CudaGlSink"),
            cmd_tx: None,
            worker: None,
            width: 0,
            height: 0,
            frames_presented: Arc::new(AtomicU64::new(0)),
            latency: Arc::new(LatencyHistogram::new()),
        }
    }

    pub fn with_title<S: Into<String>>(mut self, title: S) -> Self {
        self.title = title.into();
        self
    }

    pub fn with_app_id<S: Into<String>>(mut self, app_id: S) -> Self {
        self.app_id = app_id.into();
        self
    }

    pub fn frames_presented(&self) -> u64 {
        self.frames_presented.load(Ordering::Relaxed)
    }

    /// Glass-to-glass latency snapshot: source-side `arrival_ns` to the
    /// `eglSwapBuffers` that presents the frame. Untimed pipelines report
    /// `count = 0`.
    pub fn latency_snapshot(&self) -> LatencySnapshot {
        self.latency.snapshot()
    }

    fn shutdown(&mut self) {
        if let Some(tx) = self.cmd_tx.take() {
            let _ = tx.send(WorkerCmd::Shutdown);
        }
        if let Some(join) = self.worker.take() {
            let _ = join.join();
        }
    }
}

impl Drop for CudaGlSink {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Monotonic clock the sink offers, matching the source-side `arrival_ns`
/// epoch so the latency histogram is meaningful. Same role as `WaylandClock`.
#[derive(Debug)]
struct CudaGlClock;
impl PipelineClock for CudaGlClock {
    fn now_ns(&self) -> u64 {
        monotonic_ns()
    }
}

impl AsyncElement for CudaGlSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn provide_clock(&self) -> Option<ClockCandidate> {
        Some(ClockCandidate::new(
            ClockPriority::Provider,
            alloc::sync::Arc::new(CudaGlClock),
        ))
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // Pass-through at negotiation; NV12 is enforced in configure_pipeline.
        // The native decoder lands NV12 on this link via its DerivedOutput.
        Ok(upstream_caps.clone())
    }

    /// Native NV12-only sink constraint (mirrors `KmsSink` / `WaylandSink`): the
    /// solver intersects this against the upstream decoder's NV12 `DerivedOutput`
    /// and lands fixed NV12 on the link, so an undecoded (non-NV12) chain fails
    /// loud in negotiation. Geometry stays open; the decoder fixates it. The
    /// CUDA-vs-system memory-domain check stays per-frame in `process`.
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }))
    }

    /// Presents from CUDA device memory only; declaring it lets the M354 converter
    /// auto-plug splice a `CudaDownload`-free GPU path (or, behind a tee, a
    /// `CudaDownload` only on a sibling System branch).
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::Cuda)
    }

    /// M12 / C3 step 3: ask the producer to keep buffers in CUDA device memory
    /// so the `NvdecCuda` -> sink handoff stays on the GPU. The runner conveys
    /// this `MemoryDomainKind::Cuda` proposal to the decoder's
    /// `configure_allocation`. Returns `None` until the geometry is known (no
    /// proposal to make pre-`configure_pipeline`).
    fn propose_allocation(&self, caps: &Caps) -> Option<AllocationParams> {
        let (w, h, _) = caps.dims()?;
        let (&Dim::Fixed(w), &Dim::Fixed(h)) = (w, h) else {
            return None;
        };
        Some(AllocationParams::cuda(
            nv12_byte_size(w, h),
            CUDA_POOL_HEADROOM,
            CUDA_ALIGN,
        ))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        // NV12 only. Caps do not encode the memory domain, so the Cuda-vs-
        // System distinction is checked per frame in `process`.
        let (w, h) = match absolute_caps {
            Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(w),
                height: Dim::Fixed(h),
                ..
            } => (*w, *h),
            _ => return Err(G2gError::CapsMismatch),
        };
        if w % 2 != 0 || h % 2 != 0 {
            return Err(G2gError::CapsMismatch);
        }

        // Mid-stream geometry change: same dims is a no-op; new dims tear down
        // the worker and respawn (M16 5j), as WaylandSink does.
        if self.worker.is_some() {
            if w == self.width && h == self.height {
                return Ok(ConfigureOutcome::Accepted);
            }
            self.shutdown();
        }

        let (tx, rx) = channel::<WorkerCmd>();
        let presented = Arc::clone(&self.frames_presented);
        let latency = Arc::clone(&self.latency);
        let title = self.title.clone();
        let app_id = self.app_id.clone();

        let ready = Arc::new(Handshake::new());
        let ready_for_worker = Arc::clone(&ready);

        let join = thread::Builder::new()
            .name(String::from("g2g-cudaglsink"))
            .spawn(move || {
                if let Err(e) = worker_main(
                    w,
                    h,
                    title,
                    app_id,
                    rx,
                    presented,
                    latency,
                    ready_for_worker,
                ) {
                    std::eprintln!("g2g-cudaglsink worker error: {e:?}");
                }
            })
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

        if !ready.wait(Duration::from_secs(5)) {
            let _ = tx.send(WorkerCmd::Shutdown);
            let _ = join.join();
            return Err(G2gError::Hardware(HardwareError::Other));
        }

        self.cmd_tx = Some(tx);
        self.worker = Some(join);
        self.width = w;
        self.height = h;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(Frame { domain, timing, .. }) => {
                    // This sink consumes CUDA device memory only; a system
                    // frame means the chain forgot the NvdecCuda backend.
                    let MemoryDomain::Cuda(buf) = domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let tx = self.cmd_tx.as_ref().ok_or(G2gError::NotConfigured)?;
                    let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
                    tx.send(WorkerCmd::Frame {
                        buf,
                        arrival_ns: timing.arrival_ns,
                        ack: ack_tx,
                    })
                    .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
                    // Block until the worker presents this frame (vsync-paced
                    // by the compositor's release of the EGL back buffer).
                    ack_rx
                        .await
                        .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
                    Ok(())
                }
                PipelinePacket::CapsChanged(_)
                | PipelinePacket::Flush
                | PipelinePacket::Segment(_) => Ok(()),
                PipelinePacket::Eos => {
                    self.shutdown();
                    Ok(())
                }
                // future PipelinePacket variants: no-op (terminal sink).
                _ => Ok(()),
            }
        })
    }
}

// =================================================================
// Worker thread: Wayland window + EGL/GL + CUDA-GL upload
// =================================================================

struct WorkerState {
    registry_state: RegistryState,
    output_state: OutputState,
    window: Window,
    qh: QueueHandle<WorkerState>,
    // EGL handles: kept alive for the worker's lifetime; the WlEglSurface must
    // outlive the EGL surface, which must outlive the wl_surface.
    egl: egl::Instance<egl::Static>,
    egl_display: egl::Display,
    egl_surface: egl::Surface,
    // Kept current for the worker's life; held only as a keep-alive after setup.
    _egl_context: egl::Context,
    _wl_egl: WlEglSurface,
    gl: GlState,
    configured: bool,
    exit: bool,
    ready: Option<Arc<Handshake>>,
    presented: Arc<AtomicU64>,
    latency: Arc<LatencyHistogram>,
    /// Frame that arrived before the surface was mappable.
    pending: Option<(OwnedCudaBuffer, u64, tokio::sync::oneshot::Sender<()>)>,
}

#[allow(clippy::too_many_arguments)]
fn worker_main(
    width: u32,
    height: u32,
    title: String,
    app_id: String,
    rx: Channel<WorkerCmd>,
    presented: Arc<AtomicU64>,
    latency: Arc<LatencyHistogram>,
    ready: Arc<Handshake>,
) -> Result<(), Box<dyn std::error::Error>> {
    let conn = Connection::connect_to_env()?;
    let (globals, event_queue) = registry_queue_init(&conn)?;
    let qh = event_queue.handle();

    let mut event_loop: EventLoop<WorkerState> = EventLoop::try_new()?;
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

    // --- EGL on the Wayland surface ---
    let egl = egl::Instance::new(egl::Static);

    // The wl_display raw pointer on wayland-client 0.31. The display is a
    // special global; `backend().display_ptr()` returns the libwayland
    // `*mut wl_display` EGL wants as its native display handle.
    let display_ptr = conn.backend().display_ptr() as *mut core::ffi::c_void;
    // SAFETY: `display_ptr` is the live connection's libwayland `*mut wl_display`,
    // valid for the worker thread's lifetime (the `Connection` outlives the EGL
    // display via `conn`); `get_display` only records the handle.
    let egl_display = unsafe { egl.get_display(display_ptr) }.ok_or("eglGetDisplay failed")?;
    egl.initialize(egl_display)?;
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
        .choose_first_config(egl_display, &config_attribs)?
        .ok_or("no matching EGL config")?;

    let context_attribs = [egl::CONTEXT_MAJOR_VERSION, 3, egl::NONE];
    let egl_context = egl.create_context(egl_display, config, None, &context_attribs)?;

    // wl_egl_window from the SCTK surface; EGL window surface on top of it.
    let wl_egl = WlEglSurface::new(window.wl_surface().id(), width as i32, height as i32)?;
    // SAFETY: `wl_egl.ptr()` is a live `wl_egl_window` for this display/config;
    // `wl_egl` is moved into `WorkerState._wl_egl`, so it outlives the surface.
    let egl_surface = unsafe {
        egl.create_window_surface(
            egl_display,
            config,
            wl_egl.ptr() as *mut core::ffi::c_void,
            None,
        )
    }?;
    egl.make_current(
        egl_display,
        Some(egl_surface),
        Some(egl_surface),
        Some(egl_context),
    )?;

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

    // SAFETY: `gl` wraps the GL ES 3 context made current on this thread above.
    let gl_state = unsafe { GlState::build(gl, width, height) }?;

    let mut state = WorkerState {
        registry_state: RegistryState::new(&globals),
        output_state: OutputState::new(&globals, &qh),
        window,
        qh: qh.clone(),
        egl,
        egl_display,
        egl_surface,
        _egl_context: egl_context,
        _wl_egl: wl_egl,
        gl: gl_state,
        configured: false,
        exit: false,
        ready: Some(ready),
        presented,
        latency,
        pending: None,
    };

    loop_handle.insert_source(rx, |event, _, state: &mut WorkerState| match event {
        ChanEvent::Msg(WorkerCmd::Frame {
            buf,
            arrival_ns,
            ack,
        }) => {
            if state.configured {
                state.draw(buf, arrival_ns, ack);
            } else {
                state.pending = Some((buf, arrival_ns, ack));
            }
        }
        ChanEvent::Msg(WorkerCmd::Shutdown) | ChanEvent::Closed => {
            state.exit = true;
        }
    })?;

    while !state.exit {
        event_loop.dispatch(Some(Duration::from_millis(100)), &mut state)?;
    }
    Ok(())
}

impl Drop for WorkerState {
    fn drop(&mut self) {
        // Tear down the EGL display / context / surface the worker created. A
        // mid-stream resolution change respawns the worker (via `shutdown`), so
        // without this each resize leaks an EGL context + surface. Best-effort;
        // the worker is exiting. Releasing the current context first, then
        // destroying the surface here (Drop runs before the `_wl_egl` field drops
        // its backing `wl_egl_window`) keeps the required outlives ordering.
        let _ = self.egl.make_current(self.egl_display, None, None, None);
        let _ = self.egl.destroy_surface(self.egl_display, self.egl_surface);
        let _ = self
            .egl
            .destroy_context(self.egl_display, self._egl_context);
        let _ = self.egl.terminate(self.egl_display);
    }
}

impl WorkerState {
    /// Upload the decoded NV12 planes into the GL textures via CUDA, draw the
    /// fullscreen quad through the NV12->RGB shader, and present. Signals
    /// `ack` after `eglSwapBuffers` returns (compositor-paced backpressure).
    fn draw(
        &mut self,
        buf: OwnedCudaBuffer,
        arrival_ns: u64,
        ack: tokio::sync::oneshot::Sender<()>,
    ) {
        if let Err(e) = self.draw_inner(&buf) {
            std::eprintln!("g2g-cudaglsink draw error: {e:?}");
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

    fn draw_inner(&mut self, buf: &OwnedCudaBuffer) -> Result<(), G2gError> {
        // CUDA upload + NV12->RGB draw (shared with the KMS sink).
        self.gl.upload_and_draw(buf)?;

        // Subscribe to the next frame callback (compositor pacing) and present.
        let surface = self.window.wl_surface().clone();
        surface.frame(&self.qh, surface.clone());
        self.egl
            .swap_buffers(self.egl_display, self.egl_surface)
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        Ok(())
    }
}

impl CompositorHandler for WorkerState {
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
        // ack in `draw`, so nothing extra is needed here in v1.
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

impl WindowHandler for WorkerState {
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
            if let Some((buf, arrival_ns, ack)) = self.pending.take() {
                self.draw(buf, arrival_ns, ack);
            }
        }
    }
}

impl OutputHandler for WorkerState {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }
    fn new_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
    fn update_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
    fn output_destroyed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
}

impl ProvidesRegistryState for WorkerState {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    registry_handlers![OutputState,];
}

delegate_compositor!(WorkerState);
delegate_output!(WorkerState);
delegate_xdg_shell!(WorkerState);
delegate_xdg_window!(WorkerState);
delegate_registry!(WorkerState);

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::{Rate, VideoCodec};

    fn nv12(w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Any,
        }
    }

    #[test]
    fn intercept_passes_through() {
        let sink = CudaGlSink::new();
        let h264 = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
        };
        assert_eq!(sink.intercept_caps(&h264), Ok(h264));
    }

    #[test]
    fn configure_rejects_non_nv12() {
        let mut sink = CudaGlSink::new();
        let i420 = Caps::RawVideo {
            format: RawVideoFormat::I420,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
        };
        assert_eq!(
            sink.configure_pipeline(&i420).err(),
            Some(G2gError::CapsMismatch)
        );
        assert!(sink.worker.is_none());
    }

    #[test]
    fn configure_rejects_odd_dims() {
        let mut sink = CudaGlSink::new();
        match sink.configure_pipeline(&nv12(641, 480)) {
            Err(G2gError::CapsMismatch) => {}
            other => panic!("expected CapsMismatch on odd dims, got {other:?}"),
        }
    }

    #[test]
    fn proposes_cuda_device_memory() {
        use g2g_core::MemoryDomainKind;
        let sink = CudaGlSink::new();
        let p = sink
            .propose_allocation(&nv12(1920, 1080))
            .expect("fixed-geometry NV12 yields a proposal");
        assert_eq!(p.domain, MemoryDomainKind::Cuda);
        assert_eq!(p.size_bytes, 1920 * 1080 * 3 / 2);
        assert_eq!(p.align, CUDA_ALIGN);
        assert_eq!(p.min_buffers, CUDA_POOL_HEADROOM);
    }
}
