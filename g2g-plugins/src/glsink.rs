//! Vendor-neutral OpenGL ES display sink (M891): system-memory NV12 or RGBA
//! presented through EGL on a Wayland `xdg_toplevel`.
//!
//! The GStreamer `glimagesink` analog, and the driver-agnostic sibling of
//! [`CudaGlSink`](crate::cudaglsink): no CUDA anywhere in its path, so it runs on
//! any EGL that offers GL ES 3 (Mesa on Intel / AMD, the NVIDIA driver, a
//! headless surfaceless device). Compared with [`WaylandSink`](crate::waylandsink),
//! which converts NV12 on the CPU and pushes XRGB8888 through `wl_shm`, the
//! convert here is a fragment shader and the frame goes to the compositor as a
//! GL back buffer.
//!
//! ## Pipeline shape
//!
//! ```text
//! FileSrc ─► H264Parse ─► FfmpegH264Dec ─► GlSink
//!                                            │
//!                                            └─► EGL/GL window
//! ```
//!
//! ## Formats
//!
//! NV12 (the decoders' output) and RGBA (a GPU/overlay element's output),
//! whichever negotiation settles on; the layout is fixed at
//! `configure_pipeline` and picks the GL program. NV12 is converted BT.601
//! limited-range on the GPU; BT.709 awaits colour metadata on `Caps`.
//!
//! ## Threading
//!
//! GL and Wayland are both single-thread-affine, so the window, the EGL context
//! and the draws live on a worker thread ([`crate::glwindow`]) spun up at
//! `configure_pipeline`. The sink struct holds only `Send` handles (a `calloop`
//! channel sender plus shared atomics), so the runner can move it between
//! executor tasks.
//!
//! ## Verification status
//!
//! The GL render path (program, `R8`/`RG8`/`RGBA8` textures, quad, NV12->RGB
//! convert) is tested headlessly against a CPU BT.601 reference: the tests below
//! bring up a surfaceless EGL device, render a synthetic frame through the real
//! [`GlState`] program into an FBO, and compare `glReadPixels` output. The
//! on-screen present (Wayland surface + `eglSwapBuffers`) is validated by the
//! display smoke tests, not in CI.

use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use smithay_client_toolkit::reexports::calloop::channel::{channel, Sender as CalloopSender};

use crate::clock::wait_to_present;
use crate::glnv12::{GlMode, GlState};
use crate::glwindow::{run_gl_window, FramePresenter, WindowParams, WorkerChannels, WorkerCmd};
use crate::worker_ready::Handshake;
use g2g_core::element::QosMessage;
use g2g_core::metrics::{monotonic_ns, LatencyHistogram, LatencySnapshot};
use g2g_core::{
    AsyncElement, BusHandle, Caps, CapsConstraint, CapsSet, ClockCandidate, ClockPriority,
    ClockSync, ConfigureOutcome, Dim, ElementMetadata, Frame, G2gError, HardwareError, OutputSink,
    PadTemplate, PadTemplates, PipelineClock, PipelinePacket, PresentationPacer, PropError,
    PropKind, PropValue, PropertySpec, Rate, RawVideoFormat, MAX_LATENESS_PROPERTY,
    QOS_INTERVAL_PROPERTY,
};

/// Worker thread name, also the prefix on the worker's error lines.
const WORKER_NAME: &str = "g2g-glsink";

/// Sink-side handle set. Only `Send + Sync` state lives here so the multi-thread
/// runner can move the sink between tasks; the worker's frame payload is the
/// packed frame bytes in the negotiated layout.
pub struct GlSink {
    title: String,
    app_id: String,
    cmd_tx: Option<CalloopSender<WorkerCmd<Vec<u8>>>>,
    worker: Option<JoinHandle<()>>,
    width: u32,
    height: u32,
    frames_presented: Arc<AtomicU64>,
    latency: Arc<LatencyHistogram>,
    /// PTS pacing + QoS late-drop, on top of the worker's swap-paced ack: idle
    /// until the runner hands over a clock, and the default lateness bound never
    /// drops.
    pacer: PresentationPacer,
}

impl core::fmt::Debug for GlSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GlSink")
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

impl Default for GlSink {
    fn default() -> Self {
        Self::new()
    }
}

impl GlSink {
    pub fn new() -> Self {
        Self {
            title: String::from("glass2glass"),
            app_id: String::from("io.glass2glass.GlSink"),
            cmd_tx: None,
            worker: None,
            width: 0,
            height: 0,
            frames_presented: Arc::new(AtomicU64::new(0)),
            latency: Arc::new(LatencyHistogram::new()),
            pacer: PresentationPacer::new(),
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

    /// QoS late-drop bound: once PTS pacing is engaged, a frame past its
    /// deadline by more than `ns` is dropped instead of presented late, so the
    /// sink catches up. The default (`u64::MAX`) never drops.
    pub fn with_max_lateness_ns(mut self, ns: u64) -> Self {
        self.pacer.set_max_lateness_ns(ns);
        self
    }

    /// Post a running-stats `Qos` report every `ns` of clock time while frames
    /// present, on top of the per-drop reports. `0` (the default) reports only
    /// drops.
    pub fn with_qos_interval_ns(mut self, ns: u64) -> Self {
        self.pacer.set_report_interval_ns(ns);
        self
    }

    /// Attach the pipeline bus so QoS reports reach the application.
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.pacer.set_bus(bus);
        self
    }

    /// Frames dropped by QoS late-drop (past their deadline beyond the bound).
    pub fn late_dropped(&self) -> u64 {
        self.pacer.late_dropped()
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

impl Drop for GlSink {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Monotonic clock the sink offers, matching the source-side `arrival_ns` epoch
/// so the latency histogram is meaningful.
#[derive(Debug)]
struct GlClock;
impl PipelineClock for GlClock {
    fn now_ns(&self) -> u64 {
        monotonic_ns()
    }
}

/// The sink's accepted layouts, NV12 first (what the decoders produce).
fn accepted_caps() -> CapsSet {
    CapsSet::from_alternatives(Vec::from([
        any_geometry(RawVideoFormat::Nv12),
        any_geometry(RawVideoFormat::Rgba8),
    ]))
}

fn any_geometry(format: RawVideoFormat) -> Caps {
    Caps::RawVideo {
        format,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

impl PadTemplates for GlSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink(accepted_caps())])
    }
}

impl AsyncElement for GlSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn provide_clock(&self) -> Option<ClockCandidate> {
        Some(ClockCandidate::new(
            ClockPriority::Provider,
            alloc::sync::Arc::new(GlClock),
        ))
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // Pass-through at negotiation; the layout is enforced in
        // configure_pipeline (as WaylandSink does), so an undecoded chain fails
        // there rather than blocking the solve.
        Ok(upstream_caps.clone())
    }

    /// NV12 or RGBA, geometry open (the decoder fixates it). The solver
    /// intersects this against the upstream's output, so a compressed or planar
    /// chain fails loud in negotiation.
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(accepted_caps())
    }

    /// Adopt the elected clock + base time so frames present at their PTS
    /// deadline rather than as fast as the EGL swap allows.
    fn set_clock_sync(&mut self, sync: ClockSync) {
        self.pacer.set_clock_sync(sync);
    }

    /// Relay a late drop upstream: the runner forwards it onto the incoming
    /// link, where the producer can shed load.
    fn take_qos(&mut self) -> Option<QosMessage> {
        self.pacer.take_qos()
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "OpenGL ES video sink",
            "Sink/Video",
            "Presents NV12 / RGBA video through EGL on a Wayland surface",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new("title", PropKind::Str, "window title").with_default("glass2glass"),
            PropertySpec::new("app-id", PropKind::Str, "Wayland xdg app id")
                .with_default("io.glass2glass.GlSink"),
            MAX_LATENESS_PROPERTY,
            QOS_INTERVAL_PROPERTY,
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "title" => {
                self.title = value.as_str().ok_or(PropError::Type)?.into();
                Ok(())
            }
            "app-id" => {
                self.app_id = value.as_str().ok_or(PropError::Type)?.into();
                Ok(())
            }
            _ => self
                .pacer
                .set_property(name, &value)
                .unwrap_or(Err(PropError::Unknown)),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "title" => Some(PropValue::Str(self.title.clone())),
            "app-id" => Some(PropValue::Str(self.app_id.clone())),
            _ => self.pacer.get_property(name),
        }
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (format, w, h) = match absolute_caps {
            Caps::RawVideo {
                format,
                width: Dim::Fixed(w),
                height: Dim::Fixed(h),
                ..
            } => (*format, *w, *h),
            _ => return Err(G2gError::CapsMismatch),
        };
        let mode = match format {
            RawVideoFormat::Nv12 => GlMode::Nv12,
            RawVideoFormat::Rgba8 => GlMode::Rgba,
            _ => return Err(G2gError::CapsMismatch),
        };
        // NV12 chroma is subsampled: odd geometry has no well-defined plane.
        if mode == GlMode::Nv12 && (w % 2 != 0 || h % 2 != 0) {
            return Err(G2gError::CapsMismatch);
        }
        if w == 0 || h == 0 {
            return Err(G2gError::CapsMismatch);
        }

        // Mid-stream geometry change: same dims is a no-op; new dims tear down
        // the worker and respawn, as WaylandSink does.
        if self.worker.is_some() {
            if w == self.width && h == self.height {
                return Ok(ConfigureOutcome::Accepted);
            }
            self.shutdown();
        }

        let (tx, rx) = channel::<WorkerCmd<Vec<u8>>>();
        let presented = Arc::clone(&self.frames_presented);
        let latency = Arc::clone(&self.latency);
        let params = WindowParams {
            width: w,
            height: h,
            title: self.title.clone(),
            app_id: self.app_id.clone(),
            log_tag: WORKER_NAME,
        };

        let ready = Arc::new(Handshake::new());
        let ready_for_worker = Arc::clone(&ready);

        let join = thread::Builder::new()
            .name(String::from(WORKER_NAME))
            .spawn(move || {
                let channels = WorkerChannels {
                    rx,
                    presented,
                    latency,
                    ready: ready_for_worker,
                };
                if let Err(e) = run_gl_window(SystemPresenter { mode }, params, channels) {
                    std::eprintln!("{WORKER_NAME} worker error: {e:?}");
                }
            })
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

        // Bounded wait: a hung compositor mustn't lock us up forever.
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
                    let Some(slice) = domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    // PTS pacing: hold the frame until its deadline on the elected
                    // clock, or drop it when it is already too late (the QoS
                    // bound) or outside the segment. Unpaced without a clock:
                    // present as fast as the swap allows.
                    let presented = self.frames_presented.load(Ordering::Relaxed);
                    if !wait_to_present(self.pacer.judge(timing.pts_ns, presented)).await {
                        return Ok(());
                    }
                    let tx = self.cmd_tx.as_ref().ok_or(G2gError::NotConfigured)?;
                    let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
                    tx.send(WorkerCmd::Frame {
                        frame: slice.to_vec(),
                        arrival_ns: timing.arrival_ns,
                        ack: ack_tx,
                    })
                    .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
                    // Block until the worker presents this frame (vsync-paced by
                    // the compositor's release of the EGL back buffer).
                    ack_rx
                        .await
                        .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
                    Ok(())
                }
                PipelinePacket::CapsChanged(_) => Ok(()),
                // Track the playback segment so PTS maps to running time
                // (correct across a seek), and re-anchor after a seek flush.
                PipelinePacket::Segment(seg) => {
                    self.pacer.set_segment(seg);
                    Ok(())
                }
                PipelinePacket::Flush => {
                    self.pacer.flush();
                    Ok(())
                }
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

/// System-memory half of the shared GL worker: the frame bytes go into the
/// textures with `glTexSubImage2D`, in the layout negotiation settled on.
struct SystemPresenter {
    mode: GlMode,
}

impl FramePresenter for SystemPresenter {
    type Frame = Vec<u8>;

    fn mode(&self) -> GlMode {
        self.mode
    }

    fn present(&mut self, gl: &mut GlState, frame: &Self::Frame) -> Result<(), G2gError> {
        gl.upload_system_and_draw(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::VideoCodec;

    fn caps(format: RawVideoFormat, w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Any,
        }
    }

    #[test]
    fn intercept_passes_through() {
        let sink = GlSink::new();
        let h264 = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
        };
        assert_eq!(sink.intercept_caps(&h264), Ok(h264));
    }

    #[test]
    fn accepts_nv12_and_rgba_only() {
        let sink = GlSink::new();
        let CapsConstraint::Accepts(set) = sink.caps_constraint_as_sink() else {
            panic!("sink constraint is an Accepts set");
        };
        for format in [RawVideoFormat::Nv12, RawVideoFormat::Rgba8] {
            assert_eq!(
                set.intersect(&CapsSet::one(caps(format, 640, 480)))
                    .alternatives()
                    .len(),
                1,
                "{format:?} is accepted"
            );
        }
        assert!(set
            .intersect(&CapsSet::one(caps(RawVideoFormat::I420, 640, 480)))
            .is_empty());
    }

    /// The launch name and its `glsink` alias both build this sink, and a
    /// property set through the registry-built element sticks (the `parse_launch`
    /// path: `properties()` lookup then `set_property`).
    #[test]
    fn registry_resolves_glimagesink_and_glsink() {
        let reg = crate::registry::default_registry();
        for name in ["glimagesink", "glsink"] {
            let mut el = reg
                .make_element(name)
                .unwrap_or_else(|| panic!("{name} is registered"));
            assert!(
                el.properties().iter().any(|p| p.name == "title"),
                "{name} exposes title"
            );
            el.set_property("title", PropValue::Str(String::from("wall")))
                .expect("title is settable through the registry element");
            assert_eq!(
                el.get_property("title"),
                Some(PropValue::Str(String::from("wall")))
            );
        }
    }

    #[test]
    fn configure_rejects_unsupported_format_and_odd_nv12_dims() {
        let mut sink = GlSink::new();
        assert_eq!(
            sink.configure_pipeline(&caps(RawVideoFormat::I420, 640, 480))
                .err(),
            Some(G2gError::CapsMismatch)
        );
        assert_eq!(
            sink.configure_pipeline(&caps(RawVideoFormat::Nv12, 641, 480))
                .err(),
            Some(G2gError::CapsMismatch)
        );
        assert!(sink.worker.is_none());
    }

    #[test]
    fn properties_round_trip() {
        let mut sink = GlSink::new();
        for spec in AsyncElement::properties(&sink) {
            assert!(
                sink.get_property(spec.name).is_some(),
                "property {} is readable",
                spec.name
            );
        }
        sink.set_property("title", PropValue::Str(String::from("wall")))
            .expect("title is settable");
        assert_eq!(
            sink.get_property("title"),
            Some(PropValue::Str(String::from("wall")))
        );
        sink.set_property("max-lateness", PropValue::Uint(7_000_000))
            .expect("pacing property is settable");
        assert_eq!(
            sink.get_property("max-lateness"),
            Some(PropValue::Uint(7_000_000))
        );
        assert_eq!(
            sink.set_property("nope", PropValue::Uint(0)),
            Err(PropError::Unknown)
        );
    }

    // =============================================================
    // Headless GL render test: the real GlState program, no display server
    // =============================================================

    use glow::HasContext;
    use khronos_egl as egl;

    /// Mesa's display-server-free EGL platform (`EGL_MESA_platform_surfaceless`).
    const PLATFORM_SURFACELESS_MESA: egl::Enum = 0x31DD;

    /// An offscreen EGL context: the surfaceless platform when the EGL client
    /// offers it, else the default display, and a pbuffer surface unless the
    /// driver supports a surfaceless (`EGL_NO_SURFACE`) make-current.
    struct Headless {
        egl: egl::Instance<egl::Static>,
        display: egl::Display,
        surface: Option<egl::Surface>,
        context: egl::Context,
        /// How the context came up, for the test's evidence line.
        how: String,
    }

    impl Drop for Headless {
        fn drop(&mut self) {
            let _ = self.egl.make_current(self.display, None, None, None);
            if let Some(s) = self.surface {
                let _ = self.egl.destroy_surface(self.display, s);
            }
            let _ = self.egl.destroy_context(self.display, self.context);
            let _ = self.egl.terminate(self.display);
        }
    }

    /// Bring up an offscreen GL ES 3 context, or return why it was not possible
    /// (no EGL device on the host: the test then skips).
    fn headless_context() -> Result<(Headless, glow::Context), String> {
        let egl = egl::Instance::new(egl::Static);
        let client_exts = egl
            .query_string(None, egl::EXTENSIONS)
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();

        let (display, mut how) = if client_exts.contains("EGL_MESA_platform_surfaceless") {
            // SAFETY: the surfaceless platform takes no native display handle,
            // so EGL_DEFAULT_DISPLAY (null) is the value the extension requires.
            let d = unsafe {
                egl.get_platform_display(
                    PLATFORM_SURFACELESS_MESA,
                    egl::DEFAULT_DISPLAY,
                    &[egl::ATTRIB_NONE],
                )
            }
            .map_err(|e| alloc::format!("eglGetPlatformDisplay(surfaceless): {e:?}"))?;
            (d, String::from("EGL_MESA_platform_surfaceless"))
        } else {
            // SAFETY: EGL_DEFAULT_DISPLAY is a valid native display handle for
            // eglGetDisplay on every platform.
            let d = unsafe { egl.get_display(egl::DEFAULT_DISPLAY) }
                .ok_or("eglGetDisplay(EGL_DEFAULT_DISPLAY) failed")?;
            (d, String::from("EGL default display"))
        };

        let version = egl
            .initialize(display)
            .map_err(|e| alloc::format!("eglInitialize: {e:?}"))?;
        egl.bind_api(egl::OPENGL_ES_API)
            .map_err(|e| alloc::format!("eglBindAPI: {e:?}"))?;
        let vendor = egl
            .query_string(Some(display), egl::VENDOR)
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let display_exts = egl
            .query_string(Some(display), egl::EXTENSIONS)
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        how = alloc::format!("{how}, EGL {}.{} vendor {vendor}", version.0, version.1);

        let config_attribs = [
            egl::SURFACE_TYPE,
            egl::PBUFFER_BIT,
            egl::RENDERABLE_TYPE,
            egl::OPENGL_ES3_BIT,
            egl::RED_SIZE,
            8,
            egl::GREEN_SIZE,
            8,
            egl::BLUE_SIZE,
            8,
            egl::ALPHA_SIZE,
            8,
            egl::NONE,
        ];
        let config = egl
            .choose_first_config(display, &config_attribs)
            .map_err(|e| alloc::format!("eglChooseConfig: {e:?}"))?
            .ok_or("no GL ES 3 pbuffer config")?;

        let context = egl
            .create_context(
                display,
                config,
                None,
                &[egl::CONTEXT_MAJOR_VERSION, 3, egl::NONE],
            )
            .map_err(|e| alloc::format!("eglCreateContext: {e:?}"))?;

        // The render target is an FBO either way; a surfaceless make-current
        // needs EGL_KHR_surfaceless_context, so fall back to a 1x1 pbuffer.
        let surface = if display_exts.contains("EGL_KHR_surfaceless_context") {
            how.push_str(", surfaceless context");
            None
        } else {
            how.push_str(", pbuffer surface");
            Some(
                egl.create_pbuffer_surface(
                    display,
                    config,
                    &[egl::WIDTH, 1, egl::HEIGHT, 1, egl::NONE],
                )
                .map_err(|e| alloc::format!("eglCreatePbufferSurface: {e:?}"))?,
            )
        };
        egl.make_current(display, surface, surface, Some(context))
            .map_err(|e| alloc::format!("eglMakeCurrent: {e:?}"))?;

        // SAFETY: the context just made current on this thread resolves the GL ES
        // entry points glow asks for by name.
        let gl = unsafe {
            glow::Context::from_loader_function(|s| match egl.get_proc_address(s) {
                Some(p) => p as *const core::ffi::c_void,
                None => core::ptr::null(),
            })
        };

        Ok((
            Headless {
                egl,
                display,
                surface,
                context,
                how,
            },
            gl,
        ))
    }

    /// Render `frame` through a real [`GlState`] of `mode` into an RGBA8 FBO and
    /// read the result back, top row first (undoing GL's bottom-up readback).
    fn render_to_rgba(
        gl: glow::Context,
        w: u32,
        h: u32,
        mode: GlMode,
        frame: &[u8],
    ) -> Result<Vec<u8>, String> {
        // SAFETY: the caller's context is current on this thread; the FBO and its
        // colour texture are created here and sized w x h, which is also the
        // viewport GlState draws with and the extent read back below.
        unsafe {
            let mut state = GlState::build(gl, w, h, mode).map_err(|e| alloc::format!("{e}"))?;
            let gl = state.gl();
            let color = gl.create_texture()?;
            gl.bind_texture(glow::TEXTURE_2D, Some(color));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA8 as i32,
                w as i32,
                h as i32,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(None),
            );
            let fbo = gl.create_framebuffer()?;
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(color),
                0,
            );
            if gl.check_framebuffer_status(glow::FRAMEBUFFER) != glow::FRAMEBUFFER_COMPLETE {
                return Err(String::from("incomplete FBO"));
            }

            state
                .upload_system_and_draw(frame)
                .map_err(|e| alloc::format!("{e:?}"))?;

            let gl = state.gl();
            let mut flipped = alloc::vec![0u8; (w * h * 4) as usize];
            gl.read_pixels(
                0,
                0,
                w as i32,
                h as i32,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(Some(&mut flipped)),
            );
            let err = gl.get_error();
            if err != glow::NO_ERROR {
                return Err(alloc::format!("GL error 0x{err:x}"));
            }

            // glReadPixels starts at the bottom-left; the quad already flips V,
            // so row r of the readback is row (h - 1 - r) of the frame.
            let row = (w * 4) as usize;
            let mut out = alloc::vec![0u8; flipped.len()];
            for r in 0..h as usize {
                let src = (h as usize - 1 - r) * row;
                out[r * row..(r + 1) * row].copy_from_slice(&flipped[src..src + row]);
            }
            Ok(out)
        }
    }

    /// BT.601 limited-range NV12 -> RGB reference, the integer-fixed-point math
    /// `WaylandSink` converts with on the CPU. Chroma is nearest (`col / 2`).
    fn nv12_to_rgb_reference(src: &[u8], w: usize, h: usize) -> Vec<u8> {
        let (y_plane, uv_plane) = src.split_at(w * h);
        let mut out = alloc::vec![0u8; w * h * 3];
        for row in 0..h {
            for col in 0..w {
                let y = y_plane[row * w + col] as i32;
                let uv = (row / 2) * w + (col / 2) * 2;
                let c = y - 16;
                let d = uv_plane[uv] as i32 - 128;
                let e = uv_plane[uv + 1] as i32 - 128;
                let r = (298 * c + 409 * e + 128) >> 8;
                let g = (298 * c - 100 * d - 208 * e + 128) >> 8;
                let b = (298 * c + 516 * d + 128) >> 8;
                let dst = (row * w + col) * 3;
                out[dst] = r.clamp(0, 255) as u8;
                out[dst + 1] = g.clamp(0, 255) as u8;
                out[dst + 2] = b.clamp(0, 255) as u8;
            }
        }
        out
    }

    /// NV12 test frame: a luma ramp that changes every pixel over one constant
    /// chroma pair. Constant chroma is deliberate: the shader upsamples chroma
    /// bilinearly (`LINEAR` on the half-res `RG8` texture) while the CPU
    /// reference takes the nearest pair, and the two only agree exactly where
    /// the chroma field is flat.
    fn nv12_ramp(w: usize, h: usize, u: u8, v: u8) -> Vec<u8> {
        let mut buf = alloc::vec![0u8; w * h + w * h / 2];
        for row in 0..h {
            for col in 0..w {
                // Triangle wave, not a wrapping ramp: a 255 -> 0 step between
                // neighbouring texels would make the result hostage to the
                // sampler's sub-texel precision instead of the convert.
                let t = (row * 7 + col * 3) % 510;
                buf[row * w + col] = if t < 255 { t } else { 509 - t } as u8;
            }
        }
        for i in 0..(w * h / 4) {
            buf[w * h + i * 2] = u;
            buf[w * h + i * 2 + 1] = v;
        }
        buf
    }

    /// The milestone's correctness check: one synthetic NV12 frame through the
    /// real GL program must match the CPU BT.601 reference within a per-channel
    /// LSB or two. Two, not zero, because the shader centres chroma on 0.5 while
    /// the integer reference centres it on 128/255, a systematic ~1 LSB (worst on
    /// blue, whose chroma coefficient is 2.017) on top of float-vs-fixed-point
    /// rounding.
    #[test]
    fn nv12_gpu_convert_matches_cpu_reference() {
        let (headless, gl) = match headless_context() {
            Ok(pair) => pair,
            Err(why) => {
                std::println!("skipping: no headless EGL context ({why})");
                return;
            }
        };
        let (w, h) = (64u32, 32u32);
        let frame = nv12_ramp(w as usize, h as usize, 90, 200);
        let rgba = render_to_rgba(gl, w, h, GlMode::Nv12, &frame).expect("headless NV12 render");
        let reference = nv12_to_rgb_reference(&frame, w as usize, h as usize);

        let mut max_delta = 0i32;
        for px in 0..(w * h) as usize {
            for ch in 0..3 {
                let got = rgba[px * 4 + ch] as i32;
                let want = reference[px * 3 + ch] as i32;
                max_delta = max_delta.max((got - want).abs());
            }
            assert_eq!(rgba[px * 4 + 3], 255, "NV12 shader writes opaque alpha");
        }
        std::println!(
            "headless NV12 convert via {}: max per-channel delta {max_delta}",
            headless.how
        );
        assert!(
            max_delta <= 2,
            "GPU NV12 convert is off the CPU reference by {max_delta}"
        );
    }

    /// Chroma really reaches the shader per region, not just as one flat value:
    /// two half-frames with different chroma pairs, compared away from the seam
    /// (where the shader's bilinear chroma upsample blends the two).
    #[test]
    fn nv12_chroma_regions_reach_the_shader() {
        let (headless, gl) = match headless_context() {
            Ok(pair) => pair,
            Err(why) => {
                std::println!("skipping: no headless EGL context ({why})");
                return;
            }
        };
        let (w, h) = (64usize, 32usize);
        let mut frame = nv12_ramp(w, h, 90, 200);
        // Right half: a different chroma pair.
        for crow in 0..h / 2 {
            for ccol in (w / 4)..(w / 2) {
                let i = w * h + crow * w + ccol * 2;
                frame[i] = 200;
                frame[i + 1] = 80;
            }
        }
        let rgba = render_to_rgba(gl, w as u32, h as u32, GlMode::Nv12, &frame)
            .expect("headless NV12 render");
        let reference = nv12_to_rgb_reference(&frame, w, h);

        let mut max_delta = 0i32;
        for row in 0..h {
            for col in 0..w {
                // Skip the 4 px either side of the chroma seam and the frame
                // edges, where nearest and bilinear chroma disagree by design.
                if col.abs_diff(w / 2) <= 4 {
                    continue;
                }
                for ch in 0..3 {
                    let got = rgba[(row * w + col) * 4 + ch] as i32;
                    let want = reference[(row * w + col) * 3 + ch] as i32;
                    max_delta = max_delta.max((got - want).abs());
                }
            }
        }
        std::println!(
            "headless NV12 two-chroma-region convert via {}: max per-channel delta {max_delta}",
            headless.how
        );
        assert!(
            max_delta <= 2,
            "GPU NV12 convert is off the CPU reference by {max_delta}"
        );
    }

    /// The RGBA path is a straight texture fetch, so the readback must be the
    /// uploaded frame byte for byte.
    #[test]
    fn rgba_path_is_bit_exact() {
        let (headless, gl) = match headless_context() {
            Ok(pair) => pair,
            Err(why) => {
                std::println!("skipping: no headless EGL context ({why})");
                return;
            }
        };
        let (w, h) = (64u32, 32u32);
        let mut frame = alloc::vec![0u8; (w * h * 4) as usize];
        for row in 0..h as usize {
            for col in 0..w as usize {
                let px = (row * w as usize + col) * 4;
                frame[px] = (col * 4) as u8;
                frame[px + 1] = (row * 8) as u8;
                frame[px + 2] = ((row * 3 + col * 5) % 256) as u8;
                frame[px + 3] = 255;
            }
        }
        let rgba = render_to_rgba(gl, w, h, GlMode::Rgba, &frame).expect("headless RGBA render");
        let mismatches = rgba
            .iter()
            .zip(frame.iter())
            .filter(|(a, b)| a != b)
            .count();
        std::println!(
            "headless RGBA passthrough via {}: {mismatches} mismatched bytes",
            headless.how
        );
        assert_eq!(mismatches, 0, "RGBA passthrough must be bit-exact");
    }
}
