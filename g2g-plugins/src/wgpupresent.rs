//! Windowed wgpu display sink (M1017): the launch-line `wgpusink`.
//!
//! [`WgpuSink`] renders and presents, but only to a target the application built.
//! This element is that application: it opens its own `xdg_toplevel` on the
//! shared Wayland worker ([`crate::waylandwindow`]), builds a `wgpu::Surface`
//! over that surface's raw handles, and drives a surface-target [`WgpuSink`] on
//! the worker thread. So a text pipeline gets a GPU present with no window /
//! event-loop code of its own:
//!
//! ```text
//! videotestsrc ! wgpusink
//! filesrc ! h264parse ! nvdec ! wgpusink      (the CUDA frame is bridged, not downloaded)
//! ```
//!
//! ## Formats and domains
//!
//! A `MemoryDomain::WgpuTexture` frame is blitted where it lies (the point of the
//! zero-copy decode paths); a system-memory NV12 or RGBA frame costs one
//! `write_texture` before the same blit. A frame in another domain gets a
//! converter spliced ahead of the sink by the M354 auto-plug, which is how an
//! NVDEC CUDA frame reaches the screen without the PCIe download `waylandsink`
//! would have cost.
//!
//! ## Threading
//!
//! Wayland and the wgpu surface are both thread-affine, so the window, the device
//! and the presents live on a worker thread spun up at `configure_pipeline`. The
//! sink struct holds only `Send` handles (a `calloop` channel sender plus shared
//! atomics), so the runner can move it between executor tasks.
//!
//! Linux (Wayland) only, `wgpu-present` feature. The on-screen present is
//! validated by the display smoke runs, not in CI; the render itself is what the
//! headless [`WgpuSink`] tests cover.

use core::future::Future;
use core::pin::Pin;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use raw_window_handle::{
    RawDisplayHandle, RawWindowHandle, WaylandDisplayHandle, WaylandWindowHandle,
};
use smithay_client_toolkit::reexports::calloop::channel::{channel, Sender as CalloopSender};
use smithay_client_toolkit::reexports::client::{protocol::wl_surface, Connection, Proxy};

use crate::clock::wait_to_present;
use crate::gpu::GpuContext;
use crate::waylandwindow::{run_window, WindowParams, WindowRenderer, WorkerChannels, WorkerCmd};
use crate::wgpusink::WgpuSink;
use crate::worker_ready::Handshake;
use g2g_core::element::QosMessage;
use g2g_core::log::{short_type_name, LogName, LogSource, Target};
use g2g_core::memory::{DomainSet, MemoryDomain, MemoryDomainKind};
use g2g_core::metrics::{monotonic_ns, LatencyHistogram, LatencySnapshot};
use g2g_core::{
    g2g_error, AsyncElement, BusHandle, Caps, CapsConstraint, ClockCandidate, ClockPriority,
    ClockSync, ConfigureOutcome, ElementMetadata, Frame, G2gError, HardwareError, OutputSink,
    PadTemplate, PadTemplates, PipelineClock, PipelinePacket, PresentationPacer, PropError,
    PropKind, PropValue, PropertySpec, MAX_LATENESS_PROPERTY, QOS_INTERVAL_PROPERTY,
};

/// Worker thread name, also the prefix on the worker's error lines.
const WORKER_NAME: &str = "g2g-wgpusink";

/// How long to wait for the compositor to map the window before giving up, so a
/// hung compositor fails the pipeline instead of blocking it forever.
const WINDOW_READY_TIMEOUT: Duration = Duration::from_secs(5);

/// How long each wait on the readiness latch runs before the worker's error
/// channel is checked again.
const WINDOW_READY_POLL: Duration = Duration::from_millis(50);

/// A window dimension property left at this value follows the video geometry.
const FOLLOW_VIDEO: u32 = 0;

/// Sink-side handle set. Only `Send + Sync` state lives here so the multi-thread
/// runner can move the sink between tasks; the worker's frame payload is the
/// frame's memory domain, moved across untouched (a GPU texture handle, or the
/// system-memory slice, never a copy).
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::wgpupresent::WgpuPresentSink;
///
/// let sink = WgpuPresentSink::new()
///     .with_title("g2g preview")
///     .with_max_lateness_ns(20_000_000);
/// ```
pub struct WgpuPresentSink {
    title: String,
    app_id: String,
    fullscreen: bool,
    /// Requested window size; [`FOLLOW_VIDEO`] means the negotiated geometry.
    window_width: u32,
    window_height: u32,
    cmd_tx: Option<CalloopSender<WorkerCmd<MemoryDomain>>>,
    worker: Option<JoinHandle<()>>,
    /// Caps the window's renderer opens with, from the last `configure_pipeline`.
    caps: Option<Caps>,
    /// Negotiated video geometry, so a mid-stream change reaches the worker once.
    width: u32,
    height: u32,
    frames_presented: Arc<AtomicU64>,
    latency: Arc<LatencyHistogram>,
    /// Why the worker could not bring its window up, sent once by the worker and
    /// read while waiting for the readiness handshake, so a setup failure is
    /// reported in plain words instead of as a silent timeout.
    startup_error: Option<std::sync::mpsc::Receiver<String>>,
    /// Runner-assigned instance name, so this sink's own error lines name it.
    log_name: LogName,
    /// PTS pacing + QoS late-drop, on top of the worker's present-paced ack: idle
    /// until the runner hands over a clock, and the default lateness bound never
    /// drops.
    pacer: PresentationPacer,
}

impl core::fmt::Debug for WgpuPresentSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WgpuPresentSink")
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

impl Default for WgpuPresentSink {
    fn default() -> Self {
        Self::new()
    }
}

impl WgpuPresentSink {
    pub fn new() -> Self {
        Self {
            title: String::from(DEFAULT_TITLE),
            app_id: String::from(DEFAULT_APP_ID),
            fullscreen: false,
            window_width: FOLLOW_VIDEO,
            window_height: FOLLOW_VIDEO,
            cmd_tx: None,
            worker: None,
            caps: None,
            width: 0,
            height: 0,
            frames_presented: Arc::new(AtomicU64::new(0)),
            latency: Arc::new(LatencyHistogram::new()),
            startup_error: None,
            log_name: LogName::new(),
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

    /// Map the window fullscreen instead of as a normal toplevel.
    pub fn with_fullscreen(mut self, fullscreen: bool) -> Self {
        self.fullscreen = fullscreen;
        self
    }

    /// Open the window at this size instead of the video's, scaling each frame to
    /// fill it. `0` (the default) follows the video geometry.
    pub fn with_window_size(mut self, width: u32, height: u32) -> Self {
        self.window_width = width;
        self.window_height = height;
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

    /// Glass-to-glass latency snapshot: source-side `arrival_ns` to the present
    /// that puts the frame on screen. Untimed pipelines report `count = 0`.
    pub fn latency_snapshot(&self) -> LatencySnapshot {
        self.latency.snapshot()
    }

    /// Window width to ask for: the property when set, else the video's.
    fn window_width(&self, video_width: u32) -> u32 {
        if self.window_width == FOLLOW_VIDEO {
            video_width
        } else {
            self.window_width
        }
    }

    fn window_height(&self, video_height: u32) -> u32 {
        if self.window_height == FOLLOW_VIDEO {
            video_height
        } else {
            self.window_height
        }
    }

    /// Wait for the worker to map its window, or for it to say why it could not.
    /// Polls both so a failed setup reports its reason at once rather than after
    /// the full timeout, and a hung compositor still gives up.
    fn await_window(
        &self,
        ready: &Handshake,
        errors: &std::sync::mpsc::Receiver<String>,
    ) -> Result<(), String> {
        let deadline = Instant::now() + WINDOW_READY_TIMEOUT;
        loop {
            if ready.wait(WINDOW_READY_POLL) {
                return Ok(());
            }
            if let Ok(reason) = errors.try_recv() {
                return Err(reason);
            }
            if Instant::now() >= deadline {
                return Err(String::from("the compositor never configured the window"));
            }
        }
    }

    /// Open the window and bring its renderer up, once. Deferred to the first
    /// frame on purpose: a launch line negotiates a placeholder geometry and only
    /// learns the real one when the decoder has read the stream, so opening at
    /// `configure_pipeline` time would map a window a few pixels across.
    fn open_window(&mut self) -> Result<(), G2gError> {
        if self.cmd_tx.is_some() {
            return Ok(());
        }
        let caps = self.caps.clone().ok_or(G2gError::NotConfigured)?;
        let (tx, rx) = channel::<WorkerCmd<MemoryDomain>>();
        let presented = Arc::clone(&self.frames_presented);
        let latency = Arc::clone(&self.latency);
        let params = WindowParams {
            width: self.window_width(self.width),
            height: self.window_height(self.height),
            title: self.title.clone(),
            app_id: self.app_id.clone(),
            fullscreen: self.fullscreen,
            log_tag: short_type_name::<Self>(),
        };

        let ready = Arc::new(Handshake::new());
        let ready_for_worker = Arc::clone(&ready);
        let (error_tx, error_rx) = std::sync::mpsc::channel();

        let join = thread::Builder::new()
            .name(String::from(WORKER_NAME))
            .spawn(move || {
                let channels = WorkerChannels {
                    rx,
                    presented,
                    latency,
                    ready: ready_for_worker,
                };
                let build = |conn: &Connection,
                             surface: &wl_surface::WlSurface,
                             window_width,
                             window_height| {
                    build_renderer(conn, surface, window_width, window_height, &caps)
                };
                if let Err(e) = run_window(build, params, channels) {
                    let _ = error_tx.send(alloc::format!("{e}"));
                }
            })
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

        if let Err(reason) = self.await_window(&ready, &error_rx) {
            let _ = tx.send(WorkerCmd::Shutdown);
            let _ = join.join();
            g2g_error!(self, "cannot open a window to present on: {reason}");
            return Err(G2gError::Hardware(HardwareError::Other));
        }

        self.startup_error = Some(error_rx);
        self.cmd_tx = Some(tx);
        self.worker = Some(join);
        Ok(())
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

impl Drop for WgpuPresentSink {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Default window identity, also the property defaults `gst-inspect` reports.
const DEFAULT_TITLE: &str = "glass2glass";
const DEFAULT_APP_ID: &str = "io.glass2glass.WgpuSink";

/// Monotonic clock the sink offers, matching the source-side `arrival_ns` epoch
/// so the latency histogram is meaningful.
#[derive(Debug)]
struct PresentClock;
impl PipelineClock for PresentClock {
    fn now_ns(&self) -> u64 {
        monotonic_ns()
    }
}

impl PadTemplates for WgpuPresentSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink(crate::wgpusink::accepted_caps())])
    }
}

impl AsyncElement for WgpuPresentSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn provide_clock(&self) -> Option<ClockCandidate> {
        Some(ClockCandidate::new(
            ClockPriority::Provider,
            Arc::new(PresentClock),
        ))
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    /// The same layouts the renderer accepts: RGBA or NV12, geometry open.
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(crate::wgpusink::accepted_caps())
    }

    /// A GPU texture is presented where it lies; system memory is uploaded.
    /// Declaring this is what makes the M354 auto-plug splice a bridge (rather
    /// than a download) ahead of a decoder that emits into another GPU domain.
    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::WgpuTexture).with(MemoryDomainKind::System)
    }

    /// Adopt the elected clock + base time so frames present at their PTS
    /// deadline rather than as fast as the producer pushes them.
    fn set_clock_sync(&mut self, sync: ClockSync) {
        self.pacer.set_clock_sync(sync);
    }

    /// Relay a late drop upstream: the runner forwards it onto the incoming
    /// link, where the producer can shed load.
    fn take_qos(&mut self) -> Option<QosMessage> {
        self.pacer.take_qos()
    }

    fn set_instance_name(&mut self, name: String) {
        self.log_name.set_instance(name);
    }

    fn set_log_category(&mut self, category: String) {
        self.log_name.set_category(category);
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "wgpu video sink",
            "Sink/Video",
            "Presents GPU-resident or system-memory video on a Wayland surface through wgpu",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new("title", PropKind::Str, "window title").with_default(DEFAULT_TITLE),
            PropertySpec::new("app-id", PropKind::Str, "Wayland xdg app id")
                .with_default(DEFAULT_APP_ID),
            PropertySpec::new("fullscreen", PropKind::Bool, "map the window fullscreen")
                .with_default("false"),
            PropertySpec::new(
                "window-width",
                PropKind::Uint,
                "window width in pixels (0: follow the video)",
            )
            .with_default("0"),
            PropertySpec::new(
                "window-height",
                PropKind::Uint,
                "window height in pixels (0: follow the video)",
            )
            .with_default("0"),
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
            "fullscreen" => {
                self.fullscreen = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "window-width" => {
                self.window_width = value.as_uint().ok_or(PropError::Type)? as u32;
                Ok(())
            }
            "window-height" => {
                self.window_height = value.as_uint().ok_or(PropError::Type)? as u32;
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
            "fullscreen" => Some(PropValue::Bool(self.fullscreen)),
            "window-width" => Some(PropValue::Uint(self.window_width as u64)),
            "window-height" => Some(PropValue::Uint(self.window_height as u64)),
            _ => self.pacer.get_property(name),
        }
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        // Reject here what the renderer cannot present, before opening a window.
        let (width, height) = crate::wgpusink::source_geometry(absolute_caps)?;

        // A running window follows the new caps in place: rebuilding it (and its
        // GPU device) for a mid-stream format change would throw away a working
        // surface and flash a second window.
        if let Some(tx) = self.cmd_tx.as_ref() {
            if (width, height) != (self.width, self.height) {
                tx.send(WorkerCmd::Reconfigure {
                    caps: absolute_caps.clone(),
                    width: self.window_width(width),
                    height: self.window_height(height),
                })
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
            }
        }
        self.caps = Some(absolute_caps.clone());
        self.width = width;
        self.height = height;
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
                    self.open_window()?;
                    // PTS pacing: hold the frame until its deadline on the elected
                    // clock, or drop it when it is already too late (the QoS
                    // bound) or outside the segment. Unpaced without a clock:
                    // present as fast as the compositor allows.
                    let presented = self.frames_presented.load(Ordering::Relaxed);
                    if !wait_to_present(self.pacer.judge(timing.pts_ns, presented)).await {
                        return Ok(());
                    }
                    let tx = self.cmd_tx.as_ref().ok_or(G2gError::NotConfigured)?;
                    let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
                    tx.send(WorkerCmd::Frame {
                        frame: domain,
                        arrival_ns: timing.arrival_ns,
                        ack: ack_tx,
                    })
                    .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
                    // Block until the worker presents this frame (paced by the
                    // compositor's release of the swapchain image).
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

impl LogSource for WgpuPresentSink {
    fn log_category(&self) -> &'static str {
        short_type_name::<Self>()
    }
    fn log_instance(&self) -> Option<&str> {
        self.log_name.instance()
    }
    fn log_category_override(&self) -> Option<&str> {
        self.log_name.category()
    }
}

/// The wgpu half of the worker: a surface-target [`WgpuSink`] over the window.
struct SurfaceRenderer {
    sink: WgpuSink,
}

impl WindowRenderer for SurfaceRenderer {
    type Frame = MemoryDomain;

    fn present(&mut self, frame: &Self::Frame) -> Result<(), G2gError> {
        self.sink.present_frame(frame)
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.sink.resize(width, height);
    }

    fn reconfigure(&mut self, caps: &Caps) -> Result<(), G2gError> {
        self.sink.configure_pipeline(caps).map(|_| ())
    }
}

/// Build the surface + device + sink on the worker thread (a wgpu device and its
/// surface are bound to the thread that made them, and device creation is async,
/// so this blocks rather than spreading the setup back over the pipeline task).
fn build_renderer(
    conn: &Connection,
    surface: &wl_surface::WlSurface,
    width: u32,
    height: u32,
    caps: &Caps,
) -> Result<SurfaceRenderer, Box<dyn std::error::Error>> {
    // A GPU decoder upstream opens its own device (Vulkan Video needs queues wgpu
    // does not ask for) and its textures bind to no other, so present on that
    // device when one is offered rather than opening a second.
    let adopted = crate::gpu::present_on_producer_device(
        |instance| create_surface_on(instance, conn, surface),
        width,
        height,
    );
    let (ctx, wgpu_surface, config) = match adopted {
        Some(adopted) => adopted,
        None => {
            let (instance, wgpu_surface) = create_surface(conn, surface)?;
            let ctx = g2g_core::runtime::block_on(open_device(instance, &wgpu_surface))
                .map_err(|e| alloc::format!("no wgpu device for this surface: {e:?}"))?;
            let config = wgpu_surface
                .get_default_config(&ctx.adapter, width, height)
                .ok_or("no GPU adapter here can present to this Wayland display")?;
            (ctx, wgpu_surface, config)
        }
    };
    // wgpu answers an error it cannot return with a panic, and the release
    // profile aborts on one, so take those here: the failure names itself on this
    // sink's log category and the pipeline fails on the next frame instead of the
    // process dying.
    ctx.device
        .on_uncaptured_error(Arc::new(|error: wgpu::Error| {
            g2g_error!(
                Target::category(short_type_name::<WgpuPresentSink>()),
                "gpu error while presenting: {error}"
            );
        }));
    wgpu_surface.configure(&ctx.device, &config);
    let mut sink = WgpuSink::with_surface(ctx, wgpu_surface, config);
    sink.configure_pipeline(caps)
        .map_err(|e| alloc::format!("the sink rejected the negotiated caps: {e:?}"))?;
    Ok(SurfaceRenderer { sink })
}

/// Open the device the surface presents from. With the CUDA bridge built in, it
/// is the process-wide external-memory device, so a bridged NVDEC frame lands on
/// the very device this window presents from (a wgpu texture binds nowhere else).
async fn open_device(
    instance: wgpu::Instance,
    surface: &wgpu::Surface<'static>,
) -> Result<GpuContext, G2gError> {
    #[cfg(feature = "cuda-wgpu")]
    {
        let interop =
            Arc::new(crate::cudawgpu::create_interop_device_for_surface(instance, surface).await?);
        let interop = crate::cudawgpu::install_shared_interop_device(interop);
        Ok(GpuContext::from_wgpu(
            interop.instance.clone(),
            interop.adapter.clone(),
            interop.device.clone(),
            interop.queue.clone(),
        ))
    }
    #[cfg(not(feature = "cuda-wgpu"))]
    GpuContext::for_surface(instance, surface).await
}

/// A `wgpu::Surface` over the worker's `wl_surface`, plus the instance it belongs
/// to (an adapter is only usable with the surface's own instance).
fn create_surface(
    conn: &Connection,
    surface: &wl_surface::WlSurface,
) -> Result<(wgpu::Instance, wgpu::Surface<'static>), Box<dyn std::error::Error>> {
    #[cfg(feature = "cuda-wgpu")]
    let instance = crate::cudawgpu::vulkan_instance();
    #[cfg(not(feature = "cuda-wgpu"))]
    let instance = wgpu::Instance::default();

    let wgpu_surface = create_surface_on(&instance, conn, surface)?;
    Ok((instance, wgpu_surface))
}

/// The worker's `wl_surface` as a `wgpu::Surface` on a given instance, so a
/// producer's instance can be adopted (see [`crate::gpu::present_on_producer_device`]).
fn create_surface_on(
    instance: &wgpu::Instance,
    conn: &Connection,
    surface: &wl_surface::WlSurface,
) -> Result<wgpu::Surface<'static>, Box<dyn std::error::Error>> {
    // The libwayland `*mut wl_display` / `*mut wl_proxy` the Vulkan WSI wants as
    // its native handles, from the connection wayland-client owns.
    let display = NonNull::new(conn.backend().display_ptr().cast::<core::ffi::c_void>())
        .ok_or("no wl_display pointer on this connection")?;
    let window = NonNull::new(surface.id().as_ptr().cast::<core::ffi::c_void>())
        .ok_or("no wl_surface pointer for this surface")?;

    // SAFETY: both handles come from the live `Connection` and the `wl_surface`
    // this worker owns. The surface is stored in the renderer, which the worker
    // drops before the window (declaration order in `WindowWorker`), so it never
    // outlives the `wl_surface` it was built over.
    let wgpu_surface = unsafe {
        instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
            raw_display_handle: Some(RawDisplayHandle::Wayland(WaylandDisplayHandle::new(
                display,
            ))),
            raw_window_handle: RawWindowHandle::Wayland(WaylandWindowHandle::new(window)),
        })
    }?;
    Ok(wgpu_surface)
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::{Dim, PushOutcome, Rate, VideoCodec};

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

    /// Caps the renderer cannot present are refused at configure time, before
    /// anything tries to open a window.
    #[test]
    fn caps_it_cannot_present_are_refused_before_a_window_is_opened() {
        let mut sink = WgpuPresentSink::new();
        let compressed = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        assert!(matches!(
            sink.configure_pipeline(&compressed),
            Err(G2gError::CapsMismatch)
        ));
        assert!(sink.cmd_tx.is_none(), "no window worker was started");
    }

    /// A frame before a successful configure is an error, not a window opened on
    /// unknown geometry.
    #[tokio::test]
    async fn a_frame_before_configure_opens_no_window() {
        let mut sink = WgpuPresentSink::new();
        let frame = Frame::new(
            MemoryDomain::System(g2g_core::memory::SystemSlice::from_boxed(
                alloc::vec![0u8; 16].into_boxed_slice(),
            )),
            g2g_core::FrameTiming::default(),
            0,
        );
        assert!(matches!(
            sink.process(PipelinePacket::DataFrame(frame), &mut NullSink)
                .await,
            Err(G2gError::NotConfigured)
        ));
        assert!(sink.cmd_tx.is_none(), "no window worker was started");
    }

    /// A worker that cannot bring its window up reports why, and reports it at
    /// once rather than after the full readiness timeout (the failure path a
    /// missing compositor or an adapter that cannot present takes).
    #[test]
    fn a_worker_that_cannot_start_reports_its_reason() {
        let sink = WgpuPresentSink::new();
        let ready = Handshake::new();
        let (errors_tx, errors) = std::sync::mpsc::channel();
        errors_tx
            .send(String::from(
                "no GPU adapter here can present to this Wayland display",
            ))
            .unwrap();

        let started = Instant::now();
        let reason = sink
            .await_window(&ready, &errors)
            .expect_err("the worker reported a failure");
        assert_eq!(
            reason,
            "no GPU adapter here can present to this Wayland display"
        );
        assert!(
            started.elapsed() < WINDOW_READY_TIMEOUT,
            "reported without waiting out the timeout, took {:?}",
            started.elapsed()
        );
    }

    /// The geometry the window opens at: the video's, unless a property pins it.
    #[test]
    fn window_size_follows_the_video_unless_a_property_pins_it() {
        let mut sink = WgpuPresentSink::new();
        assert_eq!(
            (sink.window_width(1920), sink.window_height(1080)),
            (1920, 1080)
        );
        sink = sink.with_window_size(1280, 720);
        assert_eq!(
            (sink.window_width(1920), sink.window_height(1080)),
            (1280, 720)
        );
    }
}
