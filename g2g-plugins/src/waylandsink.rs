//! Wayland display sink (desktop dev convenience).
//!
//! Opens an `xdg_toplevel` window on the running Wayland compositor and
//! presents NV12 `DataFrame`s into it. The pixel path is software:
//! NV12 → XRGB8888 conversion (in the caps colorimetry's matrix and
//! range) into a `wl_shm` pool, then `attach` + `commit` per frame. Slow
//! but universal; every Wayland compositor supports `wl_shm`.
//!
//! This is the **dev sink**, not the production sink:
//! - Latency is whatever the compositor's frame callback delivers (one
//!   compositor refresh, typically ~16 ms at 60 Hz).
//! - The XRGB8888 conversion runs on the same thread that drives the
//!   pipeline; at 1080p30 the CPU cost is real (each frame is ~2 ms of
//!   YUV→RGB on a modern x86 core).
//! - No GPU upload and no `zwp_linux_dmabuf_v1` zero-copy.
//!
//! The production sink is [`crate::kmssink::KmsSink`], which scans NV12
//! out directly through KMS without colour conversion. Use the KMS sink
//! when you need low latency or are deploying to embedded; use this one
//! to *see what's going on* while iterating on the pipeline.
//!
//! ## Pipeline shape
//!
//! ```text
//! RtspSrc ─► FfmpegH264Dec(Nv12) ─► WaylandSink
//!                                       │
//!                                       └─► xdg_toplevel window
//! ```
//!
//! ## Threading
//!
//! Wayland client types (`Connection`, `EventQueue`, the SCTK state
//! struct) are designed to be single-thread-owned. We honour that by
//! pinning all Wayland state to a dedicated worker thread, spun up at
//! `configure_pipeline` time. The sink struct itself only holds a
//! `calloop` channel sender and a shared atomic counter, both of which
//! are `Send + Sync`. The runner can move us between worker tasks
//! freely.
//!
//! ## Constraints (v1)
//!
//! - NV12 input only.
//! - Mid-stream geometry change tears down the existing worker and
//!   spawns a fresh one (M16 5j). Same-dims `CapsChanged` is a no-op.
//! - No scaling: the window opens at the input video dimensions and
//!   stays there. If the compositor's `configure` event resizes us we
//!   ignore the new bounds (the video keeps drawing at its native
//!   resolution and the compositor letterboxes / clips).
//! - Once the runner elects a clock, each frame is held until its PTS
//!   deadline and dropped if it is late beyond the QoS bound. On top of
//!   that, backpressure is compositor-driven: `process()` blocks until
//!   the compositor's `frame` callback for the previously committed
//!   buffer arrives, so the producer is throttled to refresh.
//! - Window decorations are server-side if the compositor offers them
//!   (KDE, GNOME with the right protocol), otherwise the window is
//!   borderless. v1 doesn't carry CSD.

use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState},
    delegate_compositor, delegate_output, delegate_registry, delegate_shm, delegate_xdg_shell,
    delegate_xdg_window,
    output::{OutputHandler, OutputState},
    reexports::calloop::{
        channel::{channel, Channel, Event as ChanEvent, Sender as CalloopSender},
        EventLoop,
    },
    reexports::calloop_wayland_source::WaylandSource,
    reexports::client::{
        globals::registry_queue_init,
        protocol::{wl_output, wl_shm, wl_surface},
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
    shm::{
        slot::{Buffer, SlotPool},
        Shm, ShmHandler,
    },
};

use crate::clock::wait_to_present;
use crate::worker_ready::Handshake;
use crate::yuvmatrix::YuvRgbMatrix;
use g2g_core::element::{PresentationStats, QosMessage};
use g2g_core::frame::Frame;
use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::meta::Orientation;
#[cfg(feature = "metadata")]
use g2g_core::meta::OrientationMeta;
use g2g_core::metrics::{monotonic_ns, LatencyHistogram, LatencySnapshot};
use g2g_core::{
    AsyncElement, BusHandle, Caps, CapsConstraint, CapsSet, ClockCandidate, ClockPriority,
    ClockSync, Colorimetry, ConfigureOutcome, Dim, ElementMetadata, G2gError, HardwareError,
    OutputSink, PipelineClock, PipelinePacket, PresentationPacer, PropError, PropKind, PropValue,
    PropertySpec, Rate, RawVideoFormat, MAX_LATENESS_PROPERTY, QOS_INTERVAL_PROPERTY,
};

/// Worker-thread message. `Frame` carries the pre-converted XRGB8888
/// bytes (sink-side conversion keeps the worker thread free for Wayland
/// I/O) plus a one-shot `ack` the worker signals once the frame has been
/// committed *and* the compositor's next `frame` callback has fired —
/// that's the signal we use to pace the producer to refresh.
/// `Shutdown` exits the worker's event loop.
enum WorkerCmd {
    Frame(QueuedFrame),
    Shutdown,
}

/// One converted frame on its way to the surface.
struct QueuedFrame {
    bytes: Vec<u8>,
    /// Source-side wall-clock stamp from `FrameTiming::arrival_ns`.
    /// The worker records `monotonic_ns() - arrival_ns` into the
    /// latency histogram when the matching `frame` callback fires.
    /// Zero means the frame was untimed; latency is not recorded.
    arrival_ns: u64,
    /// The turn an upstream `videoflip` left for the compositor to apply
    /// (M1058). `Identity` for a frame with no `OrientationMeta`.
    orientation: Orientation,
    ack: tokio::sync::oneshot::Sender<()>,
}

/// How the sink reacts when the producer pushes faster than the
/// compositor refreshes.
///
/// - `Block` (default): `process()` waits for the matching `frame`
///   callback before returning. Producer is throttled to refresh.
///   No drops, but backpressure propagates upstream. A compositor stops
///   sending `frame` callbacks for a surface it is not painting (fully
///   occluded, minimized, or on no output), so the wait only applies
///   while callbacks are recent: a silent surface stops throttling
///   rather than freezing the branch until the window is exposed again.
/// - `DropOldest`: `process()` returns as soon as the worker accepts
///   the frame. If a previous frame is still awaiting its `frame`
///   callback, the worker overwrites it — the older frame never paints.
///   Use for live sources that prefer freshness over completeness
///   (security cameras, monitoring) and can't tolerate backpressure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PacingPolicy {
    #[default]
    Block,
    DropOldest,
}

/// `frame` callbacks older than this mean the compositor is not painting the
/// surface, and `Block` stops waiting on them. Also the bound on one ack wait,
/// so the first frames after the compositor goes silent pay at most this before
/// the sink notices. Well above any real refresh interval.
const FRAME_CALLBACK_SILENCE_NS: u64 = 250_000_000;

/// What the sink-side struct holds between `process()` calls. We keep
/// only `Send + Sync` handles here so the multi-thread runner can move
/// us between executor tasks.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::waylandsink::{PacingPolicy, WaylandSink};
///
/// let sink = WaylandSink::new()
///     .with_title("preview")
///     .with_pacing(PacingPolicy::DropOldest);
/// ```
/// Raw per-frame samples (glass-to-glass latency, presentation deadline error)
/// stop at this count (8 bytes each, about 2.4 hours at 30 fps). The latency
/// histogram keeps counting past it.
const PER_FRAME_SAMPLE_CAPACITY: usize = 1 << 18;

/// Glass-to-glass histogram plus the raw per-frame samples behind it, so a
/// bench can read exact percentiles instead of log2 bucket edges.
#[derive(Debug)]
struct LatencyRecorder {
    histogram: LatencyHistogram,
    samples: std::sync::Mutex<Vec<u64>>,
}

impl LatencyRecorder {
    fn new() -> Self {
        Self {
            histogram: LatencyHistogram::new(),
            samples: std::sync::Mutex::new(Vec::new()),
        }
    }

    fn record(&self, dur_ns: u64) {
        self.histogram.record(dur_ns);
        let mut samples = self.samples.lock().expect("latency samples lock");
        if samples.len() < PER_FRAME_SAMPLE_CAPACITY {
            samples.push(dur_ns);
        }
    }
}

/// How far the elected clock, the media timeline and the frame deadline have
/// each moved against this host's monotonic clock since the first paced frame,
/// in nanoseconds. All three stay flat on a healthy run:
///
/// - `clock_ns` walking off is the elected clock running at the wrong rate.
/// - `deadline_ns` parting from `media_ns` is the presentation anchor moving
///   under the stream.
/// - `media_ns` sagging while `clock_ns` holds is the sink failing to keep the
///   schedule: the feed ran dry, so frames present on arrival instead of on
///   their deadline.
#[derive(Clone, Copy, Debug)]
struct ScheduleDrift {
    clock_ns: i64,
    media_ns: i64,
    deadline_ns: i64,
}

/// Once-per-second sampler behind [`ScheduleDrift`]. The per-frame trace shows
/// what one frame did; this shows which of the three timelines is losing ground,
/// which is the only way to tell a bad clock from a slow feed.
#[derive(Debug, Default)]
struct ScheduleDriftTrace {
    /// `(monotonic, clock, pts, deadline)` at the first paced frame.
    base: Option<(u64, u64, u64, u64)>,
    last_log_ns: u64,
}

impl ScheduleDriftTrace {
    const INTERVAL_NS: u64 = 1_000_000_000;

    fn sample(
        &mut self,
        mono_ns: u64,
        clock_ns: u64,
        pts_ns: u64,
        deadline_ns: u64,
    ) -> Option<ScheduleDrift> {
        let base = *self
            .base
            .get_or_insert((mono_ns, clock_ns, pts_ns, deadline_ns));
        if self.last_log_ns != 0 && mono_ns.saturating_sub(self.last_log_ns) < Self::INTERVAL_NS {
            return None;
        }
        self.last_log_ns = mono_ns;
        let elapsed = |now: u64, then: u64| now as i64 - then as i64;
        let mono = elapsed(mono_ns, base.0);
        Some(ScheduleDrift {
            clock_ns: elapsed(clock_ns, base.1) - mono,
            media_ns: elapsed(pts_ns, base.2) - mono,
            deadline_ns: elapsed(deadline_ns, base.3) - mono,
        })
    }
}

pub struct WaylandSink {
    title: String,
    app_id: String,
    cmd_tx: Option<CalloopSender<WorkerCmd>>,
    worker: Option<JoinHandle<()>>,
    width: u32,
    height: u32,
    /// NV12 -> XRGB coefficients for the negotiated caps' colorimetry, re-derived
    /// whenever `configure_pipeline` runs.
    matrix: YuvRgbMatrix,
    frames_presented: Arc<AtomicU64>,
    latency: Arc<LatencyRecorder>,
    frames_dropped: Arc<AtomicU64>,
    /// Monotonic time of the compositor's most recent `frame` callback,
    /// written by the worker; how `Block` tells a painting surface from a
    /// silent one.
    last_frame_callback_ns: Arc<AtomicU64>,
    pacing: PacingPolicy,
    /// PTS pacing + QoS late-drop (M173 / M176), shared with the other
    /// synchronizing sinks: the elected clock, the segment mapping, the
    /// presentation anchor, and the lateness bound. Idle until the runner hands
    /// over a clock; the default bound never drops, presenting every frame
    /// however late.
    pacer: PresentationPacer,
    /// Monotonic stamp when the previous frame's present completed, for the
    /// per-frame pacing log (the inter-present gap is what stutter looks like).
    last_present_done_ns: u64,
    /// Signed presentation error per presented frame, nanoseconds: the elected
    /// clock's reading once the frame is up, minus the frame's deadline on that
    /// clock. Positive is late. Empty for an unpaced run (no clock, no
    /// deadlines), capped at `PER_FRAME_SAMPLE_CAPACITY`.
    deadline_errors: Vec<i64>,
    /// Once-per-second clock / media / deadline drift trace, logged under
    /// `G2G_DEBUG`.
    schedule_drift: ScheduleDriftTrace,
}

impl g2g_core::log::LogSource for WaylandSink {
    fn log_category(&self) -> &'static str {
        g2g_core::log::short_type_name::<Self>()
    }
}

impl core::fmt::Debug for WaylandSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WaylandSink")
            .field("title", &self.title)
            .field("app_id", &self.app_id)
            .field("width", &self.width)
            .field("height", &self.height)
            .field(
                "frames_presented",
                &self.frames_presented.load(Ordering::Relaxed),
            )
            .finish()
    }
}

impl Default for WaylandSink {
    fn default() -> Self {
        Self::new()
    }
}

impl WaylandSink {
    pub fn new() -> Self {
        Self {
            title: String::from("glass2glass"),
            app_id: String::from("io.glass2glass.WaylandSink"),
            cmd_tx: None,
            worker: None,
            width: 0,
            height: 0,
            matrix: YuvRgbMatrix::new(Colorimetry::UNKNOWN),
            frames_presented: Arc::new(AtomicU64::new(0)),
            latency: Arc::new(LatencyRecorder::new()),
            frames_dropped: Arc::new(AtomicU64::new(0)),
            last_frame_callback_ns: Arc::new(AtomicU64::new(0)),
            pacing: PacingPolicy::default(),
            pacer: PresentationPacer::new(),
            last_present_done_ns: 0,
            deadline_errors: Vec::new(),
            schedule_drift: ScheduleDriftTrace::default(),
        }
    }

    pub fn with_pacing(mut self, pacing: PacingPolicy) -> Self {
        self.pacing = pacing;
        self
    }

    /// Enable QoS late-drop (M173): once PTS pacing is engaged, a frame already
    /// past its deadline by more than `ns` is dropped instead of presented late,
    /// so the sink catches up rather than accumulating lag. `0` drops any frame
    /// that arrives after its deadline; the default (`u64::MAX`) never drops.
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

    pub fn frames_dropped(&self) -> u64 {
        self.frames_dropped.load(Ordering::Relaxed)
    }

    /// Frames dropped by QoS late-drop (past their deadline beyond the configured
    /// bound). Distinct from [`frames_dropped`](Self::frames_dropped), the
    /// compositor-side `DropOldest` count.
    pub fn late_dropped(&self) -> u64 {
        self.pacer.late_dropped()
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

    /// Snapshot of glass-to-glass latency: source-side
    /// `FrameTiming::arrival_ns` to the compositor's `frame` callback
    /// that confirms our commit. Only frames whose timing was stamped
    /// upstream contribute; an untimed pipeline reports `count = 0`.
    pub fn latency_snapshot(&self) -> LatencySnapshot {
        self.latency.histogram.snapshot()
    }

    /// The raw per-frame samples behind [`latency_snapshot`], nanoseconds in
    /// presentation order, capped at `PER_FRAME_SAMPLE_CAPACITY`.
    ///
    /// [`latency_snapshot`]: Self::latency_snapshot
    pub fn latency_samples(&self) -> Vec<u64> {
        self.latency
            .samples
            .lock()
            .expect("latency samples lock")
            .clone()
    }

    /// Signed presentation error per presented frame, nanoseconds in
    /// presentation order: the elected clock's reading once the frame is on
    /// screen minus the deadline the pacer held it to. Positive is late. Empty
    /// for an unpaced run, since a clockless sink presents on arrival and has no
    /// deadline to miss. Capped at `PER_FRAME_SAMPLE_CAPACITY`.
    ///
    /// In an A/V graph the elected clock is the audio sink's, so this is the
    /// video-against-audio sync error: its drift over a long run is lip sync.
    pub fn deadline_error_samples(&self) -> Vec<i64> {
        self.deadline_errors.clone()
    }

    /// The elected clock the runner handed over, `None` until one arrives (or
    /// for a graph that elects none). An A/V graph hands the video sink the
    /// audio sink's `DriftClock`, so this is how a test confirms the video is
    /// slaved to the audio timeline.
    pub fn clock_sync(&self) -> Option<&ClockSync> {
        self.pacer.clock_sync()
    }

    fn shutdown(&mut self) {
        if let Some(tx) = self.cmd_tx.take() {
            // Best-effort — if the worker is already gone the send fails
            // silently and that's the outcome we want.
            let _ = tx.send(WorkerCmd::Shutdown);
        }
        if let Some(join) = self.worker.take() {
            let _ = join.join();
        }
    }
}

impl Drop for WaylandSink {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Monotonic wall-clock the sink offers as a pipeline clock. Wraps
/// `metrics::monotonic_ns()` so the sink's timeline matches the
/// source-side `arrival_ns` stamps used by the latency histogram.
///
/// We register at `Provider` priority so a `LiveSource` (RTSP, camera)
/// still wins election when present, but in absence of one the sink
/// becomes the reference clock — the right answer for an audio-less
/// video-only pipeline once A/V sync arrives. Not yet vsync-predicting:
/// `now_ns()` is straight monotonic, no frame-callback feedback. That's
/// the upgrade needed before audio sync; tracked as Plan-1 Step 3+.
#[derive(Debug)]
struct WaylandClock;
impl PipelineClock for WaylandClock {
    fn now_ns(&self) -> u64 {
        monotonic_ns()
    }
}

impl AsyncElement for WaylandSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn provide_clock(&self) -> Option<ClockCandidate> {
        Some(ClockCandidate::new(
            ClockPriority::Provider,
            alloc::sync::Arc::new(WaylandClock),
        ))
    }

    /// This sink blits from host memory, so it takes system frames only. The
    /// allocation cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System)
    }

    /// The compositor turns the picture for us through
    /// `wl_surface::set_buffer_transform`, so a `videoflip` upstream should
    /// attach an `OrientationMeta` rather than remap every pixel (M1058). Needs
    /// the `metadata` feature: without it a frame has nowhere to carry the turn.
    fn absorbs_orientation(&self) -> bool {
        cfg!(feature = "metadata")
    }

    /// Adopt the elected clock + base time so frames present at their PTS
    /// deadline. When the elected clock is our own `WaylandClock` (the common
    /// audio-less case) its `now_ns()` shares the monotonic domain we sleep on.
    fn set_clock_sync(&mut self, sync: ClockSync) {
        self.pacer.set_clock_sync(sync);
    }

    /// Relay a late drop upstream (M174): the runner forwards this onto the
    /// incoming link, where the producer observes it as `PushOutcome::Qos` and
    /// can shed load so the sink stops running behind.
    fn take_qos(&mut self) -> Option<QosMessage> {
        self.pacer.take_qos()
    }

    fn presentation_stats(&self) -> Option<PresentationStats> {
        Some(PresentationStats {
            presented: self.frames_presented(),
            dropped: self.frames_dropped(),
            late_dropped: self.late_dropped(),
        })
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // Pass-through at negotiation; the real NV12 validation happens in
        // `configure_pipeline`. With the decoder native (`DerivedOutput`),
        // the solver assigns this link NV12 directly, so configure receives
        // NV12 at startup rather than the decoder's pre-decode H.264 caps.
        Ok(upstream_caps.clone())
    }

    /// M16 step 5: native NV12-only sink constraint. The solver intersects
    /// this against the upstream decoder's NV12 `DerivedOutput` and lands
    /// fixed NV12 on the link at startup, so a non-NV12 (undecoded) display
    /// chain fails loud in negotiation rather than reaching
    /// `configure_pipeline`. Geometry stays open (`Dim::Any`); the decoder
    /// fixates it, and a mid-stream change rebuilds the worker (5j).
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        // NV12 only. Now that every decoder is a native `DerivedOutput`,
        // the solver lands NV12 on this link at startup, so the old
        // accept-H.264-as-no-op workaround is gone: a non-NV12 sink input
        // is a real pipeline error (e.g. an undecoded display chain) and
        // fails loud here.
        let (w, h, colorimetry) = match absolute_caps {
            Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(w),
                height: Dim::Fixed(h),
                colorimetry,
                ..
            } => (*w, *h, *colorimetry),
            _ => return Err(G2gError::CapsMismatch),
        };
        if w % 2 != 0 || h % 2 != 0 {
            return Err(G2gError::CapsMismatch);
        }
        // A mid-stream colorimetry refinement (the parser's VUI colour
        // description) only re-derives the convert table: the worker's surface
        // is unaffected.
        self.matrix = YuvRgbMatrix::new(colorimetry);

        // Mid-stream geometry change: same dims is a no-op; different
        // dims means we tear down the existing worker and spawn a fresh
        // one. M16 5j: enables decoder→sink chains where the initial
        // NV12 caps carry placeholder dims (e.g. RtspSrc's `Range`
        // workaround #1, fixated to min) and the real geometry lands
        // via a mid-stream `CapsChanged` after SPS parse.
        if self.worker.is_some() {
            if w == self.width && h == self.height {
                return Ok(ConfigureOutcome::Accepted);
            }
            self.shutdown();
            // fall through to fresh-worker spawn below.
        }

        let (tx, rx) = channel::<WorkerCmd>();
        let presented = Arc::clone(&self.frames_presented);
        let dropped = Arc::clone(&self.frames_dropped);
        let latency = Arc::clone(&self.latency);
        // Seeded with now so the just-mapped surface gets one silence window of
        // ordinary blocking before its first callback has to arrive.
        self.last_frame_callback_ns
            .store(monotonic_ns(), Ordering::Relaxed);
        let last_frame_callback_ns = Arc::clone(&self.last_frame_callback_ns);
        let title = self.title.clone();
        let app_id = self.app_id.clone();

        // Synchronous handshake: the worker signals readiness once the
        // compositor's first `configure` lands. Until then `process()`
        // would be racing against an unmapped surface.
        let ready = Arc::new(Handshake::new());
        let ready_for_worker = Arc::clone(&ready);

        let join = thread::Builder::new()
            .name(String::from("g2g-waylandsink"))
            .spawn(move || {
                if let Err(e) = worker_main(
                    w,
                    h,
                    title,
                    app_id,
                    rx,
                    presented,
                    dropped,
                    latency,
                    last_frame_callback_ns,
                    ready_for_worker,
                ) {
                    std::eprintln!("g2g-waylandsink worker error: {e:?}");
                }
            })
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

        // Bounded wait: a hung compositor mustn't lock us up forever.
        if !ready.wait(Duration::from_secs(5)) {
            // Tell the worker to give up; if it already crashed, the
            // send fails and join will pick up the panic.
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

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Wayland video sink",
            "Sink/Video",
            "Presents NV12 video to a Wayland surface (software SHM)",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new("title", PropKind::Str, "window title").with_default("glass2glass"),
            PropertySpec::new("app-id", PropKind::Str, "Wayland xdg app id")
                .with_default("io.glass2glass.WaylandSink"),
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

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let orientation = frame_orientation(&frame);
                    let Frame { domain, timing, .. } = frame;
                    let slice =
                        domain.require_system_slice(g2g_core::log::short_type_name::<Self>())?;

                    // PTS pacing: hold the frame until its running-time deadline
                    // on the elected clock, or drop it if it is already too late
                    // (the QoS bound) or outside the segment. Unpaced without a
                    // clock: present immediately (pre-sync).
                    let t_in = monotonic_ns();
                    let presented = self.frames_presented.load(Ordering::Relaxed);
                    let pace = self.pacer.judge(timing.pts_ns, presented);
                    // Taken here rather than re-derived after the present: the
                    // anchor this frame was judged against is the only one its
                    // error means anything against.
                    let deadline_ns = self.pacer.last_deadline_ns();
                    // Positive slack the pacer asked us to sleep; 0 = already due.
                    let wait_ns = match pace {
                        g2g_core::Pace::Wait(n) => n,
                        _ => 0,
                    };
                    if !wait_to_present(pace).await {
                        g2g_core::g2g_log!(self, "late-drop pts={}", timing.pts_ns);
                        return Ok(());
                    }
                    let t_ready = monotonic_ns();

                    // M760: offload the NV12 -> XRGB8888 convert (pure CPU pixel
                    // math, not the Wayland calls) onto tokio's blocking pool so
                    // the cooperative runner keeps servicing sibling arms while it
                    // runs. Own the input bytes so the closure is Send.
                    #[cfg(feature = "offload")]
                    let xrgb = {
                        let (w, h) = (self.width, self.height);
                        let matrix = self.matrix;
                        let src: Vec<u8> = slice.to_vec();
                        crate::offload::run_blocking(move || nv12_to_xrgb8888(&src, w, h, &matrix))
                            .await?
                    };
                    #[cfg(not(feature = "offload"))]
                    let xrgb = nv12_to_xrgb8888(slice, self.width, self.height, &self.matrix)?;
                    let tx = self.cmd_tx.as_ref().ok_or(G2gError::NotConfigured)?;
                    let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
                    tx.send(WorkerCmd::Frame(QueuedFrame {
                        bytes: xrgb,
                        arrival_ns: timing.arrival_ns,
                        orientation,
                        ack: ack_tx,
                    }))
                    .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
                    match self.pacing {
                        PacingPolicy::Block => {
                            // Wait for the compositor's `frame` callback for
                            // this commit, but only while callbacks are
                            // arriving at all: a surface the compositor is not
                            // painting gets none, and gating on one would
                            // freeze the branch until the window is exposed.
                            // RecvError means the worker dropped the ack
                            // (shutdown / crash) — treat as a hardware fault.
                            let silence =
                                core::time::Duration::from_nanos(FRAME_CALLBACK_SILENCE_NS);
                            let last_callback_ns =
                                self.last_frame_callback_ns.load(Ordering::Relaxed);
                            if monotonic_ns().saturating_sub(last_callback_ns)
                                < FRAME_CALLBACK_SILENCE_NS
                            {
                                match tokio::time::timeout(silence, ack_rx).await {
                                    Ok(acked) => acked
                                        .map_err(|_| G2gError::Hardware(HardwareError::Other))?,
                                    Err(_) => g2g_core::g2g_log!(
                                        self,
                                        "frame callback overdue, presenting unthrottled"
                                    ),
                                }
                            }
                        }
                        PacingPolicy::DropOldest => {
                            // Fire-and-forget: producer keeps moving.
                            // If the previous frame's ack is still
                            // outstanding when this one is drawn, the
                            // worker drops it and bumps frames_dropped.
                            drop(ack_rx);
                        }
                    }
                    // Per-frame pacing trace: where this frame's wall time went
                    // (pacer sleep, convert+queue, compositor ack) and the gap
                    // since the previous present, the number stutter shows up in.
                    let t_done = monotonic_ns();
                    let gap = t_done.saturating_sub(self.last_present_done_ns);
                    self.last_present_done_ns = t_done;
                    // How far off its deadline the frame actually landed, read on
                    // the elected clock (not `t_done`, which is this host's
                    // monotonic timeline, a different one whenever audio is the
                    // master).
                    let clock_now_ns = self.pacer.clock_sync().map(|s| s.now_ns());
                    if let (Some(deadline), Some(clock_now)) = (deadline_ns, clock_now_ns) {
                        let error = clock_now as i64 - deadline as i64;
                        if self.deadline_errors.len() < PER_FRAME_SAMPLE_CAPACITY {
                            self.deadline_errors.push(error);
                        }
                        // Once a second: which of the three timelines is losing
                        // ground against wall time. See `ScheduleDrift`.
                        if let Some(drift) =
                            self.schedule_drift
                                .sample(t_done, clock_now, timing.pts_ns, deadline)
                        {
                            g2g_core::g2g_log!(
                                self,
                                "pace clock-mono={}us media-mono={}us deadline-mono={}us \
                                 wait={}us age_in={}us late_dropped={}",
                                drift.clock_ns / 1_000,
                                drift.media_ns / 1_000,
                                drift.deadline_ns / 1_000,
                                wait_ns / 1_000,
                                t_in.saturating_sub(timing.arrival_ns) / 1_000,
                                self.pacer.late_dropped()
                            );
                        }
                    }
                    g2g_core::g2g_log!(
                        self,
                        "pts={} age_in={}us wait={}us slept={}us conv+ack={}us gap={}us",
                        timing.pts_ns,
                        t_in.saturating_sub(timing.arrival_ns) / 1_000,
                        wait_ns / 1_000,
                        t_ready.saturating_sub(t_in) / 1_000,
                        t_done.saturating_sub(t_ready) / 1_000,
                        gap / 1_000
                    );
                    Ok(())
                }
                PipelinePacket::Segment(seg) => {
                    // Track the playback segment so PTS maps to running time
                    // (correct across a seek).
                    self.pacer.set_segment(seg);
                    Ok(())
                }
                PipelinePacket::Flush => {
                    // Seek flush: re-anchor presentation to the post-seek
                    // timeline; the following Segment installs the new mapping.
                    // The next frame first-frame-anchors (present the seek target
                    // now), not at the stale play-edge base time (M176).
                    self.pacer.flush();
                    Ok(())
                }
                PipelinePacket::CapsChanged(_) => Ok(()),
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
// Worker thread
// =================================================================

struct WorkerState {
    registry_state: RegistryState,
    output_state: OutputState,
    shm: Shm,
    pool: SlotPool,
    buffer: Option<Buffer>,
    window: Window,
    qh: QueueHandle<WorkerState>,
    width: u32,
    height: u32,
    configured: bool,
    exit: bool,
    ready: Option<Arc<Handshake>>,
    presented: Arc<AtomicU64>,
    dropped: Arc<AtomicU64>,
    latency: Arc<LatencyRecorder>,
    /// Shared with the producer, stamped on every `frame` callback: how
    /// `Block` pacing knows the compositor is still painting this surface.
    last_frame_callback_ns: Arc<AtomicU64>,
    /// Frame queued before the surface is mappable. Once `configure`
    /// lands we drain this into the first draw. With blocking pacing the
    /// producer is throttled to one in-flight frame, so under steady
    /// state this is None.
    pending: Option<QueuedFrame>,
    /// Buffer transform currently committed on the surface. `set_buffer_transform`
    /// is double-buffered state the compositor keeps, so it is re-issued only
    /// when the descriptor on the frames changes.
    orientation: Orientation,
    /// Ack for the most recently committed frame plus its source-side
    /// arrival timestamp. Signalled when the compositor's matching
    /// `frame` callback fires, at which point we record the latency.
    pending_ack: Option<(u64, tokio::sync::oneshot::Sender<()>)>,
}

// The worker owns the whole Wayland-thread state; threading it as one params
// struct would only move the argument list, not reduce it.
#[allow(clippy::too_many_arguments)]
fn worker_main(
    width: u32,
    height: u32,
    title: String,
    app_id: String,
    rx: Channel<WorkerCmd>,
    presented: Arc<AtomicU64>,
    dropped: Arc<AtomicU64>,
    latency: Arc<LatencyRecorder>,
    last_frame_callback_ns: Arc<AtomicU64>,
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
    let shm = Shm::bind(&globals, &qh)?;

    let surface = compositor.create_surface(&qh);
    let window = xdg_shell.create_window(surface, WindowDecorations::RequestServer, &qh);
    window.set_title(&title);
    window.set_app_id(&app_id);
    window.set_min_size(Some((width, height)));
    window.commit();

    // Allocate enough for a single XRGB8888 buffer at the input dims;
    // SlotPool grows internally if we double-buffer below.
    let pool = SlotPool::new((width * height * 4) as usize, &shm)?;

    let mut state = WorkerState {
        registry_state: RegistryState::new(&globals),
        output_state: OutputState::new(&globals, &qh),
        shm,
        pool,
        buffer: None,
        window,
        qh: qh.clone(),
        width,
        height,
        configured: false,
        exit: false,
        ready: Some(ready),
        presented,
        dropped,
        latency,
        last_frame_callback_ns,
        pending: None,
        orientation: Orientation::Identity,
        pending_ack: None,
    };

    // Wire the cmd channel into calloop so we wake on frame arrival.
    loop_handle.insert_source(rx, |event, _, state: &mut WorkerState| match event {
        ChanEvent::Msg(WorkerCmd::Frame(queued)) => {
            // Producer is blocked on `ack` until our `frame` callback
            // fires, so we should only ever see one in flight. If the
            // surface isn't mappable yet, stash it; otherwise draw now.
            if state.configured {
                state.draw(queued);
            } else {
                state.pending = Some(queued);
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

impl WorkerState {
    /// Copy `bytes` into a `SlotPool` buffer, request a `frame` callback
    /// (so the compositor tells us when it's ready for the next one),
    /// and commit. The producer's `ack` is stashed in `pending_ack`; we
    /// signal it when the matching `frame` callback fires in
    /// `CompositorHandler::frame`.
    fn draw(&mut self, queued: QueuedFrame) {
        let QueuedFrame {
            bytes,
            arrival_ns,
            orientation,
            ack,
        } = queued;
        let width = self.width as i32;
        let height = self.height as i32;
        let stride = self.width as i32 * 4;

        // Allocate or reuse the buffer. If the compositor still owns the
        // last one we double-buffer.
        let buffer = self.buffer.get_or_insert_with(|| {
            self.pool
                .create_buffer(width, height, stride, wl_shm::Format::Xrgb8888)
                .expect("create_buffer")
                .0
        });
        let canvas = match self.pool.canvas(buffer) {
            Some(canvas) => canvas,
            None => {
                let (new_buf, canvas) = self
                    .pool
                    .create_buffer(width, height, stride, wl_shm::Format::Xrgb8888)
                    .expect("create_buffer (double-buffer)");
                *buffer = new_buf;
                canvas
            }
        };

        let needed = (self.width * self.height * 4) as usize;
        if bytes.len() != needed {
            // Should never happen — sink-side conversion sizes exactly,
            // and dims are fixed at configure time. Drop quietly *and*
            // release the producer so we don't deadlock the pipeline.
            let _ = ack.send(());
            return;
        }
        canvas[..needed].copy_from_slice(&bytes[..needed]);

        let surface = self.window.wl_surface();
        if orientation != self.orientation {
            surface.set_buffer_transform(buffer_transform(orientation));
            // The surface is what the compositor lays out, and a turn that
            // transposes the picture transposes it. The buffer, and
            // `damage_buffer` with it, stays in buffer coordinates.
            let (surface_w, surface_h) = if orientation.swaps_dims() {
                (self.height, self.width)
            } else {
                (self.width, self.height)
            };
            self.window.set_min_size(Some((surface_w, surface_h)));
        }
        // Subscribe to the compositor's `frame` callback for this commit.
        // SCTK's CompositorHandler::frame routes by the WlSurface udata,
        // so we pass a clone of the surface as the callback's user data.
        surface.frame(&self.qh, surface.clone());
        surface.damage_buffer(0, 0, width, height);
        buffer.attach_to(surface).expect("attach_to");
        self.window.commit();
        self.orientation = orientation;
        self.presented.fetch_add(1, Ordering::Relaxed);

        // If a prior ack is still outstanding the compositor never sent
        // us a frame callback for it before we drew over it. Release the
        // ack (under Block this is unreachable; under DropOldest it's
        // expected and counted).
        if let Some((_, stale)) = self.pending_ack.take() {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            let _ = stale.send(());
        }
        self.pending_ack = Some((arrival_ns, ack));
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
        // The compositor is ready for the next frame. Record the
        // glass-to-glass delta (source ingest -> on-screen), then
        // release the producer blocked on this commit's ack.
        self.last_frame_callback_ns
            .store(monotonic_ns(), Ordering::Relaxed);
        if let Some((arrival_ns, ack)) = self.pending_ack.take() {
            if arrival_ns != 0 {
                let now = monotonic_ns();
                if now >= arrival_ns {
                    self.latency.record(now - arrival_ns);
                }
            }
            let _ = ack.send(());
        }
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
        // Ignore the compositor's suggested size — we render at the
        // input video dims and let the compositor letterbox/clip.
        let was_first = !self.configured;
        self.configured = true;
        if was_first {
            // Tell the sink-side handshake that the window is mappable.
            if let Some(ready) = self.ready.take() {
                ready.notify();
            }
            // Drain any frame that arrived before we were mappable.
            if let Some(queued) = self.pending.take() {
                self.draw(queued);
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

impl ShmHandler for WorkerState {
    fn shm_state(&mut self) -> &mut Shm {
        &mut self.shm
    }
}

impl ProvidesRegistryState for WorkerState {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    registry_handlers![OutputState,];
}

delegate_compositor!(WorkerState);
delegate_output!(WorkerState);
delegate_shm!(WorkerState);
delegate_xdg_shell!(WorkerState);
delegate_xdg_window!(WorkerState);
delegate_registry!(WorkerState);

// =================================================================
// Orientation descriptor -> wl_surface buffer transform
// =================================================================

/// The turn an upstream `videoflip` left for us to apply, `Identity` when the
/// frame carries no descriptor.
#[cfg(feature = "metadata")]
fn frame_orientation(frame: &Frame) -> Orientation {
    frame
        .meta
        .get::<OrientationMeta>()
        .map_or(Orientation::Identity, |m| m.orientation)
}

#[cfg(not(feature = "metadata"))]
fn frame_orientation(_frame: &Frame) -> Orientation {
    Orientation::Identity
}

/// The `wl_surface::set_buffer_transform` argument that makes the compositor
/// display a buffer turned by `orientation`.
///
/// `set_buffer_transform` takes "the transformation that the client has already
/// applied to the content of the buffer", and the compositor applies the
/// inverse; `wl_output::transform`'s rotations are counter-clockwise, and its
/// `flipped_*` entries are "an initial flip around a vertical axis followed by
/// rotation" (wayland.xml). An `OrientationMeta` says the opposite thing: the
/// turn a consumer still has to apply. So the argument is the inverse of the
/// descriptor, which moves only the two quarter rotations (every mirror is its
/// own inverse).
fn buffer_transform(orientation: Orientation) -> wl_output::Transform {
    match orientation.inverse() {
        Orientation::Identity => wl_output::Transform::Normal,
        // 90 counter-clockwise.
        Orientation::Rotate90Ccw => wl_output::Transform::_90,
        Orientation::Rotate180 => wl_output::Transform::_180,
        Orientation::Rotate90Cw => wl_output::Transform::_270,
        // Flip around a vertical axis, i.e. left to right.
        Orientation::HorizontalMirror => wl_output::Transform::Flipped,
        // That flip then 90 counter-clockwise is the upper-left diagonal.
        Orientation::Transpose => wl_output::Transform::Flipped90,
        Orientation::VerticalMirror => wl_output::Transform::Flipped180,
        Orientation::Transverse => wl_output::Transform::Flipped270,
    }
}

// =================================================================
// NV12 -> XRGB8888
// =================================================================

/// Convert a packed NV12 source buffer (`width * height` Y plane
/// followed by `width * height / 2` UV plane, interleaved as U,V,U,V)
/// into a packed XRGB8888 buffer (`width * height * 4` bytes, little-
/// endian per pixel: `[B, G, R, 0xFF]`). `matrix` carries the negotiated
/// colorimetry's coefficients.
// The `col` index drives both the luma read and the subsampled-chroma pair
// arithmetic (`col / 2`), so an iterator rewrite would not be clearer.
#[allow(clippy::needless_range_loop)]
fn nv12_to_xrgb8888(
    src: &[u8],
    width: u32,
    height: u32,
    matrix: &YuvRgbMatrix,
) -> Result<Vec<u8>, G2gError> {
    let w = width as usize;
    let h = height as usize;
    let y_size = w * h;
    let uv_size = w * (h / 2);
    if src.len() < y_size + uv_size {
        return Err(G2gError::CapsMismatch);
    }

    let mut out = alloc::vec![0u8; w * h * 4];
    let (y_plane, uv_plane) = src.split_at(y_size);

    for row in 0..h {
        let y_row = &y_plane[row * w..(row + 1) * w];
        let uv_row = &uv_plane[(row / 2) * w..(row / 2) * w + w];
        let dst_row_off = row * w * 4;
        for col in 0..w {
            let y = y_row[col] as i32;
            // UV are subsampled 2x horizontally; pair index = col / 2.
            let uv_pair = (col / 2) * 2;
            let u = uv_row[uv_pair] as i32;
            let v = uv_row[uv_pair + 1] as i32;

            let (r, g, b) = matrix.yuv_to_rgb(y, u, v);

            let dst = dst_row_off + col * 4;
            out[dst] = b as u8;
            out[dst + 1] = g as u8;
            out[dst + 2] = r as u8;
            out[dst + 3] = 0xFF;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::{BusMessage, Rate, VideoCodec};

    /// Pins all eight descriptors to the transform that makes the compositor
    /// show the picture the way the descriptor asked for. The turn is the
    /// inverse of the argument, so getting the two quarter rotations backwards
    /// (the easy mistake, `wl_output`'s names are counter-clockwise) shows up
    /// here rather than upside down on a screen.
    #[test]
    fn buffer_transform_inverts_the_descriptor() {
        use wl_output::Transform;
        assert_eq!(buffer_transform(Orientation::Identity), Transform::Normal);
        assert_eq!(buffer_transform(Orientation::Rotate90Cw), Transform::_90);
        assert_eq!(buffer_transform(Orientation::Rotate180), Transform::_180);
        assert_eq!(buffer_transform(Orientation::Rotate90Ccw), Transform::_270);
        assert_eq!(
            buffer_transform(Orientation::HorizontalMirror),
            Transform::Flipped
        );
        assert_eq!(
            buffer_transform(Orientation::Transpose),
            Transform::Flipped90
        );
        assert_eq!(
            buffer_transform(Orientation::VerticalMirror),
            Transform::Flipped180
        );
        assert_eq!(
            buffer_transform(Orientation::Transverse),
            Transform::Flipped270
        );
    }

    /// The sink is the one in-tree element that answers a `videoflip`'s
    /// question, so this is what keeps the rotation deferred.
    #[test]
    fn advertises_that_it_absorbs_orientation() {
        assert_eq!(
            WaylandSink::new().absorbs_orientation(),
            cfg!(feature = "metadata")
        );
    }

    /// The sink converts on the CPU into `wl_shm`, so it has to tell a GPU decoder to
    /// download. The allocation cascade reads this declaration, so getting it
    /// wrong means `nvdec ! waylandsink` takes a device pointer and dies with
    /// `UnsupportedDomain`.
    #[test]
    fn declares_system_memory_only() {
        assert_eq!(
            WaylandSink::new().input_domains(),
            DomainSet::only(MemoryDomainKind::System)
        );
    }

    #[test]
    fn intercept_passes_through_any_format() {
        // Negotiation-time intercept is pass-through; the NV12 requirement
        // is enforced in `configure_pipeline`. (With a native decoder the
        // solver hands this link NV12 anyway.)
        let sink = WaylandSink::new();
        let h264 = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        assert_eq!(sink.intercept_caps(&h264), Ok(h264));
    }

    #[test]
    fn intercept_passes_through_nv12() {
        let sink = WaylandSink::new();
        let nv12 = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(1280),
            height: Dim::Fixed(720),
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        assert_eq!(sink.intercept_caps(&nv12), Ok(nv12));
    }

    #[test]
    fn caps_constraint_is_accepts_nv12_any() {
        // M16 step 5: native sink constraint accepts NV12 at any geometry,
        // so a fully-native decoder->sink chain rejects non-NV12 in the
        // solver rather than via the dynamic intercept callback.
        let sink = WaylandSink::new();
        let CapsConstraint::Accepts(set) = sink.caps_constraint_as_sink() else {
            panic!("expected Accepts");
        };
        assert_eq!(
            set.alternatives(),
            &[Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
                interlace: g2g_core::Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN
            }]
        );
    }

    #[test]
    fn configure_rejects_non_nv12() {
        let mut sink = WaylandSink::new();
        let h264 = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        // A native decoder lands NV12 on this link; a non-NV12 sink input
        // is a real error (e.g. an undecoded display chain), not a no-op.
        assert_eq!(
            sink.configure_pipeline(&h264).err(),
            Some(G2gError::CapsMismatch)
        );
        assert!(
            sink.worker.is_none(),
            "no worker should be spawned on rejected caps"
        );
    }

    #[test]
    fn set_clock_sync_enables_pts_pacing() {
        // Without a clock the sink presents ASAP (pre-sync); after the runner
        // hands it the elected clock, PTS pacing engages.
        let mut sink = WaylandSink::new();
        assert!(!sink.pacer.is_paced());
        let sync = ClockSync::new(Arc::new(WaylandClock), 0);
        AsyncElement::set_clock_sync(&mut sink, sync);
        assert!(sink.pacer.is_paced(), "clock sync stored, PTS pacing on");
    }

    #[test]
    fn qos_lateness_decision_respects_the_bound() {
        // Default: never too late (u64::MAX bound).
        let sink = WaylandSink::new();
        assert!(
            !sink.pacer.is_too_late(0, u64::MAX),
            "default bound never drops"
        );

        // Bound 0: any frame past its deadline is too late.
        let strict = WaylandSink::new().with_max_lateness_ns(0);
        assert!(!strict.pacer.is_too_late(100, 100), "on time is not late");
        assert!(
            strict.pacer.is_too_late(100, 101),
            "1ns past the deadline is late"
        );

        // Bound N: late only once past the deadline by more than N.
        let tol = WaylandSink::new().with_max_lateness_ns(10);
        assert!(!tol.pacer.is_too_late(100, 110), "within tolerance");
        assert!(tol.pacer.is_too_late(100, 111), "beyond tolerance");
    }

    /// A clock whose `now_ns` the test drives by hand.
    struct ManualClock(Arc<AtomicU64>);
    impl PipelineClock for ManualClock {
        fn now_ns(&self) -> u64 {
            self.0.load(Ordering::Relaxed)
        }
    }

    /// A no-op downstream (a sink has none, but `process` takes one).
    struct NullOut;
    impl OutputSink for NullOut {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<g2g_core::PushOutcome, G2gError>> {
            packet_slot.take();
            core::task::Poll::Ready(Ok(g2g_core::PushOutcome::Accepted))
        }
    }

    fn nv12_frame(pts_ns: u64) -> Frame {
        use g2g_core::frame::FrameTiming;
        use g2g_core::memory::{MemoryDomain, SystemSlice};
        Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
            timing: FrameTiming {
                pts_ns,
                ..FrameTiming::default()
            },
            sequence: 0,
            meta: Default::default(),
        }
    }

    #[tokio::test]
    async fn qos_drops_a_late_frame_and_posts_to_the_bus() {
        // A late frame is dropped before any compositor I/O, so this exercises
        // the QoS path without a real Wayland window. The play anchor is stamped
        // at clock 0, so a PTS-0 frame is due at 0 and the clock is already past
        // it.
        use g2g_core::clock::PlayAnchor;
        let (bus, handle) = g2g_core::Bus::new(4);
        let clock = Arc::new(AtomicU64::new(1_000_000));
        let anchor = PlayAnchor::new();
        anchor.stamp(0);
        let mut sink = WaylandSink::new().with_max_lateness_ns(0).with_bus(handle);
        AsyncElement::set_clock_sync(
            &mut sink,
            ClockSync::with_play_anchor(Arc::new(ManualClock(clock)), 0, anchor),
        );

        // Clock is 1 ms; a frame with deadline 0 is 1 ms late (> 0 bound).
        let mut out = NullOut;
        sink.process(PipelinePacket::DataFrame(nv12_frame(0)), &mut out)
            .await
            .unwrap();

        assert_eq!(sink.late_dropped(), 1, "the late frame was dropped");
        // M174 relay: the runner reads this and pushes it upstream, where the
        // producer sees `PushOutcome::Qos`.
        let upstream = AsyncElement::take_qos(&mut sink).expect("upstream QoS report");
        assert_eq!(upstream.jitter_ns, 1_000_000);
        assert!(
            AsyncElement::take_qos(&mut sink).is_none(),
            "the report is consumed once"
        );
        match bus.try_recv() {
            Some(BusMessage::Qos {
                running_time_ns,
                jitter_ns,
                dropped,
                ..
            }) => {
                assert_eq!(running_time_ns, 0, "deadline reported");
                assert_eq!(jitter_ns, 1_000_000, "1 ms late");
                assert_eq!(dropped, 1, "cumulative drop count");
            }
            other => panic!("expected a Qos message, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn qos_default_does_not_drop() {
        // Default bound (u64::MAX): an on-time frame is not dropped. The anchor is
        // set on this first frame so its deadline equals now; it then proceeds to
        // present (no window here, so we only assert it was not QoS-dropped).
        let clock = Arc::new(AtomicU64::new(5_000_000));
        let mut sink = WaylandSink::new();
        AsyncElement::set_clock_sync(&mut sink, ClockSync::new(Arc::new(ManualClock(clock)), 0));
        // First frame anchors at now, so it is never late under any bound.
        assert!(
            !sink.pacer.is_too_late(0, 5_000_000),
            "anchored first frame on time"
        );
        assert_eq!(sink.late_dropped(), 0);
    }

    #[test]
    fn running_time_uses_pts_without_segment() {
        // No segment: PTS is the running time directly.
        let sink = WaylandSink::new();
        assert_eq!(sink.pacer.running_time(50_000_000), Some(50_000_000));
    }

    #[test]
    fn running_time_maps_through_segment_and_clips() {
        // Accurate seek to 70 ms: a frame before the target is clipped (None);
        // an at/after-target frame maps to running time. Mirrors SyncSink (M149).
        let mut sink = WaylandSink::new();
        let seg = g2g_core::Segment::for_flush_seek(&g2g_core::Seek::flush_to(70_000_000), None);
        sink.pacer.set_segment(seg);
        assert_eq!(
            sink.pacer.running_time(66_000_000),
            None,
            "pre-target frame clips"
        );
        assert_eq!(
            sink.pacer.running_time(70_000_000),
            Some(0),
            "the target frame is running-time zero after a flushing seek"
        );
    }

    #[test]
    fn nv12_to_xrgb_yields_correct_byte_count() {
        // 4x2 NV12: Y=8 bytes, UV=4 bytes. Output = 4*2*4 = 32 bytes.
        let src = alloc::vec![16u8; 12];
        let out = nv12_to_xrgb8888(&src, 4, 2, &YuvRgbMatrix::new(Colorimetry::UNKNOWN)).unwrap();
        assert_eq!(out.len(), 32);
    }

    #[test]
    fn nv12_to_xrgb_neutral_grey_pixel_round_trips() {
        // Y=126 (near mid-grey for limited range), U=V=128 (no chroma) →
        // R = G = B ≈ (298*(126-16) + 128) >> 8 = (298*110 + 128) >> 8
        //         = 32908 >> 8 = 128 (give or take rounding).
        // Verify the centre pixel of a 2x2 fully-uniform NV12 frame lands
        // in [125, 131] on all channels.
        let mut src = alloc::vec![0u8; 6];
        for px in &mut src[..4] {
            *px = 126; // Y
        }
        src[4] = 128; // U
        src[5] = 128; // V
        let out = nv12_to_xrgb8888(&src, 2, 2, &YuvRgbMatrix::new(Colorimetry::UNKNOWN)).unwrap();
        for px in out.as_chunks::<4>().0 {
            assert!((125..=131).contains(&px[0]), "blue out of range: {}", px[0]);
            assert!(
                (125..=131).contains(&px[1]),
                "green out of range: {}",
                px[1]
            );
            assert!((125..=131).contains(&px[2]), "red out of range: {}", px[2]);
            assert_eq!(px[3], 0xFF, "alpha must be 0xFF");
        }
    }

    /// The caps colorimetry reaches the convert table: the same NV12 bytes come
    /// out different pixels once the caps say BT.709, and an untagged stream
    /// keeps the BT.601 limited-range conversion it always had.
    #[test]
    fn configure_takes_the_convert_matrix_from_the_caps() {
        let src = alloc::vec![81u8, 81, 81, 81, 90, 240];
        let nv12_caps = |colorimetry| Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(2),
            height: Dim::Fixed(2),
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry,
        };
        // The worker never spawns: only the table is under test, and a failed
        // worker spawn (no compositor in CI) would mask it.
        let mut sink = WaylandSink::new();
        let convert = |sink: &WaylandSink| nv12_to_xrgb8888(&src, 2, 2, &sink.matrix).unwrap();

        let _ = sink.configure_pipeline(&nv12_caps(Colorimetry::UNKNOWN));
        let untagged = convert(&sink);
        let _ = sink.configure_pipeline(&nv12_caps(Colorimetry::BT601));
        assert_eq!(untagged, convert(&sink), "untagged converts as BT.601");
        let _ = sink.configure_pipeline(&nv12_caps(Colorimetry::BT709));
        assert_ne!(untagged, convert(&sink), "BT.709 converts differently");
    }

    #[test]
    fn nv12_to_xrgb_rejects_truncated_source() {
        let src = alloc::vec![0u8; 8]; // Need 12 for 4x2 NV12.
        assert!(nv12_to_xrgb8888(&src, 4, 2, &YuvRgbMatrix::new(Colorimetry::UNKNOWN)).is_err());
    }

    #[test]
    fn configure_rejects_odd_dims() {
        let mut sink = WaylandSink::new();
        let odd = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(641),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        match sink.configure_pipeline(&odd) {
            Err(G2gError::CapsMismatch) => {}
            other => panic!("expected CapsMismatch on odd dims, got {other:?}"),
        }
    }
}
