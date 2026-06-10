//! Wayland display sink (desktop dev convenience).
//!
//! Opens an `xdg_toplevel` window on the running Wayland compositor and
//! presents NV12 `DataFrame`s into it. The pixel path is software:
//! NV12 → XRGB8888 conversion (BT.601 limited range) into a `wl_shm`
//! pool, then `attach` + `commit` per frame. Slow but universal; every
//! Wayland compositor supports `wl_shm`.
//!
//! This is the **dev sink**, not the production sink:
//! - Latency is whatever the compositor's frame callback delivers (one
//!   compositor refresh, typically ~16 ms at 60 Hz).
//! - The XRGB8888 conversion runs on the same thread that drives the
//!   pipeline; at 1080p30 the CPU cost is real (each frame is ~2 ms of
//!   YUV→RGB on a modern x86 core).
//! - No GPU upload, no `zwp_linux_dmabuf_v1` zero-copy, no colour-space
//!   metadata propagation.
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
//! - No audio sync, no PTS pacing in the wall-clock sense. Backpressure
//!   is compositor-driven: `process()` blocks until the compositor's
//!   `frame` callback for the previously committed buffer arrives, so
//!   the producer is naturally throttled to refresh.
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
use alloc::sync::Arc as RcArc;
use alloc::vec::Vec;

use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState},
    delegate_compositor, delegate_output, delegate_registry, delegate_shm,
    delegate_xdg_shell, delegate_xdg_window,
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

use g2g_core::frame::Frame;
use g2g_core::metrics::{monotonic_ns, LatencyHistogram, LatencySnapshot};
use g2g_core::{
    AsyncElement, Caps, ClockCandidate, ClockPriority, ConfigureOutcome, Dim, G2gError,
    HardwareError, MemoryDomain, OutputSink, PipelineClock, PipelinePacket, VideoFormat,
};

/// Worker-thread message. `Frame` carries the pre-converted XRGB8888
/// bytes (sink-side conversion keeps the worker thread free for Wayland
/// I/O) plus a one-shot `ack` the worker signals once the frame has been
/// committed *and* the compositor's next `frame` callback has fired —
/// that's the signal we use to pace the producer to refresh.
/// `Shutdown` exits the worker's event loop.
enum WorkerCmd {
    Frame {
        bytes: Vec<u8>,
        /// Source-side wall-clock stamp from `FrameTiming::arrival_ns`.
        /// The worker records `monotonic_ns() - arrival_ns` into the
        /// latency histogram when the matching `frame` callback fires.
        /// Zero means the frame was untimed; latency is not recorded.
        arrival_ns: u64,
        ack: tokio::sync::oneshot::Sender<()>,
    },
    Shutdown,
}

/// How the sink reacts when the producer pushes faster than the
/// compositor refreshes.
///
/// - `Block` (default): `process()` waits for the matching `frame`
///   callback before returning. Producer is throttled to refresh.
///   No drops, but backpressure propagates upstream.
/// - `DropOldest`: `process()` returns as soon as the worker accepts
///   the frame. If a previous frame is still awaiting its `frame`
///   callback, the worker overwrites it — the older frame never paints.
///   Use for live sources that prefer freshness over completeness
///   (security cameras, monitoring) and can't tolerate backpressure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PacingPolicy {
    Block,
    DropOldest,
}

impl Default for PacingPolicy {
    fn default() -> Self {
        Self::Block
    }
}

/// What the sink-side struct holds between `process()` calls. We keep
/// only `Send + Sync` handles here so the multi-thread runner can move
/// us between executor tasks.
pub struct WaylandSink {
    title: String,
    app_id: String,
    cmd_tx: Option<CalloopSender<WorkerCmd>>,
    worker: Option<JoinHandle<()>>,
    width: u32,
    height: u32,
    frames_presented: Arc<AtomicU64>,
    latency: Arc<LatencyHistogram>,
    frames_dropped: Arc<AtomicU64>,
    pacing: PacingPolicy,
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
            frames_presented: Arc::new(AtomicU64::new(0)),
            latency: Arc::new(LatencyHistogram::new()),
            frames_dropped: Arc::new(AtomicU64::new(0)),
            pacing: PacingPolicy::default(),
        }
    }

    pub fn with_pacing(mut self, pacing: PacingPolicy) -> Self {
        self.pacing = pacing;
        self
    }

    pub fn frames_dropped(&self) -> u64 {
        self.frames_dropped.load(Ordering::Relaxed)
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
        self.latency.snapshot()
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
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn provide_clock(&self) -> Option<ClockCandidate> {
        Some(ClockCandidate::new(
            ClockPriority::Provider,
            alloc::sync::Arc::new(WaylandClock),
        ))
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // Pass-through at negotiation; the real NV12 validation happens in
        // `configure_pipeline`. With the decoder native (`DerivedOutput`),
        // the solver assigns this link NV12 directly, so configure receives
        // NV12 at startup rather than the decoder's pre-decode H.264 caps.
        Ok(upstream_caps.clone())
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        // NV12 only. Now that every decoder is a native `DerivedOutput`,
        // the solver lands NV12 on this link at startup, so the old
        // accept-H.264-as-no-op workaround is gone: a non-NV12 sink input
        // is a real pipeline error (e.g. an undecoded display chain) and
        // fails loud here.
        let (w, h) = match absolute_caps {
            Caps::Video {
                format: VideoFormat::Nv12,
                width: Dim::Fixed(w),
                height: Dim::Fixed(h),
                ..
            } => (*w, *h),
            _ => return Err(G2gError::CapsMismatch),
        };
        if w % 2 != 0 || h % 2 != 0 {
            return Err(G2gError::CapsMismatch);
        }

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
        let title = self.title.clone();
        let app_id = self.app_id.clone();

        // Synchronous handshake: the worker signals readiness once the
        // compositor's first `configure` lands. Until then `process()`
        // would be racing against an unmapped surface.
        let ready = Arc::new(parking_handshake::Handshake::new());
        let ready_for_worker = Arc::clone(&ready);

        let join = thread::Builder::new()
            .name(String::from("g2g-waylandsink"))
            .spawn(move || {
                if let Err(e) = worker_main(
                    w, h, title, app_id, rx, presented, dropped, latency, ready_for_worker,
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

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(Frame { domain, timing, .. }) => {
                    let MemoryDomain::System(slice) = domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let xrgb = nv12_to_xrgb8888(slice.as_slice(), self.width, self.height)?;
                    let tx = self
                        .cmd_tx
                        .as_ref()
                        .ok_or(G2gError::NotConfigured)?;
                    let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
                    tx.send(WorkerCmd::Frame {
                        bytes: xrgb,
                        arrival_ns: timing.arrival_ns,
                        ack: ack_tx,
                    })
                    .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
                    match self.pacing {
                        PacingPolicy::Block => {
                            // Wait for the compositor's `frame` callback
                            // for this commit. RecvError means the
                            // worker dropped the ack (shutdown / crash)
                            // — treat as a hardware fault.
                            ack_rx
                                .await
                                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
                        }
                        PacingPolicy::DropOldest => {
                            // Fire-and-forget: producer keeps moving.
                            // If the previous frame's ack is still
                            // outstanding when this one is drawn, the
                            // worker drops it and bumps frames_dropped.
                            drop(ack_rx);
                        }
                    }
                    Ok(())
                }
                PipelinePacket::CapsChanged(_) | PipelinePacket::Flush => Ok(()),
                PipelinePacket::Eos => {
                    self.shutdown();
                    Ok(())
                }
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
    ready: Option<Arc<parking_handshake::Handshake>>,
    presented: Arc<AtomicU64>,
    dropped: Arc<AtomicU64>,
    latency: Arc<LatencyHistogram>,
    /// Frame queued before the surface is mappable. Once `configure`
    /// lands we drain this into the first draw. With blocking pacing the
    /// producer is throttled to one in-flight frame, so under steady
    /// state this is None.
    pending: Option<(Vec<u8>, u64, tokio::sync::oneshot::Sender<()>)>,
    /// Ack for the most recently committed frame plus its source-side
    /// arrival timestamp. Signalled when the compositor's matching
    /// `frame` callback fires, at which point we record the latency.
    pending_ack: Option<(u64, tokio::sync::oneshot::Sender<()>)>,
}

fn worker_main(
    width: u32,
    height: u32,
    title: String,
    app_id: String,
    rx: Channel<WorkerCmd>,
    presented: Arc<AtomicU64>,
    dropped: Arc<AtomicU64>,
    latency: Arc<LatencyHistogram>,
    ready: Arc<parking_handshake::Handshake>,
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
        pending: None,
        pending_ack: None,
    };

    // Wire the cmd channel into calloop so we wake on frame arrival.
    loop_handle.insert_source(rx, |event, _, state: &mut WorkerState| match event {
        ChanEvent::Msg(WorkerCmd::Frame { bytes, arrival_ns, ack }) => {
            // Producer is blocked on `ack` until our `frame` callback
            // fires, so we should only ever see one in flight. If the
            // surface isn't mappable yet, stash it; otherwise draw now.
            if state.configured {
                state.draw(bytes, arrival_ns, ack);
            } else {
                state.pending = Some((bytes, arrival_ns, ack));
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
    fn draw(&mut self, bytes: Vec<u8>, arrival_ns: u64, ack: tokio::sync::oneshot::Sender<()>) {
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
        // Subscribe to the compositor's `frame` callback for this commit.
        // SCTK's CompositorHandler::frame routes by the WlSurface udata,
        // so we pass a clone of the surface as the callback's user data.
        surface.frame(&self.qh, surface.clone());
        surface.damage_buffer(0, 0, width, height);
        buffer.attach_to(surface).expect("attach_to");
        self.window.commit();
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
    fn frame(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: u32,
    ) {
        // The compositor is ready for the next frame. Record the
        // glass-to-glass delta (source ingest -> on-screen), then
        // release the producer blocked on this commit's ack.
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
            if let Some((bytes, arrival_ns, ack)) = self.pending.take() {
                self.draw(bytes, arrival_ns, ack);
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
    fn output_destroyed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: wl_output::WlOutput,
    ) {
    }
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
// NV12 -> XRGB8888 (BT.601 limited-range)
// =================================================================

/// Convert a packed NV12 source buffer (`width * height` Y plane
/// followed by `width * height / 2` UV plane, interleaved as U,V,U,V)
/// into a packed XRGB8888 buffer (`width * height * 4` bytes, little-
/// endian per pixel: `[B, G, R, 0xFF]`). Uses BT.601 limited-range
/// coefficients, which is what H.264 SD content usually carries. HDR
/// and BT.709 paths are deferred.
fn nv12_to_xrgb8888(src: &[u8], width: u32, height: u32) -> Result<Vec<u8>, G2gError> {
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

            let c = y - 16;
            let d = u - 128;
            let e = v - 128;

            // Integer-fixed-point BT.601: coefficients * 256 then >> 8.
            let r = (298 * c + 409 * e + 128) >> 8;
            let g = (298 * c - 100 * d - 208 * e + 128) >> 8;
            let b = (298 * c + 516 * d + 128) >> 8;

            let dst = dst_row_off + col * 4;
            out[dst] = b.clamp(0, 255) as u8;
            out[dst + 1] = g.clamp(0, 255) as u8;
            out[dst + 2] = r.clamp(0, 255) as u8;
            out[dst + 3] = 0xFF;
        }
    }
    Ok(out)
}

// =================================================================
// Sink-side handshake primitive (worker readiness)
// =================================================================
//
// `parking_handshake::Handshake` is a tiny one-shot: the worker calls
// `notify()` once after its first compositor `configure`, and the sink
// blocks on `wait(timeout)` until that lands (or the timeout fires).
// Implemented inline rather than pulling in `parking_lot` or `tokio::sync`
// since we already have `std::sync` available under the `wayland-sink`
// feature.

mod parking_handshake {
    use std::sync::{Condvar, Mutex};
    use std::time::Duration;

    pub(super) struct Handshake {
        flag: Mutex<bool>,
        cv: Condvar,
    }

    impl Handshake {
        pub(super) fn new() -> Self {
            Self {
                flag: Mutex::new(false),
                cv: Condvar::new(),
            }
        }

        pub(super) fn notify(&self) {
            *self.flag.lock().unwrap() = true;
            self.cv.notify_all();
        }

        /// Returns true if notified within `timeout`, false on timeout.
        pub(super) fn wait(&self, timeout: Duration) -> bool {
            let guard = self.flag.lock().unwrap();
            let (guard, _wait_result) = self
                .cv
                .wait_timeout_while(guard, timeout, |notified| !*notified)
                .unwrap();
            *guard
        }
    }
}

// Suppress unused-import warnings for the `RcArc` we kept around in
// case future code wants `alloc::sync::Arc` distinct from `std::sync::Arc`.
// `RcArc` is the same type under feature `std`, but we don't actually use it.
const _: () = {
    #[allow(dead_code)]
    fn _suppress_rcarc_unused(_: Option<RcArc<u8>>) {}
};

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::Rate;

    #[test]
    fn intercept_passes_through_any_format() {
        // Negotiation-time intercept is pass-through; the NV12 requirement
        // is enforced in `configure_pipeline`. (With a native decoder the
        // solver hands this link NV12 anyway.)
        let sink = WaylandSink::new();
        let h264 = Caps::Video {
            format: VideoFormat::H264,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
        };
        assert_eq!(sink.intercept_caps(&h264), Ok(h264));
    }

    #[test]
    fn intercept_passes_through_nv12() {
        let sink = WaylandSink::new();
        let nv12 = Caps::Video {
            format: VideoFormat::Nv12,
            width: Dim::Fixed(1280),
            height: Dim::Fixed(720),
            framerate: Rate::Any,
        };
        assert_eq!(sink.intercept_caps(&nv12), Ok(nv12));
    }

    #[test]
    fn configure_rejects_non_nv12() {
        let mut sink = WaylandSink::new();
        let h264 = Caps::Video {
            format: VideoFormat::H264,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
        };
        // A native decoder lands NV12 on this link; a non-NV12 sink input
        // is a real error (e.g. an undecoded display chain), not a no-op.
        assert_eq!(sink.configure_pipeline(&h264).err(), Some(G2gError::CapsMismatch));
        assert!(sink.worker.is_none(), "no worker should be spawned on rejected caps");
    }

    #[test]
    fn nv12_to_xrgb_yields_correct_byte_count() {
        // 4x2 NV12: Y=8 bytes, UV=4 bytes. Output = 4*2*4 = 32 bytes.
        let src = alloc::vec![16u8; 12];
        let out = nv12_to_xrgb8888(&src, 4, 2).unwrap();
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
        let out = nv12_to_xrgb8888(&src, 2, 2).unwrap();
        for px in out.chunks_exact(4) {
            assert!(
                (125..=131).contains(&px[0]),
                "blue out of range: {}",
                px[0]
            );
            assert!(
                (125..=131).contains(&px[1]),
                "green out of range: {}",
                px[1]
            );
            assert!(
                (125..=131).contains(&px[2]),
                "red out of range: {}",
                px[2]
            );
            assert_eq!(px[3], 0xFF, "alpha must be 0xFF");
        }
    }

    #[test]
    fn nv12_to_xrgb_rejects_truncated_source() {
        let src = alloc::vec![0u8; 8]; // Need 12 for 4x2 NV12.
        assert!(nv12_to_xrgb8888(&src, 4, 2).is_err());
    }

    #[test]
    fn configure_rejects_odd_dims() {
        let mut sink = WaylandSink::new();
        let odd = Caps::Video {
            format: VideoFormat::Nv12,
            width: Dim::Fixed(641),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
        };
        match sink.configure_pipeline(&odd) {
            Err(G2gError::CapsMismatch) => {}
            other => panic!("expected CapsMismatch on odd dims, got {other:?}"),
        }
    }

    #[test]
    fn handshake_round_trips() {
        let hs = Arc::new(parking_handshake::Handshake::new());
        let hs2 = Arc::clone(&hs);
        let join = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            hs2.notify();
        });
        assert!(hs.wait(Duration::from_secs(2)), "notify should land");
        join.join().unwrap();
    }

    #[test]
    fn handshake_times_out_without_notify() {
        let hs = parking_handshake::Handshake::new();
        assert!(!hs.wait(Duration::from_millis(20)));
    }
}
