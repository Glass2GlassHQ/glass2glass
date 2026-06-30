//! The sync/async impedance matcher ([`BridgeGraph`]). See the crate docs.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::JoinHandle;

use g2g_core::memory::MemoryDomain;
use g2g_core::runtime::{parse_launch, run_graph, ParseError, RunStats};
use g2g_core::{Frame, G2gError, PipelineClock};

use g2g_plugins::appsink::{register_appsink_pull, AppSinkPull, Pull};
use g2g_plugins::appsrc::{register_appsrc, AppSrcFeed};
use g2g_plugins::registry::default_registry;

/// Backpressure floor for the embedded graph's internal edges. Matches the
/// modest depth a single transform stage wants; the appsrc feed and appsink
/// pull channels carry their own bounds.
const LINK_CAPACITY: usize = 4;

/// Monotonic counter for collision-free `appsrc` / `appsink` channel names. The
/// named-feed registries those elements use are process-global (keyed by the
/// channel string), so every `BridgeGraph` must claim a unique pair. An atomic
/// counter (not a timestamp / RNG, which the no_std clock policy forbids) is
/// enough: names need only be unique within one process.
static SEQ: AtomicU64 = AtomicU64::new(0);

/// A monotonic wall clock for the embedded run. The bridge does not impose
/// real-time pacing (the host GStreamer pipeline owns the clock); frames flow as
/// fast as the embedder pushes and drains them.
#[derive(Debug)]
struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Why a [`BridgeGraph`] could not be created.
#[derive(Debug)]
pub enum BridgeError {
    /// The `appsrc ! <fragment> ! appsink` line did not parse. The wrapped error
    /// carries the usual launch diagnostics (unknown element, etc.).
    Parse(ParseError),
    /// The dedicated run thread could not be spawned.
    Spawn(std::io::Error),
}

impl fmt::Display for BridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BridgeError::Parse(e) => write!(f, "bridge sub-graph failed to parse: {e}"),
            BridgeError::Spawn(e) => write!(f, "bridge run thread failed to spawn: {e}"),
        }
    }
}

impl std::error::Error for BridgeError {}

/// An embedded `g2g` sub-graph driven from synchronous (e.g. GStreamer) code.
///
/// Wraps a launch fragment as `appsrc ! <fragment> ! appsink`, runs it on a
/// dedicated OS thread, and hands the embedder a synchronous push/pull API.
/// Construction registers the feed and drain *before* spawning, so the elements
/// claim them when the run thread configures.
///
/// # Lifecycle
///
/// Drop (or [`finish`](BridgeGraph::finish)) signals end-of-stream, releases the
/// drain so the graph cannot deadlock on an un-drained output, and joins the run
/// thread. A `BridgeGraph` therefore never outlives its thread.
#[derive(Debug)]
pub struct BridgeGraph {
    /// Push end of the embedded `appsrc`. `Option` so `finish`/`Drop` can drop it
    /// (closing the feed, which emits EOS) before joining.
    feed: Option<AppSrcFeed>,
    /// Pull end of the embedded `appsink`. Dropped first on shutdown so a full
    /// channel stops back-pressuring the graph and it can reach EOS.
    pull: Option<AppSinkPull>,
    join: Option<JoinHandle<Result<RunStats, G2gError>>>,
}

impl BridgeGraph {
    /// Build and start an embedded sub-graph.
    ///
    /// `fragment` is the g2g portion of a `gst-launch` line, *without* the
    /// surrounding `appsrc`/`appsink` (e.g. `"videoconvert ! mywgpufilter"`).
    /// `input_caps` is a GStreamer caps string describing the buffers the
    /// embedder will [`push`](BridgeGraph::push) (e.g.
    /// `"video/x-raw,format=RGBA,width=1280,height=720,framerate=30/1"`); the
    /// GStreamer shell derives it from the upstream pad's negotiated caps.
    pub fn new(fragment: &str, input_caps: &str) -> Result<Self, BridgeError> {
        let id = SEQ.fetch_add(1, Ordering::Relaxed);
        let in_ch = format!("__g2g_bridge_{id}_in");
        let out_ch = format!("__g2g_bridge_{id}_out");

        // Register before launch: the elements claim these named endpoints when
        // the run thread reaches `configure_pipeline`.
        let feed = register_appsrc(&in_ch);
        let pull = register_appsink_pull(&out_ch);

        let desc =
            format!("appsrc channel={in_ch} caps={input_caps} ! {fragment} ! appsink channel={out_ch}");

        let reg = default_registry();
        let graph = match parse_launch(&reg, &desc) {
            Ok(g) => g,
            Err(e) => return Err(BridgeError::Parse(e)),
        };
        drop(reg);

        let join = std::thread::Builder::new()
            .name(format!("g2g-bridge-{id}"))
            .spawn(move || {
                // A dedicated current-thread runtime drives the async sub-graph
                // on this OS thread, isolated from GStreamer's streaming threads
                // (DESIGN.md §7). `enable_time` matches the runtime the rest of
                // the workspace runs graphs on (sinks may use tokio timers).
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_time()
                    .build()
                    .expect("build bridge tokio runtime");
                let clock = ZeroClock;
                rt.block_on(run_graph(graph, &clock, LINK_CAPACITY))
            })
            .map_err(BridgeError::Spawn)?;

        Ok(BridgeGraph { feed: Some(feed), pull: Some(pull), join: Some(join) })
    }

    /// Push one buffer (copied) with presentation timestamp `pts_ns` into the
    /// embedded graph. Returns `false` if the feed is full (the graph is busy,
    /// retry after draining output) or the graph has gone away.
    ///
    /// Non-blocking: safe to call from a GStreamer streaming thread's `chain`.
    pub fn push(&self, data: &[u8], pts_ns: u64) -> bool {
        self.feed.as_ref().is_some_and(|f| f.push(data, pts_ns))
    }

    /// Drain one processed frame if ready. Returns [`Pull::Empty`] when the graph
    /// has not produced output yet (a stage with internal latency will lag the
    /// input by some frames) and [`Pull::Ended`] after EOS.
    ///
    /// Non-blocking: the GStreamer `chain` pushes one input, then loops
    /// `try_pull` to forward whatever is ready downstream.
    pub fn try_pull(&self) -> Pull {
        match self.pull.as_ref() {
            Some(p) => p.try_pull(),
            None => Pull::Ended,
        }
    }

    /// Block the calling thread until the next frame, or `None` at EOS / shutdown.
    /// For embedders that prefer a blocking drain over polling `try_pull`.
    pub fn pull_blocking(&self) -> Option<Frame> {
        let p = self.pull.as_ref()?;
        g2g_core::runtime::block_on(p.pull())
    }

    /// Signal end-of-stream on the feed; no more [`push`](BridgeGraph::push)es
    /// will be delivered. The graph drains its in-flight buffers and then ends.
    pub fn end_of_stream(&self) -> bool {
        self.feed.as_ref().is_some_and(g2g_plugins::appsrc::AppSrcFeed::end_of_stream)
    }

    /// Whether the run thread has finished (EOS reached or errored).
    pub fn is_done(&self) -> bool {
        match self.join.as_ref() {
            Some(j) => j.is_finished(),
            None => true,
        }
    }

    /// Signal EOS, drain-release, and join the run thread, returning its final
    /// stats. Idempotent-safe via `&mut self`; [`Drop`] calls the same path.
    pub fn finish(mut self) -> Result<RunStats, G2gError> {
        self.shutdown()
    }

    /// Drop the drain (so a full output channel stops back-pressuring), signal
    /// EOS on the feed, then join. Joining cannot deadlock: with the pull handle
    /// gone the appsink discards undeliverable frames instead of blocking.
    fn shutdown(&mut self) -> Result<RunStats, G2gError> {
        self.pull = None;
        if let Some(feed) = self.feed.take() {
            feed.end_of_stream();
            // dropping `feed` here also closes the feed, a second EOS signal.
        }
        match self.join.take() {
            Some(j) => j.join().unwrap_or(Err(G2gError::Shutdown)),
            None => Err(G2gError::Shutdown),
        }
    }
}

impl Drop for BridgeGraph {
    fn drop(&mut self) {
        if self.join.is_some() {
            let _ = self.shutdown();
        }
    }
}

/// Borrow a system-memory frame's bytes, the common case for the bridge (the
/// embedder pushed `System` buffers and the graph kept them in system memory).
/// Returns `None` for a GPU-resident frame, which the embedder must download
/// before it can be handed back to GStreamer as a `GstBuffer`.
pub fn frame_bytes(frame: &Frame) -> Option<&[u8]> {
    match &frame.domain {
        MemoryDomain::System(slice) => Some(slice.as_slice()),
        _ => None,
    }
}
