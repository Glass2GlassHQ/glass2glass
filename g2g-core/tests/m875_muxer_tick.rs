//! M875 / M877: a fan-in arm's deadline tick.
//!
//! A fan-in element that declares a `tick_interval_ns` receives
//! `PipelinePacket::Tick` while its inputs are stalled, so it can emit output on
//! its own cadence (zero-order-hold over the stalled pad) instead of freezing
//! with the input. An element that declares no interval never sees one.
//!
//! The cooperative entry points derive the timer from the pipeline clock itself
//! (M880: any clock answering `as_ticker`), so the same fixtures run through plain
//! `run_muxer_sink` (M875), plain `run_graph` on a hand-built graph, and the
//! PTS-ordered muxer arm a fan-in opts into with `input_pts_ordered` (M877). The
//! thread-per-arm runner needs an owned clock handle, so it keeps its own entry,
//! `run_graph_threaded_ticked`, where each arm runs on its own OS thread (M879).
//!
//! Time is mocked, so the tests are deterministic and finish in microseconds
//! instead of sleeping on real timers: one clock jumps virtual time to the sleep
//! deadline and resolves at once, the other never resolves a sleep at all (which
//! isolates the arm's busy-path deadline check).
#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use g2g_core::fanout::MultiInputElement;
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{
    block_on, run_graph, run_muxer_sink, DynSourceLoop, GraphNodeRef, SourceLoop,
};
use g2g_core::{
    AsyncClock, AsyncElement, Caps, ConfigureOutcome, Dim, DynAsyncClock, Frame, FrameTiming,
    G2gError, Graph, OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

const TICK_NS: u64 = 33_000_000;

/// Spin bound for the fixtures that wait on a tick count: a runner that never ticks
/// ends the run and fails an assertion instead of hanging. Generous because the
/// threaded run's waiting source races real thread startup, not just a sibling task.
const MAX_SPINS: usize = 200_000;

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(2),
        height: Dim::Fixed(2),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn frame(sequence: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 16]))),
        FrameTiming {
            pts_ns: sequence * TICK_NS,
            ..FrameTiming::default()
        },
        sequence,
    ))
}

/// Yields once so the other arms of the run get polled. Both the fake clock and
/// the stalling source need it: without a yield an arm that is always ready
/// starves the rest on the cooperative executor.
struct YieldOnce(bool);

impl Future for YieldOnce {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.0 {
            return Poll::Ready(());
        }
        self.0 = true;
        cx.waker().wake_by_ref();
        Poll::Pending
    }
}

fn yield_once() -> YieldOnce {
    YieldOnce(false)
}

/// Virtual-time clock: `sleep_until_ns` advances the clock to the deadline and
/// resolves immediately (after the yield above), so a tick period costs no real
/// time. Time is an external boundary, so mocking it is fair game.
#[derive(Debug, Default)]
struct VirtualClock {
    now_ns: AtomicU64,
}

impl PipelineClock for VirtualClock {
    fn now_ns(&self) -> u64 {
        self.now_ns.load(Ordering::Acquire)
    }

    fn as_ticker(&self) -> Option<&dyn DynAsyncClock> {
        Some(self)
    }
}

impl AsyncClock for VirtualClock {
    type SleepFuture<'a>
        = VirtualSleep<'a>
    where
        Self: 'a;

    fn sleep_until_ns(&self, deadline_ns: u64) -> VirtualSleep<'_> {
        VirtualSleep {
            clock: self,
            deadline_ns,
            yielded: false,
        }
    }
}

struct VirtualSleep<'a> {
    clock: &'a VirtualClock,
    deadline_ns: u64,
    yielded: bool,
}

impl Future for VirtualSleep<'_> {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if !self.yielded {
            self.yielded = true;
            cx.waker().wake_by_ref();
            return Poll::Pending;
        }
        let deadline = self.deadline_ns;
        self.clock.now_ns.fetch_max(deadline, Ordering::AcqRel);
        Poll::Ready(())
    }
}

/// Clock for the busy-path test: reading it advances time (work between two
/// reads takes time) and its sleep **never** resolves. So a tick can only come
/// from the arm's per-iteration deadline check, never from a parked sleep.
#[derive(Debug, Default)]
struct BusyClock {
    now_ns: AtomicU64,
}

impl PipelineClock for BusyClock {
    fn now_ns(&self) -> u64 {
        self.now_ns.fetch_add(TICK_NS / 8, Ordering::AcqRel) + TICK_NS / 8
    }

    fn as_ticker(&self) -> Option<&dyn DynAsyncClock> {
        Some(self)
    }
}

impl AsyncClock for BusyClock {
    type SleepFuture<'a>
        = core::future::Pending<()>
    where
        Self: 'a;

    fn sleep_until_ns(&self, _deadline_ns: u64) -> core::future::Pending<()> {
        core::future::pending()
    }
}

/// Pushes `count` frames back to back, then ends: traffic on one pad that keeps
/// the muxer arm out of its parked path.
struct FloodSrc {
    count: u64,
}

impl SourceLoop for FloodSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(caps()))
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for i in 0..self.count {
                out.push(frame(i)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.count)
        })
    }
}

/// Pushes nothing until the muxer has ticked `stall_until_ticks` times, then
/// sends one frame and ends. The stall is what makes the tick observable: while
/// it holds, the muxer arm has no packet to process on any pad.
struct StallSrc {
    ticks: Arc<AtomicUsize>,
    stall_until_ticks: usize,
}

impl SourceLoop for StallSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(caps()))
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let mut spins = 0;
            while self.ticks.load(Ordering::Acquire) < self.stall_until_ticks && spins < MAX_SPINS {
                yield_once().await;
                spins += 1;
            }
            out.push(frame(0)).await?;
            out.push(PipelinePacket::Eos).await?;
            Ok(1)
        })
    }
}

/// Pushes its frames back to back, then stays open (no `Eos`) until the muxer has
/// ticked `hold_until_ticks` times. The PTS-ordered arm buffers those frames while
/// another input is silent, so holding the pad open is what lets a later pad still
/// deliver an earlier PTS.
struct HoldSrc {
    seqs: &'static [u64],
    ticks: Arc<AtomicUsize>,
    hold_until_ticks: usize,
}

impl SourceLoop for HoldSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(caps()))
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for &seq in self.seqs {
                out.push(frame(seq)).await?;
            }
            let mut spins = 0;
            while self.ticks.load(Ordering::Acquire) < self.hold_until_ticks && spins < MAX_SPINS {
                yield_once().await;
                spins += 1;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.seqs.len() as u64)
        })
    }
}

/// Fan-in that emits a frame per deadline tick (the zero-order-hold shape), counts
/// the ticks it received, and records the PTS of every frame delivered to it (so a
/// test can check the arm's delivery order).
struct ZohMux {
    inputs: usize,
    interval_ns: Option<u64>,
    pts_ordered: bool,
    ticks: Arc<AtomicUsize>,
    emitted: u64,
    seen_pts: Vec<u64>,
}

impl ZohMux {
    fn new(inputs: usize, interval_ns: Option<u64>, ticks: Arc<AtomicUsize>) -> Self {
        Self {
            inputs,
            interval_ns,
            pts_ordered: false,
            ticks,
            emitted: 0,
            seen_pts: Vec::new(),
        }
    }

    /// Opt into the runner's PTS-ordered fan-in arm.
    fn pts_ordered(mut self) -> Self {
        self.pts_ordered = true;
        self
    }
}

impl MultiInputElement for ZohMux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        self.inputs
    }

    fn tick_interval_ns(&self) -> Option<u64> {
        self.interval_ns
    }

    fn input_pts_ordered(&self) -> bool {
        self.pts_ordered
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn configure_pipeline(
        &mut self,
        _input: usize,
        _absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(caps())
    }

    fn process<'a>(
        &'a mut self,
        _input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::Tick => {
                    self.ticks.fetch_add(1, Ordering::AcqRel);
                    let seq = self.emitted;
                    self.emitted += 1;
                    out.push(frame(seq)).await?;
                }
                PipelinePacket::DataFrame(f) => {
                    self.seen_pts.push(f.timing.pts_ns);
                    let seq = self.emitted;
                    self.emitted += 1;
                    out.push(frame(seq)).await?;
                }
                // Eos is the runner's to emit; caps are already negotiated.
                _ => {}
            }
            Ok(())
        })
    }
}

/// Counts the merged frames that reach the end of the pipeline.
#[derive(Default)]
struct CountSink {
    frames: u64,
}

impl AsyncElement for CountSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        if matches!(packet, PipelinePacket::DataFrame(_)) {
            self.frames += 1;
        }
        Box::pin(core::future::ready(Ok(())))
    }
}

#[test]
fn ticks_fire_while_the_input_stalls_and_the_frames_reach_the_sink() {
    let ticks = Arc::new(AtomicUsize::new(0));
    let clock = VirtualClock::default();
    let mut src = StallSrc {
        ticks: ticks.clone(),
        stall_until_ticks: 3,
    };
    let mut mux = ZohMux::new(1, Some(TICK_NS), ticks.clone());
    let mut sink = CountSink::default();

    let sources: Vec<&mut dyn DynSourceLoop> = vec![&mut src];
    let stats = block_on(run_muxer_sink(sources, &mut mux, &mut sink, &clock, 2))
        .expect("ticked muxer run");

    let fired = ticks.load(Ordering::Acquire);
    assert!(
        fired >= 3,
        "the arm must tick while the only input is stalled, got {fired}"
    );
    // Every tick emitted a zero-order-hold frame, and the input's own frame
    // followed once it finally arrived.
    assert_eq!(
        sink.frames,
        fired as u64 + 1,
        "the ZOH frames plus the input frame reached the sink"
    );
    assert_eq!(stats.frames_consumed, sink.frames);
    assert!(
        clock.now_ns() >= 3 * TICK_NS,
        "each tick advanced the deadline by one period"
    );
}

#[test]
fn an_element_that_declares_no_interval_never_ticks() {
    let ticks = Arc::new(AtomicUsize::new(0));
    let clock = VirtualClock::default();
    let mut src = StallSrc {
        ticks: ticks.clone(),
        stall_until_ticks: 0,
    };
    let mut mux = ZohMux::new(1, None, ticks.clone());
    let mut sink = CountSink::default();

    let sources: Vec<&mut dyn DynSourceLoop> = vec![&mut src];
    block_on(run_muxer_sink(sources, &mut mux, &mut sink, &clock, 2)).expect("un-ticked muxer run");

    assert_eq!(
        ticks.load(Ordering::Acquire),
        0,
        "no interval declared, so the arm delivers no Tick"
    );
    assert_eq!(
        sink.frames, 1,
        "only the input's own frame reached the sink"
    );
    assert_eq!(clock.now_ns(), 0, "nothing slept on the clock");
}

#[test]
fn a_busy_arm_still_hits_its_deadline() {
    // Pad 1 stalls while pad 0 floods, so the arm keeps finding a buffered packet
    // and never parks. With a clock whose sleep never resolves, any tick here came
    // from the arm's per-iteration deadline check.
    let ticks = Arc::new(AtomicUsize::new(0));
    let clock = BusyClock::default();
    let mut flood = FloodSrc { count: 100 };
    let mut stalled = StallSrc {
        ticks: ticks.clone(),
        stall_until_ticks: 1,
    };
    let mut mux = ZohMux::new(2, Some(TICK_NS), ticks.clone());
    let mut sink = CountSink::default();

    let sources: Vec<&mut dyn DynSourceLoop> = vec![&mut flood, &mut stalled];
    block_on(run_muxer_sink(sources, &mut mux, &mut sink, &clock, 2))
        .expect("busy ticked muxer run");

    let fired = ticks.load(Ordering::Acquire);
    assert!(
        fired >= 1,
        "a busy arm must still tick, and no sleep can have fired one"
    );
    assert_eq!(
        sink.frames,
        101 + fired as u64,
        "both inputs' frames plus one ZOH frame per tick reached the sink"
    );
}

/// M877: runs `StallSrc -> mux -> CountSink` as a hand-built graph through
/// `run_graph`, and reports the ticks the mux saw plus the frames that
/// reached the sink. Covers both fan-in arms: `pts_ordered` picks the PTS-ordered
/// one.
fn stalled_graph_run(
    interval_ns: Option<u64>,
    pts_ordered: bool,
    stall_until_ticks: usize,
) -> (usize, u64) {
    let clock = VirtualClock::default();
    let ticks = Arc::new(AtomicUsize::new(0));
    let mut src = StallSrc {
        ticks: ticks.clone(),
        stall_until_ticks,
    };
    let mut mux = ZohMux::new(1, interval_ns, ticks.clone());
    if pts_ordered {
        mux = mux.pts_ordered();
    }
    let mut sink = CountSink::default();

    let mut g: Graph<GraphNodeRef<'_>> = Graph::new();
    let mux_node = g.add_muxer(GraphNodeRef::muxer_ref(&mut mux), 1);
    let s = g.add_source(GraphNodeRef::source_ref(&mut src));
    let k = g.add_sink(GraphNodeRef::element_ref(&mut sink));
    g.link(s, mux_node.input(0)).expect("link source to pad 0");
    g.link(mux_node.output(), k)
        .expect("link merged output to sink");
    block_on(run_graph(g, &clock, 2)).expect("ticked graph run");

    (ticks.load(Ordering::Acquire), sink.frames)
}

#[test]
fn run_graph_delivers_the_tick_on_a_hand_built_graph() {
    let (fired, frames) = stalled_graph_run(Some(TICK_NS), false, 3);
    assert!(
        fired >= 3,
        "run_graph must tick the fan-in while its only input stalls, got {fired}"
    );
    assert_eq!(
        frames,
        fired as u64 + 1,
        "the ZOH frames plus the input frame reached the sink"
    );
}

#[test]
fn the_pts_ordered_arm_delivers_the_tick_too() {
    let (fired, frames) = stalled_graph_run(Some(TICK_NS), true, 3);
    assert!(
        fired >= 3,
        "the PTS-ordered arm must tick while its only input stalls, got {fired}"
    );
    assert_eq!(frames, fired as u64 + 1);
}

#[test]
fn no_interval_means_no_tick_on_either_arm() {
    for pts_ordered in [false, true] {
        let (fired, frames) = stalled_graph_run(None, pts_ordered, 0);
        assert_eq!(
            fired, 0,
            "no interval declared, so no Tick (pts_ordered={pts_ordered})"
        );
        assert_eq!(
            frames, 1,
            "only the input's own frame reached the sink (pts_ordered={pts_ordered})"
        );
    }
}

#[test]
fn the_pts_ordered_arm_ticks_while_it_holds_frames_back() {
    // Pad 0 delivers two frames and stays open; pad 1 is silent. The PTS-ordered
    // arm cannot release pad 0's frames while pad 1 might still deliver an earlier
    // PTS, so it parks with frames buffered, which is exactly where the tick has to
    // fire. Pad 1 then delivers PTS 0: it lands *before* the buffered frames, so
    // the arrival-order arm would have failed this ordering assertion.
    let ticks = Arc::new(AtomicUsize::new(0));
    let clock = VirtualClock::default();
    let mut held = HoldSrc {
        seqs: &[2, 4],
        ticks: ticks.clone(),
        hold_until_ticks: 3,
    };
    let mut stalled = StallSrc {
        ticks: ticks.clone(),
        stall_until_ticks: 3,
    };
    let mut mux = ZohMux::new(2, Some(TICK_NS), ticks.clone()).pts_ordered();
    let mut sink = CountSink::default();

    let mut g: Graph<GraphNodeRef<'_>> = Graph::new();
    let mux_node = g.add_muxer(GraphNodeRef::muxer_ref(&mut mux), 2);
    let held_node = g.add_source(GraphNodeRef::source_ref(&mut held));
    let stalled_node = g.add_source(GraphNodeRef::source_ref(&mut stalled));
    let k = g.add_sink(GraphNodeRef::element_ref(&mut sink));
    g.link(held_node, mux_node.input(0)).expect("link pad 0");
    g.link(stalled_node, mux_node.input(1)).expect("link pad 1");
    g.link(mux_node.output(), k).expect("link merged output");
    block_on(run_graph(g, &clock, 2)).expect("ticked pts graph run");

    let fired = ticks.load(Ordering::Acquire);
    assert!(
        fired >= 3,
        "the PTS-ordered arm must tick while it parks on the silent pad, got {fired}"
    );
    assert_eq!(
        mux.seen_pts,
        vec![0, 2 * TICK_NS, 4 * TICK_NS],
        "frames still arrived in global PTS order"
    );
    assert_eq!(
        sink.frames,
        fired as u64 + 3,
        "one ZOH frame per tick plus the three input frames reached the sink"
    );
}

/// M879: the same fixtures under the thread-per-arm runner, which owns its elements
/// (each arm moves onto its own OS thread), so the frame count comes from the run
/// stats rather than from a borrowed sink. Reports the ticks the mux saw, the frames
/// the sink consumed, and the virtual time the arm slept away.
#[cfg(feature = "multi-thread")]
fn threaded_stalled_run(
    interval_ns: Option<u64>,
    pts_ordered: bool,
    stall_until_ticks: usize,
) -> (usize, u64, u64) {
    use g2g_core::runtime::{run_graph_threaded_ticked, GraphNode, ThreadSpawner};
    use g2g_core::DynAsyncClock;

    let clock: Arc<dyn DynAsyncClock + Send + Sync> = Arc::new(VirtualClock::default());
    let ticks = Arc::new(AtomicUsize::new(0));
    let mut mux = ZohMux::new(1, interval_ns, ticks.clone());
    if pts_ordered {
        mux = mux.pts_ordered();
    }

    let mut g: Graph<GraphNode> = Graph::new();
    let mux_node = g.add_muxer(GraphNode::muxer(mux), 1);
    let s = g.add_source(GraphNode::source(StallSrc {
        ticks: ticks.clone(),
        stall_until_ticks,
    }));
    let k = g.add_sink(GraphNode::element(CountSink::default()));
    g.link(s, mux_node.input(0)).expect("link source to pad 0");
    g.link(mux_node.output(), k)
        .expect("link merged output to sink");
    let stats = block_on(run_graph_threaded_ticked(
        g,
        clock.clone(),
        2,
        &ThreadSpawner,
    ))
    .expect("ticked threaded graph run");

    (
        ticks.load(Ordering::Acquire),
        stats.frames_consumed,
        clock.now_ns(),
    )
}

#[cfg(feature = "multi-thread")]
#[test]
fn the_threaded_runner_ticks_both_fan_in_arms() {
    for pts_ordered in [false, true] {
        let (fired, frames, now_ns) = threaded_stalled_run(Some(TICK_NS), pts_ordered, 3);
        assert!(
            fired >= 3,
            "the threaded runner must tick the fan-in while its only input stalls, \
             got {fired} (pts_ordered={pts_ordered})"
        );
        assert_eq!(
            frames,
            fired as u64 + 1,
            "the ZOH frames plus the input frame reached the sink (pts_ordered={pts_ordered})"
        );
        assert!(
            now_ns >= 3 * TICK_NS,
            "each tick advanced the deadline by one period (pts_ordered={pts_ordered})"
        );
    }
}

#[cfg(feature = "multi-thread")]
#[test]
fn the_threaded_runner_never_ticks_without_an_interval() {
    for pts_ordered in [false, true] {
        let (fired, frames, now_ns) = threaded_stalled_run(None, pts_ordered, 0);
        assert_eq!(
            fired, 0,
            "no interval declared, so no Tick (pts_ordered={pts_ordered})"
        );
        assert_eq!(
            frames, 1,
            "only the input's own frame reached the sink (pts_ordered={pts_ordered})"
        );
        assert_eq!(
            now_ns, 0,
            "nothing slept on the clock (pts_ordered={pts_ordered})"
        );
    }
}
