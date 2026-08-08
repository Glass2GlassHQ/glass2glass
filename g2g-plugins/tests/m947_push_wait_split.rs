//! M947: the per-element summary splits compute from downstream backpressure.
//!
//! M399 timed the whole `process()` future, so an element awaiting a full
//! downstream link reported the consumer's pacing as if it were its own cost.
//! The runner now banks the time a push spends awaiting capacity on the
//! producing element's probe and takes it out of `proc`, reporting it separately
//! as `push_wait`.
//!
//! These tests drive the real graph runner twice over the same shape: once with a
//! sink that consumes slowly (the transform must read as blocked, not busy) and
//! once with a slow transform and a fast sink (the transform must read as busy,
//! not blocked).
//!
//! M951 carries the same split down to the per-frame journey: an observed run
//! over the same shape must show the stall as one stage's blocked segment rather
//! than inflating its work segment.

use core::future::Future;
use core::pin::Pin;
use core::time::Duration;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{
    run_graph, run_graph_observed, ElementLatency, GraphNode, JourneyStage, Observer, RunStats,
    SourceLoop,
};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, G2gError, Graph, MemoryDomain, OutputSink,
    PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

const MS: u64 = 1_000_000;
/// Frames per run: enough that the link fills and the majority of pushes block,
/// so the p50 (not just the max) lands in the blocked case.
const FRAMES: u64 = 12;
/// Small enough that only a couple of frames cross before the sink's pace rules.
const LINK_CAPACITY: usize = 2;
const SINK_DELAY_MS: u64 = 15;
const TRANSFORM_DELAY_MS: u64 = 3;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(64),
        height: Dim::Fixed(64),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
    }
}

fn make_frame(seq: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
        timing: FrameTiming::default(),
        sequence: seq,
        meta: Default::default(),
    }
}

/// Emits `frames` DataFrames then Eos, as fast as the link takes them.
struct FrameSrc {
    frames: u64,
}

impl SourceLoop for FrameSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(caps()))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for seq in 0..self.frames {
                out.push(PipelinePacket::DataFrame(make_frame(seq))).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.frames)
        })
    }
}

/// Identity transform that sleeps `delay_ms` per data frame before forwarding,
/// so its compute cost is a known quantity independent of downstream pacing.
struct DelayTransform {
    delay_ms: u64,
}

impl AsyncElement for DelayTransform {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        let delay_ms = self.delay_ms;
        Box::pin(async move {
            match packet {
                PipelinePacket::Eos => Ok(()),
                PipelinePacket::DataFrame(_) => {
                    if delay_ms > 0 {
                        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    }
                    out.push(packet).await?;
                    Ok(())
                }
                other => {
                    out.push(other).await?;
                    Ok(())
                }
            }
        })
    }
}

/// Terminal sink that sleeps `delay_ms` per data frame, the stand-in for a
/// clock-paced display sink: it back-pressures everything upstream of it.
struct DelaySink {
    delay_ms: u64,
}

impl AsyncElement for DelaySink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        let delay_ms = self.delay_ms;
        Box::pin(async move {
            if matches!(packet, PipelinePacket::DataFrame(_)) && delay_ms > 0 {
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            }
            Ok(())
        })
    }
}

/// source -> transform -> sink with the given per-frame delays.
fn chain(transform_delay_ms: u64, sink_delay_ms: u64) -> Graph<GraphNode> {
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(FrameSrc { frames: FRAMES }));
    let tx = g.add_transform(GraphNode::element(DelayTransform {
        delay_ms: transform_delay_ms,
    }));
    let sink = g.add_sink(GraphNode::element(DelaySink {
        delay_ms: sink_delay_ms,
    }));
    g.link(src, tx).unwrap();
    g.link(tx, sink).unwrap();
    g
}

/// The chain over the real graph runner.
async fn run_chain(transform_delay_ms: u64, sink_delay_ms: u64) -> RunStats {
    run_graph(
        chain(transform_delay_ms, sink_delay_ms),
        &ZeroClock,
        LINK_CAPACITY,
    )
    .await
    .expect("graph runs")
}

/// The same chain under an observer, which mints journey-recording probes, so
/// the snapshot carries one frame's per-stage path.
async fn run_chain_observed(transform_delay_ms: u64, sink_delay_ms: u64, obs: &Observer) {
    run_graph_observed(
        chain(transform_delay_ms, sink_delay_ms),
        &ZeroClock,
        LINK_CAPACITY,
        obs,
        None,
    )
    .await
    .expect("observed graph runs");
}

fn stage<'a>(stages: &'a [JourneyStage], name: &str) -> &'a JourneyStage {
    stages
        .iter()
        .find(|s| s.name == name)
        .unwrap_or_else(|| panic!("no journey stage named {name}"))
}

fn row<'a>(stats: &'a RunStats, name: &str) -> &'a ElementLatency {
    stats
        .per_element
        .iter()
        .find(|e| e.name == name)
        .unwrap_or_else(|| panic!("no per-element row named {name}"))
}

#[tokio::test]
async fn a_paced_sink_reads_as_blocked_not_busy() {
    let stats = run_chain(0, SINK_DELAY_MS).await;
    let tx_row = row(&stats, "DelayTransform0");
    let sink_row = row(&stats, "DelaySink0");

    assert_eq!(tx_row.proc.count, FRAMES, "every frame timed");
    assert_eq!(
        tx_row.push_wait.count, tx_row.proc.count,
        "one push-wait sample per timed frame"
    );

    // The transform does no work of its own: the sink's pace must land in
    // push_wait, and proc must stay near zero.
    assert!(
        tx_row.push_wait.p50_ns >= 4 * MS,
        "transform blocked p50 = {} ns, expected the sink's pace",
        tx_row.push_wait.p50_ns
    );
    assert!(
        tx_row.proc.p50_ns < MS,
        "transform proc p50 = {} ns, backpressure leaked into compute",
        tx_row.proc.p50_ns
    );

    // The sink itself has no output link, so nothing is attributed to it.
    assert_eq!(
        sink_row.push_wait.max_ns, 0,
        "a sink pushes nowhere, so it can never be blocked"
    );
    assert!(
        sink_row.proc.p50_ns >= MS,
        "sink proc p50 = {} ns, its own pacing is its own cost",
        sink_row.proc.p50_ns
    );

    let report = stats.report();
    assert!(report.contains("blocked p50"), "report:\n{report}");
}

#[tokio::test]
async fn a_fast_sink_leaves_the_slow_transform_reading_as_busy() {
    let stats = run_chain(TRANSFORM_DELAY_MS, 0).await;
    let tx_row = row(&stats, "DelayTransform0");

    assert_eq!(tx_row.proc.count, FRAMES);
    // Nothing back-pressures this transform, so its whole cost stays in proc.
    assert!(
        tx_row.proc.p50_ns >= MS,
        "transform proc p50 = {} ns, its sleep should dominate",
        tx_row.proc.p50_ns
    );
    assert!(
        tx_row.push_wait.p50_ns < MS,
        "transform blocked p50 = {} ns, nothing should have held it up",
        tx_row.push_wait.p50_ns
    );
    assert!(
        tx_row.proc.p50_ns > tx_row.push_wait.p50_ns,
        "compute {} ns must dominate wait {} ns on an unpaced graph",
        tx_row.proc.p50_ns,
        tx_row.push_wait.p50_ns
    );
}

/// M951: the per-frame journey splits the same way. The transform does no work
/// of its own, so the frame's time at that stage belongs to `blocked_ns`; the
/// sink pushes nowhere, so all of its time is work.
#[tokio::test]
async fn a_journey_stage_reads_a_paced_sink_as_blocked_not_work() {
    let obs = Observer::new();
    run_chain_observed(0, SINK_DELAY_MS, &obs).await;

    let j = obs.snapshot().journey.expect("a frame crossed every stage");
    assert!(!j.truncated, "the chain is linear all the way to the sink");
    let tx = stage(&j.stages, "DelayTransform0");
    let sink = stage(&j.stages, "DelaySink0");

    assert!(
        tx.blocked_ns >= 4 * MS,
        "transform blocked {} ns on frame {}, expected the sink's pace",
        tx.blocked_ns,
        j.sequence
    );
    assert!(
        tx.work_ns < MS,
        "transform work {} ns, backpressure leaked into the work segment",
        tx.work_ns
    );
    assert_eq!(sink.blocked_ns, 0, "a sink pushes nowhere");
    assert!(
        sink.work_ns >= MS,
        "sink work {} ns, its own pacing is its own cost",
        sink.work_ns
    );
}

/// The mirror case: nothing back-pressures the transform, so its journey stage
/// keeps its whole span as work.
#[tokio::test]
async fn a_journey_stage_keeps_unblocked_time_as_work() {
    let obs = Observer::new();
    run_chain_observed(TRANSFORM_DELAY_MS, 0, &obs).await;

    let j = obs.snapshot().journey.expect("a frame crossed every stage");
    let tx = stage(&j.stages, "DelayTransform0");
    assert!(
        tx.work_ns >= MS,
        "transform work {} ns, its sleep should dominate",
        tx.work_ns
    );
    assert!(
        tx.work_ns > tx.blocked_ns,
        "work {} ns must dominate blocked {} ns on an unpaced graph",
        tx.work_ns,
        tx.blocked_ns
    );
}
