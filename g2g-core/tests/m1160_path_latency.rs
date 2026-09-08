//! M1160: the DAG runner's latency fold follows paths. A fan-in waits for its
//! slowest input branch and a run reports its slowest sink, where the fold used
//! to sum every node in the graph flat.
//!
//! M1161: a fan-in floors what the fold reports for the branches feeding it
//! (`min_upstream_latency_ns`), for an input slower than the ones it started with.
#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;

use g2g_core::fanout::MultiInputElement;
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::query::LatencyReport;
use g2g_core::runtime::{block_on, run_graph, GraphNode, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, Frame, FrameTiming, G2gError, Graph, OutputSink,
    PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

const FRAME_COUNT: u64 = 4;
const LINK_CAPACITY: usize = 2;

/// Branch minimums, far enough apart that a sum is not the larger of the two.
const SLOW_BRANCH_MIN_NS: u64 = 30_000_000;
const FAST_BRANCH_MIN_NS: u64 = 5_000_000;
/// Branch ceilings, likewise distinguishable from their sum.
const WIDE_CEILING_NS: u64 = 200_000_000;
const TIGHT_CEILING_NS: u64 = 40_000_000;
/// A floor above both branch minimums, so it is what the fold has to report.
const UPSTREAM_FLOOR_NS: u64 = 50_000_000;
/// What the shared source ahead of a tee contributes to both branches.
const SOURCE_MIN_NS: u64 = 3_000_000;
const SOURCE_CEILING_NS: u64 = 100_000_000;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

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
        FrameTiming::default(),
        sequence,
    ))
}

/// Pushes `FRAME_COUNT` frames then ends, declaring whatever latency it was given.
struct LatentSrc {
    latency: LatencyReport,
}

impl SourceLoop for LatentSrc {
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

    fn latency(&self) -> LatencyReport {
        self.latency
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for i in 0..FRAME_COUNT {
                out.push(frame(i)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(FRAME_COUNT)
        })
    }
}

/// Forwards every input's frames, contributing no latency of its own.
struct ForwardingMux {
    inputs: usize,
    upstream_floor_ns: u64,
}

impl MultiInputElement for ForwardingMux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        self.inputs
    }

    fn min_upstream_latency_ns(&self) -> u64 {
        self.upstream_floor_ns
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
            if matches!(packet, PipelinePacket::DataFrame(_)) {
                out.push(packet).await?;
            }
            Ok(())
        })
    }
}

/// Counts what reaches the end of the pipeline, declaring a given latency.
struct LatentSink {
    latency: LatencyReport,
    frames: u64,
}

impl LatentSink {
    fn new(latency: LatencyReport) -> Self {
        Self { latency, frames: 0 }
    }
}

impl AsyncElement for LatentSink {
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

    fn latency(&self) -> LatencyReport {
        self.latency
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

/// `LatentSrc(a) -> ForwardingMux <- LatentSrc(b)`, then a zero-latency sink.
fn two_branches_into_a_fan_in(a: LatencyReport, b: LatencyReport) -> LatencyReport {
    branches_under_floor(a, b, 0)
}

/// The same graph, with the fan-in flooring what its branches report.
fn branches_under_floor(
    a: LatencyReport,
    b: LatencyReport,
    upstream_floor_ns: u64,
) -> LatencyReport {
    let mut g: Graph<GraphNode> = Graph::new();
    let mux = g.add_muxer(
        GraphNode::muxer(ForwardingMux {
            inputs: 2,
            upstream_floor_ns,
        }),
        2,
    );
    let src_a = g.add_source(GraphNode::source(LatentSrc { latency: a }));
    let src_b = g.add_source(GraphNode::source(LatentSrc { latency: b }));
    let sink = g.add_sink(GraphNode::element(LatentSink::new(LatencyReport::ZERO)));
    g.link(src_a, mux.input(0)).expect("link branch a");
    g.link(src_b, mux.input(1)).expect("link branch b");
    g.link(mux.output(), sink).expect("link merged output");

    let stats = block_on(run_graph(g, &ZeroClock, LINK_CAPACITY)).expect("graph runs");
    assert_eq!(stats.frames_consumed, 2 * FRAME_COUNT);
    stats.latency
}

/// One source feeding a tee, with a differently latent sink on each branch.
fn one_source_into_two_sinks(a: LatencyReport, b: LatencyReport) -> LatencyReport {
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(LatentSrc {
        latency: LatencyReport::buffered(SOURCE_MIN_NS, Some(SOURCE_CEILING_NS)),
    }));
    let tee = g.add_tee(2);
    let sink_a = g.add_sink(GraphNode::element(LatentSink::new(a)));
    let sink_b = g.add_sink(GraphNode::element(LatentSink::new(b)));
    g.link(src, tee.input()).expect("link source to tee");
    g.link(tee.out(0), sink_a).expect("link branch a");
    g.link(tee.out(1), sink_b).expect("link branch b");

    let stats = block_on(run_graph(g, &ZeroClock, LINK_CAPACITY)).expect("graph runs");
    stats.latency
}

#[test]
fn a_fan_in_waits_for_its_slowest_branch_rather_than_the_sum() {
    let folded = two_branches_into_a_fan_in(
        LatencyReport::buffered(SLOW_BRANCH_MIN_NS, Some(WIDE_CEILING_NS)),
        LatencyReport::buffered(FAST_BRANCH_MIN_NS, Some(TIGHT_CEILING_NS)),
    );
    assert_eq!(
        folded.min_ns, SLOW_BRANCH_MIN_NS,
        "the fan-in is ready when its slower branch is, not when both have run"
    );
    assert_eq!(
        folded.max_ns,
        Some(TIGHT_CEILING_NS),
        "the branch that overflows first sets the run's ceiling"
    );
}

#[test]
fn an_unbounded_branch_leaves_a_finite_ceiling_alone() {
    let folded = two_branches_into_a_fan_in(
        LatencyReport::buffered(FAST_BRANCH_MIN_NS, None),
        LatencyReport::buffered(FAST_BRANCH_MIN_NS, Some(TIGHT_CEILING_NS)),
    );
    assert_eq!(folded.max_ns, Some(TIGHT_CEILING_NS));
    assert!(!folded.is_unsatisfiable());
}

#[test]
fn one_live_branch_makes_the_whole_run_live() {
    let folded = two_branches_into_a_fan_in(
        LatencyReport::live(SLOW_BRANCH_MIN_NS, None),
        LatencyReport::buffered(FAST_BRANCH_MIN_NS, None),
    );
    assert!(folded.live, "liveness crosses the fan-in from one branch");
    assert_eq!(folded.min_ns, SLOW_BRANCH_MIN_NS);
}

#[test]
fn a_run_reports_its_slowest_sink() {
    let folded = one_source_into_two_sinks(
        LatencyReport::buffered(SLOW_BRANCH_MIN_NS, Some(WIDE_CEILING_NS)),
        LatencyReport::buffered(FAST_BRANCH_MIN_NS, Some(TIGHT_CEILING_NS)),
    );
    assert_eq!(
        folded.min_ns,
        SOURCE_MIN_NS + SLOW_BRANCH_MIN_NS,
        "the shared source adds to the slower branch, and the branches do not add to each other"
    );
    assert_eq!(
        folded.max_ns,
        Some(SOURCE_CEILING_NS + TIGHT_CEILING_NS),
        "the tighter branch sets the ceiling the source's slack adds to"
    );
}

#[test]
fn a_fan_ins_floor_lifts_what_its_branches_report() {
    let branches = (
        LatencyReport::buffered(SLOW_BRANCH_MIN_NS, Some(WIDE_CEILING_NS)),
        LatencyReport::buffered(FAST_BRANCH_MIN_NS, Some(WIDE_CEILING_NS)),
    );
    let floored = branches_under_floor(branches.0, branches.1, UPSTREAM_FLOOR_NS);
    assert_eq!(
        floored.min_ns, UPSTREAM_FLOOR_NS,
        "the floor stands in for a branch slower than either negotiated one"
    );
    assert_eq!(
        floored.max_ns,
        Some(WIDE_CEILING_NS),
        "the floor moves the minimum only"
    );
}

#[test]
fn a_floor_under_the_branches_changes_nothing() {
    let branches = (
        LatencyReport::buffered(SLOW_BRANCH_MIN_NS, Some(WIDE_CEILING_NS)),
        LatencyReport::buffered(FAST_BRANCH_MIN_NS, Some(TIGHT_CEILING_NS)),
    );
    let floored = branches_under_floor(branches.0, branches.1, FAST_BRANCH_MIN_NS);
    assert_eq!(
        floored,
        two_branches_into_a_fan_in(branches.0, branches.1),
        "the branches already exceed it"
    );
}
