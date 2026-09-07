//! M1159: a fan-in contributes to the DAG runner's latency fold. The muxer node
//! used to be skipped, so an element that holds frames back (a fallback switch
//! waiting out a stall) reported nothing to the pipeline latency query.
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
/// The fan-in's own contribution, arbitrary but nonzero.
const FANIN_LATENCY_NS: u64 = 7_000_000;

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

/// Pushes `FRAME_COUNT` frames then ends, contributing no latency of its own.
struct Src;

impl SourceLoop for Src {
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
            for i in 0..FRAME_COUNT {
                out.push(frame(i)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(FRAME_COUNT)
        })
    }
}

/// Forwards its single input and declares whatever latency the test gave it.
struct LatentMux {
    latency: LatencyReport,
}

impl MultiInputElement for LatentMux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        1
    }

    fn latency(&self) -> LatencyReport {
        self.latency
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

/// Counts what reaches the end of the pipeline.
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

/// `Src -> LatentMux -> CountSink` through the DAG runner, reporting the fold.
fn folded_latency(latency: LatencyReport) -> LatencyReport {
    let mut g: Graph<GraphNode> = Graph::new();
    let mux = g.add_muxer(GraphNode::muxer(LatentMux { latency }), 1);
    let src = g.add_source(GraphNode::source(Src));
    let sink = g.add_sink(GraphNode::element(CountSink::default()));
    g.link(src, mux.input(0)).expect("link source to pad 0");
    g.link(mux.output(), sink).expect("link merged output");

    let stats = block_on(run_graph(g, &ZeroClock, LINK_CAPACITY)).expect("graph runs");
    assert_eq!(stats.frames_consumed, FRAME_COUNT);
    stats.latency
}

#[test]
fn a_fan_in_raises_the_runs_aggregate_latency() {
    let neutral = folded_latency(LatencyReport::ZERO);
    assert_eq!(neutral.min_ns, 0, "nothing on this path declares latency");

    let declared = folded_latency(LatencyReport::buffered(
        FANIN_LATENCY_NS,
        Some(FANIN_LATENCY_NS),
    ));
    assert_eq!(
        declared.min_ns,
        neutral.min_ns + FANIN_LATENCY_NS,
        "the fan-in's declared minimum reached the fold"
    );
    assert_eq!(
        declared.max_ns,
        neutral.max_ns.map(|max| max + FANIN_LATENCY_NS),
        "and so did its ceiling"
    );
}

#[test]
fn a_live_fan_in_makes_the_run_live() {
    let declared = folded_latency(LatencyReport::live(FANIN_LATENCY_NS, None));
    assert!(
        declared.live,
        "no source here is live, so the liveness can only be the fan-in's"
    );
    assert_eq!(declared.min_ns, FANIN_LATENCY_NS);
    assert_eq!(declared.max_ns, None, "an unbounded max stays unbounded");
}
