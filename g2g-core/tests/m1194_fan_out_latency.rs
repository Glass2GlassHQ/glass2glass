#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;

use g2g_core::fanout::{MultiOutputElement, MultiOutputSink};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::query::LatencyReport;
use g2g_core::runtime::{block_on, run_graph, GraphNode, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, Frame, FrameTiming, G2gError, Graph, OutputSink,
    PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

const FRAME_COUNT: u64 = 4;
const LINK_CAPACITY: usize = 2;
const DEMUX_PORT: usize = 0;
const DEMUX_PORT_COUNT: u8 = 1;
// any nonzero value
const DEMUX_LATENCY_NS: u64 = 5_000_000;

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

struct LatentDemux {
    latency: LatencyReport,
}

impl MultiOutputElement for LatentDemux {
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
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if matches!(packet, PipelinePacket::DataFrame(_)) {
                out.push_to(DEMUX_PORT, packet).await?;
            }
            Ok(())
        })
    }

    fn latency(&self) -> LatencyReport {
        self.latency
    }
}

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

fn folded_latency(latency: LatencyReport) -> LatencyReport {
    let mut graph: Graph<GraphNode> = Graph::new();
    let src = graph.add_source(GraphNode::source(Src));
    let demux = graph.add_demux(GraphNode::demux(LatentDemux { latency }), DEMUX_PORT_COUNT);
    let sink = graph.add_sink(GraphNode::element(CountSink::default()));
    graph
        .link(src, demux.input())
        .expect("link source to demux");
    graph
        .link(demux.out(DEMUX_PORT as u8), sink)
        .expect("link demux port");

    let stats = block_on(run_graph(graph, &ZeroClock, LINK_CAPACITY)).expect("graph runs");
    assert_eq!(stats.frames_consumed, FRAME_COUNT);
    stats.latency
}

#[test]
fn a_demux_raises_the_runs_aggregate_latency() {
    let neutral = folded_latency(LatencyReport::ZERO);
    assert_eq!(neutral.min_ns, 0, "nothing on this path declares latency");

    let declared = folded_latency(LatencyReport::buffered(
        DEMUX_LATENCY_NS,
        Some(DEMUX_LATENCY_NS),
    ));
    assert_eq!(
        declared.min_ns,
        neutral.min_ns + DEMUX_LATENCY_NS,
        "the demux's declared minimum reached the fold"
    );
    assert_eq!(
        declared.max_ns,
        neutral.max_ns.map(|max| max + DEMUX_LATENCY_NS),
        "and so did its ceiling"
    );
}

#[test]
fn a_live_demux_makes_the_run_live() {
    let declared = folded_latency(LatencyReport::live(DEMUX_LATENCY_NS, None));
    assert!(
        declared.live,
        "no source here is live, so the liveness can only be the demux's"
    );
    assert_eq!(declared.min_ns, DEMUX_LATENCY_NS);
    assert_eq!(declared.max_ns, None, "an unbounded max stays unbounded");
}
