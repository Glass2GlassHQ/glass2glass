//! DAG runner fan-in: `run_graph` over muxer nodes (`InterleaveMux`), so the
//! canonical "split, process two ways, recombine" topology runs end to end.
//! Pure-fake elements (no hardware).

use core::future::Future;
use core::pin::Pin;

use g2g_core::runtime::{run_graph, GraphNode};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, G2gError, Graph, OutputSink,
    PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};
use g2g_plugins::mux::InterleaveMux;
use g2g_plugins::videocrop::VideoCrop;
use g2g_plugins::videoflip::{FlipMethod, VideoFlip};
use g2g_plugins::videotestsrc::VideoTestSrc;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Minimal sink that accepts any packet. Unlike `FakeSink` it makes no
/// monotonic-sequence assumption, so it tolerates frames a muxer interleaves
/// from multiple sources (their sequence numbers overlap). The frame count the
/// tests assert comes from `RunStats` (the runner's sink arm counts), so the
/// sink only needs to accept.
#[derive(Default)]
struct AnySink;

impl AsyncElement for AnySink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

fn rgba(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

#[tokio::test]
async fn muxer_fans_in_two_sources() {
    // Two sources of unequal length combine at the muxer; every frame reaches
    // the sink and a single merged Eos ends the stream.
    let mut g: Graph<GraphNode> = Graph::new();
    let s0 = g.add_source(GraphNode::source(VideoTestSrc::new(8, 8, 30, 4)));
    let s1 = g.add_source(GraphNode::source(VideoTestSrc::new(8, 8, 30, 3)));
    let mux = g.add_muxer(GraphNode::muxer(InterleaveMux::new(2, rgba(8, 8))), 2);
    let sink = g.add_sink(GraphNode::element(AnySink));
    g.link(s0, mux.input(0)).unwrap();
    g.link(s1, mux.input(1)).unwrap();
    g.link(mux.output(), sink).unwrap();

    let stats = run_graph(g, &NullClock, 4).await.expect("fan-in DAG runs");
    assert_eq!(stats.frames_emitted, 7, "4 + 3 source frames");
    assert_eq!(stats.frames_consumed, 7, "the muxer forwarded every frame");
}

#[tokio::test]
async fn tee_to_muxer_diamond_runs() {
    // src -> tee(2) -> {flip -> mux.in0, crop -> mux.in1} -> sink. The two
    // branches transform independently and recombine at the muxer.
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(VideoTestSrc::new(8, 8, 30, 4)));
    let tee = g.add_tee(2);
    let flip = g.add_transform(GraphNode::element(VideoFlip::new(FlipMethod::Rotate180)));
    let crop = g.add_transform(GraphNode::element(VideoCrop::new(0, 0, 4, 4)));
    let mux = g.add_muxer(GraphNode::muxer(InterleaveMux::new(2, rgba(8, 8))), 2);
    let sink = g.add_sink(GraphNode::element(AnySink));
    g.link(src, tee.input()).unwrap();
    g.link(tee.out(0), flip).unwrap();
    g.link(tee.out(1), crop).unwrap();
    g.link(flip, mux.input(0)).unwrap();
    g.link(crop, mux.input(1)).unwrap();
    g.link(mux.output(), sink).unwrap();

    let stats = run_graph(g, &NullClock, 4).await.expect("tee->mux diamond runs");
    assert_eq!(stats.frames_emitted, 4, "source emitted 4 frames");
    assert_eq!(
        stats.frames_consumed, 8,
        "both branches' 4 frames recombined at the muxer reach the sink"
    );
}
