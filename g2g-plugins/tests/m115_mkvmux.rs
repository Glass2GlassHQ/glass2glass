//! M115 Matroska / WebM mux, end to end through the DAG runner: a VP9 source
//! feeds `MkvMux`, whose `Caps::ByteStream{Matroska}` output feeds `MkvDemux`,
//! which recovers the frames to a sink. Proves the mux/demux pair composes and
//! that `ByteStream{Matroska}` negotiates as an interior link between transforms.

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_graph, GraphNode, GraphNodeRef, SourceLoop};
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError, Graph,
    MemoryDomain, OutputSink, PipelineClock, PipelinePacket, Rate, VideoCodec,
};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::mkvdemux::MkvDemux;
use g2g_plugins::mkvmux::MkvMux;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn vp9_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::Vp9,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Emits three opaque VP9 frames, each its own `DataFrame`, then EOS.
struct Vp9Source {
    remaining: u8,
}
impl SourceLoop for Vp9Source {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(vp9_caps()))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(vp9_caps()))))
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let mut n = 0u64;
            for i in 0..self.remaining {
                let payload = vec![0x10 | i, 0x20, 0x30];
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(payload.into_boxed_slice())),
                    FrameTiming {
                        pts_ns: (i as u64) * 40_000_000,
                        ..FrameTiming::default()
                    },
                    n,
                );
                out.push(PipelinePacket::DataFrame(frame)).await?;
                n += 1;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(n)
        })
    }
}

#[tokio::test]
async fn mkvmux_then_mkvdemux_round_trips_in_runner() {
    let mut graph: Graph<GraphNode> = Graph::new();
    let src = graph.add_source(GraphNodeRef::Source(Box::new(Vp9Source { remaining: 3 })));
    let mux = graph.add_transform(GraphNodeRef::element(MkvMux::new()));
    let demux = graph.add_transform(GraphNodeRef::element(MkvDemux::new()));
    let sink = graph.add_sink(GraphNodeRef::element(FakeSink::new()));
    graph.link(src, mux).unwrap();
    graph.link(mux, demux).unwrap();
    graph.link(demux, sink).unwrap();

    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .expect("mux -> demux runs");
    assert_eq!(
        stats.frames_consumed, 3,
        "three VP9 frames survive mux + demux to the sink"
    );
}
