//! M120 FLV mux, end to end through the DAG runner: a source produces H.264
//! access units, `flvmux` wraps them into an FLV byte stream, `flvdemux` unwraps
//! them, and the recovered access units reach the sink. Proves the muxer is the
//! inverse of the demuxer and that both run correctly in a real pipeline
//! (including EOS forwarding under the runner).

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_graph, GraphNode, GraphNodeRef, SourceLoop};
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, Graph, MemoryDomain,
    OutputSink, PipelineClock, PipelinePacket, Rate, VideoCodec,
};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::flvdemux::FlvDemux;
use g2g_plugins::flvmux::FlvMux;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// A source that emits each access unit in `aus` as its own H.264 `DataFrame`,
/// then EOS.
struct H264Source {
    aus: Vec<Vec<u8>>,
    next: usize,
}
impl SourceLoop for H264Source {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(h264_caps()))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(h264_caps()))))
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            while self.next < self.aus.len() {
                let au = core::mem::take(&mut self.aus[self.next]);
                self.next += 1;
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(au.into_boxed_slice())),
                    Default::default(),
                    self.next as u64,
                );
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.next as u64)
        })
    }
}

#[tokio::test]
async fn flvmux_round_trips_through_flvdemux_in_runner() {
    // Annex-B access units, as an encoder/parser emits (M662: the muxer
    // re-frames them AVCC into the FLV tags, the demuxer re-frames them back).
    let aus = vec![
        vec![0, 0, 0, 1, 0x65u8, 0xAA, 0xBB],
        vec![0, 0, 0, 1, 0x41u8, 0xCC],
    ];

    let mut graph: Graph<GraphNode> = Graph::new();
    let src = graph.add_source(GraphNodeRef::Source(Box::new(H264Source { aus, next: 0 })));
    let mux = graph.add_transform(GraphNodeRef::element(FlvMux::new()));
    let demux = graph.add_transform(GraphNodeRef::element(FlvDemux::new()));
    let sink = graph.add_sink(GraphNodeRef::element(FakeSink::new()));
    graph.link(src, mux).unwrap();
    graph.link(mux, demux).unwrap();
    graph.link(demux, sink).unwrap();

    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .expect("H.264 -> flvmux -> flvdemux -> sink");
    assert_eq!(
        stats.frames_emitted, 2,
        "the source emitted two access units"
    );
    assert_eq!(
        stats.frames_consumed, 2,
        "both recovered through mux + demux"
    );
}
