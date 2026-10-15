#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::sync::Mutex;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{run_graph, GraphNode, LatencyProfile};
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, CapsConstraint, ConfigureOutcome, G2gError, Graph,
    OutputSink, PipelinePacket, PushOutcome, Rate,
};
use g2g_plugins::clock::WallClock;
use g2g_plugins::fmp4demux::Fmp4Demux;
use g2g_plugins::mp4src::Mp4Src;

// av1_321x181.obu stream-copied by `ffmpeg -framerate 24`
const FIXTURE_FPS: u32 = 24;
const PROGRESSIVE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/av1_321x181_24fps.mp4"
);
// the same stream fragmented, its rate only in the fragment's tfhd
const FRAGMENTED: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/av1_321x181_24fps_frag.mp4"
);

static LAST_CAPS_CHANGE: Mutex<Option<Caps>> = Mutex::new(None);

struct CapsProbe;

impl AsyncElement for CapsProbe {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
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
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        if let PipelinePacket::CapsChanged(caps) = packet {
            *LAST_CAPS_CHANGE.lock().unwrap() = Some(caps);
        }
        Box::pin(async { Ok(()) })
    }
}

#[derive(Default)]
struct CapsCapture {
    framerate: Option<Rate>,
}

impl OutputSink for CapsCapture {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        if let Some(PipelinePacket::CapsChanged(Caps::CompressedVideo { framerate, .. })) =
            packet_slot.take()
        {
            self.framerate = Some(framerate);
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

#[tokio::test]
async fn mp4src_reports_the_file_frame_rate() {
    for fixture in [PROGRESSIVE, FRAGMENTED] {
        let mut graph: Graph<GraphNode> = Graph::new();
        let source = graph.add_source(GraphNode::source(Mp4Src::new(fixture)));
        let sink = graph.add_sink(GraphNode::element(CapsProbe));
        graph.link(source, sink).unwrap();
        run_graph(graph, &WallClock::new(), LatencyProfile::Throughput)
            .await
            .unwrap_or_else(|e| panic!("mp4src {fixture} runs: {e:?}"));
        let last_caps = LAST_CAPS_CHANGE
            .lock()
            .unwrap()
            .take()
            .expect("the sink saw a caps change");
        let Caps::CompressedVideo { framerate, .. } = last_caps else {
            panic!("compressed video at the sink, got {last_caps:?}");
        };
        assert_eq!(framerate, Rate::Fixed(FIXTURE_FPS << 16), "{fixture}");
    }
}

// fmp4demux's output alone cannot fixate, so it runs without a graph
#[tokio::test]
async fn fmp4demux_reads_the_frame_rate_from_the_fragment() {
    let mut demux = Fmp4Demux::new();
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::IsoBmff,
        })
        .expect("fmp4demux accepts an ISO-BMFF byte stream");
    let bytes = std::fs::read(FRAGMENTED).expect("reads the fixture");
    let frame = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming::default(),
        0,
    );
    let mut sink = CapsCapture::default();
    demux
        .process(PipelinePacket::DataFrame(frame), &mut sink)
        .await
        .expect("demuxes the fixture");
    assert_eq!(sink.framerate, Some(Rate::Fixed(FIXTURE_FPS << 16)));
}
