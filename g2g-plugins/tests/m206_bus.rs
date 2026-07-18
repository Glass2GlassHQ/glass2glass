//! M206 - bus breadth: `StreamStart`. The runner's source arm posts one
//! `BusMessage::StreamStart` per source before it produces, so an application
//! can bracket each stream's lifetime (StreamStart .. Eos), the
//! `GST_MESSAGE_STREAM_START` analog.

use std::pin::Pin;

use core::future::Future;

use g2g_core::runtime::{run_graph_with_bus, run_muxer_sink_with_bus, DynSourceLoop, GraphNode};
use g2g_core::{
    AsyncElement, Bus, BusMessage, Caps, ConfigureOutcome, Dim, G2gError, Graph, OutputSink,
    PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::mux::InterleaveMux;
use g2g_plugins::videotestsrc::VideoTestSrc;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// An order-independent sink (accepts anything): the muxer interleaves two
/// sources, so a monotonic-checking sink like `FakeSink` would reject it.
#[derive(Default)]
struct AcceptSink;
impl AsyncElement for AcceptSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;
    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }
    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
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

fn count_stream_starts(bus: &Bus) -> usize {
    let mut n = 0;
    while let Some(m) = bus.try_recv() {
        if matches!(m, BusMessage::StreamStart) {
            n += 1;
        }
    }
    n
}

#[tokio::test]
async fn single_source_posts_one_stream_start() {
    let (bus, handle) = Bus::new(64);
    {
        let mut g: Graph<GraphNode> = Graph::new();
        let src = g.add_source(GraphNode::source(VideoTestSrc::new(8, 8, 30, 4)));
        let sink = g.add_sink(GraphNode::element(FakeSink::new()));
        g.link(src, sink).unwrap();
        run_graph_with_bus(g, &NullClock, 4, &handle)
            .await
            .expect("runs with bus");
    }
    assert_eq!(
        count_stream_starts(&bus),
        1,
        "one StreamStart for the single source"
    );
}

#[tokio::test]
async fn each_source_posts_its_own_stream_start() {
    let caps = Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(8),
        height: Dim::Fixed(8),
        framerate: Rate::Fixed(30 << 16),
    };
    let (bus, handle) = Bus::new(64);
    let mut a = VideoTestSrc::new(8, 8, 30, 3);
    let mut b = VideoTestSrc::new(8, 8, 30, 3);
    let mut mux = InterleaveMux::new(2, caps);
    let mut sink = AcceptSink;
    {
        let sources: Vec<&mut dyn DynSourceLoop> = vec![&mut a, &mut b];
        run_muxer_sink_with_bus(sources, &mut mux, &mut sink, &NullClock, 4, &handle)
            .await
            .expect("muxer pipeline runs with bus");
    }
    assert_eq!(count_stream_starts(&bus), 2, "one StreamStart per source");
}
