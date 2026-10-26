//! M1242 QoS relay through the fan arms: a sink's lateness report crosses a tee,
//! a demux and a muxer on its way to the source, as M175 carried it through a
//! transform. Each graph puts a QoS-signalling sink behind the fan node and
//! checks that `VideoTestSrc` skipped ahead past its unskipped timeline.

#![cfg(all(feature = "std", feature = "metadata"))]

use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use g2g_core::element::DynAsyncElement;
use g2g_core::runtime::{run_fanin_sink, run_graph, run_source_fanout, DynSourceLoop, GraphNode};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, Frame, G2gError, Graph, Merger,
    OutputSink, PipelineClock, PipelinePacket, QosMessage, Rate, RawVideoFormat, Router,
};
use g2g_plugins::batchmux::BatchMux;
use g2g_plugins::streamdemux::StreamDemux;
use g2g_plugins::videotestsrc::VideoTestSrc;

const FPS: u32 = 30;
/// One frame period in ns at FPS.
const FRAME_NS: u64 = 1_000_000_000 / FPS as u64;
/// Frames each source emits.
const TARGET: u64 = 30;
/// The last frame's pts when the source skips nothing.
const UNSKIPPED_LAST_PTS_NS: u64 = (TARGET - 1) * FRAME_NS;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn rgba() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(8),
        height: Dim::Fixed(8),
        framerate: Rate::Fixed(FPS << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn source() -> VideoTestSrc {
    VideoTestSrc::new(8, 8, FPS, TARGET)
}

/// Sink that reports a fixed lateness on every frame it consumes, about three
/// frame periods behind, so the source upstream sheds load.
struct LateSink {
    pending: Option<QosMessage>,
    latest_pts_ns: Arc<AtomicU64>,
}

fn late_sink(latest_pts_ns: &Arc<AtomicU64>) -> LateSink {
    LateSink {
        pending: None,
        latest_pts_ns: Arc::clone(latest_pts_ns),
    }
}

impl AsyncElement for LateSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                self.latest_pts_ns
                    .fetch_max(f.timing.pts_ns, Ordering::Relaxed);
                self.pending = Some(QosMessage {
                    jitter_ns: 3 * FRAME_NS as i64,
                    running_time_ns: f.timing.pts_ns,
                    taken_ns: 0,
                    downstream_proc_ns: 0,
                });
            }
            Ok(())
        })
    }
    fn take_qos(&mut self) -> Option<QosMessage> {
        self.pending.take()
    }
}

#[tokio::test]
async fn the_source_skips_on_a_report_from_behind_a_demux() {
    let latest_pts_ns = Arc::new(AtomicU64::new(0));
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(source()));
    let demux = g.add_demux(
        GraphNode::demux(StreamDemux::new(
            rgba(),
            vec![rgba(), rgba()],
            |f: &Frame| (f.sequence % 2) as usize,
        )),
        2,
    );
    let s0 = g.add_sink(GraphNode::element(late_sink(&latest_pts_ns)));
    let s1 = g.add_sink(GraphNode::element(late_sink(&latest_pts_ns)));
    g.link(src, demux.input()).unwrap();
    g.link(demux.out(0), s0).unwrap();
    g.link(demux.out(1), s1).unwrap();

    let stats = run_graph(g, &ZeroClock, 4).await.expect("demux graph runs");
    assert!(stats.frames_consumed > 0, "frames reached the sinks");
    assert!(
        latest_pts_ns.load(Ordering::Relaxed) > UNSKIPPED_LAST_PTS_NS,
        "the source skipped on the report relayed past the demux"
    );
}

#[tokio::test]
async fn the_source_skips_on_a_report_from_behind_a_tee() {
    let latest_pts_ns = Arc::new(AtomicU64::new(0));
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(source()));
    let tee = g.add_tee(2);
    let s0 = g.add_sink(GraphNode::element(late_sink(&latest_pts_ns)));
    let s1 = g.add_sink(GraphNode::element(late_sink(&latest_pts_ns)));
    g.link(src, tee.input()).unwrap();
    g.link(tee.out(0), s0).unwrap();
    g.link(tee.out(1), s1).unwrap();

    let stats = run_graph(g, &ZeroClock, 4).await.expect("tee graph runs");
    assert!(stats.frames_consumed > 0, "frames reached the sinks");
    assert!(
        latest_pts_ns.load(Ordering::Relaxed) > UNSKIPPED_LAST_PTS_NS,
        "the source skipped on the report relayed past the tee"
    );
}

#[tokio::test]
async fn both_sources_skip_on_a_report_from_behind_a_muxer() {
    let latest_pts_ns = Arc::new(AtomicU64::new(0));
    let mut g: Graph<GraphNode> = Graph::new();
    let first = g.add_source(GraphNode::source(source()));
    let second = g.add_source(GraphNode::source(source()));
    let mux = g.add_muxer(GraphNode::muxer(BatchMux::new(2)), 2);
    let sink = g.add_sink(GraphNode::element(late_sink(&latest_pts_ns)));
    g.link(first, mux.input(0)).unwrap();
    g.link(second, mux.input(1)).unwrap();
    g.link(mux.output(), sink).unwrap();

    let stats = run_graph(g, &ZeroClock, 4).await.expect("muxer graph runs");
    assert!(stats.frames_consumed > 0, "batches reached the sink");
    assert!(
        latest_pts_ns.load(Ordering::Relaxed) > UNSKIPPED_LAST_PTS_NS,
        "the sources skipped on the report relayed past the muxer"
    );
}

#[tokio::test]
async fn the_source_skips_on_a_report_through_the_fanout_runner() {
    let latest_pts_ns = Arc::new(AtomicU64::new(0));
    let mut src = source();
    let mut router = Router::new(2);
    let mut s0 = late_sink(&latest_pts_ns);
    let mut s1 = late_sink(&latest_pts_ns);
    let sinks: Vec<&mut dyn DynAsyncElement> = vec![&mut s0, &mut s1];

    let stats = run_source_fanout(&mut src, &mut router, sinks, &ZeroClock, 4)
        .await
        .expect("fan-out runs");
    assert!(stats.frames_consumed > 0, "frames reached the sinks");
    assert!(
        latest_pts_ns.load(Ordering::Relaxed) > UNSKIPPED_LAST_PTS_NS,
        "the source skipped on the report relayed past the router"
    );
}

#[tokio::test]
async fn the_selected_source_skips_on_a_report_through_the_fanin_runner() {
    let latest_pts_ns = Arc::new(AtomicU64::new(0));
    let mut selected = source();
    let mut idle = source();
    let mut merger = Merger::new(2);
    let mut sink = late_sink(&latest_pts_ns);
    let sources: Vec<&mut dyn DynSourceLoop> = vec![&mut selected, &mut idle];

    let stats = run_fanin_sink(sources, &mut merger, &mut sink, &ZeroClock, 4)
        .await
        .expect("fan-in runs");
    assert!(stats.frames_consumed > 0, "frames reached the sink");
    assert!(
        latest_pts_ns.load(Ordering::Relaxed) > UNSKIPPED_LAST_PTS_NS,
        "the selected source skipped on the report relayed past the merger"
    );
}
