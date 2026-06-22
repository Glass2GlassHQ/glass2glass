//! M210 demux node in `run_graph` + `gst-launch` fan-out wiring. A content-
//! routing `MultiOutputElement` (here `StreamDemux`) becomes a first-class DAG
//! node via `Graph::add_demux`: structurally a tee (so it negotiates as one at
//! startup), but it routes each frame to a chosen output and announces per-output
//! caps. The symmetric fan-out counterpart to the muxer fan-in (M122/M208).
//!
//! There is no content-agnostic default demux in the registry (routing is
//! inherently stream-specific, like the muxer side has specific muxers), so the
//! launch path is exercised with a demux registered via `register_demux`.

use core::future::Future;
use core::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use g2g_core::runtime::{
    parse_launch, run_graph, DemuxFactory, GraphNode, LaunchFactory, Registry,
};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, Frame, G2gError, Graph, OutputSink,
    PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};
use g2g_plugins::streamdemux::StreamDemux;
use g2g_plugins::videotestsrc::VideoTestSrc;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn rgba() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(8),
        height: Dim::Fixed(8),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Counts the frames it receives (shared so the test can read it after the graph
/// runs). Accepts any caps, including the demux's per-port retyping `CapsChanged`.
#[derive(Default)]
struct CountSink {
    frames: Arc<AtomicUsize>,
}

impl AsyncElement for CountSink {
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
        if let PipelinePacket::DataFrame(_) = packet {
            self.frames.fetch_add(1, Ordering::Relaxed);
        }
        Box::pin(async { Ok(()) })
    }
}

#[tokio::test]
async fn demux_node_routes_frames_to_branches() {
    // Source emits 4 frames (sequence 0..4); the demux routes by sequence parity:
    // even -> port 0, odd -> port 1. Each branch must receive exactly 2 frames.
    let even = Arc::new(AtomicUsize::new(0));
    let odd = Arc::new(AtomicUsize::new(0));

    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(VideoTestSrc::new(8, 8, 30, 4)));
    // Per-port caps differ from the input, so the first frame on each port emits a
    // retyping CapsChanged (consumed harmlessly by the accepts-any sinks).
    let demux = g.add_demux(
        GraphNode::demux(StreamDemux::new(
            rgba(),
            std::vec![rgba(), rgba()],
            |f: &Frame| (f.sequence % 2) as usize,
        )),
        2,
    );
    let s0 = g.add_sink(GraphNode::element(CountSink { frames: even.clone() }));
    let s1 = g.add_sink(GraphNode::element(CountSink { frames: odd.clone() }));
    g.link(src, demux.input()).unwrap();
    g.link(demux.out(0), s0).unwrap();
    g.link(demux.out(1), s1).unwrap();

    let stats = run_graph(g, &NullClock, 4).await.expect("demux DAG runs");
    assert_eq!(stats.frames_emitted, 4, "source emitted 4 frames");
    assert_eq!(stats.frames_consumed, 4, "every frame reached one branch");
    assert_eq!(even.load(Ordering::Relaxed), 2, "even-sequence frames routed to port 0");
    assert_eq!(odd.load(Ordering::Relaxed), 2, "odd-sequence frames routed to port 1");
}

/// A parity demux built from an output count, for `register_demux`: routes by
/// sequence parity, one pass-through port-caps slot per output.
fn build_parity_demux(outputs: usize) -> Box<dyn g2g_core::runtime::DynMultiOutputElement> {
    let port_caps = (0..outputs).map(|_| rgba()).collect();
    Box::new(StreamDemux::new(rgba(), port_caps, |f: &Frame| (f.sequence % 2) as usize))
}

/// AnySink registered for the launch path (the runner counts frames via RunStats).
#[derive(Default)]
struct AnySink;
impl AsyncElement for AnySink {
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
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

#[tokio::test]
async fn gst_launch_demux_fans_out() {
    let mut reg = Registry::new();
    reg.register_source(g2g_core::runtime::SourceFactory::new("videotestsrc", rgba(), || {
        Box::new(VideoTestSrc::new(8, 8, 30, 4))
    }));
    reg.register_launch(LaunchFactory::new("anysink", std::vec::Vec::new(), || Box::new(AnySink)));
    reg.register_demux(DemuxFactory::new("paritydemux", build_parity_demux));

    // A demux node `d`: one input from the source, two outputs (the `d.` refs).
    // Without `register_demux`, two outbound links would be a FanOutWithoutTee.
    let graph = parse_launch(
        &reg,
        "videotestsrc num-buffers=4 ! paritydemux name=d   d. ! anysink   d. ! anysink",
    )
    .expect("demux fan-out pipeline parses");

    let stats = run_graph(graph, &NullClock, 4).await.expect("demux pipeline runs");
    assert_eq!(stats.frames_consumed, 4, "all 4 frames routed across the two branches");
}

#[test]
fn unregistered_fan_out_still_needs_a_tee() {
    // A non-tee, non-demux node with two outputs is still rejected, so the demux
    // exemption is scoped to registered demuxers only.
    let mut reg = Registry::new();
    reg.register_source(g2g_core::runtime::SourceFactory::new("videotestsrc", rgba(), || {
        Box::new(VideoTestSrc::new(8, 8, 30, 4))
    }));
    reg.register_launch(LaunchFactory::new("anysink", std::vec::Vec::new(), || Box::new(AnySink)));
    // `identity` is a 1-in/1-out transform, not a registered demux.
    reg.register_launch(LaunchFactory::new("identity", std::vec::Vec::new(), || {
        Box::new(g2g_plugins::identity::IdentityTransform::new())
    }));
    let err = parse_launch(
        &reg,
        "videotestsrc num-buffers=2 ! identity name=d   d. ! anysink   d. ! anysink",
    );
    assert!(err.is_err(), "a non-demux fan-out without a tee must be rejected");
}
