//! M713 terminal fan-in graph node: `run_graph` over a `FaninSink` (an N-input
//! `MultiInputElement` with no output, the `run_fanin_session` shape), so a
//! transform chain can feed a session sink's pads (`src -> tee -> transform per
//! branch -> session`). Pure-fake elements (no network): per-input Eos flush,
//! end-on-all-Eos, and per-input reverse-signal routing back to the arm feeding
//! that pad.

use core::future::Future;
use core::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use g2g_core::fanout::{MultiInputElement, ReverseChannel};
#[cfg(all(feature = "std", feature = "multi-thread"))]
use g2g_core::runtime::run_graph_threaded;
use g2g_core::runtime::{run_graph, GraphNode};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, G2gError, Graph, OutputSink,
    PipelineClock, PipelinePacket, PushOutcome, Reconfigure,
};
use g2g_plugins::videotestsrc::VideoTestSrc;
#[cfg(all(feature = "std", feature = "multi-thread"))]
use g2g_plugins::TokioThreadSpawner;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Shared per-pad counters a test asserts on after the run.
#[derive(Debug, Default)]
struct PadTallies {
    frames: Vec<AtomicU64>,
    eos: Vec<AtomicU64>,
}

impl PadTallies {
    fn new(pads: usize) -> Arc<Self> {
        Arc::new(Self {
            frames: (0..pads).map(|_| AtomicU64::new(0)).collect(),
            eos: (0..pads).map(|_| AtomicU64::new(0)).collect(),
        })
    }
}

/// Terminal fan-in fake: consumes every input, produces nothing. Optionally
/// requests an upstream keyframe on one pad's reverse channel after that pad's
/// `kf_after`th frame (the WebRTC per-(mid,rid) PLI shape).
#[derive(Debug)]
struct FakeSession {
    pads: usize,
    tallies: Arc<PadTallies>,
    reverse: Vec<ReverseChannel>,
    kf_pad: Option<usize>,
    kf_after: u64,
}

impl FakeSession {
    fn new(pads: usize, tallies: Arc<PadTallies>) -> Self {
        Self {
            pads,
            tallies,
            reverse: (0..pads).map(|_| ReverseChannel::new()).collect(),
            kf_pad: None,
            kf_after: 0,
        }
    }

    fn keyframe_on(mut self, pad: usize, after: u64) -> Self {
        self.kf_pad = Some(pad);
        self.kf_after = after;
        self
    }
}

impl MultiInputElement for FakeSession {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn input_count(&self) -> usize {
        self.pads
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
        // Terminal: no merged output exists for a downstream to negotiate.
        Err(G2gError::CapsMismatch)
    }

    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn reverse_channel(&self, input: usize) -> Option<ReverseChannel> {
        Some(self.reverse[input].clone())
    }

    fn is_terminal(&self) -> bool {
        true
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(_) => {
                    let n = self.tallies.frames[input].fetch_add(1, Ordering::SeqCst) + 1;
                    if Some(input) == self.kf_pad && n == self.kf_after {
                        self.reverse[input].request_keyframe();
                    }
                }
                PipelinePacket::Eos => {
                    self.tallies.eos[input].fetch_add(1, Ordering::SeqCst);
                }
                _ => {}
            }
            Ok(())
        })
    }
}

/// Pass-through transform that counts the `Reconfigure::ForceKeyframe` push
/// outcomes its downstream edge hands back, standing in for an encoder that
/// would force an IDR on them.
struct KeyframeObserver {
    seen: Arc<AtomicU64>,
}

impl AsyncElement for KeyframeObserver {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    // Stands in for an encoder: consumes the keyframe request (M720 would
    // otherwise relay it past this element toward the source).
    fn handles_keyframe_requests(&self) -> bool {
        true
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PushOutcome::Reconfigure(Reconfigure::ForceKeyframe) = out.push(packet).await? {
                self.seen.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        })
    }
}

/// Two sources of unequal length end on a terminal fan-in: every frame reaches
/// the session pad it was linked to, each pad gets exactly one flush `Eos`, and
/// the run ends once every input has ended.
#[tokio::test]
async fn fanin_sink_terminates_graph() {
    let tallies = PadTallies::new(2);
    let mut g: Graph<GraphNode> = Graph::new();
    let s0 = g.add_source(GraphNode::source(VideoTestSrc::new(8, 8, 30, 4)));
    let s1 = g.add_source(GraphNode::source(VideoTestSrc::new(8, 8, 30, 3)));
    let session = g.add_fanin_sink(GraphNode::muxer(FakeSession::new(2, tallies.clone())), 2);
    g.link(s0, session.input(0)).unwrap();
    g.link(s1, session.input(1)).unwrap();

    let stats = run_graph(g, &ZeroClock, 4)
        .await
        .expect("terminal fan-in runs");
    assert_eq!(stats.frames_emitted, 7, "4 + 3 source frames");
    assert_eq!(stats.frames_consumed, 7, "the session consumed every frame");
    assert_eq!(tallies.frames[0].load(Ordering::SeqCst), 4);
    assert_eq!(tallies.frames[1].load(Ordering::SeqCst), 3);
    assert_eq!(
        tallies.eos[0].load(Ordering::SeqCst),
        1,
        "pad 0 flushed once"
    );
    assert_eq!(
        tallies.eos[1].load(Ordering::SeqCst),
        1,
        "pad 1 flushed once"
    );
}

/// The simulcast fan shape: one source tees into two per-layer transform
/// branches that feed the terminal session's pads.
#[tokio::test]
async fn tee_chain_feeds_fanin_sink() {
    let tallies = PadTallies::new(2);
    let kf0 = Arc::new(AtomicU64::new(0));
    let kf1 = Arc::new(AtomicU64::new(0));
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(VideoTestSrc::new(8, 8, 30, 4)));
    let tee = g.add_tee(2);
    let t0 = g.add_transform(GraphNode::element(KeyframeObserver { seen: kf0.clone() }));
    let t1 = g.add_transform(GraphNode::element(KeyframeObserver { seen: kf1.clone() }));
    let session = g.add_fanin_sink(GraphNode::muxer(FakeSession::new(2, tallies.clone())), 2);
    g.link(src, tee.input()).unwrap();
    g.link(tee.out(0), t0).unwrap();
    g.link(tee.out(1), t1).unwrap();
    g.link(t0, session.input(0)).unwrap();
    g.link(t1, session.input(1)).unwrap();

    let stats = run_graph(g, &ZeroClock, 4).await.expect("fan graph runs");
    assert_eq!(stats.frames_emitted, 4, "source emitted 4 frames");
    assert_eq!(
        stats.frames_consumed, 8,
        "both branches' 4 frames reached the session"
    );
    assert_eq!(tallies.frames[0].load(Ordering::SeqCst), 4);
    assert_eq!(tallies.frames[1].load(Ordering::SeqCst), 4);
}

/// A per-pad reverse signal reaches only the arm feeding that pad: the session
/// PLIs pad 1 mid-stream and the branch-1 observer sees a `ForceKeyframe`
/// outcome on a later push, while branch 0 sees none.
#[tokio::test]
async fn reverse_signal_reaches_feeding_arm() {
    let tallies = PadTallies::new(2);
    let kf0 = Arc::new(AtomicU64::new(0));
    let kf1 = Arc::new(AtomicU64::new(0));
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(VideoTestSrc::new(8, 8, 30, 8)));
    let tee = g.add_tee(2);
    let t0 = g.add_transform(GraphNode::element(KeyframeObserver { seen: kf0.clone() }));
    let t1 = g.add_transform(GraphNode::element(KeyframeObserver { seen: kf1.clone() }));
    let session = g.add_fanin_sink(
        GraphNode::muxer(FakeSession::new(2, tallies.clone()).keyframe_on(1, 2)),
        2,
    );
    g.link(src, tee.input()).unwrap();
    g.link(tee.out(0), t0).unwrap();
    g.link(tee.out(1), t1).unwrap();
    g.link(t0, session.input(0)).unwrap();
    g.link(t1, session.input(1)).unwrap();

    run_graph(g, &ZeroClock, 4).await.expect("fan graph runs");
    assert_eq!(
        kf1.load(Ordering::SeqCst),
        1,
        "the PLI'd pad's feeding arm saw exactly one ForceKeyframe"
    );
    assert_eq!(
        kf0.load(Ordering::SeqCst),
        0,
        "the sibling pad saw none (per-pad routing, not broadcast)"
    );
}

/// A `gst-launch` line whose fan-in element has nothing downstream builds a
/// terminal `FaninSink` node (not a muxer missing its output) and runs.
#[tokio::test]
async fn launch_line_builds_terminal_fanin() {
    use g2g_core::graph::NodeKind;
    use g2g_core::runtime::{parse_launch, MuxerFactory, Registry, SourceFactory};
    use g2g_core::{Dim, Rate, RawVideoFormat};

    fn rgba8x8() -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(8),
            height: Dim::Fixed(8),
            framerate: Rate::Fixed(30 << 16),
        }
    }

    let mut reg = Registry::new();
    reg.register_source(SourceFactory::new("vsrc4", rgba8x8(), || {
        Box::new(VideoTestSrc::new(8, 8, 30, 4))
    }));
    reg.register_source(SourceFactory::new("vsrc3", rgba8x8(), || {
        Box::new(VideoTestSrc::new(8, 8, 30, 3))
    }));
    reg.register_muxer(MuxerFactory::new("fakesess", |n| {
        Box::new(FakeSession::new(n, PadTallies::new(n)))
    }));

    const LINE: &str = "vsrc4 ! s.   vsrc3 ! s.   fakesess name=s";
    let vg = parse_launch(&reg, LINE)
        .expect("terminal fan-in line parses")
        .finish()
        .expect("valid graph");
    let fanins: Vec<NodeKind> = vg
        .topo()
        .iter()
        .map(|&n| vg.kind(n))
        .filter(|k| matches!(k, NodeKind::FaninSink(_)))
        .collect();
    assert_eq!(
        fanins,
        [NodeKind::FaninSink(2)],
        "one 2-input terminal node"
    );

    let stats = run_graph(parse_launch(&reg, LINE).expect("parses"), &ZeroClock, 4)
        .await
        .expect("terminal fan-in launch line runs");
    assert_eq!(stats.frames_emitted, 7);
    assert_eq!(stats.frames_consumed, 7);

    // A MERGING muxer with nothing downstream stays a parse error (its output
    // would be silently dropped); only a terminal session may end the line.
    reg.register_muxer(MuxerFactory::new("funnel", |n| {
        Box::new(g2g_plugins::mux::InterleaveMux::new(n, rgba8x8()))
    }));
    let err = match parse_launch(&reg, "vsrc4 ! m.   vsrc3 ! m.   funnel name=m") {
        Err(e) => e,
        Ok(_) => panic!("merging muxer without output must be rejected"),
    };
    assert!(
        format!("{err:?}").contains("MuxerWithoutOutput"),
        "got {err:?}"
    );
}

/// The threaded runner drives the same terminal fan-in shape to the same counts.
#[cfg(all(feature = "std", feature = "multi-thread"))]
#[tokio::test]
async fn threaded_runner_matches_cooperative() {
    let build = |tallies: Arc<PadTallies>| {
        let mut g: Graph<GraphNode> = Graph::new();
        let s0 = g.add_source(GraphNode::source(VideoTestSrc::new(8, 8, 30, 4)));
        let s1 = g.add_source(GraphNode::source(VideoTestSrc::new(8, 8, 30, 3)));
        let session = g.add_fanin_sink(GraphNode::muxer(FakeSession::new(2, tallies)), 2);
        g.link(s0, session.input(0)).unwrap();
        g.link(s1, session.input(1)).unwrap();
        g
    };

    let coop_tallies = PadTallies::new(2);
    let coop = run_graph(build(coop_tallies.clone()), &ZeroClock, 4)
        .await
        .expect("cooperative run");

    let thr_tallies = PadTallies::new(2);
    let threaded = run_graph_threaded(
        build(thr_tallies.clone()),
        &ZeroClock,
        4,
        &TokioThreadSpawner,
    )
    .await
    .expect("threaded run");

    assert_eq!(threaded.frames_emitted, coop.frames_emitted);
    assert_eq!(threaded.frames_consumed, coop.frames_consumed);
    for pad in 0..2 {
        assert_eq!(
            thr_tallies.frames[pad].load(Ordering::SeqCst),
            coop_tallies.frames[pad].load(Ordering::SeqCst),
        );
        assert_eq!(thr_tallies.eos[pad].load(Ordering::SeqCst), 1);
    }
}
