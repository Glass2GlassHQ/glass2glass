//! M122 `gst-launch` muxer fan-in: a text pipeline whose element has several
//! inbound links builds a muxer node (the `funnel` registered in
//! `default_registry`), with the input count derived from link degree, and runs
//! end to end through `run_graph`.

use core::future::Future;
use core::pin::Pin;

use g2g_core::runtime::{parse_launch, run_graph, LaunchFactory, ParseError, Registry};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, G2gError, NodeKind, OutputSink,
    PipelineClock, PipelinePacket,
};
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// A sink that accepts any packet without the monotonic-sequence assertion
/// `FakeSink` makes: a muxer interleaves frames from sources whose sequence
/// numbers overlap, so the sink must tolerate a non-increasing sequence.
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

fn registry_with_anysink() -> Registry {
    let mut reg = default_registry();
    reg.register_launch(LaunchFactory::new("anysink", Vec::new(), || Box::new(AnySink)));
    reg
}

#[tokio::test]
async fn funnel_fans_in_two_sources() {
    let reg = registry_with_anysink();
    // Two sources of unequal length join at the funnel; every frame reaches the
    // sink. The funnel's input count comes from the two `m.` references. Feeding
    // chains come first (each ending `! m.`); the muxer chain is last.
    let graph = parse_launch(
        &reg,
        "videotestsrc num-buffers=4 ! m.   videotestsrc num-buffers=3 ! m.   funnel name=m ! anysink",
    )
    .expect("fan-in pipeline parses");

    let stats = run_graph(graph, &ZeroClock, 4).await.expect("fan-in pipeline runs");
    assert_eq!(stats.frames_consumed, 7, "4 + 3 source frames reached the sink");
}

#[tokio::test]
async fn funnel_builds_a_two_input_muxer_node() {
    let reg = registry_with_anysink();
    let graph = parse_launch(
        &reg,
        "videotestsrc num-buffers=1 ! m.   videotestsrc num-buffers=1 ! m.   funnel name=m ! anysink",
    )
    .expect("parses");
    let vg = graph.finish().expect("valid graph");
    let muxers: Vec<NodeKind> = vg
        .topo()
        .iter()
        .map(|&n| vg.kind(n))
        .filter(|k| matches!(k, NodeKind::Muxer(_)))
        .collect();
    assert_eq!(muxers, [NodeKind::Muxer(2)], "one muxer node with two input pads");
}

#[test]
fn fan_in_into_a_non_muxer_is_reported() {
    let reg = registry_with_anysink();
    // `videoflip` is a single-input transform; two links into it is fan-in it
    // cannot express.
    let err = parse_launch(
        &reg,
        "videotestsrc num-buffers=1 ! v.   videotestsrc num-buffers=1 ! v.   videoflip name=v ! anysink",
    )
    .unwrap_err();
    assert_eq!(err, ParseError::NotAMuxer("videoflip".into()));
}

#[test]
fn muxer_without_output_is_reported() {
    let reg = registry_with_anysink();
    // The funnel collects two inputs but nothing consumes its single output.
    let err = parse_launch(
        &reg,
        "videotestsrc num-buffers=1 ! m.   videotestsrc num-buffers=1 ! m.   funnel name=m",
    )
    .unwrap_err();
    assert_eq!(err, ParseError::MuxerWithoutOutput("funnel".into()));
}
