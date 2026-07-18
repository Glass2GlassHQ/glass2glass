//! M694 branched / threaded telemetry: a muxer fan-in run through the observed
//! entry points must report per-element telemetry for the structural fan node
//! too, not just the transforms / sinks. Before M694 the muxer / demux arms held
//! no probe, so `per_element` and the observer snapshot went blind on the fan
//! node; the threaded runner had no observed entry point at all.
#![cfg(all(feature = "std", feature = "multi-thread"))]

use core::future::Future;
use core::pin::Pin;

use g2g_core::runtime::{
    parse_launch, run_graph_observed, run_graph_threaded_observed, LaunchFactory, NodeRole,
    Observer, Registry, ThreadSpawner,
};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, G2gError, OutputSink, PipelineClock,
    PipelinePacket,
};
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// A sink that accepts any packet without a monotonic-sequence assertion: a
/// muxer interleaves frames from sources whose sequence numbers overlap.
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
    reg.register_launch(LaunchFactory::new("anysink", Vec::new(), || {
        Box::new(AnySink)
    }));
    reg
}

const FANIN: &str =
    "videotestsrc num-buffers=4 ! m.   videotestsrc num-buffers=3 ! m.   funnel name=m ! anysink";

/// The cooperative observed runner reports the muxer (fan-in) node in
/// `per_element` with a timed frame count, and the observer snapshot carries it
/// with a `Muxer` role and a live probe.
#[tokio::test]
async fn observed_muxer_fanin_reports_the_fan_node() {
    let reg = registry_with_anysink();
    let graph = parse_launch(&reg, FANIN).expect("fan-in pipeline parses");

    let obs = Observer::new();
    let stats = run_graph_observed(graph, &ZeroClock, 4, &obs, None)
        .await
        .expect("observed run");
    assert_eq!(stats.frames_consumed, 7, "4 + 3 frames reached the sink");

    // The muxer node is now probed: it appears in the end-of-run per-element
    // report with a nonzero (timed) process() count.
    let mux = stats
        .per_element
        .iter()
        .find(|e| e.name == "mux0")
        .expect("muxer node present in per_element");
    assert!(
        mux.proc.count > 0,
        "muxer process() was timed, count={}",
        mux.proc.count
    );
    assert!(!stats.per_element.is_empty());

    // The live observer snapshot carries the same fan node with a Muxer role.
    let snap = obs.snapshot();
    let mux_node = snap
        .nodes
        .iter()
        .find(|n| n.role == NodeRole::Muxer)
        .expect("muxer node in snapshot");
    let lat = mux_node.latency.as_ref().expect("muxer probed");
    assert_eq!(lat.name, "mux0");
    assert!(lat.proc.count > 0, "snapshot sees the muxer's timed frames");
}

/// The thread-per-arm observed runner (new in M694) reports the same fan-node
/// telemetry, proving the observer is wired through the threaded driver.
#[tokio::test]
async fn threaded_observed_muxer_fanin_reports_the_fan_node() {
    let reg = registry_with_anysink();
    let graph = parse_launch(&reg, FANIN).expect("fan-in pipeline parses");

    let obs = Observer::new();
    let stats = run_graph_threaded_observed(graph, &ZeroClock, 4, &obs, &ThreadSpawner)
        .await
        .expect("threaded observed run");
    assert_eq!(stats.frames_consumed, 7, "4 + 3 frames reached the sink");

    let mux = stats
        .per_element
        .iter()
        .find(|e| e.name == "mux0")
        .expect("muxer node present in per_element under the threaded runner");
    assert!(
        mux.proc.count > 0,
        "threaded muxer process() timed, count={}",
        mux.proc.count
    );

    let snap = obs.snapshot();
    assert!(
        snap.nodes
            .iter()
            .any(|n| n.role == NodeRole::Muxer && n.latency.is_some()),
        "threaded observer snapshot carries the probed muxer node"
    );
}
