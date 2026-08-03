//! M846 live per-edge telemetry: the `Observer` tap carries each link's packet /
//! byte / drop counters while the run is still going (drops previously only
//! surfaced in the end-of-run `RunStats`), and the hand-built fan-in runner has
//! an observed entry point that fills `per_element` with instance-named rows.
//!
//! std-gated (graph runner + plugin elements): `cargo test -p g2g-plugins`.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::sync::Arc;

use tokio::sync::{oneshot, Notify};

use g2g_core::element::DynAsyncElement;
use g2g_core::runtime::{
    run_fanin_sink_observed, run_graph_observed, run_source_fanout_observed, DynSourceLoop,
    GraphNode, NodeRole, Observer,
};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, G2gError, Graph, LinkPolicy, Merger, OutputSink,
    PipelineClock, PipelinePacket, Router,
};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::videotestsrc::VideoTestSrc;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Sink that parks on its first frame until released, so the run provably cannot
/// end while the test reads the live counters.
struct GateSink {
    started: Option<oneshot::Sender<()>>,
    release: Arc<Notify>,
    parked: bool,
    received: u64,
}

impl AsyncElement for GateSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        let park = match packet {
            PipelinePacket::DataFrame(_) => {
                self.received += 1;
                if self.parked {
                    false
                } else {
                    self.parked = true;
                    if let Some(tx) = self.started.take() {
                        let _ = tx.send(());
                    }
                    true
                }
            }
            _ => false,
        };
        let release = self.release.clone();
        Box::pin(async move {
            if park {
                release.notified().await;
            }
            Ok(())
        })
    }
}

#[tokio::test]
async fn live_edge_counters_and_drops_are_visible_mid_run() {
    let release = Arc::new(Notify::new());
    let (started_tx, started_rx) = oneshot::channel();

    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(VideoTestSrc::new(8, 8, 30, 64)));
    let sink = g.add_sink(GraphNode::element(GateSink {
        started: Some(started_tx),
        release: release.clone(),
        parked: false,
        received: 0,
    }));
    // Leaky + shallow, so the parked sink makes the source's pushes overflow.
    g.link_with(src, sink, LinkPolicy::DropNewest).unwrap();

    let obs = Observer::new();
    let watcher = {
        let obs = obs.clone();
        let release = release.clone();
        async move {
            if started_rx.await.is_err() {
                return None;
            }
            // The sink is parked, so anything read here is mid-run.
            let mut live = None;
            for _ in 0..100_000 {
                let snap = obs.snapshot();
                let e = &snap.edges[0];
                if e.counts.packets > 0 && e.counts.drops > 0 {
                    live = Some(e.counts);
                    break;
                }
                tokio::task::yield_now().await;
            }
            release.notify_one();
            live
        }
    };

    let (stats, live) = tokio::join!(run_graph_observed(g, &ZeroClock, 2, &obs, None), watcher);
    let stats = stats.expect("leaky run completes");
    let live = live.expect("packets and drops advanced while the run was in flight");

    assert!(live.bytes > 0, "payload bytes counted, got {}", live.bytes);
    assert_eq!(stats.frames_emitted, 64);

    // The end-of-run read is the same counter, so it can only have grown, and the
    // single edge's drops account for the whole run's drop total.
    let final_counts = obs.snapshot().edges[0].counts;
    assert!(final_counts.packets >= live.packets);
    assert!(final_counts.drops >= live.drops);
    assert_eq!(
        final_counts.drops, stats.frames_dropped,
        "per-edge drops match the run total on a single-link graph"
    );
    assert_eq!(
        stats.frames_consumed + stats.frames_dropped,
        stats.frames_emitted
    );
    // Control packets cross the link too, so the packet count is the delivered
    // frames plus this graph's CapsChanged and Eos.
    assert_eq!(final_counts.packets, stats.frames_consumed + 2);
}

#[tokio::test]
async fn observed_fanin_run_reports_named_per_element_rows() {
    let mut a = VideoTestSrc::new(8, 8, 30, 4);
    let mut b = VideoTestSrc::new(8, 8, 30, 3);
    let mut merger = Merger::new(2); // selects input 0
    let mut snk = FakeSink::new();
    let obs = Observer::new();

    let stats = {
        let sources: Vec<&mut dyn DynSourceLoop> = vec![&mut a, &mut b];
        run_fanin_sink_observed(sources, &mut merger, &mut snk, &ZeroClock, 4, &obs)
            .await
            .expect("observed fan-in run")
    };

    assert_eq!(stats.frames_emitted, 7);
    assert_eq!(stats.frames_consumed, 4, "only the selected input forwards");

    // The sink is the one node with a `process()`, so it is the reported row, and
    // its name is the instance name the runner assigned.
    assert_eq!(stats.per_element.len(), 1, "one probed element (the sink)");
    let row = &stats.per_element[0];
    assert_eq!(row.name, "FakeSink0");
    assert_eq!(row.proc.count, 4, "every merged frame was timed");

    // The tap describes the same run: two sources, the merger, the sink.
    let snap = obs.snapshot();
    assert_eq!(snap.nodes.len(), 4);
    assert_eq!(snap.nodes[0].role, NodeRole::Source);
    assert_eq!(snap.nodes[0].name, "VideoTestSrc0");
    assert_eq!(snap.nodes[1].name, "VideoTestSrc1");
    assert_eq!(snap.nodes[2].role, NodeRole::Muxer, "the merger");
    assert_eq!(snap.nodes[3].role, NodeRole::Sink);
    assert_eq!(
        snap.nodes[3].latency.as_ref().map(|l| l.name.as_str()),
        Some("FakeSink0")
    );

    assert_eq!(
        snap.edges
            .iter()
            .map(|e| (e.from, e.to))
            .collect::<Vec<_>>(),
        vec![(0, 2), (1, 2), (2, 3)],
    );
    assert!(
        snap.edges.iter().all(|e| e.caps.is_some()),
        "edges carry their negotiated caps"
    );
    // Both inputs produced; only the selected one reached the sink.
    assert_eq!(snap.edges[0].counts.packets, 5, "4 frames + Eos");
    assert_eq!(snap.edges[1].counts.packets, 4, "3 frames + Eos");
    assert_eq!(snap.edges[2].counts.packets, 5, "merged frames + Eos");
    assert!(snap.edges[2].counts.bytes > 0);
    assert_eq!(snap.edges[2].counts.drops, 0, "no leaky link here");
}

#[tokio::test]
async fn observed_fanout_run_reports_named_per_element_rows() {
    let mut src = VideoTestSrc::new(8, 8, 30, 5);
    let mut router = Router::new(2); // routes to branch 0
    let mut sink_a = FakeSink::new();
    let mut sink_b = FakeSink::new();
    let obs = Observer::new();

    let stats = {
        let sinks: Vec<&mut dyn DynAsyncElement> = vec![&mut sink_a, &mut sink_b];
        run_source_fanout_observed(&mut src, &mut router, sinks, &ZeroClock, 4, &obs)
            .await
            .expect("observed fan-out run")
    };

    assert_eq!(stats.frames_emitted, 5);
    // The fan-out element and both branch sinks are probed.
    let names: Vec<&str> = stats.per_element.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, vec!["Router0", "FakeSink0", "FakeSink1"]);
    assert_eq!(
        stats.per_element[0].proc.count, 5,
        "router timed each frame"
    );

    let snap = obs.snapshot();
    assert_eq!(snap.nodes.len(), 4, "source, router, two sinks");
    assert_eq!(snap.nodes[1].role, NodeRole::Tee);
    assert_eq!(
        snap.edges
            .iter()
            .map(|e| (e.from, e.to))
            .collect::<Vec<_>>(),
        vec![(0, 1), (1, 2), (1, 3)],
    );
    assert_eq!(snap.edges[0].counts.packets, 6, "5 frames + Eos");
    assert!(
        snap.edges[1].counts.bytes > 0,
        "the selected branch carried bytes"
    );
}
