//! M80 - the SEGMENT carrier: `PipelinePacket::Segment` propagates through a
//! transform to a sink and broadcasts across a tee, and a sink records it so
//! frame timestamps map to running time. The runner does not yet *emit* an
//! opening SEGMENT on its own (that, plus the post-flush re-emit, is the seek
//! milestone, which also needs the error-priority robustness a blocking
//! initial push would require); here a small injector element produces one so
//! the carrier is exercised end-to-end.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use g2g_core::element::{AsyncElement, BoxFuture, ConfigureOutcome, OutputSink};
use g2g_core::graph::Graph;
use g2g_core::runtime::{run_graph, run_source_transform_sink, GraphNode};
use g2g_core::{Caps, G2gError, PipelineClock, PipelinePacket, Segment};

use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::videotestsrc::VideoTestSrc;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Pass-through transform that injects one `Segment` ahead of the first
/// `DataFrame`, then forwards everything unchanged. Stands in for the
/// runner/source emission the seek milestone will add.
struct SegmentInjector {
    seg: Segment,
    emitted: bool,
}

impl SegmentInjector {
    fn new(seg: Segment) -> Self {
        Self {
            seg,
            emitted: false,
        }
    }
}

impl AsyncElement for SegmentInjector {
    type ProcessFuture<'a> = BoxFuture<'a, Result<(), G2gError>>;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(_) = &packet {
                if !self.emitted {
                    self.emitted = true;
                    out.push(PipelinePacket::Segment(self.seg)).await?;
                }
            }
            // Forward everything (including any upstream Segment) unchanged.
            out.push(packet).await?;
            Ok(())
        })
    }
}

/// A `Segment` injected mid-chain survives the runner's forwarding and the
/// sink records it; running time is then computable from the sink's segment.
#[tokio::test]
async fn segment_flows_through_transform_to_sink() {
    let target = 4u64;
    let seg = Segment {
        start: 1_000,
        base: 100,
        ..Segment::new()
    };
    let mut src = VideoTestSrc::new(32, 32, 30, target);
    let mut inj = SegmentInjector::new(seg);
    let mut sink = FakeSink::new();

    let stats = run_source_transform_sink(&mut src, &mut inj, &mut sink, &ZeroClock, 4)
        .await
        .expect("pipeline runs");

    assert_eq!(stats.frames_consumed, target);
    assert_eq!(sink.segments(), 1, "the injected SEGMENT reached the sink");
    assert_eq!(sink.last_segment(), Some(seg));
    // The sink's segment maps a timestamp to running time: base + (ts-start).
    assert_eq!(
        sink.last_segment().unwrap().to_running_time(3_000),
        Some(2_100)
    );
}

/// Sink that counts the `Segment`s it receives, readable after `run_graph`
/// consumes the elements (via a shared counter).
struct SegCountingSink {
    segments: Arc<AtomicU64>,
}

impl AsyncElement for SegCountingSink {
    type ProcessFuture<'a> = BoxFuture<'a, Result<(), G2gError>>;

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
        Box::pin(async move {
            if let PipelinePacket::Segment(_) = packet {
                self.segments.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        })
    }
}

/// A tee broadcasts the injected SEGMENT to every branch: both sinks see it.
#[tokio::test]
async fn segment_broadcasts_through_tee_to_all_sinks() {
    let target = 3u64;
    let c0 = Arc::new(AtomicU64::new(0));
    let c1 = Arc::new(AtomicU64::new(0));

    let mut g: Graph<GraphNode> = Graph::new();
    let s = g.add_source(GraphNode::source(VideoTestSrc::new(16, 16, 30, target)));
    let inj = g.add_transform(GraphNode::element(SegmentInjector::new(Segment::new())));
    let tee = g.add_tee(2);
    let s0 = g.add_sink(GraphNode::element(SegCountingSink {
        segments: c0.clone(),
    }));
    let s1 = g.add_sink(GraphNode::element(SegCountingSink {
        segments: c1.clone(),
    }));
    g.link(s, inj).unwrap();
    g.link(inj, tee.input()).unwrap();
    g.link(tee.out(0), s0).unwrap();
    g.link(tee.out(1), s1).unwrap();

    run_graph(g, &ZeroClock, 4).await.expect("tee DAG runs");

    assert_eq!(
        c0.load(Ordering::SeqCst),
        1,
        "branch 0 received the SEGMENT"
    );
    assert_eq!(
        c1.load(Ordering::SeqCst),
        1,
        "branch 1 received the SEGMENT"
    );
}
