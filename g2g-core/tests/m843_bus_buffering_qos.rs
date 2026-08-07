//! M843: bus reports that were previously missing.
//!
//! - `Buffering` on interior links: every arm with an input link (transform, not
//!   just sink) samples its fill and names the element it feeds, so an
//!   application can tell which link is starving.
//! - Periodic `Qos`: a sink given a report interval posts the running stats on
//!   pipeline-clock time, not only when it drops a frame.
//! - The late-drop `Qos`: the drop decision and its report live in the shared
//!   [`QosTracker`], so the display sinks report without a display attached.
#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;

use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{block_on, run_graph_with_bus, GraphNode, SourceLoop};
use g2g_core::{
    AsyncElement, Bus, BusMessage, Caps, ConfigureOutcome, Dim, Frame, FrameTiming, G2gError,
    Graph, OutputSink, PipelineClock, PipelinePacket, QosTracker, Rate, RawVideoFormat,
};

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(2),
        height: Dim::Fixed(2),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
    }
}

fn frame(sequence: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 16]))),
        FrameTiming {
            pts_ns: sequence * 33_000_000,
            ..FrameTiming::default()
        },
        sequence,
    ))
}

/// Pushes `count` frames then `Eos`.
struct CountSrc {
    count: u64,
}

impl SourceLoop for CountSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(caps()))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for i in 0..self.count {
                out.push(frame(i)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.count)
        })
    }
}

/// Forwards everything: an interior element in the transform position, a plain
/// consumer in the sink position.
struct Pass;

impl AsyncElement for Pass {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !matches!(packet, PipelinePacket::Eos) {
                out.push(packet).await?;
            }
            Ok(())
        })
    }
}

#[test]
fn interior_and_sink_links_both_report_buffering() {
    let (bus, handle) = Bus::new(64);
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(CountSrc { count: 8 }));
    let mid = g.add_transform(GraphNode::element(Pass));
    let sink = g.add_sink(GraphNode::element(Pass));
    g.set_node_name(mid, "mid".into());
    g.set_node_name(sink, "out".into());
    g.link(src, mid).unwrap();
    g.link(mid, sink).unwrap();
    let stats = block_on(run_graph_with_bus(g, &ZeroClock, 2, &handle)).expect("graph runs");
    assert_eq!(stats.frames_consumed, 8);

    let mut elements = Vec::new();
    while let Some(m) = bus.try_recv() {
        if let BusMessage::Buffering { percent, element } = m {
            assert!(percent <= 100, "fill percent in range");
            elements.push(element);
        }
    }
    assert!(
        elements.iter().any(|e| e.as_deref() == Some("mid")),
        "the interior link feeding `mid` reports its level, got {elements:?}"
    );
    assert!(
        elements.iter().any(|e| e.as_deref() == Some("out")),
        "the sink's link still reports its level, got {elements:?}"
    );
}

#[test]
fn periodic_qos_posts_running_stats_between_drops() {
    let (bus, handle) = Bus::new(16);
    let mut qos = QosTracker::new();
    qos.set_bus(handle);
    qos.set_report_interval_ns(20_000_000);

    // Frames on time, one every 10 ms of clock: the report lands on the frame
    // that crosses the 20 ms interval, and nothing was dropped.
    let mut posts = 0;
    for i in 0..5u64 {
        let t = i * 10_000_000;
        assert!(qos.judge_frame(t, t, i).is_none(), "frame {i} is on time");
    }
    while let Some(m) = bus.try_recv() {
        match m {
            BusMessage::Qos {
                processed, dropped, ..
            } => {
                assert_eq!(dropped, 0, "the periodic report is not a drop report");
                assert!(processed > 0);
                posts += 1;
            }
            other => panic!("unexpected message {other:?}"),
        }
    }
    assert_eq!(posts, 2, "one report per elapsed interval");
}

#[test]
fn late_drop_reports_and_hands_back_the_upstream_message() {
    let (bus, handle) = Bus::new(16);
    let mut qos = QosTracker::new();
    qos.set_bus(handle);
    qos.set_max_lateness_ns(0);

    let upstream = qos
        .judge_frame(1_000_000, 1_500_000, 7)
        .expect("half a ms past a zero bound is a drop");
    assert_eq!(upstream.jitter_ns, 500_000);
    assert_eq!(upstream.running_time_ns, 1_000_000);
    assert_eq!(
        bus.try_recv(),
        Some(BusMessage::Qos {
            running_time_ns: 1_000_000,
            jitter_ns: 500_000,
            processed: 7,
            dropped: 1,
        }),
        "the drop is reported on the bus"
    );
}
