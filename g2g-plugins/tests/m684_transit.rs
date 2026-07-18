//! M684: measured per-link transit (queue-residency) time. The observed graph
//! runner stamps each `DataFrame` as it is queued and pops the stamp when the
//! consumer pulls it, so `ElementLatency::transit` reports how long frames waited
//! on each element's input link, the "wait" half of a latency waterfall.
//!
//! This drives the real runner with a fast source and a deliberately slow
//! transform: the source runs ahead and frames pile up on the transform's input,
//! so its measured transit is positive. std-gated (needs the clock + graph
//! runner + observer).
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use core::time::Duration;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_graph, run_graph_observed, GraphNode, Observer, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, G2gError, Graph, MemoryDomain, OutputSink,
    PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

const MS: u64 = 1_000_000;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(8),
        height: Dim::Fixed(8),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Emits `frames` DataFrames as fast as the link accepts them, then Eos.
struct FastSrc {
    frames: u64,
}

impl SourceLoop for FastSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(caps()))
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(PipelinePacket::CapsChanged(caps())).await?;
            for i in 0..self.frames {
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
                    timing: FrameTiming::default(),
                    sequence: i,
                    meta: Default::default(),
                };
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.frames)
        })
    }
}

/// Forwards each frame after sleeping `delay_ms`, so upstream frames queue behind
/// it on its input link.
struct SlowTransform {
    delay_ms: u64,
}

impl AsyncElement for SlowTransform {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, up: &Caps) -> Result<Caps, G2gError> {
        Ok(up.clone())
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        let delay = self.delay_ms;
        Box::pin(async move {
            if let PipelinePacket::DataFrame(_) = &packet {
                tokio::time::sleep(Duration::from_millis(delay)).await;
                out.push(packet).await?;
            } else if !matches!(packet, PipelinePacket::Eos) {
                out.push(packet).await?;
            }
            Ok(())
        })
    }
}

struct CountingSink;
impl AsyncElement for CountingSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;
    fn intercept_caps(&self, up: &Caps) -> Result<Caps, G2gError> {
        Ok(up.clone())
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
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

fn build() -> Graph<GraphNode> {
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(FastSrc { frames: 8 }));
    let tx = g.add_transform(GraphNode::element(SlowTransform { delay_ms: 4 }));
    let sink = g.add_sink(GraphNode::element(CountingSink));
    g.link(src, tx).unwrap();
    g.link(tx, sink).unwrap();
    g
}

#[tokio::test]
async fn observed_run_measures_input_link_transit() {
    let obs = Observer::new();
    // cap=2 so the source can run a couple of frames ahead of the slow transform.
    let stats = run_graph_observed(build(), &ZeroClock, 2, &obs, None).await.expect("runs");
    assert_eq!(stats.frames_consumed, 8);

    let tx = stats
        .per_element
        .iter()
        .find(|e| e.name == "SlowTransform0")
        .expect("transform row");
    // Frames pile up behind the 4 ms processing, so the transform's input-link
    // queue-residency is measured and non-trivial.
    assert!(tx.transit.count > 0, "transit was sampled, n={}", tx.transit.count);
    assert!(
        tx.transit.p50_ns >= MS,
        "queued frames waited ~ms on the input link, p50={} ns",
        tx.transit.p50_ns
    );
}

#[tokio::test]
async fn unobserved_run_has_no_transit_overhead() {
    // Without an observer the edges are plain links: no transit stamps recorded,
    // so `transit.count` stays 0 (the zero-cost path).
    let stats = run_graph(build(), &ZeroClock, 2).await.expect("runs");
    assert_eq!(stats.frames_consumed, 8);
    for e in &stats.per_element {
        assert_eq!(e.transit.count, 0, "{} has no transit when unobserved", e.name);
    }
}
