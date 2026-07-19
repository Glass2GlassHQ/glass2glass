//! M399: measured per-element latency + input-link fill.
//!
//! `RunStats::report()` (M287) folds each element's *declared* latency. M399
//! adds the *measured* counterpart: the runner times every `DataFrame`
//! `process()` call and samples each element's input-link fill, surfacing a
//! per-element p50/p99 + fill table in `RunStats::per_element`. These tests
//! drive the real linear and graph runners with a deliberately-slow transform
//! and assert the instrumentation attributes the cost to the right element.
//!
//! The unit under test is the runner's instrumentation, not the fixtures: the
//! slow transform is a stand-in for any real element with a measurable
//! `process()` cost (a decoder, a GPU upload), exercised end to end.

use core::future::Future;
use core::pin::Pin;
use core::time::Duration;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{
    run_graph, run_simple_pipeline, run_source_transform_sink, GraphNode, SourceLoop,
};
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
        width: Dim::Fixed(64),
        height: Dim::Fixed(64),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn make_frame(seq: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
        timing: FrameTiming::default(),
        sequence: seq,
        meta: Default::default(),
    }
}

/// Emits `frames` DataFrames then Eos. No artificial delay.
struct FrameSrc {
    frames: u64,
}

impl SourceLoop for FrameSrc {
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

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for seq in 0..self.frames {
                out.push(PipelinePacket::DataFrame(make_frame(seq))).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.frames)
        })
    }
}

/// Identity transform that sleeps `delay_ms` per data frame, so its measured
/// `process()` p50 lands well above any zero-cost element's.
struct SlowTransform {
    delay_ms: u64,
}

impl AsyncElement for SlowTransform {
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
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        let delay_ms = self.delay_ms;
        Box::pin(async move {
            match packet {
                PipelinePacket::Eos => Ok(()),
                PipelinePacket::DataFrame(_) => {
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    out.push(packet).await?;
                    Ok(())
                }
                other => {
                    out.push(other).await?;
                    Ok(())
                }
            }
        })
    }
}

/// Terminal sink, no artificial delay (the fast baseline).
struct CountingSink;

impl AsyncElement for CountingSink {
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
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

#[tokio::test]
async fn linear_runner_attributes_latency_to_the_slow_transform() {
    let mut src = FrameSrc { frames: 6 };
    let mut tx = SlowTransform { delay_ms: 3 };
    let mut sink = CountingSink;
    let clock = ZeroClock;

    let stats = run_source_transform_sink(&mut src, &mut tx, &mut sink, &clock, 4)
        .await
        .expect("pipeline runs");

    assert_eq!(stats.frames_consumed, 6);
    // Two interior elements, in topological order; the source has no `process()`.
    assert_eq!(stats.per_element.len(), 2, "transform + sink rows");
    let tx_row = &stats.per_element[0];
    let sink_row = &stats.per_element[1];
    assert_eq!(tx_row.name, "SlowTransform");
    assert_eq!(sink_row.name, "CountingSink");

    // Every data frame was timed at both elements.
    assert_eq!(tx_row.proc.count, 6, "transform timed each frame");
    assert_eq!(sink_row.proc.count, 6, "sink timed each frame");

    // The 3 ms sleep dominates: the transform's measured p50 is at/above ~1 ms
    // (log2-bucket resolution) and strictly above the zero-cost sink's.
    assert!(
        tx_row.proc.p50_ns >= MS,
        "slow transform p50 = {} ns",
        tx_row.proc.p50_ns
    );
    assert!(
        tx_row.proc.p50_ns > sink_row.proc.p50_ns,
        "transform ({}) must out-measure the fast sink ({})",
        tx_row.proc.p50_ns,
        sink_row.proc.p50_ns
    );

    // The report renders the measured table with the dominating element named.
    let report = stats.report();
    assert!(
        report.contains("per-element [measured]"),
        "report:\n{report}"
    );
    assert!(report.contains("SlowTransform"), "report:\n{report}");
    assert!(report.contains("proc p50"), "report:\n{report}");
}

#[tokio::test]
async fn simple_pipeline_records_the_sink_probe() {
    let mut src = FrameSrc { frames: 4 };
    let mut sink = CountingSink;
    let clock = ZeroClock;

    let stats = run_simple_pipeline(&mut src, &mut sink, &clock, 4)
        .await
        .expect("pipeline runs");

    // The source has no `process()`; only the sink is instrumented.
    assert_eq!(stats.per_element.len(), 1);
    assert_eq!(stats.per_element[0].name, "CountingSink");
    assert_eq!(
        stats.per_element[0].proc.count, 4,
        "each frame timed at the sink"
    );
}

#[tokio::test]
async fn graph_runner_names_elements_and_measures_each() {
    // source -> slow transform -> sink, driven by the general graph runner.
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(FrameSrc { frames: 5 }));
    let tx = g.add_transform(GraphNode::element(SlowTransform { delay_ms: 3 }));
    let sink = g.add_sink(GraphNode::element(CountingSink));
    g.link(src, tx).unwrap();
    g.link(tx, sink).unwrap();

    let stats = run_graph(g, &ZeroClock, 4).await.expect("graph runs");

    assert_eq!(stats.frames_consumed, 5);
    assert_eq!(
        stats.per_element.len(),
        2,
        "transform + sink (source has no process)"
    );

    // The graph runner names instances `<category>N`; the slow transform must be
    // present with timed frames and dominate the sink.
    let tx_row = stats
        .per_element
        .iter()
        .find(|e| e.name == "SlowTransform0")
        .expect("transform row, named <category>0");
    let sink_row = stats
        .per_element
        .iter()
        .find(|e| e.name == "CountingSink0")
        .expect("sink row, named <category>0");
    assert_eq!(tx_row.proc.count, 5);
    assert!(
        tx_row.proc.p50_ns >= MS,
        "slow transform p50 = {} ns",
        tx_row.proc.p50_ns
    );
    assert!(tx_row.proc.p50_ns > sink_row.proc.p50_ns);
}
