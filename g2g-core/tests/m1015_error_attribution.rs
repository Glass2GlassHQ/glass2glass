//! M1015: a failed run says which element failed. `G2gError` carries no element
//! identity, so the runners log the name of the arm whose error they report, on
//! the `runtime` category at error level.
//!
//! The sink fails while the source is still pushing, so the source arm ends as
//! `Shutdown` (its link closed): the attributed name must be the sink's, not the
//! consequence upstream.
#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use std::sync::{Mutex, MutexGuard};

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::log::{self, LogLevel, RingSink};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{block_on, run_graph, run_source_transform_sink, GraphNodeRef, SourceLoop};
use g2g_core::{
    graph::Graph, AsyncElement, Caps, ConfigureOutcome, Dim, G2gError, HardwareError, MemoryDomain,
    OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

/// The frame the failing element rejects. Late enough that the run is under way.
const FAILING_FRAME: u64 = 2;
const FRAMES: u64 = 8;

/// The log sink is process-global, so a test that installs one must not run
/// alongside another that does.
static SINK_IN_USE: Mutex<()> = Mutex::new(());

fn failure() -> G2gError {
    G2gError::Hardware(HardwareError::Io(13))
}

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
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn frame(seq: u64) -> Frame {
    Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 2 * 2 * 4]))),
        FrameTiming::default(),
        seq,
    )
}

struct CountingSource;

impl SourceLoop for CountingSource {
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
            for seq in 0..FRAMES {
                out.push(PipelinePacket::DataFrame(frame(seq))).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(FRAMES)
        })
    }
}

struct ForwardingTransform;

impl AsyncElement for ForwardingTransform {
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
            out.push(packet).await?;
            Ok(())
        })
    }
}

/// Fails on `FAILING_FRAME`, mid-run.
struct FailingSink;

impl AsyncElement for FailingSink {
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
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(f) if f.sequence >= FAILING_FRAME => Err(failure()),
                _ => Ok(()),
            }
        })
    }
}

/// Install a fresh recorder for the duration of one test, holding the global
/// sink's lock so a sibling test cannot capture into it at the same time.
fn record_logs() -> (RingSink, MutexGuard<'static, ()>) {
    let guard = SINK_IN_USE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let ring = RingSink::new(256);
    log::set_sink(Box::new(ring.clone()));
    log::set_default_level(LogLevel::Error);
    (ring, guard)
}

/// The one attributed failure line the run logged, panicking if there is none.
fn attributed_failure(ring: &RingSink) -> String {
    let lines: Vec<String> = ring
        .drain()
        .into_iter()
        .filter(|record| record.message.starts_with("pipeline error in "))
        .map(|record| {
            assert_eq!(record.category, log::RUNTIME_CATEGORY, "{record:?}");
            record.message
        })
        .collect();
    assert_eq!(lines.len(), 1, "exactly one attributed failure: {lines:?}");
    lines.into_iter().next().expect("checked above")
}

#[test]
fn the_graph_runner_names_the_element_that_failed() {
    let (ring, _guard) = record_logs();
    let mut g: Graph<GraphNodeRef<'static>> = Graph::new();
    let src = g.add_source(GraphNodeRef::source(CountingSource));
    let transform = g.add_transform(GraphNodeRef::element(ForwardingTransform));
    let sink = g.add_sink(GraphNodeRef::element(FailingSink));
    g.set_node_name(sink, String::from("boom-sink"));
    g.link(src, transform).unwrap();
    g.link(transform, sink).unwrap();

    let err = block_on(run_graph(g, &ZeroClock, 1)).expect_err("the sink fails the run");
    assert_eq!(err, failure());
    let line = attributed_failure(&ring);
    assert!(
        line.contains("boom-sink"),
        "the failing sink's name, not an upstream consequence: {line}"
    );
    assert!(line.contains("Io(13)"), "the error itself: {line}");
}

#[test]
fn the_linear_runner_names_the_element_that_failed() {
    let (ring, _guard) = record_logs();
    let mut src = CountingSource;
    let mut transform = ForwardingTransform;
    let mut sink = FailingSink;
    let err = block_on(run_source_transform_sink(
        &mut src,
        &mut transform,
        &mut sink,
        &ZeroClock,
        1,
    ))
    .expect_err("the sink fails the run");
    assert_eq!(err, failure());
    let line = attributed_failure(&ring);
    assert!(
        line.contains("FailingSink"),
        "the failing sink's instance name: {line}"
    );
}
