//! M909: a transform whose catch-all arm forwards `Eos` must not produce two
//! EOS packets. The runner arms forward the sentinel after `process(Eos)`
//! returns, and now skip that push when the element already sent one.
//!
//! The links are sized so the second push would have to wait for the sink,
//! which has already consumed the first `Eos` and dropped its receiver: the
//! duplicate therefore fails as `Shutdown` rather than landing silently, so the
//! assertions below are scheduling-independent, not timing-dependent.
#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use std::sync::{Arc, Mutex};

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{block_on, run_graph, run_source_transform_sink, GraphNodeRef, SourceLoop};
use g2g_core::{
    graph::Graph, AsyncElement, Caps, ConfigureOutcome, Dim, G2gError, MemoryDomain, OutputSink,
    PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

const FRAMES: u64 = 8;

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

/// The shape under audit: no explicit `Eos` arm, so the catch-all forwards the
/// sentinel itself.
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
            match packet {
                PipelinePacket::DataFrame(f) => {
                    out.push(PipelinePacket::DataFrame(f)).await?;
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

#[derive(Default, Clone)]
struct Tally {
    frames: u64,
    eos: u64,
}

struct TallySink {
    tally: Arc<Mutex<Tally>>,
}

impl AsyncElement for TallySink {
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
        let tally = self.tally.clone();
        Box::pin(async move {
            let mut t = tally.lock().unwrap();
            match packet {
                PipelinePacket::DataFrame(_) => t.frames += 1,
                PipelinePacket::Eos => t.eos += 1,
                _ => {}
            }
            Ok(())
        })
    }
}

#[test]
fn source_transform_sink_emits_one_eos() {
    let tally = Arc::new(Mutex::new(Tally::default()));
    let mut src = CountingSource;
    let mut xform = ForwardingTransform;
    let mut sink = TallySink {
        tally: tally.clone(),
    };
    let stats = block_on(run_source_transform_sink(
        &mut src, &mut xform, &mut sink, &ZeroClock, 1,
    ))
    .expect("run completes without a Shutdown from a duplicate Eos");
    let t = tally.lock().unwrap();
    assert_eq!(t.eos, 1, "sink must see exactly one Eos");
    assert_eq!(t.frames, FRAMES);
    assert_eq!(stats.frames_consumed, FRAMES);
}

#[test]
fn graph_runner_emits_one_eos() {
    let tally = Arc::new(Mutex::new(Tally::default()));
    let mut g: Graph<GraphNodeRef<'static>> = Graph::new();
    let src = g.add_source(GraphNodeRef::source(CountingSource));
    let xform = g.add_transform(GraphNodeRef::element(ForwardingTransform));
    let sink = g.add_sink(GraphNodeRef::element(TallySink {
        tally: tally.clone(),
    }));
    g.link(src, xform).unwrap();
    g.link(xform, sink).unwrap();
    block_on(run_graph(g, &ZeroClock, 1)).expect("graph runs to EOS");
    let t = tally.lock().unwrap();
    assert_eq!(t.eos, 1, "sink must see exactly one Eos");
    assert_eq!(t.frames, FRAMES);
}
