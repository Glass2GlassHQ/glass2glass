//! M954: the DAG runner negotiates a source's whole produce set, not just its
//! preferred alternative. A camera-shaped source offers several formats in
//! preference order; what downstream accepts decides which one it captures in,
//! and the source learns the choice from `configure_pipeline`.
#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use std::sync::{Arc, Mutex};

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{block_on, run_graph, GraphNodeRef, SourceLoop};
use g2g_core::{
    graph::Graph, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError,
    MemoryDomain, OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat, VideoCodec,
};

const FRAMES: u64 = 3;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn raw_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Yuyv,
        width: Dim::Fixed(2),
        height: Dim::Fixed(2),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
    }
}

fn compressed_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::Mjpeg,
        width: Dim::Fixed(2),
        height: Dim::Fixed(2),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Offers raw first, then compressed, and records which one it was configured
/// with, the way a capture source reads back the mode to run.
struct TwoFormatSource {
    configured: Arc<Mutex<Option<Caps>>>,
}

impl SourceLoop for TwoFormatSource {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(raw_caps()))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::from_alternatives(
            Vec::from([raw_caps(), compressed_caps()]),
        ))))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        *self.configured.lock().unwrap() = Some(absolute_caps.clone());
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for seq in 0..FRAMES {
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 16]))),
                    FrameTiming::default(),
                    seq,
                );
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(FRAMES)
        })
    }
}

/// Sink that accepts exactly one caps, the `capsfilter` / decoder half of the
/// negotiation. `None` accepts anything.
struct PinnedSink {
    accepts: Option<Caps>,
    got: Arc<Mutex<Option<Caps>>>,
}

impl AsyncElement for PinnedSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        match &self.accepts {
            Some(pin) => pin.intersect(upstream),
            None => Ok(upstream.clone()),
        }
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        match &self.accepts {
            Some(pin) => CapsConstraint::Accepts(CapsSet::one(pin.clone())),
            None => CapsConstraint::AcceptsAny,
        }
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        *self.got.lock().unwrap() = Some(absolute_caps.clone());
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move { Ok(()) })
    }
}

/// Run the two-format source into a sink pinned to `accepts`, returning the caps
/// the source was configured with and the caps the sink received.
fn negotiate(accepts: Option<Caps>) -> (Option<Caps>, Option<Caps>) {
    let configured = Arc::new(Mutex::new(None));
    let got = Arc::new(Mutex::new(None));
    let mut g: Graph<GraphNodeRef<'static>> = Graph::new();
    let src = g.add_source(GraphNodeRef::source(TwoFormatSource {
        configured: configured.clone(),
    }));
    let sink = g.add_sink(GraphNodeRef::element(PinnedSink {
        accepts,
        got: got.clone(),
    }));
    g.link(src, sink).unwrap();
    block_on(run_graph(g, &ZeroClock, 1)).expect("graph runs to EOS");
    let configured = configured.lock().unwrap().clone();
    let got = got.lock().unwrap().clone();
    (configured, got)
}

#[test]
fn an_unpinned_sink_takes_the_sources_preferred_format() {
    let (configured, got) = negotiate(None);
    assert_eq!(configured, Some(raw_caps()));
    assert_eq!(got, Some(raw_caps()));
}

/// The point of the produce set: a downstream pin on a non-first alternative
/// selects it instead of failing the solve.
#[test]
fn a_pinned_sink_selects_a_later_alternative() {
    let (configured, got) = negotiate(Some(compressed_caps()));
    assert_eq!(configured, Some(compressed_caps()));
    assert_eq!(got, Some(compressed_caps()));
}
