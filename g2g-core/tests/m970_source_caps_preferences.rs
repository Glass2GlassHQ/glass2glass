//! M970: a source declares per-alternative costs over its produce set, the
//! same channel M965 gave transforms and sinks. Without them the produce-set
//! order decides the fixation on its own; with them a source can say it does
//! not care (letting downstream pick) or that a later alternative is the one it
//! wants.
#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use std::sync::{Arc, Mutex};

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{block_on, run_graph, GraphNodeRef, SourceLoop};
use g2g_core::{
    graph::Graph, AsyncElement, Caps, CapsConstraint, CapsPreferences, CapsSet, ConfigureOutcome,
    Dim, G2gError, MemoryDomain, OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

fn raw(format: RawVideoFormat) -> Caps {
    Caps::RawVideo {
        format,
        width: Dim::Fixed(64),
        height: Dim::Fixed(48),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Progressive,
    }
}

fn nv12() -> Caps {
    raw(RawVideoFormat::Nv12)
}

fn i420() -> Caps {
    raw(RawVideoFormat::I420)
}

/// The source produces both formats, NV12 first.
fn source_set() -> CapsSet {
    CapsSet::from_alternatives(vec![nv12(), i420()])
}

/// The source's second choice costs it far more than any downstream order saves.
const SOURCE_FALLBACK_COST: u32 = 10;

const FRAMES: u64 = 2;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Source over both formats whose declared costs the test supplies.
struct TwoFormatSource {
    preferences: Option<CapsPreferences>,
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
        core::future::ready(Ok(nv12()))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(source_set())))
    }

    fn caps_preferences(&self) -> Option<CapsPreferences> {
        self.preferences.clone()
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
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

/// Sink that accepts both formats in the order the test gives it and declares
/// no costs of its own, so only its order speaks for it.
struct RecordingSink {
    accepted: CapsSet,
    configured: Arc<Mutex<Option<Caps>>>,
}

impl AsyncElement for RecordingSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(self.accepted.clone())
    }

    fn configure_pipeline(&mut self, caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        *self.configured.lock().unwrap() = Some(caps.clone());
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

/// Run source -> sink and report the caps the sink was configured with.
fn run_scenario(preferences: Option<CapsPreferences>, sink_accepts: CapsSet) -> Caps {
    let configured = Arc::new(Mutex::new(None));
    let mut graph: Graph<GraphNodeRef<'static>> = Graph::new();
    let source = graph.add_source(GraphNodeRef::source(TwoFormatSource { preferences }));
    let sink = graph.add_sink(GraphNodeRef::element(RecordingSink {
        accepted: sink_accepts,
        configured: Arc::clone(&configured),
    }));
    graph.link(source, sink).unwrap();

    block_on(run_graph(graph, &ZeroClock, 1)).expect("graph runs to EOS");
    let caps = configured.lock().unwrap().clone();
    caps.expect("sink configured")
}

/// I420 first, NV12 second: the opposite of the source's own order.
fn sink_prefers_i420() -> CapsSet {
    CapsSet::from_alternatives(vec![i420(), nv12()])
}

/// Same order as the source, so nothing downstream argues for I420.
fn sink_prefers_nv12() -> CapsSet {
    CapsSet::from_alternatives(vec![nv12(), i420()])
}

#[test]
fn a_source_declaring_nothing_keeps_its_produce_set_order() {
    assert_eq!(
        run_scenario(None, sink_prefers_i420()),
        nv12(),
        "with no costs declared the source's first alternative wins"
    );
}

#[test]
fn an_indifferent_source_lets_the_sink_decide() {
    assert_eq!(
        run_scenario(Some(CapsPreferences::indifferent(2)), sink_prefers_i420()),
        i420(),
        "equal costs hand the pick to the sink's order"
    );
}

#[test]
fn source_costs_invert_its_own_produce_set_order() {
    let inverted = CapsPreferences::new(vec![SOURCE_FALLBACK_COST, 0]);
    assert_eq!(
        run_scenario(Some(inverted), sink_prefers_nv12()),
        i420(),
        "the source's declared costs beat both its own order and the sink's"
    );
    assert_eq!(
        run_scenario(None, sink_prefers_nv12()),
        nv12(),
        "the same graph without costs keeps NV12"
    );
}
