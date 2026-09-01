//! M965: per-alternative preference costs decide a chain with competing
//! constraints. A source prefers NV12, a middle transform declares itself
//! indifferent, and a sink prefers I420 strongly. Index order alone picks NV12
//! (the source's first choice, which nothing later can outweigh); summing the
//! declared costs picks I420, because the sink's gap dwarfs the source's.
#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use std::sync::{Arc, Mutex};

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::solver::{
    solve_graph_preferred, solve_linear, solve_linear_preferred, NodeConstraint,
};
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
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn nv12() -> Caps {
    raw(RawVideoFormat::Nv12)
}

fn i420() -> Caps {
    raw(RawVideoFormat::I420)
}

/// Source: NV12 first, I420 second.
fn source_set() -> CapsSet {
    CapsSet::from_alternatives(vec![nv12(), i420()])
}

/// Middle transform: same two formats, same order as the source.
fn middle_set() -> CapsSet {
    CapsSet::from_alternatives(vec![nv12(), i420()])
}

/// Sink: I420 first, NV12 second.
fn sink_set() -> CapsSet {
    CapsSet::from_alternatives(vec![i420(), nv12()])
}

/// The sink cares a lot: its second choice costs far more than any upstream
/// element's second choice saves.
const SINK_FALLBACK_COST: u32 = 10;

// ---------------------------------------------------------------------------
// Solver level
// ---------------------------------------------------------------------------

#[test]
fn no_preferences_chain_picks_the_source_order() {
    let source = CapsConstraint::Produces(source_set());
    let middle = CapsConstraint::Identity(middle_set());
    let sink = CapsConstraint::Accepts(sink_set());

    let links = solve_linear(&[&source, &middle, &sink]).expect("chain negotiates");
    assert_eq!(links, vec![nv12(), nv12()], "index order picks NV12 today");

    // The preference-aware entry point with nothing declared is the same solve.
    let same = solve_linear_preferred(&[&source, &middle, &sink], &[None, None, None])
        .expect("chain negotiates");
    assert_eq!(same, links, "declaring nothing changes nothing");
}

#[test]
fn declared_costs_outweigh_index_order() {
    let source = CapsConstraint::Produces(source_set());
    let middle = CapsConstraint::Identity(middle_set());
    let sink = CapsConstraint::Accepts(sink_set());

    // Source: keeps index order (NV12 0, I420 1). Middle: indifferent.
    // Sink: I420 0, NV12 10.
    let preferences = vec![
        None,
        Some(CapsPreferences::indifferent(2)),
        Some(CapsPreferences::new(vec![0, SINK_FALLBACK_COST])),
    ];
    let links =
        solve_linear_preferred(&[&source, &middle, &sink], &preferences).expect("chain negotiates");
    assert_eq!(
        links,
        vec![i420(), i420()],
        "total cost 1 (source fallback) beats 10 (sink fallback)"
    );
}

#[test]
fn a_narrow_sink_gap_leaves_the_source_choice_alone() {
    // Same shape, but the sink only mildly prefers I420: NV12 costs it 1, which
    // the source's own 1 for I420 ties. The tie breaks toward the greedy pick.
    let source = CapsConstraint::Produces(source_set());
    let middle = CapsConstraint::Identity(middle_set());
    let sink = CapsConstraint::Accepts(sink_set());
    let preferences = vec![
        None,
        Some(CapsPreferences::indifferent(2)),
        Some(CapsPreferences::new(vec![0, 1])),
    ];
    let links =
        solve_linear_preferred(&[&source, &middle, &sink], &preferences).expect("chain negotiates");
    assert_eq!(links, vec![nv12(), nv12()], "a tie keeps today's pick");
}

#[test]
fn an_indifferent_middle_defers_to_its_neighbour() {
    // The middle transform's own order is NV12-first, the sink's is I420-first,
    // and the source accepts either. Declaring the middle indifferent hands the
    // decision to the sink; leaving it on index order keeps NV12.
    let source = CapsConstraint::Produces(CapsSet::from_alternatives(vec![nv12(), i420()]));
    let middle = CapsConstraint::Identity(middle_set());
    let sink = CapsConstraint::Accepts(sink_set());

    let indifferent = vec![
        Some(CapsPreferences::indifferent(2)),
        Some(CapsPreferences::indifferent(2)),
        None,
    ];
    let deferred =
        solve_linear_preferred(&[&source, &middle, &sink], &indifferent).expect("chain negotiates");
    assert_eq!(
        deferred,
        vec![i420(), i420()],
        "with both upstream elements indifferent the sink's order decides"
    );

    let by_order = vec![
        Some(CapsPreferences::by_order(2)),
        Some(CapsPreferences::by_order(2)),
        None,
    ];
    let kept =
        solve_linear_preferred(&[&source, &middle, &sink], &by_order).expect("chain negotiates");
    assert_eq!(
        kept,
        vec![nv12(), nv12()],
        "spelling out index order reproduces the default"
    );
}

#[test]
fn same_input_gives_the_same_pick_every_time() {
    let source = CapsConstraint::Produces(source_set());
    let middle = CapsConstraint::Identity(middle_set());
    let sink = CapsConstraint::Accepts(sink_set());
    let preferences = vec![
        None,
        Some(CapsPreferences::indifferent(2)),
        Some(CapsPreferences::new(vec![0, SINK_FALLBACK_COST])),
    ];
    let first = solve_linear_preferred(&[&source, &middle, &sink], &preferences).expect("solves");
    for _ in 0..8 {
        let again =
            solve_linear_preferred(&[&source, &middle, &sink], &preferences).expect("solves");
        assert_eq!(again, first, "the solve is deterministic");
    }
}

#[test]
fn graph_solve_minimizes_the_same_chain() {
    let constraints = vec![
        NodeConstraint::Element(CapsConstraint::Produces(source_set())),
        NodeConstraint::Element(CapsConstraint::Identity(middle_set())),
        NodeConstraint::Element(CapsConstraint::Accepts(sink_set())),
    ];
    let mut graph: Graph<()> = Graph::new();
    let source = graph.add_source(());
    let middle = graph.add_transform(());
    let sink = graph.add_sink(());
    graph.link(source, middle).unwrap();
    graph.link(middle, sink).unwrap();
    let validated = graph.finish().unwrap();

    let none = solve_graph_preferred(&validated, &constraints, &[], &|n| format!("n{}", n.0))
        .expect("solves");
    assert_eq!(none, vec![nv12(), nv12()], "no costs: today's pick");

    let preferences = vec![
        None,
        Some(CapsPreferences::indifferent(2)),
        Some(CapsPreferences::new(vec![0, SINK_FALLBACK_COST])),
    ];
    let costed = solve_graph_preferred(&validated, &constraints, &preferences, &|n| {
        format!("n{}", n.0)
    })
    .expect("solves");
    assert_eq!(costed, vec![i420(), i420()], "costs flip the chain to I420");
}

// ---------------------------------------------------------------------------
// Runner level: the same scenario built from elements, through `run_graph`.
// ---------------------------------------------------------------------------

const FRAMES: u64 = 2;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

struct TwoFormatSource;

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

/// Pass-through transform over both formats that declares itself indifferent.
struct IndifferentMiddle;

impl AsyncElement for IndifferentMiddle {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Identity(middle_set())
    }

    fn caps_preferences(&self) -> Option<CapsPreferences> {
        Some(CapsPreferences::indifferent(2))
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move { out.push(packet).await.map(|_| ()) })
    }
}

/// Sink that prefers I420 and says how much: NV12 costs it ten times the gap
/// the source's own order expresses.
struct PickySink {
    configured: Arc<Mutex<Option<Caps>>>,
    declare_costs: bool,
}

impl AsyncElement for PickySink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(sink_set())
    }

    fn caps_preferences(&self) -> Option<CapsPreferences> {
        self.declare_costs
            .then(|| CapsPreferences::new(vec![0, SINK_FALLBACK_COST]))
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

fn run_scenario(declare_costs: bool) -> Caps {
    let configured = Arc::new(Mutex::new(None));
    let mut graph: Graph<GraphNodeRef<'static>> = Graph::new();
    let source = graph.add_source(GraphNodeRef::source(TwoFormatSource));
    let middle = graph.add_transform(GraphNodeRef::element(IndifferentMiddle));
    let sink = graph.add_sink(GraphNodeRef::element(PickySink {
        configured: Arc::clone(&configured),
        declare_costs,
    }));
    graph.link(source, middle).unwrap();
    graph.link(middle, sink).unwrap();

    block_on(run_graph(graph, &ZeroClock, 1)).expect("graph runs to EOS");
    let caps = configured.lock().unwrap().clone();
    caps.expect("sink configured")
}

#[test]
fn runner_reads_declared_costs() {
    assert_eq!(
        run_scenario(false),
        nv12(),
        "with nothing declared the source's order wins, as today"
    );
    assert_eq!(
        run_scenario(true),
        i420(),
        "the sink's declared gap plus the middle's indifference flips it"
    );
}
