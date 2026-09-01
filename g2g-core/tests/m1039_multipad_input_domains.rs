//! M1039: a fan-in / fan-out element's `input_domains` reaches the allocation
//! cascade, the way M1019 made a single-pad element's reach it.
//!
//! A muxer and a demux used to be hardcoded as "accepts everything" at the graph
//! node, so a GPU-capable producer feeding a muxer that only reads host memory
//! settled on the device domain and the mismatch surfaced as an
//! `UnsupportedDomain` on the first frame rather than as a download planned at
//! negotiation.
//!
//! The producer is the M1019 decoder stand-in: it can keep frames on the GPU or
//! download them, so the domain the link ends up on is entirely the consumer's
//! doing.
#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;

use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::runtime::{block_on, negotiate_graph, GraphNode, SourceLoop};
use g2g_core::{
    AllocationParams, AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, G2gError, Graph,
    MultiInputElement, MultiOutputElement, MultiOutputSink, OutputSink, PipelinePacket, Rate,
    RawVideoFormat,
};

fn nv12() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

/// A decoder stand-in that can deliver to either domain, preferring to stay on
/// the GPU. Whichever the cascade settles on is what `output_memory` reports,
/// the same contract `NvDec` follows.
struct DualDomainSource {
    settled: MemoryDomainKind,
}

impl SourceLoop for DualDomainSource {
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

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::Cuda).with(MemoryDomainKind::System)
    }

    fn output_memory(&self) -> MemoryDomainKind {
        self.settled
    }

    fn configure_allocation(&mut self, params: &AllocationParams) {
        if let Ok(resolved) = params.resolve_for_producer(self.output_domains()) {
            self.settled = resolved.domain;
        }
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(PipelinePacket::Eos).await?;
            Ok(0)
        })
    }
}

/// A one-pad muxer that accepts exactly `accepted` and proposes nothing, the
/// shape every container muxer has.
struct DeclaringMux {
    accepted: DomainSet,
}

impl MultiInputElement for DeclaringMux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        1
    }

    fn input_domains(&self) -> DomainSet {
        self.accepted
    }

    fn intercept_caps(&self, _input: usize, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(
        &mut self,
        _input: usize,
        _absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(nv12())
    }

    fn process<'a>(
        &'a mut self,
        _input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move { out.push(packet).await.map(|_| ()) })
    }
}

/// A one-port demux that accepts exactly `accepted`, the byte-parsing fan-out
/// shape.
struct DeclaringDemux {
    accepted: DomainSet,
}

impl MultiOutputElement for DeclaringDemux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_domains(&self) -> DomainSet {
        self.accepted
    }

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_input(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move { out.push_to(0, packet).await.map(|_| ()) })
    }
}

/// A sink imposing nothing, so the domain the graph settles on is the fan
/// element's doing alone.
struct AnySink;

impl AsyncElement for AnySink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
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

/// Negotiate source -> muxer -> sink and report the domain the source's link
/// settled on.
fn domain_into_mux(accepted: DomainSet) -> MemoryDomainKind {
    let mut graph: Graph<GraphNode> = Graph::new();
    let src = graph.add_source(GraphNode::source(DualDomainSource {
        settled: MemoryDomainKind::Cuda,
    }));
    let mux = graph.add_muxer(GraphNode::muxer(DeclaringMux { accepted }), 1);
    let sink = graph.add_sink(GraphNode::element(AnySink));
    graph.link(src, mux.input(0)).unwrap();
    graph.link(mux.output(), sink).unwrap();

    let (_vg, _caps, edge_memory) = block_on(negotiate_graph(graph)).expect("negotiation");
    edge_memory[0]
}

/// Negotiate source -> demux -> sink and report the domain the source's link
/// settled on.
fn domain_into_demux(accepted: DomainSet) -> MemoryDomainKind {
    let mut graph: Graph<GraphNode> = Graph::new();
    let src = graph.add_source(GraphNode::source(DualDomainSource {
        settled: MemoryDomainKind::Cuda,
    }));
    let demux = graph.add_demux(GraphNode::demux(DeclaringDemux { accepted }), 1);
    let sink = graph.add_sink(GraphNode::element(AnySink));
    graph.link(src, demux.input()).unwrap();
    graph.link(demux.out(0), sink).unwrap();

    let (_vg, _caps, edge_memory) = block_on(negotiate_graph(graph)).expect("negotiation");
    edge_memory[0]
}

#[test]
fn a_system_only_muxer_makes_the_producer_download() {
    assert_eq!(
        domain_into_mux(DomainSet::only(MemoryDomainKind::System)),
        MemoryDomainKind::System
    );
}

#[test]
fn a_gpu_muxer_keeps_the_frame_on_the_device() {
    assert_eq!(
        domain_into_mux(DomainSet::only(MemoryDomainKind::Cuda)),
        MemoryDomainKind::Cuda
    );
}

/// The all-domains default is what a fan-in that never thought about memory
/// reports, and it has to keep narrowing nothing.
#[test]
fn a_muxer_declaring_nothing_leaves_the_producer_alone() {
    assert_eq!(domain_into_mux(DomainSet::ALL), MemoryDomainKind::Cuda);
}

#[test]
fn a_system_only_demux_makes_the_producer_download() {
    assert_eq!(
        domain_into_demux(DomainSet::only(MemoryDomainKind::System)),
        MemoryDomainKind::System
    );
}

#[test]
fn a_demux_declaring_nothing_leaves_the_producer_alone() {
    assert_eq!(domain_into_demux(DomainSet::ALL), MemoryDomainKind::Cuda);
}
