//! M1019: the allocation cascade reads what a consumer declared it can take.
//!
//! An element names its memory domains in `input_domains`, but that used to
//! reach only the converter auto-plug: the cascade read `propose_allocation`
//! alone, so a consumer that accepts one domain and proposes nothing left its
//! producer free to pick any, and the mismatch surfaced as an
//! `UnsupportedDomain` on the first frame rather than at negotiation.
//!
//! The producer here is a decoder stand-in that can keep frames on the GPU or
//! download them, so the domain it ends up on is entirely the consumer's doing.
//! `negotiate_graph` runs the same cascade, so its per-edge domain (what a graph
//! dump renders) is asserted alongside.
#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;

use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::runtime::{block_on, negotiate_graph, GraphNode, SourceLoop};
use g2g_core::{
    AllocationParams, AsyncElement, Caps, ConfigureOutcome, Dim, G2gError, Graph, OutputSink,
    PipelinePacket, Rate, RawVideoFormat,
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

/// A sink that accepts exactly `accepted` and proposes nothing, the shape every
/// CPU display sink had before this landed.
struct DeclaringSink {
    accepted: DomainSet,
}

impl AsyncElement for DeclaringSink {
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

    fn input_domains(&self) -> DomainSet {
        self.accepted
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move { out.push(packet).await.map(|_| ()) })
    }
}

/// Negotiate a source -> sink graph and report the domain the link settled on.
fn settled_domain(sink_accepts: DomainSet) -> MemoryDomainKind {
    let mut graph: Graph<GraphNode> = Graph::new();
    let src = graph.add_source(GraphNode::source(DualDomainSource {
        settled: MemoryDomainKind::Cuda,
    }));
    let sink = graph.add_sink(GraphNode::element(DeclaringSink {
        accepted: sink_accepts,
    }));
    graph.link(src, sink).unwrap();

    let (_vg, _caps, edge_memory) = block_on(negotiate_graph(graph)).expect("negotiation");
    edge_memory[0]
}

#[test]
fn a_system_only_consumer_makes_the_producer_download() {
    assert_eq!(
        settled_domain(DomainSet::only(MemoryDomainKind::System)),
        MemoryDomainKind::System
    );
}

#[test]
fn a_gpu_consumer_keeps_the_frame_on_the_device() {
    assert_eq!(
        settled_domain(DomainSet::only(MemoryDomainKind::Cuda)),
        MemoryDomainKind::Cuda
    );
}

/// The all-domains default is what an element that never thought about memory
/// reports, and it has to keep narrowing nothing: the producer stays on the
/// domain it prefers, exactly as before this cascade read the declaration.
#[test]
fn a_consumer_declaring_nothing_leaves_the_producer_alone() {
    assert_eq!(settled_domain(DomainSet::ALL), MemoryDomainKind::Cuda);
}
