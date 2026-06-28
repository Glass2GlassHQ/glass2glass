//! M351: two-sided allocation-domain negotiation.
//!
//! Before M351 the memory domain of an allocation was a one-sided dictate: each
//! consumer named exactly one domain, the producer silently obeyed, and a tee
//! whose branches named different domains failed loud (`AllocationConflict`)
//! even when a domain satisfying both existed. M351 lets a producer advertise the
//! set of domains it *can* emit (`output_domains`) and a consumer the set it
//! *can* accept (`AllocationParams::accepts`); the cascade intersects them and
//! settles on the most-preferred common domain (GPU-resident before System), so
//! a graph that used to fail to negotiate now runs copy-free.
//!
//! Pure-fake elements (no hardware).

use core::future::Future;
use core::pin::Pin;
use std::boxed::Box;

use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_graph, GraphNode, SourceLoop};
use g2g_core::{
    AllocationParams, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim,
    DomainSet, Frame, FrameTiming, G2gError, Graph, MemoryDomain, MemoryDomainKind, OutputSink,
    PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn nv12(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn frame(seq: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(vec![seq as u8].into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: seq,
        meta: Default::default(),
    })
}

/// A source that can emit any of `domains`, preferring `preferred` if asked
/// unilaterally. The producer-capability half of the negotiation.
struct MultiDomainSource {
    count: u32,
    preferred: MemoryDomainKind,
    domains: DomainSet,
}

impl SourceLoop for MultiDomainSource {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(nv12(8, 8)))
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_memory(&self) -> MemoryDomainKind {
        self.preferred
    }

    fn output_domains(&self) -> DomainSet {
        self.domains
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let mut seq = 0u64;
            for _ in 0..self.count {
                out.push(frame(seq)).await?;
                seq += 1;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(seq)
        })
    }
}

/// Sink proposing `size`/`count` buffers it can accept in any domain in
/// `accepts`, preferring `prefer`. The consumer-acceptance half.
struct MultiDomainSink {
    prefer: MemoryDomainKind,
    accepts: DomainSet,
}

impl MultiDomainSink {
    fn params(&self) -> AllocationParams {
        AllocationParams { size_bytes: 64, min_buffers: 1, align: 1, domain: self.prefer, accepts: self.accepts }
    }
}

impl AsyncElement for MultiDomainSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }))
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn propose_allocation(&self, _caps: &Caps) -> Option<AllocationParams> {
        Some(self.params())
    }

    fn process<'a>(
        &'a mut self,
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

/// The headline: a `{System, Cuda}` producer feeding a tee to branch A
/// (accepts `{System, Cuda}`, prefers System) and branch B (accepts `{Cuda}`
/// only). The single-domain equality check would compare System vs Cuda and
/// conflict; the set intersection is `{Cuda}`, so the diamond negotiates Cuda
/// (zero-copy) and the graph runs.
#[tokio::test]
async fn diamond_negotiates_common_gpu_domain() {
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(MultiDomainSource {
        count: 3,
        preferred: MemoryDomainKind::System,
        domains: DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::Cuda),
    }));
    let tee = g.add_tee(2);
    let a = g.add_sink(GraphNode::element(MultiDomainSink {
        prefer: MemoryDomainKind::System,
        accepts: DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::Cuda),
    }));
    let b = g.add_sink(GraphNode::element(MultiDomainSink {
        prefer: MemoryDomainKind::Cuda,
        accepts: DomainSet::only(MemoryDomainKind::Cuda),
    }));
    g.link(src, tee.input()).unwrap();
    g.link(tee.out(0), a).unwrap();
    g.link(tee.out(1), b).unwrap();

    let stats = run_graph(g, &NullClock, 4).await.expect("overlapping accept sets negotiate");
    assert_eq!(
        stats.allocation.map(|p| p.domain),
        Some(MemoryDomainKind::Cuda),
        "branches share only Cuda, and the producer can emit it: keep the frame on the GPU",
    );
}

/// Control: the *same* topology where branch A accepts System only. Now the two
/// branches share no domain, so the negotiation fails loud exactly as a
/// single-domain conflict always has. Proves the win above comes from the
/// widened accept set, not from the conflict check going soft.
#[tokio::test]
async fn diamond_with_no_shared_domain_still_conflicts() {
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(MultiDomainSource {
        count: 3,
        preferred: MemoryDomainKind::System,
        domains: DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::Cuda),
    }));
    let tee = g.add_tee(2);
    let a = g.add_sink(GraphNode::element(MultiDomainSink {
        prefer: MemoryDomainKind::System,
        accepts: DomainSet::only(MemoryDomainKind::System),
    }));
    let b = g.add_sink(GraphNode::element(MultiDomainSink {
        prefer: MemoryDomainKind::Cuda,
        accepts: DomainSet::only(MemoryDomainKind::Cuda),
    }));
    g.link(src, tee.input()).unwrap();
    g.link(tee.out(0), a).unwrap();
    g.link(tee.out(1), b).unwrap();

    assert_eq!(
        run_graph(g, &NullClock, 4).await.err(),
        Some(G2gError::AllocationConflict),
        "System-only and Cuda-only branches still cannot share one pool",
    );
}

/// Producer capability bounds the choice: both branches would prefer Cuda, but
/// the source can only emit System, so the negotiation falls back to System
/// rather than handing the source a domain it cannot produce.
#[tokio::test]
async fn producer_capability_bounds_the_domain() {
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(MultiDomainSource {
        count: 2,
        preferred: MemoryDomainKind::System,
        domains: DomainSet::only(MemoryDomainKind::System),
    }));
    let tee = g.add_tee(2);
    let common = DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::Cuda);
    let a = g.add_sink(GraphNode::element(MultiDomainSink {
        prefer: MemoryDomainKind::Cuda,
        accepts: common,
    }));
    let b = g.add_sink(GraphNode::element(MultiDomainSink {
        prefer: MemoryDomainKind::Cuda,
        accepts: common,
    }));
    g.link(src, tee.input()).unwrap();
    g.link(tee.out(0), a).unwrap();
    g.link(tee.out(1), b).unwrap();

    let stats = run_graph(g, &NullClock, 4).await.expect("System is a shared domain");
    assert_eq!(
        stats.allocation.map(|p| p.domain),
        Some(MemoryDomainKind::System),
        "branches prefer Cuda but a System-only source forces System",
    );
}
