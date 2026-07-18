//! M343: diamond allocation join policy + β allocation crossing a muxer.
//!
//! Two startup-negotiation gaps in the DAG allocation cascade:
//!   1. A tee's two branches proposing different memory domains used to silently
//!      keep one (largest size). Now the join is a most-restrictive intersection
//!      that fails loud on a domain conflict (`AllocationConflict`).
//!   2. A muxer used to be an allocation black hole: its inputs got no proposal,
//!      so a muxer that wants device-resident input buffers could not ask for
//!      them. Now `propose_allocation_for_input` crosses the boundary at startup
//!      and re-cascades up each branch independently.
//!
//! Pure-fake elements (no hardware).

use core::future::Future;
use core::pin::Pin;
use std::boxed::Box;
use std::sync::{Arc, Mutex};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_graph, GraphNode, SourceLoop};
use g2g_core::{
    AllocationParams, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim,
    DomainSet, FrameTiming, G2gError, Graph, MemoryDomain, MemoryDomainKind, MultiInputElement,
    OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
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

/// Emits `count` NV12 frames, then EOS. No mid-stream change: these tests
/// exercise the startup allocation cascade only.
struct PlainSource {
    count: u32,
}

impl SourceLoop for PlainSource {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(nv12(8, 8)))
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    /// A test fake standing in for a domain-flexible producer (eg a hardware
    /// decoder that can deliver to System or keep frames device-resident), so the
    /// M351 source-side reconciliation honors the device-domain proposals the
    /// branches join to.
    fn output_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System)
            .with(MemoryDomainKind::Cuda)
            .with(MemoryDomainKind::D3D11Texture)
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

/// Pass-through transform recording every full `AllocationParams` it absorbs,
/// so the proposal that re-cascaded into it (from a tee join or a muxer pad) is
/// observable in its entirety, not just the size.
struct RecordingTransform {
    log: Arc<Mutex<Vec<AllocationParams>>>,
}

impl AsyncElement for RecordingTransform {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn configure_allocation(&mut self, params: &AllocationParams) {
        self.log.lock().unwrap().push(*params);
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

/// Sink that proposes a fixed allocation (any memory domain) and accepts any
/// NV12 geometry.
struct DomainSink {
    params: AllocationParams,
}

impl AsyncElement for DomainSink {
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
        Some(self.params)
    }

    fn process<'a>(
        &'a mut self,
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

/// Interleave muxer that proposes a per-pad allocation (so a device-resident
/// muxer asking each video pad for GPU buffers is expressible). `per_pad[i]` is
/// the proposal for input pad `i`.
struct AllocMux {
    inputs: usize,
    output: Caps,
    per_pad: Vec<Option<AllocationParams>>,
}

impl MultiInputElement for AllocMux {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn input_count(&self) -> usize {
        self.inputs
    }

    fn intercept_caps(&self, _input: usize, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn caps_constraint_for_output(&self) -> Result<CapsConstraint<'_>, G2gError> {
        Ok(CapsConstraint::Produces(CapsSet::one(self.output.clone())))
    }

    fn propose_allocation_for_input(&self, input: usize, _caps: &Caps) -> Option<AllocationParams> {
        self.per_pad.get(input).copied().flatten()
    }

    fn configure_pipeline(
        &mut self,
        _input: usize,
        _absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(self.output.clone())
    }

    fn process<'a>(
        &'a mut self,
        _input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(f) => {
                    out.push(PipelinePacket::DataFrame(f)).await?;
                }
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

#[derive(Default)]
struct AnySink;

impl AsyncElement for AnySink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
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

/// Two branches of a tee proposing the same domain join to the most-restrictive
/// per parameter (larger size, count, alignment); the source absorbs the joined
/// proposal (reported as `stats.allocation`).
#[tokio::test]
async fn diamond_same_domain_joins_most_restrictive() {
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(PlainSource { count: 3 }));
    let tee = g.add_tee(2);
    let a = g.add_sink(GraphNode::element(DomainSink {
        params: AllocationParams::cuda(100, 2, 4),
    }));
    let b = g.add_sink(GraphNode::element(DomainSink {
        params: AllocationParams::cuda(200, 1, 8),
    }));
    g.link(src, tee.input()).unwrap();
    g.link(tee.out(0), a).unwrap();
    g.link(tee.out(1), b).unwrap();

    let stats = run_graph(g, &NullClock, 4)
        .await
        .expect("matching domains join");
    assert_eq!(
        stats.allocation,
        Some(AllocationParams::cuda(200, 2, 8)),
        "join keeps the larger size, count, and alignment"
    );
}

/// Two branches proposing different memory domains have no common pool, so the
/// join is an empty intersection and the whole negotiation fails loud rather
/// than silently honouring one branch and copying for the other.
#[tokio::test]
async fn diamond_domain_conflict_fails_loud() {
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(PlainSource { count: 3 }));
    let tee = g.add_tee(2);
    let a = g.add_sink(GraphNode::element(DomainSink {
        params: AllocationParams::cuda(128, 1, 1),
    }));
    let b = g.add_sink(GraphNode::element(DomainSink {
        params: AllocationParams::d3d11(128, 1, 1),
    }));
    g.link(src, tee.input()).unwrap();
    g.link(tee.out(0), a).unwrap();
    g.link(tee.out(1), b).unwrap();

    let result = run_graph(g, &NullClock, 4).await;
    assert_eq!(
        result.err(),
        Some(G2gError::AllocationConflict),
        "a CUDA branch and a D3D11 branch cannot share one producer pool"
    );
}

/// A muxer's per-pad allocation proposal crosses the boundary and re-cascades up
/// that branch: the transform on pad 0's branch absorbs the muxer's CUDA
/// proposal, while pad 1 (no proposal) leaves its branch untouched.
#[tokio::test]
async fn muxer_per_pad_allocation_crosses_boundary() {
    let log0 = Arc::new(Mutex::new(Vec::new()));
    let log1 = Arc::new(Mutex::new(Vec::new()));

    let mut g: Graph<GraphNode> = Graph::new();
    let s0 = g.add_source(GraphNode::source(PlainSource { count: 2 }));
    let s1 = g.add_source(GraphNode::source(PlainSource { count: 2 }));
    let t0 = g.add_transform(GraphNode::element(RecordingTransform {
        log: Arc::clone(&log0),
    }));
    let t1 = g.add_transform(GraphNode::element(RecordingTransform {
        log: Arc::clone(&log1),
    }));
    let mux = g.add_muxer(
        GraphNode::muxer(AllocMux {
            inputs: 2,
            output: nv12(8, 8),
            per_pad: vec![Some(AllocationParams::cuda(256, 3, 64)), None],
        }),
        2,
    );
    let sink = g.add_sink(GraphNode::element(AnySink));
    g.link(s0, t0).unwrap();
    g.link(s1, t1).unwrap();
    g.link(t0, mux.input(0)).unwrap();
    g.link(t1, mux.input(1)).unwrap();
    g.link(mux.output(), sink).unwrap();

    let stats = run_graph(g, &NullClock, 4)
        .await
        .expect("muxer alloc cascade runs");
    assert_eq!(stats.frames_emitted, 4, "2 + 2 source frames");
    assert_eq!(
        *log0.lock().unwrap(),
        vec![AllocationParams::cuda(256, 3, 64)],
        "pad 0's branch absorbed the muxer's per-pad CUDA proposal"
    );
    assert!(
        log1.lock().unwrap().is_empty(),
        "pad 1 carried no proposal, so its branch was left untouched"
    );
}
