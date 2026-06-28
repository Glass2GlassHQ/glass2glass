//! M12: pipeline allocation query.
//!
//! A consumer answers its producer's allocation query with an
//! `AllocationParams` proposal; the linear runners convey it upstream so the
//! producer allocates its output pool to match (GStreamer's `ALLOCATION`
//! query). These tests drive the real runners: a source builds a real
//! `BufferPool` from the proposal it receives, a sink reads back the size of
//! the buffers it gets, and a transform folds its own requirement in.

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_simple_pipeline, run_source_transform_sink, SourceLoop};
use g2g_core::{
    AllocationParams, AsyncElement, BufferPool, Caps, ConfigureOutcome, Dim, DomainSet, G2gError,
    MemoryDomain, MemoryDomainKind, OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Source that builds its output `BufferPool` from the allocation proposal it
/// is handed (falling back to `fallback_size` if none arrives), then emits one
/// pooled DataFrame so downstream can observe the buffer size.
struct PoolSrc {
    fallback_size: usize,
    proposed: Option<AllocationParams>,
}

impl PoolSrc {
    fn new(fallback_size: usize) -> Self {
        Self { fallback_size, proposed: None }
    }

    fn pool_buffer_size(&self) -> usize {
        self.proposed.map(|p| p.size_bytes).unwrap_or(self.fallback_size)
    }
}

impl SourceLoop for PoolSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(caps()))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn configure_allocation(&mut self, params: &AllocationParams) {
        self.proposed = Some(*params);
    }

    /// A test fake standing in for a domain-flexible producer (eg `NvdecCuda`),
    /// so the M351 source-side reconciliation honors a downstream CUDA proposal
    /// rather than rejecting it against a System-only default.
    fn output_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::Cuda)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        let size = self.pool_buffer_size();
        let count = self.proposed.map(|p| p.min_buffers).unwrap_or(1).max(1);
        Box::pin(async move {
            // Real pool allocation from the negotiated parameters.
            let pool: BufferPool<Box<[u8]>> = BufferPool::new_byte_pool(count, size);
            let buf = pool.acquire().await;
            let frame = Frame {
                domain: MemoryDomain::System(SystemSlice::from_pool(buf, size)),
                timing: FrameTiming::default(),
                sequence: 0,
                meta: Default::default(),
            };
            out.push(PipelinePacket::DataFrame(frame)).await?;
            out.push(PipelinePacket::Eos).await?;
            Ok(1)
        })
    }
}

/// Sink that proposes an allocation and records the size of each System
/// buffer it receives.
struct ProposingSink {
    proposal: Option<AllocationParams>,
    received_sizes: Vec<usize>,
}

impl ProposingSink {
    fn new(proposal: Option<AllocationParams>) -> Self {
        Self { proposal, received_sizes: Vec::new() }
    }
}

impl AsyncElement for ProposingSink {
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

    fn propose_allocation(&self, _caps: &Caps) -> Option<AllocationParams> {
        self.proposal
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        if let PipelinePacket::DataFrame(f) = &packet {
            if let MemoryDomain::System(slice) = &f.domain {
                self.received_sizes.push(slice.as_slice().len());
            }
        }
        Box::pin(async { Ok(()) })
    }
}

/// Identity transform that records the downstream proposal it is configured
/// with and answers upstream by folding in its own minimum buffer size.
struct FoldingTransform {
    own_min_size: usize,
    seen_downstream: Option<AllocationParams>,
}

impl AsyncElement for FoldingTransform {
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

    fn configure_allocation(&mut self, params: &AllocationParams) {
        self.seen_downstream = Some(*params);
    }

    fn propose_allocation(&self, _caps: &Caps) -> Option<AllocationParams> {
        // Fold the downstream proposal with this element's own requirement.
        let own = AllocationParams::system(self.own_min_size, 1);
        Some(match self.seen_downstream {
            Some(downstream) => downstream.merge(own),
            None => own,
        })
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::Eos => Ok(()),
                other => {
                    out.push(other).await?;
                    Ok(())
                }
            }
        })
    }
}

/// Records, at `configure_pipeline` time, whether its allocation had already
/// been configured. A hardware decoder uses exactly this ordering to size its
/// buffer pool (`extra_hw_frames` / output texture count) from the downstream
/// `min_buffers` when it opens the codec.
struct OrderRecordingTransform {
    alloc_configured: bool,
    /// `Some(true)` if `configure_allocation` ran before `configure_pipeline`.
    alloc_seen_at_open: Option<bool>,
}

impl AsyncElement for OrderRecordingTransform {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn configure_allocation(&mut self, _params: &AllocationParams) {
        self.alloc_configured = true;
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.alloc_seen_at_open = Some(self.alloc_configured);
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::Eos => Ok(()),
                other => {
                    out.push(other).await?;
                    Ok(())
                }
            }
        })
    }
}

#[tokio::test]
async fn transform_allocation_precedes_configure_pipeline() {
    // The reorder under test: the M12 allocation query now runs before the
    // configure_pipeline cascade, so a transform (decoder) has its downstream
    // min_buffers in hand when it opens, instead of after.
    let mut src = PoolSrc::new(64);
    let mut tx = OrderRecordingTransform { alloc_configured: false, alloc_seen_at_open: None };
    let mut sink = ProposingSink::new(Some(AllocationParams::cuda(4096, 4, 256)));
    let clock = ZeroClock;

    run_source_transform_sink(&mut src, &mut tx, &mut sink, &clock, 4)
        .await
        .expect("pipeline should complete");

    assert_eq!(
        tx.alloc_seen_at_open,
        Some(true),
        "configure_allocation must run before configure_pipeline so a decoder \
         can size its pool from min_buffers at open time"
    );
}

#[tokio::test]
async fn source_sink_pool_handoff() {
    let mut src = PoolSrc::new(64);
    let mut sink = ProposingSink::new(Some(AllocationParams::system(1024, 4)));
    let clock = ZeroClock;

    let stats = run_simple_pipeline(&mut src, &mut sink, &clock, 4)
        .await
        .expect("pipeline should complete");

    assert_eq!(
        stats.allocation,
        Some(AllocationParams::system(1024, 4)),
        "the sink's proposal must be conveyed to the source"
    );
    assert_eq!(src.pool_buffer_size(), 1024, "source sized its pool from the proposal");
    assert_eq!(
        sink.received_sizes,
        vec![1024],
        "sink received a buffer sized to its own proposal"
    );
}

#[tokio::test]
async fn three_stage_transform_folds_its_requirement() {
    // Sink wants 2048-byte buffers; transform needs at least 4096. The folded
    // proposal handed to the source takes the larger size, the larger count.
    let mut src = PoolSrc::new(64);
    let mut tx = FoldingTransform { own_min_size: 4096, seen_downstream: None };
    let mut sink = ProposingSink::new(Some(AllocationParams::system(2048, 3)));
    let clock = ZeroClock;

    let stats = run_source_transform_sink(&mut src, &mut tx, &mut sink, &clock, 4)
        .await
        .expect("pipeline should complete");

    assert_eq!(
        tx.seen_downstream,
        Some(AllocationParams::system(2048, 3)),
        "transform sees the sink's proposal first"
    );
    assert_eq!(
        stats.allocation,
        Some(AllocationParams::system(4096, 3)),
        "source receives the folded (most-demanding) proposal"
    );
    assert_eq!(src.pool_buffer_size(), 4096);
    assert_eq!(sink.received_sizes, vec![4096]);
}

#[tokio::test]
async fn cuda_domain_proposal_conveyed_to_source() {
    // C3: a GPU consumer asks for device-resident buffers (MemoryDomainKind::
    // Cuda) so a CUDA producer (eg NvdecCuda) can hand frames over copy-free.
    // The allocation query must convey the consumer's domain (and size/align)
    // upstream unchanged. The fake source still allocates a System pool (it
    // can't touch CUDA on this host); the assertion is on the conveyed query,
    // which is the cross-element handoff the GPU path depends on.
    let mut src = PoolSrc::new(64);
    let mut sink = ProposingSink::new(Some(AllocationParams::cuda(4096, 4, 256)));
    let clock = ZeroClock;

    let stats = run_simple_pipeline(&mut src, &mut sink, &clock, 4)
        .await
        .expect("pipeline should complete");

    let alloc = stats.allocation.expect("the sink's CUDA proposal must be conveyed");
    assert_eq!(
        alloc.domain,
        MemoryDomainKind::Cuda,
        "the consumer's CUDA domain reaches the producer"
    );
    assert_eq!((alloc.size_bytes, alloc.align), (4096, 256));
    assert_eq!(
        src.proposed.map(|p| p.domain),
        Some(MemoryDomainKind::Cuda),
        "source's configure_allocation saw the CUDA domain"
    );
}

#[tokio::test]
async fn cuda_domain_survives_transform_fold() {
    // A CUDA consumer proposal folded with an intermediate transform's larger
    // System requirement: the most-demanding size/align win, but the consumer
    // dictates the domain, so the source is still asked for CUDA memory.
    let mut src = PoolSrc::new(64);
    let mut tx = FoldingTransform { own_min_size: 8192, seen_downstream: None };
    let mut sink = ProposingSink::new(Some(AllocationParams::cuda(4096, 2, 256)));
    let clock = ZeroClock;

    let stats = run_source_transform_sink(&mut src, &mut tx, &mut sink, &clock, 4)
        .await
        .expect("pipeline should complete");

    let alloc = stats.allocation.expect("folded proposal conveyed");
    assert_eq!(
        alloc.domain,
        MemoryDomainKind::Cuda,
        "consumer domain dictates through the fold"
    );
    assert_eq!(alloc.size_bytes, 8192, "larger transform size wins");
    assert_eq!(alloc.align, 256, "consumer alignment preserved");
}

#[tokio::test]
async fn no_proposal_leaves_source_on_fallback() {
    let mut src = PoolSrc::new(128);
    let mut sink = ProposingSink::new(None);
    let clock = ZeroClock;

    let stats = run_simple_pipeline(&mut src, &mut sink, &clock, 4)
        .await
        .expect("pipeline should complete");

    assert_eq!(stats.allocation, None, "no downstream proposal");
    assert_eq!(src.pool_buffer_size(), 128, "source used its fallback size");
    assert_eq!(sink.received_sizes, vec![128]);
}
