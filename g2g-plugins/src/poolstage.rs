//! `PoolStage`: a passthrough transform backed by a buffer pool that is rebuilt
//! to the downstream allocation proposal, including mid-stream (the β re-cascade).
//!
//! It is the "real consumer that re-sizes its pool" the allocation negotiation
//! was missing. The N-hop allocation re-cascade (M18 β) already delivers a
//! re-derived [`AllocationParams`] to an interior element's
//! [`configure_allocation`](AsyncElement::configure_allocation) when a mid-stream
//! caps change makes the sink re-propose, but the elements that received it so
//! far either only recorded it (test probes) or had pools fixed at open (the
//! decoders, whose codec pool is allocated once). `PoolStage` actually reacts:
//! each `configure_allocation` (re)builds a [`BufferPool`] sized to the proposal
//! (`min_buffers` x `size_bytes`), and every frame is staged through a buffer
//! from it, so a mid-stream geometry change visibly resizes a live pool.
//!
//! This models a staging / relay element that owns a downstream-sized pool (the
//! shape a real zero-copy hand-off pool would take). Caps are pass-through
//! (`Identity`): it does not change the format, only where the bytes live. A
//! frame larger than the current pool buffer, or one arriving before any
//! proposal, passes through untouched.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;

use g2g_core::memory::SystemSlice;
use g2g_core::{
    AllocationParams, AsyncElement, BufferPool, Caps, ConfigureOutcome, Frame, G2gError,
    MemoryDomain, OutputSink, PipelinePacket,
};

/// Passthrough transform whose buffer pool tracks the downstream allocation
/// proposal. See the module docs.
#[derive(Debug, Default)]
pub struct PoolStage {
    /// Pool sized to the last allocation proposal; rebuilt when it changes.
    pool: Option<BufferPool<Box<[u8]>>>,
    /// `(min_buffers, size_bytes)` the pool was last built with; a re-cascade
    /// with the same shape is a no-op (no needless pool churn).
    shape: Option<(usize, usize)>,
    /// How many times the pool has been (re)built. Startup is the first; each
    /// distinct mid-stream β proposal adds one. Useful in tests.
    reconfigures: usize,
    /// Frames actually staged through the pool (vs passed through untouched).
    staged: u64,
    configured: bool,
}

impl PoolStage {
    pub fn new() -> Self {
        Self::default()
    }

    /// `(min_buffers, size_bytes)` of the current pool, or `None` before the
    /// first allocation proposal.
    pub fn pool_shape(&self) -> Option<(usize, usize)> {
        self.shape
    }

    /// Buffer count of the current pool.
    pub fn pool_capacity(&self) -> Option<usize> {
        self.pool.as_ref().map(|p| p.capacity())
    }

    /// How many times the pool has been (re)built (startup + each distinct β).
    pub fn reconfigures(&self) -> usize {
        self.reconfigures
    }

    /// Frames staged through the pool so far.
    pub fn staged(&self) -> u64 {
        self.staged
    }
}

impl AsyncElement for PoolStage {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // Pass-through: the element changes only where bytes live, not the format.
        Ok(upstream_caps.clone())
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    /// Rebuild the pool to the downstream proposal. Called once at startup with
    /// the sink's initial proposal and again per mid-stream β re-cascade; a
    /// proposal of the same shape is ignored so the pool is not needlessly churned
    /// (which would drop the free buffers).
    fn configure_allocation(&mut self, params: &AllocationParams) {
        let count = params.min_buffers.max(1);
        let bytes = params.size_bytes.max(1);
        if self.shape == Some((count, bytes)) {
            return;
        }
        self.pool = Some(BufferPool::new_byte_pool(count, bytes));
        self.shape = Some((count, bytes));
        self.reconfigures += 1;
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let staged = self.stage(frame).await?;
                    out.push(PipelinePacket::DataFrame(staged)).await?;
                }
                // Caps / flush / segment / EOS flow through untouched; a
                // downstream sink re-deriving its proposal on CapsChanged is what
                // drives the β re-cascade back into `configure_allocation`.
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PoolStage {
    /// Stage one frame through the pool: copy a System frame that fits a pool
    /// buffer into one and emit it pool-backed; otherwise return it untouched.
    async fn stage(&mut self, frame: Frame) -> Result<Frame, G2gError> {
        let (Some(pool), Some((_, bytes))) = (self.pool.as_ref(), self.shape) else {
            return Ok(frame);
        };
        let MemoryDomain::System(src) = &frame.domain else {
            return Ok(frame);
        };
        let len = src.as_slice().len();
        if len > bytes {
            // Larger than the pool's buffers; pass through rather than truncate.
            return Ok(frame);
        }
        let pool = pool.clone();
        let mut buf = pool.acquire().await;
        buf[..len].copy_from_slice(src.as_slice());
        let domain = MemoryDomain::System(SystemSlice::from_pool(buf, len));
        self.staged += 1;
        Ok(Frame {
            domain,
            timing: frame.timing,
            sequence: frame.sequence,
            meta: frame.meta,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proposal(count: usize, bytes: usize) -> AllocationParams {
        AllocationParams::system(bytes, count)
    }

    #[test]
    fn rebuilds_pool_on_distinct_proposal_only() {
        let mut stage = PoolStage::new();
        assert_eq!(stage.pool_shape(), None);

        stage.configure_allocation(&proposal(2, 1024));
        assert_eq!(stage.pool_shape(), Some((2, 1024)));
        assert_eq!(stage.pool_capacity(), Some(2));
        assert_eq!(stage.reconfigures(), 1);

        // Same shape: no rebuild.
        stage.configure_allocation(&proposal(2, 1024));
        assert_eq!(stage.reconfigures(), 1);

        // New (larger) proposal: pool resized.
        stage.configure_allocation(&proposal(3, 4096));
        assert_eq!(stage.pool_shape(), Some((3, 4096)));
        assert_eq!(stage.pool_capacity(), Some(3));
        assert_eq!(stage.reconfigures(), 2);
    }
}
