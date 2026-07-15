//! Pipeline queries (M12).
//!
//! GStreamer answers a `LATENCY` query by walking the pipeline from sink to
//! source: each element folds its own latency contribution into the upstream
//! result, and the bin uses the aggregate to configure how much the sink must
//! buffer so a live source never starves. `g2g` composes paths statically, so
//! the aggregation is a fold over each element's [`LatencyReport`] rather than
//! a runtime query object travelling along pads. The linear runners compute it
//! once after negotiation and expose it on `RunStats`.

use crate::memory::{DomainSet, MemoryDomainKind};

/// Downstream-proposed buffer allocation parameters (M12 ALLOCATION query).
///
/// A consumer answers its producer's allocation query with the buffer size,
/// count, alignment, and memory domain it needs, so the producer can allocate
/// directly into a compatible pool and hand buffers over without a copy. This
/// mirrors GStreamer's `ALLOCATION` query, where downstream proposes pools and
/// allocation parameters that upstream then honours.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocationParams {
    /// Minimum buffer size in bytes the producer must allocate.
    pub size_bytes: usize,
    /// Minimum number of buffers the pool should hold so the consumer can
    /// retain references without starving the producer.
    pub min_buffers: usize,
    /// Required byte alignment of each buffer (`1` = no constraint). Hardware
    /// consumers (DMA, GPU upload) commonly need 64- or 256-byte alignment.
    pub align: usize,
    /// Memory domain the consumer *prefers* the buffers allocated in. The
    /// single concrete domain when there is no choice; the preferred pick out of
    /// [`accepts`](Self::accepts) when there is.
    pub domain: MemoryDomainKind,
    /// Every memory domain this consumer can accept, not just its preferred one.
    /// Defaults to `only(domain)` so a single-domain consumer negotiates exactly
    /// as before; a consumer that can take more (e.g. a sink that can read GPU
    /// textures *or* fall back to System) widens this so the producer can keep
    /// the frame copy-free when it is able to. The producer reconciles this
    /// against what it can emit ([`resolve_for_producer`](Self::resolve_for_producer)).
    pub accepts: DomainSet,
}

impl Default for AllocationParams {
    fn default() -> Self {
        Self {
            size_bytes: 0,
            min_buffers: 1,
            align: 1,
            domain: MemoryDomainKind::System,
            accepts: DomainSet::only(MemoryDomainKind::System),
        }
    }
}

impl AllocationParams {
    /// A System-memory proposal of `size_bytes` × `min_buffers`, no alignment
    /// constraint.
    pub const fn system(size_bytes: usize, min_buffers: usize) -> Self {
        Self {
            size_bytes,
            min_buffers,
            align: 1,
            domain: MemoryDomainKind::System,
            accepts: DomainSet::only(MemoryDomainKind::System),
        }
    }

    /// A CUDA device-memory proposal: a GPU consumer (decoder feeding a GPU
    /// sink / inference) asks its producer to keep buffers resident on the
    /// device so the handoff is copy-free.
    pub const fn cuda(size_bytes: usize, min_buffers: usize, align: usize) -> Self {
        Self {
            size_bytes,
            min_buffers,
            align,
            domain: MemoryDomainKind::Cuda,
            accepts: DomainSet::only(MemoryDomainKind::Cuda),
        }
    }

    /// A Direct3D 11 texture proposal: the Windows analog of [`cuda`](Self::cuda).
    /// A DXGI / D3D11 consumer asks its producer (a DXVA decoder) to keep
    /// buffers resident in GPU textures so the handoff is copy-free.
    pub const fn d3d11(size_bytes: usize, min_buffers: usize, align: usize) -> Self {
        Self {
            size_bytes,
            min_buffers,
            align,
            domain: MemoryDomainKind::D3D11Texture,
            accepts: DomainSet::only(MemoryDomainKind::D3D11Texture),
        }
    }

    /// Fold an upstream element's own requirement into this (downstream)
    /// proposal: the larger size, buffer count, and alignment win. `self` is
    /// the consumer-most proposal and dictates the memory `domain`.
    pub fn merge(self, upstream: Self) -> Self {
        Self {
            size_bytes: self.size_bytes.max(upstream.size_bytes),
            min_buffers: self.min_buffers.max(upstream.min_buffers),
            align: self.align.max(upstream.align),
            domain: self.domain,
            // The consumer-most side dictates the domain, so its acceptance set
            // carries forward unchanged.
            accepts: self.accepts,
        }
    }

    /// Join two *sibling* proposals that share one producer (the two branches of
    /// a tee's diamond). Unlike [`merge`](Self::merge), neither side dominates,
    /// so the memory domain cannot be picked unilaterally: the producer must
    /// allocate one pool both branches can consume. The result accepts the
    /// domains *both* branches accept ([`DomainSet::intersect`]); the preferred
    /// domain is the most-preferred survivor, and the rest is the most-restrictive
    /// per parameter (the larger size, count, and alignment). An empty
    /// intersection (no domain satisfies, say, a CUDA-only branch and a
    /// D3D11-only branch) fails loud with [`G2gError::AllocationConflict`] rather
    /// than silently honouring one branch.
    ///
    /// Single-domain branches reduce to the old behavior exactly: two matching
    /// domains intersect to that one domain; two differing single domains
    /// intersect to empty and conflict.
    pub fn join(self, other: Self) -> Result<Self, crate::error::G2gError> {
        let accepts = self.accepts.intersect(other.accepts);
        let domain = accepts.preferred().ok_or(crate::error::G2gError::AllocationConflict)?;
        Ok(Self {
            size_bytes: self.size_bytes.max(other.size_bytes),
            min_buffers: self.min_buffers.max(other.min_buffers),
            align: self.align.max(other.align),
            domain,
            accepts,
        })
    }

    /// Reconcile this downstream proposal against what the producer can actually
    /// emit (`can`): intersect the accepted domains with the producer's
    /// capability and settle on the most-preferred common domain. This turns the
    /// allocation handoff from a one-sided dictate (consumer names a domain, the
    /// producer silently obeys or mismatches) into a real two-sided negotiation.
    /// Fails [`G2gError::AllocationConflict`] when producer and consumer share no
    /// domain, which is a genuine conflict needing an auto-plugged converter
    /// rather than something either side can resolve alone.
    ///
    /// A single-domain producer/consumer reduces to today's behavior: the result
    /// is the consumer's one domain when the producer can emit it, else a
    /// conflict.
    pub fn resolve_for_producer(self, can: DomainSet) -> Result<Self, crate::error::G2gError> {
        let accepts = self.accepts.intersect(can);
        let domain = accepts.preferred().ok_or(crate::error::G2gError::AllocationConflict)?;
        Ok(Self { domain, accepts, ..self })
    }
}

/// One element's contribution to a path's latency, plus the aggregate of a
/// whole path. Mirrors GStreamer's `(live, min, max)` latency triple.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatencyReport {
    /// At least one element on the path is a live source. A live path forces
    /// the sink to buffer `min_ns` so it never runs dry waiting for capture.
    pub live: bool,
    /// Minimum latency the element (or path) introduces, in nanoseconds.
    /// Accumulates along the chain.
    pub min_ns: u64,
    /// Maximum latency that can be absorbed before buffers overflow, in
    /// nanoseconds. `Some(0)` is a non-buffering element (adds no slack);
    /// `None` means *unbounded* — the element can buffer arbitrarily and so
    /// imposes no ceiling on the path. A finite path max below the path min
    /// makes the latency unconfigurable.
    pub max_ns: Option<u64>,
}

impl Default for LatencyReport {
    fn default() -> Self {
        Self::ZERO
    }
}

impl LatencyReport {
    /// A zero-latency, non-live, non-buffering element: the default
    /// contribution and the identity for [`combine`](Self::combine). Note
    /// `max_ns` is `Some(0)` (adds no buffering slack), not `None`
    /// (unbounded), so folding it leaves a path aggregate unchanged.
    pub const ZERO: Self = Self {
        live: false,
        min_ns: 0,
        max_ns: Some(0),
    };

    /// A live element contributing `min_ns` of latency (eg a source pacing to
    /// a capture clock) with an optional buffering ceiling `max_ns`.
    pub const fn live(min_ns: u64, max_ns: Option<u64>) -> Self {
        Self {
            live: true,
            min_ns,
            max_ns,
        }
    }

    /// A non-live element that still adds latency (eg a jitter buffer or a
    /// decoder's frame-reordering depth).
    pub const fn buffered(min_ns: u64, max_ns: Option<u64>) -> Self {
        Self {
            live: false,
            min_ns,
            max_ns,
        }
    }

    /// Fold the next downstream element's contribution into this path
    /// aggregate: minimum and maximum latencies sum (an unbounded max stays
    /// unbounded), and liveness is sticky once any element is live.
    pub fn combine(self, next: Self) -> Self {
        Self {
            live: self.live || next.live,
            min_ns: self.min_ns.saturating_add(next.min_ns),
            max_ns: match (self.max_ns, next.max_ns) {
                (Some(a), Some(b)) => Some(a.saturating_add(b)),
                _ => None,
            },
        }
    }

    /// Aggregate a whole path, source first through sink last.
    pub fn aggregate<I>(reports: I) -> Self
    where
        I: IntoIterator<Item = LatencyReport>,
    {
        reports.into_iter().fold(Self::ZERO, Self::combine)
    }

    /// True when the path's maximum latency cannot absorb its minimum, i.e.
    /// GStreamer's "latency too big to configure" failure.
    pub fn is_unsatisfiable(&self) -> bool {
        matches!(self.max_ns, Some(max) if max < self.min_ns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_is_combine_identity() {
        let r = LatencyReport::live(5, Some(20));
        assert_eq!(LatencyReport::ZERO.combine(r), r);
        assert_eq!(r.combine(LatencyReport::ZERO), r);
    }

    #[test]
    fn min_and_max_sum_along_path() {
        let agg = LatencyReport::aggregate([
            LatencyReport::live(10, Some(40)),
            LatencyReport::buffered(5, Some(15)),
            LatencyReport::buffered(3, Some(10)),
        ]);
        assert!(agg.live);
        assert_eq!(agg.min_ns, 18);
        assert_eq!(agg.max_ns, Some(65));
    }

    #[test]
    fn unbounded_max_is_infectious() {
        let agg = LatencyReport::aggregate([
            LatencyReport::live(10, Some(40)),
            LatencyReport::buffered(5, None),
        ]);
        assert_eq!(agg.max_ns, None);
        assert!(!agg.is_unsatisfiable());
    }

    #[test]
    fn liveness_is_sticky() {
        let agg = LatencyReport::aggregate([
            LatencyReport::buffered(1, None),
            LatencyReport::live(2, None),
            LatencyReport::buffered(3, None),
        ]);
        assert!(agg.live);
    }

    #[test]
    fn alloc_merge_takes_most_demanding() {
        let downstream = AllocationParams {
            size_bytes: 1024,
            min_buffers: 4,
            align: 64,
            domain: MemoryDomainKind::DmaBuf,
            accepts: DomainSet::only(MemoryDomainKind::DmaBuf),
        };
        let upstream = AllocationParams::system(4096, 2);
        let merged = downstream.merge(upstream);
        assert_eq!(merged.size_bytes, 4096, "larger size wins");
        assert_eq!(merged.min_buffers, 4, "larger buffer count wins");
        assert_eq!(merged.align, 64, "stricter alignment wins");
        assert_eq!(merged.domain, MemoryDomainKind::DmaBuf, "consumer domain dictates");
    }

    #[test]
    fn join_matching_single_domains_intersects_to_that_domain() {
        // Backward-compat: two single-domain branches that agree join exactly as
        // the old equality check did.
        let a = AllocationParams::cuda(2048, 2, 256);
        let b = AllocationParams::cuda(4096, 4, 256);
        let j = a.join(b).expect("matching domains join");
        assert_eq!(j.domain, MemoryDomainKind::Cuda);
        assert_eq!(j.size_bytes, 4096);
        assert_eq!(j.min_buffers, 4);
    }

    #[test]
    fn join_disjoint_single_domains_conflicts() {
        // Backward-compat: two differing single domains still fail loud.
        let cuda = AllocationParams::cuda(1024, 2, 256);
        let d3d = AllocationParams::d3d11(1024, 2, 256);
        assert_eq!(cuda.join(d3d), Err(crate::error::G2gError::AllocationConflict));
    }

    #[test]
    fn join_overlapping_multidomain_branches_picks_common_preferred() {
        // The new win: branch A accepts {System, Cuda}, branch B accepts {Cuda}
        // only. Their intersection is {Cuda}, so the join succeeds on Cuda where
        // the old single-domain equality check (System vs Cuda) would conflict.
        let mut a = AllocationParams::system(1024, 2);
        a.accepts = DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::Cuda);
        let b = AllocationParams::cuda(1024, 2, 256);
        let j = a.join(b).expect("overlapping accept sets join");
        assert_eq!(j.domain, MemoryDomainKind::Cuda, "common domain, GPU-preferred");
    }

    #[test]
    fn resolve_for_producer_keeps_frame_on_gpu_when_both_can() {
        // Consumer accepts {System, Cuda} preferring System; producer can emit
        // {System, Cuda}. The reconciliation keeps it on the GPU (zero-copy)
        // because Cuda outranks System in the preference order.
        let mut want = AllocationParams::system(1024, 2);
        want.accepts = DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::Cuda);
        let can = DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::Cuda);
        let r = want.resolve_for_producer(can).expect("shared domain");
        assert_eq!(r.domain, MemoryDomainKind::Cuda);
    }

    #[test]
    fn resolve_for_producer_falls_back_to_system_when_gpu_unavailable() {
        // Consumer accepts {System, Cuda}; producer can only do System. The
        // reconciliation settles on System rather than blindly honouring Cuda.
        let mut want = AllocationParams::cuda(1024, 2, 256);
        want.accepts = DomainSet::only(MemoryDomainKind::Cuda).with(MemoryDomainKind::System);
        let can = DomainSet::only(MemoryDomainKind::System);
        let r = want.resolve_for_producer(can).expect("System is shared");
        assert_eq!(r.domain, MemoryDomainKind::System);
    }

    #[test]
    fn resolve_for_producer_conflicts_with_no_shared_domain() {
        // A Cuda-only consumer against a System-only producer is a genuine
        // conflict (needs a converter), surfaced loud instead of silently
        // mishandled as today's one-sided dictate would.
        let want = AllocationParams::cuda(1024, 2, 256);
        let can = DomainSet::only(MemoryDomainKind::System);
        assert_eq!(
            want.resolve_for_producer(can),
            Err(crate::error::G2gError::AllocationConflict)
        );
    }

    #[test]
    fn alloc_system_constructor_defaults() {
        let p = AllocationParams::system(512, 3);
        assert_eq!(p.align, 1);
        assert_eq!(p.domain, MemoryDomainKind::System);
        assert_eq!((p.size_bytes, p.min_buffers), (512, 3));
    }

    #[test]
    fn detects_unsatisfiable_latency() {
        // An element whose ceiling is below its own floor cannot be configured.
        let bad = LatencyReport {
            live: true,
            min_ns: 50,
            max_ns: Some(30),
        };
        assert!(bad.is_unsatisfiable());
        assert!(!LatencyReport::live(50, Some(80)).is_unsatisfiable());
    }
}
