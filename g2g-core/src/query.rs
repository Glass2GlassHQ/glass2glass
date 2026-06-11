//! Pipeline queries (M12).
//!
//! GStreamer answers a `LATENCY` query by walking the pipeline from sink to
//! source: each element folds its own latency contribution into the upstream
//! result, and the bin uses the aggregate to configure how much the sink must
//! buffer so a live source never starves. `g2g` composes paths statically, so
//! the aggregation is a fold over each element's [`LatencyReport`] rather than
//! a runtime query object travelling along pads. The linear runners compute it
//! once after negotiation and expose it on `RunStats`.

use crate::memory::MemoryDomainKind;

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
    /// Memory domain the consumer wants the buffers allocated in.
    pub domain: MemoryDomainKind,
}

impl Default for AllocationParams {
    fn default() -> Self {
        Self {
            size_bytes: 0,
            min_buffers: 1,
            align: 1,
            domain: MemoryDomainKind::System,
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
        }
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
        };
        let upstream = AllocationParams::system(4096, 2);
        let merged = downstream.merge(upstream);
        assert_eq!(merged.size_bytes, 4096, "larger size wins");
        assert_eq!(merged.min_buffers, 4, "larger buffer count wins");
        assert_eq!(merged.align, 64, "stricter alignment wins");
        assert_eq!(merged.domain, MemoryDomainKind::DmaBuf, "consumer domain dictates");
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
