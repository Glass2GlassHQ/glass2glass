//! Lightweight metrics primitives for glass-to-glass observability.
//!
//! This module is `no_std + alloc` compatible. The histogram uses log2
//! buckets in nanoseconds and an array of atomic counters, so it can be
//! shared across threads via `Arc<LatencyHistogram>` without locking.
//!
//! Under feature `std`, a process-wide `monotonic_ns()` helper exposes a
//! single shared epoch so a source-side stamp and a sink-side `now()`
//! agree on what "zero" means.

use core::sync::atomic::{AtomicU64, Ordering};

/// Number of log2 buckets. Bucket `k` covers durations in
/// `[2^k, 2^(k+1))` nanoseconds. Bucket 0 covers `[0, 2)`; bucket 31
/// covers `[2^31, 2^32) ns ≈ [2.1 s, 4.3 s)`. Anything larger lands in
/// the overflow bucket.
const NUM_BUCKETS: usize = 32;

/// Lock-free latency distribution recorded in nanoseconds.
///
/// `record(dur_ns)` is wait-free (a handful of `fetch_add` and a CAS-max
/// loop). `snapshot()` walks the buckets non-atomically — values may be
/// slightly inconsistent under concurrent writes, which is fine for an
/// observability tool.
#[derive(Debug)]
pub struct LatencyHistogram {
    buckets: [AtomicU64; NUM_BUCKETS],
    overflow: AtomicU64,
    count: AtomicU64,
    sum_ns: AtomicU64,
    max_ns: AtomicU64,
}

impl LatencyHistogram {
    pub const fn new() -> Self {
        // Can't `[AtomicU64::new(0); N]` because `AtomicU64` isn't `Copy`.
        // Spell out the array — 32 entries is annoying but const.
        Self {
            buckets: [
                AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
                AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
                AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
                AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
                AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
                AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
                AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
                AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
            ],
            overflow: AtomicU64::new(0),
            count: AtomicU64::new(0),
            sum_ns: AtomicU64::new(0),
            max_ns: AtomicU64::new(0),
        }
    }

    pub fn record(&self, dur_ns: u64) {
        let idx = bucket_of(dur_ns);
        if idx >= NUM_BUCKETS {
            self.overflow.fetch_add(1, Ordering::Relaxed);
        } else {
            self.buckets[idx].fetch_add(1, Ordering::Relaxed);
        }
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum_ns.fetch_add(dur_ns, Ordering::Relaxed);

        let mut cur = self.max_ns.load(Ordering::Relaxed);
        while dur_ns > cur {
            match self.max_ns.compare_exchange_weak(
                cur, dur_ns, Ordering::Relaxed, Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(v) => cur = v,
            }
        }
    }

    pub fn snapshot(&self) -> LatencySnapshot {
        let count = self.count.load(Ordering::Relaxed);
        let sum = self.sum_ns.load(Ordering::Relaxed);
        let max = self.max_ns.load(Ordering::Relaxed);
        let overflow = self.overflow.load(Ordering::Relaxed);

        let mut bucket_counts = [0u64; NUM_BUCKETS];
        for (i, b) in self.buckets.iter().enumerate() {
            bucket_counts[i] = b.load(Ordering::Relaxed);
        }

        let mean_ns = if count > 0 { sum / count } else { 0 };
        // Bucket-edge estimates can overshoot the actual max when many
        // samples cluster in one log2 bucket. Clamp so p99 <= max.
        let cap = if max > 0 { max } else { u64::MAX };
        let p50 = percentile_ns(&bucket_counts, overflow, count, 50).min(cap);
        let p95 = percentile_ns(&bucket_counts, overflow, count, 95).min(cap);
        let p99 = percentile_ns(&bucket_counts, overflow, count, 99).min(cap);

        LatencySnapshot { count, mean_ns, max_ns: max, p50_ns: p50, p95_ns: p95, p99_ns: p99 }
    }
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatencySnapshot {
    pub count: u64,
    pub mean_ns: u64,
    pub max_ns: u64,
    pub p50_ns: u64,
    pub p95_ns: u64,
    pub p99_ns: u64,
}

fn bucket_of(ns: u64) -> usize {
    if ns < 2 {
        return 0;
    }
    // floor(log2(ns)) via leading_zeros. ns >= 2 so this is in [1, 63].
    (63 - ns.leading_zeros()) as usize
}

/// Estimate the percentile boundary as the *upper* edge of the bucket
/// containing the rank. Coarse (factor-of-2 resolution) but enough to
/// catch regressions.
fn percentile_ns(buckets: &[u64; NUM_BUCKETS], overflow: u64, count: u64, pct: u8) -> u64 {
    if count == 0 {
        return 0;
    }
    let target = (count.saturating_mul(pct as u64) + 99) / 100;
    let mut acc: u64 = 0;
    for (i, &c) in buckets.iter().enumerate() {
        acc = acc.saturating_add(c);
        if acc >= target {
            // Upper edge of bucket i is 2^(i+1) ns. Cap at u64::MAX for
            // the top bucket to avoid overflow.
            return 1u64.checked_shl((i + 1) as u32).unwrap_or(u64::MAX);
        }
    }
    if overflow > 0 {
        u64::MAX
    } else {
        0
    }
}

/// Monotonic nanoseconds since a process-wide epoch. Both source-side
/// stamp and sink-side measurement must call this for the delta to be
/// meaningful. Std-only because it relies on `std::time::Instant`.
#[cfg(feature = "std")]
mod std_clock {
    extern crate std;
    use std::sync::OnceLock;
    use std::time::Instant;

    pub fn monotonic_ns() -> u64 {
        static EPOCH: OnceLock<Instant> = OnceLock::new();
        let epoch = EPOCH.get_or_init(Instant::now);
        // u64 ns holds ~584 years, comfortably more than process uptime.
        epoch.elapsed().as_nanos() as u64
    }
}

#[cfg(feature = "std")]
pub use std_clock::monotonic_ns;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_of_handles_edges() {
        assert_eq!(bucket_of(0), 0);
        assert_eq!(bucket_of(1), 0);
        assert_eq!(bucket_of(2), 1);
        assert_eq!(bucket_of(3), 1);
        assert_eq!(bucket_of(4), 2);
        assert_eq!(bucket_of(1_000_000), 19); // ~1ms -> 2^19 = 524288, 2^20 = 1048576
    }

    #[test]
    fn histogram_records_and_snapshots() {
        let h = LatencyHistogram::new();
        for _ in 0..100 {
            h.record(1_000_000); // 1ms
        }
        for _ in 0..5 {
            h.record(50_000_000); // 50ms
        }
        let s = h.snapshot();
        assert_eq!(s.count, 105);
        assert!(s.max_ns >= 50_000_000);
        // p50 of 100x1ms + 5x50ms should be in the ~1ms bucket.
        assert!(s.p50_ns >= 1_000_000 && s.p50_ns < 4_000_000, "p50 = {}", s.p50_ns);
        // p99 should land in the 50ms bucket range.
        assert!(s.p99_ns >= 32_000_000, "p99 = {}", s.p99_ns);
    }

    #[test]
    fn histogram_overflow_bucket() {
        let h = LatencyHistogram::new();
        h.record(u64::MAX / 2);
        let s = h.snapshot();
        assert_eq!(s.count, 1);
        assert!(s.max_ns >= u64::MAX / 2);
    }

    #[cfg(feature = "std")]
    #[test]
    fn monotonic_ns_is_nondecreasing() {
        let a = monotonic_ns();
        let b = monotonic_ns();
        assert!(b >= a);
    }
}
