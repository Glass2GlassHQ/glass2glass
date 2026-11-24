// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! Lightweight metrics primitives for glass-to-glass observability.
//!
//! This module is `no_std + alloc` compatible. The histogram uses log-linear
//! buckets in nanoseconds and an array of atomic counters, so it can be
//! shared across threads via `Arc<LatencyHistogram>` without locking.
//!
//! Under feature `std`, a process-wide `monotonic_ns()` helper exposes a
//! single shared epoch so a source-side stamp and a sink-side `now()`
//! agree on what "zero" means.

use portable_atomic::{AtomicU64, Ordering};

/// Each power of two splits into `SUB_BUCKETS` equal buckets, so a bucket's
/// upper edge is at most 1/16 above any value in it. Durations from
/// 2^32 ns (about 4.3 s) up land in the overflow bucket.
const SUB_BUCKET_BITS: u32 = 4;
const SUB_BUCKETS: usize = 1 << SUB_BUCKET_BITS;
const NUM_BUCKETS: usize = SUB_BUCKETS + (32 - SUB_BUCKET_BITS as usize) * SUB_BUCKETS;

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
        Self {
            buckets: [const { AtomicU64::new(0) }; NUM_BUCKETS],
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
                cur,
                dur_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
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

        let mean_ns = sum.checked_div(count).unwrap_or(0);
        // Bucket-edge estimates can overshoot the actual max when many
        // samples cluster in one bucket. Clamp so p99 <= max.
        let cap = if max > 0 { max } else { u64::MAX };
        let p50 = percentile_ns(&bucket_counts, overflow, count, 50).min(cap);
        let p95 = percentile_ns(&bucket_counts, overflow, count, 95).min(cap);
        let p99 = percentile_ns(&bucket_counts, overflow, count, 99).min(cap);

        LatencySnapshot {
            count,
            mean_ns,
            max_ns: max,
            p50_ns: p50,
            p95_ns: p95,
            p99_ns: p99,
        }
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
    if ns < SUB_BUCKETS as u64 {
        return ns as usize;
    }
    let shift = 63 - ns.leading_zeros() - SUB_BUCKET_BITS;
    let sub_bucket = (ns >> shift) as usize - SUB_BUCKETS;
    (shift as usize + 1) * SUB_BUCKETS + sub_bucket
}

fn bucket_upper_edge_ns(index: usize) -> u64 {
    if index < SUB_BUCKETS {
        return index as u64 + 1;
    }
    let shift = index / SUB_BUCKETS - 1;
    let sub_bucket = (index % SUB_BUCKETS) as u64;
    (SUB_BUCKETS as u64 + sub_bucket + 1) << shift
}

/// Estimate the percentile boundary as the *upper* edge of the bucket
/// containing the rank, at most 1/16 above the true value.
fn percentile_ns(buckets: &[u64; NUM_BUCKETS], overflow: u64, count: u64, pct: u8) -> u64 {
    if count == 0 {
        return 0;
    }
    let target = count.saturating_mul(pct as u64).div_ceil(100);
    let mut acc: u64 = 0;
    for (i, &c) in buckets.iter().enumerate() {
        acc = acc.saturating_add(c);
        if acc >= target {
            return bucket_upper_edge_ns(i);
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
    // Native `std::time::Instant`; on wasm32 that `now()` panics ("time not
    // implemented"), so use web-time's drop-in backed by `performance.now()`.
    #[cfg(not(target_arch = "wasm32"))]
    use std::time::Instant;
    #[cfg(target_arch = "wasm32")]
    use web_time::Instant;

    pub fn monotonic_ns() -> u64 {
        static EPOCH: OnceLock<Instant> = OnceLock::new();
        let epoch = EPOCH.get_or_init(Instant::now);
        // u64 ns holds ~584 years, comfortably more than process uptime.
        epoch.elapsed().as_nanos() as u64
    }

    pub fn unix_time_ns() -> i64 {
        #[cfg(not(target_arch = "wasm32"))]
        use std::time::{SystemTime, UNIX_EPOCH};
        #[cfg(target_arch = "wasm32")]
        use web_time::{SystemTime, UNIX_EPOCH};
        match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(after) => i64::try_from(after.as_nanos()).unwrap_or(i64::MAX),
            Err(before) => i64::try_from(before.duration().as_nanos()).map_or(i64::MIN, |n| -n),
        }
    }
}

#[cfg(feature = "std")]
pub use std_clock::{monotonic_ns, unix_time_ns};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_of_handles_edges() {
        assert_eq!(bucket_of(0), 0);
        assert_eq!(bucket_of(15), 15);
        assert_eq!(bucket_of(16), 16);
        assert_eq!(bucket_of(31), 31);
        assert_eq!(bucket_of(32), 32);
        assert_eq!(bucket_of(33), 32);
        assert_eq!(bucket_of(34), 33);
        assert_eq!(bucket_of((1 << 32) - 1), NUM_BUCKETS - 1);
        assert_eq!(bucket_of(1 << 32), NUM_BUCKETS);
        for ns in [0, 15, 16, 33, 1_000_000, 36_000_000, (1 << 32) - 1] {
            let upper = bucket_upper_edge_ns(bucket_of(ns));
            assert!(ns < upper && upper - ns <= ns / 16 + 1, "{ns} -> {upper}");
        }
    }

    #[test]
    fn percentiles_resolve_milliseconds() {
        let h = LatencyHistogram::new();
        // one sample every 10 us from 30 ms to 40 ms
        let sample_ns = |i: u64| 30_000_000 + i * 10_000;
        for i in 0..1000 {
            h.record(sample_ns(i));
        }
        let s = h.snapshot();
        for (reported, true_ns) in [(s.p50_ns, sample_ns(499)), (s.p99_ns, sample_ns(989))] {
            assert!(
                reported >= true_ns && reported <= true_ns + true_ns / 16,
                "reported {reported} ns for {true_ns} ns"
            );
        }
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
        assert!(
            s.p50_ns >= 1_000_000 && s.p50_ns < 4_000_000,
            "p50 = {}",
            s.p50_ns
        );
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
