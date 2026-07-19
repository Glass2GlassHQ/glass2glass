//! Measured per-element runtime instrumentation (M399).
//!
//! `RunStats::report()` (M287) folds each element's *declared* `latency()`. This
//! module adds the *measured* counterpart: each instrumented arm holds an
//! `Arc<ElementProbe>`, times the wall-clock cost of every `DataFrame`
//! `process()` call, and samples its input link's fill at each pull. After the
//! run the runner snapshots every probe into [`RunStats::per_element`], turning
//! the by-hand glass-to-glass analyses (the NVDEC-floor / `link_capacity`
//! studies) into a number the runner prints.
//!
//! `std`-gated where it counts: measured timing needs a real monotonic clock
//! ([`monotonic_ns`](crate::metrics::monotonic_ns), `std`-only). Under `no_std`
//! the timing calls compile to no-ops and the histogram stays empty (fill
//! sampling still works, it needs no clock); [`RunStats::per_element`] is then
//! whatever the arms recorded, typically empty.

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use portable_atomic::{AtomicU64, Ordering};

use crate::metrics::{LatencyHistogram, LatencySnapshot};

/// Per-element measured telemetry collected over a run, shared between an arm
/// (the writer) and the runner (which snapshots it once every arm has joined).
/// Writes are wait-free (the histogram is lock-free; the fill gauge is three
/// `fetch_add`/CAS), so an arm pays almost nothing on the hot path.
#[derive(Debug)]
pub struct ElementProbe {
    name: String,
    /// Wall-clock cost of each `DataFrame` `process()` call, in nanoseconds.
    proc_ns: LatencyHistogram,
    /// Queue-residency (transit) time of each `DataFrame` on this element's input
    /// link: how long the frame sat queued between the producer sending it and
    /// this element pulling it. The per-stage "wait" half of a latency waterfall
    /// (the `process()` cost is the "work" half). Empty on an uninstrumented edge
    /// or under `no_std`.
    transit_ns: LatencyHistogram,
    /// Input-link occupancy sampled at each pull (0-100), an indicator of where
    /// backpressure pools: a consistently-full input means this element is the
    /// bottleneck; a consistently-empty one means it is starved.
    fill: FillGauge,
}

impl ElementProbe {
    pub fn new(name: String) -> Arc<Self> {
        Arc::new(Self {
            name,
            proc_ns: LatencyHistogram::new(),
            transit_ns: LatencyHistogram::new(),
            fill: FillGauge::default(),
        })
    }

    /// A monotonic start stamp for the about-to-run `process()`, or `None` under
    /// `no_std` (no clock). Pair with [`record_proc_since`](Self::record_proc_since).
    #[inline]
    pub fn mark() -> Option<u64> {
        #[cfg(feature = "std")]
        {
            Some(crate::metrics::monotonic_ns())
        }
        #[cfg(not(feature = "std"))]
        {
            None
        }
    }

    /// Record the elapsed `process()` cost since `start` (from [`mark`](Self::mark)).
    /// A no-op under `no_std` or when `start` is `None`.
    #[inline]
    pub fn record_proc_since(&self, start: Option<u64>) {
        #[cfg(feature = "std")]
        if let Some(t0) = start {
            let now = crate::metrics::monotonic_ns();
            self.proc_ns.record(now.saturating_sub(t0));
        }
        #[cfg(not(feature = "std"))]
        let _ = start;
    }

    /// Sample the element's input-link fill (0-100) for this pull.
    #[inline]
    pub fn record_fill(&self, pct: u8) {
        self.fill.record(pct);
    }

    /// Record the queue-residency time (ns) of a `DataFrame` pulled off the input
    /// link (from [`LinkReceiver::pop_transit_ns`](crate::runtime::LinkReceiver::pop_transit_ns)).
    #[inline]
    pub fn record_transit(&self, ns: u64) {
        self.transit_ns.record(ns);
    }

    pub fn snapshot(&self) -> ElementLatency {
        ElementLatency {
            name: self.name.clone(),
            proc: self.proc_ns.snapshot(),
            transit: self.transit_ns.snapshot(),
            fill_mean_pct: self.fill.mean(),
            fill_max_pct: self.fill.max(),
        }
    }
}

/// A tiny lock-free mean/max accumulator for input-link fill percent. A full
/// log2 histogram is overkill for a 0-100 gauge, so this keeps just the running
/// sum, count, and max, mirroring the wait-free style of [`LatencyHistogram`].
#[derive(Debug, Default)]
struct FillGauge {
    sum: AtomicU64,
    count: AtomicU64,
    max: AtomicU64,
}

impl FillGauge {
    fn record(&self, pct: u8) {
        self.sum.fetch_add(pct as u64, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        let pct = pct as u64;
        let mut cur = self.max.load(Ordering::Relaxed);
        while pct > cur {
            match self
                .max
                .compare_exchange_weak(cur, pct, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => break,
                Err(v) => cur = v,
            }
        }
    }

    fn mean(&self) -> u8 {
        let count = self.count.load(Ordering::Relaxed);
        let sum = self.sum.load(Ordering::Relaxed);
        sum.checked_div(count).unwrap_or(0) as u8
    }

    fn max(&self) -> u8 {
        self.max.load(Ordering::Relaxed) as u8
    }
}

/// A measured per-element summary, one row of [`RunStats::per_element`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElementLatency {
    /// Instance name (`<category>N` from the graph runner, the element's log
    /// category for the linear runners).
    pub name: String,
    /// Measured `process()` latency distribution (count, p50/p95/p99, mean, max
    /// in ns). `count == 0` under `no_std` (no clock to measure with).
    pub proc: LatencySnapshot,
    /// Input-link queue-residency (transit) distribution: how long each
    /// `DataFrame` waited queued before this element pulled it. `count == 0` when
    /// the edge is not instrumented (only the graph runner enables it, on edges
    /// into transform/sink nodes) or under `no_std`.
    pub transit: LatencySnapshot,
    /// Mean input-link fill percent observed across the run (0-100).
    pub fill_mean_pct: u8,
    /// Peak input-link fill percent (0-100); 100 means the element's input was
    /// saturated at least once, i.e. it back-pressured its upstream.
    pub fill_max_pct: u8,
}

/// A nullable probe handle threaded into an arm. Cloning shares the underlying
/// [`ElementProbe`] (via `Arc`); `None` means the arm is not instrumented.
pub type Probe = Option<Arc<ElementProbe>>;

/// Snapshot a collection of optional probes into report rows, dropping the
/// un-instrumented (`None`) slots. Order is preserved (topological).
pub fn snapshot_all(probes: &[Probe]) -> Vec<ElementLatency> {
    probes
        .iter()
        .filter_map(|p| p.as_ref().map(|p| p.snapshot()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    // `g2g-core` is `#![no_std]`; the `std`-gated test below needs `std` in scope.
    #[cfg(feature = "std")]
    extern crate std;

    #[test]
    fn fill_gauge_tracks_mean_and_max() {
        let g = FillGauge::default();
        g.record(10);
        g.record(20);
        g.record(90);
        assert_eq!(g.mean(), 40, "(10+20+90)/3 = 40");
        assert_eq!(g.max(), 90);
    }

    #[test]
    fn fill_gauge_empty_is_zero() {
        let g = FillGauge::default();
        assert_eq!(g.mean(), 0);
        assert_eq!(g.max(), 0);
    }

    #[cfg(feature = "std")]
    #[test]
    fn probe_records_process_latency() {
        let p = ElementProbe::new(String::from("slowelem0"));
        // Record a few deliberate sleeps so the snapshot has a real distribution.
        for _ in 0..8 {
            let t0 = ElementProbe::mark();
            std::thread::sleep(std::time::Duration::from_millis(2));
            p.record_proc_since(t0);
        }
        p.record_fill(75);
        p.record_fill(100);
        let s = p.snapshot();
        assert_eq!(s.name, "slowelem0");
        assert_eq!(s.proc.count, 8);
        // ~2 ms sleeps land at or above the 1ms bucket; allow scheduler slop.
        assert!(s.proc.p50_ns >= 1_000_000, "p50 = {} ns", s.proc.p50_ns);
        assert_eq!(s.fill_max_pct, 100);
        assert!(s.fill_mean_pct > 0);
    }

    #[test]
    fn snapshot_all_skips_none() {
        let probes: Vec<Probe> =
            alloc::vec![None, Some(ElementProbe::new(String::from("a"))), None];
        let rows = snapshot_all(&probes);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].name, "a");
    }
}
