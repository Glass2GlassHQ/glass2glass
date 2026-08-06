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

use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use portable_atomic::{AtomicU64, Ordering};
use spin::Mutex;

use crate::element::PresentationStats;
use crate::metrics::{LatencyHistogram, LatencySnapshot};

/// How many recent frame visits a journey-recording probe keeps. Bounded and
/// preallocated, so recording never allocates and a long run never grows.
const JOURNEY_RING: usize = 64;

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
    /// M851: a bounded ring of recent per-frame visits, keyed by the frame's
    /// sequence id, so one frame's path across stages can be joined at snapshot
    /// time. `None` unless an observer is attached (the aggregate histograms
    /// above serve the end-of-run report on their own).
    journeys: Option<Mutex<VecDeque<StageVisit>>>,
    /// A paced sink's cumulative presented / dropped counters, stored once by
    /// the sink arm as it ends (from
    /// [`presentation_stats`](crate::AsyncElement::presentation_stats)).
    presentation: Mutex<Option<PresentationStats>>,
}

impl ElementProbe {
    pub fn new(name: String) -> Arc<Self> {
        Arc::new(Self {
            name,
            proc_ns: LatencyHistogram::new(),
            transit_ns: LatencyHistogram::new(),
            fill: FillGauge::default(),
            journeys: None,
            presentation: Mutex::new(None),
        })
    }

    /// As [`new`](Self::new), but also recording a bounded ring of per-frame
    /// visits for the single-frame waterfall. Only the observed graph runner
    /// mints these, so an untapped run keeps the cheaper probe.
    pub fn with_journeys(name: String) -> Arc<Self> {
        Arc::new(Self {
            name,
            proc_ns: LatencyHistogram::new(),
            transit_ns: LatencyHistogram::new(),
            fill: FillGauge::default(),
            journeys: Some(Mutex::new(VecDeque::with_capacity(JOURNEY_RING))),
            presentation: Mutex::new(None),
        })
    }

    /// The instance name of the element this probe measures.
    pub fn name(&self) -> &str {
        &self.name
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

    /// Record this element's visit by one frame: its `sequence` id, the
    /// `wait_ns` it spent queued on the input link, and the `enter` stamp from
    /// [`mark`](Self::mark) taken just before `process()` (exit is stamped here).
    /// A no-op when the probe records no journeys, under `no_std`, or when
    /// `enter` is `None`.
    #[inline]
    pub fn record_visit(&self, sequence: u64, wait_ns: u64, enter: Option<u64>) {
        #[cfg(feature = "std")]
        if let (Some(ring), Some(enter_ns)) = (self.journeys.as_ref(), enter) {
            let exit_ns = crate::metrics::monotonic_ns();
            let mut ring = ring.lock();
            if ring.len() == JOURNEY_RING {
                ring.pop_front();
            }
            ring.push_back(StageVisit {
                sequence,
                wait_ns,
                enter_ns,
                exit_ns,
            });
        }
        #[cfg(not(feature = "std"))]
        let _ = (sequence, wait_ns, enter);
    }

    /// Push a fully-stamped visit, so a test can build a deterministic journey
    /// instead of racing a real clock. Its only callers are `observe`'s tests,
    /// and that module is std-gated.
    #[cfg(all(test, feature = "std"))]
    pub(crate) fn push_visit(&self, visit: StageVisit) {
        if let Some(ring) = self.journeys.as_ref() {
            let mut ring = ring.lock();
            if ring.len() == JOURNEY_RING {
                ring.pop_front();
            }
            ring.push_back(visit);
        }
    }

    /// The recent frame visits this probe kept, oldest first. Empty unless the
    /// probe was minted with [`with_journeys`](Self::with_journeys).
    pub fn visits(&self) -> Vec<StageVisit> {
        match &self.journeys {
            Some(ring) => ring.lock().iter().copied().collect(),
            None => Vec::new(),
        }
    }

    /// Store a paced sink's end-of-run presentation counters. Called by the
    /// sink arm as it ends, so `snapshot` (taken after every arm joined) sees a
    /// settled value.
    pub fn set_presentation(&self, stats: PresentationStats) {
        *self.presentation.lock() = Some(stats);
    }

    pub fn snapshot(&self) -> ElementLatency {
        ElementLatency {
            name: self.name.clone(),
            proc: self.proc_ns.snapshot(),
            transit: self.transit_ns.snapshot(),
            fill_mean_pct: self.fill.mean(),
            fill_max_pct: self.fill.max(),
            presentation: *self.presentation.lock(),
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

/// One frame's passage through one element: the wait it served on the input
/// link plus the wall-clock window of the `process()` call that consumed it.
/// The per-frame counterpart of the aggregated `transit` / `proc` histograms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageVisit {
    /// The frame's [`Frame::sequence`](crate::Frame). Sources stamp it and
    /// 1-in-1-out transforms carry it through, which is what lets stages be
    /// joined; an element that restamps (a decoder, a parser) breaks the join
    /// rather than shifting it, and the assembled journey simply stops there.
    pub sequence: u64,
    /// Queue residency on this element's input link before the pull, `0` on an
    /// uninstrumented edge.
    pub wait_ns: u64,
    /// Monotonic stamp taken immediately before `process()`.
    pub enter_ns: u64,
    /// Monotonic stamp taken immediately after `process()` returned.
    pub exit_ns: u64,
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
    /// A paced sink's cumulative presented / dropped counters, `None` for
    /// elements that don't present.
    pub presentation: Option<PresentationStats>,
}

/// Per-edge live traffic counters (M846), shared between a link's
/// [`SenderSink`](crate::runtime::SenderSink) (the writer) and the observer tap
/// (the reader). Writes are wait-free (three relaxed `fetch_add`s), so an
/// instrumented link pays a few atomics per packet.
///
/// The end-of-run [`RunStats::frames_dropped`](crate::runtime::RunStats) folds
/// every leaky link's drops into one number; these counters keep the same events
/// per edge and readable while the run is still going.
#[derive(Debug, Default)]
pub struct EdgeCounters {
    packets: AtomicU64,
    bytes: AtomicU64,
    drops: AtomicU64,
    blocked_ns: AtomicU64,
}

impl EdgeCounters {
    /// Record one packet that entered the link, carrying `bytes` of payload,
    /// after `blocked_ns` awaiting capacity.
    #[inline]
    pub(crate) fn record_packet(&self, bytes: u64, blocked_ns: u64) {
        self.packets.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(bytes, Ordering::Relaxed);
        if blocked_ns > 0 {
            self.blocked_ns.fetch_add(blocked_ns, Ordering::Relaxed);
        }
    }

    /// Record one frame this link dropped (leaky policy, full channel).
    #[inline]
    pub(crate) fn record_drop(&self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> EdgeCounts {
        EdgeCounts {
            packets: self.packets.load(Ordering::Relaxed),
            bytes: self.bytes.load(Ordering::Relaxed),
            drops: self.drops.load(Ordering::Relaxed),
            blocked_ns: self.blocked_ns.load(Ordering::Relaxed),
        }
    }
}

/// A read of one edge's [`EdgeCounters`], carried on
/// [`EdgeInfo`](crate::runtime::EdgeInfo). All-zero on an uninstrumented edge
/// (no observer attached).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EdgeCounts {
    /// Packets (data + control) that entered the link.
    pub packets: u64,
    /// Payload bytes carried by those packets. Counts CPU-resident buffers only:
    /// a device-domain frame (CUDA / texture handle) has no bytes crossing here,
    /// so it adds `0`.
    pub bytes: u64,
    /// Frames this link dropped under a leaky
    /// [`LinkPolicy`](crate::link::LinkPolicy).
    pub drops: u64,
    /// Nanoseconds the producer spent awaiting capacity on this link. On a
    /// source's outgoing edge this is the one per-frame cost an outside observer
    /// can attribute to the source honestly: how long downstream backpressure
    /// held it up. A source's own pacing (waiting for the next captured frame)
    /// happens inside its `run` loop and is indistinguishable from work there.
    pub blocked_ns: u64,
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
    fn plain_probe_records_no_visits() {
        // The un-observed mint: recording is a no-op, so an untapped run keeps
        // nothing per frame.
        let p = ElementProbe::new(String::from("x0"));
        for i in 0..4 {
            p.record_visit(i, 10, Some(100 + i));
        }
        assert!(p.visits().is_empty());
    }

    #[cfg(feature = "std")]
    #[test]
    fn journey_ring_keeps_the_newest_and_stays_bounded() {
        let p = ElementProbe::with_journeys(String::from("x0"));
        for i in 0..(JOURNEY_RING as u64 * 2) {
            p.record_visit(i, i, Some(1_000 + i));
        }
        let v = p.visits();
        assert_eq!(v.len(), JOURNEY_RING, "ring is bounded");
        assert_eq!(v[0].sequence, JOURNEY_RING as u64, "oldest evicted");
        assert_eq!(v[v.len() - 1].sequence, JOURNEY_RING as u64 * 2 - 1);
        let last = v[v.len() - 1];
        assert_eq!(last.enter_ns, 1_000 + last.sequence);
        assert!(last.exit_ns >= last.enter_ns, "exit is stamped after enter");
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
