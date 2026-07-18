//! Throughput-driven adaptive bitrate (ABR) support (M371), shared by
//! [`HlsSrc`](crate::hlssrc) and [`DashSrc`](crate::dashsrc).
//!
//! A source times each segment download (bytes over elapsed time) and feeds it
//! to a [`BandwidthEstimator`], which keeps an exponentially-weighted moving
//! average of measured throughput. Before fetching the next segment the source
//! asks for an [`effective_cap`](BandwidthEstimator::effective_cap): the smoothed
//! estimate scaled by a safety factor, further bounded by the user's
//! `max-bandwidth`. That cap drives the *existing* variant / Representation
//! selection (`MasterPlaylist::select` / `Mpd::select`), so ABR reduces to
//! "re-pick the rendition whose declared bitrate fits the measured bandwidth"
//! and a switch is just swapping the active media playlist / Representation and
//! re-emitting the init segment.
//!
//! The estimator is pure (`no_std + alloc`, no clock of its own): the caller
//! supplies the elapsed time, measured with `g2g_core::metrics::monotonic_ns`.

/// EWMA weight on the newest sample: reactive enough to follow a real bandwidth
/// drop within a couple of segments without thrashing on one blip.
const ALPHA: f64 = 0.5;

/// Usable fraction of the estimate: leave headroom for variance so a rendition
/// selected at the estimate does not immediately rebuffer.
const SAFETY: f64 = 0.8;

/// An exponentially-weighted moving average of measured download throughput, in
/// bits per second, plus the selection rule that turns it into a bandwidth cap.
#[derive(Debug, Clone)]
pub(crate) struct BandwidthEstimator {
    /// Smoothed estimate in bits/sec; `None` until the first usable sample.
    estimate_bps: Option<f64>,
}

impl BandwidthEstimator {
    /// A fresh estimator with no samples.
    pub(crate) fn new() -> Self {
        Self { estimate_bps: None }
    }

    /// Record a completed download of `bytes` that took `elapsed_ns`. A zero (or
    /// empty) measurement carries no rate, so it is ignored rather than skewing
    /// the average (e.g. a cache hit returning instantly).
    pub(crate) fn sample(&mut self, bytes: usize, elapsed_ns: u64) {
        if elapsed_ns == 0 || bytes == 0 {
            return;
        }
        let bps = (bytes as f64) * 8.0 * 1.0e9 / (elapsed_ns as f64);
        self.estimate_bps = Some(match self.estimate_bps {
            Some(prev) => ALPHA * bps + (1.0 - ALPHA) * prev,
            None => bps,
        });
    }

    /// The bandwidth cap to feed variant / Representation selection: the estimate
    /// scaled by the safety factor, then bounded by the user's `user_cap`
    /// (`0` = no user cap). `None` until the first sample, so the initial pick
    /// uses only the user cap (the prior fixed-variant behaviour).
    pub(crate) fn effective_cap(&self, user_cap: u64) -> Option<u64> {
        let est = self.estimate_bps?;
        let abr_cap = (est * SAFETY).max(0.0) as u64;
        Some(match user_cap {
            0 => abr_cap,
            c => abr_cap.min(c),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_sample_means_no_cap_so_initial_pick_uses_user_cap() {
        let est = BandwidthEstimator::new();
        assert_eq!(
            est.effective_cap(0),
            None,
            "no estimate yet -> caller uses its own cap"
        );
        assert_eq!(est.effective_cap(5_000_000), None);
    }

    #[test]
    fn first_sample_caps_at_the_measured_rate_times_safety() {
        let mut est = BandwidthEstimator::new();
        // 1_000_000 bytes in 1s = 8 Mbit/s; safety 0.8 -> 6.4 Mbit/s cap.
        est.sample(1_000_000, 1_000_000_000);
        assert_eq!(est.effective_cap(0), Some(6_400_000));
    }

    #[test]
    fn ewma_smooths_toward_a_dropped_rate() {
        let mut est = BandwidthEstimator::new();
        est.sample(10_000_000, 1_000_000_000); // 80 Mbit/s
                                               // Bandwidth collapses to 8 Mbit/s; the EWMA (alpha 0.5) moves halfway to
                                               // 44 Mbit/s, so the cap is 0.8 * 44 = 35.2 Mbit/s.
        est.sample(1_000_000, 1_000_000_000);
        assert_eq!(est.effective_cap(0), Some(35_200_000));
    }

    #[test]
    fn effective_cap_is_bounded_by_the_user_cap() {
        let mut est = BandwidthEstimator::new();
        est.sample(10_000_000, 1_000_000_000); // 80 Mbit/s -> 64 Mbit/s after safety
                                               // User caps at 5 Mbit/s: the smaller wins.
        assert_eq!(est.effective_cap(5_000_000), Some(5_000_000));
        // No user cap: the ABR cap stands.
        assert_eq!(est.effective_cap(0), Some(64_000_000));
    }

    #[test]
    fn zero_duration_or_empty_sample_is_ignored() {
        let mut est = BandwidthEstimator::new();
        est.sample(0, 1_000_000_000);
        est.sample(1_000_000, 0);
        assert_eq!(est.effective_cap(0), None, "no usable sample recorded");
    }
}
