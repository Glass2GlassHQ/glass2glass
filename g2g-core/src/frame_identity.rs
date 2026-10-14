const RECENT_IDENTITIES: usize = 64;
const CADENCE_SAMPLES: usize = 16;
const REPORT_INTERVAL_NS: i64 = 1_000_000_000;
// a step of at least one and a half frame periods means frames went missing
const GAP_STEP_HALF_PERIODS: u64 = 3;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct FrameIdentityStats {
    pub frames: u64,
    pub lost: u64,
    pub duplicates: u64,
    pub capture_unix_ns: i64,
    pub latency_ns: i64,
    pub latency_min_ns: i64,
    pub latency_max_ns: i64,
    pub latency_mean_ns: i64,
}

#[derive(Debug, Clone, Copy)]
struct Ring<T: Copy + Default, const N: usize> {
    items: [T; N],
    len: usize,
    next: usize,
}

impl<T: Copy + Default + PartialEq, const N: usize> Ring<T, N> {
    fn new() -> Self {
        Self {
            items: [T::default(); N],
            len: 0,
            next: 0,
        }
    }

    fn push(&mut self, item: T) {
        self.items[self.next] = item;
        self.next = (self.next + 1) % N;
        self.len = (self.len + 1).min(N);
    }

    fn filled(&self) -> &[T] {
        &self.items[..self.len]
    }

    fn clear(&mut self) {
        self.len = 0;
        self.next = 0;
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct LatencyWindow {
    frames: u64,
    sum_ns: i128,
    min_ns: i64,
    max_ns: i64,
}

impl LatencyWindow {
    fn add(&mut self, latency_ns: i64) {
        if self.frames == 0 {
            self.min_ns = latency_ns;
            self.max_ns = latency_ns;
        }
        self.frames += 1;
        self.sum_ns += i128::from(latency_ns);
        self.min_ns = self.min_ns.min(latency_ns);
        self.max_ns = self.max_ns.max(latency_ns);
    }

    fn mean_ns(&self) -> i64 {
        (self.sum_ns / i128::from(self.frames.max(1))) as i64
    }
}

#[derive(Debug)]
pub struct FrameIdentityTracker {
    recent: Ring<i64, RECENT_IDENTITIES>,
    steps: Ring<u64, CADENCE_SAMPLES>,
    newest: Option<i64>,
    stats: FrameIdentityStats,
    window: LatencyWindow,
    last_report_unix_ns: Option<i64>,
}

impl Default for FrameIdentityTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameIdentityTracker {
    pub fn new() -> Self {
        Self {
            recent: Ring::new(),
            steps: Ring::new(),
            newest: None,
            stats: FrameIdentityStats::default(),
            window: LatencyWindow::default(),
            last_report_unix_ns: None,
        }
    }

    pub fn stats(&self) -> FrameIdentityStats {
        self.stats
    }

    pub fn observe(
        &mut self,
        capture_unix_ns: i64,
        present_unix_ns: i64,
    ) -> Option<FrameIdentityStats> {
        if self.recent.filled().contains(&capture_unix_ns) {
            self.stats.duplicates += 1;
            return None;
        }
        self.recent.push(capture_unix_ns);
        self.stats.frames += 1;
        match self.newest {
            Some(newest) if capture_unix_ns > newest => {
                let step = capture_unix_ns.abs_diff(newest);
                self.stats.lost = self.stats.lost.saturating_add(self.missing_before(step));
                self.steps.push(step);
                self.newest = Some(capture_unix_ns);
            }
            // a late frame fills the gap its absence opened
            Some(_) => self.stats.lost = self.stats.lost.saturating_sub(1),
            None => self.newest = Some(capture_unix_ns),
        }
        let latency_ns = present_unix_ns.saturating_sub(capture_unix_ns);
        self.stats.capture_unix_ns = capture_unix_ns;
        self.stats.latency_ns = latency_ns;
        self.window.add(latency_ns);
        let Some(last_report) = self.last_report_unix_ns else {
            self.last_report_unix_ns = Some(present_unix_ns);
            return None;
        };
        if present_unix_ns.saturating_sub(last_report) < REPORT_INTERVAL_NS {
            return None;
        }
        self.last_report_unix_ns = Some(present_unix_ns);
        Some(self.take_report())
    }

    pub fn finish(&mut self) -> Option<FrameIdentityStats> {
        (self.window.frames > 0).then(|| self.take_report())
    }

    pub fn restart(&mut self) {
        self.recent.clear();
        self.newest = None;
    }

    fn missing_before(&self, step: u64) -> u64 {
        let Some(period) = self.frame_period_ns() else {
            return 0;
        };
        if step.saturating_mul(2) < period.saturating_mul(GAP_STEP_HALF_PERIODS) {
            return 0;
        }
        (step.saturating_add(period / 2) / period).saturating_sub(1)
    }

    fn frame_period_ns(&self) -> Option<u64> {
        let filled = self.steps.filled();
        let mut sorted = [0u64; CADENCE_SAMPLES];
        let sorted = &mut sorted[..filled.len()];
        sorted.copy_from_slice(filled);
        sorted.sort_unstable();
        sorted.get(sorted.len() / 2).copied().filter(|&p| p > 0)
    }

    fn take_report(&mut self) -> FrameIdentityStats {
        let window = core::mem::take(&mut self.window);
        FrameIdentityStats {
            latency_min_ns: window.min_ns,
            latency_max_ns: window.max_ns,
            latency_mean_ns: window.mean_ns(),
            ..self.stats
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    const FRAME_PERIOD_NS: i64 = 33_333_333;
    const CAPTURE_START_NS: i64 = 1_791_849_600_000_000_000;
    const NETWORK_LATENCY_NS: i64 = 40_000_000;

    fn capture(frame: i64) -> i64 {
        CAPTURE_START_NS + frame * FRAME_PERIOD_NS
    }

    fn feed(tracker: &mut FrameIdentityTracker, frames: &[i64]) -> Vec<FrameIdentityStats> {
        frames
            .iter()
            .filter_map(|&frame| {
                tracker.observe(capture(frame), capture(frame) + NETWORK_LATENCY_NS)
            })
            .collect()
    }

    #[test]
    fn a_gap_counts_its_missing_frames_and_a_repeat_counts_as_a_duplicate() {
        let mut tracker = FrameIdentityTracker::new();
        // frames 5 and 6 never arrive, frame 8 arrives twice
        feed(&mut tracker, &[0, 1, 2, 3, 4, 7, 8, 8, 9]);
        let stats = tracker.finish().expect("frames since the last report");
        assert_eq!(stats.frames, 8);
        assert_eq!(stats.lost, 2);
        assert_eq!(stats.duplicates, 1);
        assert_eq!(stats.capture_unix_ns, capture(9));
        assert_eq!(stats.latency_ns, NETWORK_LATENCY_NS);
        assert_eq!(stats.latency_min_ns, NETWORK_LATENCY_NS);
        assert_eq!(stats.latency_max_ns, NETWORK_LATENCY_NS);
        assert_eq!(stats.latency_mean_ns, NETWORK_LATENCY_NS);
    }

    #[test]
    fn a_late_frame_fills_the_gap_it_opened() {
        let mut tracker = FrameIdentityTracker::new();
        feed(&mut tracker, &[0, 1, 2, 4, 3, 5]);
        assert_eq!(tracker.stats().lost, 0);
        assert_eq!(tracker.stats().duplicates, 0);
        assert_eq!(tracker.stats().frames, 6);
    }

    #[test]
    fn capture_jitter_is_not_loss() {
        let mut tracker = FrameIdentityTracker::new();
        let jitter = [0, 3_000_000, -2_000_000, 4_000_000, -3_000_000, 1_000_000];
        for (frame, offset) in jitter.iter().cycle().take(60).enumerate() {
            let at = capture(frame as i64) + offset;
            tracker.observe(at, at + NETWORK_LATENCY_NS);
        }
        assert_eq!(tracker.stats().lost, 0);
    }

    #[test]
    fn reports_once_per_interval_with_that_interval_latency() {
        let mut tracker = FrameIdentityTracker::new();
        let frames_per_second = (REPORT_INTERVAL_NS / FRAME_PERIOD_NS) + 1;
        let reports = feed(
            &mut tracker,
            &(0..frames_per_second * 3).collect::<Vec<_>>(),
        );
        assert_eq!(reports.len(), 2, "the first frame only arms the interval");
        assert_eq!(reports[0].frames as i64, frames_per_second + 1);
        assert!(tracker.finish().is_some(), "the tail since the last report");
        assert!(tracker.finish().is_none(), "nothing new to report");
    }
}
