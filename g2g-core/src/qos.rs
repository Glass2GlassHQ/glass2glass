// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! Sink-side QoS accounting: the late-frame drop rule and the
//! [`BusMessage::Qos`] reports that follow from it.
//!
//! Every synchronizing sink asks the same question of each frame ("is this past
//! its deadline by more than the bound?") and reports the same running stats
//! when the answer is yes, so the rule and the reports live here instead of in
//! each sink. Two cadences: one report per drop (the GStreamer
//! `GST_MESSAGE_QOS` analog), plus an optional periodic report of the running
//! stats while frames are flowing, so an application can watch the trend rather
//! than only the failures.
//!
//! The tracker reads no clock: callers pass `now_ns` from the pipeline clock
//! they already pace on, which keeps it usable under `no_std` and keeps
//! reporting cadence on pipeline time rather than wall time.

use crate::bus::{BusHandle, BusMessage};
use crate::element::QosMessage;
use crate::log::{ClockTime, Target, PACING_CATEGORY};

/// Signed lateness of a frame at `deadline_ns` observed at clock time `now_ns`:
/// positive behind the clock, negative early.
fn jitter_ns(deadline_ns: u64, now_ns: u64) -> i64 {
    if now_ns >= deadline_ns {
        i64::try_from(now_ns - deadline_ns).unwrap_or(i64::MAX)
    } else {
        i64::try_from(deadline_ns - now_ns)
            .map(|v| -v)
            .unwrap_or(i64::MIN)
    }
}

#[cfg(feature = "std")]
fn taken_ns() -> u64 {
    crate::metrics::monotonic_ns()
}

#[cfg(not(feature = "std"))]
fn taken_ns() -> u64 {
    0
}

/// The upstream report for a frame due at `deadline_ns`, judged at clock time
/// `now_ns`.
pub(crate) fn upstream_report(deadline_ns: u64, now_ns: u64, running_time_ns: u64) -> QosMessage {
    QosMessage {
        jitter_ns: jitter_ns(deadline_ns, now_ns),
        running_time_ns,
        taken_ns: taken_ns(),
        downstream_proc_ns: 0,
    }
}

/// QoS state shared by synchronizing sinks: the lateness bound, the drop count,
/// and the bus reporting.
#[derive(Debug)]
pub struct QosTracker {
    bus: Option<BusHandle>,
    /// Drop a frame whose deadline is already past by more than this many ns.
    /// `u64::MAX` (the default) never drops.
    max_lateness_ns: u64,
    /// Cadence of the running-stats report, in pipeline-clock ns. `0` disables
    /// it, leaving only the per-drop reports.
    interval_ns: u64,
    /// Clock time of the last report of either kind; `None` until the first
    /// frame arms the interval.
    last_report_ns: Option<u64>,
    dropped: u64,
    last_presented_running_time_ns: Option<u64>,
    previous_running_time_ns: Option<u64>,
    average_frame_spacing_ns: Option<u64>,
}

// gst basesink presents a late frame anyway when nothing has been presented for this long
const STARVED_PRESENT_GAP_NS: u64 = 1_000_000_000;

impl Default for QosTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl QosTracker {
    pub fn new() -> Self {
        Self {
            bus: None,
            max_lateness_ns: u64::MAX,
            interval_ns: 0,
            last_report_ns: None,
            dropped: 0,
            last_presented_running_time_ns: None,
            previous_running_time_ns: None,
            average_frame_spacing_ns: None,
        }
    }

    /// Attach the pipeline bus; without one the tracker still decides and counts
    /// drops, it just reports nothing.
    pub fn set_bus(&mut self, bus: BusHandle) {
        self.bus = Some(bus);
    }

    /// Drop a frame past its deadline by more than `ns`. `0` drops any frame
    /// that arrives after its deadline; `u64::MAX` never drops.
    pub fn set_max_lateness_ns(&mut self, ns: u64) {
        self.max_lateness_ns = ns;
    }

    pub fn max_lateness_ns(&self) -> u64 {
        self.max_lateness_ns
    }

    /// Post a running-stats [`BusMessage::Qos`] every `ns` of pipeline-clock
    /// time. `0` (the default) leaves only the per-drop reports.
    pub fn set_report_interval_ns(&mut self, ns: u64) {
        self.interval_ns = ns;
    }

    pub fn report_interval_ns(&self) -> u64 {
        self.interval_ns
    }

    /// Frames dropped for lateness so far.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    pub(crate) fn flush(&mut self) {
        self.last_presented_running_time_ns = None;
        self.previous_running_time_ns = None;
        self.average_frame_spacing_ns = None;
    }

    /// Whether a frame whose display ends at clock time `frame_end_ns` is too
    /// late to present at clock time `now_ns`. Saturating, so the `u64::MAX`
    /// default never trips.
    pub fn is_too_late(&self, frame_end_ns: u64, now_ns: u64) -> bool {
        now_ns > frame_end_ns.saturating_add(self.max_lateness_ns)
    }

    /// Verdict for one frame due at clock time `deadline_ns`, whose running
    /// time is `running_time_ns`, `processed` being the sink's presented count.
    /// `duration_ns` is the frame's running-time duration, `None` when unknown.
    ///
    /// `Some` means the frame is too late and must be dropped: the drop is
    /// counted and reported on the bus, and the returned [`QosMessage`] is the
    /// sink's upstream report, for a sink that relays it. `None` means
    /// present it, and posts the periodic running-stats report if the interval
    /// has elapsed.
    pub fn judge_frame(
        &mut self,
        deadline_ns: u64,
        duration_ns: Option<u64>,
        now_ns: u64,
        running_time_ns: u64,
        processed: u64,
    ) -> Option<QosMessage> {
        if let Some(spacing) = self
            .previous_running_time_ns
            .and_then(|previous| running_time_ns.checked_sub(previous))
            .filter(|&spacing| spacing > 0)
        {
            // gst basesink's running average over a window of eight frames
            self.average_frame_spacing_ns =
                Some(self.average_frame_spacing_ns.map_or(spacing, |average| {
                    spacing.saturating_add(average.saturating_mul(7)) / 8
                }));
        }
        self.previous_running_time_ns = Some(running_time_ns);
        // gst basesink measures lateness from a frame's end, its start plus the average spacing when it has no duration
        let frame_end_ns =
            deadline_ns.saturating_add(duration_ns.or(self.average_frame_spacing_ns).unwrap_or(0));
        let jitter = jitter_ns(deadline_ns, now_ns);
        let last_presented = *self
            .last_presented_running_time_ns
            .get_or_insert(running_time_ns);
        let starved = running_time_ns.saturating_sub(last_presented) > STARVED_PRESENT_GAP_NS;
        if self.is_too_late(frame_end_ns, now_ns) {
            if !starved {
                self.dropped += 1;
                self.post(running_time_ns, jitter, processed, now_ns);
                return Some(upstream_report(deadline_ns, now_ns, running_time_ns));
            }
            crate::g2g_warn!(
                Target::category(PACING_CATEGORY),
                "presenting a frame {} late because nothing was presented for {}: the pipeline is slower than real time",
                ClockTime(now_ns.saturating_sub(deadline_ns)),
                ClockTime(running_time_ns - last_presented)
            );
        }
        self.last_presented_running_time_ns = Some(running_time_ns);
        match self.last_report_ns {
            // First frame only arms the interval, so an application sees the
            // periodic report as a trend rather than a startup event.
            None => self.last_report_ns = Some(now_ns),
            Some(last)
                if self.interval_ns > 0 && now_ns.saturating_sub(last) >= self.interval_ns =>
            {
                self.post(running_time_ns, jitter, processed, now_ns);
            }
            Some(_) => {}
        }
        None
    }

    /// Post a report and restart the periodic interval from it: a drop report
    /// already carries the running stats, so it stands in for the next tick.
    fn post(&mut self, running_time_ns: u64, jitter: i64, processed: u64, now_ns: u64) {
        self.last_report_ns = Some(now_ns);
        if let Some(bus) = &self.bus {
            // Control message: non-blocking, never stalls the sink (a full bus
            // drops the report).
            bus.try_post(BusMessage::Qos {
                running_time_ns,
                jitter_ns: jitter,
                processed,
                dropped: self.dropped,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::Bus;

    #[test]
    fn default_bound_never_drops() {
        let mut q = QosTracker::new();
        assert!(q.judge_frame(0, None, u64::MAX, 0, 0).is_none());
        assert_eq!(q.dropped(), 0);
    }

    #[test]
    fn late_frame_drops_and_posts() {
        let (bus, handle) = Bus::new(8);
        let mut q = QosTracker::new();
        q.set_bus(handle);
        q.set_max_lateness_ns(10);
        assert!(
            q.judge_frame(100, Some(0), 105, 100, 3).is_none(),
            "within the bound"
        );
        let upstream = q
            .judge_frame(200, Some(0), 215, 200, 3)
            .expect("past the bound");
        assert_eq!(upstream.jitter_ns, 15);
        assert_eq!(upstream.running_time_ns, 200);
        assert_eq!(q.dropped(), 1);
        assert_eq!(
            bus.try_recv(),
            Some(BusMessage::Qos {
                running_time_ns: 200,
                jitter_ns: 15,
                processed: 3,
                dropped: 1,
            })
        );
    }

    #[test]
    fn a_frame_is_late_once_its_end_passes_the_bound() {
        let mut q = QosTracker::new();
        let max_lateness = 5;
        let (deadline, duration) = (100, 33);
        q.set_max_lateness_ns(max_lateness);
        let last_tolerated = deadline + duration + max_lateness;
        assert!(q
            .judge_frame(deadline, Some(duration), last_tolerated, deadline, 0)
            .is_none());
        assert!(q
            .judge_frame(deadline, Some(duration), last_tolerated + 1, deadline, 0)
            .is_some());
    }

    #[test]
    fn a_frame_without_a_duration_ends_after_the_average_spacing() {
        let mut q = QosTracker::new();
        q.set_max_lateness_ns(0);
        let spacing = 40;
        for i in 0..2 {
            let on_time = i * spacing;
            assert!(q.judge_frame(on_time, None, on_time, on_time, i).is_none());
        }
        let third = 2 * spacing;
        assert!(
            q.judge_frame(third, None, third + spacing, third, 2)
                .is_none(),
            "within one average spacing of its start"
        );
        let fourth = 3 * spacing;
        assert!(
            q.judge_frame(fourth, None, fourth + spacing + 1, fourth, 3)
                .is_some(),
            "past one average spacing"
        );
    }

    #[test]
    fn late_frame_presents_after_a_second_without_one() {
        let mut q = QosTracker::new();
        q.set_max_lateness_ns(0);
        let late_by = 50;
        let frame_period = STARVED_PRESENT_GAP_NS / 4;
        let mut presented_at = alloc::vec::Vec::new();
        for i in 0..10u64 {
            let running_time = i * frame_period;
            if q.judge_frame(
                running_time,
                Some(0),
                running_time + late_by,
                running_time,
                0,
            )
            .is_none()
            {
                presented_at.push(i);
            }
        }
        assert_eq!(presented_at, [5]);
        assert_eq!(q.dropped(), 9);
    }

    #[test]
    fn periodic_report_fires_once_per_interval() {
        let (bus, handle) = Bus::new(8);
        let mut q = QosTracker::new();
        q.set_bus(handle);
        q.set_report_interval_ns(1_000);
        // t=0 arms the interval, t=500 is inside it, t=1000 reports.
        assert!(q.judge_frame(0, None, 0, 0, 0).is_none());
        assert!(q.judge_frame(500, None, 500, 500, 1).is_none());
        assert_eq!(bus.try_recv(), None, "nothing before the interval elapses");
        assert!(q.judge_frame(1_000, None, 1_000, 1_000, 2).is_none());
        match bus.try_recv() {
            Some(BusMessage::Qos {
                processed, dropped, ..
            }) => {
                assert_eq!((processed, dropped), (2, 0));
            }
            other => panic!("expected a periodic Qos, got {other:?}"),
        }
        assert!(q.judge_frame(1_500, None, 1_500, 1_500, 3).is_none());
        assert_eq!(bus.try_recv(), None, "interval restarted from the report");
    }

    #[test]
    fn no_interval_means_no_periodic_report() {
        let (bus, handle) = Bus::new(8);
        let mut q = QosTracker::new();
        q.set_bus(handle);
        for i in 0..10u64 {
            assert!(q
                .judge_frame(i * 1_000, None, i * 1_000, i * 1_000, i)
                .is_none());
        }
        assert_eq!(bus.try_recv(), None);
    }

    #[test]
    fn early_frame_reports_negative_jitter() {
        let (bus, handle) = Bus::new(8);
        let mut q = QosTracker::new();
        q.set_bus(handle);
        q.set_report_interval_ns(100);
        q.judge_frame(0, None, 0, 0, 0);
        // Presented 40ns before its deadline, 200ns after the last report.
        q.judge_frame(240, None, 200, 240, 1);
        match bus.try_recv() {
            Some(BusMessage::Qos { jitter_ns, .. }) => assert_eq!(jitter_ns, -40),
            other => panic!("expected a periodic Qos, got {other:?}"),
        }
    }
}
