//! Presentation pacing for synchronizing sinks: the PTS -> clock deadline, the
//! play-edge / first-frame anchor, and the QoS late-drop verdict.
//!
//! Every display sink asks the same three questions of a frame ("what is its
//! deadline on the elected clock", "how long do I hold it", "is it too late to
//! bother"), so the answers live here rather than once per sink. A sink holds a
//! [`PresentationPacer`], feeds it the runner's [`ClockSync`] and the stream's
//! [`Segment`] / flush, then calls [`judge`](PresentationPacer::judge) per frame
//! and acts on the returned [`Pace`].
//!
//! The pacer never sleeps and reads the clock only through `ClockSync`: it
//! returns how long to wait, so a tokio sink, a browser `setTimeout` sink, and a
//! bare-metal sink each hold the frame on their own timer. With no clock elected
//! it always answers [`Pace::Now`], which is the pre-sync "present as fast as
//! backpressure allows" behaviour.

use crate::clock::ClockSync;
use crate::element::QosMessage;
use crate::property::{PropError, PropKind, PropValue, PropertySpec};
use crate::qos::QosTracker;
use crate::segment::Segment;

/// What a sink does with the frame it just asked [`PresentationPacer::judge`]
/// about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pace {
    /// Present it now: its deadline has passed (within the QoS bound), or the
    /// sink has no clock to pace against.
    Now,
    /// Hold it this many ns on the sink's own timer, then present.
    Wait(u64),
    /// Do not present it: either past the QoS bound (counted and reported, and
    /// [`take_qos`](PresentationPacer::take_qos) now carries the upstream
    /// report) or outside the segment (an accurate-seek pre-target clip).
    Drop,
}

/// The QoS late-drop bound, as a runtime property.
pub const MAX_LATENESS_PROPERTY: PropertySpec = PropertySpec::new(
    "max-lateness",
    PropKind::Uint,
    "QoS drop threshold, nanoseconds past the deadline",
);

/// The periodic QoS report cadence, as a runtime property.
pub const QOS_INTERVAL_PROPERTY: PropertySpec = PropertySpec::new(
    "qos-interval",
    PropKind::Uint,
    "periodic QoS report interval in nanoseconds, 0 to report drops only",
);

/// The QoS knobs a synchronizing sink exposes, so a `gst-launch` line can tune
/// the late-drop bound and the report cadence. The whole table for a sink with
/// no properties of its own; one that has others lists the two consts alongside
/// them. Handled by [`PresentationPacer::set_property`] /
/// [`get_property`](PresentationPacer::get_property).
pub const PACING_PROPERTIES: &[PropertySpec] = &[MAX_LATENESS_PROPERTY, QOS_INTERVAL_PROPERTY];

/// Per-sink presentation timing: the elected clock, the segment mapping, the
/// deadline anchor, and the QoS tracker.
#[derive(Debug, Default)]
pub struct PresentationPacer {
    /// Elected clock + base time from the runner. `Some` enables PTS pacing:
    /// each frame is held until its running-time deadline on the clock. `None`
    /// (no clock elected) keeps the pre-sync "present ASAP" behaviour.
    clock_sync: Option<ClockSync>,
    /// Active playback segment, from `PipelinePacket::Segment`, used to map a
    /// frame's PTS to running time (correct across a seek). `None` before any
    /// segment arrives, where PTS is the running time directly.
    segment: Option<Segment>,
    /// Clock time a frame's deadline is measured from: `deadline = anchor_ns +
    /// running_time`. Latched on the first frame (or re-based onto the play
    /// edge), so the stream paces by PTS deltas regardless of how long the
    /// source took to start.
    anchor_ns: Option<u64>,
    /// The anchor was set provisionally from a preroll frame consumed during
    /// `Paused`, before `Playing` stamped the base time. The next frame once
    /// `Playing` is anchored re-bases onto the play edge and clears this.
    anchor_pre_play: bool,
    /// A seek flush asks the next frame to first-frame-anchor (present the seek
    /// target immediately) rather than reuse the stale play-edge base time.
    seek_reanchor: bool,
    qos: QosTracker,
    /// The last late-drop report, for the sink's `take_qos` (the upstream
    /// reverse channel); latest wins.
    pending_qos: Option<QosMessage>,
}

impl PresentationPacer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adopt the elected clock + base time, engaging PTS pacing. The runner
    /// calls this once per element after clock election. Any anchor latched
    /// against the previous clock is dropped: the new clock has its own epoch,
    /// so the next frame re-anchors on it.
    pub fn set_clock_sync(&mut self, sync: ClockSync) {
        self.clock_sync = Some(sync);
        self.anchor_ns = None;
        self.anchor_pre_play = false;
    }

    /// The clock + base time being paced against, `None` until one is adopted.
    pub fn clock_sync(&self) -> Option<&ClockSync> {
        self.clock_sync.as_ref()
    }

    /// Whether PTS pacing is engaged (a clock was elected and handed over).
    pub fn is_paced(&self) -> bool {
        self.clock_sync.is_some()
    }

    /// Pin the anchor instead of latching it on the first frame, so a frame's
    /// deadline is `anchor_ns + running_time` from the outset. A sink whose PTS
    /// deadlines are absolute on its own clock pins `0` (deadline == running
    /// time); [`set_clock_sync`](Self::set_clock_sync) drops the pin, since an
    /// elected clock brings its own epoch.
    pub fn set_anchor_ns(&mut self, anchor_ns: u64) {
        self.anchor_ns = Some(anchor_ns);
        self.anchor_pre_play = false;
    }

    /// Install the active playback segment (from `PipelinePacket::Segment`).
    pub fn set_segment(&mut self, segment: Segment) {
        self.segment = Some(segment);
    }

    /// The active playback segment, `None` before any arrives. A sink reads it
    /// for the parts of presentation the pacer does not decide, eg trick-mode
    /// `key_units_only` frame selection.
    pub fn segment(&self) -> Option<Segment> {
        self.segment
    }

    /// Seek flush: drop the anchor so the next frame presents the seek target
    /// immediately instead of at the stale play-edge base time.
    pub fn flush(&mut self) {
        self.anchor_ns = None;
        self.anchor_pre_play = false;
        self.seek_reanchor = true;
    }

    /// Running time for a frame PTS, mapped through the active segment (the PTS
    /// directly when none). `None` means the frame is outside the segment and
    /// should be clipped (accurate-seek pre-target).
    pub fn running_time(&self, pts_ns: u64) -> Option<u64> {
        match &self.segment {
            Some(seg) => seg.to_running_time(pts_ns),
            None => Some(pts_ns),
        }
    }

    /// Clock time a frame's PTS is due at: the anchor (latched here on the first
    /// frame) plus its running time. `None` when there is nothing to wait for,
    /// either because no clock is elected or because the frame is clipped
    /// outside the segment. A caller that wants the wait, the QoS verdict and
    /// the drop counted wants [`judge`](PresentationPacer::judge) instead; this
    /// is for one that shifts the deadline itself (`clocksync`'s `ts-offset`).
    pub fn deadline_ns(&mut self, pts_ns: u64) -> Option<u64> {
        let sync = self.clock_sync.clone()?;
        let rt = self.running_time(pts_ns)?;
        Some(self.presentation_anchor(&sync, rt).saturating_add(rt))
    }

    /// Verdict for one frame: how long to hold it, or that it is not to be
    /// presented. `presented` is the sink's running presented count, reported
    /// in [`BusMessage::Qos`](crate::BusMessage::Qos).
    pub fn judge(&mut self, pts_ns: u64, presented: u64) -> Pace {
        let Some(sync) = self.clock_sync.clone() else {
            return Pace::Now;
        };
        let Some(deadline) = self.deadline_ns(pts_ns) else {
            return Pace::Drop;
        };
        let now = sync.now_ns();
        // A frame already late beyond the bound is dropped, not presented late,
        // so the sink catches up instead of accumulating lag. The same call
        // posts the periodic running-stats report when the frame is on time and
        // the interval has elapsed.
        if let Some(q) = self.qos.judge_frame(deadline, now, presented) {
            self.pending_qos = Some(q);
            return Pace::Drop;
        }
        match deadline.checked_sub(now) {
            Some(0) | None => Pace::Now,
            Some(wait) => Pace::Wait(wait),
        }
    }

    /// The QoS report a late drop left for the sink's
    /// [`take_qos`](crate::AsyncElement::take_qos), which the runner relays
    /// upstream so a source / decoder can shed load.
    pub fn take_qos(&mut self) -> Option<QosMessage> {
        self.pending_qos.take()
    }

    /// Attach the pipeline bus so QoS reports reach the application.
    pub fn set_bus(&mut self, bus: crate::BusHandle) {
        self.qos.set_bus(bus);
    }

    /// Drop a frame past its deadline by more than `ns`. `0` drops any late
    /// frame; the default (`u64::MAX`) never drops, presenting every frame
    /// however late.
    pub fn set_max_lateness_ns(&mut self, ns: u64) {
        self.qos.set_max_lateness_ns(ns);
    }

    pub fn max_lateness_ns(&self) -> u64 {
        self.qos.max_lateness_ns()
    }

    /// Post a running-stats `Qos` report every `ns` of clock time while frames
    /// present, on top of the per-drop reports. `0` (the default) reports only
    /// drops.
    pub fn set_report_interval_ns(&mut self, ns: u64) {
        self.qos.set_report_interval_ns(ns);
    }

    pub fn report_interval_ns(&self) -> u64 {
        self.qos.report_interval_ns()
    }

    /// Frames dropped for lateness so far.
    pub fn late_dropped(&self) -> u64 {
        self.qos.dropped()
    }

    /// Whether a frame due at `deadline_ns` is too late at clock time `now_ns`,
    /// without judging it (no count, no report).
    pub fn is_too_late(&self, deadline_ns: u64, now_ns: u64) -> bool {
        self.qos.is_too_late(deadline_ns, now_ns)
    }

    /// Handle one of [`PACING_PROPERTIES`]; `None` means the name is not a
    /// pacing property, so the sink should carry on with its own.
    pub fn set_property(&mut self, name: &str, value: &PropValue) -> Option<Result<(), PropError>> {
        let ns = || value.as_uint().ok_or(PropError::Type);
        match name {
            "max-lateness" => Some(ns().map(|n| self.set_max_lateness_ns(n))),
            "qos-interval" => Some(ns().map(|n| self.set_report_interval_ns(n))),
            _ => None,
        }
    }

    /// Read one of [`PACING_PROPERTIES`]; `None` means the name is not a pacing
    /// property.
    pub fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "max-lateness" => Some(PropValue::Uint(self.max_lateness_ns())),
            "qos-interval" => Some(PropValue::Uint(self.report_interval_ns())),
            _ => None,
        }
    }

    /// The clock-time anchor a frame's deadline is measured from. Three cases:
    ///
    /// - **`Playing` anchor armed and stamped** (a state-driven run): anchor on
    ///   the play-edge base time, so the first played frame presents when
    ///   streaming began, not at runner startup. A preroll frame consumed during
    ///   `Paused` (before the stamp) re-bases onto the play edge here once
    ///   `Playing` arrives.
    /// - **Seek flush pending**: first-frame-anchor (`now - running_time`) so the
    ///   seek target presents immediately, ignoring the stale play-edge base.
    /// - **Otherwise** (slow start, live, or pre-`Playing` preroll): first-frame
    ///   anchor, then pace by PTS deltas.
    fn presentation_anchor(&mut self, sync: &ClockSync, rt: u64) -> u64 {
        // Re-base a provisional preroll anchor onto the play edge once `Playing`
        // has stamped the base time.
        if self.anchor_pre_play && sync.play_anchored() && !self.seek_reanchor {
            self.anchor_ns = Some(sync.base_time());
            self.anchor_pre_play = false;
        }
        if let Some(a) = self.anchor_ns {
            return a;
        }
        // No anchor yet: establish one. Prefer the stamped play-edge base time
        // unless a seek just flushed (then present the target now).
        let use_play = sync.play_anchored() && !self.seek_reanchor;
        let anchor = if use_play {
            sync.base_time()
        } else {
            sync.now_ns().saturating_sub(rt)
        };
        self.anchor_ns = Some(anchor);
        // Mark provisional only if anchored before `Playing` stamped, so it
        // re-bases later; a post-seek first-frame anchor is final.
        self.anchor_pre_play = !use_play && !sync.play_anchored();
        self.seek_reanchor = false;
        anchor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::{Bus, BusMessage};
    use crate::clock::{PipelineClock, PlayAnchor};
    use alloc::sync::Arc;
    use core::sync::atomic::{AtomicU64, Ordering};

    /// A clock whose `now_ns` the test drives by hand.
    #[derive(Debug)]
    struct ManualClock(Arc<AtomicU64>);
    impl PipelineClock for ManualClock {
        fn now_ns(&self) -> u64 {
            self.0.load(Ordering::Relaxed)
        }
    }

    fn manual() -> (Arc<AtomicU64>, ClockSync) {
        let t = Arc::new(AtomicU64::new(0));
        let sync = ClockSync::new(Arc::new(ManualClock(t.clone())), 0);
        (t, sync)
    }

    #[test]
    fn without_a_clock_every_frame_presents_now() {
        let mut p = PresentationPacer::new();
        assert!(!p.is_paced());
        assert_eq!(p.judge(5_000_000_000, 0), Pace::Now);
        assert_eq!(p.late_dropped(), 0);
    }

    #[test]
    fn frames_wait_for_their_pts_deadline() {
        let (clock, sync) = manual();
        let mut p = PresentationPacer::new();
        p.set_clock_sync(sync);
        // First frame anchors at now (0), so it presents immediately.
        assert_eq!(p.judge(0, 0), Pace::Now);
        // 33 ms of PTS later, with the clock still at 0: hold it 33 ms.
        assert_eq!(p.judge(33_000_000, 1), Pace::Wait(33_000_000));
        // Once the clock reaches the deadline, no wait.
        clock.store(66_000_000, Ordering::Relaxed);
        assert_eq!(p.judge(66_000_000, 2), Pace::Now);
    }

    #[test]
    fn a_late_frame_drops_and_reports_both_ways() {
        let (clock, sync) = manual();
        let (bus, handle) = Bus::new(4);
        let mut p = PresentationPacer::new();
        p.set_clock_sync(sync);
        p.set_bus(handle);
        p.set_max_lateness_ns(0);
        // Anchor on the first frame at clock 0.
        assert_eq!(p.judge(0, 0), Pace::Now);
        assert!(p.take_qos().is_none(), "on-time frame reports nothing");

        // The clock jumps 1 ms past the next frame's deadline.
        clock.store(1_000_000, Ordering::Relaxed);
        assert_eq!(p.judge(0, 1), Pace::Drop);
        assert_eq!(p.late_dropped(), 1);
        let upstream = p.take_qos().expect("upstream QoS report for the relay");
        assert_eq!(upstream.jitter_ns, 1_000_000);
        assert_eq!(upstream.running_time_ns, 0);
        assert!(p.take_qos().is_none(), "consumed once");
        match bus.try_recv() {
            Some(BusMessage::Qos {
                jitter_ns, dropped, ..
            }) => assert_eq!((jitter_ns, dropped), (1_000_000, 1)),
            other => panic!("expected a Qos message, got {other:?}"),
        }
    }

    #[test]
    fn the_default_bound_never_drops_however_late() {
        let (clock, sync) = manual();
        let mut p = PresentationPacer::new();
        p.set_clock_sync(sync);
        assert_eq!(p.judge(0, 0), Pace::Now);
        clock.store(u64::MAX, Ordering::Relaxed);
        assert_eq!(p.judge(0, 1), Pace::Now, "presented, however late");
        assert_eq!(p.late_dropped(), 0);
    }

    #[test]
    fn running_time_maps_through_the_segment_and_clips() {
        let mut p = PresentationPacer::new();
        assert_eq!(p.running_time(50_000_000), Some(50_000_000), "no segment");
        // Accurate seek to 70 ms: a pre-target frame clips, the target frame is
        // running-time zero.
        p.set_segment(Segment::for_flush_seek(
            &crate::Seek::flush_to(70_000_000),
            None,
        ));
        assert_eq!(p.running_time(66_000_000), None);
        assert_eq!(p.running_time(70_000_000), Some(0));
        // A clipped frame is not presented and is not a QoS drop.
        let (_clock, sync) = manual();
        p.set_clock_sync(sync);
        assert_eq!(p.judge(66_000_000, 0), Pace::Drop);
        assert_eq!(p.late_dropped(), 0);
        assert!(p.take_qos().is_none());
    }

    #[test]
    fn anchor_uses_the_play_edge_and_rebases_preroll() {
        let clock = Arc::new(AtomicU64::new(1_000));
        let anchor = PlayAnchor::new();
        let sync =
            ClockSync::with_play_anchor(Arc::new(ManualClock(clock.clone())), 0, anchor.clone());
        let mut p = PresentationPacer::new();

        // Preroll frame consumed during `Paused` (anchor not yet stamped): the
        // pacer first-frame-anchors so it presents immediately, provisionally.
        assert_eq!(p.presentation_anchor(&sync, 0), 1_000);
        assert!(p.anchor_pre_play, "provisional, awaiting Playing");

        // Playing stamps the base time at 5_000: the next frame re-bases onto the
        // play edge, discarding the preroll-time anchor.
        anchor.stamp(5_000);
        assert_eq!(p.presentation_anchor(&sync, 100), 5_000);
        assert!(!p.anchor_pre_play, "re-base is final");

        // A seek flush forces a first-frame re-anchor (present the target now),
        // ignoring the stale play-edge base.
        clock.store(8_000, Ordering::Relaxed);
        p.flush();
        assert_eq!(p.presentation_anchor(&sync, 200), 7_800);
        assert!(!p.seek_reanchor, "seek re-anchor consumed");
    }

    #[test]
    fn qos_properties_round_trip() {
        let mut p = PresentationPacer::new();
        assert!(p.set_property("device", &PropValue::Uint(1)).is_none());
        p.set_property("max-lateness", &PropValue::Uint(20_000_000))
            .expect("known property")
            .unwrap();
        p.set_property("qos-interval", &PropValue::Uint(1_000_000_000))
            .expect("known property")
            .unwrap();
        assert_eq!(
            p.get_property("max-lateness"),
            Some(PropValue::Uint(20_000_000))
        );
        assert_eq!(
            p.get_property("qos-interval"),
            Some(PropValue::Uint(1_000_000_000))
        );
        assert_eq!(p.get_property("device"), None);
        assert_eq!(
            p.set_property("max-lateness", &PropValue::Bool(true)),
            Some(Err(PropError::Type))
        );
    }
}
