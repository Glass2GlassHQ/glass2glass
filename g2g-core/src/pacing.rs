// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

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

use crate::caps::{framerate_q16_period_ns, Caps, Rate};
use crate::clock::ClockSync;
use crate::element::QosMessage;
use crate::frame::FrameTiming;
use crate::log::{ClockTime, Target, PACING_CATEGORY};
use crate::property::{PropError, PropKind, PropValue, PropertySpec};
use crate::qos::{upstream_report, QosTracker};
use crate::query::LatencyReport;
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
    "QoS drop threshold, nanoseconds past the frame end",
);

/// The periodic QoS report cadence, as a runtime property.
pub const QOS_INTERVAL_PROPERTY: PropertySpec = PropertySpec::new(
    "qos-interval",
    PropKind::Uint,
    "periodic QoS report interval in nanoseconds, 0 to report drops only",
);

pub const PROCESSING_DEADLINE_PROPERTY: PropertySpec = PropertySpec::new(
    "processing-deadline",
    PropKind::Uint,
    "nanoseconds a live pipeline may take to process a frame, added to the latency",
);

pub const SYNC_PROPERTY: PropertySpec = PropertySpec::new(
    "sync",
    PropKind::Bool,
    "hold each frame until its timestamp on the clock; false presents on arrival and never drops late frames",
)
.with_default("true");

/// The pacing knobs a synchronizing sink exposes, so a `gst-launch` line can
/// turn clock sync off and tune the late-drop bound, the processing allowance
/// and the report cadence. The whole table for a sink with no properties of its
/// own; one that has others lists the four consts alongside them. Handled by
/// [`PresentationPacer::set_property`] /
/// [`get_property`](PresentationPacer::get_property).
pub const PACING_PROPERTIES: &[PropertySpec] = &[
    SYNC_PROPERTY,
    MAX_LATENESS_PROPERTY,
    PROCESSING_DEADLINE_PROPERTY,
    QOS_INTERVAL_PROPERTY,
];

/// A clock reading or running time as a signed nanosecond count, so anchor
/// arithmetic can go below zero. Saturates rather than wrapping: a PTS past
/// `i64::MAX` is a bogus timestamp, and pinning it at the far end of the
/// timeline keeps the frame late rather than turning it negative.
fn as_signed(ns: u64) -> i64 {
    i64::try_from(ns).unwrap_or(i64::MAX)
}

/// Per-sink presentation timing: the elected clock, the segment mapping, the
/// deadline anchor, and the QoS tracker.
#[derive(Debug, Default)]
pub struct PresentationPacer {
    /// Elected clock + base time from the runner. `Some` enables PTS pacing:
    /// each frame is held until its running-time deadline on the clock. `None`
    /// (no clock elected) keeps the pre-sync "present ASAP" behaviour.
    clock_sync: Option<ClockSync>,
    present_on_arrival: bool,
    /// Active playback segment, from `PipelinePacket::Segment`, used to map a
    /// frame's PTS to running time (correct across a seek). `None` before any
    /// segment arrives, where PTS is the running time directly.
    segment: Option<Segment>,
    /// Clock time a frame's deadline is measured from: `deadline = anchor_ns +
    /// running_time`. Latched on the first frame (or re-based onto the play
    /// edge), so the stream paces by PTS deltas regardless of how long the
    /// source took to start.
    ///
    /// Signed because a first-frame anchor is `clock - running_time`, and a
    /// stream whose PTS epoch is ahead of the elected clock's puts that below
    /// zero: an audio `DriftClock` reads the playout position (seconds since the
    /// device opened) while an HLS feed's PTS is the publisher's uptime. Clamping
    /// it at zero would push every deadline that far into the future and the sink
    /// would never present.
    anchor_ns: Option<i64>,
    /// The anchor was set provisionally from a preroll frame consumed during
    /// `Paused`, before `Playing` stamped the base time. The next frame once
    /// `Playing` is anchored re-bases onto the play edge and clears this.
    anchor_pre_play: bool,
    /// A seek flush asks the next frame to first-frame-anchor (present the seek
    /// target immediately) rather than reuse the stale play-edge base time.
    seek_reanchor: bool,
    qos: QosTracker,
    /// The last upstream report, a late drop or the periodic one, for the
    /// sink's `take_qos` (the upstream reverse channel), latest wins.
    pending_qos: Option<QosMessage>,
    /// Clock time of the last upstream report, `None` until the first judged
    /// frame arms the interval.
    last_upstream_report_ns: Option<u64>,
    /// Deadline the most recent [`judge`](Self::judge) measured against, so a
    /// sink can score its own presentation error without re-deriving it.
    /// Cleared by a `judge` that reaches no deadline (unpaced, unset PTS, or a
    /// clipped frame).
    last_deadline_ns: Option<u64>,
    // Late drops since the last presented frame, so a run of them logs once at each end.
    late_drops_in_a_row: u64,
    processing_deadline_ns: u64,
    frame_period_ns: Option<u64>,
}

/// GStreamer's `GstVideoSink` default.
pub const VIDEO_SINK_MAX_LATENESS_NS: u64 = 5_000_000;

// gstreamer's GstVideoSink default
pub const VIDEO_SINK_PROCESSING_DEADLINE_NS: u64 = 15_000_000;

// gstreamer's GstBaseSink default, which its audio and decklink sinks keep
pub const BASE_SINK_PROCESSING_DEADLINE_NS: u64 = 20_000_000;

/// How often a sink that presents frames reports its lateness upstream, so a
/// producer that shed load on a late report also learns when the sink caught up.
pub const UPSTREAM_QOS_INTERVAL_NS: u64 = 100_000_000;

impl PresentationPacer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn for_video_sink() -> Self {
        let mut pacer = Self::with_processing_deadline_ns(VIDEO_SINK_PROCESSING_DEADLINE_NS);
        pacer.set_max_lateness_ns(VIDEO_SINK_MAX_LATENESS_NS);
        pacer
    }

    pub fn with_processing_deadline_ns(processing_deadline_ns: u64) -> Self {
        Self {
            processing_deadline_ns,
            ..Self::default()
        }
    }

    /// Adopt the elected clock + base time, engaging PTS pacing. The runner
    /// calls this once per element after clock election. Any anchor latched
    /// against the previous clock is dropped: the new clock has its own epoch,
    /// so the next frame re-anchors on it.
    pub fn set_clock_sync(&mut self, sync: ClockSync) {
        self.clock_sync = Some(sync);
        self.anchor_ns = None;
        self.anchor_pre_play = false;
        self.last_upstream_report_ns = None;
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
        self.anchor_ns = Some(as_signed(anchor_ns));
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
        self.qos.flush();
        // gstreamer's flushing seek in Paused restarts running time at the next play edge's base time
        self.seek_reanchor = !self
            .clock_sync
            .as_ref()
            .is_some_and(ClockSync::awaits_play_edge);
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
    /// frame) plus its running time. `None` when there is nothing to wait for:
    /// no clock is elected, the PTS is unset ([`FrameTiming::PTS_NONE`], present
    /// it on arrival) or the frame is clipped outside the segment. A caller that
    /// distinguishes the clip from the other two reads
    /// [`FrameTiming::pts`] itself first. A caller that wants the wait, the QoS verdict and
    /// the drop counted wants [`judge`](PresentationPacer::judge) instead; this
    /// is for one that shifts the deadline itself (`clocksync`'s `ts-offset`).
    pub fn deadline_ns(&mut self, pts_ns: u64) -> Option<u64> {
        // An unset PTS has no deadline, and must not latch the anchor either:
        // the anchor belongs to the first frame that carries a real one.
        if pts_ns == FrameTiming::PTS_NONE {
            return None;
        }
        let rt = self.running_time(pts_ns)?;
        let sync = self.clock_sync.as_ref()?;
        Some(Self::due_at(
            sync,
            rt,
            &mut self.anchor_ns,
            &mut self.anchor_pre_play,
            &mut self.seek_reanchor,
        ))
    }

    /// Clock time a frame at running time `rt` is due at on `sync`, latching the
    /// anchor on first use. Takes the anchor fields apart from `self` so the
    /// clock sync is read in place rather than cloned per frame.
    fn due_at(
        sync: &ClockSync,
        rt: u64,
        anchor_ns: &mut Option<i64>,
        anchor_pre_play: &mut bool,
        seek_reanchor: &mut bool,
    ) -> u64 {
        let deadline =
            Self::presentation_anchor(sync, rt, anchor_ns, anchor_pre_play, seek_reanchor)
                .saturating_add(as_signed(rt))
                .saturating_add(as_signed(sync.deadline_latency_ns()))
                .saturating_add(as_signed(sync.paused_total_ns()));
        // A deadline below the clock's zero is simply already due.
        deadline.max(0) as u64
    }

    /// Verdict for one frame: how long to hold it, or that it is not to be
    /// presented. `presented` is the sink's running presented count, reported
    /// in [`BusMessage::Qos`](crate::BusMessage::Qos).
    pub fn judge(&mut self, timing: &FrameTiming, presented: u64) -> Pace {
        let pts_ns = timing.pts_ns;
        self.last_deadline_ns = None;
        // A frame with no presentation time is due on arrival: never late, so
        // never a QoS drop.
        if pts_ns == FrameTiming::PTS_NONE || self.present_on_arrival {
            return Pace::Now;
        }
        let Some(sync) = self.clock_sync.as_ref() else {
            return Pace::Now;
        };
        let Some(running_time) = self.running_time(pts_ns) else {
            crate::g2g_log!(
                Target::category(PACING_CATEGORY),
                "clipping frame pts {} outside the segment",
                ClockTime(pts_ns)
            );
            return Pace::Drop;
        };
        let deadline = Self::due_at(
            sync,
            running_time,
            &mut self.anchor_ns,
            &mut self.anchor_pre_play,
            &mut self.seek_reanchor,
        );
        self.last_deadline_ns = Some(deadline);
        // gst video sinks take a frame without a duration to last one frame period of the caps
        let stream_duration = (timing.duration_ns > 0)
            .then_some(timing.duration_ns)
            .or(self.frame_period_ns);
        let duration = stream_duration.map(|stream_duration| {
            // gst basesink clips a frame's end to the segment before taking its running time
            let end_pts = pts_ns.saturating_add(stream_duration);
            let end_pts = self
                .segment
                .and_then(|segment| segment.stop)
                .map_or(end_pts, |stop| end_pts.min(stop));
            self.running_time(end_pts)
                .map_or(0, |end| end.saturating_sub(running_time))
        });
        let now = sync.now_ns();
        // A frame already late beyond the bound is dropped, not presented late,
        // so the sink catches up instead of accumulating lag. The same call
        // posts the periodic running-stats report when the frame is on time and
        // the interval has elapsed.
        if let Some(q) = self
            .qos
            .judge_frame(deadline, duration, now, running_time, presented)
        {
            if self.late_drops_in_a_row == 0 {
                crate::g2g_debug!(
                    Target::category(PACING_CATEGORY),
                    "dropping late frames from pts {}, {} past its deadline",
                    ClockTime(pts_ns),
                    ClockTime(now.saturating_sub(deadline))
                );
            }
            self.late_drops_in_a_row += 1;
            self.pending_qos = Some(q);
            self.last_upstream_report_ns = Some(now);
            return Pace::Drop;
        }
        if self.late_drops_in_a_row > 0 {
            crate::g2g_debug!(
                Target::category(PACING_CATEGORY),
                "presenting again at pts {} after dropping {} late frames",
                ClockTime(pts_ns),
                self.late_drops_in_a_row
            );
            self.late_drops_in_a_row = 0;
        }
        match self.last_upstream_report_ns {
            None => self.last_upstream_report_ns = Some(now),
            Some(last) if now.saturating_sub(last) >= UPSTREAM_QOS_INTERVAL_NS => {
                self.pending_qos = Some(upstream_report(deadline, now, running_time));
                self.last_upstream_report_ns = Some(now);
            }
            Some(_) => {}
        }
        match deadline.checked_sub(now) {
            Some(0) | None => Pace::Now,
            Some(wait) => Pace::Wait(wait),
        }
    }

    /// The deadline the most recent [`judge`](Self::judge) call measured its
    /// frame against, on the elected clock. A sink reads it once the frame is on
    /// screen and subtracts it from the clock then, giving the signed
    /// presentation error. `None` when that frame had no deadline: unpaced, an
    /// unset PTS, or clipped outside the segment.
    pub fn last_deadline_ns(&self) -> Option<u64> {
        self.last_deadline_ns
    }

    /// The QoS report a late drop left for the sink's
    /// [`take_qos`](crate::AsyncElement::take_qos), which the runner relays
    /// upstream so a source / decoder can shed load, or the report a presented
    /// frame leaves at most once per [`UPSTREAM_QOS_INTERVAL_NS`].
    pub fn take_qos(&mut self) -> Option<QosMessage> {
        self.pending_qos.take()
    }

    /// Attach the pipeline bus so QoS reports reach the application.
    pub fn set_bus(&mut self, bus: crate::BusHandle) {
        self.qos.set_bus(bus);
    }

    pub fn adopt_framerate(&mut self, caps: &Caps) {
        self.frame_period_ns = match caps.framerate() {
            Rate::Fixed(q16) => framerate_q16_period_ns(q16),
            _ => None,
        };
    }

    pub fn set_sync(&mut self, sync: bool) {
        self.present_on_arrival = !sync;
    }

    pub fn set_processing_deadline_ns(&mut self, ns: u64) {
        self.processing_deadline_ns = ns;
    }

    pub fn latency(&self) -> LatencyReport {
        // gstreamer adds the processing deadline only for a sink that syncs to the clock
        if self.present_on_arrival {
            return LatencyReport::ZERO;
        }
        LatencyReport::buffered(self.processing_deadline_ns, Some(0))
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

    /// Whether a frame whose display ends at `frame_end_ns` is too late at clock
    /// time `now_ns`, without judging it (no count, no report).
    pub fn is_too_late(&self, frame_end_ns: u64, now_ns: u64) -> bool {
        self.qos.is_too_late(frame_end_ns, now_ns)
    }

    /// Handle one of [`PACING_PROPERTIES`]; `None` means the name is not a
    /// pacing property, so the sink should carry on with its own.
    pub fn set_property(&mut self, name: &str, value: &PropValue) -> Option<Result<(), PropError>> {
        let ns = || value.as_uint().ok_or(PropError::Type);
        match name {
            "sync" => Some(
                value
                    .as_bool()
                    .ok_or(PropError::Type)
                    .map(|sync| self.set_sync(sync)),
            ),
            "max-lateness" => Some(ns().map(|n| self.set_max_lateness_ns(n))),
            "processing-deadline" => Some(ns().map(|n| self.set_processing_deadline_ns(n))),
            "qos-interval" => Some(ns().map(|n| self.set_report_interval_ns(n))),
            _ => None,
        }
    }

    /// Read one of [`PACING_PROPERTIES`]; `None` means the name is not a pacing
    /// property.
    pub fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "sync" => Some(PropValue::Bool(!self.present_on_arrival)),
            "max-lateness" => Some(PropValue::Uint(self.max_lateness_ns())),
            "processing-deadline" => Some(PropValue::Uint(self.processing_deadline_ns)),
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
    fn presentation_anchor(
        sync: &ClockSync,
        rt: u64,
        anchor_ns: &mut Option<i64>,
        anchor_pre_play: &mut bool,
        seek_reanchor: &mut bool,
    ) -> i64 {
        // A live path anchors on the base time, stamped or eager, never on the
        // first frame (GStreamer's model). A startup stall (decoder init) then
        // makes the opening frames late, rendered immediately, instead of
        // becoming the standing latency of the whole run. Reality cannot be
        // paused, so the file-pacing concern behind first-frame anchoring does
        // not apply to a live path.
        if sync.path_live() {
            return as_signed(sync.base_time());
        }
        // Re-base a provisional preroll anchor onto the play edge once `Playing`
        // has stamped the base time.
        if *anchor_pre_play && sync.play_anchored() && !*seek_reanchor {
            *anchor_ns = Some(as_signed(sync.base_time()));
            *anchor_pre_play = false;
        }
        if let Some(a) = *anchor_ns {
            return a;
        }
        // No anchor yet: establish one. Prefer the stamped play-edge base time
        // unless a seek just flushed (then present the target now).
        let use_play = sync.play_anchored() && !*seek_reanchor;
        let anchor = if use_play {
            as_signed(sync.base_time())
        } else {
            as_signed(sync.now_ns())
                .saturating_sub(as_signed(rt))
                .saturating_sub(as_signed(sync.paused_total_ns()))
        };
        *anchor_ns = Some(anchor);
        // Mark provisional only if anchored before `Playing` stamped, so it
        // re-bases later; a post-seek first-frame anchor is final.
        *anchor_pre_play = !use_play && !sync.play_anchored();
        *seek_reanchor = false;
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

    fn at(pts_ns: u64) -> FrameTiming {
        FrameTiming {
            pts_ns,
            ..FrameTiming::default()
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
        assert_eq!(p.judge(&at(5_000_000_000), 0), Pace::Now);
        assert_eq!(p.late_dropped(), 0);
    }

    #[test]
    fn a_live_path_anchors_on_base_time_not_the_first_frame() {
        let t = Arc::new(AtomicU64::new(0));
        let sync = ClockSync::new(Arc::new(ManualClock(t.clone())), 0)
            .with_path_latency(crate::query::LatencyReport::live(0, None));
        let mut p = PresentationPacer::new();
        p.set_clock_sync(sync);
        // A startup stall: the first frame (rt 0) reaches the sink at 200 ms.
        // Late against the base-time anchor, so it presents immediately
        // rather than becoming the anchor.
        t.store(200_000_000, Ordering::Relaxed);
        assert_eq!(p.judge(&at(0), 0), Pace::Now);
        // The backlog frame (rt 33 ms) is judged against the base too: still
        // late at 210 ms, presented immediately. Under first-frame anchoring
        // this would have been Wait(23 ms).
        t.store(210_000_000, Ordering::Relaxed);
        assert_eq!(p.judge(&at(33_000_000), 1), Pace::Now);
        // Once real time catches up to PTS the pacer holds frames again.
        t.store(220_000_000, Ordering::Relaxed);
        assert_eq!(p.judge(&at(300_000_000), 2), Pace::Wait(80_000_000));
    }

    #[test]
    fn a_live_path_adds_the_declared_minimum_latency() {
        let t = Arc::new(AtomicU64::new(0));
        let sync = ClockSync::new(Arc::new(ManualClock(t.clone())), 0)
            .with_path_latency(crate::query::LatencyReport::live(50_000_000, None));
        let mut p = PresentationPacer::new();
        p.set_clock_sync(sync);
        // rt 0 is due at base + declared min latency, not at base.
        assert_eq!(p.judge(&at(0), 0), Pace::Wait(50_000_000));
    }

    #[test]
    fn a_non_live_path_adds_no_latency() {
        let t = Arc::new(AtomicU64::new(0));
        let sync = ClockSync::new(Arc::new(ManualClock(t.clone())), 0)
            .with_path_latency(crate::query::LatencyReport::buffered(600_000_000, None));
        let mut p = PresentationPacer::new();
        p.set_clock_sync(sync);
        assert_eq!(p.judge(&at(0), 0), Pace::Now);
        assert_eq!(p.judge(&at(40_000_000), 1), Pace::Wait(40_000_000));
    }

    #[test]
    fn frames_wait_for_their_pts_deadline() {
        let (clock, sync) = manual();
        let mut p = PresentationPacer::new();
        p.set_clock_sync(sync);
        // First frame anchors at now (0), so it presents immediately.
        assert_eq!(p.judge(&at(0), 0), Pace::Now);
        // 33 ms of PTS later, with the clock still at 0: hold it 33 ms.
        assert_eq!(p.judge(&at(33_000_000), 1), Pace::Wait(33_000_000));
        // Once the clock reaches the deadline, no wait.
        clock.store(66_000_000, Ordering::Relaxed);
        assert_eq!(p.judge(&at(66_000_000), 2), Pace::Now);
    }

    #[test]
    fn sync_false_presents_early_and_late_frames_on_arrival() {
        let (clock, sync) = manual();
        let mut p = PresentationPacer::new();
        p.set_clock_sync(sync);
        p.set_max_lateness_ns(0);
        p.set_property("sync", &PropValue::Bool(false))
            .expect("a pacing property")
            .expect("takes a bool");
        assert_eq!(p.get_property("sync"), Some(PropValue::Bool(false)));
        assert_eq!(p.judge(&at(0), 0), Pace::Now);
        assert_eq!(p.judge(&at(33_000_000), 1), Pace::Now, "early, not held");
        clock.store(1_000_000_000, Ordering::Relaxed);
        assert_eq!(p.judge(&at(66_000_000), 2), Pace::Now, "late, not dropped");
        assert_eq!(p.late_dropped(), 0);
    }

    const FRAME_DURATION_NS: u64 = 33_000_000;

    fn lasting(pts_ns: u64) -> FrameTiming {
        FrameTiming {
            pts_ns,
            duration_ns: FRAME_DURATION_NS,
            ..FrameTiming::default()
        }
    }

    fn first_frame_judged_late(
        mut p: PresentationPacer,
        timing: FrameTiming,
        late_ns: u64,
    ) -> Pace {
        let (clock, sync) = manual();
        p.set_clock_sync(sync);
        // the first frame anchors on the clock at zero, so a frame at pts zero is due then
        assert_eq!(p.judge(&timing, 0), Pace::Now);
        clock.store(late_ns, Ordering::Relaxed);
        p.judge(&timing, 1)
    }

    fn video_sink_judging_the_first_frame_at(segment: Segment, late_ns: u64) -> Pace {
        let mut p = PresentationPacer::for_video_sink();
        p.set_segment(segment);
        first_frame_judged_late(p, lasting(0), late_ns)
    }

    #[test]
    fn a_frame_without_a_duration_lasts_one_period_of_the_caps_framerate() {
        let fps = 30;
        let caps = Caps::RawVideo {
            format: crate::RawVideoFormat::Rgba8,
            width: crate::Dim::Fixed(64),
            height: crate::Dim::Fixed(64),
            framerate: Rate::Fixed(fps << 16),
            interlace: crate::Interlace::Any,
            colorimetry: crate::Colorimetry::UNKNOWN,
            pixel_aspect: crate::PixelAspect::Any,
        };
        let judged_late = |late_ns| {
            let mut p = PresentationPacer::for_video_sink();
            p.adopt_framerate(&caps);
            first_frame_judged_late(p, at(0), late_ns)
        };
        let tolerated = crate::caps::fps_period_ns(fps) + VIDEO_SINK_MAX_LATENESS_NS;
        assert_eq!(judged_late(tolerated), Pace::Now);
        assert_eq!(judged_late(tolerated + 1), Pace::Drop);
    }

    #[test]
    fn a_video_sink_presents_a_late_frame_until_its_end_passes_the_bound() {
        let tolerated = FRAME_DURATION_NS + VIDEO_SINK_MAX_LATENESS_NS;
        assert_eq!(
            video_sink_judging_the_first_frame_at(Segment::new(), tolerated),
            Pace::Now
        );
        assert_eq!(
            video_sink_judging_the_first_frame_at(Segment::new(), tolerated + 1),
            Pace::Drop
        );
    }

    #[test]
    fn a_double_speed_segment_halves_the_frame_duration() {
        let double_speed = Segment {
            rate: 2.0,
            ..Segment::new()
        };
        let tolerated = FRAME_DURATION_NS / 2 + VIDEO_SINK_MAX_LATENESS_NS;
        assert_eq!(
            video_sink_judging_the_first_frame_at(double_speed, tolerated),
            Pace::Now
        );
        assert_eq!(
            video_sink_judging_the_first_frame_at(double_speed, tolerated + 1),
            Pace::Drop
        );
    }

    #[test]
    fn a_late_frame_drops_and_reports_both_ways() {
        let (clock, sync) = manual();
        let (bus, handle) = Bus::new(4);
        let mut p = PresentationPacer::new();
        p.set_clock_sync(sync);
        p.set_bus(handle);
        p.set_max_lateness_ns(0);
        // Anchor at clock 10 ms so a deadline and a running time differ.
        clock.store(10_000_000, Ordering::Relaxed);
        assert_eq!(p.judge(&at(0), 0), Pace::Now);
        assert!(p.take_qos().is_none(), "on-time frame reports nothing");

        // The clock jumps 1 ms past the next frame's deadline.
        clock.store(11_000_000, Ordering::Relaxed);
        assert_eq!(p.judge(&at(0), 1), Pace::Drop);
        assert_eq!(p.late_dropped(), 1);
        let upstream = p.take_qos().expect("upstream QoS report for the relay");
        assert_eq!(upstream.jitter_ns, 1_000_000);
        assert_eq!(
            upstream.running_time_ns, 0,
            "the running time, not the deadline"
        );
        assert!(p.take_qos().is_none(), "consumed once");
        match bus.try_recv() {
            Some(BusMessage::Qos {
                running_time_ns,
                jitter_ns,
                dropped,
                ..
            }) => assert_eq!((running_time_ns, jitter_ns, dropped), (0, 1_000_000, 1)),
            other => panic!("expected a Qos message, got {other:?}"),
        }
    }

    #[test]
    fn a_presented_frame_reports_upstream_once_per_interval() {
        let (clock, sync) = manual();
        let mut p = PresentationPacer::for_video_sink();
        p.set_clock_sync(sync);
        assert_eq!(p.judge(&at(0), 0), Pace::Now);
        assert!(p.take_qos().is_none(), "the first frame arms the interval");

        let half_interval = UPSTREAM_QOS_INTERVAL_NS / 2;
        clock.store(half_interval, Ordering::Relaxed);
        assert_eq!(
            p.judge(&at(UPSTREAM_QOS_INTERVAL_NS), 1),
            Pace::Wait(half_interval)
        );
        assert!(
            p.take_qos().is_none(),
            "nothing before the interval elapses"
        );

        let early_ns = 5_000_000;
        let running_time = UPSTREAM_QOS_INTERVAL_NS + early_ns;
        clock.store(UPSTREAM_QOS_INTERVAL_NS, Ordering::Relaxed);
        assert_eq!(p.judge(&at(running_time), 2), Pace::Wait(early_ns));
        let report = p.take_qos().expect("a periodic upstream report");
        assert_eq!(report.jitter_ns, -(early_ns as i64), "early is negative");
        assert_eq!(report.running_time_ns, running_time);
        assert_eq!(p.late_dropped(), 0);

        assert_eq!(
            p.judge(&at(running_time + early_ns), 3),
            Pace::Wait(2 * early_ns)
        );
        assert!(p.take_qos().is_none(), "the interval restarted");
    }

    #[test]
    fn the_default_bound_never_drops_however_late() {
        let (clock, sync) = manual();
        let mut p = PresentationPacer::new();
        p.set_clock_sync(sync);
        assert_eq!(p.judge(&at(0), 0), Pace::Now);
        clock.store(u64::MAX, Ordering::Relaxed);
        assert_eq!(p.judge(&at(0), 1), Pace::Now, "presented, however late");
        assert_eq!(p.late_dropped(), 0);
    }

    #[test]
    fn a_frame_with_no_pts_presents_now_and_does_not_anchor() {
        let (clock, sync) = manual();
        let mut p = PresentationPacer::new();
        p.set_clock_sync(sync);
        clock.store(10_000_000, Ordering::Relaxed);
        assert_eq!(p.judge(&at(FrameTiming::PTS_NONE), 0), Pace::Now);
        assert!(p.anchor_ns.is_none(), "an unset PTS latches no anchor");
        assert_eq!(p.deadline_ns(FrameTiming::PTS_NONE), None);
        assert!(p.anchor_ns.is_none());

        // The next real frame is the one that anchors, on itself: it presents
        // immediately rather than waiting out the PTS the no-pts frame skipped.
        assert_eq!(p.judge(&at(4_000_000), 1), Pace::Now);
        assert_eq!(p.anchor_ns, Some(6_000_000));
    }

    #[test]
    fn a_frame_with_no_pts_is_never_a_late_drop() {
        let (clock, sync) = manual();
        let mut p = PresentationPacer::new();
        p.set_clock_sync(sync);
        p.set_max_lateness_ns(0);
        assert_eq!(p.judge(&at(0), 0), Pace::Now);
        clock.store(u64::MAX, Ordering::Relaxed);
        assert_eq!(p.judge(&at(FrameTiming::PTS_NONE), 1), Pace::Now);
        assert_eq!(p.late_dropped(), 0);
        assert!(p.take_qos().is_none());
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
        assert_eq!(p.judge(&at(66_000_000), 0), Pace::Drop);
        assert_eq!(p.late_dropped(), 0);
        assert!(p.take_qos().is_none());
    }

    fn anchor_of(p: &mut PresentationPacer, sync: &ClockSync, rt: u64) -> i64 {
        PresentationPacer::presentation_anchor(
            sync,
            rt,
            &mut p.anchor_ns,
            &mut p.anchor_pre_play,
            &mut p.seek_reanchor,
        )
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
        assert_eq!(anchor_of(&mut p, &sync, 0), 1_000);
        assert!(p.anchor_pre_play, "provisional, awaiting Playing");

        // Playing stamps the base time at 5_000: the next frame re-bases onto the
        // play edge, discarding the preroll-time anchor.
        anchor.stamp(5_000);
        assert_eq!(anchor_of(&mut p, &sync, 100), 5_000);
        assert!(!p.anchor_pre_play, "re-base is final");

        // A seek flush forces a first-frame re-anchor (present the target now),
        // ignoring the stale play-edge base.
        clock.store(8_000, Ordering::Relaxed);
        p.flush();
        assert_eq!(anchor_of(&mut p, &sync, 200), 7_800);
        assert!(!p.seek_reanchor, "seek re-anchor consumed");
    }

    #[test]
    fn a_flush_after_a_repreroll_anchors_on_the_next_play_edge() {
        const PLAY_EDGE_NS: u64 = 5_000;
        const ARRIVAL_NS: u64 = 9_000;
        const RUNNING_TIME_NS: u64 = 100;
        let clock = Arc::new(AtomicU64::new(1_000));
        let anchor = PlayAnchor::new();
        let sync =
            ClockSync::with_play_anchor(Arc::new(ManualClock(clock.clone())), 0, anchor.clone());
        let mut p = PresentationPacer::new();
        p.set_clock_sync(sync.clone());

        anchor.clear();
        p.flush();
        anchor.stamp(PLAY_EDGE_NS);
        clock.store(ARRIVAL_NS, Ordering::Relaxed);

        assert_eq!(
            anchor_of(&mut p, &sync, RUNNING_TIME_NS),
            PLAY_EDGE_NS as i64,
            "a buffer arriving after the play edge anchors on its arrival"
        );
    }

    #[test]
    fn a_pause_moves_every_later_deadline_by_its_length() {
        let clock = Arc::new(AtomicU64::new(1_000));
        let anchor = PlayAnchor::new();
        let sync =
            ClockSync::with_play_anchor(Arc::new(ManualClock(clock.clone())), 0, anchor.clone());
        let mut p = PresentationPacer::new();
        p.set_clock_sync(sync);
        anchor.stamp(1_000);
        assert_eq!(p.judge(&at(0), 0), Pace::Now);

        let pause_ns = 8_000;
        anchor.pause(1_000);
        clock.store(1_000 + pause_ns, Ordering::Relaxed);
        anchor.resume(1_000 + pause_ns);

        let next_pts = 500;
        assert_eq!(p.judge(&at(next_pts), 1), Pace::Wait(next_pts));
    }

    #[test]
    fn a_pts_epoch_ahead_of_the_clock_still_presents_the_first_frame_now() {
        // An A/V graph slaved to an audio DriftClock reads the playout position
        // (a few ms in), while an HLS feed's PTS is the publisher's uptime (here
        // 5286 s). The first-frame anchor goes negative; clamping it at zero
        // would put every deadline 88 minutes out and nothing would present.
        let (clock, sync) = manual();
        clock.store(100_000_000, Ordering::Relaxed);
        let mut p = PresentationPacer::new();
        p.set_clock_sync(sync);

        let first_pts = 5_286_000_000_000;
        assert_eq!(p.judge(&at(first_pts), 0), Pace::Now);
        assert_eq!(p.last_deadline_ns(), Some(100_000_000));
        // And the stream paces by PTS deltas from there: 33 ms later is due 33 ms
        // later on the clock.
        assert_eq!(
            p.judge(&at(first_pts + 33_000_000), 1),
            Pace::Wait(33_000_000)
        );
    }

    #[test]
    fn judge_records_the_deadline_it_measured_against() {
        let (clock, sync) = manual();
        let mut p = PresentationPacer::new();
        assert_eq!(p.judge(&at(0), 0), Pace::Now);
        assert_eq!(
            p.last_deadline_ns(),
            None,
            "unpaced frames have no deadline"
        );

        p.set_clock_sync(sync);
        // First frame anchors at clock 0, so PTS 33 ms is due at clock 33 ms.
        assert_eq!(p.judge(&at(0), 0), Pace::Now);
        assert_eq!(p.last_deadline_ns(), Some(0));
        assert_eq!(p.judge(&at(33_000_000), 1), Pace::Wait(33_000_000));
        assert_eq!(p.last_deadline_ns(), Some(33_000_000));

        // The sink reads the clock once the frame is up: 2 ms past its deadline.
        clock.store(35_000_000, Ordering::Relaxed);
        let error = clock.load(Ordering::Relaxed) as i64 - p.last_deadline_ns().unwrap() as i64;
        assert_eq!(error, 2_000_000);

        // A frame with no deadline of its own clears it rather than leaving the
        // previous frame's for the sink to misattribute.
        assert_eq!(p.judge(&at(FrameTiming::PTS_NONE), 2), Pace::Now);
        assert_eq!(p.last_deadline_ns(), None);
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
