//! Presentation sink: waits until each frame's PTS arrives on the pipeline
//! clock before reporting the frame "presented". Records per-frame drift
//! (current clock minus PTS at presentation time) for end-to-end latency
//! analysis.
//!
//! Upstream backpressure naturally paces a free-running source: the source
//! can't push faster than the sink consumes, and the sink consumes no faster
//! than the clock advances toward each frame's PTS.
//!
//! Segment (M149): the sink tracks the playback `Segment` and maps each frame's
//! PTS to running time through it, so presentation follows running time (correct
//! after a seek resets the base) rather than raw PTS. A frame outside the segment
//! is clipped, which completes accurate seek: the source snaps upstream to the
//! keyframe before the target, the decoder decodes from there, and the sink drops
//! the decoded frames before the exact target so the first presented frame is the
//! requested one. Without a segment the sink uses PTS directly, as before.
//!
//! QoS (M85): when given a max-lateness bound, a frame whose deadline is
//! already past by more than that bound is dropped rather than presented late,
//! so the sink catches up instead of compounding the lag. The decision and its
//! reporting live in the shared [`PresentationPacer`]: each drop posts a
//! [`BusMessage::Qos`] to the pipeline bus if one was attached (the GStreamer
//! `GST_MESSAGE_QOS` analog), and a report interval adds the same running stats
//! periodically. Default behaviour is unchanged (no bound, no bus): every frame
//! is presented after its deadline.
//!
//! Clock (M996): the sink is a display sink without a display, so it paces
//! through the same [`PresentationPacer`] the real ones use and adopts the
//! elected [`ClockSync`] when the runner hands one over: in an A/V graph whose
//! audio sink provides the master `DriftClock`, presentation follows the audio
//! timeline rather than this sink's own clock. Until then it paces on its own
//! clock with the anchor pinned at zero, so a frame's deadline is its running
//! time and the recorded drift is a real end-to-end latency reading. The own
//! clock stays the timer either way (it is the only thing that can sleep in
//! `no_std`); the elected clock only decides *when* each frame is due.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::sync::Arc;

use g2g_core::{
    AsyncClock, AsyncElement, BusHandle, Caps, CapsConstraint, ClockSync, ConfigureOutcome,
    ElementBound, G2gError, OutputSink, Pace, PipelineClock, PipelinePacket, PresentationPacer,
    PropError, PropValue, PropertySpec, QosMessage, PACING_PROPERTIES,
};

#[derive(Debug)]
pub struct SyncSink<C: AsyncClock> {
    /// The sink's own clock: the timer every frame is held on, and the timeline
    /// deadlines are measured against until an elected clock replaces it.
    clock: Arc<C>,
    received: u64,
    last_sequence: Option<u64>,
    eos_seen: bool,
    configured: bool,
    max_drift_ns: u64,
    total_drift_ns: u128,
    /// Deadline, segment mapping and QoS verdict, shared with the display sinks.
    /// The default lateness bound never drops, so the sink presents every frame
    /// however late, preserving the pre-QoS behaviour.
    pacer: PresentationPacer,
    /// Set once the runner hands over an elected [`ClockSync`]. On our own clock
    /// deadlines are absolute (anchor pinned at zero) and a flush has nothing to
    /// re-anchor; on an elected one they are anchored from the first frame, so a
    /// seek must re-anchor or the target lands in the past.
    elected_clock_adopted: bool,
    /// Frames dropped because they fell outside the segment (accurate-seek clip).
    clipped: u64,
    /// Non-keyframe frames dropped under a trick-mode (`key_units_only`) segment.
    trick_dropped: u64,
}

impl<C: AsyncClock + Send + Sync + 'static> SyncSink<C> {
    pub fn new(clock: C) -> Self {
        let clock = Arc::new(clock);
        let mut pacer = PresentationPacer::new();
        // Pace on our own clock until election says otherwise, with deadlines
        // absolute from running-time zero rather than anchored on the first
        // frame: that is what makes the drift readings end-to-end latency.
        pacer.set_clock_sync(ClockSync::new(clock.clone(), 0));
        pacer.set_anchor_ns(0);
        Self {
            clock,
            received: 0,
            last_sequence: None,
            eos_seen: false,
            configured: false,
            max_drift_ns: 0,
            total_drift_ns: 0,
            pacer,
            elected_clock_adopted: false,
            clipped: 0,
            trick_dropped: 0,
        }
    }

    /// Enable QoS dropping: a frame already past its deadline by more than
    /// `ns` is dropped instead of presented late. `0` drops any frame that
    /// arrives after its deadline.
    pub fn with_max_lateness_ns(mut self, ns: u64) -> Self {
        self.pacer.set_max_lateness_ns(ns);
        self
    }

    /// Post a running-stats `Qos` report every `ns` of clock time while frames
    /// flow, on top of the per-drop reports. `0` (the default) reports only
    /// drops.
    pub fn with_qos_interval_ns(mut self, ns: u64) -> Self {
        self.pacer.set_report_interval_ns(ns);
        self
    }

    /// Attach the pipeline bus so QoS reports reach the application.
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.pacer.set_bus(bus);
        self
    }

    pub fn received(&self) -> u64 {
        self.received
    }

    /// The clock + base time presentation is paced against: the elected
    /// [`ClockSync`] once the runner hands one over, otherwise this sink's own
    /// clock at base time zero.
    pub fn clock_sync(&self) -> Option<&ClockSync> {
        self.pacer.clock_sync()
    }

    /// Whether the runner handed over an elected clock, so presentation follows
    /// the pipeline's master timeline (the audio `DriftClock` in an A/V graph)
    /// rather than this sink's own clock.
    pub fn slaved_to_elected_clock(&self) -> bool {
        self.elected_clock_adopted
    }

    /// Frames dropped because they arrived too late under the QoS bound.
    pub fn dropped(&self) -> u64 {
        self.pacer.late_dropped()
    }

    /// Non-keyframe frames dropped under a trick-mode (`key_units_only`) segment.
    pub fn trick_dropped(&self) -> u64 {
        self.trick_dropped
    }

    /// Frames clipped because they fell outside the current segment, eg the
    /// decoded frames before an accurate-seek target.
    pub fn clipped(&self) -> u64 {
        self.clipped
    }

    pub fn last_sequence(&self) -> Option<u64> {
        self.last_sequence
    }

    pub fn eos_seen(&self) -> bool {
        self.eos_seen
    }

    /// Largest single-frame drift observed: `clock.now_ns() - frame.pts_ns`
    /// at presentation time. Always non-negative because the sink sleeps
    /// until the deadline has passed.
    pub fn max_drift_ns(&self) -> u64 {
        self.max_drift_ns
    }

    pub fn mean_drift_ns(&self) -> u64 {
        if self.received == 0 {
            0
        } else {
            (self.total_drift_ns / u128::from(self.received))
                .try_into()
                .unwrap_or(u64::MAX)
        }
    }

    /// Current time on the timeline deadlines are measured against: the elected
    /// clock once adopted, otherwise our own.
    fn timeline_now_ns(&self) -> u64 {
        match self.pacer.clock_sync() {
            Some(sync) => sync.now_ns(),
            None => self.clock.now_ns(),
        }
    }
}

impl<C> AsyncElement for SyncSink<C>
where
    C: AsyncClock + ElementBound + Send + Sync + 'static,
{
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    /// M16 step 5c: wildcard sink. Same rationale as `FakeSink`.
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(f) => {
                    let pts = f.timing.pts_ns;
                    // Trick-mode KEY_UNIT: under a `key_units_only` segment, present
                    // only keyframes (fast scrub), dropping dependent frames before
                    // the deadline math so they are never scheduled.
                    if self
                        .pacer
                        .segment()
                        .is_some_and(|seg| seg.key_units_only && !f.timing.keyframe)
                    {
                        self.trick_dropped += 1;
                        return Ok(());
                    }
                    // The deadline maps PTS to running time through the segment
                    // (PTS directly when there is none) and adds the anchor. `None`
                    // means the frame is outside the segment, ie one of the decoded
                    // frames before an accurate-seek target: clip it.
                    let Some(deadline) = self.pacer.deadline_ns(pts) else {
                        self.clipped += 1;
                        return Ok(());
                    };
                    // QoS: a frame already past its deadline by more than the
                    // bound is dropped, not presented late, so the sink catches
                    // up, and the same lateness travels upstream via `take_qos`
                    // so the source / decoder sheds load (M174). The same call
                    // posts the periodic running-stats report when a frame is on
                    // time and the interval has elapsed.
                    let wait_ns = match self.pacer.judge(pts, self.received) {
                        Pace::Now => 0,
                        Pace::Wait(ns) => ns,
                        Pace::Drop => return Ok(()),
                    };
                    // The wait is relative because the deadline may be on the
                    // elected clock, while our own clock is the timer.
                    self.clock
                        .sleep_until_ns(self.clock.now_ns().saturating_add(wait_ns))
                        .await;
                    let drift = self.timeline_now_ns().saturating_sub(deadline);
                    self.max_drift_ns = self.max_drift_ns.max(drift);
                    self.total_drift_ns = self.total_drift_ns.saturating_add(u128::from(drift));
                    self.last_sequence = Some(f.sequence);
                    self.received += 1;
                }
                PipelinePacket::Eos => {
                    self.eos_seen = true;
                }
                PipelinePacket::Flush => {
                    // Seek flush: drop position so presentation resumes
                    // cleanly at the post-seek timeline. The post-flush Segment
                    // that follows installs the new running-time mapping.
                    self.last_sequence = None;
                    if self.elected_clock_adopted {
                        self.pacer.flush();
                    }
                }
                PipelinePacket::CapsChanged(_) => {}
                PipelinePacket::Segment(seg) => {
                    self.pacer.set_segment(seg);
                }
                // future PipelinePacket variants: no-op (terminal sink).
                _ => {}
            }
            Ok(())
        })
    }

    /// Adopt the elected clock + base time, so presentation follows the
    /// pipeline's master timeline (in an A/V graph, the audio sink's
    /// `DriftClock`) instead of this sink's own clock.
    fn set_clock_sync(&mut self, sync: ClockSync) {
        self.pacer.set_clock_sync(sync);
        self.elected_clock_adopted = true;
    }

    fn take_qos(&mut self) -> Option<QosMessage> {
        self.pacer.take_qos()
    }

    fn properties(&self) -> &'static [PropertySpec] {
        PACING_PROPERTIES
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        self.pacer
            .set_property(name, &value)
            .unwrap_or(Err(PropError::Unknown))
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        self.pacer.get_property(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::future::Ready;
    use core::sync::atomic::{AtomicU64, Ordering};
    use g2g_core::frame::Frame;
    use g2g_core::memory::SystemSlice;
    use g2g_core::{FrameTiming, MemoryDomain, PushOutcome, Seek, SeekFlags, SeekType, Segment};

    /// A clock fixed at 0 whose sleep resolves immediately (the deadline is in the
    /// future of `now == 0`, so no QoS drop fires and no real wait happens).
    struct InstantClock;
    impl g2g_core::PipelineClock for InstantClock {
        fn now_ns(&self) -> u64 {
            0
        }
    }
    impl AsyncClock for InstantClock {
        type SleepFuture<'a> = Ready<()>;
        fn sleep_until_ns(&self, _deadline_ns: u64) -> Ready<()> {
            core::future::ready(())
        }
    }

    struct NullSink;
    impl OutputSink for NullSink {
        fn push<'a>(
            &'a mut self,
            _packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async { Ok(PushOutcome::Accepted) })
        }
    }

    fn frame(pts_ns: u64, sequence: u64) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8]))),
            FrameTiming {
                pts_ns,
                ..FrameTiming::default()
            },
            sequence,
        ))
    }

    #[tokio::test]
    async fn clips_frames_before_the_segment_start() {
        let mut sink = SyncSink::new(InstantClock);
        sink.configure_pipeline(&Caps::ByteStream {
            encoding: g2g_core::ByteStreamEncoding::Ogg,
        })
        .unwrap();
        let mut out = NullSink;
        // Accurate seek to 70 ms: the source already snapped to the keyframe at
        // 66 ms, the decoder decoded from there, and this segment starts at 70 ms.
        let seg = Segment::for_flush_seek(&Seek::flush_to(70_000_000), None);
        sink.process(PipelinePacket::Segment(seg), &mut out)
            .await
            .unwrap();
        sink.process(frame(66_000_000, 0), &mut out).await.unwrap(); // pre-target: clipped
        sink.process(frame(100_000_000, 1), &mut out).await.unwrap(); // presented

        assert_eq!(sink.clipped(), 1, "the pre-target keyframe is clipped");
        assert_eq!(
            sink.received(),
            1,
            "only the at/after-target frame is presented"
        );
        assert_eq!(sink.last_sequence(), Some(1));
    }

    /// A keyframe-tagged frame for the trick-mode test.
    fn frame_kf(pts_ns: u64, sequence: u64, keyframe: bool) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8]))),
            FrameTiming {
                pts_ns,
                keyframe,
                ..FrameTiming::default()
            },
            sequence,
        ))
    }

    #[tokio::test]
    async fn trickmode_segment_presents_only_keyframes() {
        let mut sink = SyncSink::new(InstantClock);
        sink.configure_pipeline(&Caps::ByteStream {
            encoding: g2g_core::ByteStreamEncoding::MpegTs,
        })
        .unwrap();
        let mut out = NullSink;
        // A 2x trick-mode seek: the segment asks the sink for key units only.
        let seek = Seek {
            rate: 2.0,
            flags: SeekFlags::FLUSH | SeekFlags::TRICKMODE,
            start_type: SeekType::Set,
            start: 0,
            stop_type: SeekType::None,
            stop: 0,
        };
        let seg = Segment::for_flush_seek(&seek, None);
        assert!(seg.key_units_only, "the TRICKMODE flag set key_units_only");
        sink.process(PipelinePacket::Segment(seg), &mut out)
            .await
            .unwrap();

        sink.process(frame_kf(0, 0, true), &mut out).await.unwrap(); // keyframe: presented
        sink.process(frame_kf(20_000_000, 1, false), &mut out)
            .await
            .unwrap(); // dropped
        sink.process(frame_kf(40_000_000, 2, false), &mut out)
            .await
            .unwrap(); // dropped
        sink.process(frame_kf(60_000_000, 3, true), &mut out)
            .await
            .unwrap(); // keyframe: presented

        assert_eq!(sink.received(), 2, "only the two keyframes are presented");
        assert_eq!(sink.trick_dropped(), 2, "the dependent frames are dropped");
        assert_eq!(sink.last_sequence(), Some(3));
    }

    /// The pipeline's elected master clock, moved by hand (an audio `DriftClock`
    /// in a real A/V graph).
    #[derive(Debug)]
    struct ElectedClock(Arc<AtomicU64>);
    impl g2g_core::PipelineClock for ElectedClock {
        fn now_ns(&self) -> u64 {
            self.0.load(Ordering::Relaxed)
        }
    }

    /// M996: once the runner hands over the elected `ClockSync`, deadlines are
    /// measured on *that* timeline. The sink's own clock is stuck at 0, so
    /// nothing could ever be late on it; the late drop below can only come from
    /// the elected clock's reading.
    #[tokio::test]
    async fn adopting_the_elected_clock_moves_deadlines_onto_its_timeline() {
        let elected = Arc::new(AtomicU64::new(5_000_000_000));
        let mut sink = SyncSink::new(InstantClock).with_max_lateness_ns(0);
        sink.configure_pipeline(&Caps::ByteStream {
            encoding: g2g_core::ByteStreamEncoding::Ogg,
        })
        .unwrap();
        AsyncElement::set_clock_sync(
            &mut sink,
            ClockSync::new(Arc::new(ElectedClock(elected.clone())), 5_000_000_000),
        );
        assert!(sink.slaved_to_elected_clock());
        let mut out = NullSink;

        // First frame anchors on the elected clock's epoch, so it is on time.
        sink.process(frame(0, 0), &mut out).await.unwrap();
        // 40 ms of PTS later with the elected clock 40 ms on: still on time.
        elected.store(5_040_000_000, Ordering::Relaxed);
        sink.process(frame(40_000_000, 1), &mut out).await.unwrap();
        // The elected clock jumps 120 ms past the next frame's deadline.
        elected.store(5_200_000_000, Ordering::Relaxed);
        sink.process(frame(80_000_000, 2), &mut out).await.unwrap();

        assert_eq!(sink.received(), 2, "the two on-time frames presented");
        assert_eq!(sink.dropped(), 1, "the late one dropped");
        assert_eq!(
            AsyncElement::take_qos(&mut sink).map(|q| q.jitter_ns),
            Some(120_000_000),
            "lateness measured on the elected timeline travels upstream"
        );
    }

    #[tokio::test]
    async fn without_segment_presents_every_frame() {
        let mut sink = SyncSink::new(InstantClock);
        sink.configure_pipeline(&Caps::ByteStream {
            encoding: g2g_core::ByteStreamEncoding::Ogg,
        })
        .unwrap();
        let mut out = NullSink;
        sink.process(frame(0, 0), &mut out).await.unwrap();
        sink.process(frame(50_000_000, 1), &mut out).await.unwrap();
        assert_eq!(sink.clipped(), 0);
        assert_eq!(
            sink.received(),
            2,
            "no segment: PTS is the running time, nothing clipped"
        );
    }
}
