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
//! so the sink catches up instead of compounding the lag. Each drop posts a
//! [`BusMessage::Qos`] to the pipeline bus if one was attached, the GStreamer
//! `GST_MESSAGE_QOS` analog. Default behaviour is unchanged (no bound, no bus):
//! every frame is presented after its deadline.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;

use g2g_core::{
    AsyncClock, AsyncElement, BusHandle, BusMessage, Caps, CapsConstraint, ConfigureOutcome,
    ElementBound, G2gError, OutputSink, PipelinePacket, QosMessage, Segment,
};

#[derive(Debug)]
pub struct SyncSink<C: AsyncClock> {
    clock: C,
    received: u64,
    last_sequence: Option<u64>,
    eos_seen: bool,
    configured: bool,
    max_drift_ns: u64,
    total_drift_ns: u128,
    /// QoS: drop a frame whose deadline is already past by more than this many
    /// ns. `u64::MAX` (the default) never drops, so the sink presents every
    /// frame however late, preserving the pre-QoS behaviour.
    max_lateness_ns: u64,
    dropped: u64,
    /// The current playback segment, set from `PipelinePacket::Segment`. Maps a
    /// frame's PTS to running time and clips frames outside it (the
    /// decode-to-target frames after an accurate seek). `None` before any segment
    /// arrives, where PTS is used directly as running time.
    segment: Option<Segment>,
    /// Frames dropped because they fell outside the segment (accurate-seek clip).
    clipped: u64,
    /// Non-keyframe frames dropped under a trick-mode (`key_units_only`) segment.
    trick_dropped: u64,
    bus: Option<BusHandle>,
    /// QoS signal pending delivery upstream (M174): set when a late frame is
    /// dropped, consumed by the runner via [`take_qos`](AsyncElement::take_qos)
    /// and forwarded onto the incoming link so the source can shed load.
    pending_qos: Option<QosMessage>,
}

impl<C: AsyncClock> SyncSink<C> {
    pub fn new(clock: C) -> Self {
        Self {
            clock,
            received: 0,
            last_sequence: None,
            eos_seen: false,
            configured: false,
            max_drift_ns: 0,
            total_drift_ns: 0,
            max_lateness_ns: u64::MAX,
            dropped: 0,
            segment: None,
            clipped: 0,
            trick_dropped: 0,
            bus: None,
            pending_qos: None,
        }
    }

    /// Enable QoS dropping: a frame already past its deadline by more than
    /// `ns` is dropped instead of presented late. `0` drops any frame that
    /// arrives after its deadline.
    pub fn with_max_lateness_ns(mut self, ns: u64) -> Self {
        self.max_lateness_ns = ns;
        self
    }

    /// Attach the pipeline bus so QoS drops post a [`BusMessage::Qos`].
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.bus = Some(bus);
        self
    }

    pub fn received(&self) -> u64 {
        self.received
    }

    /// Frames dropped because they arrived too late under the QoS bound.
    pub fn dropped(&self) -> u64 {
        self.dropped
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
}

impl<C> AsyncElement for SyncSink<C>
where
    C: AsyncClock + ElementBound,
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
                    if self.segment.as_ref().is_some_and(|s| s.key_units_only) && !f.timing.keyframe
                    {
                        self.trick_dropped += 1;
                        return Ok(());
                    }
                    // Map PTS to running time through the segment; a frame outside
                    // it (the decoded frames before an accurate-seek target) is
                    // clipped. Without a segment, PTS is the running time directly.
                    let deadline = match &self.segment {
                        Some(seg) => match seg.to_running_time(pts) {
                            Some(rt) => rt,
                            None => {
                                self.clipped += 1;
                                return Ok(());
                            }
                        },
                        None => pts,
                    };
                    // QoS: a frame already past its deadline by more than the
                    // bound is dropped, not presented late, so the sink catches
                    // up. `now > deadline + bound` (saturating, so the u64::MAX
                    // default never fires).
                    let now = self.clock.now_ns();
                    if now > deadline.saturating_add(self.max_lateness_ns) {
                        self.dropped += 1;
                        let jitter = i64::try_from(now - deadline).unwrap_or(i64::MAX);
                        if let Some(bus) = &self.bus {
                            // Control message: non-blocking, never stalls the
                            // sink (a full bus drops the report).
                            bus.try_post(BusMessage::Qos {
                                running_time_ns: deadline,
                                jitter_ns: jitter,
                                processed: self.received,
                                dropped: self.dropped,
                            });
                        }
                        // M174: signal the same lateness upstream so the source /
                        // decoder sheds load. The runner picks this up via
                        // `take_qos` after `process` and forwards it.
                        self.pending_qos = Some(QosMessage {
                            jitter_ns: jitter,
                            running_time_ns: deadline,
                        });
                        return Ok(());
                    }
                    self.clock.sleep_until_ns(deadline).await;
                    let drift = self.clock.now_ns().saturating_sub(deadline);
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
                }
                PipelinePacket::CapsChanged(_) => {}
                PipelinePacket::Segment(seg) => {
                    self.segment = Some(seg);
                }
                // future PipelinePacket variants: no-op (terminal sink).
                _ => {}
            }
            Ok(())
        })
    }

    fn take_qos(&mut self) -> Option<QosMessage> {
        self.pending_qos.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::future::Ready;
    use g2g_core::frame::Frame;
    use g2g_core::memory::SystemSlice;
    use g2g_core::{FrameTiming, MemoryDomain, PushOutcome, Seek, SeekFlags, SeekType};

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
