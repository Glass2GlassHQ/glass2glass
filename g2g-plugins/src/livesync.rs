//! Live gap filler (`livesync`). Passes raw video or PCM audio through and, when
//! the input stalls, keeps the output going at the input's own cadence: the last
//! video frame repeated at the next PTS, or a silence buffer the size and length
//! of the last audio one.
//!
//! It is a one-input fan-in because the deadline tick only reaches a fan-in arm
//! ([`MultiInputElement::tick_interval_ns`]), and a stall is exactly the case
//! where no packet arrives to drive the element.
//!
//! A frame behind the timeline already emitted is dropped, unless it is more than
//! `late-threshold` behind, in which case the timeline follows it instead: an
//! upstream that restarts its clock recovers rather than being dropped forever.
//!
//! `std` only: the fill deadline measures against
//! [`g2g_core::metrics::monotonic_ns`].

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;

use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, Frame, G2gError, MultiInputElement,
    OutputSink, PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate,
};

use crate::audioconvert::silence_byte;
use crate::compositor::frame_period_ns;

/// gst's `latency` default: no extra slack before a missing buffer is filled in.
const DEFAULT_LATENCY_NS: u64 = 0;
const DEFAULT_LATENCY_TEXT: &str = "0";

/// gst's `late-threshold` default: two seconds behind the output timeline before
/// the element follows the input instead of dropping it.
const DEFAULT_LATE_THRESHOLD_NS: u64 = 2_000_000_000;
const DEFAULT_LATE_THRESHOLD_TEXT: &str = "2000000000";

/// The `late-threshold` value that never resynchronises, gst's `-1` on a
/// `guint64`.
pub const LATE_THRESHOLD_NEVER: u64 = u64::MAX;

/// Tick period used before the negotiated caps or a first buffer name a cadence.
/// Fine enough that the first fill after a stall is not visibly late at any
/// ordinary frame rate.
const UNKNOWN_CADENCE_TICK_NS: u64 = 10_000_000;

/// The counters read back through `in` / `drop` / `out` / `duplicate`.
const COUNTER_DEFAULT_TEXT: &str = "0";

/// # Example
///
/// ```no_run
/// use g2g_plugins::livesync::LiveSync;
///
/// // allow upstream 100 ms of slack before a missing buffer is filled in
/// let sync = LiveSync::new().with_latency_ns(100_000_000);
/// ```
#[derive(Debug)]
pub struct LiveSync {
    latency_ns: u64,
    late_threshold_ns: u64,
    /// The negotiated input caps, which are the output caps too.
    configured: Option<Caps>,
    /// Byte a silent sample of the configured PCM format is made of, `None` on
    /// video.
    silence: Option<u8>,
    /// The last emitted frame, the template a video filler repeats.
    last_frame: Option<Frame>,
    /// Byte length of the last emitted buffer, the size an audio filler takes.
    last_bytes: usize,
    /// Span of the last emitted buffer, and so of one filler.
    last_duration_ns: u64,
    /// PTS the next emitted buffer carries, `None` until the first one.
    next_pts_ns: Option<u64>,
    /// Wall-clock time the buffer at `next_pts_ns` is due at.
    next_due_ns: u64,
    /// The caps last pushed downstream, so an unchanged re-announcement is
    /// suppressed.
    emitted_caps: Option<Caps>,
    frames_in: u64,
    frames_dropped: u64,
    frames_out: u64,
    frames_duplicated: u64,
}

impl Default for LiveSync {
    fn default() -> Self {
        Self::new()
    }
}

impl LiveSync {
    pub fn new() -> Self {
        Self {
            latency_ns: DEFAULT_LATENCY_NS,
            late_threshold_ns: DEFAULT_LATE_THRESHOLD_NS,
            configured: None,
            silence: None,
            last_frame: None,
            last_bytes: 0,
            last_duration_ns: 0,
            next_pts_ns: None,
            next_due_ns: 0,
            emitted_caps: None,
            frames_in: 0,
            frames_dropped: 0,
            frames_out: 0,
            frames_duplicated: 0,
        }
    }

    /// Nanoseconds of slack upstream gets past a buffer's due time before the
    /// gap is filled (the `latency` property).
    pub fn with_latency_ns(mut self, latency_ns: u64) -> Self {
        self.latency_ns = latency_ns;
        self
    }

    /// Nanoseconds a buffer may be behind the output timeline before the
    /// timeline follows it instead of dropping it (the `late-threshold`
    /// property). [`LATE_THRESHOLD_NEVER`] never follows.
    pub fn with_late_threshold_ns(mut self, late_threshold_ns: u64) -> Self {
        self.late_threshold_ns = late_threshold_ns;
        self
    }

    /// Buffers accepted from the input (the `in` property).
    pub fn frames_in(&self) -> u64 {
        self.frames_in
    }

    /// Buffers discarded for being behind the output timeline (`drop`).
    pub fn frames_dropped(&self) -> u64 {
        self.frames_dropped
    }

    /// Buffers emitted downstream, real and filler both (`out`).
    pub fn frames_out(&self) -> u64 {
        self.frames_out
    }

    /// Filler buffers among those emitted (`duplicate`).
    pub fn frames_duplicated(&self) -> u64 {
        self.frames_duplicated
    }

    /// Raw video and PCM audio are the two things a filler can be made of: a
    /// repeated picture and a silent sample run.
    fn fillable(caps: &Caps) -> bool {
        match caps {
            Caps::RawVideo { .. } => true,
            Caps::Audio { .. } => caps.is_raw_media(),
            _ => false,
        }
    }

    fn check_fillable(caps: &Caps) -> Result<(), G2gError> {
        if Self::fillable(caps) {
            Ok(())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    /// One video frame period from the negotiated caps, `None` on audio or on a
    /// framerate of zero.
    fn video_period_ns(&self) -> Option<u64> {
        match self.configured {
            Some(Caps::RawVideo {
                framerate: Rate::Fixed(q16),
                ..
            }) => Some(frame_period_ns(q16)).filter(|period| *period > 0),
            _ => None,
        }
    }

    fn adopt(&mut self, caps: &Caps) {
        self.silence = match caps {
            Caps::Audio { format, .. } => Some(silence_byte(*format)),
            _ => None,
        };
        self.configured = Some(caps.clone());
    }

    /// Anchor the output timeline on `pts` arriving at `now_ns`, so the buffer
    /// after it is due one duration later.
    fn advance(&mut self, pts_ns: u64, duration_ns: u64, now_ns: u64) {
        self.last_duration_ns = duration_ns;
        self.next_pts_ns = Some(pts_ns.saturating_add(duration_ns));
        self.next_due_ns = now_ns.saturating_add(duration_ns);
    }

    /// The filler for the negotiated media: the last picture again, or a silent
    /// run of the last buffer's size. `None` before the first buffer, and for
    /// audio whose bytes never lived in system memory.
    fn filler(&self) -> Option<Frame> {
        match self.silence {
            Some(byte) => {
                let last = self.last_frame.as_ref()?;
                if self.last_bytes == 0 {
                    return None;
                }
                let bytes = vec![byte; self.last_bytes].into_boxed_slice();
                Some(Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(bytes)),
                    last.timing,
                    last.sequence,
                ))
            }
            None => self.last_frame.as_ref().map(Frame::share),
        }
    }

    /// Emit the buffers whose due time has passed while the input delivered
    /// nothing, so the output timeline stays contiguous over a stall.
    async fn fill_due(&mut self, now_ns: u64, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        if self.last_duration_ns == 0 {
            return Ok(());
        }
        while let Some(pts_ns) = self.next_pts_ns {
            if now_ns < self.next_due_ns.saturating_add(self.latency_ns) {
                return Ok(());
            }
            let Some(mut frame) = self.filler() else {
                return Ok(());
            };
            let duration_ns = self.last_duration_ns;
            frame.timing.pts_ns = pts_ns;
            frame.timing.dts_ns = pts_ns;
            frame.timing.duration_ns = duration_ns;
            let due = self.next_due_ns;
            self.advance(pts_ns, duration_ns, due);
            self.frames_duplicated += 1;
            self.emit(frame, out).await?;
        }
        Ok(())
    }

    async fn emit(&mut self, mut frame: Frame, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        // Without this the held copy would deep-copy every buffer's bytes.
        frame.domain.make_shareable();
        self.last_bytes = frame
            .domain
            .as_system_slice()
            .map_or(self.last_bytes, |bytes| bytes.len());
        self.last_frame = Some(frame.share());
        self.frames_out += 1;
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }

    async fn accept(
        &mut self,
        frame: Frame,
        now_ns: u64,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        self.frames_in += 1;
        // An unstamped buffer takes the slot the timeline expects next, which is
        // what a source with no clock of its own leaves for this element to say.
        let pts_ns = frame
            .timing
            .pts()
            .unwrap_or_else(|| self.next_pts_ns.unwrap_or_default());
        if let Some(next_pts_ns) = self.next_pts_ns {
            let behind_ns = next_pts_ns.saturating_sub(pts_ns);
            let resync = self.late_threshold_ns != LATE_THRESHOLD_NEVER
                && behind_ns > self.late_threshold_ns;
            if pts_ns < next_pts_ns && !resync {
                self.frames_dropped += 1;
                return Ok(());
            }
        }
        let duration_ns = match frame.timing.duration_ns {
            0 => self.video_period_ns().unwrap_or(self.last_duration_ns),
            stamped => stamped,
        };
        self.advance(pts_ns, duration_ns, now_ns);
        self.emit(frame, out).await
    }

    /// [`process`](MultiInputElement::process) with "now" passed in, for a
    /// caller driving the fill deadline off its own clock instead of the process
    /// monotonic one.
    pub async fn handle(
        &mut self,
        _input: usize,
        packet: PipelinePacket,
        now_ns: u64,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        match packet {
            // The runner aggregates input ends and emits the merged Eos itself.
            PipelinePacket::Eos => Ok(()),
            PipelinePacket::Tick => self.fill_due(now_ns, out).await,
            PipelinePacket::DataFrame(frame) => self.accept(frame, now_ns, out).await,
            PipelinePacket::CapsChanged(caps) => {
                Self::check_fillable(&caps)?;
                self.adopt(&caps);
                if self.emitted_caps.as_ref() == Some(&caps) {
                    return Ok(());
                }
                self.emitted_caps = Some(caps.clone());
                out.push(PipelinePacket::CapsChanged(caps)).await?;
                Ok(())
            }
            other => {
                out.push(other).await?;
                Ok(())
            }
        }
    }
}

impl MultiInputElement for LiveSync {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// gst's `livesync` has one sink pad. The fan-in shape is what earns the
    /// deadline tick, not a second input.
    fn input_count(&self) -> usize {
        1
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Self::check_fillable(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn output_follows_input(&self) -> Option<usize> {
        Some(0)
    }

    /// One output buffer period, so a stall is noticed within one buffer of when
    /// the next one was due: the frame period for video, the last buffer's span
    /// for audio, and a fixed fallback until either is known.
    fn tick_interval_ns(&self) -> Option<u64> {
        self.video_period_ns()
            .or(Some(self.last_duration_ns).filter(|span| *span > 0))
            .or(Some(UNKNOWN_CADENCE_TICK_NS))
    }

    fn configure_pipeline(
        &mut self,
        _input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        Self::check_fillable(absolute_caps)?;
        self.adopt(absolute_caps);
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        self.configured.clone().ok_or(G2gError::NotConfigured)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Live sync",
            "Filter",
            "Fills a stalled live stream with repeated frames or audio silence",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        LIVESYNC_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "latency" => self.latency_ns = value.as_uint().ok_or(PropError::Type)?,
            "late-threshold" => self.late_threshold_ns = value.as_uint().ok_or(PropError::Type)?,
            "in" | "drop" | "out" | "duplicate" => return Err(PropError::ReadOnly),
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "latency" => Some(PropValue::Uint(self.latency_ns)),
            "late-threshold" => Some(PropValue::Uint(self.late_threshold_ns)),
            "in" => Some(PropValue::Uint(self.frames_in)),
            "drop" => Some(PropValue::Uint(self.frames_dropped)),
            "out" => Some(PropValue::Uint(self.frames_out)),
            "duplicate" => Some(PropValue::Uint(self.frames_duplicated)),
            _ => None,
        }
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            let now_ns = g2g_core::metrics::monotonic_ns();
            self.handle(input, packet, now_ns, out).await
        })
    }
}

/// `LiveSync`'s properties, named and defaulted as gst's `livesync`.
static LIVESYNC_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "latency",
        PropKind::Uint,
        "nanoseconds of slack upstream gets past a buffer's due time",
    )
    .with_default(DEFAULT_LATENCY_TEXT),
    PropertySpec::new(
        "late-threshold",
        PropKind::Uint,
        "nanoseconds behind the output timeline before it follows the input instead of dropping it",
    )
    .with_default(DEFAULT_LATE_THRESHOLD_TEXT),
    PropertySpec::new("in", PropKind::Uint, "buffers accepted from the input")
        .with_default(COUNTER_DEFAULT_TEXT)
        .read_only(),
    PropertySpec::new(
        "drop",
        PropKind::Uint,
        "buffers discarded for being behind the output timeline",
    )
    .with_default(COUNTER_DEFAULT_TEXT)
    .read_only(),
    PropertySpec::new("out", PropKind::Uint, "buffers emitted downstream")
        .with_default(COUNTER_DEFAULT_TEXT)
        .read_only(),
    PropertySpec::new("duplicate", PropKind::Uint, "filler buffers emitted")
        .with_default(COUNTER_DEFAULT_TEXT)
        .read_only(),
];

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use g2g_core::{
        AudioFormat, ChannelLayout, Colorimetry, Dim, FrameTiming, Interlace, PushOutcome,
        RawVideoFormat, VideoCodec,
    };

    const FPS: u32 = 30;

    fn period_ns() -> u64 {
        frame_period_ns(FPS << 16)
    }

    #[derive(Default)]
    struct CollectSink {
        pts: Vec<u64>,
    }
    impl OutputSink for CollectSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            if let PipelinePacket::DataFrame(frame) =
                packet_slot.take().expect("poll_push without a packet")
            {
                self.pts.push(frame.timing.pts_ns);
            }
            core::task::Poll::Ready(Ok(PushOutcome::Accepted))
        }
    }

    fn video() -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(2),
            height: Dim::Fixed(2),
            framerate: Rate::Fixed(FPS << 16),
            interlace: Interlace::Any,
            colorimetry: Colorimetry::UNKNOWN,
        }
    }

    fn frame(pts_ns: u64) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(vec![7u8; 16].into_boxed_slice())),
            FrameTiming {
                pts_ns,
                duration_ns: period_ns(),
                ..FrameTiming::default()
            },
            pts_ns / period_ns(),
        ))
    }

    #[tokio::test]
    async fn a_stall_is_filled_at_the_input_cadence() {
        let mut sync = LiveSync::new();
        sync.configure_pipeline(0, &video()).unwrap();
        let mut out = CollectSink::default();
        sync.handle(0, frame(0), 0, &mut out).await.unwrap();
        // Three periods with nothing arriving: the tick fills every slot.
        sync.handle(0, PipelinePacket::Tick, 3 * period_ns(), &mut out)
            .await
            .unwrap();
        let expected: Vec<u64> = (0..4).map(|slot| slot * period_ns()).collect();
        assert_eq!(out.pts, expected);
        assert_eq!(sync.frames_duplicated(), 3);
        assert_eq!(sync.frames_out(), 4);
    }

    #[test]
    fn tick_period_falls_back_before_the_caps_arrive() {
        let sync = LiveSync::new();
        assert_eq!(
            MultiInputElement::tick_interval_ns(&sync),
            Some(UNKNOWN_CADENCE_TICK_NS)
        );
        let mut sync = LiveSync::new();
        sync.configure_pipeline(0, &video()).unwrap();
        assert_eq!(
            MultiInputElement::tick_interval_ns(&sync),
            Some(period_ns())
        );
    }

    #[test]
    fn only_raw_video_and_pcm_audio_are_fillable() {
        let sync = LiveSync::new();
        assert!(sync.intercept_caps(0, &video()).is_ok());
        let pcm = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: ChannelLayout::UNSPECIFIED,
        };
        assert!(sync.intercept_caps(0, &pcm).is_ok());
        let opus = Caps::Audio {
            format: AudioFormat::Opus,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: ChannelLayout::UNSPECIFIED,
        };
        assert_eq!(sync.intercept_caps(0, &opus), Err(G2gError::CapsMismatch));
        let h264 = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(2),
            height: Dim::Fixed(2),
            framerate: Rate::Fixed(FPS << 16),
            colorimetry: Colorimetry::UNKNOWN,
        };
        assert_eq!(sync.intercept_caps(0, &h264), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn counters_are_read_only() {
        let mut sync = LiveSync::new();
        for name in ["in", "drop", "out", "duplicate"] {
            assert_eq!(
                sync.set_property(name, PropValue::Uint(1)),
                Err(PropError::ReadOnly)
            );
        }
        assert_eq!(
            sync.set_property("latency", PropValue::Bool(true)),
            Err(PropError::Type)
        );
    }
}
