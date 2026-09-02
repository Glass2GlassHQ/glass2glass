//! Playback-rate changer without pitch correction (`speed`). Resamples each
//! channel by linear interpolation, so `speed=2` consumes two input frames per
//! output frame and the stream plays twice as fast an octave higher. The caps
//! rate is unchanged: what changes is the sample count, and with it the output
//! duration, which scales by `1 / speed`. CPU-only, `no_std` baseline.
//!
//! [`ScaleTempo`](crate::scaletempo::ScaleTempo) is the pitch-preserving
//! sibling and takes its rate from the segment; this one is driven by its own
//! `speed` property, as gst's `speed` is.
//!
//! Matches GStreamer's `speed`: the read position starts at
//! `0.5 * (speed - 1)`, advances by `speed` per output frame, and each output
//! frame interpolates between the previous read sample and the one at
//! `ceil(position)`. That means the reference lags by one frame at `speed=1`,
//! and this port does too rather than special-casing the transfer function.
//!
//! Timestamps run off a running output-frame counter from the segment's start
//! divided by `speed`, and the forwarded segment carries the same division on
//! `start`, `stop` and `position`, so downstream sees an ordinary stream on the
//! compressed timeline.
//!
//! Interleaved `PcmS16Le` / `PcmF32Le` only, the audiofx format set, so
//! `audioconvert` is placed ahead for anything else.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError,
    MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec,
};

use crate::audioconvert::samples_to_ns;
use crate::audiofx;
use crate::mathf;

/// gst's `speed` bounds and default.
const DEFAULT_SPEED: f64 = 1.0;
const SPEED_MIN: f64 = 0.1;
const SPEED_MAX: f64 = 40.0;
const DEFAULT_SPEED_TEXT: &str = "1";

/// # Example
///
/// ```no_run
/// use g2g_plugins::speed::Speed;
///
/// let double = Speed::new().with_speed(2.0);
/// ```
#[derive(Debug)]
pub struct Speed {
    speed: f64,
    format: AudioFormat,
    channels: usize,
    sample_rate: u32,
    caps: Option<Caps>,
    last_caps: Option<Caps>,
    /// Start of the output timeline: the segment's start (or the first frame's
    /// pts) divided by `speed`.
    origin_ns: Option<u64>,
    emitted_frames: u64,
    emitted: u64,
}

impl Default for Speed {
    fn default() -> Self {
        Self::new()
    }
}

impl Speed {
    pub fn new() -> Self {
        Self {
            speed: DEFAULT_SPEED,
            format: AudioFormat::PcmS16Le,
            channels: 0,
            sample_rate: 0,
            caps: None,
            last_caps: None,
            origin_ns: None,
            emitted_frames: 0,
            emitted: 0,
        }
    }

    pub fn with_speed(mut self, speed: f64) -> Self {
        self.speed = speed.clamp(SPEED_MIN, SPEED_MAX);
        self
    }

    fn configure(&mut self, caps: &Caps) -> Result<(), G2gError> {
        let (format, channels, rate) = audiofx::accept_audio(caps, None)?;
        self.format = format;
        self.channels = channels;
        self.sample_rate = rate;
        self.caps = Some(caps.clone());
        Ok(())
    }

    fn reset_stream(&mut self) {
        self.origin_ns = None;
        self.emitted_frames = 0;
    }

    /// A timestamp on the compressed timeline.
    fn map_ns(&self, ns: u64) -> u64 {
        (ns as f64 / self.speed) as u64
    }

    /// Resample one interleaved buffer. Each channel is walked independently,
    /// as in the reference, so they stay phase-aligned on the same grid.
    fn resample(&self, input: &[f32]) -> Vec<f32> {
        let channels = self.channels;
        if channels == 0 || input.len() < channels {
            return Vec::new();
        }
        let input_frames = input.len() / channels;
        let mut out = Vec::new();
        for channel in 0..channels {
            let mut lower = input[channel] as f64;
            let mut position = 0.5 * (self.speed - 1.0);
            let mut index = mathf::ceil(position);
            let mut produced = 0usize;
            while index < input_frames as f64 {
                let read = if index < 0.0 { 0 } else { index as usize };
                let interpolation = position - mathf::floor(position);
                let upper = input[read * channels + channel] as f64;
                let value = (lower * (1.0 - interpolation) + upper * interpolation) as f32;
                let slot = produced * channels + channel;
                if slot >= out.len() {
                    out.resize(slot + channels, 0.0);
                }
                out[slot] = value;
                lower = upper;
                position += self.speed;
                index = mathf::ceil(position);
                produced += 1;
            }
        }
        out
    }
}

impl AsyncElement for Speed {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Reads host memory, so it takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        audiofx::accept_audio(upstream_caps, None)?;
        Ok(upstream_caps.clone())
    }

    /// Resampling in place: the caps rate is kept, only the sample count moves.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        audiofx::passthrough_constraint(None)
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configure(absolute_caps)?;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let caps = self.caps.clone().ok_or(G2gError::NotConfigured)?;
                    let src = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    let input = audiofx::decode(src, self.format);
                    let resampled = self.resample(&input);
                    if resampled.is_empty() {
                        return Ok(());
                    }
                    if self.origin_ns.is_none() {
                        self.origin_ns = Some(self.map_ns(frame.timing.pts_ns));
                    }

                    if self.last_caps.as_ref() != Some(&caps) {
                        out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                        self.last_caps = Some(caps);
                    }

                    let origin = self.origin_ns.unwrap_or(0);
                    let frames = (resampled.len() / self.channels.max(1)) as u64;
                    let pts =
                        origin.saturating_add(samples_to_ns(self.emitted_frames, self.sample_rate));
                    let end = origin.saturating_add(samples_to_ns(
                        self.emitted_frames + frames,
                        self.sample_rate,
                    ));
                    self.emitted_frames += frames;

                    let timing = FrameTiming {
                        pts_ns: pts,
                        dts_ns: pts,
                        duration_ns: end.saturating_sub(pts),
                        capture_ns: frame.timing.capture_ns,
                        arrival_ns: frame.timing.arrival_ns,
                        keyframe: frame.timing.keyframe,
                    };
                    let out_frame = Frame::new(
                        MemoryDomain::System(SystemSlice::from_boxed(audiofx::encode(
                            &resampled,
                            self.format,
                        ))),
                        timing,
                        self.emitted,
                    );
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(out_frame)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    self.configure(&c)?;
                }
                PipelinePacket::Segment(mut segment) => {
                    segment.start = self.map_ns(segment.start);
                    segment.stop = segment.stop.map(|stop| self.map_ns(stop));
                    segment.position = self.map_ns(segment.position);
                    self.reset_stream();
                    self.origin_ns = Some(segment.start);
                    out.push(PipelinePacket::Segment(segment)).await?;
                }
                PipelinePacket::Flush => {
                    self.reset_stream();
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        SPEED_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Speed",
            "Filter/Effect/Audio",
            "Set speed/pitch on audio/raw streams (resampler)",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "speed" => self.speed = audiofx::double_in_range(value, SPEED_MIN, SPEED_MAX)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "speed" => Some(PropValue::Double(self.speed)),
            _ => None,
        }
    }
}

static SPEED_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "speed",
    PropKind::Double,
    "playback rate: the output holds 1 / speed as many samples, at the caps rate",
)
.with_range("0.1", "40")
.with_default(DEFAULT_SPEED_TEXT)];

impl PadTemplates for Speed {
    fn pad_templates() -> Vec<PadTemplate> {
        audiofx::default_pad_templates()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 48_000;
    const FRAMES: usize = 100;

    fn caps(channels: u8) -> Caps {
        Caps::Audio {
            format: AudioFormat::PcmF32Le,
            channels,
            sample_rate: RATE,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }
    }

    /// A ramp, so an interpolated sample is checkable by hand.
    fn ramp(frames: usize, channels: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(frames * channels);
        for index in 0..frames {
            for _ in 0..channels {
                out.push(index as f32);
            }
        }
        out
    }

    #[test]
    fn declared_default_matches_the_constant() {
        let element = Speed::new();
        assert_eq!(
            element.get_property("speed"),
            Some(PropValue::Double(DEFAULT_SPEED))
        );
    }

    #[test]
    fn doubling_the_speed_halves_the_frame_count() {
        let mut element = Speed::new().with_speed(2.0);
        element.configure(&caps(1)).unwrap();
        let resampled = element.resample(&ramp(FRAMES, 1));
        assert_eq!(resampled.len(), FRAMES / 2);
    }

    #[test]
    fn halving_the_speed_doubles_the_frame_count() {
        let speed = 0.5;
        let mut element = Speed::new().with_speed(speed);
        element.configure(&caps(1)).unwrap();
        let resampled = element.resample(&ramp(FRAMES, 1));
        // the read grid starts half a step in, so the count lands within one
        // frame of the input divided by the speed.
        let expected = FRAMES as f64 / speed;
        assert!(
            (resampled.len() as f64 - expected).abs() <= 1.0,
            "got {} frames, expected about {expected}",
            resampled.len()
        );
    }

    #[test]
    fn a_ramp_interpolates_halfway_between_its_reads() {
        let mut element = Speed::new().with_speed(2.0);
        element.configure(&caps(1)).unwrap();
        let resampled = element.resample(&ramp(FRAMES, 1));
        // the read position starts at 0.5 and advances by 2, so output n
        // averages the samples at 2n-1 and 2n+1, both a ramp value.
        assert!((resampled[0] - 0.5).abs() < 1e-4, "got {}", resampled[0]);
        assert!((resampled[1] - 2.0).abs() < 1e-4, "got {}", resampled[1]);
        assert!((resampled[2] - 4.0).abs() < 1e-4, "got {}", resampled[2]);
    }

    #[test]
    fn every_channel_is_resampled_on_the_same_grid() {
        let mut element = Speed::new().with_speed(2.0);
        element.configure(&caps(2)).unwrap();
        let resampled = element.resample(&ramp(FRAMES, 2));
        assert_eq!(resampled.len(), FRAMES / 2 * 2);
        for pair in resampled.as_chunks::<2>().0 {
            assert_eq!(pair[0], pair[1], "the two channels carry the same ramp");
        }
    }

    #[test]
    fn a_segment_is_remapped_onto_the_compressed_timeline() {
        let element = Speed::new().with_speed(2.0);
        assert_eq!(element.map_ns(1_000_000_000), 500_000_000);
    }

    #[test]
    fn speed_is_range_checked() {
        let mut element = Speed::new();
        assert_eq!(
            element
                .set_property("speed", PropValue::Double(50.0))
                .unwrap_err(),
            PropError::Value
        );
    }
}
