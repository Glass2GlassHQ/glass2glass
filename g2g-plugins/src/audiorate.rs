//! Audio rate corrector (`audiorate`). Keeps an interleaved PCM stream
//! contiguous: a timestamp gap is filled with silence, samples that overlap
//! what was already emitted are dropped, and everything else is re-stamped, so
//! downstream sees each frame's pts follow the previous frame's last sample.
//! Format, channel count, and sample rate pass through unchanged. CPU-only,
//! `no_std` baseline.
//!
//! Output timing runs off a sample counter rather than an accumulated pts: the
//! stream start fixes `base_pts` and each output's pts is
//! `base_pts + samples * 1e9 / rate`, so a rate that does not divide a second
//! (44100) never drifts. `tolerance` (ns) is the jitter below which a frame is
//! only re-stamped; `skip-to-first` starts the grid at the first frame's
//! timestamp instead of the segment start.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
    PropError, PropKind, PropValue, PropertySpec, ANY_CHANNELS, ANY_SAMPLE_RATE,
};

use crate::audioconvert::{ns_to_samples, pcm_formats, sample_bytes, samples_to_ns, silence_byte};

/// gst `audiorate`'s default `tolerance`, 40 ms in ns.
const DEFAULT_TOLERANCE_NS: u64 = 40_000_000;

/// The same value as declared text, for `gst-inspect`.
const DEFAULT_TOLERANCE_TEXT: &str = "40000000";

/// Sample frames one silence buffer covers, so filling a long gap costs a
/// bounded allocation per push instead of one buffer the size of the gap.
const SILENCE_CHUNK_SAMPLES: u64 = 4096;

/// # Example
///
/// ```no_run
/// use g2g_plugins::audiorate::AudioRate;
///
/// let rate = AudioRate::new().with_tolerance(0);
/// ```
#[derive(Debug)]
pub struct AudioRate {
    tolerance_ns: u64,
    skip_to_first: bool,
    input: Option<(AudioFormat, u8, u32)>,
    configured: bool,
    last_caps: Option<Caps>,
    /// Timestamp of sample 0 of the contiguous run, and the count of samples
    /// emitted since it. The expected pts is derived from the pair.
    base_pts: Option<u64>,
    next_sample: u64,
    /// Start of the current segment, the grid origin when `skip-to-first` is
    /// off.
    segment_start: Option<u64>,
    in_samples: u64,
    out_samples: u64,
    added: u64,
    dropped: u64,
    emitted: u64,
}

impl Default for AudioRate {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioRate {
    /// gst's defaults: 40 ms tolerance, filling from the segment start.
    pub fn new() -> Self {
        Self {
            tolerance_ns: DEFAULT_TOLERANCE_NS,
            skip_to_first: false,
            input: None,
            configured: false,
            last_caps: None,
            base_pts: None,
            next_sample: 0,
            segment_start: None,
            in_samples: 0,
            out_samples: 0,
            added: 0,
            dropped: 0,
            emitted: 0,
        }
    }

    pub fn with_tolerance(mut self, tolerance_ns: u64) -> Self {
        self.tolerance_ns = tolerance_ns;
        self
    }

    pub fn with_skip_to_first(mut self, skip_to_first: bool) -> Self {
        self.skip_to_first = skip_to_first;
        self
    }

    /// Input samples seen, output samples emitted, silence samples added, and
    /// samples dropped, the `in` / `out` / `add` / `drop` properties.
    pub fn counters(&self) -> (u64, u64, u64, u64) {
        (self.in_samples, self.out_samples, self.added, self.dropped)
    }

    fn accept_input(&self, caps: &Caps) -> Result<(AudioFormat, u8, u32), G2gError> {
        let Caps::Audio {
            format,
            channels,
            sample_rate,
            ..
        } = caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if !pcm_formats().contains(format)
            || *channels == ANY_CHANNELS
            || *sample_rate == ANY_SAMPLE_RATE
        {
            return Err(G2gError::CapsMismatch);
        }
        Ok((*format, *channels, *sample_rate))
    }

    /// Timestamp the next output sample belongs at.
    fn expected_pts(&self, rate: u32) -> Option<u64> {
        self.base_pts
            .map(|base| base.saturating_add(samples_to_ns(self.next_sample, rate)))
    }

    /// Fix the grid origin for the first frame of a run: the segment start when
    /// it is known and behind the frame, else the frame's own timestamp.
    fn start_run(&mut self, pts: u64) -> u64 {
        let start = match self.segment_start {
            Some(start) if !self.skip_to_first && start < pts => start,
            _ => pts,
        };
        self.base_pts = Some(start);
        self.next_sample = 0;
        start
    }

    async fn emit_frame(
        &mut self,
        domain: MemoryDomain,
        samples: u64,
        rate: u32,
        source: FrameTiming,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let base = self.base_pts.unwrap_or(0);
        let pts = base.saturating_add(samples_to_ns(self.next_sample, rate));
        let end = base.saturating_add(samples_to_ns(self.next_sample + samples, rate));
        let timing = FrameTiming {
            pts_ns: pts,
            dts_ns: pts,
            duration_ns: end - pts,
            capture_ns: source.capture_ns,
            arrival_ns: source.arrival_ns,
            keyframe: source.keyframe,
        };
        self.next_sample += samples;
        self.out_samples += samples;
        let frame = Frame::new(domain, timing, self.emitted);
        self.emitted += 1;
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }

    async fn emit_silence(
        &mut self,
        mut samples: u64,
        format: AudioFormat,
        bytes_per_frame: usize,
        rate: u32,
        source: FrameTiming,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        while samples > 0 {
            let chunk = samples.min(SILENCE_CHUNK_SAMPLES);
            let bytes = vec![silence_byte(format); chunk as usize * bytes_per_frame];
            let domain = MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice()));
            self.emit_frame(domain, chunk, rate, source, out).await?;
            samples -= chunk;
        }
        Ok(())
    }
}

impl AsyncElement for AudioRate {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Audio rate adjuster",
            "Filter/Effect/Audio",
            "Fills gaps with silence and drops overlapping samples to make a contiguous stream",
            "g2g",
        )
    }

    /// Reads host memory, so it takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        self.accept_input(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    /// Retiming only: the output caps equal the input for any PCM format.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::Audio { format, .. } if pcm_formats().contains(format) => {
                CapsSet::one(input.clone())
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.input = Some(self.accept_input(absolute_caps)?);
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let (format, channels, rate) = self.input.ok_or(G2gError::NotConfigured)?;
                    let bytes_per_frame = sample_bytes(format) * channels as usize;
                    let len = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?
                        .len();
                    if bytes_per_frame == 0 || len % bytes_per_frame != 0 {
                        return Err(G2gError::CapsMismatch);
                    }
                    let samples = (len / bytes_per_frame) as u64;

                    let caps = Caps::Audio {
                        format,
                        channels,
                        sample_rate: rate,
                        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
                    };
                    if self.last_caps.as_ref() != Some(&caps) {
                        out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                        self.last_caps = Some(caps);
                    }

                    // No timestamp, so there is nothing to rate against.
                    let Some(pts) = frame.timing.pts() else {
                        out.push(PipelinePacket::DataFrame(frame)).await?;
                        return Ok(());
                    };
                    self.in_samples += samples;
                    let source = frame.timing;
                    let expected = match self.expected_pts(rate) {
                        Some(expected) => expected,
                        None => self.start_run(pts),
                    };

                    if pts > expected.saturating_add(self.tolerance_ns) {
                        let fill = ns_to_samples(pts - expected, rate);
                        self.added += fill;
                        self.emit_silence(fill, format, bytes_per_frame, rate, source, out)
                            .await?;
                    } else if expected > pts.saturating_add(self.tolerance_ns) {
                        let overlap = ns_to_samples(expected - pts, rate);
                        if overlap >= samples {
                            self.dropped += samples;
                            return Ok(());
                        }
                        self.dropped += overlap;
                        let cut = overlap as usize * bytes_per_frame;
                        let kept: Box<[u8]> = frame
                            .domain
                            .require_system_slice(g2g_core::log::short_type_name::<Self>())?[cut..]
                            .into();
                        let domain = MemoryDomain::System(SystemSlice::from_boxed(kept));
                        return self
                            .emit_frame(domain, samples - overlap, rate, source, out)
                            .await;
                    }
                    self.emit_frame(frame.domain, samples, rate, source, out)
                        .await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    let new_input = self.accept_input(&c)?;
                    if self.input != Some(new_input) {
                        // a new format or rate ends the run: the sample grid
                        // it was counted on is gone.
                        self.input = Some(new_input);
                        self.base_pts = None;
                        self.next_sample = 0;
                    }
                }
                PipelinePacket::Flush => {
                    self.base_pts = None;
                    self.next_sample = 0;
                    self.segment_start = None;
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Segment(seg) => {
                    self.segment_start = Some(seg.start);
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                // the transform arm forwards EOS.
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        AUDIORATE_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "tolerance" => self.tolerance_ns = value.as_uint().ok_or(PropError::Type)?,
            "skip-to-first" => self.skip_to_first = value.as_bool().ok_or(PropError::Type)?,
            "in" | "out" | "add" | "drop" => return Err(PropError::Value),
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "tolerance" => Some(PropValue::Uint(self.tolerance_ns)),
            "skip-to-first" => Some(PropValue::Bool(self.skip_to_first)),
            "in" => Some(PropValue::Uint(self.in_samples)),
            "out" => Some(PropValue::Uint(self.out_samples)),
            "add" => Some(PropValue::Uint(self.added)),
            "drop" => Some(PropValue::Uint(self.dropped)),
            _ => None,
        }
    }
}

/// `AudioRate`'s properties (M1066): gst `audiorate`'s two settings plus its
/// four sample counters.
static AUDIORATE_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "tolerance",
        PropKind::Uint,
        "timestamp jitter (ns) below which a frame is only re-stamped",
    )
    .with_default(DEFAULT_TOLERANCE_TEXT),
    PropertySpec::new(
        "skip-to-first",
        PropKind::Bool,
        "start at the first frame's timestamp instead of the segment start",
    )
    .with_default("false"),
    PropertySpec::new("in", PropKind::Uint, "input samples").read_only(),
    PropertySpec::new("out", PropKind::Uint, "output samples").read_only(),
    PropertySpec::new("add", PropKind::Uint, "silence samples added").read_only(),
    PropertySpec::new("drop", PropKind::Uint, "samples dropped").read_only(),
];

impl PadTemplates for AudioRate {
    fn pad_templates() -> Vec<PadTemplate> {
        let pcm = |format| Caps::Audio {
            format,
            channels: ANY_CHANNELS,
            sample_rate: ANY_SAMPLE_RATE,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        let set = CapsSet::from_alternatives(pcm_formats().map(pcm).to_vec());
        Vec::from([PadTemplate::sink(set.clone()), PadTemplate::source(set)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 48_000;

    #[test]
    fn declared_tolerance_default_matches_the_constant() {
        let spec = AUDIORATE_PROPS
            .iter()
            .find(|s| s.name == "tolerance")
            .expect("tolerance is declared");
        assert_eq!(
            spec.default.and_then(|d| d.parse::<u64>().ok()),
            Some(DEFAULT_TOLERANCE_NS)
        );
        assert_eq!(AudioRate::new().tolerance_ns, DEFAULT_TOLERANCE_NS);
    }

    #[test]
    fn sample_time_round_trips_at_an_awkward_rate() {
        // 44100 does not divide a second: 4410 samples is exactly 100 ms, and
        // the sample count comes back from the duration.
        assert_eq!(samples_to_ns(4410, 44_100), 100_000_000);
        assert_eq!(ns_to_samples(100_000_000, 44_100), 4410);
        // rounding to nearest, not down: most of a sample counts as one.
        let sample_ns = samples_to_ns(1, RATE);
        assert_eq!(ns_to_samples(sample_ns * 3 / 4, RATE), 1);
        assert_eq!(ns_to_samples(sample_ns / 4, RATE), 0);
    }

    #[test]
    fn u8_silence_is_the_offset_binary_midpoint() {
        assert_eq!(silence_byte(AudioFormat::PcmU8), 0x80);
        assert_eq!(silence_byte(AudioFormat::PcmS16Le), 0);
        assert_eq!(silence_byte(AudioFormat::PcmF32Le), 0);
    }

    #[test]
    fn configure_rejects_compressed_and_wildcard_caps() {
        let mut e = AudioRate::new();
        let opus = Caps::Audio {
            format: AudioFormat::Opus,
            channels: 2,
            sample_rate: RATE,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        assert_eq!(
            e.configure_pipeline(&opus).unwrap_err(),
            G2gError::CapsMismatch
        );
        let no_channels = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: ANY_CHANNELS,
            sample_rate: RATE,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        assert_eq!(
            e.configure_pipeline(&no_channels).unwrap_err(),
            G2gError::CapsMismatch
        );
        let no_rate = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: ANY_SAMPLE_RATE,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        assert_eq!(
            e.configure_pipeline(&no_rate).unwrap_err(),
            G2gError::CapsMismatch
        );
        let ok = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: RATE,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        assert!(e.configure_pipeline(&ok).is_ok());
    }
}
