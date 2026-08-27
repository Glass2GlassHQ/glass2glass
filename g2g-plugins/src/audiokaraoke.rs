//! Centre-channel removal (`audiokaraoke`). Subtracts each stereo channel from
//! the other, which cancels whatever is panned centre (usually the voice), and
//! adds back a band-passed mono sum so the bass and the kick survive.
//! Preserves format, channel count, and sample rate. CPU-only `no_std`.
//!
//! Matches GStreamer's `audiokaraoke` float path. The mono sum runs through the
//! reference's two-pole resonator at `filter-band` with `filter-width`, and
//! `level` / `mono-level` scale the cancellation and the re-added mono. Stereo
//! only, as in the reference's caps, so a mono or 5.1 stream needs
//! `audioconvert` ahead of it.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError,
    MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec,
};

use crate::audiofx;
use crate::mathf;

/// The effect only has a centre to cancel on a stereo pair.
const KARAOKE_CHANNELS: u8 = 2;

const DEFAULT_LEVEL: f64 = 1.0;
const LEVEL_MIN: f64 = 0.0;
const LEVEL_MAX: f64 = 1.0;
const DEFAULT_FILTER_BAND: f64 = 220.0;
const FILTER_BAND_MIN: f64 = 0.0;
const FILTER_BAND_MAX: f64 = 441.0;
const DEFAULT_FILTER_WIDTH: f64 = 100.0;
const FILTER_WIDTH_MIN: f64 = 0.0;
const FILTER_WIDTH_MAX: f64 = 100.0;

/// The reference's two-pole resonator on the mono sum: `y = A*x - B*y1 - C*y2`.
#[derive(Debug, Default, Clone, Copy)]
struct MonoResonator {
    a: f64,
    b: f64,
    c: f64,
    y1: f64,
    y2: f64,
}

impl MonoResonator {
    fn update(&mut self, band: f64, width: f64, rate: u32) {
        if rate == 0 {
            return;
        }
        let tau = core::f64::consts::TAU;
        let c = mathf::exp(-tau * width / rate as f64);
        let b = -4.0 * c / (1.0 + c) * mathf::cos(tau * band / rate as f64);
        let a = mathf::sqrt(1.0 - b * b / (4.0 * c)) * (1.0 - c);
        self.a = a;
        self.b = b;
        self.c = c;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }

    fn reset(&mut self) {
        self.y1 = 0.0;
        self.y2 = 0.0;
    }

    fn step(&mut self, x: f64) -> f64 {
        let y = (self.a * x - self.b * self.y1) - self.c * self.y2;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::audiokaraoke::AudioKaraoke;
///
/// let karaoke = AudioKaraoke::new().with_level(1.0).with_mono_level(0.5);
/// ```
#[derive(Debug)]
pub struct AudioKaraoke {
    level: f64,
    mono_level: f64,
    filter_band: f64,
    filter_width: f64,
    resonator: MonoResonator,
    format: AudioFormat,
    sample_rate: u32,
    caps: Option<Caps>,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl Default for AudioKaraoke {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioKaraoke {
    pub fn new() -> Self {
        Self {
            level: DEFAULT_LEVEL,
            mono_level: DEFAULT_LEVEL,
            filter_band: DEFAULT_FILTER_BAND,
            filter_width: DEFAULT_FILTER_WIDTH,
            resonator: MonoResonator::default(),
            format: AudioFormat::PcmS16Le,
            sample_rate: 0,
            caps: None,
            last_caps: None,
            emitted: 0,
        }
    }

    pub fn with_level(mut self, level: f64) -> Self {
        self.level = level.clamp(LEVEL_MIN, LEVEL_MAX);
        self
    }

    pub fn with_mono_level(mut self, mono_level: f64) -> Self {
        self.mono_level = mono_level.clamp(LEVEL_MIN, LEVEL_MAX);
        self
    }

    pub fn with_filter_band(mut self, band: f64) -> Self {
        self.filter_band = band.clamp(FILTER_BAND_MIN, FILTER_BAND_MAX);
        self.update_filter();
        self
    }

    pub fn with_filter_width(mut self, width: f64) -> Self {
        self.filter_width = width.clamp(FILTER_WIDTH_MIN, FILTER_WIDTH_MAX);
        self.update_filter();
        self
    }

    fn update_filter(&mut self) {
        self.resonator
            .update(self.filter_band, self.filter_width, self.sample_rate);
    }

    fn configure(&mut self, caps: &Caps) -> Result<(), G2gError> {
        let (format, _, rate) = audiofx::accept_audio(caps, Some(KARAOKE_CHANNELS))?;
        self.format = format;
        self.sample_rate = rate;
        self.caps = Some(caps.clone());
        self.update_filter();
        Ok(())
    }

    /// Cancel the centre of one interleaved stereo buffer in place.
    fn filter(&mut self, samples: &mut [f32]) {
        for pair in samples.as_chunks_mut::<{ KARAOKE_CHANNELS as usize }>().0 {
            let left = pair[0] as f64;
            let right = pair[1] as f64;
            let mono = self.resonator.step((left + right) / 2.0) * self.mono_level * self.level;
            pair[0] = (left - right * self.level + mono) as f32;
            pair[1] = (right - left * self.level + mono) as f32;
        }
    }
}

impl AsyncElement for AudioKaraoke {
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
        audiofx::accept_audio(upstream_caps, Some(KARAOKE_CHANNELS))?;
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        audiofx::passthrough_constraint(Some(KARAOKE_CHANNELS))
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
                    let mut samples = audiofx::decode(src, self.format);
                    self.filter(&mut samples);
                    let dst = audiofx::encode(&samples, self.format);

                    if self.last_caps.as_ref() != Some(&caps) {
                        out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                        self.last_caps = Some(caps);
                    }
                    let out_frame = Frame {
                        domain: MemoryDomain::System(SystemSlice::from_boxed(dst)),
                        timing: frame.timing,
                        sequence: self.emitted,
                        meta: Default::default(),
                    };
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(out_frame)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    self.configure(&c)?;
                }
                PipelinePacket::Flush => {
                    self.resonator.reset();
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
        AUDIOKARAOKE_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "AudioKaraoke",
            "Filter/Effect/Audio",
            "Removes voice from sound",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "level" => self.level = audiofx::double_in_range(value, LEVEL_MIN, LEVEL_MAX)?,
            "mono-level" => {
                self.mono_level = audiofx::double_in_range(value, LEVEL_MIN, LEVEL_MAX)?
            }
            "filter-band" => {
                self.filter_band =
                    audiofx::double_in_range(value, FILTER_BAND_MIN, FILTER_BAND_MAX)?;
                self.update_filter();
            }
            "filter-width" => {
                self.filter_width =
                    audiofx::double_in_range(value, FILTER_WIDTH_MIN, FILTER_WIDTH_MAX)?;
                self.update_filter();
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "level" => Some(PropValue::Double(self.level)),
            "mono-level" => Some(PropValue::Double(self.mono_level)),
            "filter-band" => Some(PropValue::Double(self.filter_band)),
            "filter-width" => Some(PropValue::Double(self.filter_width)),
            _ => None,
        }
    }
}

static AUDIOKARAOKE_PROPS: &[PropertySpec] = &[
    PropertySpec::new("level", PropKind::Double, "level of the effect (1 = full)")
        .with_range("0", "1")
        .with_default("1"),
    PropertySpec::new(
        "mono-level",
        PropKind::Double,
        "level of the re-added mono channel (1 = full)",
    )
    .with_range("0", "1")
    .with_default("1"),
    PropertySpec::new(
        "filter-band",
        PropKind::Double,
        "centre frequency of the mono filter in Hz",
    )
    .with_range("0", "441")
    .with_default("220"),
    PropertySpec::new(
        "filter-width",
        PropKind::Double,
        "width of the mono filter in Hz",
    )
    .with_range("0", "100")
    .with_default("100"),
];

impl PadTemplates for AudioKaraoke {
    fn pad_templates() -> Vec<PadTemplate> {
        audiofx::pad_templates(KARAOKE_CHANNELS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stereo(rate: u32) -> Caps {
        Caps::Audio {
            format: AudioFormat::PcmF32Le,
            channels: KARAOKE_CHANNELS,
            sample_rate: rate,
        }
    }

    #[test]
    fn identical_channels_cancel() {
        let mut e = AudioKaraoke::new().with_mono_level(0.0);
        e.configure(&stereo(48_000)).unwrap();
        let mut samples = [0.5f32, 0.5, -0.25, -0.25, 0.125, 0.125];
        e.filter(&mut samples);
        for value in samples {
            assert!(value.abs() < 1e-6, "centre is cancelled, got {value}");
        }
    }

    #[test]
    fn anti_phase_channels_double() {
        let mut e = AudioKaraoke::new().with_mono_level(0.0);
        e.configure(&stereo(48_000)).unwrap();
        let mut samples = [0.25f32, -0.25];
        e.filter(&mut samples);
        assert!((samples[0] - 0.5).abs() < 1e-6);
        assert!((samples[1] + 0.5).abs() < 1e-6);
    }

    #[test]
    fn mono_is_rejected() {
        let mut e = AudioKaraoke::new();
        let mono = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 1,
            sample_rate: 48_000,
        };
        assert_eq!(e.configure(&mono).unwrap_err(), G2gError::CapsMismatch);
    }
}
