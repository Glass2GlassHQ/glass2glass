//! Synthetic audio source (M25), the audio analog of `VideoTestSrc`. Emits
//! interleaved signed 16-bit PCM (`AudioFormat::PcmS16Le`) buffers of a
//! deterministic test waveform at a fixed sample rate. CPU-only, `no_std`.
//!
//! Waveforms: `sine`, `square`, `saw`, `triangle`, `white-noise`, `silence`. The
//! sine uses Bhaskara I's approximation (pure f32 arithmetic, no libm), accurate
//! to ~0.2% of full scale; the rest are exact ramps / a deterministic hash, so
//! every waveform keeps the element in the crate baseline (no libm).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, FrameTiming,
    G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec,
};

/// Samples per emitted buffer: 10 ms at the configured rate.
const BUFFER_MS: u64 = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wave {
    /// `tone_hz` sine at half full scale.
    Sine,
    /// `tone_hz` square at half full scale.
    Square,
    /// `tone_hz` rising sawtooth ramp (-half to +half full scale per period).
    Saw,
    /// `tone_hz` triangle, phased like the sine (0 at the start, peak at 1/4).
    Triangle,
    /// Deterministic per-sample pseudo-random noise at half full scale,
    /// independent of `tone_hz`.
    WhiteNoise,
    Silence,
}

#[derive(Debug)]
pub struct AudioTestSrc {
    sample_rate: u32,
    channels: u8,
    tone_hz: u32,
    wave: Wave,
    target_buffers: u64,
    configured: bool,
}

impl AudioTestSrc {
    pub fn new(sample_rate: u32, channels: u8, tone_hz: u32, target_buffers: u64) -> Self {
        assert!(sample_rate > 0 && channels > 0, "rate and channels must be non-zero");
        Self {
            sample_rate,
            channels,
            tone_hz,
            wave: Wave::Sine,
            target_buffers,
            configured: false,
        }
    }

    pub fn with_wave(mut self, wave: Wave) -> Self {
        self.wave = wave;
        self
    }

    fn caps(&self) -> Caps {
        Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: self.channels,
            sample_rate: self.sample_rate,
        }
    }

    /// Sample value at absolute sample index `n`, half full scale.
    fn sample(&self, n: u64) -> i16 {
        const HALF: f32 = (i16::MAX / 2) as f32;
        // Noise is tone-independent: a deterministic hash of the sample index.
        if self.wave == Wave::WhiteNoise {
            return (noise_unit(n) * HALF) as i16;
        }
        let period = self.sample_rate as u64 / self.tone_hz.max(1) as u64;
        if period == 0 {
            return 0;
        }
        let phase = (n % period) as f32 / period as f32; // [0, 1)
        match self.wave {
            Wave::Silence => 0,
            Wave::Square => {
                if phase < 0.5 {
                    i16::MAX / 2
                } else {
                    -(i16::MAX / 2)
                }
            }
            Wave::Sine => (crate::mathf::sin_turns(phase) * HALF) as i16,
            Wave::Saw => ((2.0 * phase - 1.0) * HALF) as i16,
            Wave::Triangle => (tri_turns(phase) * HALF) as i16,
            // Handled above before the period calculation.
            Wave::WhiteNoise => unreachable!(),
        }
    }
}

/// Triangle for t in [0, 1): 0 -> +1 -> -1 -> 0, sharing the sine's phase
/// (zero at the start, peak a quarter in). Exact, no libm.
fn tri_turns(t: f32) -> f32 {
    if t < 0.25 {
        4.0 * t
    } else if t < 0.75 {
        2.0 - 4.0 * t
    } else {
        4.0 * t - 4.0
    }
}

/// Deterministic pseudo-random value in [-1, 1) from a sample index, via a
/// SplitMix64-style integer hash. Stable per index so output is reproducible.
fn noise_unit(n: u64) -> f32 {
    let mut h = n.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    ((h & 0xFFFF) as f32 / 32_768.0) - 1.0
}

impl SourceLoop for AudioTestSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.caps()))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(self.caps()))))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let samples_per_buffer = (self.sample_rate as u64 * BUFFER_MS / 1000).max(1);
            let buffer_duration_ns = samples_per_buffer * 1_000_000_000 / self.sample_rate as u64;

            for seq in 0..self.target_buffers {
                let base = seq * samples_per_buffer;
                let mut bytes =
                    Vec::with_capacity(samples_per_buffer as usize * self.channels as usize * 2);
                for s in 0..samples_per_buffer {
                    let v = self.sample(base + s);
                    for _ in 0..self.channels {
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                }

                let pts = seq * buffer_duration_ns;
                #[cfg(feature = "std")]
                let arrival_ns = g2g_core::metrics::monotonic_ns();
                #[cfg(not(feature = "std"))]
                let arrival_ns: u64 = 0;
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(
                        bytes.into_boxed_slice(),
                    )),
                    timing: FrameTiming {
                        pts_ns: pts,
                        dts_ns: pts,
                        duration_ns: buffer_duration_ns,
                        capture_ns: pts,
                        arrival_ns,
                        keyframe: false, // audio: every buffer is independent
                    },
                    sequence: seq,
                    meta: Default::default(),
                };
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }

            out.push(PipelinePacket::Eos).await?;
            Ok(self.target_buffers)
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        AUDIOTESTSRC_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Audio test source",
            "Source/Audio",
            "Generates a synthetic audio test tone",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "samplerate" => self.sample_rate = value.as_uint().ok_or(PropError::Type)? as u32,
            "channels" => self.channels = value.as_uint().ok_or(PropError::Type)? as u8,
            "freq" => self.tone_hz = value.as_uint().ok_or(PropError::Type)? as u32,
            "num-buffers" => {
                let n = value.as_int().ok_or(PropError::Type)?;
                self.target_buffers = if n < 0 { u64::MAX } else { n as u64 };
            }
            "wave" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.wave = wave_from_str(s).ok_or(PropError::Value)?;
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "samplerate" => Some(PropValue::Uint(self.sample_rate as u64)),
            "channels" => Some(PropValue::Uint(self.channels as u64)),
            "freq" => Some(PropValue::Uint(self.tone_hz as u64)),
            "num-buffers" => Some(PropValue::Int(if self.target_buffers == u64::MAX {
                -1
            } else {
                self.target_buffers as i64
            })),
            "wave" => Some(PropValue::Str(wave_to_str(self.wave).into())),
            _ => None,
        }
    }
}

/// `AudioTestSrc`'s settable properties (M107).
static AUDIOTESTSRC_PROPS: &[PropertySpec] = &[
    PropertySpec::new("samplerate", PropKind::Uint, "samples per second"),
    PropertySpec::new("channels", PropKind::Uint, "channel count"),
    PropertySpec::new("freq", PropKind::Uint, "test tone frequency in Hz"),
    PropertySpec::new("num-buffers", PropKind::Int, "buffers to emit then EOS (-1 = forever)")
        .with_default("-1"),
    PropertySpec::new(
        "wave",
        PropKind::Str,
        "waveform: sine | square | saw | triangle | white-noise | silence",
    ),
];

/// Parse a `wave` property string to a [`Wave`].
fn wave_from_str(s: &str) -> Option<Wave> {
    match s {
        "sine" => Some(Wave::Sine),
        "square" => Some(Wave::Square),
        "saw" => Some(Wave::Saw),
        "triangle" => Some(Wave::Triangle),
        "white-noise" => Some(Wave::WhiteNoise),
        "silence" => Some(Wave::Silence),
        _ => None,
    }
}

/// The `wave` property string for a [`Wave`].
fn wave_to_str(w: Wave) -> &'static str {
    match w {
        Wave::Sine => "sine",
        Wave::Square => "square",
        Wave::Saw => "saw",
        Wave::Triangle => "triangle",
        Wave::WhiteNoise => "white-noise",
        Wave::Silence => "silence",
    }
}

impl PadTemplates for AudioTestSrc {
    /// Static superset: the type produces interleaved S16LE PCM; channel
    /// count and rate are instance configuration, but `Caps::Audio` has no
    /// open dims, so the template pins the common 48 kHz stereo shape.
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::source(CapsSet::one(Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
        }))])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_starts_at_zero_and_peaks_mid_half_period() {
        let src = AudioTestSrc::new(48_000, 1, 1_000, 1);
        assert_eq!(src.sample(0), 0);
        // 1 kHz at 48 kHz: period 48, peak near sample 12
        let peak = src.sample(12);
        assert!(
            (peak as i32 - (i16::MAX / 2) as i32).abs() < 400,
            "peak {peak} should be near half scale"
        );
    }

    #[test]
    fn square_and_silence_waves() {
        let sq = AudioTestSrc::new(48_000, 1, 1_000, 1).with_wave(Wave::Square);
        assert_eq!(sq.sample(0), i16::MAX / 2);
        assert_eq!(sq.sample(24), -(i16::MAX / 2)); // second half of period 48
        let silence = AudioTestSrc::new(48_000, 1, 1_000, 1).with_wave(Wave::Silence);
        assert_eq!(silence.sample(7), 0);
    }

    #[test]
    fn saw_ramps_from_trough_to_peak() {
        // period 48: phase 0 -> -half, phase 0.5 (sample 24) -> 0, rising overall.
        let src = AudioTestSrc::new(48_000, 1, 1_000, 1).with_wave(Wave::Saw);
        assert_eq!(src.sample(0), -(i16::MAX / 2));
        assert_eq!(src.sample(24), 0);
        assert!(src.sample(36) > src.sample(12), "sawtooth rises across the period");
    }

    #[test]
    fn triangle_is_zero_then_peaks_a_quarter_in() {
        let src = AudioTestSrc::new(48_000, 1, 1_000, 1).with_wave(Wave::Triangle);
        assert_eq!(src.sample(0), 0);
        assert_eq!(src.sample(12), i16::MAX / 2); // phase 0.25 -> +peak
        assert_eq!(src.sample(36), -(i16::MAX / 2)); // phase 0.75 -> -peak
    }

    #[test]
    fn white_noise_is_deterministic_bounded_and_varies() {
        let src = AudioTestSrc::new(48_000, 1, 1_000, 1).with_wave(Wave::WhiteNoise);
        let half = i16::MAX / 2;
        // Deterministic: the same index hashes identically.
        assert_eq!(src.sample(42), src.sample(42));
        let first = src.sample(0);
        let mut varied = false;
        for n in 0..100u64 {
            let v = src.sample(n);
            assert!(v.abs() <= half, "noise sample {v} within +/- half scale");
            varied |= v != first;
        }
        assert!(varied, "noise must vary across samples");
    }

    #[test]
    fn caps_are_pcm_s16le_at_configured_rate() {
        let src = AudioTestSrc::new(44_100, 2, 440, 1);
        assert_eq!(
            src.caps(),
            Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels: 2,
                sample_rate: 44_100,
            }
        );
    }
}
