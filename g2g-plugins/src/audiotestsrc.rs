//! Synthetic audio source (M25), the audio analog of `VideoTestSrc`. Emits
//! interleaved signed 16-bit PCM (`AudioFormat::PcmS16Le`) buffers of a
//! deterministic test tone at a fixed sample rate. CPU-only, `no_std`.
//!
//! The sine uses Bhaskara I's approximation (pure f32 arithmetic, no libm),
//! accurate to ~0.2% of full scale: more than clean enough for a test tone
//! and keeps the element in the crate baseline.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, FrameTiming, G2gError,
    MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
};

/// Samples per emitted buffer: 10 ms at the configured rate.
const BUFFER_MS: u64 = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wave {
    /// `tone_hz` sine at half full scale.
    Sine,
    /// `tone_hz` square at half full scale.
    Square,
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
            Wave::Sine => (sin_turns(phase) * (i16::MAX / 2) as f32) as i16,
        }
    }
}

/// sin(2*pi*t) for t in [0, 1) via Bhaskara I's approximation; pure f32
/// arithmetic so it works in `no_std` (core has no `sin` intrinsic).
fn sin_turns(t: f32) -> f32 {
    // map to half-turn x in [0, pi) with sign from the half
    let (x, sign) = if t < 0.5 {
        (t * 2.0, 1.0f32)
    } else {
        ((t - 0.5) * 2.0, -1.0f32)
    };
    const PI: f32 = core::f32::consts::PI;
    let xr = x * PI;
    sign * (16.0 * xr * (PI - xr)) / (5.0 * PI * PI - 4.0 * xr * (PI - xr))
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
    fn sine_approximation_hits_the_anchors() {
        // sin(0) = 0, sin(pi/2) = 1, sin(pi) = 0, sin(3pi/2) = -1
        assert!(sin_turns(0.0).abs() < 0.01);
        assert!((sin_turns(0.25) - 1.0).abs() < 0.01);
        assert!(sin_turns(0.5).abs() < 0.01);
        assert!((sin_turns(0.75) + 1.0).abs() < 0.01);
    }

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
