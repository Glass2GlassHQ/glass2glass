//! Three-band audio equalizer (`equalizer-3bands`). Cascades three RBJ peaking
//! biquads (low / mid / high) over an interleaved S16LE stream, each band's gain
//! set in dB, preserving format, channels, and rate. CPU-only `no_std`.
//!
//! The band centers are fixed (100 Hz / 1 kHz / 10 kHz, Q ~1); `band0`..`band2`
//! set each gain in dB (0 = flat, an exact pass-through). The `sin`/`cos`/`pow`
//! for the coefficients come from the crate's `libm`-free approximations, run
//! only when a gain or the rate changes, not per sample.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec,
};

const CENTERS_HZ: [f64; 3] = [100.0, 1000.0, 10000.0];
const Q: f64 = 1.0;

/// A normalized biquad (a0 folded into the other coefficients).
#[derive(Debug, Clone, Copy)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

impl Biquad {
    /// The identity filter (used before the rate is known).
    fn identity() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        }
    }

    /// RBJ cookbook peaking EQ at `f0` for sample rate `fs`, `gain_db` dB.
    fn peaking(f0: f64, fs: f64, q: f64, gain_db: f64) -> Self {
        let a = crate::mathf::powf(10.0, gain_db / 40.0);
        // w0 in turns (cycles): sin/cos take turns, so f0/fs is w0/(2*pi) already.
        let turns = (f0 / fs) as f32;
        let cos_w0 = crate::mathf::cos_turns(turns) as f64;
        let sin_w0 = crate::mathf::sin_turns(turns) as f64;
        let alpha = sin_w0 / (2.0 * q);
        let a0 = 1.0 + alpha / a;
        Self {
            b0: (1.0 + alpha * a) / a0,
            b1: (-2.0 * cos_w0) / a0,
            b2: (1.0 - alpha * a) / a0,
            a1: (-2.0 * cos_w0) / a0,
            a2: (1.0 - alpha / a) / a0,
        }
    }
}

/// Transposed direct-form II state for one biquad on one channel.
#[derive(Debug, Clone, Copy, Default)]
struct BiquadState {
    z1: f64,
    z2: f64,
}

impl BiquadState {
    fn step(&mut self, bq: &Biquad, x: f64) -> f64 {
        let y = bq.b0 * x + self.z1;
        self.z1 = bq.b1 * x - bq.a1 * y + self.z2;
        self.z2 = bq.b2 * x - bq.a2 * y;
        y
    }
}

#[derive(Debug)]
pub struct Equalizer3Bands {
    gains_db: [f64; 3],
    coeffs: [Biquad; 3],
    channels: usize,
    sample_rate: u32,
    // per-channel, per-band filter state.
    state: Vec<[BiquadState; 3]>,
    configured: bool,
}

impl Default for Equalizer3Bands {
    fn default() -> Self {
        Self::new()
    }
}

impl Equalizer3Bands {
    pub fn new() -> Self {
        Self {
            gains_db: [0.0; 3],
            coeffs: [Biquad::identity(); 3],
            channels: 0,
            sample_rate: 0,
            state: Vec::new(),
            configured: false,
        }
    }

    pub fn with_band(mut self, band: usize, gain_db: f64) -> Self {
        if band < 3 {
            self.gains_db[band] = gain_db;
        }
        self
    }

    fn recompute(&mut self) {
        if self.sample_rate == 0 {
            return;
        }
        let fs = self.sample_rate as f64;
        for (b, coeff) in self.coeffs.iter_mut().enumerate() {
            *coeff = Biquad::peaking(CENTERS_HZ[b], fs, Q, self.gains_db[b]);
        }
    }

    fn accept_input(&self, caps: &Caps) -> Result<(u32, u32), G2gError> {
        match caps {
            Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels,
                sample_rate,
            } if *channels > 0 => Ok((*channels as u32, *sample_rate)),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn configure(&mut self, caps: &Caps) -> Result<(), G2gError> {
        let (channels, rate) = self.accept_input(caps)?;
        self.channels = channels as usize;
        self.sample_rate = rate;
        self.state = vec![[BiquadState::default(); 3]; self.channels];
        self.recompute();
        Ok(())
    }

    fn filter(&mut self, src: &[u8], dst: &mut [u8]) {
        let ch = self.channels.max(1);
        for (i, (s, d)) in src.chunks_exact(2).zip(dst.chunks_exact_mut(2)).enumerate() {
            let c = i % ch;
            let mut x = i16::from_le_bytes([s[0], s[1]]) as f64;
            for b in 0..3 {
                x = self.state[c][b].step(&self.coeffs[b], x);
            }
            let rounded = if x >= 0.0 { x + 0.5 } else { x - 0.5 };
            let out = rounded.clamp(i16::MIN as f64, i16::MAX as f64) as i16;
            d.copy_from_slice(&out.to_le_bytes());
        }
    }
}

impl AsyncElement for Equalizer3Bands {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        self.accept_input(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::Audio {
                format: AudioFormat::PcmS16Le,
                ..
            } => CapsSet::one(input.clone()),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configure(absolute_caps)?;
        self.configured = true;
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
                    if !self.configured {
                        return Err(G2gError::NotConfigured);
                    }
                    let Some(src) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let mut dst = vec![0u8; src.len()].into_boxed_slice();
                    self.filter(src, &mut dst);
                    let out_frame = Frame {
                        domain: MemoryDomain::System(SystemSlice::from_boxed(dst)),
                        timing: frame.timing,
                        sequence: frame.sequence,
                        meta: Default::default(),
                    };
                    out.push(PipelinePacket::DataFrame(out_frame)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    self.configure(&c)?;
                    out.push(PipelinePacket::CapsChanged(c)).await?;
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
        EQUALIZER_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "3-band equalizer",
            "Filter/Effect/Audio",
            "Three-band audio equalizer",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        let band = match name {
            "band0" => 0,
            "band1" => 1,
            "band2" => 2,
            _ => return Err(PropError::Unknown),
        };
        self.gains_db[band] = value.as_double().ok_or(PropError::Type)?;
        self.recompute();
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "band0" => Some(PropValue::Double(self.gains_db[0])),
            "band1" => Some(PropValue::Double(self.gains_db[1])),
            "band2" => Some(PropValue::Double(self.gains_db[2])),
            _ => None,
        }
    }
}

static EQUALIZER_PROPS: &[PropertySpec] = &[
    PropertySpec::new("band0", PropKind::Double, "low band (100 Hz) gain in dB"),
    PropertySpec::new("band1", PropKind::Double, "mid band (1 kHz) gain in dB"),
    PropertySpec::new("band2", PropKind::Double, "high band (10 kHz) gain in dB"),
];

impl PadTemplates for Equalizer3Bands {
    fn pad_templates() -> Vec<PadTemplate> {
        let pcm = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
        };
        Vec::from([
            PadTemplate::sink(CapsSet::one(pcm.clone())),
            PadTemplate::source(CapsSet::one(pcm)),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack(samples: &[i16]) -> Vec<u8> {
        let mut v = vec![0u8; samples.len() * 2];
        for (i, s) in samples.iter().enumerate() {
            v[i * 2..i * 2 + 2].copy_from_slice(&s.to_le_bytes());
        }
        v
    }

    fn unpack(bytes: &[u8]) -> Vec<i16> {
        bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect()
    }

    #[test]
    fn flat_gain_is_a_pass_through() {
        // A peaking EQ at 0 dB has b == a after normalization, so H(z) = 1: the
        // output equals the input to within i16 rounding.
        let mut eq = Equalizer3Bands::new();
        eq.configure(&Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 1,
            sample_rate: 48_000,
        })
        .unwrap();
        let src = pack(&[1000, -2000, 3000, -4000, 5000, 100, -100, 0]);
        let mut dst = vec![0u8; src.len()];
        eq.filter(&src, &mut dst);
        for (a, b) in unpack(&src).iter().zip(unpack(&dst).iter()) {
            assert!((a - b).abs() <= 1, "flat EQ drifted {a} -> {b}");
        }
    }

    #[test]
    fn boost_changes_the_output() {
        let mut eq = Equalizer3Bands::new().with_band(0, 12.0);
        eq.configure(&Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 1,
            sample_rate: 48_000,
        })
        .unwrap();
        let src = pack(&[8000, 8000, 8000, 8000, 8000, 8000]);
        let mut dst = vec![0u8; src.len()];
        eq.filter(&src, &mut dst);
        assert_ne!(
            unpack(&src),
            unpack(&dst),
            "a 12 dB boost must alter the signal"
        );
    }

    #[test]
    fn peaking_at_zero_db_is_identity_transfer() {
        let bq = Biquad::peaking(1000.0, 48_000.0, 1.0, 0.0);
        assert!((bq.b0 - 1.0).abs() < 1e-9);
        assert!((bq.b1 - bq.a1).abs() < 1e-12);
        assert!((bq.b2 - bq.a2).abs() < 1e-12);
    }

    #[test]
    fn configure_rejects_non_s16le() {
        let mut eq = Equalizer3Bands::new();
        let bad = Caps::Audio {
            format: AudioFormat::PcmF32Le,
            channels: 2,
            sample_rate: 48_000,
        };
        assert_eq!(
            eq.configure_pipeline(&bad).unwrap_err(),
            G2gError::CapsMismatch
        );
    }
}
