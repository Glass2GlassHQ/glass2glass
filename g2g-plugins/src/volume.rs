//! Audio gain (`volume`). Scales every signed-16-bit PCM sample by a linear
//! `volume` factor (or zeroes it when `mute` is set), preserving format, channel
//! count, and sample rate. The audio analog of `videobalance`. CPU-only `no_std`.
//!
//! `volume` is a linear multiplier (1.0 = unchanged); results are rounded and
//! clamped to the i16 range. Only `PcmS16Le` is handled; an `f32` path is a
//! follow-up.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, G2gError,
    MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec,
};

#[derive(Debug)]
pub struct Volume {
    volume: f64,
    mute: bool,
    input: Option<Caps>,
    configured: bool,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl Default for Volume {
    fn default() -> Self {
        Self::new()
    }
}

impl Volume {
    /// Unity gain, unmuted; use the builders or properties to adjust.
    pub fn new() -> Self {
        Self { volume: 1.0, mute: false, input: None, configured: false, last_caps: None, emitted: 0 }
    }

    pub fn with_volume(mut self, volume: f64) -> Self {
        self.volume = volume;
        self
    }

    pub fn with_mute(mut self, mute: bool) -> Self {
        self.mute = mute;
        self
    }

    fn accept_input(&self, caps: &Caps) -> Result<Caps, G2gError> {
        match caps {
            Caps::Audio { format: AudioFormat::PcmS16Le, channels, .. } if *channels > 0 => {
                Ok(caps.clone())
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }
}

impl AsyncElement for Volume {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        self.accept_input(upstream_caps)
    }

    /// Native `DerivedOutput`: a gain change preserves format, channels, and rate,
    /// so the output caps equal the input for S16LE PCM.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::Audio { format: AudioFormat::PcmS16Le, .. } => CapsSet::one(input.clone()),
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
                    let caps = match &self.input {
                        Some(c) => c.clone(),
                        None => return Err(G2gError::NotConfigured),
                    };
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let src = slice.as_slice();
                    let mut dst = vec![0u8; src.len()].into_boxed_slice();
                    apply_gain(src, &mut dst, self.volume, self.mute);

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
                    self.input = Some(self.accept_input(&c)?);
                }
                PipelinePacket::Flush => {
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                PipelinePacket::Eos => {}
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        VOLUME_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "volume" => self.volume = value.as_double().ok_or(PropError::Type)?,
            "mute" => self.mute = value.as_bool().ok_or(PropError::Type)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "volume" => Some(PropValue::Double(self.volume)),
            "mute" => Some(PropValue::Bool(self.mute)),
            _ => None,
        }
    }
}

/// `Volume`'s settable properties (M104).
static VOLUME_PROPS: &[PropertySpec] = &[
    PropertySpec::new("volume", PropKind::Double, "linear gain, 0..N (1 = unchanged)"),
    PropertySpec::new("mute", PropKind::Bool, "zero the output when true"),
];

impl PadTemplates for Volume {
    fn pad_templates() -> Vec<PadTemplate> {
        let pcm = Caps::Audio { format: AudioFormat::PcmS16Le, channels: 2, sample_rate: 48_000 };
        Vec::from([PadTemplate::sink(CapsSet::one(pcm.clone())), PadTemplate::source(CapsSet::one(pcm))])
    }
}

/// Scale each interleaved S16LE sample by `volume` (or zero it when `mute`),
/// rounding and clamping to the i16 range. `dst` is the same length as `src`.
fn apply_gain(src: &[u8], dst: &mut [u8], volume: f64, mute: bool) {
    for (s, d) in src.chunks_exact(2).zip(dst.chunks_exact_mut(2)) {
        let sample = i16::from_le_bytes([s[0], s[1]]);
        let out = if mute {
            0
        } else {
            ((sample as f64) * volume).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16
        };
        d.copy_from_slice(&out.to_le_bytes());
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
        bytes.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect()
    }

    #[test]
    fn gain_scales_and_clamps() {
        let src = pack(&[1000, -1000, 20000]);
        let mut dst = vec![0u8; src.len()];
        apply_gain(&src, &mut dst, 2.0, false);
        // 20000*2 = 40000 saturates at i16::MAX.
        assert_eq!(unpack(&dst), [2000, -2000, i16::MAX]);
    }

    #[test]
    fn unity_is_identity_and_mute_silences() {
        let src = pack(&[1234, -5678, 32000]);
        let mut dst = vec![0u8; src.len()];
        apply_gain(&src, &mut dst, 1.0, false);
        assert_eq!(dst, src, "volume 1.0 reproduces the input");
        apply_gain(&src, &mut dst, 1.0, true);
        assert_eq!(unpack(&dst), [0, 0, 0], "mute zeroes every sample");
    }

    #[test]
    fn configure_rejects_non_s16le() {
        let mut v = Volume::new();
        let f32_caps =
            Caps::Audio { format: AudioFormat::PcmF32Le, channels: 2, sample_rate: 48_000 };
        assert_eq!(v.configure_pipeline(&f32_caps).unwrap_err(), G2gError::CapsMismatch);
        let ok = Caps::Audio { format: AudioFormat::PcmS16Le, channels: 1, sample_rate: 16_000 };
        assert!(v.configure_pipeline(&ok).is_ok());
    }
}
