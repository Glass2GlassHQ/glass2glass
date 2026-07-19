//! Stereo panning (`audiopanorama`). Adjusts the left/right balance of an
//! interleaved S16LE stereo stream by attenuating the channel opposite the pan
//! direction, preserving format and rate. CPU-only `no_std`.
//!
//! `panorama` is -1 (full left) to +1 (full right), 0 = centred. This is the
//! simple balance method (it attenuates one side, no cross-channel mixing); the
//! psychoacoustic method and mono-input upmix are follow-ups. Stereo only.

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
pub struct AudioPanorama {
    panorama: f64,
    input: Option<Caps>,
    configured: bool,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl Default for AudioPanorama {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioPanorama {
    /// Centred (no pan); use the builder or the `panorama` property to adjust.
    pub fn new() -> Self {
        Self {
            panorama: 0.0,
            input: None,
            configured: false,
            last_caps: None,
            emitted: 0,
        }
    }

    pub fn with_panorama(mut self, panorama: f64) -> Self {
        self.panorama = panorama;
        self
    }

    fn accept_input(&self, caps: &Caps) -> Result<Caps, G2gError> {
        match caps {
            Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels: 2,
                ..
            } => Ok(caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }
}

impl AsyncElement for AudioPanorama {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        self.accept_input(upstream_caps)
    }

    /// Native `DerivedOutput`: panning preserves format, channels, and rate, so
    /// the output caps equal the input for S16LE stereo.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels: 2,
                ..
            } => CapsSet::one(input.clone()),
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
                    apply_pan(src, &mut dst, self.panorama);

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
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        AUDIOPANORAMA_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "panorama" => self.panorama = value.as_double().ok_or(PropError::Type)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "panorama" => Some(PropValue::Double(self.panorama)),
            _ => None,
        }
    }
}

/// `AudioPanorama`'s settable properties (M104).
static AUDIOPANORAMA_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "panorama",
    PropKind::Double,
    "stereo position, -1 (left) .. 1 (right)",
)];

impl PadTemplates for AudioPanorama {
    fn pad_templates() -> Vec<PadTemplate> {
        let stereo = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
        };
        Vec::from([
            PadTemplate::sink(CapsSet::one(stereo.clone())),
            PadTemplate::source(CapsSet::one(stereo)),
        ])
    }
}

fn scale(sample: i16, gain: f64) -> i16 {
    let scaled = (sample as f64) * gain;
    // round half away from zero without libm, then clamp to i16.
    let rounded = if scaled >= 0.0 {
        scaled + 0.5
    } else {
        scaled - 0.5
    };
    rounded.clamp(i16::MIN as f64, i16::MAX as f64) as i16
}

/// Apply the balance pan to each interleaved L/R S16LE pair. A positive pan
/// attenuates the left channel, a negative pan the right; `dst` matches `src`.
fn apply_pan(src: &[u8], dst: &mut [u8], panorama: f64) {
    let pan = panorama.clamp(-1.0, 1.0);
    let left_gain = if pan > 0.0 { 1.0 - pan } else { 1.0 };
    let right_gain = if pan < 0.0 { 1.0 + pan } else { 1.0 };
    for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
        let l = scale(i16::from_le_bytes([s[0], s[1]]), left_gain);
        let r = scale(i16::from_le_bytes([s[2], s[3]]), right_gain);
        d[0..2].copy_from_slice(&l.to_le_bytes());
        d[2..4].copy_from_slice(&r.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stereo(pairs: &[(i16, i16)]) -> Vec<u8> {
        let mut v = vec![0u8; pairs.len() * 4];
        for (i, (l, r)) in pairs.iter().enumerate() {
            v[i * 4..i * 4 + 2].copy_from_slice(&l.to_le_bytes());
            v[i * 4 + 2..i * 4 + 4].copy_from_slice(&r.to_le_bytes());
        }
        v
    }

    fn pan(src: &[u8], panorama: f64) -> Vec<u8> {
        let mut dst = vec![0u8; src.len()];
        apply_pan(src, &mut dst, panorama);
        dst
    }

    #[test]
    fn centre_is_identity() {
        let src = stereo(&[(1000, -2000), (32000, 16000)]);
        assert_eq!(pan(&src, 0.0), src);
    }

    #[test]
    fn full_right_silences_left() {
        let out = pan(&stereo(&[(1000, 2000)]), 1.0);
        assert_eq!(out, stereo(&[(0, 2000)]));
    }

    #[test]
    fn full_left_silences_right() {
        let out = pan(&stereo(&[(1000, 2000)]), -1.0);
        assert_eq!(out, stereo(&[(1000, 0)]));
    }

    #[test]
    fn half_right_attenuates_left_only() {
        let out = pan(&stereo(&[(1000, 2000)]), 0.5);
        // left *0.5 = 500, right unchanged.
        assert_eq!(out, stereo(&[(500, 2000)]));
    }

    #[test]
    fn configure_requires_s16le_stereo() {
        let mut p = AudioPanorama::new();
        let mono = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 1,
            sample_rate: 48_000,
        };
        assert_eq!(
            p.configure_pipeline(&mono).unwrap_err(),
            G2gError::CapsMismatch
        );
        let stereo = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 44_100,
        };
        assert!(p.configure_pipeline(&stereo).is_ok());
    }
}
