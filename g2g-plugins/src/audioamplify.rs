//! Audio amplification (`audioamplify`). Scales every signed-16-bit PCM sample
//! by an `amplification` factor, choosing how out-of-range results are folded
//! back into the i16 range via `amplification-method`. Preserves format, channel
//! count, and sample rate. CPU-only `no_std`.
//!
//! The three methods match GStreamer's `audioamplify`: `clip` clamps to the i16
//! range (the default), `wrap-negative` is a two's-complement wrap (overflow
//! flips sign), and `wrap-positive` reflects at the boundary (overflow keeps its
//! sign). Only `PcmS16Le` is handled; an `f32` path is a follow-up.

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

/// How an amplified sample that leaves the i16 range is mapped back into it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmplifyMethod {
    /// Clamp to `[i16::MIN, i16::MAX]`.
    Clip,
    /// Two's-complement wrap (overflow flips sign).
    WrapNegative,
    /// Reflect at the boundary (overflow keeps its sign).
    WrapPositive,
}

impl AmplifyMethod {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "clip" => Some(Self::Clip),
            "wrap-negative" => Some(Self::WrapNegative),
            "wrap-positive" => Some(Self::WrapPositive),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Clip => "clip",
            Self::WrapNegative => "wrap-negative",
            Self::WrapPositive => "wrap-positive",
        }
    }
}

#[derive(Debug)]
pub struct AudioAmplify {
    amplification: f64,
    method: AmplifyMethod,
    input: Option<Caps>,
    configured: bool,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl Default for AudioAmplify {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioAmplify {
    /// Unity gain, `clip` method; use the builders or properties to adjust.
    pub fn new() -> Self {
        Self {
            amplification: 1.0,
            method: AmplifyMethod::Clip,
            input: None,
            configured: false,
            last_caps: None,
            emitted: 0,
        }
    }

    pub fn with_amplification(mut self, amplification: f64) -> Self {
        self.amplification = amplification;
        self
    }

    pub fn with_method(mut self, method: AmplifyMethod) -> Self {
        self.method = method;
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

impl AsyncElement for AudioAmplify {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        self.accept_input(upstream_caps)
    }

    /// Native `DerivedOutput`: amplification preserves format, channels, and rate,
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
                    apply_amplify(src, &mut dst, self.amplification, self.method);

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
        AUDIOAMPLIFY_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new("Audio amplify", "Filter/Effect/Audio", "Amplifies an audio stream", "g2g")
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "amplification" => self.amplification = value.as_double().ok_or(PropError::Type)?,
            "amplification-method" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.method = AmplifyMethod::from_str(s).ok_or(PropError::Value)?;
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "amplification" => Some(PropValue::Double(self.amplification)),
            "amplification-method" => Some(PropValue::Str(self.method.as_str().into())),
            _ => None,
        }
    }
}

/// `AudioAmplify`'s settable properties.
static AUDIOAMPLIFY_PROPS: &[PropertySpec] = &[
    PropertySpec::new("amplification", PropKind::Double, "linear gain (1 = unchanged)"),
    PropertySpec::new(
        "amplification-method",
        PropKind::Str,
        "overflow handling: clip | wrap-negative | wrap-positive",
    ),
];

impl PadTemplates for AudioAmplify {
    fn pad_templates() -> Vec<PadTemplate> {
        let pcm = Caps::Audio { format: AudioFormat::PcmS16Le, channels: 2, sample_rate: 48_000 };
        Vec::from([PadTemplate::sink(CapsSet::one(pcm.clone())), PadTemplate::source(CapsSet::one(pcm))])
    }
}

/// Map an amplified sample value into the i16 range by the chosen method.
fn fold(scaled: i64, method: AmplifyMethod) -> i16 {
    const MIN: i64 = i16::MIN as i64;
    const MAX: i64 = i16::MAX as i64;
    match method {
        AmplifyMethod::Clip => scaled.clamp(MIN, MAX) as i16,
        // Truncating to i16 is two's-complement wrap.
        AmplifyMethod::WrapNegative => scaled as i16,
        AmplifyMethod::WrapPositive => {
            if scaled > MAX {
                (MAX - (scaled - MAX)).clamp(MIN, MAX) as i16
            } else if scaled < MIN {
                (MIN - (scaled - MIN)).clamp(MIN, MAX) as i16
            } else {
                scaled as i16
            }
        }
    }
}

/// Scale each interleaved S16LE sample by `amplification`, folding out-of-range
/// results per `method`. `dst` is the same length as `src`.
fn apply_amplify(src: &[u8], dst: &mut [u8], amplification: f64, method: AmplifyMethod) {
    for (s, d) in src.chunks_exact(2).zip(dst.chunks_exact_mut(2)) {
        let sample = i16::from_le_bytes([s[0], s[1]]);
        let scaled = (sample as f64) * amplification;
        // round half away from zero without libm.
        let rounded = if scaled >= 0.0 { scaled + 0.5 } else { scaled - 0.5 } as i64;
        d.copy_from_slice(&fold(rounded, method).to_le_bytes());
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

    fn amp(samples: &[i16], gain: f64, method: AmplifyMethod) -> Vec<i16> {
        let src = pack(samples);
        let mut dst = vec![0u8; src.len()];
        apply_amplify(&src, &mut dst, gain, method);
        unpack(&dst)
    }

    #[test]
    fn unity_is_identity() {
        assert_eq!(amp(&[1234, -5678, 32000], 1.0, AmplifyMethod::Clip), [1234, -5678, 32000]);
    }

    #[test]
    fn clip_clamps_overflow() {
        // 20000*2 = 40000 saturates at i16::MAX; the negative saturates at MIN.
        assert_eq!(amp(&[20000, -20000], 2.0, AmplifyMethod::Clip), [i16::MAX, i16::MIN]);
    }

    #[test]
    fn wrap_negative_flips_sign_on_overflow() {
        // 20000*2 = 40000 wraps two's-complement: 40000 - 65536 = -25536.
        assert_eq!(amp(&[20000], 2.0, AmplifyMethod::WrapNegative), [-25536]);
    }

    #[test]
    fn wrap_positive_reflects_and_keeps_sign() {
        // 40000 reflects at MAX (32767): 32767 - (40000 - 32767) = 25534.
        assert_eq!(amp(&[20000], 2.0, AmplifyMethod::WrapPositive), [25534]);
    }

    #[test]
    fn configure_rejects_non_s16le() {
        let mut a = AudioAmplify::new();
        let f32_caps =
            Caps::Audio { format: AudioFormat::PcmF32Le, channels: 2, sample_rate: 48_000 };
        assert_eq!(a.configure_pipeline(&f32_caps).unwrap_err(), G2gError::CapsMismatch);
        let ok = Caps::Audio { format: AudioFormat::PcmS16Le, channels: 1, sample_rate: 16_000 };
        assert!(a.configure_pipeline(&ok).is_ok());
    }

    #[test]
    fn method_property_round_trips() {
        let mut a = AudioAmplify::new();
        a.set_property("amplification-method", PropValue::Str("wrap-positive".into())).unwrap();
        assert_eq!(a.get_property("amplification-method"), Some(PropValue::Str("wrap-positive".into())));
        assert_eq!(
            a.set_property("amplification-method", PropValue::Str("bogus".into())).unwrap_err(),
            PropError::Value
        );
    }
}
