//! Phase inversion (`audioinvert`). Blends the signal with its own negation:
//! `degree` 0 leaves it alone, 0.5 cancels it to silence, 1 flips its phase.
//! Preserves format, channel count, and sample rate. CPU-only `no_std`.
//!
//! Matches GStreamer's `audioinvert`, whose float path is `out = in * (1 -
//! degree) - in * degree`. Every g2g audiofx filter works in f32, so the S16LE
//! stream takes the same transfer function rather than the reference's integer
//! path, which folds the sample around -1 instead of 0.

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

/// `degree` 0 is a pass-through, matching the reference's default.
const DEFAULT_DEGREE: f64 = 0.0;
const DEGREE_MIN: f64 = 0.0;
const DEGREE_MAX: f64 = 1.0;

/// The reference's float transfer function: a linear blend between the signal
/// and its negation.
pub(crate) fn invert_sample(sample: f32, degree: f64) -> f32 {
    (sample as f64 * (1.0 - degree) - sample as f64 * degree) as f32
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::audioinvert::AudioInvert;
///
/// let invert = AudioInvert::new().with_degree(1.0);
/// ```
#[derive(Debug)]
pub struct AudioInvert {
    degree: f64,
    format: AudioFormat,
    caps: Option<Caps>,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl Default for AudioInvert {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioInvert {
    pub fn new() -> Self {
        Self {
            degree: DEFAULT_DEGREE,
            format: AudioFormat::PcmS16Le,
            caps: None,
            last_caps: None,
            emitted: 0,
        }
    }

    pub fn with_degree(mut self, degree: f64) -> Self {
        self.degree = degree.clamp(DEGREE_MIN, DEGREE_MAX);
        self
    }

    fn configure(&mut self, caps: &Caps) -> Result<(), G2gError> {
        let (format, _, _) = audiofx::accept_audio(caps, None)?;
        self.format = format;
        self.caps = Some(caps.clone());
        Ok(())
    }
}

impl AsyncElement for AudioInvert {
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
                    let mut samples = audiofx::decode(src, self.format);
                    for sample in samples.iter_mut() {
                        *sample = invert_sample(*sample, self.degree);
                    }
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
        AUDIOINVERT_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Audio inversion",
            "Filter/Effect/Audio",
            "Swaps upper and lower half of audio samples",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "degree" => self.degree = audiofx::double_in_range(value, DEGREE_MIN, DEGREE_MAX)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "degree" => Some(PropValue::Double(self.degree)),
            _ => None,
        }
    }
}

static AUDIOINVERT_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "degree",
    PropKind::Double,
    "degree of inversion: 0 leaves the signal, 0.5 silences it, 1 flips its phase",
)
.with_range("0", "1")
.with_default("0")];

impl PadTemplates for AudioInvert {
    fn pad_templates() -> Vec<PadTemplate> {
        audiofx::default_pad_templates()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn degree_zero_is_a_pass_through() {
        assert_eq!(invert_sample(0.25, 0.0), 0.25);
    }

    #[test]
    fn degree_one_negates() {
        assert_eq!(invert_sample(0.25, 1.0), -0.25);
    }

    #[test]
    fn half_degree_cancels() {
        assert_eq!(invert_sample(0.25, 0.5), 0.0);
    }

    #[test]
    fn degree_is_range_checked() {
        let mut e = AudioInvert::new();
        assert_eq!(
            e.set_property("degree", PropValue::Double(1.5))
                .unwrap_err(),
            PropError::Value
        );
    }
}
