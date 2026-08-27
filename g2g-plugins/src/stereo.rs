//! Stereo widener (`stereo`). Splits each stereo frame into its mono average
//! and the two per-channel differences, then scales the differences: below 1
//! the image narrows toward mono, above 1 it widens. Preserves format, channel
//! count, and sample rate. Stereo only. CPU-only `no_std`.
//!
//! Matches GStreamer's `stereo`: `avg = (l + r) / 2`, `out = avg + (in - avg) *
//! stereo * 10`. The reference stores the widening factor pre-multiplied by ten
//! and divides on read, so the `stereo` property runs 0 to 1 while the factor
//! it applies runs 0 to 10. The reference is S16-only and clamps to the integer
//! range; every g2g audiofx filter works in f32, so the saturation lands at +-1.
//!
//! The reference's loop steps its index by two over a sample (not frame) count
//! halved, so it only reaches the first half of a buffer and skips every other
//! frame in it. Here every frame is widened.

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

/// A stereo image only exists on a stereo pair.
const STEREO_CHANNELS: u8 = 2;

/// The reference's defaults, as the property reports them.
const DEFAULT_ACTIVE: bool = true;
const DEFAULT_STEREO: f64 = 0.01;
const DEFAULT_ACTIVE_TEXT: &str = "true";
const DEFAULT_STEREO_TEXT: &str = "0.01";
const STEREO_MIN: f64 = 0.0;
const STEREO_MAX: f64 = 1.0;

/// The reference keeps the widening factor ten times the `stereo` property.
const STEREO_FACTOR_SCALE: f64 = 10.0;

/// # Example
///
/// ```no_run
/// use g2g_plugins::stereo::Stereo;
///
/// let widen = Stereo::new().with_stereo(0.2);
/// ```
#[derive(Debug)]
pub struct Stereo {
    active: bool,
    stereo: f64,
    format: AudioFormat,
    caps: Option<Caps>,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl Default for Stereo {
    fn default() -> Self {
        Self::new()
    }
}

impl Stereo {
    pub fn new() -> Self {
        Self {
            active: DEFAULT_ACTIVE,
            stereo: DEFAULT_STEREO,
            format: AudioFormat::PcmS16Le,
            caps: None,
            last_caps: None,
            emitted: 0,
        }
    }

    pub fn with_active(mut self, active: bool) -> Self {
        self.active = active;
        self
    }

    pub fn with_stereo(mut self, stereo: f64) -> Self {
        self.stereo = stereo.clamp(STEREO_MIN, STEREO_MAX);
        self
    }

    fn configure(&mut self, caps: &Caps) -> Result<(), G2gError> {
        let (format, _, _) = audiofx::accept_audio(caps, Some(STEREO_CHANNELS))?;
        self.format = format;
        self.caps = Some(caps.clone());
        Ok(())
    }

    /// Widen one interleaved stereo buffer in place.
    fn widen(&self, samples: &mut [f32]) {
        if !self.active {
            return;
        }
        let factor = self.stereo * STEREO_FACTOR_SCALE;
        for pair in samples.as_chunks_mut::<{ STEREO_CHANNELS as usize }>().0 {
            let left = pair[0] as f64;
            let right = pair[1] as f64;
            let average = (left + right) / 2.0;
            pair[0] = audiofx::clamp_sample(average + (left - average) * factor);
            pair[1] = audiofx::clamp_sample(average + (right - average) * factor);
        }
    }
}

impl AsyncElement for Stereo {
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
        audiofx::accept_audio(upstream_caps, Some(STEREO_CHANNELS))?;
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        audiofx::passthrough_constraint(Some(STEREO_CHANNELS))
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
                    self.widen(&mut samples);
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
        STEREO_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Stereo effect",
            "Filter/Effect/Audio",
            "Muck with the stereo signal to enhance its 'stereo-ness'",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "active" => self.active = value.as_bool().ok_or(PropError::Type)?,
            "stereo" => self.stereo = audiofx::double_in_range(value, STEREO_MIN, STEREO_MAX)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "active" => Some(PropValue::Bool(self.active)),
            "stereo" => Some(PropValue::Double(self.stereo)),
            _ => None,
        }
    }
}

static STEREO_PROPS: &[PropertySpec] = &[
    PropertySpec::new("active", PropKind::Bool, "apply the widening")
        .with_default(DEFAULT_ACTIVE_TEXT),
    PropertySpec::new(
        "stereo",
        PropKind::Double,
        "stereo widening, a tenth of the factor the channel differences are scaled by",
    )
    .with_range("0", "1")
    .with_default(DEFAULT_STEREO_TEXT),
];

impl PadTemplates for Stereo {
    fn pad_templates() -> Vec<PadTemplate> {
        audiofx::pad_templates(STEREO_CHANNELS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stereo_caps() -> Caps {
        Caps::Audio {
            format: AudioFormat::PcmF32Le,
            channels: STEREO_CHANNELS,
            sample_rate: 48_000,
        }
    }

    #[test]
    fn declared_defaults_match_the_constants() {
        let element = Stereo::new();
        assert_eq!(
            element.get_property("stereo"),
            Some(PropValue::Double(DEFAULT_STEREO))
        );
        assert_eq!(
            element.get_property("active"),
            Some(PropValue::Bool(DEFAULT_ACTIVE))
        );
    }

    #[test]
    fn a_centred_pair_is_untouched() {
        let element = Stereo::new().with_stereo(1.0);
        let mut samples = [0.25f32, 0.25];
        element.widen(&mut samples);
        assert_eq!(samples, [0.25, 0.25]);
    }

    #[test]
    fn the_difference_is_scaled_by_ten_times_the_property() {
        let element = Stereo::new().with_stereo(0.2);
        let mut samples = [0.3f32, 0.1];
        element.widen(&mut samples);
        // average 0.2, differences +-0.1, factor 2.
        assert!((samples[0] - 0.4).abs() < 1e-6, "got {}", samples[0]);
        assert!((samples[1] - 0.0).abs() < 1e-6, "got {}", samples[1]);
    }

    #[test]
    fn inactive_leaves_the_signal_alone() {
        let element = Stereo::new().with_active(false).with_stereo(1.0);
        let mut samples = [0.3f32, 0.1];
        element.widen(&mut samples);
        assert_eq!(samples, [0.3, 0.1]);
    }

    #[test]
    fn mono_is_rejected() {
        let mut element = Stereo::new();
        let mono = Caps::Audio {
            format: AudioFormat::PcmF32Le,
            channels: 1,
            sample_rate: 48_000,
        };
        assert_eq!(
            element.configure(&mono).unwrap_err(),
            G2gError::CapsMismatch
        );
        assert!(element.configure(&stereo_caps()).is_ok());
    }
}
