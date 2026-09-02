//! Stereo channel mixer (`audiochannelmix`). Each output channel is a weighted
//! sum of the two input channels, so the four gains swap, fold to mono, or
//! cancel a side. Preserves format, channel count, and sample rate. Stereo
//! only, as in the reference's caps. CPU-only `no_std`.
//!
//! Matches GStreamer's `audiochannelmix`: `left = left-to-left * l +
//! right-to-left * r` and `right = left-to-right * l + right-to-right * r`. The
//! reference is S16-only and clamps to the integer range; every g2g audiofx
//! filter works in f32, so the saturation lands at +-1 instead.

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

/// There is nothing to cross-mix outside a stereo pair.
const CHANNEL_MIX_CHANNELS: u8 = 2;

/// The reference's defaults: the identity mix, each channel to itself.
const DEFAULT_STRAIGHT_GAIN: f64 = 1.0;
const DEFAULT_CROSS_GAIN: f64 = 0.0;
const DEFAULT_STRAIGHT_GAIN_TEXT: &str = "1";
const DEFAULT_CROSS_GAIN_TEXT: &str = "0";

/// # Example
///
/// ```no_run
/// use g2g_plugins::audiochannelmix::AudioChannelMix;
///
/// // swap the two channels.
/// let swap = AudioChannelMix::new()
///     .with_left_to_left(0.0)
///     .with_left_to_right(1.0)
///     .with_right_to_left(1.0)
///     .with_right_to_right(0.0);
/// ```
#[derive(Debug)]
pub struct AudioChannelMix {
    left_to_left: f64,
    left_to_right: f64,
    right_to_left: f64,
    right_to_right: f64,
    format: AudioFormat,
    caps: Option<Caps>,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl Default for AudioChannelMix {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioChannelMix {
    pub fn new() -> Self {
        Self {
            left_to_left: DEFAULT_STRAIGHT_GAIN,
            left_to_right: DEFAULT_CROSS_GAIN,
            right_to_left: DEFAULT_CROSS_GAIN,
            right_to_right: DEFAULT_STRAIGHT_GAIN,
            format: AudioFormat::PcmS16Le,
            caps: None,
            last_caps: None,
            emitted: 0,
        }
    }

    pub fn with_left_to_left(mut self, gain: f64) -> Self {
        self.left_to_left = gain;
        self
    }

    pub fn with_left_to_right(mut self, gain: f64) -> Self {
        self.left_to_right = gain;
        self
    }

    pub fn with_right_to_left(mut self, gain: f64) -> Self {
        self.right_to_left = gain;
        self
    }

    pub fn with_right_to_right(mut self, gain: f64) -> Self {
        self.right_to_right = gain;
        self
    }

    fn configure(&mut self, caps: &Caps) -> Result<(), G2gError> {
        let (format, _, _) = audiofx::accept_audio(caps, Some(CHANNEL_MIX_CHANNELS))?;
        self.format = format;
        self.caps = Some(caps.clone());
        Ok(())
    }

    /// Mix one interleaved stereo buffer in place.
    fn mix(&self, samples: &mut [f32]) {
        for pair in samples
            .as_chunks_mut::<{ CHANNEL_MIX_CHANNELS as usize }>()
            .0
        {
            let left = pair[0] as f64;
            let right = pair[1] as f64;
            pair[0] = audiofx::clamp_sample(self.left_to_left * left + self.right_to_left * right);
            pair[1] =
                audiofx::clamp_sample(self.left_to_right * left + self.right_to_right * right);
        }
    }
}

impl AsyncElement for AudioChannelMix {
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
        audiofx::accept_audio(upstream_caps, Some(CHANNEL_MIX_CHANNELS))?;
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        audiofx::passthrough_constraint(Some(CHANNEL_MIX_CHANNELS))
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
                    self.mix(&mut samples);
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
        AUDIOCHANNELMIX_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Simple stereo audio mixer",
            "Filter/Effect/Audio",
            "Mixes left/right channels of stereo audio",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        let gain = value.as_double().ok_or(PropError::Type)?;
        if !gain.is_finite() {
            return Err(PropError::Value);
        }
        match name {
            "left-to-left" => self.left_to_left = gain,
            "left-to-right" => self.left_to_right = gain,
            "right-to-left" => self.right_to_left = gain,
            "right-to-right" => self.right_to_right = gain,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        let gain = match name {
            "left-to-left" => self.left_to_left,
            "left-to-right" => self.left_to_right,
            "right-to-left" => self.right_to_left,
            "right-to-right" => self.right_to_right,
            _ => return None,
        };
        Some(PropValue::Double(gain))
    }
}

static AUDIOCHANNELMIX_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "left-to-left",
        PropKind::Double,
        "left channel to left channel gain",
    )
    .with_default(DEFAULT_STRAIGHT_GAIN_TEXT),
    PropertySpec::new(
        "left-to-right",
        PropKind::Double,
        "left channel to right channel gain",
    )
    .with_default(DEFAULT_CROSS_GAIN_TEXT),
    PropertySpec::new(
        "right-to-left",
        PropKind::Double,
        "right channel to left channel gain",
    )
    .with_default(DEFAULT_CROSS_GAIN_TEXT),
    PropertySpec::new(
        "right-to-right",
        PropKind::Double,
        "right channel to right channel gain",
    )
    .with_default(DEFAULT_STRAIGHT_GAIN_TEXT),
];

impl PadTemplates for AudioChannelMix {
    fn pad_templates() -> Vec<PadTemplate> {
        audiofx::pad_templates(CHANNEL_MIX_CHANNELS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stereo() -> Caps {
        Caps::Audio {
            format: AudioFormat::PcmF32Le,
            channels: CHANNEL_MIX_CHANNELS,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }
    }

    #[test]
    fn the_default_mix_is_the_identity() {
        let mut element = AudioChannelMix::new();
        element.configure(&stereo()).unwrap();
        let mut samples = [0.25f32, -0.5];
        element.mix(&mut samples);
        assert_eq!(samples, [0.25, -0.5]);
    }

    #[test]
    fn a_cross_mix_swaps_the_channels() {
        let mut element = AudioChannelMix::new()
            .with_left_to_left(0.0)
            .with_left_to_right(1.0)
            .with_right_to_left(1.0)
            .with_right_to_right(0.0);
        element.configure(&stereo()).unwrap();
        let mut samples = [0.25f32, -0.5];
        element.mix(&mut samples);
        assert_eq!(samples, [-0.5, 0.25]);
    }

    #[test]
    fn a_mix_past_full_scale_saturates() {
        let mut element = AudioChannelMix::new().with_right_to_left(1.0);
        element.configure(&stereo()).unwrap();
        let mut samples = [0.75f32, 0.75];
        element.mix(&mut samples);
        assert_eq!(samples[0], 1.0);
    }

    #[test]
    fn mono_is_rejected() {
        let mut element = AudioChannelMix::new();
        let mono = Caps::Audio {
            format: AudioFormat::PcmF32Le,
            channels: 1,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        assert_eq!(
            element.configure(&mono).unwrap_err(),
            G2gError::CapsMismatch
        );
    }
}
