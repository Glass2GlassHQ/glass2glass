//! Generic FIR filter with a hand-written kernel (`audiofirfilter`). The
//! kernel is the filter's impulse response; the element convolves it with every
//! channel. Preserves format, channel count, and sample rate. CPU-only
//! `no_std`.
//!
//! Matches GStreamer's `audiofirfilter`, which shares its convolution and its
//! group-delay bookkeeping with `audiowsinclimit`: `latency` output samples are
//! dropped at the head and the same count is convolved out of the history at
//! `Eos`, so the output keeps the input's timestamps and total sample count. An
//! arbitrary kernel has no delay the filter can work out for itself, hence the
//! separate property, whose default 0 means no compensation.
//!
//! `PropKind` has no array kind, so the reference's `kernel=<0.25,0.5,0.25>`
//! GstValueArray is written here as a string of comma-separated coefficients,
//! `kernel="0.25,0.5,0.25"`. The default is the unit impulse `"1"`, the
//! reference's, which is a pass-through.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::audiofx;

/// The reference's default kernel and latency: a unit impulse, uncompensated.
const DEFAULT_KERNEL_TEXT: &str = "1";
const DEFAULT_LATENCY: u64 = 0;
const DEFAULT_LATENCY_TEXT: &str = "0";

/// # Example
///
/// ```no_run
/// use g2g_plugins::audiofirfilter::AudioFirFilter;
///
/// // a three-tap moving average, its peak one sample in.
/// let smooth = AudioFirFilter::new()
///     .with_kernel("0.3333,0.3333,0.3333")
///     .with_latency(1);
/// ```
#[derive(Debug)]
pub struct AudioFirFilter {
    kernel: Vec<f64>,
    latency: u64,
    transform: audiofx::FirTransform,
    last_caps: Option<Caps>,
}

impl Default for AudioFirFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioFirFilter {
    pub fn new() -> Self {
        Self {
            kernel: audiofx::parse_coefficients(DEFAULT_KERNEL_TEXT).unwrap_or_default(),
            latency: DEFAULT_LATENCY,
            transform: audiofx::FirTransform::default(),
            last_caps: None,
        }
    }

    /// Set the kernel from a comma-separated coefficient list. A malformed
    /// entry leaves the kernel alone.
    pub fn with_kernel(mut self, kernel: &str) -> Self {
        if let Ok(taps) = audiofx::parse_coefficients(kernel) {
            self.kernel = taps;
            self.rebuild();
        }
        self
    }

    pub fn with_latency(mut self, latency: u64) -> Self {
        self.latency = latency;
        self.rebuild();
        self
    }

    /// The kernel's group delay in samples, as configured.
    pub fn latency_samples(&self) -> usize {
        self.transform.latency()
    }

    fn rebuild(&mut self) {
        if self.transform.rate() == 0 {
            return;
        }
        self.transform
            .set_kernel_with_latency(self.kernel.clone(), self.latency as usize);
    }

    fn configure(&mut self, caps: &Caps) -> Result<(), G2gError> {
        self.transform
            .configure_with_latency(caps, self.kernel.clone(), self.latency as usize)
    }
}

impl AsyncElement for AudioFirFilter {
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
                    let caps = self
                        .transform
                        .caps()
                        .cloned()
                        .ok_or(G2gError::NotConfigured)?;
                    let filtered = self
                        .transform
                        .filter(&frame, g2g_core::log::short_type_name::<Self>())?;
                    if let Some(out_frame) = filtered {
                        if self.last_caps.as_ref() != Some(&caps) {
                            out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                            self.last_caps = Some(caps);
                        }
                        out.push(PipelinePacket::DataFrame(out_frame)).await?;
                    }
                }
                PipelinePacket::CapsChanged(c) => {
                    self.configure(&c)?;
                }
                PipelinePacket::Flush => {
                    self.transform.reset();
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Eos => {
                    if let Some(tail) = self.transform.drain() {
                        out.push(PipelinePacket::DataFrame(tail)).await?;
                    }
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        AUDIOFIRFILTER_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Audio FIR filter",
            "Filter/Effect/Audio",
            "Generic audio FIR filter with custom filter kernel",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "kernel" => {
                let text = value.as_str().ok_or(PropError::Type)?;
                self.kernel = audiofx::parse_coefficients(text)?;
            }
            "latency" => self.latency = value.as_uint().ok_or(PropError::Type)?,
            _ => return Err(PropError::Unknown),
        }
        self.rebuild();
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "kernel" => Some(PropValue::Str(audiofx::format_coefficients(&self.kernel))),
            "latency" => Some(PropValue::Uint(self.latency)),
            _ => None,
        }
    }
}

static AUDIOFIRFILTER_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "kernel",
        PropKind::Str,
        "filter kernel for the FIR filter: comma-separated coefficients",
    )
    .with_default(DEFAULT_KERNEL_TEXT),
    PropertySpec::new(
        "latency",
        PropKind::Uint,
        "filter latency in samples, dropped from the head of the output",
    )
    .with_default(DEFAULT_LATENCY_TEXT),
];

impl PadTemplates for AudioFirFilter {
    fn pad_templates() -> Vec<PadTemplate> {
        audiofx::default_pad_templates()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::{AudioFormat, Caps};

    fn mono() -> Caps {
        Caps::Audio {
            format: AudioFormat::PcmF32Le,
            channels: 1,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }
    }

    #[test]
    fn the_default_kernel_is_a_pass_through() {
        let element = AudioFirFilter::new();
        assert_eq!(
            element.get_property("kernel"),
            Some(PropValue::Str(DEFAULT_KERNEL_TEXT.into()))
        );
        assert_eq!(element.get_property("latency"), Some(PropValue::Uint(0)));
    }

    #[test]
    fn kernel_round_trips_through_the_string_form() {
        let mut element = AudioFirFilter::new();
        element
            .set_property("kernel", PropValue::Str("0.25,0.5,0.25".into()))
            .unwrap();
        assert_eq!(
            element.get_property("kernel"),
            Some(PropValue::Str("0.25,0.5,0.25".into()))
        );
        assert_eq!(element.kernel, alloc::vec![0.25, 0.5, 0.25]);
    }

    #[test]
    fn a_malformed_coefficient_is_rejected() {
        let mut element = AudioFirFilter::new();
        assert_eq!(
            element
                .set_property("kernel", PropValue::Str("0.25,,0.5".into()))
                .unwrap_err(),
            PropError::Value
        );
    }

    #[test]
    fn latency_is_the_configured_one_not_the_kernel_centre() {
        let mut element = AudioFirFilter::new()
            .with_kernel("0.25,0.5,0.25")
            .with_latency(1);
        element.configure(&mono()).unwrap();
        assert_eq!(element.latency_samples(), 1);
        element.set_property("latency", PropValue::Uint(0)).unwrap();
        assert_eq!(element.latency_samples(), 0);
    }
}
