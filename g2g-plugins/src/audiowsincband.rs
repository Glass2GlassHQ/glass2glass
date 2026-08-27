//! Windowed-sinc band-pass / band-reject (`audiowsincband`). A linear-phase FIR
//! whose kernel is the sum of a low-pass at `lower-frequency` and a high-pass at
//! `upper-frequency`, `length` taps long and multiplied by `window`. Preserves
//! format, channel count, and sample rate. CPU-only `no_std`.
//!
//! Matches GStreamer's `audiowsincband`: that sum is the band-reject kernel and
//! `band-pass` is its spectral inversion. Group delay is handled exactly as in
//! [`crate::audiowsinclimit`].

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::audiofx::{self, BandMode, FirWindow};

const DEFAULT_FREQUENCY: f64 = 0.0;
const FREQUENCY_MIN: f64 = 0.0;
const FREQUENCY_MAX: f64 = 100_000.0;
const DEFAULT_LENGTH: i64 = 101;
const LENGTH_MIN: i64 = 3;
const LENGTH_MAX: i64 = 256_000;

/// A linear-phase kernel needs an odd tap count so its peak lands on one sample.
fn to_odd(length: i64) -> usize {
    let length = length.clamp(LENGTH_MIN, LENGTH_MAX);
    (if length % 2 == 0 { length + 1 } else { length }) as usize
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::audiowsincband::AudioWsincBand;
///
/// let band_pass = AudioWsincBand::new()
///     .with_lower_frequency(1000.0)
///     .with_upper_frequency(4000.0);
/// ```
#[derive(Debug)]
pub struct AudioWsincBand {
    mode: BandMode,
    lower_frequency: f64,
    upper_frequency: f64,
    length: usize,
    window: FirWindow,
    transform: audiofx::FirTransform,
    last_caps: Option<Caps>,
}

impl Default for AudioWsincBand {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioWsincBand {
    pub fn new() -> Self {
        Self {
            mode: BandMode::BandPass,
            lower_frequency: DEFAULT_FREQUENCY,
            upper_frequency: DEFAULT_FREQUENCY,
            length: to_odd(DEFAULT_LENGTH),
            window: FirWindow::Hamming,
            transform: audiofx::FirTransform::default(),
            last_caps: None,
        }
    }

    pub fn with_mode(mut self, mode: BandMode) -> Self {
        self.mode = mode;
        self.rebuild();
        self
    }

    pub fn with_lower_frequency(mut self, frequency: f64) -> Self {
        self.lower_frequency = frequency.clamp(FREQUENCY_MIN, FREQUENCY_MAX);
        self.rebuild();
        self
    }

    pub fn with_upper_frequency(mut self, frequency: f64) -> Self {
        self.upper_frequency = frequency.clamp(FREQUENCY_MIN, FREQUENCY_MAX);
        self.rebuild();
        self
    }

    pub fn with_length(mut self, length: i64) -> Self {
        self.length = to_odd(length);
        self.rebuild();
        self
    }

    pub fn with_window(mut self, window: FirWindow) -> Self {
        self.window = window;
        self.rebuild();
        self
    }

    /// The kernel these settings describe. Empty until the rate is known.
    fn kernel(&self, rate: u32) -> Vec<f64> {
        if rate == 0 {
            return Vec::new();
        }
        audiofx::band_kernel(
            self.mode,
            self.lower_frequency,
            self.upper_frequency,
            rate,
            self.length,
            self.window,
        )
    }

    fn rebuild(&mut self) {
        let rate = self.transform.rate();
        if rate == 0 {
            return;
        }
        self.transform.set_kernel(self.kernel(rate));
    }

    /// Group delay in samples, the latency this element adds.
    pub fn latency_samples(&self) -> usize {
        self.transform.latency()
    }

    fn configure(&mut self, caps: &Caps) -> Result<(), G2gError> {
        let (_, _, rate) = audiofx::accept_audio(caps, None)?;
        let kernel = self.kernel(rate);
        self.transform.configure(caps, kernel)
    }
}

impl AsyncElement for AudioWsincBand {
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
        AUDIOWSINCBAND_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Windowed sinc band-pass / band-reject filter",
            "Filter/Effect/Audio",
            "Band-pass and band-reject windowed sinc filter",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "mode" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.mode = BandMode::from_str(s).ok_or(PropError::Value)?;
            }
            "lower-frequency" => {
                self.lower_frequency =
                    audiofx::double_in_range(value, FREQUENCY_MIN, FREQUENCY_MAX)?
            }
            "upper-frequency" => {
                self.upper_frequency =
                    audiofx::double_in_range(value, FREQUENCY_MIN, FREQUENCY_MAX)?
            }
            "length" => self.length = to_odd(audiofx::int_in_range(value, LENGTH_MIN, LENGTH_MAX)?),
            "window" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.window = FirWindow::from_str(s).ok_or(PropError::Value)?;
            }
            _ => return Err(PropError::Unknown),
        }
        self.rebuild();
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "mode" => Some(PropValue::Str(self.mode.as_str().into())),
            "lower-frequency" => Some(PropValue::Double(self.lower_frequency)),
            "upper-frequency" => Some(PropValue::Double(self.upper_frequency)),
            "length" => Some(PropValue::Int(self.length as i64)),
            "window" => Some(PropValue::Str(self.window.as_str().into())),
            _ => None,
        }
    }
}

static AUDIOWSINCBAND_PROPS: &[PropertySpec] = &[
    PropertySpec::new("mode", PropKind::Str, "band-pass or band-reject")
        .with_enum_values(audiofx::BAND_MODE_VALUES)
        .with_default("band-pass"),
    PropertySpec::new(
        "lower-frequency",
        PropKind::Double,
        "lower cut-off frequency of the band in Hz",
    )
    .with_range("0", "100000")
    .with_default("0"),
    PropertySpec::new(
        "upper-frequency",
        PropKind::Double,
        "upper cut-off frequency of the band in Hz",
    )
    .with_range("0", "100000")
    .with_default("0"),
    PropertySpec::new(
        "length",
        PropKind::Int,
        "filter kernel length in taps, rounded up to the next odd number",
    )
    .with_range("3", "256000")
    .with_default("101"),
    PropertySpec::new(
        "window",
        PropKind::Str,
        "window function applied to the sinc",
    )
    .with_enum_values(audiofx::FIR_WINDOW_VALUES)
    .with_default("hamming"),
];

impl PadTemplates for AudioWsincBand {
    fn pad_templates() -> Vec<PadTemplate> {
        audiofx::default_pad_templates()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::{AudioFormat, Caps};

    #[test]
    fn swapped_frequencies_still_build_a_band() {
        // the kernel builder orders the pair, so an inverted band is not empty.
        let mut e = AudioWsincBand::new()
            .with_lower_frequency(4000.0)
            .with_upper_frequency(1000.0);
        e.configure(&Caps::Audio {
            format: AudioFormat::PcmF32Le,
            channels: 1,
            sample_rate: 48_000,
        })
        .unwrap();
        assert_eq!(e.latency_samples(), 50);
    }

    #[test]
    fn mode_rejects_an_unknown_spelling() {
        let mut e = AudioWsincBand::new();
        assert_eq!(
            e.set_property("mode", PropValue::Str("notch".into()))
                .unwrap_err(),
            PropError::Value
        );
    }
}
