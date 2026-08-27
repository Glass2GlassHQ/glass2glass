//! Windowed-sinc low-pass / high-pass (`audiowsinclimit`). A linear-phase FIR
//! whose kernel is a sinc at `cutoff` multiplied by `window`, `length` taps
//! long. Preserves format, channel count, and sample rate. CPU-only `no_std`.
//!
//! Matches GStreamer's `audiowsinclimit`: the kernel is normalized for unity
//! gain at DC and spectrally inverted for `high-pass`, and it is rebuilt
//! whenever `mode`, `cutoff`, `length` or `window` changes. The `(length - 1) /
//! 2` samples of group delay are removed the way
//! `gstaudiofxbasefirfilter.c` does: the leading output of that many samples is
//! dropped and the same count is convolved out of the history at `Eos`, so the
//! output keeps the input's timestamps and total sample count.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::audiofx::{self, FirWindow, LimitMode};

const DEFAULT_CUTOFF: f64 = 0.0;
const CUTOFF_MIN: f64 = 0.0;
const CUTOFF_MAX: f64 = 100_000.0;
/// The reference's default kernel length, and the bounds it accepts.
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
/// use g2g_plugins::audiowsinclimit::AudioWsincLimit;
///
/// let low_pass = AudioWsincLimit::new().with_cutoff(1000.0).with_length(101);
/// ```
#[derive(Debug)]
pub struct AudioWsincLimit {
    mode: LimitMode,
    cutoff: f64,
    length: usize,
    window: FirWindow,
    transform: audiofx::FirTransform,
    last_caps: Option<Caps>,
}

impl Default for AudioWsincLimit {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioWsincLimit {
    pub fn new() -> Self {
        Self {
            mode: LimitMode::LowPass,
            cutoff: DEFAULT_CUTOFF,
            length: to_odd(DEFAULT_LENGTH),
            window: FirWindow::Hamming,
            transform: audiofx::FirTransform::default(),
            last_caps: None,
        }
    }

    pub fn with_mode(mut self, mode: LimitMode) -> Self {
        self.mode = mode;
        self.rebuild();
        self
    }

    pub fn with_cutoff(mut self, cutoff: f64) -> Self {
        self.cutoff = cutoff.clamp(CUTOFF_MIN, CUTOFF_MAX);
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
        audiofx::limit_kernel(self.mode, self.cutoff, rate, self.length, self.window)
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

impl AsyncElement for AudioWsincLimit {
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
        AUDIOWSINCLIMIT_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Windowed sinc low-pass / high-pass filter",
            "Filter/Effect/Audio",
            "Low-pass and high-pass windowed sinc filter",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "mode" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.mode = LimitMode::from_str(s).ok_or(PropError::Value)?;
            }
            "cutoff" => self.cutoff = audiofx::double_in_range(value, CUTOFF_MIN, CUTOFF_MAX)?,
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
            "cutoff" => Some(PropValue::Double(self.cutoff)),
            "length" => Some(PropValue::Int(self.length as i64)),
            "window" => Some(PropValue::Str(self.window.as_str().into())),
            _ => None,
        }
    }
}

static AUDIOWSINCLIMIT_PROPS: &[PropertySpec] = &[
    PropertySpec::new("mode", PropKind::Str, "low-pass or high-pass")
        .with_enum_values(audiofx::LIMIT_MODE_VALUES)
        .with_default("low-pass"),
    PropertySpec::new("cutoff", PropKind::Double, "cut-off frequency in Hz")
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

impl PadTemplates for AudioWsincLimit {
    fn pad_templates() -> Vec<PadTemplate> {
        audiofx::default_pad_templates()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::{AudioFormat, Caps};

    fn mono(rate: u32) -> Caps {
        Caps::Audio {
            format: AudioFormat::PcmF32Le,
            channels: 1,
            sample_rate: rate,
        }
    }

    #[test]
    fn length_is_rounded_to_odd() {
        let mut e = AudioWsincLimit::new();
        e.set_property("length", PropValue::Int(64)).unwrap();
        assert_eq!(e.get_property("length"), Some(PropValue::Int(65)));
    }

    #[test]
    fn latency_is_half_the_kernel() {
        let mut e = AudioWsincLimit::new().with_length(101);
        e.configure(&mono(48_000)).unwrap();
        assert_eq!(e.latency_samples(), 50);
    }

    #[test]
    fn window_rejects_an_unknown_spelling() {
        let mut e = AudioWsincLimit::new();
        assert_eq!(
            e.set_property("window", PropValue::Str("kaiser".into()))
                .unwrap_err(),
            PropError::Value
        );
    }
}
