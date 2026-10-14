//! Generic IIR filter with hand-written coefficients (`audioiirfilter`). `b` is
//! the numerator of the transfer function and `a` its denominator, applied as a
//! direct-form-I recurrence with one history per channel. Preserves format,
//! channel count, and sample rate. CPU-only `no_std`.
//!
//! Matches GStreamer's `audioiirfilter`, whose recurrence is
//! `y[n] = (sum b[i] * x[n-i] - sum a[i] * y[n-i]) / a[0]`, the second sum from
//! `i = 1`. `a[0]` therefore normalizes the whole output, and an `a` of `"1"`
//! (the default, along with `b`) makes the element a pass-through.
//!
//! `PropKind` has no array kind, so the reference's `a=<1.0,-0.5>`
//! GstValueArray is written here as a string of comma-separated coefficients,
//! `a="1.0,-0.5"`. An empty `a` or `b` list, or an `a[0]` of zero, would leave
//! the recurrence undefined, so both are rejected at the property.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec,
};

use crate::audiofx;

/// The reference's defaults: a single unity tap on both sides, a pass-through.
const DEFAULT_COEFFICIENTS_TEXT: &str = "1";

// the last `len` values, each stored at `head` and `head + len` so newest-first is one slice
#[derive(Debug, Default, Clone)]
struct History {
    values: Vec<f64>,
    len: usize,
    head: usize,
}

impl History {
    fn new(len: usize) -> Self {
        Self {
            values: vec![0.0; 2 * len],
            len,
            head: 0,
        }
    }

    fn reset(&mut self) {
        self.values.fill(0.0);
        self.head = 0;
    }

    fn newest_first(&self) -> &[f64] {
        &self.values[self.head..self.head + self.len]
    }

    fn push(&mut self, value: f64) {
        if self.len == 0 {
            return;
        }
        self.head = self.head.checked_sub(1).unwrap_or(self.len - 1);
        self.values[self.head] = value;
        self.values[self.head + self.len] = value;
    }
}

/// Direct-form-I state of one channel: the last inputs and the last outputs.
#[derive(Debug, Default, Clone)]
struct ChannelHistory {
    x: History,
    y: History,
}

impl ChannelHistory {
    fn new(numerator_taps: usize, denominator_taps: usize) -> Self {
        Self {
            x: History::new(numerator_taps.saturating_sub(1)),
            y: History::new(denominator_taps.saturating_sub(1)),
        }
    }

    fn reset(&mut self) {
        self.x.reset();
        self.y.reset();
    }

    fn step(&mut self, b: &[f64], a: &[f64], x0: f64) -> f64 {
        let mut value = b[0] * x0;
        for (coefficient, past) in b[1..].iter().zip(self.x.newest_first()) {
            value += coefficient * past;
        }
        for (coefficient, past) in a[1..].iter().zip(self.y.newest_first()) {
            value -= coefficient * past;
        }
        value /= a[0];
        self.x.push(x0);
        self.y.push(value);
        value
    }
}

/// Parse a coefficient list that has to leave the recurrence defined: at least
/// one entry, and (for `a`) a non-zero leading entry to divide by.
fn parse_denominator(text: &str) -> Result<Vec<f64>, PropError> {
    let values = audiofx::parse_coefficients(text)?;
    match values.first() {
        Some(leading) if *leading != 0.0 => Ok(values),
        _ => Err(PropError::Value),
    }
}

fn parse_numerator(text: &str) -> Result<Vec<f64>, PropError> {
    let values = audiofx::parse_coefficients(text)?;
    if values.is_empty() {
        return Err(PropError::Value);
    }
    Ok(values)
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::audioiirfilter::AudioIirFilter;
///
/// // a one-pole low-pass: y[n] = 0.5 * x[n] + 0.5 * y[n-1].
/// let low_pass = AudioIirFilter::new().with_b("0.5").with_a("1,-0.5");
/// ```
#[derive(Debug)]
pub struct AudioIirFilter {
    /// Numerator of the transfer function.
    b: Vec<f64>,
    /// Denominator of the transfer function, `a[0]` non-zero.
    a: Vec<f64>,
    channels: Vec<ChannelHistory>,
    format: AudioFormat,
    caps: Option<Caps>,
    last_caps: Option<Caps>,
    emitted: u64,
    scratch: audiofx::PcmScratch,
}

impl Default for AudioIirFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioIirFilter {
    pub fn new() -> Self {
        let unity = audiofx::parse_coefficients(DEFAULT_COEFFICIENTS_TEXT).unwrap_or_default();
        Self {
            b: unity.clone(),
            a: unity,
            channels: Vec::new(),
            format: AudioFormat::PcmS16Le,
            caps: None,
            last_caps: None,
            emitted: 0,
            scratch: audiofx::PcmScratch::default(),
        }
    }

    /// Set the numerator from a comma-separated coefficient list. A malformed
    /// list leaves it alone.
    pub fn with_b(mut self, b: &str) -> Self {
        if let Ok(values) = parse_numerator(b) {
            self.b = values;
            self.reset_history();
        }
        self
    }

    /// Set the denominator from a comma-separated coefficient list. A malformed
    /// list leaves it alone.
    pub fn with_a(mut self, a: &str) -> Self {
        if let Ok(values) = parse_denominator(a) {
            self.a = values;
            self.reset_history();
        }
        self
    }

    fn reset_history(&mut self) {
        let history = ChannelHistory::new(self.b.len(), self.a.len());
        for channel in self.channels.iter_mut() {
            *channel = history.clone();
        }
    }

    fn configure(&mut self, caps: &Caps) -> Result<(), G2gError> {
        let (format, channels, _) = audiofx::accept_audio(caps, None)?;
        self.format = format;
        self.caps = Some(caps.clone());
        self.channels = vec![ChannelHistory::new(self.b.len(), self.a.len()); channels];
        Ok(())
    }

    fn reset(&mut self) {
        for channel in self.channels.iter_mut() {
            channel.reset();
        }
    }

    /// Filter one interleaved buffer in place.
    fn filter(&mut self, samples: &mut [f32]) {
        let count = self.channels.len();
        if count == 0 {
            return;
        }
        for (index, sample) in samples.iter_mut().enumerate() {
            let value = self.channels[index % count].step(&self.b, &self.a, *sample as f64);
            *sample = value as f32;
        }
    }
}

impl AsyncElement for AudioIirFilter {
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
                    let element = g2g_core::log::short_type_name::<Self>();
                    let mut scratch = core::mem::take(&mut self.scratch);
                    let filtered =
                        scratch.transform_frame(frame, self.format, element, |samples| {
                            self.filter(samples)
                        });
                    self.scratch = scratch;
                    let frame = filtered?;

                    if self.last_caps.as_ref() != Some(&caps) {
                        out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                        self.last_caps = Some(caps);
                    }
                    let out_frame = Frame {
                        domain: frame.domain,
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
                    self.reset();
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
        AUDIOIIRFILTER_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Audio IIR filter",
            "Filter/Effect/Audio",
            "Generic audio IIR filter with custom filter kernel",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        let text = value.as_str().ok_or(PropError::Type)?;
        match name {
            "a" => self.a = parse_denominator(text)?,
            "b" => self.b = parse_numerator(text)?,
            _ => return Err(PropError::Unknown),
        }
        self.reset_history();
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        let coefficients = match name {
            "a" => &self.a,
            "b" => &self.b,
            _ => return None,
        };
        Some(PropValue::Str(audiofx::format_coefficients(coefficients)))
    }
}

static AUDIOIIRFILTER_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "a",
        PropKind::Str,
        "denominator of the transfer function: comma-separated coefficients, the first non-zero",
    )
    .with_default(DEFAULT_COEFFICIENTS_TEXT),
    PropertySpec::new(
        "b",
        PropKind::Str,
        "numerator of the transfer function: comma-separated coefficients",
    )
    .with_default(DEFAULT_COEFFICIENTS_TEXT),
];

impl PadTemplates for AudioIirFilter {
    fn pad_templates() -> Vec<PadTemplate> {
        audiofx::default_pad_templates()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mono() -> Caps {
        Caps::Audio {
            format: AudioFormat::PcmF32Le,
            channels: 1,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }
    }

    #[test]
    fn the_default_coefficients_pass_the_signal_through() {
        let mut element = AudioIirFilter::new();
        element.configure(&mono()).unwrap();
        let mut samples = [0.25f32, -0.5, 0.75];
        element.filter(&mut samples);
        assert_eq!(samples, [0.25, -0.5, 0.75]);
    }

    #[test]
    fn a_one_pole_follows_its_recurrence() {
        let mut element = AudioIirFilter::new().with_b("0.5").with_a("1,-0.5");
        element.configure(&mono()).unwrap();
        let mut samples = [1.0f32, 0.0, 0.0];
        element.filter(&mut samples);
        // y[n] = 0.5 * x[n] + 0.5 * y[n-1], so an impulse decays by half.
        assert!((samples[0] - 0.5).abs() < 1e-6);
        assert!((samples[1] - 0.25).abs() < 1e-6);
        assert!((samples[2] - 0.125).abs() < 1e-6);
    }

    #[test]
    fn each_channel_keeps_its_own_history() {
        let stereo = Caps::Audio {
            format: AudioFormat::PcmF32Le,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        let mut element = AudioIirFilter::new().with_b("0.5").with_a("1,-0.5");
        element.configure(&stereo).unwrap();
        // an impulse on the left only: the right channel stays silent.
        let mut samples = [1.0f32, 0.0, 0.0, 0.0];
        element.filter(&mut samples);
        assert!((samples[0] - 0.5).abs() < 1e-6);
        assert_eq!(samples[1], 0.0);
        assert!((samples[2] - 0.25).abs() < 1e-6);
        assert_eq!(samples[3], 0.0);
    }

    #[test]
    fn a_zero_leading_denominator_is_rejected() {
        let mut element = AudioIirFilter::new();
        assert_eq!(
            element
                .set_property("a", PropValue::Str("0,1".into()))
                .unwrap_err(),
            PropError::Value
        );
        assert_eq!(
            element
                .set_property("b", PropValue::Str("".into()))
                .unwrap_err(),
            PropError::Value
        );
    }

    #[test]
    fn coefficients_round_trip_through_the_string_form() {
        let mut element = AudioIirFilter::new();
        element
            .set_property("a", PropValue::Str("1,-0.5".into()))
            .unwrap();
        assert_eq!(
            element.get_property("a"),
            Some(PropValue::Str("1,-0.5".into()))
        );
    }
}
