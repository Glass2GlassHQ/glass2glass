//! Dynamic range control (`audiodynamic`). Compresses samples above the
//! threshold or expands samples below it, with a hard or a soft knee.
//! Preserves format, channel count, and sample rate. CPU-only `no_std`.
//!
//! The four transfer functions are GStreamer's `audiodynamic` float paths:
//! outside the threshold the compressor is a straight line of slope `ratio`
//! (hard knee) or the quadratic through `f(t) = t`, `f'(t) = 1`, `f'(1) = ratio`
//! (soft knee); the expander lifts quiet samples along a line of slope `ratio`
//! through the threshold (hard knee) or the quadratic through `f(t) = t`,
//! `f'(t) = 1`, `f(z) = 0`, `f'(z) = ratio` (soft knee). The S16LE stream takes
//! the same normalized curve rather than the reference's integer path, whose
//! thresholds sit half an LSB apart either side of zero.

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

const DEFAULT_THRESHOLD: f64 = 0.0;
const THRESHOLD_MIN: f64 = 0.0;
const THRESHOLD_MAX: f64 = 1.0;
const DEFAULT_RATIO: f64 = 1.0;
const RATIO_MIN: f64 = 0.0;
const RATIO_MAX: f64 = f32::MAX as f64;
/// A threshold at full scale would divide by zero building the soft knee, so the
/// reference nudges it up, as here.
const SOFT_KNEE_THRESHOLD_CEILING: f64 = 1.000_01;

/// Whether the filter acts on loud samples or quiet ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DynamicMode {
    Compressor,
    Expander,
}

impl DynamicMode {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "compressor" => Some(Self::Compressor),
            "expander" => Some(Self::Expander),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Compressor => "compressor",
            Self::Expander => "expander",
        }
    }
}

/// Whether the ratio is applied as a kink in the transfer function or smoothed
/// into it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DynamicCharacteristics {
    HardKnee,
    SoftKnee,
}

impl DynamicCharacteristics {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "hard-knee" => Some(Self::HardKnee),
            "soft-knee" => Some(Self::SoftKnee),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::HardKnee => "hard-knee",
            Self::SoftKnee => "soft-knee",
        }
    }
}

/// The transfer function for one setting of the four properties: built once per
/// buffer, applied per sample.
#[derive(Debug, Clone, Copy)]
struct DynamicCurve {
    mode: DynamicMode,
    characteristics: DynamicCharacteristics,
    threshold: f64,
    ratio: f64,
    /// Where the expander's line crosses zero, above which quiet samples are
    /// squelched.
    zero: f64,
    /// Quadratic coefficients of the soft knee, positive then negative side.
    positive: [f64; 3],
    negative: [f64; 3],
    /// True when the settings leave the signal untouched.
    passthrough: bool,
}

impl DynamicCurve {
    fn new(
        mode: DynamicMode,
        characteristics: DynamicCharacteristics,
        threshold: f64,
        ratio: f64,
    ) -> Self {
        let passthrough = match mode {
            DynamicMode::Compressor => ratio == 1.0,
            DynamicMode::Expander => threshold == 0.0 || ratio == 1.0,
        };
        let mut curve = Self {
            mode,
            characteristics,
            threshold,
            ratio,
            zero: 0.0,
            positive: [0.0; 3],
            negative: [0.0; 3],
            passthrough,
        };
        if passthrough {
            return curve;
        }
        match mode {
            DynamicMode::Compressor => {
                let t = if threshold == 1.0 {
                    SOFT_KNEE_THRESHOLD_CEILING
                } else {
                    threshold
                };
                curve.threshold = t;
                if characteristics == DynamicCharacteristics::SoftKnee {
                    let a_p = (1.0 - ratio) / (2.0 * (t - 1.0));
                    let b_p = (ratio * t - 1.0) / (t - 1.0);
                    curve.positive = [a_p, b_p, t * (1.0 - b_p - a_p * t)];
                    let a_n = (1.0 - ratio) / (2.0 * (-t + 1.0));
                    let b_n = (-ratio * t + 1.0) / (-t + 1.0);
                    curve.negative = [a_n, b_n, -t * (1.0 - b_n + a_n * t)];
                }
            }
            DynamicMode::Expander => {
                curve.zero = match characteristics {
                    DynamicCharacteristics::HardKnee => {
                        if ratio != 0.0 {
                            threshold - threshold / ratio
                        } else {
                            0.0
                        }
                    }
                    DynamicCharacteristics::SoftKnee => (threshold * (ratio - 1.0)) / (1.0 + ratio),
                }
                .max(0.0);
                if characteristics == DynamicCharacteristics::SoftKnee {
                    let r2 = ratio * ratio;
                    let a_p = (1.0 - r2) / (4.0 * threshold);
                    let b = (1.0 + r2) / 2.0;
                    curve.positive = [a_p, b, threshold * (1.0 - b - a_p * threshold)];
                    let a_n = (1.0 - r2) / (-4.0 * threshold);
                    curve.negative = [a_n, b, -threshold * (1.0 - b + a_n * threshold)];
                }
            }
        }
        curve
    }

    fn apply(&self, sample: f32) -> f32 {
        if self.passthrough {
            return sample;
        }
        let value = sample as f64;
        let t = self.threshold;
        let r = self.ratio;
        let out = match (self.mode, self.characteristics) {
            (DynamicMode::Compressor, DynamicCharacteristics::HardKnee) => {
                if value > t {
                    t + (value - t) * r
                } else if value < -t {
                    -t + (value + t) * r
                } else {
                    value
                }
            }
            (DynamicMode::Compressor, DynamicCharacteristics::SoftKnee) => {
                let [a_p, b_p, c_p] = self.positive;
                let [a_n, b_n, c_n] = self.negative;
                if value > 1.0 {
                    1.0 + (value - 1.0) * r
                } else if value > t {
                    a_p * value * value + b_p * value + c_p
                } else if value < -1.0 {
                    -1.0 + (value + 1.0) * r
                } else if value < -t {
                    a_n * value * value + b_n * value + c_n
                } else {
                    value
                }
            }
            (DynamicMode::Expander, DynamicCharacteristics::HardKnee) => {
                if value < t && value > self.zero {
                    r * value + t * (1.0 - r)
                } else if (value <= self.zero && value > 0.0)
                    || (value >= -self.zero && value < 0.0)
                {
                    0.0
                } else if value > -t && value < -self.zero {
                    r * value - t * (1.0 - r)
                } else {
                    value
                }
            }
            (DynamicMode::Expander, DynamicCharacteristics::SoftKnee) => {
                let [a_p, b_p, c_p] = self.positive;
                let [a_n, b_n, c_n] = self.negative;
                if value < t && value > self.zero {
                    a_p * value * value + b_p * value + c_p
                } else if (value <= self.zero && value > 0.0)
                    || (value >= -self.zero && value < 0.0)
                {
                    0.0
                } else if value > -t && value < -self.zero {
                    a_n * value * value + b_n * value + c_n
                } else {
                    value
                }
            }
        };
        out as f32
    }
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::audiodynamic::{AudioDynamic, DynamicMode};
///
/// let compressor = AudioDynamic::new()
///     .with_mode(DynamicMode::Compressor)
///     .with_threshold(0.5)
///     .with_ratio(0.25);
/// ```
#[derive(Debug)]
pub struct AudioDynamic {
    mode: DynamicMode,
    characteristics: DynamicCharacteristics,
    threshold: f64,
    ratio: f64,
    format: AudioFormat,
    caps: Option<Caps>,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl Default for AudioDynamic {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioDynamic {
    /// A hard-knee compressor at unity ratio: a pass-through until tuned.
    pub fn new() -> Self {
        Self {
            mode: DynamicMode::Compressor,
            characteristics: DynamicCharacteristics::HardKnee,
            threshold: DEFAULT_THRESHOLD,
            ratio: DEFAULT_RATIO,
            format: AudioFormat::PcmS16Le,
            caps: None,
            last_caps: None,
            emitted: 0,
        }
    }

    pub fn with_mode(mut self, mode: DynamicMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn with_characteristics(mut self, characteristics: DynamicCharacteristics) -> Self {
        self.characteristics = characteristics;
        self
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold.clamp(THRESHOLD_MIN, THRESHOLD_MAX);
        self
    }

    pub fn with_ratio(mut self, ratio: f64) -> Self {
        self.ratio = ratio.max(RATIO_MIN);
        self
    }

    fn curve(&self) -> DynamicCurve {
        DynamicCurve::new(self.mode, self.characteristics, self.threshold, self.ratio)
    }

    fn configure(&mut self, caps: &Caps) -> Result<(), G2gError> {
        let (format, _, _) = audiofx::accept_audio(caps, None)?;
        self.format = format;
        self.caps = Some(caps.clone());
        Ok(())
    }
}

impl AsyncElement for AudioDynamic {
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
                    let curve = self.curve();
                    for sample in samples.iter_mut() {
                        *sample = curve.apply(*sample);
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
        AUDIODYNAMIC_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Dynamic range controller",
            "Filter/Effect/Audio",
            "Compresses or expands the dynamic range of an audio stream",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "mode" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.mode = DynamicMode::from_str(s).ok_or(PropError::Value)?;
            }
            "characteristics" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.characteristics =
                    DynamicCharacteristics::from_str(s).ok_or(PropError::Value)?;
            }
            "threshold" => {
                self.threshold = audiofx::double_in_range(value, THRESHOLD_MIN, THRESHOLD_MAX)?
            }
            "ratio" => self.ratio = audiofx::double_in_range(value, RATIO_MIN, RATIO_MAX)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "mode" => Some(PropValue::Str(self.mode.as_str().into())),
            "characteristics" => Some(PropValue::Str(self.characteristics.as_str().into())),
            "threshold" => Some(PropValue::Double(self.threshold)),
            "ratio" => Some(PropValue::Double(self.ratio)),
            _ => None,
        }
    }
}

static AUDIODYNAMIC_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "mode",
        PropKind::Str,
        "act on loud samples (compressor) or quiet ones (expander)",
    )
    .with_enum_values("compressor | expander")
    .with_default("compressor"),
    PropertySpec::new(
        "characteristics",
        PropKind::Str,
        "apply the ratio as a kink (hard-knee) or smoothly (soft-knee)",
    )
    .with_enum_values("hard-knee | soft-knee")
    .with_default("hard-knee"),
    PropertySpec::new(
        "threshold",
        PropKind::Double,
        "level the filter starts acting at, as a fraction of full scale",
    )
    .with_range("0", "1")
    .with_default("0"),
    PropertySpec::new(
        "ratio",
        PropKind::Double,
        "ratio applied past the threshold",
    )
    .with_default("1"),
];

impl PadTemplates for AudioDynamic {
    fn pad_templates() -> Vec<PadTemplate> {
        audiofx::default_pad_templates()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn curve(
        mode: DynamicMode,
        characteristics: DynamicCharacteristics,
        threshold: f64,
        ratio: f64,
    ) -> DynamicCurve {
        DynamicCurve::new(mode, characteristics, threshold, ratio)
    }

    #[test]
    fn hard_knee_compressor_bends_at_the_threshold() {
        let c = curve(
            DynamicMode::Compressor,
            DynamicCharacteristics::HardKnee,
            0.5,
            0.25,
        );
        // below the threshold nothing moves.
        assert_eq!(c.apply(0.25), 0.25);
        // full scale lands on t + (1 - t) * ratio.
        assert!((c.apply(1.0) - 0.625).abs() < 1e-6);
        assert!((c.apply(-1.0) + 0.625).abs() < 1e-6);
    }

    #[test]
    fn soft_knee_compressor_is_continuous_at_the_threshold() {
        let c = curve(
            DynamicMode::Compressor,
            DynamicCharacteristics::SoftKnee,
            0.5,
            0.25,
        );
        // f(t) == t and the slope at full scale is the ratio.
        assert!((c.apply(0.5) - 0.5).abs() < 1e-6);
        let slope = (c.apply(1.0) - c.apply(0.999)) as f64 / 0.001;
        assert!(
            (slope - 0.25).abs() < 1e-2,
            "slope at full scale is {slope}"
        );
    }

    #[test]
    fn hard_knee_expander_squelches_below_the_zero_crossing() {
        let c = curve(
            DynamicMode::Expander,
            DynamicCharacteristics::HardKnee,
            0.5,
            2.0,
        );
        // zero crossing is t - t/r = 0.25.
        assert_eq!(c.apply(0.2), 0.0);
        // above it the line has slope r through the threshold.
        assert!((c.apply(0.4) - (2.0 * 0.4 - 0.5)) < 1e-6);
        // past the threshold the signal passes.
        assert_eq!(c.apply(0.75), 0.75);
    }

    #[test]
    fn soft_knee_expander_reaches_the_threshold_untouched() {
        let c = curve(
            DynamicMode::Expander,
            DynamicCharacteristics::SoftKnee,
            0.5,
            2.0,
        );
        assert!((c.apply(0.5) - 0.5).abs() < 1e-6);
        assert_eq!(c.apply(0.9), 0.9);
    }

    #[test]
    fn unity_ratio_is_a_pass_through() {
        let c = curve(
            DynamicMode::Compressor,
            DynamicCharacteristics::SoftKnee,
            0.1,
            1.0,
        );
        assert_eq!(c.apply(0.9), 0.9);
    }

    #[test]
    fn properties_reject_out_of_range_values() {
        let mut e = AudioDynamic::new();
        assert_eq!(
            e.set_property("threshold", PropValue::Double(2.0))
                .unwrap_err(),
            PropError::Value
        );
        assert_eq!(
            e.set_property("mode", PropValue::Str("limiter".into()))
                .unwrap_err(),
            PropError::Value
        );
    }
}
