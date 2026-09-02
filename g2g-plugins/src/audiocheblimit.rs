//! Chebyshev low-pass / high-pass (`audiocheblimit`). An IIR cascade of
//! second-order sections, `poles` poles with `ripple` dB of ripple, far steeper
//! per tap than the windowed-sinc filters but without their linear phase.
//! Preserves format, channel count, and sample rate. CPU-only `no_std`.
//!
//! Matches GStreamer's `audiocheblimit`: the prototype poles sit on an ellipse
//! for `type=1` (ripple in the pass band) or on its inverse with unit-circle
//! zeros for `type=2` (ripple in the stop band), go through the bilinear
//! transform, then through the all-pass substitution that moves the cutoff onto
//! `cutoff`. The reference multiplies the sections into one high-order
//! difference equation; here they stay separate biquads, which is the same
//! transfer function with less coefficient spread. Filter state is per channel
//! and is cleared on `Flush` and on any property change.

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

use crate::audiofx::{self, IirCascade, LimitMode};

const DEFAULT_CUTOFF: f64 = 0.0;
const CUTOFF_MIN: f64 = 0.0;
const CUTOFF_MAX: f64 = 100_000.0;
const DEFAULT_POLES: i64 = 4;
const POLES_MIN: i64 = 2;
const POLES_MAX: i64 = 32;
const DEFAULT_RIPPLE_DB: f64 = 0.25;
const RIPPLE_MIN_DB: f64 = 0.0;
const RIPPLE_MAX_DB: f64 = 200.0;
/// Type 1 puts the ripple in the pass band, type 2 in the stop band.
const DEFAULT_TYPE: i64 = 1;
const TYPE_MIN: i64 = 1;
const TYPE_MAX: i64 = 2;

/// Poles come in conjugate pairs, one biquad each.
fn to_even(poles: i64) -> usize {
    let poles = poles.clamp(POLES_MIN, POLES_MAX);
    (if poles % 2 == 1 { poles + 1 } else { poles }) as usize
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::audiocheblimit::AudioChebLimit;
///
/// let low_pass = AudioChebLimit::new().with_cutoff(1000.0).with_poles(4);
/// ```
#[derive(Debug)]
pub struct AudioChebLimit {
    mode: LimitMode,
    cutoff: f64,
    poles: usize,
    kind: u8,
    ripple_db: f64,
    cascade: IirCascade,
    format: AudioFormat,
    channels: usize,
    sample_rate: u32,
    caps: Option<Caps>,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl Default for AudioChebLimit {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioChebLimit {
    pub fn new() -> Self {
        Self {
            mode: LimitMode::LowPass,
            cutoff: DEFAULT_CUTOFF,
            poles: to_even(DEFAULT_POLES),
            kind: DEFAULT_TYPE as u8,
            ripple_db: DEFAULT_RIPPLE_DB,
            cascade: IirCascade::default(),
            format: AudioFormat::PcmS16Le,
            channels: 0,
            sample_rate: 0,
            caps: None,
            last_caps: None,
            emitted: 0,
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

    pub fn with_poles(mut self, poles: i64) -> Self {
        self.poles = to_even(poles);
        self.rebuild();
        self
    }

    pub fn with_ripple(mut self, ripple_db: f64) -> Self {
        self.ripple_db = ripple_db.clamp(RIPPLE_MIN_DB, RIPPLE_MAX_DB);
        self.rebuild();
        self
    }

    /// 1 (ripple in the pass band) or 2 (ripple in the stop band).
    pub fn with_type(mut self, kind: u8) -> Self {
        self.kind = kind.clamp(TYPE_MIN as u8, TYPE_MAX as u8);
        self.rebuild();
        self
    }

    fn rebuild(&mut self) {
        if self.sample_rate == 0 {
            return;
        }
        self.cascade = audiofx::cheb_limit_filter(
            self.mode,
            self.cutoff,
            self.sample_rate,
            audiofx::ChebSettings {
                kind: self.kind,
                poles: self.poles,
                ripple_db: self.ripple_db,
                channels: self.channels,
            },
        );
    }

    fn configure(&mut self, caps: &Caps) -> Result<(), G2gError> {
        let (format, channels, rate) = audiofx::accept_audio(caps, None)?;
        self.format = format;
        self.channels = channels;
        self.sample_rate = rate;
        self.caps = Some(caps.clone());
        self.rebuild();
        Ok(())
    }

    /// The cascade's magnitude response at `frequency`, for tests and tuning.
    pub fn response_at(&self, frequency: f64) -> f64 {
        let w = core::f64::consts::TAU * frequency / self.sample_rate.max(1) as f64;
        self.cascade
            .magnitude(crate::mathf::cos(w), crate::mathf::sin(w))
    }
}

impl AsyncElement for AudioChebLimit {
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
                    self.cascade.run(&mut samples);
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
                    self.cascade.reset();
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
        AUDIOCHEBLIMIT_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Low-pass and high-pass Chebyshev filter",
            "Filter/Effect/Audio",
            "Chebyshev low-pass and high-pass filter",
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
            "poles" => self.poles = to_even(audiofx::int_in_range(value, POLES_MIN, POLES_MAX)?),
            "ripple" => {
                self.ripple_db = audiofx::double_in_range(value, RIPPLE_MIN_DB, RIPPLE_MAX_DB)?
            }
            "type" => self.kind = audiofx::int_in_range(value, TYPE_MIN, TYPE_MAX)? as u8,
            _ => return Err(PropError::Unknown),
        }
        self.rebuild();
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "mode" => Some(PropValue::Str(self.mode.as_str().into())),
            "cutoff" => Some(PropValue::Double(self.cutoff)),
            "poles" => Some(PropValue::Int(self.poles as i64)),
            "ripple" => Some(PropValue::Double(self.ripple_db)),
            "type" => Some(PropValue::Int(self.kind as i64)),
            _ => None,
        }
    }
}

static AUDIOCHEBLIMIT_PROPS: &[PropertySpec] = &[
    PropertySpec::new("mode", PropKind::Str, "low-pass or high-pass")
        .with_enum_values(audiofx::LIMIT_MODE_VALUES)
        .with_default("low-pass"),
    PropertySpec::new("cutoff", PropKind::Double, "cut-off frequency in Hz")
        .with_range("0", "100000")
        .with_default("0"),
    PropertySpec::new(
        "poles",
        PropKind::Int,
        "number of poles, rounded up to the next even number",
    )
    .with_range("2", "32")
    .with_default("4"),
    PropertySpec::new("ripple", PropKind::Double, "amount of ripple in dB")
        .with_range("0", "200")
        .with_default("0.25"),
    PropertySpec::new(
        "type",
        PropKind::Int,
        "1 ripples the pass band, 2 ripples the stop band",
    )
    .with_range("1", "2")
    .with_default("1"),
];

impl PadTemplates for AudioChebLimit {
    fn pad_templates() -> Vec<PadTemplate> {
        audiofx::default_pad_templates()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn configured(mode: LimitMode, cutoff: f64) -> AudioChebLimit {
        let mut e = AudioChebLimit::new()
            .with_mode(mode)
            .with_cutoff(cutoff)
            .with_poles(4)
            .with_ripple(0.5);
        e.configure(&Caps::Audio {
            format: AudioFormat::PcmF32Le,
            channels: 1,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        })
        .unwrap();
        e
    }

    #[test]
    fn poles_round_up_to_even() {
        let mut e = AudioChebLimit::new();
        e.set_property("poles", PropValue::Int(5)).unwrap();
        assert_eq!(e.get_property("poles"), Some(PropValue::Int(6)));
    }

    /// Gain GStreamer's own `audiocheblimit` produces for these settings, read
    /// off `audiotestsrc wave=sine freq=5000 ! audiocheblimit ... ! level` as
    /// the rms above the unfiltered rms (-4.9485 dB).
    const REFERENCE_LOW_PASS_AT_5_KHZ: f64 = 5.469_4e-4;
    /// The reference reads to five figures, so a tenth of a percent of the gain
    /// is as close as the two can be compared.
    const REFERENCE_TOLERANCE: f64 = 1e-3;

    #[test]
    fn lowpass_matches_the_reference_element() {
        let e = configured(LimitMode::LowPass, 1000.0);
        assert!((e.response_at(0.0) - 1.0).abs() < 1e-6);
        let stop = e.response_at(5000.0);
        assert!(
            (stop - REFERENCE_LOW_PASS_AT_5_KHZ).abs() / REFERENCE_LOW_PASS_AT_5_KHZ
                < REFERENCE_TOLERANCE,
            "low-pass at 5 kHz is {stop}"
        );
    }

    #[test]
    fn highpass_is_the_mirror() {
        let e = configured(LimitMode::HighPass, 1000.0);
        assert!(e.response_at(0.0) < 1e-3);
        assert!((e.response_at(20_000.0) - 1.0).abs() < 0.1);
    }

    #[test]
    fn type_is_range_checked() {
        let mut e = AudioChebLimit::new();
        assert_eq!(
            e.set_property("type", PropValue::Int(3)).unwrap_err(),
            PropError::Value
        );
    }
}
