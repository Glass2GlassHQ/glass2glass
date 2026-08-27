//! Chebyshev band-pass / band-reject (`audiochebband`). An IIR cascade whose
//! sections are fourth order, `poles` poles with `ripple` dB of ripple.
//! Preserves format, channel count, and sample rate. CPU-only `no_std`.
//!
//! Matches GStreamer's `audiochebband`: the low-pass prototype has half the
//! band filter's poles, and the second-order all-pass substitution that maps it
//! onto the band turns each pole pair into a quartic section. `type=1` ripples
//! the pass band, `type=2` the stop band. Filter state is per channel and is
//! cleared on `Flush` and on any property change.

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

use crate::audiofx::{self, BandMode, IirCascade};

const DEFAULT_FREQUENCY: f64 = 0.0;
const FREQUENCY_MIN: f64 = 0.0;
const FREQUENCY_MAX: f64 = 100_000.0;
const DEFAULT_POLES: i64 = 4;
const POLES_MIN: i64 = 4;
const POLES_MAX: i64 = 32;
const DEFAULT_RIPPLE_DB: f64 = 0.25;
const RIPPLE_MIN_DB: f64 = 0.0;
const RIPPLE_MAX_DB: f64 = 200.0;
const DEFAULT_TYPE: i64 = 1;
const TYPE_MIN: i64 = 1;
const TYPE_MAX: i64 = 2;
/// Each section is fourth order, so the pole count comes in fours.
const POLES_PER_SECTION: i64 = 4;

fn to_multiple_of_four(poles: i64) -> usize {
    let poles = poles.clamp(POLES_MIN, POLES_MAX);
    (((poles + POLES_PER_SECTION - 1) / POLES_PER_SECTION) * POLES_PER_SECTION) as usize
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::audiochebband::AudioChebBand;
///
/// let band_pass = AudioChebBand::new()
///     .with_lower_frequency(1000.0)
///     .with_upper_frequency(4000.0);
/// ```
#[derive(Debug)]
pub struct AudioChebBand {
    mode: BandMode,
    lower_frequency: f64,
    upper_frequency: f64,
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

impl Default for AudioChebBand {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioChebBand {
    pub fn new() -> Self {
        Self {
            mode: BandMode::BandPass,
            lower_frequency: DEFAULT_FREQUENCY,
            upper_frequency: DEFAULT_FREQUENCY,
            poles: to_multiple_of_four(DEFAULT_POLES),
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

    pub fn with_poles(mut self, poles: i64) -> Self {
        self.poles = to_multiple_of_four(poles);
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
        self.cascade = audiofx::cheb_band_filter(
            self.mode,
            self.lower_frequency,
            self.upper_frequency,
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

impl AsyncElement for AudioChebBand {
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
        AUDIOCHEBBAND_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Band-pass and band-reject Chebyshev filter",
            "Filter/Effect/Audio",
            "Chebyshev band-pass and band-reject filter",
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
            "poles" => {
                self.poles =
                    to_multiple_of_four(audiofx::int_in_range(value, POLES_MIN, POLES_MAX)?)
            }
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
            "lower-frequency" => Some(PropValue::Double(self.lower_frequency)),
            "upper-frequency" => Some(PropValue::Double(self.upper_frequency)),
            "poles" => Some(PropValue::Int(self.poles as i64)),
            "ripple" => Some(PropValue::Double(self.ripple_db)),
            "type" => Some(PropValue::Int(self.kind as i64)),
            _ => None,
        }
    }
}

static AUDIOCHEBBAND_PROPS: &[PropertySpec] = &[
    PropertySpec::new("mode", PropKind::Str, "band-pass or band-reject")
        .with_enum_values(audiofx::BAND_MODE_VALUES)
        .with_default("band-pass"),
    PropertySpec::new(
        "lower-frequency",
        PropKind::Double,
        "start frequency of the band in Hz",
    )
    .with_range("0", "100000")
    .with_default("0"),
    PropertySpec::new(
        "upper-frequency",
        PropKind::Double,
        "stop frequency of the band in Hz",
    )
    .with_range("0", "100000")
    .with_default("0"),
    PropertySpec::new(
        "poles",
        PropKind::Int,
        "number of poles, rounded up to the next multiple of four",
    )
    .with_range("4", "32")
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

impl PadTemplates for AudioChebBand {
    fn pad_templates() -> Vec<PadTemplate> {
        audiofx::default_pad_templates()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn configured(mode: BandMode) -> AudioChebBand {
        let mut e = AudioChebBand::new()
            .with_mode(mode)
            .with_lower_frequency(1000.0)
            .with_upper_frequency(4000.0)
            .with_poles(4)
            .with_ripple(0.5);
        e.configure(&Caps::Audio {
            format: AudioFormat::PcmF32Le,
            channels: 1,
            sample_rate: 48_000,
        })
        .unwrap();
        e
    }

    #[test]
    fn poles_round_up_to_a_multiple_of_four() {
        let mut e = AudioChebBand::new();
        e.set_property("poles", PropValue::Int(5)).unwrap();
        assert_eq!(e.get_property("poles"), Some(PropValue::Int(8)));
    }

    /// Gains GStreamer's own `audiochebband` produces for the same settings,
    /// read off `audiotestsrc wave=sine freq=F ! audiochebband ... ! level` as
    /// the rms above the unfiltered rms (-4.9485 dB).
    const REFERENCE_BAND_PASS_AT_15_KHZ: f64 = 0.027_946;
    const REFERENCE_BAND_REJECT_AT_2500_HZ: f64 = 0.132_28;
    /// The reference reads to five figures, so a tenth of a percent of the gain
    /// is as close as the two can be compared.
    const REFERENCE_TOLERANCE: f64 = 1e-3;

    #[test]
    fn band_pass_matches_the_reference_element() {
        let e = configured(BandMode::BandPass);
        // unity across the band, and the reference's own stop-band gain.
        assert!((e.response_at(2000.0) - 1.0).abs() < 0.05);
        let stop = e.response_at(15_000.0);
        assert!(
            (stop - REFERENCE_BAND_PASS_AT_15_KHZ).abs() / REFERENCE_BAND_PASS_AT_15_KHZ
                < REFERENCE_TOLERANCE,
            "band-pass at 15 kHz is {stop}"
        );
    }

    #[test]
    fn band_reject_matches_the_reference_element() {
        let e = configured(BandMode::BandReject);
        let notch = e.response_at(2500.0);
        assert!(
            (notch - REFERENCE_BAND_REJECT_AT_2500_HZ).abs() / REFERENCE_BAND_REJECT_AT_2500_HZ
                < REFERENCE_TOLERANCE,
            "band-reject at 2500 Hz is {notch}"
        );
        assert!((e.response_at(100.0) - 1.0).abs() < 0.1);
    }
}
