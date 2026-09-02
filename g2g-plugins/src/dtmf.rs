//! DTMF source and detector (`dtmfsrc` / `dtmfdetect`), ITU-T Q.23 dual tones.
//!
//! `dtmfsrc` generates 8 kHz mono S16LE (telephony) of the digit in `number`
//! (0..=16, 16 is silence) at `volume` (0..=36, dBm0 with the sign dropped),
//! packetized at `interval` ms. `min-pulse-duration` and
//! `min-inter-digit-interval` size the tone and the gap. `dtmfdetect` is an
//! 8 kHz mono passthrough that Goertzel-detects those tones. CPU-only `no_std`.

use core::fmt::Write as _;
use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    g2g_info, AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::mathf;

/// Telephony sample rate.
const DTMF_RATE: u32 = 8_000;
const DTMF_CHANNELS: u8 = 1;
const DEFAULT_INTERVAL_MS: u64 = 50;
const DEFAULT_PULSE_MS: u64 = 250;
const DEFAULT_GAP_MS: u64 = 100;
const DEFAULT_VOLUME: u32 = 16;
const MAX_NUMBER: u64 = 16;
const MAX_VOLUME: u64 = 36;

/// Low-group frequencies (rows), then high-group (columns), ITU-T Q.23.
const ROW_HZ: [f32; 4] = [697.0, 770.0, 852.0, 941.0];
const COL_HZ: [f32; 4] = [1209.0, 1336.0, 1477.0, 1633.0];

/// Digit at `(row, col)`: 0-9, `*`=10, `#`=11, A-D=12-15.
fn digit_at(row: usize, col: usize) -> u8 {
    const GRID: [[u8; 4]; 4] = [[1, 2, 3, 12], [4, 5, 6, 13], [7, 8, 9, 14], [10, 0, 11, 15]];
    GRID[row][col]
}

fn freqs_for(number: u8) -> Option<(f32, f32)> {
    for (row, &low) in ROW_HZ.iter().enumerate() {
        for (col, &high) in COL_HZ.iter().enumerate() {
            if digit_at(row, col) == number {
                return Some((low, high));
            }
        }
    }
    None
}

/// Linear amplitude for a dBm0 volume (0..=36). Two sines sum, so each is
/// half of `10^(-volume/20)` of full scale.
fn tone_amp(volume: u32) -> f32 {
    let db = volume.min(MAX_VOLUME as u32) as f64;
    let linear = mathf::powf(10.0, -db / 20.0);
    (linear * 0.5 * i16::MAX as f64) as f32
}

fn dual_tone(n: u64, low: f32, high: f32, amp: f32) -> i16 {
    // wrap in f64: the f32 cast of a raw turn count loses the phase after seconds
    let turns = |hz: f32| ((n as f64 * hz as f64 / DTMF_RATE as f64) % 1.0) as f32;
    let s = mathf::sin_turns(turns(low)) + mathf::sin_turns(turns(high));
    (s * amp) as i16
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::dtmf::DtmfSrc;
///
/// let src = DtmfSrc::new().with_number(5);
/// ```
#[derive(Debug)]
pub struct DtmfSrc {
    number: u8,
    volume: u32,
    interval_ms: u64,
    pulse_ms: u64,
    gap_ms: u64,
    target_buffers: u64,
    configured: bool,
}

impl Default for DtmfSrc {
    fn default() -> Self {
        Self::new()
    }
}

impl DtmfSrc {
    pub fn new() -> Self {
        Self {
            number: 0,
            volume: DEFAULT_VOLUME,
            interval_ms: DEFAULT_INTERVAL_MS,
            pulse_ms: DEFAULT_PULSE_MS,
            gap_ms: DEFAULT_GAP_MS,
            target_buffers: u64::MAX,
            configured: false,
        }
    }

    pub fn with_number(mut self, number: u8) -> Self {
        self.number = number.min(MAX_NUMBER as u8);
        self
    }

    fn caps(&self) -> Caps {
        Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: DTMF_CHANNELS,
            sample_rate: DTMF_RATE,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }
    }
}

impl SourceLoop for DtmfSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.caps()))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(self.caps()))))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let samples_per = ((DTMF_RATE as u64) * self.interval_ms / 1000).max(1);
            let duration_ns = samples_per * 1_000_000_000 / DTMF_RATE as u64;
            let pulse_samples = (DTMF_RATE as u64) * self.pulse_ms / 1000;
            let gap_samples = (DTMF_RATE as u64) * self.gap_ms / 1000;
            let cycle = pulse_samples.saturating_add(gap_samples).max(1);
            let amp = tone_amp(self.volume);
            let tones = freqs_for(self.number);
            for seq in 0..self.target_buffers {
                let mut bytes = Vec::with_capacity(samples_per as usize * 2);
                let base = seq * samples_per;
                for s in 0..samples_per {
                    let n = base + s;
                    let v = if n % cycle < pulse_samples {
                        match tones {
                            Some((lo, hi)) => dual_tone(n, lo, hi, amp),
                            None => 0,
                        }
                    } else {
                        0
                    };
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                let pts = seq * duration_ns;
                #[cfg(feature = "std")]
                let arrival_ns = g2g_core::metrics::monotonic_ns();
                #[cfg(not(feature = "std"))]
                let arrival_ns: u64 = 0;
                out.push(PipelinePacket::DataFrame(Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
                    timing: FrameTiming {
                        pts_ns: pts,
                        dts_ns: pts,
                        duration_ns,
                        capture_ns: pts,
                        arrival_ns,
                        keyframe: false,
                    },
                    sequence: seq,
                    meta: Default::default(),
                }))
                .await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.target_buffers)
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        DTMFSRC_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "DTMF tone generator",
            "Source/Audio",
            "Generates DTMF tones",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "interval" => {
                let v = value.as_uint().ok_or(PropError::Type)?;
                if v == 0 {
                    return Err(PropError::Value);
                }
                self.interval_ms = v;
            }
            "min-pulse-duration" => {
                self.pulse_ms = value.as_uint().ok_or(PropError::Type)?;
            }
            "min-inter-digit-interval" => {
                self.gap_ms = value.as_uint().ok_or(PropError::Type)?;
            }
            "number" => {
                let n = value.as_uint().ok_or(PropError::Type)?;
                if n > MAX_NUMBER {
                    return Err(PropError::Value);
                }
                self.number = n as u8;
            }
            "volume" => {
                let v = value.as_uint().ok_or(PropError::Type)?;
                if v > MAX_VOLUME {
                    return Err(PropError::Value);
                }
                self.volume = v as u32;
            }
            "num-buffers" => crate::numbuffers::set_num_buffers(&mut self.target_buffers, &value)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "interval" => Some(PropValue::Uint(self.interval_ms)),
            "min-pulse-duration" => Some(PropValue::Uint(self.pulse_ms)),
            "min-inter-digit-interval" => Some(PropValue::Uint(self.gap_ms)),
            "number" => Some(PropValue::Uint(self.number as u64)),
            "volume" => Some(PropValue::Uint(self.volume as u64)),
            "num-buffers" => Some(crate::numbuffers::get_num_buffers(self.target_buffers)),
            _ => None,
        }
    }
}

static DTMFSRC_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "interval",
        PropKind::Uint,
        "Interval in ms between two tone packets",
    )
    .with_default("50"),
    PropertySpec::new(
        "min-pulse-duration",
        PropKind::Uint,
        "The minimum pulse duration, in milliseconds",
    )
    .with_default("250"),
    PropertySpec::new(
        "min-inter-digit-interval",
        PropKind::Uint,
        "The minimum inter digit arrival, in milliseconds",
    )
    .with_default("100"),
    PropertySpec::new(
        "number",
        PropKind::Uint,
        "DTMF event number (0-9, *=10, #=11, A-D=12-15)",
    )
    .with_range("0", "16")
    .with_default("0"),
    PropertySpec::new(
        "volume",
        PropKind::Uint,
        "Tone power in dBm0 after dropping the sign (0..36)",
    )
    .with_range("0", "36")
    .with_default("16"),
    PropertySpec::new(
        "num-buffers",
        PropKind::Int,
        "buffers to emit then EOS (-1 = forever)",
    )
    .with_default("-1"),
];

impl PadTemplates for DtmfSrc {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::source(CapsSet::one(Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: DTMF_CHANNELS,
            sample_rate: DTMF_RATE,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }))])
    }
}

/// Goertzel power at `freq` over `samples` (i16 LE bytes).
fn goertzel(samples: &[i16], freq: f32, rate: f32) -> f32 {
    let w = freq / rate;
    let coeff = 2.0 * mathf::cos_turns(w);
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    for &x in samples {
        let s0 = x as f32 + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    s1 * s1 + s2 * s2 - coeff * s1 * s2
}

fn detect_digit(samples: &[i16]) -> Option<u8> {
    if samples.len() < 80 {
        return None;
    }
    let rate = DTMF_RATE as f32;
    let mut row_p = [0.0f32; 4];
    let mut col_p = [0.0f32; 4];
    for i in 0..4 {
        row_p[i] = goertzel(samples, ROW_HZ[i], rate);
        col_p[i] = goertzel(samples, COL_HZ[i], rate);
    }
    let (row, row_max) = argmax(&row_p);
    let (col, col_max) = argmax(&col_p);
    let row_sum: f32 = row_p.iter().sum();
    let col_sum: f32 = col_p.iter().sum();
    // Peak must dominate its group and both groups must be present.
    if row_max < 1.0e8 || col_max < 1.0e8 {
        return None;
    }
    if row_max * 2.0 < row_sum || col_max * 2.0 < col_sum {
        return None;
    }
    Some(digit_at(row, col))
}

fn argmax(v: &[f32; 4]) -> (usize, f32) {
    let mut i = 0;
    let mut m = v[0];
    for (k, &x) in v.iter().enumerate().skip(1) {
        if x > m {
            m = x;
            i = k;
        }
    }
    (i, m)
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::dtmf::DtmfDetect;
///
/// let det = DtmfDetect::new();
/// ```
#[derive(Debug)]
pub struct DtmfDetect {
    last: Option<u8>,
    configured: bool,
    log_name: LogName,
}

impl Default for DtmfDetect {
    fn default() -> Self {
        Self::new()
    }
}

impl DtmfDetect {
    pub fn new() -> Self {
        Self {
            last: None,
            configured: false,
            log_name: LogName::new(),
        }
    }

    /// Last detected digit, or `None` while silence / no tone.
    pub fn last_digit(&self) -> Option<u8> {
        self.last
    }

    fn accept(caps: &Caps) -> Result<(), G2gError> {
        match caps {
            Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels: 1,
                sample_rate: DTMF_RATE,
                ..
            } => Ok(()),
            _ => Err(G2gError::CapsMismatch),
        }
    }
}

impl AsyncElement for DtmfDetect {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Self::accept(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels: 1,
                sample_rate: DTMF_RATE,
                ..
            } => CapsSet::one(input.clone()),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Self::accept(absolute_caps)?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "DTMF detector",
            "Filter/Analyzer/Audio",
            "Detects DTMF tones",
            "g2g",
        )
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    if let Some(slice) = frame.domain.as_system_slice() {
                        let mut samples = Vec::with_capacity(slice.len() / 2);
                        for c in slice.as_chunks::<2>().0 {
                            samples.push(i16::from_le_bytes(*c));
                        }
                        let digit = detect_digit(&samples);
                        if digit != self.last {
                            if let Some(n) = digit {
                                let mut msg = String::new();
                                let _ = write!(msg, "dtmf-event number={n}");
                                g2g_info!(self, "{}", msg);
                            }
                            self.last = digit;
                        }
                    }
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl LogSource for DtmfDetect {
    fn log_category(&self) -> &'static str {
        short_type_name::<Self>()
    }
    fn log_instance(&self) -> Option<&str> {
        self.log_name.instance()
    }
    fn log_category_override(&self) -> Option<&str> {
        self.log_name.category()
    }
}

impl PadTemplates for DtmfDetect {
    fn pad_templates() -> Vec<PadTemplate> {
        let pcm = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: DTMF_CHANNELS,
            sample_rate: DTMF_RATE,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        Vec::from([
            PadTemplate::sink(CapsSet::one(pcm.clone())),
            PadTemplate::source(CapsSet::one(pcm)),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn freqs_cover_star_hash_and_digits() {
        assert_eq!(freqs_for(0), Some((941.0, 1336.0)));
        assert_eq!(freqs_for(1), Some((697.0, 1209.0)));
        assert_eq!(freqs_for(10), Some((941.0, 1209.0)));
        assert_eq!(freqs_for(11), Some((941.0, 1477.0)));
        assert_eq!(freqs_for(16), None);
    }

    #[test]
    fn goertzel_hears_digit_five() {
        let (lo, hi) = freqs_for(5).unwrap();
        let amp = tone_amp(8);
        let samples: Vec<i16> = (0..400).map(|n| dual_tone(n, lo, hi, amp)).collect();
        assert_eq!(detect_digit(&samples), Some(5));
    }

    #[test]
    fn silence_is_not_a_digit() {
        let samples = vec![0i16; 400];
        assert_eq!(detect_digit(&samples), None);
    }
}
