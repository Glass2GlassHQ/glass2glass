//! Sine-tone source (`tonegeneratesrc`): interleaved S16LE PCM at 44100 Hz
//! stereo, one sine at `freq` scaled by `volume`. CPU-only `no_std`.
//!
//! Properties: `freq` (Hz), `volume` (0..1), `samplesperbuffer`, plus the
//! `num-buffers` every g2g source takes.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, FrameTiming,
    G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec,
};

const DEFAULT_RATE: u32 = 44_100;
const DEFAULT_CHANNELS: u8 = 2;
const DEFAULT_FREQ: f64 = 440.0;
const DEFAULT_VOLUME: f64 = 0.8;
const DEFAULT_SAMPLES_PER_BUFFER: i64 = 1024;
const MAX_FREQ: f64 = (i32::MAX / 2) as f64;

/// # Example
///
/// ```no_run
/// use g2g_plugins::tonegeneratesrc::ToneGenerateSrc;
///
/// let src = ToneGenerateSrc::new().with_freq(1000.0);
/// ```
#[derive(Debug)]
pub struct ToneGenerateSrc {
    freq: f64,
    volume: f64,
    samples_per_buffer: u32,
    target_buffers: u64,
    configured: bool,
}

impl Default for ToneGenerateSrc {
    fn default() -> Self {
        Self::new()
    }
}

impl ToneGenerateSrc {
    pub fn new() -> Self {
        Self {
            freq: DEFAULT_FREQ,
            volume: DEFAULT_VOLUME,
            samples_per_buffer: DEFAULT_SAMPLES_PER_BUFFER as u32,
            target_buffers: u64::MAX,
            configured: false,
        }
    }

    pub fn with_freq(mut self, freq: f64) -> Self {
        self.freq = freq.clamp(0.0, MAX_FREQ);
        self
    }

    fn caps(&self) -> Caps {
        Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: DEFAULT_CHANNELS,
            sample_rate: DEFAULT_RATE,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }
    }

    fn sample(&self, n: u64) -> i16 {
        let period = DEFAULT_RATE as f64;
        if period <= 0.0 || self.freq <= 0.0 {
            return 0;
        }
        // wrap in f64: the f32 cast of a raw turn count loses the phase after seconds
        let phase = ((n as f64 * self.freq / period) % 1.0) as f32;
        let amp = (self.volume * (i16::MAX as f64)) as f32;
        (crate::mathf::sin_turns(phase) * amp) as i16
    }
}

impl SourceLoop for ToneGenerateSrc {
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
            let n = self.samples_per_buffer.max(1) as u64;
            let buffer_duration_ns = n.saturating_mul(1_000_000_000) / DEFAULT_RATE as u64;
            for seq in 0..self.target_buffers {
                let base = seq * n;
                let mut bytes = Vec::with_capacity(n as usize * DEFAULT_CHANNELS as usize * 2);
                for s in 0..n {
                    let v = self.sample(base + s);
                    for _ in 0..DEFAULT_CHANNELS {
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                }
                let pts = seq * buffer_duration_ns;
                #[cfg(feature = "std")]
                let arrival_ns = g2g_core::metrics::monotonic_ns();
                #[cfg(not(feature = "std"))]
                let arrival_ns: u64 = 0;
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
                    timing: FrameTiming {
                        pts_ns: pts,
                        dts_ns: pts,
                        duration_ns: buffer_duration_ns,
                        capture_ns: pts,
                        arrival_ns,
                        keyframe: false,
                    },
                    sequence: seq,
                    meta: Default::default(),
                };
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.target_buffers)
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        TONE_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Audio tone source",
            "Source/Audio",
            "Generates a sine tone at freq / volume",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "freq" => {
                let f = value.as_double().ok_or(PropError::Type)?;
                if !(0.0..=MAX_FREQ).contains(&f) {
                    return Err(PropError::Value);
                }
                self.freq = f;
            }
            "volume" => {
                let v = value.as_double().ok_or(PropError::Type)?;
                if !(0.0..=1.0).contains(&v) {
                    return Err(PropError::Value);
                }
                self.volume = v;
            }
            "samplesperbuffer" => {
                let n = value.as_int().ok_or(PropError::Type)?;
                if n < 1 {
                    return Err(PropError::Value);
                }
                self.samples_per_buffer = n as u32;
            }
            "num-buffers" => crate::numbuffers::set_num_buffers(&mut self.target_buffers, &value)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "freq" => Some(PropValue::Double(self.freq)),
            "volume" => Some(PropValue::Double(self.volume)),
            "samplesperbuffer" => Some(PropValue::Int(self.samples_per_buffer as i64)),
            "num-buffers" => Some(crate::numbuffers::get_num_buffers(self.target_buffers)),
            _ => None,
        }
    }
}

static TONE_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "freq",
        PropKind::Double,
        "Frequency of test signal. The sample rate needs to be at least 2 times higher.",
    )
    .with_range("0", "1073741823")
    .with_default("440"),
    PropertySpec::new("volume", PropKind::Double, "Volume of test signal")
        .with_range("0", "1")
        .with_default("0.8"),
    PropertySpec::new(
        "samplesperbuffer",
        PropKind::Int,
        "Number of samples in each outgoing buffer",
    )
    .with_range("1", "2147483647")
    .with_default("1024"),
    PropertySpec::new(
        "num-buffers",
        PropKind::Int,
        "buffers to emit then EOS (-1 = forever)",
    )
    .with_default("-1"),
];

impl PadTemplates for ToneGenerateSrc {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::source(CapsSet::one(Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: DEFAULT_CHANNELS,
            sample_rate: DEFAULT_RATE,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }))])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_starts_near_zero() {
        let src = ToneGenerateSrc::new();
        assert_eq!(src.sample(0), 0);
    }

    #[test]
    fn volume_zero_is_silence() {
        let mut src = ToneGenerateSrc::new();
        src.set_property("volume", PropValue::Double(0.0)).unwrap();
        assert_eq!(src.sample(100), 0);
    }
}
