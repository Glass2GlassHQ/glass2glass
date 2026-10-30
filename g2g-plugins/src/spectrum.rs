//! Audio spectrum analyzer (`spectrum`). A passthrough that runs a windowed FFT
//! over the mono-mixed S16LE signal and exposes per-band magnitudes via a getter,
//! the g2g analog of GStreamer's `spectrum` (which posts the same values on the
//! bus). Magnitudes are linear (0..1, normalized to full scale). CPU-only
//! `no_std` with a hand-rolled radix-2 FFT (no math dep).
//!
//! `bands` sets the number of output frequency bands; the FFT size is the next
//! power of two >= `2 * bands`. A Hann window reduces spectral leakage.
//! `post-messages` gates the analysis (getter empty, buffer forwarded untouched).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use crate::fft::FftPlan;

use g2g_core::{
    pcm_caps_any_rate_and_channels, AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet,
    ConfigureOutcome, ElementMetadata, G2gError, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

/// # Example
///
/// ```no_run
/// use g2g_plugins::spectrum::Spectrum;
///
/// let spectrum = Spectrum::new().with_bands(64);
/// assert!(spectrum.last_magnitudes().is_empty());
/// ```
#[derive(Debug)]
pub struct Spectrum {
    bands: usize,
    post_messages: bool,
    channels: usize,
    plan: FftPlan,
    // Hann window over one FFT window, rebuilt with the plan
    window: Vec<f64>,
    re: Vec<f64>,
    im: Vec<f64>,
    // mono-mixed samples accumulated toward the next FFT window.
    acc: Vec<f64>,
    magnitudes: Vec<f64>,
    configured: bool,
}

impl Default for Spectrum {
    fn default() -> Self {
        Self::new()
    }
}

const FULL_SCALE: f64 = 32768.0;

impl Spectrum {
    pub fn new() -> Self {
        let mut s = Self {
            bands: 128,
            post_messages: true,
            channels: 0,
            plan: FftPlan::new(1),
            window: Vec::new(),
            re: Vec::new(),
            im: Vec::new(),
            acc: Vec::new(),
            magnitudes: Vec::new(),
            configured: false,
        };
        s.set_fft_size();
        s
    }

    pub fn with_bands(mut self, bands: usize) -> Self {
        self.bands = bands.max(1);
        self.set_fft_size();
        self
    }

    /// Per-band magnitudes of the most recent full window, linear 0..1.
    pub fn last_magnitudes(&self) -> &[f64] {
        &self.magnitudes
    }

    fn set_fft_size(&mut self) {
        let size = (self.bands * 2).next_power_of_two();
        self.plan = FftPlan::new(size);
        // Hann window: 0.5*(1 - cos(2*pi*i/(n-1))), turn = i/(n-1).
        self.window = (0..size)
            .map(|i| 0.5 * (1.0 - crate::mathf::cos_turns((i as f32) / ((size - 1) as f32)) as f64))
            .collect();
        self.re = vec![0.0; size];
        self.im = vec![0.0; size];
        self.acc.clear();
    }

    fn accept_input(&self, caps: &Caps) -> Result<usize, G2gError> {
        match caps {
            Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels,
                ..
            } if *channels > 0 => Ok(*channels as usize),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    /// Feed mono-mixed samples; run an FFT whenever a full window fills.
    fn observe(&mut self, src: &[u8]) {
        let ch = self.channels.max(1);
        for frame in src.chunks_exact(2 * ch) {
            let mut sum = 0.0;
            for c in 0..ch {
                let s = &frame[c * 2..c * 2 + 2];
                sum += i16::from_le_bytes([s[0], s[1]]) as f64;
            }
            self.acc.push(sum / ch as f64 / FULL_SCALE);
            if self.acc.len() == self.plan.size() {
                self.analyze();
                self.acc.clear();
            }
        }
    }

    fn analyze(&mut self) {
        let n = self.plan.size();
        for ((slot, &x), &w) in self.re.iter_mut().zip(&self.acc).zip(&self.window) {
            *slot = x * w;
        }
        self.im.fill(0.0);
        self.plan.forward(&mut self.re, &mut self.im);
        self.magnitudes.clear();
        for k in 0..self.bands {
            let (re, im) = (self.re[k], self.im[k]);
            let mag = crate::mathf::sqrt(re * re + im * im) / (n as f64);
            self.magnitudes.push(mag);
        }
    }
}

impl AsyncElement for Spectrum {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        self.accept_input(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::Audio {
                format: AudioFormat::PcmS16Le,
                ..
            } => CapsSet::one(input.clone()),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.channels = self.accept_input(absolute_caps)?;
        self.acc.clear();
        self.configured = true;
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
                    if self.post_messages {
                        if let Some(slice) = frame.domain.as_system_slice() {
                            self.observe(slice);
                        }
                    }
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    self.channels = self.accept_input(&c)?;
                    self.acc.clear();
                    out.push(PipelinePacket::CapsChanged(c)).await?;
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
        SPECTRUM_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Spectrum",
            "Filter/Analyzer/Audio",
            "FFT spectrum analyzer",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "bands" => {
                let b = value.as_uint().ok_or(PropError::Type)? as usize;
                if b == 0 {
                    return Err(PropError::Value);
                }
                self.bands = b;
                self.set_fft_size();
            }
            "post-messages" => self.post_messages = value.as_bool().ok_or(PropError::Type)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "bands" => Some(PropValue::Uint(self.bands as u64)),
            "post-messages" => Some(PropValue::Bool(self.post_messages)),
            _ => None,
        }
    }
}

static SPECTRUM_PROPS: &[PropertySpec] = &[
    PropertySpec::new("bands", PropKind::Uint, "number of output frequency bands"),
    PropertySpec::new(
        "post-messages",
        PropKind::Bool,
        "run the analysis when true",
    ),
];

impl PadTemplates for Spectrum {
    fn pad_templates() -> Vec<PadTemplate> {
        let pcm = pcm_caps_any_rate_and_channels(|format| format == AudioFormat::PcmS16Le);
        Vec::from([PadTemplate::sink(pcm.clone()), PadTemplate::source(pcm)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observe_fills_a_window_and_produces_bands() {
        let mut s = Spectrum::new().with_bands(4); // fft_size = 8
        s.channels = 1;
        // 8 mono samples fill one window.
        let mut bytes = vec![0u8; 8 * 2];
        for (i, chunk) in bytes.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            let v = ((i as i16) * 1000) - 3500;
            chunk.copy_from_slice(&v.to_le_bytes());
        }
        s.observe(&bytes);
        assert_eq!(
            s.last_magnitudes().len(),
            4,
            "one band per requested output"
        );
    }

    #[test]
    fn configure_rejects_non_s16le() {
        let mut s = Spectrum::new();
        let bad = Caps::Audio {
            format: AudioFormat::PcmF32Le,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        assert_eq!(
            s.configure_pipeline(&bad).unwrap_err(),
            G2gError::CapsMismatch
        );
    }
}
