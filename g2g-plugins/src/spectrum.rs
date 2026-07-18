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

use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec,
};

/// In-place iterative radix-2 Cooley-Tukey FFT (decimation in time). `re`/`im`
/// have the same power-of-two length; twiddles come from the crate's `libm`-free
/// trig. Forward transform (`exp(-2*pi*i*k/N)`).
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());
    // bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        for start in (0..n).step_by(len) {
            for k in 0..half {
                // w = exp(-2*pi*i*k/len); sin/cos take turns, so the turn is -k/len.
                let turns = -(k as f32) / (len as f32);
                let wr = crate::mathf::cos_turns(turns) as f64;
                let wi = crate::mathf::sin_turns(turns) as f64;
                let (pr, pi) = (re[start + k], im[start + k]);
                let (qr, qi) = (re[start + k + half], im[start + k + half]);
                let vr = qr * wr - qi * wi;
                let vi = qr * wi + qi * wr;
                re[start + k] = pr + vr;
                im[start + k] = pi + vi;
                re[start + k + half] = pr - vr;
                im[start + k + half] = pi - vi;
            }
        }
        len <<= 1;
    }
}

#[derive(Debug)]
pub struct Spectrum {
    bands: usize,
    post_messages: bool,
    channels: usize,
    fft_size: usize,
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
            fft_size: 0,
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
        self.fft_size = (self.bands * 2).next_power_of_two();
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
            if self.acc.len() == self.fft_size {
                self.analyze();
                self.acc.clear();
            }
        }
    }

    fn analyze(&mut self) {
        let n = self.fft_size;
        let mut re = vec![0.0f64; n];
        let mut im = vec![0.0f64; n];
        for (i, (slot, &x)) in re.iter_mut().zip(self.acc.iter()).enumerate() {
            // Hann window: 0.5*(1 - cos(2*pi*i/(n-1))), turn = i/(n-1).
            let w = 0.5 * (1.0 - crate::mathf::cos_turns((i as f32) / ((n - 1) as f32)) as f64);
            *slot = x * w;
        }
        fft(&mut re, &mut im);
        self.magnitudes.clear();
        for k in 0..self.bands {
            let mag = crate::mathf::sqrt(re[k] * re[k] + im[k] * im[k]) / (n as f64);
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
                        if let MemoryDomain::System(slice) = &frame.domain {
                            self.observe(slice.as_slice());
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
        let pcm = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
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

    #[test]
    fn fft_of_a_single_cosine_peaks_at_its_bin() {
        // x[n] = cos(2*pi*3*n/N): the DFT is real with peaks at bins 3 and N-3.
        let n = 16usize;
        let mut re = vec![0.0f64; n];
        let mut im = vec![0.0f64; n];
        for (i, slot) in re.iter_mut().enumerate() {
            *slot = crate::mathf::cos_turns(3.0 * (i as f32) / (n as f32)) as f64;
        }
        fft(&mut re, &mut im);
        let mag: Vec<f64> = (0..n)
            .map(|k| crate::mathf::sqrt(re[k] * re[k] + im[k] * im[k]))
            .collect();
        // bin 3 should dominate over a non-adjacent bin like 6.
        assert!(
            mag[3] > mag[6] * 4.0,
            "bin 3 {} vs bin 6 {}",
            mag[3],
            mag[6]
        );
        assert!(mag[3] > mag[0] * 4.0, "bin 3 {} vs DC {}", mag[3], mag[0]);
    }

    #[test]
    fn dc_signal_concentrates_in_bin_zero() {
        let mut re = vec![1.0f64; 8];
        let mut im = vec![0.0f64; 8];
        fft(&mut re, &mut im);
        assert!((re[0] - 8.0).abs() < 1e-9, "DC bin should be N");
        for k in 1..8 {
            assert!(
                re[k].abs() < 1e-9 && im[k].abs() < 1e-9,
                "bin {k} should be ~0"
            );
        }
    }

    #[test]
    fn observe_fills_a_window_and_produces_bands() {
        let mut s = Spectrum::new().with_bands(4); // fft_size = 8
        s.channels = 1;
        // 8 mono samples fill one window.
        let mut bytes = vec![0u8; 8 * 2];
        for (i, chunk) in bytes.chunks_exact_mut(2).enumerate() {
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
        };
        assert_eq!(
            s.configure_pipeline(&bad).unwrap_err(),
            G2gError::CapsMismatch
        );
    }
}
