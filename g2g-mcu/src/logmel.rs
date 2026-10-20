use core::f64::consts::PI;

use g2g_core::error::G2gError;
use g2g_core::frame::Frame;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::{StaticTransform, TensorDType, TensorLayout, TensorShape};

use crate::inference::TensorFormat;
use crate::lend::lend_converted;

const FFT_SIZE_LOG2: u32 = 9;
pub const FFT_SIZE: usize = 1 << FFT_SIZE_LOG2;
const SPECTRUM_BINS: usize = FFT_SIZE / 2 + 1;
const SAMPLE_BYTES: usize = 2;
const FEATURE_BYTES: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogMelSettings {
    pub sample_rate: u32,
    pub window_samples: usize,
    pub hop_samples: usize,
    pub frames: usize,
    pub lowest_frequency: f32,
    pub highest_frequency: f32,
}

impl LogMelSettings {
    pub fn clip_samples(&self) -> usize {
        self.frames
            .saturating_sub(1)
            .saturating_mul(self.hop_samples)
            .saturating_add(self.window_samples)
    }
}

// HTK mel scale.
fn mel(frequency: f64) -> f64 {
    1127.0 * libm::log(1.0 + frequency / 700.0)
}

// Which pair of adjacent mel band edges a spectrum bin falls between, and how
// far along from the lower edge.
#[derive(Clone, Copy, Debug)]
struct BinPlacement {
    lower_edge: usize,
    position: f32,
}

pub struct LogMelSpectrogram<'r, const MELS: usize, const N: usize, const BYTES: usize> {
    settings: LogMelSettings,
    hann: [f32; FFT_SIZE],
    twiddles: [(f32, f32); FFT_SIZE / 2],
    placements: [Option<BinPlacement>; SPECTRUM_BINS],
    spectrum: [(f32, f32); FFT_SIZE],
    ring: &'r StaticLendRing<N, BYTES>,
}

impl<const MELS: usize, const N: usize, const BYTES: usize>
    LogMelSpectrogram<'static, MELS, N, BYTES>
{
    pub fn new(
        settings: LogMelSettings,
        ring: &'static StaticLendRing<N, BYTES>,
    ) -> Result<Self, G2gError> {
        // SAFETY: `'static` trivially satisfies with_ring's outlives contract.
        unsafe { Self::with_ring(settings, ring) }
    }
}

impl<'r, const MELS: usize, const N: usize, const BYTES: usize>
    LogMelSpectrogram<'r, MELS, N, BYTES>
{
    /// # Safety
    /// The ring must outlive every frame this element publishes.
    pub unsafe fn with_ring(
        settings: LogMelSettings,
        ring: &'r StaticLendRing<N, BYTES>,
    ) -> Result<Self, G2gError> {
        let nyquist = settings.sample_rate as f32 / 2.0;
        let valid = settings.sample_rate > 0
            && (1..=FFT_SIZE).contains(&settings.window_samples)
            && settings.hop_samples > 0
            && settings.frames > 0
            && MELS > 0
            && settings.lowest_frequency >= 0.0
            && settings.lowest_frequency < settings.highest_frequency
            && settings.highest_frequency <= nyquist;
        if !valid {
            return Err(G2gError::CapsMismatch);
        }

        let mut hann = [0.0f32; FFT_SIZE];
        let window = settings.window_samples as f64;
        for (index, weight) in hann.iter_mut().take(settings.window_samples).enumerate() {
            *weight = (0.5 - 0.5 * libm::cos(2.0 * PI * index as f64 / window)) as f32;
        }

        let mut twiddles = [(0.0f32, 0.0f32); FFT_SIZE / 2];
        for (index, twiddle) in twiddles.iter_mut().enumerate() {
            let angle = 2.0 * PI * index as f64 / FFT_SIZE as f64;
            *twiddle = (libm::cos(angle) as f32, -libm::sin(angle) as f32);
        }

        // MELS + 2 edges evenly spaced in mel: band b rises from edge b to its
        // peak at edge b + 1 and falls to zero at edge b + 2.
        let lowest_mel = mel(settings.lowest_frequency as f64);
        let mel_step = (mel(settings.highest_frequency as f64) - lowest_mel) / (MELS + 1) as f64;
        let mut placements = [None; SPECTRUM_BINS];
        for (bin, placement) in placements.iter_mut().enumerate() {
            let frequency = bin as f64 * settings.sample_rate as f64 / FFT_SIZE as f64;
            let steps = (mel(frequency) - lowest_mel) / mel_step;
            if steps < 0.0 || steps >= (MELS + 1) as f64 {
                continue;
            }
            let lower_edge = steps as usize;
            *placement = Some(BinPlacement {
                lower_edge,
                position: (steps - lower_edge as f64) as f32,
            });
        }

        Ok(Self {
            settings,
            hann,
            twiddles,
            placements,
            spectrum: [(0.0, 0.0); FFT_SIZE],
            ring,
        })
    }

    pub fn output_format(&self) -> TensorFormat {
        TensorFormat {
            dtype: TensorDType::F32,
            shape: TensorShape::new([1, (self.settings.frames * MELS) as u32]),
            layout: TensorLayout::Nhwc,
        }
    }

    fn power_spectrum(&mut self, window: &[u8]) {
        self.spectrum = [(0.0, 0.0); FFT_SIZE];
        // Writing each windowed sample to its bit-reversed index is the
        // reordering an in-place radix-2 FFT starts from.
        for (index, (sample, weight)) in window
            .chunks_exact(SAMPLE_BYTES)
            .zip(self.hann.iter())
            .enumerate()
        {
            let &[low, high] = sample else {
                continue;
            };
            let value = i16::from_le_bytes([low, high]) as f32 * weight;
            let reversed = index.reverse_bits() >> (usize::BITS - FFT_SIZE_LOG2);
            if let Some(slot) = self.spectrum.get_mut(reversed) {
                *slot = (value, 0.0);
            }
        }
        let mut half = 1;
        while half < FFT_SIZE {
            let stride = FFT_SIZE / (2 * half);
            for block in self.spectrum.chunks_exact_mut(2 * half) {
                let Some((lower, upper)) = block.split_at_mut_checked(half) else {
                    continue;
                };
                for (index, (a, b)) in lower.iter_mut().zip(upper.iter_mut()).enumerate() {
                    let Some(&(twiddle_real, twiddle_imaginary)) =
                        self.twiddles.get(index * stride)
                    else {
                        continue;
                    };
                    let real = b.0 * twiddle_real - b.1 * twiddle_imaginary;
                    let imaginary = b.0 * twiddle_imaginary + b.1 * twiddle_real;
                    *b = (a.0 - real, a.1 - imaginary);
                    *a = (a.0 + real, a.1 + imaginary);
                }
            }
            half *= 2;
        }
    }

    fn mel_energies(&self) -> [f32; MELS] {
        let mut energies = [0.0f32; MELS];
        for (&(real, imaginary), placement) in self.spectrum.iter().zip(self.placements.iter()) {
            let Some(placement) = placement else {
                continue;
            };
            let power = real * real + imaginary * imaginary;
            // The bin sits on the rising side of the band starting at its lower
            // edge and on the falling side of the band before it.
            if let Some(energy) = energies.get_mut(placement.lower_edge) {
                *energy += placement.position * power;
            }
            if let Some(energy) = placement
                .lower_edge
                .checked_sub(1)
                .and_then(|band| energies.get_mut(band))
            {
                *energy += (1.0 - placement.position) * power;
            }
        }
        energies
    }
}

impl<const MELS: usize, const N: usize, const BYTES: usize> StaticTransform
    for LogMelSpectrogram<'_, MELS, N, BYTES>
{
    async fn process(&mut self, input: Frame) -> Result<Option<Frame>, G2gError> {
        let Some(pcm) = input.domain.as_system_slice() else {
            return Err(G2gError::UnsupportedDomain);
        };
        if pcm.len() != self.settings.clip_samples() * SAMPLE_BYTES {
            return Err(G2gError::CapsMismatch);
        }
        let output_len = self.settings.frames * MELS * FEATURE_BYTES;
        let window_bytes = self.settings.window_samples * SAMPLE_BYTES;
        let hop_bytes = self.settings.hop_samples * SAMPLE_BYTES;
        let ring = self.ring;
        // SAFETY: the constructor established the ring-outlives-frames
        // contract (`new`: 'static, `with_ring`: the caller's contract).
        unsafe {
            lend_converted(ring, &input, output_len, |pcm, out| {
                for (frame, row) in out.chunks_exact_mut(MELS * FEATURE_BYTES).enumerate() {
                    let start = frame * hop_bytes;
                    let Some(window) = pcm.get(start..start + window_bytes) else {
                        continue;
                    };
                    self.power_spectrum(window);
                    let energies = self.mel_energies();
                    for (bytes, energy) in row.chunks_exact_mut(FEATURE_BYTES).zip(energies) {
                        bytes.copy_from_slice(&libm::logf(energy + 1.0).to_le_bytes());
                    }
                }
            })
        }
        .map(Some)
    }
}

impl<const MELS: usize, const N: usize, const BYTES: usize> core::fmt::Debug
    for LogMelSpectrogram<'_, MELS, N, BYTES>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LogMelSpectrogram")
            .field("settings", &self.settings)
            .field("mels", &MELS)
            .finish_non_exhaustive()
    }
}
