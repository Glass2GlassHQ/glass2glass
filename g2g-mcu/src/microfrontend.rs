// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
// Copyright 2018 The TensorFlow Authors (tensorflow/lite/experimental/microfrontend)
// Copyright (c) 2003-2010 Mark Borgerding (KISS FFT)
//
// SPDX-License-Identifier: MPL-2.0 AND Apache-2.0 AND BSD-3-Clause

use core::f64::consts::PI;

use g2g_core::error::G2gError;
use g2g_core::frame::Frame;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::{StaticTransform, TensorDType, TensorLayout, TensorShape};

use crate::inference::TensorFormat;
use crate::lend::lend_converted;
use crate::logmel::LogMelSettings;

const FFT_SIZE: usize = 512;
// KISS FFT runs a real FFT as a complex FFT of half the length.
const COMPLEX_FFT_SIZE: usize = FFT_SIZE / 2;
const SPECTRUM_BINS: usize = COMPLEX_FFT_SIZE + 1;
const SAMPLE_BYTES: usize = 2;
const FEATURE_BYTES: usize = 4;

const WINDOW_BITS: u32 = 12;
const FILTERBANK_BITS: u32 = 12;
const INPUT_CORRECTION_BITS: u32 = FFT_SIZE.ilog2() - FILTERBANK_BITS / 2;
const NOISE_REDUCTION_BITS: u32 = 14;
const SMOOTHING_BITS: u32 = 10;
const EVEN_SMOOTHING: f32 = 0.025;
const ODD_SMOOTHING: f32 = 0.06;
const MIN_SIGNAL_REMAINING: f32 = 0.05;
const PCAN_STRENGTH: f32 = 0.95;
const PCAN_OFFSET: f32 = 80.0;
const PCAN_GAIN_BITS: u32 = 21;
const PCAN_SNR_BITS: u32 = 12;
const PCAN_OUTPUT_BITS: u32 = 6;
const PCAN_SNR_SHIFT: u32 = PCAN_GAIN_BITS - INPUT_CORRECTION_BITS - PCAN_SNR_BITS;
const WIDE_DYNAMIC_FUNCTION_BITS: u32 = 32;
const GAIN_LUT_SIZE: usize = 4 * WIDE_DYNAMIC_FUNCTION_BITS as usize - 3;
const LOG_SCALE_SHIFT: u32 = 6;
const LOG_SCALE_LOG2: u32 = 16;
const LOG_SEGMENTS_LOG2: u32 = 7;
const LOG_COEFFICIENT: u64 = 45426;
// tensorflow speech_commands multiplies the frontend output by 10/256 before training.
const TRAINING_FEATURE_SCALE: f32 = 10.0 / 256.0;

const LOG_LUT: [u16; 130] = [
    0, 224, 442, 654, 861, 1063, 1259, 1450, 1636, 1817, 1992, 2163, 2329, 2490, 2646, 2797, 2944,
    3087, 3224, 3358, 3487, 3611, 3732, 3848, 3960, 4068, 4172, 4272, 4368, 4460, 4549, 4633, 4714,
    4791, 4864, 4934, 5001, 5063, 5123, 5178, 5231, 5280, 5326, 5368, 5408, 5444, 5477, 5507, 5533,
    5557, 5578, 5595, 5610, 5622, 5631, 5637, 5640, 5641, 5638, 5633, 5626, 5615, 5602, 5586, 5568,
    5547, 5524, 5498, 5470, 5439, 5406, 5370, 5332, 5291, 5249, 5203, 5156, 5106, 5054, 5000, 4944,
    4885, 4825, 4762, 4697, 4630, 4561, 4490, 4416, 4341, 4264, 4184, 4103, 4020, 3935, 3848, 3759,
    3668, 3575, 3481, 3384, 3286, 3186, 3084, 2981, 2875, 2768, 2659, 2549, 2437, 2323, 2207, 2090,
    1971, 1851, 1729, 1605, 1480, 1353, 1224, 1094, 963, 830, 695, 559, 421, 282, 142, 0, 0,
];

#[derive(Clone, Copy, Debug, Default)]
struct Complex {
    real: i16,
    imaginary: i16,
}

fn round_fraction(product: i32) -> i16 {
    (product.wrapping_add(1 << 14) >> 15) as i16
}

fn complex_multiply(a: Complex, b: Complex) -> Complex {
    let (ar, ai, br, bi) = (
        a.real as i32,
        a.imaginary as i32,
        b.real as i32,
        b.imaginary as i32,
    );
    Complex {
        real: round_fraction((ar * br).wrapping_sub(ai * bi)),
        imaginary: round_fraction((ar * bi).wrapping_add(ai * br)),
    }
}

fn complex_add(a: Complex, b: Complex) -> Complex {
    Complex {
        real: a.real.wrapping_add(b.real),
        imaginary: a.imaginary.wrapping_add(b.imaginary),
    }
}

fn complex_subtract(a: Complex, b: Complex) -> Complex {
    Complex {
        real: a.real.wrapping_sub(b.real),
        imaginary: a.imaginary.wrapping_sub(b.imaginary),
    }
}

fn complex_divide(c: Complex, divisor: i32) -> Complex {
    let scale = i16::MAX as i32 / divisor;
    Complex {
        real: round_fraction(c.real as i32 * scale),
        imaginary: round_fraction(c.imaginary as i32 * scale),
    }
}

fn twiddle(phase: f64) -> Complex {
    Complex {
        real: libm::floor(0.5 + i16::MAX as f64 * libm::cos(phase)) as i16,
        imaginary: libm::floor(0.5 + i16::MAX as f64 * libm::sin(phase)) as i16,
    }
}

fn most_significant_bit(value: u64) -> u32 {
    u64::BITS - value.leading_zeros()
}

fn square_root_32(mut number: u32) -> u32 {
    if number == 0 {
        return 0;
    }
    let mut result = 0u32;
    let max_bit_number = (32 - most_significant_bit(number as u64)) | 1;
    let mut bit = 1u32.wrapping_shl(31 - max_bit_number);
    for _ in 0..(31 - max_bit_number) / 2 + 1 {
        if number >= result.wrapping_add(bit) {
            number = number.wrapping_sub(result.wrapping_add(bit));
            result = (result >> 1).wrapping_add(bit);
        } else {
            result >>= 1;
        }
        bit >>= 2;
    }
    if number > result && result != 0xFFFF {
        result += 1;
    }
    result
}

fn square_root_64(mut number: u64) -> u32 {
    if number >> 32 == 0 {
        return square_root_32(number as u32);
    }
    let mut result = 0u64;
    let max_bit_number = (64 - most_significant_bit(number)) | 1;
    let mut bit = 1u64.wrapping_shl(63 - max_bit_number);
    for _ in 0..(63 - max_bit_number) / 2 + 1 {
        if number >= result.wrapping_add(bit) {
            number = number.wrapping_sub(result.wrapping_add(bit));
            result = (result >> 1).wrapping_add(bit);
        } else {
            result >>= 1;
        }
        bit >>= 2;
    }
    if number > result && result != 0xFFFF_FFFF {
        result += 1;
    }
    result as u32
}

fn frequency_to_mel(frequency: f32) -> f32 {
    (1127.0 * libm::log1p(frequency as f64 / 700.0)) as f32
}

fn pcan_gain(x: u32) -> i16 {
    let input_bits = SMOOTHING_BITS - INPUT_CORRECTION_BITS;
    let x_as_float = x as f32 / (1u32 << input_bits) as f32;
    let gain =
        (1u32 << PCAN_GAIN_BITS) as f32 * libm::powf(x_as_float + PCAN_OFFSET, -PCAN_STRENGTH);
    if gain > i16::MAX as f32 {
        return i16::MAX;
    }
    (gain + 0.5) as i16
}

fn gain_lookup_table() -> [i16; GAIN_LUT_SIZE] {
    let mut table = [0i16; GAIN_LUT_SIZE];
    let mut set = |index: usize, value: i16| {
        if let Some(slot) = table.get_mut(index) {
            *slot = value;
        }
    };
    set(0, pcan_gain(0));
    set(1, pcan_gain(1));
    for interval in 2..=WIDE_DYNAMIC_FUNCTION_BITS {
        let x0 = 1u32 << (interval - 1);
        let x1 = x0 + (x0 >> 1);
        let x2 = if interval == WIDE_DYNAMIC_FUNCTION_BITS {
            x0 + (x0 - 1)
        } else {
            2 * x0
        };
        let (y0, y1, y2) = (pcan_gain(x0), pcan_gain(x1), pcan_gain(x2));
        let difference_1 = y1 as i32 - y0 as i32;
        let difference_2 = y2 as i32 - y0 as i32;
        let a1 = 4 * difference_1 - difference_2;
        let a2 = difference_2 - a1;
        let base = 4 * interval as usize - 6;
        set(base, y0);
        set(base + 1, a1 as i16);
        set(base + 2, a2 as i16);
    }
    table
}

fn wide_dynamic_function(x: u32, table: &[i16; GAIN_LUT_SIZE]) -> i16 {
    if x <= 2 {
        return table.get(x as usize).copied().unwrap_or(0);
    }
    let interval = most_significant_bit(x as u64);
    let base = 4 * interval as usize - 6;
    let lookup = |offset: usize| table.get(base + offset).copied().unwrap_or(0) as i32;
    let fraction = if interval < 11 {
        x << (11 - interval)
    } else {
        x >> (interval - 11)
    } & 0x3FF;
    let fraction = fraction as i32;
    let mut result = (lookup(2) * fraction) >> 5;
    result = result.wrapping_add(lookup(1).wrapping_shl(5));
    result = result.wrapping_mul(fraction);
    result = result.wrapping_add(1 << 14) >> 15;
    result = result.wrapping_add(lookup(0));
    result as i16
}

fn pcan_shrink(snr: u32) -> u32 {
    if snr < (2 << PCAN_SNR_BITS) {
        snr.wrapping_mul(snr) >> (2 + 2 * PCAN_SNR_BITS - PCAN_OUTPUT_BITS)
    } else {
        (snr >> (PCAN_SNR_BITS - PCAN_OUTPUT_BITS)) - (1 << PCAN_OUTPUT_BITS)
    }
}

fn log2_fraction(x: u32, log2x: u32) -> u32 {
    let mut fraction = (x as i64 - (1i64 << log2x)) as i32;
    if log2x < LOG_SCALE_LOG2 {
        fraction = fraction.wrapping_shl(LOG_SCALE_LOG2 - log2x);
    } else {
        fraction = fraction.wrapping_shr(log2x - LOG_SCALE_LOG2);
    }
    let segment = (fraction >> (LOG_SCALE_LOG2 - LOG_SEGMENTS_LOG2)) as u32;
    let segment_unit = (1u32 << LOG_SCALE_LOG2) >> LOG_SEGMENTS_LOG2;
    let lookup = |index: u32| LOG_LUT.get(index as usize).copied().unwrap_or(0) as i32;
    let c0 = lookup(segment);
    let c1 = lookup(segment + 1);
    let segment_base = segment_unit.wrapping_mul(segment) as i32;
    let relative = (c1 - c0).wrapping_mul(fraction.wrapping_sub(segment_base)) >> LOG_SCALE_LOG2;
    fraction.wrapping_add(c0).wrapping_add(relative) as u32
}

fn log_scale(x: u32) -> u32 {
    let integer = most_significant_bit(x as u64) - 1;
    let log2 = (integer << LOG_SCALE_LOG2).wrapping_add(log2_fraction(x, integer));
    let round = 1u32 << (LOG_SCALE_LOG2 - 1);
    let loge = ((LOG_COEFFICIENT * log2 as u64 + round as u64) >> LOG_SCALE_LOG2) as u32;
    (loge << LOG_SCALE_SHIFT).wrapping_add(round) >> LOG_SCALE_LOG2
}

#[derive(Clone, Copy, Debug)]
struct BinWeight {
    channel: usize,
    weight: i16,
    unweight: i16,
}

// The TensorFlow Lite Micro audio frontend (window, FFT, mel filterbank, noise
// reduction, PCAN gain control, log scale) that micro_speech-style keyword
// models are trained on, with noise state reset per clip.
pub struct MicroFrontend<'r, const CHANNELS: usize, const N: usize, const BYTES: usize> {
    settings: LogMelSettings,
    window: [i16; FFT_SIZE],
    twiddles: [Complex; COMPLEX_FFT_SIZE],
    real_twiddles: [Complex; COMPLEX_FFT_SIZE / 2],
    bins: [Option<BinWeight>; SPECTRUM_BINS],
    gains: [i16; GAIN_LUT_SIZE],
    noise: [u32; CHANNELS],
    ring: &'r StaticLendRing<N, BYTES>,
}

impl<const CHANNELS: usize, const N: usize, const BYTES: usize>
    MicroFrontend<'static, CHANNELS, N, BYTES>
{
    pub fn new(
        settings: LogMelSettings,
        ring: &'static StaticLendRing<N, BYTES>,
    ) -> Result<Self, G2gError> {
        // SAFETY: `'static` trivially satisfies with_ring's outlives contract.
        unsafe { Self::with_ring(settings, ring) }
    }
}

impl<'r, const CHANNELS: usize, const N: usize, const BYTES: usize>
    MicroFrontend<'r, CHANNELS, N, BYTES>
{
    /// # Safety
    /// The ring must outlive every frame this element publishes.
    pub unsafe fn with_ring(
        settings: LogMelSettings,
        ring: &'r StaticLendRing<N, BYTES>,
    ) -> Result<Self, G2gError> {
        let nyquist = settings.sample_rate as f32 / 2.0;
        let valid = settings.sample_rate > 0
            && settings.window_samples.next_power_of_two() == FFT_SIZE
            && settings.hop_samples > 0
            && settings.frames > 0
            && CHANNELS > 0
            && settings.lowest_frequency >= 0.0
            && settings.lowest_frequency < settings.highest_frequency
            && settings.highest_frequency <= nyquist;
        if !valid {
            return Err(G2gError::CapsMismatch);
        }

        let mut window = [0i16; FFT_SIZE];
        let arg = (PI * 2.0 / settings.window_samples as f64) as f32;
        for (index, coefficient) in window.iter_mut().take(settings.window_samples).enumerate() {
            let value = (0.5 - 0.5 * libm::cos(arg as f64 * (index as f64 + 0.5))) as f32;
            *coefficient = libm::floor((value * (1u32 << WINDOW_BITS) as f32) as f64 + 0.5) as i16;
        }

        let mut twiddles = [Complex::default(); COMPLEX_FFT_SIZE];
        for (index, value) in twiddles.iter_mut().enumerate() {
            *value = twiddle(-2.0 * PI * index as f64 / COMPLEX_FFT_SIZE as f64);
        }
        let mut real_twiddles = [Complex::default(); COMPLEX_FFT_SIZE / 2];
        for (index, value) in real_twiddles.iter_mut().enumerate() {
            *value = twiddle(-PI * ((index + 1) as f64 / COMPLEX_FFT_SIZE as f64 + 0.5));
        }

        Ok(Self {
            settings,
            window,
            twiddles,
            real_twiddles,
            bins: filterbank_bins::<CHANNELS>(&settings)?,
            gains: gain_lookup_table(),
            noise: [0; CHANNELS],
            ring,
        })
    }

    pub fn output_format(&self) -> TensorFormat {
        TensorFormat {
            dtype: TensorDType::F32,
            shape: TensorShape::new([1, (self.settings.frames * CHANNELS) as u32]),
            layout: TensorLayout::Nhwc,
        }
    }

    fn radix4_butterflies(&self, block: &mut [Complex], m: usize, stride: usize) {
        for k in 0..m {
            let twiddle = |power: usize| {
                self.twiddles
                    .get(power * k * stride)
                    .copied()
                    .unwrap_or_default()
            };
            let read = |block: &[Complex], index: usize| {
                complex_divide(block.get(index).copied().unwrap_or_default(), 4)
            };
            let mut first = read(block, k);
            let second = read(block, k + m);
            let third = read(block, k + 2 * m);
            let fourth = read(block, k + 3 * m);
            let product_1 = complex_multiply(second, twiddle(1));
            let product_2 = complex_multiply(third, twiddle(2));
            let product_3 = complex_multiply(fourth, twiddle(3));
            let difference = complex_subtract(first, product_2);
            first = complex_add(first, product_2);
            let outer_sum = complex_add(product_1, product_3);
            let outer_difference = complex_subtract(product_1, product_3);
            let outputs = [
                (k, complex_add(first, outer_sum)),
                (
                    k + m,
                    Complex {
                        real: difference.real.wrapping_add(outer_difference.imaginary),
                        imaginary: difference.imaginary.wrapping_sub(outer_difference.real),
                    },
                ),
                (k + 2 * m, complex_subtract(first, outer_sum)),
                (
                    k + 3 * m,
                    Complex {
                        real: difference.real.wrapping_sub(outer_difference.imaginary),
                        imaginary: difference.imaginary.wrapping_add(outer_difference.real),
                    },
                ),
            ];
            for (index, value) in outputs {
                if let Some(slot) = block.get_mut(index) {
                    *slot = value;
                }
            }
        }
    }

    // KISS FFT's fixed-point real FFT, for a length whose half is a power of four.
    fn real_fft(&self, input: &[i16; FFT_SIZE]) -> [Complex; SPECTRUM_BINS] {
        let mut packed = [Complex::default(); COMPLEX_FFT_SIZE];
        let digits = COMPLEX_FFT_SIZE.ilog2() / 2;
        for (index, value) in packed.iter_mut().enumerate() {
            let mut reversed = 0;
            let mut remaining = index;
            for _ in 0..digits {
                reversed = reversed * 4 + remaining % 4;
                remaining /= 4;
            }
            *value = Complex {
                real: input.get(2 * reversed).copied().unwrap_or(0),
                imaginary: input.get(2 * reversed + 1).copied().unwrap_or(0),
            };
        }
        let mut m = 1;
        while m < COMPLEX_FFT_SIZE {
            let stride = COMPLEX_FFT_SIZE / (4 * m);
            for block in packed.chunks_exact_mut(4 * m) {
                self.radix4_butterflies(block, m, stride);
            }
            m *= 4;
        }

        let mut spectrum = [Complex::default(); SPECTRUM_BINS];
        let first = packed.first().copied().unwrap_or_default();
        let dc = complex_divide(first, 2);
        if let Some(slot) = spectrum.first_mut() {
            slot.real = dc.real.wrapping_add(dc.imaginary);
        }
        if let Some(slot) = spectrum.last_mut() {
            slot.real = dc.real.wrapping_sub(dc.imaginary);
        }
        for k in 1..=COMPLEX_FFT_SIZE / 2 {
            let forward = complex_divide(packed.get(k).copied().unwrap_or_default(), 2);
            let mirrored = packed
                .get(COMPLEX_FFT_SIZE - k)
                .copied()
                .unwrap_or_default();
            let mirrored = complex_divide(
                Complex {
                    real: mirrored.real,
                    imaginary: mirrored.imaginary.wrapping_neg(),
                },
                2,
            );
            let sum = complex_add(forward, mirrored);
            let difference = complex_subtract(forward, mirrored);
            let rotated = complex_multiply(
                difference,
                self.real_twiddles.get(k - 1).copied().unwrap_or_default(),
            );
            let half_sum = |a: i16, b: i16| ((a as i32 + b as i32) >> 1) as i16;
            let half_difference = |a: i16, b: i16| ((a as i32 - b as i32) >> 1) as i16;
            if let Some(slot) = spectrum.get_mut(k) {
                *slot = Complex {
                    real: half_sum(sum.real, rotated.real),
                    imaginary: half_sum(sum.imaginary, rotated.imaginary),
                };
            }
            if let Some(slot) = spectrum.get_mut(COMPLEX_FFT_SIZE - k) {
                *slot = Complex {
                    real: half_difference(sum.real, rotated.real),
                    imaginary: half_difference(rotated.imaginary, sum.imaginary),
                };
            }
        }
        spectrum
    }

    fn frame_features(&mut self, samples: &[u8]) -> [u16; CHANNELS] {
        let mut windowed = [0i16; FFT_SIZE];
        let mut max_magnitude = 0i16;
        for ((sample, coefficient), output) in samples
            .chunks_exact(SAMPLE_BYTES)
            .zip(self.window.iter())
            .zip(windowed.iter_mut())
        {
            let &[low, high] = sample else {
                continue;
            };
            let value = ((i16::from_le_bytes([low, high]) as i32 * *coefficient as i32)
                >> WINDOW_BITS) as i16;
            *output = value;
            let magnitude = if value < 0 {
                value.wrapping_neg()
            } else {
                value
            };
            max_magnitude = max_magnitude.max(magnitude);
        }
        let input_shift = 15 - most_significant_bit(max_magnitude as u64);
        for value in windowed.iter_mut() {
            *value = ((*value as u16 as u32) << input_shift) as u16 as i16;
        }
        let spectrum = self.real_fft(&windowed);

        let mut accumulators = [0u64; CHANNELS];
        for (bin, value) in self.bins.iter().zip(spectrum.iter()) {
            let Some(bin) = bin else {
                continue;
            };
            let (real, imaginary) = (value.real as i32, value.imaginary as i32);
            let energy = (real * real).wrapping_add(imaginary * imaginary) as i64 as u64;
            // a bin feeds its channel's rising slope and the previous channel's falling one
            if let Some(previous) = bin
                .channel
                .checked_sub(1)
                .and_then(|c| accumulators.get_mut(c))
            {
                *previous = previous.wrapping_add((bin.weight as i64 as u64).wrapping_mul(energy));
            }
            if let Some(current) = accumulators.get_mut(bin.channel) {
                *current = current.wrapping_add((bin.unweight as i64 as u64).wrapping_mul(energy));
            }
        }

        let mut features = [0u16; CHANNELS];
        for (channel, ((accumulator, noise), feature)) in accumulators
            .iter()
            .zip(self.noise.iter_mut())
            .zip(features.iter_mut())
            .enumerate()
        {
            let signal = square_root_64(*accumulator) >> input_shift;

            let smoothing = if channel % 2 == 0 {
                (EVEN_SMOOTHING * (1u32 << NOISE_REDUCTION_BITS) as f32) as u16
            } else {
                (ODD_SMOOTHING * (1u32 << NOISE_REDUCTION_BITS) as f32) as u16
            } as u64;
            let remaining = (MIN_SIGNAL_REMAINING * (1u32 << NOISE_REDUCTION_BITS) as f32) as u16;
            let scaled_up = signal << SMOOTHING_BITS;
            let mut estimate = ((scaled_up as u64 * smoothing
                + *noise as u64 * ((1u64 << NOISE_REDUCTION_BITS) - smoothing))
                >> NOISE_REDUCTION_BITS) as u32;
            *noise = estimate;
            estimate = estimate.min(scaled_up);
            let floor = ((signal as u64 * remaining as u64) >> NOISE_REDUCTION_BITS) as u32;
            let reduced = ((scaled_up - estimate) >> SMOOTHING_BITS).max(floor);

            let gain = wide_dynamic_function(*noise, &self.gains) as i32 as u32;
            let snr = ((reduced as u64 * gain as u64) >> PCAN_SNR_SHIFT) as u32;
            let controlled = pcan_shrink(snr) << INPUT_CORRECTION_BITS;

            let logged = if controlled > 1 {
                log_scale(controlled)
            } else {
                0
            };
            *feature = logged.min(u16::MAX as u32) as u16;
        }
        features
    }
}

fn filterbank_bins<const CHANNELS: usize>(
    settings: &LogMelSettings,
) -> Result<[Option<BinWeight>; SPECTRUM_BINS], G2gError> {
    let mel_low = frequency_to_mel(settings.lowest_frequency);
    let mel_spacing =
        (frequency_to_mel(settings.highest_frequency) - mel_low) / (CHANNELS + 1) as f32;
    let center = |channel: usize| mel_low + mel_spacing * (channel + 1) as f32;
    let hertz_per_bin =
        (0.5 * settings.sample_rate as f64 / (SPECTRUM_BINS as f32 - 1.0) as f64) as f32;
    let start = (1.5 + (settings.lowest_frequency / hertz_per_bin) as f64) as usize;

    let mut bins = [None; SPECTRUM_BINS];
    let mut bin = start;
    for channel in 0..=CHANNELS {
        let lower = if channel == 0 {
            mel_low
        } else {
            center(channel - 1)
        };
        loop {
            let mel = frequency_to_mel(bin as f32 * hertz_per_bin);
            if mel > center(channel) {
                break;
            }
            let Some(slot) = bins.get_mut(bin) else {
                return Err(G2gError::CapsMismatch);
            };
            let weight = (center(channel) - mel) / (center(channel) - lower);
            *slot = Some(BinWeight {
                channel,
                weight: libm::floor((weight * (1u32 << FILTERBANK_BITS) as f32) as f64 + 0.5)
                    as i16,
                unweight: libm::floor(
                    (1.0 - weight as f64) * (1u32 << FILTERBANK_BITS) as f64 + 0.5,
                ) as i16,
            });
            bin += 1;
        }
    }
    if bin >= SPECTRUM_BINS {
        return Err(G2gError::CapsMismatch);
    }
    Ok(bins)
}

impl<const CHANNELS: usize, const N: usize, const BYTES: usize> StaticTransform
    for MicroFrontend<'_, CHANNELS, N, BYTES>
{
    async fn process(&mut self, input: Frame) -> Result<Option<Frame>, G2gError> {
        let Some(pcm) = input.domain.as_system_slice() else {
            return Err(G2gError::UnsupportedDomain);
        };
        if pcm.len() != self.settings.clip_samples() * SAMPLE_BYTES {
            return Err(G2gError::CapsMismatch);
        }
        self.noise = [0; CHANNELS];
        let output_len = self.settings.frames * CHANNELS * FEATURE_BYTES;
        let window_bytes = self.settings.window_samples * SAMPLE_BYTES;
        let hop_bytes = self.settings.hop_samples * SAMPLE_BYTES;
        let ring = self.ring;
        // SAFETY: the constructor established the ring-outlives-frames
        // contract (`new`: 'static, `with_ring`: the caller's contract).
        unsafe {
            lend_converted(ring, &input, output_len, |pcm, out| {
                for (frame, row) in out.chunks_exact_mut(CHANNELS * FEATURE_BYTES).enumerate() {
                    let start = frame * hop_bytes;
                    let Some(window) = pcm.get(start..start + window_bytes) else {
                        continue;
                    };
                    let features = self.frame_features(window);
                    for (bytes, feature) in row.chunks_exact_mut(FEATURE_BYTES).zip(features) {
                        let value = feature as f32 * TRAINING_FEATURE_SCALE;
                        bytes.copy_from_slice(&value.to_le_bytes());
                    }
                }
            })
        }
        .map(Some)
    }
}

impl<const CHANNELS: usize, const N: usize, const BYTES: usize> core::fmt::Debug
    for MicroFrontend<'_, CHANNELS, N, BYTES>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MicroFrontend")
            .field("settings", &self.settings)
            .field("channels", &CHANNELS)
            .finish_non_exhaustive()
    }
}
