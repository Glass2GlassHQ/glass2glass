mod util;

use std::f64::consts::PI;

use g2g_core::error::G2gError;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::{StaticTransform, TensorDType, TensorLayout, TensorShape};
use g2g_mcu::logmel::{FFT_SIZE, MICRO_SPEECH_SETTINGS};
use g2g_mcu::{Int8Quantize, LogMelSettings, LogMelSpectrogram, TensorFormat};
use util::{block_on, frame_of, le_bytes, payload};

const MELS: usize = 40;
const FRAMES: usize = 4;
const SETTINGS: LogMelSettings = LogMelSettings {
    frames: FRAMES,
    ..MICRO_SPEECH_SETTINGS
};
const CLIP_SAMPLES: usize = SETTINGS.clip_samples();
const FEATURE_BYTES: usize = FRAMES * MELS * 4;
// Well under one int8 step of the keyword model's input scale (about 0.1).
const TOLERANCE: f64 = 0.01;

// Two tones over a pseudo-random noise floor, so every band has energy.
fn clip() -> Vec<i16> {
    let mut noise_state = 1u32;
    (0..CLIP_SAMPLES)
        .map(|n| {
            noise_state = noise_state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            let noise = (noise_state >> 24) as f64 - 128.0;
            let t = n as f64 / SETTINGS.sample_rate as f64;
            let tones =
                3000.0 * (2.0 * PI * 1000.0 * t).sin() + 800.0 * (2.0 * PI * 3100.0 * t).sin();
            (tones + noise).round() as i16
        })
        .collect()
}

fn mel(frequency: f64) -> f64 {
    1127.0 * (1.0 + frequency / 700.0).ln()
}

// A direct f64 DFT and triangular mel filterbank, written from the definition.
fn reference(samples: &[i16]) -> Vec<f64> {
    let rate = SETTINGS.sample_rate as f64;
    let window = SETTINGS.window_samples;
    let lowest = mel(SETTINGS.lowest_frequency as f64);
    let highest = mel(SETTINGS.highest_frequency as f64);
    let edges: Vec<f64> = (0..MELS + 2)
        .map(|i| lowest + (highest - lowest) * i as f64 / (MELS + 1) as f64)
        .collect();
    let mut features = Vec::new();
    for frame in 0..FRAMES {
        let start = frame * SETTINGS.hop_samples;
        let windowed: Vec<f64> = (0..window)
            .map(|n| {
                let hann = 0.5 - 0.5 * (2.0 * PI * n as f64 / window as f64).cos();
                samples[start + n] as f64 * hann
            })
            .collect();
        let power: Vec<f64> = (0..=FFT_SIZE / 2)
            .map(|k| {
                let (mut real, mut imaginary) = (0.0, 0.0);
                for (n, &x) in windowed.iter().enumerate() {
                    let angle = 2.0 * PI * (k * n) as f64 / FFT_SIZE as f64;
                    real += x * angle.cos();
                    imaginary -= x * angle.sin();
                }
                real * real + imaginary * imaginary
            })
            .collect();
        for band in 0..MELS {
            let (left, center, right) = (edges[band], edges[band + 1], edges[band + 2]);
            let energy: f64 = power
                .iter()
                .enumerate()
                .map(|(k, p)| {
                    let m = mel(k as f64 * rate / FFT_SIZE as f64);
                    let weight = if m >= left && m < center {
                        (m - left) / (center - left)
                    } else if m >= center && m < right {
                        (right - m) / (right - center)
                    } else {
                        0.0
                    };
                    weight * p
                })
                .sum();
            features.push((energy + 1.0).ln());
        }
    }
    features
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn log_mel_features_match_a_direct_dft_reference() {
    let ring: StaticLendRing<1, FEATURE_BYTES> = StaticLendRing::new();
    // SAFETY: the rings outlive every frame in this test.
    let mut element =
        unsafe { LogMelSpectrogram::<MELS, 1, FEATURE_BYTES>::with_ring(SETTINGS, &ring) }.unwrap();
    assert_eq!(
        element.output_format(),
        TensorFormat {
            dtype: TensorDType::F32,
            shape: TensorShape::new([1, (FRAMES * MELS) as u32]),
            layout: TensorLayout::Nhwc,
        }
    );
    let samples = clip();
    let input_ring: StaticLendRing<1, { CLIP_SAMPLES * 2 }> = StaticLendRing::new();
    let out = block_on(element.process(frame_of(&input_ring, &le_bytes(&samples), 7, 3)))
        .unwrap()
        .unwrap();
    assert_eq!((out.timing.pts_ns, out.sequence), (7, 3));

    let ours = floats(payload(&out));
    let expected = reference(&samples);
    assert_eq!(ours.len(), expected.len());
    let worst = ours
        .iter()
        .zip(&expected)
        .map(|(&a, &b)| (a as f64 - b).abs())
        .fold(0.0, f64::max);
    assert!(worst < TOLERANCE, "worst log-mel difference {worst}");

    // The 1 kHz tone is the loudest component, so its band peaks in every frame.
    let tone_band = (0..MELS)
        .max_by(|&a, &b| expected[a].total_cmp(&expected[b]))
        .unwrap();
    for row in ours.chunks_exact(MELS) {
        let loudest = (0..MELS)
            .max_by(|&a, &b| row[a].total_cmp(&row[b]))
            .unwrap();
        assert_eq!(loudest, tone_band);
    }
}

#[test]
fn rejects_a_clip_of_the_wrong_length_and_invalid_settings() {
    let ring: StaticLendRing<1, FEATURE_BYTES> = StaticLendRing::new();
    // SAFETY: the rings outlive every frame in this test.
    let mut element =
        unsafe { LogMelSpectrogram::<MELS, 1, FEATURE_BYTES>::with_ring(SETTINGS, &ring) }.unwrap();
    let input_ring: StaticLendRing<1, { CLIP_SAMPLES * 2 }> = StaticLendRing::new();
    let short = vec![0u8; (CLIP_SAMPLES - 1) * 2];
    let result = block_on(element.process(frame_of(&input_ring, &short, 0, 0)));
    assert_eq!(result.unwrap_err(), G2gError::CapsMismatch);

    for settings in [
        LogMelSettings {
            window_samples: FFT_SIZE + 1,
            ..SETTINGS
        },
        LogMelSettings {
            highest_frequency: 8_001.0,
            ..SETTINGS
        },
        LogMelSettings {
            frames: 0,
            ..SETTINGS
        },
    ] {
        // SAFETY: the ring outlives the element.
        let result =
            unsafe { LogMelSpectrogram::<MELS, 1, FEATURE_BYTES>::with_ring(settings, &ring) };
        assert_eq!(result.unwrap_err(), G2gError::CapsMismatch);
    }
}

#[test]
fn int8_quantize_rounds_half_away_from_zero_and_saturates() {
    const SCALE: f32 = 0.25;
    const ZERO_POINT: i32 = 5;
    let input = TensorFormat {
        dtype: TensorDType::F32,
        shape: TensorShape::new([1, 6]),
        layout: TensorLayout::Nhwc,
    };
    let values = [0.0f32, 0.125, -0.125, 0.87, 1e9, -1e9];
    let expected = [5i8, 6, 4, 8, 127, -128];

    let ring: StaticLendRing<1, 6> = StaticLendRing::new();
    // SAFETY: the rings outlive every frame in this test.
    let mut quantizer =
        unsafe { Int8Quantize::with_ring(input, SCALE, ZERO_POINT, &ring) }.unwrap();
    assert_eq!(
        quantizer.output_format(),
        TensorFormat {
            dtype: TensorDType::I8,
            ..input
        }
    );
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let input_ring: StaticLendRing<1, 24> = StaticLendRing::new();
    let out = block_on(quantizer.process(frame_of(&input_ring, &bytes, 0, 0)))
        .unwrap()
        .unwrap();
    assert_eq!(payload(&out), expected.map(|q| q as u8));
    drop(out);

    let result = block_on(quantizer.process(frame_of(&input_ring, &bytes[..20], 0, 0)));
    assert_eq!(result.unwrap_err(), G2gError::CapsMismatch);

    // SAFETY: the ring outlives the element.
    let zero_scale = unsafe { Int8Quantize::with_ring(input, 0.0, ZERO_POINT, &ring) };
    assert_eq!(zero_scale.unwrap_err(), G2gError::CapsMismatch);
    // SAFETY: the ring outlives the element.
    let wide_zero_point = unsafe { Int8Quantize::with_ring(input, SCALE, 200, &ring) };
    assert_eq!(wide_zero_point.unwrap_err(), G2gError::CapsMismatch);
}
