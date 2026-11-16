// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

mod util;

use g2g_core::error::G2gError;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::{StaticTransform, TensorDType, TensorLayout, TensorShape};
use g2g_mcu::{Int8Quantize, PixelMapping, TensorFormat};
use util::{block_on, frame_of, payload};

// person_detect.tflite's input quantization.
const SCALE: f32 = 1.0 / 127.5;
const ZERO_POINT: i32 = -1;

fn pixels_format(dtype: TensorDType) -> TensorFormat {
    TensorFormat {
        dtype,
        shape: TensorShape::new([1, 2, 2, 1]),
        layout: TensorLayout::Nhwc,
    }
}

#[test]
fn int8_range_pixels_are_shifted_by_128() {
    let pixels = [0u8, 127, 128, 255];
    let expected = [-128i8, -1, 0, 127].map(|q| q as u8);
    let ring: StaticLendRing<1, 4> = StaticLendRing::new();
    let input_ring: StaticLendRing<1, 4> = StaticLendRing::new();
    // SAFETY: the rings outlive every frame in this test.
    let mut quantizer = unsafe {
        Int8Quantize::with_ring(pixels_format(TensorDType::U8), SCALE, ZERO_POINT, &ring)
    }
    .unwrap()
    .with_pixel_mapping(PixelMapping::Int8Range)
    .unwrap();
    let out = block_on(quantizer.process(frame_of(&input_ring, &pixels, 0, 0)))
        .unwrap()
        .unwrap();
    assert_eq!(payload(&out), expected);
}

#[test]
fn pixel_mapping_is_refused_for_float_input() {
    let ring: StaticLendRing<1, 4> = StaticLendRing::new();
    // SAFETY: no frame is published.
    let quantizer = unsafe {
        Int8Quantize::with_ring(pixels_format(TensorDType::F32), SCALE, ZERO_POINT, &ring)
    }
    .unwrap();
    assert_eq!(
        quantizer
            .with_pixel_mapping(PixelMapping::Int8Range)
            .unwrap_err(),
        G2gError::CapsMismatch
    );
}
