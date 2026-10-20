mod util;

use g2g_core::error::G2gError;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::{StaticTransform, TensorDType, TensorLayout, TensorShape};
use g2g_mcu::{ImageResize, ImageSize, Int8Quantize, TensorFormat};
use util::{block_on, frame_of, payload};

const CHANNELS: u8 = 3;
const SOURCE: ImageSize = ImageSize {
    width: 7,
    height: 5,
};
const SOURCE_BYTES: usize = 7 * 5 * 3;
// Rounding to a whole level plus the 1/2048 weight steps stay under one level.
const LEVEL_TOLERANCE: f64 = 1.0;

fn image() -> Vec<u8> {
    (0..SOURCE_BYTES)
        .map(|i| ((i * 37 + 11) % 251) as u8)
        .collect()
}

// Bilinear resize with half-pixel centres in f64, from the definition.
fn reference(pixels: &[u8], target: ImageSize) -> Vec<f64> {
    let channels = CHANNELS as usize;
    let source_position = |index: u16, input: u16, output: u16| {
        let position = ((index as f64 + 0.5) * input as f64 / output as f64 - 0.5).max(0.0);
        let first = (position.floor() as usize).min(input as usize - 1);
        let second = (first + 1).min(input as usize - 1);
        let weight = if first == second {
            0.0
        } else {
            position - first as f64
        };
        (first, second, weight)
    };
    let at = |x: usize, y: usize, c: usize| {
        pixels[(y * SOURCE.width as usize + x) * channels + c] as f64
    };
    let mut out = Vec::new();
    for y in 0..target.height {
        let (y0, y1, wy) = source_position(y, SOURCE.height, target.height);
        for x in 0..target.width {
            let (x0, x1, wx) = source_position(x, SOURCE.width, target.width);
            for c in 0..channels {
                let top = at(x0, y0, c) * (1.0 - wx) + at(x1, y0, c) * wx;
                let bottom = at(x0, y1, c) * (1.0 - wx) + at(x1, y1, c) * wx;
                out.push(top * (1.0 - wy) + bottom * wy);
            }
        }
    }
    out
}

fn resize(target: ImageSize) -> (Vec<u8>, TensorFormat) {
    let ring: StaticLendRing<1, 512> = StaticLendRing::new();
    // SAFETY: the rings outlive every frame in this test.
    let mut element = unsafe { ImageResize::with_ring(CHANNELS, SOURCE, target, &ring) }.unwrap();
    let input_ring: StaticLendRing<1, SOURCE_BYTES> = StaticLendRing::new();
    let out = block_on(element.process(frame_of(&input_ring, &image(), 0, 0)))
        .unwrap()
        .unwrap();
    (payload(&out).to_vec(), element.output_format())
}

#[test]
fn bilinear_resize_matches_a_half_pixel_reference_down_and_up() {
    for target in [
        ImageSize {
            width: 3,
            height: 4,
        },
        ImageSize {
            width: 11,
            height: 9,
        },
    ] {
        let (ours, format) = resize(target);
        assert_eq!(
            format,
            TensorFormat {
                dtype: TensorDType::U8,
                shape: TensorShape::new([1, target.height as u32, target.width as u32, 3]),
                layout: TensorLayout::Nhwc,
            }
        );
        let expected = reference(&image(), target);
        assert_eq!(ours.len(), expected.len());
        for (index, (&got, want)) in ours.iter().zip(&expected).enumerate() {
            assert!(
                (got as f64 - want).abs() <= LEVEL_TOLERANCE,
                "{target:?} sample {index}: {got} vs {want}"
            );
        }
    }
}

#[test]
fn same_size_resize_is_the_identity_and_bad_input_is_rejected() {
    let (same, _) = resize(SOURCE);
    assert_eq!(same, image());

    let ring: StaticLendRing<1, 512> = StaticLendRing::new();
    // SAFETY: the rings outlive every frame in this test.
    let mut element = unsafe { ImageResize::with_ring(CHANNELS, SOURCE, SOURCE, &ring) }.unwrap();
    let input_ring: StaticLendRing<1, SOURCE_BYTES> = StaticLendRing::new();
    let result = block_on(element.process(frame_of(&input_ring, &image()[1..], 0, 0)));
    assert_eq!(result.unwrap_err(), G2gError::CapsMismatch);
    let empty = ImageSize {
        width: 0,
        height: 4,
    };
    // SAFETY: the ring outlives the element.
    let result = unsafe { ImageResize::with_ring(CHANNELS, SOURCE, empty, &ring) };
    assert_eq!(result.unwrap_err(), G2gError::CapsMismatch);
}

#[test]
fn pixels_quantize_to_int8_as_tflite_vision_models_expect() {
    // A TFLite int8 vision input with scale 1/255 and zero point -128 is the pixel minus 128.
    const SCALE: f32 = 1.0 / 255.0;
    const ZERO_POINT: i32 = -128;
    let pixels: Vec<u8> = (0..=255).collect();
    let input = TensorFormat {
        dtype: TensorDType::U8,
        shape: TensorShape::new([1, 16, 16, 1]),
        layout: TensorLayout::Nhwc,
    };
    let ring: StaticLendRing<1, 256> = StaticLendRing::new();
    // SAFETY: the rings outlive every frame in this test.
    let mut quantizer =
        unsafe { Int8Quantize::with_ring(input, SCALE, ZERO_POINT, &ring) }.unwrap();
    let input_ring: StaticLendRing<1, 256> = StaticLendRing::new();
    let out = block_on(quantizer.process(frame_of(&input_ring, &pixels, 0, 0)))
        .unwrap()
        .unwrap();
    let expected: Vec<u8> = pixels
        .iter()
        .map(|&p| (p as i32 - 128) as i8 as u8)
        .collect();
    assert_eq!(payload(&out), expected);
}
