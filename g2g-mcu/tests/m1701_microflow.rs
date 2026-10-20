#![cfg(feature = "microflow")]

mod util;

use g2g_core::staticpool::StaticLendRing;
use g2g_core::{Caps, StaticTransform, TensorDType, TensorLayout, TensorShape};
use g2g_mcu::{MicroFlowModel, TensorInference};
use microflow::model;
use util::{block_on, frame_of, payload};

// rustc runs from the workspace root, and the macro reads the path as given.
#[model("g2g-mcu/tests/fixtures/keyword/speech.tflite")]
struct Speech;

const FEATURES: usize = 1960;
const KEYWORDS: usize = 4;
const QUANTIZATION: &str = include_str!("fixtures/keyword/quantization.txt");

// MicroFlow's own speech test: every input 0.5 gives these scores.
const UPSTREAM_INPUT: f32 = 0.5;
const UPSTREAM_SCORES: [f32; KEYWORDS] = [0.15625, 0.2734375, 0.2734375, 0.296875];

fn quantization(name: &str) -> f32 {
    QUANTIZATION
        .lines()
        .find_map(|line| line.strip_prefix(name)?.trim().parse().ok())
        .unwrap_or_else(|| panic!("{name} missing from quantization.txt"))
}

#[test]
fn speech_model_through_tensor_inference_matches_microflow() {
    let ring: StaticLendRing<1, { KEYWORDS * 4 }> = StaticLendRing::new();
    // SAFETY: the rings outlive every frame in this test.
    let mut element = unsafe {
        TensorInference::with_ring(MicroFlowModel::new(Speech::predict_quantized), &ring)
    };
    assert_eq!(
        element.input_caps(),
        Caps::Tensor {
            dtype: TensorDType::I8,
            shape: TensorShape::new([1, FEATURES as u32]),
            layout: TensorLayout::Nhwc,
        }
    );
    assert_eq!(
        element.output_caps(),
        Caps::Tensor {
            dtype: TensorDType::F32,
            shape: TensorShape::new([1, KEYWORDS as u32]),
            layout: TensorLayout::Nhwc,
        }
    );

    let quantized =
        (UPSTREAM_INPUT / quantization("input_scale")).round() + quantization("input_zero_point");
    let input = vec![quantized as i8 as u8; FEATURES];
    let input_ring: StaticLendRing<1, FEATURES> = StaticLendRing::new();
    let out = block_on(element.process(frame_of(&input_ring, &input, 0, 0)))
        .unwrap()
        .unwrap();
    let scores: Vec<f32> = payload(&out)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(scores, UPSTREAM_SCORES);
}
