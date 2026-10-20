mod util;

use g2g_core::error::{G2gError, HardwareError};
use g2g_core::staticpool::StaticLendRing;
use g2g_core::{Caps, StaticTransform, TensorDType, TensorLayout, TensorShape};
use g2g_mcu::{TensorFormat, TensorInference, TensorModel};
use util::{block_on, frame_of, payload};

const INPUT: TensorFormat = TensorFormat {
    dtype: TensorDType::I8,
    shape: TensorShape::new([1, 4]),
    layout: TensorLayout::Nhwc,
};
const OUTPUT: TensorFormat = TensorFormat {
    dtype: TensorDType::F32,
    shape: TensorShape::new([1, 2]),
    layout: TensorLayout::Nhwc,
};
const OUTPUT_BYTES: usize = 8;

// Scores are the sum and the maximum of the int8 inputs.
#[derive(Default)]
struct SumAndMaximum {
    runs: usize,
    fail: bool,
}

impl TensorModel for &mut SumAndMaximum {
    fn input(&self) -> TensorFormat {
        INPUT
    }

    fn output(&self) -> TensorFormat {
        OUTPUT
    }

    async fn run(&mut self, input: &[u8], output: &mut [u8]) -> Result<(), G2gError> {
        if self.fail {
            return Err(G2gError::Hardware(HardwareError::Peripheral));
        }
        self.runs += 1;
        let values: Vec<i32> = input.iter().map(|&b| b as i8 as i32).collect();
        let sum = values.iter().sum::<i32>() as f32;
        let maximum = *values.iter().max().unwrap() as f32;
        output[..4].copy_from_slice(&sum.to_le_bytes());
        output[4..].copy_from_slice(&maximum.to_le_bytes());
        Ok(())
    }
}

fn scores(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn runs_the_model_into_a_ring_slot_and_keeps_timing() {
    let mut model = SumAndMaximum::default();
    let input_ring: StaticLendRing<1, 4> = StaticLendRing::new();
    let output_ring: StaticLendRing<1, OUTPUT_BYTES> = StaticLendRing::new();
    // SAFETY: the rings outlive every frame in this test.
    let mut element = unsafe { TensorInference::with_ring(&mut model, &output_ring) };
    assert_eq!(element.input_caps(), INPUT.caps());
    assert_eq!(
        element.output_caps(),
        Caps::Tensor {
            dtype: TensorDType::F32,
            shape: TensorShape::new([1, 2]),
            layout: TensorLayout::Nhwc,
        }
    );

    let input = [3i8, -7, 100, -128].map(|v| v as u8);
    let out = block_on(element.process(frame_of(&input_ring, &input, 1_000, 9)))
        .unwrap()
        .unwrap();
    assert_eq!(scores(payload(&out)), vec![-32.0, 100.0]);
    assert_eq!(out.timing.pts_ns, 1_000);
    assert_eq!(out.sequence, 9);
    drop(out);
    assert_eq!(model.runs, 1);
}

#[test]
fn rejects_a_tensor_of_the_wrong_size_before_running_the_model() {
    let mut model = SumAndMaximum::default();
    let input_ring: StaticLendRing<1, 8> = StaticLendRing::new();
    let output_ring: StaticLendRing<1, OUTPUT_BYTES> = StaticLendRing::new();
    // SAFETY: the rings outlive every frame in this test.
    let mut element = unsafe { TensorInference::with_ring(&mut model, &output_ring) };
    let result = block_on(element.process(frame_of(&input_ring, &[0; 5], 0, 0)));
    assert_eq!(result.unwrap_err(), G2gError::CapsMismatch);
    assert_eq!(model.runs, 0);
}

#[test]
fn an_output_larger_than_a_ring_slot_is_a_sizing_error() {
    let mut model = SumAndMaximum::default();
    let input_ring: StaticLendRing<1, 4> = StaticLendRing::new();
    let small_ring: StaticLendRing<1, { OUTPUT_BYTES - 1 }> = StaticLendRing::new();
    // SAFETY: the rings outlive every frame in this test.
    let mut element = unsafe { TensorInference::with_ring(&mut model, &small_ring) };
    let result = block_on(element.process(frame_of(&input_ring, &[0; 4], 0, 0)));
    assert_eq!(result.unwrap_err(), G2gError::CapsMismatch);
}

#[test]
fn a_backend_fault_propagates_and_a_held_slot_is_back_pressure() {
    let mut model = SumAndMaximum {
        fail: true,
        ..SumAndMaximum::default()
    };
    let input_ring: StaticLendRing<2, 4> = StaticLendRing::new();
    let output_ring: StaticLendRing<1, OUTPUT_BYTES> = StaticLendRing::new();
    // SAFETY: the rings outlive every frame in this test.
    let mut element = unsafe { TensorInference::with_ring(&mut model, &output_ring) };
    let result = block_on(element.process(frame_of(&input_ring, &[0; 4], 0, 0)));
    assert_eq!(
        result.unwrap_err(),
        G2gError::Hardware(HardwareError::Peripheral)
    );

    model.fail = false;
    // SAFETY: the rings outlive every frame in this test.
    let mut element = unsafe { TensorInference::with_ring(&mut model, &output_ring) };
    let held = block_on(element.process(frame_of(&input_ring, &[0; 4], 0, 0)))
        .unwrap()
        .unwrap();
    let result = block_on(element.process(frame_of(&input_ring, &[0; 4], 0, 1)));
    assert_eq!(result.unwrap_err(), G2gError::PoolExhausted);
    drop(held);
}
