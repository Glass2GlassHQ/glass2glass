mod util;

use core::ffi::c_void;

use g2g_core::error::{G2gError, HardwareError};
use g2g_core::staticpool::StaticLendRing;
use g2g_core::{StaticTransform, TensorDType, TensorLayout, TensorShape};
use g2g_mcu::{CTensorModel, TensorFormat, TensorInference, TensorModel};
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
const INPUT_BYTES: usize = 4;
const OUTPUT_BYTES: usize = 8;

// Same scores as the `SumAndMaximum` double in m1700_tensor_inference, restated
// because test binaries cannot share it without editing that file.
fn sum_and_maximum(input: &[u8], output: &mut [u8]) {
    let values: Vec<i32> = input.iter().map(|&b| b as i8 as i32).collect();
    let sum = values.iter().sum::<i32>() as f32;
    let maximum = *values.iter().max().unwrap() as f32;
    output[..4].copy_from_slice(&sum.to_le_bytes());
    output[4..].copy_from_slice(&maximum.to_le_bytes());
}

struct RustModel;

impl TensorModel for RustModel {
    fn input(&self) -> TensorFormat {
        INPUT
    }

    fn output(&self) -> TensorFormat {
        OUTPUT
    }

    async fn run(&mut self, input: &[u8], output: &mut [u8]) -> Result<(), G2gError> {
        sum_and_maximum(input, output);
        Ok(())
    }
}

#[repr(C)]
#[derive(Default)]
struct CInferCtx {
    fail: bool,
    runs: usize,
    input_len: usize,
    output_len: usize,
}

unsafe extern "C" fn c_infer(
    ctx: *mut c_void,
    input: *const u8,
    input_len: usize,
    output: *mut u8,
    output_len: usize,
) -> i32 {
    // SAFETY: `ctx` is the `CInferCtx` the test passed to `CTensorModel::new`.
    let ctx = unsafe { &mut *(ctx as *mut CInferCtx) };
    ctx.runs += 1;
    ctx.input_len = input_len;
    ctx.output_len = output_len;
    if ctx.fail {
        return -1;
    }
    // SAFETY: the InferFn contract: `input` is valid for `input_len` bytes and
    // `output` for `output_len` bytes, and the two do not overlap.
    let (input, output) = unsafe {
        (
            core::slice::from_raw_parts(input, input_len),
            core::slice::from_raw_parts_mut(output, output_len),
        )
    };
    sum_and_maximum(input, output);
    0
}

fn c_model(ctx: &mut CInferCtx) -> CTensorModel {
    // SAFETY: `c_infer` is a valid function; `ctx` outlives the model.
    unsafe { CTensorModel::new(c_infer, ctx as *mut _ as *mut c_void, INPUT, OUTPUT) }
}

#[test]
fn c_seam_model_matches_the_rust_model_byte_for_byte() {
    let mut ctx = CInferCtx::default();
    let input_ring: StaticLendRing<2, INPUT_BYTES> = StaticLendRing::new();
    let c_ring: StaticLendRing<1, OUTPUT_BYTES> = StaticLendRing::new();
    let rust_ring: StaticLendRing<1, OUTPUT_BYTES> = StaticLendRing::new();
    // SAFETY: the rings outlive every frame in this test.
    let mut c_element = unsafe { TensorInference::with_ring(c_model(&mut ctx), &c_ring) };
    // SAFETY: the rings outlive every frame in this test.
    let mut rust_element = unsafe { TensorInference::with_ring(RustModel, &rust_ring) };
    assert_eq!(c_element.input_format(), INPUT);
    assert_eq!(c_element.output_format(), OUTPUT);

    let inputs = [[3i8, -7, 100, -128], [1, 2, 3, 4], [-1, -2, -3, -4]];
    for (sequence, input) in inputs.iter().enumerate() {
        let input = input.map(|v| v as u8);
        let sequence = sequence as u64;
        let c_out = block_on(c_element.process(frame_of(&input_ring, &input, 0, sequence)))
            .unwrap()
            .unwrap();
        let rust_out = block_on(rust_element.process(frame_of(&input_ring, &input, 0, sequence)))
            .unwrap()
            .unwrap();
        assert_eq!(payload(&c_out), payload(&rust_out), "input {input:?}");
        assert_eq!(c_out.sequence, sequence);
    }
    assert_eq!(ctx.runs, inputs.len());
    assert_eq!(ctx.input_len, INPUT.byte_len());
    assert_eq!(ctx.output_len, OUTPUT.byte_len());
}

#[test]
fn a_negative_callback_return_is_a_peripheral_fault() {
    let mut ctx = CInferCtx {
        fail: true,
        ..CInferCtx::default()
    };
    let input_ring: StaticLendRing<1, INPUT_BYTES> = StaticLendRing::new();
    let output_ring: StaticLendRing<1, OUTPUT_BYTES> = StaticLendRing::new();
    // SAFETY: the rings outlive every frame in this test.
    let mut element = unsafe { TensorInference::with_ring(c_model(&mut ctx), &output_ring) };
    let result = block_on(element.process(frame_of(&input_ring, &[0; INPUT_BYTES], 0, 0)));
    assert_eq!(
        result.unwrap_err(),
        G2gError::Hardware(HardwareError::Peripheral)
    );
    assert_eq!(ctx.runs, 1);
    assert_eq!(ctx.input_len, INPUT.byte_len());
    assert_eq!(ctx.output_len, OUTPUT.byte_len());
}
