use g2g_core::error::G2gError;
use g2g_core::{TensorDType, TensorLayout, TensorShape};
use microflow::buffer::Buffer2D;
use nalgebra::ArrayStorage;

use crate::inference::{TensorFormat, TensorModel};

const SCORE_BYTES: usize = core::mem::size_of::<f32>();

// Wraps the `predict_quantized` function MicroFlow's `#[model]` macro generates
// for a model with one int8 `[1, INPUT]` input, scores out as dequantized f32.
pub struct MicroFlowModel<P, const INPUT: usize, const OUTPUT: usize> {
    predict_quantized: P,
}

impl<P, const INPUT: usize, const OUTPUT: usize> MicroFlowModel<P, INPUT, OUTPUT>
where
    P: FnMut(Buffer2D<i8, 1, INPUT>) -> Buffer2D<f32, 1, OUTPUT>,
{
    pub fn new(predict_quantized: P) -> Self {
        Self { predict_quantized }
    }
}

impl<P, const INPUT: usize, const OUTPUT: usize> TensorModel for MicroFlowModel<P, INPUT, OUTPUT>
where
    P: FnMut(Buffer2D<i8, 1, INPUT>) -> Buffer2D<f32, 1, OUTPUT>,
{
    fn input(&self) -> TensorFormat {
        TensorFormat {
            dtype: TensorDType::I8,
            shape: TensorShape::new([1, INPUT as u32]),
            layout: TensorLayout::Nhwc,
        }
    }

    fn output(&self) -> TensorFormat {
        TensorFormat {
            dtype: TensorDType::F32,
            shape: TensorShape::new([1, OUTPUT as u32]),
            layout: TensorLayout::Nhwc,
        }
    }

    async fn run(&mut self, input: &[u8], output: &mut [u8]) -> Result<(), G2gError> {
        // nalgebra stores a matrix column by column, and a one-row matrix has one value per column.
        let mut columns = [[0i8; 1]; INPUT];
        for (column, &byte) in columns.iter_mut().zip(input) {
            *column = [byte as i8];
        }
        let scores = (self.predict_quantized)(Buffer2D::from_data(ArrayStorage(columns)));
        for (bytes, score) in output.chunks_exact_mut(SCORE_BYTES).zip(scores.iter()) {
            bytes.copy_from_slice(&score.to_le_bytes());
        }
        Ok(())
    }
}

impl<P, const INPUT: usize, const OUTPUT: usize> core::fmt::Debug
    for MicroFlowModel<P, INPUT, OUTPUT>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MicroFlowModel")
            .field("input", &INPUT)
            .field("output", &OUTPUT)
            .finish_non_exhaustive()
    }
}
