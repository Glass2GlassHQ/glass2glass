use g2g_core::error::G2gError;
use g2g_core::frame::Frame;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::tensor::quantize;
use g2g_core::{StaticTransform, TensorDType};

use crate::inference::TensorFormat;
use crate::lend::lend_converted;

pub struct Int8Quantize<'r, const N: usize, const BYTES: usize> {
    input: TensorFormat,
    scale: f32,
    zero_point: i32,
    ring: &'r StaticLendRing<N, BYTES>,
}

impl<const N: usize, const BYTES: usize> Int8Quantize<'static, N, BYTES> {
    pub fn new(
        input: TensorFormat,
        scale: f32,
        zero_point: i32,
        ring: &'static StaticLendRing<N, BYTES>,
    ) -> Result<Self, G2gError> {
        // SAFETY: `'static` trivially satisfies with_ring's outlives contract.
        unsafe { Self::with_ring(input, scale, zero_point, ring) }
    }
}

impl<'r, const N: usize, const BYTES: usize> Int8Quantize<'r, N, BYTES> {
    /// # Safety
    /// The ring must outlive every frame this element publishes.
    pub unsafe fn with_ring(
        input: TensorFormat,
        scale: f32,
        zero_point: i32,
        ring: &'r StaticLendRing<N, BYTES>,
    ) -> Result<Self, G2gError> {
        let valid = input.dtype == TensorDType::F32
            && scale.is_finite()
            && scale > 0.0
            && (i8::MIN as i32..=i8::MAX as i32).contains(&zero_point);
        if !valid {
            return Err(G2gError::CapsMismatch);
        }
        Ok(Self {
            input,
            scale,
            zero_point,
            ring,
        })
    }

    pub fn output_format(&self) -> TensorFormat {
        TensorFormat {
            dtype: TensorDType::I8,
            ..self.input
        }
    }
}

impl<const N: usize, const BYTES: usize> StaticTransform for Int8Quantize<'_, N, BYTES> {
    async fn process(&mut self, input: Frame) -> Result<Option<Frame>, G2gError> {
        let Some(tensor) = input.domain.as_system_slice() else {
            return Err(G2gError::UnsupportedDomain);
        };
        if tensor.len() != self.input.byte_len() {
            return Err(G2gError::CapsMismatch);
        }
        let (scale, zero_point) = (self.scale, self.zero_point);
        // SAFETY: the constructor established the ring-outlives-frames
        // contract (`new`: 'static, `with_ring`: the caller's contract).
        unsafe {
            lend_converted(
                self.ring,
                &input,
                self.input.shape.elements(),
                |tensor, out| {
                    for (quantized, bytes) in out.iter_mut().zip(tensor.chunks_exact(4)) {
                        let &[b0, b1, b2, b3] = bytes else {
                            continue;
                        };
                        let value = f32::from_le_bytes([b0, b1, b2, b3]);
                        let clamped = quantize(value, scale, zero_point)
                            .clamp(i8::MIN as i32, i8::MAX as i32);
                        *quantized = clamped as i8 as u8;
                    }
                },
            )
        }
        .map(Some)
    }
}

impl<const N: usize, const BYTES: usize> core::fmt::Debug for Int8Quantize<'_, N, BYTES> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Int8Quantize")
            .field("input", &self.input)
            .field("scale", &self.scale)
            .field("zero_point", &self.zero_point)
            .finish_non_exhaustive()
    }
}
