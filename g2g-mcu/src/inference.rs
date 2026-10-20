use g2g_core::error::G2gError;
use g2g_core::frame::Frame;
use g2g_core::memory::MemoryDomain;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::{Caps, StaticTransform, TensorDType, TensorLayout, TensorShape};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorFormat {
    pub dtype: TensorDType,
    pub shape: TensorShape,
    pub layout: TensorLayout,
}

impl TensorFormat {
    pub fn caps(self) -> Caps {
        Caps::Tensor {
            dtype: self.dtype,
            shape: self.shape,
            layout: self.layout,
        }
    }

    pub fn byte_len(self) -> usize {
        self.shape.elements().saturating_mul(self.dtype.size())
    }
}

#[allow(async_fn_in_trait)]
pub trait TensorModel {
    fn input(&self) -> TensorFormat;
    fn output(&self) -> TensorFormat;
    // `input` and `output` are exactly `byte_len` of their formats.
    async fn run(&mut self, input: &[u8], output: &mut [u8]) -> Result<(), G2gError>;
}

pub struct TensorInference<'r, M, const N: usize, const BYTES: usize> {
    model: M,
    ring: &'r StaticLendRing<N, BYTES>,
}

impl<M: TensorModel, const N: usize, const BYTES: usize> TensorInference<'static, M, N, BYTES> {
    pub fn new(model: M, ring: &'static StaticLendRing<N, BYTES>) -> Self {
        // SAFETY: `'static` trivially satisfies with_ring's outlives contract.
        unsafe { Self::with_ring(model, ring) }
    }
}

impl<'r, M: TensorModel, const N: usize, const BYTES: usize> TensorInference<'r, M, N, BYTES> {
    /// # Safety
    /// The ring must outlive every frame this element publishes.
    pub unsafe fn with_ring(model: M, ring: &'r StaticLendRing<N, BYTES>) -> Self {
        Self { model, ring }
    }

    pub fn input_format(&self) -> TensorFormat {
        self.model.input()
    }

    pub fn output_format(&self) -> TensorFormat {
        self.model.output()
    }
}

impl<M: TensorModel, const N: usize, const BYTES: usize> StaticTransform
    for TensorInference<'_, M, N, BYTES>
{
    async fn process(&mut self, input: Frame) -> Result<Option<Frame>, G2gError> {
        let Some(tensor) = input.domain.as_system_slice() else {
            return Err(G2gError::UnsupportedDomain);
        };
        if tensor.len() != self.model.input().byte_len() {
            return Err(G2gError::CapsMismatch);
        }
        let output_len = self.model.output().byte_len();
        let Some(mut slot) = self.ring.acquire() else {
            return Err(G2gError::PoolExhausted);
        };
        let Some(output) = slot.buf_mut().get_mut(..output_len) else {
            return Err(G2gError::CapsMismatch);
        };
        self.model.run(tensor, output).await?;
        // SAFETY: the constructor established the ring-outlives-frames
        // contract (`new`: 'static, `with_ring`: the caller's contract).
        let out = unsafe { slot.publish(output_len) };
        Ok(Some(Frame::new(
            MemoryDomain::System(out),
            input.timing,
            input.sequence,
        )))
    }
}

impl<M, const N: usize, const BYTES: usize> core::fmt::Debug for TensorInference<'_, M, N, BYTES> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TensorInference").finish_non_exhaustive()
    }
}
