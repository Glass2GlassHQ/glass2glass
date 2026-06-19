//! ONNX Runtime inference element (`ort` backend, DESIGN.md §5).
//!
//! M21: `OrtInference` is an `AsyncElement` transform that negotiates
//! `Caps::RawVideo` on its input pad and `Caps::Tensor` on its output pad,
//! exactly like any other element in the graph. Each RGBA frame is converted
//! to a normalized f32 NCHW tensor (RGB, `value / 255`), run through the
//! session, and the model's first output is emitted as a `DataFrame` whose
//! bytes are the output tensor's f32 little-endian values, under
//! `Caps::Tensor { F32, shape, Nchw }`.
//!
//! `with_tensor_input()` switches the input pad to `Caps::Tensor` (an
//! already-normalized f32 NCHW `[1, 3, H, W]`, e.g. from `WgpuPreprocess` /
//! `WebGPUPreprocess`), fed straight to the session with no CPU normalize.
//!
//! v1 model contract (checked at construction, fails loud):
//! - exactly one input and one output, both f32 tensors
//! - input is rank-4 `[N, 3, H, W]` with `N` 1 (or dynamic, treated as 1)
//!   and static `H`/`W`: the element then accepts RGBA exactly at `W x H`
//! - output dims static (a dynamic leading batch dim is treated as 1)
//!
//! Execution providers: `from_memory` uses ONNX Runtime's default (CPU) EP;
//! `from_memory_with_directml` (M26) and `from_memory_with_cuda` (M53) register
//! a GPU EP ahead of the CPU fallback (best-effort, so the pipeline keeps
//! flowing without the device). TensorRT / CoreML are the same constructor
//! shape and remain a follow-up; the element shape doesn't change.

use core::future::Future;
use core::pin::Pin;

use ::ort::session::Session;
use ::ort::value::{Tensor, TensorElementType, ValueType};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, HardwareError,
    MemoryDomain, OutputSink, PipelinePacket, Rate, RawVideoFormat, TensorDType, TensorLayout,
    TensorShape,
};

#[derive(Debug)]
pub struct OrtInference {
    session: Session,
    input_name: String,
    output_name: String,
    /// Model input geometry (static `W x H` from the `[N, 3, H, W]` input).
    width: u32,
    height: u32,
    /// Static output dims (a dynamic leading batch dim coerced to 1).
    out_shape: Vec<u32>,
    /// When set, the input pad is a preprocessed f32 NCHW `Caps::Tensor` fed
    /// straight to the session, not RGBA normalized on the CPU.
    tensor_input: bool,
    configured: bool,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl OrtInference {
    /// Build a session from in-memory ONNX model bytes and validate the v1
    /// model contract (see module docs).
    pub fn from_memory(model_bytes: &[u8]) -> Result<Self, G2gError> {
        let mut builder = Session::builder().map_err(ort_err)?;
        let session = builder.commit_from_memory(model_bytes).map_err(ort_err)?;
        Self::from_session(session)
    }

    /// As [`from_memory`], reading the model from a file.
    pub fn from_file(path: &str) -> Result<Self, G2gError> {
        let mut builder = Session::builder().map_err(ort_err)?;
        let session = builder.commit_from_file(path).map_err(ort_err)?;
        Self::from_session(session)
    }

    /// As [`from_memory`], with the DirectML execution provider (D3D12 GPU,
    /// Windows) registered ahead of the CPU fallback. Registration is
    /// best-effort per ort's dispatch default: on a host without a usable
    /// DirectML device the session silently runs on the CPU, so the
    /// pipeline keeps flowing either way.
    #[cfg(feature = "directml")]
    pub fn from_memory_with_directml(model_bytes: &[u8]) -> Result<Self, G2gError> {
        let builder = Session::builder().map_err(ort_err)?;
        let mut builder = builder
            .with_execution_providers([::ort::ep::DirectML::default().build()])
            .map_err(ort_err)?;
        let session = builder.commit_from_memory(model_bytes).map_err(ort_err)?;
        Self::from_session(session)
    }

    /// As [`from_memory`], with the CUDA execution provider (NVIDIA GPU)
    /// registered ahead of the CPU fallback. Like the DirectML path,
    /// registration is best-effort: on a host without a usable CUDA device or
    /// runtime the session silently runs on the CPU, so the pipeline keeps
    /// flowing either way. The element shape is unchanged; the EP choice is a
    /// constructor variant.
    #[cfg(feature = "cuda")]
    pub fn from_memory_with_cuda(model_bytes: &[u8]) -> Result<Self, G2gError> {
        let builder = Session::builder().map_err(ort_err)?;
        let mut builder = builder
            .with_execution_providers([::ort::ep::CUDA::default().build()])
            .map_err(ort_err)?;
        let session = builder.commit_from_memory(model_bytes).map_err(ort_err)?;
        Self::from_session(session)
    }

    fn from_session(session: Session) -> Result<Self, G2gError> {
        let [input] = session.inputs() else {
            return Err(G2gError::CapsMismatch);
        };
        let [output] = session.outputs() else {
            return Err(G2gError::CapsMismatch);
        };
        let (input_name, in_dims) = (input.name().to_owned(), f32_tensor_dims(input.dtype())?);
        let (output_name, out_dims) = (output.name().to_owned(), f32_tensor_dims(output.dtype())?);

        // input must be [N, 3, H, W] with static H/W; N may be dynamic.
        let [n, c, h, w] = in_dims[..] else {
            return Err(G2gError::CapsMismatch);
        };
        if !(n == 1 || n == -1) || c != 3 || h <= 0 || w <= 0 {
            return Err(G2gError::CapsMismatch);
        }

        let out_shape = static_output_dims(&out_dims)?;

        Ok(Self {
            session,
            input_name,
            output_name,
            width: w as u32,
            height: h as u32,
            out_shape,
            tensor_input: false,
            configured: false,
            last_caps: None,
            emitted: 0,
        })
    }

    /// The model's expected input geometry, `(width, height)`.
    pub fn input_dims(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// The model's static output tensor dims.
    pub fn output_shape(&self) -> &[u32] {
        &self.out_shape
    }

    /// Count of tensor `DataFrame`s pushed downstream. Useful in tests.
    pub fn inferred_count(&self) -> u64 {
        self.emitted
    }

    /// Accept an already-normalized f32 NCHW `[1, 3, H, W]` tensor input
    /// (e.g. from `WgpuPreprocess` / `WebGPUPreprocess`) instead of RGBA,
    /// feeding it straight to the session with no CPU normalize. The model
    /// geometry is unchanged.
    pub fn with_tensor_input(mut self) -> Self {
        self.tensor_input = true;
        self
    }

    fn supported_input(&self) -> Caps {
        if self.tensor_input {
            Caps::Tensor {
                dtype: TensorDType::F32,
                shape: TensorShape(vec![1, 3, self.height, self.width]),
                layout: TensorLayout::Nchw,
            }
        } else {
            Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Fixed(self.width),
                height: Dim::Fixed(self.height),
                framerate: Rate::Any,
            }
        }
    }

    fn output_caps(&self) -> Caps {
        Caps::Tensor {
            dtype: TensorDType::F32,
            shape: TensorShape(self.out_shape.clone()),
            // rank-4 outputs are channel-first by the input convention;
            // for other ranks the layout tag is nominal.
            layout: TensorLayout::Nchw,
        }
    }

    /// RGBA8 -> normalized f32 NCHW RGB, then run the session.
    fn infer(&mut self, rgba: &[u8]) -> Result<(Box<[u8]>, Vec<u32>), G2gError> {
        let (w, h) = (self.width as usize, self.height as usize);
        if rgba.len() < w * h * 4 {
            return Err(G2gError::CapsMismatch);
        }
        let plane = w * h;
        let mut chw = vec![0f32; 3 * plane];
        for px in 0..plane {
            let src = px * 4;
            chw[px] = rgba[src] as f32 / 255.0;
            chw[plane + px] = rgba[src + 1] as f32 / 255.0;
            chw[2 * plane + px] = rgba[src + 2] as f32 / 255.0;
        }
        self.run_chw(chw)
    }

    /// Feed an already-normalized f32 NCHW `[1, 3, H, W]` tensor straight to
    /// the session (tensor-input mode); the bytes are the tensor's
    /// little-endian f32 values, e.g. from a GPU preprocess step.
    fn infer_tensor(&mut self, bytes: &[u8]) -> Result<(Box<[u8]>, Vec<u32>), G2gError> {
        let n = 3 * self.width as usize * self.height as usize;
        if bytes.len() < n * 4 {
            return Err(G2gError::CapsMismatch);
        }
        let chw: Vec<f32> = bytes[..n * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        self.run_chw(chw)
    }

    /// Run the session on a `[1, 3, H, W]` f32 plane, returning the first
    /// output tensor's little-endian bytes plus its actual dims.
    fn run_chw(&mut self, chw: Vec<f32>) -> Result<(Box<[u8]>, Vec<u32>), G2gError> {
        let (w, h) = (self.width as i64, self.height as i64);
        let tensor = Tensor::from_array((vec![1i64, 3, h, w], chw)).map_err(ort_err)?;
        let outputs = self
            .session
            .run(::ort::inputs![self.input_name.as_str() => tensor])
            .map_err(ort_err)?;
        let value = outputs
            .get(self.output_name.as_str())
            .ok_or(G2gError::Hardware(HardwareError::Other))?;
        let (shape, data) = value.try_extract_tensor::<f32>().map_err(ort_err)?;

        let dims: Vec<u32> = shape.iter().map(|d| (*d).max(0) as u32).collect();
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Ok((bytes.into_boxed_slice(), dims))
    }
}

impl AsyncElement for OrtInference {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&self.supported_input())
    }

    /// Native `DerivedOutput`: RGBA at the model's geometry in, the model's
    /// static tensor caps out. Non-matching input yields an empty set, so
    /// the solver rejects it at negotiation time.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let supported = self.supported_input();
        let out = self.output_caps();
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| {
            if input.intersect(&supported).is_ok() {
                CapsSet::one(out.clone())
            } else {
                CapsSet::from_alternatives(Vec::new())
            }
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        absolute_caps.intersect(&self.supported_input())?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let (bytes, dims) = if self.tensor_input {
                        self.infer_tensor(slice.as_slice())?
                    } else {
                        self.infer(slice.as_slice())?
                    };
                    let new_caps = Caps::Tensor {
                        dtype: TensorDType::F32,
                        shape: TensorShape(dims),
                        layout: TensorLayout::Nchw,
                    };
                    if self.last_caps.as_ref() != Some(&new_caps) {
                        out.push(PipelinePacket::CapsChanged(new_caps.clone())).await?;
                        self.last_caps = Some(new_caps);
                    }
                    let tensor_frame = Frame {
                        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes)),
                        // inference is per-frame: the tensor inherits the
                        // source frame's timing so latency stays traceable.
                        timing: frame.timing,
                        sequence: self.emitted,
                    };
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(tensor_frame)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    // geometry is pinned by the model; a mid-stream change
                    // to anything else is a hard error.
                    c.intersect(&self.supported_input())?;
                }
                PipelinePacket::Flush => {
                    out.push(PipelinePacket::Flush).await?;
                }
                // Segment is a timing marker: forward unchanged.
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                // stateless per-frame inference: nothing to drain.
                PipelinePacket::Eos => {}
            }
            Ok(())
        })
    }
}

/// Dims of an f32 tensor outlet; rejects every other value type.
fn f32_tensor_dims(dtype: &ValueType) -> Result<Vec<i64>, G2gError> {
    match dtype {
        ValueType::Tensor {
            ty: TensorElementType::Float32,
            shape,
            ..
        } => Ok(shape.to_vec()),
        _ => Err(G2gError::CapsMismatch),
    }
}

/// Resolve an output shape to static dims: a dynamic leading batch dim is
/// the exported-with-dynamic-batch convention and becomes 1; any other
/// dynamic dim is rejected (v1 needs static output caps for negotiation).
fn static_output_dims(dims: &[i64]) -> Result<Vec<u32>, G2gError> {
    let mut out = Vec::with_capacity(dims.len());
    for (i, d) in dims.iter().enumerate() {
        match d {
            -1 if i == 0 => out.push(1),
            d if *d > 0 => out.push(*d as u32),
            _ => return Err(G2gError::CapsMismatch),
        }
    }
    Ok(out)
}

// generic over the error's payload: builder-consuming ort calls return
// `Error<SessionBuilder>` instead of the plain `Error<()>`.
fn ort_err<T>(_e: ::ort::Error<T>) -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_output_dims_coerces_dynamic_batch_only() {
        assert_eq!(static_output_dims(&[1, 10]), Ok(vec![1, 10]));
        assert_eq!(static_output_dims(&[-1, 10]), Ok(vec![1, 10]));
        assert_eq!(
            static_output_dims(&[1, -1]),
            Err(G2gError::CapsMismatch),
            "dynamic non-batch dims are rejected"
        );
        assert_eq!(static_output_dims(&[1, 0]), Err(G2gError::CapsMismatch));
    }
}
