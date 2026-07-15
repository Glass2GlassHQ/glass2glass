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
//! flowing without the device). For Android edge, `from_memory_with_nnapi`
//! (NPU / GPU / DSP via NNAPI), `from_memory_with_xnnpack` (ARM-optimized CPU),
//! and `from_memory_for_android` (NNAPI then XNNPACK then CPU, the
//! delegate-with-fallback shape) are the same constructor pattern (M439). All EP
//! choices are constructor variants; the element shape never changes. TensorRT /
//! CoreML / QNN slot in the same way and remain follow-ups.

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
    /// Static output shape (a dynamic leading batch dim coerced to 1).
    out_shape: TensorShape,
    /// When set, the input pad is a preprocessed NCHW `Caps::Tensor` fed straight
    /// to the session, not RGBA normalized on the CPU.
    tensor_input: bool,
    /// The model's input element type: `F32` (the RGBA / f32-tensor path) or `U8`
    /// / `I8` (a quantized model, fed an integer tensor straight through, M442 the
    /// `TensorConvert::quantize` output). RGBA mode requires `F32`.
    input_dtype: TensorDType,
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

    /// As [`from_memory`], with the NNAPI execution provider (Android's system
    /// NeuralNetworks API: NPU / GPU / DSP) registered ahead of the CPU fallback.
    /// Best-effort like the CUDA / DirectML paths: with no usable NNAPI
    /// accelerator the session runs on the CPU, so the pipeline keeps flowing.
    /// Android only, the `nnapi` feature links the Android ORT's NNAPI symbol.
    #[cfg(feature = "nnapi")]
    pub fn from_memory_with_nnapi(model_bytes: &[u8]) -> Result<Self, G2gError> {
        let builder = Session::builder().map_err(ort_err)?;
        let mut builder = builder
            .with_execution_providers([::ort::ep::NNAPI::default().build()])
            .map_err(ort_err)?;
        let session = builder.commit_from_memory(model_bytes).map_err(ort_err)?;
        Self::from_session(session)
    }

    /// As [`from_memory`], with the XNNPACK execution provider (ARM-optimized
    /// CPU) registered ahead of ONNX Runtime's default CPU EP. The CPU-side
    /// companion to [`from_memory_with_nnapi`] on Android. Android only, the
    /// `xnnpack` feature links the Android ORT's XNNPACK symbol.
    #[cfg(feature = "xnnpack")]
    pub fn from_memory_with_xnnpack(model_bytes: &[u8]) -> Result<Self, G2gError> {
        let builder = Session::builder().map_err(ort_err)?;
        let mut builder = builder
            .with_execution_providers([::ort::ep::XNNPACK::default().build()])
            .map_err(ort_err)?;
        let session = builder.commit_from_memory(model_bytes).map_err(ort_err)?;
        Self::from_session(session)
    }

    /// As [`from_memory`], with the QNN execution provider (Qualcomm AI Engine
    /// Direct: the Hexagon NPU / Adreno GPU on Snapdragon) ahead of the CPU
    /// fallback. Best-effort like the CUDA / NNAPI paths: with no usable QNN
    /// backend the session runs on the CPU, so the pipeline keeps flowing. The
    /// `qnn` feature links the QNN symbol the Qualcomm ONNX Runtime build carries
    /// (a host ORT lacks it, like `nnapi`), so treat it as a Snapdragon-target
    /// feature, never enabled in a host build / CI. QNN is Qualcomm's own NPU
    /// stack, the alternative to reaching the Hexagon through NNAPI.
    #[cfg(feature = "qnn")]
    pub fn from_memory_with_qnn(model_bytes: &[u8]) -> Result<Self, G2gError> {
        let builder = Session::builder().map_err(ort_err)?;
        let mut builder = builder
            .with_execution_providers([::ort::ep::QNN::default().build()])
            .map_err(ort_err)?;
        let session = builder.commit_from_memory(model_bytes).map_err(ort_err)?;
        Self::from_session(session)
    }

    /// As [`from_memory`], with the CoreML execution provider (the Apple Neural
    /// Engine / GPU on macOS / iOS) ahead of the CPU fallback. Best-effort like the
    /// CUDA / NNAPI paths: with no usable CoreML device the session runs on the
    /// CPU. The `coreml` feature links the CoreML symbol the Apple ONNX Runtime
    /// build carries (a non-Apple ORT lacks it, like `nnapi`), so treat it as an
    /// Apple-target feature, never enabled in a host (non-Apple) build / CI. The
    /// macOS / iOS sibling of NNAPI / QNN.
    #[cfg(feature = "coreml")]
    pub fn from_memory_with_coreml(model_bytes: &[u8]) -> Result<Self, G2gError> {
        let builder = Session::builder().map_err(ort_err)?;
        let mut builder = builder
            .with_execution_providers([::ort::ep::CoreML::default().build()])
            .map_err(ort_err)?;
        let session = builder.commit_from_memory(model_bytes).map_err(ort_err)?;
        Self::from_session(session)
    }

    /// As [`from_memory`], the turnkey Android edge path: NNAPI (accelerator)
    /// preferred, then XNNPACK (ARM CPU), then ONNX Runtime's default CPU EP. ORT
    /// assigns each node to the first listed provider that supports it, so a model
    /// the accelerator cannot fully run still executes (its unsupported nodes drop
    /// to XNNPACK / CPU), the MediaPipe delegate-with-fallback shape in one call.
    #[cfg(all(feature = "nnapi", feature = "xnnpack"))]
    pub fn from_memory_for_android(model_bytes: &[u8]) -> Result<Self, G2gError> {
        let builder = Session::builder().map_err(ort_err)?;
        let mut builder = builder
            .with_execution_providers([
                ::ort::ep::NNAPI::default().build(),
                ::ort::ep::XNNPACK::default().build(),
            ])
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
        // The input may be f32 (RGBA / f32-tensor path) or quantized u8 / i8 (a
        // quantized model fed an integer tensor). The output is always f32 (a
        // quantized model dequantizes before its output).
        let (input_name, (in_dims, input_dtype)) =
            (input.name().to_owned(), input_tensor_dims(input.dtype())?);
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
            input_dtype,
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
        self.out_shape.dims()
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
                // The model's own input dtype: f32, or u8 / i8 for a quantized
                // model fed an already-quantized tensor (M442).
                dtype: self.input_dtype,
                shape: TensorShape::new([1, 3, self.height, self.width]),
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
            shape: self.out_shape,
            // rank-4 outputs are channel-first by the input convention;
            // for other ranks the layout tag is nominal.
            layout: TensorLayout::Nchw,
        }
    }

    /// RGBA8 -> normalized f32 NCHW RGB, then run the session.
    fn infer(&mut self, rgba: &[u8]) -> Result<(Box<[u8]>, Vec<u32>), G2gError> {
        let (w, h) = (self.width as usize, self.height as usize);
        // Geometry comes from the model's declared input dims; fold with checked
        // ops so absurd dimensions fail loud instead of overflowing the length
        // guard or over-allocating.
        let plane = w.checked_mul(h).ok_or(G2gError::CapsMismatch)?;
        let needed = plane.checked_mul(4).ok_or(G2gError::CapsMismatch)?;
        if rgba.len() < needed {
            return Err(G2gError::CapsMismatch);
        }
        let chw_len = plane.checked_mul(3).ok_or(G2gError::CapsMismatch)?;
        let mut chw = vec![0f32; chw_len];
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
        let (w, h) = (self.width as usize, self.height as usize);
        let plane = w.checked_mul(h).ok_or(G2gError::CapsMismatch)?;
        let n = plane.checked_mul(3).ok_or(G2gError::CapsMismatch)?;
        let nbytes = n.checked_mul(4).ok_or(G2gError::CapsMismatch)?;
        if bytes.len() < nbytes {
            return Err(G2gError::CapsMismatch);
        }
        let chw: Vec<f32> = bytes[..nbytes]
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

    /// Feed an already-quantized u8 / i8 NCHW `[1, 3, H, W]` tensor straight to the
    /// session (the quantized-model tensor-input path, M442): one byte per element,
    /// no normalize. The bytes are the integer tensor values (i8 reinterpreted from
    /// the byte). Returns the (f32) output, like [`run_chw`](Self::run_chw).
    fn infer_tensor_int(&mut self, bytes: &[u8]) -> Result<(Box<[u8]>, Vec<u32>), G2gError> {
        let (w, h) = (self.width as usize, self.height as usize);
        let plane = w.checked_mul(h).ok_or(G2gError::CapsMismatch)?;
        let n = plane.checked_mul(3).ok_or(G2gError::CapsMismatch)?;
        if bytes.len() < n {
            return Err(G2gError::CapsMismatch);
        }
        let (iw, ih) = (self.width as i64, self.height as i64);
        let shape = vec![1i64, 3, ih, iw];
        let outputs = match self.input_dtype {
            TensorDType::U8 => {
                let t = Tensor::from_array((shape, bytes[..n].to_vec())).map_err(ort_err)?;
                self.session.run(::ort::inputs![self.input_name.as_str() => t]).map_err(ort_err)?
            }
            TensorDType::I8 => {
                let data: Vec<i8> = bytes[..n].iter().map(|&b| b as i8).collect();
                let t = Tensor::from_array((shape, data)).map_err(ort_err)?;
                self.session.run(::ort::inputs![self.input_name.as_str() => t]).map_err(ort_err)?
            }
            _ => return Err(G2gError::CapsMismatch),
        };
        let value = outputs.get(self.output_name.as_str()).ok_or(G2gError::Hardware(HardwareError::Other))?;
        let (out_shape, out_data) = value.try_extract_tensor::<f32>().map_err(ort_err)?;
        let dims: Vec<u32> = out_shape.iter().map(|d| (*d).max(0) as u32).collect();
        let mut out_bytes = Vec::with_capacity(out_data.len() * 4);
        for v in out_data {
            out_bytes.extend_from_slice(&v.to_le_bytes());
        }
        Ok((out_bytes.into_boxed_slice(), dims))
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
        // RGBA mode normalizes to f32, so it only feeds an f32 model; a quantized
        // (u8 / i8) model must take a pre-quantized tensor via `with_tensor_input`.
        if !self.tensor_input && self.input_dtype != TensorDType::F32 {
            return Err(G2gError::CapsMismatch);
        }
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
                        if self.input_dtype == TensorDType::F32 {
                            self.infer_tensor(slice.as_slice())?
                        } else {
                            self.infer_tensor_int(slice.as_slice())?
                        }
                    } else {
                        self.infer(slice.as_slice())?
                    };
                    let new_caps = Caps::Tensor {
                        dtype: TensorDType::F32,
                        shape: TensorShape::from_slice(&dims).ok_or(G2gError::CapsMismatch)?,
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
                        meta: Default::default(),
                    };
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(tensor_frame)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    // Runner contract (the videoconvert / videoscale convention):
                    // the transform arm calls `configure_pipeline` (input) then
                    // `configure_output` (output) and pushes this packet carrying
                    // the pre-fixed forward *output* caps, NOT a new input. So `c`
                    // is our output tensor (the model's), not something to validate
                    // against `supported_input` (a real model's output never
                    // matches its input, so the old `c.intersect(supported_input)?`
                    // hard-errored whenever ORT was an interior transform). Forward
                    // it so a strict downstream sees the tensor caps before the
                    // first frame, and record `last_caps` to suppress the data
                    // path's duplicate emit.
                    self.last_caps = Some(c.clone());
                    out.push(PipelinePacket::CapsChanged(c)).await?;
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
                // PipelinePacket is non_exhaustive: forward variants added
                // since (a metadata-carrying packet, etc.) unchanged.
                other => {
                    out.push(other).await?;
                }
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

/// Dims and dtype of an input tensor: f32 (RGBA / f32-tensor path) or a quantized
/// u8 / i8 (a quantized model's integer input, M442). Other value types reject.
fn input_tensor_dims(dtype: &ValueType) -> Result<(Vec<i64>, TensorDType), G2gError> {
    match dtype {
        ValueType::Tensor { ty, shape, .. } => {
            let dt = match ty {
                TensorElementType::Float32 => TensorDType::F32,
                TensorElementType::Uint8 => TensorDType::U8,
                TensorElementType::Int8 => TensorDType::I8,
                _ => return Err(G2gError::CapsMismatch),
            };
            Ok((shape.to_vec(), dt))
        }
        _ => Err(G2gError::CapsMismatch),
    }
}

/// Resolve an output shape to static dims: a dynamic leading batch dim is
/// the exported-with-dynamic-batch convention and becomes 1; any other
/// dynamic dim is rejected (v1 needs static output caps for negotiation).
fn static_output_dims(dims: &[i64]) -> Result<TensorShape, G2gError> {
    let mut out = Vec::with_capacity(dims.len());
    for (i, d) in dims.iter().enumerate() {
        match d {
            -1 if i == 0 => out.push(1),
            d if *d > 0 => out.push(*d as u32),
            _ => return Err(G2gError::CapsMismatch),
        }
    }
    // A fixed-rank TensorShape carries at most MAX_TENSOR_RANK dims; a model
    // with a deeper output cannot be described by tensor caps.
    TensorShape::from_slice(&out).ok_or(G2gError::CapsMismatch)
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
        assert_eq!(static_output_dims(&[1, 10]), Ok(TensorShape::new([1, 10])));
        assert_eq!(static_output_dims(&[-1, 10]), Ok(TensorShape::new([1, 10])));
        assert_eq!(
            static_output_dims(&[1, -1]),
            Err(G2gError::CapsMismatch),
            "dynamic non-batch dims are rejected"
        );
        assert_eq!(static_output_dims(&[1, 0]), Err(G2gError::CapsMismatch));
    }
}
