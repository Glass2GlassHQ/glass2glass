//! ONNX Runtime inference element (`ort` backend, design/ml.md).
//!
//! `OrtInference` is the shared [`Inference`] element on [`OrtBackend`]: the
//! element body is backend-agnostic, and this module loads and runs an ONNX
//! session under it.
//!
//! `OrtInference` is an `AsyncElement` transform that negotiates
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
//! The model can also be loaded after construction, through the `model` property
//! (a file path) or [`load_model`](Inference::load_model), which is what lets
//! a launch line say `ortinfer model=yolov8n.onnx tensor-input=true`. Until a
//! model is loaded, negotiation and `process` fail with `NotConfigured`.
//!
//! v1 model contract (checked at construction, fails loud):
//! - exactly one input and one output, both f32 tensors
//! - input is rank-4 `[N, 3, H, W]` with `N` 1 (or dynamic, treated as 1)
//!   and static `H`/`W`: the element then accepts RGBA exactly at `W x H`
//! - output dims static (a dynamic leading batch dim is treated as 1)
//!
//! Execution providers: `from_memory` uses ONNX Runtime's default (CPU) EP;
//! `from_memory_with_directml` and `from_memory_with_cuda` register
//! a GPU EP ahead of the CPU fallback (best-effort, so the pipeline keeps
//! flowing without the device). For Android edge, `from_memory_with_nnapi`
//! (NPU / GPU / DSP via NNAPI), `from_memory_with_xnnpack` (ARM-optimized CPU),
//! and `from_memory_for_android` (NNAPI then XNNPACK then CPU, the
//! delegate-with-fallback shape) are the same constructor pattern. All EP
//! choices are constructor variants; the element shape never changes. TensorRT,
//! OpenVINO, ROCm and MIGraphX have the same constructors behind their features.
//! A launch line names the provider instead (`execution-provider=cuda`), and
//! that path fails the load loud when the provider cannot be enabled.

use crate::inference::{
    Inference, InferenceBackend, LoadedModel, ModelGeometry, MODEL_PROP, ONLY_ON_PROP,
    TENSOR_INPUT_PROP,
};
#[cfg(feature = "analytics")]
use crate::inference::{ATTACH_TENSOR_PROP, TENSOR_NAME_PROP};
use crate::model_dims::{input_geometry, static_output_dims};
use crate::tensor_bytes::f32_le_bytes;

use ::ort::session::builder::SessionBuilder;
use ::ort::session::Session;
use ::ort::value::{Tensor, TensorElementType, ValueType};

use g2g_core::{
    G2gError, HardwareError, PropError, PropKind, PropValue, PropertySpec, TensorDType,
};

/// A loaded session plus the geometry read off it, the state every graph method
/// needs. Separate from the element so the element exists before a model does:
/// a launch line builds `ortinfer` bare and the `model` property loads this.
#[derive(Debug)]
pub struct OrtModel {
    session: Session,
    input_name: String,
    output_name: String,
    geometry: ModelGeometry,
}

/// Which execution provider a text-constructed element runs the session on.
/// The `from_memory_with_*` constructors are the equivalent for code, and differ
/// in intent: those register best-effort and fall back to the CPU, where naming
/// one here is a request that fails loud if it cannot be honoured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionProvider {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "tensorrt")]
    TensorRt,
    #[cfg(feature = "openvino")]
    OpenVino,
    #[cfg(feature = "rocm")]
    Rocm,
    #[cfg(feature = "migraphx")]
    MiGraphX,
}

impl ExecutionProvider {
    fn parse(name: &str) -> Option<Self> {
        match name {
            "cpu" => Some(Self::Cpu),
            #[cfg(feature = "cuda")]
            "cuda" => Some(Self::Cuda),
            #[cfg(feature = "tensorrt")]
            "tensorrt" => Some(Self::TensorRt),
            #[cfg(feature = "openvino")]
            "openvino" => Some(Self::OpenVino),
            #[cfg(feature = "rocm")]
            "rocm" => Some(Self::Rocm),
            #[cfg(feature = "migraphx")]
            "migraphx" => Some(Self::MiGraphX),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            #[cfg(feature = "cuda")]
            Self::Cuda => "cuda",
            #[cfg(feature = "tensorrt")]
            Self::TensorRt => "tensorrt",
            #[cfg(feature = "openvino")]
            Self::OpenVino => "openvino",
            #[cfg(feature = "rocm")]
            Self::Rocm => "rocm",
            #[cfg(feature = "migraphx")]
            Self::MiGraphX => "migraphx",
        }
    }

    /// Register the provider on `builder` directly: `with_execution_providers`
    /// swallows a failure and leaves the session on the CPU, an order of
    /// magnitude slower, where a named provider has to fail loud.
    #[cfg(any(
        feature = "cuda",
        feature = "tensorrt",
        feature = "openvino",
        feature = "rocm",
        feature = "migraphx"
    ))]
    fn register(self, builder: &mut SessionBuilder) -> Result<(), G2gError> {
        use ::ort::ep::ExecutionProvider as _;
        let registered = match self {
            Self::Cpu => return Ok(()),
            #[cfg(feature = "cuda")]
            Self::Cuda => ::ort::ep::CUDA::default().register(builder),
            #[cfg(feature = "tensorrt")]
            Self::TensorRt => ::ort::ep::TensorRT::default().register(builder),
            #[cfg(feature = "openvino")]
            Self::OpenVino => ::ort::ep::OpenVINO::default().register(builder),
            #[cfg(feature = "rocm")]
            Self::Rocm => ::ort::ep::ROCm::default().register(builder),
            #[cfg(feature = "migraphx")]
            Self::MiGraphX => ::ort::ep::MIGraphX::default().register(builder),
        };
        registered.map_err(|error| {
            // the caller only sees "invalid value", so name the cause here,
            // usually a missing runtime library or an ONNX Runtime built without it
            g2g_core::g2g_error!(
                g2g_core::log::Target::category("OrtInference"),
                "{} execution provider could not be enabled, check its runtime \
                 libraries and the ONNX Runtime build: {error}",
                self.name()
            );
            G2gError::Hardware(HardwareError::Other)
        })
    }

    #[cfg(not(any(
        feature = "cuda",
        feature = "tensorrt",
        feature = "openvino",
        feature = "rocm",
        feature = "migraphx"
    )))]
    fn register(self, _builder: &mut SessionBuilder) -> Result<(), G2gError> {
        Ok(())
    }
}

#[derive(Debug)]
pub struct OrtBackend {
    /// Execution provider registered when the model loads.
    provider: ExecutionProvider,
}

impl Default for OrtBackend {
    fn default() -> Self {
        Self {
            provider: ExecutionProvider::Cpu,
        }
    }
}

/// # Example
///
/// ```no_run
/// use g2g_ml::ortinfer::OrtInference;
///
/// // gst-launch style: ortinfer model=yolov8n.onnx
/// let element = OrtInference::from_file("yolov8n.onnx").unwrap();
/// let tensor_fed = OrtInference::from_file("yolov8n.onnx")
///     .unwrap()
///     .with_tensor_input();
/// ```
pub type OrtInference = Inference<OrtBackend>;

impl InferenceBackend for OrtBackend {
    type Model = OrtModel;

    const PROPERTIES: &'static [PropertySpec] = ORT_INFER_PROPS;

    fn load(&self, path: &str) -> Result<OrtModel, G2gError> {
        let mut builder = Session::builder().map_err(ort_err)?;
        self.provider.register(&mut builder)?;
        let session = builder.commit_from_file(path).map_err(ort_err)?;
        OrtModel::load(session)
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "execution-provider" => {
                let name = value.as_str().ok_or(PropError::Type)?;
                self.provider = ExecutionProvider::parse(name).ok_or(PropError::Value)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "execution-provider" => Some(PropValue::Str(String::from(self.provider.name()))),
            _ => None,
        }
    }
}

impl Inference<OrtBackend> {
    /// Build a session from in-memory ONNX model bytes and validate the v1
    /// model contract (see module docs).
    pub fn from_memory(model_bytes: &[u8]) -> Result<Self, G2gError> {
        let mut builder = Session::builder().map_err(ort_err)?;
        let session = builder.commit_from_memory(model_bytes).map_err(ort_err)?;
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

    /// As [`from_memory`], with the TensorRT execution provider (NVIDIA GPU
    /// through TensorRT's graph compiler) registered ahead of the CPU fallback.
    /// Best-effort like the CUDA path: without the TensorRT runtime libraries
    /// the session silently runs on the CPU.
    #[cfg(feature = "tensorrt")]
    pub fn from_memory_with_tensorrt(model_bytes: &[u8]) -> Result<Self, G2gError> {
        let builder = Session::builder().map_err(ort_err)?;
        let mut builder = builder
            .with_execution_providers([::ort::ep::TensorRT::default().build()])
            .map_err(ort_err)?;
        let session = builder.commit_from_memory(model_bytes).map_err(ort_err)?;
        Self::from_session(session)
    }

    /// As [`from_memory`], with the OpenVINO execution provider (Intel CPU, iGPU
    /// or NPU) registered ahead of the CPU fallback. Best-effort like the CUDA
    /// path: the stock ONNX Runtime download lacks OpenVINO, so without an
    /// OpenVINO-enabled build the session silently runs on the CPU.
    #[cfg(feature = "openvino")]
    pub fn from_memory_with_openvino(model_bytes: &[u8]) -> Result<Self, G2gError> {
        let builder = Session::builder().map_err(ort_err)?;
        let mut builder = builder
            .with_execution_providers([::ort::ep::OpenVINO::default().build()])
            .map_err(ort_err)?;
        let session = builder.commit_from_memory(model_bytes).map_err(ort_err)?;
        Self::from_session(session)
    }

    /// As [`from_memory`], with the ROCm execution provider (AMD GPU) registered
    /// ahead of the CPU fallback. Best-effort like the CUDA path: `ort` has no
    /// prebuilt ROCm binaries and falls back to the stock download, so without a
    /// ROCm-enabled build the session silently runs on the CPU.
    #[cfg(feature = "rocm")]
    pub fn from_memory_with_rocm(model_bytes: &[u8]) -> Result<Self, G2gError> {
        let builder = Session::builder().map_err(ort_err)?;
        let mut builder = builder
            .with_execution_providers([::ort::ep::ROCm::default().build()])
            .map_err(ort_err)?;
        let session = builder.commit_from_memory(model_bytes).map_err(ort_err)?;
        Self::from_session(session)
    }

    /// As [`from_memory`], with the MIGraphX execution provider (AMD's graph
    /// compiler on ROCm) registered ahead of the CPU fallback. Best-effort like
    /// the CUDA path: the stock ONNX Runtime download lacks MIGraphX, so without
    /// a MIGraphX-enabled build the session silently runs on the CPU.
    #[cfg(feature = "migraphx")]
    pub fn from_memory_with_migraphx(model_bytes: &[u8]) -> Result<Self, G2gError> {
        let builder = Session::builder().map_err(ort_err)?;
        let mut builder = builder
            .with_execution_providers([::ort::ep::MIGraphX::default().build()])
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
        let mut element = Self::new();
        element.model = Some(OrtModel::load(session)?);
        Ok(element)
    }
}

impl OrtModel {
    fn load(session: Session) -> Result<Self, G2gError> {
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

        let (width, height) = input_geometry(&in_dims)?;
        let output_shape = static_output_dims(&out_dims)?;

        Ok(Self {
            session,
            input_name,
            output_name,
            geometry: ModelGeometry {
                width,
                height,
                input_dtype,
                output_shape,
            },
        })
    }
}

impl LoadedModel for OrtModel {
    fn geometry(&self) -> &ModelGeometry {
        &self.geometry
    }

    /// Run the session on a `[1, 3, H, W]` f32 plane, returning the first
    /// output tensor's little-endian bytes plus its actual dims.
    fn run_f32(&mut self, chw: Vec<f32>) -> Result<(Box<[u8]>, Vec<u32>), G2gError> {
        let (w, h) = (self.geometry.width as i64, self.geometry.height as i64);
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
        Ok((f32_le_bytes(data).into_boxed_slice(), dims))
    }

    /// Feed an already-quantized u8 / i8 NCHW `[1, 3, H, W]` tensor straight to the
    /// session (the quantized-model tensor-input path): one byte per element,
    /// no normalize. The bytes are the integer tensor values (i8 reinterpreted from
    /// the byte). Returns the (f32) output, like [`run_f32`](Self::run_f32).
    fn run_int(&mut self, bytes: &[u8]) -> Result<(Box<[u8]>, Vec<u32>), G2gError> {
        let (iw, ih) = (self.geometry.width as i64, self.geometry.height as i64);
        let shape = vec![1i64, 3, ih, iw];
        let outputs = match self.geometry.input_dtype {
            TensorDType::U8 => {
                let t = Tensor::from_array((shape, bytes.to_vec())).map_err(ort_err)?;
                self.session
                    .run(::ort::inputs![self.input_name.as_str() => t])
                    .map_err(ort_err)?
            }
            TensorDType::I8 => {
                let data: Vec<i8> = bytes.iter().map(|&b| b as i8).collect();
                let t = Tensor::from_array((shape, data)).map_err(ort_err)?;
                self.session
                    .run(::ort::inputs![self.input_name.as_str() => t])
                    .map_err(ort_err)?
            }
            _ => return Err(G2gError::CapsMismatch),
        };
        let value = outputs
            .get(self.output_name.as_str())
            .ok_or(G2gError::Hardware(HardwareError::Other))?;
        let (out_shape, out_data) = value.try_extract_tensor::<f32>().map_err(ort_err)?;
        let dims: Vec<u32> = out_shape.iter().map(|d| (*d).max(0) as u32).collect();
        Ok((f32_le_bytes(out_data).into_boxed_slice(), dims))
    }
}

const PROVIDER_PROP: PropertySpec = PropertySpec::new(
    "execution-provider",
    PropKind::Str,
    "execution provider to run the session on: cpu, or one the build includes \
     (cuda, tensorrt, openvino, rocm, migraphx)",
);

#[cfg(feature = "analytics")]
static ORT_INFER_PROPS: &[PropertySpec] = &[
    MODEL_PROP,
    TENSOR_INPUT_PROP,
    PROVIDER_PROP,
    ATTACH_TENSOR_PROP,
    TENSOR_NAME_PROP,
    ONLY_ON_PROP,
];
#[cfg(not(feature = "analytics"))]
static ORT_INFER_PROPS: &[PropertySpec] =
    &[MODEL_PROP, TENSOR_INPUT_PROP, PROVIDER_PROP, ONLY_ON_PROP];

/// Dims of an f32 tensor outlet; rejects every other value type.
pub(crate) fn f32_tensor_dims(dtype: &ValueType) -> Result<Vec<i64>, G2gError> {
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
/// u8 / i8 (a quantized model's integer input). Other value types reject.
pub(crate) fn input_tensor_dims(dtype: &ValueType) -> Result<(Vec<i64>, TensorDType), G2gError> {
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

// generic over the error's payload: builder-consuming ort calls return
// `Error<SessionBuilder>` instead of the plain `Error<()>`.
pub(crate) fn ort_err<T>(_e: ::ort::Error<T>) -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}
