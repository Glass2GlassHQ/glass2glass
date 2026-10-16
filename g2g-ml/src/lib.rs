//! ML inference elements for `glass2glass`.
//!
//! Experimental (Tier 3 in `STABILITY.md`): no stability promise. Device
//! execution providers and GPU preprocess are host-validated, not CI-run.
//!
//! Three backends, all implementing the same `AsyncElement` contract and
//! negotiating `Caps::Tensor` on their output pads:
//! - `burn`: pure-Rust, `wgpu`-backed, for embedded / Wasm / RTOS targets.
//! - `ort`: ONNX Runtime bindings for high-performance server inference.
//! - `executorch`: PyTorch's ExecuTorch runtime, running a `.pte` program.

#![forbid(unsafe_op_in_unsafe_fn)]

// ONNX Runtime inference element. The module is `ortinfer` (not `ort`) so
// in-crate paths can't collide with the `ort` dependency crate.
#[cfg(feature = "ort")]
pub mod ortinfer;

// The inference element body shared by every model runtime, generic over the
// runtime that loads and runs the model.
#[cfg(any(feature = "ort", feature = "executorch"))]
pub mod inference;

// ExecuTorch inference element: a `.pte` program on ExecuTorch's portable CPU
// kernels, the same element as `ortinfer` on a different runtime.
#[cfg(feature = "executorch")]
pub mod executorchinfer;

// Launch-line registration of the ML elements into a g2g-core Registry, so
// `ortinfer model=... ! detectionpostprocess` parses. Behind `launch`, which
// pulls g2g-core's runtime layer.
#[cfg(feature = "launch")]
pub mod registry;
#[cfg(feature = "launch")]
pub use registry::register;

// Bounded multi-stream tensor batcher (design/ml.md, M22; moved from the
// dissolved g2g-enterprise crate, M635): gathers one tensor frame per input
// stream into a single batched frame for dynamic-batch inference.
#[cfg(feature = "std")]
pub mod batcher;

// tensor post-processing (softmax / argmax classification head); pure Rust,
// no feature gate.
pub mod postprocess;

pub mod tensor_bytes;

// The dims checks a model's declared input and output must pass, shared by the
// model runtimes.
#[cfg(any(feature = "ort", feature = "executorch"))]
mod model_dims;

// Dependency-free safetensors weight reader/writer (M262): import trained
// weights at runtime into the hand-rolled inference elements without serde or
// the safetensors crate. Architecture stays compiled; only weights load.
pub mod safetensors;

// Detection post-processing (design/ml.md): decode a YOLO-style model output
// tensor into AnalyticsMeta bounding-box detections (confidence threshold + NMS),
// attached to the frame. Gated behind `analytics` (pulls g2g-core's `metadata`).
#[cfg(feature = "analytics")]
pub mod detect;

// Instance-segmentation decode (design/ml.md): a YOLO -seg model's box and
// mask-prototype outputs -> AnalyticsMeta Segmentation + Roi nodes. Pure Rust,
// no inference engine, so a browser ort-web caller can decode with it too.
#[cfg(feature = "analytics")]
pub mod segmentation;

// The in-graph instance-segmentation producer (M992): runs a YOLO -seg ONNX and
// attaches the decoded masks / ROIs to the frame it forwards. Needs both the ORT
// session and the metadata graph.
#[cfg(all(feature = "ort", feature = "analytics"))]
pub mod ortsegment;

// Inline GPU tensor preprocessing (design/ml.md): NV12 -> normalized f32
// NCHW RGB tensor in a wgpu compute shader, the hardware-first preprocessing
// counterpart of OrtInference's CPU path.
#[cfg(feature = "wgpu")]
pub mod wgpupreprocess;

#[cfg(feature = "wgpu")]
mod wgpu_transfer;

// GPU-resident tensor inference (design/ml.md, M216): a wgpu matmul compute
// pass that binds WgpuPreprocess's GPU-resident output tensor directly, so the
// tensor never leaves the GPU between preprocess and inference. The consumer
// half of the keep-on-GPU branch with_gpu_output (M215) opened.
#[cfg(feature = "wgpu")]
pub mod wgpuinfer;

// CudaToWgpu: bridges NVDEC CUDA NV12 to WgpuPreprocess's surface-import path.
// It lives with the Vulkan/CUDA external-memory interop it drives (and with the
// launch registry that auto-plugs it); re-exported here for its ML consumers.
// Linux + NVIDIA.
#[cfg(all(target_os = "linux", feature = "cuda-wgpu"))]
pub use g2g_plugins::cudatowgpu;

// Pure-Rust Burn inference element (design/ml.md): a linear layer run on
// burn's wgpu backend, the no-C++ counterpart of OrtInference. The module is
// `burninfer` (not `burn`) so in-crate paths can't collide with the dependency.
#[cfg(feature = "burn")]
pub mod burninfer;
