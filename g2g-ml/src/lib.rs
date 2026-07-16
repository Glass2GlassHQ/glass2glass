//! ML inference elements for `glass2glass`.
//!
//! Two backends, both implementing the same `AsyncElement` contract and
//! negotiating `Caps::Tensor` on their output pads:
//! - `burn`: pure-Rust, `wgpu`-backed, for embedded / Wasm / RTOS targets.
//! - `ort`: ONNX Runtime bindings for high-performance server inference.

#![forbid(unsafe_op_in_unsafe_fn)]

// ONNX Runtime inference element. The module is `ortinfer` (not `ort`) so
// in-crate paths can't collide with the `ort` dependency crate.
#[cfg(feature = "ort")]
pub mod ortinfer;

// Bounded multi-stream tensor batcher (DESIGN.md §5.3, M22; moved from the
// dissolved g2g-enterprise crate, M635): gathers one tensor frame per input
// stream into a single batched frame for dynamic-batch inference.
#[cfg(feature = "std")]
pub mod batcher;

// tensor post-processing (softmax / argmax classification head); pure Rust,
// no feature gate.
pub mod postprocess;

// Dependency-free safetensors weight reader/writer (M262): import trained
// weights at runtime into the hand-rolled inference elements without serde or
// the safetensors crate. Architecture stays compiled; only weights load.
pub mod safetensors;

// Detection post-processing (DESIGN.md §5.3): decode a YOLO-style model output
// tensor into AnalyticsMeta bounding-box detections (confidence threshold + NMS),
// attached to the frame. Gated behind `analytics` (pulls g2g-core's `metadata`).
#[cfg(feature = "analytics")]
pub mod detect;

// Inline GPU tensor preprocessing (DESIGN.md §5.1): NV12 -> normalized f32
// NCHW RGB tensor in a wgpu compute shader, the hardware-first preprocessing
// counterpart of OrtInference's CPU path.
#[cfg(feature = "wgpu")]
pub mod wgpupreprocess;

// GPU-resident tensor inference (DESIGN.md §5.2, M216): a wgpu matmul compute
// pass that binds WgpuPreprocess's GPU-resident output tensor directly, so the
// tensor never leaves the GPU between preprocess and inference. The consumer
// half of the keep-on-GPU branch with_gpu_output (M215) opened.
#[cfg(feature = "wgpu")]
pub mod wgpuinfer;

// CudaToWgpu: bridges NVDEC CUDA NV12 to WgpuPreprocess's surface-import path
// via g2g-plugins's Vulkan/CUDA external-memory interop. Linux + NVIDIA.
#[cfg(all(target_os = "linux", feature = "cuda-wgpu"))]
pub mod cudatowgpu;

// Pure-Rust Burn inference element (DESIGN.md §5.2): a linear layer run on
// burn's wgpu backend, the no-C++ counterpart of OrtInference. The module is
// `burninfer` (not `burn`) so in-crate paths can't collide with the dependency.
#[cfg(feature = "burn")]
pub mod burninfer;
