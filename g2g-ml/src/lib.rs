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

// tensor post-processing (softmax / argmax classification head); pure Rust,
// no feature gate.
pub mod postprocess;

// Inline GPU tensor preprocessing (DESIGN.md §5.1): NV12 -> normalized f32
// NCHW RGB tensor in a wgpu compute shader, the hardware-first preprocessing
// counterpart of OrtInference's CPU path.
#[cfg(feature = "wgpu")]
pub mod wgpupreprocess;

// Pure-Rust Burn inference element (DESIGN.md §5.2): a linear layer run on
// burn's wgpu backend, the no-C++ counterpart of OrtInference. The module is
// `burninfer` (not `burn`) so in-crate paths can't collide with the dependency.
#[cfg(feature = "burn")]
pub mod burninfer;
