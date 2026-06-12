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
