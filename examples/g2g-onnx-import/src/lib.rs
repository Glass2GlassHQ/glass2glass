//! ONNX topology import for g2g's Burn backend.
//!
//! `build.rs` runs `burn-onnx`'s `ModelGen` over `model/tiny_classifier.onnx`,
//! which emits a burn `Module` (topology) plus a burnpack weight blob embedded
//! in the binary. [`TinyClassifier`] puts that generated `Model` behind g2g's
//! [`BurnModule`] trait, so `BurnInference::module` runs it frame by frame like
//! its built-in linear layer.
//!
//! The imported graph is `Conv2d -> BatchNorm -> ReLU -> global average pool ->
//! linear`. Attention topologies are not validated through this path yet.

use burn::backend::Wgpu;
use burn::tensor::Tensor;
use g2g_core::G2gError;
use g2g_ml::burninfer::{BurnInference, BurnModule};

/// Rust generated from the ONNX file at build time. `Model::default()` loads the
/// embedded weights onto the default wgpu device.
pub mod tiny_classifier {
    include!(concat!(env!("OUT_DIR"), "/model/tiny_classifier.rs"));
}

/// Input geometry and class count baked into the ONNX file.
pub const WIDTH: u32 = 4;
/// See [`WIDTH`].
pub const HEIGHT: u32 = 4;
/// See [`WIDTH`].
pub const NUM_CLASSES: usize = 2;

/// The generated model behind g2g's element-facing trait.
#[derive(Debug)]
pub struct TinyClassifier(tiny_classifier::Model<Wgpu>);

impl TinyClassifier {
    /// Build the model with its embedded trained weights.
    pub fn new() -> Self {
        Self(tiny_classifier::Model::default())
    }
}

impl Default for TinyClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl BurnModule for TinyClassifier {
    fn forward(&self, input: Tensor<Wgpu, 4>) -> Tensor<Wgpu, 2> {
        self.0.forward(input)
    }
}

/// A `BurnInference` element running the imported classifier.
pub fn inference_element() -> Result<BurnInference, G2gError> {
    BurnInference::module(WIDTH, HEIGHT, NUM_CLASSES, Box::new(TinyClassifier::new()))
}
