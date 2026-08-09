//! ONNX topology import for g2g's Burn backend.
//!
//! `build.rs` runs `burn-onnx`'s `ModelGen` over each fixture in `model/`, which
//! emits a burn `Module` (topology) plus a burnpack weight blob embedded in the
//! binary. Each generated `Model` is wrapped in g2g's [`BurnModule`] trait, so
//! `BurnInference::module` runs it frame by frame like its built-in linear layer.
//!
//! Two topologies, both taking the element's `[1, 3, H, W]` NCHW input and
//! emitting `[1, NUM_CLASSES]` logits:
//!
//! - [`TinyClassifier`]: `Conv2d -> BatchNorm -> ReLU -> global average pool -> linear`.
//! - [`TinyAttention`]: the 16 pixels as a token sequence through multi-head
//!   self-attention (the standard ONNX `Attention` op, which `burn-onnx` lowers
//!   onto `burn::tensor::module::attention`) -> mean pool -> linear.

use burn::backend::Wgpu;
use burn::tensor::Tensor;
use g2g_core::G2gError;
use g2g_ml::burninfer::{BurnInference, BurnModule};

/// Input geometry and class count baked into both ONNX files.
pub const WIDTH: u32 = 4;
/// See [`WIDTH`].
pub const HEIGHT: u32 = 4;
/// See [`WIDTH`].
pub const NUM_CLASSES: usize = 2;

/// Wrap one build-time-generated model as a [`BurnModule`] plus the
/// `BurnInference` element that drives it. Every generated model has the same
/// shape here (`Model::default()` loads the embedded weights onto the default
/// wgpu device, and `forward` is the 4D-in / 2D-out pass the trait wants), so a
/// further fixture is one more invocation rather than another copy of the block.
macro_rules! imported_onnx_model {
    ($module:ident, $wrapper:ident, $element:ident, $what:literal) => {
        /// Rust generated from the ONNX file at build time.
        pub mod $module {
            include!(concat!(
                env!("OUT_DIR"),
                "/model/",
                stringify!($module),
                ".rs"
            ));
        }

        #[doc = concat!("The generated ", $what, " behind g2g's element-facing trait.")]
        #[derive(Debug)]
        pub struct $wrapper($module::Model<Wgpu>);

        impl $wrapper {
            /// Build the model with its embedded trained weights.
            pub fn new() -> Self {
                Self($module::Model::default())
            }
        }

        impl Default for $wrapper {
            fn default() -> Self {
                Self::new()
            }
        }

        impl BurnModule for $wrapper {
            fn forward(&self, input: Tensor<Wgpu, 4>) -> Tensor<Wgpu, 2> {
                self.0.forward(input)
            }
        }

        #[doc = concat!("A `BurnInference` element running the imported ", $what, ".")]
        pub fn $element() -> Result<BurnInference, G2gError> {
            BurnInference::module(WIDTH, HEIGHT, NUM_CLASSES, Box::new($wrapper::new()))
        }
    };
}

imported_onnx_model!(
    tiny_classifier,
    TinyClassifier,
    classifier_element,
    "conv / batch-norm classifier"
);
imported_onnx_model!(
    tiny_attention,
    TinyAttention,
    attention_element,
    "multi-head self-attention model"
);
