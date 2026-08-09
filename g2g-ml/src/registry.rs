//! Launch-line registration for the ML elements (M820).
//!
//! The stock element registry is assembled in `g2g-plugins`, which does not
//! depend on `g2g-ml`, so an app that wants the ML elements by name calls
//! [`register`] on the registry it built:
//!
//! ```ignore
//! let mut reg = g2g_plugins::registry::default_registry();
//! g2g_ml::register(&mut reg);
//! let graph = parse_launch(&reg, "... ! ortinfer model=yolov8n.onnx ! detectionpostprocess")?;
//! ```
//!
//! Behind the `launch` feature: the `Registry` and the pad templates live behind
//! g2g-core's `runtime`, which the default `g2g-ml` build does not pull.

#[cfg(any(feature = "ort", feature = "wgpu", feature = "analytics"))]
use g2g_core::runtime::LaunchFactory;
use g2g_core::runtime::Registry;

/// Register the text-constructible ML elements under their launch names:
/// `ortinfer` (`ort`), `wgpupreprocess` (`wgpu`), `detectionpostprocess`
/// (`analytics`). Each is present only when the feature that builds the element
/// is on, so this is a no-op in a build with none of them. `WgpuInference` is
/// excluded: it is constructed from weight tensors and shapes, which a text line
/// cannot express.
///
/// Returns the registry so the call can chain onto a builder expression.
pub fn register(reg: &mut Registry) -> &mut Registry {
    #[cfg(feature = "ort")]
    reg.register_launch(LaunchFactory::of::<crate::ortinfer::OrtInference>(
        "ortinfer",
        // Model-less: the `model` property loads the session (M820).
        || Box::new(crate::ortinfer::OrtInference::new()),
    ));
    #[cfg(feature = "wgpu")]
    reg.register_launch(LaunchFactory::of::<crate::wgpupreprocess::WgpuPreprocess>(
        "wgpupreprocess",
        || Box::new(crate::wgpupreprocess::WgpuPreprocess::new()),
    ));
    #[cfg(feature = "analytics")]
    reg.register_launch(LaunchFactory::of::<crate::detect::DetectionPostprocess>(
        "detectionpostprocess",
        // The YOLO defaults the in-tree detector graphs use; `conf-threshold` /
        // `iou-threshold` override them.
        || Box::new(crate::detect::DetectionPostprocess::new(0.25, 0.45)),
    ));
    #[cfg(all(feature = "ort", feature = "analytics"))]
    reg.register_launch(LaunchFactory::of::<crate::ortsegment::OrtSegmentation>(
        "ortsegment",
        // Model-less: the `model` property loads the session.
        || Box::new(crate::ortsegment::OrtSegmentation::new()),
    ));
    reg
}
