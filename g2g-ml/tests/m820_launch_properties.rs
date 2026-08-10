//! M820: the ML elements from a launch line. Each property `parse_launch` can
//! set must (a) be declared in `properties()` so the parser knows its kind, and
//! (b) round-trip through `set_property` / `get_property` onto the field the
//! element acts on, with a bad value rejected. `ortinfer` additionally loads its
//! model from the `model` property, so it is constructible before a model exists.
//!
//! Run with:
//!
//! ```powershell
//! cargo test -p g2g-ml --features ort,analytics,launch --test m820_launch_properties
//! ```

#![cfg(feature = "launch")]

use g2g_core::{AsyncElement, PropError, PropValue, PropertySpec};

/// True when a spec table declares a property of this name (the half
/// `parse_launch` reads to determine the value kind).
fn declares(specs: &[PropertySpec], name: &str) -> bool {
    specs.iter().any(|s| s.name == name)
}

// shared hand-encoded ONNX fixture builder (tests/util/onnx_fixture.rs)
#[cfg(feature = "ort")]
mod onnx {
    include!("util/onnx_fixture.rs");
}

/// Write the identity fixture model to a temp file and return its path, so the
/// `model=` property has a real file to load (no network, no checked-in blob).
#[cfg(feature = "ort")]
fn fixture_model(name: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("g2g_m820_{name}.onnx"));
    std::fs::write(&path, onnx::identity_model(&[1, 3, 2, 2])).expect("write fixture model");
    path
}

#[cfg(feature = "ort")]
#[test]
fn ortinfer_loads_the_model_property_in_either_order() {
    use g2g_ml::ortinfer::OrtInference;

    let path = fixture_model("either_order");
    let path = path.to_str().unwrap();

    let mut e = OrtInference::new();
    assert!(declares(e.properties(), "model"));
    assert!(declares(e.properties(), "tensor-input"));

    // tensor-input set *before* the model must survive the load.
    e.set_property("tensor-input", PropValue::Bool(true))
        .unwrap();
    e.set_property("model", PropValue::Str(path.into()))
        .unwrap();
    assert_eq!(e.get_property("model"), Some(PropValue::Str(path.into())));
    assert_eq!(e.get_property("tensor-input"), Some(PropValue::Bool(true)));
    assert_eq!(
        e.input_dims(),
        (2, 2),
        "geometry read from the loaded model"
    );
    assert_eq!(e.output_shape(), &[1, 3, 2, 2]);

    // and set *after* the model it still applies.
    let mut e = OrtInference::new();
    e.set_property("model", PropValue::Str(path.into()))
        .unwrap();
    assert_eq!(e.get_property("tensor-input"), Some(PropValue::Bool(false)));
    e.set_property("tensor-input", PropValue::Bool(true))
        .unwrap();
    assert_eq!(e.get_property("tensor-input"), Some(PropValue::Bool(true)));
    assert_eq!(e.input_dims(), (2, 2));
}

#[cfg(feature = "ort")]
#[test]
fn ortinfer_rejects_a_bad_model_and_a_mistyped_flag() {
    use g2g_ml::ortinfer::OrtInference;

    let mut e = OrtInference::new();
    assert_eq!(
        e.set_property("model", PropValue::Str("/nonexistent/nope.onnx".into())),
        Err(PropError::Value),
        "a path that does not load is a bad value"
    );
    assert_eq!(
        e.set_property("tensor-input", PropValue::Str("yes".into())),
        Err(PropError::Type)
    );
    assert_eq!(
        e.set_property("nosuchprop", PropValue::Bool(true)),
        Err(PropError::Unknown)
    );
}

#[cfg(feature = "ort")]
#[test]
fn ortinfer_without_a_model_fails_loud() {
    use g2g_core::{Caps, CapsConstraint, Dim, G2gError, Rate, RawVideoFormat};
    use g2g_ml::ortinfer::OrtInference;

    let rgba = Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(2),
        height: Dim::Fixed(2),
        framerate: Rate::Any,
        interlace: g2g_core::Interlace::Any,
    };
    let mut e = OrtInference::new();
    assert_eq!(e.intercept_caps(&rgba), Err(G2gError::NotConfigured));
    assert_eq!(
        e.configure_pipeline(&rgba).err(),
        Some(G2gError::NotConfigured),
        "no model means no geometry to negotiate against"
    );
    // The solver sees an empty derived set, so a graph holding it cannot fixate.
    let CapsConstraint::DerivedOutput(derive) = e.caps_constraint_as_transform() else {
        panic!("ortinfer derives its output caps");
    };
    assert!(derive(&rgba).alternatives().is_empty());
}

#[cfg(feature = "analytics")]
#[test]
fn detectionpostprocess_thresholds_and_input_size_round_trip() {
    use g2g_ml::detect::DetectionPostprocess;

    let mut e = DetectionPostprocess::new(0.25, 0.45);
    for name in [
        "conf-threshold",
        "iou-threshold",
        "input-width",
        "input-height",
    ] {
        assert!(declares(e.properties(), name), "declares {name}");
    }
    // f32-exact values, so the f64 round-trip is exact.
    e.set_property("conf-threshold", PropValue::Double(0.5))
        .unwrap();
    e.set_property("iou-threshold", PropValue::Double(0.25))
        .unwrap();
    e.set_property("input-width", PropValue::Uint(416)).unwrap();
    e.set_property("input-height", PropValue::Uint(320))
        .unwrap();
    assert_eq!(
        e.get_property("conf-threshold"),
        Some(PropValue::Double(0.5))
    );
    assert_eq!(
        e.get_property("iou-threshold"),
        Some(PropValue::Double(0.25))
    );
    assert_eq!(e.get_property("input-width"), Some(PropValue::Uint(416)));
    assert_eq!(e.get_property("input-height"), Some(PropValue::Uint(320)));

    // Scores are in [0, 1] and the input size divides the box coordinates.
    assert_eq!(
        e.set_property("conf-threshold", PropValue::Double(1.5)),
        Err(PropError::Value)
    );
    assert_eq!(
        e.set_property("iou-threshold", PropValue::Double(-0.1)),
        Err(PropError::Value)
    );
    assert_eq!(
        e.set_property("input-width", PropValue::Uint(0)),
        Err(PropError::Value)
    );
    assert_eq!(
        e.set_property("input-height", PropValue::Str("640".into())),
        Err(PropError::Type)
    );
}

/// The threshold property must reach the decode, not just the getter: a box
/// scoring 0.4 is emitted under conf-threshold 0.3 and dropped under 0.5.
#[cfg(feature = "analytics")]
#[tokio::test]
async fn conf_threshold_property_changes_what_is_detected() {
    use g2g_core::element::{OutputSink, PushOutcome};
    use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
    use g2g_core::memory::{MemoryDomain, SystemSlice};
    use g2g_core::{AnalyticsMeta, Caps, G2gError, TensorDType, TensorLayout, TensorShape};
    use g2g_ml::detect::DetectionPostprocess;

    #[derive(Default)]
    struct MetaSink {
        detections: usize,
    }
    impl OutputSink for MetaSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");
            core::task::Poll::Ready({
                if let PipelinePacket::DataFrame(frame) = packet {
                    if let Some(a) = frame.meta.get::<AnalyticsMeta>() {
                        self.detections = a.detections().count();
                    }
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }

    // One anchor, one class: cx, cy, w, h, then the class score 0.4.
    let values: [f32; 5] = [100.0, 100.0, 40.0, 40.0, 0.4];
    let mut bytes = Vec::new();
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let caps = Caps::Tensor {
        dtype: TensorDType::F32,
        shape: TensorShape::new([1, 5, 1]),
        layout: TensorLayout::Nchw,
    };

    let mut counts = Vec::new();
    for threshold in [0.3, 0.5] {
        let mut e = DetectionPostprocess::new(0.25, 0.45);
        e.set_property("conf-threshold", PropValue::Double(threshold))
            .unwrap();
        e.configure_pipeline(&caps).unwrap();
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.clone().into_boxed_slice())),
            timing: FrameTiming::default(),
            sequence: 0,
            meta: Default::default(),
        };
        let mut sink = MetaSink::default();
        e.process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();
        counts.push(sink.detections);
    }
    assert_eq!(
        counts,
        vec![1, 0],
        "the threshold property gates the decode"
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpupreprocess_gpu_output_round_trips() {
    use g2g_ml::wgpupreprocess::WgpuPreprocess;

    // Construction is GPU-free (the device is built lazily on the first frame),
    // so this runs on a host without an adapter.
    let mut e = WgpuPreprocess::new();
    assert!(declares(e.properties(), "gpu-output"));
    assert_eq!(e.get_property("gpu-output"), Some(PropValue::Bool(false)));
    e.set_property("gpu-output", PropValue::Bool(true)).unwrap();
    assert_eq!(e.get_property("gpu-output"), Some(PropValue::Bool(true)));
    assert_eq!(
        e.set_property("gpu-output", PropValue::Uint(1)),
        Err(PropError::Type)
    );
}

#[test]
fn register_adds_the_ml_elements_by_name() {
    use g2g_core::runtime::Registry;

    let mut reg = Registry::new();
    g2g_ml::register(&mut reg);

    #[cfg(feature = "ort")]
    {
        assert!(reg.inspect("ortinfer").is_some());
        let e = reg.make_element("ortinfer").expect("ortinfer builds bare");
        assert_eq!(e.get_property("model"), Some(PropValue::Str(String::new())));
    }
    #[cfg(feature = "wgpu")]
    assert!(reg.inspect("wgpupreprocess").is_some());
    #[cfg(feature = "analytics")]
    {
        let e = reg
            .make_element("detectionpostprocess")
            .expect("detectionpostprocess builds");
        // The YOLO defaults the in-tree detector graphs use (0.45 is not
        // f32-exact, hence the tolerance).
        assert_eq!(
            e.get_property("conf-threshold"),
            Some(PropValue::Double(0.25))
        );
        let Some(PropValue::Double(iou)) = e.get_property("iou-threshold") else {
            panic!("iou-threshold reads back");
        };
        assert!((iou - 0.45).abs() < 1e-6);
    }
    #[cfg(all(feature = "ort", feature = "analytics"))]
    {
        let e = reg
            .make_element("ortsegment")
            .expect("ortsegment builds bare");
        assert_eq!(e.get_property("model"), Some(PropValue::Str(String::new())));
        let Some(PropValue::Double(mask)) = e.get_property("mask-threshold") else {
            panic!("mask-threshold reads back");
        };
        assert!((mask - 0.5).abs() < 1e-6);
    }
    // WgpuInference is not text-constructible (it takes weight tensors).
    assert!(reg.inspect("wgpuinference").is_none());
}

/// The DESIGN_TODO target line, end to end: parse it against a registry the ML
/// elements were registered on, and read the properties back off the built graph.
#[cfg(all(feature = "ort", feature = "analytics"))]
#[test]
fn target_launch_line_parses_with_properties_applied() {
    use g2g_core::runtime::{parse_launch, GraphNodeRef};
    use g2g_core::NodeId;
    use g2g_plugins::registry::default_registry;

    let path = fixture_model("launch_line");
    let path = path.to_str().unwrap();

    let mut reg = default_registry();
    g2g_ml::register(&mut reg);

    let line = format!(
        "videotestsrc num-buffers=1 ! ortinfer model={path} tensor-input=true \
         ! detectionpostprocess conf-threshold=0.3 ! fakesink"
    );
    let graph = parse_launch(&reg, &line).expect("the target line parses");

    let mut seen = 0;
    for i in 0..graph.node_count() {
        let Some(GraphNodeRef::Element(e)) = graph.element(NodeId(i as u32)) else {
            continue;
        };
        if e.get_property("model") == Some(PropValue::Str(path.into())) {
            assert_eq!(e.get_property("tensor-input"), Some(PropValue::Bool(true)));
            seen += 1;
        }
        if let Some(PropValue::Double(conf)) = e.get_property("conf-threshold") {
            assert!(
                (conf - 0.3).abs() < 1e-6,
                "conf-threshold reached the decoder"
            );
            seen += 1;
        }
    }
    assert_eq!(seen, 2, "both ML elements were built with their properties");
}

/// An undeclared property fails the parse, so the line cannot silently ignore it.
#[cfg(feature = "analytics")]
#[test]
fn unknown_ml_property_is_rejected_by_the_parser() {
    use g2g_core::runtime::parse_launch;
    use g2g_plugins::registry::default_registry;

    let mut reg = default_registry();
    g2g_ml::register(&mut reg);
    assert!(parse_launch(
        &reg,
        "videotestsrc num-buffers=1 ! detectionpostprocess nosuchprop=1 ! fakesink"
    )
    .is_err());
}
