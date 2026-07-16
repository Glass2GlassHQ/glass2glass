//! M321: element-property forwarding to the hosted Python instance.
//!
//! `PyTransform` forwards its declared element properties (e.g. `model-name`,
//! `device`, `batch-size`) onto the Python object at construction, mapping gst
//! names to Python attributes (`model-name` -> `model_name`) and PropValue
//! scalars to Python scalars. The `PropEcho` fixture reads those attributes back
//! out as a blob / detection, and this asserts they arrived (and kept their
//! type). Needs libpython + the `metadata`-enabled core, so the file compiles
//! away without the `analytics` feature.
#![cfg(feature = "analytics")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::memory::SystemSlice;
use g2g_core::{
    AnalyticsMeta, AsyncElement, BlobMeta, Caps, Dim, Frame, FrameTiming, G2gError, MemoryDomain,
    OutputSink, PipelinePacket, PropValue, PushOutcome, Rate, RawVideoFormat,
};
use g2g_python::PyTransform;

#[derive(Default)]
struct CollectSink {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for CollectSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        self.packets.push(packet);
        Box::pin(async { Ok(PushOutcome::Accepted) })
    }
}

fn frame_2x1_rgba() -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(vec![0u8; 8].into_boxed_slice())),
        timing: FrameTiming {
            pts_ns: 0,
            dts_ns: 0,
            duration_ns: 0,
            capture_ns: 0,
            arrival_ns: 0,
            keyframe: false,
        },
        sequence: 0,
        meta: Default::default(),
    }
}

#[test]
fn element_properties_reach_the_python_instance() {
    std::env::set_var(
        "PYTHONPATH",
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"),
    );

    let mut el = PyTransform::new("echo_element", "PropEcho");
    // Set declared element properties (the gst-launch face). These are forwarded
    // to the Python instance; the str ones as Python str, the int as Python int.
    el.set_property("model-name", PropValue::Str("yolo11m.onnx".into())).unwrap();
    el.set_property("device", PropValue::Str("cuda:0".into())).unwrap();
    el.set_property("batch-size", PropValue::Int(4)).unwrap();

    let caps = Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(2),
        height: Dim::Fixed(1),
        framerate: Rate::Fixed(30),
    };
    el.configure_pipeline(&caps).unwrap();

    let mut sink = CollectSink::default();
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    rt.block_on(el.process(PipelinePacket::DataFrame(frame_2x1_rgba()), &mut sink))
        .unwrap();

    let PipelinePacket::DataFrame(frame) = &sink.packets[0] else {
        panic!("expected a DataFrame downstream");
    };

    // The int property kept its type: it was usable directly as the (integer)
    // detection label, which would have raised had it arrived as a string.
    let analytics = frame
        .meta
        .get::<AnalyticsMeta>()
        .expect("PropEcho should attach a detection labelled by batch-size");
    let dets: Vec<_> = analytics.detections().collect();
    assert_eq!(dets.len(), 1);
    assert_eq!(dets[0].label, 4, "batch-size=4 forwarded as an int and used as the label");

    // The string properties reached self.model_name / self.device verbatim.
    let blobs = frame.meta.get::<BlobMeta>().expect("PropEcho should attach blobs");
    let by_header = |h: &str| {
        blobs
            .iter()
            .find(|b| b.header == h)
            .map(|b| b.payload.clone())
            .unwrap_or_default()
    };
    assert_eq!(by_header("model_name"), b"yolo11m.onnx", "model-name -> self.model_name");
    assert_eq!(by_header("device"), b"cuda:0", "device -> self.device");
}
