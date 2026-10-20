// Shared by yolo_pose.rs and yolo_obb.rs (included via `include!`, not a test
// crate itself, so plain comments only). Runs a fixture built by
// tools/pose-obb-fixture.sh through OrtInference then DetectionPostprocess and
// reads Ultralytics' reference predictions for the same picture.

use std::path::PathBuf;

use g2g_core::element::{AsyncElement, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{AnalyticsMeta, Caps, G2gError, PropValue, TensorDType, TensorLayout, TensorShape};
use g2g_ml::detect::DetectionPostprocess;
use g2g_ml::ortinfer::OrtInference;

// The fixture scripts export at imgsz 640 and predict with these thresholds.
const SIZE: u32 = 640;
const ANCHORS: u32 = 8400;
const CONFIDENCE: f64 = 0.25;
const IOU: f64 = 0.45;
// Ultralytics rounds the letterbox pad and resized size to whole pixels and the
// decoder places the picture exactly, so coordinates differ by under a pixel.
const COORDINATE_TOLERANCE: f32 = 0.01;
// Scores come from the same ONNX file, but the execution provider may differ.
const SCORE_TOLERANCE: f32 = 0.01;

// expected.txt: picture width and height, then one row of numbers per detection.
struct Reference {
    picture_w: u32,
    picture_h: u32,
    rows: Vec<Vec<f32>>,
}

#[derive(Default)]
struct LastFrame {
    frame: Option<Frame>,
}

impl OutputSink for LastFrame {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        if let Some(PipelinePacket::DataFrame(frame)) = packet_slot.take() {
            self.frame = Some(frame);
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn packet(bytes: Vec<u8>) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    })
}

fn tensor_caps(shape: &[u32]) -> Caps {
    Caps::Tensor {
        dtype: TensorDType::F32,
        shape: TensorShape::from_slice(shape).unwrap(),
        layout: TensorLayout::Nchw,
    }
}

fn read_reference(text: &str) -> Reference {
    let mut lines = text.lines();
    let size: Vec<u32> = lines
        .next()
        .expect("picture size line")
        .split_whitespace()
        .map(|v| v.parse().expect("picture size"))
        .collect();
    let rows = lines
        .map(|line| {
            line.split_whitespace()
                .map(|v| v.parse().expect("reference value"))
                .collect()
        })
        .collect();
    Reference {
        picture_w: size[0],
        picture_h: size[1],
        rows,
    }
}

// Decode the fixture in `fixtures/<name>` with `head`, or None when the
// fixture has not been built.
async fn decode_fixture(
    name: &str,
    head: &str,
    channels: u32,
) -> Option<(AnalyticsMeta, Reference)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    let paths = ["model.onnx", "input_f32.bin", "expected.txt"].map(|f| dir.join(f));
    if !paths.iter().all(|p| p.exists()) {
        eprintln!(
            "{name} fixtures absent ({}); run tools/pose-obb-fixture.sh first. skipping.",
            dir.display()
        );
        return None;
    }
    let [model, input, expected] = paths.map(|p| std::fs::read(p).expect("read fixture"));
    let reference = read_reference(std::str::from_utf8(&expected).expect("utf-8 reference"));

    let mut infer = OrtInference::from_memory(&model)
        .expect("model loads")
        .with_tensor_input();
    infer
        .configure_pipeline(&tensor_caps(&[1, 3, SIZE, SIZE]))
        .expect("configure inference");
    let mut raw = LastFrame::default();
    infer
        .process(packet(input), &mut raw)
        .await
        .expect("inference runs");
    let raw = raw.frame.expect("model output frame");
    let raw_bytes = raw
        .domain
        .as_system_slice()
        .expect("system memory")
        .to_vec();
    assert_eq!(
        raw_bytes.len(),
        (channels * ANCHORS) as usize * 4,
        "[1, {channels}, {ANCHORS}] f32"
    );

    let mut decode = DetectionPostprocess::new(0.0, 0.0).with_input_size(SIZE, SIZE);
    for (property, value) in [
        ("head", PropValue::Str(head.into())),
        ("conf-threshold", PropValue::Double(CONFIDENCE)),
        ("iou-threshold", PropValue::Double(IOU)),
        ("picture-width", PropValue::Uint(reference.picture_w as u64)),
        (
            "picture-height",
            PropValue::Uint(reference.picture_h as u64),
        ),
    ] {
        decode.set_property(property, value).expect(property);
    }
    decode
        .configure_pipeline(&tensor_caps(&[1, channels, ANCHORS]))
        .expect("configure decode");
    let mut sink = LastFrame::default();
    decode
        .process(packet(raw_bytes), &mut sink)
        .await
        .expect("decode runs");
    let frame = sink.frame.expect("decoded frame");
    let analytics = frame
        .meta
        .get::<AnalyticsMeta>()
        .expect("analytics attached")
        .clone();
    Some((analytics, reference))
}

// Index of the centre in `centres` nearest to `(x, y)`.
fn nearest(centres: &[(f32, f32)], x: f32, y: f32) -> usize {
    let distance = |&(cx, cy): &(f32, f32)| (cx - x).powi(2) + (cy - y).powi(2);
    (0..centres.len())
        .min_by(|&a, &b| distance(&centres[a]).total_cmp(&distance(&centres[b])))
        .expect("at least one decoded centre")
}

fn assert_within(got: f32, want: f32, tolerance: f32, what: &str) {
    assert!(
        (got - want).abs() <= tolerance,
        "{what}: decoded {got}, Ultralytics {want}"
    );
}
