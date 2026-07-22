//! M448: a real object detector end to end through the g2g element graph, the
//! detection sibling of the M446 classifier. A real YOLO (Ultralytics YOLO11 /
//! YOLOv8 share the `[1, 4+C, A]` channel-major output) runs through
//! `OrtInference -> DetectionPostprocess` and detects the dog in a real photo:
//! the model's raw `[1, 84, 8400]` tensor becomes structured `AnalyticsMeta`
//! detections (anchor decode + per-class NMS), the reusable element that closes
//! the perception-postprocessing gap vs a MediaPipe Solution.
//!
//! The model is ~tens of MB so it is not committed (repo fixtures are KB-scale);
//! `tools/detect-fixture.sh` / `fixtures/detect/gen.py` obtain it on demand into a
//! gitignored dir (HF blocks anonymous downloads here, so the model comes from an
//! `ultralytics` export or a path in `$G2G_YOLO_MODEL`) and preprocess the input.
//! The test skips when the fixtures are absent.
//!
//! Run:
//!   G2G_YOLO_MODEL=/path/to/yolo11m.onnx tools/detect-fixture.sh
//!   cargo test -p g2g-ml --features "ort analytics" --test yolo_detect -- --nocapture

#![cfg(all(feature = "ort", feature = "analytics"))]

use std::path::PathBuf;

use g2g_core::element::{AsyncElement, BoxFuture, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{AnalyticsMeta, Caps, G2gError, TensorDType, TensorLayout, TensorShape};
use g2g_ml::detect::DetectionPostprocess;
use g2g_ml::ortinfer::OrtInference;

const SIZE: u32 = 640;
/// YOLO output `[1, 4 + 80, 8400]`: 4 box channels + 80 COCO classes, 8400 anchors.
const CHANNELS: u32 = 84;
const ANCHORS: u32 = 8400;
/// COCO class index 16 = "dog"; the sample image is a Samoyed.
const COCO_DOG: u32 = 16;

#[derive(Default)]
struct OneFrame {
    bytes: Option<Vec<u8>>,
}
impl OutputSink for OneFrame {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if self.bytes.is_none() {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.bytes = Some(s.to_vec());
                    }
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

/// Records each detection (label, confidence) off the frame's `AnalyticsMeta`.
#[derive(Default)]
struct MetaSink {
    dets: Vec<(u32, f32)>,
}
impl OutputSink for MetaSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = &packet {
                if let Some(a) = f.meta.get::<AnalyticsMeta>() {
                    self.dets = a.detections().map(|d| (d.label, d.confidence)).collect();
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/detect")
}

fn frame(bytes: Vec<u8>) -> PipelinePacket {
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

#[tokio::test]
async fn yolo_detects_the_dog() {
    let dir = fixture_dir();
    let model_path = dir.join("model.onnx");
    let input_path = dir.join("input_f32.bin");
    if !model_path.exists() || !input_path.exists() {
        eprintln!(
            "detect fixtures absent ({}); run tools/detect-fixture.sh first. skipping.",
            dir.display()
        );
        return;
    }

    let model = std::fs::read(&model_path).expect("read model");
    let input = std::fs::read(&input_path).expect("read input");
    assert_eq!(
        input.len(),
        3 * SIZE as usize * SIZE as usize * 4,
        "f32 NCHW [1,3,640,640]"
    );

    // Stage 1: the real YOLO, image tensor in -> [1,84,8400] raw detections.
    let mut infer = OrtInference::from_memory(&model)
        .expect("model loads")
        .with_tensor_input();
    infer
        .configure_pipeline(&tensor_caps(&[1, 3, SIZE, SIZE]))
        .expect("configure inference");
    let mut raw = OneFrame::default();
    infer
        .process(frame(input), &mut raw)
        .await
        .expect("inference runs");
    let raw_bytes = raw.bytes.expect("raw detection tensor");
    assert_eq!(
        raw_bytes.len(),
        (CHANNELS * ANCHORS) as usize * 4,
        "[1,84,8400] f32"
    );

    // Stage 2: decode + per-class NMS -> structured detections on the frame.
    let mut decode = DetectionPostprocess::new(0.25, 0.45).with_input_size(SIZE, SIZE);
    decode
        .configure_pipeline(&tensor_caps(&[1, CHANNELS, ANCHORS]))
        .expect("configure decode");
    let mut sink = MetaSink::default();
    decode
        .process(frame(raw_bytes), &mut sink)
        .await
        .expect("decode runs");

    eprintln!(">> YOLO detections after NMS: {:?}", sink.dets);
    assert!(!sink.dets.is_empty(), "expected at least one detection");
    // NMS must have collapsed the ~10 overlapping raw dog boxes to a handful.
    assert!(
        sink.dets.len() <= 10,
        "NMS should suppress overlaps, got {}",
        sink.dets.len()
    );
    let dog = sink.dets.iter().find(|(label, _)| *label == COCO_DOG);
    let (_, conf) = dog.expect("a 'dog' (COCO class 16) detection");
    assert!(
        *conf > 0.5,
        "dog detected with confidence > 0.5, got {conf}"
    );
    eprintln!(">> detected COCO class {COCO_DOG} (dog) at confidence {conf:.3}");
}
