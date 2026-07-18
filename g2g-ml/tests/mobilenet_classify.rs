//! M446: a real (non-toy) quantized vision model end to end through the g2g
//! element graph. Where the M440-M444 probes proved the on-NPU *plumbing* (a
//! [1,3,4,4] toy conv on the Edge TPU), this proves *utility*: the validated
//! int8 MobileNetV2 (ImageNet, [1,3,224,224] -> [1,1000]) classifies a real
//! image through `OrtInference -> TensorPostprocess::argmax`, and its top-1
//! matches the ONNX Runtime reference exactly, so the g2g chain is correct
//! independent of whether that class is the "true" label.
//!
//! The 3.6 MB model is not committed (repo fixtures are KB-scale); it is fetched
//! on demand by `tools/mobilenet-fixture.sh` into a gitignored dir, the same
//! "validated locally, not CI" pattern as the GPU / Android probes. The test
//! skips (does not fail) when the fixtures are absent.
//!
//! Run:
//!   tools/mobilenet-fixture.sh
//!   cargo test -p g2g-ml --features ort --test mobilenet_classify -- --nocapture

#![cfg(feature = "ort")]

use std::path::PathBuf;

use g2g_core::element::{AsyncElement, BoxFuture, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{Caps, G2gError, TensorDType, TensorLayout, TensorShape};
use g2g_ml::ortinfer::OrtInference;
use g2g_ml::postprocess::TensorPostprocess;

const SIZE: u32 = 224;
const CLASSES: usize = 1000;
/// The ONNX Runtime CPU top-1 for the committed input (see `expected.txt` /
/// `fetch.py`): class 332 = "Angora". The g2g chain must reproduce it exactly.
const EXPECTED_IDX: usize = 332;

/// Captures the first `DataFrame`'s system bytes (ignores `CapsChanged`), the
/// same one-shot sink the Android probes use to drive an element by hand.
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
                    if let MemoryDomain::System(s) = &f.domain {
                        self.bytes = Some(s.as_slice().to_vec());
                    }
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mobilenet")
}

fn frame(bytes: Vec<u8>) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    })
}

fn tensor_caps(dtype: TensorDType, shape: &[u32]) -> Caps {
    Caps::Tensor {
        dtype,
        shape: TensorShape::from_slice(shape).unwrap(),
        layout: TensorLayout::Nchw,
    }
}

#[tokio::test]
async fn mobilenetv2_int8_classifies_a_real_image() {
    let dir = fixture_dir();
    let model_path = dir.join("model.onnx");
    let input_path = dir.join("input_f32.bin");
    if !model_path.exists() || !input_path.exists() {
        eprintln!(
            "mobilenet fixtures absent ({}); run tools/mobilenet-fixture.sh first. skipping.",
            dir.display()
        );
        return;
    }

    let model = std::fs::read(&model_path).expect("read model");
    let input = std::fs::read(&input_path).expect("read input");
    assert_eq!(
        input.len(),
        3 * SIZE as usize * SIZE as usize * 4,
        "f32 NCHW [1,3,224,224]"
    );

    // Stage 1: the real quantized MobileNetV2, f32 tensor in -> [1,1000] logits.
    let mut infer = OrtInference::from_memory(&model)
        .expect("model loads")
        .with_tensor_input();
    infer
        .configure_pipeline(&tensor_caps(TensorDType::F32, &[1, 3, SIZE, SIZE]))
        .expect("configure inference");
    let mut logits = OneFrame::default();
    infer
        .process(frame(input), &mut logits)
        .await
        .expect("inference runs");
    let logit_bytes = logits.bytes.expect("logits tensor");
    assert_eq!(logit_bytes.len(), CLASSES * 4, "1000 f32 logits");

    // Stage 2: the classification head, logits -> winning [index, value].
    let mut argmax = TensorPostprocess::argmax();
    argmax
        .configure_pipeline(&tensor_caps(TensorDType::F32, &[1, CLASSES as u32]))
        .expect("configure argmax");
    let mut top = OneFrame::default();
    argmax
        .process(frame(logit_bytes), &mut top)
        .await
        .expect("argmax runs");
    let top_bytes = top.bytes.expect("argmax output");
    let idx = f32::from_le_bytes([top_bytes[0], top_bytes[1], top_bytes[2], top_bytes[3]]) as usize;

    let label = std::fs::read_to_string(dir.join("expected.txt")).unwrap_or_default();
    eprintln!(">> MobileNetV2-int8 top-1 = class {idx} ({})", label.trim());
    assert_eq!(
        idx, EXPECTED_IDX,
        "g2g argmax must match the ONNX Runtime reference top-1"
    );
}
