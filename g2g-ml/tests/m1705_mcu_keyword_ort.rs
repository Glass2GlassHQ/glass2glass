#![cfg(feature = "ort")]

use std::path::PathBuf;

use g2g_core::conformance::{ConformanceDimension, Evidence};
use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::MemoryDomain;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::{drive_ready, StaticTransform};
use g2g_mcu::logmel::MICRO_SPEECH_SETTINGS;
use g2g_mcu::{Int8Quantize, LogMelSpectrogram, MicroFlowModel, TensorInference};
use microflow::model;
use ort::session::Session;
use ort::value::Tensor;

// rustc runs from the workspace root, and the macro reads the path as given.
#[model("g2g-mcu/tests/fixtures/keyword/speech.tflite")]
struct Speech;

const MELS: usize = 40;
const FEATURES: usize = MICRO_SPEECH_SETTINGS.frames * MELS;
const KEYWORDS: usize = 4;
const CLIP_BYTES: usize = MICRO_SPEECH_SETTINGS.clip_samples() * 2;
// ONNX Runtime runs the QDQ twin in float between quantize steps, while MicroFlow
// and the TFLite interpreter keep int8 arithmetic, so the twin's int8 scores may
// differ by a few output steps.
const TOLERANCE_STEPS: i32 = 6;

fn fixture(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../g2g-mcu/tests/fixtures/keyword")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn quantization(name: &str) -> f64 {
    String::from_utf8(fixture("quantization.txt"))
        .unwrap()
        .lines()
        .find_map(|line| line.strip_prefix(name)?.trim().parse().ok())
        .unwrap_or_else(|| panic!("{name} missing from quantization.txt"))
}

fn lend<const BYTES: usize>(ring: &StaticLendRing<1, BYTES>, bytes: &[u8]) -> Frame {
    let mut slot = ring.acquire().expect("free slot");
    slot.buf_mut()[..bytes.len()].copy_from_slice(bytes);
    // SAFETY: every ring in this test outlives its frames.
    let slice = unsafe { slot.publish(bytes.len()) };
    Frame::new(MemoryDomain::System(slice), FrameTiming::default(), 0)
}

fn step<T: StaticTransform>(element: &mut T, frame: Frame) -> Vec<u8> {
    let out = drive_ready(element.process(frame))
        .expect("the static chain never suspends")
        .expect("element process")
        .expect("one output frame");
    out.domain.as_system_slice().unwrap().to_vec()
}

#[test]
fn keyword_graph_on_onnx_runtime_matches_the_heap_free_elements() {
    let clip_ring: StaticLendRing<1, CLIP_BYTES> = StaticLendRing::new();
    let log_mel_ring: StaticLendRing<1, { FEATURES * 4 }> = StaticLendRing::new();
    let feature_ring: StaticLendRing<1, FEATURES> = StaticLendRing::new();
    let score_ring: StaticLendRing<1, { KEYWORDS * 4 }> = StaticLendRing::new();
    let input_scale = quantization("input_scale") as f32;
    let input_zero_point = quantization("input_zero_point") as i32;
    // SAFETY: the rings outlive every frame in this test.
    let (mut log_mel, mut quantize, mut inference) = unsafe {
        let log_mel = LogMelSpectrogram::<MELS, 1, { FEATURES * 4 }>::with_ring(
            MICRO_SPEECH_SETTINGS,
            &log_mel_ring,
        )
        .unwrap();
        let quantize = Int8Quantize::with_ring(
            log_mel.output_format(),
            input_scale,
            input_zero_point,
            &feature_ring,
        )
        .unwrap();
        let inference =
            TensorInference::with_ring(MicroFlowModel::new(Speech::predict_quantized), &score_ring);
        (log_mel, quantize, inference)
    };

    // Preprocessing: the same bytes the QEMU Cortex-M4 run is checked against.
    let log_mel_bytes = step(&mut log_mel, lend(&clip_ring, &fixture("yes.s16le")));
    let features = step(&mut quantize, lend(&log_mel_ring, &log_mel_bytes));
    assert_eq!(
        features,
        fixture("features.i8"),
        "host features differ from the pinned run"
    );

    let microflow_scores = step(&mut inference, lend(&feature_ring, &features));
    assert_eq!(
        microflow_scores,
        fixture("scores.f32le"),
        "host scores differ from the pinned run"
    );

    let mut session = Session::builder()
        .unwrap()
        .commit_from_memory(&fixture("speech.onnx"))
        .unwrap();
    let input: Vec<i8> = features.iter().map(|&b| b as i8).collect();
    let tensor = Tensor::from_array((vec![1i64, FEATURES as i64], input)).unwrap();
    let outputs = session.run(ort::inputs![tensor]).unwrap();
    let (_, onnx_steps) = outputs[0].try_extract_tensor::<i8>().unwrap();

    let output_scale = quantization("output_scale");
    let output_zero_point = quantization("output_zero_point");
    let microflow_steps: Vec<i32> = microflow_scores
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&bytes| {
            (f32::from_le_bytes(bytes) as f64 / output_scale + output_zero_point).round() as i32
        })
        .collect();
    let tflite_steps: Vec<i32> = fixture("tflite_scores.i8")
        .iter()
        .map(|&q| q as i8 as i32)
        .collect();
    assert_eq!(
        microflow_steps, tflite_steps,
        "MicroFlow differs from the TFLite interpreter"
    );

    let onnx_steps: Vec<i32> = onnx_steps.iter().map(|&q| q as i32).collect();
    let worst = microflow_steps
        .iter()
        .zip(&onnx_steps)
        .map(|(a, b)| (a - b).abs())
        .max()
        .unwrap();
    assert!(
        worst <= TOLERANCE_STEPS,
        "MicroFlow {microflow_steps:?} vs ONNX Runtime {onnx_steps:?}: {worst} steps apart"
    );

    let pinned =
        "features and scores equal the pinned run tools/qemu-check.sh reproduces on Cortex-M4";
    for (peer, detail) in [
        (
            "tflite",
            format!("MicroFlow int8 scores {microflow_steps:?} equal the TFLite 2.15 interpreter's, {pinned}"),
        ),
        (
            "onnxruntime",
            format!(
                "MicroFlow int8 scores {microflow_steps:?} vs ONNX Runtime {onnx_steps:?} on the \
                 tf2onnx twin, {worst} output steps apart, {pinned}"
            ),
        ),
    ] {
        g2g_plugins::conformance::persist::record_evidence(
            "tensorinference",
            &Evidence::new(ConformanceDimension::Oracle)
                .peer(peer)
                .codec("micro_speech int8 tflite")
                .detail(detail),
        )
        .expect("persist oracle evidence");
    }
}
