//! The ONNX-imported topologies run through `BurnInference` on the real GPU and
//! match the ONNX Runtime reference for the same input.
//!
//! `RGBA_FRAME` and the `*_LOGITS` constants are the output of the fixture
//! script, run from the repository root as:
//!
//! ```text
//! uv run --with onnx --with onnxruntime --with numpy tools/onnx-fixture.py \
//!     classifier examples/g2g-onnx-import/model/tiny_classifier.onnx
//! uv run --with onnx --with onnxruntime --with numpy tools/onnx-fixture.py \
//!     attention examples/g2g-onnx-import/model/tiny_attention.onnx
//! ```
//!
//! Skips when burn's wgpu backend finds no adapter. There is no ndarray-backend
//! variant: `BurnInference` is pinned to the wgpu backend.

use g2g_core::element::{BoxFuture, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, Caps, Dim, G2gError, OutputSink, Rate, RawVideoFormat, TensorDType, TensorLayout,
    TensorShape,
};
use g2g_ml::burninfer::{gpu_available, BurnInference};
use g2g_onnx_import::{attention_element, classifier_element, HEIGHT, NUM_CLASSES, WIDTH};

const RGBA_FRAME: [u8; 64] = [
    11, 48, 85, 122, 159, 196, 233, 19, 56, 93, 130, 167, 204, 241, 27, 64, 101, 138, 175, 212,
    249, 35, 72, 109, 146, 183, 220, 6, 43, 80, 117, 154, 191, 228, 14, 51, 88, 125, 162, 199, 236,
    22, 59, 96, 133, 170, 207, 244, 30, 67, 104, 141, 178, 215, 1, 38, 75, 112, 149, 186, 223, 9,
    46, 83,
];

/// `Conv2d -> BatchNorm -> ReLU -> global average pool -> linear`.
const CLASSIFIER_LOGITS: [f32; 2] = [4.43818, -0.9364319];

/// Multi-head self-attention over the 16 pixels as a token sequence, mean-pooled
/// into a linear head. The fixture script cross-checks these against a numpy fold
/// of the attention formula, so they are not just ORT agreeing with itself.
const ATTENTION_LOGITS: [f32; 2] = [-6.184183, 1.546977];

/// f32 GPU execution of conv / batch-norm and softmax chains, so a few ulps of
/// drift from the ONNX Runtime reference is expected.
const TOLERANCE: f32 = 1e-3;

const PTS_NS: u64 = 4242;
const SEQUENCE: u64 = 7;

/// burn builds its wgpu device lazily on first use, so keep the two model runs
/// off each other's device setup.
static GPU_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[derive(Default)]
struct Collect {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for Collect {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            self.packets.push(packet);
            Ok(PushOutcome::Accepted)
        })
    }
}

fn rgba_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
    }
}

fn frame_f32(f: &Frame) -> Vec<f32> {
    let Some(slice) = f.domain.as_system_slice() else {
        panic!("tensor frame must be System memory");
    };
    slice
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// Push `RGBA_FRAME` through the element, assert the tensor caps / timing
/// contract both imported models share, and return the logits.
async fn logits_from(mut element: BurnInference) -> Vec<f32> {
    element.configure_pipeline(&rgba_caps()).expect("configure");

    let frame = Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(
            RGBA_FRAME.to_vec().into_boxed_slice(),
        )),
        timing: FrameTiming {
            pts_ns: PTS_NS,
            dts_ns: PTS_NS,
            ..FrameTiming::default()
        },
        sequence: SEQUENCE,
        meta: Default::default(),
    };

    let mut out = Collect::default();
    element
        .process(PipelinePacket::DataFrame(frame), &mut out)
        .await
        .expect("imported model inference");

    let caps_changes: Vec<&Caps> = out
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::CapsChanged(c) => Some(c),
            _ => None,
        })
        .collect();
    let frames: Vec<&Frame> = out
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f),
            _ => None,
        })
        .collect();

    assert_eq!(caps_changes.len(), 1, "tensor caps emitted exactly once");
    assert_eq!(
        *caps_changes[0],
        Caps::Tensor {
            dtype: TensorDType::F32,
            shape: TensorShape::new([1, NUM_CLASSES as u32]),
            layout: TensorLayout::Nchw,
        }
    );
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].timing.pts_ns, PTS_NS);
    assert_eq!(element.inferred_count(), 1);

    let logits = frame_f32(frames[0]);
    assert_eq!(logits.len(), NUM_CLASSES);
    logits
}

fn assert_matches_reference(got: &[f32], expected: &[f32]) {
    for (i, (g, e)) in got.iter().zip(expected).enumerate() {
        assert!(
            (g - e).abs() < TOLERANCE,
            "logit {i}: imported model {g} vs onnxruntime reference {e}"
        );
    }
}

#[tokio::test]
async fn imported_onnx_classifier_matches_onnxruntime_reference() {
    if !gpu_available() {
        eprintln!("skipping: no burn wgpu adapter on this host");
        return;
    }
    let _gpu = GPU_LOCK.lock().await;
    let logits = logits_from(classifier_element().expect("imported classifier element")).await;
    assert_matches_reference(&logits, &CLASSIFIER_LOGITS);
}

#[tokio::test]
async fn imported_onnx_attention_matches_onnxruntime_reference() {
    if !gpu_available() {
        eprintln!("skipping: no burn wgpu adapter on this host");
        return;
    }
    let _gpu = GPU_LOCK.lock().await;
    let logits = logits_from(attention_element().expect("imported attention element")).await;
    assert_matches_reference(&logits, &ATTENTION_LOGITS);
}
