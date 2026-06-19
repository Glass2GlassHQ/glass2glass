//! M27: `TensorPostprocess` element-level behavior, plus the full
//! classification chain through real ONNX Runtime when `ort` is enabled.

use g2g_core::element::{AsyncElement, BoxFuture, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{Caps, G2gError, TensorDType, TensorLayout, TensorShape};
use g2g_ml::postprocess::TensorPostprocess;

#[derive(Default)]
struct Collect {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            self.packets.push(packet);
            Ok(PushOutcome::Accepted)
        })
    }
}

fn logits_caps(n: u32) -> Caps {
    Caps::Tensor {
        dtype: TensorDType::F32,
        shape: TensorShape(vec![1, n]),
        layout: TensorLayout::Nchw,
    }
}

fn f32_frame(values: &[f32], sequence: u64) -> Frame {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence,
        meta: Default::default(),
    }
}

fn frame_values(f: &Frame) -> Vec<f32> {
    let MemoryDomain::System(slice) = &f.domain else {
        panic!("System frames expected");
    };
    slice
        .as_slice()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[tokio::test]
async fn softmax_element_normalizes_and_keeps_input_caps() {
    let mut el = TensorPostprocess::softmax();
    el.configure_pipeline(&logits_caps(4)).expect("configure");
    let mut out = Collect::default();
    el.process(
        PipelinePacket::DataFrame(f32_frame(&[0.0, 1.0, 2.0, 3.0], 0)),
        &mut out,
    )
    .await
    .expect("frame");

    let caps: Vec<_> = out
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::CapsChanged(c) => Some(c.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(caps, vec![logits_caps(4)], "softmax echoes the input caps");

    let frames: Vec<_> = out
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f),
            _ => None,
        })
        .collect();
    let probs = frame_values(frames[0]);
    assert_eq!(probs.len(), 4);
    assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    assert!(probs[3] > probs[2] && probs[2] > probs[1] && probs[1] > probs[0]);
}

#[tokio::test]
async fn argmax_element_emits_index_value_pair() {
    let mut el = TensorPostprocess::argmax();
    el.configure_pipeline(&logits_caps(5)).expect("configure");
    let mut out = Collect::default();
    el.process(
        PipelinePacket::DataFrame(f32_frame(&[0.1, 0.9, 7.5, -2.0, 3.0], 0)),
        &mut out,
    )
    .await
    .expect("frame");

    let frames: Vec<_> = out
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f),
            _ => None,
        })
        .collect();
    assert_eq!(frame_values(frames[0]), vec![2.0, 7.5], "[index, value]");
    let caps: Vec<_> = out
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::CapsChanged(c) => Some(c.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(caps, vec![logits_caps(2)], "argmax output is [1, 2]");
}

#[tokio::test]
async fn non_f32_tensor_is_rejected() {
    let mut el = TensorPostprocess::softmax();
    let bytes = Caps::Tensor {
        dtype: TensorDType::U8,
        shape: TensorShape(vec![1, 4]),
        layout: TensorLayout::Nchw,
    };
    let err = el.configure_pipeline(&bytes).expect_err("u8 rejected");
    assert_eq!(err, G2gError::CapsMismatch);
}

/// Full classification chain: real ONNX Runtime inference (identity model)
/// into argmax. The brightest input pixel's red channel must win.
#[cfg(feature = "ort")]
#[tokio::test]
async fn real_inference_into_argmax_finds_the_peak() {
    use g2g_ml::ortinfer::OrtInference;

    // identity model fixture, same encoding as ort_inference.rs but inline
    // minimal: reuse the inference element only as a producer of tensors.
    mod onnx {
        include!("util/onnx_fixture.rs");
    }
    let model = onnx::identity_model(&[1, 3, 2, 2]);
    let mut inf = OrtInference::from_memory(&model).expect("model loads");
    let rgba = Caps::RawVideo {
        format: g2g_core::RawVideoFormat::Rgba8,
        width: g2g_core::Dim::Fixed(2),
        height: g2g_core::Dim::Fixed(2),
        framerate: g2g_core::Rate::Any,
    };
    inf.configure_pipeline(&rgba).expect("configure inference");

    // pixel 3's red channel (offset 12) is the global maximum.
    let mut pixels = vec![10u8; 16];
    pixels[12] = 250;
    let frame = Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(pixels.into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    };

    let mut tensors = Collect::default();
    inf.process(PipelinePacket::DataFrame(frame), &mut tensors)
        .await
        .expect("inference");

    let mut head = TensorPostprocess::argmax();
    let mut out = Collect::default();
    for p in tensors.packets {
        match p {
            PipelinePacket::CapsChanged(c) => {
                head.configure_pipeline(&c).expect("configure head");
            }
            PipelinePacket::DataFrame(f) => {
                head.process(PipelinePacket::DataFrame(f), &mut out)
                    .await
                    .expect("argmax");
            }
            _ => {}
        }
    }

    let frames: Vec<_> = out
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f),
            _ => None,
        })
        .collect();
    let result = frame_values(frames[0]);
    // NCHW: R plane holds pixels 0..4, so pixel 3's red is flat index 3.
    assert_eq!(result[0], 3.0, "winning flat index");
    assert!((result[1] - 250.0 / 255.0).abs() < 1e-6, "winning value");
}
