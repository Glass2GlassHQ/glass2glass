//! M21: `OrtInference` end-to-end against real ONNX Runtime (CPU EP).
//!
//! The model fixture is built in-test by hand-encoding the ONNX protobuf
//! (a single `Identity` node), so the test needs no network or checked-in
//! binary blob and the expected output is exactly the normalized input.
//!
//! Run with:
//!
//! ```powershell
//! cargo test -p g2g-ml --features ort --test ort_inference
//! ```

#![cfg(feature = "ort")]

use g2g_core::element::{AsyncElement, BoxFuture, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    Caps, ConfigureOutcome, Dim, G2gError, Rate, RawVideoFormat, TensorDType, TensorLayout,
    TensorShape,
};
use g2g_ml::ortinfer::OrtInference;

// shared hand-encoded ONNX fixture builder (tests/util/onnx_fixture.rs)
mod onnx {
    include!("util/onnx_fixture.rs");
}
use onnx::identity_model;

// --- test harness ---------------------------------------------------------

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

fn rgba_caps(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Any,
    }
}

fn rgba_frame(bytes: Vec<u8>, sequence: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence,
        meta: Default::default(),
    }
}

#[test]
fn model_contract_is_validated_at_construction() {
    // 4 channels violates the [N, 3, H, W] contract.
    let four_chan = identity_model(&[1, 4, 2, 2]);
    assert_eq!(
        OrtInference::from_memory(&four_chan).err(),
        Some(G2gError::CapsMismatch)
    );

    // rank 2 violates it too.
    let rank2 = identity_model(&[1, 3]);
    assert_eq!(
        OrtInference::from_memory(&rank2).err(),
        Some(G2gError::CapsMismatch)
    );

    let good = identity_model(&[1, 3, 2, 2]);
    let inf = OrtInference::from_memory(&good).expect("contract-conforming model loads");
    assert_eq!(inf.input_dims(), (2, 2));
    assert_eq!(inf.output_shape(), &[1, 3, 2, 2]);
}

/// M26: the DirectML-registered session (with CPU fallback) must produce
/// the same byte-exact results as the CPU path.
#[cfg(feature = "directml")]
#[tokio::test]
async fn directml_session_infers_identically() {
    let model = identity_model(&[1, 3, 2, 2]);
    let mut inf = OrtInference::from_memory_with_directml(&model).expect("model loads");
    assert_eq!(inf.input_dims(), (2, 2));
    let narrowed = inf.intercept_caps(&rgba_caps(2, 2)).expect("2x2 accepted");
    inf.configure_pipeline(&narrowed).expect("configure");

    let mut sink = Collect::default();
    inf.process(
        PipelinePacket::DataFrame(rgba_frame((0..16).collect(), 0)),
        &mut sink,
    )
    .await
    .expect("frame");
    let values: Vec<f32> = sink
        .packets
        .iter()
        .find_map(|p| match p {
            PipelinePacket::DataFrame(f) => {
                let MemoryDomain::System(slice) = &f.domain else {
                    return None;
                };
                Some(
                    slice
                        .as_slice()
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect(),
                )
            }
            _ => None,
        })
        .expect("one tensor frame");
    let expected: Vec<f32> = [0u8, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14]
        .iter()
        .map(|b| *b as f32 / 255.0)
        .collect();
    assert_eq!(values, expected);
}

/// The CUDA-registered session (best-effort, CPU fallback when no NVIDIA
/// device) must produce the same byte-exact results as the CPU path. On this
/// host without CUDA it proves the EP wires/registers/runs via fallback, not
/// that CUDA executed.
#[cfg(feature = "cuda")]
#[tokio::test]
async fn cuda_session_infers_identically() {
    let model = identity_model(&[1, 3, 2, 2]);
    let mut inf = OrtInference::from_memory_with_cuda(&model).expect("model loads");
    assert_eq!(inf.input_dims(), (2, 2));
    let narrowed = inf.intercept_caps(&rgba_caps(2, 2)).expect("2x2 accepted");
    inf.configure_pipeline(&narrowed).expect("configure");

    let mut sink = Collect::default();
    inf.process(
        PipelinePacket::DataFrame(rgba_frame((0..16).collect(), 0)),
        &mut sink,
    )
    .await
    .expect("frame");
    let values: Vec<f32> = sink
        .packets
        .iter()
        .find_map(|p| match p {
            PipelinePacket::DataFrame(f) => {
                let MemoryDomain::System(slice) = &f.domain else {
                    return None;
                };
                Some(
                    slice
                        .as_slice()
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect(),
                )
            }
            _ => None,
        })
        .expect("one tensor frame");
    let expected: Vec<f32> = [0u8, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14]
        .iter()
        .map(|b| *b as f32 / 255.0)
        .collect();
    assert_eq!(values, expected);
}

#[tokio::test]
async fn inference_emits_tensor_caps_and_normalized_values() {
    let model = identity_model(&[1, 3, 2, 2]);
    let mut inf = OrtInference::from_memory(&model).expect("model loads");

    // geometry-pinned negotiation: wrong dims are rejected, right dims pass.
    assert_eq!(
        inf.intercept_caps(&rgba_caps(4, 4)),
        Err(G2gError::CapsMismatch)
    );
    let narrowed = inf.intercept_caps(&rgba_caps(2, 2)).expect("2x2 accepted");
    let outcome = inf.configure_pipeline(&narrowed).expect("configure");
    assert!(matches!(outcome, ConfigureOutcome::Accepted));

    // 2x2 RGBA: pixel p holds [4p, 4p+1, 4p+2, 4p+3].
    let rgba: Vec<u8> = (0..16).collect();
    let mut sink = Collect::default();
    inf.process(PipelinePacket::DataFrame(rgba_frame(rgba, 0)), &mut sink)
        .await
        .expect("first frame");
    inf.process(
        PipelinePacket::DataFrame(rgba_frame((16..32).collect(), 1)),
        &mut sink,
    )
    .await
    .expect("second frame");
    inf.process(PipelinePacket::Eos, &mut sink).await.expect("eos");
    assert_eq!(inf.inferred_count(), 2);

    // exactly one CapsChanged (suppressed on the unchanged second frame).
    let caps: Vec<_> = sink
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::CapsChanged(c) => Some(c.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        caps,
        vec![Caps::Tensor {
            dtype: TensorDType::F32,
            shape: TensorShape::new([1, 3, 2, 2]),
            layout: TensorLayout::Nchw,
        }]
    );

    // Identity model: output = input preprocessing, i.e. RGB planes / 255
    // in NCHW order. R plane = bytes [0,4,8,12], G = [1,5,9,13], B = [2,6,10,14].
    let frames: Vec<&Frame> = sink
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f),
            _ => None,
        })
        .collect();
    assert_eq!(frames.len(), 2);
    let MemoryDomain::System(slice) = &frames[0].domain else {
        panic!("tensor frames are System-domain");
    };
    let bytes = slice.as_slice();
    assert_eq!(bytes.len(), 12 * 4, "1x3x2x2 f32 tensor");
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let expected: Vec<f32> = [0u8, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14]
        .iter()
        .map(|b| *b as f32 / 255.0)
        .collect();
    assert_eq!(values, expected);
    assert_eq!(frames[0].sequence, 0);
    assert_eq!(frames[1].sequence, 1);
}

#[tokio::test]
async fn tensor_input_mode_feeds_preprocessed_tensor_directly() {
    // tensor-input mode (M59): a GPU preprocess hands an already-normalized
    // f32 NCHW tensor straight in, with no second CPU /255 normalize.
    let model = identity_model(&[1, 3, 2, 2]);
    let mut inf = OrtInference::from_memory(&model)
        .expect("model loads")
        .with_tensor_input();

    let tensor_caps = Caps::Tensor {
        dtype: TensorDType::F32,
        shape: TensorShape::new([1, 3, 2, 2]),
        layout: TensorLayout::Nchw,
    };
    // negotiation flips to the tensor pad: matching tensor accepted, RGBA rejected.
    assert_eq!(inf.intercept_caps(&tensor_caps), Ok(tensor_caps.clone()));
    assert_eq!(
        inf.intercept_caps(&rgba_caps(2, 2)),
        Err(G2gError::CapsMismatch)
    );
    inf.configure_pipeline(&tensor_caps).expect("configure");

    // identity model returns the fed tensor unchanged (no extra normalization).
    let input: Vec<f32> = (0..12).map(|i| i as f32 / 255.0).collect();
    let bytes: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut sink = Collect::default();
    inf.process(PipelinePacket::DataFrame(rgba_frame(bytes, 0)), &mut sink)
        .await
        .expect("tensor frame");

    let values: Vec<f32> = sink
        .packets
        .iter()
        .find_map(|p| match p {
            PipelinePacket::DataFrame(f) => {
                let MemoryDomain::System(slice) = &f.domain else {
                    return None;
                };
                Some(
                    slice
                        .as_slice()
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect(),
                )
            }
            _ => None,
        })
        .expect("one tensor frame");
    assert_eq!(values, input);
}

#[tokio::test]
async fn uint8_tensor_input_runs_a_quantized_model() {
    // M442: a quantized model has a uint8 input; OrtInference feeds the integer
    // tensor straight through (no f32 normalize). Uses the committed uint8-input
    // QDQ Conv->ReLU fixture; on the CPU EP here it proves the integer-input path
    // (the Edge TPU placement is the on-device probe's job).
    let model = include_bytes!("fixtures/qconv_relu_u8in.onnx");
    let mut inf = OrtInference::from_memory(model).expect("uint8 model loads").with_tensor_input();
    assert_eq!(inf.input_dims(), (4, 4));

    // The input pad is a uint8 NCHW tensor; an f32 tensor (or RGBA) is rejected.
    let u8_caps = Caps::Tensor {
        dtype: TensorDType::U8,
        shape: TensorShape::new([1, 3, 4, 4]),
        layout: TensorLayout::Nchw,
    };
    assert_eq!(inf.intercept_caps(&u8_caps), Ok(u8_caps.clone()));
    let f32_caps = Caps::Tensor {
        dtype: TensorDType::F32,
        shape: TensorShape::new([1, 3, 4, 4]),
        layout: TensorLayout::Nchw,
    };
    assert_eq!(inf.intercept_caps(&f32_caps), Err(G2gError::CapsMismatch));
    inf.configure_pipeline(&u8_caps).expect("configure");

    // 48 uint8 elements ([1,3,4,4]), one byte each (pre-quantized pixel-like data).
    let data: Vec<u8> = (0..48u16).map(|i| (i * 5) as u8).collect();
    let mut sink = Collect::default();
    inf.process(PipelinePacket::DataFrame(rgba_frame(data, 0)), &mut sink)
        .await
        .expect("uint8 tensor frame runs");

    let out: Vec<f32> = sink
        .packets
        .iter()
        .find_map(|p| match p {
            PipelinePacket::DataFrame(f) => {
                let MemoryDomain::System(slice) = &f.domain else { return None };
                Some(slice.as_slice().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
            }
            _ => None,
        })
        .expect("one tensor frame");
    // Output is [1,4,4,4] = 64 floats, post-ReLU so non-negative.
    assert_eq!(out.len(), 64);
    assert!(out.iter().all(|v| v.is_finite() && *v >= 0.0));
}
