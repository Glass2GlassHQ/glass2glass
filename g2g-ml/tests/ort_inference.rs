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

// --- minimal ONNX protobuf encoding -------------------------------------
// Wire format only needs varints and length-delimited fields. Field numbers
// follow onnx.proto3 (ModelProto / GraphProto / NodeProto / ValueInfoProto).

fn varint(mut v: u64, out: &mut Vec<u8>) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

fn field_varint(field: u64, v: u64, out: &mut Vec<u8>) {
    varint(field << 3, out); // wire type 0
    varint(v, out);
}

fn field_bytes(field: u64, bytes: &[u8], out: &mut Vec<u8>) {
    varint((field << 3) | 2, out); // wire type 2
    varint(bytes.len() as u64, out);
    out.extend_from_slice(bytes);
}

fn field_str(field: u64, s: &str, out: &mut Vec<u8>) {
    field_bytes(field, s.as_bytes(), out);
}

/// ValueInfoProto: name=1, type=2; TypeProto.tensor_type=1 holds
/// elem_type=1 (FLOAT = 1) and shape=2 (dims as Dimension.dim_value=1).
fn value_info(name: &str, dims: &[u64]) -> Vec<u8> {
    let mut shape = Vec::new();
    for d in dims {
        let mut dim = Vec::new();
        field_varint(1, *d, &mut dim);
        field_bytes(1, &dim, &mut shape);
    }
    let mut tensor_type = Vec::new();
    field_varint(1, 1, &mut tensor_type); // elem_type = FLOAT
    field_bytes(2, &shape, &mut tensor_type);
    let mut type_proto = Vec::new();
    field_bytes(1, &tensor_type, &mut type_proto);
    let mut vi = Vec::new();
    field_str(1, name, &mut vi);
    field_bytes(2, &type_proto, &mut vi);
    vi
}

/// A complete ModelProto holding one `Identity` node from input "x" to
/// output "y", both f32 tensors of `dims` (so a test can also build a
/// contract-violating model, e.g. 4 channels).
fn identity_model(dims: &[u64]) -> Vec<u8> {
    let mut node = Vec::new();
    field_str(1, "x", &mut node); // input
    field_str(2, "y", &mut node); // output
    field_str(4, "Identity", &mut node); // op_type

    let mut graph = Vec::new();
    field_bytes(1, &node, &mut graph);
    field_str(2, "g", &mut graph); // graph name
    field_bytes(11, &value_info("x", dims), &mut graph); // input
    field_bytes(12, &value_info("y", dims), &mut graph); // output

    let mut opset = Vec::new();
    field_str(1, "", &mut opset); // default domain
    field_varint(2, 13, &mut opset); // opset 13

    let mut model = Vec::new();
    field_varint(1, 8, &mut model); // ir_version = 8
    field_bytes(7, &graph, &mut model);
    field_bytes(8, &opset, &mut model);
    model
}

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
            shape: TensorShape(vec![1, 3, 2, 2]),
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
