#![cfg(feature = "wgpu")]
//! §5.2 (M216): `WgpuInference` runs a linear layer on the GPU directly against
//! the GPU-resident tensor `WgpuPreprocess::with_gpu_output` (M215) emits, so the
//! tensor never makes the GPU->CPU->GPU round-trip. The tests chain the two real
//! GPU elements (NV12 -> preprocess -> inference) and assert the logits, read
//! back only at the very end, match a full CPU reference and the burn / ort
//! linear contract. Skips when no wgpu adapter is present.

use g2g_core::element::{BoxFuture, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, Caps, Dim, G2gError, OutputSink, Rate, RawVideoFormat, TensorDType, TensorLayout,
    TensorShape,
};
use g2g_ml::wgpuinfer::{linear_reference, WgpuInference};
use g2g_ml::wgpupreprocess::{
    gpu_available, nv12_to_gpu_texture, nv12_to_rgb_tensor, WgpuBufferOwner, WgpuPreprocess,
};

const W: u32 = 4;
const H: u32 = 2;
const K: usize = 3 * W as usize * H as usize; // flat NCHW length
const N: usize = 2; // outputs

fn nv12_caps(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn tensor_in_caps() -> Caps {
    Caps::Tensor {
        dtype: TensorDType::F32,
        shape: TensorShape(vec![1, 3, H, W]),
        layout: TensorLayout::Nchw,
    }
}

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

fn nv12_frame(bytes: Vec<u8>, pts_ns: u64, sequence: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming { pts_ns, dts_ns: pts_ns, ..FrameTiming::default() },
        sequence,
        meta: Default::default(),
    }
}

fn nv12_texture_frame(domain: MemoryDomain) -> Frame {
    Frame { domain, timing: FrameTiming { pts_ns: 99, dts_ns: 99, ..FrameTiming::default() }, sequence: 0, meta: Default::default() }
}

fn sample_nv12() -> Vec<u8> {
    let y_plane = [16u8, 81, 145, 235, 41, 100, 200, 128];
    let uv_plane = [128u8, 128, 90, 200]; // block 0 neutral, block 1 coloured
    y_plane.iter().chain(&uv_plane).copied().collect()
}

/// Deterministic `[K, N]` weights (row-major) + `[N]` bias. Column 0 sums every
/// input; column 1 is a position-weighted ramp, so the two outputs differ and a
/// transposed / mis-indexed weight matrix would be caught.
fn weights_bias() -> (Vec<f32>, Vec<f32>) {
    let mut weights = vec![0f32; K * N];
    for k in 0..K {
        weights[k * N] = 1.0; // column 0
        weights[k * N + 1] = k as f32 * 0.01; // column 1
    }
    (weights, vec![0.5, -0.25])
}

/// Run the NV12 frame through `WgpuPreprocess` in GPU-output mode and return the
/// resulting GPU-resident tensor frame (a `MemoryDomain::WgpuBuffer`).
async fn preprocess_to_gpu_tensor(nv12: Vec<u8>) -> Frame {
    let mut pre = WgpuPreprocess::new().with_gpu_output();
    pre.configure_pipeline(&nv12_caps(W, H)).expect("configure NV12");
    let mut out = Collect::default();
    pre.process(PipelinePacket::DataFrame(nv12_frame(nv12, 4242, 7)), &mut out)
        .await
        .expect("gpu-output preprocess");
    out.packets
        .into_iter()
        .find_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f),
            _ => None,
        })
        .expect("a GPU-resident tensor frame")
}

fn logits_from_system(f: &Frame) -> Vec<f32> {
    let MemoryDomain::System(slice) = &f.domain else {
        panic!("default mode must read logits back to System, got {:?}", f.domain.kind());
    };
    slice.as_slice().chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
}

#[tokio::test]
async fn infers_gpu_resident_tensor_and_matches_cpu_reference() {
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }
    let nv12 = sample_nv12();
    let (weights, bias) = weights_bias();

    // The tensor enters the inference element GPU-resident: never read back.
    let tensor_frame = preprocess_to_gpu_tensor(nv12.clone()).await;
    assert!(
        matches!(tensor_frame.domain, MemoryDomain::WgpuBuffer(_)),
        "preprocess must hand off a GPU buffer"
    );

    let mut infer = WgpuInference::linear(W, H, weights.clone(), bias.clone()).unwrap();
    infer.configure_pipeline(&tensor_in_caps()).expect("configure tensor input");

    let mut out = Collect::default();
    infer
        .process(PipelinePacket::DataFrame(tensor_frame), &mut out)
        .await
        .expect("gpu inference on the resident tensor");

    let caps: Vec<&Caps> = out
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::CapsChanged(c) => Some(c),
            _ => None,
        })
        .collect();
    let frame = out
        .packets
        .iter()
        .find_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f),
            _ => None,
        })
        .expect("a logits frame");

    assert_eq!(caps.len(), 1, "logits caps emitted once");
    assert_eq!(
        *caps[0],
        Caps::Tensor {
            dtype: TensorDType::F32,
            shape: TensorShape(vec![1, N as u32]),
            layout: TensorLayout::Nchw,
        }
    );

    // Full CPU reference: the exact tensor the GPU preprocess produced, fed
    // through the same linear math. This pins both the preprocess and the
    // inference end-to-end.
    let cpu_tensor = nv12_to_rgb_tensor(&nv12, W as usize, H as usize);
    let expected = linear_reference(&cpu_tensor, &weights, &bias);

    let got = logits_from_system(frame);
    assert_eq!(got.len(), N, "[1, N] logits");
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert!((g - e).abs() < 1e-2, "logit {i}: gpu {g} vs cpu reference {e}");
    }
    // The two columns differ, so column 0 != column 1 unless the ramp happens to
    // sum equal, which it does not for this input: proves the weight matrix was
    // indexed, not collapsed.
    assert!((got[0] - got[1]).abs() > 1e-3, "the two outputs must differ");

    // timing flows through preprocess -> inference unchanged.
    assert_eq!(frame.timing.pts_ns, 4242);
    assert_eq!(infer.inferred_count(), 1);
}

/// `with_gpu_output`: the logits also stay GPU-resident, so the whole
/// preprocess -> inference branch keeps the data on the device until the final
/// read-back. The recovered owner is the same `WgpuBufferOwner` downcast the
/// preprocess stage uses.
#[tokio::test]
async fn gpu_output_logits_stay_resident_and_match() {
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }
    let nv12 = sample_nv12();
    let (weights, bias) = weights_bias();
    let tensor_frame = preprocess_to_gpu_tensor(nv12.clone()).await;

    let mut infer =
        WgpuInference::linear(W, H, weights.clone(), bias.clone()).unwrap().with_gpu_output();
    infer.configure_pipeline(&tensor_in_caps()).expect("configure");

    let mut out = Collect::default();
    infer.process(PipelinePacket::DataFrame(tensor_frame), &mut out).await.expect("gpu inference");

    let frame = out
        .packets
        .iter()
        .find_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f),
            _ => None,
        })
        .expect("a logits frame");

    let MemoryDomain::WgpuBuffer(owned) = &frame.domain else {
        panic!("gpu-output mode must keep logits resident, got {:?}", frame.domain.kind());
    };
    assert_eq!(owned.len, N * 4, "buffer holds the [1, N] f32 logits");

    let owner = owned
        .keep_alive()
        .as_any()
        .downcast_ref::<WgpuBufferOwner>()
        .expect("recover the wgpu buffer owner");
    let bytes = owner.read_back().expect("read logits back");
    let got: Vec<f32> =
        bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();

    let cpu_tensor = nv12_to_rgb_tensor(&nv12, W as usize, H as usize);
    let expected = linear_reference(&cpu_tensor, &weights, &bias);
    assert_eq!(got.len(), N);
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert!((g - e).abs() < 1e-2, "logit {i}: gpu-resident {g} vs cpu reference {e}");
    }
}

/// The element is GPU-input only: a System tensor frame (the CPU path's job) is
/// rejected, not silently wrong.
#[tokio::test]
async fn rejects_system_memory_input() {
    let (weights, bias) = weights_bias();
    let mut infer = WgpuInference::linear(W, H, weights, bias).unwrap();
    infer.configure_pipeline(&tensor_in_caps()).expect("configure");

    let mut out = Collect::default();
    let sys = Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(vec![0u8; K * 4].into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    };
    assert_eq!(
        infer.process(PipelinePacket::DataFrame(sys), &mut out).await,
        Err(G2gError::UnsupportedDomain),
        "System input is the CPU path's job (BurnInference)"
    );
}

/// The full keep-on-GPU branch (M215 + M216 + M217): a GPU NV12 surface ->
/// `WgpuPreprocess` (surface-import in, GPU-resident tensor out) -> `WgpuInference`
/// (binds that tensor) -> logits, with the pixels never touching the CPU until
/// the logits are read back at the very end. The result matches a full CPU
/// reference (NV12 -> RGB tensor -> linear).
#[tokio::test]
async fn surface_to_logits_keeps_everything_on_gpu() {
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }
    let nv12 = sample_nv12();
    let (weights, bias) = weights_bias();

    // GPU NV12 surface in (no CPU upload inside the element).
    let domain = nv12_to_gpu_texture(&nv12, W, H).await.expect("gpu nv12 surface");
    let mut pre = WgpuPreprocess::new().with_gpu_output();
    pre.configure_pipeline(&nv12_caps(W, H)).expect("configure preprocess");
    let mut pout = Collect::default();
    pre.process(PipelinePacket::DataFrame(nv12_texture_frame(domain)), &mut pout)
        .await
        .expect("surface-import preprocess");
    let tensor_frame = pout
        .packets
        .into_iter()
        .find_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f),
            _ => None,
        })
        .expect("a GPU-resident tensor frame");
    assert!(
        matches!(tensor_frame.domain, MemoryDomain::WgpuBuffer(_)),
        "tensor stays on the GPU between preprocess and inference"
    );

    // Inference binds the resident tensor directly.
    let mut infer = WgpuInference::linear(W, H, weights.clone(), bias.clone()).unwrap();
    infer.configure_pipeline(&tensor_in_caps()).expect("configure inference");
    let mut iout = Collect::default();
    infer.process(PipelinePacket::DataFrame(tensor_frame), &mut iout).await.expect("gpu inference");

    let frame = iout
        .packets
        .iter()
        .find_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f),
            _ => None,
        })
        .expect("a logits frame");
    let got = logits_from_system(frame);

    let cpu_tensor = nv12_to_rgb_tensor(&nv12, W as usize, H as usize);
    let expected = linear_reference(&cpu_tensor, &weights, &bias);
    assert_eq!(got.len(), N);
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert!((g - e).abs() < 1e-2, "logit {i}: gpu chain {g} vs cpu reference {e}");
    }
}

#[test]
fn linear_validates_weight_dimensions() {
    assert!(WgpuInference::linear(2, 2, vec![0.0; 3 * 4 * 2], vec![0.0; 2]).is_ok());
    assert_eq!(
        WgpuInference::linear(2, 2, vec![0.0; 23], vec![0.0; 2]).err(),
        Some(G2gError::CapsMismatch),
        "weights must be K*N"
    );
}
