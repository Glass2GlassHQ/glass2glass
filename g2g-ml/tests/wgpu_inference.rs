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
use g2g_ml::safetensors::{serialize, SafeTensors};
use g2g_ml::wgpuinfer::{
    avgpool2d_reference, conv2d_reference, linear_reference, maxpool2d_reference, relu_reference,
    sigmoid_reference, WgpuInference,
};
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

/// Serialize GPU work across the parallel test tasks: creating several wgpu
/// devices and dispatching on a single adapter concurrently can fault the driver,
/// so each GPU test holds this lock for its device work. (CI has no adapter and
/// skips these tests entirely.)
fn gpu_guard() -> &'static tokio::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<tokio::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
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

fn nchw_caps(shape: Vec<u32>) -> Caps {
    Caps::Tensor { dtype: TensorDType::F32, shape: TensorShape(shape), layout: TensorLayout::Nchw }
}

/// Configure `op` for `in_caps`, run it on `frame`, and return the single output
/// `DataFrame`. Lets the layer-zoo tests chain ops (each one's GPU-resident
/// output is the next one's input) without repeating the boilerplate.
async fn run_op(mut op: WgpuInference, in_caps: Caps, frame: Frame) -> Frame {
    op.configure_pipeline(&in_caps).expect("configure op");
    let mut out = Collect::default();
    op.process(PipelinePacket::DataFrame(frame), &mut out).await.expect("op process");
    out.packets
        .into_iter()
        .find_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f),
            _ => None,
        })
        .expect("an output frame")
}

#[tokio::test]
async fn infers_gpu_resident_tensor_and_matches_cpu_reference() {
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }
    let _gpu = gpu_guard().lock().await;
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
    let _gpu = gpu_guard().lock().await;
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

/// A real 2D convolution layer on the GPU-resident tensor: NV12 -> preprocess ->
/// conv2d, the keystone that lets the on-device chain run an actual CNN layer (not
/// just the matmul). 2 output channels, a 3x3 same-pad kernel over the `[1,3,2,4]`
/// preprocess tensor; the read-back `[1,2,2,4]` map matches the CPU conv reference
/// over the exact tensor the GPU preprocess produced.
#[tokio::test]
async fn conv2d_on_gpu_resident_tensor_matches_cpu_reference() {
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }
    let _gpu = gpu_guard().lock().await;
    const CIN: u32 = 3;
    const COUT: u32 = 2;
    const KH: u32 = 3;
    const KW: u32 = 3;
    // Deterministic, non-symmetric weights/bias so the full [Cout,Cin,KH,KW] index
    // and the spatial accumulation are exercised, not collapsed.
    let weights: Vec<f32> =
        (0..(COUT * CIN * KH * KW)).map(|i| i as f32 * 0.013 - 0.25).collect();
    let bias = vec![0.1f32, -0.2];

    let nv12 = sample_nv12();
    let tensor_frame = preprocess_to_gpu_tensor(nv12.clone()).await;
    assert!(
        matches!(tensor_frame.domain, MemoryDomain::WgpuBuffer(_)),
        "preprocess must hand off a GPU buffer"
    );

    let mut conv = WgpuInference::conv2d(CIN, COUT, KH, KW, H, W, weights.clone(), bias.clone())
        .expect("valid conv dims");
    conv.configure_pipeline(&tensor_in_caps()).expect("configure tensor input");

    let mut out = Collect::default();
    conv.process(PipelinePacket::DataFrame(tensor_frame), &mut out)
        .await
        .expect("gpu conv on the resident tensor");

    let caps: Vec<&Caps> = out
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::CapsChanged(c) => Some(c),
            _ => None,
        })
        .collect();
    assert_eq!(caps.len(), 1, "conv output caps emitted once");
    assert_eq!(
        *caps[0],
        Caps::Tensor {
            dtype: TensorDType::F32,
            shape: TensorShape(vec![1, COUT, H, W]),
            layout: TensorLayout::Nchw,
        },
        "[1, Cout, H, W] feature map"
    );

    let frame = out
        .packets
        .iter()
        .find_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f),
            _ => None,
        })
        .expect("a conv output frame");
    let got = logits_from_system(frame);

    let cpu_tensor = nv12_to_rgb_tensor(&nv12, W as usize, H as usize);
    let expected = conv2d_reference(
        &cpu_tensor,
        CIN as usize,
        COUT as usize,
        KH as usize,
        KW as usize,
        H as usize,
        W as usize,
        &weights,
        &bias,
    );
    assert_eq!(got.len(), (COUT * H * W) as usize, "[1, Cout, H, W] = 16 values");
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert!((g - e).abs() < 1e-2, "conv out {i}: gpu {g} vs cpu reference {e}");
    }
    // The kernel actually mixed inputs: a same-pad conv over a non-constant tensor
    // does not produce a flat map.
    assert!(
        got.iter().any(|&v| (v - got[0]).abs() > 1e-3),
        "the feature map must vary spatially (the conv was applied, not a constant)"
    );
}

/// M262: import trained conv weights from a safetensors file at runtime and run
/// them on the GPU. The architecture stays our compiled `WgpuInference`; only the
/// weights are loaded (here from an in-test safetensors blob, exactly as a real
/// `.safetensors` from PyTorch would arrive). The GPU output of the imported
/// layer matches the CPU conv reference fed the same decoded weights, proving the
/// weight-file -> GPU round-trip.
#[tokio::test]
async fn conv2d_imports_safetensors_weights_and_runs_on_gpu() {
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }
    let _gpu = gpu_guard().lock().await;
    const CIN: u32 = 3;
    const COUT: u32 = 2;
    const KH: u32 = 3;
    const KW: u32 = 3;
    let weights: Vec<f32> =
        (0..(COUT * CIN * KH * KW)).map(|i| (i as f32).sin() * 0.3).collect();
    let bias = vec![0.05f32, -0.1];

    // The trained-weights file, as PyTorch's safetensors.save_file would write it.
    let blob = serialize(&[
        ("conv.weight", &[COUT as usize, CIN as usize, KH as usize, KW as usize], &weights),
        ("conv.bias", &[COUT as usize], &bias),
    ]);
    let st = SafeTensors::parse(&blob).expect("parse safetensors weights");

    let mut conv = WgpuInference::conv2d_from_safetensors(&st, "conv.weight", "conv.bias", H, W)
        .expect("build conv from imported weights");
    conv.configure_pipeline(&tensor_in_caps()).expect("configure tensor input");

    let nv12 = sample_nv12();
    let tensor_frame = preprocess_to_gpu_tensor(nv12.clone()).await;
    let mut out = Collect::default();
    conv.process(PipelinePacket::DataFrame(tensor_frame), &mut out)
        .await
        .expect("gpu conv with imported weights");

    let frame = out
        .packets
        .iter()
        .find_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f),
            _ => None,
        })
        .expect("a conv output frame");
    let got = logits_from_system(frame);

    // Reference uses the weights decoded back out of the same file, so this pins
    // the loader (shape + f32 decode) and the GPU conv together.
    let w_ref = st.get("conv.weight").unwrap().to_f32().unwrap();
    let b_ref = st.get("conv.bias").unwrap().to_f32().unwrap();
    assert_eq!(w_ref, weights, "weights survive the safetensors round-trip");
    let cpu_tensor = nv12_to_rgb_tensor(&nv12, W as usize, H as usize);
    let expected = conv2d_reference(
        &cpu_tensor,
        CIN as usize,
        COUT as usize,
        KH as usize,
        KW as usize,
        H as usize,
        W as usize,
        &w_ref,
        &b_ref,
    );
    assert_eq!(got.len(), (COUT * H * W) as usize);
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert!((g - e).abs() < 1e-2, "conv out {i}: gpu {g} vs cpu reference {e}");
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
    let _gpu = gpu_guard().lock().await;
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

/// The layer zoo chained on-device: NV12 -> preprocess -> conv2d -> relu ->
/// maxpool, every stage GPU-resident (`with_gpu_output`) until the final pool is
/// read back. A real small-CNN body: the data never leaves the GPU between
/// layers. The result matches a CPU reference that folds the same ops over the
/// exact tensor the GPU preprocess produced, and the relu actually clamps (the
/// conv output has negatives), so a missing nonlinearity would be caught.
#[tokio::test]
async fn conv_relu_pool_chain_runs_on_gpu_and_matches_cpu_reference() {
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }
    let _gpu = gpu_guard().lock().await;
    const CIN: u32 = 3;
    const COUT: u32 = 2;
    const KH: u32 = 3;
    const KW: u32 = 3;
    const PK: u32 = 2; // 2x2 pool, stride 2

    // Weights/bias chosen so the conv produces both signs, making the relu bite.
    let weights: Vec<f32> =
        (0..(COUT * CIN * KH * KW)).map(|i| i as f32 * 0.05 - 0.6).collect();
    let bias = vec![-0.3f32, 0.2];

    let nv12 = sample_nv12();
    let tensor_frame = preprocess_to_gpu_tensor(nv12.clone()).await;

    // conv2d -> relu -> maxpool, intermediates kept on the GPU.
    let conv = WgpuInference::conv2d(CIN, COUT, KH, KW, H, W, weights.clone(), bias.clone())
        .expect("valid conv")
        .with_gpu_output();
    let conv_out = run_op(conv, tensor_in_caps(), tensor_frame).await;
    assert!(
        matches!(conv_out.domain, MemoryDomain::WgpuBuffer(_)),
        "conv output stays GPU-resident for the next layer"
    );

    let relu = WgpuInference::relu(COUT, H, W).expect("valid relu").with_gpu_output();
    let relu_out = run_op(relu, nchw_caps(vec![1, COUT, H, W]), conv_out).await;
    assert!(
        matches!(relu_out.domain, MemoryDomain::WgpuBuffer(_)),
        "relu output stays GPU-resident for the pool"
    );

    // The pool reads back to System at the end of the chain.
    let pool = WgpuInference::maxpool2d(COUT, H, W, PK, PK, PK, PK).expect("valid pool");
    let pool_out = run_op(pool, nchw_caps(vec![1, COUT, H, W]), relu_out).await;
    let got = logits_from_system(&pool_out);

    // CPU reference: the same ops folded over the exact preprocess tensor.
    let cpu_tensor = nv12_to_rgb_tensor(&nv12, W as usize, H as usize);
    let conv_ref = conv2d_reference(
        &cpu_tensor,
        CIN as usize,
        COUT as usize,
        KH as usize,
        KW as usize,
        H as usize,
        W as usize,
        &weights,
        &bias,
    );
    let relu_ref = relu_reference(&conv_ref);
    let expected = maxpool2d_reference(
        &relu_ref,
        COUT as usize,
        H as usize,
        W as usize,
        PK as usize,
        PK as usize,
        PK as usize,
        PK as usize,
    );
    // 2x2 stride-2 over [COUT, 2, 4] -> [COUT, 1, 2] = 4 values.
    let (oh, ow) = ((H - PK) / PK + 1, (W - PK) / PK + 1);
    assert_eq!(got.len(), (COUT * oh * ow) as usize, "[1, COUT, OH, OW] pooled map");
    assert_eq!(expected.len(), got.len());
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert!((g - e).abs() < 1e-2, "chain out {i}: gpu {g} vs cpu reference {e}");
    }
    // The relu must have zeroed at least one conv output, else it was a no-op for
    // this input and the test would not prove the nonlinearity ran.
    assert!(
        conv_ref.iter().any(|&v| v < 0.0),
        "test setup: the conv must produce negatives for the relu to clamp"
    );
}

/// `avgpool2d` standalone, pinning the weightless (meta, input, out) bind path
/// and the average-pool math independently of the chain. A 2x2 stride-2 pool over
/// the `[1, 3, 2, 4]` preprocess tensor, read back and compared to the reference.
#[tokio::test]
async fn avgpool2d_on_gpu_resident_tensor_matches_cpu_reference() {
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }
    let _gpu = gpu_guard().lock().await;
    const C: u32 = 3;
    const PK: u32 = 2;

    let nv12 = sample_nv12();
    let tensor_frame = preprocess_to_gpu_tensor(nv12.clone()).await;

    let pool = WgpuInference::avgpool2d(C, H, W, PK, PK, PK, PK).expect("valid avgpool");
    let out = run_op(pool, tensor_in_caps(), tensor_frame).await;
    let got = logits_from_system(&out);

    let cpu_tensor = nv12_to_rgb_tensor(&nv12, W as usize, H as usize);
    let expected = avgpool2d_reference(
        &cpu_tensor,
        C as usize,
        H as usize,
        W as usize,
        PK as usize,
        PK as usize,
        PK as usize,
        PK as usize,
    );
    assert_eq!(got.len(), expected.len(), "[1, C, OH, OW] pooled map");
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert!((g - e).abs() < 1e-3, "avgpool {i}: gpu {g} vs cpu reference {e}");
    }
}

/// `sigmoid` standalone, pinning the activation shader's sigmoid branch (kind 1)
/// independently of the relu the chain exercises. Monotonic and bounded in (0, 1),
/// so a wrong formula is caught regardless of input sign.
#[tokio::test]
async fn sigmoid_on_gpu_resident_tensor_matches_cpu_reference() {
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }
    let _gpu = gpu_guard().lock().await;
    let nv12 = sample_nv12();
    let tensor_frame = preprocess_to_gpu_tensor(nv12.clone()).await;

    let act = WgpuInference::sigmoid(3, H, W).expect("valid sigmoid");
    let out = run_op(act, tensor_in_caps(), tensor_frame).await;
    let got = logits_from_system(&out);

    let cpu_tensor = nv12_to_rgb_tensor(&nv12, W as usize, H as usize);
    let expected = sigmoid_reference(&cpu_tensor);
    assert_eq!(got.len(), expected.len(), "shape-preserving activation");
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert!((g - e).abs() < 1e-3, "sigmoid {i}: gpu {g} vs cpu reference {e}");
        assert!(*g > 0.0 && *g < 1.0, "sigmoid output {i} = {g} must lie in (0, 1)");
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

#[test]
fn pool_validates_window_and_dims() {
    // A 2x2 pool fits a 2x4 input.
    assert!(WgpuInference::maxpool2d(3, 2, 4, 2, 2, 2, 2).is_ok());
    // A window larger than the input is rejected, not silently clamped.
    assert_eq!(
        WgpuInference::maxpool2d(3, 2, 4, 3, 2, 1, 1).err(),
        Some(G2gError::CapsMismatch),
        "kh > h must fail loud"
    );
    // Zero stride / channels are rejected.
    assert_eq!(WgpuInference::avgpool2d(3, 2, 4, 2, 2, 0, 1).err(), Some(G2gError::CapsMismatch));
    assert_eq!(WgpuInference::relu(0, 2, 4).err(), Some(G2gError::CapsMismatch));
}

#[test]
fn conv2d_overflowing_dims_fail_loud_not_panic() {
    // conv2d dims can come from an untrusted safetensors shape. A kernel whose
    // element-count product overflows must return CapsMismatch, not panic
    // (debug) or wrap to a value that admits a short weight buffer / undersized
    // GPU buffers. 65536^4 overflows u64, so the weight-length fold rejects it.
    assert_eq!(
        WgpuInference::conv2d(0x10000, 0x10000, 0x10000, 0x10000, 0x10000, 0x10000, vec![], vec![])
            .err(),
        Some(G2gError::CapsMismatch),
        "overflowing conv2d geometry must fail loud"
    );
    // Valid kernel dims but a spatial size whose in/out element count overflows
    // usize must also fail at the size fold rather than panicking.
    assert_eq!(
        WgpuInference::conv2d(3, 3, 3, 3, 0xFFFF_FFFF, 0xFFFF_FFFF, vec![0.0; 81], vec![0.0; 3])
            .err(),
        Some(G2gError::CapsMismatch),
        "overflowing conv2d spatial size must fail loud"
    );
}
