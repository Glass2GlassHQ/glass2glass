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
    add_reference, attention_reference, avgpool2d_reference, batch_norm_reference,
    conv2d_reference, layer_norm_reference, linear_reference, matmul_reference,
    maxpool2d_reference, relu_reference, sigmoid_reference, softmax_reference, StackLayer,
    WgpuInference,
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
        shape: TensorShape::new([1, 3, H, W]),
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
        timing: FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            ..FrameTiming::default()
        },
        sequence,
        meta: Default::default(),
    }
}

fn nv12_texture_frame(domain: MemoryDomain) -> Frame {
    Frame {
        domain,
        timing: FrameTiming {
            pts_ns: 99,
            dts_ns: 99,
            ..FrameTiming::default()
        },
        sequence: 0,
        meta: Default::default(),
    }
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
    pre.configure_pipeline(&nv12_caps(W, H))
        .expect("configure NV12");
    let mut out = Collect::default();
    pre.process(
        PipelinePacket::DataFrame(nv12_frame(nv12, 4242, 7)),
        &mut out,
    )
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
    let Some(slice) = f.domain.as_system_slice() else {
        panic!(
            "default mode must read logits back to System, got {:?}",
            f.domain.kind()
        );
    };
    slice
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn nchw_caps(shape: &[u32]) -> Caps {
    Caps::Tensor {
        dtype: TensorDType::F32,
        shape: TensorShape::from_slice(shape).unwrap(),
        layout: TensorLayout::Nchw,
    }
}

/// Configure `op` for `in_caps`, run it on `frame`, and return the single output
/// `DataFrame`. Lets the layer-zoo tests chain ops (each one's GPU-resident
/// output is the next one's input) without repeating the boilerplate.
async fn run_op(mut op: WgpuInference, in_caps: Caps, frame: Frame) -> Frame {
    op.configure_pipeline(&in_caps).expect("configure op");
    let mut out = Collect::default();
    op.process(PipelinePacket::DataFrame(frame), &mut out)
        .await
        .expect("op process");
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
    infer
        .configure_pipeline(&tensor_in_caps())
        .expect("configure tensor input");

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
            shape: TensorShape::new([1, N as u32]),
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
        assert!(
            (g - e).abs() < 1e-2,
            "logit {i}: gpu {g} vs cpu reference {e}"
        );
    }
    // The two columns differ, so column 0 != column 1 unless the ramp happens to
    // sum equal, which it does not for this input: proves the weight matrix was
    // indexed, not collapsed.
    assert!(
        (got[0] - got[1]).abs() > 1e-3,
        "the two outputs must differ"
    );

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

    let mut infer = WgpuInference::linear(W, H, weights.clone(), bias.clone())
        .unwrap()
        .with_gpu_output();
    infer
        .configure_pipeline(&tensor_in_caps())
        .expect("configure");

    let mut out = Collect::default();
    infer
        .process(PipelinePacket::DataFrame(tensor_frame), &mut out)
        .await
        .expect("gpu inference");

    let frame = out
        .packets
        .iter()
        .find_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f),
            _ => None,
        })
        .expect("a logits frame");

    let MemoryDomain::WgpuBuffer(owned) = &frame.domain else {
        panic!(
            "gpu-output mode must keep logits resident, got {:?}",
            frame.domain.kind()
        );
    };
    assert_eq!(owned.len, N * 4, "buffer holds the [1, N] f32 logits");

    let owner = owned
        .keep_alive()
        .as_any()
        .downcast_ref::<WgpuBufferOwner>()
        .expect("recover the wgpu buffer owner");
    let bytes = owner.read_back().expect("read logits back");
    let got: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();

    let cpu_tensor = nv12_to_rgb_tensor(&nv12, W as usize, H as usize);
    let expected = linear_reference(&cpu_tensor, &weights, &bias);
    assert_eq!(got.len(), N);
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert!(
            (g - e).abs() < 1e-2,
            "logit {i}: gpu-resident {g} vs cpu reference {e}"
        );
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
    let weights: Vec<f32> = (0..(COUT * CIN * KH * KW))
        .map(|i| i as f32 * 0.013 - 0.25)
        .collect();
    let bias = vec![0.1f32, -0.2];

    let nv12 = sample_nv12();
    let tensor_frame = preprocess_to_gpu_tensor(nv12.clone()).await;
    assert!(
        matches!(tensor_frame.domain, MemoryDomain::WgpuBuffer(_)),
        "preprocess must hand off a GPU buffer"
    );

    let mut conv = WgpuInference::conv2d(CIN, COUT, KH, KW, H, W, weights.clone(), bias.clone())
        .expect("valid conv dims");
    conv.configure_pipeline(&tensor_in_caps())
        .expect("configure tensor input");

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
            shape: TensorShape::new([1, COUT, H, W]),
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
    assert_eq!(
        got.len(),
        (COUT * H * W) as usize,
        "[1, Cout, H, W] = 16 values"
    );
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert!(
            (g - e).abs() < 1e-2,
            "conv out {i}: gpu {g} vs cpu reference {e}"
        );
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
    let weights: Vec<f32> = (0..(COUT * CIN * KH * KW))
        .map(|i| (i as f32).sin() * 0.3)
        .collect();
    let bias = vec![0.05f32, -0.1];

    // The trained-weights file, as PyTorch's safetensors.save_file would write it.
    let blob = serialize(&[
        (
            "conv.weight",
            &[COUT as usize, CIN as usize, KH as usize, KW as usize],
            &weights,
        ),
        ("conv.bias", &[COUT as usize], &bias),
    ]);
    let st = SafeTensors::parse(&blob).expect("parse safetensors weights");

    let mut conv = WgpuInference::conv2d_from_safetensors(&st, "conv.weight", "conv.bias", H, W)
        .expect("build conv from imported weights");
    conv.configure_pipeline(&tensor_in_caps())
        .expect("configure tensor input");

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
        assert!(
            (g - e).abs() < 1e-2,
            "conv out {i}: gpu {g} vs cpu reference {e}"
        );
    }
}

/// The element is GPU-input only: a System tensor frame (the CPU path's job) is
/// rejected, not silently wrong.
#[tokio::test]
async fn rejects_system_memory_input() {
    let (weights, bias) = weights_bias();
    let mut infer = WgpuInference::linear(W, H, weights, bias).unwrap();
    infer
        .configure_pipeline(&tensor_in_caps())
        .expect("configure");

    let mut out = Collect::default();
    let sys = Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(vec![0u8; K * 4].into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    };
    assert_eq!(
        infer
            .process(PipelinePacket::DataFrame(sys), &mut out)
            .await,
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
    let domain = nv12_to_gpu_texture(&nv12, W, H)
        .await
        .expect("gpu nv12 surface");
    let mut pre = WgpuPreprocess::new().with_gpu_output();
    pre.configure_pipeline(&nv12_caps(W, H))
        .expect("configure preprocess");
    let mut pout = Collect::default();
    pre.process(
        PipelinePacket::DataFrame(nv12_texture_frame(domain)),
        &mut pout,
    )
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
    infer
        .configure_pipeline(&tensor_in_caps())
        .expect("configure inference");
    let mut iout = Collect::default();
    infer
        .process(PipelinePacket::DataFrame(tensor_frame), &mut iout)
        .await
        .expect("gpu inference");

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
        assert!(
            (g - e).abs() < 1e-2,
            "logit {i}: gpu chain {g} vs cpu reference {e}"
        );
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
    let weights: Vec<f32> = (0..(COUT * CIN * KH * KW))
        .map(|i| i as f32 * 0.05 - 0.6)
        .collect();
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

    let relu = WgpuInference::relu(COUT, H, W)
        .expect("valid relu")
        .with_gpu_output();
    let relu_out = run_op(relu, nchw_caps(&[1, COUT, H, W]), conv_out).await;
    assert!(
        matches!(relu_out.domain, MemoryDomain::WgpuBuffer(_)),
        "relu output stays GPU-resident for the pool"
    );

    // The pool reads back to System at the end of the chain.
    let pool = WgpuInference::maxpool2d(COUT, H, W, PK, PK, PK, PK).expect("valid pool");
    let pool_out = run_op(pool, nchw_caps(&[1, COUT, H, W]), relu_out).await;
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
    assert_eq!(
        got.len(),
        (COUT * oh * ow) as usize,
        "[1, COUT, OH, OW] pooled map"
    );
    assert_eq!(expected.len(), got.len());
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert!(
            (g - e).abs() < 1e-2,
            "chain out {i}: gpu {g} vs cpu reference {e}"
        );
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
        assert!(
            (g - e).abs() < 1e-3,
            "avgpool {i}: gpu {g} vs cpu reference {e}"
        );
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
        assert!(
            (g - e).abs() < 1e-3,
            "sigmoid {i}: gpu {g} vs cpu reference {e}"
        );
        assert!(
            *g > 0.0 && *g < 1.0,
            "sigmoid output {i} = {g} must lie in (0, 1)"
        );
    }
}

/// M524: a *whole* small CNN classifier imported from one safetensors file and
/// run end to end on the GPU: conv-BN-ReLU-pool x2 -> global-avg-pool -> linear
/// head. `stack_from_safetensors` builds the chain (tracking the running shape),
/// every layer but the last stays GPU-resident, and the read-back logits match a
/// CPU reference folding the same ops over the exact preprocess tensor. This is
/// the step past a single conv (M262): a full trained model runs on the
/// hand-rolled GPU path, architecture compiled, weights a file. It exercises the
/// new batch-norm op and the general (post-pool) linear head.
#[tokio::test]
async fn full_cnn_from_safetensors_matches_cpu_reference() {
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }
    let _gpu = gpu_guard().lock().await;
    const EPS: f32 = 1e-5;

    // Deterministic pseudo-weights for a [1,3,2,4] -> 2-logit classifier.
    // conv1 [4,3,3,3], bn1 [4], conv2 [3,4,3,3], bn2 [3], fc [K=3,N=2].
    let conv1_w: Vec<f32> = (0..(4 * 3 * 3 * 3))
        .map(|i| (i as f32 * 0.7).sin() * 0.3)
        .collect();
    let conv1_b = vec![0.05f32, -0.1, 0.02, 0.0];
    let bn1_g = vec![1.1f32, 0.9, 1.0, 1.2];
    let bn1_b = vec![0.0f32, 0.1, -0.05, 0.2];
    let bn1_m = vec![0.1f32, -0.2, 0.0, 0.05];
    let bn1_v = vec![0.8f32, 1.2, 1.0, 0.6];
    let conv2_w: Vec<f32> = (0..(3 * 4 * 3 * 3))
        .map(|i| (i as f32 * 0.5).cos() * 0.25)
        .collect();
    let conv2_b = vec![-0.03f32, 0.07, 0.0];
    let bn2_g = vec![1.0f32, 1.05, 0.95];
    let bn2_b = vec![0.02f32, -0.03, 0.0];
    let bn2_m = vec![0.0f32, 0.1, -0.05];
    let bn2_v = vec![1.0f32, 0.7, 1.1];
    // fc weight [K=3, N=2] row-major (input-major, matching the matmul shader).
    let fc_w = vec![0.5f32, -0.3, 0.2, 0.4, -0.1, 0.25];
    let fc_b = vec![0.1f32, -0.2];

    let blob = serialize(&[
        ("conv1.weight", &[4, 3, 3, 3], &conv1_w),
        ("conv1.bias", &[4], &conv1_b),
        ("bn1.weight", &[4], &bn1_g),
        ("bn1.bias", &[4], &bn1_b),
        ("bn1.running_mean", &[4], &bn1_m),
        ("bn1.running_var", &[4], &bn1_v),
        ("conv2.weight", &[3, 4, 3, 3], &conv2_w),
        ("conv2.bias", &[3], &conv2_b),
        ("bn2.weight", &[3], &bn2_g),
        ("bn2.bias", &[3], &bn2_b),
        ("bn2.running_mean", &[3], &bn2_m),
        ("bn2.running_var", &[3], &bn2_v),
        ("fc.weight", &[3, 2], &fc_w),
        ("fc.bias", &[2], &fc_b),
    ]);
    let st = SafeTensors::parse(&blob).expect("parse model weights");

    let specs = vec![
        StackLayer::Conv2d {
            name: "conv1".into(),
        },
        StackLayer::BatchNorm {
            name: "bn1".into(),
            eps: EPS,
        },
        StackLayer::Relu,
        StackLayer::MaxPool2d {
            kh: 2,
            kw: 2,
            sh: 2,
            sw: 2,
        },
        StackLayer::Conv2d {
            name: "conv2".into(),
        },
        StackLayer::BatchNorm {
            name: "bn2".into(),
            eps: EPS,
        },
        StackLayer::Relu,
        StackLayer::GlobalAvgPool,
        StackLayer::Linear { name: "fc".into() },
    ];
    let chain = WgpuInference::stack_from_safetensors(&specs, &st, 3, H, W)
        .expect("import the whole stack from one file");
    assert_eq!(chain.len(), specs.len(), "one op per spec");

    // Run the chain on the GPU-resident preprocess tensor; the input caps to each
    // op is the previous op's output shape.
    let in_shapes = [
        vec![1, 3, 2, 4], // conv1
        vec![1, 4, 2, 4], // bn1
        vec![1, 4, 2, 4], // relu
        vec![1, 4, 2, 4], // maxpool
        vec![1, 4, 1, 2], // conv2
        vec![1, 3, 1, 2], // bn2
        vec![1, 3, 1, 2], // relu
        vec![1, 3, 1, 2], // global-avg-pool
        vec![1, 3, 1, 1], // fc
    ];
    let nv12 = sample_nv12();
    let mut frame = preprocess_to_gpu_tensor(nv12.clone()).await;
    for (op, shape) in chain.into_iter().zip(in_shapes) {
        frame = run_op(op, nchw_caps(&shape), frame).await;
    }
    let got = logits_from_system(&frame);

    // CPU reference: the same ops folded over the exact preprocess tensor.
    let x = nv12_to_rgb_tensor(&nv12, W as usize, H as usize);
    let c1 = conv2d_reference(&x, 3, 4, 3, 3, 2, 4, &conv1_w, &conv1_b);
    let n1 = batch_norm_reference(&c1, 4, 2 * 4, &bn1_g, &bn1_b, &bn1_m, &bn1_v, EPS);
    let r1 = relu_reference(&n1);
    let p1 = maxpool2d_reference(&r1, 4, 2, 4, 2, 2, 2, 2); // -> [4,1,2]
    let c2 = conv2d_reference(&p1, 4, 3, 3, 3, 1, 2, &conv2_w, &conv2_b); // -> [3,1,2]
    let n2 = batch_norm_reference(&c2, 3, 2, &bn2_g, &bn2_b, &bn2_m, &bn2_v, EPS);
    let r2 = relu_reference(&n2);
    let gap = avgpool2d_reference(&r2, 3, 1, 2, 1, 2, 1, 2); // -> [3,1,1]
    let expected = linear_reference(&gap, &fc_w, &fc_b);

    assert_eq!(got.len(), 2, "two class logits");
    assert_eq!(expected.len(), 2);
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert!(
            (g - e).abs() < 1e-2,
            "logit {i}: gpu full-model {g} vs cpu reference {e}"
        );
    }
    // The two logits must differ (the fc columns and the whole stack actually ran,
    // not collapsed to a constant).
    assert!(
        (got[0] - got[1]).abs() > 1e-4,
        "the classifier's two logits must differ"
    );
}

/// M531: a residual/skip block imported from one safetensors file and run
/// GPU-resident through `ResidualStack`: `y = conv2(relu(conv1(x))) + x`. The
/// `SaveSkip` records the input tensor, the two shape-preserving convs + ReLU are
/// the residual branch `f`, and `AddSkip` joins the saved input back with the new
/// elementwise-add GPU op. The read-back matches a CPU reference folding the same
/// ops (including the final add) over the exact preprocess tensor. This is the
/// step past the straight chain (M524): a non-linear topology now imports and runs
/// on the hand-rolled GPU path.
#[tokio::test]
async fn residual_block_from_safetensors_matches_cpu_reference() {
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }
    let _gpu = gpu_guard().lock().await;

    // Two shape-preserving 3x3 convs over the [1,3,2,4] preprocess tensor (cout=3
    // so the residual branch's output matches the 3-channel skip). Deterministic
    // pseudo-weights.
    let conv1_w: Vec<f32> = (0..(3 * 3 * 3 * 3))
        .map(|i| (i as f32 * 0.3).sin() * 0.2)
        .collect();
    let conv1_b = vec![0.01f32, -0.02, 0.0];
    let conv2_w: Vec<f32> = (0..(3 * 3 * 3 * 3))
        .map(|i| (i as f32 * 0.4).cos() * 0.15)
        .collect();
    let conv2_b = vec![0.0f32, 0.03, -0.01];

    let blob = serialize(&[
        ("conv1.weight", &[3, 3, 3, 3], &conv1_w),
        ("conv1.bias", &[3], &conv1_b),
        ("conv2.weight", &[3, 3, 3, 3], &conv2_w),
        ("conv2.bias", &[3], &conv2_b),
    ]);
    let st = SafeTensors::parse(&blob).expect("parse residual weights");

    let specs = vec![
        StackLayer::SaveSkip { slot: "id".into() },
        StackLayer::Conv2d {
            name: "conv1".into(),
        },
        StackLayer::Relu,
        StackLayer::Conv2d {
            name: "conv2".into(),
        },
        StackLayer::AddSkip { slot: "id".into() },
    ];
    let mut stack = WgpuInference::residual_stack_from_safetensors(&specs, &st, 3, H, W)
        .expect("import the residual block from one file");

    let nv12 = sample_nv12();
    let frame = preprocess_to_gpu_tensor(nv12.clone()).await;
    let out = stack.run(frame).expect("run the residual block on the GPU");
    let got = logits_from_system(&out);

    // CPU reference: f(x) + x over the exact preprocess tensor.
    let x = nv12_to_rgb_tensor(&nv12, W as usize, H as usize);
    let c1 = conv2d_reference(&x, 3, 3, 3, 3, H as usize, W as usize, &conv1_w, &conv1_b);
    let r1 = relu_reference(&c1);
    let c2 = conv2d_reference(&r1, 3, 3, 3, 3, H as usize, W as usize, &conv2_w, &conv2_b);
    let expected = add_reference(&c2, &x);

    assert_eq!(
        got.len(),
        expected.len(),
        "residual output is the [1,3,H,W] tensor"
    );
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert!(
            (g - e).abs() < 1e-3,
            "residual elem {i}: gpu {g} vs cpu reference {e}"
        );
    }
    // The skip must actually be added: the residual output must differ from the
    // branch output f(x) alone (else AddSkip was a no-op).
    let branch_only = &c2;
    let differs = got
        .iter()
        .zip(branch_only)
        .any(|(g, f)| (g - f).abs() > 1e-4);
    assert!(
        differs,
        "AddSkip must add the saved input, not pass f(x) through"
    );
}

/// The residual builder rejects an `AddSkip` against a shape that does not match
/// the saved skip tensor (a mis-wired block), rather than producing garbage.
#[test]
fn residual_addskip_shape_mismatch_fails_loud() {
    // Save x [3,2,4], then a 2x2 stride-2 pool shrinks it to [3,1,2]; adding the
    // saved [3,2,4] skip back is a shape mismatch.
    let blob = serialize(&[
        ("c.weight", &[3, 3, 3, 3], &[0.0f32; 81]),
        ("c.bias", &[3], &[0.0f32; 3]),
    ]);
    let st = SafeTensors::parse(&blob).unwrap();
    let specs = vec![
        StackLayer::SaveSkip { slot: "id".into() },
        StackLayer::MaxPool2d {
            kh: 2,
            kw: 2,
            sh: 2,
            sw: 2,
        },
        StackLayer::AddSkip { slot: "id".into() },
    ];
    assert!(
        WgpuInference::residual_stack_from_safetensors(&specs, &st, 3, H, W).is_err(),
        "an AddSkip whose saved shape != running shape must fail loud"
    );
    // An AddSkip naming an unsaved slot also fails.
    let specs2 = vec![StackLayer::AddSkip {
        slot: "missing".into(),
    }];
    assert!(WgpuInference::residual_stack_from_safetensors(&specs2, &st, 3, H, W).is_err());
}

#[test]
fn add_reference_is_elementwise() {
    assert_eq!(
        add_reference(&[1.0, 2.0, 3.0], &[0.5, -1.0, 10.0]),
        vec![1.5, 1.0, 13.0]
    );
}

// ---------------------------------------------------------------------------
// M856: transformer ops. The `[1, 3, 2, 4]` tensor the GPU preprocess leaves on
// the device is 24 f32s; the transformer layers read the same buffer as a
// `[1, 1, S, D]` token matrix (S = 4 tokens of D = 6 features). Same bytes, same
// order, so the CPU reference starts from the identical `nv12_to_rgb_tensor`
// values. The references walk the reductions in the shaders' order, but the GPU
// still contracts multiply-adds and evaluates `exp` its own way, so these check
// to ~1e-5 (measured worst case on a 3060: 4e-7) rather than bit-equality.
// ---------------------------------------------------------------------------

/// Sequence length and model dim the token tests read out of the 24-value
/// preprocess tensor.
const S: u32 = 4;
const D: u32 = 6;

fn token_caps() -> Caps {
    nchw_caps(&[1, 1, S, D])
}

/// Deterministic `[K, N]` input-major matrix and `[N]` bias, non-symmetric so a
/// transposed or collapsed index is caught.
fn matrix(k: u32, n: u32, phase: f32) -> (Vec<f32>, Vec<f32>) {
    let w = (0..k * n)
        .map(|i| (i as f32 * 0.37 + phase).sin() * 0.6)
        .collect();
    let b = (0..n).map(|i| i as f32 * 0.11 - 0.2 + phase).collect();
    (w, b)
}

/// The per-token matmul: `[S, D] . [D, N] + [N]`, the projection every
/// transformer layer is built from. Run on the GPU-resident preprocess tensor
/// read as a token matrix, checked against the CPU reference.
#[tokio::test]
async fn matmul_on_gpu_resident_tokens_matches_cpu_reference() {
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }
    let _gpu = gpu_guard().lock().await;
    const N: u32 = 5;
    let (weights, bias) = matrix(D, N, 0.3);

    let nv12 = sample_nv12();
    let tensor_frame = preprocess_to_gpu_tensor(nv12.clone()).await;
    let op = WgpuInference::matmul(S, D, N, weights.clone(), bias.clone()).expect("valid matmul");
    let out = run_op(op, token_caps(), tensor_frame).await;
    let got = logits_from_system(&out);

    let x = nv12_to_rgb_tensor(&nv12, W as usize, H as usize);
    let expected = matmul_reference(&x, S as usize, D as usize, N as usize, &weights, &bias);
    assert_eq!(got.len(), (S * N) as usize, "[S, N] token matrix");
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert!(
            (g - e).abs() < 1e-5,
            "matmul {i}: gpu {g} vs cpu reference {e}"
        );
    }
    // Every row must differ: a matmul that ignored the row index would repeat.
    assert!(
        got[..N as usize] != got[N as usize..2 * N as usize],
        "each token must get its own projection"
    );
}

/// LayerNorm over the last dim: each token normalized by its own mean and
/// variance, then the learned affine. Rows must come out zero-mean / unit-var
/// before the affine, which the reference encodes and the assertion below
/// double-checks with an identity gamma/beta run.
#[tokio::test]
async fn layer_norm_on_gpu_matches_cpu_reference() {
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }
    let _gpu = gpu_guard().lock().await;
    const EPS: f32 = 1e-5;
    let gamma = vec![1.3f32, 0.7, 1.0, 0.4, 2.0, -0.5];
    let beta = vec![0.1f32, -0.2, 0.0, 0.3, -0.1, 0.05];

    let nv12 = sample_nv12();
    let tensor_frame = preprocess_to_gpu_tensor(nv12.clone()).await;
    let op = WgpuInference::layer_norm(S, D, gamma.clone(), beta.clone(), EPS).expect("valid ln");
    let out = run_op(op, token_caps(), tensor_frame).await;
    let got = logits_from_system(&out);

    let x = nv12_to_rgb_tensor(&nv12, W as usize, H as usize);
    let expected = layer_norm_reference(&x, S as usize, D as usize, &gamma, &beta, EPS);
    assert_eq!(got.len(), (S * D) as usize, "shape-preserving");
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert!(
            (g - e).abs() < 1e-5,
            "layernorm {i}: gpu {g} vs cpu reference {e}"
        );
    }
    // With the affine removed each row is zero-mean and unit-variance: proves the
    // per-row statistics were used, not a global or per-column normalization.
    let plain = layer_norm_reference(&x, S as usize, D as usize, &[1.0; 6], &[0.0; 6], EPS);
    for row in plain.chunks_exact(D as usize) {
        let mean: f32 = row.iter().sum::<f32>() / D as f32;
        let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / D as f32;
        assert!(mean.abs() < 1e-4, "row mean {mean} must be ~0");
        assert!((var - 1.0).abs() < 1e-3, "row variance {var} must be ~1");
    }
}

/// Softmax over the last dim, standalone: each row sums to 1 and matches the
/// max-subtracted CPU reference.
#[tokio::test]
async fn softmax_on_gpu_matches_cpu_reference() {
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }
    let _gpu = gpu_guard().lock().await;
    let nv12 = sample_nv12();
    let tensor_frame = preprocess_to_gpu_tensor(nv12.clone()).await;
    let op = WgpuInference::softmax(S, D).expect("valid softmax");
    let out = run_op(op, token_caps(), tensor_frame).await;
    let got = logits_from_system(&out);

    let x = nv12_to_rgb_tensor(&nv12, W as usize, H as usize);
    let expected = softmax_reference(&x, S as usize, D as usize);
    assert_eq!(got.len(), (S * D) as usize, "shape-preserving");
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert!(
            (g - e).abs() < 1e-6,
            "softmax {i}: gpu {g} vs cpu reference {e}"
        );
    }
    for (r, row) in got.chunks_exact(D as usize).enumerate() {
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "row {r} sums to {sum}, not 1");
    }
}

/// Self-attention on the GPU for 1, 2 and 3 heads: the QKV projection runs as a
/// GPU-resident matmul (`[S, D] -> [S, 3D]`, the packed layout the attention op
/// indexes) and the attention pass reads it in place. Each head count is checked
/// against the CPU reference over the same projected values, so a head-stride or
/// softmax-axis error shows up as a mismatch rather than a plausible number.
#[tokio::test]
async fn attention_on_gpu_matches_cpu_reference_for_each_head_count() {
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }
    let _gpu = gpu_guard().lock().await;
    let nv12 = sample_nv12();
    let x = nv12_to_rgb_tensor(&nv12, W as usize, H as usize);
    let (qkv_w, qkv_b) = matrix(D, 3 * D, 0.9);

    for heads in [1u32, 2, 3] {
        let tensor_frame = preprocess_to_gpu_tensor(nv12.clone()).await;
        let proj = WgpuInference::matmul(S, D, 3 * D, qkv_w.clone(), qkv_b.clone())
            .expect("valid qkv projection")
            .with_gpu_output();
        let projected = run_op(proj, token_caps(), tensor_frame).await;
        assert!(
            matches!(projected.domain, MemoryDomain::WgpuBuffer(_)),
            "the packed QKV stays on the GPU for the attention pass"
        );

        let attn = WgpuInference::attention(S, D, heads).expect("valid attention");
        let out = run_op(attn, nchw_caps(&[1, 1, S, 3 * D]), projected).await;
        let got = logits_from_system(&out);

        let qkv = matmul_reference(&x, S as usize, D as usize, 3 * D as usize, &qkv_w, &qkv_b);
        let expected = attention_reference(&qkv, S as usize, D as usize, heads as usize);
        assert_eq!(got.len(), (S * D) as usize, "[S, D] context, {heads} heads");
        for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (g - e).abs() < 1e-5,
                "attention ({heads} heads) elem {i}: gpu {g} vs cpu reference {e}"
            );
        }
        // The context must be a genuine mix: with a non-degenerate input no two
        // tokens attend identically.
        assert!(
            got[..D as usize] != got[D as usize..2 * D as usize],
            "{heads} heads: tokens must get distinct contexts"
        );
    }
}

/// Head counts that do not divide the model dim (and zero dims) are rejected at
/// construction rather than reading past a head's slice at dispatch.
#[test]
fn attention_validates_head_split() {
    for heads in [1u32, 2, 3, 6] {
        assert!(WgpuInference::attention(4, 6, heads).is_ok());
    }
    for bad in [0u32, 4, 5, 7] {
        assert_eq!(
            WgpuInference::attention(4, 6, bad).err(),
            Some(G2gError::CapsMismatch),
            "{bad} heads must not split a 6-dim model"
        );
    }
    assert!(WgpuInference::attention(0, 6, 2).is_err());
    assert!(WgpuInference::attention(4, 0, 2).is_err());
    assert!(WgpuInference::layer_norm(4, 6, vec![1.0; 5], vec![0.0; 6], 1e-5).is_err());
    assert!(WgpuInference::matmul(4, 6, 2, vec![0.0; 11], vec![0.0; 2]).is_err());
}

/// A whole pre-norm transformer block imported from one safetensors file and run
/// GPU-resident through `ResidualStack`:
/// `x + attn(ln1(x))` then `+ mlp(ln2(.))`, with 2-head attention. The four
/// projections load under the torch names (`attn.q_proj.weight`, ...) and the
/// output projection is deliberately exported without a bias, so the
/// missing-bias path is exercised too. The read-back matches a CPU reference
/// that projects Q, K and V as three separate matmuls and interleaves them,
/// which is a different code path from the packed `[D, 3D]` matrix the importer
/// builds: a mis-packed projection would not agree.
#[tokio::test]
async fn transformer_block_from_safetensors_matches_cpu_reference() {
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }
    let _gpu = gpu_guard().lock().await;
    const EPS: f32 = 1e-5;
    const HEADS: usize = 2;
    const HIDDEN: u32 = 12;

    let ln1_g = vec![1.2f32, 0.8, 1.0, 0.9, 1.1, 0.7];
    let ln1_b = vec![0.0f32, 0.05, -0.05, 0.1, 0.0, -0.1];
    let ln2_g = vec![0.9f32, 1.0, 1.3, 0.6, 1.0, 1.1];
    let ln2_b = vec![0.02f32, 0.0, -0.03, 0.0, 0.07, 0.0];
    let (qw, qb) = matrix(D, D, 0.1);
    let (kw, kb) = matrix(D, D, 1.7);
    let (vw, vb) = matrix(D, D, 2.9);
    let (ow, _) = matrix(D, D, 3.3);
    let ob = vec![0f32; D as usize]; // o_proj.bias is absent from the file
    let (fc1_w, fc1_b) = matrix(D, HIDDEN, 0.5);
    let (fc2_w, fc2_b) = matrix(HIDDEN, D, 1.1);

    let d = D as usize;
    let blob = serialize(&[
        ("ln1.weight", &[d], &ln1_g),
        ("ln1.bias", &[d], &ln1_b),
        ("attn.q_proj.weight", &[d, d], &qw),
        ("attn.q_proj.bias", &[d], &qb),
        ("attn.k_proj.weight", &[d, d], &kw),
        ("attn.k_proj.bias", &[d], &kb),
        ("attn.v_proj.weight", &[d, d], &vw),
        ("attn.v_proj.bias", &[d], &vb),
        ("attn.o_proj.weight", &[d, d], &ow),
        ("ln2.weight", &[d], &ln2_g),
        ("ln2.bias", &[d], &ln2_b),
        ("mlp.fc1.weight", &[d, HIDDEN as usize], &fc1_w),
        ("mlp.fc1.bias", &[HIDDEN as usize], &fc1_b),
        ("mlp.fc2.weight", &[HIDDEN as usize, d], &fc2_w),
        ("mlp.fc2.bias", &[d], &fc2_b),
    ]);
    let st = SafeTensors::parse(&blob).expect("parse transformer weights");

    let specs = vec![
        StackLayer::SaveSkip {
            slot: "attn_in".into(),
        },
        StackLayer::LayerNorm {
            name: "ln1".into(),
            eps: EPS,
        },
        StackLayer::Attention {
            name: "attn".into(),
            heads: HEADS as u32,
        },
        StackLayer::AddSkip {
            slot: "attn_in".into(),
        },
        StackLayer::SaveSkip {
            slot: "mlp_in".into(),
        },
        StackLayer::LayerNorm {
            name: "ln2".into(),
            eps: EPS,
        },
        StackLayer::Matmul {
            name: "mlp.fc1".into(),
        },
        StackLayer::Relu,
        StackLayer::Matmul {
            name: "mlp.fc2".into(),
        },
        StackLayer::AddSkip {
            slot: "mlp_in".into(),
        },
    ];
    let mut stack = WgpuInference::residual_stack_from_safetensors(&specs, &st, 1, S, D)
        .expect("import the transformer block from one file");

    let nv12 = sample_nv12();
    let frame = preprocess_to_gpu_tensor(nv12.clone()).await;
    let out = stack.run(frame).expect("run the block on the GPU");
    let got = logits_from_system(&out);

    // CPU reference over the same 24 values, read as [S, D] tokens.
    let (s, d) = (S as usize, D as usize);
    let x = nv12_to_rgb_tensor(&nv12, W as usize, H as usize);
    let n1 = layer_norm_reference(&x, s, d, &ln1_g, &ln1_b, EPS);
    let q = matmul_reference(&n1, s, d, d, &qw, &qb);
    let k = matmul_reference(&n1, s, d, d, &kw, &kb);
    let v = matmul_reference(&n1, s, d, d, &vw, &vb);
    // Interleave Q, K, V per token into the packed [S, 3D] layout.
    let mut qkv = Vec::with_capacity(s * 3 * d);
    for t in 0..s {
        for m in [&q, &k, &v] {
            qkv.extend_from_slice(&m[t * d..(t + 1) * d]);
        }
    }
    let ctx = attention_reference(&qkv, s, d, HEADS);
    let attn_out = matmul_reference(&ctx, s, d, d, &ow, &ob);
    let after_attn = add_reference(&attn_out, &x);
    let n2 = layer_norm_reference(&after_attn, s, d, &ln2_g, &ln2_b, EPS);
    let h1 = relu_reference(&matmul_reference(
        &n2,
        s,
        d,
        HIDDEN as usize,
        &fc1_w,
        &fc1_b,
    ));
    let h2 = matmul_reference(&h1, s, HIDDEN as usize, d, &fc2_w, &fc2_b);
    let expected = add_reference(&h2, &after_attn);

    assert_eq!(got.len(), s * d, "the block preserves the [S, D] shape");
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert!(
            (g - e).abs() < 1e-5,
            "block elem {i}: gpu {g} vs cpu reference {e}"
        );
    }
    // Both residuals must have been added: the output cannot equal the MLP branch
    // alone, nor the attention branch alone.
    assert!(
        got.iter().zip(&h2).any(|(g, b)| (g - b).abs() > 1e-4),
        "the MLP skip must be added"
    );
    assert!(
        got.iter().zip(&attn_out).any(|(g, b)| (g - b).abs() > 1e-4),
        "the attention skip must be added"
    );
}

/// The transformer layers read the running tensor as `[1, 1, S, D]` tokens, so a
/// multi-channel feature map is rejected at import instead of being silently
/// reinterpreted, and a missing / mis-shaped projection fails loud.
#[test]
fn transformer_import_validates_shape_and_tensors() {
    let d = D as usize;
    let (qw, qb) = matrix(D, D, 0.1);
    let blob = serialize(&[
        ("attn.q_proj.weight", &[d, d], &qw),
        ("attn.q_proj.bias", &[d], &qb),
        ("attn.k_proj.weight", &[d, d], &qw),
        ("attn.v_proj.weight", &[d, d], &qw),
        ("attn.o_proj.weight", &[d, d], &qw),
        ("ln1.weight", &[d], &qb),
        ("ln1.bias", &[d], &qb),
    ]);
    let st = SafeTensors::parse(&blob).unwrap();

    let attn = vec![StackLayer::Attention {
        name: "attn".into(),
        heads: 2,
    }];
    assert!(WgpuInference::stack_from_safetensors(&attn, &st, 1, S, D).is_ok());
    // C != 1: not a token matrix.
    assert_eq!(
        WgpuInference::stack_from_safetensors(&attn, &st, 3, S, D).err(),
        Some(G2gError::CapsMismatch),
        "a [1, C, H, W] feature map is not a token matrix"
    );
    let ln = vec![StackLayer::LayerNorm {
        name: "ln1".into(),
        eps: 1e-5,
    }];
    assert!(WgpuInference::stack_from_safetensors(&ln, &st, 1, S, D).is_ok());
    assert!(WgpuInference::stack_from_safetensors(&ln, &st, 2, S, D).is_err());
    // A projection the file does not carry.
    let missing = vec![StackLayer::Attention {
        name: "other".into(),
        heads: 2,
    }];
    assert!(WgpuInference::stack_from_safetensors(&missing, &st, 1, S, D).is_err());
    // A matmul whose K does not match the running feature count.
    let mm = vec![StackLayer::Matmul {
        name: "attn.q_proj".into(),
    }];
    assert!(WgpuInference::stack_from_safetensors(&mm, &st, 1, S, D).is_ok());
    assert!(WgpuInference::stack_from_safetensors(&mm, &st, 1, S, D + 1).is_err());
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
    assert_eq!(
        WgpuInference::avgpool2d(3, 2, 4, 2, 2, 0, 1).err(),
        Some(G2gError::CapsMismatch)
    );
    assert_eq!(
        WgpuInference::relu(0, 2, 4).err(),
        Some(G2gError::CapsMismatch)
    );
}

#[test]
fn conv2d_overflowing_dims_fail_loud_not_panic() {
    // conv2d dims can come from an untrusted safetensors shape. A kernel whose
    // element-count product overflows must return CapsMismatch, not panic
    // (debug) or wrap to a value that admits a short weight buffer / undersized
    // GPU buffers. 65536^4 overflows u64, so the weight-length fold rejects it.
    assert_eq!(
        WgpuInference::conv2d(
            0x10000,
            0x10000,
            0x10000,
            0x10000,
            0x10000,
            0x10000,
            vec![],
            vec![]
        )
        .err(),
        Some(G2gError::CapsMismatch),
        "overflowing conv2d geometry must fail loud"
    );
    // Valid kernel dims but a spatial size whose in/out element count overflows
    // usize must also fail at the size fold rather than panicking.
    assert_eq!(
        WgpuInference::conv2d(
            3,
            3,
            3,
            3,
            0xFFFF_FFFF,
            0xFFFF_FFFF,
            vec![0.0; 81],
            vec![0.0; 3]
        )
        .err(),
        Some(G2gError::CapsMismatch),
        "overflowing conv2d spatial size must fail loud"
    );
}
