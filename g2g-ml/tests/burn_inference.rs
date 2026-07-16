#![cfg(feature = "burn")]
//! §5.2: `BurnInference` runs a linear layer (`input . W + b`) on burn's wgpu
//! backend. The test feeds a known RGBA frame through the element on the real
//! GPU and asserts the `[1, N]` logits match a CPU matmul of the same
//! deterministic weights within float tolerance, with the tensor caps emitted
//! once and timing inherited. Skips when burn's wgpu backend has no adapter.

use g2g_core::element::{BoxFuture, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, Caps, Dim, G2gError, OutputSink, Rate, RawVideoFormat, TensorDType, TensorLayout,
    TensorShape,
};
use g2g_ml::burninfer::{gpu_available, normalize_rgba_nchw, BurnInference};

const W: u32 = 2;
const H: u32 = 2;
const N: usize = 2;

fn rgba_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(W),
        height: Dim::Fixed(H),
        framerate: Rate::Fixed(30 << 16),
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

fn frame_f32(f: &Frame) -> Vec<f32> {
    let MemoryDomain::System(slice) = &f.domain else {
        panic!("tensor frame must be System memory");
    };
    slice
        .as_slice()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn rgba_frame(bytes: Vec<u8>, pts_ns: u64, sequence: u64) -> Frame {
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

/// Reference `input . W + b` on the CPU for the same weights the element uses.
fn cpu_linear(rgba: &[u8], weights: &[f32], bias: &[f32]) -> Vec<f32> {
    let flat = normalize_rgba_nchw(rgba, W as usize, H as usize);
    (0..N)
        .map(|n| {
            let dot: f32 = flat
                .iter()
                .enumerate()
                .map(|(k, x)| x * weights[k * N + n])
                .sum();
            dot + bias[n]
        })
        .collect()
}

#[tokio::test]
async fn gpu_linear_matches_cpu_matmul() {
    if !gpu_available() {
        eprintln!("skipping: no burn wgpu adapter on this host");
        return;
    }

    let k = 3 * (W * H) as usize; // 12
    let weights: Vec<f32> = (0..k * N).map(|i| i as f32 * 0.01).collect();
    let bias: Vec<f32> = vec![0.5, -0.25];
    let rgba: Vec<u8> = vec![
        10, 20, 30, 255, 40, 50, 60, 255, 70, 80, 90, 255, 100, 110, 120, 255,
    ];

    let mut element =
        BurnInference::linear(W, H, weights.clone(), bias.clone()).expect("valid linear layer");
    element.configure_pipeline(&rgba_caps()).expect("configure");

    let mut out = Collect::default();
    element
        .process(PipelinePacket::DataFrame(rgba_frame(rgba.clone(), 1234, 5)), &mut out)
        .await
        .expect("burn infer frame 1");
    // second frame: caps must not re-emit.
    element
        .process(PipelinePacket::DataFrame(rgba_frame(rgba.clone(), 5678, 6)), &mut out)
        .await
        .expect("burn infer frame 2");

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
            shape: TensorShape::new([1, N as u32]),
            layout: TensorLayout::Nchw,
        }
    );
    assert_eq!(frames.len(), 2);

    let expected = cpu_linear(&rgba, &weights, &bias);
    let got = frame_f32(frames[0]);
    assert_eq!(got.len(), N, "[1, N] logits");
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert!(
            (g - e).abs() < 1e-3,
            "logit {i}: gpu {g} vs cpu reference {e}"
        );
    }
    assert_eq!(frames[0].timing.pts_ns, 1234);
    assert_eq!(frames[1].timing.pts_ns, 5678);
    assert_eq!(element.inferred_count(), 2);
}
