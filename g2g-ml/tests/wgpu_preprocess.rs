#![cfg(feature = "wgpu")]
//! §5.1: `WgpuPreprocess` converts an NV12 frame to a normalized f32 NCHW RGB
//! tensor in a wgpu compute shader. The test runs a known NV12 frame (distinct
//! luma, one neutral and one coloured chroma block) through the element on the
//! real GPU and asserts the read-back tensor matches the host BT.601 reference
//! within float tolerance, with the tensor caps emitted once. Skips when no
//! wgpu adapter is present, like the other hardware-gated elements.

use g2g_core::element::{BoxFuture, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, Caps, Dim, G2gError, OutputSink, Rate, RawVideoFormat, TensorDType, TensorLayout,
    TensorShape,
};
use g2g_ml::wgpupreprocess::{gpu_available, nv12_to_rgb_tensor, WgpuPreprocess};

fn nv12_caps(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
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

#[tokio::test]
async fn gpu_nv12_to_rgb_tensor_matches_cpu_reference() {
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }

    let (w, h) = (4usize, 2usize);
    // Y plane (w*h), then interleaved Cb,Cr per 2x2 block (two blocks).
    let y_plane = [16u8, 81, 145, 235, 41, 100, 200, 128];
    let uv_plane = [128u8, 128, 90, 200]; // block 0 neutral, block 1 coloured
    let nv12: Vec<u8> = y_plane.iter().chain(&uv_plane).copied().collect();
    assert_eq!(nv12.len(), w * h * 3 / 2);

    let mut element = WgpuPreprocess::new();
    element
        .configure_pipeline(&nv12_caps(w as u32, h as u32))
        .expect("configure NV12 geometry");

    let mut out = Collect::default();
    element
        .process(PipelinePacket::DataFrame(nv12_frame(nv12.clone(), 4242, 7)), &mut out)
        .await
        .expect("gpu preprocess frame 1");
    // a second frame: caps must not re-emit.
    element
        .process(PipelinePacket::DataFrame(nv12_frame(nv12.clone(), 9001, 8)), &mut out)
        .await
        .expect("gpu preprocess frame 2");

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
            shape: TensorShape(vec![1, 3, h as u32, w as u32]),
            layout: TensorLayout::Nchw,
        }
    );
    assert_eq!(frames.len(), 2);

    let expected = nv12_to_rgb_tensor(&nv12, w, h);
    let got = frame_f32(frames[0]);
    assert_eq!(got.len(), 3 * w * h, "NCHW RGB tensor length");
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert!(
            (g - e).abs() < 1e-3,
            "element {i}: gpu {g} vs cpu reference {e}"
        );
    }
    // a coloured chroma block must produce non-equal R/G/B somewhere (proves
    // the chroma path ran, not just luma).
    let area = w * h;
    let colored = (0..area).any(|px| (got[px] - got[area + px]).abs() > 1e-3);
    assert!(colored, "coloured block should break the grayscale R==G==B");

    // timing is inherited from the source frame.
    assert_eq!(frames[0].timing.pts_ns, 4242);
    assert_eq!(frames[1].timing.pts_ns, 9001);
    assert_eq!(element.emitted(), 2);
}
