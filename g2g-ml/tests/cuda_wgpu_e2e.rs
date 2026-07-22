#![cfg(all(target_os = "linux", feature = "cuda-wgpu-e2e"))]
//! M220 Stage 1 end-to-end: the zero-copy GPU keep-on-GPU pipeline.
//!
//! ```text
//! H.264 -> FfmpegVideoDec(NvdecCuda) -> CudaToWgpu -> WgpuPreprocess -> WgpuInference
//!          (MemoryDomain::Cuda)         (-> WgpuTexture, (-> GPU tensor)  (-> logits)
//!                                         no PCIe download)
//! ```
//!
//! Unlike the Stage 0 scaffold (which paid a device->host download via
//! `CudaDownload`), `CudaToWgpu` copies the NVDEC NV12 planes device->device
//! into a Vulkan image shared with CUDA and hands `WgpuPreprocess` a GPU
//! texture: the pixels never touch the CPU. Each decoded frame is shared (M213
//! refcount) so one copy drives this zero-copy path and one copy is downloaded
//! purely to compute the CPU reference the logits must match.
//!
//! Any Annex-B H.264 elementary stream works as a fixture; generate one with:
//!
//! ```sh
//! ffmpeg -f lavfi -i testsrc=size=320x240:rate=30:duration=1 -c:v libx264 \
//!     -pix_fmt yuv420p -g 15 -bsf:v h264_mp4toannexb -f h264 /tmp/clip.h264
//! G2G_H264_FIXTURE=/tmp/clip.h264 cargo test -p g2g-ml \
//!     --features cuda-wgpu-e2e --test cuda_wgpu_e2e -- --nocapture
//! ```
//!
//! Skips when the fixture env var is unset or no wgpu adapter is present.
//! Validated on an RTX 3060 (M251): all frames match the CPU reference, no
//! PCIe download.

use std::time::Instant;

use g2g_core::element::{BoxFuture, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::MemoryDomain;
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, G2gError, OutputSink, Rate, RawVideoFormat,
    SystemSlice, TensorDType, TensorLayout, TensorShape, VideoCodec,
};
use g2g_ml::cudatowgpu::CudaToWgpu;
use g2g_ml::wgpuinfer::{linear_reference, WgpuInference};
use g2g_ml::wgpupreprocess::{gpu_available, nv12_to_rgb_tensor, WgpuPreprocess};
use g2g_plugins::cuda::CudaDownload;
use g2g_plugins::ffmpegdec::{Backend, FfmpegVideoDec};

const MAX_FRAMES: usize = 30;
const N: usize = 2;

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

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

fn nv12_caps(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn tensor_caps(w: u32, h: u32) -> Caps {
    Caps::Tensor {
        dtype: TensorDType::F32,
        shape: TensorShape::new([1, 3, h, w]),
        layout: TensorLayout::Nchw,
    }
}

fn weights_bias(k: usize) -> (Vec<f32>, Vec<f32>) {
    let mut weights = vec![0f32; k * N];
    for i in 0..k {
        weights[i * N] = 1.0;
        weights[i * N + 1] = i as f32 * 1e-4;
    }
    (weights, vec![0.5, -0.25])
}

fn data_frames(packets: Vec<PipelinePacket>) -> Vec<Frame> {
    packets
        .into_iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f),
            _ => None,
        })
        .collect()
}

fn first_nv12_dims(packets: &[PipelinePacket]) -> Option<(u32, u32)> {
    packets.iter().find_map(|p| match p {
        PipelinePacket::CapsChanged(Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            ..
        }) => Some((*w, *h)),
        _ => None,
    })
}

fn logits_from(frame: &Frame) -> Vec<f32> {
    let Some(slice) = frame.domain.as_system_slice() else {
        panic!(
            "inference must read logits back to System, got {:?}",
            frame.domain.kind()
        );
    };
    slice
        .as_slice()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn p50_ms(mut s: Vec<f64>) -> f64 {
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

/// Annex-B access-unit splitter (see the Stage 0 scaffold for why H264Parse
/// can't do this): a new AU starts at the first VCL slice once the current AU
/// has one.
fn split_access_units(bs: &[u8]) -> Vec<Vec<u8>> {
    let mut codes: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i + 3 <= bs.len() {
        if bs[i] == 0 && bs[i + 1] == 0 && bs[i + 2] == 1 {
            codes.push((i, i + 3));
            i += 3;
        } else if i + 4 <= bs.len()
            && bs[i] == 0
            && bs[i + 1] == 0
            && bs[i + 2] == 0
            && bs[i + 3] == 1
        {
            codes.push((i, i + 4));
            i += 4;
        } else {
            i += 1;
        }
    }
    let mut aus = Vec::new();
    let mut start: Option<usize> = None;
    let mut has_vcl = false;
    for &(sc, nal) in &codes {
        let is_vcl = (1..=5).contains(&(bs[nal] & 0x1f));
        if is_vcl && has_vcl {
            aus.push(bs[start.take().unwrap()..sc].to_vec());
            has_vcl = false;
        }
        if start.is_none() {
            start = Some(sc);
        }
        has_vcl |= is_vcl;
    }
    if let Some(s) = start {
        aus.push(bs[s..].to_vec());
    }
    aus
}

async fn run_preprocess_infer(
    pre: &mut WgpuPreprocess,
    infer: &mut WgpuInference,
    frame: Frame,
) -> Vec<f32> {
    let mut pout = Collect::default();
    pre.process(PipelinePacket::DataFrame(frame), &mut pout)
        .await
        .expect("preprocess");
    let tensor = data_frames(pout.packets)
        .into_iter()
        .next()
        .expect("GPU tensor");
    assert!(
        matches!(tensor.domain, MemoryDomain::WgpuBuffer(_)),
        "tensor stays on the GPU"
    );
    let mut iout = Collect::default();
    infer
        .process(PipelinePacket::DataFrame(tensor), &mut iout)
        .await
        .expect("infer");
    logits_from(data_frames(iout.packets).first().expect("logits"))
}

#[tokio::test]
async fn cuda_to_wgpu_zero_copy_matches_cpu_reference() {
    let Some(path) = std::env::var_os("G2G_H264_FIXTURE") else {
        eprintln!("skipping: set G2G_H264_FIXTURE=/path/to/clip.h264 to run");
        return;
    };
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }
    let bitstream = std::fs::read(&path).expect("read fixture");
    let access_units = split_access_units(&bitstream);
    assert!(!access_units.is_empty(), "no access units");

    // Decode on the GPU: NV12 stays in CUDA device memory.
    let mut dec = FfmpegVideoDec::new().with_backend(Backend::NvdecCuda);
    let narrowed = dec.intercept_caps(&h264_caps()).expect("H.264 supported");
    assert!(matches!(
        dec.configure_pipeline(&narrowed).expect("NVDEC init"),
        ConfigureOutcome::Accepted
    ));
    let mut decoded = Collect::default();
    for (seq, au) in access_units.into_iter().enumerate() {
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(au.into_boxed_slice())),
            timing: FrameTiming::default(),
            sequence: seq as u64,
            meta: Default::default(),
        };
        dec.process(PipelinePacket::DataFrame(frame), &mut decoded)
            .await
            .expect("decode");
    }
    dec.process(PipelinePacket::Eos, &mut decoded)
        .await
        .expect("flush");

    let (w, h) = first_nv12_dims(&decoded.packets).expect("NV12 caps");
    eprintln!("NVDEC decoded NV12 {w}x{h}");
    let cuda_frames = data_frames(decoded.packets);
    assert!(
        matches!(
            cuda_frames.first().map(|f| &f.domain),
            Some(MemoryDomain::Cuda(_))
        ),
        "NvdecCuda"
    );

    let k = 3 * w as usize * h as usize;
    let (weights, bias) = weights_bias(k);

    // Reference branch: download to NV12 system bytes (only to derive the CPU
    // reference; the zero-copy branch never downloads).
    let mut download = CudaDownload::new();
    download
        .configure_pipeline(&nv12_caps(w, h))
        .expect("configure download");

    // Zero-copy branch under test.
    let mut bridge = CudaToWgpu::new();
    bridge
        .configure_pipeline(&nv12_caps(w, h))
        .expect("configure bridge");
    let mut pre = WgpuPreprocess::new().with_gpu_output();
    pre.configure_pipeline(&nv12_caps(w, h))
        .expect("configure preprocess");
    let mut infer = WgpuInference::linear(w, h, weights.clone(), bias.clone()).expect("linear");
    infer
        .configure_pipeline(&tensor_caps(w, h))
        .expect("configure inference");

    // G2G_CUDAWGPU_NOPOOL=1 drops the reuse pool before each frame, forcing the
    // per-frame allocate + CUDA-import path, to A/B the pool against it.
    let nopool = std::env::var_os("G2G_CUDAWGPU_NOPOOL").is_some();

    let mut gpu_ms = Vec::new();
    let mut bridge_ms = Vec::new();
    let mut matched = 0usize;

    for frame in cuda_frames.into_iter().take(MAX_FRAMES) {
        // Share the CUDA frame (refcount): one copy downloads for the reference,
        // one drives the zero-copy bridge.
        let ref_frame = Frame {
            domain: frame.domain.share(),
            timing: frame.timing,
            sequence: frame.sequence,
            meta: Default::default(),
        };
        let mut dl = Collect::default();
        download
            .process(PipelinePacket::DataFrame(ref_frame), &mut dl)
            .await
            .expect("download");
        let nv12 = data_frames(dl.packets)
            .into_iter()
            .next()
            .expect("downloaded NV12");
        let Some(slice) = nv12.domain.as_system_slice() else {
            panic!("System NV12")
        };
        let cpu_tensor = nv12_to_rgb_tensor(slice.as_slice(), w as usize, h as usize);
        let expected = linear_reference(&cpu_tensor, &weights, &bias);

        // Zero-copy: bridge -> preprocess (surface-import) -> infer.
        if nopool {
            bridge.reset_pool();
        }
        let t0 = Instant::now();
        let mut bout = Collect::default();
        let tb = Instant::now();
        bridge
            .process(PipelinePacket::DataFrame(frame), &mut bout)
            .await
            .expect("bridge");
        bridge_ms.push(tb.elapsed().as_secs_f64() * 1e3);
        let tex_frame = data_frames(bout.packets)
            .into_iter()
            .next()
            .expect("wgpu texture frame");
        assert!(
            matches!(tex_frame.domain, MemoryDomain::WgpuTexture(_)),
            "bridge must emit a GPU texture (no download)"
        );
        let got = run_preprocess_infer(&mut pre, &mut infer, tex_frame).await;
        gpu_ms.push(t0.elapsed().as_secs_f64() * 1e3);

        assert_eq!(got.len(), N);
        for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
            let tol = 1e-2 * e.abs().max(1.0) + 1e-1;
            assert!(
                (g - e).abs() <= tol,
                "logit {i}: zero-copy {g} vs cpu reference {e} (tol {tol})"
            );
        }
        assert!((got[0] - got[1]).abs() > 1e-3, "outputs must differ");
        matched += 1;
    }

    assert!(matched > 0, "no frames bridged");
    assert_eq!(
        matched as u64,
        bridge.converted(),
        "every frame went through the bridge"
    );
    eprintln!(
        "zero-copy: bridged + matched {matched} frame(s) against CPU reference; \
         p50 bridge+preprocess+infer {:.2} ms (no PCIe download)",
        p50_ms(gpu_ms),
    );
    eprintln!(
        "bridge step only ({}): p50 {:.2} ms",
        if nopool {
            "per-frame alloc, G2G_CUDAWGPU_NOPOOL"
        } else {
            "pooled reuse"
        },
        p50_ms(bridge_ms),
    );
}
