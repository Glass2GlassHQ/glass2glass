#![cfg(all(target_os = "linux", feature = "cuda-wgpu-e2e"))]
//! M220 Stage 0: the CUDA<->wgpu correctness scaffold.
//!
//! Validates the *composition* of the GPU keep-on-GPU chain end to end on real
//! hardware, before the zero-copy `CudaToWgpu` bridge (Stage 1) exists:
//!
//! ```text
//! H.264 -> FfmpegVideoDec(NvdecCuda) -> CudaDownload -> WgpuPreprocess -> WgpuInference
//!          (MemoryDomain::Cuda)         (-> System NV12)  (-> GPU tensor)   (-> logits)
//! ```
//!
//! This pays the PCIe device->host download `CudaDownload` performs (that is
//! exactly the structural loss Stage 1 deletes), so it is NOT the zero-copy
//! payoff. Its job is to (1) prove the real NVDEC NV12 output flows correctly
//! through the inference chain at real video dimensions, by matching a full CPU
//! reference computed from the *same* downloaded NV12 bytes, and (2) print a
//! per-frame latency baseline the zero-copy version must beat.
//!
//! Each link is already independently tested (`m213_gpu_tee` for NvdecCuda,
//! `CudaDownload`'s own test, `wgpu_inference` for preprocess+infer on synthetic
//! NV12). This is the integration signal across all of them on the 3060.
//!
//! Run on a Linux + NVIDIA + wgpu host:
//!
//! ```sh
//! G2G_H264_FIXTURE=/path/to/clip.h264 cargo test -p g2g-enterprise \
//!     --features cuda-wgpu-e2e --test cuda_wgpu_scaffold -- --nocapture
//! ```
//!
//! Skips (does not fail) when the fixture env var is unset or no wgpu adapter is
//! present, matching the other hardware-gated tests in the workspace.

use std::time::Instant;

use g2g_core::element::{BoxFuture, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::MemoryDomain;
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, G2gError, OutputSink, Rate, RawVideoFormat,
    SystemSlice, TensorDType, TensorLayout, TensorShape, VideoCodec,
};
use g2g_plugins::cuda::CudaDownload;
use g2g_plugins::ffmpegdec::{Backend, FfmpegVideoDec};
use g2g_ml::wgpuinfer::{linear_reference, WgpuInference};
use g2g_ml::wgpupreprocess::{gpu_available, nv12_to_rgb_tensor, WgpuPreprocess};

/// Cap the GPU work so a long fixture still finishes quickly; 30 frames is the
/// synthetic testsrc clip, plenty for a composition + baseline signal.
const MAX_FRAMES: usize = 30;
/// Two linear outputs, as in the `wgpu_inference` reference test.
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
        shape: TensorShape(vec![1, 3, h, w]),
        layout: TensorLayout::Nchw,
    }
}

/// Deterministic `[K, N]` weights (row-major) + `[N]` bias. Column 0 sums every
/// input (so logit 0 ~ the tensor's total); column 1 is a position-weighted
/// ramp. The two columns differ, so a transposed / mis-indexed weight matrix is
/// caught. Identical math runs on the GPU and in `linear_reference`.
fn weights_bias(k: usize) -> (Vec<f32>, Vec<f32>) {
    let mut weights = vec![0f32; k * N];
    for i in 0..k {
        weights[i * N] = 1.0;
        weights[i * N + 1] = i as f32 * 1e-4;
    }
    (weights, vec![0.5, -0.25])
}

fn logits_from_system(f: &Frame) -> Vec<f32> {
    let MemoryDomain::System(slice) = &f.domain else {
        panic!("inference must read logits back to System, got {:?}", f.domain.kind());
    };
    slice.as_slice().chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
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

fn p50_ms(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

/// Split an Annex-B H.264 bitstream into access units (one coded picture each),
/// the shape libavcodec wants per `AVPacket`: feeding the whole stream at once
/// decodes only the first picture. A new AU starts at the first VCL slice NAL
/// (types 1..=5) once the current AU already holds one; non-VCL NALs (SPS/PPS/
/// SEI) attach to the AU that follows. Test-local because the crate's `annexb`
/// NAL iterator is `pub(crate)`.
fn split_access_units(bs: &[u8]) -> Vec<Vec<u8>> {
    // Start-code offsets: (start of 00 00 01 / 00 00 00 01, start of NAL byte).
    let mut codes: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i + 3 <= bs.len() {
        if bs[i] == 0 && bs[i + 1] == 0 && bs[i + 2] == 1 {
            codes.push((i, i + 3));
            i += 3;
        } else if i + 4 <= bs.len() && bs[i] == 0 && bs[i + 1] == 0 && bs[i + 2] == 0 && bs[i + 3] == 1
        {
            codes.push((i, i + 4));
            i += 4;
        } else {
            i += 1;
        }
    }
    let mut aus = Vec::new();
    let mut au_start: Option<usize> = None;
    let mut au_has_vcl = false;
    for &(sc, nal) in &codes {
        let is_vcl = (1..=5).contains(&(bs[nal] & 0x1f));
        if is_vcl && au_has_vcl {
            aus.push(bs[au_start.take().unwrap()..sc].to_vec());
            au_has_vcl = false;
        }
        if au_start.is_none() {
            au_start = Some(sc);
        }
        au_has_vcl |= is_vcl;
    }
    if let Some(s) = au_start {
        aus.push(bs[s..].to_vec());
    }
    aus
}

#[tokio::test]
async fn cuda_to_wgpu_scaffold_matches_cpu_reference() {
    let Some(path) = std::env::var_os("G2G_H264_FIXTURE") else {
        eprintln!("skipping: set G2G_H264_FIXTURE=/path/to/clip.h264 to run");
        return;
    };
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }
    let bitstream = std::fs::read(&path).expect("read H.264 fixture");
    assert!(!bitstream.is_empty(), "fixture is empty");

    let access_units = split_access_units(&bitstream);
    eprintln!("split into {} access unit(s)", access_units.len());
    assert!(!access_units.is_empty(), "no access units in fixture");

    // --- Decode on the GPU: NV12 stays in CUDA device memory. -------------
    let mut dec = FfmpegVideoDec::new().with_backend(Backend::NvdecCuda);
    let narrowed = dec.intercept_caps(&h264_caps()).expect("H.264 supported");
    assert!(matches!(
        dec.configure_pipeline(&narrowed).expect("NVDEC must initialise"),
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
        dec.process(PipelinePacket::DataFrame(frame), &mut decoded).await.expect("decode");
    }
    dec.process(PipelinePacket::Eos, &mut decoded).await.expect("flush decoder");

    let (w, h) = first_nv12_dims(&decoded.packets).expect("NVDEC must emit NV12 caps");
    eprintln!("NVDEC decoded NV12 {w}x{h}");
    let cuda_frames = data_frames(decoded.packets);
    eprintln!("NVDEC emitted {} frame(s)", cuda_frames.len());
    assert!(!cuda_frames.is_empty(), "NVDEC produced no frames");
    assert!(
        matches!(cuda_frames[0].domain, MemoryDomain::Cuda(_)),
        "NvdecCuda must keep frames in CUDA device memory, got {:?}",
        cuda_frames[0].domain.kind()
    );

    // --- Configure the consumer chain once for these dimensions. ----------
    let k = 3 * w as usize * h as usize;
    let (weights, bias) = weights_bias(k);

    let mut download = CudaDownload::new();
    assert!(matches!(
        download.configure_pipeline(&nv12_caps(w, h)).expect("configure download"),
        ConfigureOutcome::Accepted
    ));

    let mut pre = WgpuPreprocess::new().with_gpu_output();
    pre.configure_pipeline(&nv12_caps(w, h)).expect("configure preprocess");

    let mut infer = WgpuInference::linear(w, h, weights.clone(), bias.clone()).expect("linear");
    infer.configure_pipeline(&tensor_caps(w, h)).expect("configure inference");

    let mut download_ms = Vec::new();
    let mut gpu_ms = Vec::new();
    let mut matched = 0usize;

    for (idx, cuda_frame) in cuda_frames.into_iter().take(MAX_FRAMES).enumerate() {
        // Device -> host NV12 (the PCIe cost Stage 1 removes).
        let t0 = Instant::now();
        let mut dl = Collect::default();
        download.process(PipelinePacket::DataFrame(cuda_frame), &mut dl).await.expect("download");
        download_ms.push(t0.elapsed().as_secs_f64() * 1e3);

        let nv12_frame = data_frames(dl.packets).into_iter().next().expect("downloaded NV12");
        let MemoryDomain::System(slice) = &nv12_frame.domain else {
            panic!("CudaDownload must produce a System NV12 frame");
        };
        let nv12_bytes = slice.as_slice().to_vec();

        // NV12 -> GPU RGB tensor -> logits, tensor never leaving the device
        // between the two GPU elements.
        let t1 = Instant::now();
        let mut pre_out = Collect::default();
        pre.process(PipelinePacket::DataFrame(nv12_frame), &mut pre_out).await.expect("preprocess");
        let tensor_frame = data_frames(pre_out.packets).into_iter().next().expect("GPU tensor");
        assert!(
            matches!(tensor_frame.domain, MemoryDomain::WgpuBuffer(_)),
            "preprocess must hand off a GPU-resident tensor"
        );

        let mut infer_out = Collect::default();
        infer.process(PipelinePacket::DataFrame(tensor_frame), &mut infer_out).await.expect("infer");
        gpu_ms.push(t1.elapsed().as_secs_f64() * 1e3);

        let got = logits_from_system(
            data_frames(infer_out.packets).first().expect("a logits frame"),
        );
        assert_eq!(got.len(), N, "[1, N] logits");

        // Full CPU reference from the identical NV12 bytes: pins the whole
        // NVDEC -> download -> preprocess -> infer path numerically.
        let cpu_tensor = nv12_to_rgb_tensor(&nv12_bytes, w as usize, h as usize);
        let expected = linear_reference(&cpu_tensor, &weights, &bias);
        for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
            // Relative tolerance: the sum-column logit is ~1e5 at this K, where
            // GPU vs CPU accumulation order diverges by more than a fixed eps.
            let tol = 1e-2 * e.abs().max(1.0) + 1e-1;
            assert!(
                (g - e).abs() <= tol,
                "frame {idx} logit {i}: gpu {g} vs cpu reference {e} (tol {tol})"
            );
        }
        assert!((got[0] - got[1]).abs() > 1e-3, "the two outputs must differ");
        matched += 1;
    }

    assert!(matched > 0, "no frames made it through the chain");
    eprintln!(
        "matched {matched} frame(s) against CPU reference; \
         baseline p50: download {:.2} ms, gpu preprocess+infer {:.2} ms",
        p50_ms(download_ms),
        p50_ms(gpu_ms),
    );
}
