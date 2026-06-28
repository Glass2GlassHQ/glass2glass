//! M353: `CudaUpload` (System NV12 -> CUDA), the host->device mirror of
//! `CudaDownload` and the converter that lets a CPU-side NV12 stream feed
//! `NvEnc`, which ingests `MemoryDomain::Cuda` only. Validated on a real NVIDIA
//! GPU; skips gracefully (no panic) where CUDA is unavailable, so it is a no-op
//! on a machine without the hardware.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use core::future::Future;
use core::pin::Pin;
use std::boxed::Box;

use g2g_core::element::{BoxFuture, PushOutcome};
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, Dim, Frame, FrameTiming, G2gError, MemoryDomain, MemoryDomainKind,
    OutputSink, PipelinePacket, Rate, RawVideoFormat,
};
use g2g_plugins::cuda::CudaUpload;

// NVENC has a minimum encode resolution, so use a size the encoder accepts (the
// same 320x240 the NvEnc round-trip test uses); the upload itself is size-agnostic.
const W: u32 = 320;
const H: u32 = 240;

fn nv12_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(W),
        height: Dim::Fixed(H),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// A packed system-memory NV12 frame (gradient luma, neutral chroma).
fn system_nv12_frame(seq: u64) -> Frame {
    let total = (W * H + 2 * (W / 2) * (H / 2)) as usize;
    let mut buf = vec![0u8; total];
    for (i, b) in buf[..(W * H) as usize].iter_mut().enumerate() {
        *b = ((i as u64 + seq * 7) & 0xff) as u8;
    }
    for b in &mut buf[(W * H) as usize..] {
        *b = 128;
    }
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(buf.into_boxed_slice())),
        timing: FrameTiming { pts_ns: seq * 33_000_000, ..FrameTiming::default() },
        sequence: seq,
        meta: Default::default(),
    }
}

#[derive(Default)]
struct Collect {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for Collect {
    fn push<'a>(&'a mut self, packet: PipelinePacket) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            self.packets.push(packet);
            Ok(PushOutcome::Accepted)
        })
    }
}

fn skip_if_no_gpu(err: &G2gError) -> bool {
    if matches!(err, G2gError::Hardware(_)) {
        std::eprintln!("skipping m353: CUDA unavailable ({err:?})");
        true
    } else {
        false
    }
}

/// A System NV12 frame uploaded by `CudaUpload` comes out as `MemoryDomain::Cuda`
/// with the configured geometry, the host->device promotion.
#[tokio::test]
async fn cuda_upload_promotes_system_nv12_to_cuda() {
    let mut up = CudaUpload::new();
    match up.configure_pipeline(&nv12_caps()) {
        Ok(_) => {}
        Err(e) if skip_if_no_gpu(&e) => return,
        Err(e) => panic!("unexpected configure error: {e:?}"),
    }

    let mut out = Collect::default();
    up.process(PipelinePacket::DataFrame(system_nv12_frame(0)), &mut out)
        .await
        .expect("upload a system NV12 frame");

    let frame = out.packets.iter().find_map(|p| match p {
        PipelinePacket::DataFrame(f) => Some(f),
        _ => None,
    });
    let frame = frame.expect("upload emitted a frame");
    assert_eq!(frame.domain.kind(), MemoryDomainKind::Cuda, "frame promoted to CUDA");
    if let MemoryDomain::Cuda(buf) = &frame.domain {
        assert_eq!((buf.width, buf.height), (W, H), "geometry preserved");
    }
    assert_eq!(up.uploaded(), 1, "one host->device upload");
    assert_eq!(up.forwarded(), 0, "no pass-through (input was System)");
}

/// A frame already in CUDA memory passes through untouched (no redundant copy),
/// so the converter is a safe no-op on a GPU-resident path. Driven by routing a
/// `CudaUpload` output back into a second `CudaUpload`.
#[tokio::test]
async fn cuda_upload_passes_through_gpu_frames() {
    let mut up = CudaUpload::new();
    match up.configure_pipeline(&nv12_caps()) {
        Ok(_) => {}
        Err(e) if skip_if_no_gpu(&e) => return,
        Err(e) => panic!("unexpected configure error: {e:?}"),
    }
    let mut mid = Collect::default();
    up.process(PipelinePacket::DataFrame(system_nv12_frame(0)), &mut mid).await.expect("upload");
    let cuda_frame = mid
        .packets
        .into_iter()
        .find_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f),
            _ => None,
        })
        .expect("uploaded frame");

    let mut pass = CudaUpload::new();
    pass.configure_pipeline(&nv12_caps()).expect("second upload configures (GPU already up)");
    let mut out = Collect::default();
    pass.process(PipelinePacket::DataFrame(cuda_frame), &mut out).await.expect("pass through");
    assert_eq!(pass.uploaded(), 0, "no upload for an already-CUDA frame");
    assert_eq!(pass.forwarded(), 1, "GPU frame forwarded untouched");
}

/// The headline: a CPU-side NV12 stream feeds `NvEnc` (CUDA-only input) through
/// `CudaUpload`, producing H.264 Annex-B. Answers "do we need a CudaUpload for
/// NvEnc" with a working pipeline.
#[cfg(feature = "nvenc")]
#[tokio::test]
async fn system_nv12_feeds_nvenc_via_cuda_upload() {
    use g2g_plugins::nvenc::NvEnc;

    let mut up = CudaUpload::new();
    match up.configure_pipeline(&nv12_caps()) {
        Ok(_) => {}
        Err(e) if skip_if_no_gpu(&e) => return,
        Err(e) => panic!("unexpected configure error: {e:?}"),
    }
    let mut enc = NvEnc::new();
    match enc.configure_pipeline(&nv12_caps()) {
        Ok(_) => {}
        Err(e) if skip_if_no_gpu(&e) => return,
        Err(e) => panic!("unexpected NvEnc configure error: {e:?}"),
    }

    // Upload each System frame to CUDA, then feed the device-resident frame to
    // the encoder.
    let mut aus: Vec<Vec<u8>> = Vec::new();
    for seq in 0..8u64 {
        let mut mid = Collect::default();
        up.process(PipelinePacket::DataFrame(system_nv12_frame(seq)), &mut mid)
            .await
            .expect("upload frame");
        for p in mid.packets {
            if let PipelinePacket::DataFrame(f) = &p {
                assert_eq!(f.domain.kind(), MemoryDomainKind::Cuda, "NvEnc requires CUDA input");
            }
            let mut enc_out = CollectAus { aus: &mut aus };
            enc.process(p, &mut enc_out).await.expect("encode frame");
        }
    }
    // Flush.
    let mut enc_out = CollectAus { aus: &mut aus };
    enc.process(PipelinePacket::Eos, &mut enc_out).await.expect("flush encoder");

    assert!(!aus.is_empty(), "encoder produced H.264 access units");
    assert!(
        aus.iter().any(|au| au.windows(3).any(|w| w == [0, 0, 1])),
        "access units carry Annex-B start codes",
    );
}

/// Sink that records the System-memory H.264 access units the encoder emits.
#[cfg(feature = "nvenc")]
struct CollectAus<'a> {
    aus: &'a mut Vec<Vec<u8>>,
}

#[cfg(feature = "nvenc")]
impl OutputSink for CollectAus<'_> {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let MemoryDomain::System(s) = &f.domain {
                    self.aus.push(s.as_slice().to_vec());
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}
