//! On-display smoke test for `CudaGlSink`, the CUDA-GL zero-copy-ish sink.
//!
//! Pipeline (file-driven, no RTSP so it runs fully in-session):
//!
//! ```text
//! H.264 file -> FfmpegVideoDec(NvdecCuda) -> CudaGlSink -> EGL/GL window
//!               (MemoryDomain::Cuda NV12)    (device->texture + NV12->RGB shader)
//! ```
//!
//! Unlike `wayland_smoke` (which uses `NvdecCuvid` -> system-memory NV12 ->
//! `WaylandSink`, a CPU convert + upload), this keeps the decoded NV12 in CUDA
//! device memory and presents it via CUDA-GL interop, the device-resident
//! display path.
//!
//! Ignored by default. Requires a running Wayland session, an NVIDIA GPU, and an
//! H.264 Annex-B fixture. On a hybrid (iGPU + NVIDIA) laptop the GL context must
//! land on the NVIDIA GPU or `cuGraphicsGLRegisterImage` fails, so render-offload
//! env vars are usually needed:
//!
//! ```sh
//! ffmpeg -f lavfi -i testsrc=size=320x240:rate=30:duration=2 -c:v libx264 \
//!     -pix_fmt yuv420p -g 15 -bsf:v h264_mp4toannexb -f h264 /tmp/clip.h264
//! __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
//!     __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
//!     G2G_H264_FIXTURE=/tmp/clip.h264 cargo test -p g2g-plugins \
//!     --features "ffmpeg cuda-gl" --test cudagl_smoke -- --ignored --nocapture
//! ```
//!
//! A window titled "glass2glass cudagl smoke" should show the test pattern.

#![cfg(all(target_os = "linux", feature = "ffmpeg", feature = "cuda-gl"))]

use g2g_core::element::{BoxFuture, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::MemoryDomain;
use g2g_core::metrics::monotonic_ns;
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, G2gError, OutputSink, Rate, RawVideoFormat,
    SystemSlice, VideoCodec,
};
use g2g_plugins::cudaglsink::CudaGlSink;
use g2g_plugins::ffmpegdec::{Backend, FfmpegVideoDec};

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

/// Annex-B access-unit splitter (a new AU starts at the first VCL slice once the
/// current AU already has one). Same helper as `cuda_wgpu_e2e`.
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

#[tokio::test]
#[ignore = "needs a Wayland session + NVIDIA GPU + G2G_H264_FIXTURE"]
async fn cudagl_sink_presents_nvdec_cuda_frames() {
    if std::env::var_os("WAYLAND_DISPLAY").is_none() {
        eprintln!("skipping: no WAYLAND_DISPLAY in env (run under a Wayland session)");
        return;
    }
    let Some(path) = std::env::var_os("G2G_H264_FIXTURE") else {
        eprintln!("skipping: set G2G_H264_FIXTURE=/path/to/clip.h264 to run");
        return;
    };
    let bitstream = std::fs::read(&path).expect("read fixture");
    let access_units = split_access_units(&bitstream);
    assert!(!access_units.is_empty(), "no access units in fixture");

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

    let (w, h) = first_nv12_dims(&decoded.packets).expect("NV12 caps from decoder");
    eprintln!("NVDEC decoded NV12 {w}x{h}");

    let cuda_frames: Vec<Frame> = decoded
        .packets
        .into_iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) if matches!(f.domain, MemoryDomain::Cuda(_)) => Some(f),
            _ => None,
        })
        .collect();
    assert!(
        !cuda_frames.is_empty(),
        "decoder produced no CUDA frames (NvdecCuda backend missing?)"
    );

    // Present each device-resident frame through CudaGlSink on the live display.
    let mut sink = CudaGlSink::new().with_title("glass2glass cudagl smoke");
    sink.configure_pipeline(&nv12_caps(w, h))
        .expect("CudaGlSink configure (spawns GL worker)");

    let mut out = Collect::default();
    for frame in cuda_frames {
        // Stamp arrival so the sink's glass-to-glass histogram is populated.
        let frame = Frame {
            timing: FrameTiming {
                arrival_ns: monotonic_ns(),
                ..frame.timing
            },
            ..frame
        };
        sink.process(PipelinePacket::DataFrame(frame), &mut out)
            .await
            .expect("present");
    }
    sink.process(PipelinePacket::Eos, &mut out)
        .await
        .expect("eos");

    let presented = sink.frames_presented();
    let lat = sink.latency_snapshot();
    eprintln!(
        "CudaGlSink presented {presented} frame(s); glass-to-glass n={} p50={:.1}ms p95={:.1}ms max={:.1}ms",
        lat.count,
        lat.p50_ns as f64 / 1e6,
        lat.p95_ns as f64 / 1e6,
        lat.max_ns as f64 / 1e6,
    );
    assert!(presented > 0, "no frames presented through CudaGlSink");
}
