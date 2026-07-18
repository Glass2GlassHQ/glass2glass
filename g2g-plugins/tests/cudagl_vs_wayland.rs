//! A/B latency benchmark: `CudaGlSink` (device-resident CUDA-GL present) vs the
//! `NvdecCuvid -> WaylandSink` baseline (system-memory NV12 + CPU NV12->XRGB
//! convert + `wl_shm` upload).
//!
//! Both legs hardware-decode the *same* H.264 fixture on NVIDIA, so the only
//! difference is the present path:
//!
//! ```text
//! A:  H.264 -> FfmpegVideoDec(NvdecCuda)  -> CudaGlSink    (NV12 stays on GPU,
//!              MemoryDomain::Cuda            shader convert)  GPU convert
//! B:  H.264 -> FfmpegVideoDec(NvdecCuvid) -> WaylandSink   (download to system
//!              system NV12                   CPU convert)     NV12, CPU convert
//! ```
//!
//! Arrival is stamped after decode and before the sink, so each sink's
//! glass-to-glass histogram isolates its present cost (the NvdecCuvid
//! device->host download is charged to decode, not to the sink window). The
//! difference grows with resolution: at 1080p the Wayland leg's per-frame CPU
//! convert of ~3 MB is real work the CudaGl leg does on the GPU.
//!
//! Ignored by default. Needs a Wayland session, an NVIDIA GPU (with `h264_cuvid`
//! in libavcodec), and on a hybrid iGPU+NVIDIA host the GL context forced onto
//! NVIDIA (see `cudagl_smoke`). Default fixture is 1080p; override with your own.
//!
//! ```sh
//! ffmpeg -f lavfi -i testsrc=size=1920x1080:rate=30:duration=4 -c:v libx264 \
//!     -pix_fmt yuv420p -g 30 -bsf:v h264_mp4toannexb -f h264 /tmp/clip1080.h264
//! __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
//!     __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
//!     G2G_H264_FIXTURE=/tmp/clip1080.h264 cargo test -p g2g-plugins \
//!     --features "ffmpeg cuda-gl wayland-sink" --test cudagl_vs_wayland \
//!     -- --ignored --nocapture
//! ```

#![cfg(all(
    target_os = "linux",
    feature = "ffmpeg",
    feature = "cuda-gl",
    feature = "wayland-sink"
))]

use std::time::Instant;

use g2g_core::element::{BoxFuture, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::MemoryDomain;
use g2g_core::metrics::{monotonic_ns, LatencySnapshot};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, G2gError, OutputSink, Rate, RawVideoFormat,
    SystemSlice, VideoCodec,
};
use g2g_plugins::cudaglsink::CudaGlSink;
use g2g_plugins::ffmpegdec::{Backend, FfmpegVideoDec, OutputFormat};
use g2g_plugins::waylandsink::WaylandSink;

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

/// Annex-B access-unit splitter (matches `cudagl_smoke` / `cuda_wgpu_e2e`).
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

/// Decode every access unit through `dec`, returning the emitted data frames and
/// the NV12 dimensions the decoder fixated.
async fn decode_all(dec: &mut FfmpegVideoDec, access_units: &[Vec<u8>]) -> (Vec<Frame>, u32, u32) {
    let narrowed = dec.intercept_caps(&h264_caps()).expect("H.264 supported");
    assert!(matches!(
        dec.configure_pipeline(&narrowed).expect("decoder init"),
        ConfigureOutcome::Accepted
    ));
    let mut out = Collect::default();
    for (seq, au) in access_units.iter().enumerate() {
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(au.clone().into_boxed_slice())),
            timing: FrameTiming::default(),
            sequence: seq as u64,
            meta: Default::default(),
        };
        dec.process(PipelinePacket::DataFrame(frame), &mut out)
            .await
            .expect("decode");
    }
    dec.process(PipelinePacket::Eos, &mut out)
        .await
        .expect("flush");
    let (w, h) = first_nv12_dims(&out.packets).expect("NV12 caps from decoder");
    let frames = out
        .packets
        .into_iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f),
            _ => None,
        })
        .collect();
    (frames, w, h)
}

/// Present every frame through `sink` (already configured), stamping arrival
/// just before each push so the sink's histogram measures its present cost.
async fn present_all<S: AsyncElement>(sink: &mut S, frames: Vec<Frame>) -> usize {
    let mut out = Collect::default();
    let mut n = 0;
    for frame in frames {
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
        n += 1;
    }
    sink.process(PipelinePacket::Eos, &mut out)
        .await
        .expect("eos");
    n
}

fn report(label: &str, presented: u64, fed: usize, lat: &LatencySnapshot, elapsed_s: f64) {
    eprintln!(
        "{label}: presented={presented}/{fed} elapsed={elapsed_s:.2}s \
         glass-to-glass n={} p50={:.2}ms p95={:.2}ms max={:.2}ms",
        lat.count,
        lat.p50_ns as f64 / 1e6,
        lat.p95_ns as f64 / 1e6,
        lat.max_ns as f64 / 1e6,
    );
}

#[tokio::test]
#[ignore = "needs a Wayland session + NVIDIA GPU + G2G_H264_FIXTURE"]
async fn cudagl_vs_wayland_present_latency() {
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

    // ---- A: NvdecCuda -> CudaGlSink (device-resident, GPU convert) ----
    let mut dec_a = FfmpegVideoDec::new().with_backend(Backend::NvdecCuda);
    let (frames_a, wa, ha) = decode_all(&mut dec_a, &access_units).await;
    assert!(
        frames_a
            .first()
            .map(|f| matches!(f.domain, MemoryDomain::Cuda(_)))
            .unwrap_or(false),
        "NvdecCuda must emit CUDA device frames"
    );
    eprintln!(
        "A decoded {} NvdecCuda NV12 {wa}x{ha} frame(s)",
        frames_a.len()
    );
    let mut cudagl = CudaGlSink::new().with_title("glass2glass A: CudaGlSink");
    cudagl
        .configure_pipeline(&nv12_caps(wa, ha))
        .expect("CudaGlSink configure");
    let t = Instant::now();
    let fed_a = present_all(&mut cudagl, frames_a).await;
    let elapsed_a = t.elapsed().as_secs_f64();
    let lat_a = cudagl.latency_snapshot();
    let pres_a = cudagl.frames_presented();
    drop(cudagl); // tear down the window before opening the next

    // ---- B: NvdecCuvid -> WaylandSink (system NV12, CPU convert + SHM) ----
    let mut dec_b = FfmpegVideoDec::new()
        .with_output_format(OutputFormat::Nv12)
        .with_backend(Backend::NvdecCuvid);
    let (frames_b, wb, hb) = decode_all(&mut dec_b, &access_units).await;
    assert!(
        frames_b
            .first()
            .map(|f| matches!(f.domain, MemoryDomain::System(_)))
            .unwrap_or(false),
        "NvdecCuvid must emit system-memory frames"
    );
    eprintln!(
        "B decoded {} NvdecCuvid NV12 {wb}x{hb} frame(s)",
        frames_b.len()
    );
    let mut wayland = WaylandSink::new().with_title("glass2glass B: WaylandSink");
    wayland
        .configure_pipeline(&nv12_caps(wb, hb))
        .expect("WaylandSink configure");
    let t = Instant::now();
    let fed_b = present_all(&mut wayland, frames_b).await;
    let elapsed_b = t.elapsed().as_secs_f64();
    let lat_b = wayland.latency_snapshot();
    let pres_b = wayland.frames_presented();
    drop(wayland);

    eprintln!("\n=== A/B present-path latency ({wa}x{ha}) ===");
    report(
        "A CudaGlSink  (GPU convert) ",
        pres_a,
        fed_a,
        &lat_a,
        elapsed_a,
    );
    report(
        "B WaylandSink (CPU convert) ",
        pres_b,
        fed_b,
        &lat_b,
        elapsed_b,
    );
    if lat_a.count > 0 && lat_b.count > 0 {
        let ratio = lat_b.p50_ns as f64 / lat_a.p50_ns.max(1) as f64;
        eprintln!("p50 ratio (B/A) = {ratio:.2}x  (>1 means the device-resident path is faster)");
    }

    assert!(pres_a > 0, "CudaGlSink presented nothing");
    assert!(pres_b > 0, "WaylandSink presented nothing");
    assert!(
        lat_a.count > 0 && lat_b.count > 0,
        "both sinks must record latency samples"
    );
}
