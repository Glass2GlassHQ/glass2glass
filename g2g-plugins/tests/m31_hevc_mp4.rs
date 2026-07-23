//! M31: HEVC through the fMP4 container. `Mp4Mux` muxes an `hvc1`/`hvcC`
//! track from synthetic H.265 access units, and `Mp4Src` reads them back
//! byte-exactly with the codec/geometry recovered during the caps probe. No
//! encoder needed: the elements only frame the bitstream, so hand-built NALUs
//! exercise the full container path on any platform.

use g2g_core::element::{AsyncElement, BoxFuture, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::SourceLoop;
use g2g_core::{Caps, Dim, G2gError, Rate, VideoCodec};
use g2g_plugins::mp4mux::Mp4Mux;
use g2g_plugins::mp4src::Mp4Src;

use std::path::PathBuf;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;
const FRAMES: usize = 6;
const FRAME_DURATION_NS: u64 = 33_333_333;

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g_m31_{}_{}.mp4", std::process::id(), name))
}

/// An H.265 NAL: byte 0 carries `nal_type` in bits 1..6, byte 1 is the
/// layer/temporal-id (use 0x01), then payload.
fn hevc_nal(nal_type: u8, payload: &[u8]) -> Vec<u8> {
    let mut n = vec![(nal_type << 1) & 0x7E, 0x01];
    n.extend_from_slice(payload);
    n
}

fn annexb(nalus: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    for n in nalus {
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(n);
    }
    out
}

/// First HEVC access unit: VPS(32)+SPS(33)+PPS(34)+IDR(19). The SPS is padded
/// past 15 bytes so the sink's hvcC can copy a full profile_tier_level.
fn keyframe_au() -> Vec<u8> {
    let vps = hevc_nal(32, &[0x0A, 0x0B]);
    let sps = hevc_nal(33, &(0u8..20).collect::<Vec<_>>());
    let pps = hevc_nal(34, &[0xC0, 0xC1]);
    let idr = hevc_nal(19, &[0x11, 0x22, 0x33, 0x44]);
    annexb(&[vps, sps, pps, idr])
}

/// A non-keyframe access unit: a single TRAIL_R(1) slice.
fn delta_au(index: usize) -> Vec<u8> {
    let slice = hevc_nal(1, &[index as u8, 0xAB, 0xCD]);
    annexb(&[slice])
}

fn frame(bytes: Vec<u8>, index: usize) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming {
            pts_ns: index as u64 * FRAME_DURATION_NS,
            dts_ns: index as u64 * FRAME_DURATION_NS,
            duration_ns: FRAME_DURATION_NS,
            capture_ns: index as u64 * FRAME_DURATION_NS,
            ..FrameTiming::default()
        },
        sequence: index as u64,
        meta: Default::default(),
    }
}

fn hevc_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H265,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Fixed(30 << 16),
    }
}

#[derive(Default)]
struct Collect {
    frames: Vec<Vec<u8>>,
    eos: bool,
}

impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(f) => {
                    let Some(slice) = f.domain.as_system_slice() else {
                        panic!("expected system frame");
                    };
                    self.frames.push(slice.to_vec());
                }
                PipelinePacket::Eos => self.eos = true,
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

#[tokio::test]
async fn hevc_round_trips_through_the_fmp4_container() {
    let path = temp_path("roundtrip");

    // --- mux ---
    let access_units: Vec<Vec<u8>> = (0..FRAMES)
        .map(|i| if i == 0 { keyframe_au() } else { delta_au(i) })
        .collect();

    let mut mux = Mp4Mux::new();
    let narrowed = mux.intercept_caps(&hevc_caps()).expect("intercept H.265");
    assert!(matches!(
        narrowed,
        Caps::CompressedVideo {
            codec: VideoCodec::H265,
            ..
        }
    ));
    mux.configure_pipeline(&narrowed).expect("configure mux");
    let mut cap = Capture::default();
    for (i, au) in access_units.iter().enumerate() {
        mux.process(PipelinePacket::DataFrame(frame(au.clone(), i)), &mut cap)
            .await
            .expect("mux frame");
    }
    mux.process(PipelinePacket::Eos, &mut cap)
        .await
        .expect("mux eos");
    assert_eq!(mux.emitted(), FRAMES as u64);
    std::fs::write(&path, &cap.bytes).expect("write fmp4 file");

    // --- probe + demux ---
    let mut src = Mp4Src::new(&path);
    let probed = src.intercept_caps().await.expect("probe");
    assert_eq!(
        probed,
        Caps::CompressedVideo {
            codec: VideoCodec::H265,
            width: Dim::Fixed(WIDTH),
            height: Dim::Fixed(HEIGHT),
            // advisory range, not `Any`: per-frame PTS carries the real timing
            // and `Any` would abort fixate when nothing downstream pins the rate.
            framerate: Rate::Range {
                min_q16: 1 << 16,
                max_q16: 240 << 16
            },
        },
        "probe recovers HEVC codec and geometry"
    );
    src.configure_pipeline(&probed).expect("configure src");

    let mut out = Collect::default();
    let emitted = src.run(&mut out).await.expect("demux");
    assert_eq!(emitted, FRAMES as u64);
    assert!(out.eos);

    // The writer keeps parameter sets in-band, so each access unit returns
    // byte-exactly (no prepend on the first frame).
    assert_eq!(
        out.frames, access_units,
        "every access unit recovered exactly"
    );

    let _ = std::fs::remove_file(&path);
}

/// Concatenates the ISO-BMFF byte-stream frames `Mp4Mux` forwards downstream.
#[derive(Default)]
struct Capture {
    bytes: Vec<u8>,
}
impl OutputSink for Capture {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.bytes.extend_from_slice(s);
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}
