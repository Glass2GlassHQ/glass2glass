//! M1109 decoder pipeline depth: a zerolatency H.264 stream (no B-frames, so
//! POC type 2) must leave `FfmpegH264Dec` with at most one access unit of
//! internal buffering. Feeding one AU at a time and counting outputs catches
//! any change that silently adds frames of pipeline depth, the class of
//! latency regression the level-DPB reorder seed caused (5 buffered frames,
//! 167 ms at 720p30) before it was confined to streams that can reorder.
//!
//! ffmpeg writes the fixture from the parameters below, so the assertions are
//! against what the stream was asked to be, not transcribed literals. The CI
//! workflows do not build the `ffmpeg` feature (libav* dev packages are
//! deliberately not pinned there), so this runs in the local full-feature
//! suite:
//! `cargo test -p g2g-plugins --features ffmpeg --test m1109_decoder_depth`.
#![cfg(all(target_os = "linux", feature = "ffmpeg"))]

use std::process::Command;

use g2g_core::element::{AsyncElement, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{Caps, ConfigureOutcome, Dim, G2gError, Rate, VideoCodec};
use g2g_plugins::ffmpegdec::FfmpegVideoDec;

/// The shape the fixture is encoded at. Assertions read these.
const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
const FRAMERATE: u64 = 30;
const FRAME_COUNT: usize = 30;

/// Access units the decoder may hold before its first output. One is the
/// packet being decoded; anything more is added latency every downstream
/// element inherits.
const MAX_DEPTH_FRAMES: usize = 1;

const NANOSECONDS_PER_SECOND: u64 = 1_000_000_000;
const NAL_TYPE_ACCESS_UNIT_DELIMITER: u8 = 9;

#[derive(Default)]
struct Collect {
    decoded: usize,
}

impl OutputSink for Collect {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        if matches!(packet, PipelinePacket::DataFrame(_)) {
            self.decoded += 1;
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

/// Encode the fixture: zerolatency (B-frames off, so x264 emits POC type 2)
/// with access unit delimiters, so the stream splits into AUs without a
/// slice-header parse.
fn write_fixture(path: &std::path::Path) -> bool {
    let source = format!("testsrc=size={WIDTH}x{HEIGHT}:rate={FRAMERATE}");
    let status = Command::new("ffmpeg")
        .args(["-y", "-f", "lavfi", "-i", &source])
        .args(["-frames:v", &FRAME_COUNT.to_string()])
        .args([
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
        ])
        .args(["-pix_fmt", "yuv420p", "-x264-params", "aud=1"])
        .args(["-f", "h264"])
        .arg(path)
        .output();
    status.is_ok_and(|o| o.status.success())
}

/// Split Annex-B bytes into access units on the AUD NALs the fixture was
/// encoded with.
fn split_access_units(bitstream: &[u8]) -> Vec<Vec<u8>> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 < bitstream.len() {
        let (offset, is_start) = if bitstream[i..].starts_with(&[0, 0, 0, 1]) {
            (4, true)
        } else if bitstream[i..].starts_with(&[0, 0, 1]) {
            (3, true)
        } else {
            (1, false)
        };
        if is_start {
            let nal_type = bitstream[i + offset] & 0x1F;
            if nal_type == NAL_TYPE_ACCESS_UNIT_DELIMITER {
                starts.push(i);
            }
            i += offset;
        } else {
            i += offset;
        }
    }
    starts
        .iter()
        .enumerate()
        .map(|(n, &s)| {
            let end = starts.get(n + 1).copied().unwrap_or(bitstream.len());
            bitstream[s..end].to_vec()
        })
        .collect()
}

#[tokio::test]
async fn zerolatency_h264_decodes_with_at_most_one_frame_of_depth() {
    run_depth_check(FfmpegVideoDec::new(), Some(MAX_DEPTH_FRAMES)).await;
}

/// Diagnostic twin for the NVDEC path: reports the observed depth instead of
/// bounding it. Ignored because CI has no GPU; run on the GPU host with
/// `-- --ignored --nocapture` when chasing hardware-decode latency.
#[tokio::test]
#[ignore = "needs an NVIDIA GPU with libnvcuvid"]
async fn zerolatency_h264_nvdec_depth_report() {
    use g2g_plugins::ffmpegdec::{Backend, OutputFormat};
    let dec = FfmpegVideoDec::new()
        .with_output_format(OutputFormat::Nv12)
        .with_backend(Backend::NvdecCuvid);
    run_depth_check(dec, None).await;
}

async fn run_depth_check(mut dec: FfmpegVideoDec, max_depth: Option<usize>) {
    let fixture = std::env::temp_dir().join(format!("g2g_m1109_{}.h264", std::process::id()));
    if !write_fixture(&fixture) {
        eprintln!("skipping: ffmpeg (with libx264) not available to write the fixture");
        return;
    }
    let bitstream = std::fs::read(&fixture).expect("read fixture");
    let _ = std::fs::remove_file(&fixture);

    let access_units = split_access_units(&bitstream);
    assert_eq!(
        access_units.len(),
        FRAME_COUNT,
        "fixture did not split into one AU per encoded frame"
    );

    let upstream = Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    };
    let narrowed = dec.intercept_caps(&upstream).expect("intercept H.264");
    let outcome = dec
        .configure_pipeline(&narrowed)
        .expect("libavcodec must initialise");
    assert!(matches!(outcome, ConfigureOutcome::Accepted));

    let mut sink = Collect::default();
    let mut worst_depth = 0;
    for (index, au) in access_units.into_iter().enumerate() {
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(au.into_boxed_slice())),
            timing: FrameTiming {
                pts_ns: index as u64 * NANOSECONDS_PER_SECOND / FRAMERATE,
                ..FrameTiming::default()
            },
            sequence: index as u64,
            meta: Default::default(),
        };
        dec.process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .expect("decode access unit");
        let pushed = index + 1;
        worst_depth = worst_depth.max(pushed - sink.decoded);
        if let Some(bound) = max_depth {
            assert!(
                sink.decoded + bound >= pushed,
                "decoder is buffering {} frames after {pushed} AUs in / {} out: \
                 pipeline depth above {bound} adds latency every downstream \
                 element inherits",
                pushed - sink.decoded,
                sink.decoded,
            );
        }
    }
    eprintln!("worst pipeline depth: {worst_depth} frame(s)");

    dec.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("drain on Eos");
    assert_eq!(sink.decoded, FRAME_COUNT, "decoder dropped frames");
}
