//! M663 `H265Parse` against a real H.265 elementary stream: ffmpeg (libx265)
//! encodes a short Annex-B stream with B-frames (so the SPS carries real
//! short-term RPS sets on the way to the VUI), and the re-framing parser must
//! recover the concrete geometry **and the VUI `timing_info` framerate** in its
//! `CapsChanged`, emitting one `DataFrame` per access unit. Self-skips where
//! ffmpeg or libx265 is absent.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::process::Command;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{AsyncElement, Caps, Dim, G2gError, OutputSink, PushOutcome, Rate, VideoCodec};
use g2g_plugins::h265parse::H265Parse;

#[derive(Default)]
struct CaptureSink {
    caps: Vec<Caps>,
    frames: usize,
    keyframes: usize,
}
impl OutputSink for CaptureSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(f) => {
                    self.frames += 1;
                    if f.timing.keyframe {
                        self.keyframes += 1;
                    }
                }
                PipelinePacket::CapsChanged(c) => self.caps.push(c),
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

#[tokio::test]
async fn recovers_geometry_and_vui_framerate_from_a_libx265_stream() {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("ffmpeg not present; skipping the H.265 real-stream test");
        return;
    }
    let path = std::env::temp_dir().join("g2g_m663.h265");
    let out = Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=320x240:rate=25",
        ])
        .args([
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx265",
            "-preset",
            "ultrafast",
            "-f",
            "hevc",
        ])
        .arg(&path)
        .output()
        .expect("ffmpeg runs");
    if !out.status.success() {
        eprintln!(
            "ffmpeg has no libx265; skipping: {}",
            String::from_utf8_lossy(&out.stderr)
                .lines()
                .last()
                .unwrap_or("")
        );
        return;
    }
    let es = std::fs::read(&path).expect("read the elementary stream");

    let caps = Caps::CompressedVideo {
        codec: VideoCodec::H265,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    };
    // Re-framing mode: byte chunks in, one access unit per DataFrame out (the
    // shape the TS/HLS auto-plug inserts). Feed uneven chunks to exercise
    // reassembly across buffer boundaries.
    let mut parse = H265Parse::reframing();
    parse.configure_pipeline(&caps).unwrap();
    let mut sink = CaptureSink::default();
    for (i, chunk) in es.chunks(4096).enumerate() {
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(chunk.to_vec().into_boxed_slice())),
            FrameTiming::default(),
            i as u64,
        );
        parse
            .process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();
    }
    parse.process(PipelinePacket::Eos, &mut sink).await.unwrap();

    // The refined caps carry the concrete geometry and the VUI framerate.
    assert_eq!(
        sink.caps,
        vec![Caps::CompressedVideo {
            codec: VideoCodec::H265,
            width: Dim::Fixed(320),
            height: Dim::Fixed(240),
            framerate: Rate::Fixed(25 << 16),
        }],
        "one CapsChanged with geometry + VUI timing_info framerate"
    );
    // 25 fps x 1 s re-framed to one DataFrame per access unit, IRAP flagged.
    assert_eq!(sink.frames, 25, "one frame per coded picture");
    assert!(sink.keyframes >= 1, "the IRAP access unit is flagged");

    let _ = std::fs::remove_file(&path);
}
