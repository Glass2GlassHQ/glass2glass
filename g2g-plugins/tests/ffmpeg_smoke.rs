//! M13: smoke test for `FfmpegH264Dec` (libavcodec software H.264 decoder).
//!
//! Ignored by default — requires:
//! - Linux with system libavcodec/libavformat/libavutil (Fedora:
//!   `ffmpeg-free-devel`; Debian: `libavcodec-dev libavformat-dev libavutil-dev`).
//! - An H.264 Annex-B fixture file path in `G2G_H264_FIXTURE`.
//!
//! Run with:
//!
//! ```sh
//! G2G_H264_FIXTURE=/path/to/clip.h264 cargo test -p g2g-plugins \
//!     --features ffmpeg --test ffmpeg_smoke -- --ignored --nocapture
//! ```
//!
//! Unlike `vaapi_smoke`, this test asserts decoded frames are produced —
//! ffmpeg's software decoder is portable enough that a green run is a real
//! end-to-end signal.

#![cfg(all(target_os = "linux", feature = "ffmpeg"))]

use g2g_core::element::{AsyncElement, BoxFuture, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{Caps, ConfigureOutcome, Dim, G2gError, Rate, VideoFormat};
use g2g_plugins::ffmpegdec::{FfmpegH264Dec, OutputFormat};

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

#[tokio::test]
#[ignore = "requires libav* and a G2G_H264_FIXTURE path"]
async fn ffmpeg_h264_decodes_fixture() {
    decode_once(OutputFormat::I420).await;
}

#[tokio::test]
#[ignore = "requires libav* and a G2G_H264_FIXTURE path"]
async fn ffmpeg_h264_decodes_fixture_nv12() {
    decode_once(OutputFormat::Nv12).await;
}

async fn decode_once(output: OutputFormat) {
    let Some(path) = std::env::var_os("G2G_H264_FIXTURE") else {
        eprintln!("skipping: set G2G_H264_FIXTURE=/path/to/clip.h264 to run");
        return;
    };
    let bitstream = std::fs::read(&path).expect("read H.264 fixture");
    assert!(!bitstream.is_empty(), "fixture is empty");

    let mut dec = FfmpegH264Dec::new().with_output_format(output);
    let upstream = Caps::Video {
        format: VideoFormat::H264,
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

    // Feed the whole fixture as one packet. `H264Parse` will normally
    // deliver one *access unit* (SPS + PPS + SEI + slices for one picture)
    // per `DataFrame`; libavcodec's bitstream filter accepts that shape
    // happily. Splitting further (one NAL per packet) breaks the SPS/PPS
    // bookkeeping that h264 expects to see alongside the first slice. This
    // smoke test just validates that the path produces a decoded frame; a
    // multi-frame test belongs alongside the real `H264Parse` element.
    let frame = Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bitstream.into_boxed_slice())),
        caps: narrowed,
        timing: FrameTiming {
            pts_ns: 0,
            dts_ns: 0,
            duration_ns: 0,
            capture_ns: 0,
        },
        sequence: 0,
    };
    dec.process(PipelinePacket::DataFrame(frame), &mut sink)
        .await
        .expect("process DataFrame");
    dec.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("process Eos");

    let caps_changes: Vec<_> = sink
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::CapsChanged(c) => Some(c.clone()),
            _ => None,
        })
        .collect();
    let data_frames: Vec<_> = sink
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f),
            _ => None,
        })
        .collect();

    eprintln!(
        "decoded {} frame(s); {} CapsChanged emitted",
        data_frames.len(),
        caps_changes.len()
    );
    assert!(!caps_changes.is_empty(), "expected at least one CapsChanged");
    assert!(!data_frames.is_empty(), "expected at least one decoded frame");

    // I420 and NV12 have identical byte length (w*h*3/2 for even dims); only
    // the chroma layout differs. The runner checks length + format tag.
    let expected_format = match output {
        OutputFormat::I420 => VideoFormat::I420,
        OutputFormat::Nv12 => VideoFormat::Nv12,
    };
    let first = caps_changes.first().unwrap();
    match first {
        Caps::Video {
            format,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            ..
        } if *format == expected_format => {
            eprintln!("first {:?} caps: {}x{}", expected_format, w, h);
            let f = data_frames.first().unwrap();
            let cw = ((*w + 1) / 2) as usize;
            let ch = ((*h + 1) / 2) as usize;
            let expected = (*w as usize) * (*h as usize) + 2 * cw * ch;
            match &f.domain {
                MemoryDomain::System(slice) => {
                    assert_eq!(
                        slice.as_slice().len(),
                        expected,
                        "{:?} byte length mismatch",
                        expected_format,
                    );
                }
                _ => panic!("decoder must emit System-domain frames"),
            }
        }
        other => panic!("expected fixed {expected_format:?} caps, got {other:?}"),
    }
}

