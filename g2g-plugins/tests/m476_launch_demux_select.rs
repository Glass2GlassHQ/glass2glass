//! M476 - explicit-demux fan-out in a `gst-launch` line. A named demuxer fed by a
//! file source splits the file into its elementary streams, honoring GStreamer's
//! `d.video_0` / `d.audio_0` pad-name selection:
//!
//! ```text
//! filesrc location=movie.mkv bytestream-format=matroska ! matroskademux name=d
//!   d.video_0 ! h264parse ! fakesink
//!   d.audio_0 ! aacparse ! fakesink
//! ```
//!
//! Routing fidelity is proved by a *strict* parser on each branch (`h264parse`
//! only accepts H.264, `aacparse` only AAC): if `video_0` / `audio_0` were
//! mis-resolved, negotiation would fail. A swapped-reference-order case proves the
//! selection is by pad *name*, not position.

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{
    AudioFormat, Caps, Dim, G2gError, MultiInputElement, OutputSink, PipelineClock, PushOutcome,
    Rate, VideoCodec,
};
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        framerate: Rate::Fixed(30 << 16),
    }
}
fn aac_caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::Aac,
        channels: 2,
        sample_rate: 48_000,
    }
}

#[derive(Default)]
struct Collect {
    bytes: Vec<u8>,
}
impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
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
fn frame(data: Vec<u8>, pts_ns: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            ..FrameTiming::default()
        },
        0,
    ))
}
fn annexb(nals: &[&[u8]]) -> Vec<u8> {
    let mut v = Vec::new();
    for n in nals {
        v.extend_from_slice(&[0, 0, 0, 1]);
        v.extend_from_slice(n);
    }
    v
}
fn adts_au(payload: &[u8]) -> Vec<u8> {
    let frame_len = payload.len() + 7;
    let mut au = vec![
        0xFF,
        0xF1,
        (1 << 6) | (3 << 2),
        ((2 & 3) << 6) | ((frame_len >> 11) & 3) as u8,
        ((frame_len >> 3) & 0xFF) as u8,
        (((frame_len & 7) << 5) as u8) | 0x1F,
        0xFC,
    ];
    au.extend_from_slice(payload);
    au
}

/// Drive a two-input A/V muxer (H.264 video + AAC audio) and return its bytes.
async fn mux_av<M: MultiInputElement>(mut mux: M) -> Vec<u8> {
    let sps = [0x67u8, 0x42, 0x00, 0x1e, 0x88];
    let pps = [0x68u8, 0xce, 0x3c, 0x80];
    let idr = [0x65u8, 0x88, 0x84, 0x00];
    mux.configure_pipeline(0, &h264_caps()).unwrap();
    mux.configure_pipeline(1, &aac_caps()).unwrap();
    let mut sink = Collect::default();
    mux.process(0, frame(annexb(&[&sps, &pps, &idr]), 0), &mut sink)
        .await
        .unwrap();
    mux.process(1, frame(adts_au(&[0xA1, 0xA2, 0xA3]), 0), &mut sink)
        .await
        .unwrap();
    mux.process(
        0,
        frame(annexb(&[&[0x41u8, 0x9a, 0x00]]), 33_000_000),
        &mut sink,
    )
    .await
    .unwrap();
    mux.process(1, frame(adts_au(&[0xB4, 0xB5]), 21_000_000), &mut sink)
        .await
        .unwrap();
    mux.process(0, PipelinePacket::Eos, &mut sink)
        .await
        .unwrap();
    mux.process(1, PipelinePacket::Eos, &mut sink)
        .await
        .unwrap();
    sink.bytes
}

/// Write bytes to a uniquely-named temp file, returning its path. Removed by the
/// caller after the run.
fn write_temp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("g2g-m476-{}-{}", std::process::id(), name));
    std::fs::write(&path, bytes).expect("write temp media");
    path
}

async fn run_line(line: &str) -> u64 {
    let reg = default_registry();
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"))
        .frames_consumed
}

#[tokio::test]
async fn matroskademux_named_pads_fan_out() {
    let bytes = mux_av(g2g_plugins::mkvmuxn::MkvMuxN::new(2)).await;
    let path = write_temp("mkv.mkv", &bytes);
    let p = path.display();

    // Strict parser per branch: h264parse only accepts H.264, aacparse only AAC.
    // A successful run proves d.video_0 -> the video stream, d.audio_0 -> audio.
    let line = format!(
        "filesrc location={p} bytestream-format=matroska ! matroskademux name=d  \
         d.video_0 ! h264parse ! fakesink  d.audio_0 ! aacparse ! fakesink"
    );
    let consumed = run_line(&line).await;
    assert!(
        consumed >= 4,
        "all four access units flowed through the fan-out: {consumed}"
    );

    // Swapped reference order: the audio branch is written first. Named selection
    // (not position) must still route audio to aacparse and video to h264parse.
    let swapped = format!(
        "filesrc location={p} bytestream-format=matroska ! matroskademux name=d  \
         d.audio_0 ! aacparse ! fakesink  d.video_0 ! h264parse ! fakesink"
    );
    let consumed = run_line(&swapped).await;
    assert!(
        consumed >= 4,
        "named pads route by name regardless of reference order: {consumed}"
    );

    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn tsdemux_named_pads_fan_out() {
    let bytes = mux_av(g2g_plugins::tsmuxn::TsMux::new(2)).await;
    let path = write_temp("ts.ts", &bytes);
    let p = path.display();
    let line = format!(
        "filesrc location={p} bytestream-format=mpegts ! tsdemux name=d  \
         d.video_0 ! h264parse ! fakesink  d.audio_0 ! aacparse ! fakesink"
    );
    let consumed = run_line(&line).await;
    assert!(
        consumed >= 4,
        "MPEG-TS fan-out routed both streams: {consumed}"
    );
    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn qtdemux_named_pads_fan_out() {
    let bytes = mux_av(g2g_plugins::mp4muxn::Mp4MuxN::new(2)).await;
    let path = write_temp("mp4.mp4", &bytes);
    let p = path.display();
    let line = format!(
        "filesrc location={p} bytestream-format=mp4 ! qtdemux name=d  \
         d.video_0 ! h264parse ! fakesink  d.audio_0 ! aacparse ! fakesink"
    );
    let consumed = run_line(&line).await;
    assert!(
        consumed >= 4,
        "MP4 (qtdemux) fan-out routed both streams: {consumed}"
    );
    let _ = std::fs::remove_file(&path);
}
