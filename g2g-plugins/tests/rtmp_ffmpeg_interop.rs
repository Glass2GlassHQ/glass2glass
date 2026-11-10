// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! Real-peer RTMP interop (M527): ffmpeg publishes an RTMP stream
//! (`-f flv rtmp://...`, ffmpeg's native RTMP client) into the g2g `RtmpSrc`
//! *listener*, which demuxes it to an FLV byte stream. This proves the g2g RTMP
//! server handshake (auto-detecting ffmpeg's simple or digest C1), chunk-stream
//! reassembly, and AMF0 publish flow match a reference RTMP client on the wire,
//! which the in-tree `RtmpPublisher <-> RtmpSession` loopback cannot (both sides
//! are g2g).
//!
//! Ignored by default (needs ffmpeg + ffprobe, opens a local TCP port). Run:
//!
//! ```sh
//! cargo test -p g2g-plugins --features rtmp --test rtmp_ffmpeg_interop \
//!     -- --ignored --nocapture
//! ```
#![cfg(feature = "rtmp")]

use std::net::TcpListener as StdTcpListener;
use std::path::Path;
use std::process::Command;
use std::time::Duration;

use g2g_core::runtime::SourceLoop;
use g2g_core::{ByteStreamEncoding, Caps, G2gError, OutputSink, PipelinePacket, PushOutcome};

use g2g_plugins::rtmpsrc::RtmpSrc;

mod ffprobe;
use ffprobe::fixture_path;

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::registry::default_registry;

/// Concatenates every FLV byte-stream `DataFrame` the source emits.
#[derive(Default)]
struct FlvCollect {
    bytes: Vec<u8>,
}

impl OutputSink for FlvCollect {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let p = packet_slot.take().expect("poll_push without a packet");

        if let PipelinePacket::DataFrame(frame) = &p {
            if let Some(slice) = frame.domain.as_system_slice() {
                self.bytes.extend_from_slice(slice);
            }
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn ffprobe_video_frame_count(path: &Path) -> u64 {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            path.to_str().unwrap(),
        ])
        .output()
        .expect("spawn ffprobe");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .find_map(|l| l.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

#[tokio::test]
#[ignore = "needs ffmpeg+ffprobe; opens a local TCP port + a temp file"]
async fn ffmpeg_rtmp_publisher_into_rtmpsrc() {
    // Bind the RTMP listener up front so ffmpeg's TCP connect (queued in the
    // listen backlog) succeeds before RtmpSrc calls accept().
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind rtmp listener");
    let port = listener.local_addr().unwrap().port();
    let mut src = RtmpSrc::from_listener(listener).expect("adopt listener");
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::Flv,
    })
    .expect("configure RtmpSrc");

    // ffmpeg publishes ~2 s of synthetic H.264 over RTMP to our listener.
    let url = format!("rtmp://127.0.0.1:{port}/live/stream");
    let ffmpeg = tokio::task::spawn_blocking(move || {
        Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-re",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=320x240:rate=15:duration=2",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-f",
                "flv",
                &url,
            ])
            .status()
    });

    // RtmpSrc accepts the publisher, drives the handshake + publish, and streams
    // the demuxed FLV until ffmpeg closes the connection (then EOS).
    let mut sink = FlvCollect::default();
    let emitted = tokio::time::timeout(Duration::from_secs(20), src.run(&mut sink))
        .await
        .expect("RtmpSrc completes within 20s")
        .expect("RtmpSrc runs");
    let _ = ffmpeg.await;

    assert!(
        emitted > 0,
        "RtmpSrc emitted FLV frames from the ffmpeg publisher"
    );
    assert!(sink.bytes.len() > 3, "collected an FLV byte stream");
    // The demuxed stream is a proper FLV file: the "FLV" signature + version.
    assert_eq!(
        &sink.bytes[0..3],
        b"FLV",
        "output starts with the FLV signature"
    );

    // ffprobe confirms the demuxed FLV decodes back to video frames intact.
    let out_flv = std::env::temp_dir().join(format!("g2g_rtmp_{}_{port}.flv", std::process::id()));
    std::fs::write(&out_flv, &sink.bytes).expect("write collected FLV");
    let frames = ffprobe_video_frame_count(&out_flv);
    let _ = std::fs::remove_file(&out_flv);
    assert!(
        frames > 0,
        "ffprobe decoded {frames} video frames from the RTMP-ingested FLV stream"
    );

    // g2g's RTMP server handshake (ffmpeg's native client) + chunk-stream reassembly
    // produced a valid FLV a reference demuxer decoded: persist `Oracle` evidence.
    use g2g_core::conformance::{ConformanceDimension, Evidence};
    g2g_plugins::conformance::persist::record_evidence(
        "rtmpsrc",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("h264")
            .detail(
                "ffmpeg RTMP publisher: handshake + chunk-stream demuxed to FLV, ffprobe-decoded",
            ),
    )
    .expect("record oracle evidence");
}

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

// the RTMP peer has bound its port by then
const PEER_READY: Duration = Duration::from_millis(500);

fn free_port() -> u16 {
    StdTcpListener::bind("127.0.0.1:0")
        .expect("bind a probe socket")
        .local_addr()
        .expect("address")
        .port()
}

// each stream's codec and decoded frame count
fn ffprobe_streams(path: &Path) -> Vec<(String, u64)> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-count_frames", "-show_entries"])
        .arg("stream=codec_name,nb_read_frames")
        .args(["-of", "csv=p=0"])
        .arg(path)
        .output()
        .expect("spawn ffprobe");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|line| {
            let (codec, frames) = line.trim().split_once(',')?;
            Some((codec.to_string(), frames.parse().ok()?))
        })
        .collect()
}

#[tokio::test]
#[ignore = "needs ffmpeg+ffprobe; opens a local TCP port + temp files"]
async fn ffmpeg_publishes_two_audio_tracks_into_rtmpsrc() {
    let port = free_port();
    let dir = std::env::temp_dir().join(format!("g2g_rtmp_tracks_in_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("work dir");
    let tracks = [dir.join("track0.aac"), dir.join("track1.aac")];
    let line = format!(
        "rtmpsrc address=127.0.0.1 port={port} ! flvdemux name=d \
         d.audio ! aacparse ! filesink location={} d.audio_1 ! aacparse ! filesink location={}",
        tracks[0].display(),
        tracks[1].display()
    );
    let graph = parse_launch(&default_registry(), &line).expect("the line parses");
    let url = format!("rtmp://127.0.0.1:{port}/live/stream");
    let ffmpeg = std::thread::spawn(move || {
        std::thread::sleep(PEER_READY);
        Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-re",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=1",
            ])
            .args(["-f", "lavfi", "-i", "sine=frequency=880:duration=1"])
            .args(["-map", "0", "-map", "1", "-c:a", "aac", "-f", "flv", &url])
            .status()
            .expect("spawn ffmpeg")
    });
    tokio::time::timeout(Duration::from_secs(20), run_graph(graph, &ZeroClock, 4))
        .await
        .expect("the ingest ends when ffmpeg disconnects")
        .expect("the ingest runs");
    assert!(ffmpeg.join().expect("ffmpeg thread").success());
    for track in &tracks {
        let streams = ffprobe_streams(track);
        assert!(
            matches!(streams.as_slice(), [(codec, frames)] if codec == "aac" && *frames > 0),
            "{track:?}: {streams:?}"
        );
    }
    assert_ne!(
        std::fs::read(&tracks[0]).expect("track 0"),
        std::fs::read(&tracks[1]).expect("track 1")
    );
}

#[tokio::test]
#[ignore = "needs ffmpeg+ffprobe; opens a local TCP port + temp files"]
async fn eflvmux_publishes_two_audio_tracks_to_ffmpeg() {
    let port = free_port();
    let received =
        std::env::temp_dir().join(format!("g2g_rtmp_tracks_out_{}.flv", std::process::id()));
    let url = format!("rtmp://127.0.0.1:{port}/live/stream");
    let listener_url = url.clone();
    let listener_out = received.clone();
    let ffmpeg = std::thread::spawn(move || {
        Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-listen", "1", "-i", &listener_url])
            .args(["-map", "0", "-c", "copy", "-f", "flv"])
            .arg(&listener_out)
            .status()
            .expect("spawn ffmpeg")
    });
    tokio::time::sleep(PEER_READY).await;
    let line = format!(
        "eflvmux name=m ! rtmpsink location={url} \
         filesrc location={} ! typefind ! aacparse ! m.audio \
         filesrc location={} ! typefind ! mpegaudioparse ! m.audio_1",
        fixture_path("aac_stereo_44100_adts.aac").display(),
        fixture_path("mp3_stereo_44100_id3v2.mp3").display()
    );
    let graph = parse_launch(&default_registry(), &line).expect("the line parses");
    run_graph(graph, &ZeroClock, 4)
        .await
        .expect("the publish runs");
    ffmpeg.join().expect("ffmpeg thread");
    let streams = ffprobe_streams(&received);
    let _ = std::fs::remove_file(&received);
    let codecs: Vec<&str> = streams.iter().map(|(codec, _)| codec.as_str()).collect();
    assert_eq!(codecs, ["aac", "mp3"], "{streams:?}");
    assert!(streams.iter().all(|(_, frames)| *frames > 0), "{streams:?}");
}
