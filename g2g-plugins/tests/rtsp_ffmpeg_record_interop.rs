// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! Real-peer RTSP ingest interop (M529): ffmpeg *publishes* to the g2g
//! `RtspServerSrc` over RTSP (the OPTIONS/ANNOUNCE/SETUP/RECORD direction), over
//! UDP and TCP-interleaved (`-rtsp_transport tcp`, RFC 2326 §10.12) RTP, and the
//! server depayloads it into Annex-B access units that ffmpeg must then decode.
//! This drives the g2g RTSP responder + the shared RTP receive path (jitter
//! buffer + depayload, M520) against a reference RTSP client, which the in-tree
//! `record_loopback` (a hand-rolled g2g publisher) cannot.
//!
//! Ignored by default (needs ffmpeg + ffprobe, opens local TCP + UDP sockets). Run:
//!
//! ```sh
//! cargo test -p g2g-plugins --features rtsp-server --test rtsp_ffmpeg_record_interop \
//!     -- --ignored --nocapture
//! ```
#![cfg(feature = "rtsp-server")]

use std::io::Write;
use std::net::TcpListener as StdTcpListener;
use std::process::{Command, Stdio};
use std::time::Duration;

use g2g_core::runtime::SourceLoop;
use g2g_core::{G2gError, OutputSink, PipelinePacket, PropValue, PushOutcome};

use g2g_plugins::rtspserversrc::RtspServerSrc;

const ACCESS_UNITS: u64 = 20;

#[derive(Default)]
struct AuCollect {
    access_units: Vec<Vec<u8>>,
    keyframes: usize,
}

impl OutputSink for AuCollect {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let p = packet_slot.take().expect("poll_push without a packet");

        if let PipelinePacket::DataFrame(frame) = &p {
            let slice = frame.domain.as_system_slice().expect("system frame");
            self.access_units.push(slice.to_vec());
            self.keyframes += usize::from(frame.timing.keyframe);
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

struct Codec {
    encoding_name: &'static str,
    encoder_args: &'static [&'static str],
    ffmpeg_demuxer: &'static str,
    evidence_name: &'static str,
}

const H264: Codec = Codec {
    encoding_name: "H264",
    encoder_args: &[
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
    ],
    ffmpeg_demuxer: "h264",
    evidence_name: "h264",
};

const H265: Codec = Codec {
    encoding_name: "H265",
    encoder_args: &[
        "-c:v",
        "libx265",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-x265-params",
        "log-level=error",
    ],
    ffmpeg_demuxer: "hevc",
    evidence_name: "h265",
};

async fn record_from_ffmpeg(codec: &Codec, rtsp_transport: &'static str) -> AuCollect {
    // Bind the RTSP control listener up front so ffmpeg's TCP connect queues in
    // the backlog before RtspServerSrc calls accept().
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind rtsp control");
    let port = listener.local_addr().unwrap().port();
    let mut src = RtspServerSrc::from_listener(listener)
        .expect("adopt listener")
        .with_video_size(320, 240)
        .with_frame_limit(ACCESS_UNITS);
    src.set_property("encoding-name", PropValue::Str(codec.encoding_name.into()))
        .expect("an ingest encoding");

    let url = format!("rtsp://127.0.0.1:{port}/stream");
    let encoder_args = codec.encoder_args;
    let ffmpeg = tokio::task::spawn_blocking(move || {
        Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-re",
                "-an",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=320x240:rate=15:duration=4",
            ])
            .args(encoder_args)
            .args(["-f", "rtsp", "-rtsp_transport", rtsp_transport, &url])
            .status()
    });

    let mut sink = AuCollect::default();
    let received = tokio::time::timeout(Duration::from_secs(25), src.run(&mut sink))
        .await
        .expect("RtspServerSrc receives the access units within 25s")
        .expect("RtspServerSrc runs");
    let _ = ffmpeg.await;
    assert_eq!(received, ACCESS_UNITS);
    sink
}

fn assert_ffmpeg_decodes_every_access_unit(codec: &Codec, recorded: &AuCollect) {
    // ffmpeg's RTSP publisher sends the parameter sets only in the ANNOUNCE
    let mut ffprobe = Command::new("ffprobe")
        .args(["-v", "error", "-f", codec.ffmpeg_demuxer, "-i", "pipe:0"])
        .args(["-count_frames", "-select_streams", "v:0"])
        .args(["-show_entries", "stream=nb_read_frames", "-of", "csv=p=0"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("run ffprobe");
    let mut stdin = ffprobe.stdin.take().unwrap();
    stdin
        .write_all(&recorded.access_units.concat())
        .expect("feed ffprobe");
    drop(stdin);
    let output = ffprobe.wait_with_output().expect("ffprobe exits");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success() && stderr.is_empty(), "{stderr}");
    let decoded: u64 = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .expect("ffprobe prints the decoded frame count");
    assert_eq!(decoded, ACCESS_UNITS, "every access unit decodes");
    assert!(recorded.keyframes > 0, "a keyframe is flagged");
}

/// Persist the peer-tagged `Oracle` evidence both ingest directions earn, so
/// `--maturity` derives rtspserversrc as InteropTested against ffmpeg.
fn record_ingest_evidence(codec: &Codec, detail: &str) {
    use g2g_core::conformance::{ConformanceDimension, Evidence};
    g2g_plugins::conformance::persist::record_evidence(
        "rtspserversrc",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec(codec.evidence_name)
            .detail(detail),
    )
    .expect("record oracle evidence");
}

#[tokio::test]
#[ignore = "needs ffmpeg with RTSP; opens local TCP + UDP sockets"]
async fn ffmpeg_rtsp_publisher_records_into_rtspserversrc() {
    let recorded = record_from_ffmpeg(&H264, "udp").await;
    assert_ffmpeg_decodes_every_access_unit(&H264, &recorded);
    record_ingest_evidence(&H264, "ffmpeg RECORDs over UDP RTP, decoded by ffmpeg");
}

#[tokio::test]
#[ignore = "needs ffmpeg with RTSP; opens a local TCP socket"]
async fn ffmpeg_rtsp_publisher_records_interleaved_into_rtspserversrc() {
    let recorded = record_from_ffmpeg(&H264, "tcp").await;
    assert_ffmpeg_decodes_every_access_unit(&H264, &recorded);
    record_ingest_evidence(
        &H264,
        "ffmpeg RECORDs over TCP-interleaved RTP, decoded by ffmpeg",
    );
}

#[tokio::test]
#[ignore = "needs ffmpeg with RTSP and libx265; opens a local TCP socket"]
async fn ffmpeg_rtsp_publisher_records_h265_into_rtspserversrc() {
    let recorded = record_from_ffmpeg(&H265, "tcp").await;
    assert_ffmpeg_decodes_every_access_unit(&H265, &recorded);
    record_ingest_evidence(
        &H265,
        "ffmpeg RECORDs H.265 over TCP-interleaved RTP, decoded by ffmpeg",
    );
}
