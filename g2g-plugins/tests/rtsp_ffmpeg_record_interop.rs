// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! Real-peer RTSP ingest interop (M529): ffmpeg, and gst `rtspclientsink` for
//! audio and video, *publish* to the g2g `RtspServerSrc` over RTSP (the
//! OPTIONS/ANNOUNCE/SETUP/RECORD direction), over UDP and TCP-interleaved
//! (`-rtsp_transport tcp`, RFC 2326 §10.12) RTP, and the server depayloads it
//! into access units that ffmpeg must then decode.
//! This drives the g2g RTSP responder + the shared RTP receive path (jitter
//! buffer + depayload, M520) against a reference RTSP client, which the in-tree
//! `record_loopback` (a hand-rolled g2g publisher) cannot.
//!
//! Ignored by default (needs ffmpeg + ffprobe, and gst-launch-1.0 for the gst
//! tests, opens local TCP + UDP sockets). Run:
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
use g2g_core::{
    G2gError, MultiOutputSink, MultiOutputSource, OutputSink, PipelinePacket, PropValue,
    PushOutcome,
};

use g2g_plugins::rtspserversrc::RtspServerSrc;
use g2g_plugins::rtspserversrcn::RtspServerSrcN;

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
    SourceLoop::set_property(
        &mut src,
        "encoding-name",
        PropValue::Str(codec.encoding_name.into()),
    )
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
    let received = tokio::time::timeout(
        Duration::from_secs(25),
        SourceLoop::run(&mut src, &mut sink),
    )
    .await
    .expect("RtspServerSrc receives the access units within 25s")
    .expect("RtspServerSrc runs");
    let _ = ffmpeg.await;
    assert_eq!(received, ACCESS_UNITS);
    sink
}

// what ffprobe decodes of `stream` in `bytes` read as `demuxer`, failing on any decode error
fn ffprobe_decoded_frames(demuxer: &str, stream: &str, bytes: &[u8]) -> u64 {
    let mut ffprobe = Command::new("ffprobe")
        .args(["-v", "error", "-f", demuxer, "-i", "pipe:0"])
        .args(["-count_frames", "-select_streams", stream])
        .args(["-show_entries", "stream=nb_read_frames", "-of", "csv=p=0"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("run ffprobe");
    let mut stdin = ffprobe.stdin.take().unwrap();
    stdin.write_all(bytes).expect("feed ffprobe");
    drop(stdin);
    let output = ffprobe.wait_with_output().expect("ffprobe exits");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success() && stderr.is_empty(), "{stderr}");
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .expect("ffprobe prints the decoded frame count")
}

fn assert_ffmpeg_decodes_every_access_unit(codec: &Codec, recorded: &AuCollect) {
    // ffmpeg's RTSP publisher sends the parameter sets only in the ANNOUNCE
    let decoded =
        ffprobe_decoded_frames(codec.ffmpeg_demuxer, "v:0", &recorded.access_units.concat());
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

const VIDEO_PORT: usize = 0;
const AUDIO_PORT: usize = 1;
// what `-itsoffset` delays ffmpeg's audio input by
const AUDIO_DELAY_MS: i64 = 500;
// ffmpeg's aac encoder starts one 1024-sample frame early, 21 ms at 48 kHz
const AAC_PRIMING_MS: i64 = 21;
// both first PTS are truncated to whole milliseconds
const SYNC_TOLERANCE_MS: i64 = 2;
const AV_ACCESS_UNITS: u64 = 30;

#[derive(Default)]
struct PortPts {
    pts_ns: [Vec<u64>; 2],
    audio: Vec<u8>,
}

impl MultiOutputSink for PortPts {
    fn poll_push_to(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        port: usize,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push_to without a packet");
        if let PipelinePacket::DataFrame(frame) = packet {
            self.pts_ns[port].push(frame.timing.pts_ns);
            if port == AUDIO_PORT {
                let slice = frame.domain.as_system_slice().expect("system frame");
                self.audio.extend_from_slice(slice);
            }
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }

    fn port_count(&self) -> usize {
        2
    }
}

fn av_server(listener: StdTcpListener) -> RtspServerSrc {
    RtspServerSrc::from_listener(listener)
        .expect("adopt listener")
        .with_video_size(320, 240)
        .with_audio_pad(true)
        .with_frame_limit(AV_ACCESS_UNITS)
}

// one publisher's video and AAC pads
fn av_server_n(listener: StdTcpListener) -> RtspServerSrcN {
    const VIDEO_AND_AUDIO_PADS: usize = 2;
    RtspServerSrcN::from_listener(listener, VIDEO_AND_AUDIO_PADS)
        .expect("adopt listener")
        .with_video_size(320, 240)
        .with_audio(true)
        .with_frame_limit(AV_ACCESS_UNITS)
}

// `publisher` is the command that publishes video and AAC to the URL it is given
async fn record_av<S: MultiOutputSource>(
    server: impl FnOnce(StdTcpListener) -> S,
    publisher: impl FnOnce(&str) -> Command,
) -> PortPts {
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind rtsp control");
    let port = listener.local_addr().unwrap().port();
    let mut src = server(listener);

    let url = format!("rtsp://127.0.0.1:{port}/stream");
    let mut publisher = publisher(&url);
    let publish = tokio::task::spawn_blocking(move || publisher.status());

    let mut sink = PortPts::default();
    let received = tokio::time::timeout(
        Duration::from_secs(25),
        MultiOutputSource::run(&mut src, &mut sink),
    )
    .await
    .expect("the server receives both tracks within 25s")
    .expect("the server runs");
    let _ = publish.await;
    assert_eq!(received, 2 * AV_ACCESS_UNITS);
    sink
}

// ffmpeg without `-re` sends as fast as it encodes, so arrival times alone put
// the audio right on top of the video
fn ffmpeg_av_publisher(rtsp_transport: &'static str) -> impl FnOnce(&str) -> Command {
    move |url| {
        let mut ffmpeg = Command::new("ffmpeg");
        ffmpeg
            .args(["-hide_banner", "-loglevel", "error"])
            .args([
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=320x240:rate=15:duration=4",
            ])
            .args(["-itsoffset", &format!("{AUDIO_DELAY_MS}ms")])
            .args([
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=48000:duration=4",
            ])
            .args(["-map", "0:v", "-map", "1:a"])
            .args(H264.encoder_args)
            .args(["-c:a", "aac", "-b:a", "64k"])
            .args(["-f", "rtsp", "-rtsp_transport", rtsp_transport, url]);
        ffmpeg
    }
}

// gst `rtspclientsink` payloads AAC with `rtpmp4apay`, as RFC 6416 MP4A-LATM
fn gst_av_publisher(protocols: &'static str) -> impl FnOnce(&str) -> Command {
    move |url| {
        let mut gst = Command::new("gst-launch-1.0");
        gst.args(["-q", "videotestsrc", "num-buffers=60", "!"])
            .args(["video/x-raw,width=320,height=240,framerate=15/1", "!"])
            .args(["x264enc", "tune=zerolatency", "speed-preset=ultrafast", "!"])
            .args(["h264parse", "!", "s.sink_0"])
            .args([
                "audiotestsrc",
                "num-buffers=100",
                "!",
                "audio/x-raw,rate=48000",
                "!",
            ])
            .args(["avenc_aac", "!", "aacparse", "!", "s.sink_1"])
            .args([
                "rtspclientsink",
                "name=s",
                &format!("protocols={protocols}"),
            ])
            .arg(format!("location={url}"));
        gst
    }
}

fn assert_audio_follows_video_by_the_publish_offset(recorded: &PortPts) {
    let first_ms = |port: usize| recorded.pts_ns[port][0] as i64 / 1_000_000;
    let offset_ms = first_ms(AUDIO_PORT) - first_ms(VIDEO_PORT);
    let expected_ms = AUDIO_DELAY_MS - AAC_PRIMING_MS;
    assert!(
        (offset_ms - expected_ms).abs() <= SYNC_TOLERANCE_MS,
        "audio starts {offset_ms} ms after video, expected {expected_ms} ms"
    );
}

#[tokio::test]
#[ignore = "needs ffmpeg with RTSP; opens local TCP + UDP sockets"]
async fn ffmpeg_av_publish_over_udp_keeps_the_audio_offset() {
    let recorded = record_av(av_server, ffmpeg_av_publisher("udp")).await;
    assert_audio_follows_video_by_the_publish_offset(&recorded);
}

#[tokio::test]
#[ignore = "needs ffmpeg with RTSP; opens a local TCP socket"]
async fn ffmpeg_av_publish_interleaved_keeps_the_audio_offset() {
    let recorded = record_av(av_server, ffmpeg_av_publisher("tcp")).await;
    assert_audio_follows_video_by_the_publish_offset(&recorded);
}

#[tokio::test]
#[ignore = "needs ffmpeg with RTSP; opens local TCP + UDP sockets"]
async fn rtspserversrcn_records_a_publishers_aac_on_its_second_pad() {
    let recorded = record_av(av_server_n, ffmpeg_av_publisher("udp")).await;
    assert_audio_follows_video_by_the_publish_offset(&recorded);
}

#[tokio::test]
#[ignore = "needs gst-launch-1.0 with rtspclientsink, x264enc and avenc_aac; opens local TCP + UDP sockets"]
async fn gst_av_publish_over_udp_records_decodable_aac() {
    let recorded = record_av(av_server, gst_av_publisher("udp")).await;
    assert_eq!(
        ffprobe_decoded_frames("aac", "a:0", &recorded.audio),
        AV_ACCESS_UNITS
    );
}

#[tokio::test]
#[ignore = "needs gst-launch-1.0 with rtspclientsink, x264enc and avenc_aac; opens a local TCP socket"]
async fn gst_av_publish_interleaved_records_decodable_aac() {
    let recorded = record_av(av_server, gst_av_publisher("tcp")).await;
    assert_eq!(
        ffprobe_decoded_frames("aac", "a:0", &recorded.audio),
        AV_ACCESS_UNITS
    );
}
