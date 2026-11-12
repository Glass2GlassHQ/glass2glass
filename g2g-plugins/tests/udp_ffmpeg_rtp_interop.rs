// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! Real-peer RTP interop (M528): ffmpeg's RTP H.264 payloader (`-f rtp`) streams
//! into the g2g `UdpSrc`, which depayloads RFC 6184 RTP (single-NAL / FU-A /
//! STAP-A) back into Annex-B access units. This proves the g2g depayloader +
//! jitter buffer match ffmpeg's payloader on the wire (FU-A fragmentation of a
//! large IDR is where depay bugs hide), which the g2g<->g2g `udp_loopback` (both
//! sides g2g) cannot.
//!
//! Ignored by default (needs ffmpeg, opens a local UDP socket). Run:
//!
//! ```sh
//! cargo test -p g2g-plugins --features udp-ingress --test udp_ffmpeg_rtp_interop \
//!     -- --ignored --nocapture
//! ```
#![cfg(feature = "udp-ingress")]

use std::net::UdpSocket as StdUdpSocket;
use std::process::Command;
use std::time::Duration;

use g2g_core::runtime::SourceLoop;
use g2g_core::{G2gError, OutputSink, PipelinePacket, PushOutcome};

use g2g_plugins::udpsrc::UdpSrc;

/// Records each emitted access unit's leading bytes, length, and keyframe flag.
#[derive(Default)]
struct AuCollect {
    aus: Vec<(Vec<u8>, usize, bool)>,
}

impl OutputSink for AuCollect {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let p = packet_slot.take().expect("poll_push without a packet");

        if let PipelinePacket::DataFrame(frame) = &p {
            if let Some(slice) = frame.domain.as_system_slice() {
                let s = slice;
                let head = s[..s.len().min(5)].to_vec();
                self.aus.push((head, s.len(), frame.timing.keyframe));
            }
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

/// An Annex-B access unit starts with a 3- or 4-byte start code.
fn is_annex_b(head: &[u8]) -> bool {
    head.starts_with(&[0, 0, 0, 1]) || head.starts_with(&[0, 0, 1])
}

#[tokio::test]
#[ignore = "needs ffmpeg with the rtp muxer; opens a local UDP socket"]
async fn ffmpeg_rtp_h264_streams_into_udpsrc() {
    // Bind the receive socket up front; ffmpeg blasts RTP with no handshake, so
    // the kernel buffers early datagrams until UdpSrc reads them.
    let sock = StdUdpSocket::bind("127.0.0.1:0").expect("bind udp");
    let port = sock.local_addr().unwrap().port();
    const N: u64 = 20;
    let mut src = UdpSrc::from_socket(sock)
        .expect("adopt socket")
        .with_frame_limit(N);
    src.configure_pipeline(&src_caps()).expect("configure");

    // ffmpeg payloads ~3 s of H.264 as RTP to our port (PT 96, its own SSRC).
    let url = format!("rtp://127.0.0.1:{port}");
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
                "testsrc=size=320x240:rate=15:duration=3",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-payload_type",
                "96",
                "-f",
                "rtp",
                &url,
            ])
            .status()
    });

    let mut sink = AuCollect::default();
    let received = tokio::time::timeout(Duration::from_secs(20), src.run(&mut sink))
        .await
        .expect("UdpSrc receives N access units within 20s")
        .expect("UdpSrc runs");
    let _ = ffmpeg.await;

    assert_eq!(
        received, N,
        "depayloaded the requested number of access units"
    );
    assert_eq!(sink.aus.len() as u64, N);
    // Every emitted AU must be Annex-B framed (the depayloader re-framed the RTP
    // NALs correctly, including reassembling FU-A fragments).
    for (i, (head, len, _)) in sink.aus.iter().enumerate() {
        assert!(*len > 4, "AU {i} has content");
        assert!(
            is_annex_b(head),
            "AU {i} is Annex-B framed (got {head:02x?})"
        );
    }
    // At least one IDR access unit must be recognized as a keyframe (proves an
    // FU-A-fragmented IDR reassembled into a valid access unit).
    assert!(
        sink.aus.iter().any(|(_, _, kf)| *kf),
        "at least one keyframe among the depayloaded access units"
    );

    // The depayloader consumed a real ffmpeg RTP payloader on the wire: persist
    // peer-tagged `Oracle` evidence so `--maturity` derives udpsrc as InteropTested.
    use g2g_core::conformance::{ConformanceDimension, Evidence};
    g2g_plugins::conformance::persist::record_evidence(
        "udpsrc",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("h264")
            .detail("ffmpeg RTP (RFC 6184) payloader depayloaded to Annex-B, FU-A IDR reassembled"),
    )
    .expect("record oracle evidence");
}

fn src_caps() -> g2g_core::Caps {
    use g2g_core::{Caps, Dim, Rate, VideoCodec};
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        framerate: Rate::Fixed(15 << 16),
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        pixel_aspect: g2g_core::PixelAspect::Any,
        alignment: g2g_core::BitstreamAlignment::Unspecified,
    }
}

#[derive(Default)]
struct PtsCollect {
    pts_ns: Vec<u64>,
}

impl OutputSink for PtsCollect {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        if let Some(PipelinePacket::DataFrame(frame)) = packet_slot.take() {
            self.pts_ns.push(frame.timing.pts_ns);
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

// an RTP port whose next port is free too, for ffmpeg's RTCP
fn free_rtp_port() -> u16 {
    loop {
        let rtp = StdUdpSocket::bind("127.0.0.1:0").expect("bind udp");
        let port = rtp.local_addr().unwrap().port();
        let Some(rtcp_port) = port.checked_add(1) else {
            continue;
        };
        if StdUdpSocket::bind(("127.0.0.1", rtcp_port)).is_ok() {
            return port;
        }
    }
}

// what `-itsoffset` delays ffmpeg's audio input by
const AUDIO_DELAY_MS: u64 = 500;
const AV_FRAMES: u64 = 30;
const NANOS_PER_MS: u64 = 1_000_000;
const NANOS_PER_SEC: u64 = 1_000_000_000;
const OPUS_SAMPLE_RATE_HZ: u64 = 48_000;
// libopus's lookahead at 48 kHz, which ffmpeg starts the stream early by
const OPUS_LOOKAHEAD_SAMPLES: u64 = 312;
// each udpsrc fixes the sender's clock offset from the arrival of its own first report
const SYNC_TOLERANCE_NS: u64 = 2 * NANOS_PER_MS;

#[tokio::test]
#[ignore = "needs ffmpeg with libx264 and libopus; opens local UDP sockets"]
async fn two_udpsrc_keep_ffmpegs_audio_offset_on_the_pipeline_clock() {
    use g2g_core::clock::{ClockSync, MonotonicClock, PipelineClock};
    use g2g_core::PropValue;

    let sync = ClockSync::new(std::sync::Arc::new(MonotonicClock), MonotonicClock.now_ns());
    let (video_port, audio_port) = (free_rtp_port(), free_rtp_port());
    let mut video = UdpSrc::new(([127, 0, 0, 1], video_port).into())
        .with_video_size(320, 240)
        .with_frame_limit(AV_FRAMES);
    let mut audio = UdpSrc::new(([127, 0, 0, 1], audio_port).into()).with_frame_limit(AV_FRAMES);
    audio
        .set_property("encoding-name", PropValue::Str("OPUS".into()))
        .expect("opus");
    for src in [&mut video, &mut audio] {
        src.set_clock_sync(sync.clone());
        let caps = src.intercept_caps().await.expect("caps");
        src.configure_pipeline(&caps).expect("configure");
    }

    // ffmpeg maps each output's PTS zero to the time that output's header was
    // written unless both are given one start time
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    // a second ahead, since ffmpeg's encoder setup before its first packet must
    // not put PTS zero before running-time zero
    let start_us = (now + Duration::from_secs(1)).as_micros().to_string();
    // without -re ffmpeg sends as fast as it encodes, so arrival times alone put
    // the audio right on top of the video
    let ffmpeg = tokio::task::spawn_blocking(move || {
        Command::new("ffmpeg")
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
                &format!("sine=frequency=440:sample_rate={OPUS_SAMPLE_RATE_HZ}:duration=4"),
            ])
            .args(["-map", "0:v", "-c:v", "libx264", "-preset", "ultrafast"])
            .args(["-tune", "zerolatency", "-payload_type", "96"])
            .args(["-start_time_realtime", &start_us, "-f", "rtp"])
            .arg(format!("rtp://127.0.0.1:{video_port}"))
            .args(["-map", "1:a", "-c:a", "libopus", "-payload_type", "97"])
            .args(["-start_time_realtime", &start_us, "-f", "rtp"])
            .arg(format!("rtp://127.0.0.1:{audio_port}"))
            .status()
    });

    let (mut video_out, mut audio_out) = (PtsCollect::default(), PtsCollect::default());
    let (video_run, audio_run) = tokio::time::timeout(Duration::from_secs(20), async {
        tokio::join!(video.run(&mut video_out), audio.run(&mut audio_out))
    })
    .await
    .expect("both udpsrc receive their frames within 20s");
    video_run.expect("video runs");
    audio_run.expect("audio runs");
    let _ = ffmpeg.await;

    let offset_ns = audio_out.pts_ns[0] as i64 - video_out.pts_ns[0] as i64;
    let expected_ns = (AUDIO_DELAY_MS * NANOS_PER_MS
        - OPUS_LOOKAHEAD_SAMPLES * NANOS_PER_SEC / OPUS_SAMPLE_RATE_HZ)
        as i64;
    assert!(
        offset_ns.abs_diff(expected_ns) <= SYNC_TOLERANCE_NS,
        "audio starts {offset_ns} ns after video, expected {expected_ns} ns"
    );
}
