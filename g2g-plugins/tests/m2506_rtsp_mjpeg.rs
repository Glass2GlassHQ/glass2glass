// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(feature = "rtsp", feature = "mjpeg"))]

use std::time::Duration;

use g2g_core::runtime::{parse_launch, run_graph, LatencyProfile, SourceLoop};
use g2g_core::{Caps, Dim, PipelinePacket, VideoCodec};
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;
use g2g_plugins::rtspsrc::RtspSrc;
use g2g_plugins::rtspsrcn::{probe_tracks, DEFAULT_USER_AGENT};

#[path = "common/collect_sink.rs"]
mod collect_sink;
#[path = "common/ffmpeg_tool.rs"]
mod ffmpeg_tool;
#[path = "common/rtsp_h265_relay.rs"]
mod rtsp_relay;

const FRAME_COUNT: u64 = 8;
const DEADLINE: Duration = Duration::from_secs(10);

#[tokio::test]
async fn rtspsrc_refines_jpeg_caps_before_the_first_image() {
    if !ffmpeg_tool::have_ffmpeg() {
        eprintln!("skipping RTP/JPEG interop without ffmpeg");
        return;
    }
    let relay = rtsp_relay::serve_mjpeg();
    let mut source = RtspSrc::new(&relay.url).with_frame_limit(FRAME_COUNT);
    let mut output = collect_sink::CollectSink::default();
    tokio::time::timeout(DEADLINE, async {
        let tracks = probe_tracks(&relay.url, DEFAULT_USER_AGENT)
            .await
            .expect("DESCRIBE finds JPEG");
        let caps = source
            .intercept_caps()
            .await
            .expect("SETUP negotiates JPEG");
        assert_eq!(caps, tracks.video);
        assert!(matches!(
            caps,
            Caps::CompressedVideo {
                codec: VideoCodec::Mjpeg,
                ..
            }
        ));
        source.run(&mut output).await.expect("RTSP delivers JPEG");
    })
    .await
    .expect("the RTSP run completes");

    let mut exact_caps = false;
    let mut frames = 0;
    for packet in output.packets {
        match packet {
            PipelinePacket::CapsChanged(Caps::CompressedVideo {
                codec: VideoCodec::Mjpeg,
                width: Dim::Fixed(64),
                height: Dim::Fixed(48),
                ..
            }) => exact_caps = true,
            PipelinePacket::DataFrame(frame) => {
                assert!(exact_caps, "real dimensions precede the first image");
                assert!(frame.timing.keyframe);
                let jpeg = frame.domain.as_system_slice().unwrap();
                assert!(jpeg.starts_with(&[0xff, 0xd8]) && jpeg.ends_with(&[0xff, 0xd9]));
                frames += 1;
            }
            _ => {}
        }
    }
    assert_eq!(frames, FRAME_COUNT);
}

#[tokio::test]
async fn rtspsrcn_feeds_a_motion_jpeg_decoder() {
    if !ffmpeg_tool::have_ffmpeg() {
        eprintln!("skipping RTP/JPEG interop without ffmpeg");
        return;
    }
    let relay = rtsp_relay::serve_mjpeg();
    let graph = parse_launch(
        &default_registry(),
        &format!(
            "rtspsrcn name=cam location={} num-buffers={FRAME_COUNT} cam. ! mjpegdec ! fakesink",
            relay.url
        ),
    )
    .expect("parse the RTSP graph");
    let clock = WallClock::new();
    let stats = tokio::time::timeout(
        DEADLINE,
        run_graph(graph, &clock, LatencyProfile::Throughput),
    )
    .await
    .expect("the RTSP graph completes")
    .expect("decode RTP/JPEG");
    assert_eq!(stats.frames_emitted, FRAME_COUNT);
    assert_eq!(stats.frames_consumed, FRAME_COUNT);
}
