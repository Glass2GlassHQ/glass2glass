// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(feature = "udp-ingress", feature = "mjpeg"))]

use std::net::UdpSocket;
use std::process::Command;
use std::time::Duration;

use g2g_core::runtime::SourceLoop;
use g2g_core::{AsyncElement, Caps, Dim, PipelinePacket, Rate, VideoCodec};
use g2g_plugins::{mjpegdec::MjpegDec, stillparse::JpegParse, udpsrc::UdpSrc};

#[path = "common/collect_sink.rs"]
mod collect_sink;
#[path = "common/ffmpeg_mjpeg.rs"]
mod ffmpeg_mjpeg;
#[path = "common/ffmpeg_tool.rs"]
mod ffmpeg_tool;
#[path = "common/gstreamer_tool.rs"]
mod gstreamer_tool;

use collect_sink::{feed_in_chunks, CollectSink};

const FRAME_COUNT: u64 = 4;

fn ffmpeg_mjpeg() -> Command {
    let mut ffmpeg = Command::new("ffmpeg");
    ffmpeg.args(["-hide_banner", "-loglevel", "error"]);
    ffmpeg_mjpeg::mjpeg_source(&mut ffmpeg);
    ffmpeg.args(["-frames:v", &FRAME_COUNT.to_string()]);
    ffmpeg
}

fn gstreamer_mjpeg() -> Command {
    let mut gst = Command::new("gst-launch-1.0");
    gst.args([
        "-q",
        "videotestsrc",
        &format!("num-buffers={FRAME_COUNT}"),
        "pattern=smpte",
        "!",
        "video/x-raw,format=Y42B,width=64,height=48,framerate=25/1",
        "!",
        "jpegenc",
    ]);
    gst
}

async fn receive_jpeg(
    peer: &'static str,
    mut reference: Command,
    sender: impl FnOnce(u16) -> Command + Send + 'static,
) {
    let reference = reference.output().unwrap();
    assert!(
        reference.status.success(),
        "{peer} JPEG oracle: {}",
        String::from_utf8_lossy(&reference.stderr)
    );
    let socket = UdpSocket::bind("127.0.0.1:0").unwrap();
    let port = socket.local_addr().unwrap().port();
    let mut source = UdpSrc::from_socket(socket)
        .unwrap()
        .with_sdp(&format!(
            "v=0\r\nm=video {port} RTP/AVP 26\r\na=framerate:25\r\n"
        ))
        .unwrap()
        .with_frame_limit(FRAME_COUNT);
    let caps = source.intercept_caps().await.unwrap();
    assert!(matches!(
        caps,
        Caps::CompressedVideo {
            codec: VideoCodec::Mjpeg,
            ..
        }
    ));
    source.configure_pipeline(&caps).unwrap();

    let sender = tokio::task::spawn_blocking(move || {
        let output = sender(port).output().unwrap();
        assert!(
            output.status.success(),
            "{peer} RTP/JPEG sender: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    });
    let mut received = CollectSink::default();
    let result = tokio::time::timeout(Duration::from_secs(5), source.run(&mut received)).await;
    sender.await.unwrap();
    assert_eq!(
        result
            .unwrap_or_else(|_| panic!("{peer}: timed out receiving JPEG frames"))
            .unwrap(),
        FRAME_COUNT
    );

    let mut decoder = MjpegDec::new();
    decoder.configure_pipeline(&caps).unwrap();
    let mut output = CollectSink::default();
    let mut shapes = 0;
    let mut frames = 0;
    for packet in received.packets {
        match &packet {
            PipelinePacket::CapsChanged(Caps::CompressedVideo {
                width: Dim::Fixed(64),
                height: Dim::Fixed(48),
                framerate: Rate::Fixed(rate),
                ..
            }) => {
                assert_eq!(*rate, 25 << 16);
                shapes += 1;
            }
            PipelinePacket::DataFrame(frame) => {
                assert_eq!(shapes, 1, "image dimensions precede each frame");
                assert!(frame.timing.keyframe);
                assert_eq!(frame.timing.pts_ns, frames * 40_000_000);
                let jpeg = frame.domain.as_system_slice().unwrap();
                assert!(jpeg.len() > 1024, "the JPEG scan spans multiple datagrams");
                assert!(jpeg.starts_with(&[0xff, 0xd8]) && jpeg.ends_with(&[0xff, 0xd9]));
                frames += 1;
            }
            _ => {}
        }
        decoder.process(packet, &mut output).await.unwrap();
    }
    assert_eq!(frames, FRAME_COUNT);

    let mut reference_caps = caps;
    let Caps::CompressedVideo { width, height, .. } = &mut reference_caps else {
        unreachable!("the source selected JPEG");
    };
    *width = Dim::Fixed(64);
    *height = Dim::Fixed(48);
    let mut reference_parser = JpegParse::new();
    reference_parser
        .configure_pipeline(&reference_caps)
        .unwrap();
    let mut parsed_reference = CollectSink::default();
    feed_in_chunks(
        &mut reference_parser,
        &reference.stdout,
        &mut parsed_reference,
    )
    .await;
    let mut reference_decoder = MjpegDec::new();
    reference_decoder
        .configure_pipeline(&reference_caps)
        .unwrap();
    let mut expected = CollectSink::default();
    for packet in parsed_reference.packets {
        reference_decoder
            .process(packet, &mut expected)
            .await
            .unwrap();
    }
    let pixels = |sink: CollectSink| {
        sink.packets
            .into_iter()
            .filter_map(|packet| match packet {
                PipelinePacket::DataFrame(frame) => {
                    Some(frame.domain.as_system_slice().unwrap().to_vec())
                }
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    let (expected, actual) = (pixels(expected), pixels(output));
    assert_eq!(actual.len() as u64, FRAME_COUNT);
    assert_eq!(expected.len(), actual.len(), "{peer} reference frame count");
    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            actual == expected,
            "{peer} frame {index} differs from the reference"
        );
    }
}

#[tokio::test]
async fn udpsrc_receives_420_jpeg_from_ffmpeg() {
    if !ffmpeg_tool::have_ffmpeg() {
        eprintln!("skipping RTP/JPEG interop without FFmpeg");
        return;
    }
    let mut reference = ffmpeg_mjpeg();
    reference.args(["-f", "mjpeg", "-"]);
    receive_jpeg("FFmpeg 4:2:0", reference, |port| {
        let mut sender = ffmpeg_mjpeg();
        sender.args([
            "-f",
            "rtp",
            &format!("rtp://127.0.0.1:{port}?rtcpport={port}&pkt_size=256"),
        ]);
        sender
    })
    .await;
}

#[tokio::test]
async fn udpsrc_receives_422_jpeg_from_gstreamer() {
    if !gstreamer_tool::have_gstreamer_elements(&[
        "videotestsrc",
        "jpegenc",
        "rtpjpegpay",
        "udpsink",
        "fdsink",
    ]) {
        eprintln!("skipping RTP/JPEG interop without GStreamer elements");
        return;
    }
    let mut reference = gstreamer_mjpeg();
    reference.args(["!", "fdsink", "fd=1"]);
    receive_jpeg("GStreamer 4:2:2", reference, |port| {
        let mut sender = gstreamer_mjpeg();
        sender.args([
            "!",
            "rtpjpegpay",
            "mtu=256",
            "pt=26",
            "!",
            "udpsink",
            "host=127.0.0.1",
            &format!("port={port}"),
        ]);
        sender
    })
    .await;
}
