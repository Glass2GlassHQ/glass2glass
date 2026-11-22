// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::net::UdpSocket;
use std::process::Command;
use std::time::Duration;

use g2g_core::runtime::SourceLoop;
use g2g_core::{AsyncElement, ByteStreamEncoding, Caps, Dim, PipelinePacket, Rate, VideoCodec};
use g2g_plugins::{ivfdemux::IvfDemux, udpsrc::UdpSrc};

#[path = "collect_sink.rs"]
mod collect_sink;
#[path = "ffmpeg_tool.rs"]
mod ffmpeg_tool;

use collect_sink::{feed_in_chunks, CollectSink};

const FRAME_COUNT: u64 = 6;
const DATAGRAM_BYTES: usize = 256;
const RTP_TICK_NS: u64 = 1_000_000_000 / 90_000 + 1;
const DEADLINE: Duration = Duration::from_secs(10);

// Copy the same packets to IVF and RTP so neither path re-encodes the fixture.
fn copy_frames(file: &str, paced: bool, count: u64) -> Command {
    let mut command = Command::new("ffmpeg");
    command.args(["-hide_banner", "-loglevel", "error"]);
    if paced {
        command.arg("-re");
    }
    command
        .args(["-i", file, "-map", "0:v:0", "-c", "copy", "-frames:v"])
        .arg(count.to_string());
    command
}

async fn frames_from_ivf(file: &str, codec: VideoCodec) -> Vec<Vec<u8>> {
    let ivf = copy_frames(file, false, FRAME_COUNT)
        .args(["-f", "ivf", "-"])
        .output()
        .expect("ffmpeg starts");
    assert!(
        ivf.status.success(),
        "copy the reference frames: {}",
        String::from_utf8_lossy(&ivf.stderr)
    );
    let mut demux = IvfDemux::new();
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Ivf,
        })
        .unwrap();
    let mut sink = CollectSink::default();
    feed_in_chunks(&mut demux, &ivf.stdout, &mut sink).await;
    sink.packets
        .into_iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => {
                let bytes = frame.domain.as_system_slice().unwrap();
                let bytes = if codec == VideoCodec::Av1 {
                    // FFmpeg's IVF muxer prefixes an empty temporal delimiter OBU.
                    bytes.strip_prefix(&[0x12, 0]).expect("IVF AV1 delimiter")
                } else {
                    bytes
                };
                Some(bytes.to_vec())
            }
            _ => None,
        })
        .collect()
}

pub(crate) async fn receive(
    fixture: &'static str,
    codec: VideoCodec,
    width: u32,
    height: u32,
    fps: u32,
    configure: impl FnOnce(UdpSrc, u16) -> UdpSrc,
) {
    if !ffmpeg_tool::have_ffmpeg() {
        eprintln!("skipping RTP interop without ffmpeg");
        return;
    }
    if codec == VideoCodec::Av1 {
        let probe = copy_frames(fixture, false, 1)
            .args([
                "-strict",
                "experimental",
                "-packetsize",
                "256",
                "-f",
                "rtp",
                "pipe:1",
            ])
            .output()
            .expect("FFmpeg RTP capability probe starts");
        let stderr = String::from_utf8_lossy(&probe.stderr);
        if !probe.status.success() && stderr.contains("Unsupported codec av1") {
            eprintln!("skipping AV1 interop with an older FFmpeg RTP muxer");
            return;
        }
        assert!(
            probe.status.success(),
            "FFmpeg RTP capability probe: {stderr}"
        );
    }
    let expected = frames_from_ivf(fixture, codec).await;
    assert_eq!(expected.len() as u64, FRAME_COUNT);
    assert!(
        expected[0].len() > DATAGRAM_BYTES,
        "the payloader must fragment the first frame"
    );

    let socket = UdpSocket::bind("127.0.0.1:0").expect("bind RTP");
    let port = socket.local_addr().unwrap().port();
    let source = UdpSrc::from_socket(socket)
        .unwrap()
        .with_frame_limit(FRAME_COUNT);
    let mut source = configure(source, port);
    let caps = source.intercept_caps().await.unwrap();
    assert!(matches!(caps, Caps::CompressedVideo { codec: actual, .. } if actual == codec));
    source.configure_pipeline(&caps).unwrap();

    let ffmpeg = tokio::task::spawn_blocking(move || {
        let url = format!("rtp://127.0.0.1:{port}?rtcpport={port}&pkt_size={DATAGRAM_BYTES}");
        let output = copy_frames(fixture, true, FRAME_COUNT)
            // FFmpeg still gates its VP9 and AV1 RTP packetizers behind this flag.
            .args(["-strict", "experimental", "-payload_type", "96", "-f", "rtp", &url])
            .output()
            .expect("FFmpeg RTP sender starts");
        assert!(
            output.status.success(),
            "FFmpeg RTP sender: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    });
    let mut output = CollectSink::default();
    let count = tokio::time::timeout(DEADLINE, source.run(&mut output)).await;
    ffmpeg.await.expect("FFmpeg sends RTP");
    assert_eq!(
        count.expect("receiver finishes").expect("receiver runs"),
        FRAME_COUNT
    );

    let mut shapes = 0;
    let mut frame_index = 0;
    let mut interframes = 0;
    for packet in output.packets {
        match packet {
            PipelinePacket::CapsChanged(Caps::CompressedVideo {
                codec: actual,
                width: Dim::Fixed(w),
                height: Dim::Fixed(h),
                framerate: Rate::Fixed(rate),
                ..
            }) => {
                assert_eq!((actual, w, h, rate), (codec, width, height, fps << 16));
                shapes += 1;
            }
            PipelinePacket::DataFrame(frame) => {
                assert_eq!(
                    shapes, 1,
                    "geometry precedes its frames and is emitted once"
                );
                if frame_index == 0 {
                    assert!(frame.timing.keyframe);
                }
                interframes += usize::from(!frame.timing.keyframe);
                let actual = frame.domain.as_system_slice().unwrap();
                let reference = &expected[frame_index];
                assert!(
                    actual == reference,
                    "frame {frame_index} differs from the reference ({} vs {} bytes)",
                    actual.len(),
                    reference.len()
                );
                let expected_pts = frame_index as u64 * 1_000_000_000 / u64::from(fps);
                assert!(frame.timing.pts_ns.abs_diff(expected_pts) <= RTP_TICK_NS);
                frame_index += 1;
            }
            _ => {}
        }
    }
    assert_eq!(frame_index, expected.len());
    assert!(interframes > 0);
}
