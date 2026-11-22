// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "udp-ingress")]

use std::net::UdpSocket;
use std::time::Duration;

use g2g_core::rtp::RtpHeader;
use g2g_core::runtime::SourceLoop;
use g2g_core::{Caps, Dim, PipelinePacket, VideoCodec};
use g2g_plugins::udpsrc::UdpSrc;

#[path = "common/collect_sink.rs"]
mod collect_sink;
#[path = "common/ivf.rs"]
mod ivf;

const FIXTURE: &[u8] = include_bytes!("fixtures/vp90-2-22-svc_1280x720_3_1packet.ivf");
const LAYER_LENGTHS: [usize; 3] = [4682, 1740, 5193];
const FRAGMENT_BYTES: usize = 1000;

#[tokio::test]
async fn vp9_rtp_rebuilds_the_three_layer_superframe() {
    let pictures = ivf::ivf_frames(FIXTURE);
    let &[picture] = pictures.as_slice() else {
        panic!("expected one IVF packet");
    };
    let receive_socket = UdpSocket::bind("127.0.0.1:0").unwrap();
    let destination = receive_socket.local_addr().unwrap();
    let mut source = UdpSrc::from_socket(receive_socket)
        .unwrap()
        .with_sdp(&format!(
            "v=0\r\nm=video {} RTP/AVP 96\r\na=rtpmap:96 VP9/90000\r\n",
            destination.port()
        ))
        .unwrap()
        .with_video_size(16, 16)
        .with_frame_limit(1);
    let caps = source.intercept_caps().await.unwrap();
    source.configure_pipeline(&caps).unwrap();

    let sender = UdpSocket::bind("127.0.0.1:0").unwrap();
    let mut sequence = u16::MAX - 2;
    let mut offset = 0;
    for (sid, length) in LAYER_LENGTHS.into_iter().enumerate() {
        let layer = &picture[offset..offset + length];
        offset += length;
        let fragments = layer.chunks(FRAGMENT_BYTES);
        let last = fragments.len() - 1;
        for (index, fragment) in fragments.enumerate() {
            let beginning = index == 0;
            let end = index == last;
            let marker = sid == LAYER_LENGTHS.len() - 1 && end;
            let rtp = RtpHeader {
                marker,
                payload_type: 96,
                sequence,
                timestamp: 90_000,
                ssrc: 1,
            };
            sequence = sequence.wrapping_add(1);
            let has_structure = sid == 0 && beginning;
            let descriptor = 0xa0
                | (u8::from(beginning) << 3)
                | (u8::from(end) << 2)
                | (u8::from(has_structure) << 1);
            let layer_indices = ((sid as u8) << 1) | u8::from(sid != 0);
            let mut datagram = rtp.to_bytes().to_vec();
            datagram.extend_from_slice(&[descriptor, 7, layer_indices, 7]);
            if has_structure {
                // Three spatial layers, each with width and height in network order.
                datagram.extend_from_slice(&[0x50, 1, 64, 0, 180, 2, 128, 1, 104, 5, 0, 2, 208]);
            }
            datagram.extend_from_slice(fragment);
            sender.send_to(&datagram, destination).unwrap();
        }
    }

    let mut output = collect_sink::CollectSink::default();
    let count = tokio::time::timeout(Duration::from_secs(5), source.run(&mut output))
        .await
        .expect("the full spatial picture arrived")
        .unwrap();
    assert_eq!(count, 1);
    let mut announced = false;
    let mut pictures = 0;
    for packet in output.packets {
        match packet {
            PipelinePacket::CapsChanged(Caps::CompressedVideo {
                codec: VideoCodec::Vp9,
                width: Dim::Fixed(1280),
                height: Dim::Fixed(720),
                ..
            }) => announced = true,
            PipelinePacket::DataFrame(frame) => {
                assert!(announced, "highest layer geometry precedes the picture");
                assert!(frame.timing.keyframe);
                let bytes = frame.domain.as_system_slice().unwrap();
                assert!(
                    bytes == picture,
                    "the RTP picture matches the IVF superframe"
                );
                pictures += 1;
            }
            _ => {}
        }
    }
    assert_eq!(pictures, 1);
}
