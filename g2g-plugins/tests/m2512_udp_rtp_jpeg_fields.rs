// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(feature = "udp-ingress", feature = "mjpeg", feature = "mjpeg-encode"))]

use std::net::UdpSocket;
use std::time::Duration;

use g2g_core::rtp::RtpHeader;
use g2g_core::runtime::SourceLoop;
use g2g_core::{AsyncElement, BitstreamAlignment, Caps, Dim, Interlace, PipelinePacket};
use g2g_plugins::{mjpegdec::MjpegDec, stillparse::JpegParse, udpsrc::UdpSrc};
use jpeg_encoder::{ColorType, Encoder, SamplingFactor};

#[path = "common/collect_sink.rs"]
mod collect_sink;
use collect_sink::CollectSink;

const WIDTH: u16 = 32;
const FIELD_HEIGHT: u16 = 16;
const FRAME_HEIGHT: u16 = FIELD_HEIGHT * 2;
const QUALITY: u8 = 50;
const ODD: u8 = 1;
const EVEN: u8 = 2;
const REPEAT: u8 = 3;

fn encode_scan(height: u16, rgb: [u8; 3]) -> Vec<u8> {
    let mut jpeg = Vec::new();
    let mut encoder = Encoder::new(&mut jpeg, QUALITY);
    encoder.set_sampling_factor(SamplingFactor::F_2_2);
    encoder
        .encode(
            &rgb.repeat(usize::from(WIDTH) * usize::from(height)),
            WIDTH,
            height,
            ColorType::Rgb,
        )
        .unwrap();
    // This encoder writes one scan with the standard tables at the declared quality.
    let sos = jpeg.windows(2).position(|b| b == [0xff, 0xda]).unwrap();
    let length = usize::from(u16::from_be_bytes([jpeg[sos + 2], jpeg[sos + 3]]));
    jpeg[sos + 2 + length..]
        .strip_suffix(&[0xff, 0xd9])
        .unwrap()
        .to_vec()
}

fn packet(
    sequence: u16,
    timestamp: u32,
    field: u8,
    height: u16,
    offset: usize,
    marker: bool,
    scan: &[u8],
) -> Vec<u8> {
    let mut bytes = RtpHeader {
        marker,
        payload_type: 26,
        sequence,
        timestamp,
        ssrc: 1,
    }
    .to_bytes()
    .to_vec();
    bytes.push(field);
    bytes.extend_from_slice(&(offset as u32).to_be_bytes()[1..]);
    bytes.extend_from_slice(&[1, QUALITY, (WIDTH / 8) as u8, (height / 8) as u8]);
    bytes.extend_from_slice(scan);
    bytes
}

#[tokio::test]
async fn jpeg_rtp_weaves_fields_and_returns_to_progressive_at_the_same_size() {
    let odd = encode_scan(FIELD_HEIGHT, [30, 30, 210]);
    let even = encode_scan(FIELD_HEIGHT, [210, 30, 30]);
    let full = encode_scan(FRAME_HEIGHT, [30, 210, 30]);
    let socket = UdpSocket::bind("127.0.0.1:0").unwrap();
    let address = socket.local_addr().unwrap();
    let mut source = UdpSrc::from_socket(socket)
        .unwrap()
        .with_sdp(&format!(
            "v=0\r\nm=video {} RTP/AVP 26\r\na=framerate:25\r\n",
            address.port()
        ))
        .unwrap()
        .with_frame_limit(3);
    let caps = source.intercept_caps().await.unwrap();
    source.configure_pipeline(&caps).unwrap();

    let sender = UdpSocket::bind("127.0.0.1:0").unwrap();
    let split = even.len() / 2;
    for datagram in [
        packet(u16::MAX, 90_000, ODD, FIELD_HEIGHT, 0, true, &odd),
        packet(0, 91_800, EVEN, FIELD_HEIGHT, 0, false, &even[..split]),
        packet(1, 91_800, EVEN, FIELD_HEIGHT, split, true, &even[split..]),
        packet(2, 93_600, REPEAT, FIELD_HEIGHT, 0, true, &odd),
        packet(3, 97_200, 0, FRAME_HEIGHT, 0, true, &full),
    ] {
        sender.send_to(&datagram, address).unwrap();
    }
    let mut received = CollectSink::default();
    let count = tokio::time::timeout(Duration::from_secs(5), source.run(&mut received))
        .await
        .expect("the JPEG frames arrived")
        .unwrap();
    assert_eq!(count, 3);

    let mut parser = JpegParse::new();
    parser.configure_pipeline(&caps).unwrap();
    let mut parsed = CollectSink::default();
    let mut alignment = None;
    let mut frame_count = 0;
    for packet in received.packets {
        match &packet {
            PipelinePacket::CapsChanged(Caps::CompressedVideo {
                height: Dim::Fixed(height),
                alignment: mode,
                ..
            }) => {
                assert_eq!(*height, u32::from(FRAME_HEIGHT));
                alignment = Some(*mode);
            }
            PipelinePacket::DataFrame(frame) => {
                let expected = if frame_count < 2 {
                    BitstreamAlignment::FieldPair
                } else {
                    BitstreamAlignment::Unspecified
                };
                assert_eq!(alignment, Some(expected));
                assert_eq!(frame.timing.pts_ns, frame_count * 40_000_000);
                frame_count += 1;
            }
            _ => {}
        }
        parser.process(packet, &mut parsed).await.unwrap();
    }
    assert_eq!(frame_count, 3);

    let mut decoder = MjpegDec::new();
    decoder.configure_pipeline(&caps).unwrap();
    let mut decoded = CollectSink::default();
    let mut compressed_modes = Vec::new();
    for packet in parsed.packets {
        if let PipelinePacket::CapsChanged(Caps::CompressedVideo { alignment, .. }) = &packet {
            compressed_modes.push(*alignment);
        }
        decoder.process(packet, &mut decoded).await.unwrap();
    }
    assert_eq!(
        compressed_modes,
        [
            BitstreamAlignment::FieldPair,
            BitstreamAlignment::Unspecified
        ]
    );
    let mut raw_modes = Vec::new();
    let mut frames = 0;
    for packet in decoded.packets {
        match packet {
            PipelinePacket::CapsChanged(Caps::RawVideo {
                height, interlace, ..
            }) => {
                assert_eq!(height, Dim::Fixed(u32::from(FRAME_HEIGHT)));
                raw_modes.push(interlace);
            }
            PipelinePacket::DataFrame(frame) => {
                let bytes = frame.domain.as_system_slice().unwrap();
                let rows = bytes.chunks_exact(usize::from(WIDTH) * 4);
                assert_eq!(rows.len(), usize::from(FRAME_HEIGHT));
                for (row, pixels) in rows.enumerate() {
                    let primary = match frames {
                        0 if row % 2 == 0 => 0,
                        0 | 1 => 2,
                        _ => 1,
                    };
                    for pixel in pixels.as_chunks::<4>().0 {
                        assert!(pixel[primary] > 180, "frame {frames} row {row}: {pixel:?}");
                        assert!(pixel[..3]
                            .iter()
                            .enumerate()
                            .all(|(c, value)| c == primary || *value < 60));
                    }
                }
                frames += 1;
            }
            _ => {}
        }
    }
    assert_eq!(frames, 3);
    assert_eq!(raw_modes, [Interlace::Interleaved, Interlace::Any]);
}
