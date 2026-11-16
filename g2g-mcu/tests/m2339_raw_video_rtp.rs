// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

mod util;

use g2g_core::error::G2gError;
use g2g_core::rtp::{RtpHeader, RTP_HEADER_LEN};
use g2g_core::st2110video::{Sampling, St2110VideoDepacketizer};
use g2g_core::staticpool::StaticLendRing;
use g2g_core::{RawVideoFormat, StaticTransform};
use g2g_mcu::rtp::PacketSender;
use g2g_mcu::RawVideoRtpSend;
use util::{block_on, frame_of, payload};

#[derive(Default)]
struct Collect {
    packets: Vec<Vec<u8>>,
}

impl PacketSender for Collect {
    async fn send(
        &mut self,
        header: &[u8; RTP_HEADER_LEN],
        payload: &[u8],
    ) -> Result<(), G2gError> {
        let mut packet = header.to_vec();
        packet.extend_from_slice(payload);
        self.packets.push(packet);
        Ok(())
    }
}

const SIDE: usize = 96;
const IMAGE_BYTES: usize = SIDE * SIDE;
const PACKET_BYTES: usize = 1400;
const PAYLOAD_TYPE: u8 = 96;
const SSRC: u32 = 0x6262;

#[test]
fn a_gray_frame_reaches_a_422_receiver_and_passes_through() {
    let image: Vec<u8> = (0..IMAGE_BYTES).map(|i| (i * 7 + 3) as u8).collect();
    let ring: StaticLendRing<1, IMAGE_BYTES> = StaticLendRing::new();
    let mut send = RawVideoRtpSend::<_, PACKET_BYTES>::new(
        Collect::default(),
        Sampling::Gray8,
        SIDE,
        SIDE,
        PAYLOAD_TYPE,
        SSRC,
    )
    .expect("96x96 has a 4:2:2 layout");

    let out = block_on(send.process(frame_of(&ring, &image, 0, 0)))
        .expect("sends")
        .expect("passes the frame on");
    assert_eq!(payload(&out), image.as_slice());
    drop(out);

    let packets = send.free().packets;
    assert!(packets.len() > 1, "a 96x96 frame needs several packets");
    assert!(packets.iter().all(|packet| packet.len() <= PACKET_BYTES));
    let mut receiver = St2110VideoDepacketizer::new(RawVideoFormat::Yuyv, SIDE, SIDE).unwrap();
    let yuyv = packets
        .iter()
        .filter_map(|packet| receiver.depacketize(packet))
        .last()
        .expect("the marker packet completes the frame")
        .bytes;
    let luma: Vec<u8> = yuyv.iter().step_by(2).copied().collect();
    assert_eq!(luma, image);
}

#[test]
fn frame_bytes_past_the_picture_go_out_as_the_marker_packet_extension() {
    const EXTENSION_ID: u8 = 1;
    const LABEL: &[u8] = &[230, b'p', b'e', b'r', b's', b'o', b'n'];
    const FRAME_BYTES: usize = IMAGE_BYTES + LABEL.len();
    let image: Vec<u8> = (0..IMAGE_BYTES).map(|i| (i * 5 + 1) as u8).collect();
    let labelled = [image.as_slice(), LABEL].concat();
    let ring: StaticLendRing<1, FRAME_BYTES> = StaticLendRing::new();
    let mut send = RawVideoRtpSend::<_, PACKET_BYTES>::new(
        Collect::default(),
        Sampling::Gray8,
        SIDE,
        SIDE,
        PAYLOAD_TYPE,
        SSRC,
    )
    .expect("96x96 has a 4:2:2 layout")
    .with_trailer_extension(EXTENSION_ID);

    drop(block_on(send.process(frame_of(&ring, &labelled, 0, 0))).expect("sends"));

    let packets = send.free().packets;
    let (last, earlier) = packets.split_last().unwrap();
    assert_eq!(RtpHeader::find_extension(last, EXTENSION_ID), Some(LABEL));
    assert!(earlier
        .iter()
        .all(|packet| RtpHeader::find_extension(packet, EXTENSION_ID).is_none()));
    let mut receiver = St2110VideoDepacketizer::new(RawVideoFormat::Yuyv, SIDE, SIDE).unwrap();
    let yuyv = packets
        .iter()
        .find_map(|packet| receiver.depacketize(packet))
        .expect("the marker packet completes the frame")
        .bytes;
    let luma: Vec<u8> = yuyv.iter().step_by(2).copied().collect();
    assert_eq!(luma, image);
}

#[test]
fn refuses_a_packet_buffer_too_small_for_one_pixel_group() {
    let sender = RawVideoRtpSend::<_, RTP_HEADER_LEN>::new(
        Collect::default(),
        Sampling::Gray8,
        SIDE,
        SIDE,
        PAYLOAD_TYPE,
        SSRC,
    );
    assert!(sender.is_none());
}
