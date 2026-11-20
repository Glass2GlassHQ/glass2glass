// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(target_os = "linux", feature = "v4l2"))]

#[path = "common/annexb.rs"]
mod annexb;
mod decode_ref;
#[path = "common/h264_hardware_decode.rs"]
mod h264_hardware_decode;
#[path = "common/v4l2_encode.rs"]
mod v4l2_encode;

use g2g_core::element::AsyncElement;
use g2g_core::property::PropValue;
use g2g_plugins::v4l2vp8enc::V4l2SlVp8Enc;
use v4l2_encode::{encode, nv12_frame, HEIGHT, WIDTH};

const FRAME_COUNT: u64 = 12;
const KEYFRAME_INTERVAL: u64 = 5;
// RFC 6386 section 9.1: a clear low bit of the frame tag marks a keyframe, whose start code and sizes follow the tag
const INTER_FRAME_BIT: u8 = 0x01;
const FRAME_TAG_LENGTH: usize = 3;
const KEYFRAME_START_CODE: [u8; 3] = [0x9d, 0x01, 0x2a];
const SIZE_BITS: u16 = 0x3fff;
const FINE_QUANTIZER: u64 = 10;
const COARSE_QUANTIZER: u64 = 100;

fn is_keyframe(frame: &[u8]) -> bool {
    frame[0] & INTER_FRAME_BIT == 0
}

fn keyframe_size(frame: &[u8]) -> (u32, u32) {
    let start_code = &frame[FRAME_TAG_LENGTH..FRAME_TAG_LENGTH + KEYFRAME_START_CODE.len()];
    assert_eq!(start_code, KEYFRAME_START_CODE);
    let sizes = FRAME_TAG_LENGTH + KEYFRAME_START_CODE.len();
    let size = |offset: usize| {
        u32::from(
            u16::from_le_bytes([frame[sizes + offset], frame[sizes + offset + 1]]) & SIZE_BITS,
        )
    };
    (size(0), size(2))
}

fn encoder(quantizer: u64) -> V4l2SlVp8Enc {
    let mut encoder = V4l2SlVp8Enc::new();
    encoder
        .set_property("keyframe-interval", PropValue::Uint(KEYFRAME_INTERVAL))
        .expect("keyframe-interval is a property");
    encoder
        .set_property("quantizer", PropValue::Uint(quantizer))
        .expect("quantizer is a property");
    encoder
}

#[test]
fn keyframes_follow_the_interval_and_carry_the_picture_size() {
    if !V4l2SlVp8Enc::is_available() {
        eprintln!("skipping: no V4L2 stateless VP8 encoder");
        return;
    }
    let frames: Vec<Vec<u8>> = (0..FRAME_COUNT).map(nv12_frame).collect();
    let coded = encode(encoder(FINE_QUANTIZER), &frames);
    assert_eq!(coded.len(), frames.len());
    for (index, frame) in (0..).zip(&coded) {
        let keyframe = index % KEYFRAME_INTERVAL == 0;
        assert_eq!(is_keyframe(frame), keyframe, "frame {index}");
        if keyframe {
            assert_eq!(keyframe_size(frame), (WIDTH, HEIGHT), "frame {index}");
        }
    }
}

#[test]
fn a_coarser_quantizer_codes_fewer_bytes() {
    if !V4l2SlVp8Enc::is_available() {
        eprintln!("skipping: no V4L2 stateless VP8 encoder");
        return;
    }
    let frames: Vec<Vec<u8>> = (0..FRAME_COUNT).map(nv12_frame).collect();
    let bytes = |quantizer| encode(encoder(quantizer), &frames).concat().len();
    let (fine, coarse) = (bytes(FINE_QUANTIZER), bytes(COARSE_QUANTIZER));
    assert!(
        coarse < fine,
        "quantizer {COARSE_QUANTIZER} coded {coarse} bytes, {FINE_QUANTIZER} coded {fine}"
    );
}
