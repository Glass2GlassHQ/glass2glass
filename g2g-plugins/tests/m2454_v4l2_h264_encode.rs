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
use g2g_plugins::v4l2h264dec::V4l2H264Decoder;
use g2g_plugins::v4l2h264enc::V4l2SlH264Enc;
use h264_hardware_decode::system_frames_of;
use v4l2_encode::{encode, nv12_frame, HEIGHT, WIDTH};

const FRAME_COUNT: u64 = 12;
const KEYFRAME_INTERVAL: u64 = 5;
const START_CODE_LENGTH: usize = 4;
const NAL_TYPE_MASK: u8 = 0x1f;
const NAL_SLICE: u8 = 1;
const NAL_SPS: u8 = 7;
// the worst frame of this gradient measures a mean absolute luma error under 1 at the default quantizer
const MAX_MEAN_LUMA_ERROR: f64 = 2.0;

#[test]
fn the_hardware_decoder_reads_back_what_it_encodes() {
    if !V4l2SlH264Enc::is_available() {
        eprintln!("skipping: no V4L2 stateless H.264 encoder");
        return;
    }
    let frames: Vec<Vec<u8>> = (0..FRAME_COUNT).map(nv12_frame).collect();
    let mut encoder = V4l2SlH264Enc::new();
    encoder
        .set_property("keyframe-interval", PropValue::Uint(KEYFRAME_INTERVAL))
        .expect("keyframe-interval is a property");
    let access_units = encode(encoder, &frames);
    assert_eq!(access_units.len(), frames.len());
    for (index, access_unit) in (0..).zip(&access_units) {
        let first_nal = access_unit[START_CODE_LENGTH] & NAL_TYPE_MASK;
        let expected = if index % KEYFRAME_INTERVAL == 0 {
            NAL_SPS
        } else {
            NAL_SLICE
        };
        assert_eq!(first_nal, expected, "access unit {index}");
    }

    // one buffer holds more pictures than the decoder has picture buffers
    let stream = access_units.concat();
    let decoded = system_frames_of::<V4l2H264Decoder>("encoded gradient", &stream, (WIDTH, HEIGHT));
    assert_eq!(decoded.len(), frames.len());
    let luma_length = (WIDTH * HEIGHT) as usize;
    for (index, (source, picture)) in frames.iter().zip(&decoded).enumerate() {
        let error: u64 = source[..luma_length]
            .iter()
            .zip(&picture[..luma_length])
            .map(|(&a, &b)| u64::from(a.abs_diff(b)))
            .sum();
        let mean = error as f64 / luma_length as f64;
        assert!(
            mean < MAX_MEAN_LUMA_ERROR,
            "frame {index}: mean luma error {mean}"
        );
    }
}
