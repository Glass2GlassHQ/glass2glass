// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(target_os = "linux", feature = "v4l2"))]

#[path = "common/annexb.rs"]
mod annexb;
mod decode_ref;
#[path = "common/h264_hardware_decode.rs"]
mod h264_hardware_decode;

use g2g_plugins::v4l2h264dec::{
    find_v4l2_h264_decoder, V4l2DecodeError, V4l2H264Decoder, V4l2SlH264Dec,
};
use h264_hardware_decode::{h264_caps, BFRAMES};

fn on_a_v4l2_decoder(test: fn()) {
    match find_v4l2_h264_decoder() {
        Ok(_) => test(),
        Err(V4l2DecodeError::NoStatelessH264Decoder) => {
            eprintln!("skipping: no V4L2 stateless H.264 decoder");
        }
        Err(error) => panic!("look for a V4L2 stateless H.264 decoder: {error:?}"),
    }
}

#[test]
fn system_output_is_bit_exact_in_display_order() {
    on_a_v4l2_decoder(
        h264_hardware_decode::system_output_is_bit_exact_in_display_order::<V4l2H264Decoder>,
    );
}

#[test]
fn latency_counts_the_pictures_the_reorder_holds() {
    on_a_v4l2_decoder(
        h264_hardware_decode::latency_counts_the_pictures_the_reorder_holds::<V4l2H264Decoder>,
    );
}

#[test]
fn a_flush_resumes_bit_exact_from_the_next_keyframe() {
    on_a_v4l2_decoder(
        h264_hardware_decode::a_flush_resumes_bit_exact_from_the_next_keyframe::<V4l2H264Decoder>,
    );
}

#[test]
fn high_bit_depth_stream_is_refused() {
    on_a_v4l2_decoder(h264_hardware_decode::high_bit_depth_stream_is_refused::<V4l2H264Decoder>);
}

#[test]
fn a_damaged_access_unit_drops_pictures_to_the_next_keyframe_and_asks_for_one() {
    on_a_v4l2_decoder(
        h264_hardware_decode::a_damaged_access_unit_drops_pictures_to_the_next_keyframe_and_asks_for_one::<V4l2H264Decoder>,
    );
}

#[test]
fn a_flush_resumes_bit_exact_from_a_recovery_point_i_picture() {
    on_a_v4l2_decoder(
        h264_hardware_decode::a_flush_resumes_bit_exact_from_a_recovery_point_i_picture::<
            V4l2H264Decoder,
        >,
    );
}

#[test]
fn decodebin_offers_it_for_h264_on_a_v4l2_decoder() {
    on_a_v4l2_decoder(|| assert!(V4l2SlH264Dec::accepts_input(&h264_caps(BFRAMES.picture))));
}

#[test]
fn decode_order_timestamps_leave_in_rising_order() {
    on_a_v4l2_decoder(
        h264_hardware_decode::decode_order_timestamps_leave_in_rising_order::<V4l2H264Decoder>,
    );
}
