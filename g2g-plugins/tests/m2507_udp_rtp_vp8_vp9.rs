// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "udp-ingress")]

#[path = "common/udp_rtp_video.rs"]
mod udp_rtp_video;

use g2g_core::runtime::SourceLoop;
use g2g_core::{PropValue, VideoCodec};
use udp_rtp_video::receive;

const VP8: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/vp8_64x48_25fps_gop10_offset.ogg"
);
const VP9: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/vp9_320x240.ivf"
);

#[tokio::test]
async fn udp_vp8_reassembles_ffmpeg_rtp_packets() {
    receive(VP8, VideoCodec::Vp8, 64, 48, 25, |mut source, _port| {
        source
            .set_property("encoding-name", PropValue::Str("VP8".into()))
            .unwrap();
        source.with_framerate(25)
    })
    .await;
}

#[tokio::test]
async fn udp_vp9_uses_sdp_and_reassembles_ffmpeg_rtp_packets() {
    receive(VP9, VideoCodec::Vp9, 320, 240, 30, |source, port| {
        source
            .with_sdp(&format!(
                "v=0\r\nm=video {port} RTP/AVP 96\r\na=rtpmap:96 VP9/90000\r\na=framerate:30\r\n"
            ))
            .unwrap()
    })
    .await;
}
