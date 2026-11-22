// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "udp-ingress")]

#[path = "common/udp_rtp_video.rs"]
mod udp_rtp_video;

use g2g_core::VideoCodec;
use udp_rtp_video::receive;

const AV1: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/av1_321x181.ivf"
);

#[tokio::test]
async fn udp_av1_uses_sdp_and_reassembles_ffmpeg_rtp_packets() {
    receive(AV1, VideoCodec::Av1, 321, 181, 25, |source, port| {
        source
            .with_sdp(&format!(
                "v=0\r\nm=video {port} RTP/AVP 96\r\na=rtpmap:96 AV1/90000\r\na=framerate:25\r\n"
            ))
            .unwrap()
    })
    .await;
}
