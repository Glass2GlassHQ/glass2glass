// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! Every video codec the WebRTC elements carry round-trips a P2P duplex session
//! over localhost UDP: two `WebRtcDuplexSession`s built for the codec negotiate
//! it, and each received access unit is byte-identical to one that was sent,
//! which proves str0m's packetizer and depacketizer keep g2g's framing (Annex-B,
//! raw VP8 / VP9 frames, low-overhead AV1 temporal units). Default CI gate, no
//! media server.

#![cfg(all(target_os = "linux", feature = "webrtc"))]

mod webrtc_duplex_common;
mod webrtc_streams_common;

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use g2g_core::Caps;
use g2g_plugins::webrtc_codec::WebRtcVideoCodec;
use g2g_plugins::webrtcduplex::{SdpChannel, SignalRole, WebRtcDuplexSession};

use webrtc_duplex_common::{run_peer, Received, ZeroClock};
use webrtc_streams_common::synthetic_stream;

const FRAMES: usize = 60;
/// The handshake drops the first units, so most but not all arrive.
const MIN_RECEIVED: usize = FRAMES / 2;
const RUN_TIMEOUT: Duration = Duration::from_secs(40);

async fn round_trip(codec: WebRtcVideoCodec) -> (Arc<Vec<Vec<u8>>>, Received, Received) {
    let units = Arc::new(synthetic_stream(codec, FRAMES));
    let (chan_a, chan_b) = SdpChannel::pair();
    let mut session_a =
        WebRtcDuplexSession::new(SignalRole::Offerer, chan_a, 1).with_video_codec(codec);
    let mut session_b =
        WebRtcDuplexSession::new(SignalRole::Answerer, chan_b, 1).with_video_codec(codec);
    let received_a = Arc::new(Mutex::new(Received::default()));
    let received_b = Arc::new(Mutex::new(Received::default()));
    let clock = ZeroClock;

    let (result_a, result_b) = tokio::time::timeout(RUN_TIMEOUT, async {
        tokio::join!(
            run_peer(
                codec,
                units.clone(),
                &mut session_a,
                received_a.clone(),
                &clock
            ),
            run_peer(
                codec,
                units.clone(),
                &mut session_b,
                received_b.clone(),
                &clock
            )
        )
    })
    .await
    .unwrap_or_else(|_| panic!("{codec:?} P2P run completes in time"));
    result_a.unwrap_or_else(|e| panic!("{codec:?} peer A: {e:?}"));
    result_b.unwrap_or_else(|e| panic!("{codec:?} peer B: {e:?}"));
    let take = |cell: &Arc<Mutex<Received>>| core::mem::take(&mut *cell.lock().unwrap());
    (units, take(&received_a), take(&received_b))
}

fn assert_received(codec: WebRtcVideoCodec, sent: &[Vec<u8>], received: &Received, peer: &str) {
    let caps = received
        .caps
        .as_ref()
        .unwrap_or_else(|| panic!("{codec:?} {peer}: caps announced before frames"));
    assert!(
        matches!(caps, Caps::CompressedVideo { codec: c, .. } if *c == codec.video_codec()),
        "{codec:?} {peer}: output caps carry the negotiated codec, got {caps:?}"
    );
    assert!(
        received.units.len() >= MIN_RECEIVED,
        "{codec:?} {peer}: {} of {FRAMES} units arrived",
        received.units.len()
    );
    let sent_set: HashSet<&[u8]> = sent.iter().map(Vec::as_slice).collect();
    for (unit, _) in &received.units {
        assert!(
            sent_set.contains(unit.as_slice()),
            "{codec:?} {peer}: a received unit is not byte-identical to any sent unit"
        );
    }
    let keyframe_units: HashSet<&[u8]> = sent
        .iter()
        .step_by(webrtc_streams_common::GOP)
        .map(Vec::as_slice)
        .collect();
    let keyframes_flagged = received
        .units
        .iter()
        .filter(|(unit, keyframe)| *keyframe && keyframe_units.contains(unit.as_slice()))
        .count();
    assert!(
        keyframes_flagged >= 1,
        "{codec:?} {peer}: a received keyframe carries the keyframe flag"
    );
    let inter_flagged = received
        .units
        .iter()
        .filter(|(unit, keyframe)| *keyframe && !keyframe_units.contains(unit.as_slice()))
        .count();
    assert_eq!(
        inter_flagged, 0,
        "{codec:?} {peer}: no inter frame is flagged as a keyframe"
    );
}

#[tokio::test]
async fn every_codec_round_trips_a_p2p_session() {
    for codec in WebRtcVideoCodec::ALL {
        let (sent, received_a, received_b) = round_trip(codec).await;
        eprintln!(
            "{codec:?}: A received {} units, B received {} units",
            received_a.units.len(),
            received_b.units.len()
        );
        assert_received(codec, &sent, &received_a, "peer A");
        assert_received(codec, &sent, &received_b, "peer B");
    }
}
