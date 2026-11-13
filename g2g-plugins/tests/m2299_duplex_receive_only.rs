// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(target_os = "linux", feature = "webrtc"))]

mod webrtc_duplex_common;
mod webrtc_streams_common;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use g2g_core::element::DynAsyncElement;
use g2g_core::runtime::{run_duplex_session, DynSourceLoop};
use g2g_plugins::webrtc_codec::WebRtcVideoCodec;
use g2g_plugins::webrtcduplex::{SdpChannel, SignalRole, WebRtcDuplexSession};
use tokio::sync::mpsc;

use webrtc_duplex_common::{run_peer, Received, RecordingSink, ZeroClock};
use webrtc_streams_common::synthetic_stream;

const FRAMES: usize = 60;
// the handshake drops the first units
const MIN_RECEIVED: usize = FRAMES / 2;
// str0m reports an ICE disconnect about 25 seconds after a peer leaves without closing
const RUN_TIMEOUT: Duration = Duration::from_secs(15);
const SIGNALING_DEPTH: usize = 4;
const LINK_CAPACITY: usize = 8;

#[tokio::test]
async fn receive_only_client_gets_host_video_with_no_send_source() {
    let codec = WebRtcVideoCodec::default();
    let units = Arc::new(synthetic_stream(codec, FRAMES));
    let (client_offer_tx, mut client_offer_rx) = mpsc::channel(SIGNALING_DEPTH);
    let (host_offer_tx, host_offer_rx) = mpsc::channel(SIGNALING_DEPTH);
    let (host_answer_tx, client_answer_rx) = mpsc::channel(SIGNALING_DEPTH);
    let mut client = WebRtcDuplexSession::new(
        SignalRole::Offerer,
        SdpChannel::from_halves(client_offer_tx, client_answer_rx),
        1,
    )
    .with_receive_only();
    let mut host = WebRtcDuplexSession::new(
        SignalRole::Answerer,
        SdpChannel::from_halves(host_answer_tx, host_offer_rx),
        1,
    );

    let client_video = Arc::new(Mutex::new(Received::default()));
    let host_video = Arc::new(Mutex::new(Received::default()));
    let clock = ZeroClock;
    let mut client_sink = RecordingSink {
        received: client_video.clone(),
    };
    let relay_offer = async {
        let offer: String = client_offer_rx.recv().await.expect("client offer");
        host_offer_tx.send(offer.clone()).await.expect("host reads");
        offer
    };
    let run_client = async {
        let sources: Vec<&mut dyn DynSourceLoop> = Vec::new();
        let sinks: Vec<&mut dyn DynAsyncElement> = vec![&mut client_sink];
        run_duplex_session(sources, &mut client, sinks, &clock, LINK_CAPACITY).await
    };
    let (offer, client_result, host_result) = tokio::time::timeout(RUN_TIMEOUT, async {
        tokio::join!(
            relay_offer,
            run_client,
            run_peer(codec, units.clone(), &mut host, host_video.clone(), &clock)
        )
    })
    .await
    .expect("client ends on the host's close, before an ICE timeout");
    client_result.expect("client session");
    host_result.expect("host session");

    assert!(offer.contains("a=recvonly"), "{offer}");
    assert!(!offer.contains("a=sendrecv"), "{offer}");
    let received = client_video.lock().unwrap().units.len();
    assert!(
        received >= MIN_RECEIVED,
        "client: {received} of {FRAMES} video units arrived"
    );
    assert_eq!(host_video.lock().unwrap().units.len(), 0);
}
