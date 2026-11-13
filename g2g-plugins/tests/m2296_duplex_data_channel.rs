// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(target_os = "linux", feature = "webrtc"))]

mod webrtc_duplex_common;
mod webrtc_streams_common;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use g2g_plugins::datachannel::MAX_MESSAGE_SIZE;
use g2g_plugins::webrtc_codec::WebRtcVideoCodec;
use g2g_plugins::webrtcduplex::{SdpChannel, SignalRole, WebRtcDuplexSession};
use tokio::sync::mpsc::UnboundedReceiver;

use webrtc_duplex_common::{run_peer, Received, ZeroClock};
use webrtc_streams_common::{payload, synthetic_stream};

const FRAMES: usize = 60;
// the handshake drops the first units
const MIN_RECEIVED: usize = FRAMES / 2;
const RUN_TIMEOUT: Duration = Duration::from_secs(40);
const LABEL: &str = "input";
const SMALL_MESSAGES: usize = 100;

fn messages(salt: u8) -> Vec<Vec<u8>> {
    let mut messages: Vec<Vec<u8>> = (0..SMALL_MESSAGES)
        .map(|index| payload(1 + index, salt.wrapping_add(index as u8)))
        .collect();
    messages.push(payload(MAX_MESSAGE_SIZE, salt));
    messages
}

fn drained(incoming: &mut UnboundedReceiver<Vec<u8>>) -> Vec<Vec<u8>> {
    let mut messages = Vec::new();
    while let Ok(message) = incoming.try_recv() {
        messages.push(message);
    }
    messages
}

#[tokio::test]
async fn data_channel_messages_cross_the_session_while_video_flows() {
    let codec = WebRtcVideoCodec::default();
    let units = Arc::new(synthetic_stream(codec, FRAMES));
    let (client_signal, host_signal) = SdpChannel::pair();
    let mut client = WebRtcDuplexSession::new(SignalRole::Offerer, client_signal, 1);
    let mut host = WebRtcDuplexSession::new(SignalRole::Answerer, host_signal, 1);
    let mut client_channel = client.open_data_channel(LABEL);
    let mut host_channel = host.open_data_channel(LABEL);

    let client_messages = messages(1);
    let host_messages = messages(2);
    for message in &client_messages {
        client_channel.outgoing.send(message.clone()).unwrap();
    }
    for message in &host_messages {
        host_channel.outgoing.send(message.clone()).unwrap();
    }

    let client_video = Arc::new(Mutex::new(Received::default()));
    let host_video = Arc::new(Mutex::new(Received::default()));
    let clock = ZeroClock;
    let (client_result, host_result) = tokio::time::timeout(RUN_TIMEOUT, async {
        tokio::join!(
            run_peer(
                codec,
                units.clone(),
                &mut client,
                client_video.clone(),
                &clock
            ),
            run_peer(codec, units.clone(), &mut host, host_video.clone(), &clock)
        )
    })
    .await
    .expect("P2P run completes in time");
    client_result.expect("client session");
    host_result.expect("host session");

    for (peer, video) in [("client", &client_video), ("host", &host_video)] {
        let received = video.lock().unwrap().units.len();
        assert!(
            received >= MIN_RECEIVED,
            "{peer}: {received} of {FRAMES} video units arrived"
        );
    }
    assert!(
        drained(&mut host_channel.incoming) == client_messages,
        "host receives every client message intact and in order"
    );
    assert!(
        drained(&mut client_channel.incoming) == host_messages,
        "client receives every host message intact and in order"
    );
}
