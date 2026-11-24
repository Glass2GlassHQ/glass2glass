// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "webtransport")]

use std::time::Duration;

use g2g_core::runtime::{run_simple_pipeline, LatencyProfile};
use g2g_core::{AsyncElement, PipelinePacket};
use g2g_plugins::datachannel::DataChannelEnds;
use g2g_plugins::remotewtsink::RemoteWtSink;
use g2g_plugins::remotewtsrc::RemoteWtSrc;

mod webtransport_common;

use webtransport_common::{
    test_caps, test_frame, CollectSink, NullOut, TestCert, ZeroClock, FRAME_LEN,
};

const FRAMES: u8 = 4;
const SESSION_TIMEOUT: Duration = Duration::from_secs(20);
const CLIENT_MESSAGE: &[u8] = b"token from the client";
const HOST_MESSAGE: &[u8] = b"cursor from the host";

async fn data_channel_runs_both_ways_beside_the_video(scheme: &str) {
    let tls = TestCert::generate(&format!("m2317-{scheme}"));
    let mut remote =
        RemoteWtSink::new(format!("{scheme}://127.0.0.1:0")).with_listener(tls.cert(), tls.key());
    let mut host_channel = remote.open_data_channel();
    let port = remote.listen().expect("bind the listening sink").port();

    let mut source = RemoteWtSrc::new("127.0.0.1:0".parse().unwrap())
        .with_caller(format!("{scheme}://127.0.0.1:{port}"), tls.hash_hex.clone())
        .with_frame_limit(u64::from(FRAMES));
    let DataChannelEnds {
        outgoing: client_outgoing,
        incoming: mut client_incoming,
    } = source.open_data_channel();
    client_outgoing
        .send(CLIENT_MESSAGE.to_vec())
        .expect("queue the client message");
    // a client with no input finishes its side after its first message, and still hears the host
    drop(client_outgoing);

    let (client_received_sender, client_received) = tokio::sync::oneshot::channel();
    let client = async move {
        let message = client_incoming.recv().await;
        client_received_sender
            .send(())
            .expect("the host waits for the client");
        message
    };
    let host = async {
        remote.configure_pipeline(&test_caps()).expect("configure");
        let mut null = NullOut;
        // the first frame makes the sink accept the client
        remote
            .process(
                PipelinePacket::DataFrame(test_frame(0, FRAME_LEN)),
                &mut null,
            )
            .await
            .expect("first frame");
        let from_client = host_channel.incoming.recv().await;
        host_channel
            .outgoing
            .send(HOST_MESSAGE.to_vec())
            .expect("queue the host message");
        // Eos closes the session, so the reply must land first
        client_received.await.expect("client got the host message");
        for i in 1..FRAMES {
            remote
                .process(
                    PipelinePacket::DataFrame(test_frame(i, FRAME_LEN)),
                    &mut null,
                )
                .await
                .expect("frame");
        }
        remote
            .process(PipelinePacket::Eos, &mut null)
            .await
            .expect("eos");
        from_client
    };
    let mut sink = CollectSink::default();
    let receive = run_simple_pipeline(
        &mut source,
        &mut sink,
        &ZeroClock,
        LatencyProfile::Live.link_capacity(),
    );
    let (from_client, from_host, received) = tokio::time::timeout(SESSION_TIMEOUT, async {
        tokio::join!(host, client, receive)
    })
    .await
    .expect("the session finishes in time");

    assert_eq!(from_client.as_deref(), Some(CLIENT_MESSAGE));
    assert_eq!(from_host.as_deref(), Some(HOST_MESSAGE));
    let stats = received.expect("receive pipeline");
    assert_eq!(stats.frames_emitted, u64::from(FRAMES));
    assert_eq!(sink.frames.len(), usize::from(FRAMES));
}

#[tokio::test]
async fn data_channel_runs_both_ways_over_raw_quic() {
    data_channel_runs_both_ways_beside_the_video("quic").await;
}

#[tokio::test]
async fn data_channel_runs_both_ways_over_webtransport() {
    data_channel_runs_both_ways_beside_the_video("https").await;
}
