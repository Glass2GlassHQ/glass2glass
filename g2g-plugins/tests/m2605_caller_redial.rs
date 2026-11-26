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

const FRAMES_PER_HOST: u8 = 3;
const SESSION_TIMEOUT: Duration = Duration::from_secs(30);
const PORT_FREE_TIMEOUT: Duration = Duration::from_secs(5);
const PORT_POLL_INTERVAL: Duration = Duration::from_millis(50);
const GREETING: &[u8] = b"token";
const QUEUED_WHILE_DISCONNECTED: &[u8] = b"click while the host was gone";
const AFTER_RECONNECT: &[u8] = b"click on the new host";
// long enough for the client to see the first host close and stop writing to it
const CLOSE_NOTICED: Duration = Duration::from_millis(300);
// the second host binds later, so the client queues while it redials
const SECOND_HOST_DELAY: Duration = Duration::from_millis(1500);

async fn bind_when_free(location: &str, tls: &TestCert) -> RemoteWtSink {
    let deadline = tokio::time::Instant::now() + PORT_FREE_TIMEOUT;
    loop {
        let mut sink = RemoteWtSink::new(location).with_listener(tls.cert(), tls.key());
        match sink.listen() {
            Ok(_) => return sink,
            Err(e) if tokio::time::Instant::now() >= deadline => {
                panic!("{location} stays taken: {e:?}")
            }
            Err(_) => tokio::time::sleep(PORT_POLL_INTERVAL).await,
        }
    }
}

async fn send_frames(host: &mut RemoteWtSink, first: u8) {
    let mut null = NullOut;
    for i in first..first + FRAMES_PER_HOST {
        host.process(
            PipelinePacket::DataFrame(test_frame(i, FRAME_LEN)),
            &mut null,
        )
        .await
        .expect("frame");
    }
}

async fn caller_redials_a_restarted_host_and_greets_it_first(scheme: &str) {
    let tls = TestCert::generate(&format!("m2605-{scheme}"));
    let mut first_host =
        RemoteWtSink::new(format!("{scheme}://127.0.0.1:0")).with_listener(tls.cert(), tls.key());
    let mut first_channel = first_host.open_data_channel();
    let port = first_host.listen().expect("bind the first host").port();
    let location = format!("{scheme}://127.0.0.1:{port}");

    let mut source = RemoteWtSrc::new("127.0.0.1:0".parse().unwrap())
        .with_caller(location.clone(), tls.hash_hex.clone())
        .with_reconnect();
    let DataChannelEnds {
        outgoing: client_outgoing,
        incoming: _client_incoming,
    } = source.open_data_channel_with_greeting(GREETING.to_vec());
    let (first_host_gone, host_gone) = tokio::sync::oneshot::channel();
    let (second_host_greeted, greeted) = tokio::sync::oneshot::channel();

    let client = async move {
        host_gone.await.expect("the first host goes away");
        tokio::time::sleep(CLOSE_NOTICED).await;
        client_outgoing
            .send(QUEUED_WHILE_DISCONNECTED.to_vec())
            .expect("queue while disconnected");
        greeted.await.expect("the second host reads the greeting");
        client_outgoing
            .send(AFTER_RECONNECT.to_vec())
            .expect("queue after reconnecting");
    };
    let hosts = async {
        first_host
            .configure_pipeline(&test_caps())
            .expect("configure");
        send_frames(&mut first_host, 0).await;
        let first_greeting = first_channel.incoming.recv().await;
        // dropping the sink without Eos stands in for a killed host
        drop(first_host);
        first_host_gone.send(()).expect("the client waits");
        tokio::time::sleep(SECOND_HOST_DELAY).await;

        let mut second_host = bind_when_free(&location, &tls).await;
        let mut second_channel = second_host.open_data_channel();
        second_host
            .configure_pipeline(&test_caps())
            .expect("configure");
        send_frames(&mut second_host, FRAMES_PER_HOST).await;
        let second_greeting = second_channel.incoming.recv().await;
        second_host_greeted.send(()).expect("the client waits");
        let after = second_channel.incoming.recv().await;
        second_host
            .process(PipelinePacket::Eos, &mut NullOut)
            .await
            .expect("eos");
        (first_greeting, second_greeting, after)
    };
    let mut sink = CollectSink::default();
    let receive = run_simple_pipeline(
        &mut source,
        &mut sink,
        &ZeroClock,
        LatencyProfile::Live.link_capacity(),
    );
    let ((first_greeting, second_greeting, after), (), received) =
        tokio::time::timeout(SESSION_TIMEOUT, async {
            tokio::join!(hosts, client, receive)
        })
        .await
        .expect("the session finishes in time");

    assert_eq!(first_greeting.as_deref(), Some(GREETING));
    assert_eq!(second_greeting.as_deref(), Some(GREETING));
    assert_eq!(after.as_deref(), Some(AFTER_RECONNECT));
    received.expect("receive pipeline");
    // test_frame fills a frame with its index
    let markers: Vec<u8> = sink.frames.iter().map(|(_, _, bytes)| bytes[0]).collect();
    let second_host_markers: Vec<u8> = (FRAMES_PER_HOST..2 * FRAMES_PER_HOST).collect();
    assert!(
        markers.ends_with(&second_host_markers),
        "every frame of the second host arrives after the redial: {markers:?}"
    );
    assert!(sink.eos, "the second host's Eos ends the stream");
}

#[tokio::test]
async fn caller_redials_a_restarted_host_over_raw_quic() {
    caller_redials_a_restarted_host_and_greets_it_first("quic").await;
}

#[tokio::test]
async fn caller_redials_a_restarted_host_over_webtransport() {
    caller_redials_a_restarted_host_and_greets_it_first("https").await;
}
