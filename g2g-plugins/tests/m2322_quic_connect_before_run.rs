// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "webtransport")]

use std::time::Duration;

use g2g_core::runtime::{run_simple_pipeline, LatencyProfile};
use g2g_core::{AsyncElement, PipelinePacket};
use g2g_plugins::remotewtsink::RemoteWtSink;
use g2g_plugins::remotewtsrc::RemoteWtSrc;

mod webtransport_common;

use webtransport_common::{
    test_caps, test_frame, CollectSink, NullOut, TestCert, ZeroClock, FRAME_LEN,
};

const FRAMES: u8 = 3;
const SESSION_TIMEOUT: Duration = Duration::from_secs(20);
const TOKEN: &[u8] = b"bearer token";

async fn listener_reads_the_token_before_any_video(scheme: &str) {
    let tls = TestCert::generate(&format!("m2322-{scheme}"));
    let mut remote =
        RemoteWtSink::new(format!("{scheme}://127.0.0.1:0")).with_listener(tls.cert(), tls.key());
    let mut host_channel = remote.open_data_channel();
    let port = remote.listen().expect("bind the listening sink").port();

    let mut source = RemoteWtSrc::new("127.0.0.1:0".parse().unwrap())
        .with_caller(format!("{scheme}://127.0.0.1:{port}"), tls.hash_hex.clone())
        .with_frame_limit(u64::from(FRAMES));
    let client_channel = source.open_data_channel();
    client_channel
        .outgoing
        .send(TOKEN.to_vec())
        .expect("queue the token");

    let host = async {
        remote.connect().await.expect("accept the client");
        let token = host_channel.incoming.recv().await;
        let sent_before_token = remote.sent();
        remote.configure_pipeline(&test_caps()).expect("configure");
        let mut null = NullOut;
        for i in 0..FRAMES {
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
        (token, sent_before_token)
    };
    let mut sink = CollectSink::default();
    let receive = run_simple_pipeline(
        &mut source,
        &mut sink,
        &ZeroClock,
        LatencyProfile::Live.link_capacity(),
    );
    let ((token, sent_before_token), received) =
        tokio::time::timeout(SESSION_TIMEOUT, async { tokio::join!(host, receive) })
            .await
            .expect("the session finishes in time");

    assert_eq!(token.as_deref(), Some(TOKEN));
    assert_eq!(sent_before_token, 0);
    let stats = received.expect("receive pipeline");
    assert_eq!(stats.frames_emitted, u64::from(FRAMES));
    assert_eq!(sink.caps.first(), Some(&test_caps()));
    assert_eq!(sink.frames.len(), usize::from(FRAMES));
    drop(client_channel);
}

#[tokio::test]
async fn listener_reads_the_token_before_any_video_over_raw_quic() {
    listener_reads_the_token_before_any_video("quic").await;
}

#[tokio::test]
async fn listener_reads_the_token_before_any_video_over_webtransport() {
    listener_reads_the_token_before_any_video("https").await;
}
