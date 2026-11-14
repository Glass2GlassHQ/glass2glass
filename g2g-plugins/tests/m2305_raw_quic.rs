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

const FRAMES: u8 = 8;
const RECEIVE_TIMEOUT: Duration = Duration::from_secs(20);

async fn send_frames(location: String, hashes: String) -> Result<u64, g2g_core::G2gError> {
    let mut remote = RemoteWtSink::new(location).with_server_certificate_hashes(hashes);
    remote.configure_pipeline(&test_caps()).expect("configure");
    let mut null = NullOut;
    for i in 0..FRAMES {
        remote
            .process(
                PipelinePacket::DataFrame(test_frame(i, FRAME_LEN)),
                &mut null,
            )
            .await?;
    }
    remote.process(PipelinePacket::Eos, &mut null).await?;
    Ok(remote.sent())
}

#[tokio::test]
async fn raw_quic_carries_a_split_graph_edge() {
    let tls = TestCert::generate("m2305-raw");
    let mut source = RemoteWtSrc::new("127.0.0.1:0".parse().unwrap())
        .with_certificate(tls.cert(), tls.key())
        .with_frame_limit(u64::from(FRAMES));
    let port = source.listen().await.expect("bind quic endpoint").port();
    let mut sink = CollectSink::default();
    let receive = tokio::time::timeout(
        RECEIVE_TIMEOUT,
        run_simple_pipeline(
            &mut source,
            &mut sink,
            &ZeroClock,
            LatencyProfile::Live.link_capacity(),
        ),
    );
    let send = send_frames(format!("quic://127.0.0.1:{port}"), tls.hash_hex.clone());
    let (received, sent) = tokio::join!(receive, send);
    let stats = received
        .expect("receiver finishes in time")
        .expect("receive pipeline");

    assert_eq!(stats.frames_emitted, u64::from(FRAMES));
    assert!(sent.expect("sender") > u64::from(FRAMES));
    assert_eq!(sink.caps.first(), Some(&test_caps()));
    assert_eq!(sink.frames.len(), usize::from(FRAMES));
    for (i, (sequence, _, bytes)) in sink.frames.iter().enumerate() {
        assert_eq!(*sequence, i as u64);
        assert_eq!(
            bytes,
            test_frame(i as u8, FRAME_LEN)
                .domain
                .as_system_slice()
                .expect("system frame")
        );
    }
}

#[tokio::test]
async fn raw_quic_refuses_a_certificate_that_is_not_pinned() {
    let tls = TestCert::generate("m2305-pinned");
    let other = TestCert::generate("m2305-other");
    let mut source = RemoteWtSrc::new("127.0.0.1:0".parse().unwrap())
        .with_certificate(tls.cert(), tls.key())
        .with_frame_limit(u64::from(FRAMES));
    let port = source.listen().await.expect("bind quic endpoint").port();
    let mut sink = CollectSink::default();
    // quinn only completes a handshake the server side has accepted
    let receive = tokio::time::timeout(
        RECEIVE_TIMEOUT,
        run_simple_pipeline(
            &mut source,
            &mut sink,
            &ZeroClock,
            LatencyProfile::Live.link_capacity(),
        ),
    );
    let send = send_frames(format!("quic://127.0.0.1:{port}"), other.hash_hex.clone());
    let (received, sent) = tokio::join!(receive, send);
    assert!(
        sent.is_err(),
        "a client pinned to another certificate must not connect"
    );
    assert!(
        received.expect("receiver ends in time").is_err(),
        "the server sees the refused handshake fail"
    );
    assert!(sink.frames.is_empty());
}

async fn listener_sink_sends_to_a_caller_source(scheme: &str) {
    let tls = TestCert::generate(&format!("m2305-roles-{scheme}"));
    let mut remote =
        RemoteWtSink::new(format!("{scheme}://127.0.0.1:0")).with_listener(tls.cert(), tls.key());
    let port = remote.listen().expect("bind the listening sink").port();
    let send = async {
        remote.configure_pipeline(&test_caps()).expect("configure");
        let mut null = NullOut;
        for i in 0..FRAMES {
            remote
                .process(
                    PipelinePacket::DataFrame(test_frame(i, FRAME_LEN)),
                    &mut null,
                )
                .await?;
        }
        remote.process(PipelinePacket::Eos, &mut null).await?;
        Ok::<u64, g2g_core::G2gError>(remote.sent())
    };
    let mut source = RemoteWtSrc::new("127.0.0.1:0".parse().unwrap())
        .with_caller(format!("{scheme}://127.0.0.1:{port}"), tls.hash_hex.clone())
        .with_frame_limit(u64::from(FRAMES));
    let mut sink = CollectSink::default();
    let receive = tokio::time::timeout(
        RECEIVE_TIMEOUT,
        run_simple_pipeline(
            &mut source,
            &mut sink,
            &ZeroClock,
            LatencyProfile::Live.link_capacity(),
        ),
    );
    let (received, sent) = tokio::join!(receive, send);
    let stats = received
        .expect("receiver finishes in time")
        .expect("receive pipeline");
    assert!(sent.expect("sender") > u64::from(FRAMES));
    assert_eq!(stats.frames_emitted, u64::from(FRAMES));
    assert_eq!(sink.caps.first(), Some(&test_caps()));
    assert_eq!(sink.frames.len(), usize::from(FRAMES));
}

#[tokio::test]
async fn a_listening_sink_sends_to_a_calling_source_over_raw_quic() {
    listener_sink_sends_to_a_caller_source("quic").await;
}

#[tokio::test]
async fn a_listening_sink_sends_to_a_calling_source_over_webtransport() {
    listener_sink_sends_to_a_caller_source("https").await;
}
