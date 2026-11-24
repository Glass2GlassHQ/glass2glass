// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "webtransport")]

use std::time::Duration;

use g2g_core::runtime::{run_simple_pipeline, LatencyProfile};
use g2g_core::{AsyncElement, AudioFormat, Caps, ChannelLayout, PipelinePacket};
use g2g_plugins::remotewtsink::RemoteWtSink;
use g2g_plugins::remotewtsrc::RemoteWtSrc;

mod webtransport_common;

use webtransport_common::{test_caps, test_frame, CollectSink, NullOut, TestCert, ZeroClock};

const FRAMES: u8 = 6;
const TRACK_FRAMES: u8 = 5;
const TRACK_FRAME_LEN: usize = 40;
// the track drops a frame that finds its queue full, so the test paces them
const TRACK_FRAME_GAP: Duration = Duration::from_millis(10);
const RECEIVE_TIMEOUT: Duration = Duration::from_secs(20);

fn track_caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::Opus,
        channels: 2,
        sample_rate: 48_000,
        channel_layout: ChannelLayout::UNSPECIFIED,
    }
}

async fn track_rides_beside_the_main_stream(listen_scheme: &str, dial_scheme: &str) {
    let tls = TestCert::generate(&format!("m2575-{dial_scheme}"));
    let mut remote = RemoteWtSink::new(format!("{listen_scheme}://127.0.0.1:0"))
        .with_listener(tls.cert(), tls.key());
    let mut track = remote.open_track();
    let port = remote.listen().expect("bind the listening sink").port();
    let mut source = RemoteWtSrc::new("127.0.0.1:0".parse().unwrap())
        .with_caller(
            format!("{dial_scheme}://127.0.0.1:{port}"),
            tls.hash_hex.clone(),
        )
        .with_frame_limit(u64::from(FRAMES));
    let mut track_source = source.open_track();

    let send_main = async {
        remote.configure_pipeline(&test_caps()).expect("configure");
        let mut null = NullOut;
        for i in 0..FRAMES {
            // the websocket link drops delta frames behind a busy socket, a keyframe waits for room
            let mut frame = test_frame(i, 16);
            frame.timing.keyframe = true;
            remote
                .process(PipelinePacket::DataFrame(frame), &mut null)
                .await?;
        }
        Ok::<(), g2g_core::G2gError>(())
    };
    let send_track = async {
        track.configure_pipeline(&track_caps()).expect("configure");
        let mut null = NullOut;
        for i in 0..TRACK_FRAMES {
            track
                .process(
                    PipelinePacket::DataFrame(test_frame(i, TRACK_FRAME_LEN)),
                    &mut null,
                )
                .await?;
            tokio::time::sleep(TRACK_FRAME_GAP).await;
        }
        track.process(PipelinePacket::Eos, &mut null).await
    };
    let mut main_sink = CollectSink::default();
    let mut track_sink = CollectSink::default();
    let link_capacity = LatencyProfile::Live.link_capacity();
    let receive_main = tokio::time::timeout(
        RECEIVE_TIMEOUT,
        run_simple_pipeline(&mut source, &mut main_sink, &ZeroClock, link_capacity),
    );
    let receive_track = tokio::time::timeout(
        RECEIVE_TIMEOUT,
        run_simple_pipeline(
            &mut track_source,
            &mut track_sink,
            &ZeroClock,
            link_capacity,
        ),
    );
    let (main_received, track_received, main_sent, track_sent) =
        tokio::join!(receive_main, receive_track, send_main, send_track);
    main_sent.expect("main stream sent");
    track_sent.expect("track sent");
    main_received
        .expect("main stream received in time")
        .expect("main receive pipeline");
    track_received
        .expect("track received in time")
        .expect("track receive pipeline");

    assert_eq!(main_sink.caps.first(), Some(&test_caps()));
    assert_eq!(main_sink.frames.len(), usize::from(FRAMES));
    assert_eq!(track_sink.caps.first(), Some(&track_caps()));
    assert!(track_sink.eos, "the track's Eos ends the track source");
    let track_frames: Vec<u64> = track_sink
        .frames
        .iter()
        .map(|(sequence, ..)| *sequence)
        .collect();
    assert_eq!(
        track_frames,
        (0..u64::from(TRACK_FRAMES)).collect::<Vec<_>>()
    );
    for (_, _, bytes) in &track_sink.frames {
        assert_eq!(
            bytes.len(),
            TRACK_FRAME_LEN,
            "track frames keep their length"
        );
    }
}

#[tokio::test]
async fn a_track_rides_beside_the_main_stream_over_raw_quic() {
    track_rides_beside_the_main_stream("quic", "quic").await;
}

#[tokio::test]
async fn a_track_rides_beside_the_main_stream_over_webtransport() {
    track_rides_beside_the_main_stream("https", "https").await;
}

#[tokio::test]
async fn a_track_rides_beside_the_main_stream_over_the_websocket_fallback() {
    track_rides_beside_the_main_stream("quic", "wss").await;
}
