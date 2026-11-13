// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(target_os = "linux", feature = "webrtc"))]

mod webrtc_duplex_common;
mod webrtc_streams_common;

use core::future::{ready, Future, Ready};
use core::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use g2g_core::element::DynAsyncElement;
use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_duplex_session, DynSourceLoop, SourceLoop};
use g2g_core::{
    Caps, ConfigureOutcome, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket,
    PushOutcome, Reconfigure,
};
use g2g_plugins::webrtc_codec::WebRtcVideoCodec;
use g2g_plugins::webrtcduplex::{SdpChannel, SignalRole, WebRtcDuplexSession};
use tokio::sync::mpsc;

use webrtc_duplex_common::{run_peer, Received, RecordingSink, ZeroClock, FRAME_INTERVAL};
use webrtc_streams_common::{fixed_caps, synthetic_stream, GOP};

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

// sends one keyframe at the start, then a keyframe only when one is requested
struct KeyframeOnRequestSrc {
    codec: WebRtcVideoCodec,
    keyframe: Vec<u8>,
    delta: Vec<u8>,
    requests: Arc<AtomicU64>,
}

impl SourceLoop for KeyframeOnRequestSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = Ready<Result<Caps, G2gError>>;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        ready(Ok(fixed_caps(self.codec)))
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(PipelinePacket::CapsChanged(fixed_caps(self.codec)))
                .await?;
            let mut send_keyframe = true;
            for seq in 0..FRAMES as u64 {
                let unit = if send_keyframe {
                    &self.keyframe
                } else {
                    &self.delta
                };
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(unit.clone().into_boxed_slice())),
                    FrameTiming {
                        pts_ns: seq * FRAME_INTERVAL.as_nanos() as u64,
                        ..FrameTiming::default()
                    },
                    seq,
                );
                let outcome = out.push(PipelinePacket::DataFrame(frame)).await?;
                send_keyframe = matches!(
                    outcome,
                    PushOutcome::Reconfigure(Reconfigure::ForceKeyframe)
                );
                if send_keyframe {
                    self.requests.fetch_add(1, Ordering::Relaxed);
                }
                tokio::time::sleep(FRAME_INTERVAL).await;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(FRAMES as u64)
        })
    }
}

#[tokio::test]
async fn receive_only_client_asks_for_the_keyframe_the_handshake_dropped() {
    let codec = WebRtcVideoCodec::default();
    let units = synthetic_stream(codec, 2);
    const { assert!(GOP > 1, "the second synthetic unit is a delta unit") };
    let requests = Arc::new(AtomicU64::new(0));
    let mut source = KeyframeOnRequestSrc {
        codec,
        keyframe: units[0].clone(),
        delta: units[1].clone(),
        requests: requests.clone(),
    };
    let (client_signal, host_signal) = SdpChannel::pair();
    let mut client =
        WebRtcDuplexSession::new(SignalRole::Offerer, client_signal, 1).with_receive_only();
    let mut host = WebRtcDuplexSession::new(SignalRole::Answerer, host_signal, 1);
    let client_video = Arc::new(Mutex::new(Received::default()));
    let mut client_sink = RecordingSink {
        received: client_video.clone(),
    };
    let mut host_sink = RecordingSink {
        received: Arc::new(Mutex::new(Received::default())),
    };
    let clock = ZeroClock;
    let run_client = async {
        let sources: Vec<&mut dyn DynSourceLoop> = Vec::new();
        let sinks: Vec<&mut dyn DynAsyncElement> = vec![&mut client_sink];
        run_duplex_session(sources, &mut client, sinks, &clock, LINK_CAPACITY).await
    };
    let run_host = async {
        let sources: Vec<&mut dyn DynSourceLoop> = vec![&mut source];
        let sinks: Vec<&mut dyn DynAsyncElement> = vec![&mut host_sink];
        run_duplex_session(sources, &mut host, sinks, &clock, LINK_CAPACITY).await
    };
    let (client_result, host_result) =
        tokio::time::timeout(RUN_TIMEOUT, async { tokio::join!(run_client, run_host) })
            .await
            .expect("client ends on the host's close");
    client_result.expect("client session");
    host_result.expect("host session");

    assert!(
        requests.load(Ordering::Relaxed) > 0,
        "the client asked for a keyframe"
    );
    let received = client_video.lock().unwrap();
    let (first, keyframe) = received.units.first().expect("the client got video");
    assert!(
        *keyframe && *first == source.keyframe,
        "video starts on a keyframe"
    );
}
