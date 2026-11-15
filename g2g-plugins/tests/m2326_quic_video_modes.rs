// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "webtransport")]

use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use g2g_core::caps::framerate_q16_period_ns;
use g2g_core::query::LatencyReport;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AsyncElement, G2gError, OutputSink, PipelinePacket, PropValue, PushOutcome, Reconfigure,
};
use g2g_plugins::remotewtio::generate_certificate;
use g2g_plugins::remotewtsink::RemoteWtSink;
use g2g_plugins::remotewtsrc::RemoteWtSrc;
use g2g_plugins::remotewtvideo::VideoMode;
use tokio::net::UdpSocket;

mod webtransport_common;

use webtransport_common::{test_caps, test_frame, NullOut, FRAME_LEN};

const SESSION_TIMEOUT: Duration = Duration::from_secs(30);
const RELAY_SEED: u64 = 0x2326;

// forwards QUIC packets between one client and the listening host, dropping host-to-client packets on request
struct UdpRelay {
    address: SocketAddr,
    blackhole: Arc<AtomicBool>,
    host_to_client_bytes: Arc<AtomicU64>,
}

impl UdpRelay {
    async fn start(host: SocketAddr, host_to_client_loss_percent: u64) -> Self {
        let socket = Arc::new(UdpSocket::bind("127.0.0.1:0").await.expect("bind relay"));
        let address = socket.local_addr().expect("relay address");
        let blackhole = Arc::new(AtomicBool::new(false));
        let dropping = blackhole.clone();
        let host_to_client_bytes = Arc::new(AtomicU64::new(0));
        let forwarded = host_to_client_bytes.clone();
        tokio::spawn(async move {
            let mut random = RELAY_SEED;
            let mut client = None;
            let mut buffer = vec![0u8; 65536];
            loop {
                let Ok((length, from)) = socket.recv_from(&mut buffer).await else {
                    return;
                };
                if dropping.load(Ordering::Relaxed) {
                    continue;
                }
                let to = if from == host {
                    // xorshift64, so a run drops the same packets each time
                    random ^= random << 13;
                    random ^= random >> 7;
                    random ^= random << 17;
                    if random % 100 < host_to_client_loss_percent {
                        continue;
                    }
                    let Some(client) = client else {
                        continue;
                    };
                    forwarded.fetch_add(length as u64, Ordering::Relaxed);
                    client
                } else {
                    client = Some(from);
                    host
                };
                let _ = socket.send_to(&buffer[..length], to).await;
            }
        });
        Self {
            address,
            blackhole,
            host_to_client_bytes,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Received {
    sequence: u64,
    keyframe: bool,
}

// records what the source emits and answers one data frame with a keyframe request when asked
#[derive(Default)]
struct RecordingOut {
    frames: Vec<Received>,
    controls: Vec<&'static str>,
    ask_for_keyframe_at: Option<u64>,
    latency: Option<LatencyReport>,
}

impl OutputSink for RecordingOut {
    fn latency_changed(&mut self, latency: LatencyReport) {
        self.latency = Some(latency);
    }

    fn poll_push(
        &mut self,
        _context: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let outcome = match packet_slot.take() {
            Some(PipelinePacket::DataFrame(frame)) => {
                self.frames.push(Received {
                    sequence: frame.sequence,
                    keyframe: frame.timing.keyframe,
                });
                if self.ask_for_keyframe_at == Some(frame.sequence) {
                    PushOutcome::Reconfigure(Reconfigure::ForceKeyframe)
                } else {
                    PushOutcome::Accepted
                }
            }
            Some(PipelinePacket::CapsChanged(_)) => {
                self.controls.push("caps");
                PushOutcome::Accepted
            }
            Some(PipelinePacket::Eos) => {
                self.controls.push("eos");
                PushOutcome::Accepted
            }
            _ => PushOutcome::Accepted,
        };
        core::task::Poll::Ready(Ok(outcome))
    }
}

struct Host {
    sink: RemoteWtSink,
    port: u16,
    hash: String,
}

fn host(mode: VideoMode) -> Host {
    let certificate = generate_certificate().expect("generate the host certificate");
    let hash = certificate.sha256_hex.clone();
    let mut sink = RemoteWtSink::new("quic://127.0.0.1:0")
        .with_self_signed_listener(certificate)
        .with_video_mode(mode);
    let port = sink.listen().expect("bind the listening sink").port();
    Host { sink, port, hash }
}

fn client(location: String, hash: String) -> RemoteWtSrc {
    RemoteWtSrc::new("127.0.0.1:0".parse().unwrap()).with_caller(location, hash)
}

async fn receive(source: &mut RemoteWtSrc, out: &mut RecordingOut) -> u64 {
    let caps = source
        .intercept_caps()
        .await
        .expect("connect and read caps");
    source
        .configure_pipeline(&caps)
        .expect("configure the source");
    source.run(out).await.expect("receive until eos")
}

// the encoder's side of a keyframe request: the next frame after one is a keyframe
struct FakeEncoder {
    next: u64,
    frame_len: usize,
    keyframe_pending: bool,
    keyframe_requests: u64,
}

impl FakeEncoder {
    fn new() -> Self {
        Self::with_frame_len(FRAME_LEN)
    }

    fn with_frame_len(frame_len: usize) -> Self {
        Self {
            next: 0,
            frame_len,
            keyframe_pending: true,
            keyframe_requests: 0,
        }
    }

    async fn send_next(&mut self, sink: &mut RemoteWtSink) {
        let mut frame = test_frame(self.next as u8, self.frame_len);
        frame.sequence = self.next;
        frame.timing.keyframe = core::mem::take(&mut self.keyframe_pending);
        sink.process(PipelinePacket::DataFrame(frame), &mut NullOut)
            .await
            .expect("send a frame");
        self.next += 1;
        if sink.take_reconfigure() == Some(Reconfigure::ForceKeyframe) {
            self.keyframe_requests += 1;
            self.keyframe_pending = true;
        }
    }
}

// every delivered delta frame needs the frame before it, or the decoder shows garbage
fn assert_decodable(frames: &[Received]) {
    assert!(frames.first().is_some_and(|first| first.keyframe));
    for pair in frames.windows(2) {
        assert!(
            pair[1].sequence > pair[0].sequence,
            "frames arrive in send order: {pair:?}"
        );
        assert!(
            pair[1].keyframe || pair[1].sequence == pair[0].sequence + 1,
            "a delta frame follows its reference: {pair:?}"
        );
    }
}

#[tokio::test]
async fn stream_per_frame_keeps_frames_and_caps_changes_in_send_order() {
    const FRAMES: u8 = 40;
    let Host {
        mut sink,
        port,
        hash,
    } = host(VideoMode::StreamPerFrame);
    let mut source = client(format!("quic://127.0.0.1:{port}"), hash);
    let mut out = RecordingOut::default();

    let send = async {
        sink.connect().await.expect("accept the client");
        sink.configure_pipeline(&test_caps()).expect("configure");
        let mut encoder = FakeEncoder::new();
        for index in 0..FRAMES {
            if index == FRAMES / 2 {
                sink.process(PipelinePacket::CapsChanged(test_caps()), &mut NullOut)
                    .await
                    .expect("caps change");
                let mut changed = test_caps();
                if let g2g_core::Caps::RawVideo { width, .. } = &mut changed {
                    *width = g2g_core::Dim::Fixed(4);
                }
                sink.process(PipelinePacket::CapsChanged(changed), &mut NullOut)
                    .await
                    .expect("caps change");
            }
            encoder.send_next(&mut sink).await;
        }
        sink.process(PipelinePacket::Eos, &mut NullOut)
            .await
            .expect("eos");
    };
    let (_, emitted) = tokio::time::timeout(SESSION_TIMEOUT, async {
        tokio::join!(send, receive(&mut source, &mut out))
    })
    .await
    .expect("the session finishes in time");

    assert_eq!(emitted, u64::from(FRAMES));
    let sequences: Vec<u64> = out.frames.iter().map(|frame| frame.sequence).collect();
    assert_eq!(sequences, (0..u64::from(FRAMES)).collect::<Vec<_>>());
    // the leading caps, the changed caps (the unchanged resend is deduplicated), then eos
    assert_eq!(out.controls, ["caps", "caps", "eos"]);
}

#[tokio::test]
async fn a_downstream_keyframe_request_reaches_the_sending_encoder() {
    const ASK_AT: u64 = 3;
    const FRAMES: u8 = 60;
    let Host {
        mut sink,
        port,
        hash,
    } = host(VideoMode::Stream);
    let mut source = client(format!("quic://127.0.0.1:{port}"), hash);
    let mut out = RecordingOut {
        ask_for_keyframe_at: Some(ASK_AT),
        ..RecordingOut::default()
    };

    let send = async {
        sink.connect().await.expect("accept the client");
        sink.configure_pipeline(&test_caps()).expect("configure");
        let mut encoder = FakeEncoder::new();
        for _ in 0..FRAMES {
            encoder.send_next(&mut sink).await;
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        sink.process(PipelinePacket::Eos, &mut NullOut)
            .await
            .expect("eos");
        encoder.keyframe_requests
    };
    let (keyframe_requests, _) = tokio::time::timeout(SESSION_TIMEOUT, async {
        tokio::join!(send, receive(&mut source, &mut out))
    })
    .await
    .expect("the session finishes in time");

    assert_eq!(keyframe_requests, 1);
    let keyframes: Vec<u64> = out
        .frames
        .iter()
        .filter(|frame| frame.keyframe)
        .map(|frame| frame.sequence)
        .collect();
    assert_eq!(keyframes.len(), 2, "the first frame and the requested one");
    assert!(keyframes[1] > ASK_AT);
}

#[tokio::test]
async fn lost_datagrams_skip_to_a_requested_keyframe() {
    const FRAMES: u16 = 400;
    const LOSS_PERCENT: u64 = 5;
    let Host {
        mut sink,
        port,
        hash,
    } = host(VideoMode::Datagrams);
    let relay = UdpRelay::start(([127, 0, 0, 1], port).into(), LOSS_PERCENT).await;
    let mut source = client(format!("quic://{}", relay.address), hash);
    let mut out = RecordingOut::default();

    let send = async {
        sink.connect().await.expect("accept the client");
        sink.configure_pipeline(&test_caps()).expect("configure");
        let mut encoder = FakeEncoder::new();
        for _ in 0..FRAMES {
            encoder.send_next(&mut sink).await;
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        sink.process(PipelinePacket::Eos, &mut NullOut)
            .await
            .expect("eos");
        encoder.keyframe_requests
    };
    let (keyframe_requests, _) = tokio::time::timeout(SESSION_TIMEOUT, async {
        tokio::join!(send, receive(&mut source, &mut out))
    })
    .await
    .expect("the session finishes in time");

    assert_decodable(&out.frames);
    assert!(
        out.frames.len() < usize::from(FRAMES),
        "the relay dropped some frames"
    );
    assert!(keyframe_requests > 0, "the client asked for a keyframe");
    assert!(
        out.frames.iter().skip(1).any(|frame| frame.keyframe),
        "a requested keyframe arrived"
    );
    assert_eq!(out.controls.last(), Some(&"eos"));
    let g2g_core::Rate::Fixed(framerate_q16) = test_caps().framerate() else {
        panic!("the test caps have a fixed framerate");
    };
    let frame_period_ns =
        framerate_q16_period_ns(framerate_q16).expect("the test caps have a framerate");
    assert_eq!(
        out.latency,
        Some(LatencyReport::live(frame_period_ns, None)),
        "datagram video reports the wait for a missing frame"
    );
}

#[tokio::test]
async fn repair_shards_rebuild_frames_the_path_loses_pieces_of() {
    const FRAMES: u16 = 300;
    const LOSS_PERCENT: u64 = 5;
    // without repair about a quarter of six-datagram frames lose a piece at 5% loss
    const MULTI_SHARD_FRAME_LEN: usize = 6_000;
    const DELIVERED_AT_LEAST_PERCENT: usize = 97;
    let Host {
        mut sink,
        port,
        hash,
    } = host(VideoMode::Datagrams);
    let relay = UdpRelay::start(([127, 0, 0, 1], port).into(), LOSS_PERCENT).await;
    let mut source = client(format!("quic://{}", relay.address), hash);
    let mut out = RecordingOut::default();

    let send = async {
        sink.connect().await.expect("accept the client");
        sink.configure_pipeline(&test_caps()).expect("configure");
        for sequence in 0..u64::from(FRAMES) {
            // keyframes depend on no earlier frame
            let mut frame = test_frame(sequence as u8, MULTI_SHARD_FRAME_LEN);
            frame.sequence = sequence;
            frame.timing.keyframe = true;
            sink.process(PipelinePacket::DataFrame(frame), &mut NullOut)
                .await
                .expect("send a frame");
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        sink.process(PipelinePacket::Eos, &mut NullOut)
            .await
            .expect("eos");
    };
    tokio::time::timeout(SESSION_TIMEOUT, async {
        tokio::join!(send, receive(&mut source, &mut out))
    })
    .await
    .expect("the session finishes in time");

    assert_decodable(&out.frames);
    assert!(
        out.frames.len() * 100 >= usize::from(FRAMES) * DELIVERED_AT_LEAST_PERCENT,
        "{} of {FRAMES} frames arrived",
        out.frames.len()
    );
    assert_eq!(out.controls.last(), Some(&"eos"));
}

#[tokio::test]
async fn blocked_udp_falls_back_to_websocket() {
    const FRAMES: u8 = 40;
    const ASK_AT: u64 = 10;
    const TO_HOST: &[u8] = b"input for the host";
    const TO_CLIENT: &[u8] = b"cursor for the client";
    let Host {
        mut sink,
        port,
        hash,
    } = host(VideoMode::Stream);
    let mut host_messages = sink.open_data_channel();
    let relay = UdpRelay::start(([127, 0, 0, 1], port).into(), 0).await;
    relay.blackhole.store(true, Ordering::Relaxed);
    let mut source = client(format!("quic://{}", relay.address), hash);
    SourceLoop::set_property(
        &mut source,
        "fallback-location",
        PropValue::Str(format!("wss://127.0.0.1:{port}")),
    )
    .expect("fallback-location");
    let mut client_messages = source.open_data_channel();
    client_messages
        .outgoing
        .send(TO_HOST.to_vec())
        .expect("queue a message for the host");
    let mut out = RecordingOut {
        ask_for_keyframe_at: Some(ASK_AT),
        ..RecordingOut::default()
    };

    let send = async {
        sink.connect().await.expect("accept the client");
        sink.configure_pipeline(&test_caps()).expect("configure");
        host_messages
            .outgoing
            .send(TO_CLIENT.to_vec())
            .expect("queue a message for the client");
        let mut encoder = FakeEncoder::new();
        for _ in 0..FRAMES {
            encoder.send_next(&mut sink).await;
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        sink.process(PipelinePacket::Eos, &mut NullOut)
            .await
            .expect("eos");
        let from_client = host_messages.incoming.recv().await;
        (encoder.keyframe_requests, from_client)
    };
    let ((keyframe_requests, from_client), _) = tokio::time::timeout(SESSION_TIMEOUT, async {
        tokio::join!(send, receive(&mut source, &mut out))
    })
    .await
    .expect("the session finishes in time");

    let sequences: Vec<u64> = out.frames.iter().map(|frame| frame.sequence).collect();
    assert_eq!(sequences, (0..u64::from(FRAMES)).collect::<Vec<_>>());
    assert_eq!(out.controls.last(), Some(&"eos"));
    assert_eq!(
        keyframe_requests, 1,
        "the keyframe request crossed the websocket"
    );
    assert_eq!(from_client.as_deref(), Some(TO_HOST));
    assert_eq!(
        client_messages.incoming.recv().await.as_deref(),
        Some(TO_CLIENT)
    );
    assert_eq!(
        relay.host_to_client_bytes.load(Ordering::Relaxed),
        0,
        "nothing crossed the blocked UDP path"
    );
}

#[tokio::test]
async fn a_slow_websocket_reader_drops_delta_frames_until_a_keyframe() {
    const FRAMES: u16 = 200;
    // big enough that a paused reader fills the loopback TCP buffers within the pause
    const LARGE_FRAME_LEN: usize = 64_000;
    const READER_PAUSE: Duration = Duration::from_secs(1);
    let Host {
        mut sink,
        port,
        hash,
    } = host(VideoMode::Stream);
    let mut source = client(format!("wss://127.0.0.1:{port}"), hash);
    let mut out = RecordingOut::default();

    let send = async {
        sink.connect().await.expect("accept the client");
        sink.configure_pipeline(&test_caps()).expect("configure");
        let mut encoder = FakeEncoder::with_frame_len(LARGE_FRAME_LEN);
        for _ in 0..FRAMES {
            encoder.send_next(&mut sink).await;
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        sink.process(PipelinePacket::Eos, &mut NullOut)
            .await
            .expect("eos");
        encoder.keyframe_requests
    };
    let receive_late = async {
        let caps = source
            .intercept_caps()
            .await
            .expect("connect and read caps");
        source
            .configure_pipeline(&caps)
            .expect("configure the source");
        tokio::time::sleep(READER_PAUSE).await;
        source.run(&mut out).await.expect("receive until eos")
    };
    let (keyframe_requests, _) =
        tokio::time::timeout(SESSION_TIMEOUT, async { tokio::join!(send, receive_late) })
            .await
            .expect("the session finishes in time");

    assert!(
        keyframe_requests > 0,
        "a dropped frame asked for a keyframe"
    );
    assert!(
        out.frames.len() < usize::from(FRAMES),
        "frames that found the queue full were dropped"
    );
    assert_decodable(&out.frames);
    assert_eq!(out.controls.last(), Some(&"eos"));
}

#[tokio::test]
async fn a_stalled_link_resets_old_frames_behind_a_new_keyframe() {
    const FRAMES: u8 = 120;
    const STALL_AT: u8 = 20;
    const STALL: Duration = Duration::from_millis(300);
    const FRAME_TIMEOUT_MS: u64 = 50;
    const STALL_FRAME_LEN: usize = 20_000;
    // the stall lasts about sixty frames, and only the ones after the last keyframe in it still need sending
    const FRAMES_RESET_AT_LEAST: usize = 20;
    let Host {
        mut sink,
        port,
        hash,
    } = host(VideoMode::StreamPerFrame);
    AsyncElement::set_property(
        &mut sink,
        "frame-timeout",
        PropValue::Uint(FRAME_TIMEOUT_MS),
    )
    .expect("frame-timeout");
    let relay = UdpRelay::start(([127, 0, 0, 1], port).into(), 0).await;
    let mut source = client(format!("quic://{}", relay.address), hash);
    let mut out = RecordingOut::default();
    let stall_ended = Arc::new(AtomicU64::new(0));

    let send = async {
        sink.connect().await.expect("accept the client");
        sink.configure_pipeline(&test_caps()).expect("configure");
        let mut encoder = FakeEncoder::with_frame_len(STALL_FRAME_LEN);
        let mut stall_started = None;
        for index in 0..FRAMES {
            if index == STALL_AT {
                relay.blackhole.store(true, Ordering::Relaxed);
                stall_started = Some(tokio::time::Instant::now());
            }
            if stall_started.is_some_and(|started| started.elapsed() >= STALL) {
                relay.blackhole.store(false, Ordering::Relaxed);
                stall_started = None;
                stall_ended.store(u64::from(index), Ordering::Relaxed);
            }
            encoder.send_next(&mut sink).await;
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        sink.process(PipelinePacket::Eos, &mut NullOut)
            .await
            .expect("eos");
        encoder.keyframe_requests
    };
    let (keyframe_requests, _) = tokio::time::timeout(SESSION_TIMEOUT, async {
        tokio::join!(send, receive(&mut source, &mut out))
    })
    .await
    .expect("the session finishes in time");

    assert_decodable(&out.frames);
    assert!(
        keyframe_requests > 0,
        "the stall asked the encoder for a keyframe"
    );
    let stall_ended = stall_ended.load(Ordering::Relaxed);
    assert!(stall_ended > 0, "the stall ended before the last frame");
    assert!(
        out.frames.len() < usize::from(FRAMES),
        "frames reset during the stall never showed"
    );
    assert!(out
        .frames
        .iter()
        .any(|frame| frame.keyframe && frame.sequence > u64::from(STALL_AT)));
    assert_eq!(
        out.frames.last().map(|frame| frame.sequence),
        Some(u64::from(FRAMES) - 1)
    );
    let sent_bytes = usize::from(FRAMES) * STALL_FRAME_LEN;
    let forwarded = relay.host_to_client_bytes.load(Ordering::Relaxed);
    assert!(
        forwarded < (sent_bytes - FRAMES_RESET_AT_LEAST * STALL_FRAME_LEN) as u64,
        "reset frames are not retransmitted: {forwarded} of {sent_bytes} bytes crossed the link"
    );
}
