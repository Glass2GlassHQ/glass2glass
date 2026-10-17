//! `WhipServerSrc` against the g2g WHIP client (`WebRtcSink`) on localhost, once
//! per carried video codec and once for Opus audio. Synthetic access units, no
//! media server and no fixture files, so it runs by default.

#![cfg(all(target_os = "linux", feature = "webrtc"))]

mod webrtc_streams_common;

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::mpsc;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, FrameTiming, G2gError, MemoryDomain, MultiOutputSink,
    MultiOutputSource, OutputSink, PipelinePacket, PropValue, PushOutcome,
};
use g2g_plugins::webrtc_codec::WebRtcVideoCodec;
use g2g_plugins::webrtc_stats::WebRtcStats;
use g2g_plugins::webrtcsink::WebRtcSink;
use g2g_plugins::whipserversrc::WhipServerSrc;
use webrtc_streams_common::{fixed_caps, payload, synthetic_stream, FPS};

const VIDEO_PORT: usize = 0;
const AUDIO_PORT: usize = 1;
const VIDEO_UNITS: usize = 60;
const AUDIO_PACKETS: usize = 50;
const AUDIO_PACKET_BYTES: usize = 20;
const AUDIO_PACKET_INTERVAL: Duration = Duration::from_millis(20);
const OPUS_CHANNELS: u8 = 2;
const OPUS_SAMPLE_RATE: u32 = 48_000;
// let the sender's pacer flush before EOS deletes the resource
const DRAIN_BEFORE_EOS: Duration = Duration::from_millis(500);
const RUN_TIMEOUT: Duration = Duration::from_secs(20);
const TLS_CERT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/whip_tls_test_cert.pem"
);
const TLS_KEY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/whip_tls_test_key.pem"
);
const MIN_DELIVERED_PERCENT: usize = 90;
/// Every packet the server pushed, per output port.
struct Collector {
    packets: mpsc::UnboundedSender<(usize, PipelinePacket)>,
    ports: usize,
}

impl MultiOutputSink for Collector {
    fn poll_push_to(
        &mut self,
        _cx: &mut Context<'_>,
        port: usize,
        packet: &mut Option<PipelinePacket>,
    ) -> Poll<Result<PushOutcome, G2gError>> {
        let packet = packet.take().expect("push without a packet");
        let _ = self.packets.send((port, packet));
        Poll::Ready(Ok(PushOutcome::Accepted))
    }

    fn port_count(&self) -> usize {
        self.ports
    }
}

struct Discard;

impl OutputSink for Discard {
    fn poll_push(
        &mut self,
        _cx: &mut Context<'_>,
        packet: &mut Option<PipelinePacket>,
    ) -> Poll<Result<PushOutcome, G2gError>> {
        packet.take();
        Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn opus_caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::Opus,
        channels: OPUS_CHANNELS,
        sample_rate: OPUS_SAMPLE_RATE,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    }
}

fn free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

/// Publish `units` through a `WebRtcSink` at `interval`, then EOS. Returns the
/// sink's final stats.
async fn publish(
    whip_url: String,
    caps: Caps,
    units: &[Vec<u8>],
    interval: Duration,
) -> WebRtcStats {
    let mut sink = WebRtcSink::new(whip_url);
    sink.configure_pipeline(&caps).unwrap();
    for (sequence, unit) in units.iter().enumerate() {
        let timing = FrameTiming {
            pts_ns: sequence as u64 * interval.as_nanos() as u64,
            ..FrameTiming::default()
        };
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(unit.clone().into_boxed_slice())),
            timing,
            sequence as u64,
        );
        sink.process(PipelinePacket::DataFrame(frame), &mut Discard)
            .await
            .unwrap();
        tokio::time::sleep(interval).await;
    }
    tokio::time::sleep(DRAIN_BEFORE_EOS).await;
    sink.process(PipelinePacket::Eos, &mut Discard)
        .await
        .unwrap();
    sink.stats_handle().snapshot()
}

/// What one loopback run produced: the packets per output port and both ends'
/// final stats.
struct Loopback {
    ports: [Vec<PipelinePacket>; 2],
    server_stats: WebRtcStats,
    sink_stats: WebRtcStats,
}

/// Run a server for `video_codec` and publish `units` with `caps` into it.
async fn loopback(
    video_codec: WebRtcVideoCodec,
    caps: Caps,
    units: Vec<Vec<u8>>,
    interval: Duration,
) -> Loopback {
    let host_addr = format!("http://127.0.0.1:{}", free_port());
    let mut server = WhipServerSrc::new(host_addr.clone());
    server
        .set_property("video-codec", PropValue::Str(video_codec.name().into()))
        .unwrap();
    let server_stats = server.stats_handle();
    let (packets, mut received) = mpsc::unbounded_channel();
    let mut collector = Collector { packets, ports: 2 };

    let mut ports: [Vec<PipelinePacket>; 2] = [Vec::new(), Vec::new()];
    let run: Pin<Box<dyn Future<Output = Result<u64, G2gError>> + '_>> = server.run(&mut collector);
    let client = async {
        // both caps announcements follow the bind, so the server is listening
        let mut announced = Vec::new();
        for _ in 0..2 {
            announced.push(received.recv().await.unwrap());
        }
        let sink_stats = publish(format!("{host_addr}/whip"), caps, &units, interval).await;
        (announced, sink_stats)
    };
    let (result, (announced, sink_stats)) =
        tokio::time::timeout(RUN_TIMEOUT, async { tokio::join!(run, client) })
            .await
            .expect("loopback run timed out");
    result.expect("server run failed");
    for (port, packet) in announced {
        ports[port].push(packet);
    }
    while let Ok((port, packet)) = received.try_recv() {
        ports[port].push(packet);
    }
    Loopback {
        ports,
        server_stats: server_stats.snapshot(),
        sink_stats,
    }
}

fn data_frames(packets: &[PipelinePacket]) -> Vec<&Frame> {
    packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(frame) => Some(frame),
            _ => None,
        })
        .collect()
}

fn bytes(frame: &Frame) -> &[u8] {
    match &frame.domain {
        MemoryDomain::System(slice) => slice.as_slice(),
        _ => panic!("expected system memory"),
    }
}

fn assert_ends_with_eos(ports: &[Vec<PipelinePacket>; 2]) {
    for (port, packets) in ports.iter().enumerate() {
        assert!(
            matches!(packets.last(), Some(PipelinePacket::Eos)),
            "port {port} did not end with EOS"
        );
    }
}

fn min_delivered(sent: usize) -> usize {
    sent * MIN_DELIVERED_PERCENT / 100
}

#[tokio::test]
async fn every_video_codec_publishes_into_the_server() {
    let interval = Duration::from_secs(1) / FPS;
    for codec in WebRtcVideoCodec::ALL {
        let units = synthetic_stream(codec, VIDEO_UNITS);
        let run = loopback(codec, fixed_caps(codec), units.clone(), interval).await;
        let ports = &run.ports;

        let Some(PipelinePacket::CapsChanged(Caps::CompressedVideo {
            codec: announced, ..
        })) = ports[VIDEO_PORT].first()
        else {
            panic!("{codec:?}: video port did not open with video caps");
        };
        assert_eq!(*announced, codec.video_codec());

        let frames = data_frames(&ports[VIDEO_PORT]);
        assert!(
            frames.len() >= min_delivered(units.len()),
            "{codec:?}: {} of {} units arrived",
            frames.len(),
            units.len()
        );
        assert_eq!(bytes(frames[0]), &units[0][..], "{codec:?}");
        assert!(frames[0].timing.keyframe, "{codec:?}");
        assert!(data_frames(&ports[AUDIO_PORT]).is_empty(), "{codec:?}");
        assert_ends_with_eos(ports);
        assert_session_counted(&run, frames.len() as u64);
    }
}

/// Both ends' stats saw the session: the server counted the packets it received
/// and the peer bytes, the sink counted what it sent.
fn assert_session_counted(run: &Loopback, frames_received: u64) {
    let server = &run.server_stats;
    let peer = server.peer.as_ref().expect("server peer stats");
    assert!(peer.bytes_rx > 0, "{server:?}");
    assert!(
        server.packets_received() >= frames_received,
        "server counted {} packets for {frames_received} frames",
        server.packets_received()
    );
    assert!(server.egress.is_empty(), "{server:?}");
    let sink = &run.sink_stats;
    assert!(sink.packets_sent() >= frames_received, "{sink:?}");
    assert!(sink.ingress.is_empty(), "{sink:?}");
}

#[tokio::test]
async fn opus_publishes_into_the_audio_output() {
    let packets: Vec<Vec<u8>> = (0..AUDIO_PACKETS)
        .map(|index| payload(AUDIO_PACKET_BYTES, index as u8))
        .collect();
    let run = loopback(
        WebRtcVideoCodec::default(),
        opus_caps(),
        packets.clone(),
        AUDIO_PACKET_INTERVAL,
    )
    .await;
    let ports = &run.ports;

    assert!(matches!(
        ports[AUDIO_PORT].first(),
        Some(PipelinePacket::CapsChanged(Caps::Audio {
            format: AudioFormat::Opus,
            ..
        }))
    ));
    let frames = data_frames(&ports[AUDIO_PORT]);
    assert!(
        frames.len() >= min_delivered(packets.len()),
        "{} of {} packets arrived",
        frames.len(),
        packets.len()
    );
    assert_eq!(bytes(frames[0]), &packets[0][..]);
    assert!(data_frames(&ports[VIDEO_PORT]).is_empty());
    assert_ends_with_eos(ports);
    assert_session_counted(&run, frames.len() as u64);
}

#[tokio::test]
async fn https_host_addr_serves_whip_over_tls() {
    let port = free_port();
    let mut server =
        WhipServerSrc::new(format!("https://127.0.0.1:{port}")).with_tls(TLS_CERT, TLS_KEY);
    let (packets, mut received) = mpsc::unbounded_channel();
    let mut collector = Collector { packets, ports: 2 };
    let run = server.run(&mut collector);
    let client = async {
        // both caps announcements follow the bind, so the server is listening
        for _ in 0..2 {
            received.recv().await.unwrap();
        }
        let connector = tokio_native_tls::TlsConnector::from(
            tokio_native_tls::native_tls::TlsConnector::builder()
                .danger_accept_invalid_certs(true)
                .build()
                .unwrap(),
        );
        let tcp = tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .unwrap();
        let mut tls = connector
            .connect("localhost", tcp)
            .await
            .expect("tls handshake");
        tls.write_all(
            b"POST /whip HTTP/1.1\r\nHost: localhost\r\nContent-Type: text/plain\r\nContent-Length: 3\r\n\r\nv=0",
        )
        .await
        .unwrap();
        let mut response = Vec::new();
        tls.read_to_end(&mut response).await.unwrap();
        String::from_utf8(response).unwrap()
    };
    // a refused POST leaves the server waiting, so the client decides the end
    let text = tokio::time::timeout(RUN_TIMEOUT, async {
        tokio::select! {
            result = run => panic!("server ended early: {result:?}"),
            text = client => text,
        }
    })
    .await
    .expect("tls request timed out");
    assert!(text.starts_with("HTTP/1.1 415"), "{text}");
    assert!(text.contains("Access-Control-Allow-Origin: *"), "{text}");
}

const PUBLISHERS: usize = 2;

/// `max-publishers=2`: two publishers land on their own pad pairs, a third POST
/// is refused with 409 while both slots are taken, and the run ends with EOS on
/// every pad once the last publisher leaves.
#[tokio::test]
async fn publishers_land_on_their_own_pad_pairs() {
    let codec = WebRtcVideoCodec::default();
    let interval = Duration::from_secs(1) / FPS;
    let units = synthetic_stream(codec, VIDEO_UNITS);
    let host_addr = format!("http://127.0.0.1:{}", free_port());
    let whip_url = format!("{host_addr}/whip");
    let mut server = WhipServerSrc::new(host_addr.clone());
    server
        .set_property("max-publishers", PropValue::Uint(PUBLISHERS as u64))
        .unwrap();
    let slot_stats = server.publisher_stats_handles();
    let port_count = PUBLISHERS * 2;
    let (packets, mut received) = mpsc::unbounded_channel();
    let mut collector = Collector {
        packets,
        ports: port_count,
    };

    let run: Pin<Box<dyn Future<Output = Result<u64, G2gError>> + '_>> = server.run(&mut collector);
    let clients = async {
        let mut announced = Vec::new();
        for _ in 0..port_count {
            announced.push(received.recv().await.unwrap());
        }
        let first = publish(whip_url.clone(), fixed_caps(codec), &units, interval);
        // the second publisher starts a little later so the third POST below
        // lands while both slots are taken
        let second = async {
            tokio::time::sleep(interval * 4).await;
            publish(whip_url.clone(), fixed_caps(codec), &units, interval).await
        };
        let refused = async {
            tokio::time::sleep(interval * 12).await;
            reqwest::Client::new()
                .post(&whip_url)
                .header("Content-Type", "application/sdp")
                .body("v=0\r\n")
                .send()
                .await
                .unwrap()
                .status()
                .as_u16()
        };
        let (first, second, refused) = tokio::join!(first, second, refused);
        (announced, [first, second], refused)
    };
    let (result, (announced, sink_stats, refused)) =
        tokio::time::timeout(RUN_TIMEOUT, async { tokio::join!(run, clients) })
            .await
            .expect("loopback run timed out");
    result.expect("server run failed");
    assert_eq!(refused, 409);

    let mut ports: Vec<Vec<PipelinePacket>> = (0..port_count).map(|_| Vec::new()).collect();
    for (port, packet) in announced {
        ports[port].push(packet);
    }
    while let Ok((port, packet)) = received.try_recv() {
        ports[port].push(packet);
    }
    for slot in 0..PUBLISHERS {
        let video = &ports[slot * 2 + VIDEO_PORT];
        assert!(
            matches!(
                video.first(),
                Some(PipelinePacket::CapsChanged(Caps::CompressedVideo { .. }))
            ),
            "slot {slot}: video pad did not open with video caps"
        );
        let frames = data_frames(video);
        assert!(
            frames.len() >= min_delivered(units.len()),
            "slot {slot}: {} of {} units arrived",
            frames.len(),
            units.len()
        );
        assert_eq!(bytes(frames[0]), &units[0][..], "slot {slot}");
        assert!(
            data_frames(&ports[slot * 2 + AUDIO_PORT]).is_empty(),
            "slot {slot}"
        );
        let stats = slot_stats[slot].snapshot();
        assert!(
            stats.packets_received() >= frames.len() as u64,
            "slot {slot} counted {} packets for {} frames",
            stats.packets_received(),
            frames.len()
        );
        assert!(sink_stats[slot].packets_sent() >= frames.len() as u64);
    }
    for (port, packets) in ports.iter().enumerate() {
        assert!(
            matches!(packets.last(), Some(PipelinePacket::Eos)),
            "port {port} did not end with EOS"
        );
    }
}
