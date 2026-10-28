//! `WhipServerSrc` against the g2g WHIP client (`WebRtcSink`) on localhost, once
//! per carried video codec and once for Opus audio. Synthetic access units, no
//! media server and no fixture files, so it runs by default.

#![cfg(all(target_os = "linux", feature = "webrtc"))]

mod webrtc_streams_common;

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};
use std::net::IpAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use str0m::change::SdpOffer;
use str0m::crypto::from_feature_flags;
use str0m::net::{Protocol, Receive};
use str0m::{Candidate, Event, Input, Output, RtcConfig};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream, UdpSocket};
use tokio::sync::mpsc;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, FrameTiming, G2gError, MemoryDomain, MultiOutputSink,
    MultiOutputSource, OutputSink, PipelinePacket, PropValue, PushOutcome, Reconfigure,
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

const RECONNECT_ATTEMPTS: u64 = 5;
// str0m takes about 22 s to call the lost session disconnected
const RESTART_TIMEOUT: Duration = Duration::from_secs(60);
// the stopped server's port frees once the runtime drops its aborted listener task
const RESTART_GAP: Duration = Duration::from_millis(100);

/// Send `unit` as frame `sequence` of a stream at `FPS`, then wait one frame
/// interval.
async fn send_unit(sink: &mut WebRtcSink, sequence: u64, unit: &[u8]) {
    let interval = Duration::from_secs(1) / FPS;
    let timing = FrameTiming {
        pts_ns: sequence * interval.as_nanos() as u64,
        ..FrameTiming::default()
    };
    let frame = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(unit.to_vec().into_boxed_slice())),
        timing,
        sequence,
    );
    sink.process(PipelinePacket::DataFrame(frame), &mut Discard)
        .await
        .unwrap();
    tokio::time::sleep(interval).await;
}

/// Publish `inter` until the server emits a frame, sending `keyframe` whenever
/// the sink asks the encoder for one. Returns the server's first frame.
async fn publish_until_received(
    sink: &mut WebRtcSink,
    received: &mut mpsc::UnboundedReceiver<(usize, PipelinePacket)>,
    keyframe: &[u8],
    inter: &[u8],
) -> Frame {
    for sequence in 0u64.. {
        let unit = match sink.take_reconfigure() {
            Some(Reconfigure::ForceKeyframe) => keyframe,
            _ => inter,
        };
        send_unit(sink, sequence, unit).await;
        while let Ok((_, packet)) = received.try_recv() {
            if let PipelinePacket::DataFrame(frame) = packet {
                return frame;
            }
        }
    }
    unreachable!("the sequence counter never runs out")
}

/// Run a server on `host_addr` while `sink` publishes into it, and return the
/// first frame it emits. Dropping the run on return stops the server.
async fn first_frame_of_a_server(
    host_addr: &str,
    sink: &mut WebRtcSink,
    keyframe: &[u8],
    inter: &[u8],
) -> Frame {
    let mut server = WhipServerSrc::new(host_addr);
    let (packets, mut received) = mpsc::unbounded_channel();
    let mut collector = Collector { packets, ports: 2 };
    let run = server.run(&mut collector);
    let client = async {
        // both caps announcements follow the bind, so the server is listening
        for _ in 0..2 {
            received.recv().await.unwrap();
        }
        publish_until_received(sink, &mut received, keyframe, inter).await
    };
    tokio::time::timeout(RESTART_TIMEOUT, async {
        tokio::select! {
            result = run => panic!("server ended early: {result:?}"),
            frame = client => frame,
        }
    })
    .await
    .expect("no frame reached the server")
}

/// A server that restarts on the same port has forgotten the sink's session:
/// the sink POSTs a new offer, asks the encoder for a keyframe, and the new
/// server's first frame is that keyframe.
#[tokio::test]
async fn sink_republishes_into_a_restarted_server() {
    let codec = WebRtcVideoCodec::H264;
    let units = synthetic_stream(codec, 2);
    let (keyframe, inter) = (&units[0], &units[1]);
    let host_addr = format!("http://127.0.0.1:{}", free_port());
    let mut sink = WebRtcSink::new(format!("{host_addr}/whip"));
    sink.set_property("reconnect", PropValue::Uint(RECONNECT_ATTEMPTS))
        .unwrap();
    sink.configure_pipeline(&fixed_caps(codec)).unwrap();

    first_frame_of_a_server(&host_addr, &mut sink, keyframe, keyframe).await;
    tokio::time::sleep(RESTART_GAP).await;
    let frame = first_frame_of_a_server(&host_addr, &mut sink, keyframe, inter).await;

    assert_eq!(bytes(&frame), &keyframe[..]);
    assert!(frame.timing.keyframe);
}

// read one HTTP request from `stream` and return its body
async fn read_request_body(stream: &mut TcpStream) -> String {
    let mut request = Vec::new();
    let mut buf = [0u8; 4096];
    let header_end = loop {
        let n = stream.read(&mut buf).await.unwrap();
        assert_ne!(n, 0, "the client closed before sending a request");
        request.extend_from_slice(&buf[..n]);
        if let Some(at) = request.windows(4).position(|w| w == b"\r\n\r\n") {
            break at + 4;
        }
    };
    let headers = String::from_utf8(request[..header_end].to_vec()).unwrap();
    let content_length: usize = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse().unwrap())
        })
        .expect("the request has a Content-Length");
    while request.len() < header_end + content_length {
        let n = stream.read(&mut buf).await.unwrap();
        assert_ne!(n, 0, "the client closed mid-body");
        request.extend_from_slice(&buf[..n]);
    }
    String::from_utf8(request[header_end..].to_vec()).unwrap()
}

/// Answer one WHIP offer on `listener`, wait for DTLS to connect, then stop
/// listening and close the session with a DTLS close_notify, the way mediamtx
/// ends its sessions on SIGTERM.
async fn answer_then_close(listener: TcpListener) {
    let (mut stream, _) = listener.accept().await.unwrap();
    let offer = read_request_body(&mut stream).await;
    let candidate_ip: IpAddr = offer
        .lines()
        .find(|l| l.starts_with("a=candidate:"))
        .and_then(|l| l.split_whitespace().nth(4))
        .expect("the offer carries a host candidate")
        .parse()
        .unwrap();
    let socket = UdpSocket::bind((candidate_ip, 0)).await.unwrap();
    let local = socket.local_addr().unwrap();
    let mut rtc = RtcConfig::new()
        .set_crypto_provider(Arc::new(from_feature_flags()))
        .build(Instant::now());
    rtc.add_local_candidate(Candidate::host(local, "udp").unwrap());
    let answer = rtc
        .sdp_api()
        .accept_offer(SdpOffer::from_sdp_string(&offer).unwrap())
        .unwrap()
        .to_sdp_string();
    let response = format!(
        "HTTP/1.1 201 Created\r\nContent-Type: application/sdp\r\n\
         Location: /whip/closing\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{answer}",
        answer.len()
    );
    stream.write_all(response.as_bytes()).await.unwrap();
    drop(stream);
    drop(listener);

    let mut buf = vec![0u8; 2048];
    loop {
        let deadline = loop {
            match rtc.poll_output().unwrap() {
                Output::Timeout(t) => break t,
                Output::Transmit(t) => {
                    socket.send_to(&t.contents, t.destination).await.unwrap();
                }
                Output::Event(Event::Connected) => rtc.close().unwrap(),
                Output::Event(_) => {}
            }
        };
        if !rtc.is_alive() {
            return;
        }
        let timeout = deadline.saturating_duration_since(Instant::now());
        tokio::select! {
            received = socket.recv_from(&mut buf) => {
                let (n, source) = received.unwrap();
                let receive = Receive {
                    proto: Protocol::Udp,
                    source,
                    destination: local,
                    contents: buf[..n].try_into().unwrap(),
                };
                rtc.handle_input(Input::Receive(Instant::now(), receive)).unwrap();
            }
            _ = tokio::time::sleep(timeout) => {
                rtc.handle_input(Input::Timeout(Instant::now())).unwrap();
            }
        }
    }
}

/// A server that shuts down cleanly sends the sink a DTLS close_notify: the sink
/// POSTs a new offer, and the server that takes over the port gets a keyframe.
#[tokio::test]
async fn sink_republishes_after_the_server_closes_the_session() {
    let codec = WebRtcVideoCodec::H264;
    let units = synthetic_stream(codec, 2);
    let (keyframe, inter) = (&units[0], &units[1]);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let host_addr = format!("http://{}", listener.local_addr().unwrap());
    let mut sink = WebRtcSink::new(format!("{host_addr}/whip"));
    sink.set_property("reconnect", PropValue::Uint(RECONNECT_ATTEMPTS))
        .unwrap();
    sink.configure_pipeline(&fixed_caps(codec)).unwrap();

    let closed = tokio::spawn(answer_then_close(listener));
    for sequence in 0u64.. {
        if closed.is_finished() {
            break;
        }
        send_unit(&mut sink, sequence, keyframe).await;
    }
    closed.await.unwrap();
    let frame = first_frame_of_a_server(&host_addr, &mut sink, keyframe, inter).await;

    assert_eq!(bytes(&frame), &keyframe[..]);
    assert!(frame.timing.keyframe);
}
