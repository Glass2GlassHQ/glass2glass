//! M834: multi-client RTSP *ingest* (`RtspServerSrc`). One publisher records at a
//! time onto the single output pad, but publishers are now sequential: when one
//! disconnects, tears down, or falls silent, its session is dropped and the next
//! publisher takes over the same pad without restarting the graph. A publisher
//! that connects while another is recording is refused with 503 rather than
//! queued, and per-session state (RTP port, negotiated transport, responder)
//! never carries over to the next one.
//!
//! The loopback tests drive a hand-rolled publisher; the ffmpeg interop tests
//! (ignored by default) drive the same paths from a reference RTSP client:
//!
//! ```sh
//! cargo test -p g2g-plugins --features rtsp-server \
//!     --test m834_rtsp_ingest_multiclient -- --ignored --nocapture
//! ```
#![cfg(feature = "rtsp-server")]

use core::future::Future;
use core::pin::Pin;
use std::net::{SocketAddr, TcpListener as StdTcpListener, UdpSocket as StdUdpSocket};
use std::time::Duration;

use g2g_core::runtime::SourceLoop;
use g2g_core::{Caps, Dim, G2gError, OutputSink, PipelinePacket, PushOutcome, Rate, VideoCodec};

use g2g_plugins::rtppay::RtpH264Packetizer;
use g2g_plugins::rtspserversrc::RtspServerSrc;

use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Every access unit the source emitted: its tag byte, PTS and sequence number.
#[derive(Default)]
struct Capture {
    frames: Vec<(u8, u64, u64)>,
    eos: usize,
}

impl Capture {
    fn tags(&self) -> Vec<u8> {
        self.frames.iter().map(|(tag, _, _)| *tag).collect()
    }
}

impl OutputSink for Capture {
    fn push<'a>(
        &'a mut self,
        p: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        match &p {
            PipelinePacket::DataFrame(frame) => {
                if let Some(slice) = frame.domain.as_system_slice() {
                    self.frames.push((
                        slice.get(5).copied().unwrap_or(0),
                        frame.timing.pts_ns,
                        frame.sequence,
                    ));
                }
            }
            PipelinePacket::Eos => self.eos += 1,
            _ => {}
        }
        Box::pin(async { Ok(PushOutcome::Accepted) })
    }
}

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// One small Annex-B IDR access unit, tagged at byte 5.
fn access_unit(tag: u8) -> Vec<u8> {
    vec![0u8, 0, 0, 1, 0x65, tag, 0xAB, 0xCD]
}

/// Frame an RTP packet as interleaved binary (RFC 2326 §10.12).
fn interleaved(channel: u8, rtp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + rtp.len());
    out.push(0x24);
    out.push(channel);
    out.extend_from_slice(&(rtp.len() as u16).to_be_bytes());
    out.extend_from_slice(rtp);
    out
}

/// Read one full RTSP response (header block + any Content-Length body).
async fn read_response(sock: &mut tokio::net::TcpStream) -> String {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 2048];
    loop {
        let n = sock.read(&mut tmp).await.expect("read response");
        assert!(n > 0, "server closed the control connection");
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&buf[..pos]).to_string();
            let want_body = head
                .lines()
                .find_map(|l| l.strip_prefix("Content-Length:"))
                .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                .unwrap_or(0);
            if buf.len() >= pos + 4 + want_body {
                return String::from_utf8_lossy(&buf).to_string();
            }
        }
    }
}

const SDP: &str =
    "v=0\r\no=- 0 0 IN IP4 0.0.0.0\r\ns=g2g\r\nm=video 0 RTP/AVP 96\r\na=rtpmap:96 H264/90000\r\n";
const URL: &str = "rtsp://127.0.0.1/stream";

/// OPTIONS + ANNOUNCE, the prelude both transports share. A publisher that
/// reconnects the instant the previous one dropped can still catch the tail of
/// that session and be refused, so retry until the pad is free, as a real
/// reconnecting encoder does.
async fn announce(rtsp_addr: SocketAddr) -> tokio::net::TcpStream {
    let mut ctrl = loop {
        let mut ctrl = tokio::net::TcpStream::connect(rtsp_addr)
            .await
            .expect("connect rtsp");
        ctrl.write_all(format!("OPTIONS {URL} RTSP/1.0\r\nCSeq: 1\r\n\r\n").as_bytes())
            .await
            .unwrap();
        let options = read_response(&mut ctrl).await;
        if options.contains("200 OK") {
            break ctrl;
        }
        assert!(
            options.contains("503 Service Unavailable"),
            "unexpected OPTIONS answer: {options}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    ctrl.write_all(
        format!(
            "ANNOUNCE {URL} RTSP/1.0\r\nCSeq: 2\r\nContent-Type: application/sdp\r\nContent-Length: {}\r\n\r\n{SDP}",
            SDP.len()
        )
        .as_bytes(),
    )
    .await
    .unwrap();
    assert!(read_response(&mut ctrl).await.contains("200 OK"));
    ctrl
}

/// Publish over unicast UDP: returns the control stream, an RTP socket already
/// pointed at the server, and the server RTP port SETUP advertised.
async fn publish_udp(rtsp_addr: SocketAddr) -> (tokio::net::TcpStream, tokio::net::UdpSocket, u16) {
    let mut ctrl = announce(rtsp_addr).await;
    let rtp = tokio::net::UdpSocket::bind("127.0.0.1:0")
        .await
        .expect("bind publisher rtp");
    let client_rtp_port = rtp.local_addr().unwrap().port();
    ctrl.write_all(
        format!(
            "SETUP {URL}/streamid=0 RTSP/1.0\r\nCSeq: 3\r\nTransport: RTP/AVP;unicast;client_port={client_rtp_port}-{};mode=record\r\n\r\n",
            client_rtp_port + 1
        )
        .as_bytes(),
    )
    .await
    .unwrap();
    let setup = read_response(&mut ctrl).await;
    assert!(
        !setup.contains("interleaved="),
        "a UDP SETUP must not be answered with an interleaved transport: {setup}"
    );
    let server_port: u16 = setup
        .split("server_port=")
        .nth(1)
        .and_then(|s| s.split(['-', ';', '\r']).next())
        .and_then(|s| s.trim().parse().ok())
        .expect("SETUP advertises a server_port");
    ctrl.write_all(
        format!("RECORD {URL} RTSP/1.0\r\nCSeq: 4\r\nSession: 12345678\r\n\r\n").as_bytes(),
    )
    .await
    .unwrap();
    assert!(read_response(&mut ctrl).await.contains("200 OK"));
    rtp.connect(("127.0.0.1", server_port))
        .await
        .expect("connect rtp dest");
    (ctrl, rtp, server_port)
}

/// Publish over TCP-interleaved: the control stream also carries the RTP.
async fn publish_interleaved(rtsp_addr: SocketAddr) -> tokio::net::TcpStream {
    let mut ctrl = announce(rtsp_addr).await;
    ctrl.write_all(
        format!("SETUP {URL}/streamid=0 RTSP/1.0\r\nCSeq: 3\r\nTransport: RTP/AVP/TCP;unicast;interleaved=0-1;mode=record\r\n\r\n")
            .as_bytes(),
    )
    .await
    .unwrap();
    let setup = read_response(&mut ctrl).await;
    assert!(setup.contains("interleaved=0-1"), "{setup}");
    ctrl.write_all(
        format!("RECORD {URL} RTSP/1.0\r\nCSeq: 4\r\nSession: 12345678\r\n\r\n").as_bytes(),
    )
    .await
    .unwrap();
    assert!(read_response(&mut ctrl).await.contains("200 OK"));
    ctrl
}

/// Send access units `tags` as RTP over UDP, one packet each. The packetizer is
/// the session's, so RTP sequence numbers keep rising within a session.
async fn send_udp_aus(rtp: &tokio::net::UdpSocket, pktz: &mut RtpH264Packetizer, tags: &[u8]) {
    for tag in tags {
        for pkt in pktz.packetize(&access_unit(*tag), *tag as u32 * 3000) {
            rtp.send(&pkt).await.expect("send rtp");
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    tokio::time::sleep(Duration::from_millis(100)).await;
}

/// Send access units `tags` as `$`-framed RTP on the control connection.
async fn send_interleaved_aus(
    ctrl: &mut tokio::net::TcpStream,
    pktz: &mut RtpH264Packetizer,
    tags: &[u8],
) {
    for tag in tags {
        for pkt in pktz.packetize(&access_unit(*tag), *tag as u32 * 3000) {
            ctrl.write_all(&interleaved(0, &pkt))
                .await
                .expect("send interleaved rtp");
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    tokio::time::sleep(Duration::from_millis(100)).await;
}

/// Bind `port` to prove the session that owned it released its socket, keeping
/// the binding so the next session cannot silently reuse the same port.
async fn claim_released_port(port: u16) -> StdUdpSocket {
    for _ in 0..200 {
        if let Ok(sock) = StdUdpSocket::bind(("0.0.0.0", port)) {
            return sock;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("session RTP port {port} was never released");
}

/// Run the source to completion (within `budget`) and report what it emitted
/// alongside its session bookkeeping.
async fn serve(mut src: RtspServerSrc, budget: Duration) -> (u64, Capture, u64, u64) {
    src.configure_pipeline(&h264_caps()).expect("configure");
    let mut cap = Capture::default();
    let n = tokio::time::timeout(budget, src.run(&mut cap))
        .await
        .expect("source completes within the budget")
        .expect("source runs");
    (n, cap, src.sessions_served(), src.sessions_refused())
}

fn bound_src() -> (RtspServerSrc, SocketAddr) {
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind rtsp");
    let src = RtspServerSrc::from_listener(listener)
        .expect("adopt listener")
        .with_rtp(96, 0x1234_5678)
        .with_video_size(320, 240);
    let addr = ([127, 0, 0, 1], src.local_port().expect("bound")).into();
    (src, addr)
}

/// A publisher that leaves is replaced by the next one on the same output pad:
/// both streams are emitted, the sequence numbering continues, PTS keeps moving
/// forward across the handover, and the finished session's RTP port is released.
#[tokio::test]
async fn a_second_publisher_takes_over_the_pad_after_the_first_leaves() {
    let (src, addr) = bound_src();
    let src = src.with_max_sessions(2);

    let publisher = async move {
        let (ctrl, rtp, port_a) = publish_udp(addr).await;
        let mut pktz = RtpH264Packetizer::new(96, 0x1234_5678);
        send_udp_aus(&rtp, &mut pktz, &[0, 1, 2, 3]).await;
        drop(ctrl);
        drop(rtp);

        // The session's socket goes with the session: hold its port so the next
        // one is forced to allocate a fresh socket rather than inherit this one.
        let held = claim_released_port(port_a).await;

        let (ctrl, rtp, port_b) = publish_udp(addr).await;
        assert_ne!(port_a, port_b, "each session binds its own RTP port");
        let mut pktz = RtpH264Packetizer::new(96, 0x1234_5678);
        send_udp_aus(&rtp, &mut pktz, &[4, 5, 6, 7]).await;
        drop(ctrl);
        drop(rtp);
        drop(held);
    };

    let (received, cap, served, refused) =
        tokio::join!(publisher, serve(src, Duration::from_secs(15))).1;

    assert_eq!(received, 8, "both publishes were ingested");
    assert_eq!(cap.tags(), (0..8).collect::<Vec<u8>>());
    assert_eq!(served, 2, "two publishers reached RECORD");
    assert_eq!(refused, 0, "neither publisher overlapped the other");
    assert_eq!(cap.eos, 1, "EOS once, after the last session");
    let seqs: Vec<u64> = cap.frames.iter().map(|(_, _, seq)| *seq).collect();
    assert_eq!(
        seqs,
        (0..8).collect::<Vec<u64>>(),
        "sequence numbering continues across the handover"
    );
    for pair in cap.frames.windows(2) {
        assert!(
            pair[1].1 > pair[0].1,
            "PTS keeps moving forward across the handover: {:?}",
            cap.frames
        );
    }
}

/// A publisher that connects while another is recording is refused with 503,
/// and the refusal does not disturb the live session.
#[tokio::test]
async fn a_publisher_arriving_mid_session_is_refused_with_503() {
    let (src, addr) = bound_src();
    let src = src.with_max_sessions(1);

    let publisher = async move {
        let (ctrl, rtp, _) = publish_udp(addr).await;
        let mut pktz = RtpH264Packetizer::new(96, 0x1234_5678);
        send_udp_aus(&rtp, &mut pktz, &[0, 1]).await;

        let mut second = tokio::net::TcpStream::connect(addr)
            .await
            .expect("a second publisher still reaches the listener");
        second
            .write_all(format!("OPTIONS {URL} RTSP/1.0\r\nCSeq: 7\r\n\r\n").as_bytes())
            .await
            .unwrap();
        let refusal = read_response(&mut second).await;
        assert!(
            refusal.starts_with("RTSP/1.0 503 Service Unavailable"),
            "the second publisher is refused, not queued: {refusal}"
        );
        assert!(refusal.contains("CSeq: 7"), "the refusal echoes CSeq");
        drop(second);

        send_udp_aus(&rtp, &mut pktz, &[2, 3]).await;
        drop(ctrl);
        drop(rtp);
    };

    let (received, cap, served, refused) =
        tokio::join!(publisher, serve(src, Duration::from_secs(15))).1;

    assert_eq!(
        received, 4,
        "the recording session ran on through the refusal"
    );
    assert_eq!(cap.tags(), vec![0, 1, 2, 3]);
    assert_eq!(served, 1);
    assert_eq!(refused, 1, "the overlapping publisher was refused");
}

/// A UDP publisher that disconnects mid-RECORD ends its session promptly: the
/// receive loop only ever sees datagrams, so the control-channel close is what
/// has to end it (before M834 the source waited for RTP forever).
#[tokio::test]
async fn a_disconnect_mid_record_does_not_stall_the_graph() {
    let (src, addr) = bound_src();
    let src = src.with_max_sessions(1);

    let publisher = async move {
        let (ctrl, rtp, _) = publish_udp(addr).await;
        let mut pktz = RtpH264Packetizer::new(96, 0x1234_5678);
        send_udp_aus(&rtp, &mut pktz, &[0]).await;
        drop(ctrl);
        drop(rtp);
    };

    // A budget far under any timeout: only the disconnect can end this.
    let (received, cap, served, _) = tokio::join!(publisher, serve(src, Duration::from_secs(5))).1;
    assert_eq!(received, 1);
    assert_eq!(cap.eos, 1, "the graph was ended, not left waiting");
    assert_eq!(served, 1);
}

/// A publisher that holds its control connection open but goes silent is reaped
/// by the session timeout, so a peer that vanishes without closing its socket
/// cannot pin the pad.
#[tokio::test]
async fn a_silent_publisher_is_reaped_by_the_session_timeout() {
    let (src, addr) = bound_src();
    let src = src
        .with_max_sessions(1)
        .with_session_timeout(Duration::from_millis(300));

    let publisher = async move {
        let (ctrl, rtp, _) = publish_udp(addr).await;
        let mut pktz = RtpH264Packetizer::new(96, 0x1234_5678);
        send_udp_aus(&rtp, &mut pktz, &[0]).await;
        // Neither media nor control traffic, but the sockets stay open.
        tokio::time::sleep(Duration::from_secs(2)).await;
        drop(ctrl);
        drop(rtp);
    };

    let (received, cap, served, _) = tokio::join!(publisher, serve(src, Duration::from_secs(5))).1;
    assert_eq!(received, 1);
    assert_eq!(cap.eos, 1, "the silent session was torn down");
    assert_eq!(served, 1);
}

/// Transport state does not survive a session: after an interleaved publisher
/// leaves, the next one negotiates plain UDP and its media is received (a leaked
/// interleaved channel would answer the SETUP wrong and swallow the RTP).
#[tokio::test]
async fn the_next_publisher_negotiates_its_own_transport() {
    let (src, addr) = bound_src();
    let src = src.with_max_sessions(2);

    let publisher = async move {
        let mut ctrl = publish_interleaved(addr).await;
        let mut pktz = RtpH264Packetizer::new(96, 0x1234_5678);
        send_interleaved_aus(&mut ctrl, &mut pktz, &[0, 1]).await;
        drop(ctrl);

        // publish_udp asserts the SETUP answer is a UDP transport, not the
        // interleaved one the previous session negotiated.
        let (ctrl, rtp, _) = publish_udp(addr).await;
        let mut pktz = RtpH264Packetizer::new(96, 0x1234_5678);
        send_udp_aus(&rtp, &mut pktz, &[2, 3]).await;
        drop(ctrl);
        drop(rtp);
    };

    let (received, cap, served, _) = tokio::join!(publisher, serve(src, Duration::from_secs(15))).1;
    assert_eq!(received, 4, "both transports were ingested in turn");
    assert_eq!(cap.tags(), vec![0, 1, 2, 3]);
    assert_eq!(served, 2);
}

/// The session knobs are runtime properties, not just builders.
#[test]
fn session_properties_round_trip() {
    use g2g_core::{PropValue, PropertySpec};
    let mut src = RtspServerSrc::new("0.0.0.0:8554".parse().unwrap());
    let declares =
        |specs: &[PropertySpec], name: &str| specs.iter().any(|s: &PropertySpec| s.name == name);
    assert!(declares(SourceLoop::properties(&src), "max-sessions"));
    assert!(declares(SourceLoop::properties(&src), "timeout"));

    src.set_property("max-sessions", PropValue::Uint(3))
        .unwrap();
    assert_eq!(src.get_property("max-sessions"), Some(PropValue::Uint(3)));
    src.set_property("timeout", PropValue::Uint(15)).unwrap();
    assert_eq!(src.get_property("timeout"), Some(PropValue::Uint(15)));
    // 0 disables reaping and reads back as 0.
    src.set_property("timeout", PropValue::Uint(0)).unwrap();
    assert_eq!(src.get_property("timeout"), Some(PropValue::Uint(0)));
    assert!(src
        .set_property("timeout", PropValue::Uint(604_801))
        .is_err());
}

/// Reference peer: two ffmpeg publishes in sequence into one running source, the
/// reconnecting-encoder case. Both must be ingested on the same output pad.
#[tokio::test]
#[ignore = "needs ffmpeg with RTSP; opens local TCP + UDP sockets"]
async fn ffmpeg_publishes_twice_in_sequence() {
    let (src, addr) = bound_src();
    let src = src.with_max_sessions(2);
    let url = format!("rtsp://127.0.0.1:{}/stream", addr.port());

    let publisher = tokio::task::spawn_blocking(move || {
        let first = ffmpeg_publish(&url, "tcp", 2);
        let second = ffmpeg_publish(&url, "udp", 2);
        (first, second)
    });

    let (received, cap, served, refused) = tokio::join!(
        async { publisher.await.unwrap() },
        serve(src, Duration::from_secs(60))
    )
    .1;

    assert_eq!(served, 2, "both ffmpeg publishes reached RECORD");
    assert_eq!(refused, 0, "the publishes did not overlap");
    assert!(
        received > 20,
        "both publishes contributed access units (got {received})"
    );
    for pair in cap.frames.windows(2) {
        assert!(
            pair[1].1 >= pair[0].1,
            "PTS never goes backwards across the re-publish"
        );
    }
}

/// Reference peer: a second ffmpeg publishing while the first records is refused
/// cleanly (it exits with an error), and the first stream is unaffected.
#[tokio::test]
#[ignore = "needs ffmpeg with RTSP; opens local TCP + UDP sockets"]
async fn ffmpeg_second_concurrent_publisher_is_refused() {
    let (src, addr) = bound_src();
    let src = src.with_max_sessions(1);
    let url = format!("rtsp://127.0.0.1:{}/stream", addr.port());

    let first_url = url.clone();
    let first = tokio::task::spawn_blocking(move || ffmpeg_publish(&first_url, "tcp", 5));
    let second = tokio::task::spawn_blocking(move || {
        std::thread::sleep(Duration::from_secs(2));
        ffmpeg_publish(&url, "tcp", 5)
    });

    let publishers = async {
        let (a, b) = tokio::join!(first, second);
        (a.unwrap(), b.unwrap())
    };
    let ((_, second_ok), (received, _cap, served, refused)) =
        tokio::join!(publishers, serve(src, Duration::from_secs(60)));

    assert!(
        !second_ok,
        "the overlapping publisher fails instead of hanging"
    );
    assert_eq!(served, 1, "only one publisher recorded");
    assert!(refused >= 1, "the overlapping publisher was refused");
    assert!(received > 10, "the first publish was unaffected");
}

/// Publish `secs` seconds of H.264 to `url` with ffmpeg. Returns whether it
/// exited successfully.
fn ffmpeg_publish(url: &str, transport: &str, secs: u32) -> bool {
    std::process::Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-re",
            "-an",
            "-f",
            "lavfi",
            "-i",
            &format!("testsrc=size=320x240:rate=15:duration={secs}"),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-f",
            "rtsp",
            "-rtsp_transport",
            transport,
            url,
        ])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}
