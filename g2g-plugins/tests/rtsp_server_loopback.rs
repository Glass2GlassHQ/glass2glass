//! RtspServerSink end-to-end over loopback: a minimal RTSP player connects,
//! runs OPTIONS / DESCRIBE / SETUP / PLAY against the server sink, then receives
//! the RTP it streams and depayloads the access units back. Proves the serving
//! path (RTSP control handshake + RTP/UDP transport) without an external client.
#![cfg(feature = "rtsp-server")]

use core::future::Future;
use core::pin::Pin;
use std::net::TcpListener as StdTcpListener;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket,
    PushOutcome, Rate, VideoCodec,
};

use g2g_plugins::rtcp::{self, RtcpPacket};
use g2g_plugins::rtpdepay::RtpH264Depayloader;
use g2g_plugins::rtspserversink::RtspServerSink;

use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// A no-op downstream for driving a sink's `process` directly.
struct NullOut;
impl OutputSink for NullOut {
    fn push<'a>(
        &'a mut self,
        _p: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
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

/// One small Annex-B IDR access unit, tagged at byte 5 so the receiver can tell
/// them apart.
fn access_unit(tag: u8) -> Vec<u8> {
    vec![0u8, 0, 0, 1, 0x65, tag, 0xAB, 0xCD]
}

/// Read one full RTSP response (headers, plus any Content-Length body) from the
/// control socket.
async fn read_response(sock: &mut tokio::net::TcpStream) -> String {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 2048];
    loop {
        let n = sock.read(&mut tmp).await.expect("read response");
        assert!(n > 0, "server closed the control connection");
        buf.extend_from_slice(&tmp[..n]);
        // Have we got the full header block?
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

#[tokio::test]
async fn rtsp_player_handshakes_then_receives_rtp() {
    const N: u8 = 8;

    // Server sink on an ephemeral RTSP port.
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind rtsp");
    let rtsp_addr = listener.local_addr().unwrap();
    let mut sink = RtspServerSink::from_listener(listener)
        .unwrap()
        .with_rtp(96, 0x1234_5678);
    sink.configure_pipeline(&h264_caps()).expect("configure");

    // Client RTP socket; its port is what we put in the SETUP Transport header.
    let rtp = tokio::net::UdpSocket::bind("127.0.0.1:0")
        .await
        .expect("bind client rtp");
    let client_rtp_port = rtp.local_addr().unwrap().port();

    // The player: connect, run the handshake, then receive RTP and depayload.
    let client = async move {
        let mut ctrl = tokio::net::TcpStream::connect(rtsp_addr)
            .await
            .expect("connect rtsp");
        let url = "rtsp://127.0.0.1/stream";

        ctrl.write_all(format!("OPTIONS {url} RTSP/1.0\r\nCSeq: 1\r\n\r\n").as_bytes())
            .await
            .unwrap();
        assert!(read_response(&mut ctrl).await.contains("200 OK"));

        ctrl.write_all(
            format!("DESCRIBE {url} RTSP/1.0\r\nCSeq: 2\r\nAccept: application/sdp\r\n\r\n")
                .as_bytes(),
        )
        .await
        .unwrap();
        assert!(read_response(&mut ctrl)
            .await
            .contains("m=video 0 RTP/AVP 96"));

        let setup = format!(
            "SETUP {url}/streamid=0 RTSP/1.0\r\nCSeq: 3\r\nTransport: RTP/AVP;unicast;client_port={client_rtp_port}-{}\r\n\r\n",
            client_rtp_port + 1,
        );
        ctrl.write_all(setup.as_bytes()).await.unwrap();
        let setup_resp = read_response(&mut ctrl).await;
        assert!(setup_resp.contains("Session:"), "SETUP assigns a session");
        assert!(setup_resp.contains("server_port="));

        ctrl.write_all(
            format!("PLAY {url} RTSP/1.0\r\nCSeq: 4\r\nSession: 12345678\r\n\r\n").as_bytes(),
        )
        .await
        .unwrap();
        assert!(read_response(&mut ctrl).await.contains("200 OK"));

        // Now receive the RTP stream and recover the access units.
        let mut depay = RtpH264Depayloader::new();
        let mut tags = Vec::new();
        let mut pkt = [0u8; 2048];
        while tags.len() < N as usize {
            let recv =
                tokio::time::timeout(std::time::Duration::from_secs(5), rtp.recv(&mut pkt)).await;
            let n = recv.expect("rtp arrives within 5s").expect("recv rtp");
            if let Some(au) = depay.depacketize(&pkt[..n]) {
                // Annex-B payload: [0,0,0,1][NAL][tag ..]; recover the tag byte.
                tags.push(au.data.get(5).copied().unwrap_or(0));
            }
        }
        tags
    };

    // Drive the sink: the first frame blocks until the player connects + PLAYs.
    // Stream more than N so the player reliably drains N even if it starts its
    // recv loop a beat after PLAY; once the player leaves, a send may fail with
    // ECONNREFUSED, which (after N) just means it is done.
    let server = async move {
        let mut null = NullOut;
        for i in 0u8..(N * 3) {
            let au = access_unit(i % N);
            let frame = Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(au.into_boxed_slice())),
                timing: FrameTiming {
                    pts_ns: i as u64 * 33_000_000,
                    ..FrameTiming::default()
                },
                sequence: i as u64,
                meta: Default::default(),
            };
            match sink
                .process(PipelinePacket::DataFrame(frame), &mut null)
                .await
            {
                Ok(()) => {}
                Err(_) if sink.frames_sent() >= N as u64 => break, // player left after draining
                Err(e) => panic!("stream frame: {e:?}"),
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        sink.frames_sent()
    };

    let (tags, frames_sent) = tokio::join!(client, server);
    assert!(frames_sent >= N as u64, "server streamed frames after PLAY");
    let expected: Vec<u8> = (0..N).collect();
    assert_eq!(
        tags, expected,
        "player received and depayloaded every AU in order"
    );
}

/// Bind an RTP/RTCP client socket pair on adjacent ports (the `client_port`
/// pair a SETUP advertises).
async fn bind_client_pair() -> (tokio::net::UdpSocket, tokio::net::UdpSocket) {
    for _ in 0..16 {
        let rtp = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let port = rtp.local_addr().unwrap().port();
        let Some(rtcp_port) = port.checked_add(1) else {
            continue;
        };
        if let Ok(rtcp) = tokio::net::UdpSocket::bind(("127.0.0.1", rtcp_port)).await {
            return (rtp, rtcp);
        }
    }
    panic!("no adjacent udp port pair available");
}

#[tokio::test]
async fn sender_reports_flow_and_a_silent_player_is_reaped() {
    // Timeline (generous margins for CI): the player keepalives via RTCP RRs
    // until KEEPALIVE_MS, well past the session timeout, proving RRs count as
    // activity; then it goes silent with the control connection still open, so
    // the reap that follows can only come from the timeout.
    const TIMEOUT_MS: u64 = 300;
    const KEEPALIVE_MS: u64 = 700;
    const END_MS: u64 = 1800;

    let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind rtsp");
    let rtsp_addr = listener.local_addr().unwrap();
    let mut sink = RtspServerSink::from_listener(listener)
        .unwrap()
        .with_rtp(96, 0x1234_5678)
        .with_rtcp_sr_interval(std::time::Duration::from_millis(50))
        .with_session_timeout(std::time::Duration::from_millis(TIMEOUT_MS));
    sink.configure_pipeline(&h264_caps()).expect("configure");

    let (rtp, rtcp_sock) = bind_client_pair().await;
    let client_rtp_port = rtp.local_addr().unwrap().port();

    let client = async move {
        let mut ctrl = tokio::net::TcpStream::connect(rtsp_addr)
            .await
            .expect("connect rtsp");
        let url = "rtsp://127.0.0.1/stream";

        ctrl.write_all(format!("OPTIONS {url} RTSP/1.0\r\nCSeq: 1\r\n\r\n").as_bytes())
            .await
            .unwrap();
        read_response(&mut ctrl).await;
        ctrl.write_all(format!("DESCRIBE {url} RTSP/1.0\r\nCSeq: 2\r\n\r\n").as_bytes())
            .await
            .unwrap();
        read_response(&mut ctrl).await;

        let setup = format!(
            "SETUP {url}/streamid=0 RTSP/1.0\r\nCSeq: 3\r\nTransport: RTP/AVP;unicast;client_port={client_rtp_port}-{}\r\n\r\n",
            client_rtp_port + 1,
        );
        ctrl.write_all(setup.as_bytes()).await.unwrap();
        let setup_resp = read_response(&mut ctrl).await;
        // 300 ms rounds up to a whole-second advertisement.
        assert!(
            setup_resp.contains(";timeout=1"),
            "SETUP advertises the session timeout: {setup_resp}"
        );
        let server_rtp_port: u16 = setup_resp
            .split("server_port=")
            .nth(1)
            .and_then(|s| s.split('-').next())
            .and_then(|p| p.trim().parse().ok())
            .expect("server_port in SETUP response");

        ctrl.write_all(
            format!("PLAY {url} RTSP/1.0\r\nCSeq: 4\r\nSession: 12345678\r\n\r\n").as_bytes(),
        )
        .await
        .unwrap();
        read_response(&mut ctrl).await;
        let started = std::time::Instant::now();

        // Keepalive phase: receive RTP + RTCP, send an RR every 100 ms.
        let server_rtcp = std::net::SocketAddr::from(([127, 0, 0, 1], server_rtp_port + 1));
        let mut sr_seen = false;
        let mut last_rtp_at_ms = 0u64;
        let mut pkt = [0u8; 2048];
        let mut ctl = [0u8; 2048];
        while started.elapsed().as_millis() < KEEPALIVE_MS as u128 {
            let rr = rtcp::build_receiver_report(0x0BAD_CAFE, &[]);
            rtcp_sock.send_to(&rr, server_rtcp).await.unwrap();
            let window = tokio::time::sleep(std::time::Duration::from_millis(100));
            tokio::pin!(window);
            loop {
                tokio::select! {
                    _ = &mut window => break,
                    r = rtp.recv(&mut pkt) => {
                        r.expect("recv rtp");
                        last_rtp_at_ms = started.elapsed().as_millis() as u64;
                    }
                    r = rtcp_sock.recv(&mut ctl) => {
                        let n = r.expect("recv rtcp");
                        for p in rtcp::parse_compound(&ctl[..n]) {
                            if let RtcpPacket::SenderReport { ssrc, .. } = p {
                                assert_eq!(ssrc, 0x1234_5678, "SR carries the server SSRC");
                                sr_seen = true;
                            }
                        }
                    }
                }
            }
        }
        assert!(sr_seen, "a sender report arrived on the client RTCP port");
        // Media still flowed well past the session timeout: only the RRs kept
        // the session alive (the control channel sent nothing after PLAY).
        assert!(
            last_rtp_at_ms > TIMEOUT_MS + 200,
            "RTP still flowing at {last_rtp_at_ms} ms, past the {TIMEOUT_MS} ms timeout"
        );

        // Silent phase: no RRs, no requests, control connection held open.
        tokio::time::sleep(std::time::Duration::from_millis(END_MS - KEEPALIVE_MS)).await;
        ctrl // keep the socket alive until the server side has asserted
    };

    let server = async move {
        let mut null = NullOut;
        let started = std::time::Instant::now();
        let mut i = 0u64;
        while started.elapsed().as_millis() < END_MS as u128 {
            let frame = Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(
                    access_unit((i % 8) as u8).into_boxed_slice(),
                )),
                timing: FrameTiming {
                    pts_ns: i * 33_000_000,
                    ..FrameTiming::default()
                },
                sequence: i,
                meta: Default::default(),
            };
            sink.process(PipelinePacket::DataFrame(frame), &mut null)
                .await
                .expect("stream frame");
            i += 1;
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        sink
    };

    let (_ctrl, sink) = tokio::join!(client, server);
    assert!(
        sink.sender_reports_sent() >= 1,
        "server sent sender reports"
    );
    assert_eq!(
        sink.client_count(),
        0,
        "the silent player was reaped by the session timeout"
    );
}

/// Drain the RTSP response header block from `buf`, reading more if needed;
/// leaves any trailing bytes (interleaved data) in `buf`.
async fn consume_response(sock: &mut tokio::net::TcpStream, buf: &mut Vec<u8>) -> String {
    let mut tmp = [0u8; 2048];
    loop {
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&buf[..pos]).to_string();
            buf.drain(..pos + 4);
            return head;
        }
        let n = sock.read(&mut tmp).await.expect("read");
        assert!(n > 0, "server closed control connection");
        buf.extend_from_slice(&tmp[..n]);
    }
}

#[tokio::test]
async fn rtsp_player_receives_interleaved_rtp_over_control() {
    // A player that SETUPs TCP-interleaved (RFC 2326 §10.12) receives the RTP as
    // `$`-framed binary on the control connection, no UDP port. Mirrors
    // `ffmpeg -rtsp_transport tcp`.
    const N: u8 = 8;

    let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind rtsp");
    let rtsp_addr = listener.local_addr().unwrap();
    let mut sink = RtspServerSink::from_listener(listener)
        .unwrap()
        .with_rtp(96, 0x1234_5678)
        .with_rtcp_sr_interval(std::time::Duration::from_millis(50));
    sink.configure_pipeline(&h264_caps()).expect("configure");

    let client = async move {
        let mut ctrl = tokio::net::TcpStream::connect(rtsp_addr)
            .await
            .expect("connect rtsp");
        let url = "rtsp://127.0.0.1/stream";
        let mut buf: Vec<u8> = Vec::new();

        ctrl.write_all(format!("OPTIONS {url} RTSP/1.0\r\nCSeq: 1\r\n\r\n").as_bytes())
            .await
            .unwrap();
        assert!(consume_response(&mut ctrl, &mut buf)
            .await
            .contains("200 OK"));

        ctrl.write_all(
            format!("DESCRIBE {url} RTSP/1.0\r\nCSeq: 2\r\nAccept: application/sdp\r\n\r\n")
                .as_bytes(),
        )
        .await
        .unwrap();
        consume_response(&mut ctrl, &mut buf).await;

        let setup = format!(
            "SETUP {url}/streamid=0 RTSP/1.0\r\nCSeq: 3\r\nTransport: RTP/AVP/TCP;unicast;interleaved=0-1\r\n\r\n"
        );
        ctrl.write_all(setup.as_bytes()).await.unwrap();
        let setup_resp = consume_response(&mut ctrl, &mut buf).await;
        assert!(
            setup_resp.contains("RTP/AVP/TCP"),
            "server negotiates interleaved: {setup_resp}"
        );
        assert!(
            setup_resp.contains("interleaved=0-1"),
            "server echoes the channels: {setup_resp}"
        );

        ctrl.write_all(
            format!("PLAY {url} RTSP/1.0\r\nCSeq: 4\r\nSession: 12345678\r\n\r\n").as_bytes(),
        )
        .await
        .unwrap();
        assert!(consume_response(&mut ctrl, &mut buf)
            .await
            .contains("200 OK"));

        // The client's RTCP (a receiver report) rides the control connection
        // `$`-framed on channel 1; the server must consume it as keepalive, not
        // trip over the binary while parsing requests.
        let rr = rtcp::build_receiver_report(0x0BAD_CAFE, &[]);
        let mut framed = vec![0x24u8, 1];
        framed.extend_from_slice(&(rr.len() as u16).to_be_bytes());
        framed.extend_from_slice(&rr);
        ctrl.write_all(&framed).await.unwrap();

        // Read `$`-framed RTP (channel 0) and RTCP (channel 1) off the control
        // connection; depayload the RTP, collect sender reports from the RTCP.
        let mut depay = RtpH264Depayloader::new();
        let mut tags = Vec::new();
        let mut sr_seen = false;
        let mut tmp = [0u8; 2048];
        while tags.len() < N as usize {
            while buf.len() < 4 {
                let n =
                    tokio::time::timeout(std::time::Duration::from_secs(5), ctrl.read(&mut tmp))
                        .await
                        .expect("interleaved data within 5s")
                        .expect("read control");
                assert!(n > 0, "server closed before N frames");
                buf.extend_from_slice(&tmp[..n]);
            }
            assert_eq!(buf[0], 0x24, "interleaved frame starts with $");
            let channel = buf[1];
            let len = u16::from_be_bytes([buf[2], buf[3]]) as usize;
            while buf.len() < 4 + len {
                let n = ctrl.read(&mut tmp).await.expect("read control");
                assert!(n > 0, "server closed mid-frame");
                buf.extend_from_slice(&tmp[..n]);
            }
            let payload: Vec<u8> = buf[4..4 + len].to_vec();
            buf.drain(..4 + len);
            match channel {
                0 => {
                    if let Some(au) = depay.depacketize(&payload) {
                        tags.push(au.data.get(5).copied().unwrap_or(0));
                    }
                }
                1 => {
                    for p in rtcp::parse_compound(&payload) {
                        if let RtcpPacket::SenderReport { ssrc, .. } = p {
                            assert_eq!(ssrc, 0x1234_5678, "SR carries the server SSRC");
                            sr_seen = true;
                        }
                    }
                }
                other => panic!("unexpected interleaved channel {other}"),
            }
        }
        assert!(sr_seen, "a sender report arrived on the RTCP channel");
        tags
    };

    let server = async move {
        let mut null = NullOut;
        for i in 0u8..(N * 3) {
            let au = access_unit(i % N);
            let frame = Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(au.into_boxed_slice())),
                timing: FrameTiming {
                    pts_ns: i as u64 * 33_000_000,
                    ..FrameTiming::default()
                },
                sequence: i as u64,
                meta: Default::default(),
            };
            match sink
                .process(PipelinePacket::DataFrame(frame), &mut null)
                .await
            {
                Ok(()) => {}
                Err(_) if sink.frames_sent() >= N as u64 => break,
                Err(e) => panic!("stream frame: {e:?}"),
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        sink.frames_sent()
    };

    let (tags, frames_sent) = tokio::join!(client, server);
    assert!(frames_sent >= N as u64, "server streamed frames after PLAY");
    let expected: Vec<u8> = (0..N).collect();
    assert_eq!(
        tags, expected,
        "player received every AU in order over the interleaved channel"
    );
}
