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
    let mut sink = RtspServerSink::from_listener(listener).unwrap().with_rtp(96, 0x1234_5678);
    sink.configure_pipeline(&h264_caps()).expect("configure");

    // Client RTP socket; its port is what we put in the SETUP Transport header.
    let rtp = tokio::net::UdpSocket::bind("127.0.0.1:0").await.expect("bind client rtp");
    let client_rtp_port = rtp.local_addr().unwrap().port();

    // The player: connect, run the handshake, then receive RTP and depayload.
    let client = async move {
        let mut ctrl = tokio::net::TcpStream::connect(rtsp_addr).await.expect("connect rtsp");
        let url = "rtsp://127.0.0.1/stream";

        ctrl.write_all(format!("OPTIONS {url} RTSP/1.0\r\nCSeq: 1\r\n\r\n").as_bytes()).await.unwrap();
        assert!(read_response(&mut ctrl).await.contains("200 OK"));

        ctrl.write_all(format!("DESCRIBE {url} RTSP/1.0\r\nCSeq: 2\r\nAccept: application/sdp\r\n\r\n").as_bytes()).await.unwrap();
        assert!(read_response(&mut ctrl).await.contains("m=video 0 RTP/AVP 96"));

        let setup = format!(
            "SETUP {url}/streamid=0 RTSP/1.0\r\nCSeq: 3\r\nTransport: RTP/AVP;unicast;client_port={client_rtp_port}-{}\r\n\r\n",
            client_rtp_port + 1,
        );
        ctrl.write_all(setup.as_bytes()).await.unwrap();
        let setup_resp = read_response(&mut ctrl).await;
        assert!(setup_resp.contains("Session:"), "SETUP assigns a session");
        assert!(setup_resp.contains("server_port="));

        ctrl.write_all(format!("PLAY {url} RTSP/1.0\r\nCSeq: 4\r\nSession: 12345678\r\n\r\n").as_bytes()).await.unwrap();
        assert!(read_response(&mut ctrl).await.contains("200 OK"));

        // Now receive the RTP stream and recover the access units.
        let mut depay = RtpH264Depayloader::new();
        let mut tags = Vec::new();
        let mut pkt = [0u8; 2048];
        while tags.len() < N as usize {
            let recv = tokio::time::timeout(std::time::Duration::from_secs(5), rtp.recv(&mut pkt)).await;
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
                timing: FrameTiming { pts_ns: i as u64 * 33_000_000, ..FrameTiming::default() },
                sequence: i as u64,
                meta: Default::default(),
            };
            match sink.process(PipelinePacket::DataFrame(frame), &mut null).await {
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
    assert_eq!(tags, expected, "player received and depayloaded every AU in order");
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
    let mut sink = RtspServerSink::from_listener(listener).unwrap().with_rtp(96, 0x1234_5678);
    sink.configure_pipeline(&h264_caps()).expect("configure");

    let client = async move {
        let mut ctrl = tokio::net::TcpStream::connect(rtsp_addr).await.expect("connect rtsp");
        let url = "rtsp://127.0.0.1/stream";
        let mut buf: Vec<u8> = Vec::new();

        ctrl.write_all(format!("OPTIONS {url} RTSP/1.0\r\nCSeq: 1\r\n\r\n").as_bytes()).await.unwrap();
        assert!(consume_response(&mut ctrl, &mut buf).await.contains("200 OK"));

        ctrl.write_all(format!("DESCRIBE {url} RTSP/1.0\r\nCSeq: 2\r\nAccept: application/sdp\r\n\r\n").as_bytes()).await.unwrap();
        consume_response(&mut ctrl, &mut buf).await;

        let setup = format!(
            "SETUP {url}/streamid=0 RTSP/1.0\r\nCSeq: 3\r\nTransport: RTP/AVP/TCP;unicast;interleaved=0-1\r\n\r\n"
        );
        ctrl.write_all(setup.as_bytes()).await.unwrap();
        let setup_resp = consume_response(&mut ctrl, &mut buf).await;
        assert!(setup_resp.contains("RTP/AVP/TCP"), "server negotiates interleaved: {setup_resp}");
        assert!(setup_resp.contains("interleaved=0-1"), "server echoes the channels: {setup_resp}");

        ctrl.write_all(format!("PLAY {url} RTSP/1.0\r\nCSeq: 4\r\nSession: 12345678\r\n\r\n").as_bytes()).await.unwrap();
        assert!(consume_response(&mut ctrl, &mut buf).await.contains("200 OK"));

        // Read `$`-framed RTP off the control connection and depayload.
        let mut depay = RtpH264Depayloader::new();
        let mut tags = Vec::new();
        let mut tmp = [0u8; 2048];
        while tags.len() < N as usize {
            while buf.len() < 4 {
                let n = tokio::time::timeout(std::time::Duration::from_secs(5), ctrl.read(&mut tmp))
                    .await
                    .expect("interleaved data within 5s")
                    .expect("read control");
                assert!(n > 0, "server closed before N frames");
                buf.extend_from_slice(&tmp[..n]);
            }
            assert_eq!(buf[0], 0x24, "interleaved frame starts with $");
            assert_eq!(buf[1], 0, "RTP rides the negotiated channel 0");
            let len = u16::from_be_bytes([buf[2], buf[3]]) as usize;
            while buf.len() < 4 + len {
                let n = ctrl.read(&mut tmp).await.expect("read control");
                assert!(n > 0, "server closed mid-frame");
                buf.extend_from_slice(&tmp[..n]);
            }
            let rtp: Vec<u8> = buf[4..4 + len].to_vec();
            buf.drain(..4 + len);
            if let Some(au) = depay.depacketize(&rtp) {
                tags.push(au.data.get(5).copied().unwrap_or(0));
            }
        }
        tags
    };

    let server = async move {
        let mut null = NullOut;
        for i in 0u8..(N * 3) {
            let au = access_unit(i % N);
            let frame = Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(au.into_boxed_slice())),
                timing: FrameTiming { pts_ns: i as u64 * 33_000_000, ..FrameTiming::default() },
                sequence: i as u64,
                meta: Default::default(),
            };
            match sink.process(PipelinePacket::DataFrame(frame), &mut null).await {
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
    assert_eq!(tags, expected, "player received every AU in order over the interleaved channel");
}
