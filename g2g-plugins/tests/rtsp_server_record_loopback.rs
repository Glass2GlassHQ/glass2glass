//! RtspServerSrc end-to-end over loopback: a minimal RTSP *publisher* connects,
//! runs OPTIONS / ANNOUNCE / SETUP / RECORD against the server source, then
//! pushes RTP to the server's negotiated port. Proves the ingest (RECORD)
//! direction (RTSP control handshake + RTP/UDP receive + depayload) without an
//! external publisher.
#![cfg(feature = "rtsp-server")]

use core::future::Future;
use core::pin::Pin;
use std::net::TcpListener as StdTcpListener;

use g2g_core::runtime::SourceLoop;
use g2g_core::{Caps, Dim, G2gError, OutputSink, PipelinePacket, PushOutcome, Rate, VideoCodec};

use g2g_plugins::rtppay::RtpH264Packetizer;
use g2g_plugins::rtspserversrc::RtspServerSrc;

use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Captures the access units `RtspServerSrc` emits, keyed by their tag byte.
#[derive(Default)]
struct Capture {
    tags: Vec<u8>,
}
impl OutputSink for Capture {
    fn push<'a>(
        &'a mut self,
        p: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        if let PipelinePacket::DataFrame(frame) = &p {
            if let Some(slice) = frame.domain.as_system_slice() {
                // Annex-B payload [0,0,0,1][NAL][tag ..]; recover the tag byte.
                self.tags.push(slice.get(5).copied().unwrap_or(0));
            }
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

/// Pull `server_port=NNNN` out of a SETUP response Transport header.
fn parse_server_port(resp: &str) -> u16 {
    resp.split("server_port=")
        .nth(1)
        .and_then(|s| s.split(['-', ';', '\r']).next())
        .and_then(|s| s.trim().parse().ok())
        .expect("SETUP advertises a server_port")
}

/// Drive the RTSP control handshake (OPTIONS/ANNOUNCE/SETUP/RECORD) from a
/// publisher and return the control stream (kept open for the session), an RTP
/// socket already connected to the server's negotiated port, and that port.
async fn handshake_publisher(
    rtsp_addr: std::net::SocketAddr,
) -> (tokio::net::TcpStream, tokio::net::UdpSocket, u16) {
    let mut ctrl = tokio::net::TcpStream::connect(rtsp_addr)
        .await
        .expect("connect rtsp");
    let url = "rtsp://127.0.0.1/stream";

    ctrl.write_all(format!("OPTIONS {url} RTSP/1.0\r\nCSeq: 1\r\n\r\n").as_bytes())
        .await
        .unwrap();
    assert!(read_response(&mut ctrl).await.contains("200 OK"));

    let sdp = "v=0\r\no=- 0 0 IN IP4 0.0.0.0\r\ns=g2g\r\nm=video 0 RTP/AVP 96\r\na=rtpmap:96 H264/90000\r\n";
    ctrl.write_all(
        format!(
            "ANNOUNCE {url} RTSP/1.0\r\nCSeq: 2\r\nContent-Type: application/sdp\r\nContent-Length: {}\r\n\r\n{sdp}",
            sdp.len()
        )
        .as_bytes(),
    )
    .await
    .unwrap();
    assert!(read_response(&mut ctrl).await.contains("200 OK"));

    // We send RTP from this socket; client_port is advertised for symmetry.
    let rtp = tokio::net::UdpSocket::bind("127.0.0.1:0")
        .await
        .expect("bind publisher rtp");
    let client_rtp_port = rtp.local_addr().unwrap().port();
    ctrl.write_all(
        format!(
            "SETUP {url}/streamid=0 RTSP/1.0\r\nCSeq: 3\r\nTransport: RTP/AVP;unicast;client_port={client_rtp_port}-{};mode=record\r\n\r\n",
            client_rtp_port + 1
        )
        .as_bytes(),
    )
    .await
    .unwrap();
    let setup_resp = read_response(&mut ctrl).await;
    let server_port = parse_server_port(&setup_resp);

    ctrl.write_all(
        format!("RECORD {url} RTSP/1.0\r\nCSeq: 4\r\nSession: 12345678\r\n\r\n").as_bytes(),
    )
    .await
    .unwrap();
    assert!(read_response(&mut ctrl).await.contains("200 OK"));

    rtp.connect(("127.0.0.1", server_port))
        .await
        .expect("connect rtp dest");
    (ctrl, rtp, server_port)
}

/// Run the server source to completion against a `publisher` coroutine, both
/// sharing the ephemeral RTSP listener. Returns (emitted-count, received tags).
async fn run_ingest<F, Fut>(src: RtspServerSrc, publisher: F) -> (u64, Vec<u8>)
where
    F: FnOnce(std::net::SocketAddr) -> Fut,
    Fut: Future<Output = ()>,
{
    let rtsp_addr = src
        .local_port()
        .map(|p| ([127, 0, 0, 1], p).into())
        .expect("bound");
    let server = async move {
        let mut src = src;
        src.configure_pipeline(&h264_caps()).expect("configure");
        let mut cap = Capture::default();
        let n = tokio::time::timeout(std::time::Duration::from_secs(5), src.run(&mut cap))
            .await
            .expect("server completes within 5s")
            .expect("server runs");
        (n, cap.tags)
    };
    let (_, out) = tokio::join!(publisher(rtsp_addr), server);
    out
}

#[tokio::test]
async fn rtsp_publisher_handshakes_then_pushes_rtp() {
    const N: u8 = 8;
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind rtsp");
    let src = RtspServerSrc::from_listener(listener)
        .unwrap()
        .with_rtp(96, 0x1234_5678)
        .with_video_size(320, 240)
        .with_frame_limit(N as u64);

    let publisher = |rtsp_addr| async move {
        let (ctrl, rtp, _) = handshake_publisher(rtsp_addr).await;
        // Push N access units as RTP, in order, to the server's negotiated port.
        let mut pktz = RtpH264Packetizer::new(96, 0x1234_5678);
        for i in 0u8..N {
            for pkt in pktz.packetize(&access_unit(i), i as u32 * 3000) {
                rtp.send(&pkt).await.expect("send rtp");
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        drop(ctrl);
    };

    let (n, tags) = run_ingest(src, publisher).await;
    assert_eq!(n, N as u64, "server emitted every access unit then EOS");
    let expected: Vec<u8> = (0..N).collect();
    assert_eq!(
        tags, expected,
        "server received and depayloaded every AU in order"
    );
}

/// The ingest jitter buffer (M520) reorders RTP that arrives out of sequence:
/// each single-packet access unit is sent with its neighbour swapped, yet the
/// server must still emit the access units in RTP-sequence order.
#[tokio::test]
async fn rtsp_ingest_reorders_out_of_order_rtp() {
    const N: u8 = 8;
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind rtsp");
    let src = RtspServerSrc::from_listener(listener)
        .unwrap()
        .with_rtp(96, 0x1234_5678)
        .with_video_size(320, 240)
        // A generous hold so a swapped pair is always reordered, not flushed.
        .with_jitter(200, 64)
        .with_frame_limit(N as u64);

    let publisher = |rtsp_addr| async move {
        let (ctrl, rtp, _) = handshake_publisher(rtsp_addr).await;
        // Each AU is one RTP packet (single small NAL); build them in order,
        // then send the first in order (it baselines the release sequence) and
        // swap each following adjacent pair, so the socket sees 0, 2,1, 4,3, ...
        // The jitter buffer must restore 0,1,2,3,...
        let mut pktz = RtpH264Packetizer::new(96, 0x1234_5678);
        let mut pkts: Vec<Vec<u8>> = Vec::new();
        for i in 0u8..N {
            let mut group = pktz.packetize(&access_unit(i), i as u32 * 3000);
            assert_eq!(group.len(), 1, "each tiny AU packetizes to one RTP packet");
            pkts.append(&mut group);
        }
        rtp.send(&pkts[0]).await.expect("send rtp");
        let mut i = 1;
        while i < pkts.len() {
            if i + 1 < pkts.len() {
                rtp.send(&pkts[i + 1]).await.expect("send rtp");
                rtp.send(&pkts[i]).await.expect("send rtp");
                i += 2;
            } else {
                rtp.send(&pkts[i]).await.expect("send rtp");
                i += 1;
            }
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        drop(ctrl);
    };

    let (n, tags) = run_ingest(src, publisher).await;
    assert_eq!(n, N as u64, "server emitted every access unit then EOS");
    let expected: Vec<u8> = (0..N).collect();
    assert_eq!(
        tags, expected,
        "jitter buffer restored RTP-sequence order despite swapped arrival"
    );
}
