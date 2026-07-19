//! RtspServerSink serving multiple players: two RTSP players connect (one at
//! startup, one mid-stream) and each receives its own ordered RTP session of the
//! same H.264. Proves the multi-client serving path (per-client handshake +
//! per-client RTP session + broadcast) without an external client.
#![cfg(feature = "rtsp-server")]

use core::future::Future;
use core::pin::Pin;
use std::net::{SocketAddr, TcpListener as StdTcpListener};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket,
    PushOutcome, Rate, VideoCodec,
};

use g2g_plugins::rtpdepay::RtpH264Depayloader;
use g2g_plugins::rtspserversink::RtspServerSink;

use tokio::io::{AsyncReadExt, AsyncWriteExt};

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

/// One small Annex-B IDR access unit, tagged with the frame index at byte 5.
fn access_unit(tag: u8) -> Vec<u8> {
    vec![0u8, 0, 0, 1, 0x65, tag, 0xAB, 0xCD]
}

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

/// Connect a player, run OPTIONS/DESCRIBE/SETUP/PLAY, then receive RTP and
/// return the first `want` access-unit tags it sees (in receive order).
async fn run_player(rtsp_addr: SocketAddr, want: usize) -> Vec<u8> {
    let rtp = tokio::net::UdpSocket::bind("127.0.0.1:0")
        .await
        .expect("bind player rtp");
    let client_rtp_port = rtp.local_addr().unwrap().port();
    let mut ctrl = tokio::net::TcpStream::connect(rtsp_addr)
        .await
        .expect("connect rtsp");
    let url = "rtsp://127.0.0.1/stream";

    ctrl.write_all(format!("OPTIONS {url} RTSP/1.0\r\nCSeq: 1\r\n\r\n").as_bytes())
        .await
        .unwrap();
    assert!(read_response(&mut ctrl).await.contains("200 OK"));
    ctrl.write_all(
        format!("DESCRIBE {url} RTSP/1.0\r\nCSeq: 2\r\nAccept: application/sdp\r\n\r\n").as_bytes(),
    )
    .await
    .unwrap();
    assert!(read_response(&mut ctrl).await.contains("m=video"));
    ctrl.write_all(
        format!(
            "SETUP {url}/streamid=0 RTSP/1.0\r\nCSeq: 3\r\nTransport: RTP/AVP;unicast;client_port={client_rtp_port}-{}\r\n\r\n",
            client_rtp_port + 1
        )
        .as_bytes(),
    )
    .await
    .unwrap();
    assert!(read_response(&mut ctrl).await.contains("server_port="));
    ctrl.write_all(
        format!("PLAY {url} RTSP/1.0\r\nCSeq: 4\r\nSession: 12345678\r\n\r\n").as_bytes(),
    )
    .await
    .unwrap();
    assert!(read_response(&mut ctrl).await.contains("200 OK"));

    let mut depay = RtpH264Depayloader::new();
    let mut tags = Vec::new();
    let mut pkt = [0u8; 2048];
    while tags.len() < want {
        let recv =
            tokio::time::timeout(std::time::Duration::from_secs(8), rtp.recv(&mut pkt)).await;
        let n = recv.expect("rtp arrives within 8s").expect("recv rtp");
        if let Some(au) = depay.depacketize(&pkt[..n]) {
            tags.push(au.data.get(5).copied().unwrap_or(0));
        }
    }
    // Keep the control connection open while the harness checks results.
    let _ = ctrl;
    tags
}

/// Every received run must be strictly consecutive: a clean, ordered per-client
/// RTP session (each client gets its own sequence space, no interleave).
fn assert_contiguous(tags: &[u8], who: &str) {
    assert!(
        tags.len() >= 10,
        "{who} received {} frames (want >= 10)",
        tags.len()
    );
    for w in tags.windows(2) {
        assert_eq!(
            w[1],
            w[0] + 1,
            "{who} frames must be consecutive, saw {:?}",
            tags
        );
    }
}

#[tokio::test]
async fn two_players_each_receive_an_ordered_stream() {
    const COUNT: u8 = 60;

    let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind rtsp");
    let rtsp_addr = listener.local_addr().unwrap();

    // Player 1 connects immediately (it bootstraps the server); player 2 joins
    // mid-stream and must still get a clean ordered session.
    let p1 = run_player(rtsp_addr, 12);
    let p2 = async move {
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;
        run_player(rtsp_addr, 10).await
    };

    // The server streams COUNT tagged frames; the first blocks until a player
    // PLAYs, the rest also accept + advance new players and broadcast.
    let server = async move {
        let mut sink = RtspServerSink::from_listener(listener)
            .unwrap()
            .with_rtp(96, 0x1111_1111);
        sink.configure_pipeline(&h264_caps()).expect("configure");
        let mut null = NullOut;
        for i in 0u8..COUNT {
            let au = access_unit(i);
            let frame = Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(au.into_boxed_slice())),
                timing: FrameTiming {
                    pts_ns: i as u64 * 33_000_000,
                    ..FrameTiming::default()
                },
                sequence: i as u64,
                meta: Default::default(),
            };
            sink.process(PipelinePacket::DataFrame(frame), &mut null)
                .await
                .expect("process");
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        sink.frames_sent()
    };

    let (tags1, tags2, frames_sent) = tokio::join!(p1, p2, server);

    assert_eq!(frames_sent, COUNT as u64, "server streamed every frame");
    assert_contiguous(&tags1, "player 1");
    assert_contiguous(&tags2, "player 2");
    // Player 1 bootstraps the server, so it sees the stream from the very start.
    assert_eq!(
        tags1[0], 0,
        "the bootstrapping player receives from the first frame"
    );
    // Player 2 joined later, so its first frame is strictly after player 1's.
    assert!(
        tags2[0] > 0,
        "the late player joined mid-stream (saw {:?})",
        tags2
    );
}
