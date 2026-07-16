//! M532: RtspServerSrc TCP-interleaved ingest (RFC 2326 §10.12) over loopback. A
//! minimal RTSP *publisher* connects, runs OPTIONS / ANNOUNCE / SETUP (asking for
//! `RTP/AVP/TCP;interleaved=0-1`) / RECORD, then pushes RTP as `$`-framed binary
//! on the *same* TCP control connection, the transport `ffmpeg -rtsp_transport
//! tcp` uses. Proves the interleaved receive + depayload path (no UDP socket)
//! without an external publisher.
#![cfg(feature = "rtsp-server")]

use core::future::Future;
use core::pin::Pin;
use std::net::TcpListener as StdTcpListener;

use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, Dim, G2gError, MemoryDomain, OutputSink, PipelinePacket, PushOutcome, Rate, VideoCodec,
};

use g2g_plugins::rtppay::RtpH264Packetizer;
use g2g_plugins::rtspserversrc::RtspServerSrc;

use tokio::io::{AsyncReadExt, AsyncWriteExt};

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
            if let MemoryDomain::System(slice) = &frame.domain {
                self.tags.push(slice.as_slice().get(5).copied().unwrap_or(0));
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

/// Frame an RTP packet as interleaved binary: `$`, channel, 2-byte big-endian
/// length, then the packet (RFC 2326 §10.12).
fn interleaved(channel: u8, rtp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + rtp.len());
    out.push(0x24);
    out.push(channel);
    out.extend_from_slice(&(rtp.len() as u16).to_be_bytes());
    out.extend_from_slice(rtp);
    out
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

/// Drive the interleaved control handshake and return the control stream (which
/// also carries the RTP).
async fn handshake_interleaved(rtsp_addr: std::net::SocketAddr) -> tokio::net::TcpStream {
    let mut ctrl = tokio::net::TcpStream::connect(rtsp_addr).await.expect("connect rtsp");
    let url = "rtsp://127.0.0.1/stream";

    ctrl.write_all(format!("OPTIONS {url} RTSP/1.0\r\nCSeq: 1\r\n\r\n").as_bytes()).await.unwrap();
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

    ctrl.write_all(
        format!(
            "SETUP {url}/streamid=0 RTSP/1.0\r\nCSeq: 3\r\nTransport: RTP/AVP/TCP;unicast;interleaved=0-1;mode=record\r\n\r\n"
        )
        .as_bytes(),
    )
    .await
    .unwrap();
    let setup = read_response(&mut ctrl).await;
    assert!(setup.contains("RTP/AVP/TCP"), "server negotiates interleaved: {setup}");
    assert!(setup.contains("interleaved=0-1"), "server echoes the channels: {setup}");

    ctrl.write_all(
        format!("RECORD {url} RTSP/1.0\r\nCSeq: 4\r\nSession: 12345678\r\n\r\n").as_bytes(),
    )
    .await
    .unwrap();
    assert!(read_response(&mut ctrl).await.contains("200 OK"));
    ctrl
}

async fn run_ingest<F, Fut>(src: RtspServerSrc, publisher: F) -> (u64, Vec<u8>)
where
    F: FnOnce(std::net::SocketAddr) -> Fut,
    Fut: Future<Output = ()>,
{
    let rtsp_addr = src.local_port().map(|p| ([127, 0, 0, 1], p).into()).expect("bound");
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
async fn rtsp_interleaved_publisher_pushes_rtp_over_control_channel() {
    const N: u8 = 8;
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind rtsp");
    let src = RtspServerSrc::from_listener(listener)
        .unwrap()
        .with_rtp(96, 0x1234_5678)
        .with_video_size(320, 240)
        .with_frame_limit(N as u64);

    let publisher = |rtsp_addr| async move {
        let mut ctrl = handshake_interleaved(rtsp_addr).await;
        // Push N access units as `$`-framed RTP on channel 0 (the RTP channel).
        let mut pktz = RtpH264Packetizer::new(96, 0x1234_5678);
        for i in 0u8..N {
            for pkt in pktz.packetize(&access_unit(i), i as u32 * 3000) {
                ctrl.write_all(&interleaved(0, &pkt)).await.expect("send interleaved rtp");
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        drop(ctrl);
    };

    let (n, tags) = run_ingest(src, publisher).await;
    assert_eq!(n, N as u64, "server emitted every access unit then EOS");
    let expected: Vec<u8> = (0..N).collect();
    assert_eq!(tags, expected, "interleaved RTP depayloaded in order over the control channel");
}

/// RTCP interleaved on the sibling channel (1) is skipped, not misread as RTP:
/// the publisher interleaves a dummy channel-1 frame between RTP frames and the
/// server still emits exactly the RTP access units.
#[tokio::test]
async fn rtsp_interleaved_ignores_the_rtcp_channel() {
    const N: u8 = 6;
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind rtsp");
    let src = RtspServerSrc::from_listener(listener)
        .unwrap()
        .with_rtp(96, 0x1234_5678)
        .with_frame_limit(N as u64);

    let publisher = |rtsp_addr| async move {
        let mut ctrl = handshake_interleaved(rtsp_addr).await;
        let mut pktz = RtpH264Packetizer::new(96, 0x1234_5678);
        for i in 0u8..N {
            // A dummy RTCP-channel (1) frame interleaved before each RTP frame.
            ctrl.write_all(&interleaved(1, &[0x80, 0xC9, 0, 1, 0, 0, 0, 0]))
                .await
                .expect("send rtcp");
            for pkt in pktz.packetize(&access_unit(i), i as u32 * 3000) {
                ctrl.write_all(&interleaved(0, &pkt)).await.expect("send rtp");
            }
            tokio::time::sleep(std::time::Duration::from_millis(4)).await;
        }
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        drop(ctrl);
    };

    let (n, tags) = run_ingest(src, publisher).await;
    assert_eq!(n, N as u64, "only the RTP channel produced access units");
    assert_eq!(tags, (0..N).collect::<Vec<u8>>(), "RTCP-channel frames were skipped");
}
