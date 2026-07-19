#![cfg(feature = "udp-egress")]
//! M47: `UdpSink` drives the M46 `RtpH264Packetizer` and sends the RTP packets
//! over UDP. The test binds a loopback receiver, runs access units through the
//! sink, and parses the datagrams back: the RTP timestamp tracks `pts_ns` at
//! 90 kHz, sequence numbers are contiguous, the marker bit lands on each access
//! unit's last packet, and a fragmented NAL reassembles byte-exactly. Loopback
//! UDP is used because the live RTSP/RTP port (554) is sandbox-blocked.

use std::net::SocketAddr;
use std::time::Duration;

use g2g_core::element::AsyncElement;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{Caps, Dim, G2gError, Rate, VideoCodec};
use g2g_plugins::rtppay::RtpH264Packetizer;
use g2g_plugins::udpsink::UdpSink;

const SSRC: u32 = 0xCAFE_F00D;
const PAYLOAD_TYPE: u8 = 96;
const MAX_PAYLOAD: usize = 16;

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30 << 16),
    }
}

struct NullOut;
impl g2g_core::OutputSink for NullOut {
    fn push<'a>(
        &'a mut self,
        _packet: PipelinePacket,
    ) -> g2g_core::element::BoxFuture<'a, Result<g2g_core::element::PushOutcome, G2gError>> {
        Box::pin(async { Ok(g2g_core::element::PushOutcome::Accepted) })
    }
}

fn annexb(nals: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    for nal in nals {
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(nal);
    }
    out
}

fn au_frame(bytes: Vec<u8>, pts_ns: u64, sequence: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            duration_ns: 33_333_333,
            ..FrameTiming::default()
        },
        sequence,
        meta: Default::default(),
    }
}

fn seq(pkt: &[u8]) -> u16 {
    u16::from_be_bytes([pkt[2], pkt[3]])
}
fn timestamp(pkt: &[u8]) -> u32 {
    u32::from_be_bytes([pkt[4], pkt[5], pkt[6], pkt[7]])
}
fn ssrc(pkt: &[u8]) -> u32 {
    u32::from_be_bytes([pkt[8], pkt[9], pkt[10], pkt[11]])
}
fn marker(pkt: &[u8]) -> bool {
    pkt[1] & 0x80 != 0
}

async fn recv_n(sock: &tokio::net::UdpSocket, n: usize) -> Vec<Vec<u8>> {
    let mut out = Vec::with_capacity(n);
    let mut buf = [0u8; 2048];
    for _ in 0..n {
        let len = tokio::time::timeout(Duration::from_secs(2), sock.recv(&mut buf))
            .await
            .expect("recv timed out: a loopback datagram was lost")
            .expect("recv");
        out.push(buf[..len].to_vec());
    }
    out
}

#[tokio::test]
async fn sink_sends_rtp_matching_packetizer_with_pts_derived_timestamp() {
    let receiver = tokio::net::UdpSocket::bind("127.0.0.1:0")
        .await
        .expect("bind receiver");
    let dest: SocketAddr = receiver.local_addr().expect("local addr");

    let mut sink = UdpSink::new(dest)
        .with_rtp(PAYLOAD_TYPE, SSRC)
        .with_max_payload(MAX_PAYLOAD);
    sink.configure_pipeline(&h264_caps()).expect("configure");

    // AU1 (pts 0): two small NALs, each a single-NAL packet.
    let sps: &[u8] = &[0x67, 0x42, 0xC0, 0x1E];
    let pps: &[u8] = &[0x68, 0xCE, 0x3C, 0x80];
    let au1 = annexb(&[sps, pps]);
    // AU2 (pts 1/30 s): one oversized IDR NAL that must fragment into FU-A.
    let idr_body: Vec<u8> = (0..40u8).collect();
    let idr = idr_nal(&idr_body);
    let au2 = annexb(&[&idr]);

    let pts2 = 33_333_333u64;

    // Reference packets the packetizer would produce for the same stream.
    let mut reference = RtpH264Packetizer::new(PAYLOAD_TYPE, SSRC).with_max_payload(MAX_PAYLOAD);
    let mut expected: Vec<Vec<u8>> = Vec::new();
    expected.extend(reference.packetize(&au1, 0));
    expected.extend(reference.packetize(&au2, 2999)); // 33_333_333 ns at 90 kHz
    let expected_count = expected.len();
    assert_eq!(expected_count, 5, "2 single-NAL + 3 FU-A fragments");

    let mut out = NullOut;
    sink.process(
        PipelinePacket::DataFrame(au_frame(au1.clone(), 0, 0)),
        &mut out,
    )
    .await
    .expect("send AU1");
    let first = recv_n(&receiver, 2).await;
    sink.process(
        PipelinePacket::DataFrame(au_frame(au2.clone(), pts2, 1)),
        &mut out,
    )
    .await
    .expect("send AU2");
    let second = recv_n(&receiver, 3).await;
    sink.process(PipelinePacket::Eos, &mut out)
        .await
        .expect("eos");

    let mut received: Vec<Vec<u8>> = first.into_iter().chain(second).collect();
    received.sort_by_key(|p| seq(p)); // loopback is in-order in practice; be robust anyway

    assert_eq!(received.len(), expected_count);
    assert_eq!(
        received, expected,
        "datagrams on the wire must match the packetizer byte-for-byte"
    );

    // Sequence is contiguous from zero, PT/SSRC/version consistent.
    for (i, pkt) in received.iter().enumerate() {
        assert_eq!(seq(pkt), i as u16, "contiguous sequence");
        assert_eq!(pkt[0] >> 6, 2, "RTP version 2");
        assert_eq!(pkt[1] & 0x7F, PAYLOAD_TYPE, "payload type");
        assert_eq!(ssrc(pkt), SSRC, "ssrc");
    }

    // RTP timestamp is the 90 kHz image of pts (independent of the reference).
    assert_eq!(timestamp(&received[0]), 0, "AU1 pts 0 -> ts 0");
    assert_eq!(timestamp(&received[2]), 2999, "AU2 pts 1/30 s -> ts 2999");
    // One timestamp per access unit.
    assert_eq!(timestamp(&received[0]), timestamp(&received[1]));
    assert_eq!(timestamp(&received[2]), timestamp(&received[3]));
    assert_eq!(timestamp(&received[3]), timestamp(&received[4]));

    // Marker only on each AU's last packet.
    assert!(!marker(&received[0]));
    assert!(marker(&received[1]), "AU1 ends at packet 1");
    assert!(!marker(&received[2]));
    assert!(!marker(&received[3]));
    assert!(marker(&received[4]), "AU2 ends at packet 4");

    // FU-A fragments (AU2) reassemble to the original IDR NAL byte-exactly.
    let mut reassembled = Vec::new();
    let mut nal_type = 0u8;
    let mut f_nri = 0u8;
    for pkt in &received[2..] {
        let payload = &pkt[12..];
        assert_eq!(payload[0] & 0x1F, 28, "FU-A indicator type");
        f_nri = payload[0] & 0xE0;
        nal_type = payload[1] & 0x1F;
        reassembled.extend_from_slice(&payload[2..]);
    }
    assert_eq!(
        reassembled, idr_body,
        "FU-A body reassembles the IDR payload"
    );
    assert_eq!(
        (f_nri | nal_type),
        0x65,
        "reconstructed NAL header is the IDR header"
    );

    assert_eq!(sink.frames_sent(), 2);
    assert_eq!(sink.packets_sent(), 5);
    assert_eq!(
        sink.bytes_sent() as usize,
        expected.iter().map(|p| p.len()).sum::<usize>()
    );
    assert!(sink.eos_seen(), "EOS reaches the sink");
}

fn idr_nal(body: &[u8]) -> Vec<u8> {
    let mut idr = vec![0x65u8]; // F=0, NRI=3, type=5 (IDR)
    idr.extend_from_slice(body);
    idr
}

#[tokio::test]
async fn sink_emits_rtcp_sender_report_when_enabled() {
    use g2g_plugins::rtcp::{self, RtcpPacket};

    let receiver = tokio::net::UdpSocket::bind("127.0.0.1:0")
        .await
        .expect("bind receiver");
    let dest: SocketAddr = receiver.local_addr().expect("local addr");

    // interval 0: a sender report is due on the first frame, after its media.
    let mut sink = UdpSink::new(dest)
        .with_rtp(PAYLOAD_TYPE, SSRC)
        .with_max_payload(MAX_PAYLOAD)
        .with_rtcp_sender_reports(0);
    sink.configure_pipeline(&h264_caps()).expect("configure");

    // One AU of two small NALs -> two single-NAL RTP packets (4-byte payloads).
    let sps: &[u8] = &[0x67, 0x42, 0xC0, 0x1E];
    let pps: &[u8] = &[0x68, 0xCE, 0x3C, 0x80];
    let au1 = annexb(&[sps, pps]);

    let mut out = NullOut;
    sink.process(PipelinePacket::DataFrame(au_frame(au1, 0, 0)), &mut out)
        .await
        .expect("send AU1");

    // Two media datagrams plus one RTCP sender report.
    let dgrams = recv_n(&receiver, 3).await;
    let sr = dgrams
        .iter()
        .find(|d| {
            matches!(
                rtcp::parse_compound(d).first(),
                Some(RtcpPacket::SenderReport { .. })
            )
        })
        .expect("an RTCP sender report arrived on the muxed socket");

    let Some(RtcpPacket::SenderReport {
        ssrc: sr_ssrc,
        ntp,
        rtp_ts,
        ..
    }) = rtcp::parse_compound(sr).into_iter().next()
    else {
        panic!("first compound packet is a sender report");
    };
    assert_eq!(sr_ssrc, SSRC, "SR carries the media SSRC");
    assert_eq!(rtp_ts, 0, "SR reports the last media RTP timestamp (pts 0)");
    assert_ne!(ntp, 0, "SR carries an NTP wall-clock timestamp");

    // packet_count / octet_count sit at fixed SR offsets (after header+ssrc+ntp+ts).
    let packet_count = u32::from_be_bytes(sr[20..24].try_into().unwrap());
    let octet_count = u32::from_be_bytes(sr[24..28].try_into().unwrap());
    assert_eq!(packet_count, 2, "two media packets counted");
    assert_eq!(octet_count, 8, "two 4-byte NAL payloads = 8 octets");

    assert_eq!(sink.sender_reports_sent(), 1);
}
