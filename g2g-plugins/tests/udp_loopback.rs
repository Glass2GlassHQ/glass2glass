//! Loopback integration test for the UDP RTP receive path: a sender packetizes
//! synthetic H.264 access units with `RtpH264Packetizer` and sends them over a
//! UDP socket to `UdpSrc`, which depayloads them back to Annex-B and pushes
//! them into a `FakeSink`. Exercises single-NAL, STAP-style multi-NAL, and FU-A
//! fragmentation end-to-end over real localhost UDP.

#![cfg(all(feature = "udp-ingress", feature = "udp-egress"))]

use std::net::UdpSocket as StdUdpSocket;

use g2g_core::runtime::{run_simple_pipeline, LatencyProfile};
use g2g_core::PipelineClock;
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::rtppay::RtpH264Packetizer;
use g2g_plugins::udpsrc::UdpSrc;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// One synthetic access unit: SPS, PPS, then an oversized IDR that forces FU-A
/// fragmentation under the small MTU below. Content is arbitrary; the receive
/// path only reassembles bytes.
fn access_unit() -> Vec<u8> {
    let mut au = Vec::new();
    for nal in [vec![0x67u8, 0x42, 0x00, 0x1f], vec![0x68u8, 0xce, 0x38, 0x80]] {
        au.extend_from_slice(&[0, 0, 0, 1]);
        au.extend_from_slice(&nal);
    }
    let mut idr = vec![0x65u8];
    idr.extend_from_slice(&(0..200u8).collect::<Vec<_>>());
    au.extend_from_slice(&[0, 0, 0, 1]);
    au.extend_from_slice(&idr);
    au
}

#[tokio::test]
async fn udpsrc_receives_and_depayloads_rtp_over_loopback() {
    // Receiver: bind an ephemeral port and hand the socket to UdpSrc so the
    // sender knows exactly where to send (no port race).
    let recv = StdUdpSocket::bind("127.0.0.1:0").expect("bind receiver");
    let recv_addr = recv.local_addr().unwrap();

    const TARGET: u64 = 20;
    let mut src = UdpSrc::from_socket(recv).unwrap().with_frame_limit(TARGET);
    let mut sink = FakeSink::new();
    let clock = ZeroClock;

    let au = access_unit();
    // Sender: a separate socket, FU-A fragmenting via a small max payload.
    let sender = StdUdpSocket::bind("127.0.0.1:0").expect("bind sender");
    sender.connect(recv_addr).unwrap();
    let send_task = tokio::task::spawn_blocking(move || {
        let mut pkt = RtpH264Packetizer::new(96, 0x1234).with_max_payload(64);
        // Send more access units than the receiver consumes; it stops at the
        // frame limit and any surplus is harmlessly dropped. A short gap keeps
        // the receiver's recv loop ahead and tolerates startup latency.
        for i in 0..(TARGET as u32 * 4) {
            let ts = i.wrapping_mul(3000);
            for rtp in pkt.packetize(&au, ts) {
                let _ = sender.send(&rtp);
            }
            std::thread::sleep(std::time::Duration::from_millis(3));
        }
    });

    let stats = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        run_simple_pipeline(&mut src, &mut sink, &clock, LatencyProfile::Live.link_capacity()),
    )
    .await
    .expect("pipeline should finish within 15s")
    .expect("udp receive pipeline should succeed");

    send_task.abort();

    let expected_au = access_unit();
    eprintln!(
        "emitted={} received={} last_seq={:?} au_len={}",
        stats.frames_emitted,
        sink.received(),
        sink.last_sequence(),
        expected_au.len()
    );
    assert_eq!(stats.frames_emitted, TARGET, "source emits exactly the frame-limit count");
    assert_eq!(sink.received(), TARGET, "sink receives every depayloaded access unit");
    assert_eq!(sink.last_sequence(), Some(TARGET - 1), "access units arrive in order");
}
