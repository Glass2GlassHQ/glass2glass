//! Loopback integration test for the UDP RTP receive path: a sender packetizes
//! synthetic H.264 access units with `RtpH264Packetizer` and sends them over a
//! UDP socket to `UdpSrc`, which depayloads them back to Annex-B and pushes
//! them into a `FakeSink`. Exercises single-NAL, STAP-style multi-NAL, and FU-A
//! fragmentation end-to-end over real localhost UDP.

#![cfg(all(feature = "udp-ingress", feature = "udp-egress"))]

use core::future::Future;
use core::pin::Pin;
use std::collections::HashSet;
use std::net::{SocketAddr, UdpSocket as StdUdpSocket};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_simple_pipeline, LatencyProfile};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PipelineClock, PipelinePacket, PushOutcome, Rate, VideoCodec,
};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::rtcp;
use g2g_plugins::rtppay::RtpH264Packetizer;
use g2g_plugins::udpsink::UdpSink;
use g2g_plugins::udpsrc::UdpSrc;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Records the AU index marker (the byte just past the NAL header) of every
/// received access unit, so a test can assert content arrives in order.
#[derive(Default)]
struct MarkerSink {
    markers: Vec<u8>,
}

impl AsyncElement for MarkerSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;
    fn intercept_caps(&self, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(frame) = packet {
                if let Some(slice) = frame.domain.as_system_slice() {
                    // Annex-B: [0,0,0,1][NAL header][index marker ..].
                    if let Some(&marker) = slice.get(5) {
                        self.markers.push(marker);
                    }
                }
            }
            Ok(())
        })
    }
}

/// One synthetic access unit: SPS, PPS, then an oversized IDR that forces FU-A
/// fragmentation under the small MTU below. Content is arbitrary; the receive
/// path only reassembles bytes.
fn access_unit() -> Vec<u8> {
    let mut au = Vec::new();
    for nal in [
        vec![0x67u8, 0x42, 0x00, 0x1f],
        vec![0x68u8, 0xce, 0x38, 0x80],
    ] {
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
        run_simple_pipeline(
            &mut src,
            &mut sink,
            &clock,
            LatencyProfile::Live.link_capacity(),
        ),
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
    assert_eq!(
        stats.frames_emitted, TARGET,
        "source emits exactly the frame-limit count"
    );
    assert_eq!(
        sink.received(),
        TARGET,
        "sink receives every depayloaded access unit"
    );
    assert_eq!(
        sink.last_sequence(),
        Some(TARGET - 1),
        "access units arrive in order"
    );
}

#[tokio::test]
async fn jitter_buffer_reorders_out_of_order_packets() {
    // Each access unit is one small single-NAL packet whose marker byte encodes
    // its index. We packetize them in sequence (so RTP seq = index) but send
    // them with later pairs swapped. The jitter buffer must restore order, so
    // the sink sees indices 0,1,2,... with nothing dropped. Without it, every
    // swapped pair is a sequence gap that resets reassembly and loses an AU.
    let recv = StdUdpSocket::bind("127.0.0.1:0").expect("bind receiver");
    let recv_addr = recv.local_addr().unwrap();

    const N: u8 = 24;
    // Generous hold + depth so reorder is absorbed, never declared lost.
    let mut src = UdpSrc::from_socket(recv)
        .unwrap()
        .with_jitter(200, 64)
        .with_frame_limit(N as u64);
    let mut sink = MarkerSink::default();
    let clock = ZeroClock;

    // Pre-build one RTP packet per AU (seq = index, marker set).
    let mut pkt = RtpH264Packetizer::new(96, 0x1234);
    let mut packets: Vec<Vec<u8>> = Vec::new();
    for i in 0..N {
        let au = alloc_au(i);
        let mut p = pkt.packetize(&au, (i as u32).wrapping_mul(3000));
        assert_eq!(p.len(), 1, "single small NAL is one packet");
        packets.push(p.pop().unwrap());
    }
    // Send order: 0 first (sets the baseline), then swap each later pair.
    let mut order: Vec<usize> = vec![0];
    let mut i = 1usize;
    while i < N as usize {
        if i + 1 < N as usize {
            order.push(i + 1);
            order.push(i);
        } else {
            order.push(i);
        }
        i += 2;
    }

    let sender = StdUdpSocket::bind("127.0.0.1:0").expect("bind sender");
    sender.connect(recv_addr).unwrap();
    let send_task = tokio::task::spawn_blocking(move || {
        // Resend the whole reordered burst a few times so startup latency never
        // costs the receiver an AU; duplicates are dropped by the buffer.
        for _ in 0..6 {
            for &idx in &order {
                let _ = sender.send(&packets[idx]);
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        }
    });

    let stats = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        run_simple_pipeline(
            &mut src,
            &mut sink,
            &clock,
            LatencyProfile::Live.link_capacity(),
        ),
    )
    .await
    .expect("pipeline finishes within 15s")
    .expect("udp receive pipeline succeeds");

    send_task.abort();

    eprintln!(
        "emitted={} markers={:?}",
        stats.frames_emitted, sink.markers
    );
    assert_eq!(
        stats.frames_emitted, N as u64,
        "every AU emitted, none lost to reorder"
    );
    let expected: Vec<u8> = (0..N).collect();
    assert_eq!(
        sink.markers, expected,
        "access units delivered in sequence order"
    );
}

/// A single-NAL access unit (Annex-B) whose marker byte is `index`.
fn alloc_au(index: u8) -> Vec<u8> {
    let mut au = vec![0u8, 0, 0, 1, 0x65, index];
    au.extend_from_slice(&[0xAA; 8]);
    au
}

/// Discards an OutputSink's pushes; the UDP sink ignores its downstream.
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
        width: Dim::Fixed(1280),
        height: Dim::Fixed(720),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// A lossy UDP relay between the sender and receiver. RTP from the sender is
/// forwarded to the receiver, except sequences in `drop_once` are dropped the
/// first time they are seen (a retransmit of the same sequence gets through, so
/// the NACK loop can recover). RTCP from the receiver is relayed back to the
/// sender, so the feedback channel survives the lossy link.
async fn lossy_proxy(
    proxy: tokio::net::UdpSocket,
    recv_addr: SocketAddr,
    mut drop_once: HashSet<u16>,
) {
    let mut sink_addr: Option<SocketAddr> = None;
    let mut buf = [0u8; 2048];
    loop {
        let Ok((n, from)) = proxy.recv_from(&mut buf).await else {
            return;
        };
        if from == recv_addr {
            // RTCP (RR / NACK) heading back to the sender.
            if let Some(dest) = sink_addr {
                let _ = proxy.send_to(&buf[..n], dest).await;
            }
            continue;
        }
        sink_addr = Some(from);
        // Media from the sender. Drop a target sequence exactly once.
        if !rtcp::is_rtcp(&buf[..n]) && n >= 4 {
            let seq = u16::from_be_bytes([buf[2], buf[3]]);
            if drop_once.remove(&seq) {
                continue;
            }
        }
        let _ = proxy.send_to(&buf[..n], recv_addr).await;
    }
}

#[tokio::test]
async fn nack_recovers_dropped_packets_via_retransmission() {
    // sender(UdpSink) -> lossy proxy (drops seq 5,12,20 once) -> receiver(UdpSrc).
    // The receiver NACKs the gaps; the sender resends from its history; the
    // retransmits get through the proxy and the receiver recovers every AU.
    let proxy = tokio::net::UdpSocket::bind("127.0.0.1:0")
        .await
        .expect("bind proxy");
    let proxy_addr = proxy.local_addr().unwrap();
    let recv_std = StdUdpSocket::bind("127.0.0.1:0").expect("bind receiver");
    let recv_addr = recv_std.local_addr().unwrap();

    const N: u8 = 30;
    let dropped: HashSet<u16> = [5u16, 12, 20].into_iter().collect();
    let proxy_task = tokio::spawn(lossy_proxy(proxy, recv_addr, dropped));

    // Receiver: generous jitter hold so a gap waits for its retransmit; NACK on.
    let mut src = UdpSrc::from_socket(recv_std)
        .unwrap()
        .with_jitter(500, 256)
        .with_rtcp(200, true)
        .with_frame_limit(N as u64);
    let mut marker = MarkerSink::default();
    let clock = ZeroClock;

    // Sender: drive UdpSink manually so we can read its retransmit count after.
    // Send well past N so process() keeps servicing NACKs while the receiver
    // recovers the early gaps; the receiver stops at its frame limit.
    let sink_fut = async {
        let mut sink = UdpSink::new(proxy_addr);
        sink.configure_pipeline(&h264_caps())
            .expect("configure sink");
        let mut null = NullOut;
        for i in 0u32..(N as u32 * 2) {
            let au = alloc_au((i & 0xFF) as u8);
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
                .expect("send");
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        sink
    };

    let recv_fut = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        run_simple_pipeline(
            &mut src,
            &mut marker,
            &clock,
            LatencyProfile::Live.link_capacity(),
        ),
    );

    let (recv_res, sink) = tokio::join!(recv_fut, sink_fut);
    proxy_task.abort();

    let stats = recv_res
        .expect("receiver finishes within 15s")
        .expect("receive pipeline succeeds");
    eprintln!(
        "emitted={} retransmits={} markers={:?}",
        stats.frames_emitted,
        sink.retransmits_sent(),
        marker.markers,
    );
    assert_eq!(
        stats.frames_emitted, N as u64,
        "every AU recovered despite the drops"
    );
    let expected: Vec<u8> = (0..N).collect();
    assert_eq!(
        marker.markers, expected,
        "AUs delivered in order after recovery"
    );
    assert!(
        sink.retransmits_sent() >= 3,
        "sender retransmitted the dropped packets"
    );
}

#[tokio::test]
async fn rtx_resends_recover_dropped_packets() {
    // Same lossy link, but the sender resends as RFC 4588 RTX packets (a distinct
    // payload type, on its own SSRC, with the original sequence prepended) and the
    // receiver reconstructs them. The dropped AUs (5, 12, 20) have no other path to
    // recovery, so success proves the RTX wrap + reconstruct round trip end to end.
    let proxy = tokio::net::UdpSocket::bind("127.0.0.1:0")
        .await
        .expect("bind proxy");
    let proxy_addr = proxy.local_addr().unwrap();
    let recv_std = StdUdpSocket::bind("127.0.0.1:0").expect("bind receiver");
    let recv_addr = recv_std.local_addr().unwrap();

    const N: u8 = 30;
    let dropped: HashSet<u16> = [5u16, 12, 20].into_iter().collect();
    let proxy_task = tokio::spawn(lossy_proxy(proxy, recv_addr, dropped));

    // RTX payload type 97 carries the original PT 96; reconstruct on the receiver.
    let mut src = UdpSrc::from_socket(recv_std)
        .unwrap()
        .with_jitter(500, 256)
        .with_rtcp(200, true)
        .with_rtx(97, 96)
        .with_frame_limit(N as u64);
    let mut marker = MarkerSink::default();
    let clock = ZeroClock;

    let sink_fut = async {
        // RTX resends on PT 97, a distinct RTX SSRC (SSRC-multiplexed RTX).
        let mut sink = UdpSink::new(proxy_addr).with_rtx(97, 0x5254_5800);
        sink.configure_pipeline(&h264_caps())
            .expect("configure sink");
        let mut null = NullOut;
        for i in 0u32..(N as u32 * 2) {
            let au = alloc_au((i & 0xFF) as u8);
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
                .expect("send");
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        sink
    };

    let recv_fut = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        run_simple_pipeline(
            &mut src,
            &mut marker,
            &clock,
            LatencyProfile::Live.link_capacity(),
        ),
    );

    let (recv_res, sink) = tokio::join!(recv_fut, sink_fut);
    proxy_task.abort();

    let stats = recv_res
        .expect("receiver finishes within 15s")
        .expect("receive pipeline succeeds");
    assert_eq!(
        stats.frames_emitted, N as u64,
        "every AU recovered via RTX resends"
    );
    let expected: Vec<u8> = (0..N).collect();
    assert_eq!(
        marker.markers, expected,
        "AUs delivered in order after RTX recovery"
    );
    assert!(
        sink.retransmits_sent() >= 3,
        "sender resent the dropped packets as RTX"
    );
}

#[tokio::test]
async fn fec_recovers_dropped_packets_without_retransmission() {
    // One-way path: the receiver sends NO feedback (RTCP off) and the sender does
    // NOT retransmit, so the only way the dropped AUs can be recovered is the
    // ULPFEC repair packets. The proxy drops one media packet per FEC group.
    let proxy = tokio::net::UdpSocket::bind("127.0.0.1:0")
        .await
        .expect("bind proxy");
    let proxy_addr = proxy.local_addr().unwrap();
    let recv_std = StdUdpSocket::bind("127.0.0.1:0").expect("bind receiver");
    let recv_addr = recv_std.local_addr().unwrap();

    const N: u8 = 12;
    // Groups of 4: seqs 0-3, 4-7, 8-11. Drop one media packet from each group.
    let dropped: HashSet<u16> = [2u16, 6, 10].into_iter().collect();
    let proxy_task = tokio::spawn(lossy_proxy(proxy, recv_addr, dropped));

    let mut src = UdpSrc::from_socket(recv_std)
        .unwrap()
        .with_jitter(500, 256)
        .with_rtcp(0, false) // no RR, no NACK: a strictly one-way receiver
        .with_fec(98)
        .with_frame_limit(N as u64);
    let mut marker = MarkerSink::default();
    let clock = ZeroClock;

    let sink_fut = async {
        // FEC group of 4 on payload type 98; retransmit OFF so FEC alone recovers.
        let mut sink =
            UdpSink::new(proxy_addr)
                .with_retransmit(false, 1)
                .with_fec(4, 98, 0xFEC0_0001);
        sink.configure_pipeline(&h264_caps())
            .expect("configure sink");
        let mut null = NullOut;
        for i in 0u32..N as u32 {
            let au = alloc_au(i as u8);
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
                .expect("send");
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        sink
    };

    let recv_fut = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        run_simple_pipeline(
            &mut src,
            &mut marker,
            &clock,
            LatencyProfile::Live.link_capacity(),
        ),
    );

    let (recv_res, sink) = tokio::join!(recv_fut, sink_fut);
    proxy_task.abort();

    let stats = recv_res
        .expect("receiver finishes within 15s")
        .expect("receive pipeline succeeds");
    assert_eq!(
        stats.frames_emitted, N as u64,
        "every dropped AU recovered by FEC"
    );
    let expected: Vec<u8> = (0..N).collect();
    assert_eq!(
        marker.markers, expected,
        "AUs delivered in order after FEC recovery"
    );
    assert_eq!(
        sink.retransmits_sent(),
        0,
        "recovery was forward error correction, not resends"
    );
}

#[tokio::test]
async fn interleaved_fec_recovers_a_consecutive_burst() {
    // The burst single-level FEC cannot handle: a one-way path (no feedback, no
    // retransmit) loses 4 *consecutive* packets, which would put >1 loss in a
    // contiguous group. Interleaved (column) FEC over a 4x4 block spreads the
    // burst one-per-column, so each column repair recovers its loss.
    let proxy = tokio::net::UdpSocket::bind("127.0.0.1:0")
        .await
        .expect("bind proxy");
    let proxy_addr = proxy.local_addr().unwrap();
    let recv_std = StdUdpSocket::bind("127.0.0.1:0").expect("bind receiver");
    let recv_addr = recv_std.local_addr().unwrap();

    const N: u8 = 16; // one full 4x4 interleaving block (1 RTP packet per AU)
                      // Drop the consecutive burst seqs 5..=8: columns 1,2,3,0 respectively, so
                      // exactly one loss per column.
    let dropped: HashSet<u16> = [5u16, 6, 7, 8].into_iter().collect();
    let proxy_task = tokio::spawn(lossy_proxy(proxy, recv_addr, dropped));

    let mut src = UdpSrc::from_socket(recv_std)
        .unwrap()
        .with_jitter(500, 256)
        .with_rtcp(0, false) // strictly one-way receiver
        .with_fec(98) // the decoder handles interleaved masks unchanged
        .with_frame_limit(N as u64);
    let mut marker = MarkerSink::default();
    let clock = ZeroClock;

    let sink_fut = async {
        // 4 rows x 4 columns; retransmit OFF so only the column repairs can recover.
        let mut sink = UdpSink::new(proxy_addr)
            .with_retransmit(false, 1)
            .with_interleaved_fec(4, 4, 98, 0xFEC0_0002);
        sink.configure_pipeline(&h264_caps())
            .expect("configure sink");
        let mut null = NullOut;
        for i in 0u32..N as u32 {
            let au = alloc_au(i as u8);
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
                .expect("send");
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        sink
    };

    let recv_fut = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        run_simple_pipeline(
            &mut src,
            &mut marker,
            &clock,
            LatencyProfile::Live.link_capacity(),
        ),
    );

    let (recv_res, sink) = tokio::join!(recv_fut, sink_fut);
    proxy_task.abort();

    let stats = recv_res
        .expect("receiver finishes within 15s")
        .expect("receive pipeline succeeds");
    assert_eq!(
        stats.frames_emitted, N as u64,
        "the whole consecutive burst recovered"
    );
    let expected: Vec<u8> = (0..N).collect();
    assert_eq!(
        marker.markers, expected,
        "AUs delivered in order after burst recovery"
    );
    assert_eq!(
        sink.retransmits_sent(),
        0,
        "recovery was interleaved FEC, not resends"
    );
}

#[tokio::test]
async fn flexfec_2d_recovers_a_burst_row_of_losses() {
    // M758 multi-level burst FEC: a 4x3 block (row repairs every 4 packets +
    // 4 column repairs at stride 4). Dropping 4 CONSECUTIVE packets (a whole
    // row) defeats every row repair; the column repairs recover all four.
    // One-way path: retransmit off, so only FEC can bring them back.
    let proxy = tokio::net::UdpSocket::bind("127.0.0.1:0")
        .await
        .expect("bind proxy");
    let proxy_addr = proxy.local_addr().unwrap();
    let recv_std = StdUdpSocket::bind("127.0.0.1:0").expect("bind receiver");
    let recv_addr = recv_std.local_addr().unwrap();

    const N: u8 = 12; // one 4x3 FlexFEC block (1 RTP packet per AU)
    let dropped: HashSet<u16> = [4u16, 5, 6, 7].into_iter().collect();
    let proxy_task = tokio::spawn(lossy_proxy(proxy, recv_addr, dropped));

    let mut src = UdpSrc::from_socket(recv_std)
        .unwrap()
        .with_jitter(500, 256)
        .with_rtcp(0, false)
        .with_flexfec(98)
        .with_frame_limit(N as u64);
    let mut marker = MarkerSink::default();
    let clock = ZeroClock;

    let sink_fut = async {
        let mut sink = UdpSink::new(proxy_addr)
            .with_retransmit(false, 1)
            .with_flexfec_2d(4, 3, 98, 0xFEC0_0004);
        sink.configure_pipeline(&h264_caps())
            .expect("configure sink");
        let mut null = NullOut;
        for i in 0u32..N as u32 {
            let au = alloc_au(i as u8);
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
                .expect("send");
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        sink
    };

    let recv_fut = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        run_simple_pipeline(
            &mut src,
            &mut marker,
            &clock,
            LatencyProfile::Live.link_capacity(),
        ),
    );

    let (recv_res, sink) = tokio::join!(recv_fut, sink_fut);
    proxy_task.abort();

    let stats = recv_res
        .expect("receiver finishes within 15s")
        .expect("receive pipeline succeeds");
    assert_eq!(
        stats.frames_emitted, N as u64,
        "the burst row recovered via column repairs"
    );
    let expected: Vec<u8> = (0..N).collect();
    assert_eq!(
        marker.markers, expected,
        "AUs delivered in order after 2-D burst recovery"
    );
    assert_eq!(sink.retransmits_sent(), 0, "recovery was FEC, not resends");
}

#[tokio::test]
async fn flexfec_recovers_a_loss_in_a_group_beyond_ulpfec() {
    // FlexFEC's wide mask protects a 20-packet group with one repair, past
    // ULPFEC's 16-packet ceiling. One-way path (no feedback / no retransmit): the
    // single FlexFEC repair recovers the one dropped packet.
    let proxy = tokio::net::UdpSocket::bind("127.0.0.1:0")
        .await
        .expect("bind proxy");
    let proxy_addr = proxy.local_addr().unwrap();
    let recv_std = StdUdpSocket::bind("127.0.0.1:0").expect("bind receiver");
    let recv_addr = recv_std.local_addr().unwrap();

    const N: u8 = 20; // one FlexFEC group of 20 (1 RTP packet per AU)
    let dropped: HashSet<u16> = [10u16].into_iter().collect();
    let proxy_task = tokio::spawn(lossy_proxy(proxy, recv_addr, dropped));

    let mut src = UdpSrc::from_socket(recv_std)
        .unwrap()
        .with_jitter(500, 256)
        .with_rtcp(0, false) // strictly one-way receiver
        .with_flexfec(98)
        .with_frame_limit(N as u64);
    let mut marker = MarkerSink::default();
    let clock = ZeroClock;

    let sink_fut = async {
        // FlexFEC over a group of 20 on PT 98; retransmit OFF so only the repair recovers.
        let mut sink = UdpSink::new(proxy_addr)
            .with_retransmit(false, 1)
            .with_flexfec(20, 98, 0xFEC0_0003);
        sink.configure_pipeline(&h264_caps())
            .expect("configure sink");
        let mut null = NullOut;
        for i in 0u32..N as u32 {
            let au = alloc_au(i as u8);
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
                .expect("send");
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        sink
    };

    let recv_fut = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        run_simple_pipeline(
            &mut src,
            &mut marker,
            &clock,
            LatencyProfile::Live.link_capacity(),
        ),
    );

    let (recv_res, sink) = tokio::join!(recv_fut, sink_fut);
    proxy_task.abort();

    let stats = recv_res
        .expect("receiver finishes within 15s")
        .expect("receive pipeline succeeds");
    assert_eq!(
        stats.frames_emitted, N as u64,
        "the dropped packet recovered by FlexFEC"
    );
    let expected: Vec<u8> = (0..N).collect();
    assert_eq!(
        marker.markers, expected,
        "AUs delivered in order after FlexFEC recovery"
    );
    assert_eq!(
        sink.retransmits_sent(),
        0,
        "recovery was FlexFEC, not resends"
    );
}
