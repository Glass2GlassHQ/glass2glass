//! SRT end-to-end over loopback: a caller (`SrtSink`) connects to a listener
//! (`SrtSrc`) through a lossy proxy that drops one data packet once. The HSv5
//! handshake completes, the listener detects the gap and NAKs, the caller
//! retransmits, and every payload is delivered in order. Proves the handshake +
//! data transport + NAK-based ARQ across the real socket path (g2g <-> g2g; real
//! libsrt/ffmpeg interop is operator-validated).
#![cfg(feature = "srt")]

use core::future::Future;
use core::pin::Pin;
use std::net::{SocketAddr, UdpSocket as StdUdpSocket};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_simple_pipeline, LatencyProfile};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PipelineClock, PipelinePacket, PushOutcome,
};

use g2g_plugins::srt;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}
use g2g_plugins::srtsink::SrtSink;
use g2g_plugins::srtsrc::SrtSrc;

/// Sink that records the first byte of each received payload (its index tag).
#[derive(Default)]
struct TagSink {
    tags: Vec<u8>,
}

impl AsyncElement for TagSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>> where Self: 'a;
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
                if let MemoryDomain::System(slice) = &frame.domain {
                    if let Some(&tag) = slice.as_slice().first() {
                        self.tags.push(tag);
                    }
                }
            }
            Ok(())
        })
    }
}

/// A no-op downstream for driving the sink directly.
struct NullOut;
impl OutputSink for NullOut {
    fn push<'a>(
        &'a mut self,
        _p: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async { Ok(PushOutcome::Accepted) })
    }
}

/// Relay caller<->listener, dropping the SRT data packet with `drop_seq` once.
async fn lossy_proxy(proxy: tokio::net::UdpSocket, listener_addr: SocketAddr, drop_seq: u32) {
    let mut caller: Option<SocketAddr> = None;
    let mut dropped = false;
    let mut buf = [0u8; 2048];
    loop {
        let Ok((n, from)) = proxy.recv_from(&mut buf).await else { return };
        if Some(from) == caller || (caller.is_none() && from != listener_addr) {
            caller = Some(from);
            // Caller -> listener. Drop the target data packet exactly once.
            if !srt::is_control(&buf[..n]) {
                if let Some(d) = srt::parse_data_packet(&buf[..n]) {
                    if d.seq == drop_seq && !dropped {
                        dropped = true;
                        continue;
                    }
                }
            }
            let _ = proxy.send_to(&buf[..n], listener_addr).await;
        } else {
            // Listener -> caller (NAK / ACK / handshake replies).
            if let Some(dest) = caller {
                let _ = proxy.send_to(&buf[..n], dest).await;
            }
        }
    }
}

#[tokio::test]
async fn srt_handshake_and_arq_recover_a_dropped_payload() {
    const N: u8 = 12;

    // Listener on an ephemeral port; proxy in front of it.
    let listener_std = StdUdpSocket::bind("127.0.0.1:0").expect("bind listener");
    let listener_addr = listener_std.local_addr().unwrap();
    let proxy = tokio::net::UdpSocket::bind("127.0.0.1:0").await.expect("bind proxy");
    let proxy_addr = proxy.local_addr().unwrap();

    // Init seq is 1, so the 3rd data packet is seq 3; drop it once.
    let proxy_task = tokio::spawn(lossy_proxy(proxy, listener_addr, 3));

    let mut src = SrtSrc::from_socket(listener_std).unwrap().with_frame_limit(N as u64);
    let mut sink_collect = TagSink::default();
    let clock = ZeroClock;

    // Caller: one small payload per frame, tagged 0..N by its first byte.
    let caller = async {
        let mut sink = SrtSink::new(proxy_addr);
        sink.configure_pipeline(&Caps::ByteStream { encoding: g2g_core::ByteStreamEncoding::MpegTs })
            .expect("configure");
        let mut null = NullOut;
        for i in 0u8..(N * 2) {
            let payload = vec![i % N; 100];
            let frame = Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(payload.into_boxed_slice())),
                timing: FrameTiming { pts_ns: i as u64 * 10_000_000, ..FrameTiming::default() },
                sequence: i as u64,
                meta: Default::default(),
            };
            // Once the listener hits its limit + leaves, a send may fail; that is fine.
            if sink.process(PipelinePacket::DataFrame(frame), &mut null).await.is_err() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(8)).await;
        }
        sink.retransmits()
    };

    let recv = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        run_simple_pipeline(&mut src, &mut sink_collect, &clock, LatencyProfile::Live.link_capacity()),
    );

    let (recv_res, retransmits) = tokio::join!(recv, caller);
    proxy_task.abort();

    let stats = recv_res.expect("listener finishes within 15s").expect("receive pipeline ok");
    assert_eq!(stats.frames_emitted, N as u64, "every payload delivered despite the drop");
    let expected: Vec<u8> = (0..N).collect();
    assert_eq!(sink_collect.tags, expected, "payloads delivered in order after ARQ recovery");
    assert!(retransmits >= 1, "caller retransmitted the dropped packet on NAK");
}

#[tokio::test]
async fn congestion_control_paces_egress_to_the_bandwidth_cap() {
    // The caller sends as fast as the loop allows (no inter-frame sleep) but with
    // a bandwidth cap, so the only throttle is the congestion-control pacing.
    // Unpaced this loop is ~instant; the cap stretches it to ~packets*size/bw.
    const N: u8 = 30;
    const PAYLOAD: usize = 1316; // one SRT packet per frame
    const BW: u64 = 200_000; // bytes/sec cap

    let listener_std = StdUdpSocket::bind("127.0.0.1:0").expect("bind listener");
    let listener_addr = listener_std.local_addr().unwrap();

    let mut src = SrtSrc::from_socket(listener_std).unwrap().with_frame_limit(N as u64);
    let mut sink_collect = TagSink::default();
    let clock = ZeroClock;

    let caller = async {
        let mut sink = SrtSink::new(listener_addr).with_max_bandwidth(BW);
        sink.configure_pipeline(&Caps::ByteStream { encoding: g2g_core::ByteStreamEncoding::MpegTs })
            .expect("configure");
        let mut null = NullOut;
        let start = std::time::Instant::now();
        for i in 0u8..N {
            let payload = vec![i; PAYLOAD];
            let frame = Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(payload.into_boxed_slice())),
                timing: FrameTiming { pts_ns: i as u64 * 10_000_000, ..FrameTiming::default() },
                sequence: i as u64,
                meta: Default::default(),
            };
            // No inter-frame sleep: the pacing is the only thing slowing this down.
            if sink.process(PipelinePacket::DataFrame(frame), &mut null).await.is_err() {
                break;
            }
        }
        start.elapsed()
    };

    let recv = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        run_simple_pipeline(&mut src, &mut sink_collect, &clock, LatencyProfile::Live.link_capacity()),
    );

    let (recv_res, elapsed) = tokio::join!(recv, caller);

    let stats = recv_res.expect("listener finishes within 15s").expect("receive pipeline ok");
    assert_eq!(stats.frames_emitted, N as u64, "all paced packets delivered");

    // Expected ~ N*PAYLOAD/BW seconds. Assert a lower bound (pacing cannot send
    // faster than the cap); a generous 60% margin absorbs scheduler slack.
    let expected_ms = (N as u64 * PAYLOAD as u64 * 1000) / BW;
    assert!(
        elapsed.as_millis() as u64 >= expected_ms * 6 / 10,
        "pacing held egress near the {BW} B/s cap (elapsed {elapsed:?}, expected ~{expected_ms}ms)",
    );
}
