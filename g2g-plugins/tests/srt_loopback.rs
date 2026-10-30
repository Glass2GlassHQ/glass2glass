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
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_simple_pipeline, LatencyProfile, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, FrameTiming, G2gError, MemoryDomain, NetworkFailure,
    OutputSink, PipelineClock, PipelinePacket, PropValue, PushOutcome,
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
                    if let Some(&tag) = slice.first() {
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
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        packet_slot.take();
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

#[derive(Clone, Copy)]
struct DroppedSequence {
    seq: u32,
    copies: usize,
}

/// Relay caller<->listener, dropping the first `copies` of the SRT data packet
/// `dropped.seq` and counting the ACKs the listener sends.
async fn lossy_proxy(
    proxy: tokio::net::UdpSocket,
    listener_addr: SocketAddr,
    dropped: Option<DroppedSequence>,
    acks_seen: Arc<AtomicUsize>,
) {
    let mut caller: Option<SocketAddr> = None;
    let mut copies_dropped = 0;
    let mut buf = [0u8; 2048];
    loop {
        let Ok((n, from)) = proxy.recv_from(&mut buf).await else {
            return;
        };
        if Some(from) == caller || (caller.is_none() && from != listener_addr) {
            caller = Some(from);
            // Caller -> listener. Drop the target data packet's first copies.
            let data_seq = srt::parse_data_packet(&buf[..n]).map(|d| d.seq);
            if let Some(dropped) = dropped.filter(|d| Some(d.seq) == data_seq) {
                if copies_dropped < dropped.copies {
                    copies_dropped += 1;
                    continue;
                }
            }
            let _ = proxy.send_to(&buf[..n], listener_addr).await;
        } else {
            // Listener -> caller (NAK / ACK / handshake replies).
            if matches!(
                srt::parse_control(&buf[..n]),
                Some(srt::Control::Ack { .. })
            ) {
                acks_seen.fetch_add(1, Ordering::Relaxed);
            }
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
    let proxy = tokio::net::UdpSocket::bind("127.0.0.1:0")
        .await
        .expect("bind proxy");
    let proxy_addr = proxy.local_addr().unwrap();

    // Init seq is 1, so the 3rd data packet is seq 3; drop it once.
    let dropped = DroppedSequence { seq: 3, copies: 1 };
    let proxy_task = tokio::spawn(lossy_proxy(
        proxy,
        listener_addr,
        Some(dropped),
        Arc::default(),
    ));

    let mut src = SrtSrc::from_socket(listener_std)
        .unwrap()
        .with_frame_limit(N as u64);
    let mut sink_collect = TagSink::default();
    let clock = ZeroClock;

    // Caller: one small payload per frame, tagged 0..N by its first byte.
    let caller = async {
        let mut sink = SrtSink::new(proxy_addr);
        sink.configure_pipeline(&Caps::ByteStream {
            encoding: g2g_core::ByteStreamEncoding::MpegTs,
        })
        .expect("configure");
        let mut null = NullOut;
        for i in 0u8..(N * 2) {
            let payload = vec![i % N; 100];
            let frame = Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(payload.into_boxed_slice())),
                timing: FrameTiming {
                    pts_ns: i as u64 * 10_000_000,
                    ..FrameTiming::default()
                },
                sequence: i as u64,
                meta: Default::default(),
            };
            // Once the listener hits its limit + leaves, a send may fail; that is fine.
            if sink
                .process(PipelinePacket::DataFrame(frame), &mut null)
                .await
                .is_err()
            {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(8)).await;
        }
        sink.retransmits()
    };

    let recv = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        run_simple_pipeline(
            &mut src,
            &mut sink_collect,
            &clock,
            LatencyProfile::Live.link_capacity(),
        ),
    );

    let (recv_res, retransmits) = tokio::join!(recv, caller);
    proxy_task.abort();

    let stats = recv_res
        .expect("listener finishes within 15s")
        .expect("receive pipeline ok");
    assert_eq!(
        stats.frames_emitted, N as u64,
        "every payload delivered despite the drop"
    );
    let expected: Vec<u8> = (0..N).collect();
    assert_eq!(
        sink_collect.tags, expected,
        "payloads delivered in order after ARQ recovery"
    );
    assert!(
        retransmits >= 1,
        "caller retransmitted the dropped packet on NAK"
    );
}

#[tokio::test]
async fn listener_acks_on_its_timer_not_per_datagram() {
    const N: u8 = 200;
    const ACK_INTERVAL: std::time::Duration = std::time::Duration::from_millis(10);
    // The first ACK goes out at once, and a timer wake can land at the boundary.
    const ACK_SLACK: usize = 2;

    let listener_std = StdUdpSocket::bind("127.0.0.1:0").expect("bind listener");
    let listener_addr = listener_std.local_addr().unwrap();
    let proxy = tokio::net::UdpSocket::bind("127.0.0.1:0")
        .await
        .expect("bind proxy");
    let proxy_addr = proxy.local_addr().unwrap();
    let acks_seen = Arc::new(AtomicUsize::new(0));
    let proxy_task = tokio::spawn(lossy_proxy(proxy, listener_addr, None, acks_seen.clone()));

    let mut src = SrtSrc::from_socket(listener_std)
        .unwrap()
        .with_frame_limit(N as u64);
    let mut sink_collect = TagSink::default();
    let clock = ZeroClock;

    let caller = async {
        let mut sink = SrtSink::new(proxy_addr);
        sink.configure_pipeline(&Caps::ByteStream {
            encoding: g2g_core::ByteStreamEncoding::MpegTs,
        })
        .expect("configure");
        let mut null = NullOut;
        for i in 0..N {
            let frame = Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(
                    vec![i; 100].into_boxed_slice(),
                )),
                timing: FrameTiming::default(),
                sequence: i as u64,
                meta: Default::default(),
            };
            sink.process(PipelinePacket::DataFrame(frame), &mut null)
                .await
                .expect("send");
        }
    };

    let start = std::time::Instant::now();
    let recv = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        run_simple_pipeline(
            &mut src,
            &mut sink_collect,
            &clock,
            LatencyProfile::Live.link_capacity(),
        ),
    );
    let (recv_res, ()) = tokio::join!(recv, caller);
    let elapsed = start.elapsed();
    proxy_task.abort();

    let stats = recv_res
        .expect("listener finishes within 15s")
        .expect("receive pipeline ok");
    assert_eq!(stats.frames_emitted, N as u64);
    let ack_bound = (elapsed.as_nanos() / ACK_INTERVAL.as_nanos()) as usize + ACK_SLACK;
    let acks = acks_seen.load(Ordering::Relaxed);
    assert!(
        acks <= ack_bound,
        "{acks} ACKs in {elapsed:?} for {N} datagrams, the 10 ms timer allows {ack_bound}"
    );
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

    let mut src = SrtSrc::from_socket(listener_std)
        .unwrap()
        .with_frame_limit(N as u64);
    let mut sink_collect = TagSink::default();
    let clock = ZeroClock;

    let caller = async {
        let mut sink = SrtSink::new(listener_addr).with_max_bandwidth(BW);
        sink.configure_pipeline(&Caps::ByteStream {
            encoding: g2g_core::ByteStreamEncoding::MpegTs,
        })
        .expect("configure");
        let mut null = NullOut;
        let start = std::time::Instant::now();
        for i in 0u8..N {
            let payload = vec![i; PAYLOAD];
            let frame = Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(payload.into_boxed_slice())),
                timing: FrameTiming {
                    pts_ns: i as u64 * 10_000_000,
                    ..FrameTiming::default()
                },
                sequence: i as u64,
                meta: Default::default(),
            };
            // No inter-frame sleep: the pacing is the only thing slowing this down.
            if sink
                .process(PipelinePacket::DataFrame(frame), &mut null)
                .await
                .is_err()
            {
                break;
            }
        }
        start.elapsed()
    };

    let recv = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        run_simple_pipeline(
            &mut src,
            &mut sink_collect,
            &clock,
            LatencyProfile::Live.link_capacity(),
        ),
    );

    let (recv_res, elapsed) = tokio::join!(recv, caller);

    let stats = recv_res
        .expect("listener finishes within 15s")
        .expect("receive pipeline ok");
    assert_eq!(
        stats.frames_emitted, N as u64,
        "all paced packets delivered"
    );

    // Expected ~ N*PAYLOAD/BW seconds. Assert a lower bound (pacing cannot send
    // faster than the cap); a generous 60% margin absorbs scheduler slack.
    let expected_ms = (N as u64 * PAYLOAD as u64 * 1000) / BW;
    assert!(
        elapsed.as_millis() as u64 >= expected_ms * 6 / 10,
        "pacing held egress near the {BW} B/s cap (elapsed {elapsed:?}, expected ~{expected_ms}ms)",
    );
}

fn tagged_frame(tag: u8) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(vec![tag; 100].into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: tag as u64,
        meta: Default::default(),
    }
}

async fn send_tagged(sink: &mut SrtSink, tag: u8) -> Result<(), G2gError> {
    sink.process(PipelinePacket::DataFrame(tagged_frame(tag)), &mut NullOut)
        .await
}

#[tokio::test]
async fn a_packet_the_sender_never_resends_is_skipped_once_the_next_is_due() {
    const N: u8 = 12;
    // the caller's first sequence is 1, so the third payload is sequence 3
    const LOST_TAG: u8 = 2;

    let listener_std = StdUdpSocket::bind("127.0.0.1:0").expect("bind listener");
    let listener_addr = listener_std.local_addr().unwrap();
    let proxy = tokio::net::UdpSocket::bind("127.0.0.1:0")
        .await
        .expect("bind proxy");
    let proxy_addr = proxy.local_addr().unwrap();
    let dropped = DroppedSequence {
        seq: u32::from(LOST_TAG) + 1,
        copies: usize::MAX,
    };
    let proxy_task = tokio::spawn(lossy_proxy(
        proxy,
        listener_addr,
        Some(dropped),
        Arc::default(),
    ));

    let mut src = SrtSrc::from_socket(listener_std)
        .unwrap()
        .with_frame_limit(N as u64);
    let mut sink_collect = TagSink::default();
    let clock = ZeroClock;

    let caller = async {
        let mut sink = SrtSink::new(proxy_addr);
        sink.configure_pipeline(&Caps::ByteStream {
            encoding: g2g_core::ByteStreamEncoding::MpegTs,
        })
        .expect("configure");
        for tag in 0..N * 2 {
            // Once the listener hits its limit + leaves, a send may fail; that is fine.
            if send_tagged(&mut sink, tag).await.is_err() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(8)).await;
        }
    };

    let recv = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        run_simple_pipeline(
            &mut src,
            &mut sink_collect,
            &clock,
            LatencyProfile::Live.link_capacity(),
        ),
    );
    let (recv_res, ()) = tokio::join!(recv, caller);
    proxy_task.abort();

    recv_res
        .expect("delivery carries on past the lost packet")
        .expect("receive pipeline ok");
    let expected: Vec<u8> = (0..=N).filter(|&tag| tag != LOST_TAG).collect();
    assert_eq!(sink_collect.tags, expected);
}

#[derive(Debug, PartialEq)]
enum Received {
    Data(u8),
    Flush,
    Eos,
}

struct ReceivedLog {
    events: Vec<Received>,
    flushed: Arc<tokio::sync::Notify>,
}

impl OutputSink for ReceivedLog {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        match packet_slot.take().expect("poll_push without a packet") {
            PipelinePacket::DataFrame(frame) => {
                let slice = frame.domain.as_system_slice().expect("system frame");
                self.events.push(Received::Data(slice[0]));
            }
            PipelinePacket::Flush => {
                self.events.push(Received::Flush);
                self.flushed.notify_one();
            }
            PipelinePacket::Eos => self.events.push(Received::Eos),
            _ => {}
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

const PER_CALLER: u8 = 5;
const LATENCY_MS: u16 = 50;
// TSBPD holds each payload for the latency, and a caller's end drops what is still held
const DELIVERY_WAIT: std::time::Duration = std::time::Duration::from_millis(4 * LATENCY_MS as u64);
// past the 5 s peer idle timeout
const RUN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);

fn listening_source(keep_listening: bool) -> (SrtSrc, SocketAddr) {
    let listener_std = StdUdpSocket::bind("127.0.0.1:0").expect("bind listener");
    let listener_addr = listener_std.local_addr().unwrap();
    let mut src = SrtSrc::from_socket(listener_std)
        .unwrap()
        .with_frame_limit(2 * PER_CALLER as u64);
    src.set_property("latency", PropValue::Uint(LATENCY_MS.into()))
        .unwrap();
    src.set_property("keep-listening", PropValue::Bool(keep_listening))
        .unwrap();
    (src, listener_addr)
}

async fn caller_sending(listener_addr: SocketAddr, tags: core::ops::Range<u8>) -> SrtSink {
    let mut sink = SrtSink::new(listener_addr).with_latency(LATENCY_MS);
    sink.configure_pipeline(&Caps::ByteStream {
        encoding: g2g_core::ByteStreamEncoding::MpegTs,
    })
    .expect("configure");
    for tag in tags {
        send_tagged(&mut sink, tag).await.expect("send");
    }
    tokio::time::sleep(DELIVERY_WAIT).await;
    sink
}

fn both_callers_then_eos() -> Vec<Received> {
    let mut expected: Vec<Received> = (0..PER_CALLER).map(Received::Data).collect();
    expected.push(Received::Flush);
    expected.extend((PER_CALLER..2 * PER_CALLER).map(Received::Data));
    expected.push(Received::Eos);
    expected
}

#[tokio::test]
async fn keep_listening_takes_a_new_caller_after_a_shutdown() {
    let (mut src, listener_addr) = listening_source(true);
    let flushed = Arc::new(tokio::sync::Notify::new());
    let mut log = ReceivedLog {
        events: Vec::new(),
        flushed: flushed.clone(),
    };

    let callers = async {
        let mut first = caller_sending(listener_addr, 0..PER_CALLER).await;
        first
            .process(PipelinePacket::Eos, &mut NullOut)
            .await
            .expect("shutdown");
        flushed.notified().await;
        caller_sending(listener_addr, PER_CALLER..2 * PER_CALLER).await
    };
    // the second caller waits for the Flush, so the timeout covers both
    let (result, _second) = tokio::time::timeout(RUN_TIMEOUT, async {
        tokio::join!(src.run(&mut log), callers)
    })
    .await
    .expect("the second caller streams within the timeout");

    let emitted = result.expect("run ok");
    assert_eq!(emitted, 2 * PER_CALLER as u64);
    assert_eq!(log.events, both_callers_then_eos());
}

#[tokio::test]
async fn keep_listening_takes_a_new_caller_after_the_first_goes_silent() {
    let (mut src, listener_addr) = listening_source(true);
    let flushed = Arc::new(tokio::sync::Notify::new());
    let mut log = ReceivedLog {
        events: Vec::new(),
        flushed: flushed.clone(),
    };

    let callers = async {
        drop(caller_sending(listener_addr, 0..PER_CALLER).await);
        flushed.notified().await;
        caller_sending(listener_addr, PER_CALLER..2 * PER_CALLER).await
    };
    // the second caller waits for the Flush, so the timeout covers both
    let (result, _second) = tokio::time::timeout(RUN_TIMEOUT, async {
        tokio::join!(src.run(&mut log), callers)
    })
    .await
    .expect("the second caller streams within the timeout");

    let emitted = result.expect("run ok");
    assert_eq!(emitted, 2 * PER_CALLER as u64);
    assert_eq!(log.events, both_callers_then_eos());
}

#[tokio::test]
async fn a_silent_caller_fails_the_source_without_keep_listening() {
    let (mut src, listener_addr) = listening_source(false);
    let mut log = ReceivedLog {
        events: Vec::new(),
        flushed: Arc::default(),
    };

    let caller = async { drop(caller_sending(listener_addr, 0..PER_CALLER).await) };
    let (result, ()) = tokio::join!(tokio::time::timeout(RUN_TIMEOUT, src.run(&mut log)), caller);

    assert_eq!(
        result.expect("run ends"),
        Err(G2gError::Network(NetworkFailure::Stream))
    );
    let expected: Vec<Received> = (0..PER_CALLER).map(Received::Data).collect();
    assert_eq!(log.events, expected);
}

// replies go to the newest caller address, and nothing passes while `blackhole` is set
async fn blackhole_proxy(
    proxy: tokio::net::UdpSocket,
    listener_addr: SocketAddr,
    blackhole: Arc<AtomicBool>,
    first_timestamps: Arc<std::sync::Mutex<Vec<u32>>>,
) {
    let mut caller = None;
    let mut timestamped_caller = None;
    let mut buf = [0u8; 2048];
    loop {
        let Ok((n, from)) = proxy.recv_from(&mut buf).await else {
            return;
        };
        if blackhole.load(Ordering::Relaxed) {
            continue;
        }
        if from == listener_addr {
            if let Some(caller) = caller {
                let _ = proxy.send_to(&buf[..n], caller).await;
            }
        } else {
            caller = Some(from);
            if let Some(data) = srt::parse_data_packet(&buf[..n]) {
                if timestamped_caller != Some(from) {
                    timestamped_caller = Some(from);
                    first_timestamps.lock().unwrap().push(data.timestamp);
                }
            }
            let _ = proxy.send_to(&buf[..n], listener_addr).await;
        }
    }
}

#[tokio::test]
async fn the_caller_calls_again_after_the_listener_goes_silent() {
    // longer than the 5 s peer idle timeout of both ends
    const SILENCE: std::time::Duration = std::time::Duration::from_millis(5_500);
    const FRAME_INTERVAL: std::time::Duration = std::time::Duration::from_millis(50);
    let (mut src, listener_addr) = listening_source(true);
    let proxy = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let proxy_addr = proxy.local_addr().unwrap();
    let blackhole = Arc::new(AtomicBool::new(false));
    let first_timestamps = Arc::new(std::sync::Mutex::new(Vec::new()));
    tokio::spawn(blackhole_proxy(
        proxy,
        listener_addr,
        blackhole.clone(),
        first_timestamps.clone(),
    ));
    let mut log = ReceivedLog {
        events: Vec::new(),
        flushed: Arc::default(),
    };

    // frames keep coming through the silence, as from a live encoder
    let caller = async {
        let mut sink = caller_sending(proxy_addr, 0..PER_CALLER).await;
        blackhole.store(true, Ordering::Relaxed);
        let blackhole = blackhole.clone();
        // sends block in the new handshake while the proxy is silent
        tokio::spawn(async move {
            tokio::time::sleep(SILENCE).await;
            blackhole.store(false, Ordering::Relaxed);
        });
        for tag in PER_CALLER..=u8::MAX {
            let mut frame = tagged_frame(tag);
            frame.timing.pts_ns = FRAME_INTERVAL.as_nanos() as u64 * u64::from(tag);
            sink.process(PipelinePacket::DataFrame(frame), &mut NullOut)
                .await
                .expect("send");
            tokio::time::sleep(FRAME_INTERVAL).await;
        }
    };
    let result = tokio::time::timeout(RUN_TIMEOUT, async {
        tokio::select! {
            result = src.run(&mut log) => result,
            () = caller => panic!("the listener never took {PER_CALLER} frames from a second call"),
        }
    })
    .await
    .expect("the caller reconnects within the timeout");

    assert_eq!(result.expect("run ok"), 2 * PER_CALLER as u64);
    let first_call: Vec<Received> = (0..PER_CALLER).map(Received::Data).collect();
    assert_eq!(log.events[..first_call.len()], first_call[..]);
    let rest = &log.events[first_call.len()..];
    let Some((Received::Flush, rest)) = rest.split_first() else {
        panic!(
            "the silent first call ends in a flush, got {:?}",
            log.events
        );
    };
    let Some(Received::Data(first_tag)) = rest.first() else {
        panic!("the second call delivers data, got {:?}", log.events);
    };
    let second_call: Vec<Received> = (*first_tag..first_tag + PER_CALLER)
        .map(Received::Data)
        .chain([Received::Eos])
        .collect();
    assert_eq!(
        rest, second_call,
        "the second call delivers consecutive frames"
    );
    assert_eq!(
        first_timestamps.lock().unwrap()[1],
        0,
        "the second call's timestamps count from its own start"
    );
}

const PAYLOADS: u8 = 10;
const PAYLOAD_SPACING: std::time::Duration = std::time::Duration::from_millis(10);
const PASSPHRASE: &str = "g2g_srt_loopback_passphrase";
// the cookie contest is random, so several meetings give each end both roles
const RENDEZVOUS_MEETINGS: usize = 8;

fn free_port() -> u16 {
    StdUdpSocket::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn sink_with_uri(uri: String) -> SrtSink {
    let mut sink = SrtSink::new(([127, 0, 0, 1], 0).into());
    sink.set_property("uri", PropValue::Str(uri)).expect("uri");
    sink.configure_pipeline(&Caps::ByteStream {
        encoding: g2g_core::ByteStreamEncoding::MpegTs,
    })
    .expect("configure");
    sink
}

fn source_with_uri(uri: String) -> SrtSrc {
    let socket = StdUdpSocket::bind("127.0.0.1:0").expect("bind source");
    let mut src = SrtSrc::from_socket(socket)
        .unwrap()
        .with_frame_limit(PAYLOADS.into());
    src.set_property("uri", PropValue::Str(uri)).expect("uri");
    src
}

// sends tags 0, 1, 2... as a live encoder keeps sending, until the source has its payloads
async fn received_from(src: &mut SrtSrc, mut sink: SrtSink) -> Vec<Received> {
    let mut log = ReceivedLog {
        events: Vec::new(),
        flushed: Arc::default(),
    };
    let sending = async {
        for tag in 0..=u8::MAX {
            send_tagged(&mut sink, tag).await.expect("send");
            tokio::time::sleep(PAYLOAD_SPACING).await;
        }
    };
    let result = tokio::time::timeout(RUN_TIMEOUT, async {
        tokio::select! {
            result = src.run(&mut log) => result,
            () = sending => panic!("the source never took {PAYLOADS} payloads"),
        }
    })
    .await
    .expect("streams within the timeout");
    assert_eq!(result.expect("run ok"), u64::from(PAYLOADS));
    log.events
}

fn payloads_then_eos() -> Vec<Received> {
    (0..PAYLOADS)
        .map(Received::Data)
        .chain([Received::Eos])
        .collect()
}

// the listener sends under the AES-256 key the caller made
#[tokio::test]
async fn a_caller_srtsrc_receives_from_a_listening_srtsink() {
    let port = free_port();
    let sink = sink_with_uri(format!(
        "srt://127.0.0.1:{port}?mode=listener&passphrase={PASSPHRASE}"
    ));
    let mut src = source_with_uri(format!(
        "srt://127.0.0.1:{port}?passphrase={PASSPHRASE}&pbkeylen=32"
    ));
    assert_eq!(received_from(&mut src, sink).await, payloads_then_eos());
}

#[tokio::test]
async fn rendezvous_srtsink_and_srtsrc_meet_and_stream() {
    for _ in 0..RENDEZVOUS_MEETINGS {
        let sink_port = free_port();
        let mut src = source_with_uri(format!(
            "srt://127.0.0.1:{sink_port}?mode=rendezvous&passphrase={PASSPHRASE}"
        ));
        let src_port = src.local_port().expect("bound");
        let sink = sink_with_uri(format!(
            "srt://127.0.0.1:{src_port}?mode=rendezvous&localport={sink_port}&passphrase={PASSPHRASE}"
        ));
        assert_eq!(received_from(&mut src, sink).await, payloads_then_eos());
    }
}
