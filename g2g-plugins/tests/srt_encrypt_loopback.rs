//! SRT encrypted end-to-end over loopback: a caller (`SrtSink`) and listener
//! (`SrtSrc`) sharing a passphrase. The random stream key is wrapped under the
//! passphrase-derived KEK and exchanged in the handshake KM extension; payloads
//! are AES-CTR encrypted. Proves the KM exchange + AES-CTR round-trip, including
//! across a NAK retransmission of an encrypted packet, and that a wrong
//! passphrase fails to derive the key (g2g <-> g2g; real libsrt interop is
//! operator-validated).
#![cfg(feature = "srt")]

use core::future::Future;
use core::pin::Pin;
use std::net::{SocketAddr, UdpSocket as StdUdpSocket};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_simple_pipeline, LatencyProfile};
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, ConfigureOutcome, FrameTiming, G2gError, MemoryDomain,
    OutputSink, PipelineClock, PipelinePacket, PushOutcome,
};

use g2g_plugins::srt;
use g2g_plugins::srtsink::SrtSink;
use g2g_plugins::srtsrc::SrtSrc;

const PASS: &str = "correct horse battery staple";

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Records the first byte of each received payload (its index tag). If the
/// payload is not decrypted, the tag bytes come out as ciphertext garbage, not
/// the 0..N sequence, so this also proves decryption.
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

struct NullOut;
impl OutputSink for NullOut {
    fn push<'a>(
        &'a mut self,
        _p: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async { Ok(PushOutcome::Accepted) })
    }
}

/// Relay caller<->listener, dropping the SRT data packet with `drop_seq` once
/// (the encrypted ciphertext packet, exercising encrypted retransmission).
async fn lossy_proxy(proxy: tokio::net::UdpSocket, listener_addr: SocketAddr, drop_seq: u32) {
    let mut caller: Option<SocketAddr> = None;
    let mut dropped = false;
    let mut buf = [0u8; 2048];
    loop {
        let Ok((n, from)) = proxy.recv_from(&mut buf).await else { return };
        if Some(from) == caller || (caller.is_none() && from != listener_addr) {
            caller = Some(from);
            if !srt::is_control(&buf[..n]) {
                if let Some(d) = srt::parse_data_packet(&buf[..n]) {
                    if d.seq == drop_seq && !dropped {
                        dropped = true;
                        continue;
                    }
                }
            }
            let _ = proxy.send_to(&buf[..n], listener_addr).await;
        } else if let Some(dest) = caller {
            let _ = proxy.send_to(&buf[..n], dest).await;
        }
    }
}

/// Drive the caller, sending one tagged payload per frame until it can't.
async fn drive_caller(mut sink: SrtSink, n: u8) -> u64 {
    sink.configure_pipeline(&Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs })
        .expect("configure");
    let mut null = NullOut;
    for i in 0u8..(n * 2) {
        let payload = vec![i % n; 100];
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(payload.into_boxed_slice())),
            timing: FrameTiming { pts_ns: i as u64 * 10_000_000, ..FrameTiming::default() },
            sequence: i as u64,
            meta: Default::default(),
        };
        if sink.process(PipelinePacket::DataFrame(frame), &mut null).await.is_err() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(8)).await;
    }
    sink.retransmits()
}

#[tokio::test]
async fn encrypted_stream_round_trips_through_lossy_proxy() {
    const N: u8 = 12;

    let listener_std = StdUdpSocket::bind("127.0.0.1:0").expect("bind listener");
    let listener_addr = listener_std.local_addr().unwrap();
    let proxy = tokio::net::UdpSocket::bind("127.0.0.1:0").await.expect("bind proxy");
    let proxy_addr = proxy.local_addr().unwrap();
    let proxy_task = tokio::spawn(lossy_proxy(proxy, listener_addr, 3));

    let mut src =
        SrtSrc::from_socket(listener_std).unwrap().with_frame_limit(N as u64).with_passphrase(PASS);
    let mut sink_collect = TagSink::default();
    let clock = ZeroClock;

    let caller = drive_caller(SrtSink::new(proxy_addr).with_passphrase(PASS), N);

    let recv = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        run_simple_pipeline(&mut src, &mut sink_collect, &clock, LatencyProfile::Live.link_capacity()),
    );

    let (recv_res, retransmits) = tokio::join!(recv, caller);
    proxy_task.abort();

    let stats = recv_res.expect("listener finishes within 15s").expect("receive pipeline ok");
    assert_eq!(stats.frames_emitted, N as u64, "every encrypted payload delivered despite the drop");
    let expected: Vec<u8> = (0..N).collect();
    assert_eq!(
        sink_collect.tags, expected,
        "payloads decrypt to the original tags in order (incl. the retransmitted one)"
    );
    assert!(retransmits >= 1, "the dropped encrypted packet was retransmitted on NAK");
}

#[tokio::test]
async fn aes256_encrypted_stream_round_trips() {
    // AES-256 (opt-in on the caller): the key size rides in the KM KLen field, so
    // the listener recovers a 256-bit key from the handshake with no extra config
    // and the payloads decrypt to their original tags. Direct connection (the KM
    // exchange + AES-256 CTR is the unit under test, not loss recovery).
    const N: u8 = 10;

    let listener_std = StdUdpSocket::bind("127.0.0.1:0").expect("bind listener");
    let listener_addr = listener_std.local_addr().unwrap();

    let mut src =
        SrtSrc::from_socket(listener_std).unwrap().with_frame_limit(N as u64).with_passphrase(PASS);
    let mut sink_collect = TagSink::default();
    let clock = ZeroClock;

    let caller = drive_caller(SrtSink::new(listener_addr).with_passphrase(PASS).with_aes256(), N);

    let recv = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        run_simple_pipeline(&mut src, &mut sink_collect, &clock, LatencyProfile::Live.link_capacity()),
    );
    let (recv_res, _) = tokio::join!(recv, caller);

    let stats = recv_res.expect("listener finishes within 15s").expect("receive pipeline ok");
    assert_eq!(stats.frames_emitted, N as u64, "every AES-256 payload delivered");
    let expected: Vec<u8> = (0..N).collect();
    assert_eq!(sink_collect.tags, expected, "AES-256 payloads decrypt to the original tags in order");
}

#[tokio::test]
async fn key_rotation_rolls_the_cipher_mid_stream() {
    // The caller rekeys every 3 packets: the cipher key changes several times
    // over the stream, alternating even/odd parity, and the listener installs
    // each new key from the KM control packet and keeps decrypting in order.
    const N: u8 = 12;

    let listener_std = StdUdpSocket::bind("127.0.0.1:0").expect("bind listener");
    let listener_addr = listener_std.local_addr().unwrap();

    let mut src =
        SrtSrc::from_socket(listener_std).unwrap().with_frame_limit(N as u64).with_passphrase(PASS);
    let mut sink_collect = TagSink::default();
    let clock = ZeroClock;

    let caller = async {
        let mut sink =
            SrtSink::new(listener_addr).with_passphrase(PASS).with_key_rotation(3);
        sink.configure_pipeline(&Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs })
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
            if sink.process(PipelinePacket::DataFrame(frame), &mut null).await.is_err() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(8)).await;
        }
        sink.rekeys()
    };

    let recv = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        run_simple_pipeline(&mut src, &mut sink_collect, &clock, LatencyProfile::Live.link_capacity()),
    );

    let (recv_res, rekeys) = tokio::join!(recv, caller);

    let stats = recv_res.expect("listener finishes within 15s").expect("receive pipeline ok");
    assert_eq!(stats.frames_emitted, N as u64, "every payload delivered across the rekeys");
    let expected: Vec<u8> = (0..N).collect();
    assert_eq!(
        sink_collect.tags, expected,
        "payloads decrypt to the original tags in order across multiple key rotations"
    );
    assert!(rekeys >= 2, "the cipher rotated mid-stream at least twice (got {rekeys})");
}

#[tokio::test]
async fn wrong_passphrase_fails_to_decrypt() {
    const N: u8 = 6;

    let listener_std = StdUdpSocket::bind("127.0.0.1:0").expect("bind listener");
    let listener_addr = listener_std.local_addr().unwrap();

    // Listener's passphrase differs from the caller's: the KM unwrap (AES key
    // wrap integrity check) fails, so the receive pipeline errors out.
    let mut src = SrtSrc::from_socket(listener_std)
        .unwrap()
        .with_frame_limit(N as u64)
        .with_passphrase("a different secret");
    let mut sink_collect = TagSink::default();
    let clock = ZeroClock;

    let caller = drive_caller(SrtSink::new(listener_addr).with_passphrase(PASS), N);

    let recv = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        run_simple_pipeline(&mut src, &mut sink_collect, &clock, LatencyProfile::Live.link_capacity()),
    );
    let (recv_res, _) = tokio::join!(recv, caller);

    let inner = recv_res.expect("listener finishes within 15s");
    assert!(inner.is_err(), "a wrong passphrase must fail the stream, not deliver garbage");
}

/// Relay caller<->listener, dropping the first KMREQ (rekey Keying Material)
/// caller->listener once, to force the sender's retransmit-until-KMRSP path.
async fn drop_first_kmreq_proxy(proxy: tokio::net::UdpSocket, listener_addr: SocketAddr) {
    let mut caller: Option<SocketAddr> = None;
    let mut dropped = false;
    let mut buf = [0u8; 2048];
    loop {
        let Ok((n, from)) = proxy.recv_from(&mut buf).await else { return };
        if Some(from) == caller || (caller.is_none() && from != listener_addr) {
            caller = Some(from);
            if !dropped && srt::is_control(&buf[..n]) {
                if let Some(srt::Control::KeyMaterial { rsp: false, .. }) =
                    srt::parse_control(&buf[..n])
                {
                    dropped = true;
                    continue; // drop this one KMREQ
                }
            }
            let _ = proxy.send_to(&buf[..n], listener_addr).await;
        } else if let Some(dest) = caller {
            let _ = proxy.send_to(&buf[..n], dest).await;
        }
    }
}

/// Drive the caller sending `n*2` tagged frames with key rotation, returning the
/// finished sink so the test can read its rekey / KM-retransmit counters.
async fn drive_rotating_caller(mut sink: SrtSink, n: u8) -> SrtSink {
    sink.configure_pipeline(&Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs })
        .expect("configure");
    let mut null = NullOut;
    for i in 0u8..(n * 2) {
        let payload = vec![i % n; 100];
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(payload.into_boxed_slice())),
            timing: FrameTiming { pts_ns: i as u64 * 10_000_000, ..FrameTiming::default() },
            sequence: i as u64,
            meta: Default::default(),
        };
        if sink.process(PipelinePacket::DataFrame(frame), &mut null).await.is_err() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(8)).await;
    }
    sink
}

/// Relay caller<->listener, counting the KMRSP (rekey acknowledgement) control
/// packets the listener sends back. Deterministic (packet-count, not timing).
async fn count_kmrsp_proxy(
    proxy: tokio::net::UdpSocket,
    listener_addr: SocketAddr,
    kmrsps: Arc<AtomicU64>,
) {
    let mut caller: Option<SocketAddr> = None;
    let mut buf = [0u8; 2048];
    loop {
        let Ok((n, from)) = proxy.recv_from(&mut buf).await else { return };
        if from == listener_addr {
            if srt::is_control(&buf[..n]) {
                if let Some(srt::Control::KeyMaterial { rsp: true, .. }) =
                    srt::parse_control(&buf[..n])
                {
                    kmrsps.fetch_add(1, Ordering::Relaxed);
                }
            }
            if let Some(dest) = caller {
                let _ = proxy.send_to(&buf[..n], dest).await;
            }
        } else {
            caller = Some(from);
            let _ = proxy.send_to(&buf[..n], listener_addr).await;
        }
    }
}

#[tokio::test]
async fn receiver_kmrsps_each_rekey() {
    // Clean loopback with rotation: the listener must return a KMRSP for each
    // rekey it installs (proven by counting KMRSPs at the proxy, timing-free).
    const N: u8 = 12;

    let listener_std = StdUdpSocket::bind("127.0.0.1:0").expect("bind listener");
    let listener_addr = listener_std.local_addr().unwrap();
    let proxy = tokio::net::UdpSocket::bind("127.0.0.1:0").await.expect("bind proxy");
    let proxy_addr = proxy.local_addr().unwrap();
    let kmrsps = Arc::new(AtomicU64::new(0));
    let proxy_task = tokio::spawn(count_kmrsp_proxy(proxy, listener_addr, kmrsps.clone()));

    let mut src =
        SrtSrc::from_socket(listener_std).unwrap().with_frame_limit(N as u64).with_passphrase(PASS);
    let mut sink_collect = TagSink::default();
    let clock = ZeroClock;

    let caller =
        drive_rotating_caller(SrtSink::new(proxy_addr).with_passphrase(PASS).with_key_rotation(3), N);

    let recv = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        run_simple_pipeline(&mut src, &mut sink_collect, &clock, LatencyProfile::Live.link_capacity()),
    );
    let (recv_res, sink) = tokio::join!(recv, caller);
    proxy_task.abort();

    let stats = recv_res.expect("listener finishes within 15s").expect("receive pipeline ok");
    assert_eq!(stats.frames_emitted, N as u64, "every payload delivered across the rekeys");
    assert!(sink.rekeys() >= 2, "the cipher rotated at least twice");
    // Delivering all N frames required installing the intervening rekeys, each of
    // which the listener must have KMRSP'd. (The sender rotates over the extra
    // 2N sent frames too, so KMRSP count tracks the receiver's ~N/3 rekeys, not
    // the sender's total.)
    assert!(
        kmrsps.load(Ordering::Relaxed) >= 2,
        "listener KMRSP'd its installed rekeys (saw {})",
        kmrsps.load(Ordering::Relaxed)
    );
}

#[tokio::test]
async fn dropped_rekey_km_is_retransmitted_and_recovers() {
    // The first rekey KMREQ is dropped by the proxy: without retransmission the
    // receiver would never install that key. The sender resends the KM until the
    // KMRSP arrives, so the stream recovers and completes.
    const N: u8 = 30;

    let listener_std = StdUdpSocket::bind("127.0.0.1:0").expect("bind listener");
    let listener_addr = listener_std.local_addr().unwrap();
    let proxy = tokio::net::UdpSocket::bind("127.0.0.1:0").await.expect("bind proxy");
    let proxy_addr = proxy.local_addr().unwrap();
    let proxy_task = tokio::spawn(drop_first_kmreq_proxy(proxy, listener_addr));

    let mut src =
        SrtSrc::from_socket(listener_std).unwrap().with_frame_limit(N as u64).with_passphrase(PASS);
    let mut sink_collect = TagSink::default();
    let clock = ZeroClock;

    let caller =
        drive_rotating_caller(SrtSink::new(proxy_addr).with_passphrase(PASS).with_key_rotation(3), N);

    let recv = tokio::time::timeout(
        std::time::Duration::from_secs(20),
        run_simple_pipeline(&mut src, &mut sink_collect, &clock, LatencyProfile::Live.link_capacity()),
    );
    let (recv_res, sink) = tokio::join!(recv, caller);
    proxy_task.abort();

    let stats = recv_res.expect("listener finishes within 20s").expect("receive pipeline ok");
    assert_eq!(stats.frames_emitted, N as u64, "stream recovered and completed after the dropped rekey KM");
    assert!(sink.km_retransmits() >= 1, "the dropped rekey KM was retransmitted until acknowledged");
}
