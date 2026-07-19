//! Lossy-link P2P resilience for the native duplex WebRTC session
//! ([`WebRtcDuplexSession`]): two sendrecv peers connect over a test-local UDP
//! relay that deterministically drops ~10% of large video SRTP packets, and we
//! confirm H.264 frame delivery stays near-complete because str0m's NACK / RTX
//! resend path recovers the losses. RTX is negotiated automatically by str0m
//! (each H.264 payload type is offered with an RTX resend PT and `fb_nack`, and
//! `add_media` allocates the RTX SSRC), so this is a proof it works end to end,
//! not new resilience code.
//!
//! Runs fully on localhost UDP with a synthetic H.264 stream (no ffmpeg fixture,
//! no media server), so it is a default CI gate. Two phases:
//!   - relay at 0% drop: proves the relay plumbing (frames flow through it),
//!   - relay at ~10% drop of large video RTP: frames still arrive near-complete
//!     and NACK feedback is observed on both peers.

#![cfg(all(target_os = "linux", feature = "webrtc"))]

use core::future::{ready, Future, Ready};
use core::pin::Pin;
use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::net::UdpSocket;
use tokio::sync::mpsc;

use g2g_core::element::DynAsyncElement;
use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_duplex_session, DynSourceLoop, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, FrameTiming, G2gError, MemoryDomain,
    OutputSink, PipelineClock, PipelinePacket, Rate, VideoCodec,
};
use g2g_plugins::webrtcduplex::{SdpChannel, SignalRole, WebRtcDuplexSession};

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Build a synthetic Annex-B H.264 stream of `frames` access units. Every 30th
/// AU is a keyframe (SPS + PPS + a large IDR slice); the rest are large P slices.
/// The slice payloads are big enough (>MTU) that str0m fragments each AU across
/// several RTP packets, so dropping one packet damages the frame unless RTX
/// recovers it. Payload bytes are all in `1..=251`, so they can never emulate an
/// Annex-B start code (`00 00 01`) and split the NAL wrongly.
fn synthetic_h264_stream(frames: usize) -> Vec<Vec<u8>> {
    const SC: [u8; 4] = [0, 0, 0, 1];
    let pad = |n: usize, salt: u8| -> Vec<u8> {
        (0..n)
            .map(|i| (((i as u32 + salt as u32) % 251) as u8) + 1)
            .collect::<Vec<u8>>()
    };
    let mut aus = Vec::with_capacity(frames);
    for f in 0..frames {
        let mut au = Vec::new();
        if f % 30 == 0 {
            // SPS (type 7), PPS (type 8), IDR slice (type 5, nal_ref_idc = 3).
            au.extend_from_slice(&SC);
            au.extend_from_slice(&[0x67, 0x42, 0x00, 0x1f]);
            au.extend_from_slice(&pad(8, 1));
            au.extend_from_slice(&SC);
            au.extend_from_slice(&[0x68, 0xce, 0x3c, 0x80]);
            au.extend_from_slice(&SC);
            au.push(0x65);
            au.extend_from_slice(&pad(4000, f as u8));
        } else {
            // Non-IDR slice (type 1, nal_ref_idc = 2).
            au.extend_from_slice(&SC);
            au.push(0x41);
            au.extend_from_slice(&pad(1500, f as u8));
        }
        aus.push(au);
    }
    aus
}

/// Source that paces a fixed set of H.264 access units in real time (~30 fps) for
/// `duration`, looping, and counts what it emitted into `sent`.
struct SyntheticH264Src {
    aus: Arc<Vec<Vec<u8>>>,
    duration: Duration,
    interval: Duration,
    sent: Arc<AtomicU64>,
}

impl SourceLoop for SyntheticH264Src {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = Ready<Result<Caps, G2gError>>;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        ready(Ok(h264_caps()))
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(PipelinePacket::CapsChanged(h264_caps())).await?;
            let start = Instant::now();
            let mut seq = 0u64;
            let mut idx = 0usize;
            while Instant::now().duration_since(start) < self.duration {
                let au = self.aus[idx % self.aus.len()].clone();
                idx += 1;
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(au.into_boxed_slice())),
                    FrameTiming {
                        pts_ns: seq * 33_000_000,
                        ..FrameTiming::default()
                    },
                    seq,
                );
                out.push(PipelinePacket::DataFrame(frame)).await?;
                seq += 1;
                self.sent.fetch_add(1, Ordering::Relaxed);
                tokio::time::sleep(self.interval).await;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(seq)
        })
    }
}

/// Sink that counts received `DataFrame`s into a shared atomic.
struct CountingSink {
    frames: Arc<AtomicU64>,
}

impl AsyncElement for CountingSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
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
            if let PipelinePacket::DataFrame(_) = packet {
                self.frames.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        })
    }
}

/// Parse the addr (`ip port`) of the first ICE host candidate in an SDP blob.
fn first_candidate_addr(sdp: &str) -> Option<SocketAddr> {
    for line in sdp.lines() {
        if let Some(rest) = line.trim().strip_prefix("a=candidate:") {
            // "<foundation> <comp> <transport> <prio> <ip> <port> typ host ..."
            let toks: Vec<&str> = rest.split(' ').collect();
            if toks.len() >= 6 {
                let ip: IpAddr = toks[4].parse().ok()?;
                let port: u16 = toks[5].parse().ok()?;
                return Some(SocketAddr::new(ip, port));
            }
        }
    }
    None
}

/// Rewrite every `a=candidate:` line's addr to `relay`, preserving line endings.
fn rewrite_candidates(sdp: &str, relay: SocketAddr) -> String {
    let mut out = String::new();
    for line in sdp.split_inclusive('\n') {
        let (content, ending) = match line.strip_suffix("\r\n") {
            Some(c) => (c, "\r\n"),
            None => match line.strip_suffix('\n') {
                Some(c) => (c, "\n"),
                None => (line, ""),
            },
        };
        if content.starts_with("a=candidate:") {
            let toks: Vec<&str> = content.split(' ').collect();
            if toks.len() >= 6 {
                let mut rebuilt: Vec<String> = toks.iter().map(|s| (*s).to_string()).collect();
                rebuilt[4] = relay.ip().to_string();
                rebuilt[5] = relay.port().to_string();
                out.push_str(&rebuilt.join(" "));
                out.push_str(ending);
                continue;
            }
        }
        out.push_str(content);
        out.push_str(ending);
    }
    out
}

/// True if this datagram is a large video SRTP/RTX packet (first byte in the
/// RTP/SRTP range, payload over 200 bytes), the class we deterministically drop.
/// STUN (first byte 0..=3), DTLS (20..=63), and small RTCP/NACK/audio are left
/// alone so ICE, the handshake, and the loss-recovery feedback all survive.
fn is_droppable_media(pkt: &[u8]) -> bool {
    matches!(pkt.first(), Some(b) if (128..=191).contains(b)) && pkt.len() > 200
}

/// Shared relay counters.
#[derive(Default)]
struct RelayStats {
    forwarded: AtomicU64,
    dropped: AtomicU64,
}

/// Forward datagrams between the two relay sockets, dropping every `drop_every`-th
/// droppable media packet (`None` = drop nothing). `sock_a` faces peer A (which
/// sends here and expects replies from here); `sock_b` faces peer B.
async fn relay_forward(
    sock_a: UdpSocket,
    sock_b: UdpSocket,
    addr_a: SocketAddr,
    addr_b: Arc<Mutex<Option<SocketAddr>>>,
    drop_every: Option<u64>,
    stats: Arc<RelayStats>,
) {
    let matched = AtomicU64::new(0);
    let should_drop = |pkt: &[u8]| -> bool {
        let Some(every) = drop_every else {
            return false;
        };
        if is_droppable_media(pkt) {
            let n = matched.fetch_add(1, Ordering::Relaxed);
            return n % every == every - 1;
        }
        false
    };
    let mut buf_a = [0u8; 2048];
    let mut buf_b = [0u8; 2048];
    loop {
        tokio::select! {
            r = sock_a.recv_from(&mut buf_a) => {
                let Ok((n, _)) = r else { return };
                if should_drop(&buf_a[..n]) {
                    stats.dropped.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                let dest_b = *addr_b.lock().unwrap();
                if let Some(b) = dest_b {
                    let _ = sock_b.send_to(&buf_a[..n], b).await;
                    stats.forwarded.fetch_add(1, Ordering::Relaxed);
                }
            }
            r = sock_b.recv_from(&mut buf_b) => {
                let Ok((n, _)) = r else { return };
                if should_drop(&buf_b[..n]) {
                    stats.dropped.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                let _ = sock_a.send_to(&buf_b[..n], addr_a).await;
                stats.forwarded.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// Splice a candidate-rewriting relay between the offerer and answerer SDP
/// channels: parse each peer's real host candidate, bind two relay sockets, point
/// each peer at the relay, and forward media (dropping `drop_every`-th large video
/// packet). Returns the two spliced channels plus the relay stats handle.
async fn spliced_channels(drop_every: Option<u64>) -> (SdpChannel, SdpChannel, Arc<RelayStats>) {
    let (off_out_tx, mut off_out_rx) = mpsc::channel::<String>(4);
    let (relay_to_off_tx, relay_to_off_rx) = mpsc::channel::<String>(4);
    let (ans_out_tx, mut ans_out_rx) = mpsc::channel::<String>(4);
    let (relay_to_ans_tx, relay_to_ans_rx) = mpsc::channel::<String>(4);

    let off_chan = SdpChannel::from_halves(off_out_tx, relay_to_off_rx);
    let ans_chan = SdpChannel::from_halves(ans_out_tx, relay_to_ans_rx);

    let stats = Arc::new(RelayStats::default());
    let stats_task = stats.clone();

    tokio::spawn(async move {
        // Offer from A carries A's real candidate; stand up the relay on A's IP.
        let offer = off_out_rx.recv().await.expect("offer");
        let addr_a = first_candidate_addr(&offer).expect("offer candidate");
        let sock_a = UdpSocket::bind((addr_a.ip(), 0))
            .await
            .expect("bind sock_a");
        let sock_b = UdpSocket::bind((addr_a.ip(), 0))
            .await
            .expect("bind sock_b");
        let relay_for_a = sock_a.local_addr().expect("sock_a addr");
        let relay_for_b = sock_b.local_addr().expect("sock_b addr");

        // B will send to sock_b; forward B -> A out of sock_a to addr_a.
        let addr_b = Arc::new(Mutex::new(None));
        tokio::spawn(relay_forward(
            sock_a,
            sock_b,
            addr_a,
            addr_b.clone(),
            drop_every,
            stats_task,
        ));

        let offer = rewrite_candidates(&offer, relay_for_b);
        relay_to_ans_tx.send(offer).await.expect("forward offer");

        // Answer from B carries B's real candidate; A will send to sock_a.
        let answer = ans_out_rx.recv().await.expect("answer");
        let b = first_candidate_addr(&answer).expect("answer candidate");
        *addr_b.lock().unwrap() = Some(b);
        let answer = rewrite_candidates(&answer, relay_for_a);
        relay_to_off_tx.send(answer).await.expect("forward answer");
    });

    (off_chan, ans_chan, stats)
}

/// Per-run result: received frame counts, sent counts, per-peer peak NACKs, and
/// relay traffic.
struct RunResult {
    a_recv: u64,
    b_recv: u64,
    a_sent: u64,
    b_sent: u64,
    a_nacks: u64,
    b_nacks: u64,
    forwarded: u64,
    dropped: u64,
}

/// Run two duplex peers (video-only sendrecv) through a lossy relay for
/// `media_secs` of media and report the outcome.
async fn run_over_relay(drop_every: Option<u64>, media_secs: u64) -> RunResult {
    let aus = Arc::new(synthetic_h264_stream(240));
    let (off_chan, ans_chan, stats) = spliced_channels(drop_every).await;

    let mut sess_a = WebRtcDuplexSession::new(SignalRole::Offerer, off_chan, 1);
    let mut sess_b = WebRtcDuplexSession::new(SignalRole::Answerer, ans_chan, 1);

    let a_recv = Arc::new(AtomicU64::new(0));
    let b_recv = Arc::new(AtomicU64::new(0));
    let a_sent = Arc::new(AtomicU64::new(0));
    let b_sent = Arc::new(AtomicU64::new(0));
    let clock = ZeroClock;
    let clock_ref = &clock;
    let duration = Duration::from_secs(media_secs);
    let interval = Duration::from_millis(33);

    let peer_a = {
        let (aus, recv, sent) = (aus.clone(), a_recv.clone(), a_sent.clone());
        let sess = &mut sess_a;
        async move {
            let mut src = SyntheticH264Src {
                aus,
                duration,
                interval,
                sent,
            };
            let mut sink = CountingSink { frames: recv };
            let sources: Vec<&mut dyn DynSourceLoop> = std::vec![&mut src];
            let sinks: Vec<&mut dyn DynAsyncElement> = std::vec![&mut sink];
            run_duplex_session(sources, sess, sinks, clock_ref, 8).await
        }
    };
    let peer_b = {
        let (aus, recv, sent) = (aus.clone(), b_recv.clone(), b_sent.clone());
        let sess = &mut sess_b;
        async move {
            let mut src = SyntheticH264Src {
                aus,
                duration,
                interval,
                sent,
            };
            let mut sink = CountingSink { frames: recv };
            let sources: Vec<&mut dyn DynSourceLoop> = std::vec![&mut src];
            let sinks: Vec<&mut dyn DynAsyncElement> = std::vec![&mut sink];
            run_duplex_session(sources, sess, sinks, clock_ref, 8).await
        }
    };

    let (ra, rb) = tokio::time::timeout(Duration::from_secs(media_secs + 25), async {
        tokio::join!(peer_a, peer_b)
    })
    .await
    .expect("P2P loss run completes in time");
    ra.expect("peer A duplex ok");
    rb.expect("peer B duplex ok");

    RunResult {
        a_recv: a_recv.load(Ordering::SeqCst),
        b_recv: b_recv.load(Ordering::SeqCst),
        a_sent: a_sent.load(Ordering::Relaxed),
        b_sent: b_sent.load(Ordering::Relaxed),
        a_nacks: sess_a.nacks_seen(),
        b_nacks: sess_b.nacks_seen(),
        forwarded: stats.forwarded.load(Ordering::Relaxed),
        dropped: stats.dropped.load(Ordering::Relaxed),
    }
}

/// Baseline: relay in the media path, no drops. Proves the relay plumbing, i.e.
/// both peers connect through it and receive the other's frames.
#[tokio::test]
async fn duplex_relay_no_loss_delivers_frames() {
    let r = run_over_relay(None, 6).await;
    eprintln!(
        "no-loss relay: A recv {}/{} sent, B recv {}/{} sent; relay forwarded {} dropped {}",
        r.a_recv, r.b_sent, r.b_recv, r.a_sent, r.forwarded, r.dropped
    );
    assert_eq!(r.dropped, 0, "no-loss phase must not drop");
    assert!(r.forwarded > 0, "relay should have forwarded media");
    assert!(
        r.a_recv >= 60 && r.b_recv >= 60,
        "both peers should receive most frames through the relay (A={}, B={})",
        r.a_recv,
        r.b_recv
    );
}

/// Lossy link: relay drops ~10% of large video RTP packets. str0m's NACK / RTX
/// recovers them, so frame delivery stays near-complete, and NACK feedback is
/// observed on both peers.
#[tokio::test]
async fn duplex_relay_with_loss_recovers_via_rtx() {
    let r = run_over_relay(Some(10), 8).await;
    eprintln!(
        "lossy relay: A recv {}/{} sent, B recv {}/{} sent; nacks A={} B={}; relay forwarded {} dropped {}",
        r.a_recv, r.b_sent, r.b_recv, r.a_sent, r.a_nacks, r.b_nacks, r.forwarded, r.dropped
    );
    assert!(r.dropped > 0, "loss phase should have dropped packets");
    // With ~10% packet loss and working RTX, delivery should stay near-complete.
    // Threshold is conservative to absorb warmup + timing jitter, but well above
    // what survives without resends (many frames die from missing fragments).
    let a_target = (r.b_sent as f64 * 0.7) as u64;
    let b_target = (r.a_sent as f64 * 0.7) as u64;
    assert!(
        r.a_recv >= a_target,
        "peer A got {} frames, expected >= {} (70% of {} sent by B) under loss+RTX",
        r.a_recv,
        a_target,
        r.b_sent
    );
    assert!(
        r.b_recv >= b_target,
        "peer B got {} frames, expected >= {} (70% of {} sent by A) under loss+RTX",
        r.b_recv,
        b_target,
        r.a_sent
    );
    // NACK feedback must have flowed (proves the recovery path was exercised).
    assert!(
        r.a_nacks > 0 && r.b_nacks > 0,
        "expected NACKs on both peers under loss (A={}, B={})",
        r.a_nacks,
        r.b_nacks
    );
}
