// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! Lossy-link P2P resilience for the native duplex WebRTC session
//! ([`WebRtcDuplexSession`]): two sendrecv peers connect over a test-local UDP
//! relay that deterministically drops ~10% of original video SRTP packets, and we
//! confirm H.264 frame delivery stays near-complete because str0m's NACK / RTX
//! resend path recovers the losses. RTX is negotiated automatically by str0m
//! (each H.264 payload type is offered with an RTX resend PT and `fb_nack`, and
//! `add_media` allocates the RTX SSRC), so this is a proof it works end to end,
//! not new resilience code.
//!
//! Runs fully on localhost UDP with a synthetic H.264 stream (no ffmpeg fixture,
//! no media server), so it is a default CI gate:
//!   - relay at 0% drop: proves the relay plumbing (frames flow through it),
//!   - relay at ~10% drop of large video RTP: frames still arrive near-complete
//!     and NACK feedback is observed on both peers;
//!   - receive-only with bounded single-packet frame loss: every frame after
//!     startup arrives, and NACK feedback is observed.

#![cfg(all(target_os = "linux", feature = "webrtc"))]

mod webrtc_duplex_common;
mod webrtc_streams_common;

use g2g_plugins::webrtc_codec::WebRtcVideoCodec;
use webrtc_duplex_common::{PacedSrc, Received, RecordingSink};
use webrtc_streams_common::{synthetic_stream, synthetic_stream_sized, Sizes, KEYFRAME_PAYLOAD};

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
use g2g_core::rtp::RtpHeader;
use g2g_core::runtime::{run_duplex_session, DynSourceLoop, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, FrameTiming, G2gError, MemoryDomain,
    OutputSink, PipelineClock, PipelinePacket, Rate, VideoCodec,
};
use g2g_plugins::sdp::SdpMedia;
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
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        pixel_aspect: g2g_core::PixelAspect::Any,
        alignment: g2g_core::BitstreamAlignment::Unspecified,
    }
}

/// H.264 units bigger than the MTU (see `webrtc_streams_common`), so one
/// dropped packet damages a frame unless RTX recovers it.
fn synthetic_h264_stream(frames: usize) -> Vec<Vec<u8>> {
    synthetic_stream(WebRtcVideoCodec::H264, frames)
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

#[derive(Clone, Copy)]
enum VideoLoss {
    Sustained { every: u64 },
    SinglePacketFrames { every: u64, count: u64 },
}

/// Shared relay counters.
#[derive(Default)]
struct RelayStats {
    forwarded: AtomicU64,
    dropped: AtomicU64,
}

/// Forward datagrams between the two relay sockets and inject loss into the
/// negotiated original video payload type. RTX and control traffic pass through.
/// `sock_a` faces peer A (which sends here and expects replies from here);
/// `sock_b` faces peer B.
async fn relay_forward(
    sock_a: UdpSocket,
    sock_b: UdpSocket,
    addr_a: SocketAddr,
    addr_b: SocketAddr,
    video_pt: u8,
    loss: Option<VideoLoss>,
    stats: Arc<RelayStats>,
) {
    let mut matched = 0u64;
    let mut should_drop = |pkt: &[u8], previous_timestamp: &mut Option<u32>| -> bool {
        let Some(loss) = loss else { return false };
        let Some(parsed) = RtpHeader::parse_header(pkt) else {
            return false;
        };
        let header = parsed.header;
        if header.payload_type != video_pt {
            return false;
        }
        let starts_frame = previous_timestamp.replace(header.timestamp) != Some(header.timestamp);
        let every = match loss {
            VideoLoss::Sustained { every } if pkt.len() > 200 => every,
            VideoLoss::SinglePacketFrames { every, count }
                if starts_frame
                    && header.marker
                    && stats.dropped.load(Ordering::Relaxed) < count =>
            {
                every
            }
            _ => return false,
        };
        matched += 1;
        matched.is_multiple_of(every)
    };
    let mut previous_a = None;
    let mut previous_b = None;
    let mut buf_a = [0u8; 2048];
    let mut buf_b = [0u8; 2048];
    loop {
        tokio::select! {
            r = sock_a.recv_from(&mut buf_a) => {
                let Ok((n, _)) = r else { return };
                if should_drop(&buf_a[..n], &mut previous_a) {
                    stats.dropped.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                let _ = sock_b.send_to(&buf_a[..n], addr_b).await;
                stats.forwarded.fetch_add(1, Ordering::Relaxed);
            }
            r = sock_b.recv_from(&mut buf_b) => {
                let Ok((n, _)) = r else { return };
                if should_drop(&buf_b[..n], &mut previous_b) {
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
/// each peer at the relay, and forward media with the requested loss schedule.
/// Returns the two spliced channels plus the relay stats handle.
async fn spliced_channels(loss: Option<VideoLoss>) -> (SdpChannel, SdpChannel, Arc<RelayStats>) {
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

        let offer = rewrite_candidates(&offer, relay_for_b);
        relay_to_ans_tx.send(offer).await.expect("forward offer");

        // Answer from B carries B's address and the accepted video payload type.
        let answer = ans_out_rx.recv().await.expect("answer");
        let addr_b = first_candidate_addr(&answer).expect("answer candidate");
        let video_pt = SdpMedia::parse(&answer).expect("answer video").payload_type;
        tokio::spawn(relay_forward(
            sock_a, sock_b, addr_a, addr_b, video_pt, loss, stats_task,
        ));
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
    let (off_chan, ans_chan, stats) =
        spliced_channels(drop_every.map(|every| VideoLoss::Sustained { every })).await;

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

// one RTP packet carries the whole unit, as it does for a static desktop's P-frames
const ONE_PACKET_INTER_PAYLOAD: usize = 700;
const RECEIVE_ONLY_FRAMES: usize = 120;
const RECEIVE_ONLY_DROP_EVERY: u64 = 11;
// Forward a loss-free tail, so later sequence numbers expose every injected gap.
const RECEIVE_ONLY_DROPS: u64 = 6;
const RECEIVE_ONLY_TIMEOUT: Duration = Duration::from_secs(30);
const LINK_CAPACITY: usize = 8;

#[tokio::test]
async fn receive_only_client_recovers_lost_single_packet_frames() {
    let units = Arc::new(synthetic_stream_sized(
        WebRtcVideoCodec::H264,
        RECEIVE_ONLY_FRAMES,
        Sizes {
            keyframe: KEYFRAME_PAYLOAD,
            inter: ONE_PACKET_INTER_PAYLOAD,
        },
    ));
    let (offer_channel, answer_channel, stats) =
        spliced_channels(Some(VideoLoss::SinglePacketFrames {
            every: RECEIVE_ONLY_DROP_EVERY,
            count: RECEIVE_ONLY_DROPS,
        }))
        .await;
    let mut client =
        WebRtcDuplexSession::new(SignalRole::Offerer, offer_channel, 1).with_receive_only();
    let mut host = WebRtcDuplexSession::new(SignalRole::Answerer, answer_channel, 1);
    let received = Arc::new(Mutex::new(Received::default()));
    let mut client_sink = RecordingSink {
        received: received.clone(),
    };
    let mut host_src = PacedSrc {
        codec: WebRtcVideoCodec::H264,
        units: units.clone(),
        first_pts_ns: 0,
    };
    let mut host_sink = RecordingSink {
        received: Arc::new(Mutex::new(Received::default())),
    };
    let clock = ZeroClock;
    let run_client = async {
        let sources: Vec<&mut dyn DynSourceLoop> = Vec::new();
        let sinks: Vec<&mut dyn DynAsyncElement> = std::vec![&mut client_sink];
        run_duplex_session(sources, &mut client, sinks, &clock, LINK_CAPACITY).await
    };
    let run_host = async {
        let sources: Vec<&mut dyn DynSourceLoop> = std::vec![&mut host_src];
        let sinks: Vec<&mut dyn DynAsyncElement> = std::vec![&mut host_sink];
        run_duplex_session(sources, &mut host, sinks, &clock, LINK_CAPACITY).await
    };
    let (client_result, host_result) = tokio::time::timeout(RECEIVE_ONLY_TIMEOUT, async {
        tokio::join!(run_client, run_host)
    })
    .await
    .expect("both sessions end after the host's units run out");
    client_result.expect("client session");
    host_result.expect("host session");

    let indices: Vec<usize> = received
        .lock()
        .unwrap()
        .units
        .iter()
        .map(|(bytes, _)| {
            units
                .iter()
                .position(|unit| unit == bytes)
                .expect("each received frame is a sent unit")
        })
        .collect();
    let first = *indices.first().expect("frames arrive");
    let missing: Vec<usize> = (first..units.len())
        .filter(|index| !indices.contains(index))
        .collect();
    assert_eq!(
        stats.dropped.load(Ordering::Relaxed),
        RECEIVE_ONLY_DROPS,
        "the relay injected every scheduled loss"
    );
    assert!(
        client.nacks_seen() > 0 && host.nacks_seen() > 0,
        "NACKs crossed the relay"
    );
    assert!(missing.is_empty(), "frames {missing:?} never arrived");
}
