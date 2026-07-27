//! Mid-session track REMOVE and pad recycling on the duplex WebRTC session
//! (M785), the inverse of M784's spare-pad ADD. `DuplexControl::remove_track`
//! stops the track's m-line (port 0, out of the BUNDLE group), which frees its
//! pads on BOTH peers: the removed track stops arriving, and the freed pad is
//! claimable again by a later track, which negotiates a NEW m-line (a stopped
//! one never reactivates) and re-announces its caps.
//!
//! Three scenarios, one test each: remove, recycle, and a raced ADD where both
//! peers offer a new track at once (the answerer yields, retracts its own offer,
//! and both still end up with the peer's track).
//!
//! Runs fully on localhost UDP with a synthetic H.264 / Opus stream (no fixture,
//! no media server), so it is a default CI gate under the `webrtc` feature.

#![cfg(all(target_os = "linux", feature = "webrtc"))]

use core::future::{ready, Future, Ready};
use core::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use g2g_core::element::DynAsyncElement;
use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_duplex_session, DynSourceLoop, SourceLoop};
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, ConfigureOutcome, Dim, FrameTiming, G2gError,
    MemoryDomain, OutputSink, PipelineClock, PipelinePacket, Rate, VideoCodec,
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

fn opus_caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::Opus,
        channels: 2,
        sample_rate: 48_000,
    }
}

/// Synthetic Annex-B H.264 access units: every 30th is a keyframe (SPS + PPS +
/// IDR slice), the rest are P slices. Payload bytes stay in `1..=251` so they can
/// never emulate a start code.
fn synthetic_h264_stream(frames: usize) -> Vec<Vec<u8>> {
    const SC: [u8; 4] = [0, 0, 0, 1];
    let pad = |n: usize, salt: u8| -> Vec<u8> {
        (0..n)
            .map(|i| (((i as u32 + salt as u32) % 251) as u8) + 1)
            .collect()
    };
    (0..frames)
        .map(|f| {
            let mut au = Vec::new();
            if f % 30 == 0 {
                au.extend_from_slice(&SC);
                au.extend_from_slice(&[0x67, 0x42, 0x00, 0x1f]);
                au.extend_from_slice(&pad(8, 1));
                au.extend_from_slice(&SC);
                au.extend_from_slice(&[0x68, 0xce, 0x3c, 0x80]);
                au.extend_from_slice(&SC);
                au.push(0x65);
                au.extend_from_slice(&pad(600, f as u8));
            } else {
                au.extend_from_slice(&SC);
                au.push(0x41);
                au.extend_from_slice(&pad(400, f as u8));
            }
            au
        })
        .collect()
}

/// Opus is packetized as-is by str0m, so any bytes round-trip.
fn synthetic_opus_stream(packets: usize) -> Vec<Vec<u8>> {
    (0..packets)
        .map(|p| (0..80u8).map(|i| i.wrapping_add(p as u8)).collect())
        .collect()
}

/// Paces `payloads` in real time for `duration`, looping, after waiting
/// `start_delay`. An empty payload set makes a source that only negotiates and
/// then ends.
struct PacedSrc {
    caps: Caps,
    payloads: Arc<Vec<Vec<u8>>>,
    start_delay: Duration,
    duration: Duration,
    interval: Duration,
}

impl SourceLoop for PacedSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = Ready<Result<Caps, G2gError>>;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        ready(Ok(self.caps.clone()))
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(PipelinePacket::CapsChanged(self.caps.clone()))
                .await?;
            let mut seq = 0u64;
            if !self.payloads.is_empty() {
                tokio::time::sleep(self.start_delay).await;
                let start = Instant::now();
                let step_ns = self.interval.as_nanos() as u64;
                while start.elapsed() < self.duration {
                    let payload = self.payloads[seq as usize % self.payloads.len()].clone();
                    let frame = Frame::new(
                        MemoryDomain::System(SystemSlice::from_boxed(payload.into_boxed_slice())),
                        FrameTiming {
                            pts_ns: seq * step_ns,
                            ..FrameTiming::default()
                        },
                        seq,
                    );
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                    seq += 1;
                    tokio::time::sleep(self.interval).await;
                }
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(seq)
        })
    }
}

/// What a recv sink saw, in order, so caps events can be placed relative to the
/// frames around them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Seen {
    Caps,
    Frame,
    Eos,
}

type Log = Arc<Mutex<Vec<Seen>>>;

/// Recv sink recording its packet order into a shared log.
struct RecordSink {
    log: Log,
}

impl RecordSink {
    fn new() -> (Self, Log) {
        let log: Log = Arc::new(Mutex::new(Vec::new()));
        (RecordSink { log: log.clone() }, log)
    }
}

impl AsyncElement for RecordSink {
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
            let seen = match packet {
                PipelinePacket::CapsChanged(_) => Some(Seen::Caps),
                PipelinePacket::DataFrame(_) => Some(Seen::Frame),
                PipelinePacket::Eos => Some(Seen::Eos),
                _ => None,
            };
            if let Some(seen) = seen {
                self.log.lock().unwrap().push(seen);
            }
            Ok(())
        })
    }
}

fn frames(log: &Log) -> usize {
    log.lock()
        .unwrap()
        .iter()
        .filter(|s| **s == Seen::Frame)
        .count()
}

fn caps_events(log: &Log) -> usize {
    log.lock()
        .unwrap()
        .iter()
        .filter(|s| **s == Seen::Caps)
        .count()
}

/// Frames recorded after the pad's last caps event, i.e. what the track that
/// currently owns the pad delivered.
fn frames_after_last_caps(log: &Log) -> usize {
    let log = log.lock().unwrap();
    match log.iter().rposition(|s| *s == Seen::Caps) {
        Some(i) => log[i..].iter().filter(|s| **s == Seen::Frame).count(),
        None => 0,
    }
}

fn video_src(payloads: Arc<Vec<Vec<u8>>>, start_delay: Duration, duration: Duration) -> PacedSrc {
    PacedSrc {
        caps: h264_caps(),
        payloads,
        start_delay,
        duration,
        interval: Duration::from_millis(33),
    }
}

fn audio_src(payloads: Arc<Vec<Vec<u8>>>, duration: Duration) -> PacedSrc {
    PacedSrc {
        caps: opus_caps(),
        payloads,
        start_delay: Duration::ZERO,
        duration,
        interval: Duration::from_millis(20),
    }
}

/// (a) The offerer removes its active video track mid-run: the answerer's video
/// output stops receiving while its audio keeps flowing.
#[tokio::test]
async fn remove_track_stops_its_m_line_and_leaves_the_others_alone() {
    const MEDIA: Duration = Duration::from_secs(6);
    let aus = Arc::new(synthetic_h264_stream(150));
    let opus = Arc::new(synthetic_opus_stream(100));
    let (off_sig, ans_sig) = SdpChannel::pair();

    let sess_a = WebRtcDuplexSession::new(SignalRole::Offerer, off_sig, 2);
    let control = sess_a.control();
    let mut sess_a = sess_a;
    let mut sess_b = WebRtcDuplexSession::new(SignalRole::Answerer, ans_sig, 2);

    let (mut a_v_sink, _a_v) = RecordSink::new();
    let (mut a_a_sink, _a_a) = RecordSink::new();
    let (mut b_v_sink, b_v) = RecordSink::new();
    let (mut b_a_sink, b_a) = RecordSink::new();

    let clock = ZeroClock;
    let clock_ref = &clock;

    // Remove video two seconds in, then sample the answerer's counters over a
    // settled window on each side of the removal.
    let remover = {
        let (b_v, b_a) = (b_v.clone(), b_a.clone());
        async move {
            tokio::time::sleep(Duration::from_secs(2)).await;
            control.remove_track(0);
            tokio::time::sleep(Duration::from_secs(1)).await; // settle
            let (v0, a0) = (frames(&b_v), frames(&b_a));
            tokio::time::sleep(Duration::from_secs(2)).await;
            (frames(&b_v) - v0, frames(&b_a) - a0)
        }
    };
    let peer_a = {
        let (aus, opus) = (aus.clone(), opus.clone());
        let sess = &mut sess_a;
        async move {
            let mut vsrc = video_src(aus, Duration::ZERO, MEDIA);
            let mut asrc = audio_src(opus, MEDIA);
            let sources: Vec<&mut dyn DynSourceLoop> = std::vec![&mut vsrc, &mut asrc];
            let sinks: Vec<&mut dyn DynAsyncElement> = std::vec![&mut a_v_sink, &mut a_a_sink];
            run_duplex_session(sources, sess, sinks, clock_ref, 8).await
        }
    };
    let peer_b = {
        let (aus, opus) = (aus.clone(), opus.clone());
        let sess = &mut sess_b;
        async move {
            let mut vsrc = video_src(aus, Duration::ZERO, MEDIA);
            let mut asrc = audio_src(opus, MEDIA);
            let sources: Vec<&mut dyn DynSourceLoop> = std::vec![&mut vsrc, &mut asrc];
            let sinks: Vec<&mut dyn DynAsyncElement> = std::vec![&mut b_v_sink, &mut b_a_sink];
            run_duplex_session(sources, sess, sinks, clock_ref, 8).await
        }
    };

    let (window, ra, rb) = tokio::time::timeout(Duration::from_secs(40), async {
        tokio::join!(remover, peer_a, peer_b)
    })
    .await
    .expect("remove run completes in time");
    ra.expect("offerer duplex ok");
    rb.expect("answerer duplex ok");

    let (video_after, audio_after) = window;
    eprintln!("after the remove, the answerer got video={video_after} audio={audio_after}");
    assert!(
        video_after <= 2,
        "the removed video m-line should deliver nothing, got {video_after}"
    );
    assert!(
        audio_after >= 40,
        "audio must keep flowing through the video removal, got {audio_after}"
    );
    // The removal freed the pad rather than ending it: no Eos before the run end.
    assert_eq!(
        b_v.lock()
            .unwrap()
            .iter()
            .filter(|s| **s == Seen::Eos)
            .count(),
        1,
        "the freed pad ends on the session's own EOS, once"
    );
}

/// (b) A removed track's pads are recycled: the same spare source's next frame
/// negotiates a NEW m-line onto the freed pad, which re-announces its caps
/// before the new track's frames.
#[tokio::test]
async fn a_freed_pad_is_recycled_by_the_next_track() {
    const MEDIA: Duration = Duration::from_secs(8);
    let aus = Arc::new(synthetic_h264_stream(150));
    let opus = Arc::new(synthetic_opus_stream(100));
    let silent = Arc::new(Vec::new());
    let (off_sig, ans_sig) = SdpChannel::pair();

    let sess_a = WebRtcDuplexSession::new(SignalRole::Offerer, off_sig, 2).with_spare_tracks(1, 0);
    let control = sess_a.control();
    let mut sess_a = sess_a;
    let mut sess_b =
        WebRtcDuplexSession::new(SignalRole::Answerer, ans_sig, 2).with_spare_tracks(1, 0);

    let (mut a_v_sink, _a_v) = RecordSink::new();
    let (mut a_a_sink, _a_a) = RecordSink::new();
    let (mut a_s_sink, _a_s) = RecordSink::new();
    let (mut b_v_sink, _b_v) = RecordSink::new();
    let (mut b_a_sink, _b_a) = RecordSink::new();
    let (mut b_s_sink, b_s) = RecordSink::new();

    let clock = ZeroClock;
    let clock_ref = &clock;

    // The spare track starts half a second in; three seconds later it is removed,
    // and the source's next frame claims the freed pad with a new m-line.
    let remover = async move {
        tokio::time::sleep(Duration::from_secs(3)).await;
        control.remove_track(2);
    };
    let peer_a = {
        let (aus, opus) = (aus.clone(), opus.clone());
        let sess = &mut sess_a;
        async move {
            let mut vsrc = video_src(aus.clone(), Duration::ZERO, MEDIA);
            let mut asrc = audio_src(opus, MEDIA);
            let mut ssrc = video_src(aus, Duration::from_millis(500), MEDIA);
            let sources: Vec<&mut dyn DynSourceLoop> = std::vec![&mut vsrc, &mut asrc, &mut ssrc];
            let sinks: Vec<&mut dyn DynAsyncElement> =
                std::vec![&mut a_v_sink, &mut a_a_sink, &mut a_s_sink];
            run_duplex_session(sources, sess, sinks, clock_ref, 8).await
        }
    };
    let peer_b = {
        let (aus, opus, silent) = (aus.clone(), opus.clone(), silent.clone());
        let sess = &mut sess_b;
        async move {
            let mut vsrc = video_src(aus, Duration::ZERO, MEDIA);
            let mut asrc = audio_src(opus, MEDIA);
            let mut ssrc = video_src(silent, Duration::ZERO, MEDIA);
            let sources: Vec<&mut dyn DynSourceLoop> = std::vec![&mut vsrc, &mut asrc, &mut ssrc];
            let sinks: Vec<&mut dyn DynAsyncElement> =
                std::vec![&mut b_v_sink, &mut b_a_sink, &mut b_s_sink];
            run_duplex_session(sources, sess, sinks, clock_ref, 8).await
        }
    };

    let (_, ra, rb) = tokio::time::timeout(Duration::from_secs(45), async {
        tokio::join!(remover, peer_a, peer_b)
    })
    .await
    .expect("recycle run completes in time");
    ra.expect("offerer duplex ok");
    rb.expect("answerer duplex ok");

    eprintln!(
        "answerer spare pad: {} frames over {} caps events, {} after the last",
        frames(&b_s),
        caps_events(&b_s),
        frames_after_last_caps(&b_s)
    );
    assert!(
        caps_events(&b_s) >= 2,
        "the recycled pad must re-announce caps for its new track, log: {:?}",
        b_s.lock().unwrap()
    );
    assert!(
        frames_after_last_caps(&b_s) >= 10,
        "the recycled pad must carry the new track's frames, got {}",
        frames_after_last_caps(&b_s)
    );
}

/// (c) Both peers offer a new track at the same moment: the answerer yields its
/// own offer to the offerer's (glare rule) and retracts it, and both peers still
/// end up receiving the other's spare track.
#[tokio::test]
async fn a_raced_add_still_lands_on_both_peers() {
    const MEDIA: Duration = Duration::from_secs(6);
    let aus = Arc::new(synthetic_h264_stream(150));
    let opus = Arc::new(synthetic_opus_stream(100));
    let (off_sig, ans_sig) = SdpChannel::pair();

    let mut sess_a =
        WebRtcDuplexSession::new(SignalRole::Offerer, off_sig, 2).with_spare_tracks(1, 0);
    let mut sess_b =
        WebRtcDuplexSession::new(SignalRole::Answerer, ans_sig, 2).with_spare_tracks(1, 0);

    let (mut a_v_sink, _a_v) = RecordSink::new();
    let (mut a_a_sink, _a_a) = RecordSink::new();
    let (mut a_s_sink, a_s) = RecordSink::new();
    let (mut b_v_sink, _b_v) = RecordSink::new();
    let (mut b_a_sink, _b_a) = RecordSink::new();
    let (mut b_s_sink, b_s) = RecordSink::new();

    let clock = ZeroClock;
    let clock_ref = &clock;
    // Both spare sources start together, so the two ADD offers race.
    let spare_delay = Duration::from_millis(500);

    let peer_a = {
        let (aus, opus) = (aus.clone(), opus.clone());
        let sess = &mut sess_a;
        async move {
            let mut vsrc = video_src(aus.clone(), Duration::ZERO, MEDIA);
            let mut asrc = audio_src(opus, MEDIA);
            let mut ssrc = video_src(aus, spare_delay, MEDIA);
            let sources: Vec<&mut dyn DynSourceLoop> = std::vec![&mut vsrc, &mut asrc, &mut ssrc];
            let sinks: Vec<&mut dyn DynAsyncElement> =
                std::vec![&mut a_v_sink, &mut a_a_sink, &mut a_s_sink];
            run_duplex_session(sources, sess, sinks, clock_ref, 8).await
        }
    };
    let peer_b = {
        let (aus, opus) = (aus.clone(), opus.clone());
        let sess = &mut sess_b;
        async move {
            let mut vsrc = video_src(aus.clone(), Duration::ZERO, MEDIA);
            let mut asrc = audio_src(opus, MEDIA);
            let mut ssrc = video_src(aus, spare_delay, MEDIA);
            let sources: Vec<&mut dyn DynSourceLoop> = std::vec![&mut vsrc, &mut asrc, &mut ssrc];
            let sinks: Vec<&mut dyn DynAsyncElement> =
                std::vec![&mut b_v_sink, &mut b_a_sink, &mut b_s_sink];
            run_duplex_session(sources, sess, sinks, clock_ref, 8).await
        }
    };

    let (ra, rb) = tokio::time::timeout(Duration::from_secs(40), async {
        tokio::join!(peer_a, peer_b)
    })
    .await
    .expect("raced-add run completes in time");
    ra.expect("offerer duplex ok");
    rb.expect("answerer duplex ok");

    let (a, b) = (frames(&a_s), frames(&b_s));
    eprintln!("raced add: offerer spare={a} frames, answerer spare={b} frames");
    assert!(
        b >= 20,
        "the answerer should receive the offerer's spare track, got {b}"
    );
    assert!(
        a >= 20,
        "the offerer should receive the answerer's spare track, got {a}"
    );
}
