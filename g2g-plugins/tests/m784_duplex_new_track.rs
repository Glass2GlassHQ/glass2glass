//! Mid-session NEW track on the duplex WebRTC session (M784): both peers declare
//! two active tracks (video + audio) plus one SPARE video pad, which carries no
//! m-line at the initial handshake. Half a second in, the offerer's spare send
//! pad starts producing frames: the session offers the peer a new sendrecv
//! m-line, binds it to the free video output pad on both sides, and the media
//! flows there. Asserts the active tracks keep flowing both directions, that the
//! answerer receives the late track on its spare output pad, and that the pad's
//! `CapsChanged` precedes its first frame.
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
/// then ends (the answerer's spare pad).
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

/// What a recv sink saw, in order, so a caps event can be placed relative to the
/// first frame on that pad.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Seen {
    Caps,
    Frame,
    Eos,
}

/// Recv sink recording its packet order into a shared log.
struct RecordSink {
    log: Arc<Mutex<Vec<Seen>>>,
}

impl RecordSink {
    fn new() -> (Self, Arc<Mutex<Vec<Seen>>>) {
        let log = Arc::new(Mutex::new(Vec::new()));
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

fn frames(log: &Arc<Mutex<Vec<Seen>>>) -> usize {
    log.lock()
        .unwrap()
        .iter()
        .filter(|s| **s == Seen::Frame)
        .count()
}

/// True if the pad's first recorded packet was its caps event (and it got one).
fn caps_before_first_frame(log: &Arc<Mutex<Vec<Seen>>>) -> bool {
    matches!(log.lock().unwrap().first(), Some(Seen::Caps))
}

const MEDIA: Duration = Duration::from_secs(5);
const SPARE_DELAY: Duration = Duration::from_millis(500);

#[tokio::test]
async fn spare_pad_carries_a_mid_session_track_to_the_peer() {
    let aus = Arc::new(synthetic_h264_stream(150));
    let opus = Arc::new(synthetic_opus_stream(100));
    let silent = Arc::new(Vec::new());
    let (off_sig, ans_sig) = SdpChannel::pair();

    // Two active tracks (video + audio) plus one spare video pad per peer. Only
    // the offerer's spare source produces media, so there is no renegotiation
    // glare to resolve.
    let mut sess_a =
        WebRtcDuplexSession::new(SignalRole::Offerer, off_sig, 2).with_spare_tracks(1, 0);
    let mut sess_b =
        WebRtcDuplexSession::new(SignalRole::Answerer, ans_sig, 2).with_spare_tracks(1, 0);

    let video = |payloads: Arc<Vec<Vec<u8>>>, start_delay: Duration| PacedSrc {
        caps: h264_caps(),
        payloads,
        start_delay,
        duration: MEDIA,
        interval: Duration::from_millis(33),
    };
    let audio = |payloads: Arc<Vec<Vec<u8>>>| PacedSrc {
        caps: opus_caps(),
        payloads,
        start_delay: Duration::ZERO,
        duration: MEDIA,
        interval: Duration::from_millis(20),
    };

    let (mut a_v_sink, a_v) = RecordSink::new();
    let (mut a_a_sink, a_a) = RecordSink::new();
    let (mut a_s_sink, a_s) = RecordSink::new();
    let (mut b_v_sink, b_v) = RecordSink::new();
    let (mut b_a_sink, b_a) = RecordSink::new();
    let (mut b_s_sink, b_s) = RecordSink::new();

    let clock = ZeroClock;
    let clock_ref = &clock;

    let peer_a = {
        let (aus, opus) = (aus.clone(), opus.clone());
        let sess = &mut sess_a;
        async move {
            // Pad 2 (the spare) starts late: its first frame triggers the
            // mid-session m-line ADD.
            let mut vsrc = video(aus.clone(), Duration::ZERO);
            let mut asrc = audio(opus);
            let mut ssrc = video(aus, SPARE_DELAY);
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
            let mut vsrc = video(aus, Duration::ZERO);
            let mut asrc = audio(opus);
            let mut ssrc = video(silent, Duration::ZERO);
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
    .expect("duplex new-track run completes in time");
    ra.expect("offerer duplex ok");
    rb.expect("answerer duplex ok");

    let (avf, aaf, asf) = (frames(&a_v), frames(&a_a), frames(&a_s));
    let (bvf, baf, bsf) = (frames(&b_v), frames(&b_a), frames(&b_s));
    eprintln!("offerer got video={avf} audio={aaf} spare={asf}");
    eprintln!("answerer got video={bvf} audio={baf} spare={bsf}");

    // (a) The active tracks flow both directions, as before.
    assert!(
        avf >= 30 && aaf >= 30,
        "offerer should receive the answerer's active tracks (video={avf}, audio={aaf})"
    );
    assert!(
        bvf >= 30 && baf >= 30,
        "answerer should receive the offerer's active tracks (video={bvf}, audio={baf})"
    );
    // (b) The late track lands on the answerer's spare pad.
    assert!(
        bsf >= 20,
        "answerer should receive the mid-session track on its spare pad, got {bsf}"
    );
    // The offerer's own spare pad stays silent (the answerer never sends on it).
    assert_eq!(asf, 0, "the answerer sent no media on the spare m-line");
    // (c) The spare pad's caps event preceded its first frame.
    assert!(
        caps_before_first_frame(&b_s),
        "spare pad must announce caps before its first frame, log: {:?}",
        b_s.lock().unwrap()
    );
    assert!(
        caps_before_first_frame(&b_v) && caps_before_first_frame(&b_a),
        "active pads announce caps first too"
    );
}
