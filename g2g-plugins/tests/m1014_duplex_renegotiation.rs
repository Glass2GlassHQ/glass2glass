//! Growing the pad COUNT of a live duplex WebRTC session (M1014), where M784
//! needed a spare pad declared up front: both peers start with ONE video track
//! and no spare, and are driven by the dynamic duplex runner.
//!
//! Half a second in, the offerer attaches a second video source through
//! `DynamicDuplexHandle::add_send_track`. The session learns of the pad from its
//! caps, offers the peer a new sendrecv m-line, and asks the runner for the recv
//! port to match. On the answerer the same m-line arrives as a remote track with
//! no free pad, so it grows one too, and the runner's sink factory builds the
//! element that drains it. Later the answerer attaches its own source, which
//! binds to that same m-line's send direction, so the grown track carries media
//! both ways.
//!
//! Runs fully on localhost UDP with a synthetic H.264 stream (no fixture, no
//! media server), so it is a default CI gate under the `webrtc` feature.

#![cfg(all(target_os = "linux", feature = "webrtc"))]

use core::future::{ready, Future, Ready};
use core::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use g2g_core::element::DynAsyncElement;
use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_duplex_session_dynamic, DynSourceLoop, SourceLoop};
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
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
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

/// Paces `payloads` in real time for `duration`, looping.
struct PacedSrc {
    caps: Caps,
    payloads: Arc<Vec<Vec<u8>>>,
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

type SeenLog = Arc<Mutex<Vec<Seen>>>;

/// Recv sink recording its packet order into a shared log.
struct RecordSink {
    log: SeenLog,
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

/// Every port the runner's sink factory was asked for, with that port's log.
type GrownPorts = Arc<Mutex<Vec<(usize, Caps, SeenLog)>>>;

fn frames(log: &SeenLog) -> usize {
    log.lock()
        .unwrap()
        .iter()
        .filter(|s| **s == Seen::Frame)
        .count()
}

/// True if the pad's first recorded packet was its caps event (and it got one).
fn caps_before_first_frame(log: &SeenLog) -> bool {
    matches!(log.lock().unwrap().first(), Some(Seen::Caps))
}

const MEDIA: Duration = Duration::from_secs(4);
/// When the offerer attaches its extra track. Long enough that the initial
/// session is streaming, short enough to leave media time on the grown one.
const OFFERER_ADD: Duration = Duration::from_millis(500);
/// When the answerer attaches its own, well after the m-line the offerer's add
/// negotiated exists, so it binds to that one's send direction instead of
/// offering a second m-line.
const ANSWERER_ADD: Duration = Duration::from_secs(2);

#[tokio::test]
async fn a_track_beyond_the_declared_pads_negotiates_a_new_m_line() {
    let aus = Arc::new(synthetic_h264_stream(150));
    let (off_sig, ans_sig) = SdpChannel::pair();

    // One active video track per peer, NO spare pads: every later track has to
    // grow the session.
    let mut sess_a = WebRtcDuplexSession::new(SignalRole::Offerer, off_sig, 1);
    let mut sess_b = WebRtcDuplexSession::new(SignalRole::Answerer, ans_sig, 1);

    // Every source, however late it attaches, stops streaming at the same moment:
    // a source still pushing when its session ends would see the run tear the
    // inbound channel out from under it.
    let video = |payloads: Arc<Vec<Vec<u8>>>, attached_at: Duration| PacedSrc {
        caps: h264_caps(),
        payloads,
        duration: MEDIA.saturating_sub(attached_at),
        interval: Duration::from_millis(33),
    };

    let mut a_src = video(aus.clone(), Duration::ZERO);
    let mut b_src = video(aus.clone(), Duration::ZERO);
    let a_declared: SeenLog = Arc::new(Mutex::new(Vec::new()));
    let b_declared: SeenLog = Arc::new(Mutex::new(Vec::new()));
    let mut a_sink = RecordSink {
        log: a_declared.clone(),
    };
    let mut b_sink = RecordSink {
        log: b_declared.clone(),
    };
    let a_grown: GrownPorts = Arc::new(Mutex::new(Vec::new()));
    let b_grown: GrownPorts = Arc::new(Mutex::new(Vec::new()));

    let clock = ZeroClock;
    let factory = |grown: GrownPorts| {
        move |port: usize, caps: &Caps| -> Option<Box<dyn DynAsyncElement>> {
            let log: SeenLog = Arc::new(Mutex::new(Vec::new()));
            grown
                .lock()
                .unwrap()
                .push((port, caps.clone(), log.clone()));
            Some(Box::new(RecordSink { log }))
        }
    };

    let a_sources: Vec<&mut dyn DynSourceLoop> = std::vec![&mut a_src];
    let a_sinks: Vec<&mut dyn DynAsyncElement> = std::vec![&mut a_sink];
    let (a_handle, a_run) = run_duplex_session_dynamic(
        a_sources,
        &mut sess_a,
        a_sinks,
        &clock,
        8,
        factory(a_grown.clone()),
    );
    let b_sources: Vec<&mut dyn DynSourceLoop> = std::vec![&mut b_src];
    let b_sinks: Vec<&mut dyn DynAsyncElement> = std::vec![&mut b_sink];
    let (b_handle, b_run) = run_duplex_session_dynamic(
        b_sources,
        &mut sess_b,
        b_sinks,
        &clock,
        8,
        factory(b_grown.clone()),
    );

    let adds = {
        let aus = aus.clone();
        async move {
            tokio::time::sleep(OFFERER_ADD).await;
            let input = a_handle
                .add_send_track(Box::new(video(aus.clone(), OFFERER_ADD)))
                .expect("the running offerer takes a second video track");
            // Dropping each handle is what lets that peer's send side end.
            drop(a_handle);
            tokio::time::sleep(ANSWERER_ADD - OFFERER_ADD).await;
            let back = b_handle
                .add_send_track(Box::new(video(aus, ANSWERER_ADD)))
                .expect("the running answerer takes one too");
            drop(b_handle);
            (input, back)
        }
    };

    let (ra, rb, (a_input, b_input)) = tokio::time::timeout(Duration::from_secs(60), async {
        tokio::join!(a_run, b_run, adds)
    })
    .await
    .expect("dynamic duplex renegotiation completes in time");
    ra.expect("offerer duplex ok");
    rb.expect("answerer duplex ok");

    assert_eq!(
        a_input, 1,
        "the offerer's track takes the index past its one"
    );
    assert_eq!(b_input, 1, "so does the answerer's");

    // (a) The declared track keeps flowing both directions.
    assert!(
        frames(&a_declared) >= 30 && frames(&b_declared) >= 30,
        "declared pads should keep streaming (offerer={}, answerer={})",
        frames(&a_declared),
        frames(&b_declared)
    );

    // (b) The answerer grew a recv pad for the m-line the offerer added, and its
    // sink came from the runner's factory.
    let b_ports = b_grown.lock().unwrap().clone();
    assert_eq!(
        b_ports.len(),
        1,
        "the answerer's factory is asked for exactly the one grown port, got {:?}",
        b_ports
            .iter()
            .map(|(p, c, _)| (*p, c.clone()))
            .collect::<Vec<_>>()
    );
    let (b_port, b_caps, b_log) = &b_ports[0];
    assert_eq!(*b_port, 1, "the grown port follows the declared one");
    assert!(
        matches!(
            b_caps,
            Caps::CompressedVideo {
                codec: VideoCodec::H264,
                ..
            }
        ),
        "the grown port carries the session's video caps, got {b_caps:?}"
    );
    assert!(
        frames(b_log) >= 20,
        "the answerer should receive the offerer's grown track, got {}",
        frames(b_log)
    );
    assert!(
        caps_before_first_frame(b_log),
        "the grown pad announces caps before its first frame, log: {:?}",
        b_log.lock().unwrap()
    );

    // (c) The answerer's own late track rides the same m-line back, so the offerer
    // receives on a grown pad too.
    let a_ports = a_grown.lock().unwrap().clone();
    assert!(
        !a_ports.is_empty(),
        "the offerer grows a recv pad for the m-line it added"
    );
    let a_frames: usize = a_ports.iter().map(|(_, _, log)| frames(log)).sum();
    assert!(
        a_frames >= 10,
        "the grown track should carry media back to the offerer, got {a_frames} frames over {} grown pad(s)",
        a_ports.len()
    );
    assert!(
        a_ports
            .iter()
            .all(|(_, _, log)| frames(log) == 0 || caps_before_first_frame(log)),
        "every grown pad that carried media announced caps first"
    );
}
