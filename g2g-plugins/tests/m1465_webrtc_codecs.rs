//! Every video codec the WebRTC elements carry round-trips a P2P duplex session
//! over localhost UDP: two `WebRtcDuplexSession`s built for the codec negotiate
//! it, and each received access unit is byte-identical to one that was sent,
//! which proves str0m's packetizer and depacketizer keep g2g's framing (Annex-B,
//! raw VP8 / VP9 frames, low-overhead AV1 temporal units). Default CI gate, no
//! media server.

#![cfg(all(target_os = "linux", feature = "webrtc"))]

mod webrtc_streams_common;

use core::future::{ready, Future, Ready};
use core::pin::Pin;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use g2g_core::element::DynAsyncElement;
use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_duplex_session, DynSourceLoop, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, FrameTiming, G2gError, MemoryDomain,
    OutputSink, PipelineClock, PipelinePacket,
};
use g2g_plugins::webrtc_codec::WebRtcVideoCodec;
use g2g_plugins::webrtcduplex::{SdpChannel, SignalRole, WebRtcDuplexSession};

use webrtc_streams_common::{fixed_caps, synthetic_stream};

const FRAMES: usize = 60;
const FRAME_INTERVAL: Duration = Duration::from_millis(33);
/// The handshake drops the first units, so most but not all arrive.
const MIN_RECEIVED: usize = FRAMES / 2;
const RUN_TIMEOUT: Duration = Duration::from_secs(40);

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Paces the synthetic units at frame rate, then EOS.
struct PacedSrc {
    codec: WebRtcVideoCodec,
    units: Arc<Vec<Vec<u8>>>,
}

impl SourceLoop for PacedSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = Ready<Result<Caps, G2gError>>;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        ready(Ok(fixed_caps(self.codec)))
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(PipelinePacket::CapsChanged(fixed_caps(self.codec)))
                .await?;
            for (seq, unit) in self.units.iter().enumerate() {
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(unit.clone().into_boxed_slice())),
                    FrameTiming {
                        pts_ns: seq as u64 * FRAME_INTERVAL.as_nanos() as u64,
                        ..FrameTiming::default()
                    },
                    seq as u64,
                );
                out.push(PipelinePacket::DataFrame(frame)).await?;
                tokio::time::sleep(FRAME_INTERVAL).await;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.units.len() as u64)
        })
    }
}

#[derive(Default)]
struct Received {
    caps: Option<Caps>,
    units: Vec<(Vec<u8>, bool)>,
}

struct RecordingSink {
    received: Arc<Mutex<Received>>,
}

impl AsyncElement for RecordingSink {
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
            let mut received = self.received.lock().unwrap();
            match packet {
                PipelinePacket::CapsChanged(caps) => {
                    received.caps.get_or_insert(caps);
                }
                PipelinePacket::DataFrame(frame) => {
                    let bytes = frame
                        .domain
                        .as_system_slice()
                        .expect("system memory")
                        .to_vec();
                    received.units.push((bytes, frame.timing.keyframe));
                }
                _ => {}
            }
            Ok(())
        })
    }
}

async fn round_trip(codec: WebRtcVideoCodec) -> (Arc<Vec<Vec<u8>>>, Received, Received) {
    let units = Arc::new(synthetic_stream(codec, FRAMES));
    let (chan_a, chan_b) = SdpChannel::pair();
    let mut session_a =
        WebRtcDuplexSession::new(SignalRole::Offerer, chan_a, 1).with_video_codec(codec);
    let mut session_b =
        WebRtcDuplexSession::new(SignalRole::Answerer, chan_b, 1).with_video_codec(codec);
    let received_a = Arc::new(Mutex::new(Received::default()));
    let received_b = Arc::new(Mutex::new(Received::default()));
    let clock = ZeroClock;

    async fn peer(
        codec: WebRtcVideoCodec,
        units: Arc<Vec<Vec<u8>>>,
        session: &mut WebRtcDuplexSession,
        received: Arc<Mutex<Received>>,
        clock: &ZeroClock,
    ) -> Result<g2g_core::runtime::RunStats, G2gError> {
        let mut src = PacedSrc { codec, units };
        let mut sink = RecordingSink { received };
        let sources: Vec<&mut dyn DynSourceLoop> = vec![&mut src];
        let sinks: Vec<&mut dyn DynAsyncElement> = vec![&mut sink];
        run_duplex_session(sources, session, sinks, clock, 8).await
    }
    let (result_a, result_b) = tokio::time::timeout(RUN_TIMEOUT, async {
        tokio::join!(
            peer(
                codec,
                units.clone(),
                &mut session_a,
                received_a.clone(),
                &clock
            ),
            peer(
                codec,
                units.clone(),
                &mut session_b,
                received_b.clone(),
                &clock
            )
        )
    })
    .await
    .unwrap_or_else(|_| panic!("{codec:?} P2P run completes in time"));
    result_a.unwrap_or_else(|e| panic!("{codec:?} peer A: {e:?}"));
    result_b.unwrap_or_else(|e| panic!("{codec:?} peer B: {e:?}"));
    let take = |cell: &Arc<Mutex<Received>>| core::mem::take(&mut *cell.lock().unwrap());
    (units, take(&received_a), take(&received_b))
}

fn assert_received(codec: WebRtcVideoCodec, sent: &[Vec<u8>], received: &Received, peer: &str) {
    let caps = received
        .caps
        .as_ref()
        .unwrap_or_else(|| panic!("{codec:?} {peer}: caps announced before frames"));
    assert!(
        matches!(caps, Caps::CompressedVideo { codec: c, .. } if *c == codec.video_codec()),
        "{codec:?} {peer}: output caps carry the negotiated codec, got {caps:?}"
    );
    assert!(
        received.units.len() >= MIN_RECEIVED,
        "{codec:?} {peer}: {} of {FRAMES} units arrived",
        received.units.len()
    );
    let sent_set: HashSet<&[u8]> = sent.iter().map(Vec::as_slice).collect();
    for (unit, _) in &received.units {
        assert!(
            sent_set.contains(unit.as_slice()),
            "{codec:?} {peer}: a received unit is not byte-identical to any sent unit"
        );
    }
    let keyframe_units: HashSet<&[u8]> = sent
        .iter()
        .step_by(webrtc_streams_common::GOP)
        .map(Vec::as_slice)
        .collect();
    let keyframes_flagged = received
        .units
        .iter()
        .filter(|(unit, keyframe)| *keyframe && keyframe_units.contains(unit.as_slice()))
        .count();
    assert!(
        keyframes_flagged >= 1,
        "{codec:?} {peer}: a received keyframe carries the keyframe flag"
    );
    let inter_flagged = received
        .units
        .iter()
        .filter(|(unit, keyframe)| *keyframe && !keyframe_units.contains(unit.as_slice()))
        .count();
    assert_eq!(
        inter_flagged, 0,
        "{codec:?} {peer}: no inter frame is flagged as a keyframe"
    );
}

#[tokio::test]
async fn every_codec_round_trips_a_p2p_session() {
    for codec in WebRtcVideoCodec::ALL {
        let (sent, received_a, received_b) = round_trip(codec).await;
        eprintln!(
            "{codec:?}: A received {} units, B received {} units",
            received_a.units.len(),
            received_b.units.len()
        );
        assert_received(codec, &sent, &received_a, "peer A");
        assert_received(codec, &sent, &received_b, "peer B");
    }
}
