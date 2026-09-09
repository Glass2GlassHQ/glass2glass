//! M1170: `fallbacksrc` restarts a dead *session*, a source that produces every
//! track itself with no demuxer after it. `RestartFanoutSrc` is the multi-output
//! sibling of M1163's `RestartSrc`: the same policy (rebuild on failure, on a
//! `restart-timeout` stall, and on EOS under `restart-on-eos`, giving up once
//! `retry-timeout` of failures has passed), with every port moved by one shared
//! timeline offset so the tracks keep their alignment across a rebuild.
//!
//! The wrapper's rules run against a stub two-port session and a recording
//! multi-output sink, so nothing here needs a network. The RTSP hook that puts a
//! real session behind it (`rtsp_uri_fanout`) is network-coupled; its
//! network-free half (`rtsp_track_fanout`, the port shape) is checked at the
//! bottom under the `rtsp` feature.
//!
//! Run with `cargo test -p g2g-plugins --features std` (and `--features rtsp`
//! for the last check).
#![cfg(feature = "std")]

use core::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use g2g_core::fanout::{MultiOutputSink, MultiOutputSource};
use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{FanoutRebuild, RestartPolicy, UriError};
use g2g_core::{
    AudioFormat, Caps, ChannelLayout, Colorimetry, Dim, G2gError, MemoryDomain, PipelinePacket,
    PushOutcome, Rate, VideoCodec,
};
use g2g_plugins::fallbacksrc::{RestartFanoutSrc, RETRY_DELAY};

type DynSession = Box<dyn g2g_core::fanout::DynMultiOutputSource>;

/// The stub session's two ports, in the order `RtspSrcN` uses.
const VIDEO: usize = 0;
const AUDIO: usize = 1;
const PORTS: usize = 2;

/// Frames each life pushes per port.
const FRAMES_PER_LIFE_PER_PORT: u64 = 2;

/// Per-frame spacing of each port, so the two ports end their life at different
/// times and the shared offset is provably the later of the two.
const VIDEO_FRAME_NS: u64 = 40_000_000;
const AUDIO_FRAME_NS: u64 = 10_000_000;

/// A retry budget no test spends, so the budget is never what ends a run.
const GENEROUS_RETRY_NS: u64 = 3_600_000_000_000;

/// Disables the stall watchdog (gst's zero `restart-timeout`).
const NO_STALL_CHECK: u64 = 0;

/// A stall timeout short enough that a hung life is caught well inside one
/// retry delay.
const SHORT_STALL: Duration = Duration::from_millis(200);

/// Slack past the last rebuild a timed run waits for, so the life it starts has
/// time to deliver.
const SETTLE: Duration = Duration::from_millis(500);

fn h264_any() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        colorimetry: Colorimetry::UNKNOWN,
    }
}

fn aac_any() -> Caps {
    Caps::Audio {
        format: AudioFormat::Aac,
        channels: 2,
        sample_rate: 48_000,
        channel_layout: ChannelLayout::UNSPECIFIED,
    }
}

/// A multi-output sink recording each port's packets in order.
#[derive(Default)]
struct MultiCollect {
    ports: Vec<Vec<PipelinePacket>>,
}

impl MultiCollect {
    fn new() -> Self {
        Self {
            ports: (0..PORTS).map(|_| Vec::new()).collect(),
        }
    }

    fn pts(&self, port: usize) -> Vec<u64> {
        self.ports[port]
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some(f.timing.pts_ns),
                _ => None,
            })
            .collect()
    }

    fn eos_count(&self, port: usize) -> usize {
        self.ports[port]
            .iter()
            .filter(|p| matches!(p, PipelinePacket::Eos))
            .count()
    }
}

impl MultiOutputSink for MultiCollect {
    fn poll_push_to(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        port: usize,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push_to without a packet");
        self.ports[port].push(packet);
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }

    fn port_count(&self) -> usize {
        self.ports.len()
    }
}

fn frame(pts_ns: u64, duration_ns: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
        FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            duration_ns,
            ..FrameTiming::default()
        },
        0,
    ))
}

/// A stub two-port session: it pushes `FRAMES_PER_LIFE_PER_PORT` frames on each
/// port from PTS 0, then ends. Every life restarts at 0, so what the sink sees
/// is only monotonic if the wrapper moved the life onto the running timeline.
#[derive(Debug)]
struct TwoTrackSession {
    /// Push nothing and never return, so the stall watchdog is what ends the
    /// life.
    hang: bool,
}

impl MultiOutputSource for TwoTrackSession {
    type RunFuture<'a>
        = Pin<Box<dyn core::future::Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    fn output_count(&self) -> usize {
        PORTS
    }

    fn output_caps(&self, output: usize) -> Result<Caps, G2gError> {
        match output {
            VIDEO => Ok(h264_any()),
            AUDIO => Ok(aac_any()),
            _ => Err(G2gError::NotConfigured),
        }
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn MultiOutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if self.hang {
                core::future::pending::<()>().await;
            }
            let mut pushed = 0;
            for (port, step) in [(VIDEO, VIDEO_FRAME_NS), (AUDIO, AUDIO_FRAME_NS)] {
                for i in 0..FRAMES_PER_LIFE_PER_PORT {
                    out.push_to(port, frame(i * step, step)).await?;
                    pushed += 1;
                }
            }
            for port in 0..PORTS {
                out.push_to(port, PipelinePacket::Eos).await?;
            }
            Ok(pushed)
        })
    }
}

fn session() -> DynSession {
    Box::new(TwoTrackSession { hang: false })
}

fn hanging_session() -> DynSession {
    Box::new(TwoTrackSession { hang: true })
}

/// A rebuild closure over `build` that counts how often it was called.
fn counting_rebuild(
    build: impl Fn() -> DynSession + Send + 'static,
) -> (FanoutRebuild, Arc<AtomicUsize>) {
    let count = Arc::new(AtomicUsize::new(0));
    let seen = count.clone();
    let rebuild: FanoutRebuild = Box::new(move || {
        seen.fetch_add(1, Ordering::SeqCst);
        Ok(build())
    });
    (rebuild, count)
}

fn policy(restart_on_eos: bool, stall_ns: u64, retry_ns: u64) -> RestartPolicy {
    RestartPolicy {
        restart_on_eos,
        restart_timeout_ns: stall_ns,
        retry_timeout_ns: retry_ns,
    }
}

/// The port count and caps are answered from the first session, so the graph can
/// be shaped before any life has run and a rebuild cannot change it.
#[test]
fn the_wrapper_answers_the_sessions_ports() {
    let (rebuild, _) = counting_rebuild(session);
    let wrapper = RestartFanoutSrc::new(session(), rebuild, policy(false, 0, 0), None);
    assert_eq!(wrapper.output_count(), PORTS);
    assert_eq!(wrapper.output_caps(VIDEO).unwrap(), h264_any());
    assert_eq!(wrapper.output_caps(AUDIO).unwrap(), aac_any());
    assert!(wrapper.output_caps(PORTS).is_err(), "no third port");
}

/// Every life restarts at PTS 0, and one shared offset moves both ports onto the
/// running timeline: each port's own stream rises, and both are moved by the same
/// amount, so the tracks keep the alignment they had.
#[tokio::test]
async fn both_ports_share_one_timeline_across_restarts() {
    let (rebuild, rebuilds) = counting_rebuild(session);
    let mut wrapper = RestartFanoutSrc::new(
        session(),
        rebuild,
        policy(true, NO_STALL_CHECK, GENEROUS_RETRY_NS),
        None,
    );
    let mut sink = MultiCollect::new();
    let _ = tokio::time::timeout(
        RETRY_DELAY * 2 + SETTLE,
        MultiOutputSource::run(&mut wrapper, &mut sink),
    )
    .await;

    let lives = rebuilds.load(Ordering::SeqCst) as u64 + 1;
    assert!(
        lives >= 3,
        "two retry delays fit two rebuilds, got {lives} lives"
    );
    for port in [VIDEO, AUDIO] {
        let pts = sink.pts(port);
        assert_eq!(
            pts.len() as u64,
            lives * FRAMES_PER_LIFE_PER_PORT,
            "every life's frames reached port {port}: {pts:?}"
        );
        assert!(
            pts.windows(2).all(|w| w[0] < w[1]),
            "port {port} PTS must rise across the restart boundary: {pts:?}"
        );
        assert_eq!(sink.eos_count(port), 0, "inner EOS packets are swallowed");
    }
    // The offset is the later of the two ports' end times, applied to both, so
    // each life starts both ports at the same instant.
    let (video, audio) = (sink.pts(VIDEO), sink.pts(AUDIO));
    let per_life = FRAMES_PER_LIFE_PER_PORT as usize;
    for life in 0..lives as usize {
        assert_eq!(
            video[life * per_life],
            audio[life * per_life],
            "life {life} starts both ports together: {video:?} / {audio:?}"
        );
    }
}

/// A session that delivers nothing for `restart-timeout` is rebuilt, the stall
/// rule of M1163 over a multi-port life.
#[tokio::test]
async fn a_stalled_session_is_rebuilt() {
    let (rebuild, rebuilds) = counting_rebuild(hanging_session);
    let mut wrapper = RestartFanoutSrc::new(
        hanging_session(),
        rebuild,
        policy(false, SHORT_STALL.as_nanos() as u64, GENEROUS_RETRY_NS),
        None,
    );
    let mut sink = MultiCollect::new();
    let _ = tokio::time::timeout(
        RETRY_DELAY * 2 + SETTLE,
        MultiOutputSource::run(&mut wrapper, &mut sink),
    )
    .await;

    assert!(
        rebuilds.load(Ordering::SeqCst) >= 2,
        "a hung session is caught by the stall watchdog and rebuilt"
    );
}

/// Once the retry budget is spent the wrapper stops rebuilding and ends its
/// stream, which is what hands every switch to its fallback for good. The
/// terminal EOS reaches every port, so no branch is stranded.
#[tokio::test]
async fn a_spent_retry_budget_ends_every_port() {
    let failures = Arc::new(AtomicUsize::new(0));
    let seen = failures.clone();
    let rebuild: FanoutRebuild = Box::new(move || {
        seen.fetch_add(1, Ordering::SeqCst);
        Err(UriError::Malformed)
    });
    // Two rebuilds start inside the budget; the third would start past it.
    let retry_budget = RETRY_DELAY * 2 + RETRY_DELAY / 2;
    let mut wrapper = RestartFanoutSrc::new(
        session(),
        rebuild,
        policy(true, NO_STALL_CHECK, retry_budget.as_nanos() as u64),
        None,
    );
    let mut sink = MultiCollect::new();
    let started = Instant::now();
    let frames = MultiOutputSource::run(&mut wrapper, &mut sink)
        .await
        .expect("giving up ends the stream cleanly rather than failing the run");

    assert_eq!(
        frames,
        FRAMES_PER_LIFE_PER_PORT * PORTS as u64,
        "the first life delivered; every rebuild after it failed"
    );
    assert_eq!(
        failures.load(Ordering::SeqCst),
        2,
        "rebuilds attempted inside the budget"
    );
    let elapsed = started.elapsed();
    assert!(
        elapsed >= RETRY_DELAY * 2 && elapsed < retry_budget,
        "each rebuild waited its delay and the third was not attempted: {elapsed:?}"
    );
    for port in [VIDEO, AUDIO] {
        assert_eq!(
            sink.eos_count(port),
            1,
            "one terminal EOS on port {port}: {:?}",
            sink.ports[port]
        );
    }
}

/// The network-free half of the RTSP hook: a stream carrying both tracks becomes
/// a session-headed fan-out with a video port and an audio port, and a video-only
/// stream is declined (one kind does not fan out).
#[cfg(feature = "rtsp")]
#[test]
fn an_rtsp_stream_with_audio_becomes_a_session_fanout() {
    use g2g_core::runtime::UriFanoutHead;
    use g2g_core::stream::StreamType;
    use g2g_plugins::rtspsrcn::{negotiation_audio_caps, RtspTracks};
    use g2g_plugins::uridecodebin::rtsp_track_fanout;

    const URL: &str = "rtsp://camera.invalid:554/stream1";
    let declared = aac_any();
    let tracks = RtspTracks {
        video: h264_any(),
        audio: Some(declared.clone()),
        onvif_metadata: false,
    };
    let fanout = rtsp_track_fanout(URL, &tracks).expect("a stream with audio fans out");
    assert!(
        matches!(fanout.head, UriFanoutHead::FanoutSource { .. }),
        "the session is the head: there is no demuxer to put after it"
    );
    assert_eq!(
        fanout
            .ports
            .iter()
            .map(|p| p.stream_type)
            .collect::<Vec<_>>(),
        vec![StreamType::Video, StreamType::Audio]
    );
    assert_eq!(fanout.ports[VIDEO].caps, h264_any());
    assert_eq!(
        fanout.ports[AUDIO].caps,
        negotiation_audio_caps(&declared),
        "the audio port negotiates with the decoder-facing form of the SDP's caps"
    );

    let video_only = RtspTracks {
        video: h264_any(),
        audio: None,
        onvif_metadata: false,
    };
    assert!(
        rtsp_track_fanout(URL, &video_only).is_none(),
        "one kind falls back to the single-stream expansion"
    );
}
