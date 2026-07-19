//! M593 phase B: a locked `PtpClock` is elected pipeline master over local
//! audio/video clocks and slaved to sinks through the real `run_graph`.
//!
//! Phase A proved the servo math; this proves the election + distribution: in a
//! graph carrying a PTP arm (a locked `PtpClock` at the `PtpGrandmaster` tier),
//! an audio arm (`AudioProvider`) and a video arm (`Provider`), the runner elects
//! the PTP clock and hands *it* to the video sink. That is what makes A/V sync
//! hold across machines: every device locked to the same grandmaster reads the
//! same timeline. Deterministic (a hand-advanced clock, synthetic exchanges), so
//! it runs in CI; a real multi-machine PTP soak is host/reference-gear gated.

use core::future::{ready, Ready};
use core::pin::Pin;
use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_graph, GraphNodeRef, SourceLoop};
use g2g_core::{
    graph::Graph, AsyncElement, AudioFormat, Caps, ClockCandidate, ClockPriority, ClockSync,
    ConfigureOutcome, Dim, G2gError, MemoryDomain, MonotonicClock, OutputSink, PipelineClock,
    PipelinePacket, PtpClock, Rate, RawVideoFormat, RefNs, TaiNs,
};

/// A monotonic reference advanced by hand (the PTP servo's local clock).
#[derive(Debug, Default)]
struct ManualClock(AtomicU64);
impl ManualClock {
    fn set(&self, v: u64) {
        self.0.store(v, Ordering::Release);
    }
}
impl PipelineClock for ManualClock {
    fn now_ns(&self) -> u64 {
        self.0.load(Ordering::Acquire)
    }
}

const EPOCH: i128 = 1_700_000_000_000_000_000; // TAI-scale grandmaster epoch
const DELAY: i128 = 100_000; // 100 us one-way link delay
const GAP: u64 = 1_000_000; // 1 ms slave Sync->Delay_Req gap

/// The four timestamps of a PTP exchange at reference time `local`, master time
/// `= local + EPOCH` (rate 1.0).
fn exchange(local: u64) -> (u64, u64, u64, u64) {
    let master = |x: u64| -> i128 { EPOCH + x as i128 };
    let t1 = (master(local) - DELAY) as u64;
    let t2 = local;
    let t3 = local + GAP;
    let t4 = (master(local + GAP) + DELAY) as u64;
    (t1, t2, t3, t4)
}

/// Build a `PtpClock` already driven to lock over the given reference.
fn locked_ptp_clock(clk: Arc<ManualClock>) -> Arc<PtpClock> {
    let ptp = Arc::new(PtpClock::new(clk.clone()));
    let mut local = 1_000_000_000u64;
    for _ in 0..24 {
        clk.set(local);
        let (t1, t2, t3, t4) = exchange(local);
        ptp.sync_exchange(TaiNs(t1), RefNs(t2), RefNs(t3), TaiNs(t4));
        local += 125_000_000;
    }
    assert!(ptp.is_locked(), "test setup: PTP clock should be locked");
    ptp
}

const VIDEO: fn() -> Caps = || Caps::RawVideo {
    format: RawVideoFormat::Rgba8,
    width: Dim::Fixed(8),
    height: Dim::Fixed(8),
    framerate: Rate::Fixed(30 << 16),
};
const AUDIO: fn() -> Caps = || Caps::Audio {
    format: AudioFormat::PcmS16Le,
    channels: 2,
    sample_rate: 48_000,
};

/// Source: two frames of `caps` then EOS.
struct EmitSrc {
    caps: Caps,
}
impl SourceLoop for EmitSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = Ready<Result<Caps, G2gError>>
    where
        Self: 'a;
    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        ready(Ok(self.caps.clone()))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for seq in 0..2u64 {
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
                    timing: FrameTiming::default(),
                    sequence: seq,
                    meta: Default::default(),
                };
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(2)
        })
    }
}

/// Sink that provides a locked `PtpClock` to election (a PTP-timed element).
struct PtpMasterSink {
    clock: Arc<PtpClock>,
}
impl AsyncElement for PtpMasterSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;
    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn provide_clock(&self) -> Option<ClockCandidate> {
        self.clock.candidate()
    }
    fn process<'a>(
        &'a mut self,
        _p: PipelinePacket,
        _o: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

/// Audio sink standing in for AlsaSink: an `AudioProvider` clock.
struct AudioProviderSink;
impl AsyncElement for AudioProviderSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;
    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn provide_clock(&self) -> Option<ClockCandidate> {
        let clock: Arc<dyn PipelineClock + Send + Sync> = Arc::new(MonotonicClock);
        Some(ClockCandidate::new(ClockPriority::AudioProvider, clock))
    }
    fn process<'a>(
        &'a mut self,
        _p: PipelinePacket,
        _o: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

type ElectedClock = (Arc<dyn PipelineClock + Send + Sync>, u64);

/// Video sink: offers a `Provider` clock and adopts whatever `ClockSync` the
/// runner elects, so the test can confirm it was slaved to the PTP master.
struct RecordingVideoSink {
    got: Arc<Mutex<Option<ElectedClock>>>,
}
impl AsyncElement for RecordingVideoSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;
    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn provide_clock(&self) -> Option<ClockCandidate> {
        let clock: Arc<dyn PipelineClock + Send + Sync> = Arc::new(MonotonicClock);
        Some(ClockCandidate::new(ClockPriority::Provider, clock))
    }
    fn set_clock_sync(&mut self, sync: ClockSync) {
        *self.got.lock().unwrap() = Some((sync.clock.clone(), sync.base_time_ns));
    }
    fn process<'a>(
        &'a mut self,
        _p: PipelinePacket,
        _o: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

#[tokio::test]
async fn ptp_clock_is_elected_master_and_slaves_the_video_sink() {
    let clk = Arc::new(ManualClock::default());
    let ptp = locked_ptp_clock(clk.clone());

    // Pin "wall time" so the elected base time is known.
    clk.set(5_000_000_000);
    let expected_base = ptp.now_ns();

    let got = Arc::new(Mutex::new(None));
    let mut g: Graph<GraphNodeRef<'static>> = Graph::new();

    // PTP arm (grandmaster), audio arm (AudioProvider), video arm (Provider).
    let psrc = g.add_source(GraphNodeRef::source(EmitSrc { caps: VIDEO() }));
    let psink = g.add_sink(GraphNodeRef::element(PtpMasterSink { clock: ptp.clone() }));
    g.link(psrc, psink).unwrap();

    let asrc = g.add_source(GraphNodeRef::source(EmitSrc { caps: AUDIO() }));
    let asink = g.add_sink(GraphNodeRef::element(AudioProviderSink));
    g.link(asrc, asink).unwrap();

    let vsrc = g.add_source(GraphNodeRef::source(EmitSrc { caps: VIDEO() }));
    let vsink = g.add_sink(GraphNodeRef::element(RecordingVideoSink {
        got: got.clone(),
    }));
    g.link(vsrc, vsink).unwrap();

    let stats = run_graph(g, &ManualClock::default(), 4)
        .await
        .expect("graph runs");

    // PTP outranks both the audio and video local clocks.
    assert_eq!(
        stats.clock_priority,
        ClockPriority::PtpGrandmaster,
        "the PTP grandmaster clock wins election over audio and video"
    );

    let (elected, base) = got
        .lock()
        .unwrap()
        .clone()
        .expect("video sink got a ClockSync");
    assert_eq!(
        base, expected_base,
        "video sink's base time is the PTP clock's reading"
    );

    // The video sink is slaved to the PTP (grandmaster/TAI) timeline: advancing
    // the reference, its clock tracks the PTP estimate, not raw wall time.
    clk.set(9_000_000_000);
    assert_eq!(
        elected.now_ns(),
        ptp.now_ns(),
        "video sink slaved to the PTP clock"
    );
    assert!(
        elected.now_ns() > EPOCH as u64,
        "and that timeline is grandmaster TAI, not wall time"
    );
}
