//! M592: A/V lip-sync validation (clock-sync track, phase 3).
//!
//! M590 built the `DriftClock` (least-squares fit of a real playout rate) and
//! M591 made `AlsaSink` provide it to election at the `AudioProvider` tier so
//! audio is the sync master. This validates the payoff end to end:
//!
//! 1. `audio_clock_is_elected_and_handed_to_the_video_sink` runs a real two-arm
//!    graph through `run_graph` (an audio arm whose sink provides a disciplined
//!    `DriftClock` at `AudioProvider`, a video arm whose source provides a plain
//!    `Provider` clock and whose sink is the real headless `SyncSink`) and asserts
//!    the runner elects the audio clock and hands *that* clock to the video sink,
//!    so the video sink is slaved to the audio timeline, not wall time.
//!
//! 2. `slaving_to_the_drift_clock_keeps_av_in_lipsync` drives the real
//!    `DriftClock` with a synthetic 0.1%-fast audio master and shows the point of
//!    all this: a video sink whose presentation deadlines are read off the drift
//!    clock stays locked to the audio (~0 skew), while one paced to wall time
//!    drifts ~10 ms out over 10 s. Deterministic (a hand-advanced clock, no real
//!    sleeping), so it runs in CI; a real-time host run would be 10 s+ and flaky.

use core::future::{ready, Ready};
use core::pin::Pin;
use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_graph, GraphNodeRef, SourceLoop};
use g2g_core::{
    graph::Graph, AsyncClock, AsyncElement, AudioFormat, Caps, ClockCandidate, ClockPriority,
    ConfigureOutcome, Dim, DriftClock, G2gError, MemoryDomain, MonotonicClock, OutputSink,
    PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};
use g2g_plugins::syncsink::SyncSink;

/// A monotonic clock the test advances by hand, standing in for the system
/// clock the audio DAC (and the drift fit) are measured against.
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

/// A `DriftClock` disciplined to `slope` (master ns per reference ns) over a
/// hand-advanced reference, i.e. an audio DAC running `slope`x wall time. Returns
/// the clock and the reference so the test can move "wall time" underneath it.
fn disciplined_drift_clock(slope: f64) -> (Arc<DriftClock>, Arc<ManualClock>) {
    let manual = Arc::new(ManualClock::default());
    let drift = Arc::new(DriftClock::new(manual.clone()));
    // Feed a few seconds of observations at a 10 ms cadence so the least-squares
    // fit locks onto the rate, exactly as AlsaSink's worker would from snd_pcm_delay.
    for k in 0..64u64 {
        let local = k * 10_000_000;
        manual.set(local);
        drift.observe(drift.reference_now(), (local as f64 * slope) as u64);
    }
    (drift, manual)
}

const VIDEO: fn() -> Caps = || Caps::RawVideo {
    format: RawVideoFormat::Rgba8,
    width: Dim::Fixed(8),
    height: Dim::Fixed(8),
    framerate: Rate::Fixed(30 << 16),
    interlace: g2g_core::Interlace::Any,
    colorimetry: g2g_core::Colorimetry::UNKNOWN,
};

const AUDIO: fn() -> Caps = || Caps::Audio {
    format: AudioFormat::PcmS16Le,
    channels: 2,
    sample_rate: 48_000,
    channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
};

/// The timer the video sink holds frames on: reads zero and never really
/// sleeps, so the graph completes instantly. Presentation deadlines come from
/// the elected clock, not from this.
#[derive(Debug)]
struct InstantClock;
impl PipelineClock for InstantClock {
    fn now_ns(&self) -> u64 {
        0
    }
}
impl AsyncClock for InstantClock {
    type SleepFuture<'a> = Ready<()>;
    fn sleep_until_ns(&self, _deadline_ns: u64) -> Ready<()> {
        ready(())
    }
}

/// Source: emit two frames of `caps` then EOS. `clock_priority` offers a plain
/// monotonic clock to election at that tier, so the video arm has a contender
/// the audio master has to beat.
struct EmitSrc {
    caps: Caps,
    clock_priority: Option<ClockPriority>,
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
    fn provide_clock(&self) -> Option<ClockCandidate> {
        let clock: Arc<dyn PipelineClock + Send + Sync> = Arc::new(MonotonicClock);
        self.clock_priority
            .map(|priority| ClockCandidate::new(priority, clock))
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

/// Audio sink standing in for `AlsaSink`: provides a disciplined `DriftClock` to
/// election at the `AudioProvider` tier.
struct AudioMasterSink {
    clock: Arc<DriftClock>,
}
impl AsyncElement for AudioMasterSink {
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
        let clock: Arc<dyn PipelineClock + Send + Sync> = self.clock.clone();
        Some(ClockCandidate::new(ClockPriority::AudioProvider, clock))
    }
    fn process<'a>(
        &'a mut self,
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

#[tokio::test]
async fn audio_clock_is_elected_and_handed_to_the_video_sink() {
    let (audio_clock, manual) = disciplined_drift_clock(1.001);
    // Pin "wall time" so the base the runner reads off the elected clock is known.
    manual.set(2_000_000_000);
    let expected_base = audio_clock.now_ns();

    // The real headless presentation sink on the video arm: it holds its own
    // clock as a timer but adopts whatever the pipeline elects for deadlines.
    let mut vsink = SyncSink::new(InstantClock);
    assert!(
        !vsink.slaved_to_elected_clock(),
        "before the run it paces on its own clock"
    );

    let stats = {
        let mut g: Graph<GraphNodeRef> = Graph::new();

        // Audio arm: source -> clock-providing sink (the master).
        let asrc = g.add_source(GraphNodeRef::source(EmitSrc {
            caps: AUDIO(),
            clock_priority: None,
        }));
        let asink = g.add_sink(GraphNodeRef::element(AudioMasterSink {
            clock: audio_clock.clone(),
        }));
        g.link(asrc, asink).unwrap();

        // Video arm: a source offering a plain `Provider` clock (which must lose
        // to audio) -> the sink that adopts the elected one.
        let vsrc = g.add_source(GraphNodeRef::source(EmitSrc {
            caps: VIDEO(),
            clock_priority: Some(ClockPriority::Provider),
        }));
        let vnode = g.add_sink(GraphNodeRef::element_ref(&mut vsink));
        g.link(vsrc, vnode).unwrap();

        run_graph(g, &ManualClock::default(), 4)
            .await
            .expect("graph runs")
    };

    // Audio outranks the video arm's Provider clock.
    assert_eq!(
        stats.clock_priority,
        ClockPriority::AudioProvider,
        "the audio sink's clock is elected master"
    );

    assert_eq!(vsink.received(), 2, "the video sink presented both frames");
    assert!(
        vsink.slaved_to_elected_clock(),
        "the runner handed the video sink the elected clock"
    );
    let sync = vsink.clock_sync().expect("video sink got a ClockSync");
    assert_eq!(
        sync.base_time_ns, expected_base,
        "video sink's base time is the audio clock's reading"
    );

    // The clinching check: the clock the video sink is slaved to *is* the audio
    // drift clock. Move wall time and confirm the video sink's clock tracks the
    // audio timeline (drifted), not raw wall time.
    manual.set(5_000_000_000);
    assert_eq!(
        sync.now_ns(),
        audio_clock.now_ns(),
        "video sink is slaved to the audio drift clock",
    );
    assert_ne!(
        sync.now_ns(),
        manual.now_ns(),
        "and that timeline is the drifted audio one, not wall time",
    );
}

#[tokio::test]
async fn slaving_to_the_drift_clock_keeps_av_in_lipsync() {
    // Audio DAC runs 0.1% fast (a realistic ppm-scale drift, exaggerated).
    let (drift, manual) = disciplined_drift_clock(1.001);
    let slope = drift.slope();
    assert!(
        (slope - 1.001).abs() < 1e-4,
        "drift fit locked onto the rate: {slope}"
    );

    // Election instant: base time = the audio clock's reading now.
    let w0 = 1_000_000_000u64;
    manual.set(w0);
    let base = drift.now_ns();

    const FRAME_PERIOD: u64 = 40_000_000; // 25 fps
    const FRAMES: u64 = 250; // 10 s

    let mut max_slaved_skew = 0i64;
    let mut wallclock_skew_at_end = 0i64;

    for i in 0..=FRAMES {
        let pts = i * FRAME_PERIOD; // frame running time
        let deadline = (base + pts) as i64; // presentation deadline on the elected clock

        // A video sink *slaved to the audio clock* presents frame `i` when the
        // audio clock reaches the deadline. That is the wall instant the same
        // audio running time actually leaves the DAC, so audio and video land
        // together. Verify via the real projection: at that wall time the audio
        // clock reads the deadline (to within fit rounding).
        let w_audio = w0 + (pts as f64 / slope) as u64;
        manual.set(w_audio);
        let audio_position = drift.now_ns() as i64;
        max_slaved_skew = max_slaved_skew.max((audio_position - deadline).abs());

        // A video sink paced to *wall time* presents frame `i` at wall `w0 + pts`.
        // By then the faster audio has already played past the deadline; that gap
        // is the lip-sync error the drift clock removes.
        let w_video_wall = w0 + pts;
        manual.set(w_video_wall);
        let audio_ahead = drift.now_ns() as i64 - deadline;
        if i == FRAMES {
            wallclock_skew_at_end = audio_ahead;
        }
    }

    // Slaved to the audio clock: essentially perfect sync (only fit rounding).
    assert!(
        max_slaved_skew < 1_000_000,
        "audio-slaved video drifted {max_slaved_skew} ns from audio (want <1 ms)",
    );
    // Paced to wall time instead: ~0.1% of 10 s = ~10 ms out by the end.
    assert!(
        wallclock_skew_at_end > 8_000_000,
        "wall-clock video should visibly drift from audio; got {wallclock_skew_at_end} ns",
    );

    eprintln!(
        "m592 lip-sync: audio-slaved max skew {max_slaved_skew} ns; \
         wall-clock skew at 10 s {} ms",
        wallclock_skew_at_end / 1_000_000
    );
}
