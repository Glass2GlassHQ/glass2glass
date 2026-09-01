//! M953: the fan-in deadline tick under the thread-per-arm runner.
//!
//! The cooperative runner derives the tick timer from the pipeline clock itself
//! (M880), so `compositor timed-output=true` holds its output rate over a stalled
//! input with no wiring of its own. This is the same launch line through the entry
//! `g2g-launch --threads` uses (`run_graph_threaded_with_progress`, one OS thread
//! per arm against the wall clock): the arms take the timer as a shared handle
//! (`PipelineClock::shared_ticker`), so the behaviour has to match. A clock that
//! only tells time leaves the same graph frozen with its input.
#![cfg(all(feature = "std", feature = "multi-thread"))]

use core::future::Future;
use core::pin::Pin;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{
    parse_launch, run_graph_threaded_with_progress, LaunchFactory, PipelineProgress, Registry,
    SourceFactory, SourceLoop,
};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, G2gError, MemoryDomain, OutputSink, PipelineClock,
    PipelinePacket, Rate, RawVideoFormat,
};
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;
use g2g_plugins::TokioThreadSpawner;

/// One output frame period at the compositor's default 30 fps.
const PERIOD_NS: u64 = 1_000_000_000 * 65536 / (30 << 16);

/// How long input 0 stalls before it ends the stream. Twelve frame periods, so a
/// loaded CI host still gets well past [`WANT_HELD`] ticks in the window.
const STALL: Duration = Duration::from_millis(400);

/// Held frames the stall must produce for the timed run to count as ticking.
const WANT_HELD: usize = 3;

/// The PTS of every frame the sink consumed, in a process-global cell: a launch
/// factory builds its element from a plain `fn`, so the sink cannot carry a
/// per-test handle.
fn consumed_pts() -> &'static Mutex<Vec<u64>> {
    static PTS: OnceLock<Mutex<Vec<u64>>> = OnceLock::new();
    PTS.get_or_init(Mutex::default)
}

/// A clock that only tells time: the negative control, since both ticker
/// accessors default to `None`.
#[derive(Debug)]
struct MuteClock;

impl PipelineClock for MuteClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn rgba(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn solid(w: u32, h: u32, color: [u8; 4], pts_ns: u64) -> PipelinePacket {
    let mut buf = vec![0u8; (w * h) as usize * 4];
    for px in buf.as_chunks_mut::<4>().0 {
        *px = color;
    }
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(buf.into_boxed_slice())),
        g2g_core::FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            ..Default::default()
        },
        0,
    ))
}

/// The compositor's timing input (pad 0): one frame, then a stall that holds the
/// pad open while only the deadline tick can produce output.
struct StallingBaseSrc;

impl SourceLoop for StallingBaseSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(rgba(16, 16)))
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(solid(16, 16, [0, 0, 255, 255], 0)).await?;
            tokio::time::sleep(STALL).await;
            out.push(PipelinePacket::Eos).await?;
            Ok(1)
        })
    }
}

/// The overlay (pad 1): one frame, so the compositor finishes priming, then ends.
struct OverlaySrc;

impl SourceLoop for OverlaySrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(rgba(8, 8)))
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(solid(8, 8, [255, 0, 0, 255], 0)).await?;
            out.push(PipelinePacket::Eos).await?;
            Ok(1)
        })
    }
}

/// Records the PTS of every composited frame that reaches the end of the pipeline.
struct CountSink;

impl AsyncElement for CountSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        if let PipelinePacket::DataFrame(frame) = packet {
            consumed_pts().lock().unwrap().push(frame.timing.pts_ns);
        }
        Box::pin(core::future::ready(Ok(())))
    }
}

fn test_registry() -> Registry {
    let mut reg = default_registry();
    reg.register_source(SourceFactory::new("stallbase", rgba(16, 16), || {
        Box::new(StallingBaseSrc)
    }));
    reg.register_source(SourceFactory::new("overlay", rgba(8, 8), || {
        Box::new(OverlaySrc)
    }));
    reg.register_launch(LaunchFactory::new("countsink", Vec::new(), || {
        Box::new(CountSink)
    }));
    reg
}

const LINE: &str = "stallbase ! c.   overlay ! c.   \
                    compositor name=c width=32 height=32 timed-output=true ! countsink";

/// Runs the launch line on one OS thread per arm, and reports the PTS of every
/// frame the sink consumed.
async fn launched_pts<Clk: PipelineClock>(clock: &Clk) -> Vec<u64> {
    let reg = test_registry();
    let graph = parse_launch(&reg, LINE).expect("the timed compositor line parses");
    consumed_pts().lock().unwrap().clear();
    let progress = PipelineProgress::new();
    run_graph_threaded_with_progress(graph, clock, 2, &progress, None, &TokioThreadSpawner)
        .await
        .expect("pipeline runs");
    consumed_pts().lock().unwrap().clone()
}

// Both scenarios run the same launch line, and a launch factory can only reach
// the recorded frames through process-global state, so they share one test rather
// than racing as two.
#[tokio::test]
async fn a_threaded_launch_line_ticks_from_the_pipeline_clock() {
    // Negative control first: a clock that only tells time offers no timer, so
    // the composite freezes with its stalled input.
    let untimed = launched_pts(&MuteClock).await;
    assert_eq!(
        untimed,
        vec![0],
        "no ticker on the clock, so only the input's own frame was composited"
    );

    let pts = launched_pts(&WallClock::new()).await;
    assert!(
        pts.len() > WANT_HELD,
        "the arms must tick on the wall clock: {pts:?}"
    );
    // One real frame at PTS 0, then one zero-order-hold frame per tick, each a
    // frame period later, exactly as the cooperative runner produces them.
    let expected: Vec<u64> = (0..pts.len() as u64).map(|i| i * PERIOD_NS).collect();
    assert_eq!(pts, expected, "held frames walk forward one period each");
}
