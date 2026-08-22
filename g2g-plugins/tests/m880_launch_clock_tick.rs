//! M880: the fan-in deadline tick reaches a `parse_launch` pipeline.
//!
//! `compositor timed-output=true` declares a tick interval, and the cooperative
//! runner derives the timer from the pipeline clock itself (`PipelineClock::as_ticker`),
//! so a text pipeline run through the entry point `g2g-launch` uses
//! (`run_graph_with_progress`, against a clock that can sleep) holds its output rate
//! over a stalled input with no ticked entry point of its own. A clock that only
//! tells time leaves the same graph frozen with its input.

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{
    parse_launch, run_graph_with_progress, LaunchFactory, PipelineProgress, Registry,
    SourceFactory, SourceLoop,
};
use g2g_core::{
    AsyncClock, AsyncElement, Caps, ConfigureOutcome, Dim, DynAsyncClock, G2gError, MemoryDomain,
    OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};
use g2g_plugins::registry::default_registry;

/// One output frame period at the compositor's default 30 fps.
const PERIOD_NS: u64 = 1_000_000_000 * 65536 / (30 << 16);

/// Held frames the stalling base source waits for before it ends the stream.
const WANT_HELD: usize = 3;

/// Those plus input 0's own frame: what the sink must consume.
const WANT_FRAMES: usize = WANT_HELD + 1;

/// Spin bound for that wait: a runner that never ticks ends the run and fails an
/// assertion below instead of hanging. Each tick costs a couple of yields on the
/// virtual clock, so this is generous for the timed case and short for the
/// untimed one, which always exhausts it.
const MAX_SPINS: usize = 2_000;

/// Yields once so the other arms of the run get polled.
struct YieldOnce(bool);

impl Future for YieldOnce {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.0 {
            return Poll::Ready(());
        }
        self.0 = true;
        cx.waker().wake_by_ref();
        Poll::Pending
    }
}

/// Virtual-time clock: `sleep_until_ns` advances the clock to the deadline and
/// resolves at once (after a yield), so a tick period costs no real time.
#[derive(Debug, Default)]
struct VirtualClock {
    now_ns: AtomicU64,
}

impl PipelineClock for VirtualClock {
    fn now_ns(&self) -> u64 {
        self.now_ns.load(Ordering::Acquire)
    }

    fn as_ticker(&self) -> Option<&dyn DynAsyncClock> {
        Some(self)
    }
}

impl AsyncClock for VirtualClock {
    type SleepFuture<'a>
        = VirtualSleep<'a>
    where
        Self: 'a;

    fn sleep_until_ns(&self, deadline_ns: u64) -> VirtualSleep<'_> {
        VirtualSleep {
            clock: self,
            deadline_ns,
            yielded: false,
        }
    }
}

struct VirtualSleep<'a> {
    clock: &'a VirtualClock,
    deadline_ns: u64,
    yielded: bool,
}

impl Future for VirtualSleep<'_> {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if !self.yielded {
            self.yielded = true;
            cx.waker().wake_by_ref();
            return Poll::Pending;
        }
        let deadline = self.deadline_ns;
        self.clock.now_ns.fetch_max(deadline, Ordering::AcqRel);
        Poll::Ready(())
    }
}

/// A clock that only tells time: the negative control, since `as_ticker` is the
/// default `None`.
#[derive(Debug)]
struct MuteClock;

impl PipelineClock for MuteClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// The PTS of every frame the sink consumed, in a process-global cell: a launch
/// factory builds its element from a plain `fn`, so the sink cannot carry a
/// per-test handle.
fn consumed_pts() -> &'static Mutex<Vec<u64>> {
    static PTS: OnceLock<Mutex<Vec<u64>>> = OnceLock::new();
    PTS.get_or_init(Mutex::default)
}

fn consumed_count() -> &'static AtomicUsize {
    static COUNT: OnceLock<AtomicUsize> = OnceLock::new();
    COUNT.get_or_init(AtomicUsize::default)
}

fn rgba(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
    }
}

fn solid(w: u32, h: u32, color: [u8; 4], pts_ns: u64) -> PipelinePacket {
    let mut buf = vec![0u8; (w * h) as usize * 4];
    for px in buf.as_chunks_mut::<4>().0 {
        px.copy_from_slice(&color);
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
/// pad open until the sink has seen `WANT_HELD` more frames, which only the
/// deadline tick can produce.
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
            let mut spins = 0;
            while consumed_count().load(Ordering::Acquire) < WANT_FRAMES && spins < MAX_SPINS {
                YieldOnce(false).await;
                spins += 1;
            }
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
            // Published after the PTS, so a waiting source never observes a
            // count without its frame.
            consumed_count().fetch_add(1, Ordering::AcqRel);
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

/// Runs the launch line through the same entry `g2g-launch` uses, and reports the
/// PTS of every frame the sink consumed.
async fn launched_pts<Clk: PipelineClock>(clock: &Clk) -> Vec<u64> {
    let reg = test_registry();
    let graph = parse_launch(&reg, LINE).expect("the timed compositor line parses");
    consumed_pts().lock().unwrap().clear();
    consumed_count().store(0, Ordering::Release);
    let progress = PipelineProgress::new();
    run_graph_with_progress(graph, clock, 2, &progress, None)
        .await
        .expect("pipeline runs");
    consumed_pts().lock().unwrap().clone()
}

// Both scenarios run the same launch line, and a launch factory can only reach
// the recorded frames through process-global state, so they share one test rather
// than racing as two.
#[tokio::test]
async fn a_launch_line_ticks_from_the_pipeline_clock() {
    // Negative control first: a clock that only tells time offers no ticker, so
    // the composite freezes with its stalled input.
    let untimed = launched_pts(&MuteClock).await;
    assert_eq!(
        untimed,
        vec![0],
        "no ticker on the clock, so only the input's own frame was composited"
    );

    let clock = VirtualClock::default();
    let pts = launched_pts(&clock).await;
    assert!(
        pts.len() >= WANT_FRAMES,
        "timed-output must hold the rate over the stalled input: {pts:?}"
    );
    // One real frame at PTS 0, then one zero-order-hold frame per tick, each a
    // frame period later: the launch-exposed property actually reached the arm.
    let expected: Vec<u64> = (0..pts.len() as u64).map(|i| i * PERIOD_NS).collect();
    assert_eq!(pts, expected, "held frames walk forward one period each");
    assert!(
        clock.now_ns() >= WANT_HELD as u64 * PERIOD_NS,
        "the arm slept on the pipeline clock"
    );
}
