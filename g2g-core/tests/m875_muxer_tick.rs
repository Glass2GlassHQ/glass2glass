//! M875: the muxer arm's deadline tick.
//!
//! A fan-in element that declares a `tick_interval_ns` receives
//! `PipelinePacket::Tick` while its inputs are stalled, so it can emit output on
//! its own cadence (zero-order-hold over the stalled pad) instead of freezing
//! with the input. An element that declares no interval never sees one.
//!
//! Time is mocked, so the tests are deterministic and finish in microseconds
//! instead of sleeping on real timers: one clock jumps virtual time to the sleep
//! deadline and resolves at once, the other never resolves a sleep at all (which
//! isolates the arm's busy-path deadline check).
#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use g2g_core::fanout::MultiInputElement;
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{block_on, run_muxer_sink_ticked, DynSourceLoop, SourceLoop};
use g2g_core::{
    AsyncClock, AsyncElement, Caps, ConfigureOutcome, Dim, Frame, FrameTiming, G2gError,
    OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

const TICK_NS: u64 = 33_000_000;

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(2),
        height: Dim::Fixed(2),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn frame(sequence: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 16]))),
        FrameTiming {
            pts_ns: sequence * TICK_NS,
            ..FrameTiming::default()
        },
        sequence,
    ))
}

/// Yields once so the other arms of the run get polled. Both the fake clock and
/// the stalling source need it: without a yield an arm that is always ready
/// starves the rest on the cooperative executor.
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

fn yield_once() -> YieldOnce {
    YieldOnce(false)
}

/// Virtual-time clock: `sleep_until_ns` advances the clock to the deadline and
/// resolves immediately (after the yield above), so a tick period costs no real
/// time. Time is an external boundary, so mocking it is fair game.
#[derive(Debug, Default)]
struct VirtualClock {
    now_ns: AtomicU64,
}

impl PipelineClock for VirtualClock {
    fn now_ns(&self) -> u64 {
        self.now_ns.load(Ordering::Acquire)
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

/// Clock for the busy-path test: reading it advances time (work between two
/// reads takes time) and its sleep **never** resolves. So a tick can only come
/// from the arm's per-iteration deadline check, never from a parked sleep.
#[derive(Debug, Default)]
struct BusyClock {
    now_ns: AtomicU64,
}

impl PipelineClock for BusyClock {
    fn now_ns(&self) -> u64 {
        self.now_ns.fetch_add(TICK_NS / 8, Ordering::AcqRel) + TICK_NS / 8
    }
}

impl AsyncClock for BusyClock {
    type SleepFuture<'a>
        = core::future::Pending<()>
    where
        Self: 'a;

    fn sleep_until_ns(&self, _deadline_ns: u64) -> core::future::Pending<()> {
        core::future::pending()
    }
}

/// Pushes `count` frames back to back, then ends: traffic on one pad that keeps
/// the muxer arm out of its parked path.
struct FloodSrc {
    count: u64,
}

impl SourceLoop for FloodSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(caps()))
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for i in 0..self.count {
                out.push(frame(i)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.count)
        })
    }
}

/// Pushes nothing until the muxer has ticked `stall_until_ticks` times, then
/// sends one frame and ends. The stall is what makes the tick observable: while
/// it holds, the muxer arm has no packet to process on any pad.
struct StallSrc {
    ticks: Arc<AtomicUsize>,
    stall_until_ticks: usize,
}

impl SourceLoop for StallSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(caps()))
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            // Bounded: a runner that never ticks then ends the run and fails the
            // assertion instead of stalling the test forever.
            let mut spins = 0;
            while self.ticks.load(Ordering::Acquire) < self.stall_until_ticks && spins < 10_000 {
                yield_once().await;
                spins += 1;
            }
            out.push(frame(0)).await?;
            out.push(PipelinePacket::Eos).await?;
            Ok(1)
        })
    }
}

/// Fan-in that emits a frame per deadline tick (the zero-order-hold shape) and
/// counts the ticks it received.
struct ZohMux {
    inputs: usize,
    interval_ns: Option<u64>,
    ticks: Arc<AtomicUsize>,
    emitted: u64,
}

impl MultiInputElement for ZohMux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        self.inputs
    }

    fn tick_interval_ns(&self) -> Option<u64> {
        self.interval_ns
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn configure_pipeline(
        &mut self,
        _input: usize,
        _absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(caps())
    }

    fn process<'a>(
        &'a mut self,
        _input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::Tick => {
                    self.ticks.fetch_add(1, Ordering::AcqRel);
                    let seq = self.emitted;
                    self.emitted += 1;
                    out.push(frame(seq)).await?;
                }
                PipelinePacket::DataFrame(_) => {
                    let seq = self.emitted;
                    self.emitted += 1;
                    out.push(frame(seq)).await?;
                }
                // Eos is the runner's to emit; caps are already negotiated.
                _ => {}
            }
            Ok(())
        })
    }
}

/// Counts the merged frames that reach the end of the pipeline.
#[derive(Default)]
struct CountSink {
    frames: u64,
}

impl AsyncElement for CountSink {
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

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        if matches!(packet, PipelinePacket::DataFrame(_)) {
            self.frames += 1;
        }
        Box::pin(core::future::ready(Ok(())))
    }
}

#[test]
fn ticks_fire_while_the_input_stalls_and_the_frames_reach_the_sink() {
    let ticks = Arc::new(AtomicUsize::new(0));
    let clock = VirtualClock::default();
    let mut src = StallSrc {
        ticks: ticks.clone(),
        stall_until_ticks: 3,
    };
    let mut mux = ZohMux {
        inputs: 1,
        interval_ns: Some(TICK_NS),
        ticks: ticks.clone(),
        emitted: 0,
    };
    let mut sink = CountSink::default();

    let sources: Vec<&mut dyn DynSourceLoop> = vec![&mut src];
    let stats = block_on(run_muxer_sink_ticked(
        sources, &mut mux, &mut sink, &clock, 2,
    ))
    .expect("ticked muxer run");

    let fired = ticks.load(Ordering::Acquire);
    assert!(
        fired >= 3,
        "the arm must tick while the only input is stalled, got {fired}"
    );
    // Every tick emitted a zero-order-hold frame, and the input's own frame
    // followed once it finally arrived.
    assert_eq!(
        sink.frames,
        fired as u64 + 1,
        "the ZOH frames plus the input frame reached the sink"
    );
    assert_eq!(stats.frames_consumed, sink.frames);
    assert!(
        clock.now_ns() >= 3 * TICK_NS,
        "each tick advanced the deadline by one period"
    );
}

#[test]
fn an_element_that_declares_no_interval_never_ticks() {
    let ticks = Arc::new(AtomicUsize::new(0));
    let clock = VirtualClock::default();
    let mut src = StallSrc {
        ticks: ticks.clone(),
        stall_until_ticks: 0,
    };
    let mut mux = ZohMux {
        inputs: 1,
        interval_ns: None,
        ticks: ticks.clone(),
        emitted: 0,
    };
    let mut sink = CountSink::default();

    let sources: Vec<&mut dyn DynSourceLoop> = vec![&mut src];
    block_on(run_muxer_sink_ticked(
        sources, &mut mux, &mut sink, &clock, 2,
    ))
    .expect("un-ticked muxer run");

    assert_eq!(
        ticks.load(Ordering::Acquire),
        0,
        "no interval declared, so the arm delivers no Tick"
    );
    assert_eq!(
        sink.frames, 1,
        "only the input's own frame reached the sink"
    );
    assert_eq!(clock.now_ns(), 0, "nothing slept on the clock");
}

#[test]
fn a_busy_arm_still_hits_its_deadline() {
    // Pad 1 stalls while pad 0 floods, so the arm keeps finding a buffered packet
    // and never parks. With a clock whose sleep never resolves, any tick here came
    // from the arm's per-iteration deadline check.
    let ticks = Arc::new(AtomicUsize::new(0));
    let clock = BusyClock::default();
    let mut flood = FloodSrc { count: 100 };
    let mut stalled = StallSrc {
        ticks: ticks.clone(),
        stall_until_ticks: 1,
    };
    let mut mux = ZohMux {
        inputs: 2,
        interval_ns: Some(TICK_NS),
        ticks: ticks.clone(),
        emitted: 0,
    };
    let mut sink = CountSink::default();

    let sources: Vec<&mut dyn DynSourceLoop> = vec![&mut flood, &mut stalled];
    block_on(run_muxer_sink_ticked(
        sources, &mut mux, &mut sink, &clock, 2,
    ))
    .expect("busy ticked muxer run");

    let fired = ticks.load(Ordering::Acquire);
    assert!(
        fired >= 1,
        "a busy arm must still tick, and no sleep can have fired one"
    );
    assert_eq!(
        sink.frames,
        101 + fired as u64,
        "both inputs' frames plus one ZOH frame per tick reached the sink"
    );
}
