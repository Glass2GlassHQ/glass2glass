//! M1004 clock loss + re-election.
//!
//! The runner elects a clock once at startup and hands it to every sink. When
//! that clock loses the reference it disciplines to (`PipelineClock::healthy`
//! goes false, as a PTP servo does when its master goes away), the health
//! monitor posts `BusMessage::ClockLost`, elects again over the candidates that
//! are still healthy, and retargets the sinks' `ClockSync` in place through the
//! shared `ElectedClock` handle.
//!
//! The two clock candidates read visibly different times, so "the sink now reads
//! the other candidate" is an assertion on the value it sees, not on internals.
//! Time is mocked (the monitor's period costs no real time).
#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{block_on, run_graph, run_graph_with_bus, GraphNodeRef, SourceLoop};
use g2g_core::{
    AsyncClock, AsyncElement, Bus, BusMessage, Caps, ClockCandidate, ClockPriority, ClockSync,
    ConfigureOutcome, Dim, DynAsyncClock, Frame, FrameTiming, G2gError, Graph, OutputSink,
    PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

/// Time the elected (degradable) clock reads, and the time its healthy
/// replacement reads. Far apart so the sink's reading names its source.
const PRIMARY_NOW_NS: u64 = 1_000_000;
const BACKUP_NOW_NS: u64 = 9_000_000_000;

/// Spin bound for the source's wait on the re-election: a monitor that never
/// swaps ends the run and fails an assertion instead of hanging.
const MAX_SPINS: usize = 100_000;

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(2),
        height: Dim::Fixed(2),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
    }
}

fn frame(sequence: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 16]))),
        FrameTiming::default(),
        sequence,
    ))
}

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

fn yield_once() -> YieldOnce {
    YieldOnce(false)
}

/// Virtual-time clock: `sleep_until_ns` advances to the deadline and resolves
/// after one yield, so the monitor's one-second period costs no real time. This
/// is the runner's clock (the timer the monitor sleeps on), never a candidate.
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

/// A clock candidate that can lose its reference on demand: the stand-in for a
/// PTP clock going free-running. Reads a fixed time, so a sink's reading says
/// which candidate it is paced by.
#[derive(Debug)]
struct TestClock {
    now_ns: u64,
    healthy: AtomicBool,
}

impl TestClock {
    fn new(now_ns: u64) -> Self {
        Self {
            now_ns,
            healthy: AtomicBool::new(true),
        }
    }

    fn degrade(&self) {
        self.healthy.store(false, Ordering::Release);
    }
}

impl PipelineClock for TestClock {
    fn now_ns(&self) -> u64 {
        self.now_ns
    }

    fn healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire)
    }
}

/// Offers the degradable clock at the top election tier and pushes frames until
/// the sink has been retargeted (or the spin bound gives up), degrading its clock
/// after `degrade_after` frames.
struct DegradingSrc {
    clock: Arc<TestClock>,
    /// The clock reading the sink last saw, so the source can stop once the
    /// re-election has visibly reached it.
    sink_reading: Arc<AtomicU64>,
    degrade_after: u64,
    /// `false` runs the same graph with a clock that never degrades.
    degrade: bool,
    frames: u64,
}

impl SourceLoop for DegradingSrc {
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

    fn provide_clock(&self) -> Option<ClockCandidate> {
        Some(ClockCandidate::new(
            ClockPriority::LiveSource,
            self.clock.clone(),
        ))
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let mut spins = 0;
            loop {
                out.push(frame(self.frames)).await?;
                self.frames += 1;
                yield_once().await;

                if self.degrade && self.frames == self.degrade_after {
                    self.clock.degrade();
                }
                let done = if self.degrade {
                    self.sink_reading.load(Ordering::Acquire) == BACKUP_NOW_NS
                } else {
                    self.frames >= self.degrade_after
                };
                spins += 1;
                if done || spins >= MAX_SPINS {
                    break;
                }
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.frames)
        })
    }
}

/// Offers the always-healthy clock (a lower tier, so it only wins a re-election)
/// and records what its `ClockSync` reads for every frame.
struct ReadingSink {
    clock: Arc<TestClock>,
    sync: Option<ClockSync>,
    readings: Vec<u64>,
    last_reading: Arc<AtomicU64>,
}

impl AsyncElement for ReadingSink {
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
        Some(ClockCandidate::new(
            ClockPriority::Provider,
            self.clock.clone(),
        ))
    }

    fn set_clock_sync(&mut self, sync: ClockSync) {
        self.sync = Some(sync);
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        if matches!(packet, PipelinePacket::DataFrame(_)) {
            let now = self
                .sync
                .as_ref()
                .expect("the runner elected a clock")
                .now_ns();
            self.readings.push(now);
            self.last_reading.store(now, Ordering::Release);
        }
        Box::pin(core::future::ready(Ok(())))
    }
}

fn fixture(degrade: bool) -> (DegradingSrc, ReadingSink, Arc<TestClock>) {
    let primary = Arc::new(TestClock::new(PRIMARY_NOW_NS));
    let backup = Arc::new(TestClock::new(BACKUP_NOW_NS));
    let sink_reading = Arc::new(AtomicU64::new(0));
    let src = DegradingSrc {
        clock: primary.clone(),
        sink_reading: sink_reading.clone(),
        degrade_after: 4,
        degrade,
        frames: 0,
    };
    let sink = ReadingSink {
        clock: backup,
        sync: None,
        readings: Vec::new(),
        last_reading: sink_reading,
    };
    (src, sink, primary)
}

#[test]
fn a_lost_clock_is_reported_and_the_sink_is_retargeted() {
    let clock = VirtualClock::default();
    let (mut src, mut sink, primary) = fixture(true);
    let (bus, handle) = Bus::new(64);

    let mut g: Graph<GraphNodeRef<'_>> = Graph::new();
    let s = g.add_source(GraphNodeRef::source_ref(&mut src));
    let k = g.add_sink(GraphNodeRef::element_ref(&mut sink));
    g.link(s, k).expect("link source to sink");
    block_on(run_graph_with_bus(g, &clock, 2, &handle)).expect("graph run");

    assert!(!primary.healthy(), "the elected clock did degrade");

    let lost = {
        let mut n = 0;
        while let Some(m) = bus.try_recv() {
            if matches!(m, BusMessage::ClockLost) {
                n += 1;
            }
        }
        n
    };
    assert_eq!(lost, 1, "one ClockLost per loss, not one per health check");

    assert_eq!(
        sink.readings.first(),
        Some(&PRIMARY_NOW_NS),
        "the sink starts on the elected (live source) clock"
    );
    assert_eq!(
        sink.readings.last(),
        Some(&BACKUP_NOW_NS),
        "after the loss the same ClockSync reads the re-elected clock"
    );
}

#[test]
fn a_healthy_clock_is_never_re_elected() {
    let clock = VirtualClock::default();
    let (mut src, mut sink, primary) = fixture(false);
    let (bus, handle) = Bus::new(64);

    let mut g: Graph<GraphNodeRef<'_>> = Graph::new();
    let s = g.add_source(GraphNodeRef::source_ref(&mut src));
    let k = g.add_sink(GraphNodeRef::element_ref(&mut sink));
    g.link(s, k).expect("link source to sink");
    block_on(run_graph_with_bus(g, &clock, 2, &handle)).expect("graph run");

    assert!(primary.healthy());
    while let Some(m) = bus.try_recv() {
        assert!(
            !matches!(m, BusMessage::ClockLost),
            "a healthy clock is never reported lost"
        );
    }
    assert!(
        sink.readings.iter().all(|&r| r == PRIMARY_NOW_NS),
        "every frame paced on the elected clock"
    );
}

/// Without a bus there is no monitor and no swappable handle: the sink reads the
/// elected clock directly, degraded or not.
#[test]
fn without_a_bus_the_elected_clock_is_never_swapped() {
    let clock = VirtualClock::default();
    let (mut src, mut sink, primary) = fixture(true);
    src.degrade_after = 2;
    // Nothing will retarget the sink, so run a fixed number of frames instead of
    // waiting for a swap that cannot come.
    src.degrade = false;
    let degrading = primary.clone();

    let mut g: Graph<GraphNodeRef<'_>> = Graph::new();
    let s = g.add_source(GraphNodeRef::source_ref(&mut src));
    let k = g.add_sink(GraphNodeRef::element_ref(&mut sink));
    g.link(s, k).expect("link source to sink");
    degrading.degrade();
    block_on(run_graph(g, &clock, 2)).expect("graph run");

    assert!(
        sink.readings.iter().all(|&r| r == PRIMARY_NOW_NS),
        "no bus, no monitor: the sink stays on the clock it was given"
    );
}
