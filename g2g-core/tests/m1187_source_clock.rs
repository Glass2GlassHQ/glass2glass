//! M1187 - the elected clock reaches the sources.
//!
//! A capture source stamps its own zero-based timeline (a sample count, the
//! driver's first timestamp), which says nothing about when the pipeline started
//! or what a foreign master reads. The runner now hands every source the same
//! `ClockSync` it hands the sinks, and `CaptureAnchor` maps the source timeline
//! onto the elected clock's running time.
//!
//! Here the master is the sink's clock (the duplex case: capture paced against
//! another element's hardware), and it reads a second past the base time, so
//! the source's stamps have to start there rather than at zero.
#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use std::sync::Arc;

use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{block_on, run_graph, GraphNodeRef, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CaptureAnchor, ClockCandidate, ClockPriority, ClockSync, ConfigureOutcome,
    Dim, Frame, FrameTiming, G2gError, Graph, OutputSink, PipelineClock, PipelinePacket, Rate,
    RawVideoFormat,
};

/// What the master clock reads, and the base time the runner samples off it.
/// The gap is the capture source's late start.
const BASE_NOW_NS: u64 = 4_000_000_000;
const CAPTURE_LEAD_NS: u64 = 1_000_000_000;
/// The source's own frame spacing.
const PERIOD_NS: u64 = 20_000_000;
const FRAMES: u64 = 3;

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(2),
        height: Dim::Fixed(2),
        framerate: Rate::Fixed(50 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

/// Reads `BASE_NOW_NS` while the runner samples the base time, then jumps ahead
/// by the capture lead, so "the source started capturing late" is a reading the
/// test can assert on rather than a timing race.
#[derive(Debug)]
struct SteppingClock {
    reads: std::sync::atomic::AtomicU64,
}

impl PipelineClock for SteppingClock {
    fn now_ns(&self) -> u64 {
        let nth = self.reads.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        // The runner's own base-time sample is the first read; every read after
        // it is the source anchoring, a second later.
        if nth == 0 {
            BASE_NOW_NS
        } else {
            BASE_NOW_NS + CAPTURE_LEAD_NS
        }
    }
}

/// A capture source with no clock of its own: it stamps a zero-based timeline
/// and translates it through whatever master the runner elected.
struct CaptureSrc {
    anchor: CaptureAnchor,
    sync: Option<ClockSync>,
    stamps: Vec<u64>,
}

impl SourceLoop for CaptureSrc {
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

    fn set_clock_sync(&mut self, sync: ClockSync) {
        self.sync = Some(sync);
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for seq in 0..FRAMES {
                let elapsed_ns = seq * PERIOD_NS;
                let pts_ns = match &self.sync {
                    Some(sync) => self.anchor.stamp(sync, elapsed_ns, PERIOD_NS),
                    None => elapsed_ns,
                };
                self.stamps.push(pts_ns);
                out.push(PipelinePacket::DataFrame(Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 16]))),
                    FrameTiming {
                        pts_ns,
                        dts_ns: pts_ns,
                        duration_ns: PERIOD_NS,
                        ..FrameTiming::default()
                    },
                    seq,
                )))
                .await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(FRAMES)
        })
    }
}

/// The master: an element that paces to its own hardware, as an audio sink does.
struct ClockingSink {
    clock: Arc<SteppingClock>,
}

impl AsyncElement for ClockingSink {
    type ProcessFuture<'a>
        = core::future::Ready<Result<(), G2gError>>
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

    fn process<'a>(
        &'a mut self,
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        core::future::ready(Ok(()))
    }
}

#[test]
fn a_source_stamps_running_time_on_the_elected_clock() {
    let clock = g2g_core::clock::MonotonicClock;
    let mut src = CaptureSrc {
        anchor: CaptureAnchor::new(),
        sync: None,
        stamps: Vec::new(),
    };
    let mut sink = ClockingSink {
        clock: Arc::new(SteppingClock {
            reads: std::sync::atomic::AtomicU64::new(0),
        }),
    };

    let mut g: Graph<GraphNodeRef<'_>> = Graph::new();
    let s = g.add_source(GraphNodeRef::source_ref(&mut src));
    let k = g.add_sink(GraphNodeRef::element_ref(&mut sink));
    g.link(s, k).expect("link source to sink");
    block_on(run_graph(g, &clock, 2)).expect("graph run");

    assert!(src.sync.is_some(), "the runner handed the source the clock");
    let first = CAPTURE_LEAD_NS - PERIOD_NS;
    assert_eq!(
        src.stamps,
        (0..FRAMES)
            .map(|n| first + n * PERIOD_NS)
            .collect::<Vec<_>>(),
        "capture lands on the master's running time, keeping its own spacing"
    );
}

/// Without an elected clock the source keeps its own zero, the pre-M1187 path.
#[test]
fn a_source_with_no_elected_clock_keeps_its_own_zero() {
    let clock = g2g_core::clock::MonotonicClock;
    let mut src = CaptureSrc {
        anchor: CaptureAnchor::new(),
        sync: None,
        stamps: Vec::new(),
    };
    let mut sink = SilentSink;

    let mut g: Graph<GraphNodeRef<'_>> = Graph::new();
    let s = g.add_source(GraphNodeRef::source_ref(&mut src));
    let k = g.add_sink(GraphNodeRef::element_ref(&mut sink));
    g.link(s, k).expect("link source to sink");
    block_on(run_graph(g, &clock, 2)).expect("graph run");

    assert!(src.sync.is_none(), "nothing offered a clock");
    assert_eq!(
        src.stamps,
        (0..FRAMES).map(|n| n * PERIOD_NS).collect::<Vec<_>>()
    );
}

/// A sink that offers no clock, so the election finds no candidate.
struct SilentSink;

impl AsyncElement for SilentSink {
    type ProcessFuture<'a>
        = core::future::Ready<Result<(), G2gError>>
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
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        core::future::ready(Ok(()))
    }
}
