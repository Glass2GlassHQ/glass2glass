// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

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
        pixel_aspect: g2g_core::PixelAspect::Any,
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

#[test]
fn sources_with_no_elected_clock_share_a_fallback_running_time() {
    let clock = g2g_core::clock::MonotonicClock;
    let capture = || CaptureSrc {
        anchor: CaptureAnchor::new(),
        sync: None,
        stamps: Vec::new(),
    };
    let mut sources = [capture(), capture()];
    let mut sinks = [SilentSink::default(), SilentSink::default()];

    let mut g: Graph<GraphNodeRef<'_>> = Graph::new();
    for (src, sink) in sources.iter_mut().zip(sinks.iter_mut()) {
        let s = g.add_source(GraphNodeRef::source_ref(src));
        let k = g.add_sink(GraphNodeRef::element_ref(sink));
        g.link(s, k).expect("link source to sink");
    }
    block_on(run_graph(g, &clock, 2)).expect("graph run");

    let base_times: Vec<u64> = sources
        .iter()
        .map(|src| {
            src.sync
                .as_ref()
                .expect("the runner handed each source a clock")
                .base_time()
        })
        .collect();
    assert_eq!(
        base_times[0], base_times[1],
        "both sources share one base time"
    );
    assert!(
        sinks.iter().all(|sink| !sink.got_clock),
        "sinks get no clock to pace on"
    );
    for src in &sources {
        let spacing: Vec<u64> = src.stamps.windows(2).map(|w| w[1] - w[0]).collect();
        assert_eq!(spacing, [PERIOD_NS; FRAMES as usize - 1]);
    }
}

/// A sink that offers no clock, so the election finds no candidate.
#[derive(Default)]
struct SilentSink {
    got_clock: bool,
}

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

    fn set_clock_sync(&mut self, _sync: ClockSync) {
        self.got_clock = true;
    }

    fn process<'a>(
        &'a mut self,
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        core::future::ready(Ok(()))
    }
}
