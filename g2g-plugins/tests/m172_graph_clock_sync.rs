//! M172: the DAG runner (`run_graph`) hands the elected pipeline clock to its
//! sinks via `set_clock_sync`, so a sink in a graph pipeline PTS-paces like one
//! in the linear runners (M169). Before this, `run_graph` elected a clock but
//! never delivered the resulting `ClockSync`, so display sinks presented ASAP.

use core::future::Future;
use core::pin::Pin;
use std::sync::{Arc, Mutex};

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_graph, GraphNodeRef, SourceLoop};
use g2g_core::{
    graph::Graph, AsyncElement, Caps, ClockCandidate, ClockPriority, ClockSync, ConfigureOutcome,
    Dim, G2gError, MemoryDomain, OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

/// A clock pinned to a fixed instant, so the test can assert the exact base time
/// the runner read from the elected clock.
struct FixedClock(u64);
impl PipelineClock for FixedClock {
    fn now_ns(&self) -> u64 {
        self.0
    }
}

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(8),
        height: Dim::Fixed(8),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Source emitting two frames then EOS, optionally offering a clock so a
/// non-fallback clock is elected.
struct EmitSrc {
    provide: Option<(ClockPriority, u64)>,
}

impl SourceLoop for EmitSrc {
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
        self.provide
            .map(|(p, now)| ClockCandidate::new(p, Arc::new(FixedClock(now))))
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for seq in 0..2u64 {
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(Box::new(
                        [0u8; 8 * 8 * 4],
                    ))),
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

/// Sink that records the `base_time_ns` of any `ClockSync` the runner delivers,
/// into a shared cell the test reads after the run.
struct RecordingSink {
    got_base: Arc<Mutex<Option<u64>>>,
}

impl AsyncElement for RecordingSink {
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
    fn set_clock_sync(&mut self, sync: ClockSync) {
        *self.got_base.lock().unwrap() = Some(sync.base_time_ns);
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
async fn run_graph_delivers_clock_sync_to_the_sink() {
    let got = Arc::new(Mutex::new(None));
    let mut g: Graph<GraphNodeRef<'static>> = Graph::new();
    // Source offers a live-source clock at a fixed instant; that instant is the
    // base time the runner should hand the sink.
    let src = g.add_source(GraphNodeRef::source(EmitSrc {
        provide: Some((ClockPriority::LiveSource, 7_000)),
    }));
    let sink = g.add_sink(GraphNodeRef::element(RecordingSink {
        got_base: got.clone(),
    }));
    g.link(src, sink).unwrap();

    let stats = run_graph(g, &FixedClock(0), 4).await.expect("graph runs");

    assert_eq!(
        stats.clock_priority,
        ClockPriority::LiveSource,
        "live-source clock elected"
    );
    assert_eq!(
        *got.lock().unwrap(),
        Some(7_000),
        "sink received ClockSync with the elected clock's base time"
    );
}

#[tokio::test]
async fn run_graph_skips_clock_sync_without_an_elected_clock() {
    // No element offers a clock: the runner falls back to the system clock and
    // must NOT call set_clock_sync (the pre-M169 "present ASAP" behaviour).
    let got = Arc::new(Mutex::new(None));
    let mut g: Graph<GraphNodeRef<'static>> = Graph::new();
    let src = g.add_source(GraphNodeRef::source(EmitSrc { provide: None }));
    let sink = g.add_sink(GraphNodeRef::element(RecordingSink {
        got_base: got.clone(),
    }));
    g.link(src, sink).unwrap();

    let stats = run_graph(g, &FixedClock(0), 4).await.expect("graph runs");

    assert_eq!(stats.clock_priority, ClockPriority::SystemFallback);
    assert_eq!(
        *got.lock().unwrap(),
        None,
        "no ClockSync delivered without an elected clock"
    );
}
