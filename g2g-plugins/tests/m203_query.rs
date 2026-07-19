//! M203 - application query system: position + duration.
//!
//! A source reports its total duration via `SourceLoop::query_duration`; the
//! runner publishes it on a `PipelineProgress` handle the app polls and posts a
//! `BusMessage::DurationChanged`. The sink arm publishes the stream-time
//! position of each buffer it consumes. This is the GStreamer POSITION /
//! DURATION query analog, the "can back a media player" primitive.

use std::pin::Pin;

use core::future::Future;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{
    run_graph_with_bus, run_graph_with_progress, GraphNode, PipelineProgress, SourceLoop,
};
use g2g_core::{
    Bus, BusMessage, Caps, ConfigureOutcome, Dim, FrameTiming, G2gError, Graph, MemoryDomain,
    OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

use g2g_plugins::fakesink::FakeSink;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(64),
        height: Dim::Fixed(64),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// A finite source that knows its duration up front and stamps each frame with
/// an increasing PTS, so the runner can publish position and duration. Models a
/// file / container source (the `Mp4Src` shape) for the query path.
struct ProgressSrc {
    total: u64,
    step_ns: u64,
    sequence: u64,
}

impl ProgressSrc {
    fn duration_ns(&self) -> u64 {
        self.total * self.step_ns
    }
}

impl SourceLoop for ProgressSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(caps()))
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn query_duration(&self) -> Option<u64> {
        Some(self.duration_ns())
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let _ = out.push(PipelinePacket::CapsChanged(caps())).await?;
            for i in 0..self.total {
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(
                        vec![0u8; 4].into_boxed_slice(),
                    )),
                    timing: FrameTiming {
                        pts_ns: i * self.step_ns,
                        ..FrameTiming::default()
                    },
                    sequence: self.sequence,
                    meta: Default::default(),
                };
                let _ = out.push(PipelinePacket::DataFrame(frame)).await?;
                self.sequence += 1;
                tokio::task::yield_now().await;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.total)
        })
    }
}

#[tokio::test]
async fn progress_handle_reports_duration_and_position() {
    let progress = PipelineProgress::new();
    // Before the run there is nothing to report.
    assert_eq!(progress.position(), None);
    assert_eq!(progress.duration(), None);

    let total = 8u64;
    let step_ns = 33_000_000u64; // ~30 fps
    let stats = {
        let mut g: Graph<GraphNode> = Graph::new();
        let src = g.add_source(GraphNode::source(ProgressSrc {
            total,
            step_ns,
            sequence: 0,
        }));
        let sink = g.add_sink(GraphNode::element(FakeSink::new()));
        g.link(src, sink).unwrap();
        run_graph_with_progress(g, &NullClock, 4, &progress)
            .await
            .expect("graph runs")
    };
    assert_eq!(stats.frames_consumed, total);

    // Duration came from the source's query_duration (published before frames).
    assert_eq!(progress.duration(), Some(total * step_ns));
    // Position is the stream time of the last buffer the sink consumed.
    assert_eq!(progress.position(), Some((total - 1) * step_ns));
}

#[tokio::test]
async fn bus_gets_duration_changed_once() {
    let (bus, handle) = Bus::new(64);
    let total = 5u64;
    let step_ns = 40_000_000u64;
    let duration_ns = total * step_ns;

    {
        let mut g: Graph<GraphNode> = Graph::new();
        let src = g.add_source(GraphNode::source(ProgressSrc {
            total,
            step_ns,
            sequence: 0,
        }));
        let sink = g.add_sink(GraphNode::element(FakeSink::new()));
        g.link(src, sink).unwrap();
        run_graph_with_bus(g, &NullClock, 4, &handle)
            .await
            .expect("graph runs with bus");
    }

    let mut durations = Vec::new();
    while let Some(m) = bus.try_recv() {
        if let BusMessage::DurationChanged { duration_ns } = m {
            durations.push(duration_ns);
        }
    }
    assert_eq!(
        durations,
        vec![duration_ns],
        "exactly one DurationChanged with the source's duration"
    );
}
