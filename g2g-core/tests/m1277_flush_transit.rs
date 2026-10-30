#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use std::time::Duration;

use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{
    block_on, GraphNode, Join2, Observer, RunBuilder, SeekController, SourceLoop, StateController,
};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, Frame, FrameTiming, G2gError, Graph, OutputSink,
    PipelineClock, PipelinePacket, PipelineState, Rate, RawVideoFormat, Seek, Segment,
    StateChangeReturn,
};

const LINK_CAPACITY: usize = 2;
const FRAME_COUNT: u64 = 16;
const FRAME_STEP_NS: u64 = 1_000;
const SEEK_TARGET_NS: u64 = 1_000_000;
// The stale frames sit queued this long before the seek drains them.
const PAUSE: Duration = Duration::from_millis(300);

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(16),
        height: Dim::Fixed(16),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

struct SeekableSource {
    seek: SeekController,
}

impl SourceLoop for SeekableSource {
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
            let mut position = 0;
            for sequence in 0..FRAME_COUNT {
                if let Some(seek) = self.seek.take_seek(Ok) {
                    out.push(PipelinePacket::Flush).await?;
                    position = seek.start;
                    let segment = Segment::for_flush_seek(&seek, None);
                    out.push(PipelinePacket::Segment(segment)).await?;
                }
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
                    FrameTiming {
                        pts_ns: position,
                        ..FrameTiming::default()
                    },
                    sequence,
                );
                out.push(PipelinePacket::DataFrame(frame)).await?;
                position += FRAME_STEP_NS;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(FRAME_COUNT)
        })
    }
}

struct DiscardSink;

impl AsyncElement for DiscardSink {
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

#[test]
fn stale_frames_drained_by_a_paused_flushing_seek_take_their_transit_stamps() {
    let seek = SeekController::new();
    let mut graph: Graph<GraphNode> = Graph::new();
    let source = graph.add_source(GraphNode::source(SeekableSource { seek: seek.clone() }));
    let sink = graph.add_sink(GraphNode::element(DiscardSink));
    graph.link(source, sink).unwrap();

    let observer = Observer::new();
    let state = StateController::new(PipelineState::Ready);
    let run = RunBuilder::new(graph, &ZeroClock, LINK_CAPACITY)
        .with_state(&state)
        .with_observer(&observer)
        .run();
    let driver = async {
        assert_eq!(
            state.set_state(PipelineState::Paused),
            StateChangeReturn::Async
        );
        state.await_prerolled().await;
        // Blocks the executor too, so nothing drains the stale frames meanwhile.
        std::thread::sleep(PAUSE);
        seek.seek(Seek::flush_to(SEEK_TARGET_NS));
        state.request_repreroll();
        state.await_prerolled().await;
        state.set_state(PipelineState::Playing);
    };
    let (stats, ()) = block_on(Join2::new(run, driver));
    let stats = stats.expect("the run reaches end of stream");

    let sink_row = stats
        .per_element
        .iter()
        .find(|row| row.name == "DiscardSink0")
        .expect("the sink has a telemetry row");
    assert!(sink_row.transit.count > 0, "the sink sampled transit");
    assert!(
        sink_row.transit.max_ns < PAUSE.as_nanos() as u64,
        "no frame after the seek reports the pause as its queue time, max {} ns",
        sink_row.transit.max_ns
    );
}
