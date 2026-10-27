//! M1950: `set_state(Null)` on a running graph ends the run cleanly. The sink
//! arms stop at the gate and the arms feeding them end on the closed links, which
//! is the stop that was asked for, not a pipeline failure.
#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;

use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{block_on, GraphNode, RunBuilder, SourceLoop, StateController};
use g2g_core::{
    Bus, BusMessage, Caps, ConfigureOutcome, Dim, Frame, FrameTiming, G2gError, Graph, OutputSink,
    PipelineClock, PipelinePacket, PipelineState, Rate, RawVideoFormat,
};

#[path = "common/count_sink.rs"]
mod count_sink;
use count_sink::CountSink;

const LINK_CAPACITY: usize = 2;
const BUS_DEPTH: usize = 64;
// frames the source pushes before it stops the pipeline
const FRAMES_BEFORE_NULL: u64 = 4;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

// a live source that never ends on its own, and sets the pipeline to Null partway
struct EndlessSource {
    state: StateController,
}

impl SourceLoop for EndlessSource {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(16),
            height: Dim::Fixed(16),
            framerate: Rate::Fixed(30 << 16),
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        }))
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for sequence in 0.. {
                if sequence == FRAMES_BEFORE_NULL {
                    self.state.set_state(PipelineState::Null);
                }
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
                    FrameTiming::default(),
                    sequence,
                );
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            unreachable!("the source pushes until its link closes")
        })
    }
}

#[test]
fn null_ends_a_running_graph_without_an_error_or_an_eos() {
    let state = StateController::new(PipelineState::Playing);
    let mut graph: Graph<GraphNode> = Graph::new();
    let source = graph.add_source(GraphNode::source(EndlessSource {
        state: state.clone(),
    }));
    let sink = graph.add_sink(GraphNode::element(CountSink::default()));
    graph.link(source, sink).unwrap();

    let (bus, bus_handle) = Bus::new(BUS_DEPTH);
    let run = RunBuilder::new(graph, &ZeroClock, LINK_CAPACITY)
        .with_bus(&bus_handle)
        .with_state(&state)
        .run();
    let stats = block_on(run).expect("a Null stop is not a pipeline failure");
    assert!(stats.frames_consumed > 0, "frames flowed before the stop");

    let mut messages = Vec::new();
    while let Some(message) = bus.try_recv() {
        messages.push(message);
    }
    let reported_end = messages
        .iter()
        .any(|message| matches!(message, BusMessage::Eos | BusMessage::ElementError { .. }));
    assert!(
        !reported_end,
        "a Null stop posts neither an end of stream nor an element error: {messages:?}"
    );
}
