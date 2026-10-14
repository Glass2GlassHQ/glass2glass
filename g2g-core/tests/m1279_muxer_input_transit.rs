#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use std::time::Duration;

use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{block_on, run_graph_observed, GraphNode, Observer, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, Frame, FrameTiming, G2gError, Graph,
    MultiInputElement, OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

const LINK_CAPACITY: usize = 2;
const FRAME_COUNT: u64 = 8;
// Slow enough that the source runs ahead and frames queue on the muxer's input.
const MUX_WORK: Duration = Duration::from_millis(4);
const MUX_NAME: &str = "mux";

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::I420,
        width: Dim::Fixed(16),
        height: Dim::Fixed(16),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

struct FastSource;

impl SourceLoop for FastSource {
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
            for sequence in 0..FRAME_COUNT {
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
                    FrameTiming::default(),
                    sequence,
                );
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(FRAME_COUNT)
        })
    }
}

struct SlowMux;

impl MultiInputElement for SlowMux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        1
    }
    fn intercept_caps(&self, _input: usize, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }
    fn configure_pipeline(
        &mut self,
        _input: usize,
        _caps: &Caps,
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
            if let PipelinePacket::DataFrame(frame) = packet {
                std::thread::sleep(MUX_WORK);
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            Ok(())
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
fn an_observed_run_measures_queue_time_on_a_muxer_input() {
    let mut graph: Graph<GraphNode> = Graph::new();
    let source = graph.add_source(GraphNode::source(FastSource));
    let mux = graph.add_muxer(GraphNode::muxer(SlowMux), 1);
    graph.set_node_name(mux.node(), MUX_NAME.into());
    let sink = graph.add_sink(GraphNode::element(DiscardSink));
    graph.link(source, mux.input(0)).unwrap();
    graph.link(mux.output(), sink).unwrap();

    let observer = Observer::new();
    let stats = block_on(run_graph_observed(
        graph,
        &ZeroClock,
        LINK_CAPACITY,
        &observer,
        None,
    ))
    .expect("the run reaches end of stream");
    assert_eq!(stats.frames_consumed, FRAME_COUNT);

    let mux_row = stats
        .per_element
        .iter()
        .find(|row| row.name == MUX_NAME)
        .unwrap_or_else(|| panic!("no muxer row in {:?}", stats.per_element));
    assert_eq!(
        mux_row.transit.count, FRAME_COUNT,
        "every frame into the muxer has a queue time"
    );
    assert!(
        mux_row.transit.max_ns >= MUX_WORK.as_nanos() as u64,
        "frames waited behind the muxer's work, max {} ns",
        mux_row.transit.max_ns
    );
}
