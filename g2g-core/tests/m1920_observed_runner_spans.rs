#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use g2g_core::element::DynAsyncElement;
use g2g_core::fanout::{MultiOutputElement, MultiOutputSink};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{
    block_on, run_fanin_session_observed, run_muxer_sink_dynamic_observed,
    run_source_fanout_observed, run_source_tee_dynamic_observed, DynSourceLoop, GraphNode,
    Observer, RunBuilder, SourceLoop, StageVisit, VisitRecorder,
};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, Frame, FrameTiming, G2gError, Graph,
    MultiInputElement, OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

const FRAME_COUNT: usize = 6;
const LINK_CAPACITY: usize = 2;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[derive(Debug, Default)]
struct RecordedVisits(Mutex<Vec<(String, StageVisit)>>);

impl VisitRecorder for RecordedVisits {
    fn record(&self, element: &str, visit: StageVisit) {
        self.0.lock().unwrap().push((element.to_string(), visit));
    }
}

impl RecordedVisits {
    fn spans_per_element(&self) -> BTreeMap<String, usize> {
        let mut spans = BTreeMap::new();
        for (element, _) in self.0.lock().unwrap().iter() {
            *spans.entry(element.clone()).or_default() += 1;
        }
        spans
    }
}

fn traced_observer() -> (Observer, Arc<RecordedVisits>) {
    let visits = Arc::new(RecordedVisits::default());
    let observer = Observer::new().with_visit_recorder(visits.clone());
    (observer, visits)
}

fn spans(entries: &[(&str, usize)]) -> BTreeMap<String, usize> {
    entries
        .iter()
        .map(|&(element, count)| (element.to_string(), count))
        .collect()
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

fn frame(sequence: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
        FrameTiming::default(),
        sequence,
    ))
}

struct CountingSource;

impl SourceLoop for CountingSource {
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
            for sequence in 0..FRAME_COUNT as u64 {
                out.push(frame(sequence)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(FRAME_COUNT as u64)
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

struct PassMux {
    inputs: usize,
    terminal: bool,
}

impl MultiInputElement for PassMux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        self.inputs
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

    fn is_terminal(&self) -> bool {
        self.terminal
    }

    fn process<'a>(
        &'a mut self,
        _input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.terminal && matches!(packet, PipelinePacket::DataFrame(_)) {
                out.push(packet).await?;
            }
            Ok(())
        })
    }
}

struct Splitter;

impl MultiOutputElement for Splitter {
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
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(data) = packet {
                let port = data.sequence as usize % 2;
                out.push_to(port, PipelinePacket::DataFrame(data)).await?;
            }
            Ok(())
        })
    }
}

#[test]
fn the_fan_in_session_runner_traces_its_sources_and_session() {
    let (observer, visits) = traced_observer();
    let mut first = CountingSource;
    let mut second = CountingSource;
    let sources: Vec<&mut dyn DynSourceLoop> = vec![&mut first, &mut second];
    let mut session = PassMux {
        inputs: 2,
        terminal: true,
    };
    block_on(run_fanin_session_observed(
        sources,
        &mut session,
        &ZeroClock,
        LINK_CAPACITY,
        &observer,
    ))
    .expect("the session run ends");

    assert_eq!(
        visits.spans_per_element(),
        spans(&[
            ("CountingSource0", FRAME_COUNT),
            ("CountingSource1", FRAME_COUNT),
            ("PassMux0", 2 * FRAME_COUNT),
        ])
    );
}

#[test]
fn the_dynamic_muxer_sink_runner_traces_an_attached_input_the_muxer_and_the_sink() {
    let (observer, visits) = traced_observer();
    let mut mux = PassMux {
        inputs: 1,
        terminal: false,
    };
    let mut sink = DiscardSink;
    let (handle, run) =
        run_muxer_sink_dynamic_observed(&mut mux, &mut sink, LINK_CAPACITY, &observer);
    handle
        .add_input(Box::new(CountingSource))
        .expect("the input is queued");
    drop(handle);
    block_on(run).expect("the muxer run ends");

    assert_eq!(
        visits.spans_per_element(),
        spans(&[
            ("CountingSource0", FRAME_COUNT),
            ("PassMux0", FRAME_COUNT),
            ("DiscardSink0", FRAME_COUNT),
        ])
    );
}

#[test]
fn the_dynamic_tee_runner_traces_its_source_and_an_attached_branch() {
    let (observer, visits) = traced_observer();
    let mut source = CountingSource;
    let (handle, run) = run_source_tee_dynamic_observed(&mut source, LINK_CAPACITY, &observer);
    handle
        .add_branch(Box::new(DiscardSink))
        .expect("the branch is queued");
    drop(handle);
    block_on(run).expect("the tee run ends");

    assert_eq!(
        visits.spans_per_element(),
        spans(&[
            ("CountingSource0", FRAME_COUNT),
            ("DiscardSink0", FRAME_COUNT)
        ])
    );
}

#[test]
fn the_source_fan_out_runner_hands_the_recorder_to_the_graph_runner() {
    let (observer, visits) = traced_observer();
    let mut source = CountingSource;
    let mut splitter = Splitter;
    let mut left = DiscardSink;
    let mut right = DiscardSink;
    let sinks: Vec<&mut dyn DynAsyncElement> = vec![&mut left, &mut right];
    block_on(run_source_fanout_observed(
        &mut source,
        &mut splitter,
        sinks,
        &ZeroClock,
        LINK_CAPACITY,
        &observer,
    ))
    .expect("the fan-out run ends");

    assert_eq!(
        visits.spans_per_element(),
        spans(&[
            ("CountingSource0", FRAME_COUNT),
            ("Splitter0", FRAME_COUNT),
            ("DiscardSink0", FRAME_COUNT / 2),
            ("DiscardSink1", FRAME_COUNT / 2),
        ])
    );
}

fn source_to_sink() -> Graph<GraphNode> {
    let mut graph: Graph<GraphNode> = Graph::new();
    let source = graph.add_source(GraphNode::source(CountingSource));
    let sink = graph.add_sink(GraphNode::element(DiscardSink));
    graph.link(source, sink).unwrap();
    graph
}

#[test]
fn a_graph_run_traces_through_its_observer_recorder() {
    let (observer, visits) = traced_observer();
    block_on(
        RunBuilder::new(source_to_sink(), &ZeroClock, LINK_CAPACITY)
            .with_observer(&observer)
            .run(),
    )
    .expect("the run reaches end of stream");

    assert_eq!(
        visits.spans_per_element(),
        spans(&[
            ("CountingSource0", FRAME_COUNT),
            ("DiscardSink0", FRAME_COUNT)
        ])
    );
}

#[test]
fn a_graph_run_given_two_visit_recorders_fails() {
    let (observer, _) = traced_observer();
    let result = block_on(
        RunBuilder::new(source_to_sink(), &ZeroClock, LINK_CAPACITY)
            .with_observer(&observer)
            .with_visit_recorder(Arc::new(RecordedVisits::default()))
            .run(),
    );

    assert_eq!(result.err(), Some(G2gError::VisitRecorderConflict));
}
