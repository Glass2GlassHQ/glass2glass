#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use g2g_core::fanout::{MultiOutputSink, MultiOutputSource};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{
    block_on, GraphMutator, GraphNode, Join2, RunBuilder, SourceLoop, StageVisit, VisitRecorder,
};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, Frame, FrameTiming, G2gError, Graph,
    MultiInputElement, OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

const FRAME_COUNT: u64 = 8;
const LINK_CAPACITY: usize = 2;
// slow enough that frames queue on the fan-in sink's input
const FANIN_WORK: Duration = Duration::from_millis(4);
// how long the late muxer input holds back its first frame
const LATE_INPUT_DELAY: Duration = Duration::from_millis(20);
// spans an element has to record before the mutation driver moves on
const SPANS_PER_STEP: usize = 3;

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
    fn of(&self, element: &str) -> Vec<StageVisit> {
        self.0
            .lock()
            .unwrap()
            .iter()
            .filter(|(name, _)| name == element)
            .map(|(_, visit)| *visit)
            .collect()
    }

    fn sequences_of(&self, element: &str) -> Vec<u64> {
        let mut sequences: Vec<u64> = self.of(element).iter().map(|v| v.sequence).collect();
        sequences.sort_unstable();
        sequences
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

fn frame(sequence: u64) -> PipelinePacket {
    let timing = FrameTiming {
        pts_ns: sequence,
        ..FrameTiming::default()
    };
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
        timing,
        sequence,
    ))
}

// re-wakes itself until the instant passes, so the cooperative run keeps going
struct Deadline(Instant);

impl Future for Deadline {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if Instant::now() >= self.0 {
            return Poll::Ready(());
        }
        cx.waker().wake_by_ref();
        Poll::Pending
    }
}

struct Yield(bool);

impl Future for Yield {
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

#[derive(Default)]
struct CountingSource {
    first_frame_delay: Option<Duration>,
    stop: Option<Arc<AtomicBool>>,
}

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
            if let Some(delay) = self.first_frame_delay {
                Deadline(Instant::now() + delay).await;
            }
            let frames = match self.stop {
                Some(_) => u64::MAX,
                None => FRAME_COUNT,
            };
            let mut emitted = 0;
            for sequence in 0..frames {
                if self.stop.as_ref().is_some_and(|s| s.load(Ordering::SeqCst)) {
                    break;
                }
                out.push(frame(sequence)).await?;
                emitted += 1;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(emitted)
        })
    }
}

struct TwoPortSource;

impl MultiOutputSource for TwoPortSource {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    fn output_count(&self) -> usize {
        2
    }

    fn output_caps(&self, _output: usize) -> Result<Caps, G2gError> {
        Ok(caps())
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn MultiOutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for sequence in 0..FRAME_COUNT {
                out.push_to(sequence as usize % 2, frame(sequence)).await?;
            }
            out.push_to(0, PipelinePacket::Eos).await?;
            out.push_to(1, PipelinePacket::Eos).await?;
            Ok(FRAME_COUNT)
        })
    }
}

struct PassThrough;

impl AsyncElement for PassThrough {
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
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !matches!(packet, PipelinePacket::Eos) {
                out.push(packet).await?;
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

struct TestMux {
    inputs: usize,
    pts_ordered: bool,
    terminal_work: Option<Duration>,
}

impl MultiInputElement for TestMux {
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

    fn input_pts_ordered(&self) -> bool {
        self.pts_ordered
    }

    fn is_terminal(&self) -> bool {
        self.terminal_work.is_some()
    }

    fn process<'a>(
        &'a mut self,
        _input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            let PipelinePacket::DataFrame(data) = packet else {
                return Ok(());
            };
            match self.terminal_work {
                Some(work) => std::thread::sleep(work),
                None => {
                    out.push(PipelinePacket::DataFrame(data)).await?;
                }
            }
            Ok(())
        })
    }
}

fn run_traced(graph: Graph<GraphNode>) -> Arc<RecordedVisits> {
    let visits = Arc::new(RecordedVisits::default());
    block_on(
        RunBuilder::new(graph, &ZeroClock, LINK_CAPACITY)
            .with_visit_recorder(visits.clone())
            .run(),
    )
    .expect("the run reaches end of stream");
    visits
}

#[test]
fn a_source_records_one_span_per_pushed_frame_back_to_back() {
    let mut graph: Graph<GraphNode> = Graph::new();
    let source = graph.add_source(GraphNode::source(CountingSource::default()));
    graph.set_node_name(source, "src".into());
    let sink = graph.add_sink(GraphNode::element(DiscardSink));
    graph.link(source, sink).unwrap();

    let visits = run_traced(graph);
    let spans = visits.of("src");
    assert_eq!(
        spans.iter().map(|v| v.sequence).collect::<Vec<_>>(),
        (0..FRAME_COUNT).collect::<Vec<_>>()
    );
    for pair in spans.windows(2) {
        assert!(
            pair[1].enter_ns >= pair[0].exit_ns,
            "each span starts once the previous push returned: {pair:?}"
        );
    }
    assert!(spans
        .iter()
        .all(|v| v.wait_ns == 0 && v.exit_ns >= v.enter_ns));
}

#[test]
fn a_fan_out_source_records_a_span_for_the_frames_of_every_port() {
    let mut graph: Graph<GraphNode> = Graph::new();
    let source = graph.add_fanout_src(GraphNode::fanout_source(TwoPortSource), 2);
    graph.set_node_name(source.node(), "session".into());
    for port in 0..2 {
        let sink = graph.add_sink(GraphNode::element(DiscardSink));
        graph.link(source.output(port), sink).unwrap();
    }

    let visits = run_traced(graph);
    assert_eq!(
        visits.sequences_of("session"),
        (0..FRAME_COUNT).collect::<Vec<_>>()
    );
}

#[test]
fn a_terminal_fan_in_records_each_frame_with_its_queue_wait() {
    let mut graph: Graph<GraphNode> = Graph::new();
    let source = graph.add_source(GraphNode::source(CountingSource::default()));
    let session = graph.add_fanin_sink(
        GraphNode::muxer(TestMux {
            inputs: 1,
            pts_ordered: false,
            terminal_work: Some(FANIN_WORK),
        }),
        1,
    );
    graph.set_node_name(session.node(), "session".into());
    graph.link(source, session.input(0)).unwrap();

    let visits = run_traced(graph);
    assert_eq!(
        visits.sequences_of("session"),
        (0..FRAME_COUNT).collect::<Vec<_>>()
    );
    let longest_wait = visits.of("session").iter().map(|v| v.wait_ns).max();
    assert!(
        longest_wait >= Some(FANIN_WORK.as_nanos() as u64),
        "frames queued behind the fan-in's work, longest wait {longest_wait:?} ns"
    );
}

#[test]
fn a_pts_ordered_muxer_counts_the_time_a_frame_is_held_for_its_peer_as_wait() {
    let mut graph: Graph<GraphNode> = Graph::new();
    let early = graph.add_source(GraphNode::source(CountingSource::default()));
    let late = graph.add_source(GraphNode::source(CountingSource {
        first_frame_delay: Some(LATE_INPUT_DELAY),
        stop: None,
    }));
    let mux = graph.add_muxer(
        GraphNode::muxer(TestMux {
            inputs: 2,
            pts_ordered: true,
            terminal_work: None,
        }),
        2,
    );
    graph.set_node_name(mux.node(), "mux".into());
    let sink = graph.add_sink(GraphNode::element(DiscardSink));
    graph.link(early, mux.input(0)).unwrap();
    graph.link(late, mux.input(1)).unwrap();
    graph.link(mux.output(), sink).unwrap();

    let visits = run_traced(graph);
    let spans = visits.of("mux");
    assert_eq!(
        spans.len() as u64,
        2 * FRAME_COUNT,
        "one span per input frame"
    );
    let longest_wait = spans.iter().map(|v| v.wait_ns).max();
    assert!(
        longest_wait >= Some(LATE_INPUT_DELAY.as_nanos() as u64),
        "the early input's frames waited for the late one, longest wait {longest_wait:?} ns"
    );
}

async fn until_spans(visits: &RecordedVisits, element: &str) {
    while visits.of(element).len() < SPANS_PER_STEP {
        Yield(false).await;
    }
}

async fn splice_and_replace(
    mutator: &GraphMutator<'_>,
    visits: &RecordedVisits,
    stop: &AtomicBool,
) -> [String; 3] {
    until_spans(visits, "src").await;
    let spliced = mutator
        .insert_after("src", Box::new(PassThrough))
        .await
        .expect("the splice lands");
    until_spans(visits, &spliced).await;
    let (sink, _) = mutator
        .replace_sink("sink", Box::new(DiscardSink))
        .await
        .expect("the sink is replaced");
    until_spans(visits, &sink).await;
    let replacement = CountingSource {
        first_frame_delay: None,
        stop: Some(Arc::new(AtomicBool::new(false))),
    };
    let replacement_stop = replacement.stop.clone().expect("a stop flag");
    let (source, _) = mutator
        .replace_source("src", Box::new(replacement))
        .await
        .expect("the source is replaced");
    until_spans(visits, &source).await;
    stop.store(true, Ordering::SeqCst);
    replacement_stop.store(true, Ordering::SeqCst);
    [spliced, sink, source]
}

#[test]
fn elements_a_mutator_adds_record_spans() {
    let stop = Arc::new(AtomicBool::new(false));
    let mut graph: Graph<GraphNode> = Graph::new();
    let source = graph.add_source(GraphNode::source(CountingSource {
        first_frame_delay: None,
        stop: Some(stop.clone()),
    }));
    graph.set_node_name(source, "src".into());
    let sink = graph.add_sink(GraphNode::element(DiscardSink));
    graph.set_node_name(sink, "sink".into());
    graph.link(source, sink).unwrap();

    let visits = Arc::new(RecordedVisits::default());
    let (mutator, run) = RunBuilder::new(graph, &ZeroClock, LINK_CAPACITY)
        .with_visit_recorder(visits.clone())
        .run_mutable();
    let driver = splice_and_replace(&mutator, &visits, &stop);
    let (stats, added) = block_on(Join2::new(run, driver));
    stats.expect("the run survives the mutations");
    for name in added {
        assert!(
            visits.of(&name).len() >= SPANS_PER_STEP,
            "{name} recorded spans"
        );
    }
}
