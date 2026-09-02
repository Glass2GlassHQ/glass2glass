//! M1149: the source and the sink of a running graph are replaceable in place.
//! The stream keeps flowing across the swap: the old sink renders what was
//! queued for it and finalizes before the replacement takes a single frame, and
//! a replacement source picks up the running time its predecessor reached while
//! stamping from its own zero.
//!
//! Both sinks record into one shared log, tagged with which of them took each
//! frame, so a test reads the swap off the log: no frame lost, none reordered,
//! and one clean transition from the old element to the new one. The
//! caps-changing cases add the `CapsChanged` position, and the source cases the
//! `Segment` position, both of which must reach the sink ahead of the first
//! frame they describe.
//!
//! The mutator is driven concurrently with the run (`Join2`), cooperatively for
//! every case and through the thread-per-arm runner in the `threaded` module.
#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use g2g_core::caps::CapsSet;
use g2g_core::clock::{ClockCandidate, ClockPriority, ClockSync};
use g2g_core::element::DynAsyncElement;
use g2g_core::format_element::CapsConstraint;
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::property::PropValue;
use g2g_core::runtime::{
    block_on, run_graph_mutable, select2, DynSourceLoop, Either, GraphMutator, GraphNode, Join2,
    MutationError, SourceLoop,
};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, Frame, FrameTiming, G2gError, Graph,
    MultiInputElement, OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat, Segment,
};

/// Small on purpose: a shallow link keeps packets queued across the swap.
const LINK_CAPACITY: usize = 2;
/// One frame's worth of presentation time, so a stitched timeline is measurable.
const FRAME_DURATION_NS: u64 = 33_000_000;
/// Where the replacement source's sequence numbers start, so the sink can tell
/// whose frame it is holding no matter how far the first source got.
const REPLACEMENT_BASE_SEQUENCE: u64 = 1_000_000;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn i420() -> Caps {
    video_caps(RawVideoFormat::I420)
}

fn nv12() -> Caps {
    video_caps(RawVideoFormat::Nv12)
}

fn video_caps(format: RawVideoFormat) -> Caps {
    Caps::RawVideo {
        format,
        width: Dim::Fixed(16),
        height: Dim::Fixed(16),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

/// One frame: `sequence` in the header, `mark` in the payload (which source or
/// transform it came through), and a PTS on this source's own timeline.
fn frame(sequence: u64, mark: u8, pts_ns: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(Box::new([mark; 4]))),
        FrameTiming {
            pts_ns,
            duration_ns: FRAME_DURATION_NS,
            ..FrameTiming::default()
        },
        sequence,
    ))
}

fn mark_of(frame: &Frame) -> u8 {
    match &frame.domain {
        MemoryDomain::System(s) => s.as_slice()[0],
        other => panic!("unexpected frame domain {other:?}"),
    }
}

/// Emits frames until the driver stops it, then `Eos`. Every frame carries this
/// source's own `mark`, sequences counted from `first_sequence`, and a PTS
/// counted from zero: a replacement stamps its own timeline, which is what the
/// mutator's segment stitch has to absorb.
struct CountingSource {
    mark: u8,
    first_sequence: u64,
    caps: Caps,
    pushed: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    /// Bounds the stream where a test wants one that ends on its own.
    frames: u64,
}

impl CountingSource {
    fn new(mark: u8, pushed: &Arc<AtomicUsize>, stop: &Arc<AtomicBool>) -> Self {
        Self {
            mark,
            first_sequence: 0,
            caps: i420(),
            pushed: Arc::clone(pushed),
            stop: Arc::clone(stop),
            frames: u64::MAX,
        }
    }

    /// A replacement: its sequences are unmistakably its own.
    fn replacement(mark: u8, pushed: &Arc<AtomicUsize>, stop: &Arc<AtomicBool>) -> Self {
        Self {
            first_sequence: REPLACEMENT_BASE_SEQUENCE,
            ..Self::new(mark, pushed, stop)
        }
    }

    fn producing(mut self, caps: Caps) -> Self {
        self.caps = caps;
        self
    }

    fn frames(mut self, frames: u64) -> Self {
        self.frames = frames;
        self
    }
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
        core::future::ready(Ok(self.caps.clone()))
    }
    fn configure_pipeline(&mut self, caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match *caps == self.caps {
            true => Ok(ConfigureOutcome::Accepted),
            false => Err(G2gError::CapsMismatch),
        }
    }
    /// The count this instance pushed, so a handed-back source is identifiable.
    fn get_property(&self, name: &str) -> Option<PropValue> {
        (name == "pushed").then(|| PropValue::Uint(self.pushed.load(Ordering::SeqCst) as u64))
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let mut emitted = 0;
            while emitted < self.frames {
                if self.stop.load(Ordering::SeqCst) {
                    break;
                }
                let pts = emitted * FRAME_DURATION_NS;
                out.push(frame(self.first_sequence + emitted, self.mark, pts))
                    .await?;
                self.pushed.fetch_add(1, Ordering::SeqCst);
                emitted += 1;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(emitted)
        })
    }
}

/// Forwards everything unchanged: a middle position, so a test can name a
/// transform where a source or a sink is expected.
#[derive(Default)]
struct Pass;

impl AsyncElement for Pass {
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

/// Keeps what reached a terminal fan-in: enough to run one, so a test can try to
/// replace it as if it were a sink.
struct PassMux;

impl MultiInputElement for PassMux {
    type ProcessFuture<'a>
        = core::future::Ready<Result<(), G2gError>>
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
        Ok(i420())
    }
    fn is_terminal(&self) -> bool {
        true
    }
    fn process<'a>(
        &'a mut self,
        _input: usize,
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        core::future::ready(Ok(()))
    }
}

/// One frame as a sink saw it: which sink took it, what the source called it,
/// and the timeline it arrived on.
#[derive(Debug, Clone)]
struct Seen {
    sink: u8,
    sequence: u64,
    mark: u8,
    pts_ns: u64,
    segment: Segment,
}

impl Seen {
    /// This frame's running time, the number the segment stitch has to keep
    /// moving forward across a source swap.
    fn running_ns(&self) -> u64 {
        self.segment
            .to_running_time(self.pts_ns)
            .unwrap_or_else(|| panic!("frame {} is outside its own segment", self.sequence))
    }
}

/// What the sinks saw, in order, and where each control packet landed in it.
#[derive(Default, Debug)]
struct Record {
    frames: Vec<Seen>,
    caps: Vec<(usize, Caps)>,
    segments: Vec<(usize, Segment)>,
}

impl Record {
    fn sequences(&self) -> Vec<u64> {
        self.frames.iter().map(|f| f.sequence).collect()
    }

    fn of_sink(&self, sink: u8) -> Vec<Seen> {
        self.frames
            .iter()
            .filter(|f| f.sink == sink)
            .cloned()
            .collect()
    }
}

/// Records every packet into the shared log under its own `id`, and accepts the
/// formats `accepts` names (which is what the runner's downstream-feasibility
/// snapshot is built from).
struct RecordingSink {
    id: u8,
    record: Arc<Mutex<Record>>,
    accepts: Vec<Caps>,
    /// This instance's own tallies, so a handed-back sink is identifiable and
    /// its end of stream is observable.
    taken: u64,
    finalized: u64,
    segment: Segment,
    /// Offer a clock to election, so the run hands every sink a `ClockSync`.
    provides_clock: bool,
    /// How many times the runner called `set_clock_sync` on this instance.
    synced: u64,
}

impl RecordingSink {
    fn new(id: u8, record: &Arc<Mutex<Record>>, accepts: Vec<Caps>) -> Self {
        Self {
            id,
            record: Arc::clone(record),
            accepts,
            taken: 0,
            finalized: 0,
            segment: Segment::new(),
            provides_clock: false,
            synced: 0,
        }
    }

    fn with_clock(mut self) -> Self {
        self.provides_clock = true;
        self
    }
}

impl AsyncElement for RecordingSink {
    type ProcessFuture<'a>
        = core::future::Ready<Result<(), G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        match self.accepts.contains(upstream) {
            true => Ok(upstream.clone()),
            false => Err(G2gError::CapsMismatch),
        }
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::from_alternatives(self.accepts.clone()))
    }
    fn configure_pipeline(&mut self, caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match self.accepts.contains(caps) {
            true => Ok(ConfigureOutcome::Accepted),
            false => Err(G2gError::CapsMismatch),
        }
    }
    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "taken" => Some(PropValue::Uint(self.taken)),
            "finalized" => Some(PropValue::Uint(self.finalized)),
            "synced" => Some(PropValue::Uint(self.synced)),
            _ => None,
        }
    }
    fn provide_clock(&self) -> Option<ClockCandidate> {
        self.provides_clock
            .then(|| ClockCandidate::new(ClockPriority::AudioProvider, Arc::new(ZeroClock)))
    }
    fn set_clock_sync(&mut self, _sync: ClockSync) {
        self.synced += 1;
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        let mut record = self.record.lock().unwrap();
        match packet {
            PipelinePacket::DataFrame(f) => {
                self.taken += 1;
                record.frames.push(Seen {
                    sink: self.id,
                    sequence: f.sequence,
                    mark: mark_of(&f),
                    pts_ns: f.timing.pts_ns,
                    segment: self.segment,
                });
            }
            PipelinePacket::Segment(segment) => {
                self.segment = segment;
                let at = record.frames.len();
                record.segments.push((at, segment));
            }
            PipelinePacket::CapsChanged(caps) => {
                let at = record.frames.len();
                record.caps.push((at, caps));
            }
            PipelinePacket::Eos => self.finalized += 1,
            _ => {}
        }
        core::future::ready(Ok(()))
    }
}

/// Yields once, re-waking immediately, so a test task can let the run future
/// make progress between checks.
struct Yield(bool);

impl Yield {
    fn once() -> Self {
        Yield(false)
    }
}

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

/// Let the run progress until `ready` holds.
async fn until(ready: impl Fn() -> bool) {
    while !ready() {
        Yield::once().await;
    }
}

/// Resolves once the deadline has passed, re-waking so the run keeps being
/// polled meanwhile.
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

/// A replacement must land while the stream is flowing. The sources here stop
/// only when the driver tells them to, so an operation that ends up waiting for
/// the stream waits for something that is waiting on it: this bound turns that
/// into a failure rather than a hang.
const OP_DEADLINE: Duration = Duration::from_secs(2);

async fn within_deadline<F: Future>(what: &str, op: F) -> F::Output {
    let started = Instant::now();
    match select2(op, Deadline(started + OP_DEADLINE)).await {
        Either::Left(value) => value,
        Either::Right(()) => panic!(
            "{what} did not complete within {OP_DEADLINE:?} while the stream was flowing; \
             a replacement must not wait for the end of the stream"
        ),
    }
}

fn frames_seen(record: &Arc<Mutex<Record>>) -> usize {
    record.lock().unwrap().frames.len()
}

fn frames_from(record: &Arc<Mutex<Record>>, mark: u8) -> usize {
    record
        .lock()
        .unwrap()
        .frames
        .iter()
        .filter(|f| f.mark == mark)
        .count()
}

fn frames_into(record: &Arc<Mutex<Record>>, sink: u8) -> usize {
    record
        .lock()
        .unwrap()
        .frames
        .iter()
        .filter(|f| f.sink == sink)
        .count()
}

/// The value changes exactly once, from `before` to `after`, with both sides
/// non-empty (so the swap really landed mid-stream). Returns the index of the
/// first frame on the far side.
fn assert_single_transition(values: &[u8], before: u8, after: u8, what: &str) -> usize {
    let at = values.iter().position(|&v| v == after).unwrap_or_else(|| {
        panic!("{what}: nothing carries the post-swap value {after}: {values:?}")
    });
    assert!(at > 0, "{what}: the swap must land mid-stream");
    assert!(
        values[..at].iter().all(|&v| v == before),
        "{what}: everything before the swap must come from the old element: {values:?}"
    );
    assert!(
        values[at..].iter().all(|&v| v == after),
        "{what}: nothing may cross the swap out of order: {values:?}"
    );
    at
}

/// `src -> sink`, with a second sink held back for the replacement.
fn source_sink_graph(
    record: &Arc<Mutex<Record>>,
    pushed: &Arc<AtomicUsize>,
    stop: &Arc<AtomicBool>,
    accepts: Vec<Caps>,
) -> Graph<GraphNode> {
    let mut graph: Graph<GraphNode> = Graph::new();
    let source = graph.add_source(GraphNode::source(CountingSource::new(0, pushed, stop)));
    graph.set_node_name(source, "src".into());
    let sink = graph.add_sink(GraphNode::element(RecordingSink::new(0, record, accepts)));
    graph.set_node_name(sink, "sink".into());
    graph.link(source, sink).unwrap();
    graph
}

/// `src -> mid -> sink`, for the addressing cases.
fn source_transform_sink_graph(
    record: &Arc<Mutex<Record>>,
    pushed: &Arc<AtomicUsize>,
    stop: &Arc<AtomicBool>,
) -> Graph<GraphNode> {
    let mut graph: Graph<GraphNode> = Graph::new();
    let source = graph.add_source(GraphNode::source(CountingSource::new(0, pushed, stop)));
    graph.set_node_name(source, "src".into());
    let mid = graph.add_transform(GraphNode::element(Pass));
    graph.set_node_name(mid, "mid".into());
    let sink = graph.add_sink(GraphNode::element(RecordingSink::new(
        0,
        record,
        vec![i420()],
    )));
    graph.set_node_name(sink, "sink".into());
    graph.link(source, mid).unwrap();
    graph.link(mid, sink).unwrap();
    graph
}

/// Swap the sink mid-stream, let the replacement show, and stop the stream.
async fn replace_the_sink(
    mutator: &GraphMutator<'static>,
    record: &Arc<Mutex<Record>>,
    replacement: GraphNode,
    stop: &Arc<AtomicBool>,
) -> (String, Box<dyn DynAsyncElement + 'static>) {
    until(|| frames_seen(record) >= 3).await;
    let GraphNode::Element(element) = replacement else {
        panic!("the replacement sink must be an element");
    };
    let (name, old) = within_deadline(
        "the sink replacement",
        mutator.replace_sink("sink", element),
    )
    .await
    .expect("the replacement accepts the caps on the wire");
    within_deadline(
        "the replacement sink taking frames",
        until(|| frames_into(record, 1) >= 2),
    )
    .await;
    stop.store(true, Ordering::SeqCst);
    (name, old)
}

/// A handed-back element, read through the property surface every element has.
fn property(element: &dyn DynAsyncElement, name: &str) -> u64 {
    match element.get_property(name) {
        Some(PropValue::Uint(v)) => v,
        other => panic!("property {name} reads {other:?}"),
    }
}

fn source_property(source: &dyn DynSourceLoop, name: &str) -> u64 {
    match source.get_property(name) {
        Some(PropValue::Uint(v)) => v,
        other => panic!("property {name} reads {other:?}"),
    }
}

/// The stream partitions cleanly between the two sinks: every frame arrives
/// exactly once, in order, and the old sink finalizes before it comes back.
#[test]
fn a_sink_is_replaced_mid_stream() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let graph = source_sink_graph(&record, &pushed, &stop, vec![i420()]);

    let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
    let observed = Arc::clone(&record);
    let halt = Arc::clone(&stop);
    let replacement = GraphNode::element(RecordingSink::new(1, &record, vec![i420()]));
    let driver = async move { replace_the_sink(&mutator, &observed, replacement, &halt).await };
    let (stats, (name, old)) = block_on(Join2::new(run, driver));
    stats.expect("the run survives the sink swap");

    assert_ne!(name, "sink", "the replacement gets a name of its own");
    let record = record.lock().unwrap();
    let expected: Vec<u64> = (0..record.frames.len() as u64).collect();
    assert_eq!(
        record.sequences(),
        expected,
        "no frame may be lost or reordered across a sink swap"
    );
    let sinks: Vec<u8> = record.frames.iter().map(|f| f.sink).collect();
    assert_single_transition(&sinks, 0, 1, "the sink swap");
    assert_eq!(
        property(&*old, "finalized"),
        1,
        "the old sink must see its end of stream before it is handed back"
    );
    assert_eq!(
        property(&*old, "taken"),
        record.of_sink(0).len() as u64,
        "the sink handed back is the instance that recorded the first stretch"
    );
}

/// The wait-before-unpark guarantee: the old sink's last frame lands before the
/// replacement's first, so nothing is rendered twice or out of order.
#[test]
fn the_old_sink_finishes_before_the_replacement_starts() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let graph = source_sink_graph(&record, &pushed, &stop, vec![i420()]);

    let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
    let observed = Arc::clone(&record);
    let halt = Arc::clone(&stop);
    let replacement = GraphNode::element(RecordingSink::new(1, &record, vec![i420()]));
    let driver = async move { replace_the_sink(&mutator, &observed, replacement, &halt).await };
    let (stats, (_name, _old)) = block_on(Join2::new(run, driver));
    stats.expect("the run survives the sink swap");

    let record = record.lock().unwrap();
    let old = record.of_sink(0);
    let new = record.of_sink(1);
    assert!(!old.is_empty() && !new.is_empty(), "both sinks must render");
    assert!(
        old.last().expect("checked above").sequence < new[0].sequence,
        "the old sink's last frame must land before the replacement's first"
    );
}

/// The elected clock reaches a replacement sink the way it reached the
/// negotiated ones. Two swaps in a row, so both the original and the first
/// replacement come back readable.
#[test]
fn a_replacement_sink_is_handed_the_elected_clock() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let mut graph: Graph<GraphNode> = Graph::new();
    let source = graph.add_source(GraphNode::source(CountingSource::new(0, &pushed, &stop)));
    graph.set_node_name(source, "src".into());
    let sink = graph.add_sink(GraphNode::element(
        RecordingSink::new(0, &record, vec![i420()]).with_clock(),
    ));
    graph.set_node_name(sink, "sink".into());
    graph.link(source, sink).unwrap();

    let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
    let observed = Arc::clone(&record);
    let halt = Arc::clone(&stop);
    let first = GraphNode::element(RecordingSink::new(1, &record, vec![i420()]));
    let second = GraphNode::element(RecordingSink::new(2, &record, vec![i420()]));
    let driver = async move {
        until(|| frames_seen(&observed) >= 3).await;
        let GraphNode::Element(first) = first else {
            panic!("the replacement sink must be an element");
        };
        let (first_name, old) = within_deadline(
            "the first sink replacement",
            mutator.replace_sink("sink", first),
        )
        .await
        .expect("the replacement accepts the caps on the wire");
        within_deadline(
            "the first replacement taking frames",
            until(|| frames_into(&observed, 1) >= 2),
        )
        .await;
        let GraphNode::Element(second) = second else {
            panic!("the replacement sink must be an element");
        };
        let (_, first_back) = within_deadline(
            "the second sink replacement",
            mutator.replace_sink(&first_name, second),
        )
        .await
        .expect("the second replacement accepts the caps on the wire");
        within_deadline(
            "the second replacement taking frames",
            until(|| frames_into(&observed, 2) >= 1),
        )
        .await;
        halt.store(true, Ordering::SeqCst);
        (old, first_back)
    };
    let (stats, (old, first_back)) = block_on(Join2::new(run, driver));
    stats.expect("the run survives both swaps");

    assert_eq!(
        property(&*old, "synced"),
        1,
        "the negotiated sink was handed the elected clock at startup"
    );
    assert_eq!(
        property(&*first_back, "synced"),
        1,
        "a replacement sink is handed the elected clock at the swap"
    );
}

/// A replacement source picks up where its predecessor left off: the consumer
/// sees the old frames, then a fresh segment, then the new source's, whose raw
/// timestamps restart at zero while the running time keeps climbing.
#[test]
fn a_source_is_replaced_mid_stream() {
    let record = Arc::new(Mutex::new(Record::default()));
    let first = Arc::new(AtomicUsize::new(0));
    let second = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let graph = source_sink_graph(&record, &first, &stop, vec![i420()]);

    let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
    let observed = Arc::clone(&record);
    let halt = Arc::clone(&stop);
    let replacement: Box<dyn DynSourceLoop> =
        Box::new(CountingSource::replacement(1, &second, &stop));
    let driver = async move {
        until(|| frames_seen(&observed) >= 3).await;
        let (name, old) = within_deadline(
            "the source replacement",
            mutator.replace_source("src", replacement),
        )
        .await
        .expect("the replacement produces the shape on the wire");
        within_deadline(
            "the replacement source's frames reaching the sink",
            until(|| frames_from(&observed, 1) >= 2),
        )
        .await;
        halt.store(true, Ordering::SeqCst);
        (name, old)
    };
    let (stats, (name, old)) = block_on(Join2::new(run, driver));
    stats.expect("the run survives the source swap");

    assert_ne!(name, "src", "the replacement gets a name of its own");
    assert_eq!(
        source_property(&*old, "pushed"),
        first.load(Ordering::SeqCst) as u64,
        "the source handed back is the instance that fed the first stretch"
    );

    let record = record.lock().unwrap();
    let marks: Vec<u8> = record.frames.iter().map(|f| f.mark).collect();
    let at = assert_single_transition(&marks, 0, 1, "the source swap");
    // Everything the retired source got onto the link is still there, in order.
    let delivered: Vec<u64> = record.frames[..at].iter().map(|f| f.sequence).collect();
    let expected: Vec<u64> = (0..delivered.len() as u64).collect();
    assert_eq!(
        delivered, expected,
        "what the old source had already pushed must all arrive, in order"
    );
    assert_eq!(
        record.frames[at].sequence, REPLACEMENT_BASE_SEQUENCE,
        "the replacement's stream starts at its own first frame"
    );
    assert_eq!(
        record.frames[at].pts_ns, 0,
        "the replacement stamps from its own zero"
    );
    assert_eq!(
        record.segments.iter().map(|&(a, _)| a).collect::<Vec<_>>(),
        vec![0, at],
        "the stitched segment must reach the sink exactly once, ahead of the first frame it times"
    );
    assert!(
        record.segments[1].1.base > 0,
        "the stitched segment must carry the running time the first source reached"
    );
    let last_old = record.frames[at - 1].running_ns();
    assert!(
        last_old > 0,
        "the first source's running time must have moved off zero"
    );
    for frame in &record.frames[at..] {
        assert!(
            frame.running_ns() >= last_old,
            "running time must not go backwards across a source swap: {} after {last_old}",
            frame.running_ns()
        );
    }
    let running: Vec<u64> = record.frames.iter().map(|f| f.running_ns()).collect();
    assert!(
        running.windows(2).all(|w| w[1] > w[0]),
        "running time must keep climbing frame by frame: {running:?}"
    );
}

/// A replacement that changes the shape needs the chain below to accept it, and
/// announces the change ahead of its first frame.
#[test]
fn a_caps_changing_source_replacement_needs_consent() {
    let record = Arc::new(Mutex::new(Record::default()));
    let first = Arc::new(AtomicUsize::new(0));
    let second = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    // The sink takes both shapes, so the swap is allowed.
    let graph = source_sink_graph(&record, &first, &stop, vec![i420(), nv12()]);

    let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
    let observed = Arc::clone(&record);
    let halt = Arc::clone(&stop);
    let recolored: Box<dyn DynSourceLoop> =
        Box::new(CountingSource::replacement(1, &second, &stop).producing(nv12()));
    let driver = async move {
        until(|| frames_seen(&observed) >= 3).await;
        within_deadline(
            "the caps-changing source replacement",
            mutator.replace_source("src", recolored),
        )
        .await
        .expect("the sink accepts NV12, so the swap is allowed");
        within_deadline(
            "the recolored frames reaching the sink",
            until(|| frames_from(&observed, 1) >= 2),
        )
        .await;
        halt.store(true, Ordering::SeqCst);
    };
    let (stats, ()) = block_on(Join2::new(run, driver));
    stats.expect("the run survives the caps-changing source swap");

    let record = record.lock().unwrap();
    let marks: Vec<u8> = record.frames.iter().map(|f| f.mark).collect();
    let at = assert_single_transition(&marks, 0, 1, "the caps-changing source swap");
    assert_eq!(
        record.caps,
        vec![(at, nv12())],
        "the sink must be told the new shape once, before the first frame carrying it"
    );
}

/// The same swap with a sink that takes only the shape on the wire: no consent,
/// no swap, and the stream carries on unchanged.
#[test]
fn a_source_replacement_the_sink_cannot_take_is_refused() {
    let record = Arc::new(Mutex::new(Record::default()));
    let first = Arc::new(AtomicUsize::new(0));
    let second = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let graph = source_sink_graph(&record, &first, &stop, vec![i420()]);

    let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
    let observed = Arc::clone(&record);
    let halt = Arc::clone(&stop);
    let recolored: Box<dyn DynSourceLoop> =
        Box::new(CountingSource::replacement(1, &second, &stop).producing(nv12()));
    let driver = async move {
        until(|| frames_seen(&observed) >= 3).await;
        let refused = within_deadline(
            "the refused source replacement",
            mutator.replace_source("src", recolored),
        )
        .await;
        // The stream is untouched, so a later valid swap still lands.
        let seen = frames_seen(&observed);
        within_deadline(
            "the stream continuing past a refusal",
            until(|| frames_seen(&observed) > seen + 2),
        )
        .await;
        let valid: Box<dyn DynSourceLoop> =
            Box::new(CountingSource::replacement(1, &second, &halt));
        let accepted = within_deadline(
            "the valid source replacement after a refusal",
            mutator.replace_source("src", valid),
        )
        .await;
        within_deadline(
            "the replacement source's frames reaching the sink",
            until(|| frames_from(&observed, 1) >= 2),
        )
        .await;
        halt.store(true, Ordering::SeqCst);
        (refused, accepted)
    };
    let (stats, (refused, accepted)) = block_on(Join2::new(run, driver));
    stats.expect("a refused replacement leaves the run going");

    assert_eq!(
        refused.err(),
        Some(MutationError::DownstreamRefused),
        "a shape the chain below cannot take must be refused"
    );
    accepted.expect("the refusal left nothing behind, so the next swap lands");
    let record = record.lock().unwrap();
    assert_eq!(
        record.caps,
        vec![],
        "a refused swap must not announce a shape change"
    );
    let marks: Vec<u8> = record.frames.iter().map(|f| f.mark).collect();
    assert_single_transition(&marks, 0, 1, "the swap after the refusal");
}

/// Nothing but the end each operation names is replaceable.
#[test]
fn the_addressable_positions_are_the_two_ends() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let graph = source_transform_sink_graph(&record, &pushed, &stop);

    let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
    let observed = Arc::clone(&record);
    let halt = Arc::clone(&stop);
    let spare = Arc::new(AtomicUsize::new(0));
    let driver = async move {
        until(|| frames_seen(&observed) >= 3).await;
        let mut refusals = Vec::new();
        for node in ["mid", "sink"] {
            let source: Box<dyn DynSourceLoop> =
                Box::new(CountingSource::replacement(1, &spare, &halt));
            refusals.push(
                within_deadline(
                    "replace_source on a non-source",
                    mutator.replace_source(node, source),
                )
                .await
                .err(),
            );
        }
        for node in ["src", "mid"] {
            refusals.push(
                within_deadline(
                    "replace_sink on a non-sink",
                    mutator.replace_sink(
                        node,
                        Box::new(RecordingSink::new(9, &observed, vec![i420()])),
                    ),
                )
                .await
                .err(),
            );
        }
        let unknown: Box<dyn DynSourceLoop> =
            Box::new(CountingSource::replacement(1, &spare, &halt));
        refusals.push(
            within_deadline(
                "replace_source on a name nothing carries",
                mutator.replace_source("nope", unknown),
            )
            .await
            .err(),
        );
        halt.store(true, Ordering::SeqCst);
        refusals
    };
    let (stats, refusals) = block_on(Join2::new(run, driver));
    stats.expect("refused addressing leaves the run going");

    assert_eq!(
        refusals,
        vec![
            Some(MutationError::NotMutable("mid".into())),
            Some(MutationError::NotMutable("sink".into())),
            Some(MutationError::NotMutable("src".into())),
            Some(MutationError::NotMutable("mid".into())),
            Some(MutationError::UnknownNode("nope".into())),
        ],
        "each end is addressed by its own operation and nothing else is"
    );
}

/// A replaced sink carries its own set of accepted shapes onto the edge, so a
/// caps change the sink that replaced it takes is still allowed afterwards.
#[test]
fn a_replaced_sink_leaves_the_edge_as_wide_as_it_takes() {
    let record = Arc::new(Mutex::new(Record::default()));
    let first = Arc::new(AtomicUsize::new(0));
    let second = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let graph = source_sink_graph(&record, &first, &stop, vec![i420(), nv12()]);

    let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
    let observed = Arc::clone(&record);
    let halt = Arc::clone(&stop);
    let replacement_sink = RecordingSink::new(1, &record, vec![i420(), nv12()]);
    let recolored: Box<dyn DynSourceLoop> =
        Box::new(CountingSource::replacement(1, &second, &stop).producing(nv12()));
    let driver = async move {
        until(|| frames_seen(&observed) >= 3).await;
        within_deadline(
            "the sink replacement",
            mutator.replace_sink("sink", Box::new(replacement_sink)),
        )
        .await
        .expect("the replacement takes the shape on the wire");
        let recolor = within_deadline(
            "the caps-changing source replacement below a replaced sink",
            mutator.replace_source("src", recolored),
        )
        .await;
        within_deadline(
            "the recolored frames reaching the replacement sink",
            until(|| frames_from(&observed, 1) >= 2),
        )
        .await;
        halt.store(true, Ordering::SeqCst);
        recolor
    };
    let (stats, recolor) = block_on(Join2::new(run, driver));
    stats.expect("the run survives both swaps");

    recolor.expect("the replacement sink accepts NV12, so the source swap is allowed");
    let record = record.lock().unwrap();
    let marks: Vec<u8> = record.frames.iter().map(|f| f.mark).collect();
    let at = assert_single_transition(&marks, 0, 1, "the source swap below a replaced sink");
    assert_eq!(
        record.caps,
        vec![(at, nv12())],
        "the new shape must reach the replacement sink before the first frame carrying it"
    );
}

/// A terminal fan-in is the consumer of its edge and still no sink position: the
/// runner never lent its element out, so there is nothing to hand back.
#[test]
fn a_terminal_fanin_is_not_a_sink_position() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let mut graph: Graph<GraphNode> = Graph::new();
    let session = graph.add_fanin_sink(GraphNode::muxer(PassMux), 1);
    graph.set_node_name(session.node(), "session".into());
    let source = graph.add_source(GraphNode::source(CountingSource::new(0, &pushed, &stop)));
    graph.set_node_name(source, "src".into());
    graph.link(source, session.input(0)).unwrap();

    let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
    let counted = Arc::clone(&pushed);
    let halt = Arc::clone(&stop);
    let observed = Arc::clone(&record);
    let driver = async move {
        until(|| counted.load(Ordering::SeqCst) >= 3).await;
        let refused = within_deadline(
            "replace_sink on a terminal fan-in",
            mutator.replace_sink(
                "session",
                Box::new(RecordingSink::new(9, &observed, vec![i420()])),
            ),
        )
        .await;
        halt.store(true, Ordering::SeqCst);
        refused
    };
    let (stats, refused) = block_on(Join2::new(run, driver));
    stats.expect("the refusal leaves the run going");

    assert_eq!(
        refused.err(),
        Some(MutationError::NotMutable("session".into())),
        "a fan-in's element is not the runner's to hand back"
    );
}

/// Once the stream has ended there is no graph left to change.
#[test]
fn a_replacement_after_the_stream_ended_is_refused() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let mut graph: Graph<GraphNode> = Graph::new();
    let source = graph.add_source(GraphNode::source(
        CountingSource::new(0, &pushed, &stop).frames(4),
    ));
    graph.set_node_name(source, "src".into());
    let sink = graph.add_sink(GraphNode::element(RecordingSink::new(
        0,
        &record,
        vec![i420()],
    )));
    graph.set_node_name(sink, "sink".into());
    graph.link(source, sink).unwrap();

    let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
    let after = mutator.clone();
    let (stats, ()) = block_on(Join2::new(run, async {}));
    stats.expect("the short stream ends on its own");

    let refused = block_on(after.replace_sink(
        "sink",
        Box::new(RecordingSink::new(1, &record, vec![i420()])),
    ));
    assert_eq!(
        refused.err(),
        Some(MutationError::GraphEnded),
        "a replacement has nowhere to land once the run is over"
    );
    assert_eq!(
        record.lock().unwrap().frames.len(),
        4,
        "the stream ran to its end"
    );
}

/// The same swaps on the thread-per-arm runner, where the producer parks on a
/// worker thread of its own.
#[cfg(feature = "multi-thread")]
mod threaded {
    use super::*;
    use g2g_core::runtime::{run_graph_threaded_mutable, ThreadSpawner};

    #[test]
    fn a_sink_is_replaced_in_a_threaded_run() {
        let record = Arc::new(Mutex::new(Record::default()));
        let pushed = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let graph = source_sink_graph(&record, &pushed, &stop, vec![i420()]);

        let spawner = ThreadSpawner;
        let (mutator, run) = run_graph_threaded_mutable(graph, &ZeroClock, LINK_CAPACITY, &spawner);
        let observed = Arc::clone(&record);
        let halt = Arc::clone(&stop);
        let replacement = GraphNode::element(RecordingSink::new(1, &record, vec![i420()]));
        let driver = async move { replace_the_sink(&mutator, &observed, replacement, &halt).await };
        let (stats, (_name, old)) = block_on(Join2::new(run, driver));
        stats.expect("the threaded run survives the sink swap");

        let record = record.lock().unwrap();
        let expected: Vec<u64> = (0..record.frames.len() as u64).collect();
        assert_eq!(
            record.sequences(),
            expected,
            "no frame may be lost or reordered across a sink swap"
        );
        let sinks: Vec<u8> = record.frames.iter().map(|f| f.sink).collect();
        assert_single_transition(&sinks, 0, 1, "the threaded sink swap");
        assert_eq!(
            property(&*old, "finalized"),
            1,
            "the old sink finalizes on its own thread too"
        );
        let old_frames = record.of_sink(0);
        let new_frames = record.of_sink(1);
        assert!(
            old_frames.last().expect("the old sink rendered").sequence < new_frames[0].sequence,
            "the old sink's last frame must land before the replacement's first"
        );
    }

    #[test]
    fn a_source_is_replaced_in_a_threaded_run() {
        let record = Arc::new(Mutex::new(Record::default()));
        let first = Arc::new(AtomicUsize::new(0));
        let second = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let graph = source_sink_graph(&record, &first, &stop, vec![i420()]);

        let spawner = ThreadSpawner;
        let (mutator, run) = run_graph_threaded_mutable(graph, &ZeroClock, LINK_CAPACITY, &spawner);
        let observed = Arc::clone(&record);
        let halt = Arc::clone(&stop);
        let replacement: Box<dyn DynSourceLoop> =
            Box::new(CountingSource::replacement(1, &second, &stop));
        let driver = async move {
            until(|| frames_seen(&observed) >= 3).await;
            within_deadline(
                "the source replacement",
                mutator.replace_source("src", replacement),
            )
            .await
            .expect("the replacement produces the shape on the wire");
            within_deadline(
                "the replacement source's frames reaching the sink",
                until(|| frames_from(&observed, 1) >= 2),
            )
            .await;
            halt.store(true, Ordering::SeqCst);
        };
        let (stats, ()) = block_on(Join2::new(run, driver));
        stats.expect("the threaded run survives the source swap");

        let record = record.lock().unwrap();
        let marks: Vec<u8> = record.frames.iter().map(|f| f.mark).collect();
        let at = assert_single_transition(&marks, 0, 1, "the threaded source swap");
        assert_eq!(
            record.frames[at].pts_ns, 0,
            "the replacement stamps from its own zero"
        );
        let running: Vec<u64> = record.frames.iter().map(|f| f.running_ns()).collect();
        assert!(
            running.windows(2).all(|w| w[1] > w[0]),
            "running time must keep climbing across the swap: {running:?}"
        );
    }
}
