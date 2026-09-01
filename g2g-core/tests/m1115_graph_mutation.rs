//! M1115: structural mutation of a running graph. A transform is spliced onto a
//! live edge, and lifted back off, while frames keep flowing.
//!
//! Each frame carries its sequence number and a mark counting the transforms it
//! crossed, so every test reads the splice off the sink's own record: no frame
//! lost, none reordered, and the marked ones sitting on exactly one side of the
//! change. The caps-changing cases add the `CapsChanged` position: it must reach
//! the sink before the first frame of the new shape.
//!
//! The mutator is driven concurrently with the run (`Join2`), cooperatively for
//! most cases and through the thread-per-arm runner for the last two.
#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use g2g_core::caps::CapsSet;
use g2g_core::format_element::CapsConstraint;
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::property::PropValue;
use g2g_core::runtime::{
    block_on, run_graph_mutable, select2, Either, GraphNode, Join2, MutationError, SourceLoop,
};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, Frame, FrameTiming, G2gError, Graph, OutputSink,
    PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

/// Long enough that a mutation lands with most of the stream still to come.
const FRAME_COUNT: u64 = 400;
/// Small on purpose: a shallow link keeps packets queued across the splice.
const LINK_CAPACITY: usize = 2;
/// Frames the gated transform takes before it stops, so a remove has to drain a
/// queue that has built up at its input.
const GATE_AT: u64 = 4;

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

/// One frame: `sequence` in the frame header, `marks` in the payload.
fn frame(sequence: u64, marks: u8) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(Box::new([marks; 4]))),
        FrameTiming {
            pts_ns: sequence * 33_000_000,
            ..FrameTiming::default()
        },
        sequence,
    ))
}

fn marks_of(frame: &Frame) -> u8 {
    match &frame.domain {
        MemoryDomain::System(s) => s.as_slice()[0],
        other => panic!("unexpected frame domain {other:?}"),
    }
}

/// Emits `FRAME_COUNT` unmarked frames then `Eos`, counting its completed
/// pushes so a test can tell how far the stream has got. With `eos` off it just
/// stops, which is how a failing source ends: the graph then winds down because
/// the link closes behind it.
struct CountingSource {
    pushed: Arc<AtomicUsize>,
    eos: bool,
    frames: u64,
    /// Ends the stream when the driver raises it, for the tests that want a
    /// stream running for as long as they need one.
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
        core::future::ready(Ok(i420()))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let mut emitted = 0;
            for sequence in 0..self.frames {
                if self.stop.as_ref().is_some_and(|s| s.load(Ordering::SeqCst)) {
                    break;
                }
                out.push(frame(sequence, 0)).await?;
                self.pushed.fetch_add(1, Ordering::SeqCst);
                emitted += 1;
            }
            if self.eos {
                out.push(PipelinePacket::Eos).await?;
            }
            Ok(emitted)
        })
    }
}

/// Marks every frame it forwards, so the sink can tell which frames crossed it,
/// and counts them on a property so a removed instance can be identified.
/// `allow` caps how many frames it takes: past that it waits, which is what
/// leaves a queue at its input for a remove to drain.
#[derive(Default)]
struct Marker {
    marked: u64,
    allow: Option<Arc<AtomicUsize>>,
}

impl Marker {
    fn gated(allow: Arc<AtomicUsize>) -> Self {
        Self {
            marked: 0,
            allow: Some(allow),
        }
    }
}

impl AsyncElement for Marker {
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
    fn get_property(&self, name: &str) -> Option<PropValue> {
        (name == "marked").then_some(PropValue::Uint(self.marked))
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            let PipelinePacket::DataFrame(mut data) = packet else {
                if !matches!(packet, PipelinePacket::Eos) {
                    out.push(packet).await?;
                }
                return Ok(());
            };
            if let Some(allow) = &self.allow {
                while (self.marked as usize) >= allow.load(Ordering::SeqCst) {
                    Yield::once().await;
                }
            }
            if let MemoryDomain::System(bytes) = &mut data.domain {
                bytes.as_mut_slice()[0] += 1;
            }
            self.marked += 1;
            out.push(PipelinePacket::DataFrame(data)).await?;
            Ok(())
        })
    }
}

/// Marks like [`Marker`] and changes the caps while doing it: I420 in, NV12 out.
#[derive(Default)]
struct Recolor {
    marked: u64,
}

impl AsyncElement for Recolor {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        match upstream {
            Caps::RawVideo {
                format: RawVideoFormat::I420,
                ..
            } => Ok(nv12()),
            _ => Err(G2gError::CapsMismatch),
        }
    }
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Mapping(alloc_pairs())
    }
    fn configure_pipeline(&mut self, caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match caps {
            Caps::RawVideo {
                format: RawVideoFormat::I420,
                ..
            } => Ok(ConfigureOutcome::Accepted),
            _ => Err(G2gError::CapsMismatch),
        }
    }
    fn get_property(&self, name: &str) -> Option<PropValue> {
        (name == "marked").then_some(PropValue::Uint(self.marked))
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            let PipelinePacket::DataFrame(mut data) = packet else {
                if !matches!(packet, PipelinePacket::Eos) {
                    out.push(packet).await?;
                }
                return Ok(());
            };
            if let MemoryDomain::System(bytes) = &mut data.domain {
                bytes.as_mut_slice()[0] += 1;
            }
            self.marked += 1;
            out.push(PipelinePacket::DataFrame(data)).await?;
            Ok(())
        })
    }
}

fn alloc_pairs() -> Vec<(CapsSet, CapsSet)> {
    vec![(CapsSet::one(i420()), CapsSet::one(nv12()))]
}

/// Accepts nothing this graph carries, so a splice attempt is refused.
struct Picky;

impl AsyncElement for Picky {
    type ProcessFuture<'a>
        = core::future::Ready<Result<(), G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&self, _upstream: &Caps) -> Result<Caps, G2gError> {
        Err(G2gError::CapsMismatch)
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Err(G2gError::CapsMismatch)
    }
    fn process<'a>(
        &'a mut self,
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        core::future::ready(Ok(()))
    }
}

/// What the sink saw, in order: every frame's sequence and mark count, and the
/// frame position each `CapsChanged` arrived at.
#[derive(Default, Debug)]
struct Record {
    frames: Vec<(u64, u8)>,
    caps: Vec<(usize, Caps)>,
}

impl Record {
    fn marks(&self) -> Vec<u8> {
        self.frames.iter().map(|&(_, m)| m).collect()
    }
}

/// Records every packet, and accepts the formats `accepts` names (which is what
/// the runner's downstream-feasibility snapshot is built from). `pace` makes it
/// take its time over each frame, which keeps the shallow link full and so keeps
/// the producer inside a blocking send: that is the state a mutation has to be
/// able to interrupt.
struct RecordingSink {
    record: Arc<Mutex<Record>>,
    accepts: Vec<Caps>,
    pace: Option<Duration>,
}

impl AsyncElement for RecordingSink {
    type ProcessFuture<'a>
        = core::future::Ready<Result<(), G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        if self.accepts.contains(upstream) {
            Ok(upstream.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::from_alternatives(self.accepts.clone()))
    }
    fn configure_pipeline(&mut self, caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if self.accepts.contains(caps) {
            Ok(ConfigureOutcome::Accepted)
        } else {
            Err(G2gError::CapsMismatch)
        }
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        let mut record = self.record.lock().unwrap();
        match packet {
            PipelinePacket::DataFrame(f) => {
                let entry = (f.sequence, marks_of(&f));
                record.frames.push(entry);
                if let Some(pace) = self.pace {
                    drop(record);
                    std::thread::sleep(pace);
                    return core::future::ready(Ok(()));
                }
            }
            PipelinePacket::CapsChanged(caps) => {
                let at = record.frames.len();
                record.caps.push((at, caps));
            }
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

/// A mutation must land while the stream is flowing. The sources in the cycling
/// tests below stop only when the driver tells them to, so an operation that
/// ends up waiting for the stream instead waits for something that is waiting on
/// it: this bound is what turns that into a failure rather than a hang. Orders
/// of magnitude above the microseconds an operation actually takes.
const OP_DEADLINE: Duration = Duration::from_secs(2);

/// Run one mutation, failing if it does not land inside [`OP_DEADLINE`].
async fn within_deadline<F: Future>(what: &str, op: F) -> F::Output {
    let started = Instant::now();
    match select2(op, Deadline(started + OP_DEADLINE)).await {
        Either::Left(value) => value,
        Either::Right(()) => panic!(
            "{what} did not complete within {OP_DEADLINE:?} while the stream was flowing; \
             a mutation must not wait for the end of the stream"
        ),
    }
}

fn frames_seen(record: &Arc<Mutex<Record>>) -> usize {
    record.lock().unwrap().frames.len()
}

/// Every frame arrived exactly once, in the order the source emitted them.
fn assert_complete_in_order(record: &Record) {
    let sequences: Vec<u64> = record.frames.iter().map(|&(s, _)| s).collect();
    let expected: Vec<u64> = (0..FRAME_COUNT).collect();
    assert_eq!(
        sequences, expected,
        "every frame must arrive exactly once, in order, across the splice"
    );
}

/// The mark count changes exactly once, from `before` to `after`, and both sides
/// are non-empty (so the mutation really landed mid-stream). Returns the index
/// of the first frame on the far side.
fn assert_single_transition(marks: &[u8], before: u8, after: u8) -> usize {
    let at = marks
        .iter()
        .position(|&m| m == after)
        .unwrap_or_else(|| panic!("no frame carries the post-mutation mark {after}: {marks:?}"));
    assert!(
        at > 0,
        "the mutation must land mid-stream, not before the first frame"
    );
    assert!(
        marks[..at].iter().all(|&m| m == before),
        "frames before the mutation must be untouched by it"
    );
    assert!(
        marks[at..].iter().all(|&m| m == after),
        "no frame may cross the mutation point out of order: {marks:?}"
    );
    at
}

fn source_transform_sink(
    record: &Arc<Mutex<Record>>,
    pushed: &Arc<AtomicUsize>,
    middle: Option<GraphNode>,
    accepts: Vec<Caps>,
) -> Graph<GraphNode> {
    graph_with_source(record, pushed, middle, StreamShape::new(accepts))
}

/// How a test wants its stream to run, beyond the elements in it.
struct StreamShape {
    accepts: Vec<Caps>,
    eos: bool,
    frames: u64,
    stop: Option<Arc<AtomicBool>>,
    pace: Option<Duration>,
}

impl StreamShape {
    /// `FRAME_COUNT` frames then `Eos`, taken by a sink accepting `accepts`.
    fn new(accepts: Vec<Caps>) -> Self {
        Self {
            accepts,
            eos: true,
            frames: FRAME_COUNT,
            stop: None,
            pace: None,
        }
    }

    fn frames(mut self, frames: u64) -> Self {
        self.frames = frames;
        self
    }

    /// Ends by stopping rather than by saying so.
    fn without_eos(mut self) -> Self {
        self.eos = false;
        self
    }

    /// Runs for as long as the test needs it, ending when `stop` is raised.
    fn until_stopped(mut self, stop: &Arc<AtomicBool>) -> Self {
        self.frames = u64::MAX;
        self.stop = Some(Arc::clone(stop));
        self
    }

    fn paced(mut self, pace: Duration) -> Self {
        self.pace = Some(pace);
        self
    }
}

fn graph_with_source(
    record: &Arc<Mutex<Record>>,
    pushed: &Arc<AtomicUsize>,
    middle: Option<GraphNode>,
    shape: StreamShape,
) -> Graph<GraphNode> {
    let mut graph: Graph<GraphNode> = Graph::new();
    let source = graph.add_source(GraphNode::source(CountingSource {
        pushed: Arc::clone(pushed),
        eos: shape.eos,
        frames: shape.frames,
        stop: shape.stop,
    }));
    graph.set_node_name(source, "src".into());
    let sink = graph.add_sink(GraphNode::element(RecordingSink {
        record: Arc::clone(record),
        accepts: shape.accepts,
        pace: shape.pace,
    }));
    graph.set_node_name(sink, "sink".into());
    match middle {
        Some(element) => {
            let mid = graph.add_transform(element);
            graph.set_node_name(mid, "mid".into());
            graph.link(source, mid).unwrap();
            graph.link(mid, sink).unwrap();
        }
        None => graph.link(source, sink).unwrap(),
    }
    graph
}

#[test]
fn a_passthrough_transform_splices_into_a_flowing_graph() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let graph = source_transform_sink(&record, &pushed, None, vec![i420()]);

    let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
    let observed = Arc::clone(&record);
    let driver = async {
        until(|| frames_seen(&observed) >= 3).await;
        mutator
            .insert_after("src", Box::new(Marker::default()))
            .await
            .expect("a passthrough splices onto the edge it can carry")
    };
    let (stats, name) = block_on(Join2::new(run, driver));
    let stats = stats.expect("the run survives the splice");

    // `<category>N`, the runner's own naming, the category being this element's
    // (defaulted) log category.
    assert_eq!(
        name, "Marker0",
        "the spliced element is named like any other"
    );
    assert_eq!(stats.frames_consumed, FRAME_COUNT);
    let record = record.lock().unwrap();
    assert_complete_in_order(&record);
    assert_single_transition(&record.marks(), 0, 1);
    assert!(
        record.caps.is_empty(),
        "a passthrough splice changes no caps, so the sink sees no re-solve"
    );
}

#[test]
fn a_caps_changing_splice_announces_itself_before_its_first_frame() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let graph = source_transform_sink(&record, &pushed, None, vec![i420(), nv12()]);

    let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
    let observed = Arc::clone(&record);
    let driver = async {
        until(|| frames_seen(&observed) >= 3).await;
        mutator
            .insert_after("src", Box::new(Recolor::default()))
            .await
            .expect("the sink accepts NV12, so the splice is allowed")
    };
    let (stats, _name) = block_on(Join2::new(run, driver));
    stats.expect("the run survives the splice");

    let record = record.lock().unwrap();
    assert_complete_in_order(&record);
    let at = assert_single_transition(&record.marks(), 0, 1);
    assert_eq!(
        record.caps,
        vec![(at, nv12())],
        "the new shape must reach the sink once, before the first frame carrying it"
    );
}

#[test]
fn an_element_that_refuses_the_edge_caps_leaves_the_stream_alone() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let graph = source_transform_sink(&record, &pushed, None, vec![i420()]);

    let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
    let observed = Arc::clone(&record);
    let driver = async {
        until(|| frames_seen(&observed) >= 3).await;
        mutator.insert_after("src", Box::new(Picky)).await
    };
    let (stats, refused) = block_on(Join2::new(run, driver));
    let stats = stats.expect("a refused splice leaves the run going");

    assert_eq!(
        refused,
        Err(MutationError::Refused(G2gError::CapsMismatch)),
        "an element that cannot take the edge's caps must be refused"
    );
    assert_eq!(stats.frames_consumed, FRAME_COUNT);
    let record = record.lock().unwrap();
    assert_complete_in_order(&record);
    assert!(
        record.marks().iter().all(|&m| m == 0),
        "nothing was spliced in, so no frame is marked"
    );
}

#[test]
fn a_position_that_is_not_a_transform_edge_is_refused() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let graph = source_transform_sink(&record, &pushed, None, vec![i420()]);

    let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
    let driver = async {
        let unknown = mutator
            .insert_after("nowhere", Box::new(Marker::default()))
            .await;
        let past_the_sink = mutator
            .insert_after("sink", Box::new(Marker::default()))
            .await;
        let sink_removal = mutator.remove("sink").await.err();
        (unknown, past_the_sink, sink_removal)
    };
    let (stats, (unknown, past_the_sink, sink_removal)) = block_on(Join2::new(run, driver));
    stats.expect("refused addresses leave the run going");

    assert_eq!(unknown, Err(MutationError::UnknownNode("nowhere".into())));
    assert_eq!(
        past_the_sink,
        Err(MutationError::NotMutable("sink".into())),
        "a sink has no edge below it to splice onto"
    );
    assert_eq!(
        sink_removal,
        Some(MutationError::NotMutable("sink".into())),
        "only a transform can be lifted out"
    );
}

#[test]
fn a_removed_element_drains_first_and_comes_back_to_the_caller() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let allow = Arc::new(AtomicUsize::new(GATE_AT as usize));
    let graph = source_transform_sink(
        &record,
        &pushed,
        Some(GraphNode::element(Marker::gated(Arc::clone(&allow)))),
        vec![i420()],
    );

    let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
    let observed = Arc::clone(&record);
    let counted = Arc::clone(&pushed);
    let release = Arc::clone(&allow);
    let driver = async {
        // Wait until the element is holding at its gate with a full input link
        // behind it, so the remove has a real queue to drain through it.
        let queued = GATE_AT as usize + LINK_CAPACITY;
        until(|| counted.load(Ordering::SeqCst) >= queued).await;
        let seen_before = frames_seen(&observed);
        let removed = Join2::new(mutator.remove("mid"), async {
            for _ in 0..8 {
                Yield::once().await;
            }
            release.store(usize::MAX, Ordering::SeqCst);
        })
        .await
        .0
        .expect("a transform on a 1:1 edge can be lifted out");
        (removed, seen_before)
    };
    let (stats, (removed, seen_before)) = block_on(Join2::new(run, driver));
    let stats = stats.expect("the run survives the removal");
    assert_eq!(stats.frames_consumed, FRAME_COUNT);

    let record = record.lock().unwrap();
    assert_complete_in_order(&record);
    let marks = record.marks();
    let at = assert_single_transition(&marks, 1, 0);
    let marked = match removed.get_property("marked") {
        Some(PropValue::Uint(n)) => n,
        other => panic!("the removed element must come back live, got {other:?}"),
    };
    assert_eq!(
        marked as usize, at,
        "the element that came back is the one that marked the frames the sink saw"
    );
    assert!(
        marked as usize > seen_before,
        "frames queued at its input when the remove began drained through it \
         ({marked} marked vs {seen_before} already delivered)"
    );
}

#[test]
fn removing_a_caps_changing_element_renegotiates_downstream() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let graph = source_transform_sink(
        &record,
        &pushed,
        Some(GraphNode::element(Recolor::default())),
        vec![i420(), nv12()],
    );

    let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
    let observed = Arc::clone(&record);
    let driver = async {
        until(|| frames_seen(&observed) >= 3).await;
        mutator
            .remove("mid")
            .await
            .expect("the sink accepts the producer's I420, so the bypass is allowed")
    };
    let (stats, _removed) = block_on(Join2::new(run, driver));
    stats.expect("the run survives the removal");

    let record = record.lock().unwrap();
    assert_complete_in_order(&record);
    let at = assert_single_transition(&record.marks(), 1, 0);
    assert_eq!(
        record.caps,
        vec![(at, i420())],
        "the sink must be told the producer's shape before the first frame in it"
    );
}

#[test]
fn removing_a_caps_changing_element_a_sink_depends_on_is_refused() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    // The sink takes NV12 only, which only the transform produces.
    let graph = source_transform_sink(
        &record,
        &pushed,
        Some(GraphNode::element(Recolor::default())),
        vec![nv12()],
    );

    let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
    let observed = Arc::clone(&record);
    let driver = async {
        until(|| frames_seen(&observed) >= 3).await;
        mutator.remove("mid").await.err()
    };
    let (stats, refused) = block_on(Join2::new(run, driver));
    let stats = stats.expect("a refused removal leaves the run going");

    assert_eq!(
        refused,
        Some(MutationError::DownstreamRefused),
        "the sink cannot take what the producer emits, so the element stays"
    );
    assert_eq!(stats.frames_consumed, FRAME_COUNT);
    let record = record.lock().unwrap();
    assert_complete_in_order(&record);
    assert!(
        record.marks().iter().all(|&m| m == 1),
        "every frame still crosses the element that was not removed"
    );
}

/// A second splice at the same position is checked against what the first one
/// was configured to take, not waved through. Without that check the caps change
/// reaches an element that never agreed to it and kills the run mid-stream.
#[test]
fn a_splice_above_a_spliced_element_still_needs_downstream_consent() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let graph = source_transform_sink(&record, &pushed, None, vec![i420()]);

    let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
    let observed = Arc::clone(&record);
    let driver = async {
        until(|| frames_seen(&observed) >= 3).await;
        mutator
            .insert_after("src", Box::new(Marker::default()))
            .await
            .expect("the passthrough splices onto an I420 edge");
        mutator
            .insert_after("src", Box::new(Recolor::default()))
            .await
    };
    let (stats, refused) = block_on(Join2::new(run, driver));
    let stats = stats.expect("the refused second splice leaves the run going");

    assert_eq!(
        refused,
        Err(MutationError::DownstreamRefused),
        "NV12 is not what the element below was configured to take"
    );
    assert_eq!(stats.frames_consumed, FRAME_COUNT);
    let record = record.lock().unwrap();
    assert_complete_in_order(&record);
    assert_single_transition(&record.marks(), 0, 1);
    assert!(
        record.caps.is_empty(),
        "the refused splice changed no caps downstream"
    );
}

/// A mutation asked for as the stream ends resolves instead of hanging: the
/// request can be queued at the very moment the run future (and with it the
/// service that would serve it) goes away.
#[test]
fn a_mutation_racing_the_end_of_the_stream_resolves() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let graph = graph_with_source(
        &record,
        &pushed,
        None,
        StreamShape::new(vec![i420()]).frames(1),
    );

    let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
    let counted = Arc::clone(&pushed);
    let driver = async {
        until(|| counted.load(Ordering::SeqCst) >= 1).await;
        mutator
            .insert_after("src", Box::new(Marker::default()))
            .await
    };
    let (stats, spliced) = block_on(Join2::new(run, driver));
    let stats = stats.expect("the run ends normally");

    assert_eq!(stats.frames_consumed, 1);
    assert_eq!(
        spliced,
        Err(MutationError::GraphEnded),
        "the source is already gone by the time the request is made"
    );
}

/// Cycles of insert-then-remove with nothing in between, over a stream that
/// runs until the driver stops it. Back to back is the case where the next
/// operation asks the producer to park while it is still on its way back from
/// the last one's resume, and the request has to survive that.
const MUTATION_CYCLES: usize = 5;
/// How long the sink spends on each frame during those cycles. Enough that the
/// shallow link stays full, so the producer is parked in a send when the next
/// operation asks it to stop, which is where the two used to be able to cancel
/// each other out.
const SINK_PACE: Duration = Duration::from_micros(200);

fn cycle_graph(
    record: &Arc<Mutex<Record>>,
    pushed: &Arc<AtomicUsize>,
    stop: &Arc<AtomicBool>,
) -> Graph<GraphNode> {
    graph_with_source(
        record,
        pushed,
        None,
        StreamShape::new(vec![i420()])
            .until_stopped(stop)
            .paced(SINK_PACE),
    )
}

/// Every operation landed, and the stream came through whole. `emitted` is what
/// the source reported.
fn assert_cycles_delivered(record: &Record, emitted: u64) {
    let sequences: Vec<u64> = record.frames.iter().map(|&(s, _)| s).collect();
    let expected: Vec<u64> = (0..emitted).collect();
    assert_eq!(
        sequences, expected,
        "every frame the source emitted must arrive exactly once, in order, across the cycles"
    );
    assert!(
        emitted > MUTATION_CYCLES as u64,
        "the stream must outlast the cycles, got {emitted} frames"
    );
}

#[test]
fn back_to_back_cycles_keep_landing_while_the_stream_flows() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let graph = cycle_graph(&record, &pushed, &stop);

    let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
    let observed = Arc::clone(&record);
    let halt = Arc::clone(&stop);
    let driver = async move {
        until(|| frames_seen(&observed) >= 3).await;
        for cycle in 0..MUTATION_CYCLES {
            let name = within_deadline(
                "an insert",
                mutator.insert_after("src", Box::new(Marker::default())),
            )
            .await
            .unwrap_or_else(|e| panic!("insert {cycle} refused: {e:?}"));
            within_deadline(&format!("the remove of {name}"), mutator.remove(&name))
                .await
                .unwrap_or_else(|e| panic!("remove {cycle} refused: {e:?}"));
        }
        halt.store(true, Ordering::SeqCst);
    };
    let (stats, ()) = block_on(Join2::new(run, driver));
    let stats = stats.expect("the run survives the cycles");

    let record = record.lock().unwrap();
    assert_cycles_delivered(&record, stats.frames_emitted);
}

/// A mutable run winds down the way any other does. The splice machinery holds
/// no link of its own behind the arms, so a source that stops without an `Eos`
/// still closes the link below it and ends the graph.
#[test]
fn a_source_that_stops_without_an_eos_still_ends_a_mutable_run() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let graph = graph_with_source(
        &record,
        &pushed,
        None,
        StreamShape::new(vec![i420()]).without_eos(),
    );

    let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
    let observed = Arc::clone(&record);
    let driver = async {
        until(|| frames_seen(&observed) >= 3).await;
        mutator
            .insert_after("src", Box::new(Marker::default()))
            .await
            .expect("the splice lands before the source stops")
    };
    let (stats, _name) = block_on(Join2::new(run, driver));
    stats.expect("the run ends when the source's link closes behind it");

    let record = record.lock().unwrap();
    assert_complete_in_order(&record);
}

/// The thread-per-arm runner performs both operations the same way: the splice
/// happens between packets on whichever OS thread the producer is running on,
/// and a spliced element gets a worker thread of its own.
#[cfg(feature = "multi-thread")]
mod threaded {
    use super::*;
    use g2g_core::runtime::{run_graph_threaded_mutable, ThreadSpawner};

    #[test]
    fn a_transform_splices_into_a_threaded_run() {
        let record = Arc::new(Mutex::new(Record::default()));
        let pushed = Arc::new(AtomicUsize::new(0));
        let graph = source_transform_sink(&record, &pushed, None, vec![i420()]);

        let spawner = ThreadSpawner;
        let (mutator, run) = run_graph_threaded_mutable(graph, &ZeroClock, LINK_CAPACITY, &spawner);
        let observed = Arc::clone(&record);
        let driver = async {
            until(|| frames_seen(&observed) >= 3).await;
            mutator
                .insert_after("src", Box::new(Marker::default()))
                .await
                .expect("the splice works the same on its own thread")
        };
        let (stats, _name) = block_on(Join2::new(run, driver));
        let stats = stats.expect("the threaded run survives the splice");
        assert_eq!(stats.frames_consumed, FRAME_COUNT);

        let record = record.lock().unwrap();
        assert_complete_in_order(&record);
        assert_single_transition(&record.marks(), 0, 1);
    }

    /// The back-to-back cycles with the producer on its own thread, where the
    /// next park lands in the window between the last resume and the producer
    /// waking up to see it.
    #[test]
    fn back_to_back_cycles_keep_landing_while_the_stream_flows() {
        let record = Arc::new(Mutex::new(Record::default()));
        let pushed = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let graph = cycle_graph(&record, &pushed, &stop);

        let spawner = ThreadSpawner;
        let (mutator, run) = run_graph_threaded_mutable(graph, &ZeroClock, LINK_CAPACITY, &spawner);
        let observed = Arc::clone(&record);
        let halt = Arc::clone(&stop);
        let driver = async move {
            until(|| frames_seen(&observed) >= 3).await;
            for cycle in 0..MUTATION_CYCLES {
                let name = within_deadline(
                    "an insert",
                    mutator.insert_after("src", Box::new(Marker::default())),
                )
                .await
                .unwrap_or_else(|e| panic!("insert {cycle} refused: {e:?}"));
                within_deadline(&format!("the remove of {name}"), mutator.remove(&name))
                    .await
                    .unwrap_or_else(|e| panic!("remove {cycle} refused: {e:?}"));
            }
            halt.store(true, Ordering::SeqCst);
        };
        let (stats, ()) = block_on(Join2::new(run, driver));
        let stats = stats.expect("the threaded run survives the cycles");

        let record = record.lock().unwrap();
        assert_cycles_delivered(&record, stats.frames_emitted);
    }

    /// The end-of-stream race on worker threads: whichever side wins, the call
    /// resolves and the run finishes.
    #[test]
    fn a_mutation_racing_the_end_of_the_stream_resolves() {
        let record = Arc::new(Mutex::new(Record::default()));
        let pushed = Arc::new(AtomicUsize::new(0));
        let graph = graph_with_source(
            &record,
            &pushed,
            None,
            StreamShape::new(vec![i420()]).frames(1),
        );

        let spawner = ThreadSpawner;
        let (mutator, run) = run_graph_threaded_mutable(graph, &ZeroClock, LINK_CAPACITY, &spawner);
        let counted = Arc::clone(&pushed);
        let driver = async {
            until(|| counted.load(Ordering::SeqCst) >= 1).await;
            mutator
                .insert_after("src", Box::new(Marker::default()))
                .await
        };
        let (stats, spliced) = block_on(Join2::new(run, driver));
        let stats = stats.expect("the run ends normally");

        assert_eq!(stats.frames_consumed, 1);
        // The source arm may or may not still be alive when the request lands,
        // so either verdict is legitimate; hanging is not, and neither is any
        // other error.
        assert!(
            matches!(spliced, Ok(_) | Err(MutationError::GraphEnded)),
            "the racing splice must resolve one way or the other, got {spliced:?}"
        );
    }

    #[test]
    fn a_transform_is_lifted_out_of_a_threaded_run() {
        let record = Arc::new(Mutex::new(Record::default()));
        let pushed = Arc::new(AtomicUsize::new(0));
        // Driver-stopped stream: a bounded one can finish before the driver's
        // remove lands on a slow thread schedule, turning the op into a
        // legitimate GraphEnded and the test into a coin flip.
        let stop = Arc::new(AtomicBool::new(false));
        let graph = graph_with_source(
            &record,
            &pushed,
            Some(GraphNode::element(Marker::default())),
            StreamShape::new(vec![i420()]).until_stopped(&stop),
        );

        let spawner = ThreadSpawner;
        let (mutator, run) = run_graph_threaded_mutable(graph, &ZeroClock, LINK_CAPACITY, &spawner);
        let observed = Arc::clone(&record);
        let halt = Arc::clone(&stop);
        let driver = async move {
            until(|| frames_seen(&observed) >= 3).await;
            let removed = mutator
                .remove("mid")
                .await
                .expect("the element is lifted out on its own thread");
            halt.store(true, Ordering::SeqCst);
            removed
        };
        let (stats, removed) = block_on(Join2::new(run, driver));
        let stats = stats.expect("the threaded run survives the removal");

        let record = record.lock().unwrap();
        let sequences: Vec<u64> = record.frames.iter().map(|&(s, _)| s).collect();
        let expected: Vec<u64> = (0..stats.frames_emitted).collect();
        assert_eq!(
            sequences, expected,
            "every frame must arrive exactly once, in order, across the removal"
        );
        let at = assert_single_transition(&record.marks(), 1, 0);
        assert_eq!(
            removed.get_property("marked"),
            Some(PropValue::Uint(at as u64)),
            "the element that came back is the one that marked those frames"
        );
    }
}
