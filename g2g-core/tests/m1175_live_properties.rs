//! M1175: setting and reading an element's properties while its graph runs. The
//! `GraphMutator` queues the operation and the arm that owns the element
//! performs it at its next packet boundary, so the element is only ever touched
//! from the task driving it.
//!
//! Every frame carries the step its transform added, so a test reads the new
//! value off the sink's own record rather than off the element: the value took
//! effect on the stream, not just in a field. The refusal cases check the
//! element's own verdict comes back and that the stream is unchanged by them.
#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::property::{PropError, PropKind, PropValue, PropertySpec};
use g2g_core::runtime::{
    block_on, run_graph_mutable, select2, Either, GraphNode, Join2, MutationError, SourceLoop,
};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, Frame, FrameTiming, G2gError, Graph, OutputSink,
    PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

/// A shallow link keeps packets queued, so a set has to cross what is in flight
/// before the sink sees its effect.
const LINK_CAPACITY: usize = 2;
/// The step the transform starts on, so the stream carries a value before any
/// set and a change is visible as a transition rather than as a first mark.
const INITIAL_STEP: u64 = 1;
/// What a test sets it to.
const NEW_STEP: u64 = 5;
/// The largest step the element accepts: one past it is what a refused set is.
const MAX_STEP: u64 = 9;
/// Frames the bounded stream carries, for the tests that run to the end.
const FRAME_COUNT: u64 = 8;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn i420() -> Caps {
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
        FrameTiming {
            pts_ns: sequence * 33_000_000,
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

/// Emits frames until it reaches `frames` or the driver raises `stop`, then
/// `Eos`.
struct CountingSource {
    pushed: Arc<AtomicUsize>,
    frames: u64,
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
                out.push(frame(sequence)).await?;
                self.pushed.fetch_add(1, Ordering::SeqCst);
                emitted += 1;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(emitted)
        })
    }
}

static STEPPER_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "step",
    PropKind::Uint,
    "how much this element adds to each frame's first payload byte",
)
.with_range("0", "9")];

/// Adds `step` to every frame's first payload byte, so the value the element is
/// carrying right now is readable off the sink's record. Refuses anything past
/// [`MAX_STEP`], which is the refusal a live set has to carry back.
struct Stepper {
    step: u64,
}

impl AsyncElement for Stepper {
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
    fn properties(&self) -> &'static [PropertySpec] {
        STEPPER_PROPS
    }
    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "step" => {
                let step = value.as_uint().ok_or(PropError::Type)?;
                if step > MAX_STEP {
                    return Err(PropError::Value);
                }
                self.step = step;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }
    fn get_property(&self, name: &str) -> Option<PropValue> {
        (name == "step").then_some(PropValue::Uint(self.step))
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            let PipelinePacket::DataFrame(mut data) = packet else {
                return Ok(());
            };
            if let MemoryDomain::System(bytes) = &mut data.domain {
                bytes.as_mut_slice()[0] += self.step as u8;
            }
            out.push(PipelinePacket::DataFrame(data)).await?;
            Ok(())
        })
    }
}

/// Every frame's mark, in arrival order.
#[derive(Default, Debug)]
struct Record {
    marks: Vec<u8>,
    sequences: Vec<u64>,
}

struct RecordingSink {
    record: Arc<Mutex<Record>>,
}

impl AsyncElement for RecordingSink {
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
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        if let PipelinePacket::DataFrame(f) = packet {
            let mut record = self.record.lock().unwrap();
            record.marks.push(mark_of(&f));
            record.sequences.push(f.sequence);
        }
        core::future::ready(Ok(()))
    }
}

/// `src -> mid -> sink`, the shape every test below drives.
fn stepper_graph(
    record: &Arc<Mutex<Record>>,
    pushed: &Arc<AtomicUsize>,
    frames: u64,
    stop: Option<&Arc<AtomicBool>>,
) -> Graph<GraphNode> {
    let mut graph: Graph<GraphNode> = Graph::new();
    let source = graph.add_source(GraphNode::source(CountingSource {
        pushed: Arc::clone(pushed),
        frames,
        stop: stop.map(Arc::clone),
    }));
    graph.set_node_name(source, "src".into());
    let mid = graph.add_transform(GraphNode::element(Stepper { step: INITIAL_STEP }));
    graph.set_node_name(mid, "mid".into());
    let sink = graph.add_sink(GraphNode::element(RecordingSink {
        record: Arc::clone(record),
    }));
    graph.set_node_name(sink, "sink".into());
    graph.link(source, mid).unwrap();
    graph.link(mid, sink).unwrap();
    graph
}

/// A stream the driver stops when it has seen what it needs.
fn driven_graph(
    record: &Arc<Mutex<Record>>,
    pushed: &Arc<AtomicUsize>,
    stop: &Arc<AtomicBool>,
) -> Graph<GraphNode> {
    stepper_graph(record, pushed, u64::MAX, Some(stop))
}

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

/// A property operation must land while the stream is flowing: the sources here
/// stop only when the driver says so, so an operation that waits for the stream
/// instead waits for something waiting on it.
const OP_DEADLINE: Duration = Duration::from_secs(2);

async fn within_deadline<F: Future>(what: &str, op: F) -> F::Output {
    let started = Instant::now();
    match select2(op, Deadline(started + OP_DEADLINE)).await {
        Either::Left(value) => value,
        Either::Right(()) => panic!(
            "{what} did not complete within {OP_DEADLINE:?} while the stream was flowing; \
             a property operation must not wait for the end of the stream"
        ),
    }
}

/// A wedged run has nothing left to poll it, so a deadline inside the run future
/// cannot catch it.
const RUN_DEADLINE: Duration = Duration::from_secs(10);

fn run_within_deadline<T: Send + 'static>(
    what: &str,
    run: impl FnOnce() -> T + Send + 'static,
) -> T {
    let (done, finished) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = done.send(run());
    });
    match finished.recv_timeout(RUN_DEADLINE) {
        Ok(value) => value,
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            panic!("{what} was still running after {RUN_DEADLINE:?}: the run is wedged")
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            panic!("{what} panicked; its own failure is above")
        }
    }
}

fn frames_seen(record: &Arc<Mutex<Record>>) -> usize {
    record.lock().unwrap().marks.len()
}

fn marks(record: &Arc<Mutex<Record>>) -> Vec<u8> {
    record.lock().unwrap().marks.clone()
}

/// The marks the sink saw, with each run of equal ones collapsed, so a test
/// reads the changes rather than their lengths.
fn mark_runs(marks: &[u8]) -> Vec<u8> {
    let mut runs: Vec<u8> = Vec::new();
    for &m in marks {
        if runs.last() != Some(&m) {
            runs.push(m);
        }
    }
    runs
}

/// Every frame arrived exactly once, in the order the source emitted them.
fn assert_in_order(record: &Record) {
    let expected: Vec<u64> = (0..record.sequences.len() as u64).collect();
    assert_eq!(
        record.sequences, expected,
        "every frame must arrive exactly once, in order, across a live property set"
    );
}

/// Let the stream carry `count` more frames past what it has, so a test can see
/// what a refused operation did (or did not do) to it.
async fn carry_on(record: &Arc<Mutex<Record>>, count: usize) {
    let target = frames_seen(record) + count;
    within_deadline(
        "the stream carrying on",
        until(|| frames_seen(record) >= target),
    )
    .await;
}

#[test]
fn a_live_set_changes_the_frames_reaching_the_sink() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let graph = driven_graph(&record, &pushed, &stop);

    let seen = Arc::clone(&record);
    let halt = Arc::clone(&stop);
    let (stats, ()) = run_within_deadline("a live property set", move || {
        let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
        let driver = async move {
            until(|| frames_seen(&seen) >= 3).await;
            within_deadline(
                "setting step on a running transform",
                mutator.set_property("mid", "step", PropValue::Uint(NEW_STEP)),
            )
            .await
            .expect("the element accepts a step inside its range");
            within_deadline(
                "the stepped frames reaching the sink",
                until(|| marks(&seen).contains(&(NEW_STEP as u8))),
            )
            .await;
            halt.store(true, Ordering::SeqCst);
        };
        block_on(Join2::new(run, driver))
    });
    stats.expect("the run survives a live property set");

    let record = record.lock().unwrap();
    assert_in_order(&record);
    assert_eq!(
        mark_runs(&record.marks),
        vec![INITIAL_STEP as u8, NEW_STEP as u8],
        "the stream carries the old step, then the new one, and changes exactly once"
    );
}

#[test]
fn a_live_get_reads_back_what_was_set() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let graph = driven_graph(&record, &pushed, &stop);

    let seen = Arc::clone(&record);
    let halt = Arc::clone(&stop);
    let (stats, (before, after, unknown)) = run_within_deadline("a live property get", move || {
        let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
        let driver = async move {
            until(|| frames_seen(&seen) >= 3).await;
            let before = within_deadline("reading step", mutator.get_property("mid", "step"))
                .await
                .expect("the transform takes property reads");
            within_deadline(
                "setting step",
                mutator.set_property("mid", "step", PropValue::Uint(NEW_STEP)),
            )
            .await
            .expect("the element accepts a step inside its range");
            let after = within_deadline("reading step back", mutator.get_property("mid", "step"))
                .await
                .expect("the transform takes property reads");
            let unknown = within_deadline(
                "reading a name the element has not got",
                mutator.get_property("mid", "nope"),
            )
            .await
            .expect("an unknown name is the element's own answer, not an error");
            halt.store(true, Ordering::SeqCst);
            (before, after, unknown)
        };
        block_on(Join2::new(run, driver))
    });
    stats.expect("the run survives a live property get");

    assert_eq!(
        before,
        Some(PropValue::Uint(INITIAL_STEP)),
        "a read before any set returns the value the element was built with"
    );
    assert_eq!(
        after,
        Some(PropValue::Uint(NEW_STEP)),
        "a read after a set sees that set"
    );
    assert_eq!(
        unknown, None,
        "a name the element does not carry is `Ok(None)`, the element's own answer"
    );
}

#[test]
fn a_refused_value_comes_back_as_the_elements_own_error() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let graph = driven_graph(&record, &pushed, &stop);

    let seen = Arc::clone(&record);
    let halt = Arc::clone(&stop);
    let (stats, (out_of_range, unknown)) = run_within_deadline("a refused live set", move || {
        let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
        let driver = async move {
            until(|| frames_seen(&seen) >= 3).await;
            let out_of_range = within_deadline(
                "setting step past the element's range",
                mutator.set_property("mid", "step", PropValue::Uint(MAX_STEP + 1)),
            )
            .await;
            let unknown = within_deadline(
                "setting a name the element has not got",
                mutator.set_property("mid", "nope", PropValue::Uint(1)),
            )
            .await;
            // The refusals must leave the stream as it was, which only frames
            // that arrive after them can show.
            carry_on(&seen, 3).await;
            halt.store(true, Ordering::SeqCst);
            (out_of_range, unknown)
        };
        block_on(Join2::new(run, driver))
    });
    stats.expect("a refused property set leaves the run to carry on");

    assert_eq!(
        out_of_range,
        Err(MutationError::PropertyRejected(PropError::Value)),
        "the element's own verdict on an out-of-range value comes back"
    );
    assert_eq!(
        unknown,
        Err(MutationError::PropertyRejected(PropError::Unknown)),
        "a property the element does not declare is refused by the element, not by the mutator"
    );
    let record = record.lock().unwrap();
    assert_in_order(&record);
    assert_eq!(
        mark_runs(&record.marks),
        vec![INITIAL_STEP as u8],
        "a refused set changes nothing about the frames"
    );
}

#[test]
fn an_unknown_node_is_refused_by_name() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let graph = driven_graph(&record, &pushed, &stop);

    let seen = Arc::clone(&record);
    let halt = Arc::clone(&stop);
    let (stats, (set, get)) = run_within_deadline("a set on an unknown node", move || {
        let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
        let driver = async move {
            until(|| frames_seen(&seen) >= 3).await;
            let set = within_deadline(
                "setting on a node that does not exist",
                mutator.set_property("nope", "step", PropValue::Uint(NEW_STEP)),
            )
            .await;
            let get = within_deadline(
                "reading from a node that does not exist",
                mutator.get_property("nope", "step"),
            )
            .await;
            halt.store(true, Ordering::SeqCst);
            (set, get)
        };
        block_on(Join2::new(run, driver))
    });
    stats.expect("naming a node that is not there leaves the run alone");

    assert_eq!(set, Err(MutationError::UnknownNode("nope".into())));
    assert_eq!(get, Err(MutationError::UnknownNode("nope".into())));
}

#[test]
fn a_source_has_no_packet_boundary_to_act_at() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let graph = driven_graph(&record, &pushed, &stop);

    let seen = Arc::clone(&record);
    let halt = Arc::clone(&stop);
    let (stats, refused) = run_within_deadline("a set on a source", move || {
        let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
        let driver = async move {
            until(|| frames_seen(&seen) >= 3).await;
            let refused = within_deadline(
                "setting a property on the source",
                mutator.set_property("src", "step", PropValue::Uint(NEW_STEP)),
            )
            .await;
            halt.store(true, Ordering::SeqCst);
            refused
        };
        block_on(Join2::new(run, driver))
    });
    stats.expect("a refused set on a source leaves the run alone");

    assert_eq!(
        refused,
        Err(MutationError::NotMutable("src".into())),
        "a source drives itself, so no arm hands it packets one at a time"
    );
}

#[test]
fn a_set_after_the_run_ended_is_refused() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let graph = stepper_graph(&record, &pushed, FRAME_COUNT, None);

    let refused = run_within_deadline("a set after the run ended", move || {
        let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
        let stats = block_on(run).expect("the bounded stream ends on its own");
        assert_eq!(stats.frames_consumed, FRAME_COUNT);
        // The run future is gone, so the service that would serve this is too.
        block_on(mutator.set_property("mid", "step", PropValue::Uint(NEW_STEP)))
    });

    assert_eq!(refused, Err(MutationError::GraphEnded));
}

/// The thread-per-arm runner takes the same operations: the element is touched
/// on whichever worker thread its arm runs on, at that arm's packet boundary.
#[cfg(feature = "multi-thread")]
mod threaded {
    use super::*;
    use g2g_core::runtime::{run_graph_threaded_mutable, ThreadSpawner};

    #[test]
    fn a_live_set_changes_the_frames_in_a_threaded_run() {
        let record = Arc::new(Mutex::new(Record::default()));
        let pushed = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let graph = driven_graph(&record, &pushed, &stop);

        let seen = Arc::clone(&record);
        let halt = Arc::clone(&stop);
        let (stats, ()) = run_within_deadline("a threaded live property set", move || {
            let (mutator, run) =
                run_graph_threaded_mutable(graph, &ZeroClock, LINK_CAPACITY, &ThreadSpawner);
            let driver = async move {
                until(|| frames_seen(&seen) >= 3).await;
                within_deadline(
                    "setting step on a threaded transform",
                    mutator.set_property("mid", "step", PropValue::Uint(NEW_STEP)),
                )
                .await
                .expect("the element accepts a step inside its range");
                within_deadline(
                    "the stepped frames reaching the sink",
                    until(|| marks(&seen).contains(&(NEW_STEP as u8))),
                )
                .await;
                halt.store(true, Ordering::SeqCst);
            };
            block_on(Join2::new(run, driver))
        });
        stats.expect("the threaded run survives a live property set");

        let record = record.lock().unwrap();
        assert_in_order(&record);
        assert_eq!(
            mark_runs(&record.marks),
            vec![INITIAL_STEP as u8, NEW_STEP as u8],
            "the threaded stream carries the old step, then the new one"
        );
    }
}
