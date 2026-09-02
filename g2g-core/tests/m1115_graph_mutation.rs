//! M1115: structural mutation of a running graph. A transform is spliced onto a
//! live edge, and lifted back off, while frames keep flowing. M1132: a removed
//! element's internal queue reaches the consumer before the bypass. M1133: the
//! edges next to a tee, a demux and a fan-in take a splice too. M1146: so does
//! the edge into a tee, addressed by the name the runner gives it, and a refused
//! remove leaves no drain flag behind.
//!
//! Each frame carries its sequence number and a mark counting the transforms it
//! crossed, so every test reads the splice off the sink's own record: no frame
//! lost, none reordered, and the marked ones sitting on exactly one side of the
//! change. The caps-changing cases add the `CapsChanged` position: it must reach
//! the sink before the first frame of the new shape.
//!
//! The mutator is driven concurrently with the run (`Join2`), cooperatively for
//! most cases and through the thread-per-arm runner in the `threaded` module.
#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use g2g_core::caps::CapsSet;
use g2g_core::format_element::CapsConstraint;
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::property::PropValue;
use g2g_core::runtime::{
    block_on, run_graph_mutable, select2, Either, GraphMutator, GraphNode, Join2, MutationError,
    SourceLoop,
};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, Frame, FrameTiming, G2gError, Graph,
    MultiInputElement, OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
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

/// Frames a [`Holdback`] keeps inside itself at any moment.
const HOLD_DEPTH: usize = 3;

/// Runs [`HOLD_DEPTH`] frames behind its input the way a reordering element
/// does: each frame joins the queue and the oldest leaves, so that many frames
/// sit inside the element the whole time. It releases them on `Eos` and nowhere
/// else, so a remove that does not flush it loses exactly those frames. Marks
/// what it forwards, like [`Marker`].
#[derive(Default)]
struct Holdback {
    held: VecDeque<Frame>,
    released: u64,
}

impl AsyncElement for Holdback {
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
        (name == "released").then_some(PropValue::Uint(self.released))
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(mut data) => {
                    if let MemoryDomain::System(bytes) = &mut data.domain {
                        bytes.as_mut_slice()[0] += 1;
                    }
                    self.held.push_back(data);
                    if self.held.len() > HOLD_DEPTH {
                        let due = self.held.pop_front().expect("the queue is over depth");
                        out.push(PipelinePacket::DataFrame(due)).await?;
                    }
                    Ok(())
                }
                PipelinePacket::Eos => {
                    while let Some(due) = self.held.pop_front() {
                        self.released += 1;
                        out.push(PipelinePacket::DataFrame(due)).await?;
                    }
                    // Forwarded like any element's: during a flush-on-remove the
                    // adapter swallows it, since the run carries on without this
                    // element.
                    out.push(PipelinePacket::Eos).await?;
                    Ok(())
                }
                other => {
                    out.push(other).await?;
                    Ok(())
                }
            }
        })
    }
}

/// Holds one frame back and releases it on `Eos`, like [`Holdback`], and parks
/// inside the `process` call for the frame numbered `park_at` until the driver
/// releases it. That park is what lets a test put an operation and the end of
/// the source's stream inside one `process` call, deterministically.
struct ParkingHoldback {
    park_at: u64,
    parked: Arc<AtomicBool>,
    release: Arc<AtomicBool>,
    held: Option<Frame>,
}

impl AsyncElement for ParkingHoldback {
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
            match packet {
                PipelinePacket::DataFrame(mut data) => {
                    if data.sequence == self.park_at {
                        self.parked.store(true, Ordering::SeqCst);
                        until(|| self.release.load(Ordering::SeqCst)).await;
                    }
                    if let MemoryDomain::System(bytes) = &mut data.domain {
                        bytes.as_mut_slice()[0] += 1;
                    }
                    if let Some(due) = self.held.replace(data) {
                        out.push(PipelinePacket::DataFrame(due)).await?;
                    }
                    Ok(())
                }
                PipelinePacket::Eos => {
                    if let Some(due) = self.held.take() {
                        out.push(PipelinePacket::DataFrame(due)).await?;
                    }
                    out.push(PipelinePacket::Eos).await?;
                    Ok(())
                }
                other => {
                    out.push(other).await?;
                    Ok(())
                }
            }
        })
    }
}

/// Forwards every frame from any input pad to the merged output unchanged:
/// enough to run a fan-in position while a splice lands on one of its input
/// edges. The runner emits the merged `Eos`, so this never forwards one.
///
/// With `record` set it is a *terminal* fan-in instead (no output edge at all),
/// and keeps what reached it itself, since there is no sink below it to read the
/// stream off.
struct PassMux {
    inputs: usize,
    record: Option<Arc<Mutex<Record>>>,
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
        Ok(i420())
    }
    fn is_terminal(&self) -> bool {
        self.record.is_some()
    }
    fn process<'a>(
        &'a mut self,
        _input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            // Only the frames: each pad announces the same caps, and a merged
            // output that repeated them would tell the sink nothing new.
            if let PipelinePacket::DataFrame(data) = packet {
                match &self.record {
                    Some(record) => record
                        .lock()
                        .unwrap()
                        .frames
                        .push((data.sequence, marks_of(&data))),
                    None => {
                        out.push(PipelinePacket::DataFrame(data)).await?;
                    }
                }
            }
            Ok(())
        })
    }
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
    eos_count: usize,
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
            PipelinePacket::Eos => record.eos_count += 1,
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

/// A wedged run has nothing left to poll it, so no deadline inside the run
/// future can catch it. Driving the whole run from a thread of its own turns
/// that into a failing test rather than a suite that never finishes.
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
    record.lock().unwrap().frames.len()
}

/// Frames the sink has seen carrying `mark`, i.e. that crossed that many
/// marking transforms.
fn marked_seen(record: &Arc<Mutex<Record>>, mark: u8) -> usize {
    record
        .lock()
        .unwrap()
        .marks()
        .iter()
        .filter(|&&m| m == mark)
        .count()
}

/// Every frame arrived exactly once, in the order the source emitted them.
fn assert_complete_in_order(record: &Record) {
    assert_no_gap(record, FRAME_COUNT);
}

/// As [`assert_complete_in_order`], for a stream the driver stops rather than
/// one of known length: `emitted` is what the source reported.
fn assert_no_gap(record: &Record, emitted: u64) {
    let sequences: Vec<u64> = record.frames.iter().map(|&(s, _)| s).collect();
    let expected: Vec<u64> = (0..emitted).collect();
    assert_eq!(
        sequences, expected,
        "every frame must arrive exactly once, in order, across the mutation"
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

/// A source the driver stops, and a sink recording what reaches it. The two
/// halves every structural graph below is built from.
fn driven_source(pushed: &Arc<AtomicUsize>, stop: &Arc<AtomicBool>) -> GraphNode {
    GraphNode::source(CountingSource {
        pushed: Arc::clone(pushed),
        eos: true,
        frames: u64::MAX,
        stop: Some(Arc::clone(stop)),
    })
}

fn recording_sink(record: &Arc<Mutex<Record>>) -> GraphNode {
    sink_accepting(record, vec![i420()])
}

fn sink_accepting(record: &Arc<Mutex<Record>>, accepts: Vec<Caps>) -> GraphNode {
    GraphNode::element(RecordingSink {
        record: Arc::clone(record),
        accepts,
        pace: None,
    })
}

/// M1133: a tee with a sink on each branch. The splice goes onto the branch
/// edge, which is named by its consumer since the tee has several. The tee
/// itself is left unnamed on purpose: the runner gives it [`TEE_NAME`], which is
/// what M1146 addresses the edge above it by.
fn tee_graph(
    left: &Arc<Mutex<Record>>,
    right: &Arc<Mutex<Record>>,
    pushed: &Arc<AtomicUsize>,
    stop: &Arc<AtomicBool>,
) -> Graph<GraphNode> {
    tee_graph_accepting(left, right, pushed, stop, vec![i420()])
}

/// The name the runner gives the first unnamed broadcast tee (M1146), the
/// `<category>N` convention every other instance is named by.
const TEE_NAME: &str = "tee0";

fn tee_graph_accepting(
    left: &Arc<Mutex<Record>>,
    right: &Arc<Mutex<Record>>,
    pushed: &Arc<AtomicUsize>,
    stop: &Arc<AtomicBool>,
    accepts: Vec<Caps>,
) -> Graph<GraphNode> {
    let mut graph: Graph<GraphNode> = Graph::new();
    let source = graph.add_source(driven_source(pushed, stop));
    graph.set_node_name(source, "src".into());
    let tee = graph.add_tee(2);
    let left_sink = graph.add_sink(sink_accepting(left, accepts.clone()));
    graph.set_node_name(left_sink, "left".into());
    let right_sink = graph.add_sink(sink_accepting(right, accepts));
    graph.set_node_name(right_sink, "right".into());
    graph.link(source, tee.input()).unwrap();
    graph.link(tee.out(0), left_sink).unwrap();
    graph.link(tee.out(1), right_sink).unwrap();
    graph
}

/// M1133: a two-pad muxer, one pad driven and one silent. The splice goes onto
/// the driven pad's edge, which is named by its producer since the muxer has
/// several inbound edges.
fn muxer_graph(
    record: &Arc<Mutex<Record>>,
    pushed: &Arc<AtomicUsize>,
    stop: &Arc<AtomicBool>,
) -> Graph<GraphNode> {
    let mut graph: Graph<GraphNode> = Graph::new();
    let mux = graph.add_muxer(
        GraphNode::muxer(PassMux {
            inputs: 2,
            record: None,
        }),
        2,
    );
    graph.set_node_name(mux.node(), "mux".into());
    let source = graph.add_source(driven_source(pushed, stop));
    graph.set_node_name(source, "src".into());
    // Ends at once: the fan-in arm holds the merged `Eos` until every pad has
    // one, so a pad with nothing to say costs the run nothing.
    let quiet = graph.add_source(GraphNode::source(CountingSource {
        pushed: Arc::new(AtomicUsize::new(0)),
        eos: true,
        frames: 0,
        stop: None,
    }));
    graph.set_node_name(quiet, "quiet".into());
    let sink = graph.add_sink(recording_sink(record));
    graph.set_node_name(sink, "sink".into());
    graph.link(source, mux.input(0)).unwrap();
    graph.link(quiet, mux.input(1)).unwrap();
    graph.link(mux.output(), sink).unwrap();
    graph
}

/// M1133: a terminal fan-in (no merged output at all) taking one driven pad.
/// The consumer it records for is the fan-in element itself.
fn fanin_sink_graph(
    record: &Arc<Mutex<Record>>,
    pushed: &Arc<AtomicUsize>,
    stop: &Arc<AtomicBool>,
) -> Graph<GraphNode> {
    let mut graph: Graph<GraphNode> = Graph::new();
    let session = graph.add_fanin_sink(
        GraphNode::muxer(PassMux {
            inputs: 1,
            record: Some(Arc::clone(record)),
        }),
        1,
    );
    let source = graph.add_source(driven_source(pushed, stop));
    graph.set_node_name(source, "src".into());
    graph.link(source, session.input(0)).unwrap();
    graph
}

/// Splice a marker onto the edge on one side of `at`, let it show at the sink,
/// lift it back off, and stop the stream once the bypass shows too. `before`
/// picks the side: the edge entering `at`, else the one leaving it.
async fn splice_and_lift(
    mutator: &GraphMutator<'_>,
    watched: &Arc<Mutex<Record>>,
    at: &str,
    before: bool,
    stop: &Arc<AtomicBool>,
) {
    until(|| frames_seen(watched) >= 3).await;
    let element = Box::new(Marker::default());
    let name = within_deadline("the splice onto a structural edge", async {
        match before {
            true => mutator.insert_before(at, element).await,
            false => mutator.insert_after(at, element).await,
        }
    })
    .await
    .unwrap_or_else(|e| panic!("the splice at {at} was refused: {e:?}"));
    within_deadline(
        "the spliced frames reaching the sink",
        until(|| marked_seen(watched, 1) >= 2),
    )
    .await;
    let seen = frames_seen(watched);
    within_deadline(&format!("the remove of {name}"), mutator.remove(&name))
        .await
        .unwrap_or_else(|e| panic!("the remove of {name} was refused: {e:?}"));
    within_deadline(
        "the bypassed frames reaching the sink",
        until(|| frames_seen(watched) > seen + 2),
    )
    .await;
    stop.store(true, Ordering::SeqCst);
}

/// The marks a branch saw, collapsed to the runs they form: `[0, 1, 0]` is the
/// whole shape of a splice that landed and was lifted off again.
fn mark_runs(record: &Record) -> Vec<u8> {
    let mut runs: Vec<u8> = Vec::new();
    for mark in record.marks() {
        if runs.last() != Some(&mark) {
            runs.push(mark);
        }
    }
    runs
}

/// Every frame this consumer saw arrived in order from the first, none lost.
/// Used where the stream outlives the run's own frame count (a tee branch, a
/// muxer pad), so the source's total is not the number to compare against.
fn assert_contiguous_from_start(record: &Record) {
    let sequences: Vec<u64> = record.frames.iter().map(|&(s, _)| s).collect();
    let expected: Vec<u64> = (0..sequences.len() as u64).collect();
    assert_eq!(
        sequences, expected,
        "no frame may be lost or reordered across a structural-edge mutation"
    );
}

/// M1133: the edge from a tee output to its branch consumer takes a splice and
/// gives it back, and the sibling branch never notices.
#[test]
fn a_transform_splices_onto_a_tee_branch() {
    let left = Arc::new(Mutex::new(Record::default()));
    let right = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let graph = tee_graph(&left, &right, &pushed, &stop);

    let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
    let watched = Arc::clone(&left);
    let halt = Arc::clone(&stop);
    let driver = async move { splice_and_lift(&mutator, &watched, "left", true, &halt).await };
    let (stats, ()) = block_on(Join2::new(run, driver));
    stats.expect("the run survives a splice on one of its tee branches");

    let left = left.lock().unwrap();
    let right = right.lock().unwrap();
    assert_contiguous_from_start(&left);
    assert_contiguous_from_start(&right);
    assert_eq!(
        mark_runs(&left),
        vec![0, 1, 0],
        "the spliced element marked a stretch of the branch and stopped when it was lifted off"
    );
    assert_eq!(
        mark_runs(&right),
        vec![0],
        "the sibling branch carried nothing the splice touched"
    );
}

/// M1146: a broadcast tee carries no element, and the runner names it all the
/// same, so the one edge into it is addressable by that name. The splice lands
/// above the fan-out, so every branch carries it.
#[test]
fn a_transform_splices_onto_the_edge_into_a_tee() {
    let left = Arc::new(Mutex::new(Record::default()));
    let right = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let graph = tee_graph(&left, &right, &pushed, &stop);

    let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
    let watched = Arc::clone(&left);
    let halt = Arc::clone(&stop);
    let driver = async move { splice_and_lift(&mutator, &watched, TEE_NAME, true, &halt).await };
    let (stats, ()) = block_on(Join2::new(run, driver));
    stats.expect("the run survives a splice on the edge above its tee");

    let left = left.lock().unwrap();
    let right = right.lock().unwrap();
    assert_contiguous_from_start(&left);
    assert_contiguous_from_start(&right);
    assert_eq!(
        mark_runs(&left),
        vec![0, 1, 0],
        "the splice above the tee marked a stretch of this branch"
    );
    assert_eq!(
        mark_runs(&right),
        vec![0, 1, 0],
        "a splice above the tee reaches every branch, not just the watched one"
    );
}

/// M1146: the shape a splice above a tee changes the edge to has to reach every
/// branch, ahead of the first frame carrying it, the way it reaches a single
/// consumer.
#[test]
fn a_caps_changing_splice_above_a_tee_announces_itself_on_every_branch() {
    let left = Arc::new(Mutex::new(Record::default()));
    let right = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let graph = tee_graph_accepting(&left, &right, &pushed, &stop, vec![i420(), nv12()]);

    let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
    let watched = Arc::clone(&left);
    let sibling = Arc::clone(&right);
    let halt = Arc::clone(&stop);
    let driver = async move {
        until(|| frames_seen(&watched) >= 3).await;
        within_deadline(
            "a caps-changing splice above a tee",
            mutator.insert_before(TEE_NAME, Box::new(Recolor::default())),
        )
        .await
        .expect("both branches accept NV12, so the splice is allowed");
        within_deadline(
            "the recolored frames reaching both branches",
            until(|| marked_seen(&watched, 1) >= 2 && marked_seen(&sibling, 1) >= 2),
        )
        .await;
        halt.store(true, Ordering::SeqCst);
    };
    let (stats, ()) = block_on(Join2::new(run, driver));
    stats.expect("the run survives a caps-changing splice above its tee");

    for branch in [&left, &right] {
        let record = branch.lock().unwrap();
        assert_contiguous_from_start(&record);
        let at = assert_single_transition(&record.marks(), 0, 1);
        assert_eq!(
            record.caps,
            vec![(at, nv12())],
            "each branch must be told the new shape once, before its first frame carrying it"
        );
    }
}

/// M1146 names the tee, which makes the edge above it addressable; the edges
/// below it stay ambiguous, so the tee is still no `insert_after` position.
#[test]
fn insert_after_a_tee_is_still_refused() {
    let left = Arc::new(Mutex::new(Record::default()));
    let right = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let graph = tee_graph(&left, &right, &pushed, &stop);

    let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
    let watched = Arc::clone(&left);
    let halt = Arc::clone(&stop);
    let driver = async move {
        until(|| frames_seen(&watched) >= 3).await;
        let answer = within_deadline(
            "insert_after on a tee",
            mutator.insert_after(TEE_NAME, Box::new(Marker::default())),
        )
        .await;
        halt.store(true, Ordering::SeqCst);
        answer
    };
    let (stats, answer) = block_on(Join2::new(run, driver));
    stats.expect("a refused address leaves the run going");

    assert_eq!(
        answer,
        Err(MutationError::NotMutable(TEE_NAME.into())),
        "a tee has a branch per consumer below it, so insert_after must not pick one"
    );
    assert!(
        left.lock().unwrap().marks().iter().all(|&m| m == 0),
        "nothing was spliced in, so no branch carries a marked frame"
    );
}

/// M1133: the edge from an upstream element into a muxer input pad takes a
/// splice and gives it back.
#[test]
fn a_transform_splices_onto_a_muxer_input() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let graph = muxer_graph(&record, &pushed, &stop);

    let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
    let watched = Arc::clone(&record);
    let halt = Arc::clone(&stop);
    let driver = async move { splice_and_lift(&mutator, &watched, "src", false, &halt).await };
    let (stats, ()) = block_on(Join2::new(run, driver));
    stats.expect("the run survives a splice on one of its muxer inputs");

    let record = record.lock().unwrap();
    assert_contiguous_from_start(&record);
    assert_eq!(
        mark_runs(&record),
        vec![0, 1, 0],
        "the spliced element marked a stretch of the pad's stream and stopped when it was lifted off"
    );
}

/// M1133: the edge into a terminal fan-in's input pad, which has no merged
/// output below it, takes a splice on the same terms as a muxer's.
#[test]
fn a_transform_splices_onto_a_terminal_fanin_input() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let graph = fanin_sink_graph(&record, &pushed, &stop);

    let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
    let watched = Arc::clone(&record);
    let halt = Arc::clone(&stop);
    let driver = async move { splice_and_lift(&mutator, &watched, "src", false, &halt).await };
    let (stats, ()) = block_on(Join2::new(run, driver));
    stats.expect("the run survives a splice on its terminal fan-in's input");

    let record = record.lock().unwrap();
    assert_contiguous_from_start(&record);
    assert_eq!(mark_runs(&record), vec![0, 1, 0]);
}

/// A structural node is still not a splice point: neither end of one names a
/// single edge, so both sides of it are refused.
#[test]
fn a_structural_node_itself_is_still_refused() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let graph = muxer_graph(&record, &pushed, &stop);

    let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
    let observed = Arc::clone(&record);
    let halt = Arc::clone(&stop);
    let driver = async move {
        until(|| frames_seen(&observed) >= 3).await;
        let below_the_pads = mutator
            .insert_before("mux", Box::new(Marker::default()))
            .await;
        let lifted = mutator.remove("mux").await.err();
        halt.store(true, Ordering::SeqCst);
        (below_the_pads, lifted)
    };
    let (stats, (below_the_pads, lifted)) = block_on(Join2::new(run, driver));
    stats.expect("refused addresses leave the run going");

    assert_eq!(
        below_the_pads,
        Err(MutationError::NotMutable("mux".into())),
        "a muxer has several inbound edges, so none of them is the edge above it"
    );
    assert_eq!(
        lifted,
        Some(MutationError::NotMutable("mux".into())),
        "only a transform can be lifted out"
    );
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

/// M1132. Built so the flush is the only way the held frames can arrive: the
/// element is [`HOLD_DEPTH`] frames behind its input, so removing it without one
/// leaves a gap of exactly that size in what the sink sees.
fn flush_graph(
    record: &Arc<Mutex<Record>>,
    pushed: &Arc<AtomicUsize>,
    stop: &Arc<AtomicBool>,
) -> Graph<GraphNode> {
    graph_with_source(
        record,
        pushed,
        Some(GraphNode::element(Holdback::default())),
        StreamShape::new(vec![i420()]).until_stopped(stop),
    )
}

/// What both runners must show after lifting a [`Holdback`] out mid-stream.
/// `released` is the property read off the element the mutator handed back.
fn assert_flushed_before_bypass(record: &Record, emitted: u64, released: Option<PropValue>) {
    assert_no_gap(record, emitted);
    let at = assert_single_transition(&record.marks(), 1, 0);
    assert!(
        at >= HOLD_DEPTH,
        "the element must have been holding a full queue when it was lifted out, \
         only {at} frames crossed it"
    );
    assert_eq!(
        released,
        Some(PropValue::Uint(HOLD_DEPTH as u64)),
        "the element released exactly the frames it was holding when its input closed"
    );
}

/// A buffering element's held frames reach the consumer when it is lifted out,
/// ahead of the first frame that bypasses it. Without the flush they leave with
/// the element and the sink's sequence jumps.
#[test]
fn a_removed_element_is_flushed_before_the_stream_bypasses_it() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let graph = flush_graph(&record, &pushed, &stop);

    let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
    let observed = Arc::clone(&record);
    let halt = Arc::clone(&stop);
    let driver = async move {
        until(|| frames_seen(&observed) >= 3).await;
        let removed = within_deadline("the remove of a buffering element", mutator.remove("mid"))
            .await
            .expect("a buffering transform on a 1:1 edge can be lifted out");
        // Let the bypass show itself before ending the stream, so the transition
        // the assertions look for has a far side.
        until(|| marked_seen(&observed, 0) >= 2).await;
        halt.store(true, Ordering::SeqCst);
        removed
    };
    let (stats, removed) = block_on(Join2::new(run, driver));
    let stats = stats.expect("the run survives the flush and the removal");

    let record = record.lock().unwrap();
    assert_flushed_before_bypass(
        &record,
        stats.frames_emitted,
        removed.get_property("released"),
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
    assert_no_gap(record, emitted);
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

/// Frames the source of the drain-race graph pushes before it ends: few enough
/// that it can enqueue every one of them past an element parked on the first
/// (a link's worth, plus the one the element holds) and finish while that
/// element is still inside the call.
const FRAMES_BEFORE_END: u64 = 3;
/// The frame the element parks on: the first, so the whole of that short stream
/// is pushed and finished while the element sits inside that one call.
const PARK_AT: u64 = 0;

/// A stream whose source ends while the element below it is inside one
/// `process` call: a remove asked for in that window is refused, because the
/// producer it would have to park is already gone.
fn parking_graph(
    record: &Arc<Mutex<Record>>,
    pushed: &Arc<AtomicUsize>,
    parked: &Arc<AtomicBool>,
    release: &Arc<AtomicBool>,
) -> Graph<GraphNode> {
    graph_with_source(
        record,
        pushed,
        Some(GraphNode::element(ParkingHoldback {
            park_at: PARK_AT,
            parked: Arc::clone(parked),
            release: Arc::clone(release),
            held: None,
        })),
        StreamShape::new(vec![i420()])
            .without_eos()
            .frames(FRAMES_BEFORE_END),
    )
}

/// Ask for the remove in the window the refusal has to survive: the element is
/// inside `process` for frame 0, and the source has pushed its last frame and
/// ended behind it. The drain flag a remove raises here would outlive the
/// refusal and strip the element's own end of stream.
async fn remove_while_parked(
    mutator: GraphMutator<'_>,
    parked: &AtomicBool,
    pushed: &AtomicUsize,
    release: &AtomicBool,
) -> Option<MutationError> {
    until(|| {
        parked.load(Ordering::SeqCst) && pushed.load(Ordering::SeqCst) as u64 == FRAMES_BEFORE_END
    })
    .await;
    let refused = mutator.remove("mid").await.err();
    release.store(true, Ordering::SeqCst);
    refused
}

/// The remove is refused (its producer's arm has already ended), so the element
/// stays and the run must wind down exactly as it would have without the
/// request: the element is never flushed, and its consumer's link still closes
/// behind it. Returns what the operation answered and what the sink saw.
fn refused_remove_over_an_ended_source(threaded: bool) -> (Option<MutationError>, Vec<(u64, u8)>) {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let parked = Arc::new(AtomicBool::new(false));
    let release = Arc::new(AtomicBool::new(false));
    let seen = Arc::clone(&record);
    let refused = run_within_deadline("a run whose remove was refused by its ended source", {
        let record = Arc::clone(&record);
        move || {
            let graph = parking_graph(&record, &pushed, &parked, &release);
            let driver = |mutator| remove_while_parked(mutator, &parked, &pushed, &release);
            let (stats, refused) = match threaded {
                #[cfg(feature = "multi-thread")]
                true => {
                    let spawner = g2g_core::runtime::ThreadSpawner;
                    let (mutator, run) = g2g_core::runtime::run_graph_threaded_mutable(
                        graph,
                        &ZeroClock,
                        LINK_CAPACITY,
                        &spawner,
                    );
                    block_on(Join2::new(run, driver(mutator)))
                }
                #[cfg(not(feature = "multi-thread"))]
                true => unreachable!("the threaded case is only asked for with the feature on"),
                false => {
                    let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
                    block_on(Join2::new(run, driver(mutator)))
                }
            };
            stats.expect("a refused remove leaves the run to end on its own terms");
            refused
        }
    });
    let frames = seen.lock().unwrap().frames.clone();
    (refused, frames)
}

/// A remove the mutator refuses must leave nothing of itself behind. It raises
/// a drain flag and claims the element's output link to lift it out; refused
/// after either, the element's arm would strip its own end of stream on the way
/// out and the claimed link would hold its consumer's channel open, so the
/// consumer waits for an end that was swallowed and the run never finishes.
#[test]
fn a_refused_remove_leaves_no_drain_behind() {
    let (refused, frames) = refused_remove_over_an_ended_source(false);
    assert_eq!(
        refused,
        Some(MutationError::GraphEnded),
        "the producer's arm has ended, so there is no edge left to lift the element off"
    );
    assert_eq!(
        frames,
        vec![(0, 1), (1, 1)],
        "the element was never removed, so it forwards what it forwards and keeps holding the rest"
    );
}

/// The thread-per-arm runner performs both operations the same way: the splice
/// happens between packets on whichever OS thread the producer is running on,
/// and a spliced element gets a worker thread of its own.
#[cfg(feature = "multi-thread")]
/// Forwards every packet, the `Eos` included, to both ports: the shape of a
/// demux whose catch-all arm passes control packets through itself.
struct EosForwardingDemux;

impl g2g_core::fanout::MultiOutputElement for EosForwardingDemux {
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
        out: &'a mut dyn g2g_core::fanout::MultiOutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::Eos => {
                    out.push_to(0, PipelinePacket::Eos).await?;
                    out.push_to(1, PipelinePacket::Eos).await?;
                }
                PipelinePacket::CapsChanged(caps) => {
                    out.push_to(0, PipelinePacket::CapsChanged(caps.clone()))
                        .await?;
                    out.push_to(1, PipelinePacket::CapsChanged(caps)).await?;
                }
                other => {
                    out.push_to(0, other).await?;
                }
            }
            Ok(())
        })
    }
}

fn eos_forwarding_graph(
    left: &Arc<Mutex<Record>>,
    right: &Arc<Mutex<Record>>,
    pushed: &Arc<AtomicUsize>,
    stop: &Arc<AtomicBool>,
) -> Graph<GraphNode> {
    let mut graph: Graph<GraphNode> = Graph::new();
    let source = graph.add_source(driven_source(pushed, stop));
    let demux = graph.add_demux(GraphNode::demux(EosForwardingDemux), 2);
    let left_sink = graph.add_sink(recording_sink(left));
    let right_sink = graph.add_sink(recording_sink(right));
    graph.link(source, demux.input()).unwrap();
    graph.link(demux.out(0), left_sink).unwrap();
    graph.link(demux.out(1), right_sink).unwrap();
    graph
}

/// The runner ends every demux branch itself; an element that forwards the
/// `Eos` too must not leave a second terminal behind it on any port.
#[test]
fn a_demux_forwarding_eos_ends_each_branch_once() {
    let left = Arc::new(Mutex::new(Record::default()));
    let right = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let graph = eos_forwarding_graph(&left, &right, &pushed, &stop);

    let (_mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
    let watched = Arc::clone(&left);
    let halt = Arc::clone(&stop);
    let driver = async move {
        until(|| frames_seen(&watched) >= 3).await;
        halt.store(true, Ordering::SeqCst);
    };
    let (stats, ()) = block_on(Join2::new(run, driver));
    stats.expect("the run ends cleanly");
    assert_eq!(left.lock().unwrap().eos_count, 1, "one terminal per branch");
    assert_eq!(
        right.lock().unwrap().eos_count,
        1,
        "one terminal per branch"
    );
}

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

    /// The flush happens on the removed element's own worker thread, and the
    /// frames it releases still land ahead of the first bypassed one.
    #[test]
    fn a_removed_element_is_flushed_in_a_threaded_run() {
        let record = Arc::new(Mutex::new(Record::default()));
        let pushed = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let graph = flush_graph(&record, &pushed, &stop);

        let spawner = ThreadSpawner;
        let (mutator, run) = run_graph_threaded_mutable(graph, &ZeroClock, LINK_CAPACITY, &spawner);
        let observed = Arc::clone(&record);
        let halt = Arc::clone(&stop);
        let driver = async move {
            until(|| frames_seen(&observed) >= 3).await;
            let removed =
                within_deadline("the remove of a buffering element", mutator.remove("mid"))
                    .await
                    .expect("a buffering transform is flushed on its own thread too");
            until(|| marked_seen(&observed, 0) >= 2).await;
            halt.store(true, Ordering::SeqCst);
            removed
        };
        let (stats, removed) = block_on(Join2::new(run, driver));
        let stats = stats.expect("the threaded run survives the flush and the removal");

        let record = record.lock().unwrap();
        assert_flushed_before_bypass(
            &record,
            stats.frames_emitted,
            removed.get_property("released"),
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
        assert_no_gap(&record, stats.frames_emitted);
        let at = assert_single_transition(&record.marks(), 1, 0);
        assert_eq!(
            removed.get_property("marked"),
            Some(PropValue::Uint(at as u64)),
            "the element that came back is the one that marked those frames"
        );
    }

    /// M1133 on worker threads: the tee arm is the producer that parks, and it
    /// resumes onto the branch's new hop without its siblings stalling.
    #[test]
    fn a_transform_splices_onto_a_tee_branch_in_a_threaded_run() {
        let left = Arc::new(Mutex::new(Record::default()));
        let right = Arc::new(Mutex::new(Record::default()));
        let pushed = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let graph = tee_graph(&left, &right, &pushed, &stop);

        let spawner = ThreadSpawner;
        let (mutator, run) = run_graph_threaded_mutable(graph, &ZeroClock, LINK_CAPACITY, &spawner);
        let watched = Arc::clone(&left);
        let halt = Arc::clone(&stop);
        let driver = async move { splice_and_lift(&mutator, &watched, "left", true, &halt).await };
        let (stats, ()) = block_on(Join2::new(run, driver));
        stats.expect("the threaded run survives a splice on one of its tee branches");

        let left = left.lock().unwrap();
        let right = right.lock().unwrap();
        assert_contiguous_from_start(&left);
        assert_contiguous_from_start(&right);
        assert_eq!(mark_runs(&left), vec![0, 1, 0]);
        assert_eq!(
            mark_runs(&right),
            vec![0],
            "the sibling branch carried nothing the splice touched"
        );
    }

    /// M1146: the same refusal on worker threads, where the arm reads the drain
    /// flag on its own thread while the mutator is raising it.
    #[test]
    fn a_refused_remove_leaves_no_drain_behind_in_a_threaded_run() {
        let (refused, frames) = refused_remove_over_an_ended_source(true);
        assert_eq!(refused, Some(MutationError::GraphEnded));
        assert_eq!(frames, vec![(0, 1), (1, 1)]);
    }

    /// M1146 on worker threads: the source feeding the tee parks and resumes
    /// onto the spliced hop, and every branch carries what the splice added.
    #[test]
    fn a_transform_splices_onto_the_edge_into_a_tee_in_a_threaded_run() {
        let left = Arc::new(Mutex::new(Record::default()));
        let right = Arc::new(Mutex::new(Record::default()));
        let pushed = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let graph = tee_graph(&left, &right, &pushed, &stop);

        let spawner = ThreadSpawner;
        let (mutator, run) = run_graph_threaded_mutable(graph, &ZeroClock, LINK_CAPACITY, &spawner);
        let watched = Arc::clone(&left);
        let halt = Arc::clone(&stop);
        let driver =
            async move { splice_and_lift(&mutator, &watched, TEE_NAME, true, &halt).await };
        let (stats, ()) = block_on(Join2::new(run, driver));
        stats.expect("the threaded run survives a splice on the edge above its tee");

        let left = left.lock().unwrap();
        let right = right.lock().unwrap();
        assert_contiguous_from_start(&left);
        assert_contiguous_from_start(&right);
        assert_eq!(mark_runs(&left), vec![0, 1, 0]);
        assert_eq!(
            mark_runs(&right),
            vec![0, 1, 0],
            "a splice above the tee reaches every branch, not just the watched one"
        );
    }

    /// M1133 on worker threads: the producer feeding a muxer pad parks and
    /// resumes the same way any other does.
    #[test]
    fn a_transform_splices_onto_a_muxer_input_in_a_threaded_run() {
        let record = Arc::new(Mutex::new(Record::default()));
        let pushed = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let graph = muxer_graph(&record, &pushed, &stop);

        let spawner = ThreadSpawner;
        let (mutator, run) = run_graph_threaded_mutable(graph, &ZeroClock, LINK_CAPACITY, &spawner);
        let watched = Arc::clone(&record);
        let halt = Arc::clone(&stop);
        let driver = async move { splice_and_lift(&mutator, &watched, "src", false, &halt).await };
        let (stats, ()) = block_on(Join2::new(run, driver));
        stats.expect("the threaded run survives a splice on one of its muxer inputs");

        let record = record.lock().unwrap();
        assert_contiguous_from_start(&record);
        assert_eq!(mark_runs(&record), vec![0, 1, 0]);
    }
}

/// The uniqueness test in `edge_at` counts every graph edge on the side asked
/// for, not just the mutable ones: with one mutable edge among several, the
/// splice would land on a branch the caller never named.
mod ambiguous_addressing {
    use super::*;
    use g2g_core::fanout::{MultiOutputElement, MultiOutputSink};

    /// Alternates packets between its two ports, so both branches carry traffic.
    struct Alternating {
        next: usize,
    }

    impl MultiOutputElement for Alternating {
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
            out: &'a mut dyn MultiOutputSink,
        ) -> Self::ProcessFuture<'a> {
            Box::pin(async move {
                if matches!(packet, PipelinePacket::Eos) {
                    return Ok(());
                }
                let port = self.next % 2;
                self.next += 1;
                out.push_to(port, packet).await?;
                Ok(())
            })
        }
    }

    fn nested_graph(
        deep: &Arc<Mutex<Record>>,
        shallow: &Arc<Mutex<Record>>,
        pushed: &Arc<AtomicUsize>,
        stop: &Arc<AtomicBool>,
    ) -> Graph<GraphNode> {
        let mut graph: Graph<GraphNode> = Graph::new();
        let source = graph.add_source(driven_source(pushed, stop));
        graph.set_node_name(source, "src".into());
        let outer = graph.add_demux(GraphNode::demux(Alternating { next: 0 }), 2);
        graph.set_node_name(outer.node(), "dmx".into());
        let inner = graph.add_tee(2);
        let deep_a = graph.add_sink(recording_sink(deep));
        let deep_b = graph.add_sink(recording_sink(deep));
        let shallow_sink = graph.add_sink(recording_sink(shallow));
        graph.set_node_name(shallow_sink, "shallow".into());
        graph.link(source, outer.input()).unwrap();
        graph.link(outer.out(0), inner.input()).unwrap();
        graph.link(outer.out(1), shallow_sink).unwrap();
        graph.link(inner.out(0), deep_a).unwrap();
        graph.link(inner.out(1), deep_b).unwrap();
        graph
    }

    /// A muxer output is no mutation position, so the pad it feeds leaves this
    /// muxer with one mutable edge above it among two: the count that refuses
    /// the address is over both, or the splice would land on the pad the caller
    /// never named.
    fn stacked_muxer_graph(
        record: &Arc<Mutex<Record>>,
        pushed: &Arc<AtomicUsize>,
        stop: &Arc<AtomicBool>,
    ) -> Graph<GraphNode> {
        let mut graph: Graph<GraphNode> = Graph::new();
        let inner = graph.add_muxer(
            GraphNode::muxer(PassMux {
                inputs: 1,
                record: None,
            }),
            1,
        );
        let outer = graph.add_muxer(
            GraphNode::muxer(PassMux {
                inputs: 2,
                record: None,
            }),
            2,
        );
        graph.set_node_name(outer.node(), "outer".into());
        let source = graph.add_source(driven_source(pushed, stop));
        graph.set_node_name(source, "src".into());
        let quiet = graph.add_source(GraphNode::source(CountingSource {
            pushed: Arc::new(AtomicUsize::new(0)),
            eos: true,
            frames: 0,
            stop: None,
        }));
        let sink = graph.add_sink(recording_sink(record));
        graph.link(quiet, inner.input(0)).unwrap();
        graph.link(inner.output(), outer.input(0)).unwrap();
        graph.link(source, outer.input(1)).unwrap();
        graph.link(outer.output(), sink).unwrap();
        graph
    }

    #[test]
    fn insert_before_a_muxer_fed_by_another_is_refused() {
        let record = Arc::new(Mutex::new(Record::default()));
        let pushed = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let graph = stacked_muxer_graph(&record, &pushed, &stop);

        let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
        let watched = Arc::clone(&record);
        let halt = Arc::clone(&stop);
        let driver = async move {
            until(|| frames_seen(&watched) >= 3).await;
            let answer = within_deadline(
                "insert_before on a muxer with two inbound edges",
                mutator.insert_before("outer", Box::new(Marker::default())),
            )
            .await;
            halt.store(true, Ordering::SeqCst);
            answer
        };
        let (stats, answer) = block_on(Join2::new(run, driver));
        stats.expect("the run survives");
        assert_eq!(
            answer,
            Err(MutationError::NotMutable("outer".into())),
            "one of the two pads is fed by a muxer, which is no splice point, and that must \
             refuse the address rather than pick the other pad"
        );
        assert!(
            record.lock().unwrap().marks().iter().all(|&m| m == 0),
            "nothing was spliced in, so no frame is marked"
        );
    }

    #[test]
    fn insert_after_on_a_two_branch_demux_is_refused() {
        let deep = Arc::new(Mutex::new(Record::default()));
        let shallow = Arc::new(Mutex::new(Record::default()));
        let pushed = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let graph = nested_graph(&deep, &shallow, &pushed, &stop);

        let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LINK_CAPACITY);
        let watched = Arc::clone(&shallow);
        let halt = Arc::clone(&stop);
        let driver = async move {
            until(|| frames_seen(&watched) >= 3).await;
            let answer = within_deadline(
                "insert_after on a demux with two branches",
                mutator.insert_after("dmx", Box::new(Marker::default())),
            )
            .await;
            if answer.is_ok() {
                within_deadline(
                    "the unnamed branch showing the splice",
                    until(|| marked_seen(&watched, 1) >= 2),
                )
                .await;
            }
            halt.store(true, Ordering::SeqCst);
            answer
        };
        let (stats, answer) = block_on(Join2::new(run, driver));
        stats.expect("the run survives");
        let shallow_marks = mark_runs(&shallow.lock().unwrap());
        assert_eq!(
            answer,
            Err(MutationError::NotMutable("dmx".into())),
            "a demux has two branches below it, so insert_after must not pick one; \
             it returned {answer:?} and branch 1 shows {shallow_marks:?}"
        );
    }
}
