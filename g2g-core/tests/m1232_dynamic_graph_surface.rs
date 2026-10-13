//! M1232-M1236: the client surface over a running graph. `RunBuilder` composes
//! a mutator with the other run options; a handle's timeout bounds the wait for
//! a packet boundary and withdraws what it asked for; the bus reports every
//! operation; a chain splices as one, with converters where the chain below
//! would refuse the output; a tee takes and drops whole branches; a muxer takes
//! an input on a free pad.
//!
//! Every frame carries the marks of the transforms it crossed, so what a sink
//! recorded says which elements were in the stream and when.
#![cfg(all(feature = "std", feature = "runtime", feature = "multi-thread"))]

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use g2g_core::caps::CapsSet;
use g2g_core::element::DynAsyncElement;
use g2g_core::format_element::CapsConstraint;
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::pad_template::PadTemplate;
use g2g_core::property::{PropError, PropKind, PropValue, PropertySpec};
use g2g_core::runtime::{
    block_on, select2, Either, ElementFactory, GraphNode, Join2, MutationError, MutationKind,
    Registry, RunBuilder, SourceLoop, ThreadSpawner,
};
use g2g_core::{
    AsyncClock, AsyncElement, Bus, BusMessage, Caps, ConfigureOutcome, Dim, DynAsyncClock, Frame,
    FrameTiming, G2gError, Graph, MultiInputElement, OutputSink, PipelineClock, PipelinePacket,
    Rate, RawVideoFormat,
};

/// Shallow, so a stream keeps its producer inside a blocking send: the state a
/// mutation has to interrupt.
const LINK_CAPACITY: usize = 2;
/// Frames a bounded stream carries.
const FRAME_COUNT: u64 = 8;
/// The mark a source added mid-run stamps its frames with, so they can be told
/// from the frames the graph's own source emits (which carry 0).
const LATE_SOURCE_MARK: u8 = 7;
/// Virtual nanoseconds a timed handle waits.
const TIMEOUT: Duration = Duration::from_millis(500);
/// The name the runner gives the first unnamed broadcast tee.
const TEE_NAME: &str = "tee0";
/// A mutation must land while the stream flows; this turns a wait on the end of
/// the stream into a failure rather than a hang.
const OP_DEADLINE: Duration = Duration::from_secs(5);

struct ZeroClock;

impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Virtual-time clock: a sleep advances the clock to its deadline and resolves
/// after one yield, so a timeout is reached as soon as nothing else is ready.
#[derive(Debug, Default)]
struct VirtualClock {
    now_ns: AtomicU64,
}

impl PipelineClock for VirtualClock {
    fn now_ns(&self) -> u64 {
        self.now_ns.load(Ordering::Acquire)
    }

    fn as_ticker(&self) -> Option<&dyn DynAsyncClock> {
        Some(self)
    }
}

impl AsyncClock for VirtualClock {
    type SleepFuture<'a>
        = VirtualSleep<'a>
    where
        Self: 'a;

    fn sleep_until_ns(&self, deadline_ns: u64) -> VirtualSleep<'_> {
        VirtualSleep {
            clock: self,
            deadline_ns,
            yielded: false,
        }
    }
}

struct VirtualSleep<'a> {
    clock: &'a VirtualClock,
    deadline_ns: u64,
    yielded: bool,
}

impl Future for VirtualSleep<'_> {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if !self.yielded {
            self.yielded = true;
            cx.waker().wake_by_ref();
            return Poll::Pending;
        }
        let deadline = self.deadline_ns;
        self.clock.now_ns.fetch_max(deadline, Ordering::AcqRel);
        Poll::Ready(())
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

fn frame(sequence: u64, mark: u8) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(Box::new([mark; 4]))),
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

/// Emits frames stamped `mark` until it reaches `frames` or the driver raises
/// `stop`, then `Eos`. With `stall` set it stops pushing after `stall_after`
/// frames until the driver releases it, without ever reaching a packet
/// boundary: the producer a timed operation gives up on.
struct CountingSource {
    pushed: Arc<AtomicUsize>,
    frames: u64,
    mark: u8,
    stop: Option<Arc<AtomicBool>>,
    stall: Option<Stall>,
}

struct Stall {
    after: u64,
    release: Arc<AtomicBool>,
}

impl CountingSource {
    fn driven(pushed: &Arc<AtomicUsize>, stop: &Arc<AtomicBool>) -> Self {
        Self {
            pushed: Arc::clone(pushed),
            frames: u64::MAX,
            mark: 0,
            stop: Some(Arc::clone(stop)),
            stall: None,
        }
    }

    fn bounded(frames: u64, mark: u8) -> Self {
        Self {
            pushed: Arc::new(AtomicUsize::new(0)),
            frames,
            mark,
            stop: None,
            stall: None,
        }
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
                if let Some(stall) = &self.stall {
                    if sequence == stall.after {
                        until(|| stall.release.load(Ordering::SeqCst)).await;
                    }
                }
                out.push(frame(sequence, self.mark)).await?;
                self.pushed.fetch_add(1, Ordering::SeqCst);
                emitted += 1;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(emitted)
        })
    }
}

/// Adds one to every frame's mark, so the sink can count the transforms a
/// frame crossed, and counts what it marked on a property.
#[derive(Default)]
struct Marker {
    marked: u64,
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
            if let MemoryDomain::System(bytes) = &mut data.domain {
                bytes.as_mut_slice()[0] += 1;
            }
            self.marked += 1;
            out.push(PipelinePacket::DataFrame(data)).await?;
            Ok(())
        })
    }
}

/// Marks like [`Marker`] and numbers its output from 0, as most transforms do.
#[derive(Default)]
struct Renumber {
    emitted: u64,
}

impl AsyncElement for Renumber {
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
            let PipelinePacket::DataFrame(mut data) = packet else {
                if !matches!(packet, PipelinePacket::Eos) {
                    out.push(packet).await?;
                }
                return Ok(());
            };
            if let MemoryDomain::System(bytes) = &mut data.domain {
                bytes.as_mut_slice()[0] += 1;
            }
            data.sequence = self.emitted;
            self.emitted += 1;
            out.push(PipelinePacket::DataFrame(data)).await?;
            Ok(())
        })
    }
}

/// Marks like [`Marker`] and changes the raw format from `from` to `to`.
struct Reformat {
    from: RawVideoFormat,
    to: RawVideoFormat,
}

impl AsyncElement for Reformat {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        match upstream {
            Caps::RawVideo { format, .. } if *format == self.from => Ok(video_caps(self.to)),
            _ => Err(G2gError::CapsMismatch),
        }
    }
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Mapping(vec![(
            CapsSet::one(video_caps(self.from)),
            CapsSet::one(video_caps(self.to)),
        )])
    }
    fn configure_pipeline(&mut self, caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match caps {
            Caps::RawVideo { format, .. } if *format == self.from => Ok(ConfigureOutcome::Accepted),
            _ => Err(G2gError::CapsMismatch),
        }
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
            out.push(PipelinePacket::DataFrame(data)).await?;
            Ok(())
        })
    }
}

/// A registry whose one factory converts NV12 to I420.
fn nv12_converter_registry() -> Registry {
    let mut registry = Registry::new();
    registry.register(ElementFactory::new(
        "nv12toi420",
        vec![
            PadTemplate::sink(CapsSet::one(nv12())),
            PadTemplate::source(CapsSet::one(i420())),
        ],
        |_| {
            Box::new(Reformat {
                from: RawVideoFormat::Nv12,
                to: RawVideoFormat::I420,
            })
        },
    ));
    registry
}

static STEPPER_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "step",
    PropKind::Uint,
    "how much this element adds to each frame's mark",
)];

/// Adds `step` to every frame's mark. While `park` is raised it stays inside
/// `process` for the next frame until `release`, so a property operation finds
/// no packet boundary to land at.
struct Stepper {
    step: u64,
    park: Arc<AtomicBool>,
    parked: Arc<AtomicBool>,
    release: Arc<AtomicBool>,
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
                self.step = value.as_uint().ok_or(PropError::Type)?;
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
                if !matches!(packet, PipelinePacket::Eos) {
                    out.push(packet).await?;
                }
                return Ok(());
            };
            if self.park.load(Ordering::SeqCst) {
                self.parked.store(true, Ordering::SeqCst);
                until(|| self.release.load(Ordering::SeqCst)).await;
            }
            if let MemoryDomain::System(bytes) = &mut data.domain {
                bytes.as_mut_slice()[0] += self.step as u8;
            }
            out.push(PipelinePacket::DataFrame(data)).await?;
            Ok(())
        })
    }
}

/// Forwards every frame from any pad to the merged output unchanged. Declares
/// `declared` pads, of which the graph links only some, so a free one is left
/// for a runtime input; `veto` refuses every runtime input.
struct PassMux {
    declared: usize,
    veto: bool,
}

impl MultiInputElement for PassMux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        self.declared
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
    fn accepts_runtime_input(&self, _pad: usize, _caps: &Caps) -> bool {
        !self.veto
    }
    fn process<'a>(
        &'a mut self,
        _input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(data) = packet {
                out.push(PipelinePacket::DataFrame(data)).await?;
            }
            Ok(())
        })
    }
}

/// What a sink saw, in order: each frame's sequence and mark, the frame
/// position of each `CapsChanged`, and how many ends of stream.
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

/// Records every packet and accepts the formats in `accepts`, which is what
/// the runner's downstream-feasibility snapshot is built from.
struct RecordingSink {
    record: Arc<Mutex<Record>>,
    accepts: Vec<Caps>,
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
            PipelinePacket::DataFrame(f) => record.frames.push((f.sequence, mark_of(&f))),
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
        Yield(false).await;
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

async fn within_deadline<F: Future>(what: &str, op: F) -> F::Output {
    match select2(op, Deadline(Instant::now() + OP_DEADLINE)).await {
        Either::Left(value) => value,
        Either::Right(()) => panic!("{what} did not complete within {OP_DEADLINE:?}"),
    }
}

fn frames_seen(record: &Arc<Mutex<Record>>) -> usize {
    record.lock().unwrap().frames.len()
}

fn frames_marked(record: &Arc<Mutex<Record>>, mark: u8) -> usize {
    record
        .lock()
        .unwrap()
        .frames
        .iter()
        .filter(|&&(_, m)| m == mark)
        .count()
}

fn recording_sink(record: &Arc<Mutex<Record>>) -> GraphNode {
    GraphNode::element(RecordingSink {
        record: Arc::clone(record),
        accepts: vec![i420()],
    })
}

/// A recording sink as the last element of a branch.
fn sink_element(record: &Arc<Mutex<Record>>) -> Box<dyn DynAsyncElement> {
    Box::new(RecordingSink {
        record: Arc::clone(record),
        accepts: vec![i420()],
    })
}

fn marker() -> Box<dyn DynAsyncElement> {
    Box::new(Marker::default())
}

/// `src -> sink`, the source running until the driver stops it.
fn source_sink_graph(
    record: &Arc<Mutex<Record>>,
    pushed: &Arc<AtomicUsize>,
    stop: &Arc<AtomicBool>,
) -> Graph<GraphNode> {
    source_sink_graph_with(CountingSource::driven(pushed, stop), record)
}

fn source_sink_graph_with(source: CountingSource, record: &Arc<Mutex<Record>>) -> Graph<GraphNode> {
    let mut graph: Graph<GraphNode> = Graph::new();
    let src = graph.add_source(GraphNode::source(source));
    graph.set_node_name(src, "src".into());
    let sink = graph.add_sink(recording_sink(record));
    graph.set_node_name(sink, "sink".into());
    graph.link(src, sink).unwrap();
    graph
}

/// `src -> tee -> {left, right}`.
fn tee_graph(
    left: &Arc<Mutex<Record>>,
    right: &Arc<Mutex<Record>>,
    pushed: &Arc<AtomicUsize>,
    stop: &Arc<AtomicBool>,
) -> Graph<GraphNode> {
    let mut graph: Graph<GraphNode> = Graph::new();
    let src = graph.add_source(GraphNode::source(CountingSource::driven(pushed, stop)));
    graph.set_node_name(src, "src".into());
    let tee = graph.add_tee(2);
    let left_sink = graph.add_sink(recording_sink(left));
    graph.set_node_name(left_sink, "left".into());
    let right_sink = graph.add_sink(recording_sink(right));
    graph.set_node_name(right_sink, "right".into());
    graph.link(src, tee.input()).unwrap();
    graph.link(tee.out(0), left_sink).unwrap();
    graph.link(tee.out(1), right_sink).unwrap();
    graph
}

/// `{src, quiet} -> mux -> sink`, the muxer declaring one more pad than the
/// graph links.
fn muxer_graph(
    record: &Arc<Mutex<Record>>,
    pushed: &Arc<AtomicUsize>,
    stop: &Arc<AtomicBool>,
    declared: usize,
    veto: bool,
) -> Graph<GraphNode> {
    let mut graph: Graph<GraphNode> = Graph::new();
    let mux = graph.add_muxer(GraphNode::muxer(PassMux { declared, veto }), 2);
    graph.set_node_name(mux.node(), "mux".into());
    let src = graph.add_source(GraphNode::source(CountingSource::driven(pushed, stop)));
    graph.set_node_name(src, "src".into());
    // Ends at once: the fan-in arm holds the merged `Eos` until every pad has
    // one, so a pad with nothing to say costs the run nothing.
    let quiet = graph.add_source(GraphNode::source(CountingSource::bounded(0, 0)));
    graph.set_node_name(quiet, "quiet".into());
    let sink = graph.add_sink(recording_sink(record));
    graph.set_node_name(sink, "sink".into());
    graph.link(src, mux.input(0)).unwrap();
    graph.link(quiet, mux.input(1)).unwrap();
    graph.link(mux.output(), sink).unwrap();
    graph
}

/// The mutation reports the bus carried, in order.
fn mutations_on(bus: &Bus) -> Vec<(MutationKind, String, Result<Vec<String>, MutationError>)> {
    let mut reports = Vec::new();
    while let Some(message) = bus.try_recv() {
        if let BusMessage::GraphMutated {
            operation,
            node,
            outcome,
        } = message
        {
            reports.push((operation, node, outcome));
        }
    }
    reports
}

/// M1232: a mutator and a bus on one run, which only an observer could pair
/// before; M1233: the bus hears every operation, landed or refused.
#[test]
fn the_builder_pairs_a_mutator_with_a_bus_and_the_bus_reports_each_operation() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let graph = source_sink_graph(&record, &pushed, &stop);
    let (bus, handle) = Bus::new(64);

    let (mutator, run) = RunBuilder::new(graph, &ZeroClock, LINK_CAPACITY)
        .with_bus(&handle)
        .run_mutable();
    let observed = Arc::clone(&record);
    let halt = Arc::clone(&stop);
    let driver = async move {
        until(|| frames_seen(&observed) >= 2).await;
        let name = within_deadline("the insert", mutator.insert_after("src", marker()))
            .await
            .expect("a marker splices below the source");
        until(|| frames_marked(&observed, 1) >= 2).await;
        let refused = mutator
            .remove("nobody")
            .await
            .map(|_| ())
            .expect_err("an unknown node is refused");
        within_deadline("the remove", mutator.remove(&name))
            .await
            .expect("the marker lifts back off");
        halt.store(true, Ordering::SeqCst);
        (name, refused)
    };
    let (stats, (name, refused)) = block_on(Join2::new(run, driver));
    stats.expect("the run survives the splice");

    assert_eq!(refused, MutationError::UnknownNode("nobody".into()));
    let reports = mutations_on(&bus);
    assert_eq!(
        reports,
        vec![
            (MutationKind::Insert, "src".into(), Ok(vec![name.clone()])),
            (
                MutationKind::Remove,
                "nobody".into(),
                Err(MutationError::UnknownNode("nobody".into()))
            ),
            (MutationKind::Remove, name, Ok(vec![])),
        ],
        "the bus reports each operation with what it created or why it was refused"
    );
}

/// M1233: a producer that never reaches a packet boundary does not hold a timed
/// handle; the park request is withdrawn, so the stream runs on untouched once
/// the producer does move, and an untimed operation then lands.
#[test]
fn a_timed_operation_gives_up_on_a_stalled_producer_and_leaves_it_running() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let release = Arc::new(AtomicBool::new(false));
    let mut source = CountingSource::driven(&pushed, &stop);
    source.stall = Some(Stall {
        after: 3,
        release: Arc::clone(&release),
    });
    let graph = source_sink_graph_with(source, &record);
    let clock = VirtualClock::default();

    let (mutator, run) = RunBuilder::new(graph, &clock, LINK_CAPACITY).run_mutable();
    let timed = mutator.clone().with_timeout(TIMEOUT);
    let observed = Arc::clone(&record);
    let halt = Arc::clone(&stop);
    let driver = async move {
        until(|| frames_seen(&observed) >= 3).await;
        let refused = within_deadline("the timed insert", timed.insert_after("src", marker()))
            .await
            .expect_err("a stalled producer never reaches the boundary the insert waits for");
        release.store(true, Ordering::SeqCst);
        until(|| frames_seen(&observed) >= 6).await;
        let unmarked_after_timeout = frames_marked(&observed, 0);
        let name = within_deadline("the untimed insert", mutator.insert_after("src", marker()))
            .await
            .expect("once the producer moves again, an insert lands");
        until(|| frames_marked(&observed, 1) >= 2).await;
        halt.store(true, Ordering::SeqCst);
        (refused, unmarked_after_timeout, name)
    };
    let (stats, (refused, unmarked_after_timeout, _)) = block_on(Join2::new(run, driver));
    stats.expect("the run survives a timed-out operation");

    assert_eq!(refused, MutationError::TimedOut);
    assert!(
        unmarked_after_timeout >= 6,
        "the withdrawn park left the producer running on its own link: {unmarked_after_timeout} unmarked frames"
    );
    let record = record.lock().unwrap();
    let sequences: Vec<u64> = record.frames.iter().map(|&(s, _)| s).collect();
    let expected: Vec<u64> = (0..sequences.len() as u64).collect();
    assert_eq!(sequences, expected, "no frame was lost across the timeout");
}

/// A spliced transform that numbers its own output from 0 does not send the
/// sequence the sink sees backwards, going in or coming back out.
#[test]
fn a_renumbering_splice_keeps_the_sink_sequence_rising() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let graph = source_sink_graph(&record, &pushed, &stop);

    let (mutator, run) = RunBuilder::new(graph, &ZeroClock, LINK_CAPACITY).run_mutable();
    let observed = Arc::clone(&record);
    let halt = Arc::clone(&stop);
    let driver = async move {
        until(|| frames_seen(&observed) >= 3).await;
        let name = within_deadline(
            "the insert",
            mutator.insert_after("src", Box::new(Renumber::default())),
        )
        .await
        .expect("the renumbering transform goes in");
        until(|| frames_marked(&observed, 1) >= 3).await;
        within_deadline("the remove", mutator.remove(&name))
            .await
            .expect("the renumbering transform comes back out");
        let seen = frames_seen(&observed);
        until(|| frames_seen(&observed) >= seen + 3).await;
        halt.store(true, Ordering::SeqCst);
    };
    let (stats, ()) = block_on(Join2::new(run, driver));
    stats.expect("the run survives both splices");

    let record = record.lock().unwrap();
    let sequences: Vec<u64> = record.frames.iter().map(|&(s, _)| s).collect();
    assert!(
        sequences.windows(2).all(|pair| pair[0] < pair[1]),
        "the sink's sequence only rises: {sequences:?}"
    );
}

/// M1233: a timed property set that finds no packet boundary is withdrawn, so
/// it never lands late; the next untimed one does.
#[test]
fn a_timed_property_set_is_withdrawn_rather_than_applied_late() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let park = Arc::new(AtomicBool::new(false));
    let parked = Arc::new(AtomicBool::new(false));
    let release = Arc::new(AtomicBool::new(false));
    let mut graph: Graph<GraphNode> = Graph::new();
    let src = graph.add_source(GraphNode::source(CountingSource::driven(&pushed, &stop)));
    graph.set_node_name(src, "src".into());
    let stepper = graph.add_transform(GraphNode::element(Stepper {
        step: 0,
        park: Arc::clone(&park),
        parked: Arc::clone(&parked),
        release: Arc::clone(&release),
    }));
    graph.set_node_name(stepper, "step".into());
    let sink = graph.add_sink(recording_sink(&record));
    graph.set_node_name(sink, "sink".into());
    graph.link(src, stepper).unwrap();
    graph.link(stepper, sink).unwrap();
    let clock = VirtualClock::default();

    let (mutator, run) = RunBuilder::new(graph, &clock, LINK_CAPACITY).run_mutable();
    let timed = mutator.clone().with_timeout(TIMEOUT);
    let observed = Arc::clone(&record);
    let halt = Arc::clone(&stop);
    let driver = async move {
        until(|| frames_seen(&observed) >= 2).await;
        park.store(true, Ordering::SeqCst);
        until(|| parked.load(Ordering::SeqCst)).await;
        let refused = within_deadline(
            "the timed set",
            timed.set_property("step", "step", PropValue::Uint(3)),
        )
        .await
        .expect_err("an arm inside process reaches no boundary");
        park.store(false, Ordering::SeqCst);
        release.store(true, Ordering::SeqCst);
        until(|| frames_seen(&observed) >= 6).await;
        let after_timeout = within_deadline("the read", mutator.get_property("step", "step"))
            .await
            .expect("a read lands between packets");
        within_deadline(
            "the untimed set",
            mutator.set_property("step", "step", PropValue::Uint(3)),
        )
        .await
        .expect("the set lands once the arm moves again");
        until(|| frames_marked(&observed, 3) >= 2).await;
        halt.store(true, Ordering::SeqCst);
        (refused, after_timeout)
    };
    let (stats, (refused, after_timeout)) = block_on(Join2::new(run, driver));
    stats.expect("the run survives the withdrawn set");

    assert_eq!(refused, MutationError::TimedOut);
    assert_eq!(
        after_timeout,
        Some(PropValue::Uint(0)),
        "the timed-out set was withdrawn, not applied once the arm reached a boundary"
    );
}

/// M1233: a timeout needs a clock that sleeps; a run without one says so
/// instead of waiting unbounded under a handle that promised not to.
#[test]
fn a_timed_handle_on_a_run_without_a_timer_is_refused() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let graph = source_sink_graph(&record, &pushed, &stop);

    let (mutator, run) = RunBuilder::new(graph, &ZeroClock, LINK_CAPACITY).run_mutable();
    let timed = mutator.with_timeout(TIMEOUT);
    let observed = Arc::clone(&record);
    let halt = Arc::clone(&stop);
    let driver = async move {
        until(|| frames_seen(&observed) >= 2).await;
        let refused = timed
            .insert_after("src", marker())
            .await
            .expect_err("no timer");
        halt.store(true, Ordering::SeqCst);
        refused
    };
    let (stats, refused) = block_on(Join2::new(run, driver));
    stats.expect("the run is untouched");
    assert_eq!(refused, MutationError::NoTimer);
}

/// M1234: a chain goes in as one operation, in stream order, and each element
/// comes back out on its own.
#[test]
fn a_chain_splices_as_one_operation() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let graph = source_sink_graph(&record, &pushed, &stop);

    let (mutator, run) = RunBuilder::new(graph, &ZeroClock, LINK_CAPACITY).run_mutable();
    let observed = Arc::clone(&record);
    let halt = Arc::clone(&stop);
    let driver = async move {
        until(|| frames_seen(&observed) >= 2).await;
        let names = within_deadline(
            "the chain insert",
            mutator.insert_chain_after("src", vec![marker(), marker()]),
        )
        .await
        .expect("two markers splice below the source as one");
        assert_eq!(names.len(), 2);
        assert_ne!(names[0], names[1], "each element gets its own name");
        until(|| frames_marked(&observed, 2) >= 2).await;
        let first = within_deadline("the first remove", mutator.remove(&names[0]))
            .await
            .expect("the upstream element lifts out");
        until(|| frames_marked(&observed, 1) >= 2).await;
        within_deadline("the second remove", mutator.remove(&names[1]))
            .await
            .expect("the downstream element lifts out");
        until(|| {
            let record = observed.lock().unwrap();
            record.frames.last().is_some_and(|&(_, m)| m == 0) && record.frames.len() > 6
        })
        .await;
        halt.store(true, Ordering::SeqCst);
        first
    };
    let (stats, first) = block_on(Join2::new(run, driver));
    stats.expect("the run survives the chain");

    assert!(
        first
            .get_property("marked")
            .is_some_and(|m| m != PropValue::Uint(0)),
        "the element handed back is the one that ran"
    );
    let record = record.lock().unwrap();
    let marks = record.marks();
    let runs: Vec<u8> = marks.iter().fold(Vec::new(), |mut runs, &m| {
        if runs.last() != Some(&m) {
            runs.push(m);
        }
        runs
    });
    assert_eq!(
        runs,
        vec![0, 2, 1, 0],
        "both marks arrive together, then leave one at a time"
    );
    let sequences: Vec<u64> = record.frames.iter().map(|&(s, _)| s).collect();
    let expected: Vec<u64> = (0..sequences.len() as u64).collect();
    assert_eq!(sequences, expected, "no frame was lost or reordered");
}

/// M1234: an element whose output the sink refuses is refused on a plain
/// handle, and goes in behind a converter on a handle that has one.
#[test]
fn a_caps_changing_insert_takes_its_converter_from_the_handle() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let graph = source_sink_graph(&record, &pushed, &stop);

    let (mutator, run) = RunBuilder::new(graph, &ZeroClock, LINK_CAPACITY).run_mutable();
    let converting = mutator
        .clone()
        .with_converters(Arc::new(nv12_converter_registry()).converters());
    let recolor = || {
        Box::new(Reformat {
            from: RawVideoFormat::I420,
            to: RawVideoFormat::Nv12,
        }) as Box<dyn DynAsyncElement>
    };
    let observed = Arc::clone(&record);
    let halt = Arc::clone(&stop);
    let driver = async move {
        until(|| frames_seen(&observed) >= 2).await;
        let refused = mutator
            .insert_after("src", recolor())
            .await
            .expect_err("the sink takes no NV12 and the plain handle has no converter");
        let names = within_deadline(
            "the converted insert",
            converting.insert_chain_after("src", vec![recolor()]),
        )
        .await
        .expect("the converter takes the output to a shape the sink accepts");
        until(|| frames_marked(&observed, 2) >= 2).await;
        halt.store(true, Ordering::SeqCst);
        (refused, names)
    };
    let (stats, (refused, names)) = block_on(Join2::new(run, driver));
    stats.expect("the run survives the converted splice");

    assert_eq!(refused, MutationError::DownstreamRefused);
    assert_eq!(
        names.len(),
        2,
        "the element and its converter went in together"
    );
    let record = record.lock().unwrap();
    assert!(
        record
            .caps
            .iter()
            .all(|(_, caps)| *caps == nv12() || *caps == i420()),
        "the sink only ever heard shapes it takes"
    );
    assert!(
        !record.caps.iter().any(|(_, caps)| *caps == nv12()),
        "the converter kept NV12 off the sink's edge"
    );
}

/// M1235: a branch of a transform and a sink joins a running tee, sees every
/// frame from then on, and comes back out finalized; the sibling branches
/// never notice. The tee's own branches drop the same way.
#[test]
fn a_branch_is_added_under_a_tee_and_dropped_again() {
    let left = Arc::new(Mutex::new(Record::default()));
    let right = Arc::new(Mutex::new(Record::default()));
    let third = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let graph = tee_graph(&left, &right, &pushed, &stop);

    let (mutator, run) = RunBuilder::new(graph, &ZeroClock, LINK_CAPACITY).run_mutable();
    let watched = Arc::clone(&left);
    let remaining = Arc::clone(&right);
    let joined = Arc::clone(&third);
    let halt = Arc::clone(&stop);
    let driver = async move {
        until(|| frames_seen(&watched) >= 3).await;
        let names = within_deadline(
            "the branch add",
            mutator.add_branch(TEE_NAME, vec![marker(), sink_element(&joined)]),
        )
        .await
        .expect("a tee takes a branch while it runs");
        assert_eq!(names.len(), 2, "the transform and the sink are both named");
        until(|| frames_seen(&joined) >= 3).await;
        let back = within_deadline("the branch drop", mutator.remove_branch(&names[1]))
            .await
            .expect("the branch comes back out through its sink");
        let own = within_deadline("the own branch drop", mutator.remove_branch("left"))
            .await
            .expect("a branch the tee was built with drops the same way");
        let seen = frames_seen(&remaining);
        until(|| frames_seen(&remaining) > seen + 2).await;
        halt.store(true, Ordering::SeqCst);
        (back, own)
    };
    let (stats, (back, own)) = block_on(Join2::new(run, driver));
    stats.expect("the run survives a branch coming and going");

    assert_eq!(
        back.len(),
        2,
        "the transform and the sink come back, in stream order"
    );
    assert!(
        back[0]
            .get_property("marked")
            .is_some_and(|m| m != PropValue::Uint(0)),
        "the first element back is the transform that ran"
    );
    assert_eq!(own.len(), 1, "a built-in branch is its sink alone");
    let third = third.lock().unwrap();
    assert!(third.frames.len() >= 3, "the late branch saw the stream");
    assert!(
        third.marks().iter().all(|&m| m == 1),
        "every frame on the late branch crossed its transform"
    );
    let sequences: Vec<u64> = third.frames.iter().map(|&(s, _)| s).collect();
    let from = sequences[0];
    let expected: Vec<u64> = (from..from + sequences.len() as u64).collect();
    assert_eq!(
        sequences, expected,
        "the late branch missed nothing once attached"
    );
    assert_eq!(
        third.eos_count, 1,
        "the dropped branch's sink took an end of stream"
    );
    let left = left.lock().unwrap();
    assert_eq!(
        left.eos_count, 1,
        "the dropped built-in branch finalized too"
    );
    let right = right.lock().unwrap();
    assert!(
        right.marks().iter().all(|&m| m == 0),
        "the sibling never saw the late transform"
    );
    assert_eq!(
        right.eos_count, 1,
        "the remaining branch ended with the run"
    );
}

/// M1235: the positions a branch operation refuses.
#[test]
fn branch_operations_refuse_nodes_that_are_not_under_a_tee() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let graph = source_sink_graph(&record, &pushed, &stop);

    let (mutator, run) = RunBuilder::new(graph, &ZeroClock, LINK_CAPACITY).run_mutable();
    let observed = Arc::clone(&record);
    let halt = Arc::clone(&stop);
    let driver = async move {
        until(|| frames_seen(&observed) >= 2).await;
        let not_a_tee = mutator
            .add_branch("src", vec![marker(), sink_element(&observed)])
            .await
            .expect_err("a source takes no branch");
        let not_under_a_tee = mutator
            .remove_branch("sink")
            .await
            .map(|_| ())
            .expect_err("a sink fed by a source is not a branch");
        let not_a_muxer = mutator
            .add_input("sink", Box::new(CountingSource::bounded(1, 0)))
            .await
            .expect_err("a sink takes no input");
        halt.store(true, Ordering::SeqCst);
        (not_a_tee, not_under_a_tee, not_a_muxer)
    };
    let (stats, (not_a_tee, not_under_a_tee, not_a_muxer)) = block_on(Join2::new(run, driver));
    stats.expect("the run is untouched by refused operations");

    assert_eq!(not_a_tee, MutationError::NotMutable("src".into()));
    assert_eq!(not_under_a_tee, MutationError::NotMutable("sink".into()));
    assert_eq!(not_a_muxer, MutationError::NotMutable("sink".into()));
}

/// M1236: a source joins a running muxer on its free pad and its frames reach
/// the merged output; a second one finds no pad.
#[test]
fn an_input_is_added_to_a_running_muxer() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let graph = muxer_graph(&record, &pushed, &stop, 3, false);

    let (mutator, run) = RunBuilder::new(graph, &ZeroClock, LINK_CAPACITY).run_mutable();
    let observed = Arc::clone(&record);
    let halt = Arc::clone(&stop);
    let driver = async move {
        until(|| frames_seen(&observed) >= 3).await;
        let name = within_deadline(
            "the input add",
            mutator.add_input(
                "mux",
                Box::new(CountingSource::bounded(FRAME_COUNT, LATE_SOURCE_MARK)),
            ),
        )
        .await
        .expect("the muxer has a pad to spare");
        until(|| frames_marked(&observed, LATE_SOURCE_MARK) as u64 >= FRAME_COUNT).await;
        let no_pad = mutator
            .add_input("mux", Box::new(CountingSource::bounded(1, 0)))
            .await
            .expect_err("every declared pad is taken now");
        halt.store(true, Ordering::SeqCst);
        (name, no_pad)
    };
    let (stats, (name, no_pad)) = block_on(Join2::new(run, driver));
    stats.expect("the run ends once every pad, the late one included, has ended");

    assert!(!name.is_empty());
    assert_eq!(no_pad, MutationError::NoFreePad);
    let record = record.lock().unwrap();
    let late: Vec<u64> = record
        .frames
        .iter()
        .filter(|&&(_, m)| m == LATE_SOURCE_MARK)
        .map(|&(s, _)| s)
        .collect();
    let expected: Vec<u64> = (0..FRAME_COUNT).collect();
    assert_eq!(
        late, expected,
        "every frame of the late input reached the merged output in order"
    );
}

/// M1236: the element's veto on a runtime input comes back to the caller and
/// the run goes on.
#[test]
fn a_muxer_vetoes_a_runtime_input() {
    let record = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let graph = muxer_graph(&record, &pushed, &stop, 3, true);

    let (mutator, run) = RunBuilder::new(graph, &ZeroClock, LINK_CAPACITY).run_mutable();
    let observed = Arc::clone(&record);
    let halt = Arc::clone(&stop);
    let driver = async move {
        until(|| frames_seen(&observed) >= 3).await;
        let refused = within_deadline(
            "the vetoed add",
            mutator.add_input(
                "mux",
                Box::new(CountingSource::bounded(1, LATE_SOURCE_MARK)),
            ),
        )
        .await
        .expect_err("the element vetoes it");
        until(|| frames_seen(&observed) >= 6).await;
        halt.store(true, Ordering::SeqCst);
        refused
    };
    let (stats, refused) = block_on(Join2::new(run, driver));
    stats.expect("a vetoed input leaves the run on its existing pads");

    assert_eq!(refused, MutationError::Refused(G2gError::InputRefused));
    assert_eq!(frames_marked(&record, LATE_SOURCE_MARK), 0);
}

/// M1235 on the thread-per-arm runner: the tee takes the port on its own
/// thread, and the branch's arms get threads of their own.
#[test]
fn a_branch_is_added_under_a_threaded_tee() {
    let left = Arc::new(Mutex::new(Record::default()));
    let right = Arc::new(Mutex::new(Record::default()));
    let third = Arc::new(Mutex::new(Record::default()));
    let pushed = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let graph = tee_graph(&left, &right, &pushed, &stop);

    let spawner = ThreadSpawner;
    let (mutator, run) =
        RunBuilder::new(graph, &ZeroClock, LINK_CAPACITY).run_threaded_mutable(&spawner);
    let watched = Arc::clone(&left);
    let joined = Arc::clone(&third);
    let halt = Arc::clone(&stop);
    let driver = async move {
        until(|| frames_seen(&watched) >= 3).await;
        let names = within_deadline(
            "the branch add",
            mutator.add_branch(TEE_NAME, vec![marker(), sink_element(&joined)]),
        )
        .await
        .expect("a threaded tee takes a branch while it runs");
        until(|| frames_seen(&joined) >= 3).await;
        within_deadline("the branch drop", mutator.remove_branch(&names[1]))
            .await
            .expect("the branch comes back out");
        halt.store(true, Ordering::SeqCst);
    };
    let (stats, ()) = block_on(Join2::new(run, driver));
    stats.expect("the threaded run survives a branch coming and going");

    let third = third.lock().unwrap();
    assert!(third.frames.len() >= 3);
    assert!(third.marks().iter().all(|&m| m == 1));
    assert_eq!(third.eos_count, 1);
}
