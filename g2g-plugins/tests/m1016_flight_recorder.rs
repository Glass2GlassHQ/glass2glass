//! M1016 flight recorder: while a run goes, every graph edge keeps a bounded
//! ring of its most recent packets; when the run ends in an error the rings are
//! dumped as `replaysrc` recordings, so a crash after an hour of live streaming
//! hands back the last moments of traffic as a repro.
//!
//! std-gated (file I/O + graph runner): `cargo test -p g2g-plugins --features std
//! --test m1016_flight_recorder`.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::graph::Graph;
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{
    run_graph, run_graph_recorded, run_graph_with_progress, FlightRecorder, GraphNodeRef,
    PipelineProgress, SourceLoop, FLIGHT_RING_PACKETS,
};
#[cfg(feature = "multi-thread")]
use g2g_core::runtime::{run_graph_threaded_recorded, GraphNode, ThreadSpawner};
use g2g_core::wire::read_records;
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, CapsConstraint, ConfigureOutcome, G2gError,
    HardwareError, OutputSink, PipelineClock, PipelinePacket,
};
use g2g_plugins::record::ReplaySrc;

/// Bytes per test frame: big enough that each frame is distinguishable, small
/// enough that the packet bound (not the byte bound) is what the window tests
/// exercise.
const FRAME_BYTES: usize = 16;

/// Link depth for the runs under test.
const LINK_CAPACITY: usize = 2;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn caps(encoding: ByteStreamEncoding) -> Caps {
    Caps::ByteStream { encoding }
}

/// The frame the source emits for `sequence`: a byte pattern derived from the
/// sequence id, so a replayed frame can be checked against what was sent.
fn frame_bytes(sequence: u64) -> Vec<u8> {
    (0..FRAME_BYTES)
        .map(|i| (sequence as usize + i) as u8)
        .collect()
}

fn system_frame(sequence: u64) -> Frame {
    Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(
            frame_bytes(sequence).into_boxed_slice(),
        )),
        FrameTiming::default(),
        sequence,
    )
}

fn failure() -> G2gError {
    G2gError::Hardware(HardwareError::Io(13))
}

/// What the source pushes: `frames` frames, optionally switching caps after
/// `caps_change_after` frames, all in system memory unless `device_memory` is set
/// (which makes the packets unserializable, the GPU-edge case).
#[derive(Clone, Copy)]
struct SourcePlan {
    frames: u64,
    caps_change_after: Option<u64>,
    device_memory: bool,
}

impl SourcePlan {
    fn frames(frames: u64) -> Self {
        Self {
            frames,
            caps_change_after: None,
            device_memory: false,
        }
    }
}

struct CountingSource {
    plan: SourcePlan,
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
        core::future::ready(Ok(caps(ByteStreamEncoding::Ogg)))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        let plan = self.plan;
        Box::pin(async move {
            for sequence in 0..plan.frames {
                let frame = if plan.device_memory {
                    // SAFETY: fd -1 is never a live DMABUF; `from_raw` only
                    // stores it (no I/O) and the Drop `close(-1)` is a harmless
                    // no-op. This exercises the recorder's refusal to record
                    // device memory, not real DMABUF handling.
                    let dmabuf = unsafe { g2g_core::memory::OwnedDmaBuf::from_raw(-1, 0, 0) };
                    Frame::new(
                        MemoryDomain::DmaBuf(dmabuf),
                        FrameTiming::default(),
                        sequence,
                    )
                } else {
                    system_frame(sequence)
                };
                out.push(PipelinePacket::DataFrame(frame)).await?;
                if plan.caps_change_after == Some(sequence + 1) {
                    out.push(PipelinePacket::CapsChanged(caps(
                        ByteStreamEncoding::Matroska,
                    )))
                    .await?;
                }
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(plan.frames)
        })
    }
}

/// Fails once it has consumed the frame at `failing_frame`, mid-stream.
struct FailingSink {
    failing_frame: u64,
}

impl AsyncElement for FailingSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        let failing_frame = self.failing_frame;
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(f) if f.sequence >= failing_frame => Err(failure()),
                _ => Ok(()),
            }
        })
    }
}

/// Sink that keeps every packet it is given, for checking a replay.
struct CollectingSink {
    packets: Arc<Mutex<Vec<PipelinePacket>>>,
}

impl AsyncElement for CollectingSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        self.packets.lock().expect("collect lock").push(packet);
        Box::pin(core::future::ready(Ok(())))
    }
}

/// An empty directory of this test's own, cleared if a previous run left one.
fn scratch_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("g2g_m1016_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// Run `plan` into a sink that fails at `failing_frame` with `recorder`
/// attached, asserting the run fails for the sink's reason (not the recorder's).
async fn run_failing(plan: SourcePlan, failing_frame: u64, recorder: &FlightRecorder) {
    let mut g: Graph<GraphNodeRef<'static>> = Graph::new();
    let src = g.add_source(GraphNodeRef::source(CountingSource { plan }));
    let sink = g.add_sink(GraphNodeRef::element(FailingSink { failing_frame }));
    g.link(src, sink).expect("linked");
    let err = run_graph_recorded(g, &ZeroClock, LINK_CAPACITY, None, None, recorder)
        .await
        .expect_err("the sink fails the run");
    assert_eq!(err, failure(), "the sink's failure, not a recording error");
}

/// As [`run_failing`], with every arm on its own OS thread: the rings are filled
/// by the worker threads rather than by one cooperative executor.
#[cfg(feature = "multi-thread")]
async fn run_failing_threaded(plan: SourcePlan, failing_frame: u64, recorder: &FlightRecorder) {
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(CountingSource { plan }));
    let sink = g.add_sink(GraphNode::element(FailingSink { failing_frame }));
    g.link(src, sink).expect("linked");
    let err = run_graph_threaded_recorded(
        g,
        &ZeroClock,
        LINK_CAPACITY,
        None,
        None,
        recorder,
        &ThreadSpawner,
    )
    .await
    .expect_err("the sink fails the run");
    assert_eq!(err, failure(), "the sink's failure, not a recording error");
}

/// The one dump file the single-edge graphs above produce, and its packets.
fn only_dump(dir: &Path, written: &[PathBuf]) -> (PathBuf, Vec<PipelinePacket>) {
    assert_eq!(written.len(), 1, "one edge, one recording: {written:?}");
    let path = written[0].clone();
    assert_eq!(path.parent(), Some(dir), "written into the dump directory");
    let records = read_records(&std::fs::read(&path).expect("read dump")).expect("dump parses");
    (path, records)
}

fn leading_caps(records: &[PipelinePacket]) -> Caps {
    match records.first() {
        Some(PipelinePacket::CapsChanged(c)) => c.clone(),
        other => panic!("a dump must lead with its caps, got {other:?}"),
    }
}

/// The `(sequence, bytes)` of every `DataFrame` in `packets`, in order.
fn data_frames(packets: &[PipelinePacket]) -> Vec<(u64, Vec<u8>)> {
    packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some((
                f.sequence,
                f.domain.as_system_slice().expect("system bytes").to_vec(),
            )),
            _ => None,
        })
        .collect()
}

/// Replay `recording` through a real `ReplaySrc` and return what came out.
async fn replay(recording: &Path) -> Vec<PipelinePacket> {
    let packets = Arc::new(Mutex::new(Vec::new()));
    let mut g: Graph<GraphNodeRef<'static>> = Graph::new();
    let src = g.add_source(GraphNodeRef::source(ReplaySrc::new(recording)));
    let sink = g.add_sink(GraphNodeRef::element(CollectingSink {
        packets: packets.clone(),
    }));
    g.link(src, sink).expect("linked");
    run_graph(g, &ZeroClock, LINK_CAPACITY)
        .await
        .expect("the replay runs");
    let mut collected = packets.lock().expect("collect lock");
    core::mem::take(&mut *collected)
}

/// Dump the single edge of a failed run and check it is a usable repro: named
/// for the hop, led by the negotiated caps, holding every frame that reached the
/// failure, and byte-identical when `ReplaySrc` plays it back.
async fn assert_dump_is_a_replayable_repro(
    recorder: &FlightRecorder,
    dir: &Path,
    failing_frame: u64,
) {
    assert_eq!(recorder.edge_count(), 1, "the graph's single edge");
    let written = recorder.dump_to_dir(dir).expect("dump written");
    let (path, records) = only_dump(dir, &written);
    assert_eq!(
        path.file_name().and_then(|n| n.to_str()),
        Some("0-CountingSource0-to-FailingSink0.g2grec"),
        "the file names the hop by its element instances"
    );
    assert_eq!(leading_caps(&records), caps(ByteStreamEncoding::Ogg));

    let recorded = data_frames(&records);
    assert!(
        recorded.len() > failing_frame as usize,
        "the frames up to and including the failing one: {:?}",
        recorded.iter().map(|(s, _)| *s).collect::<Vec<_>>()
    );
    for (sequence, bytes) in &recorded {
        assert_eq!(bytes, &frame_bytes(*sequence), "recorded frame {sequence}");
    }

    // The recording is a repro: replayed, it produces the same frames.
    let replayed = data_frames(&replay(&path).await);
    assert_eq!(replayed, recorded, "replay is byte-identical to the ring");
}

/// The headline case: a run that dies mid-stream leaves a recording of the
/// traffic that led there, and `ReplaySrc` re-emits those frames byte for byte.
#[tokio::test]
async fn a_failed_run_dumps_a_replayable_recording_per_edge() {
    let dir = scratch_dir("replayable");
    const FRAMES: u64 = 8;
    const FAILING_FRAME: u64 = 6;

    let recorder = FlightRecorder::new();
    run_failing(SourcePlan::frames(FRAMES), FAILING_FRAME, &recorder).await;
    assert_dump_is_a_replayable_repro(&recorder, &dir, FAILING_FRAME).await;
    let _ = std::fs::remove_dir_all(&dir);
}

/// The same repro from the thread-per-arm runner. Heavy pipelines are the ones
/// run under `--threads`, so they are exactly the crashes that need one; the
/// rings are shared with the worker threads that fill them.
#[cfg(feature = "multi-thread")]
#[tokio::test]
async fn a_failed_threaded_run_dumps_a_replayable_recording_per_edge() {
    let dir = scratch_dir("replayable_threaded");
    const FRAMES: u64 = 8;
    const FAILING_FRAME: u64 = 6;

    let recorder = FlightRecorder::new();
    run_failing_threaded(SourcePlan::frames(FRAMES), FAILING_FRAME, &recorder).await;
    assert_dump_is_a_replayable_repro(&recorder, &dir, FAILING_FRAME).await;
    let _ = std::fs::remove_dir_all(&dir);
}

/// The ring is bounded: feed it more than it holds and only the most recent
/// window survives, as one contiguous run of frames ending at the failure.
#[tokio::test]
async fn the_ring_keeps_only_the_most_recent_window() {
    let dir = scratch_dir("bounded");
    let frames = FLIGHT_RING_PACKETS as u64 * 2;
    let failing_frame = frames - 2;

    let recorder = FlightRecorder::new();
    run_failing(SourcePlan::frames(frames), failing_frame, &recorder).await;
    let written = recorder.dump_to_dir(&dir).expect("dump written");
    let (_, records) = only_dump(&dir, &written);

    let recorded = data_frames(&records);
    assert_eq!(
        records.len(),
        FLIGHT_RING_PACKETS + 1,
        "the leading caps plus exactly one ring's worth"
    );
    assert_eq!(recorded.len(), FLIGHT_RING_PACKETS, "all of it is frames");
    let sequences: Vec<u64> = recorded.iter().map(|(s, _)| *s).collect();
    assert!(
        sequences.windows(2).all(|w| w[1] == w[0] + 1),
        "one contiguous window: {sequences:?}"
    );
    assert!(
        *sequences.last().expect("non-empty") >= failing_frame,
        "the window ends at the failure, not at the start of the run: {sequences:?}"
    );
    assert!(
        sequences[0] > 0,
        "the oldest frames were dropped: {sequences:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A mid-run caps change still inside the retained window stays an ordinary
/// record there, and the dump leads with the caps the older frames were carried
/// under.
#[tokio::test]
async fn a_caps_change_inside_the_window_stays_in_place() {
    let dir = scratch_dir("caps_inside");
    const FRAMES: u64 = 8;
    const CAPS_CHANGE_AFTER: u64 = 2;

    let recorder = FlightRecorder::new();
    run_failing(
        SourcePlan {
            frames: FRAMES,
            caps_change_after: Some(CAPS_CHANGE_AFTER),
            device_memory: false,
        },
        FRAMES - 2,
        &recorder,
    )
    .await;
    let written = recorder.dump_to_dir(&dir).expect("dump written");
    let (_, records) = only_dump(&dir, &written);

    assert_eq!(
        leading_caps(&records),
        caps(ByteStreamEncoding::Ogg),
        "the negotiated caps, which the first retained frames used"
    );
    let inside: Vec<&PipelinePacket> = records[1..]
        .iter()
        .filter(|p| matches!(p, PipelinePacket::CapsChanged(_)))
        .collect();
    assert_eq!(inside.len(), 1, "the change is a record in the window");
    match inside[0] {
        PipelinePacket::CapsChanged(c) => {
            assert_eq!(*c, caps(ByteStreamEncoding::Matroska))
        }
        other => panic!("expected caps, got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The same change, but so long ago that it has scrolled out of the ring: the
/// dump must lead with the *post-change* caps, or the replay types the retained
/// frames wrong.
#[tokio::test]
async fn a_caps_change_that_scrolled_out_leads_the_dump() {
    let dir = scratch_dir("caps_scrolled");
    let frames = FLIGHT_RING_PACKETS as u64 * 2;

    let recorder = FlightRecorder::new();
    run_failing(
        SourcePlan {
            frames,
            caps_change_after: Some(1),
            device_memory: false,
        },
        frames - 2,
        &recorder,
    )
    .await;
    let written = recorder.dump_to_dir(&dir).expect("dump written");
    let (path, records) = only_dump(&dir, &written);

    assert_eq!(
        leading_caps(&records),
        caps(ByteStreamEncoding::Matroska),
        "the change that scrolled out of the window leads the dump"
    );
    assert!(
        !records[1..]
            .iter()
            .any(|p| matches!(p, PipelinePacket::CapsChanged(_))),
        "it is not also inside the window"
    );
    // Still a valid recording: `ReplaySrc` types the stream from that record.
    assert_eq!(
        data_frames(&replay(&path).await).len(),
        FLIGHT_RING_PACKETS,
        "the retained window replays under the leading caps"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// An edge carrying device memory cannot be serialized, so it is skipped at dump
/// time. The run still fails for its own reason and the dump still succeeds.
#[tokio::test]
async fn a_device_memory_edge_is_skipped_not_fatal() {
    let dir = scratch_dir("device");
    let recorder = FlightRecorder::new();
    run_failing(
        SourcePlan {
            frames: 8,
            caps_change_after: None,
            device_memory: true,
        },
        6,
        &recorder,
    )
    .await;
    assert_eq!(
        recorder.edge_count(),
        1,
        "the edge was still being recorded"
    );
    let written = recorder.dump_to_dir(&dir).expect("the dump does not fail");
    assert!(
        written.is_empty(),
        "nothing replayable to write: {written:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Recorder off: the run takes the plain entry point, so no ring is attached to
/// any edge and there is nothing to dump.
#[tokio::test]
async fn an_unrecorded_run_attaches_no_rings() {
    let dir = scratch_dir("off");
    let recorder = FlightRecorder::new();

    let mut g: Graph<GraphNodeRef<'static>> = Graph::new();
    let src = g.add_source(GraphNodeRef::source(CountingSource {
        plan: SourcePlan::frames(8),
    }));
    let sink = g.add_sink(GraphNodeRef::element(FailingSink { failing_frame: 6 }));
    g.link(src, sink).expect("linked");
    let progress = PipelineProgress::new();
    let err = run_graph_with_progress(g, &ZeroClock, LINK_CAPACITY, &progress, None)
        .await
        .expect_err("the sink fails the run");
    assert_eq!(err, failure());

    assert_eq!(recorder.edge_count(), 0, "no edge was instrumented");
    assert!(
        recorder
            .dump_to_dir(&dir)
            .expect("dump is a no-op")
            .is_empty(),
        "nothing was recorded"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
