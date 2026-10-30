#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use core::task::Poll;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{
    attach_seek_controller, block_on, eos_then_wait_for_seek, GraphNode, Join2, RunBuilder,
    RunStats, SeekController, SeekFormat, SourceLoop,
};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, Frame, FrameTiming, G2gError, Graph, OutputSink,
    PipelineClock, PipelinePacket, Rate, RawVideoFormat, Seek,
};

const LINK_CAPACITY: usize = 2;
const FRAME_COUNT: u64 = 8;
const FRAME_STEP_NS: u64 = 1_000;
const SEEK_TARGET_NS: u64 = 5_000;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(16),
        height: Dim::Fixed(16),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

// plays frames at FRAME_STEP_NS apart up to FRAME_COUNT frames' worth, from the latest seek's target
#[derive(Default)]
struct SeekableSource {
    seek: Option<SeekController>,
}

impl SourceLoop for SeekableSource {
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

    fn seek_format(&self) -> Option<SeekFormat> {
        Some(SeekFormat::Time)
    }

    fn set_seek(&mut self, controller: SeekController) {
        self.seek = Some(controller);
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let seek = self.seek.clone().expect("the test attaches a controller");
            let mut sequence = 0;
            let mut position = 0;
            loop {
                if let Some(taken) = seek.take_seek(Ok) {
                    out.push(PipelinePacket::Flush).await?;
                    position = taken.start;
                }
                if position < FRAME_COUNT * FRAME_STEP_NS {
                    let frame = Frame::new(
                        MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
                        FrameTiming {
                            pts_ns: position,
                            ..FrameTiming::default()
                        },
                        sequence,
                    );
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                    sequence += 1;
                    position += FRAME_STEP_NS;
                    continue;
                }
                if !eos_then_wait_for_seek(Some(&seek), out).await? {
                    return Ok(sequence);
                }
            }
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
            out.push(packet).await?;
            Ok(())
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Seen {
    Frame(u64),
    Flush,
    Eos,
}

struct RecordingSink {
    seen: Arc<Mutex<Vec<Seen>>>,
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
        let seen = match packet {
            PipelinePacket::DataFrame(frame) => Some(Seen::Frame(frame.timing.pts_ns)),
            PipelinePacket::Flush => Some(Seen::Flush),
            PipelinePacket::Eos => Some(Seen::Eos),
            _ => None,
        };
        self.seen.lock().unwrap().extend(seen);
        core::future::ready(Ok(()))
    }
}

fn seekable_graph(seen: &Arc<Mutex<Vec<Seen>>>) -> Graph<GraphNode> {
    let mut graph: Graph<GraphNode> = Graph::new();
    let source = graph.add_source(GraphNode::source(SeekableSource::default()));
    let transform = graph.add_transform(GraphNode::element(PassThrough));
    let sink = graph.add_sink(GraphNode::element(RecordingSink { seen: seen.clone() }));
    graph.link(source, transform).unwrap();
    graph.link(transform, sink).unwrap();
    graph
}

fn frames_from(start_ns: u64) -> impl Iterator<Item = Seen> {
    (start_ns..FRAME_COUNT * FRAME_STEP_NS)
        .step_by(FRAME_STEP_NS as usize)
        .map(Seen::Frame)
}

fn played_through_then_from_the_target() -> Vec<Seen> {
    frames_from(0)
        .chain([Seen::Eos, Seen::Flush])
        .chain(frames_from(SEEK_TARGET_NS))
        .chain([Seen::Eos])
        .collect()
}

// once every sink took the end of the stream, seeks to the target, and runs `run` alongside
fn run_with_a_seek_after_the_end(
    run: impl Future<Output = Result<RunStats, G2gError>>,
    controller: &SeekController,
) {
    let run_ended = AtomicBool::new(false);
    let run = async {
        let stats = run.await;
        run_ended.store(true, Ordering::SeqCst);
        stats
    };
    let seek = async {
        core::future::poll_fn(|cx| {
            assert!(
                !run_ended.load(Ordering::SeqCst),
                "the run ended while the application kept it open"
            );
            if controller.stream_ended() {
                return Poll::Ready(());
            }
            cx.waker().wake_by_ref();
            Poll::Pending
        })
        .await;
        controller.seek(Seek::flush_to(SEEK_TARGET_NS));
        controller.let_run_end();
    };
    let (stats, ()) = block_on(Join2::new(run, seek));
    stats.expect("the run ends once nothing more is sought");
}

#[test]
fn a_parked_source_takes_a_seek_after_the_end_and_the_run_continues() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut graph = seekable_graph(&seen);
    let controller = attach_seek_controller(&mut graph).expect("the source can seek");
    controller.keep_run_open();
    let run = RunBuilder::new(graph, &ZeroClock, LINK_CAPACITY)
        .with_seek_controller(&controller)
        .run();
    run_with_a_seek_after_the_end(run, &controller);
    assert_eq!(*seen.lock().unwrap(), played_through_then_from_the_target());
}

#[cfg(feature = "multi-thread")]
#[test]
fn the_threaded_runner_takes_a_seek_after_the_end() {
    use g2g_core::runtime::ThreadSpawner;

    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut graph = seekable_graph(&seen);
    let controller = attach_seek_controller(&mut graph).expect("the source can seek");
    controller.keep_run_open();
    let run = RunBuilder::new(graph, &ZeroClock, LINK_CAPACITY)
        .with_seek_controller(&controller)
        .run_threaded(&ThreadSpawner);
    run_with_a_seek_after_the_end(run, &controller);
    assert_eq!(*seen.lock().unwrap(), played_through_then_from_the_target());
}

#[test]
fn the_run_ends_by_itself_when_nothing_more_is_sought() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut graph = seekable_graph(&seen);
    let controller = attach_seek_controller(&mut graph).expect("the source can seek");
    let run = RunBuilder::new(graph, &ZeroClock, LINK_CAPACITY)
        .with_seek_controller(&controller)
        .run();
    block_on(run).expect("the run ends at the end of the stream");
    assert!(controller.takes_seeks_after_end());
    assert!(controller.stream_ended());
    let played: Vec<Seen> = frames_from(0).chain([Seen::Eos]).collect();
    assert_eq!(*seen.lock().unwrap(), played);
}

#[test]
fn a_run_without_the_seek_controller_ends_at_the_end_of_the_stream() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut graph = seekable_graph(&seen);
    let controller = attach_seek_controller(&mut graph).expect("the source can seek");
    controller.keep_run_open();
    block_on(RunBuilder::new(graph, &ZeroClock, LINK_CAPACITY).run())
        .expect("the run ends at the end of the stream");
    assert!(!controller.takes_seeks_after_end());
    let played: Vec<Seen> = frames_from(0).chain([Seen::Eos]).collect();
    assert_eq!(*seen.lock().unwrap(), played);
}
