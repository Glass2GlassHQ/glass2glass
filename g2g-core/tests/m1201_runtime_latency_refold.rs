#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use std::sync::{Arc, Mutex};

use g2g_core::clock::{ClockCandidate, ClockPriority, ClockSync};
use g2g_core::fanout::{MultiOutputElement, MultiOutputSink, MultiOutputSource};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::query::LatencyReport;
use g2g_core::runtime::{block_on, run_graph, GraphNode, RunStats, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, Frame, FrameTiming, G2gError, Graph, OutputSink,
    PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

const FRAME_COUNT: u64 = 6;
const LINK_CAPACITY: usize = 2;
const PORT: usize = 0;
const PORT_COUNT: u8 = 1;
// any nonzero value
const HELD_UNIT_NS: u64 = 40_000_000;
const HELD_UNIT: LatencyReport = LatencyReport::buffered(HELD_UNIT_NS, Some(HELD_UNIT_NS));

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(2),
        height: Dim::Fixed(2),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn frame(sequence: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 16]))),
        FrameTiming::default(),
        sequence,
    ))
}

struct Src;

impl SourceLoop for Src {
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
            for i in 0..FRAME_COUNT {
                out.push(frame(i)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(FRAME_COUNT)
        })
    }
}

// Declares nothing until it has routed a unit, like a demux that learns the unit
// duration from the stream.
#[derive(Default)]
struct LearningDemux {
    routed: bool,
}

impl MultiOutputElement for LearningDemux {
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
            if matches!(packet, PipelinePacket::DataFrame(_)) {
                self.routed = true;
                out.push_to(PORT, packet).await?;
            }
            Ok(())
        })
    }

    fn latency(&self) -> LatencyReport {
        if self.routed {
            HELD_UNIT
        } else {
            LatencyReport::ZERO
        }
    }
}

#[derive(Debug, Default)]
struct SinkLog {
    startup_path_latency_ns: Option<u64>,
    per_frame_path_latency_ns: Vec<u64>,
}

// Offers a clock so the runner elects one and hands this sink a ClockSync.
struct SyncedSink {
    sync: Option<ClockSync>,
    log: Arc<Mutex<SinkLog>>,
}

impl AsyncElement for SyncedSink {
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

    fn provide_clock(&self) -> Option<ClockCandidate> {
        Some(ClockCandidate::new(
            ClockPriority::AudioProvider,
            Arc::new(ZeroClock),
        ))
    }

    fn set_clock_sync(&mut self, sync: ClockSync) {
        self.log.lock().unwrap().startup_path_latency_ns = Some(sync.path_latency_min_ns());
        self.sync = Some(sync);
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        if let (PipelinePacket::DataFrame(_), Some(sync)) = (&packet, &self.sync) {
            self.log
                .lock()
                .unwrap()
                .per_frame_path_latency_ns
                .push(sync.path_latency_min_ns());
        }
        Box::pin(core::future::ready(Ok(())))
    }
}

fn demux_graph(log: &Arc<Mutex<SinkLog>>) -> Graph<GraphNode> {
    let mut graph: Graph<GraphNode> = Graph::new();
    let src = graph.add_source(GraphNode::source(Src));
    let demux = graph.add_demux(GraphNode::demux(LearningDemux::default()), PORT_COUNT);
    let sink = graph.add_sink(GraphNode::element(SyncedSink {
        sync: None,
        log: log.clone(),
    }));
    graph
        .link(src, demux.input())
        .expect("link source to demux");
    graph
        .link(demux.out(PORT as u8), sink)
        .expect("link demux port");
    graph
}

fn assert_refolded(stats: &RunStats, log: &SinkLog) {
    assert_eq!(stats.frames_consumed, FRAME_COUNT);
    assert_eq!(
        log.startup_path_latency_ns,
        Some(0),
        "before any unit the demux declares nothing"
    );
    // the first frame can reach the sink before its arm re-folds
    let after_first = &log.per_frame_path_latency_ns[1..];
    assert_eq!(after_first.len() as u64, FRAME_COUNT - 1);
    assert!(
        after_first.iter().all(|&ns| ns == HELD_UNIT_NS),
        "every frame after the first is paced with the held unit: {:?}",
        log.per_frame_path_latency_ns
    );
    assert_eq!(
        stats.latency.min_ns, HELD_UNIT_NS,
        "the run reports the last fold"
    );
}

#[test]
fn a_demux_that_learns_its_latency_retargets_the_running_sink() {
    let log = Arc::new(Mutex::new(SinkLog::default()));
    let stats = block_on(run_graph(demux_graph(&log), &ZeroClock, LINK_CAPACITY)).expect("run");
    assert_refolded(&stats, &log.lock().unwrap());
}

#[cfg(feature = "multi-thread")]
#[test]
fn the_threaded_runner_refolds_the_same_way() {
    use g2g_core::runtime::{run_graph_threaded, ThreadSpawner};

    let log = Arc::new(Mutex::new(SinkLog::default()));
    let stats = block_on(run_graph_threaded(
        demux_graph(&log),
        &ZeroClock,
        LINK_CAPACITY,
        &ThreadSpawner,
    ))
    .expect("threaded run");
    assert_refolded(&stats, &log.lock().unwrap());
}

struct LatentFanoutSource;

impl MultiOutputSource for LatentFanoutSource {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    fn output_count(&self) -> usize {
        PORT_COUNT as usize
    }

    fn output_caps(&self, _output: usize) -> Result<Caps, G2gError> {
        Ok(caps())
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn MultiOutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for i in 0..FRAME_COUNT {
                out.push_to(PORT, frame(i)).await?;
            }
            out.push_to(PORT, PipelinePacket::Eos).await?;
            Ok(FRAME_COUNT)
        })
    }

    fn latency(&self) -> LatencyReport {
        HELD_UNIT
    }
}

#[test]
fn a_fan_out_source_reaches_the_fold() {
    let log = Arc::new(Mutex::new(SinkLog::default()));
    let mut graph: Graph<GraphNode> = Graph::new();
    let src = graph.add_fanout_src(GraphNode::fanout_source(LatentFanoutSource), PORT_COUNT);
    let sink = graph.add_sink(GraphNode::element(SyncedSink {
        sync: None,
        log: log.clone(),
    }));
    graph
        .link(src.output(PORT as u8), sink)
        .expect("link fan-out port");
    let stats = block_on(run_graph(graph, &ZeroClock, LINK_CAPACITY)).expect("run");
    assert_eq!(stats.latency, HELD_UNIT);
    assert_eq!(
        log.lock().unwrap().startup_path_latency_ns,
        Some(HELD_UNIT_NS)
    );
}
