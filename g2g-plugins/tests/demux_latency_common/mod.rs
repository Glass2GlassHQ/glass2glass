use core::future::Future;
use core::pin::Pin;
use std::sync::{Arc, Mutex};

use g2g_core::clock::{ClockCandidate, ClockPriority, ClockSync};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{block_on, run_graph, GraphNode, RunStats, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Frame, FrameTiming, G2gError, Graph, MultiOutputElement,
    OutputSink, PipelineClock, PipelinePacket,
};

const LINK_CAPACITY: usize = 2;
const PORT: u8 = 0;
const PORT_COUNT: u8 = 1;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

pub(crate) fn data_frame(payload: Vec<u8>, timing: FrameTiming, sequence: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(payload.into_boxed_slice())),
        timing,
        sequence,
    ))
}

// Pushes one chunk per frame, the way a live capture delivers the container.
struct ChunkSource {
    caps: Caps,
    chunks: Vec<Vec<u8>>,
}

impl SourceLoop for ChunkSource {
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

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let chunks = core::mem::take(&mut self.chunks);
            let count = chunks.len() as u64;
            for (sequence, chunk) in chunks.into_iter().enumerate() {
                out.push(data_frame(chunk, FrameTiming::default(), sequence as u64))
                    .await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(count)
        })
    }
}

#[derive(Debug, Default)]
pub(crate) struct SinkLog {
    pub(crate) startup_path_latency_ns: Option<u64>,
    pub(crate) per_frame_path_latency_ns: Vec<u64>,
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

pub(crate) fn run_demux<D>(demux: D, caps: Caps, chunks: Vec<Vec<u8>>) -> (SinkLog, RunStats)
where
    D: MultiOutputElement + 'static,
{
    let log = Arc::new(Mutex::new(SinkLog::default()));
    let mut graph: Graph<GraphNode> = Graph::new();
    let src = graph.add_source(GraphNode::source(ChunkSource { caps, chunks }));
    let demux = graph.add_demux(GraphNode::demux(demux), PORT_COUNT);
    let sink = graph.add_sink(GraphNode::element(SyncedSink {
        sync: None,
        log: log.clone(),
    }));
    graph
        .link(src, demux.input())
        .expect("link source to demux");
    graph.link(demux.out(PORT), sink).expect("link demux port");
    let stats = block_on(run_graph(graph, &ZeroClock, LINK_CAPACITY)).expect("run");
    let log = std::mem::take(&mut *log.lock().unwrap());
    (log, stats)
}
