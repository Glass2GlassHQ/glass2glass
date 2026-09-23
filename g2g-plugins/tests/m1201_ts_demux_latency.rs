#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::sync::{Arc, Mutex};

use g2g_core::clock::{ClockCandidate, ClockPriority, ClockSync};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{block_on, run_graph, GraphNode, SourceLoop};
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, ConfigureOutcome, Frame, FrameTiming, G2gError, Graph,
    OutputSink, PipelineClock, PipelinePacket,
};
use g2g_plugins::mpegts::{TsMuxer, STREAM_TYPE_H264};
use g2g_plugins::tsdemux::{TsDemuxN, TsStream};

const ACCESS_UNIT_COUNT: u64 = 6;
const FIRST_PTS_90KHZ: u64 = 900_000;
const FRAME_INTERVAL_90KHZ: u64 = 3_600;
const PTS_CLOCK_HZ: u64 = 90_000;
const NANOS_PER_SECOND: u64 = 1_000_000_000;
const FRAME_INTERVAL_NS: u64 = FRAME_INTERVAL_90KHZ * NANOS_PER_SECOND / PTS_CLOCK_HZ;
const LINK_CAPACITY: usize = 2;
const VIDEO_PORT: u8 = 0;
const PORT_COUNT: u8 = 1;
const IDR_ACCESS_UNIT: [u8; 6] = [0, 0, 0, 1, 0x65, 0x11];
const P_ACCESS_UNIT: [u8; 6] = [0, 0, 0, 1, 0x41, 0x22];

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn transport_stream_caps() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    }
}

// One access unit's TS packets per frame, the way a live capture delivers them.
fn transport_stream_chunks() -> Vec<Vec<u8>> {
    let mut muxer = TsMuxer::with_streams(&[STREAM_TYPE_H264]);
    (0..ACCESS_UNIT_COUNT)
        .map(|i| {
            let access_unit = if i == 0 {
                &IDR_ACCESS_UNIT
            } else {
                &P_ACCESS_UNIT
            };
            let pts = FIRST_PTS_90KHZ + i * FRAME_INTERVAL_90KHZ;
            muxer.push_au(access_unit, Some(pts), None)
        })
        .collect()
}

struct ChunkSource {
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
        core::future::ready(Ok(transport_stream_caps()))
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let chunks = core::mem::take(&mut self.chunks);
            let count = chunks.len() as u64;
            for (sequence, chunk) in chunks.into_iter().enumerate() {
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(chunk.into_boxed_slice())),
                    FrameTiming::default(),
                    sequence as u64,
                );
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(count)
        })
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

#[test]
fn tsdemux_paces_its_sinks_with_the_access_unit_it_holds() {
    let log = Arc::new(Mutex::new(SinkLog::default()));
    let mut graph: Graph<GraphNode> = Graph::new();
    let src = graph.add_source(GraphNode::source(ChunkSource {
        chunks: transport_stream_chunks(),
    }));
    let demux = graph.add_demux(
        GraphNode::demux(TsDemuxN::new(vec![TsStream::H264])),
        PORT_COUNT,
    );
    let sink = graph.add_sink(GraphNode::element(SyncedSink {
        sync: None,
        log: log.clone(),
    }));
    graph
        .link(src, demux.input())
        .expect("link source to demux");
    graph
        .link(demux.out(VIDEO_PORT), sink)
        .expect("link video port");

    let stats = block_on(run_graph(graph, &ZeroClock, LINK_CAPACITY)).expect("run");

    let log = log.lock().unwrap();
    assert_eq!(
        log.per_frame_path_latency_ns.len() as u64,
        ACCESS_UNIT_COUNT,
        "every access unit reaches the sink"
    );
    assert_eq!(
        log.startup_path_latency_ns,
        Some(0),
        "nothing is parsed before the run starts"
    );
    assert_eq!(
        log.per_frame_path_latency_ns.last(),
        Some(&FRAME_INTERVAL_NS),
        "the sink ends paced with one held access unit: {:?}",
        log.per_frame_path_latency_ns
    );
    assert_eq!(stats.latency.min_ns, FRAME_INTERVAL_NS);
}
