//! M1123: the latency fold's liveness reaches every element, the return path of
//! the latency query. Each element here records what it was told and how many
//! frames it had already processed, so a test can check both the answer and that
//! it arrived before the first frame.
#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use g2g_core::query::LatencyReport;
use g2g_core::runtime::{
    block_on, run_graph, run_graph_mutable, run_simple_pipeline, run_source_transform_sink,
    GraphNode, Join2, SourceLoop,
};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, Frame, FrameTiming, G2gError, Graph, OutputSink,
    PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

const FRAME_COUNT: u64 = 32;
const LINK_CAPACITY: usize = 2;
/// Frames for the splice test, enough that the mutation lands mid-stream.
const LONG_STREAM: u64 = 400;
/// A live source's own contribution to the fold, arbitrary but nonzero.
const SOURCE_LATENCY_NS: u64 = 5_000_000;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(4),
        height: Dim::Fixed(4),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn frame(sequence: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        g2g_core::memory::MemoryDomain::System(g2g_core::memory::SystemSlice::from_boxed(
            Box::new([0u8; 16]),
        )),
        FrameTiming {
            pts_ns: sequence * 33_000_000,
            ..FrameTiming::default()
        },
        sequence,
    ))
}

/// Pushes `frames` frames then `Eos`, reporting itself live or not.
struct Src {
    live: bool,
    frames: u64,
}

impl Src {
    fn new(live: bool) -> Self {
        Self {
            live,
            frames: FRAME_COUNT,
        }
    }
}

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
    fn latency(&self) -> LatencyReport {
        match self.live {
            true => LatencyReport::live(SOURCE_LATENCY_NS, Some(SOURCE_LATENCY_NS)),
            false => LatencyReport::ZERO,
        }
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for i in 0..self.frames {
                out.push(frame(i)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.frames)
        })
    }
}

/// What one element was told, and how many frames it had processed by then.
type Told = Arc<Mutex<Vec<(bool, usize)>>>;

/// Forwards everything, recording each liveness signal against the number of
/// frames it had already processed when it arrived.
#[derive(Default)]
struct Probe {
    told: Told,
    seen: Arc<AtomicUsize>,
}

impl Probe {
    fn new() -> Self {
        Self::default()
    }

    fn record(&self) -> Told {
        Arc::clone(&self.told)
    }
}

impl AsyncElement for Probe {
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
    fn configure_liveness(&mut self, live: bool) {
        self.told
            .lock()
            .unwrap()
            .push((live, self.seen.load(Ordering::SeqCst)));
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if matches!(packet, PipelinePacket::DataFrame(_)) {
                self.seen.fetch_add(1, Ordering::SeqCst);
            }
            if !matches!(packet, PipelinePacket::Eos) {
                out.push(packet).await?;
            }
            Ok(())
        })
    }
}

fn assert_told(told: &Told, live: bool, who: &str) {
    assert_eq!(
        told.lock().unwrap().as_slice(),
        &[(live, 0)],
        "{who} is told live={live} once, before its first frame"
    );
}

/// A graph over a source, one transform and a sink, all three recording.
fn run_probed_graph(live: bool) -> (Told, Told) {
    let transform = Probe::new();
    let sink = Probe::new();
    let (transform_told, sink_told) = (transform.record(), sink.record());

    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(Src::new(live)));
    let mid = g.add_transform(GraphNode::element(transform));
    let snk = g.add_sink(GraphNode::element(sink));
    g.link(src, mid).unwrap();
    g.link(mid, snk).unwrap();

    let stats = block_on(run_graph(g, &ZeroClock, LINK_CAPACITY)).expect("graph runs");
    assert_eq!(stats.frames_consumed, FRAME_COUNT);
    assert_eq!(stats.latency.live, live, "the fold agrees with the source");
    (transform_told, sink_told)
}

#[test]
fn a_live_source_makes_the_whole_graph_live() {
    let (transform, sink) = run_probed_graph(true);
    assert_told(&transform, true, "the transform");
    assert_told(&sink, true, "the sink");
}

#[test]
fn a_graph_with_no_live_source_is_told_so() {
    let (transform, sink) = run_probed_graph(false);
    assert_told(&transform, false, "the transform");
    assert_told(&sink, false, "the sink");
}

#[test]
fn the_linear_runners_hand_liveness_down() {
    let mut source = Src::new(true);
    let mut transform = Probe::new();
    let mut sink = Probe::new();
    let (transform_told, sink_told) = (transform.record(), sink.record());
    block_on(run_source_transform_sink(
        &mut source,
        &mut transform,
        &mut sink,
        &ZeroClock,
        LINK_CAPACITY,
    ))
    .expect("pipeline runs");
    assert_told(&transform_told, true, "the transform");
    assert_told(&sink_told, true, "the sink");

    let mut source = Src::new(false);
    let mut sink = Probe::new();
    let told = sink.record();
    block_on(run_simple_pipeline(
        &mut source,
        &mut sink,
        &ZeroClock,
        LINK_CAPACITY,
    ))
    .expect("pipeline runs");
    assert_told(&told, false, "the source -> sink runner's sink");
}

/// M1115 splices an element into a graph that is already running, so the
/// liveness the runner folded at startup has to reach it there too.
#[test]
fn a_spliced_element_is_told_the_runs_liveness() {
    let sink = Probe::new();
    let sink_told = sink.record();
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(Src {
        live: true,
        frames: LONG_STREAM,
    }));
    let snk = g.add_sink(GraphNode::element(sink));
    g.set_node_name(src, "src".into());
    g.link(src, snk).unwrap();

    let spliced = Probe::new();
    let spliced_told = spliced.record();
    let (mutator, run) = run_graph_mutable(g, &ZeroClock, LINK_CAPACITY);
    let driver = async {
        mutator
            .insert_after("src", Box::new(spliced))
            .await
            .expect("a passthrough splices onto the edge")
    };
    let (stats, _name) = block_on(Join2::new(run, driver));
    stats.expect("the run survives the splice");

    assert_told(&sink_told, true, "the sink");
    assert_told(&spliced_told, true, "the spliced element");
}
