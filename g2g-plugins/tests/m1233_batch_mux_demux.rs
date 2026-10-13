#![cfg(all(feature = "std", feature = "metadata"))]

use core::future::Future;
use core::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use g2g_core::runtime::{parse_launch, run_graph, GraphNode, LaunchFactory};
use g2g_core::{
    AsyncElement, BatchMeta, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming,
    G2gError, Graph, MultiOutputElement, OutputSink, PipelineClock, PipelinePacket, Rate,
    RawVideoFormat,
};
use g2g_plugins::batchdemux::BatchDemux;
use g2g_plugins::batchmux::BatchMux;
use g2g_plugins::registry::default_registry;
use g2g_plugins::videotestsrc::VideoTestSrc;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[derive(Debug, Default)]
struct Received {
    caps: Option<Caps>,
    frames: Vec<(FrameTiming, u64)>,
    eos: bool,
}

fn rgba(width: Dim, height: Dim, framerate: Rate) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width,
        height,
        framerate,
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

struct RecordingSink(Arc<Mutex<Received>>);

impl AsyncElement for RecordingSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(rgba(Dim::Any, Dim::Any, Rate::Any)))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.0.lock().unwrap().caps = Some(absolute_caps.clone());
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        let mut received = self.0.lock().unwrap();
        match packet {
            PipelinePacket::DataFrame(frame) => {
                received.frames.push((frame.timing, frame.sequence));
            }
            PipelinePacket::Eos => received.eos = true,
            _ => {}
        }
        Box::pin(async { Ok(()) })
    }
}

struct BatchProbe(Arc<Mutex<Vec<Vec<usize>>>>);

impl AsyncElement for BatchProbe {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        if let PipelinePacket::DataFrame(frame) = &packet {
            let indices = frame
                .meta
                .get::<BatchMeta>()
                .map(|batch| batch.streams.iter().map(|stream| stream.index).collect())
                .unwrap_or_default();
            self.0.lock().unwrap().push(indices);
        }
        Box::pin(async move {
            out.push(packet).await?;
            Ok(())
        })
    }
}

fn expected_frames(fps: u64, count: u64) -> Vec<(u64, u64)> {
    let step = 1_000_000_000 / fps;
    (0..count).map(|i| (i * step, i)).collect()
}

fn pts_and_sequence(received: &Received) -> Vec<(u64, u64)> {
    received
        .frames
        .iter()
        .map(|(timing, sequence)| (timing.pts_ns, *sequence))
        .collect()
}

async fn run_batch_graph(
    first_frames: u64,
    second_frames: u64,
) -> (Received, Received, Vec<Vec<usize>>) {
    let sinks = [
        Arc::new(Mutex::new(Received::default())),
        Arc::new(Mutex::new(Received::default())),
    ];
    let batches = Arc::new(Mutex::new(Vec::new()));
    let mut g: Graph<GraphNode> = Graph::new();
    let first = g.add_source(GraphNode::source(VideoTestSrc::new(8, 8, 30, first_frames)));
    let second = g.add_source(GraphNode::source(VideoTestSrc::new(
        16,
        8,
        25,
        second_frames,
    )));
    let mux = g.add_muxer(GraphNode::muxer(BatchMux::new(2)), 2);
    let probe = g.add_transform(GraphNode::element(BatchProbe(batches.clone())));
    let demux = g.add_demux(GraphNode::demux(BatchDemux::new(2)), 2);
    g.link(first, mux.input(0)).unwrap();
    g.link(second, mux.input(1)).unwrap();
    g.link(mux.output(), probe).unwrap();
    g.link(probe, demux.input()).unwrap();
    for (port, received) in sinks.iter().enumerate() {
        let sink = g.add_sink(GraphNode::element(RecordingSink(received.clone())));
        g.link(demux.out(port as u8), sink).unwrap();
    }
    run_graph(g, &ZeroClock, 4).await.expect("batch graph runs");
    let [first, second] = sinks.map(|s| std::mem::take(&mut *s.lock().unwrap()));
    let batches = std::mem::take(&mut *batches.lock().unwrap());
    (first, second, batches)
}

#[tokio::test]
async fn batches_reach_each_branch_with_their_original_timing() {
    let (first, second, batches) = run_batch_graph(4, 4).await;
    assert_eq!(
        first.caps,
        Some(rgba(Dim::Fixed(8), Dim::Fixed(8), Rate::Fixed(30 << 16))),
        "branch 0 negotiated its own stream, not the batch"
    );
    assert_eq!(
        second.caps,
        Some(rgba(Dim::Fixed(16), Dim::Fixed(8), Rate::Fixed(25 << 16)))
    );
    assert_eq!(pts_and_sequence(&first), expected_frames(30, 4));
    assert_eq!(pts_and_sequence(&second), expected_frames(25, 4));
    assert!(first.eos && second.eos);
    assert_eq!(
        batches,
        vec![vec![0, 1]; 4],
        "every batch held both streams"
    );
}

#[tokio::test]
async fn an_ended_input_drops_out_of_later_batches() {
    let (first, second, batches) = run_batch_graph(2, 4).await;
    assert_eq!(pts_and_sequence(&first), expected_frames(30, 2));
    assert!(first.eos, "the short branch still ends");
    assert_eq!(pts_and_sequence(&second), expected_frames(25, 4));
    assert_eq!(
        batches,
        vec![vec![0, 1], vec![0, 1], vec![1], vec![1]],
        "the last two batches hold only stream 1"
    );
}

static FIRST_SINK_FRAMES: AtomicUsize = AtomicUsize::new(0);
static SECOND_SINK_FRAMES: AtomicUsize = AtomicUsize::new(0);

struct CountingSink(&'static AtomicUsize);

impl AsyncElement for CountingSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        if let PipelinePacket::DataFrame(_) = packet {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
        Box::pin(async { Ok(()) })
    }
}

#[tokio::test]
async fn launch_line_batches_and_splits_two_streams() {
    let mut reg = default_registry();
    let line = "videotestsrc num-buffers=4 ! m.sink_0   videotestsrc num-buffers=4 ! m.sink_1   \
                streammux name=m ! streamdemux name=d   d.src_0 ! fakesink   d.src_1 ! fakesink";
    let graph = parse_launch(&reg, line).expect("batch line parses");
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .expect("batch line runs to EOS");
    assert_eq!(stats.frames_consumed, 8, "4 frames reached each fakesink");

    reg.register_launch(LaunchFactory::new("firstsink", Vec::new(), || {
        Box::new(CountingSink(&FIRST_SINK_FRAMES))
    }));
    reg.register_launch(LaunchFactory::new("secondsink", Vec::new(), || {
        Box::new(CountingSink(&SECOND_SINK_FRAMES))
    }));
    let line = "videotestsrc num-buffers=4 ! m.sink_0   videotestsrc num-buffers=4 ! m.sink_1   \
                streammux name=m ! streamdemux name=d   d.src_0 ! firstsink   d.src_1 ! secondsink";
    let graph = parse_launch(&reg, line).expect("batch line parses");
    run_graph(graph, &ZeroClock, 4)
        .await
        .expect("batch line runs to EOS");
    assert_eq!(FIRST_SINK_FRAMES.load(Ordering::Relaxed), 4);
    assert_eq!(SECOND_SINK_FRAMES.load(Ordering::Relaxed), 4);
}

#[test]
fn registry_lists_and_describes_the_batch_elements() {
    let reg = default_registry();
    let names = reg.element_names();
    assert!(names.contains(&"streammux"));
    assert!(names.contains(&"streamdemux"));
    let docs = reg.describe_all();
    let doc = |name: &str| docs.iter().find(|d| d.name == name).cloned();
    assert_eq!(
        doc("streammux").expect("streammux described").klass,
        "Mux/Batch"
    );
    let demux = doc("streamdemux").expect("streamdemux described");
    assert_eq!(demux.long_name, "Stream demux");
    assert_eq!(demux.klass, "Demux/Batch");
    let inspected = reg.inspect("streamdemux").expect("streamdemux inspects");
    assert!(inspected.contains("Stream demux"), "{inspected}");
    assert!(inspected.contains("Demux/Batch"), "{inspected}");
}

#[test]
fn demux_with_more_ports_than_streams_fails_at_configure() {
    let stream = rgba(Dim::Fixed(8), Dim::Fixed(8), Rate::Fixed(30 << 16));
    let batch = Caps::Batch {
        streams: vec![stream.clone(), stream],
    };
    let mut demux = BatchDemux::new(3);
    let err = demux
        .configure_pipeline(&batch)
        .expect_err("3 ports cannot split 2 streams");
    assert_eq!(
        err,
        G2gError::PortCountMismatch {
            ports: 3,
            streams: 2
        }
    );
    assert_eq!(
        format!("{err:?}"),
        "PortCountMismatch { ports: 3, streams: 2 }"
    );
}
