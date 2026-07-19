//! M727 terminal fan-out source graph node: `run_graph` over a `FanoutSrc`
//! (a 0-in / N-out `MultiOutputSource`, the receive-side mirror of the M713
//! terminal fan-in), so a session source's tracks feed downstream chains
//! (`livekitsrc -> h264parse -> ...` per pad) and a launch line can express it.
//! Pure-fake elements.

use core::future::Future;
use core::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use g2g_core::fanout::{MultiOutputSink, MultiOutputSource};
use g2g_core::graph::NodeKind;
#[cfg(all(feature = "std", feature = "multi-thread"))]
use g2g_core::runtime::run_graph_threaded;
use g2g_core::runtime::{parse_launch, run_graph, FanoutSrcFactory, GraphNode, Registry};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, G2gError, Graph, OutputSink,
    PipelineClock, PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate,
    RawVideoFormat,
};
#[cfg(all(feature = "std", feature = "multi-thread"))]
use g2g_plugins::TokioThreadSpawner;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn rgba(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Two-port fake session source: `frames` DataFrames per port, then Eos on
/// both (the `run` contract). Port 0 is 8x8, port 1 is 16x16.
#[derive(Debug, Default)]
struct FakeSessionSrc {
    frames: u64,
    label: String,
}

impl MultiOutputSource for FakeSessionSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;

    fn output_count(&self) -> usize {
        2
    }
    fn output_caps(&self, output: usize) -> Result<Caps, G2gError> {
        match output {
            0 => Ok(rgba(8, 8)),
            1 => Ok(rgba(16, 16)),
            _ => Err(G2gError::CapsMismatch),
        }
    }
    fn properties(&self) -> &'static [PropertySpec] {
        static PROPS: &[PropertySpec] = &[PropertySpec::new("label", PropKind::Str, "test label")];
        PROPS
    }
    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "label" => {
                self.label = value.as_str().ok_or(PropError::Type)?.into();
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn MultiOutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let mut pushed = 0u64;
            for port in 0..2usize {
                let (w, h) = if port == 0 { (8u32, 8u32) } else { (16, 16) };
                out.push_to(port, PipelinePacket::CapsChanged(rgba(w, h)))
                    .await?;
                for seq in 0..self.frames {
                    let buf = vec![0u8; (w * h * 4) as usize];
                    let frame = g2g_core::frame::Frame::new(
                        g2g_core::MemoryDomain::System(g2g_core::memory::SystemSlice::from_boxed(
                            buf.into_boxed_slice(),
                        )),
                        g2g_core::FrameTiming::default(),
                        seq,
                    );
                    out.push_to(port, PipelinePacket::DataFrame(frame)).await?;
                    pushed += 1;
                }
            }
            out.push_to(0, PipelinePacket::Eos).await?;
            out.push_to(1, PipelinePacket::Eos).await?;
            Ok(pushed)
        })
    }
}

/// Counting sink for one branch.
struct CountSink {
    frames: Arc<AtomicU64>,
}

impl AsyncElement for CountSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if matches!(packet, PipelinePacket::DataFrame(_)) {
                self.frames.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        })
    }
}

fn build(frames: u64, c0: Arc<AtomicU64>, c1: Arc<AtomicU64>) -> Graph<GraphNode> {
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_fanout_src(
        GraphNode::fanout_source(FakeSessionSrc {
            frames,
            label: String::new(),
        }),
        2,
    );
    let s0 = g.add_sink(GraphNode::element(CountSink { frames: c0 }));
    let s1 = g.add_sink(GraphNode::element(CountSink { frames: c1 }));
    g.link(src.output(0), s0).unwrap();
    g.link(src.output(1), s1).unwrap();
    g
}

/// Each port's frames reach its own branch, and the run ends on both Eos.
#[tokio::test]
async fn fanout_src_feeds_both_branches() {
    let (c0, c1) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
    let stats = run_graph(build(4, c0.clone(), c1.clone()), &ZeroClock, 4)
        .await
        .expect("fan-out source graph runs");
    assert_eq!(stats.frames_emitted, 8, "4 frames per port");
    assert_eq!(stats.frames_consumed, 8);
    assert_eq!(c0.load(Ordering::SeqCst), 4);
    assert_eq!(c1.load(Ordering::SeqCst), 4);
}

/// The threaded runner drives the same shape to the same counts.
#[cfg(all(feature = "std", feature = "multi-thread"))]
#[tokio::test]
async fn threaded_matches_cooperative() {
    let (c0, c1) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
    let stats = run_graph_threaded(
        build(3, c0.clone(), c1.clone()),
        &ZeroClock,
        4,
        &TokioThreadSpawner,
    )
    .await
    .expect("threaded run");
    assert_eq!(stats.frames_emitted, 6);
    assert_eq!(c0.load(Ordering::SeqCst), 3);
    assert_eq!(c1.load(Ordering::SeqCst), 3);
}

/// A launch line expresses the fan-out source with named output pads and
/// applies its properties.
#[tokio::test]
async fn launch_line_builds_fanout_src() {
    let mut reg = Registry::new();
    reg.register_fanout_src(FanoutSrcFactory::new("fakesession", |_n| {
        Box::new(FakeSessionSrc {
            frames: 3,
            label: String::new(),
        })
    }));
    reg.register_launch(g2g_core::runtime::LaunchFactory::new(
        "anysink",
        Vec::new(),
        || {
            Box::new(CountSink {
                frames: Arc::new(AtomicU64::new(0)),
            })
        },
    ));

    const LINE: &str = "fakesession name=s label=hello   s. ! anysink   s. ! anysink";
    let vg = parse_launch(&reg, LINE)
        .expect("fan-out source line parses")
        .finish()
        .expect("valid graph");
    let nodes: Vec<NodeKind> = vg
        .topo()
        .iter()
        .map(|&n| vg.kind(n))
        .filter(|k| matches!(k, NodeKind::FanoutSrc(_)))
        .collect();
    assert_eq!(nodes, [NodeKind::FanoutSrc(2)], "one 2-output source node");

    let stats = run_graph(parse_launch(&reg, LINE).expect("parses"), &ZeroClock, 4)
        .await
        .expect("launch line runs");
    assert_eq!(stats.frames_emitted, 6);
    assert_eq!(stats.frames_consumed, 6);
}
