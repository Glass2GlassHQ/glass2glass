//! M724 mid-stream re-solve through the fan-in session runners: a source that
//! changes caps mid-stream re-configures its session input pad (both the
//! standalone `run_fanin_session` and the graph `FaninSink` arm), so a
//! resolution change reaches e.g. a WebRTC session's layer metadata instead of
//! dying between the pad and the element.

use core::future::{ready, Future, Ready};
use core::pin::Pin;
use std::sync::{Arc, Mutex};

use g2g_core::fanout::MultiInputElement;
use g2g_core::runtime::{run_fanin_session, run_graph, DynSourceLoop, GraphNode, SourceLoop};
use g2g_core::{
    Caps, CapsConstraint, ConfigureOutcome, Dim, G2gError, Graph, OutputSink, PipelineClock,
    PipelinePacket, Rate, RawVideoFormat,
};

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

/// Terminal session recording every `(input, caps)` configure call.
struct RecordingSession {
    pads: usize,
    configures: Arc<Mutex<Vec<(usize, Caps)>>>,
}

impl MultiInputElement for RecordingSession {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn input_count(&self) -> usize {
        self.pads
    }
    fn intercept_caps(&self, _i: usize, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn configure_pipeline(&mut self, i: usize, c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configures.lock().unwrap().push((i, c.clone()));
        Ok(ConfigureOutcome::Accepted)
    }
    fn output_caps(&self) -> Result<Caps, G2gError> {
        Err(G2gError::CapsMismatch)
    }
    fn caps_constraint_as_input(&self, _i: usize) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn is_terminal(&self) -> bool {
        true
    }
    fn process<'a>(
        &'a mut self,
        _i: usize,
        _p: PipelinePacket,
        _o: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

/// Source that switches geometry mid-stream: caps A, 2 frames, caps B, 2 frames.
struct SwitchingSrc;

impl SourceLoop for SwitchingSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = Ready<Result<Caps, G2gError>>;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        ready(Ok(rgba(8, 8)))
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let mut seq = 0u64;
            for (w, h) in [(8u32, 8u32), (16, 16)] {
                out.push(PipelinePacket::CapsChanged(rgba(w, h))).await?;
                for _ in 0..2 {
                    let buf = vec![0u8; (w * h * 4) as usize];
                    let frame = g2g_core::frame::Frame::new(
                        g2g_core::MemoryDomain::System(g2g_core::memory::SystemSlice::from_boxed(
                            buf.into_boxed_slice(),
                        )),
                        g2g_core::FrameTiming::default(),
                        seq,
                    );
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                    seq += 1;
                }
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(seq)
        })
    }
}

fn configures_of(log: &Arc<Mutex<Vec<(usize, Caps)>>>, input: usize) -> Vec<(u32, u32)> {
    log.lock()
        .unwrap()
        .iter()
        .filter(|(i, _)| *i == input)
        .filter_map(|(_, c)| match c {
            Caps::RawVideo {
                width: Dim::Fixed(w),
                height: Dim::Fixed(h),
                ..
            } => Some((*w, *h)),
            _ => None,
        })
        .collect()
}

/// The standalone fan-in session runner re-configures the changed pad.
#[tokio::test]
async fn fanin_session_reconfigures_on_midstream_caps() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut src = SwitchingSrc;
    let mut session = RecordingSession {
        pads: 1,
        configures: log.clone(),
    };
    let clock = ZeroClock;
    let sources: Vec<&mut dyn DynSourceLoop> = vec![&mut src];
    run_fanin_session(sources, &mut session, &clock, 4)
        .await
        .expect("session runs");
    let dims = configures_of(&log, 0);
    assert!(
        dims.ends_with(&[(8, 8), (16, 16)]),
        "startup + both mid-stream configures, in order: {dims:?}"
    );
}

/// The graph `FaninSink` arm re-configures the changed pad too.
#[tokio::test]
async fn graph_fanin_node_reconfigures_on_midstream_caps() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(SwitchingSrc));
    let session = g.add_fanin_sink(
        GraphNode::muxer(RecordingSession {
            pads: 1,
            configures: log.clone(),
        }),
        1,
    );
    g.link(src, session.input(0)).unwrap();
    run_graph(g, &ZeroClock, 4).await.expect("graph runs");
    let dims = configures_of(&log, 0);
    assert!(
        dims.ends_with(&[(8, 8), (16, 16)]),
        "startup + both mid-stream configures, in order: {dims:?}"
    );
}
