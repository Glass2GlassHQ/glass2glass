//! DAG runner D4: mid-stream re-solve + β allocation re-cascade over a graph.
//! A source emits a `CapsChanged` mid-stream; `run_graph` re-solves each branch
//! independently, walks the sink's re-derived allocation proposal upstream
//! through the branch transform (β), fails a rejecting branch loud, and
//! re-configures muxer inputs per pad. Pure-fake elements (no hardware).

use core::future::Future;
use core::pin::Pin;
use std::boxed::Box;
use std::sync::{Arc, Mutex};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_graph, GraphNode, SourceLoop};
use g2g_core::{
    AllocationParams, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim,
    FrameTiming, G2gError, Graph, MemoryDomain, MultiInputElement, OutputSink, PipelineClock,
    PipelinePacket, Rate, RawVideoFormat,
};

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn nv12(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn nv12_any() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
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

/// Pool size as a pure function of caps geometry, so a geometry change shows up
/// as a different recorded size.
fn geometry_size(caps: &Caps) -> Option<usize> {
    match caps.dims()? {
        (Dim::Fixed(w), Dim::Fixed(h), _) => Some(*w as usize * *h as usize),
        _ => None,
    }
}

/// Scripted source: emits `before` frames under `initial` caps, then an optional
/// mid-stream `CapsChanged(switch)`, then `after` frames, then EOS.
struct ScriptedSource {
    initial: Caps,
    switch: Option<Caps>,
    before: u32,
    after: u32,
}

impl SourceLoop for ScriptedSource {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.initial.clone()))
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let mut seq = 0u64;
            for _ in 0..self.before {
                out.push(frame(seq)).await?;
                seq += 1;
            }
            if let Some(caps) = self.switch.clone() {
                out.push(PipelinePacket::CapsChanged(caps)).await?;
            }
            for _ in 0..self.after {
                out.push(frame(seq)).await?;
                seq += 1;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(seq)
        })
    }
}

fn frame(seq: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(vec![seq as u8].into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: seq,
        meta: Default::default(),
    })
}

/// Pass-through transform that records every `configure_allocation` size it
/// receives. The β re-cascade made visible: an entry per proposal that reached
/// it (here, the sink's re-derived proposal on a mid-stream caps change).
struct RecordingTransform {
    alloc_log: Arc<Mutex<Vec<usize>>>,
}

impl AsyncElement for RecordingTransform {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn configure_allocation(&mut self, params: &AllocationParams) {
        self.alloc_log.lock().unwrap().push(params.size_bytes);
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

/// NV12 sink whose pool size is a function of caps geometry, restricted to the
/// `accept` set (so a mid-stream change to an unaccepted format fails loud).
struct PoolSink {
    accept: Caps,
}

impl AsyncElement for PoolSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(self.accept.clone()))
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn propose_allocation(&self, caps: &Caps) -> Option<AllocationParams> {
        geometry_size(caps).map(|size| AllocationParams::system(size, 1))
    }

    fn process<'a>(
        &'a mut self,
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

/// Muxer that records every per-pad `configure_pipeline(pad, caps)`, so the
/// per-input mid-stream re-solve is observable. Inputs are wildcard (per-frame
/// caps); the output produces a fixed merged caps.
struct RecordingMux {
    inputs: usize,
    output: Caps,
    config_log: Arc<Mutex<Vec<(usize, Caps)>>>,
}

impl MultiInputElement for RecordingMux {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn input_count(&self) -> usize {
        self.inputs
    }

    fn intercept_caps(&self, _input: usize, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn caps_constraint_for_output(&self) -> Result<CapsConstraint<'_>, G2gError> {
        Ok(CapsConstraint::Produces(CapsSet::one(self.output.clone())))
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        self.config_log
            .lock()
            .unwrap()
            .push((input, absolute_caps.clone()));
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(self.output.clone())
    }

    fn process<'a>(
        &'a mut self,
        _input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(f) => {
                    out.push(PipelinePacket::DataFrame(f)).await?;
                }
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

/// Sink that accepts any packet, for the muxer test.
#[derive(Default)]
struct AnySink;

impl AsyncElement for AnySink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

/// A mid-stream geometry change fans across a tee and β re-cascades the sink's
/// new proposal one hop upstream to each branch transform, independently. Both
/// branch transforms record the new geometry's size.
#[tokio::test]
async fn tee_diamond_recascades_each_branch() {
    let log_a = Arc::new(Mutex::new(Vec::new()));
    let log_b = Arc::new(Mutex::new(Vec::new()));

    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(ScriptedSource {
        initial: nv12(8, 8),
        switch: Some(nv12(16, 16)),
        before: 2,
        after: 2,
    }));
    let tee = g.add_tee(2);
    let ta = g.add_transform(GraphNode::element(RecordingTransform {
        alloc_log: Arc::clone(&log_a),
    }));
    let tb = g.add_transform(GraphNode::element(RecordingTransform {
        alloc_log: Arc::clone(&log_b),
    }));
    let sa = g.add_sink(GraphNode::element(PoolSink { accept: nv12_any() }));
    let sb = g.add_sink(GraphNode::element(PoolSink { accept: nv12_any() }));
    g.link(src, tee.input()).unwrap();
    g.link(tee.out(0), ta).unwrap();
    g.link(tee.out(1), tb).unwrap();
    g.link(ta, sa).unwrap();
    g.link(tb, sb).unwrap();

    let stats = run_graph(g, &NullClock, 4)
        .await
        .expect("tee diamond re-solves");
    assert_eq!(stats.frames_emitted, 4);
    assert_eq!(
        stats.frames_consumed, 8,
        "both branches delivered all 4 frames"
    );
    assert_eq!(
        *log_a.lock().unwrap(),
        vec![8 * 8, 16 * 16],
        "branch A transform records the startup proposal then the β re-cascade"
    );
    assert_eq!(
        *log_b.lock().unwrap(),
        vec![8 * 8, 16 * 16],
        "branch B re-cascades independently of branch A"
    );
    assert!(
        stats.coordinator_events >= 2,
        "each sink reported its mid-stream change"
    );
}

/// No mid-stream change: only the startup allocation cascade configures each
/// branch transform; β never fires. Proves the second entry in the test above
/// is the re-cascade, not a duplicate startup configure.
#[tokio::test]
async fn no_change_leaves_branches_at_startup_proposal() {
    let log_a = Arc::new(Mutex::new(Vec::new()));
    let log_b = Arc::new(Mutex::new(Vec::new()));

    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(ScriptedSource {
        initial: nv12(8, 8),
        switch: None,
        before: 4,
        after: 0,
    }));
    let tee = g.add_tee(2);
    let ta = g.add_transform(GraphNode::element(RecordingTransform {
        alloc_log: Arc::clone(&log_a),
    }));
    let tb = g.add_transform(GraphNode::element(RecordingTransform {
        alloc_log: Arc::clone(&log_b),
    }));
    let sa = g.add_sink(GraphNode::element(PoolSink { accept: nv12_any() }));
    let sb = g.add_sink(GraphNode::element(PoolSink { accept: nv12_any() }));
    g.link(src, tee.input()).unwrap();
    g.link(tee.out(0), ta).unwrap();
    g.link(tee.out(1), tb).unwrap();
    g.link(ta, sa).unwrap();
    g.link(tb, sb).unwrap();

    let stats = run_graph(g, &NullClock, 4).await.expect("tee diamond runs");
    assert_eq!(stats.frames_consumed, 8);
    assert_eq!(
        *log_a.lock().unwrap(),
        vec![8 * 8],
        "startup cascade only, no β"
    );
    assert_eq!(*log_b.lock().unwrap(), vec![8 * 8]);
    assert_eq!(
        stats.coordinator_events, 0,
        "no reports without a mid-stream change"
    );
}

/// A branch whose sink rejects the mid-stream format fails the whole graph loud
/// (strict default), even though the other branch could carry the change.
#[tokio::test]
async fn rejecting_branch_fails_loud() {
    let mut g: Graph<GraphNode> = Graph::new();
    // Source negotiates NV12, then switches to RGBA mid-stream.
    let src = g.add_source(GraphNode::source(ScriptedSource {
        initial: nv12(8, 8),
        switch: Some(rgba(8, 8)),
        before: 2,
        after: 2,
    }));
    let tee = g.add_tee(2);
    // Branch A accepts any NV12 or RGBA geometry; branch B only NV12.
    let any = g.add_sink(GraphNode::element(PoolSink {
        accept: Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        },
    }));
    let nv12_only = g.add_sink(GraphNode::element(PoolSink { accept: nv12_any() }));
    g.link(src, tee.input()).unwrap();
    g.link(tee.out(0), any).unwrap();
    g.link(tee.out(1), nv12_only).unwrap();

    let result = run_graph(g, &NullClock, 4).await;
    assert_eq!(
        result.err(),
        Some(G2gError::CapsMismatch),
        "the NV12-only branch rejects the RGBA mid-stream change"
    );
}

/// A mid-stream change on one muxer input re-configures only that pad. The
/// recorded per-pad configures show the startup caps plus the mid-stream caps on
/// the changed pad.
#[tokio::test]
async fn muxer_resolves_per_input() {
    let config_log = Arc::new(Mutex::new(Vec::new()));

    let mut g: Graph<GraphNode> = Graph::new();
    let s0 = g.add_source(GraphNode::source(ScriptedSource {
        initial: nv12(8, 8),
        switch: Some(nv12(16, 16)),
        before: 2,
        after: 2,
    }));
    let s1 = g.add_source(GraphNode::source(ScriptedSource {
        initial: nv12(8, 8),
        switch: None,
        before: 3,
        after: 0,
    }));
    let mux = g.add_muxer(
        GraphNode::muxer(RecordingMux {
            inputs: 2,
            output: nv12(8, 8),
            config_log: Arc::clone(&config_log),
        }),
        2,
    );
    let sink = g.add_sink(GraphNode::element(AnySink));
    g.link(s0, mux.input(0)).unwrap();
    g.link(s1, mux.input(1)).unwrap();
    g.link(mux.output(), sink).unwrap();

    let stats = run_graph(g, &NullClock, 4)
        .await
        .expect("muxer per-input re-solve runs");
    assert_eq!(stats.frames_emitted, 7, "4 + 3 source frames");

    let log = config_log.lock().unwrap();
    // pad 0 mid-stream re-config to 16x16 happened; pad 1 never changed.
    assert!(
        log.contains(&(0, nv12(16, 16))),
        "pad 0 re-configured mid-stream: {log:?}"
    );
    assert!(
        !log.iter()
            .any(|(pad, caps)| *pad == 1 && *caps == nv12(16, 16)),
        "pad 1 did not change"
    );
}
