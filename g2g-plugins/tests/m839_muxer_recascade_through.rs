//! M839: the β allocation re-cascade walks *through* a muxer.
//!
//! M344 made a mid-stream change on one muxer input re-cascade up that pad's
//! branch alone. The walk stopped there: the merged output is derived from every
//! pad, so a change on one input can move the output pool, and a pool the output
//! settles on can move the demand on the *other* pads. This drives the
//! continuation: the muxer re-derives its output allocation from the changed
//! input plus the unchanged others, tells the consumer arm reading its output,
//! and re-cascades up the other pads only when the new output actually moved
//! their demand. The loop runs to a fixed point, and a mutually-constraining pair
//! that cannot settle fails loud instead of spinning.
//!
//! Graph shape: `src ! tee ! {t0, t1} ! mux ! sink`, where t1 pins its output
//! caps, so a mid-stream geometry change reaches muxer pad 0 only.
//!
//! Pure-fake elements (no hardware).

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

type SizeLog = Arc<Mutex<Vec<usize>>>;

fn log() -> SizeLog {
    Arc::new(Mutex::new(Vec::new()))
}

fn nv12(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
    }
}

/// Pool bytes as a pure function of caps geometry, so a geometry change shows up
/// as a different recorded size.
fn geometry_size(caps: &Caps) -> Option<usize> {
    match caps.dims()? {
        (Dim::Fixed(w), Dim::Fixed(h), _) => Some(*w as usize * *h as usize),
        _ => None,
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

/// Emits `before` frames under `initial`, an optional mid-stream `CapsChanged`,
/// then `after` frames, then EOS.
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

/// Branch transform recording every allocation size it absorbs, so the proposals
/// that re-cascade into it (startup, then whatever the muxer walks up this
/// branch) are observable. `pin` fixes the forwarded output caps, which is how
/// one branch is held at its startup geometry while the other changes.
struct RecordingTransform {
    alloc_log: SizeLog,
    pin: Option<Caps>,
}

impl RecordingTransform {
    fn passthrough(alloc_log: &SizeLog) -> Self {
        Self {
            alloc_log: Arc::clone(alloc_log),
            pin: None,
        }
    }

    fn pinned(alloc_log: &SizeLog, pin: Caps) -> Self {
        Self {
            alloc_log: Arc::clone(alloc_log),
            pin: Some(pin),
        }
    }
}

impl AsyncElement for RecordingTransform {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(self.pin.clone().unwrap_or_else(|| upstream.clone()))
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        match &self.pin {
            Some(caps) => {
                let pinned = caps.clone();
                CapsConstraint::DerivedOutput(Box::new(move |_| CapsSet::one(pinned.clone())))
            }
            None => CapsConstraint::IdentityAny,
        }
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

/// Sink recording the allocation sizes handed to it. With `demand` clear it
/// proposes nothing of its own, so every entry is a pool a producer settled on
/// and pushed downstream. With it set, a mid-stream caps change makes the sink
/// demand a pool of its own, which re-cascades upstream into the muxer.
struct RecordingSink {
    alloc_log: SizeLog,
    demand: bool,
}

impl AsyncElement for RecordingSink {
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

    fn propose_allocation(&self, caps: &Caps) -> Option<AllocationParams> {
        if !self.demand {
            return None;
        }
        Some(AllocationParams::system(geometry_size(caps)? * 4, 3))
    }

    fn configure_allocation(&mut self, params: &AllocationParams) {
        self.alloc_log.lock().unwrap().push(params.size_bytes);
    }

    fn process<'a>(
        &'a mut self,
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

/// An interleave muxer whose merged output is written into one pool sized by the
/// largest input, and whose inputs must in turn be at least that size (a shared
/// staging surface). So a change on one pad moves the output pool, and the output
/// pool moves the demand on the other pads: the mutually-constraining pair the
/// walk has to settle.
///
/// `output_constrains_inputs = false` turns off the second half: the output pool
/// is still re-derived and pushed downstream, but it imposes nothing back on the
/// pads.
struct FloorMux {
    inputs: usize,
    output: Caps,
    pad_caps: Vec<Option<Caps>>,
    /// Bytes the settled output pool imposes on every input pad.
    floor: usize,
    output_constrains_inputs: bool,
    /// Headroom each pad adds on top of the floor, and the pool size it stops at.
    /// Non-zero makes the pads climb toward `limit` a step per round instead of
    /// settling immediately, so the walk has to iterate to reach its fixed point.
    step: usize,
    limit: usize,
    /// Whether the merged output takes pad 0's caps, so a change there reaches
    /// the consumer as a `CapsChanged` and its answer comes back as a demand.
    follow_pad0: bool,
}

impl FloorMux {
    fn new(inputs: usize, output: Caps, output_constrains_inputs: bool) -> Self {
        Self {
            inputs,
            output,
            pad_caps: vec![None; inputs],
            floor: 0,
            output_constrains_inputs,
            step: 0,
            limit: usize::MAX,
            follow_pad0: false,
        }
    }

    fn climbing(inputs: usize, output: Caps, step: usize, limit: usize) -> Self {
        Self {
            step,
            limit,
            ..Self::new(inputs, output, true)
        }
    }

    fn following(inputs: usize, output: Caps) -> Self {
        Self {
            follow_pad0: true,
            ..Self::new(inputs, output, true)
        }
    }

    fn pad_demand(&self, caps: &Caps) -> Option<usize> {
        Some(
            geometry_size(caps)?
                .max(self.floor + self.step)
                .min(self.limit),
        )
    }
}

impl MultiInputElement for FloorMux {
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
        Ok(CapsConstraint::Produces(CapsSet::one(self.output_caps()?)))
    }

    fn output_follows_input(&self) -> Option<usize> {
        self.follow_pad0.then_some(0)
    }

    fn propose_allocation_for_input(&self, _input: usize, caps: &Caps) -> Option<AllocationParams> {
        Some(AllocationParams::system(self.pad_demand(caps)?, 2))
    }

    fn propose_allocation_for_output(&self, _caps: &Caps) -> Option<AllocationParams> {
        let largest = self
            .pad_caps
            .iter()
            .flatten()
            .filter_map(|caps| self.pad_demand(caps))
            .max()?;
        Some(AllocationParams::system(largest, 2))
    }

    fn configure_allocation_for_output(&mut self, params: &AllocationParams) {
        if self.output_constrains_inputs {
            self.floor = params.size_bytes;
        }
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        self.pad_caps[input] = Some(absolute_caps.clone());
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        match (self.follow_pad0, &self.pad_caps[0]) {
            (true, Some(caps)) => Ok(caps.clone()),
            _ => Ok(self.output.clone()),
        }
    }

    fn process<'a>(
        &'a mut self,
        _input: usize,
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

/// A muxer that can never settle: every input pad demands one byte more than the
/// output pool, and the output pool demands one byte more than the largest input.
/// Each round of the walk moves both, so the fixed point does not exist.
struct RunawayMux {
    inputs: usize,
    output: Caps,
    pad_caps: Vec<Option<Caps>>,
    floor: usize,
}

impl MultiInputElement for RunawayMux {
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

    fn propose_allocation_for_input(&self, _input: usize, caps: &Caps) -> Option<AllocationParams> {
        let size = geometry_size(caps)?.max(self.floor + 1);
        Some(AllocationParams::system(size, 1))
    }

    fn propose_allocation_for_output(&self, _caps: &Caps) -> Option<AllocationParams> {
        let largest = self
            .pad_caps
            .iter()
            .enumerate()
            .filter_map(|(pad, caps)| {
                self.propose_allocation_for_input(pad, caps.as_ref()?)
                    .map(|p| p.size_bytes)
            })
            .max()?;
        Some(AllocationParams::system(largest + 1, 1))
    }

    fn configure_allocation_for_output(&mut self, params: &AllocationParams) {
        self.floor = params.size_bytes;
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        self.pad_caps[input] = Some(absolute_caps.clone());
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
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

/// Build `src ! tee ! {t0, t1} ! mux ! sink`, where branch 1 pins its output at
/// `pinned` so only muxer pad 0 sees the mid-stream geometry change.
fn tee_mux_graph(
    mux: GraphNode,
    switch: Option<Caps>,
    pinned: Caps,
    log0: &SizeLog,
    log1: &SizeLog,
    sink_log: &SizeLog,
) -> Graph<GraphNode> {
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(ScriptedSource {
        initial: nv12(8, 8),
        switch,
        before: 2,
        after: 2,
    }));
    let tee = g.add_tee(2);
    let t0 = g.add_transform(GraphNode::element(RecordingTransform::passthrough(log0)));
    let t1 = g.add_transform(GraphNode::element(RecordingTransform::pinned(log1, pinned)));
    let m = g.add_muxer(mux, 2);
    let sink = g.add_sink(GraphNode::element(RecordingSink {
        alloc_log: Arc::clone(sink_log),
        demand: false,
    }));
    g.link(src, tee.input()).unwrap();
    g.link(tee.out(0), t0).unwrap();
    g.link(tee.out(1), t1).unwrap();
    g.link(t0, m.input(0)).unwrap();
    g.link(t1, m.input(1)).unwrap();
    g.link(m.output(), sink).unwrap();
    g
}

/// The walk also crosses the boundary in the other direction: the consumer's own
/// re-derived demand used to die at the muxer (its upstream is a multi-input node
/// with no single continuation). Now the muxer absorbs it as its output pool and
/// carries it onto every input pad, so both branches end up at the sink's demand.
#[tokio::test]
async fn consumer_demand_crosses_into_every_input_pad() {
    let (log0, log1, sink_log) = (log(), log(), log());
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(ScriptedSource {
        initial: nv12(8, 8),
        switch: Some(nv12(16, 16)),
        before: 2,
        after: 4,
    }));
    let tee = g.add_tee(2);
    let t0 = g.add_transform(GraphNode::element(RecordingTransform::passthrough(&log0)));
    let t1 = g.add_transform(GraphNode::element(RecordingTransform::pinned(
        &log1,
        nv12(8, 8),
    )));
    // The merged output follows pad 0, so the change reaches the sink as a
    // `CapsChanged` and the sink answers with a pool of its own.
    let m = g.add_muxer(GraphNode::muxer(FloorMux::following(2, nv12(8, 8))), 2);
    let sink = g.add_sink(GraphNode::element(RecordingSink {
        alloc_log: Arc::clone(&sink_log),
        demand: true,
    }));
    g.link(src, tee.input()).unwrap();
    g.link(tee.out(0), t0).unwrap();
    g.link(tee.out(1), t1).unwrap();
    g.link(t0, m.input(0)).unwrap();
    g.link(t1, m.input(1)).unwrap();
    g.link(m.output(), sink).unwrap();

    run_graph(g, &NullClock, 4)
        .await
        .expect("the consumer's demand crosses the muxer");
    // 16x16 x 4 bytes: the sink's demand, which only reaches the branches by
    // crossing the muxer boundary.
    assert_eq!(
        log0.lock().unwrap().last().copied(),
        Some(16 * 16 * 4),
        "the changed pad's branch ends at the consumer's demand"
    );
    assert_eq!(
        log1.lock().unwrap().last().copied(),
        Some(16 * 16 * 4),
        "the unchanged pad's branch does too"
    );
}

/// The walk continues through the muxer: a mid-stream geometry change on pad 0
/// re-cascades up pad 0's branch, re-derives the merged-output pool (8x8 -> 16x16
/// worth of bytes), pushes it to the consumer reading that output, and lifts pad
/// 1's demand to the same floor, which re-cascades up pad 1's branch even though
/// pad 1's own caps never changed.
#[tokio::test]
async fn input_change_walks_through_muxer_onto_the_other_pad() {
    let (log0, log1, sink_log) = (log(), log(), log());
    let g = tee_mux_graph(
        GraphNode::muxer(FloorMux::new(2, nv12(8, 8), true)),
        Some(nv12(16, 16)),
        nv12(8, 8),
        &log0,
        &log1,
        &sink_log,
    );

    let stats = run_graph(g, &NullClock, 4)
        .await
        .expect("the walk through the muxer runs");
    assert_eq!(
        stats.frames_emitted, 4,
        "the source's frames, tee'd to both pads"
    );
    assert_eq!(
        *log0.lock().unwrap(),
        vec![8 * 8, 16 * 16],
        "the changed pad's branch absorbed the startup proposal then the re-cascade"
    );
    assert_eq!(
        *log1.lock().unwrap(),
        vec![8 * 8, 16 * 16],
        "the unchanged pad's branch was lifted by the muxer's new output pool"
    );
    assert_eq!(
        *sink_log.lock().unwrap(),
        vec![16 * 16],
        "the consumer learned the pool the muxer settled on for its output"
    );
}

/// Convergence over several rounds: each pad wants a step of headroom above the
/// output pool, and the output pool is the largest pad demand, so the two climb
/// each other until they hit the muxer's ceiling. The walk iterates to that fixed
/// point and stops there, and the run completes.
#[tokio::test]
async fn mutually_constraining_pads_converge() {
    let (log0, log1, sink_log) = (log(), log(), log());
    let g = tee_mux_graph(
        GraphNode::muxer(FloorMux::climbing(2, nv12(8, 8), 128, 512)),
        Some(nv12(16, 16)),
        nv12(8, 8),
        &log0,
        &log1,
        &sink_log,
    );

    run_graph(g, &NullClock, 4).await.expect("the walk settles");
    assert_eq!(
        *log0.lock().unwrap(),
        vec![128, 256, 384, 512],
        "the changed pad climbs a step per round and stops at the ceiling"
    );
    assert_eq!(
        *log1.lock().unwrap(),
        vec![128, 384, 512],
        "the unchanged pad is pulled up with it, joining once the pool moved"
    );
    assert_eq!(
        *sink_log.lock().unwrap(),
        vec![256, 384, 512],
        "the consumer sees each settled pool, ending at the fixed point"
    );
}

/// A pair that cannot settle (each side always demands one more byte than the
/// other) is stopped by the round bound and fails the run loud, rather than
/// re-cascading forever.
#[tokio::test]
async fn runaway_pair_fails_loud() {
    let (log0, log1, sink_log) = (log(), log(), log());
    let g = tee_mux_graph(
        GraphNode::muxer(RunawayMux {
            inputs: 2,
            output: nv12(8, 8),
            pad_caps: vec![None; 2],
            floor: 0,
        }),
        Some(nv12(16, 16)),
        nv12(8, 8),
        &log0,
        &log1,
        &sink_log,
    );

    assert_eq!(
        run_graph(g, &NullClock, 4).await.err(),
        Some(G2gError::AllocationConflict),
        "a non-converging boundary is a real allocation conflict"
    );
}

/// When the re-derived output pool imposes nothing on the inputs, the unchanged
/// pad keeps the allocation it already has: the consumer still learns the new
/// output pool, but only the changed pad's branch is re-cascaded.
#[tokio::test]
async fn unchanged_pad_untouched_when_output_imposes_nothing() {
    let (log0, log1, sink_log) = (log(), log(), log());
    let g = tee_mux_graph(
        GraphNode::muxer(FloorMux::new(2, nv12(8, 8), false)),
        Some(nv12(16, 16)),
        nv12(8, 8),
        &log0,
        &log1,
        &sink_log,
    );

    run_graph(g, &NullClock, 4)
        .await
        .expect("the walk through the muxer runs");
    assert_eq!(*log0.lock().unwrap(), vec![8 * 8, 16 * 16]);
    assert_eq!(
        *log1.lock().unwrap(),
        vec![8 * 8],
        "nothing constrains this pad, so it keeps its startup allocation"
    );
    assert_eq!(
        *sink_log.lock().unwrap(),
        vec![16 * 16],
        "the output pool still moved and reached the consumer"
    );
}

/// No mid-stream change: only the startup cascade configures the branches and the
/// output pool is never re-derived, so nothing reaches the consumer. Proves the
/// second entry in the tests above is the walk, not a duplicate startup
/// configure.
#[tokio::test]
async fn no_change_leaves_every_pad_at_its_startup_proposal() {
    let (log0, log1, sink_log) = (log(), log(), log());
    let g = tee_mux_graph(
        GraphNode::muxer(FloorMux::new(2, nv12(8, 8), true)),
        None,
        nv12(8, 8),
        &log0,
        &log1,
        &sink_log,
    );

    let stats = run_graph(g, &NullClock, 4)
        .await
        .expect("static graph runs");
    assert_eq!(*log0.lock().unwrap(), vec![8 * 8]);
    assert_eq!(*log1.lock().unwrap(), vec![8 * 8]);
    assert!(
        sink_log.lock().unwrap().is_empty(),
        "no output pool change to push downstream"
    );
    assert_eq!(
        stats.coordinator_events, 0,
        "no reports without a mid-stream change"
    );
}
