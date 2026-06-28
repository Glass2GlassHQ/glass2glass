//! M344: mid-stream β allocation re-cascade across a muxer.
//!
//! M343 made a muxer's per-pad allocation demand cross the boundary at *startup*
//! negotiation. This closes the *mid-stream* gap: when one muxer input emits a
//! `CapsChanged`, the muxer re-derives that pad's allocation demand and
//! re-cascades it up only that pad's branch (`Recascade::target`), leaving the
//! other inputs untouched. The branch transform on the changed pad records both
//! the startup proposal and the mid-stream re-cascaded one; the unchanged pad's
//! transform records only the startup proposal.
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

fn nv12(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// A CUDA allocation sized from caps geometry, so a geometry change shows up as a
/// different proposal (the mid-stream re-cascade made visible).
fn geom_alloc(caps: &Caps) -> Option<AllocationParams> {
    match caps.dims()? {
        (Dim::Fixed(w), Dim::Fixed(h), _) => {
            Some(AllocationParams::cuda(*w as usize * *h as usize, 2, 64))
        }
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
    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
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

/// Pass-through transform recording every full `AllocationParams` it absorbs, so
/// the proposals that re-cascade into it (startup + per-pad mid-stream) are
/// observable.
struct RecordingTransform {
    log: Arc<Mutex<Vec<AllocationParams>>>,
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
        self.log.lock().unwrap().push(*params);
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

/// Interleave muxer proposing a per-pad allocation derived from that pad's caps
/// geometry, so its demand tracks a mid-stream geometry change.
struct GeomMux {
    inputs: usize,
    output: Caps,
}

impl MultiInputElement for GeomMux {
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

    fn propose_allocation_for_input(
        &self,
        _input: usize,
        caps: &Caps,
    ) -> Option<AllocationParams> {
        geom_alloc(caps)
    }

    fn configure_pipeline(
        &mut self,
        _input: usize,
        _absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
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

/// A mid-stream geometry change on muxer pad 0 re-cascades the muxer's
/// re-derived per-pad proposal up pad 0's branch alone: that transform records
/// the startup proposal (8x8) then the mid-stream one (16x16), while pad 1's
/// transform (no change) records only its startup proposal.
#[tokio::test]
async fn muxer_midstream_recascades_only_changed_pad() {
    let log0 = Arc::new(Mutex::new(Vec::new()));
    let log1 = Arc::new(Mutex::new(Vec::new()));

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
    let t0 = g.add_transform(GraphNode::element(RecordingTransform { log: Arc::clone(&log0) }));
    let t1 = g.add_transform(GraphNode::element(RecordingTransform { log: Arc::clone(&log1) }));
    let mux = g.add_muxer(GraphNode::muxer(GeomMux { inputs: 2, output: nv12(8, 8) }), 2);
    let sink = g.add_sink(GraphNode::element(AnySink));
    g.link(s0, t0).unwrap();
    g.link(s1, t1).unwrap();
    g.link(t0, mux.input(0)).unwrap();
    g.link(t1, mux.input(1)).unwrap();
    g.link(mux.output(), sink).unwrap();

    let stats = run_graph(g, &NullClock, 4).await.expect("muxer mid-stream re-cascade runs");
    assert_eq!(stats.frames_emitted, 7, "4 + 3 source frames");
    assert_eq!(
        *log0.lock().unwrap(),
        vec![AllocationParams::cuda(64, 2, 64), AllocationParams::cuda(256, 2, 64)],
        "pad 0's branch absorbed the startup proposal then the mid-stream re-cascade"
    );
    assert_eq!(
        *log1.lock().unwrap(),
        vec![AllocationParams::cuda(64, 2, 64)],
        "pad 1 never changed, so its branch only saw the startup proposal"
    );
    assert!(
        stats.coordinator_events >= 1,
        "the muxer reported the per-pad mid-stream re-cascade"
    );
}
