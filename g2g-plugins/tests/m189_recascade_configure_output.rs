//! M189: a caps-driven transform re-resolves its output target on a mid-stream
//! `CapsChanged`, not only at startup. M185/M186 delivered `configure_output`
//! during startup negotiation (graph + linear-coordinator paths); this wires it
//! into the mid-stream re-cascade arms too, so a `videoscale` / `videoconvert`
//! fed by a downstream capsfilter retargets when the upstream caps shift.
//!
//! A `DerivedOutput` transform whose `configure_output` records the output caps
//! is driven by a scripted source that switches caps mid-stream. The recorded
//! log shows [startup, mid-stream], proving the re-cascade delivers it.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::boxed::Box;
use std::sync::{Arc, Mutex};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_graph, run_source_transform_sink, GraphNode, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError,
    Graph, MemoryDomain, OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

struct NullClock;
impl PipelineClock for NullClock {
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

fn frame(seq: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(vec![seq as u8].into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: seq,
        meta: Default::default(),
    })
}

/// Scripted source: `before` frames under `initial`, then a mid-stream
/// `CapsChanged(switch)`, then `after` frames, then EOS.
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

/// Caps-driven (auto) transform standing in for videoscale/videoconvert: a
/// passthrough `DerivedOutput` so the runner delivers it an output target, and a
/// `configure_output` that records every target it is handed. The recorded log
/// is the test's window onto when the target is (re-)resolved.
struct RecordingConvert {
    out_log: Arc<Mutex<Vec<Caps>>>,
}

impl AsyncElement for RecordingConvert {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| CapsSet::one(input.clone())))
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn configure_output(&mut self, output_caps: &Caps) -> Result<(), G2gError> {
        self.out_log.lock().unwrap().push(output_caps.clone());
        Ok(())
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

/// Wildcard sink: accepts any caps so the transform forwards greedily (the
/// mid-stream forward resolve defers to the incoming caps).
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

/// Linear-coordinator path (`run_source_transform_sink`): the transform's target
/// is configured at startup (initial caps) and again on the mid-stream switch.
#[tokio::test]
async fn linear_coordinator_recascades_configure_output() {
    let out_log = Arc::new(Mutex::new(Vec::new()));
    let mut source = ScriptedSource {
        initial: rgba(320, 240),
        switch: Some(rgba(640, 480)),
        before: 1,
        after: 1,
    };
    let mut transform = RecordingConvert {
        out_log: Arc::clone(&out_log),
    };
    let mut sink = AnySink;

    run_source_transform_sink(&mut source, &mut transform, &mut sink, &NullClock, 4)
        .await
        .expect("linear chain runs");

    assert_eq!(
        *out_log.lock().unwrap(),
        vec![rgba(320, 240), rgba(640, 480)],
        "configure_output fires at startup then again on the mid-stream re-cascade"
    );
}

/// Graph path (`run_graph`): same expectation through the DAG runner's transform
/// arm.
#[tokio::test]
async fn graph_runner_recascades_configure_output() {
    let out_log = Arc::new(Mutex::new(Vec::new()));
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(ScriptedSource {
        initial: rgba(320, 240),
        switch: Some(rgba(640, 480)),
        before: 1,
        after: 1,
    }));
    let t = g.add_transform(GraphNode::element(RecordingConvert {
        out_log: Arc::clone(&out_log),
    }));
    let snk = g.add_sink(GraphNode::element(AnySink));
    g.link(src, t).unwrap();
    g.link(t, snk).unwrap();

    run_graph(g, &NullClock, 4).await.expect("graph runs");

    assert_eq!(
        *out_log.lock().unwrap(),
        vec![rgba(320, 240), rgba(640, 480)],
        "graph transform arm re-resolves the output target on the mid-stream change"
    );
}

/// Control: with no mid-stream change, the target is configured exactly once (at
/// startup), proving the second entry above is the re-cascade, not a duplicate.
#[tokio::test]
async fn no_change_configures_output_once() {
    let out_log = Arc::new(Mutex::new(Vec::new()));
    let mut source = ScriptedSource {
        initial: rgba(320, 240),
        switch: None,
        before: 2,
        after: 0,
    };
    let mut transform = RecordingConvert {
        out_log: Arc::clone(&out_log),
    };
    let mut sink = AnySink;

    run_source_transform_sink(&mut source, &mut transform, &mut sink, &NullClock, 4)
        .await
        .expect("linear chain runs");

    assert_eq!(
        *out_log.lock().unwrap(),
        vec![rgba(320, 240)],
        "startup configure_output only; no re-cascade without a mid-stream change"
    );
}
