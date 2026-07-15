//! M347: `GraphTemplate` for graph re-run / replay. `run_graph` consumes its
//! elements, so a graph can be run only once; a `GraphTemplate` rebuilds a fresh
//! graph (fresh elements) per run, the foundation for seek-and-replay and retry.
//! Pure-fake elements (no hardware).

use core::future::Future;
use core::pin::Pin;
use std::boxed::Box;
use std::sync::{Arc, Mutex};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_graph, GraphNode, GraphTemplate, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, FrameTiming, G2gError, Graph,
    MemoryDomain, OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn nv12() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(8),
        height: Dim::Fixed(8),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Emits `count` frames, then EOS. Single-use: its run loop drains its budget, so
/// a second run on the *same* instance would yield nothing, which is what makes
/// the per-run rebuild observable.
struct CountingSource {
    count: u32,
}

impl SourceLoop for CountingSource {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(nv12()))
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for seq in 0..self.count as u64 {
                out.push(PipelinePacket::DataFrame(Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(
                        vec![seq as u8].into_boxed_slice(),
                    )),
                    timing: FrameTiming::default(),
                    sequence: seq,
                    meta: Default::default(),
                }))
                .await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.count as u64)
        })
    }
}

struct CountingSink {
    count: Arc<Mutex<u64>>,
}

impl AsyncElement for CountingSink {
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
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(_) = packet {
                *self.count.lock().unwrap() += 1;
            }
            Ok(())
        })
    }
}

/// One template, instantiated and run twice. Each run gets fresh elements, so
/// both runs deliver the full frame count, the replay the seek path relies on.
#[tokio::test]
async fn template_instantiates_a_fresh_graph_each_run() {
    let run_a = Arc::new(Mutex::new(0u64));
    let run_b = Arc::new(Mutex::new(0u64));

    // The build closure captures the per-run sink counter via a selector so each
    // instantiation wires a distinct sink, proving the rebuild is real.
    let counters = [Arc::clone(&run_a), Arc::clone(&run_b)];
    let next = Arc::new(Mutex::new(0usize));
    let template = GraphTemplate::new(move || {
        let idx = {
            let mut n = next.lock().unwrap();
            let i = *n;
            *n += 1;
            i
        };
        let mut g: Graph<GraphNode> = Graph::new();
        let src = g.add_source(GraphNode::source(CountingSource { count: 3 }));
        let sink = g.add_sink(GraphNode::element(CountingSink {
            count: Arc::clone(&counters[idx]),
        }));
        g.link(src, sink).unwrap();
        g
    });

    let stats_a = run_graph(template.instantiate(), &NullClock, 4).await.expect("run A");
    let stats_b = run_graph(template.instantiate(), &NullClock, 4).await.expect("run B");

    assert_eq!(stats_a.frames_consumed, 3);
    assert_eq!(stats_b.frames_consumed, 3, "the second run got fresh elements, not an exhausted graph");
    assert_eq!(*run_a.lock().unwrap(), 3);
    assert_eq!(*run_b.lock().unwrap(), 3);
}
