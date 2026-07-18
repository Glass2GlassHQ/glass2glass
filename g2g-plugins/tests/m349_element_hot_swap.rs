//! M349: mid-stream element hot-swap, live under load. `ElementSlot` (M8) is
//! unit-tested in isolation; this proves a swap works *inside a running graph*:
//! a slot transform sits in `source -> slot -> sink`, and the element inside it
//! swaps itself out for a replacement after a threshold of frames, so the
//! remainder of the stream flows through the new element without draining or
//! rebuilding the pipeline. The swap is sequenced from inside the slotted
//! element's own `process` (the slot reads its contents at the start of each
//! call), so the split point is deterministic. Pure-fake elements (no hardware).

use core::future::Future;
use core::pin::Pin;
use std::boxed::Box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use g2g_core::element::{BoxFuture, DynAsyncElement};
use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_graph, GraphNode, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, ElementSlot, FrameTiming, G2gError,
    Graph, MemoryDomain, OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
    SwapHandle,
};

/// The (handle, replacement element) a slotted element holds to swap itself out
/// once, filled in after the slot is built.
type SwapCell = Arc<Mutex<Option<(SwapHandle, Box<dyn DynAsyncElement + Send>)>>>;

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

struct CountedSource {
    n: u64,
}

impl SourceLoop for CountedSource {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
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
            for seq in 0..self.n {
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
            Ok(self.n)
        })
    }
}

/// The replacement element: a pass-through transform that counts its frames.
struct ElementB {
    count: Arc<AtomicU64>,
}

impl DynAsyncElement for ElementB {
    fn intercept_caps(&self, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
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
        out: &'a mut dyn OutputSink,
    ) -> BoxFuture<'a, Result<(), G2gError>> {
        let count = self.count.clone();
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(f) => {
                    count.fetch_add(1, Ordering::SeqCst);
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
    fn propose_allocation(&self, _: &Caps) -> Option<g2g_core::AllocationParams> {
        None
    }
    fn configure_allocation(&mut self, _: &g2g_core::AllocationParams) {}
}

/// The initial element: pass-through + counter, but on its `threshold`-th frame
/// it swaps the slot to `ElementB` via a held `SwapHandle`, so the remaining
/// frames route to B. The (handle, B) pair is filled in after the slot is built
/// (the slot owns this element, so the handle can only be taken afterwards) and
/// taken exactly once.
struct ElementA {
    count: Arc<AtomicU64>,
    threshold: u64,
    swap: SwapCell,
}

impl DynAsyncElement for ElementA {
    fn intercept_caps(&self, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
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
        out: &'a mut dyn OutputSink,
    ) -> BoxFuture<'a, Result<(), G2gError>> {
        let count = self.count.clone();
        let threshold = self.threshold;
        let swap = self.swap.clone();
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(f) => {
                    let n = count.fetch_add(1, Ordering::SeqCst) + 1;
                    out.push(PipelinePacket::DataFrame(f)).await?;
                    if n == threshold {
                        // Swap the slot to B; the next slot.process sees B.
                        if let Some((handle, b)) = swap.lock().unwrap().take() {
                            handle.swap(b);
                        }
                    }
                }
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
    fn propose_allocation(&self, _: &Caps) -> Option<g2g_core::AllocationParams> {
        None
    }
    fn configure_allocation(&mut self, _: &g2g_core::AllocationParams) {}
}

#[tokio::test]
async fn element_hot_swaps_mid_stream_inside_a_running_graph() {
    let a_count = Arc::new(AtomicU64::new(0));
    let b_count = Arc::new(AtomicU64::new(0));
    let sink_count = Arc::new(Mutex::new(0u64));

    let swap_cell: SwapCell = Arc::new(Mutex::new(None));

    // A is built into the slot; it swaps to B after 3 frames.
    let a = ElementA {
        count: Arc::clone(&a_count),
        threshold: 3,
        swap: Arc::clone(&swap_cell),
    };
    let slot = ElementSlot::new(Box::new(a));
    // B is configured against the same caps before being installed (the slot
    // contract: a swapped-in element is not re-negotiated).
    let mut b = ElementB {
        count: Arc::clone(&b_count),
    };
    b.configure_pipeline(&nv12()).unwrap();
    *swap_cell.lock().unwrap() = Some((slot.handle(), Box::new(b)));

    // Counting sink.
    struct CountingSink {
        count: Arc<Mutex<u64>>,
    }
    impl AsyncElement for CountingSink {
        type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;
        fn intercept_caps(&self, c: &Caps) -> Result<Caps, G2gError> {
            Ok(c.clone())
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
            let count = self.count.clone();
            Box::pin(async move {
                if let PipelinePacket::DataFrame(_) = packet {
                    *count.lock().unwrap() += 1;
                }
                Ok(())
            })
        }
    }

    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(CountedSource { n: 5 }));
    let slot_node = g.add_transform(GraphNode::element(slot));
    let sink = g.add_sink(GraphNode::element(CountingSink {
        count: Arc::clone(&sink_count),
    }));
    g.link(src, slot_node).unwrap();
    g.link(slot_node, sink).unwrap();

    // link_capacity 1 keeps the slot processing in lock-step with the source, so
    // the swap on A's 3rd frame lands before frame 4 reaches the slot.
    let stats = run_graph(g, &NullClock, 1)
        .await
        .expect("hot-swap graph runs to EOS");

    assert_eq!(
        a_count.load(Ordering::SeqCst),
        3,
        "A handled the first 3 frames"
    );
    assert_eq!(
        b_count.load(Ordering::SeqCst),
        2,
        "B handled the remaining 2 after the swap"
    );
    assert_eq!(
        *sink_count.lock().unwrap(),
        5,
        "every frame reached the sink across the swap"
    );
    assert_eq!(stats.frames_consumed, 5);
}
