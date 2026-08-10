//! M1009: the fan-in / fan-out arms of `run_graph` are heap-free in steady
//! state, the way M1000 made the transform / sink arms. The demux arm and the
//! two muxer arms are monomorphized over their element via the `drive_*_arm`
//! hooks on `DynMultiOutputElement` / `DynMultiInputElement`, so `process` runs
//! as the element's own future type instead of a boxed one per packet, and the
//! multi-port pushes go through the poll-based `MultiOutputSink`.
//!
//! Measured inside each run: a sink snapshots the global allocation counter at a
//! warmup frame and at the last frame, and the difference must be zero.
#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::alloc::{GlobalAlloc, Layout, System};

use g2g_core::element::PushFuture;
use g2g_core::fanout::{MultiInputElement, MultiOutputElement, MultiOutputSink, PushToFuture};
use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::runtime::{block_on, run_graph, GraphNode, SourceLoop};
use g2g_core::{
    graph::Graph, AsyncElement, Caps, ConfigureOutcome, Dim, G2gError, MemoryDomain, OutputSink,
    PipelineClock, PipelinePacket, Rate, RawVideoFormat, StaticLendRing,
};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

struct Counting;

// SAFETY: forwards to `System` unchanged, only counting alloc / realloc first.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `layout` is the valid layout forwarded from our caller.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` / `layout` come from a prior `alloc` of this allocator.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `ptr` / `layout` come from a prior `alloc`; `new_size` is the
        // caller's valid request, forwarded unchanged.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

const SLOTS: usize = 16;
const BYTES: usize = 16;
const PAYLOAD: usize = 4;
const WARMUP: u64 = 100;
const FRAMES: u64 = 100_000;

/// Counter snapshot at the warmup frame, taken by a sink mid-run so setup
/// (channels, arms, negotiation) and teardown are excluded.
static AT_WARMUP: AtomicUsize = AtomicUsize::new(0);
/// Counter snapshot at the last frame.
static AT_END: AtomicUsize = AtomicUsize::new(0);

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(2),
        height: Dim::Fixed(2),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
    }
}

/// Emits `FRAMES` ring-lent frames. The one box is its run future (per run,
/// not per frame); each push is an unboxed `PushFuture`.
struct RingSource {
    ring: &'static StaticLendRing<SLOTS, BYTES>,
}

impl SourceLoop for RingSource {
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
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for seq in 0..FRAMES {
                let mut slot = self.ring.acquire().expect("a slot is free");
                for b in slot.buf_mut()[..PAYLOAD].iter_mut() {
                    *b = seq as u8;
                }
                // SAFETY: the ring is 'static, so it outlives every frame.
                let payload = unsafe { slot.publish(PAYLOAD) };
                let frame = Frame::new(MemoryDomain::System(payload), FrameTiming::default(), seq);
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(FRAMES)
        })
    }
}

/// A port push mapped to `process`'s `Result<(), _>`, or nothing at all for the
/// packets a fan-out element must swallow (the arm owns the per-branch `Eos`).
/// The concrete future a non-boxing demux returns.
struct RouteFuture<'a> {
    push: Option<PushToFuture<'a, dyn MultiOutputSink + 'a>>,
}

impl Future for RouteFuture<'_> {
    type Output = Result<(), G2gError>;

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<Self::Output> {
        // SAFETY: the push is polled in place; nothing moves out of it.
        let Some(push) = unsafe { self.get_unchecked_mut() }.push.as_mut() else {
            return core::task::Poll::Ready(Ok(()));
        };
        // SAFETY: `push` is structurally pinned inside this future.
        let push = unsafe { Pin::new_unchecked(push) };
        push.poll(cx).map(|r| r.map(|_outcome| ()))
    }
}

/// The single-output analog of [`RouteFuture`], for the muxer element.
struct MergeFuture<'a> {
    push: Option<PushFuture<'a, dyn OutputSink + 'a>>,
}

impl Future for MergeFuture<'_> {
    type Output = Result<(), G2gError>;

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<Self::Output> {
        // SAFETY: the push is polled in place; nothing moves out of it.
        let Some(push) = unsafe { self.get_unchecked_mut() }.push.as_mut() else {
            return core::task::Poll::Ready(Ok(()));
        };
        // SAFETY: `push` is structurally pinned inside this future.
        let push = unsafe { Pin::new_unchecked(push) };
        push.poll(cx).map(|r| r.map(|_outcome| ()))
    }
}

/// Two-port demux alternating by frame sequence, with [`RouteFuture`] as its
/// `ProcessFuture`, so no heap anywhere on the packet path.
struct AlternatingDemux;

impl MultiOutputElement for AlternatingDemux {
    type ProcessFuture<'a>
        = RouteFuture<'a>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn port_output_caps(&self, _port: usize) -> Option<Caps> {
        Some(caps())
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::ProcessFuture<'a> {
        let port = match &packet {
            // the arm closes each branch with its own Eos
            PipelinePacket::Eos => return RouteFuture { push: None },
            PipelinePacket::DataFrame(frame) => (frame.sequence % 2) as usize,
            _ => 0,
        };
        RouteFuture {
            push: Some(out.push_to(port, packet)),
        }
    }
}

/// Two-input muxer forwarding every input packet to the merged output, with
/// [`MergeFuture`] as its `ProcessFuture`.
struct ForwardingMux;

impl MultiInputElement for ForwardingMux {
    type ProcessFuture<'a>
        = MergeFuture<'a>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        2
    }
    fn intercept_caps(&self, _input: usize, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }
    fn configure_pipeline(
        &mut self,
        _input: usize,
        _caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(caps())
    }
    fn process<'a>(
        &'a mut self,
        _input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        // the runner emits the merged Eos itself
        if matches!(packet, PipelinePacket::Eos) {
            return MergeFuture { push: None };
        }
        MergeFuture {
            push: Some(out.push(packet)),
        }
    }
}

/// Terminal sink: drops each frame (returning its ring slot) and snapshots the
/// allocation counter at the warmup and final frames. `mark_at` is the frame
/// count this instance marks the end on; a sink that never reaches it (the
/// demux's second branch) only drains.
struct MarkingSink {
    seen: u64,
    mark_at: u64,
}

impl MarkingSink {
    fn new(mark_at: u64) -> Self {
        Self { seen: 0, mark_at }
    }
}

impl AsyncElement for MarkingSink {
    type ProcessFuture<'a>
        = core::future::Ready<Result<(), G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        if matches!(packet, PipelinePacket::DataFrame(_)) && self.mark_at > 0 {
            self.seen += 1;
            if self.seen == WARMUP {
                AT_WARMUP.store(ALLOCS.load(Ordering::Relaxed), Ordering::Relaxed);
            }
            if self.seen == self.mark_at {
                AT_END.store(ALLOCS.load(Ordering::Relaxed), Ordering::Relaxed);
            }
        }
        core::future::ready(Ok(()))
    }
}

/// The allocations counted between the two marks, with both marks taken.
fn steady_state_allocations() -> usize {
    let at_warmup = AT_WARMUP.load(Ordering::Relaxed);
    let at_end = AT_END.load(Ordering::Relaxed);
    assert!(at_warmup > 0, "the warmup mark was taken");
    assert!(
        at_end >= at_warmup,
        "the end mark was taken after the warmup"
    );
    at_end - at_warmup
}

fn demux_graph_allocations() -> usize {
    static RING: StaticLendRing<SLOTS, BYTES> = StaticLendRing::new();

    let mut graph: Graph<GraphNode> = Graph::new();
    let src = graph.add_source(GraphNode::source(RingSource { ring: &RING }));
    let demux = graph.add_demux(GraphNode::demux(AlternatingDemux), 2);
    // Only the even branch marks; the counter is process-wide, so it still sees
    // whatever the odd branch allocated between the two marks.
    let even = graph.add_sink(GraphNode::element(MarkingSink::new(FRAMES / 2)));
    let odd = graph.add_sink(GraphNode::element(MarkingSink::new(0)));
    graph.link(src, demux.input()).unwrap();
    graph.link(demux.out(0), even).unwrap();
    graph.link(demux.out(1), odd).unwrap();

    let stats = block_on(run_graph(graph, &ZeroClock, 2)).expect("run to EOS");
    assert_eq!(stats.frames_consumed, FRAMES);
    steady_state_allocations()
}

fn muxer_graph_allocations() -> usize {
    static RING_A: StaticLendRing<SLOTS, BYTES> = StaticLendRing::new();
    static RING_B: StaticLendRing<SLOTS, BYTES> = StaticLendRing::new();

    let mut graph: Graph<GraphNode> = Graph::new();
    let a = graph.add_source(GraphNode::source(RingSource { ring: &RING_A }));
    let b = graph.add_source(GraphNode::source(RingSource { ring: &RING_B }));
    let mux = graph.add_muxer(GraphNode::muxer(ForwardingMux), 2);
    let sink = graph.add_sink(GraphNode::element(MarkingSink::new(2 * FRAMES)));
    graph.link(a, mux.input(0)).unwrap();
    graph.link(b, mux.input(1)).unwrap();
    graph.link(mux.output(), sink).unwrap();

    let stats = block_on(run_graph(graph, &ZeroClock, 2)).expect("run to EOS");
    assert_eq!(stats.frames_consumed, 2 * FRAMES);
    steady_state_allocations()
}

/// One test for both graphs: a second `#[test]` in this binary would have the
/// harness reporting its result (on another thread) inside the counted window.
#[test]
fn fan_arms_steady_state_make_zero_heap_allocations() {
    let demux = demux_graph_allocations();
    let muxer = muxer_graph_allocations();
    assert_eq!(
        demux,
        0,
        "the demux arm allocated {demux} times between frame {WARMUP} and frame {} \
         (expected none: poll-based multi-output sink + monomorphized arm)",
        FRAMES / 2
    );
    assert_eq!(
        muxer,
        0,
        "the muxer arm allocated {muxer} times between frame {WARMUP} and frame {} \
         (expected none: monomorphized arm + poll-based sender sink)",
        2 * FRAMES
    );
}
