//! M1000: an arbitrary graph through `run_graph` is heap-free in steady state.
//! The two per-frame boxes the dyn layer used to impose are gone: `OutputSink`
//! is poll-based (a push through `&mut dyn OutputSink` builds a stack
//! `PushFuture`), and the arms are monomorphized over the element type via the
//! `drive_*_arm` hooks, so `process` runs as the element's own future type.
//! An element opts in by declaring a concrete (non-boxed) `ProcessFuture`, as
//! the ones here do; frames come from a `StaticLendRing`, so the data path is
//! the proven zero-alloc lend.
//!
//! Measured inside the run: the sink snapshots the global allocation counter
//! at a warmup frame and at the last frame, and the difference must be zero.
#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::alloc::{GlobalAlloc, Layout, System};

use g2g_core::element::PushFuture;
use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::runtime::{block_on, run_graph, GraphNodeRef, SourceLoop};
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

const SLOTS: usize = 4;
const BYTES: usize = 16;
const PAYLOAD: usize = 4;
const WARMUP: u64 = 100;
const FRAMES: u64 = 100_000;

/// Counter snapshot at the warmup frame, taken by the sink mid-run so setup
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
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
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

/// A push mapped to `process`'s `Result<(), _>`: the concrete future a
/// non-boxing pass-through transform returns.
struct ForwardFuture<'a> {
    push: PushFuture<'a, dyn OutputSink + 'a>,
}

impl Future for ForwardFuture<'_> {
    type Output = Result<(), G2gError>;

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<Self::Output> {
        // SAFETY: `push` is structurally pinned; nothing moves out of it.
        let push = unsafe { self.map_unchecked_mut(|s| &mut s.push) };
        push.poll(cx).map(|r| r.map(|_outcome| ()))
    }
}

/// Pass-through transform whose `ProcessFuture` is the concrete
/// [`ForwardFuture`], no heap anywhere.
struct PassThrough;

impl AsyncElement for PassThrough {
    type ProcessFuture<'a>
        = ForwardFuture<'a>
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
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        ForwardFuture {
            push: out.push(packet),
        }
    }
}

/// Terminal sink: drops each frame (returning its ring slot) and snapshots the
/// allocation counter at the warmup and final frames.
struct MarkingSink;

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
        if let PipelinePacket::DataFrame(frame) = &packet {
            if frame.sequence == WARMUP {
                AT_WARMUP.store(ALLOCS.load(Ordering::Relaxed), Ordering::Relaxed);
            }
            if frame.sequence == FRAMES - 1 {
                AT_END.store(ALLOCS.load(Ordering::Relaxed), Ordering::Relaxed);
            }
        }
        core::future::ready(Ok(()))
    }
}

#[test]
fn dyn_graph_steady_state_makes_zero_heap_allocations() {
    static RING: StaticLendRing<SLOTS, BYTES> = StaticLendRing::new();

    let mut graph: Graph<GraphNodeRef<'static>> = Graph::new();
    let src = graph.add_source(GraphNodeRef::source(RingSource { ring: &RING }));
    let mid = graph.add_transform(GraphNodeRef::element(PassThrough));
    let snk = graph.add_sink(GraphNodeRef::element(MarkingSink));
    graph.link(src, mid).unwrap();
    graph.link(mid, snk).unwrap();

    let stats = block_on(run_graph(graph, &ZeroClock, 2)).expect("run to EOS");
    assert_eq!(stats.frames_consumed, FRAMES);

    let at_warmup = AT_WARMUP.load(Ordering::Relaxed);
    let at_end = AT_END.load(Ordering::Relaxed);
    assert!(at_warmup > 0, "the warmup mark was taken");
    assert_eq!(
        at_end - at_warmup,
        0,
        "run_graph steady state allocated {} times between frame {WARMUP} and \
         frame {} (expected none: poll-based sinks + monomorphized arms)",
        at_end - at_warmup,
        FRAMES - 1
    );
}
