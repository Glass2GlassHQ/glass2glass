//! M1965: a graph mutated for as long as it runs keeps a flat heap. Each cycle
//! splices a transform in and lifts it out, then adds a tee branch and drops
//! it, and the bytes still allocated after many cycles match those after a few.
#![cfg(all(feature = "std", feature = "runtime"))]

#[path = "common/count_sink.rs"]
mod count_sink;
use count_sink::CountSink;

use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicBool, AtomicIsize, AtomicUsize, Ordering};
use core::task::{Context, Poll};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{block_on, GraphMutator, GraphNode, Join2, RunBuilder, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, G2gError, Graph, OutputSink, PipelineClock,
    PipelinePacket, Rate, RawVideoFormat,
};

static LIVE_BYTES: AtomicIsize = AtomicIsize::new(0);

struct LiveBytes;

// SAFETY: forwards to `System` unchanged, only adding up the sizes first.
unsafe impl GlobalAlloc for LiveBytes {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LIVE_BYTES.fetch_add(layout.size() as isize, Ordering::Relaxed);
        // SAFETY: `layout` is the valid layout forwarded from our caller.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        // SAFETY: `ptr` / `layout` come from a prior `alloc` of this allocator.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LIVE_BYTES.fetch_add(
            new_size as isize - layout.size() as isize,
            Ordering::Relaxed,
        );
        // SAFETY: `ptr` / `layout` come from a prior `alloc`; `new_size` is the
        // caller's valid request, forwarded unchanged.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: LiveBytes = LiveBytes;

const LINK_CAPACITY: usize = 2;
const TEE_NAME: &str = "tee0";
const WARMUP_CYCLES: usize = 20;
const MEASURED_CYCLES: usize = 200;
// one buffer growing once fits under this, eight bytes a cycle do not
const MAX_GROWTH_BYTES: isize = 1024;

struct ZeroClock;

impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn i420() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::I420,
        width: Dim::Fixed(16),
        height: Dim::Fixed(16),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

/// Pushes frames until the driver raises `stop`, counting each one in `pushed`.
struct DrivenSource {
    pushed: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
}

impl SourceLoop for DrivenSource {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(i420()))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let mut sequence = 0;
            while !self.stop.load(Ordering::SeqCst) {
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
                    FrameTiming::default(),
                    sequence,
                );
                out.push(PipelinePacket::DataFrame(frame)).await?;
                self.pushed.fetch_add(1, Ordering::SeqCst);
                sequence += 1;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(sequence)
        })
    }
}

#[derive(Default)]
struct Passthrough;

impl AsyncElement for Passthrough {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
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
        Box::pin(async move {
            if !matches!(packet, PipelinePacket::Eos) {
                out.push(packet).await?;
            }
            Ok(())
        })
    }
}

/// Pending once, so the run gets polled between the driver's checks.
struct Yield(bool);

impl Future for Yield {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.0 {
            return Poll::Ready(());
        }
        self.0 = true;
        cx.waker().wake_by_ref();
        Poll::Pending
    }
}

async fn frames_flow(pushed: &AtomicUsize) {
    let from = pushed.load(Ordering::SeqCst);
    while pushed.load(Ordering::SeqCst) < from + LINK_CAPACITY + 1 {
        Yield(false).await;
    }
}

async fn mutation_cycle(mutator: &GraphMutator<'_>, pushed: &AtomicUsize) {
    let spliced = mutator
        .insert_after("src", Box::new(Passthrough))
        .await
        .expect("a transform splices below the source");
    frames_flow(pushed).await;
    mutator
        .remove(&spliced)
        .await
        .expect("the transform lifts back out");
    let branch = mutator
        .add_branch(TEE_NAME, vec![Box::new(CountSink::default())])
        .await
        .expect("the tee takes a branch");
    frames_flow(pushed).await;
    mutator
        .remove_branch(&branch[0])
        .await
        .expect("the branch drops");
    frames_flow(pushed).await;
}

#[test]
fn a_mutated_run_keeps_a_flat_heap() {
    let pushed = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let mut graph: Graph<GraphNode> = Graph::new();
    let src = graph.add_source(GraphNode::source(DrivenSource {
        pushed: Arc::clone(&pushed),
        stop: Arc::clone(&stop),
    }));
    graph.set_node_name(src, "src".into());
    let tee = graph.add_tee(1);
    let sink = graph.add_sink(GraphNode::element(CountSink::default()));
    graph.link(src, tee.input()).unwrap();
    graph.link(tee.out(0), sink).unwrap();

    let (mutator, run) = RunBuilder::new(graph, &ZeroClock, LINK_CAPACITY).run_mutable();
    let driver = async {
        for _ in 0..WARMUP_CYCLES {
            mutation_cycle(&mutator, &pushed).await;
        }
        let after_warmup = LIVE_BYTES.load(Ordering::Relaxed);
        for _ in 0..MEASURED_CYCLES {
            mutation_cycle(&mutator, &pushed).await;
        }
        let growth = LIVE_BYTES.load(Ordering::Relaxed) - after_warmup;
        stop.store(true, Ordering::SeqCst);
        growth
    };
    let (stats, growth) = block_on(Join2::new(run, driver));
    stats.expect("the run survives every mutation");

    assert!(
        growth < MAX_GROWTH_BYTES,
        "{MEASURED_CYCLES} mutation cycles left {growth} bytes allocated"
    );
}
