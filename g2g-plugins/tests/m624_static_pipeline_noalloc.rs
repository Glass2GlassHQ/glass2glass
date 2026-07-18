//! M624: the static element model (`g2g_core::staticelem`, Phase 2 of the
//! alloc-optional core) runs a whole `source -> transform -> sink` pipeline with
//! zero heap allocations, proven with a counting global allocator.
//!
//! Where M616 proved the raw capture *data path* is zero-alloc and M620 proved a
//! hand-wired concrete chain is, this proves the generic *runner + trait API* is:
//! `run_source_transform_sink` over `async fn`-in-trait stages compiles to unboxed
//! futures (no `Pin<Box<dyn Future>>`, unlike the object-safe `OutputSink`), so
//! driving 100k frames through the runner allocates nothing. This is the runtime
//! complement to the link-time proof (`examples/g2g-noalloc`, which links the same
//! model for bare Cortex-M with no allocator at all).

use core::cell::Cell;
use core::future::Future;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::alloc::{GlobalAlloc, Layout, System};

use g2g_core::error::G2gError;
use g2g_core::memory::MemoryDomain;
use g2g_core::{
    run_source_transform_sink, Frame, FrameTiming, StaticLendRing, StaticSink, StaticSource,
    StaticTransform,
};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// Counting is scoped to the thread running the pipeline. The allocator is
    /// process-global, but libtest's main thread also allocates while the test
    /// runs (it blocks in `mpsc::recv_timeout` awaiting the result), and whether
    /// that lands inside the measured window is a scheduling race, so an
    /// unscoped counter attributes another thread's allocations to the pipeline.
    /// Const-init and `Drop`-free, so reading it from the allocator cannot
    /// allocate or recurse.
    static MEASURING: Cell<bool> = const { Cell::new(false) };
}

fn measuring() -> bool {
    MEASURING.with(Cell::get)
}

/// Wraps the system allocator, counting every allocating call (alloc + realloc)
/// made by the measuring thread.
struct Counting;

// SAFETY: every method forwards to `System` unchanged; the only added work is a
// thread-local read and a relaxed counter increment. All `GlobalAlloc` contracts
// are the system's.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if measuring() {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: `layout` is forwarded unchanged from our caller to System.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` / `layout` came from this allocator's `alloc` (forwarded to
        // System), so System may free them.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if measuring() {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: `ptr` / `layout` came from a prior `alloc`; `new_size` forwarded.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

const SLOTS: usize = 2;
const BYTES: usize = 16;

/// A capture source over a fixed ring: acquire a slot, write one byte, lend it.
struct RingSource<'r> {
    ring: &'r StaticLendRing<SLOTS, BYTES>,
    remaining: u64,
    seq: u64,
}
impl StaticSource for RingSource<'_> {
    async fn next(&mut self) -> Result<Option<Frame>, G2gError> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        let mut slot = self
            .ring
            .acquire()
            .expect("a slot is free (each frame drops before the next)");
        slot.buf_mut()[0] = self.seq as u8;
        // SAFETY: `ring` outlives every frame (the caller owns it for the whole run).
        let payload = unsafe { slot.publish(1) };
        let frame = Frame::new(
            MemoryDomain::System(payload),
            FrameTiming {
                pts_ns: self.seq,
                ..FrameTiming::default()
            },
            self.seq,
        );
        self.seq = self.seq.wrapping_add(1);
        Ok(Some(frame))
    }
}

/// Pass-through transform that reads the payload byte (real work), forwards it.
struct Touch;
impl StaticTransform for Touch {
    async fn process(&mut self, input: Frame) -> Result<Option<Frame>, G2gError> {
        if let MemoryDomain::System(s) = &input.domain {
            let _ = core::hint::black_box(s.as_slice()[0]);
        }
        Ok(Some(input))
    }
}

/// Folds each frame's first byte into a checksum.
struct SumSink {
    sum: u64,
}
impl StaticSink for SumSink {
    async fn consume(&mut self, frame: Frame) -> Result<(), G2gError> {
        if let MemoryDomain::System(s) = &frame.domain {
            self.sum = self.sum.wrapping_add(u64::from(s.as_slice()[0]));
        }
        Ok(())
    }
}

/// Drive the always-ready chain with g2g-core's safe single-poll executor
/// (the M634 de-dup of the local noop-waker copies; no executor, no heap).
fn block_on<F: Future>(fut: F) -> F::Output {
    g2g_core::drive_ready(fut).expect("the static chain never suspends")
}

/// Run `frames` through the static runner and return the sink checksum.
fn run(ring: &StaticLendRing<SLOTS, BYTES>, frames: u64) -> u64 {
    let source = RingSource {
        ring,
        remaining: frames,
        seq: 0,
    };
    let mut sink = SumSink { sum: 0 };
    block_on(run_source_transform_sink(source, Touch, &mut sink)).expect("pipeline runs");
    sink.sum
}

#[test]
fn static_runner_pipeline_makes_zero_heap_allocations() {
    let ring: StaticLendRing<SLOTS, BYTES> = StaticLendRing::new();

    // Warm up any one-time lazy init outside the measured region.
    let _ = run(&ring, 1_000);

    MEASURING.set(true);
    let acc = run(&ring, 100_000);
    MEASURING.set(false);
    let allocs = ALLOCS.load(Ordering::Relaxed);

    assert_eq!(
        allocs, 0,
        "the static source->transform->sink runner made {allocs} heap allocations over 100k frames"
    );
    // Use the checksum so the run cannot be elided.
    assert!(acc <= 100_000 * 255);
}
