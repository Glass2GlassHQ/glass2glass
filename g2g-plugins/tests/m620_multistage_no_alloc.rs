//! M620: prove a multi-stage pipeline (source -> transform -> sink) is zero heap
//! allocation in steady state when the stages hand off over concrete (non-`dyn`)
//! calls, extending the single-stage data-path proof of `m616_no_steady_state_alloc`
//! to a whole (small) graph.
//!
//! M616 pinned the boundary: the object-safe `OutputSink::push` returns
//! `Pin<Box<dyn Future>>`, so the general dyn runner boxes one future per frame. The
//! zero-alloc contract is therefore the data path plus a *concrete* link. This test
//! is that concrete link: three stages wired by direct method calls over a shared
//! `StaticLendRing`, so a frame flows capture -> transform -> sink and its slot
//! recycles, with the counting allocator confirming the whole chain allocates
//! nothing across 100k frames. (A fully zero-alloc *dyn* runner, monomorphized with
//! unboxed `process` futures, is a larger effort deferred; the load-bearing MCU
//! claim, a heap-free data plane, is proven here and in M616.)

use core::sync::atomic::{AtomicUsize, Ordering};
use std::alloc::{GlobalAlloc, Layout, System};

use g2g_core::memory::MemoryDomain;
use g2g_core::{Frame, FrameTiming, StaticLendRing};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

struct Counting;

// SAFETY: forwards to `System` unchanged, only counting alloc / realloc first.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `layout` is the caller's valid layout, forwarded unchanged.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` / `layout` come from a prior `alloc` forwarded to System.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `ptr` / `layout` from a prior `alloc`; `new_size` the caller's.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

const SLOTS: usize = 2;
const BYTES: usize = 16;
const PAYLOAD: usize = 4;

/// Stage 1: a capture source over the fixed ring. Produces one frame per call by
/// filling the next free slot and lending it zero-copy. Returns `None` when no slot
/// is free (the ring is full), the genuine back-pressure.
struct RingSource<'r> {
    ring: &'r StaticLendRing<SLOTS, BYTES>,
    seq: u64,
}
impl RingSource<'_> {
    fn produce(&mut self) -> Option<Frame> {
        let mut slot = self.ring.acquire()?;
        for b in slot.buf_mut()[..PAYLOAD].iter_mut() {
            *b = self.seq as u8;
        }
        // SAFETY: the ring outlives the caller's loop, so the lent slice never dangles.
        let payload = unsafe { slot.publish(PAYLOAD) };
        let frame = Frame::new(
            MemoryDomain::System(payload),
            FrameTiming::default(),
            self.seq,
        );
        self.seq += 1;
        Some(frame)
    }
}

/// Stage 2: an inspecting transform. Reads the frame's first byte (folding it into a
/// running value) and forwards the frame unchanged, allocating nothing.
#[derive(Default)]
struct InspectTransform {
    fold: u64,
}
impl InspectTransform {
    fn apply(&mut self, frame: Frame) -> Frame {
        if let Some(s) = frame.domain.as_system_slice() {
            self.fold = self.fold.wrapping_add(u64::from(s[0]));
        }
        frame
    }
}

/// Stage 3: a sink. Consumes the frame (checksums it, then drops it, returning the
/// slot to the ring), allocating nothing.
#[derive(Default)]
struct SumSink {
    sum: u64,
}
impl SumSink {
    fn consume(&mut self, frame: Frame) {
        if let Some(s) = frame.domain.as_system_slice() {
            self.sum = self.sum.wrapping_add(u64::from(s[PAYLOAD - 1]));
        }
        // frame drops here: the slot returns to the ring.
    }
}

/// Drive the three concrete stages for `frames` iterations over one ring.
fn run_chain(ring: &StaticLendRing<SLOTS, BYTES>, frames: u64) -> (u64, u64) {
    let mut src = RingSource { ring, seq: 0 };
    let mut xform = InspectTransform::default();
    let mut sink = SumSink::default();
    for _ in 0..frames {
        let frame = src
            .produce()
            .expect("a slot is free (each frame drops before the next)");
        let frame = xform.apply(frame);
        sink.consume(frame);
    }
    (xform.fold, sink.sum)
}

#[test]
fn concrete_multistage_pipeline_makes_zero_heap_allocations() {
    let ring: StaticLendRing<SLOTS, BYTES> = StaticLendRing::new();

    // Warm up one-time lazy init outside the measured region.
    let _ = run_chain(&ring, 1_000);

    let before = ALLOCS.load(Ordering::Relaxed);
    let (fold, sum) = run_chain(&ring, 100_000);
    let allocs = ALLOCS.load(Ordering::Relaxed) - before;

    assert_eq!(
        allocs, 0,
        "source -> transform -> sink over 100k frames made {allocs} heap allocations"
    );
    // Use the folded values so the chain cannot be optimized away.
    assert!(fold <= 100_000 * 255 && sum <= 100_000 * 255);
}
