//! M616: prove, with a counting global allocator, that the fixed-pool frame *data
//! path* performs zero heap allocations in steady state, the load-bearing claim for
//! an MCU / RTOS deployment (no heap after startup).
//!
//! The path exercised is the embedded capture -> frame -> consume -> drop hot loop
//! over a `StaticLendRing`: acquire a pre-allocated slot, fill it (the DMA-write
//! stand-in), lend it zero-copy as a `System` frame (`SystemSlice::from_foreign`, no
//! heap), read it, and drop it (the slot returns to the ring, no `free`). The ring's
//! buffers are inline (`no_std`, no `alloc`), so 100k frames recycle a fixed 2 slots
//! and allocate nothing.
//!
//! Honest boundary: this is the *data* path. The object-safe `OutputSink::push`
//! returns `Pin<Box<dyn Future>>` and so boxes one future per frame, a control-plane
//! allocation; that cost is pinned separately in `m616_dyn_push_allocates`. The
//! zero-alloc contract is the data path plus a concrete (non-dyn) link.

use core::sync::atomic::{AtomicUsize, Ordering};
use std::alloc::{GlobalAlloc, Layout, System};

use g2g_core::memory::MemoryDomain;
use g2g_core::{Frame, FrameTiming, StaticLendRing};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

/// Wraps the system allocator, counting every allocating call (alloc + realloc).
struct Counting;

// SAFETY: every method forwards to `System` unchanged; the only added work is a
// relaxed counter increment before allocating. Pointers, layouts, and semantics are
// exactly the system allocator's, so all `GlobalAlloc` contracts are upheld.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `layout` is the valid layout forwarded from our caller; System's
        // alloc contract is identical to ours, so forwarding upholds it.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` / `layout` come from a prior `alloc` of this same allocator
        // (which forwarded to System), so System may free them.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `ptr` / `layout` come from a prior `alloc`; `new_size` is the
        // caller's valid request, forwarded unchanged to System.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

const SLOTS: usize = 2;
const BYTES: usize = 16;
const PAYLOAD: usize = 4;

/// Run the fixed-ring capture -> frame -> consume -> drop hot path for `frames`
/// iterations, returning a checksum so the work is not optimized away. Allocates
/// nothing: the ring buffers are inline and each frame borrows a slot.
fn run_data_path(ring: &StaticLendRing<SLOTS, BYTES>, frames: u64) -> u64 {
    let mut acc = 0u64;
    for i in 0..frames {
        let mut slot = ring
            .acquire()
            .expect("a slot is free (each frame is dropped before the next)");
        for b in slot.buf_mut()[..PAYLOAD].iter_mut() {
            *b = i as u8;
        }
        // SAFETY: `ring` outlives every frame produced here (the caller owns it for
        // the whole loop), so the lent slice never dangles.
        let payload = unsafe { slot.publish(PAYLOAD) };
        let frame = Frame::new(MemoryDomain::System(payload), FrameTiming::default(), i);
        if let MemoryDomain::System(s) = &frame.domain {
            acc = acc.wrapping_add(u64::from(s.as_slice()[0]));
        }
        drop(frame); // the slot returns to the ring; nothing is freed to the heap
    }
    acc
}

#[test]
fn fixed_pool_frame_path_makes_zero_heap_allocations() {
    let ring: StaticLendRing<SLOTS, BYTES> = StaticLendRing::new();

    // Warm up any one-time lazy initialization outside the measured region.
    let _ = run_data_path(&ring, 1_000);

    let before = ALLOCS.load(Ordering::Relaxed);
    let acc = run_data_path(&ring, 100_000);
    let allocs = ALLOCS.load(Ordering::Relaxed) - before;

    assert_eq!(
        allocs, 0,
        "the fixed-pool frame data path made {allocs} heap allocations over 100k frames"
    );
    // Use the checksum so the loop cannot be elided (100k frames, first byte 0..=255).
    assert!(acc <= 100_000 * 255);
}
