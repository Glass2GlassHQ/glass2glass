//! M616 (honest boundary): the object-safe `OutputSink::push` returns
//! `Pin<Box<dyn Future>>`, so driving frames through a `dyn OutputSink` boxes one
//! future per frame, a control-plane heap allocation. This is the counterpart to
//! `m616_no_steady_state_alloc` (the zero-alloc *data* path): it pins the control
//! path's per-frame cost with the same counting allocator, so the zero-alloc claim
//! is scoped honestly (data path + a concrete non-dyn link) and the boxing cost
//! cannot creep without a test noticing.
//!
//! The frames themselves come from a `StaticLendRing` (zero-alloc), so the counted
//! allocations are the push futures, not frame buffers: the measured count is at
//! least one per frame.

use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::alloc::{GlobalAlloc, Layout, System};

use g2g_core::memory::MemoryDomain;
use g2g_core::{Frame, FrameTiming, G2gError, OutputSink, PipelinePacket, PushOutcome, StaticLendRing};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

struct Counting;

// SAFETY: forwards to `System` unchanged, only counting alloc / realloc first.
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

/// A sink that discards frames. Its `push` still returns a boxed future (the trait is
/// object-safe), so each call heap-allocates a box.
struct NullSink;

impl OutputSink for NullSink {
    fn push<'a>(
        &'a mut self,
        _packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async { Ok(PushOutcome::Accepted) })
    }
}

/// Push `frames` zero-alloc ring frames through the dyn sink, driving each push to
/// completion. The only per-frame heap traffic is the boxed push future.
fn push_frames(ring: &StaticLendRing<SLOTS, BYTES>, sink: &mut NullSink, frames: u64) {
    for i in 0..frames {
        let mut slot = ring.acquire().expect("a slot is free");
        for b in slot.buf_mut()[..PAYLOAD].iter_mut() {
            *b = i as u8;
        }
        // SAFETY: `ring` outlives the frame (the caller owns it for the loop).
        let payload = unsafe { slot.publish(PAYLOAD) };
        let frame = Frame::new(MemoryDomain::System(payload), FrameTiming::default(), i);
        let _ = embassy_futures::block_on(sink.push(PipelinePacket::DataFrame(frame)));
    }
}

#[test]
fn dyn_output_sink_push_boxes_a_future_per_frame() {
    let ring: StaticLendRing<SLOTS, BYTES> = StaticLendRing::new();
    let mut sink = NullSink;

    // Warm up outside the measured region.
    push_frames(&ring, &mut sink, 100);

    const N: u64 = 1_000;
    let before = ALLOCS.load(Ordering::Relaxed);
    push_frames(&ring, &mut sink, N);
    let allocs = ALLOCS.load(Ordering::Relaxed) - before;

    assert!(
        allocs >= N as usize,
        "the dyn OutputSink::push path allocated {allocs} times for {N} frames (expected >= one box each); \
         the zero-alloc contract is the data path, not the dyn control path"
    );
}
