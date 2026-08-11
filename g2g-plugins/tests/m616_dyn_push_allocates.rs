//! M616 -> M1000: this test used to pin the honest boundary of the zero-alloc
//! claim, the one box `OutputSink::push` allocated per frame through a `dyn`
//! sink. The poll-based `OutputSink` (M1000) removed that box: `push` builds a
//! concrete `PushFuture` on the stack and `poll_push` is a plain dyn call. The
//! assertion is therefore inverted: pushing through `&mut dyn OutputSink` must
//! not allocate at all, so the control path now carries the same zero-alloc
//! contract as the data path and a reintroduced per-push box fails loudly here.
//!
//! The frames themselves come from a `StaticLendRing` (zero-alloc), so any
//! counted allocation would be control-plane traffic, not frame buffers.

use core::sync::atomic::{AtomicUsize, Ordering};
use std::alloc::{GlobalAlloc, Layout, System};

use g2g_core::memory::MemoryDomain;
use g2g_core::{
    Frame, FrameTiming, G2gError, OutputSink, PipelinePacket, PushOutcome, StaticLendRing,
};

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

/// A sink that discards frames through the poll form, so a push through the
/// trait object costs no heap.
struct NullSink;

impl OutputSink for NullSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        packet_slot.take();
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

/// Push `frames` zero-alloc ring frames through the sink as a `dyn OutputSink`
/// (the trait-object path the graph runner drives), completing each push.
fn push_frames(ring: &StaticLendRing<SLOTS, BYTES>, sink: &mut dyn OutputSink, frames: u64) {
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
fn dyn_output_sink_push_never_allocates() {
    let ring: StaticLendRing<SLOTS, BYTES> = StaticLendRing::new();
    let mut sink = NullSink;

    // Warm up outside the measured region.
    push_frames(&ring, &mut sink, 100);

    const N: u64 = 1_000;
    let before = ALLOCS.load(Ordering::Relaxed);
    push_frames(&ring, &mut sink, N);
    let allocs = ALLOCS.load(Ordering::Relaxed) - before;

    assert_eq!(
        allocs, 0,
        "the dyn OutputSink push path allocated {allocs} times for {N} frames; \
         the poll-based sink contract (M1000) is zero-alloc on the control path too"
    );
}
