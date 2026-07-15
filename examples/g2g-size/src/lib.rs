//! Minimal embedded-slice footprint harness. The `#[no_mangle]` entry below
//! touches a representative no-alloc embedded path through g2g-core (caps
//! algebra, a `Frame`, the no-alloc `StaticLendRing` DMA capture ring, the
//! runtime bounded channel), `black_box`'d so dead-code elimination keeps it.
//! Build size-optimised for Cortex-M, then `size` the staticlib to read the
//! real `.text` footprint.
#![no_std]

use core::alloc::{GlobalAlloc, Layout};
use core::hint::black_box;

use g2g_core::runtime::bounded;
use g2g_core::{Caps, Dim, Frame, FrameTiming, MemoryDomain, Rate, RawVideoFormat, StaticLendRing};

// A non-allocating global allocator: this harness is built and measured, never
// run, so the allocator only needs to exist for the link. The code paths that
// would call it are still codegen'd (that is what we measure).
struct NullAlloc;
// SAFETY: never actually invoked (the harness is not executed); returning null is
// a valid GlobalAlloc response (allocation failure).
unsafe impl GlobalAlloc for NullAlloc {
    unsafe fn alloc(&self, _: Layout) -> *mut u8 {
        core::ptr::null_mut()
    }
    unsafe fn dealloc(&self, _: *mut u8, _: Layout) {}
}
#[global_allocator]
static ALLOC: NullAlloc = NullAlloc;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

/// Exercise a minimal embedded pipeline slice so the linker emits real code for
/// it. Returns a value derived from each path so nothing is optimised away.
#[no_mangle]
pub extern "C" fn g2g_min() -> u64 {
    // Caps algebra: intersect a rate-open cap with a fixed one (the negotiation
    // core narrows the rate).
    let a = Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Any,
    };
    let b = Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30 << 16),
    };
    let narrowed = black_box(&a).intersect(black_box(&b)).is_ok();

    // No-alloc capture ring: reserve a slot, fill it, lend it as a Frame.
    let ring: StaticLendRing<4, 64> = StaticLendRing::new();
    let mut acc = 0u64;
    if let Some(mut slot) = ring.acquire() {
        slot.buf_mut()[0] = 0xAB;
        // SAFETY: `ring` outlives this borrow (it lives to the end of the fn);
        // the lent slice is dropped before the ring.
        let payload = unsafe { slot.publish(8) };
        let frame = Frame {
            domain: MemoryDomain::System(payload),
            timing: FrameTiming::default(),
            sequence: 1,
            meta: Default::default(),
        };
        acc += frame.sequence;
        black_box(&frame.domain);
    }
    acc += ring.leased_count() as u64;

    // Runtime bounded channel: construct the inter-element link.
    let (tx, rx) = bounded::<u32>(4);
    black_box(&tx);
    black_box(&rx);

    acc + narrowed as u64
}
