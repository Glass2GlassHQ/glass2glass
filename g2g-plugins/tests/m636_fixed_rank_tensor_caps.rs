//! M636: `TensorShape` is a fixed-rank inline array (no heap), so
//! `Caps::Tensor` is part of the no-alloc MCU subset like every other caps
//! kind (closing the carve-out the M623 `alloc` feature seam left open).
//!
//! One test fn, not several: the zero-alloc section reads a process-global
//! allocation counter, so nothing else in this binary may allocate
//! concurrently. The link-time complement is `examples/g2g-noalloc`, whose
//! pipeline now negotiates tensor caps on its transform link and still links
//! for bare Cortex-M with no allocator and no panic machinery.

use core::sync::atomic::{AtomicUsize, Ordering};
use std::alloc::{GlobalAlloc, Layout, System};

use g2g_core::{Caps, TensorDType, TensorLayout, TensorShape, MAX_TENSOR_RANK};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

/// Wraps the system allocator, counting every allocating call (alloc + realloc).
struct Counting;

// SAFETY: every method forwards to `System` unchanged; the only added work is a
// relaxed counter increment. All `GlobalAlloc` contracts are the system's.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `layout` is forwarded unchanged from our caller to System.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` / `layout` came from this allocator's `alloc` (forwarded to
        // System), so System may free them.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `ptr` / `layout` came from a prior `alloc`; `new_size` forwarded.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn tensor(dtype: TensorDType, shape: TensorShape) -> Caps {
    Caps::Tensor { dtype, shape, layout: TensorLayout::Nchw }
}

#[test]
fn fixed_rank_tensor_caps() {
    // -- shape API --
    let s = TensorShape::new([1, 3, 224, 224]);
    assert_eq!(s.dims(), &[1, 3, 224, 224]);
    assert_eq!(s.elements(), 3 * 224 * 224);
    assert_eq!(TensorShape::from_slice(&[1, 3, 224, 224]), Some(s));
    // rank bounds: empty and > MAX_TENSOR_RANK are rejected, the max fits
    assert_eq!(TensorShape::from_slice(&[]), None);
    assert_eq!(TensorShape::from_slice(&[1; MAX_TENSOR_RANK + 1]), None);
    let full = TensorShape::from_slice(&[2; MAX_TENSOR_RANK]).expect("max rank fits");
    assert_eq!(full.dims().len(), MAX_TENSOR_RANK);
    // rank is part of equality even when the used dims match as a prefix
    assert_ne!(TensorShape::new([1, 3]), TensorShape::new([1, 3, 1]));
    // in-place edit keeps rank (the batcher's batch-dim rewrite)
    let mut batched = s;
    batched.dims_mut()[0] = 8;
    assert_eq!(batched.dims(), &[8, 3, 224, 224]);
    // overflow saturates instead of wrapping or panicking
    let huge = TensorShape::new([u32::MAX, u32::MAX, u32::MAX]);
    assert_eq!(huge.elements(), usize::MAX);
    // Debug prints like the old tuple struct (used by Caps::to_gst_string)
    assert_eq!(format!("{s:?}"), "TensorShape([1, 3, 224, 224])");

    // -- caps algebra --
    let a = tensor(TensorDType::F32, s);
    assert!(a.is_raw_media());
    assert!(a.is_fixed());
    assert_eq!(a.intersect(&a), Ok(a.clone()));
    assert_eq!(a.fixate(), Ok(a.clone()));
    let b = tensor(TensorDType::F32, TensorShape::new([1, 3, 224]));
    assert!(a.intersect(&b).is_err(), "shape mismatch must not intersect");
    let c = tensor(TensorDType::U8, s);
    assert!(a.intersect(&c).is_err(), "dtype mismatch must not intersect");

    // -- zero-alloc: the negotiation path allocates nothing --
    // (`Caps` has no heap fields; intersect/fixate of tensor caps clones by
    // value. This is the runtime complement of the g2g-noalloc link proof.)
    let before = ALLOCS.load(Ordering::Relaxed);
    for _ in 0..10_000 {
        let s = TensorShape::new([1, 3, 64, 64]);
        let t = tensor(TensorDType::U8, s);
        let got = t.intersect(&t).expect("intersects");
        let fixed = got.fixate().expect("fixates");
        assert!(std::hint::black_box(&fixed).is_raw_media());
        assert_eq!(s.elements(), 3 * 64 * 64);
    }
    assert_eq!(
        ALLOCS.load(Ordering::Relaxed) - before,
        0,
        "tensor caps negotiation must be heap-free"
    );
}
