//! Strict no-heap buffer pool for `no_std` RTOS targets that cannot tolerate
//! `alloc`, the counterpart of [`crate::pool::BufferPool`] (which is
//! `Arc`/`Vec`-backed). Sized at construction with a fixed `[T; N]`; acquiring
//! moves a buffer out and the RAII handle returns it on drop. Pure `core`: no
//! `alloc`, no `Arc`, no OS.
//!
//! `!Sync` (a `RefCell` free list), which suits the single-core cooperative
//! Embassy executor: tasks share the pool by reference and never run in
//! parallel, so the borrows are always short and non-overlapping. The async
//! `acquire` parks a single waiter; a multi-consumer pool must poll
//! `try_acquire` instead.

use core::cell::{RefCell, UnsafeCell};
use core::ffi::c_void;
use core::fmt;
use core::future::Future;
use core::ops::{Deref, DerefMut};
use core::pin::Pin;
use core::sync::atomic::{AtomicBool, Ordering};
use core::task::{Context, Poll, Waker};

use crate::memory::SystemSlice;

#[derive(Debug)]
pub struct StaticBufferPool<T, const N: usize> {
    inner: RefCell<Inner<T, N>>,
}

#[derive(Debug)]
struct Inner<T, const N: usize> {
    slots: [Option<T>; N],
    waker: Option<Waker>,
}

impl<T, const N: usize> StaticBufferPool<T, N> {
    /// Build a pool from `N` pre-allocated buffers; capacity is fixed at `N`.
    pub fn new(buffers: [T; N]) -> Self {
        Self {
            inner: RefCell::new(Inner { slots: buffers.map(Some), waker: None }),
        }
    }

    /// Capacity (the const `N`).
    pub fn capacity(&self) -> usize {
        N
    }

    /// Buffers currently available for acquisition.
    pub fn available(&self) -> usize {
        self.inner.borrow().slots.iter().filter(|s| s.is_some()).count()
    }

    /// Buffers currently checked out (`capacity - available`).
    pub fn outstanding(&self) -> usize {
        N - self.available()
    }

    /// Try to check out one buffer; `None` if the pool is exhausted.
    pub fn try_acquire(&self) -> Option<StaticPooled<'_, T, N>> {
        let mut inner = self.inner.borrow_mut();
        for slot in inner.slots.iter_mut() {
            if let Some(value) = slot.take() {
                return Some(StaticPooled { pool: self, value: Some(value) });
            }
        }
        None
    }

    /// Acquire one buffer, awaiting until one is free. Parks a single waiter
    /// (the embedded single-consumer model); use [`Self::try_acquire`] for
    /// multi-consumer pools.
    pub fn acquire(&self) -> StaticAcquire<'_, T, N> {
        StaticAcquire { pool: self }
    }

    /// Return a buffer to the first free slot and wake the parked acquirer.
    fn release(&self, value: T) {
        // Take the waker out before releasing the borrow, then wake outside it
        // so a re-entrant acquire can't hit a double borrow.
        let waker = {
            let mut inner = self.inner.borrow_mut();
            for slot in inner.slots.iter_mut() {
                if slot.is_none() {
                    *slot = Some(value);
                    break;
                }
            }
            inner.waker.take()
        };
        if let Some(w) = waker {
            w.wake();
        }
    }
}

#[derive(Debug)]
pub struct StaticPooled<'a, T, const N: usize> {
    pool: &'a StaticBufferPool<T, N>,
    value: Option<T>,
}

impl<T, const N: usize> Deref for StaticPooled<'_, T, N> {
    type Target = T;
    fn deref(&self) -> &T {
        self.value.as_ref().expect("StaticPooled accessed after drop")
    }
}

impl<T, const N: usize> DerefMut for StaticPooled<'_, T, N> {
    fn deref_mut(&mut self) -> &mut T {
        self.value.as_mut().expect("StaticPooled accessed after drop")
    }
}

impl<T: AsRef<[u8]>, const N: usize> AsRef<[u8]> for StaticPooled<'_, T, N> {
    fn as_ref(&self) -> &[u8] {
        self.deref().as_ref()
    }
}

impl<T: AsMut<[u8]>, const N: usize> AsMut<[u8]> for StaticPooled<'_, T, N> {
    fn as_mut(&mut self) -> &mut [u8] {
        self.deref_mut().as_mut()
    }
}

impl<T, const N: usize> Drop for StaticPooled<'_, T, N> {
    fn drop(&mut self) {
        if let Some(v) = self.value.take() {
            self.pool.release(v);
        }
    }
}

#[allow(missing_debug_implementations)]
pub struct StaticAcquire<'a, T, const N: usize> {
    pool: &'a StaticBufferPool<T, N>,
}

impl<'a, T, const N: usize> Future for StaticAcquire<'a, T, N> {
    type Output = StaticPooled<'a, T, N>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.pool.try_acquire() {
            Some(buf) => Poll::Ready(buf),
            None => {
                self.pool.inner.borrow_mut().waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}

/// A fixed, heap-free ring of byte buffers that lends each slot to the pipeline
/// *zero-copy* as a [`SystemSlice`], the capture-side sibling of
/// [`StaticBufferPool`] (which moves an owned buffer out; this keeps the bytes in
/// place and lends a borrow). It models a DMA capture ring: `N` slots of `BYTES`
/// bytes live inline (no `alloc`), the producer fills the next free slot and
/// [`publish`](RingSlot::publish)es it as a frame that *borrows* the slot, and the
/// slot is reclaimed when that frame is dropped downstream (the lend's free
/// callback clears the lease). A slot is never reused while still lent, so the
/// producer stalls when every slot is in flight, the genuine ring back-pressure.
///
/// The borrow is runtime-guarded, not a Rust lifetime: a `PipelinePacket` crosses
/// the `OutputSink` / stack-channel boundary by value (`'static`), so the lent
/// slice is the `'static` foreign-lend ([`SystemSlice::from_foreign`]) with the
/// lease standing in for the borrow. `new()` is not `const`; place the ring in a
/// `StaticCell` (or a `static` via a const-init wrapper) on real hardware, or keep
/// it alive on the stack for the duration of a `block_on` pipeline.
pub struct StaticLendRing<const N: usize, const BYTES: usize> {
    slots: [UnsafeCell<[u8; BYTES]>; N],
    leased: [AtomicBool; N],
}

// SAFETY: interior mutability of `slots` is guarded by the per-slot `leased`
// flags. A slot is written only through the unique `RingSlot` that holds its
// lease (between acquire and publish) and is read-only once published until its
// lease clears, so there is never an aliasing `&`/`&mut` to the same slot. The
// flags are independent atomics written with plain store (no RMW), so a
// DMA-completion ISR clearing one slot's lease never races a store to another and
// the type builds on targets without atomic CAS (eg `thumbv6m`). Acquire's
// scan-then-set is not atomic, so the single-executor capture contract holds: one
// task *sets* leases; only *clears* (a frame drop, or an ISR) may come from
// elsewhere.
unsafe impl<const N: usize, const BYTES: usize> Sync for StaticLendRing<N, BYTES> {}

impl<const N: usize, const BYTES: usize> StaticLendRing<N, BYTES> {
    /// Build a ring of `N` zeroed `BYTES`-sized slots.
    pub fn new() -> Self {
        Self {
            slots: core::array::from_fn(|_| UnsafeCell::new([0u8; BYTES])),
            leased: core::array::from_fn(|_| AtomicBool::new(false)),
        }
    }

    /// Slot count (the const `N`).
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Per-slot byte capacity (the const `BYTES`).
    pub const fn slot_bytes(&self) -> usize {
        BYTES
    }

    /// Slots currently lent and not yet reclaimed.
    pub fn leased_count(&self) -> usize {
        self.leased.iter().filter(|f| f.load(Ordering::Acquire)).count()
    }

    /// Reserve a free slot for capture, or `None` if all `N` are still in flight
    /// (ring full: the producer must wait for a downstream drop). Fill the
    /// returned handle, then [`publish`](RingSlot::publish) it as a frame slice.
    pub fn acquire(&self) -> Option<RingSlot<'_, N, BYTES>> {
        for idx in 0..N {
            if !self.leased[idx].load(Ordering::Acquire) {
                self.leased[idx].store(true, Ordering::Release);
                return Some(RingSlot { ring: self, idx });
            }
        }
        None
    }

    /// True if `ptr` points inside one of this ring's slots, the zero-copy witness
    /// a test uses to prove a received frame aliases the ring (no copy).
    pub fn contains(&self, ptr: *const u8) -> bool {
        let p = ptr as usize;
        self.slots.iter().any(|s| {
            let base = s.get() as usize;
            p >= base && p < base + BYTES
        })
    }
}

impl<const N: usize, const BYTES: usize> Default for StaticLendRing<N, BYTES> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize, const BYTES: usize> fmt::Debug for StaticLendRing<N, BYTES> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StaticLendRing")
            .field("capacity", &N)
            .field("slot_bytes", &BYTES)
            .field("leased", &self.leased_count())
            .finish()
    }
}

/// An exclusive lease on one [`StaticLendRing`] slot: fill it via
/// [`buf_mut`](Self::buf_mut), then [`publish`](Self::publish) it as a zero-copy
/// frame slice. Dropping it without publishing releases the lease (the slot was
/// reserved for a capture that never happened).
pub struct RingSlot<'r, const N: usize, const BYTES: usize> {
    ring: &'r StaticLendRing<N, BYTES>,
    idx: usize,
}

impl<const N: usize, const BYTES: usize> RingSlot<'_, N, BYTES> {
    /// The slot's backing bytes, for the capture (DMA target / test fill) to write.
    pub fn buf_mut(&mut self) -> &mut [u8] {
        // SAFETY: this `RingSlot` is the unique holder of slot `idx`'s lease and the
        // slot is not yet published, so this is the only reference to those bytes.
        let arr: &mut [u8; BYTES] = unsafe { &mut *self.ring.slots[self.idx].get() };
        arr.as_mut_slice()
    }

    /// Publish the first `len` captured bytes as a zero-copy [`SystemSlice`] that
    /// borrows this slot; the slot is reclaimed for reuse when the returned slice
    /// (the frame carrying it) is dropped downstream.
    ///
    /// # Safety
    /// The ring must outlive the returned `SystemSlice` and any frame holding it.
    /// On a `static` / `StaticCell` ring this is automatic; a stack ring must be
    /// kept alive until the pipeline drains.
    pub unsafe fn publish(self, len: usize) -> SystemSlice {
        debug_assert!(len <= BYTES, "published len exceeds slot capacity");
        let ptr = self.ring.slots[self.idx].get() as *const u8;
        let flag = &self.ring.leased[self.idx] as *const AtomicBool as *mut c_void;
        // Hand lease-clearing from this handle's `Drop` to the lend's free callback,
        // so the slot stays leased until the published frame is dropped.
        core::mem::forget(self);
        // SAFETY: `ptr` covers `len <= BYTES` bytes in a slot valid for the ring's
        // lifetime (caller's contract); the slot is read-only while lent and is not
        // reused until `release_slot` clears its lease on the frame's drop.
        unsafe { SystemSlice::from_foreign(ptr, len, Some(release_slot), flag) }
    }
}

impl<const N: usize, const BYTES: usize> fmt::Debug for RingSlot<'_, N, BYTES> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RingSlot").field("idx", &self.idx).finish()
    }
}

impl<const N: usize, const BYTES: usize> Drop for RingSlot<'_, N, BYTES> {
    fn drop(&mut self) {
        // Acquired but never published: release the lease so the slot is reusable.
        self.ring.leased[self.idx].store(false, Ordering::Release);
    }
}

/// Free callback for a published [`RingSlot`]: clears the slot's lease flag so the
/// ring can hand it out again. `user` is the slot's `&AtomicBool` lease flag.
unsafe extern "C" fn release_slot(user: *mut c_void) {
    // SAFETY: `user` is the lease-flag pointer `RingSlot::publish` passed; it is
    // valid for the ring's lifetime (the publish contract) and only stored to here.
    unsafe { (*(user as *const AtomicBool)).store(false, Ordering::Release) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::task::{RawWaker, RawWakerVTable};

    fn noop_waker() -> Waker {
        fn clone(_: *const ()) -> RawWaker {
            RawWaker::new(core::ptr::null(), &VTABLE)
        }
        fn no_op(_: *const ()) {}
        static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, no_op, no_op, no_op);
        // SAFETY: every vtable fn is a no-op over a null data pointer and never
        // dereferences it, so the RawWaker contract holds trivially.
        unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &VTABLE)) }
    }

    fn poll_once<F: Future + Unpin>(fut: &mut F) -> Poll<F::Output> {
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        Pin::new(fut).poll(&mut cx)
    }

    #[test]
    fn capacity_and_available_match_on_construction() {
        let pool: StaticBufferPool<[u8; 4], 3> = StaticBufferPool::new([[0u8; 4]; 3]);
        assert_eq!(pool.capacity(), 3);
        assert_eq!(pool.available(), 3);
        assert_eq!(pool.outstanding(), 0);
    }

    #[test]
    fn acquire_decrements_available_drop_returns() {
        let pool: StaticBufferPool<[u8; 4], 3> = StaticBufferPool::new([[0u8; 4]; 3]);
        {
            let _a = pool.try_acquire().expect("a");
            let _b = pool.try_acquire().expect("b");
            assert_eq!(pool.available(), 1);
            assert_eq!(pool.outstanding(), 2);
        }
        assert_eq!(pool.available(), 3, "dropping handles returns buffers");
    }

    #[test]
    fn exhausted_pool_returns_none() {
        let pool: StaticBufferPool<u32, 2> = StaticBufferPool::new([0; 2]);
        let _a = pool.try_acquire().unwrap();
        let _b = pool.try_acquire().unwrap();
        assert!(pool.try_acquire().is_none());
    }

    #[test]
    fn handle_derefs_to_buffer_and_writes_through() {
        let pool: StaticBufferPool<[u8; 4], 1> = StaticBufferPool::new([[0u8; 4]; 1]);
        let mut buf = pool.try_acquire().unwrap();
        buf[0] = 0xAB;
        assert_eq!(buf.as_ref(), &[0xAB, 0, 0, 0]);
    }

    #[test]
    fn async_acquire_parks_then_resolves_when_a_buffer_is_freed() {
        let pool: StaticBufferPool<u32, 1> = StaticBufferPool::new([7; 1]);
        let held = pool.try_acquire().unwrap();
        // Pool exhausted: acquire parks.
        let mut fut = pool.acquire();
        assert!(matches!(poll_once(&mut fut), Poll::Pending));
        // Freeing the buffer wakes the acquirer; the next poll resolves.
        drop(held);
        let Poll::Ready(buf) = poll_once(&mut fut) else {
            panic!("acquire must resolve once a buffer is free");
        };
        assert_eq!(pool.available(), 0, "the resolved acquire holds the buffer");
        drop(buf);
        assert_eq!(pool.available(), 1, "dropping it returns the buffer");
    }

    // --- StaticLendRing (zero-copy DMA ring) ---

    #[test]
    fn lend_ring_publish_borrows_slot_and_drop_reclaims() {
        let ring: StaticLendRing<2, 8> = StaticLendRing::new();
        assert_eq!((ring.capacity(), ring.slot_bytes(), ring.leased_count()), (2, 8, 0));

        let mut slot = ring.acquire().expect("free slot");
        slot.buf_mut()[..3].copy_from_slice(&[1, 2, 3]);
        assert_eq!(ring.leased_count(), 1, "acquire leases the slot");
        // SAFETY: `ring` outlives `frame` (both drop at end of scope, frame first).
        let frame = unsafe { slot.publish(3) };
        assert_eq!(ring.leased_count(), 1, "publish keeps the lease until the frame drops");
        // Zero-copy witness: the published bytes alias the ring slot, not a copy.
        assert_eq!(frame.as_slice(), &[1, 2, 3]);
        assert!(ring.contains(frame.as_slice().as_ptr()), "frame bytes live in the ring");

        drop(frame);
        assert_eq!(ring.leased_count(), 0, "dropping the frame reclaims the slot");
    }

    #[test]
    fn lend_ring_acquired_but_unpublished_slot_is_released_on_drop() {
        let ring: StaticLendRing<1, 4> = StaticLendRing::new();
        {
            let _slot = ring.acquire().expect("free slot");
            assert!(ring.acquire().is_none(), "ring full while the lease is held");
        }
        assert_eq!(ring.leased_count(), 0, "dropping an unpublished lease frees the slot");
        assert!(ring.acquire().is_some(), "slot reusable again");
    }

    #[test]
    fn lend_ring_full_when_all_slots_in_flight_then_recycles() {
        let ring: StaticLendRing<2, 4> = StaticLendRing::new();
        // SAFETY: the ring outlives every published frame in this scope. (len 1 so
        // the slice pointer is the slot base, not the empty-slice sentinel.)
        let f0 = unsafe { ring.acquire().unwrap().publish(1) };
        let f1 = unsafe { ring.acquire().unwrap().publish(1) };
        let p0 = f0.as_slice().as_ptr();
        assert!(ring.acquire().is_none(), "both slots lent: ring is full (back-pressure)");

        drop(f0); // a downstream drop frees one slot
        let f2 = unsafe { ring.acquire().expect("slot freed by the drop").publish(1) };
        // The recycled frame reuses slot 0's physical buffer: no fresh allocation.
        assert_eq!(f2.as_slice().as_ptr(), p0, "the freed slot's buffer is recycled");
        drop(f1);
        drop(f2);
        assert_eq!(ring.leased_count(), 0);
    }
}
