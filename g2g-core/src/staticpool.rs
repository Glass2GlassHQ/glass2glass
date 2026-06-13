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

use core::cell::RefCell;
use core::future::Future;
use core::ops::{Deref, DerefMut};
use core::pin::Pin;
use core::task::{Context, Poll, Waker};

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
}
