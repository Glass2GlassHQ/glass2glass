//! Buffer pools.
//!
//! M4 provides the `Arc`-recycled variant for `alloc` targets. A buffer is
//! checked out of the pool with `try_acquire` and is returned to the pool
//! automatically when the resulting `PooledBuffer` is dropped. The pool is
//! cheaply cloneable: each clone is an `Arc` bump, and clones share the same
//! underlying free list.
//!
//! Future milestones will add:
//! - Compile-time sized `BufferPool<T, const N: usize>` for strict no-heap
//!   RTOS targets that cannot tolerate `alloc`.
//! - Async `acquire().await` that suspends until a buffer is available
//!   (today's `acquire()` returns `PoolExhausted` on contention).

#![cfg(feature = "runtime")]

use core::future::Future;
use core::ops::{Deref, DerefMut};
use core::pin::Pin;
use core::task::{Context, Poll, Waker};

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::sync::Arc;
use alloc::vec::Vec;

use spin::Mutex;

use crate::error::G2gError;

#[derive(Debug)]
pub struct BufferPool<T> {
    inner: Arc<PoolInner<T>>,
}

#[derive(Debug)]
struct PoolInner<T> {
    state: Mutex<PoolState<T>>,
    capacity: usize,
}

#[derive(Debug)]
struct PoolState<T> {
    free: Vec<T>,
    waiters: VecDeque<Waker>,
}

impl<T> BufferPool<T> {
    /// Build a pool from a pre-allocated set of buffers. The number of
    /// buffers fixes the pool's capacity; the pool never grows.
    pub fn from_buffers(buffers: Vec<T>) -> Self {
        let capacity = buffers.len();
        Self {
            inner: Arc::new(PoolInner {
                state: Mutex::new(PoolState {
                    free: buffers,
                    waiters: VecDeque::new(),
                }),
                capacity,
            }),
        }
    }

    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    /// Number of buffers currently available for acquisition.
    pub fn available(&self) -> usize {
        self.inner.state.lock().free.len()
    }

    /// Number of buffers currently checked out (capacity − available).
    pub fn outstanding(&self) -> usize {
        self.inner.capacity - self.available()
    }

    /// Try to acquire one buffer. Returns `None` if the pool is exhausted.
    pub fn try_acquire(&self) -> Option<PooledBuffer<T>> {
        let value = self.inner.state.lock().free.pop()?;
        Some(PooledBuffer {
            value: Some(value),
            pool: self.inner.clone(),
        })
    }

    /// Sync convenience: acquire one buffer, or fail with `PoolExhausted`.
    /// Prefer [`Self::acquire`] inside async element loops — it awaits
    /// capacity instead of failing fast.
    pub fn try_acquire_or_err(&self) -> Result<PooledBuffer<T>, G2gError> {
        self.try_acquire().ok_or(G2gError::PoolExhausted)
    }

    /// Acquire one buffer, awaiting until one becomes available.
    pub fn acquire(&self) -> AcquireFuture<'_, T> {
        AcquireFuture { pool: self }
    }
}

#[allow(missing_debug_implementations)]
pub struct AcquireFuture<'a, T> {
    pool: &'a BufferPool<T>,
}

impl<'a, T> Future for AcquireFuture<'a, T> {
    type Output = PooledBuffer<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let inner = &self.pool.inner;
        let mut state = inner.state.lock();
        if let Some(v) = state.free.pop() {
            return Poll::Ready(PooledBuffer {
                value: Some(v),
                pool: inner.clone(),
            });
        }
        // Park this waker. Dedupe so a re-poll (a spurious wakeup, or this future
        // living in a `select!`) does not push a second entry for the same task;
        // any leftover entries are drained, not consumed one-at-a-time, on return.
        if !state.waiters.iter().any(|w| w.will_wake(cx.waker())) {
            state.waiters.push_back(cx.waker().clone());
        }
        Poll::Pending
    }
}

impl<T> Clone for BufferPool<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl BufferPool<Box<[u8]>> {
    /// Allocate a CPU byte pool of `count` buffers, each `bytes` long.
    pub fn new_byte_pool(count: usize, bytes: usize) -> Self {
        let mut buffers: Vec<Box<[u8]>> = Vec::with_capacity(count);
        for _ in 0..count {
            buffers.push(alloc::vec![0u8; bytes].into_boxed_slice());
        }
        Self::from_buffers(buffers)
    }
}

#[derive(Debug)]
pub struct PooledBuffer<T> {
    value: Option<T>,
    pool: Arc<PoolInner<T>>,
}

impl<T> Deref for PooledBuffer<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.value
            .as_ref()
            .expect("PooledBuffer accessed after drop")
    }
}

impl<T> DerefMut for PooledBuffer<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.value
            .as_mut()
            .expect("PooledBuffer accessed after drop")
    }
}

impl<T: AsRef<[u8]>> AsRef<[u8]> for PooledBuffer<T> {
    fn as_ref(&self) -> &[u8] {
        self.deref().as_ref()
    }
}

impl<T: AsMut<[u8]>> AsMut<[u8]> for PooledBuffer<T> {
    fn as_mut(&mut self) -> &mut [u8] {
        self.deref_mut().as_mut()
    }
}

impl<T> Drop for PooledBuffer<T> {
    fn drop(&mut self) {
        if let Some(v) = self.value.take() {
            // Wake every parked acquirer so each re-polls and races for the freed
            // buffer (the loser re-parks). Draining all, rather than popping one,
            // means a waker left behind by a cancelled / dropped `AcquireFuture`
            // is a harmless no-op rather than a consumed wake that would starve a
            // still-live waiter. Wake outside the lock to avoid re-entry.
            let waiters = {
                let mut state = self.pool.state.lock();
                state.free.push(v);
                core::mem::take(&mut state.waiters)
            };
            for w in waiters {
                w.wake();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_capacity_and_available_match_on_construction() {
        let pool = BufferPool::new_byte_pool(4, 16);
        assert_eq!(pool.capacity(), 4);
        assert_eq!(pool.available(), 4);
        assert_eq!(pool.outstanding(), 0);
    }

    #[test]
    fn acquire_decrements_available_drop_returns_buffer() {
        let pool = BufferPool::new_byte_pool(4, 16);
        {
            let _a = pool.try_acquire_or_err().expect("a");
            let _b = pool.try_acquire_or_err().expect("b");
            assert_eq!(pool.available(), 2);
            assert_eq!(pool.outstanding(), 2);
        }
        assert_eq!(pool.available(), 4);
        assert_eq!(pool.outstanding(), 0);
    }

    #[test]
    fn exhausted_pool_returns_pool_exhausted() {
        let pool = BufferPool::new_byte_pool(2, 8);
        let _a = pool.try_acquire_or_err().unwrap();
        let _b = pool.try_acquire_or_err().unwrap();
        assert!(matches!(
            pool.try_acquire_or_err(),
            Err(G2gError::PoolExhausted)
        ));
        assert!(pool.try_acquire().is_none());
    }

    #[test]
    fn clones_share_the_same_free_list() {
        let pool = BufferPool::new_byte_pool(2, 8);
        let pool2 = pool.clone();
        let _a = pool.try_acquire_or_err().unwrap();
        assert_eq!(pool2.available(), 1);
    }

    #[test]
    fn buffer_size_matches_byte_pool_argument() {
        let pool = BufferPool::new_byte_pool(1, 32);
        let buf = pool.try_acquire_or_err().unwrap();
        assert_eq!(buf.as_ref().len(), 32);
    }

    /// A counting `Waker` so the async-acquire tests can observe wakes without
    /// an executor.
    struct CountWaker(core::sync::atomic::AtomicUsize);
    impl alloc::task::Wake for CountWaker {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
        }
    }

    #[test]
    fn dropped_pending_acquirer_does_not_starve_a_live_waiter() {
        use core::task::{Context, Poll};

        let pool = BufferPool::new_byte_pool(1, 8);
        let held = pool.try_acquire_or_err().unwrap(); // pool now exhausted

        let live = Arc::new(CountWaker(core::sync::atomic::AtomicUsize::new(0)));
        let live_waker = live.clone().into();
        let mut live_cx = Context::from_waker(&live_waker);

        // A live acquirer parks, then a second acquirer parks and is *cancelled*
        // (dropped) while pending, leaving its waker behind.
        let mut live_fut = pool.acquire();
        assert!(matches!(
            core::pin::Pin::new(&mut live_fut).poll(&mut live_cx),
            Poll::Pending
        ));
        {
            let cancelled = Arc::new(CountWaker(core::sync::atomic::AtomicUsize::new(0)));
            let cancelled_waker = cancelled.clone().into();
            let mut cancelled_cx = Context::from_waker(&cancelled_waker);
            let mut cancelled_fut = pool.acquire();
            assert!(matches!(
                core::pin::Pin::new(&mut cancelled_fut).poll(&mut cancelled_cx),
                Poll::Pending
            ));
            // cancelled_fut dropped here with its waker still parked.
        }

        // Returning the one buffer must wake the live waiter (a one-at-a-time
        // pop could have consumed the wake on the cancelled future instead).
        drop(held);
        assert!(
            live.0.load(core::sync::atomic::Ordering::SeqCst) >= 1,
            "the live acquirer must be woken when the buffer returns"
        );
        assert!(matches!(
            core::pin::Pin::new(&mut live_fut).poll(&mut live_cx),
            Poll::Ready(_)
        ));
    }

    #[test]
    fn repolling_a_pending_acquirer_does_not_accumulate_wakers() {
        use core::task::{Context, Poll};

        let pool = BufferPool::new_byte_pool(1, 8);
        let _held = pool.try_acquire_or_err().unwrap();

        let w = Arc::new(CountWaker(core::sync::atomic::AtomicUsize::new(0)));
        let waker = w.clone().into();
        let mut cx = Context::from_waker(&waker);

        let mut fut = pool.acquire();
        for _ in 0..5 {
            assert!(matches!(
                core::pin::Pin::new(&mut fut).poll(&mut cx),
                Poll::Pending
            ));
        }
        assert_eq!(
            pool.inner.state.lock().waiters.len(),
            1,
            "re-polling the same future must not push duplicate wakers"
        );
    }
}
