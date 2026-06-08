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
        state.waiters.push_back(cx.waker().clone());
        Poll::Pending
    }
}

impl<T> Clone for BufferPool<T> {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
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
        self.value.as_ref().expect("PooledBuffer accessed after drop")
    }
}

impl<T> DerefMut for PooledBuffer<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.value.as_mut().expect("PooledBuffer accessed after drop")
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
            let mut state = self.pool.state.lock();
            state.free.push(v);
            if let Some(w) = state.waiters.pop_front() {
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
}
