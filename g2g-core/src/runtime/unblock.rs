//! Manual-unblock channel (M1166): the app -> source side-channel a
//! `fallbacksrc` restart wrapper waits on before each life of its source, so the
//! application decides when a restarted source's data reaches downstream. This
//! is gst `fallbacksrc`'s `manual-unblock` property plus its `unblock` action
//! signal; g2g has no signals, so the release is a shared handle the app
//! constructs and registers, the same app-holds / source-holds-a-clone shape as
//! [`GaplessController`](crate::runtime::GaplessController).
//!
//! One [`unblock`](UnblockHandle::unblock) releases one life. The handle re-arms
//! after each release, so an application that asked for manual unblocking
//! releases again after every restart.

use core::fmt;
use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use core::task::{Context, Poll, Waker};

extern crate std;

use alloc::sync::Arc;

use std::sync::Mutex;

#[derive(Default)]
struct UnblockInner {
    /// App -> source release, set by `unblock` and taken by the waiting life.
    released: AtomicBool,
    /// Releases a source has taken (monotonic), so the app can tell one life's
    /// release from the next.
    taken: AtomicU64,
    /// Waker a source parked in [`UnblockHandle::wait_release`] registered, woken
    /// by `unblock`. `None` when no source is parked.
    waker: Mutex<Option<Waker>>,
}

/// Cloneable manual-unblock channel. Every clone shares one release slot and one
/// taken count. The app holds one handle and registers a clone on the
/// [`Registry`](crate::runtime::Registry); the `fallbacksrc` restart wrapper
/// holds another and waits on it before each life.
#[derive(Clone, Default)]
pub struct UnblockHandle {
    inner: Arc<UnblockInner>,
}

impl fmt::Debug for UnblockHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UnblockHandle")
            .field("release_pending", &self.release_pending())
            .field("released_count", &self.released_count())
            .finish()
    }
}

impl UnblockHandle {
    /// A handle holding every life until the app releases it.
    pub fn new() -> Self {
        Self::default()
    }

    /// Application side: release the source waiting on this handle, and wake it.
    /// Releases exactly one life: the handle re-arms, so a source that restarts
    /// waits again. A second call before a source takes the first is a no-op.
    pub fn unblock(&self) {
        self.inner.released.store(true, Ordering::SeqCst);
        if let Some(waker) = self.inner.waker.lock().unwrap().take() {
            waker.wake();
        }
    }

    /// Whether a release is set and no source has taken it yet.
    pub fn release_pending(&self) -> bool {
        self.inner.released.load(Ordering::SeqCst)
    }

    /// Application side: how many lives this handle has released (monotonic), so
    /// the app can tell a taken release from an untaken one.
    pub fn released_count(&self) -> u64 {
        self.inner.taken.load(Ordering::SeqCst)
    }

    /// Source side: park until the app releases this life, then resolve. The
    /// release is consumed as it resolves, so the next life parks again.
    pub fn wait_release(&self) -> UnblockWait<'_> {
        UnblockWait { handle: self }
    }

    /// Consume a pending release, `false` when none is set.
    fn take_release(&self) -> bool {
        if self
            .inner
            .released
            .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return false;
        }
        self.inner.taken.fetch_add(1, Ordering::SeqCst);
        true
    }
}

/// Future returned by [`UnblockHandle::wait_release`]. Resolves when the app
/// calls [`unblock`](UnblockHandle::unblock), consuming that release.
#[derive(Debug)]
pub struct UnblockWait<'a> {
    handle: &'a UnblockHandle,
}

impl Future for UnblockWait<'_> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.handle.take_release() {
            return Poll::Ready(());
        }
        // Register before the final check so an `unblock` that lands in the gap
        // still wakes us (it either sees the waker, or we see its release).
        *self.handle.inner.waker.lock().unwrap() = Some(cx.waker().clone());
        if self.handle.take_release() {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noop_waker() -> Waker {
        use core::task::{RawWaker, RawWakerVTable};
        const VT: RawWakerVTable = RawWakerVTable::new(
            |_| RawWaker::new(core::ptr::null(), &VT),
            |_| {},
            |_| {},
            |_| {},
        );
        // SAFETY: the vtable's clone returns a valid RawWaker over the same
        // vtable, and wake/wake_by_ref/drop are no-ops on a null data pointer.
        unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &VT)) }
    }

    #[test]
    fn a_release_frees_one_wait_and_re_arms() {
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        let app = UnblockHandle::new();
        let source = app.clone();

        let mut first = core::pin::pin!(source.wait_release());
        assert!(first.as_mut().poll(&mut cx).is_pending(), "held");
        app.unblock();
        assert!(first.as_mut().poll(&mut cx).is_ready(), "released");
        assert_eq!(app.released_count(), 1);
        assert!(!app.release_pending(), "the release was consumed");

        let mut second = core::pin::pin!(source.wait_release());
        assert!(
            second.as_mut().poll(&mut cx).is_pending(),
            "the next life waits again"
        );
        app.unblock();
        assert!(second.as_mut().poll(&mut cx).is_ready());
        assert_eq!(app.released_count(), 2);
    }

    #[test]
    fn a_release_set_before_the_wait_resolves_it_immediately() {
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        let app = UnblockHandle::new();
        app.unblock();
        app.unblock(); // still one release: a second call before a take is a no-op
        let mut wait = core::pin::pin!(app.wait_release());
        assert!(wait.as_mut().poll(&mut cx).is_ready());
        let mut next = core::pin::pin!(app.wait_release());
        assert!(next.as_mut().poll(&mut cx).is_pending(), "one life only");
    }
}
