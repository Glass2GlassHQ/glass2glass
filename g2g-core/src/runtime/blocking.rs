//! Minimal park-based blocking executor, shared by the synchronous FFI front
//! ends (`g2g-capi`, `g2g-pyapi`) that need to drive one runtime future to
//! completion on the calling thread without pulling in a full async runtime.

extern crate std;

use core::future::Future;

/// Drive `fut` to completion on the calling thread, parking between polls.
///
/// The future's waker unparks this thread, so a cross-thread waker (e.g. the
/// runtime channel's producer waking a blocked `recv`) resumes the loop. Use
/// only for a single blocking call, not as a general executor.
pub fn block_on<F: Future>(fut: F) -> F::Output {
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake, Waker};

    struct ThreadWaker(std::thread::Thread);
    impl Wake for ThreadWaker {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.unpark();
        }
    }

    let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    let mut fut = core::pin::pin!(fut);
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => std::thread::park(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::pin::Pin;
    use core::task::{Context, Poll};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn returns_a_ready_value_without_parking() {
        assert_eq!(block_on(async { 42u32 }), 42);
    }

    /// A future that pends until a spawned thread sets a flag and wakes the
    /// stored waker, exercising the cross-thread park/unpark path.
    struct WakeFromOtherThread {
        done: Arc<AtomicBool>,
        spawned: bool,
    }

    impl Future for WakeFromOtherThread {
        type Output = u32;
        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<u32> {
            if self.done.load(Ordering::SeqCst) {
                return Poll::Ready(7);
            }
            if !self.spawned {
                self.spawned = true;
                let done = self.done.clone();
                let waker = cx.waker().clone();
                std::thread::spawn(move || {
                    std::thread::sleep(Duration::from_millis(20));
                    done.store(true, Ordering::SeqCst);
                    waker.wake();
                });
            }
            Poll::Pending
        }
    }

    #[test]
    fn resumes_on_a_cross_thread_wake() {
        let fut = WakeFromOtherThread {
            done: Arc::new(AtomicBool::new(false)),
            spawned: false,
        };
        assert_eq!(block_on(fut), 7);
    }
}
