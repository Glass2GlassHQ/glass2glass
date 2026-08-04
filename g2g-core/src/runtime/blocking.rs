//! Minimal park-based blocking executor, shared by the synchronous FFI front
//! ends (`g2g-capi`, `g2g-pyapi`) that need to drive one runtime future to
//! completion on the calling thread without pulling in a full async runtime.

extern crate std;

use core::future::Future;
use core::sync::atomic::{AtomicBool, Ordering};

/// Drive `fut` to completion on the calling thread, parking between polls.
///
/// The future's waker raises a flag and unparks this thread, so a cross-thread
/// waker (e.g. the runtime channel's producer waking a blocked `recv`) resumes
/// the loop. Use only for a single blocking call, not as a general executor.
pub fn block_on<F: Future>(fut: F) -> F::Output {
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake, Waker};

    /// The unpark token alone cannot carry the wake: any code running on this
    /// thread between a wake and our `park` may park itself and swallow the
    /// token. `std::sync::Once` does exactly that on every platform whose `Once`
    /// is queue-based rather than futex-based (macOS and the other non-Linux
    /// unixes), so one contended `OnceLock::get_or_init` inside a poll is enough
    /// to lose a wake. `woken` is ours alone, so no one else can consume it.
    struct ThreadWaker {
        thread: std::thread::Thread,
        woken: AtomicBool,
    }
    impl Wake for ThreadWaker {
        fn wake(self: Arc<Self>) {
            self.wake_by_ref();
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.woken.store(true, Ordering::SeqCst);
            self.thread.unpark();
        }
    }

    let state = Arc::new(ThreadWaker {
        thread: std::thread::current(),
        woken: AtomicBool::new(false),
    });
    let waker = Waker::from(state.clone());
    let mut cx = Context::from_waker(&waker);
    let mut fut = core::pin::pin!(fut);
    loop {
        // Cleared before the poll, so a wake raised during it is still seen.
        state.woken.store(false, Ordering::SeqCst);
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => {
                while !state.woken.swap(false, Ordering::SeqCst) {
                    std::thread::park();
                }
            }
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

    /// Wakes itself, then parks once, the way a contended `OnceLock` does on the
    /// platforms whose `Once` is queue-based: the park consumes the unpark token
    /// the wake just set.
    struct StealsTheParkToken {
        polls: u32,
    }

    impl Future for StealsTheParkToken {
        type Output = u32;
        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<u32> {
            self.polls += 1;
            if self.polls == 2 {
                return Poll::Ready(self.polls);
            }
            cx.waker().wake_by_ref();
            std::thread::park();
            Poll::Pending
        }
    }

    #[test]
    fn a_wake_survives_a_stolen_park_token() {
        // On its own thread with a timeout: a regression parks forever, and a
        // hung suite is worse diagnostics than a failed assertion.
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(block_on(StealsTheParkToken { polls: 0 }));
        });
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(10)),
            Ok(2),
            "block_on must not depend on the unpark token surviving the poll"
        );
    }
}
