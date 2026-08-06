//! The link between a blocking audio device worker thread and the async element
//! that feeds it, shared by `AlsaSink` and `PulseSink`.
//!
//! An audio device is a blocking API, so these sinks run it on their own thread
//! and hand samples over a channel. Two things about that hand-off have to hold,
//! and neither did before this module existed:
//!
//! The runner's executor is cooperative and single-threaded, so **nothing on it
//! may block**. A `Sender::send` on a full bounded channel blocks, and a
//! `JoinHandle::join` blocks for as long as the device takes to play out what is
//! queued. A sink that joined at `Eos` stalled every other arm in the pipeline
//! for the length of the audio: a two-branch graph played its audio while the
//! video branch sat frozen, because the cheap audio branch reached `Eos` first
//! and then held the executor.
//!
//! And the queue has to be **bounded**. An unbounded one lets the element push
//! a whole decoded stream into memory as fast as it decodes, which is both a
//! large allocation on a long file and the reason the drain at `Eos` was long
//! enough to notice.
//!
//! So: a bounded queue whose full case yields to the executor instead of
//! blocking it, and an end-of-stream drain that polls the worker's own
//! completion flag, yielding between polls. `Eos` still completes only once the
//! audio has actually played out, which is what a sink owes the pipeline; it
//! just no longer owns the executor while that happens.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

use g2g_core::{G2gError, HardwareError};

/// Commands buffered between the element and its device thread. One command is
/// one buffer of samples, so this is a queue depth in buffers: enough that a
/// scheduling hiccup on the element side cannot underrun the device, small
/// enough that a long file does not accumulate in memory. At the ~21 ms buffers
/// an AC-3 or AAC stream decodes to, 64 is well over a second of slack.
const QUEUE_DEPTH: usize = 64;

/// Yield once to the executor. Deliberately not `tokio::task::yield_now`: the
/// audio sink features do not pull tokio, and this works on the cooperative
/// runner and a tokio runtime alike.
///
/// The waker fires immediately, so a retry loop built on this polls hot rather
/// than sleeping: waiting on a full queue or on playout costs CPU. That is the
/// price of a cooperative single-threaded executor with no timer primitive, and
/// it buys the thing that matters, that no other arm is starved. If the runtime
/// grows a cooperative sleep, the two wait loops below should back off with it.
struct YieldNow(bool);

impl Future for YieldNow {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.0 {
            return Poll::Ready(());
        }
        self.0 = true;
        cx.waker().wake_by_ref();
        Poll::Pending
    }
}

fn yield_now() -> YieldNow {
    YieldNow(false)
}

/// A device worker thread plus the bounded queue feeding it.
#[derive(Debug)]
pub(crate) struct WorkerLink<C: Send + 'static> {
    tx: Option<SyncSender<C>>,
    worker: Option<JoinHandle<()>>,
    /// Set by the worker as it returns, so the element can wait for playout
    /// without blocking on `join`.
    finished: Arc<AtomicBool>,
}

impl<C: Send + 'static> WorkerLink<C> {
    /// Spawn `body` on a named thread with a bounded command queue. `body` owns
    /// the device and drains the receiver; the completion flag is set for it.
    pub(crate) fn spawn<F>(name: &str, body: F) -> Result<Self, G2gError>
    where
        F: FnOnce(Receiver<C>) + Send + 'static,
    {
        let (tx, rx) = mpsc::sync_channel::<C>(QUEUE_DEPTH);
        let finished = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&finished);
        let worker = thread::Builder::new()
            .name(alloc::string::String::from(name))
            .spawn(move || {
                body(rx);
                flag.store(true, Ordering::Release);
            })
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        Ok(Self {
            tx: Some(tx),
            worker: Some(worker),
            finished,
        })
    }

    /// Hand one command to the worker, yielding while the queue is full rather
    /// than blocking the executor. This is where back-pressure from the device
    /// reaches the pipeline, which is what bounds the memory.
    pub(crate) async fn send(&self, mut cmd: C, err: G2gError) -> Result<(), G2gError> {
        loop {
            let Some(tx) = self.tx.as_ref() else {
                return Err(err);
            };
            match tx.try_send(cmd) {
                Ok(()) => return Ok(()),
                Err(TrySendError::Full(back)) => {
                    cmd = back;
                    yield_now().await;
                }
                Err(TrySendError::Disconnected(_)) => return Err(err),
            }
        }
    }

    /// End of stream: queue `shutdown`, then wait for the worker to finish
    /// playing out, yielding to the executor throughout. Returns once the audio
    /// has actually been played, so a downstream `Eos` still means "the sound
    /// finished", but the rest of the pipeline keeps running while it drains.
    pub(crate) async fn finish(&mut self, shutdown: C, err: G2gError) -> Result<(), G2gError> {
        // A disconnected worker is already gone; nothing left to drain.
        if self.send(shutdown, err.clone()).await.is_ok() {
            while !self.finished.load(Ordering::Acquire) {
                yield_now().await;
            }
        }
        self.tx = None;
        // The flag is set as the worker returns, so this join is immediate.
        if let Some(join) = self.worker.take() {
            let _ = join.join();
        }
        Ok(())
    }

    /// Tear down without waiting for playout: drop the queue so the worker sees
    /// a disconnected channel and returns, then reap it. For `Drop` and for a
    /// reconfigure, neither of which is a place to keep playing.
    pub(crate) fn abort(&mut self) {
        self.tx = None;
        if let Some(join) = self.worker.take() {
            let _ = join.join();
        }
    }

    /// Whether a worker is running (the element is configured).
    pub(crate) fn is_running(&self) -> bool {
        self.worker.is_some()
    }
}

impl<C: Send + 'static> Drop for WorkerLink<C> {
    fn drop(&mut self) {
        self.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::runtime::block_on;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    /// The device boundary is the only thing mocked: a worker that sleeps per
    /// buffer the way a real one blocks on playout.
    fn slow_worker(per_buffer: Duration, played: Arc<AtomicUsize>) -> WorkerLink<u8> {
        WorkerLink::spawn("test-audio", move |rx| {
            while let Ok(cmd) = rx.recv() {
                if cmd == 0 {
                    break; // shutdown
                }
                thread::sleep(per_buffer);
                played.fetch_add(1, Ordering::Release);
            }
        })
        .expect("spawn")
    }

    fn err() -> G2gError {
        G2gError::Hardware(HardwareError::Other)
    }

    /// Drive two futures on the one executor, so "did the other task run while
    /// the sink drained" is a question this test can actually ask.
    async fn zip<A: Future, B: Future>(a: A, b: B) -> (A::Output, B::Output) {
        let mut a = alloc::boxed::Box::pin(a);
        let mut b = alloc::boxed::Box::pin(b);
        let (mut ra, mut rb) = (None, None);
        core::future::poll_fn(move |cx| {
            if ra.is_none() {
                if let Poll::Ready(v) = a.as_mut().poll(cx) {
                    ra = Some(v);
                }
            }
            if rb.is_none() {
                if let Poll::Ready(v) = b.as_mut().poll(cx) {
                    rb = Some(v);
                }
            }
            match (ra.take(), rb.take()) {
                (Some(x), Some(y)) => Poll::Ready((x, y)),
                (x, y) => {
                    ra = x;
                    rb = y;
                    Poll::Pending
                }
            }
        })
        .await
    }

    /// The regression this module exists for: while one branch drains its audio
    /// at `Eos`, another task on the same executor keeps running. Before, the
    /// blocking `join` meant the counter below never advanced during the drain.
    #[test]
    fn a_draining_sink_does_not_starve_another_task() {
        let played = Arc::new(AtomicUsize::new(0));
        let mut link = slow_worker(Duration::from_millis(5), Arc::clone(&played));
        let ticks = Arc::new(AtomicUsize::new(0));
        let t = Arc::clone(&ticks);

        block_on(async move {
            for _ in 0..8u8 {
                link.send(1, err()).await.expect("queue a buffer");
            }
            // The other arm: a task interleaved with the drain on the same
            // single-threaded executor.
            let other = async {
                for _ in 0..200 {
                    t.fetch_add(1, Ordering::Release);
                    yield_now().await;
                }
            };
            let drain = async {
                link.finish(0, err()).await.expect("drain");
            };
            zip(other, drain).await;
        });

        assert_eq!(played.load(Ordering::Acquire), 8, "every buffer played out");
        assert!(
            ticks.load(Ordering::Acquire) > 0,
            "the other task ran while the sink drained"
        );
    }

    /// The queue is bounded: a producer that outruns the device is held by
    /// back-pressure instead of accumulating the whole stream in memory.
    #[test]
    fn the_queue_is_bounded_and_back_pressures() {
        let played = Arc::new(AtomicUsize::new(0));
        let mut link = slow_worker(Duration::from_millis(1), Arc::clone(&played));
        let sent = Arc::new(AtomicUsize::new(0));
        let s = Arc::clone(&sent);

        block_on(async move {
            // Far more buffers than the queue holds: the sends can only complete
            // as the worker consumes, so this cannot be a memory sink.
            for _ in 0..(QUEUE_DEPTH * 3) {
                link.send(1, err()).await.expect("queue");
                s.fetch_add(1, Ordering::Release);
            }
            link.finish(0, err()).await.expect("drain");
        });

        assert_eq!(sent.load(Ordering::Acquire), QUEUE_DEPTH * 3);
        assert_eq!(
            played.load(Ordering::Acquire),
            QUEUE_DEPTH * 3,
            "everything queued was played"
        );
    }

    /// `abort` is the teardown path: it does not wait for playout.
    #[test]
    fn abort_stops_without_draining() {
        let played = Arc::new(AtomicUsize::new(0));
        let mut link = slow_worker(Duration::from_millis(50), Arc::clone(&played));
        block_on(async {
            for _ in 0..8u8 {
                link.send(1, err()).await.expect("queue");
            }
        });
        link.abort();
        assert!(!link.is_running(), "the worker is reaped");
    }
}
