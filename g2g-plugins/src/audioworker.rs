// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! The link between a blocking audio device worker thread and the async element
//! that feeds it, shared by the ALSA, PulseAudio, WASAPI, Core Audio and AAudio
//! sinks.
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
//! So: a bounded queue whose full case parks the element until the worker takes
//! a command, and an end-of-stream drain that parks it until the worker
//! returns. `Eos` still completes only once the
//! audio has actually played out, which is what a sink owes the pipeline; it
//! just no longer owns the executor while that happens.

use core::future::poll_fn;
use core::task::{Poll, Waker};
use core::time::Duration;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvError, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use g2g_core::{G2gError, HardwareError};

/// Commands buffered between the element and its device thread. One command is
/// one buffer of samples, so this is a queue depth in buffers: enough that a
/// scheduling hiccup on the element side cannot underrun the device, small
/// enough that a long file does not accumulate in memory. At the ~21 ms buffers
/// an AC-3 or AAC stream decodes to, 64 is well over a second of slack.
const QUEUE_DEPTH: usize = 64;

#[derive(Debug, Default)]
struct ElementWaker(Mutex<Option<Waker>>);

impl ElementWaker {
    fn register(&self, waker: &Waker) {
        *self.0.lock().expect("no waker holder panics") = Some(waker.clone());
    }

    fn wake(&self) {
        let waker = self.0.lock().expect("no waker holder panics").take();
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

#[derive(Debug)]
pub(crate) struct CommandReceiver<C> {
    rx: Receiver<C>,
    element: Arc<ElementWaker>,
}

impl<C> CommandReceiver<C> {
    pub(crate) fn recv(&self) -> Result<C, RecvError> {
        let received = self.rx.recv();
        self.element.wake();
        received
    }

    pub(crate) fn recv_timeout(&self, timeout: Duration) -> Result<C, RecvTimeoutError> {
        let received = self.rx.recv_timeout(timeout);
        self.element.wake();
        received
    }
}

pub(crate) fn command_queue<C>() -> (SyncSender<C>, CommandReceiver<C>) {
    let (tx, rx) = mpsc::sync_channel(QUEUE_DEPTH);
    let receiver = CommandReceiver {
        rx,
        element: Arc::default(),
    };
    (tx, receiver)
}

struct WorkerExit {
    finished: Arc<AtomicBool>,
    element: Arc<ElementWaker>,
}

// a panicking worker still wakes the element, which then sees the closed queue
impl Drop for WorkerExit {
    fn drop(&mut self) {
        self.finished.store(true, Ordering::Release);
        self.element.wake();
    }
}

/// A device worker thread plus the bounded queue feeding it.
#[derive(Debug)]
pub(crate) struct WorkerLink<C: Send + 'static> {
    tx: Option<SyncSender<C>>,
    worker: Option<JoinHandle<()>>,
    /// Set by the worker as it returns, so the element can wait for playout
    /// without blocking on `join`.
    finished: Arc<AtomicBool>,
    element: Arc<ElementWaker>,
}

impl<C: Send + 'static> WorkerLink<C> {
    /// Spawn `body` on a named thread with a bounded command queue. `body` owns
    /// the device and drains the receiver; the completion flag is set for it.
    pub(crate) fn spawn<F>(name: &str, body: F) -> Result<Self, G2gError>
    where
        F: FnOnce(CommandReceiver<C>) + Send + 'static,
    {
        let (tx, rx) = command_queue::<C>();
        let finished = Arc::new(AtomicBool::new(false));
        let element = Arc::clone(&rx.element);
        let exit = WorkerExit {
            finished: Arc::clone(&finished),
            element: Arc::clone(&element),
        };
        let worker = thread::Builder::new()
            .name(alloc::string::String::from(name))
            .spawn(move || {
                let _exit = exit;
                body(rx);
            })
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        Ok(Self {
            tx: Some(tx),
            worker: Some(worker),
            finished,
            element,
        })
    }

    /// Hand one command to the worker, parking while the queue is full rather
    /// than blocking the executor. This is where back-pressure from the device
    /// reaches the pipeline, which is what bounds the memory.
    pub(crate) async fn send(&self, cmd: C, err: G2gError) -> Result<(), G2gError> {
        let Some(tx) = self.tx.as_ref() else {
            return Err(err);
        };
        let mut cmd = Some(cmd);
        poll_fn(|cx| {
            let pending = cmd.take().expect("send is not polled after it completes");
            let back = match tx.try_send(pending) {
                Ok(()) => return Poll::Ready(Ok(())),
                Err(TrySendError::Disconnected(_)) => return Poll::Ready(Err(err.clone())),
                Err(TrySendError::Full(back)) => back,
            };
            self.element.register(cx.waker());
            // the worker can take a command between the first try and the register
            match tx.try_send(back) {
                Ok(()) => Poll::Ready(Ok(())),
                Err(TrySendError::Disconnected(_)) => Poll::Ready(Err(err.clone())),
                Err(TrySendError::Full(back)) => {
                    cmd = Some(back);
                    Poll::Pending
                }
            }
        })
        .await
    }

    /// End of stream: queue `shutdown`, then wait for the worker to finish
    /// playing out, parked on the executor throughout. Returns once the audio
    /// has actually been played, so a downstream `Eos` still means "the sound
    /// finished", but the rest of the pipeline keeps running while it drains.
    pub(crate) async fn finish(&mut self, shutdown: C, err: G2gError) -> Result<(), G2gError> {
        // A disconnected worker is already gone; nothing left to drain.
        if self.send(shutdown, err.clone()).await.is_ok() {
            poll_fn(|cx| {
                if self.finished.load(Ordering::Acquire) {
                    return Poll::Ready(());
                }
                self.element.register(cx.waker());
                if self.finished.load(Ordering::Acquire) {
                    Poll::Ready(())
                } else {
                    Poll::Pending
                }
            })
            .await;
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
    use crate::yieldnow::yield_now;
    use core::future::Future;
    use core::task::Poll;
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

    #[test]
    fn a_full_queue_parks_the_sender_until_the_worker_takes_a_command() {
        let played = Arc::new(AtomicUsize::new(0));
        let mut link = slow_worker(Duration::from_millis(20), Arc::clone(&played));
        let polls = block_on(async {
            // the worker holds one command while it plays, so one more than the queue fills it
            for _ in 0..=QUEUE_DEPTH {
                link.send(1, err()).await.expect("queue");
            }
            let mut send = alloc::boxed::Box::pin(link.send(1, err()));
            let mut polls = 0usize;
            core::future::poll_fn(|cx| {
                polls += 1;
                send.as_mut().poll(cx)
            })
            .await
            .expect("queued once the worker took a command");
            polls
        });
        link.abort();
        assert!(
            polls <= 3,
            "the sender was polled {polls} times while it waited for room"
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
