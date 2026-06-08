use alloc::collections::VecDeque;
use alloc::sync::Arc;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, Waker};

use spin::Mutex;

use crate::element::OutputSink;
use crate::error::G2gError;
use crate::frame::PipelinePacket;

pub fn bounded<T>(capacity: usize) -> (Sender<T>, Receiver<T>) {
    assert!(capacity > 0, "channel capacity must be > 0");
    let inner = Arc::new(Mutex::new(Inner {
        queue: VecDeque::with_capacity(capacity),
        capacity,
        send_waker: None,
        recv_waker: None,
        senders: 1,
        receivers: 1,
    }));
    (Sender { inner: inner.clone() }, Receiver { inner })
}

#[derive(Debug)]
struct Inner<T> {
    queue: VecDeque<T>,
    capacity: usize,
    send_waker: Option<Waker>,
    recv_waker: Option<Waker>,
    senders: usize,
    receivers: usize,
}

#[derive(Debug)]
pub struct Sender<T> {
    inner: Arc<Mutex<Inner<T>>>,
}

#[derive(Debug)]
pub struct Receiver<T> {
    inner: Arc<Mutex<Inner<T>>>,
}

impl<T> Clone for Sender<T> {
    fn clone(&self) -> Self {
        self.inner.lock().senders += 1;
        Self { inner: self.inner.clone() }
    }
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        let mut g = self.inner.lock();
        g.senders -= 1;
        if g.senders == 0 {
            if let Some(w) = g.recv_waker.take() {
                w.wake();
            }
        }
    }
}

impl<T> Drop for Receiver<T> {
    fn drop(&mut self) {
        let mut g = self.inner.lock();
        g.receivers -= 1;
        if g.receivers == 0 {
            if let Some(w) = g.send_waker.take() {
                w.wake();
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SendError {
    /// All receivers dropped.
    Closed,
    /// Channel at capacity (only from `try_send`).
    Full,
}

impl<T> Sender<T> {
    /// Best-effort synchronous push. Returns the rejected value plus a
    /// reason if the channel is full or closed.
    pub fn try_send(&self, value: T) -> Result<(), (T, SendError)> {
        let mut g = self.inner.lock();
        if g.receivers == 0 {
            return Err((value, SendError::Closed));
        }
        if g.queue.len() >= g.capacity {
            return Err((value, SendError::Full));
        }
        g.queue.push_back(value);
        if let Some(w) = g.recv_waker.take() {
            w.wake();
        }
        Ok(())
    }

    pub fn send(&self, value: T) -> SendFuture<'_, T> {
        SendFuture { sender: self, value: Some(value) }
    }
}

#[allow(missing_debug_implementations)]
pub struct SendFuture<'a, T> {
    sender: &'a Sender<T>,
    value: Option<T>,
}

impl<'a, T: Unpin> Future for SendFuture<'a, T> {
    type Output = Result<(), SendError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let mut g = this.sender.inner.lock();
        if g.receivers == 0 {
            return Poll::Ready(Err(SendError::Closed));
        }
        if g.queue.len() < g.capacity {
            let v = this.value.take().expect("SendFuture polled after completion");
            g.queue.push_back(v);
            if let Some(w) = g.recv_waker.take() {
                w.wake();
            }
            return Poll::Ready(Ok(()));
        }
        g.send_waker = Some(cx.waker().clone());
        Poll::Pending
    }
}

impl<T> Receiver<T> {
    pub fn recv(&self) -> RecvFuture<'_, T> {
        RecvFuture { receiver: self }
    }
}

#[allow(missing_debug_implementations)]
pub struct RecvFuture<'a, T> {
    receiver: &'a Receiver<T>,
}

impl<'a, T> Future for RecvFuture<'a, T> {
    type Output = Option<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let mut g = this.receiver.inner.lock();
        if let Some(v) = g.queue.pop_front() {
            if let Some(w) = g.send_waker.take() {
                w.wake();
            }
            return Poll::Ready(Some(v));
        }
        if g.senders == 0 {
            return Poll::Ready(None);
        }
        g.recv_waker = Some(cx.waker().clone());
        Poll::Pending
    }
}

/// Adapter from a `Sender<PipelinePacket>` to the synchronous `OutputSink`
/// trait. Returns `PoolExhausted` on backpressure — elements that need to
/// await capacity should hold a `Sender` directly and call `send().await`.
#[derive(Debug)]
pub struct SenderSink {
    sender: Sender<PipelinePacket>,
}

impl SenderSink {
    pub fn new(sender: Sender<PipelinePacket>) -> Self {
        Self { sender }
    }
}

impl OutputSink for SenderSink {
    fn push(&mut self, packet: PipelinePacket) -> Result<(), G2gError> {
        match self.sender.try_send(packet) {
            Ok(()) => Ok(()),
            Err((_, SendError::Full)) => Err(G2gError::PoolExhausted),
            Err((_, SendError::Closed)) => Err(G2gError::Shutdown),
        }
    }
}
