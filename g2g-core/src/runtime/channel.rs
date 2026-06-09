use alloc::collections::VecDeque;
use alloc::sync::Arc;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, Waker};

use alloc::boxed::Box;
use spin::Mutex;

use crate::element::{BoxFuture, OutputSink, PushOutcome, Reconfigure};
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

    /// Non-blocking pop. Returns `None` when the queue is empty (whether or
    /// not senders remain). Lets a consumer drain without awaiting.
    pub fn try_recv(&self) -> Option<T> {
        let mut g = self.inner.lock();
        let v = g.queue.pop_front();
        if v.is_some() {
            if let Some(w) = g.send_waker.take() {
                w.wake();
            }
        }
        v
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

/// Capacity-1 latest-wins slot carrying the upstream-traveling
/// `Reconfigure` signal of a bidirectional link. Stores overwrite any
/// pending value; takes consume it. Cheap: one `Arc<Mutex<Option<_>>>`.
#[derive(Debug, Clone, Default)]
pub struct ReconfigureSlot {
    inner: Arc<Mutex<Option<Reconfigure>>>,
}

impl ReconfigureSlot {
    pub fn store(&self, value: Reconfigure) {
        *self.inner.lock() = Some(value);
    }

    pub fn take(&self) -> Option<Reconfigure> {
        self.inner.lock().take()
    }
}

/// Upstream end of a bidirectional inter-element link: forward
/// `PipelinePacket` channel + reverse `Reconfigure` slot. Held by the
/// producing element (wrapped in [`SenderSink`]). Cloneable so a fan-in
/// merger can share one output link across N forwarders; the link closes
/// when the last clone drops.
#[derive(Debug, Clone)]
pub struct LinkSender {
    pub(crate) data: Sender<PipelinePacket>,
    pub(crate) reconfigure: ReconfigureSlot,
}

/// Downstream end of a bidirectional inter-element link. Held by the
/// consuming element (or the runner loop driving it). `request_reconfigure`
/// fires an upstream signal that the producer observes on its next
/// [`OutputSink::push`].
#[derive(Debug)]
pub struct LinkReceiver {
    pub(crate) data: Receiver<PipelinePacket>,
    pub(crate) reconfigure: ReconfigureSlot,
}

impl LinkReceiver {
    pub fn recv(&self) -> RecvFuture<'_, PipelinePacket> {
        self.data.recv()
    }

    /// Latest-wins: overwrites any pending request that the producer
    /// hasn't yet observed. Reconfigure is a control signal, not a
    /// stream — older proposals are stale by definition.
    pub fn request_reconfigure(&self, r: Reconfigure) {
        self.reconfigure.store(r);
    }
}

/// Build a bidirectional inter-element link with `capacity` forward
/// slots and a capacity-1 reverse `Reconfigure` slot.
pub fn link(capacity: usize) -> (LinkSender, LinkReceiver) {
    let (data_tx, data_rx) = bounded::<PipelinePacket>(capacity);
    let slot = ReconfigureSlot::default();
    (
        LinkSender { data: data_tx, reconfigure: slot.clone() },
        LinkReceiver { data: data_rx, reconfigure: slot },
    )
}

/// Adapter from a [`LinkSender`] to the async `OutputSink` trait. The
/// packet is *always enqueued* (so no in-flight data is lost mid-stream);
/// then the reverse `Reconfigure` slot is checked. If downstream fired
/// a request — either before this push or while the producer was
/// awaiting capacity — the producer sees `PushOutcome::Reconfigure(...)`
/// and must handle it before producing the next packet. The just-pushed
/// packet continues downstream under the still-current caps, exactly as
/// GStreamer's CAPS handshake allows.
#[derive(Debug)]
pub struct SenderSink {
    link: LinkSender,
}

impl SenderSink {
    pub fn new(link: LinkSender) -> Self {
        Self { link }
    }
}

impl OutputSink for SenderSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            match self.link.data.send(packet).await {
                Ok(()) => match self.link.reconfigure.take() {
                    Some(r) => Ok(PushOutcome::Reconfigure(r)),
                    None => Ok(PushOutcome::Accepted),
                },
                Err(SendError::Closed) => Err(G2gError::Shutdown),
                Err(SendError::Full) => unreachable!("send().await never returns Full"),
            }
        })
    }
}

#[cfg(test)]
mod link_tests {
    use super::*;
    use crate::caps::{Caps, Dim, Rate, VideoFormat};
    use crate::frame::{Frame, FrameTiming};
    use crate::memory::{MemoryDomain, SystemSlice};
    use alloc::boxed::Box;
    use core::pin::Pin;
    use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

    // Hand-rolled noop waker so this test module has no extra dev-dep.
    // The link's send/recv futures resolve in a single poll whenever
    // capacity is non-zero, so we never need to actually re-wake.
    static NOOP_VTABLE: RawWakerVTable = RawWakerVTable::new(
        |_| RawWaker::new(core::ptr::null(), &NOOP_VTABLE),
        |_| {},
        |_| {},
        |_| {},
    );

    fn noop_waker() -> Waker {
        // SAFETY: NOOP_VTABLE's functions are all no-ops and never
        // dereference the data pointer; passing null is safe.
        unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &NOOP_VTABLE)) }
    }

    fn run_to_ready<F: core::future::Future>(mut fut: F) -> F::Output {
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        // SAFETY: `fut` lives on the stack for the duration of this fn
        // and we never move it after pinning.
        let mut pinned = unsafe { Pin::new_unchecked(&mut fut) };
        loop {
            match pinned.as_mut().poll(&mut cx) {
                Poll::Ready(v) => return v,
                Poll::Pending => panic!("link_tests::run_to_ready saw Pending"),
            }
        }
    }

    fn dummy_frame() -> PipelinePacket {
        PipelinePacket::DataFrame(Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
            caps: Caps::Video {
                format: VideoFormat::H264,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
            },
            timing: FrameTiming { pts_ns: 0, dts_ns: 0, duration_ns: 0, capture_ns: 0 },
            sequence: 0,
        })
    }

    fn proposed_caps() -> Caps {
        Caps::Video {
            format: VideoFormat::H264,
            width: Dim::Fixed(1280),
            height: Dim::Fixed(720),
            framerate: Rate::Any,
        }
    }

    #[test]
    fn push_returns_accepted_when_no_reconfigure_pending() {
        let (tx, _rx) = link(2);
        let mut sink = SenderSink::new(tx);
        let outcome = run_to_ready(sink.push(dummy_frame())).expect("send ok");
        assert_eq!(outcome, PushOutcome::Accepted);
    }

    #[test]
    fn request_reconfigure_surfaces_on_next_push() {
        let (tx, rx) = link(2);
        let mut sink = SenderSink::new(tx);

        // Downstream fires reconfigure before upstream pushes.
        rx.request_reconfigure(Reconfigure::Propose(proposed_caps()));

        // Packet is still enqueued (no data loss); reconfigure surfaces
        // as the push outcome so the producer can react before the next.
        let outcome = run_to_ready(sink.push(dummy_frame())).expect("send ok");
        match outcome {
            PushOutcome::Reconfigure(Reconfigure::Propose(c)) => {
                assert_eq!(c, proposed_caps());
            }
            other => panic!("expected Reconfigure::Propose, got {other:?}"),
        }

        let received = run_to_ready(rx.recv());
        assert!(matches!(received, Some(PipelinePacket::DataFrame(_))));
    }

    #[test]
    fn second_push_returns_accepted_after_reconfigure_drained() {
        let (tx, rx) = link(2);
        let mut sink = SenderSink::new(tx);

        rx.request_reconfigure(Reconfigure::Renegotiate);
        let first = run_to_ready(sink.push(dummy_frame())).unwrap();
        assert!(matches!(first, PushOutcome::Reconfigure(_)));

        let second = run_to_ready(sink.push(dummy_frame())).unwrap();
        assert_eq!(second, PushOutcome::Accepted);
    }

    #[test]
    fn try_recv_returns_value_then_none() {
        let (tx, rx) = bounded::<u32>(2);
        assert_eq!(rx.try_recv(), None, "empty queue");
        tx.try_send(7).unwrap();
        assert_eq!(rx.try_recv(), Some(7));
        assert_eq!(rx.try_recv(), None, "drained");
    }

    #[test]
    fn try_recv_drains_then_none_after_senders_drop() {
        let (tx, rx) = bounded::<u32>(2);
        tx.try_send(1).unwrap();
        drop(tx);
        assert_eq!(rx.try_recv(), Some(1), "remaining value still drains");
        assert_eq!(rx.try_recv(), None, "empty and closed");
    }

    #[test]
    fn latest_reconfigure_overwrites_older_pending() {
        let (tx, rx) = link(2);
        let mut sink = SenderSink::new(tx);

        // Stale: must be overwritten by the next request.
        rx.request_reconfigure(Reconfigure::Renegotiate);
        rx.request_reconfigure(Reconfigure::Propose(proposed_caps()));

        let outcome = run_to_ready(sink.push(dummy_frame())).unwrap();
        match outcome {
            PushOutcome::Reconfigure(Reconfigure::Propose(c)) => {
                assert_eq!(c, proposed_caps(), "newest proposal must win");
            }
            other => panic!("expected newest Propose, got {other:?}"),
        }
    }
}
