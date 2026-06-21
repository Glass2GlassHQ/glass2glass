use alloc::collections::VecDeque;
use alloc::sync::Arc;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, Waker};

use alloc::boxed::Box;
use spin::Mutex;

use crate::element::{BoxFuture, OutputSink, PushOutcome, QosMessage, Reconfigure};
use crate::error::G2gError;
use crate::frame::PipelinePacket;
use crate::link::LinkPolicy;

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

    /// Remove and return the front-most queued value matching `pred`, or
    /// `None` if none match. Used by a leaky `DropOldest` link to evict the
    /// oldest data frame and make room without disturbing queued control
    /// packets. No waker is signalled: a receiver only parks when the queue is
    /// empty, and eviction only runs on a full queue.
    pub(crate) fn evict_front_matching(&self, pred: impl Fn(&T) -> bool) -> Option<T> {
        let mut g = self.inner.lock();
        let idx = g.queue.iter().position(pred)?;
        g.queue.remove(idx)
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

    /// Current fill of the channel as a percent (0 = empty, 100 = full),
    /// a snapshot for buffering observability. Capacity is always > 0.
    pub fn fill_percent(&self) -> u8 {
        let g = self.inner.lock();
        ((g.queue.len() * 100) / g.capacity) as u8
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

/// Capacity-1 latest-wins slot carrying the upstream-traveling [`QosMessage`] of
/// a bidirectional link (M174). Same shape as [`ReconfigureSlot`]: a later QoS
/// report supersedes an unobserved earlier one (lateness is a current condition,
/// not a stream).
#[derive(Debug, Clone, Default)]
pub struct QosSlot {
    inner: Arc<Mutex<Option<QosMessage>>>,
}

impl QosSlot {
    pub fn store(&self, value: QosMessage) {
        *self.inner.lock() = Some(value);
    }

    pub fn take(&self) -> Option<QosMessage> {
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
    /// Reverse QoS slot (M174): a downstream sink stores a lateness report here;
    /// the producer observes it on its next push as [`PushOutcome::Qos`].
    pub(crate) qos: QosSlot,
    /// Backpressure policy for this link. `Block` (the default) awaits
    /// capacity; the leaky variants drop data frames under a full channel.
    pub(crate) policy: LinkPolicy,
    /// Cumulative count of frames this link has dropped, shared with the
    /// runner so the drop total surfaces in `RunStats`. `None` until the
    /// runner installs one (leaky links only).
    pub(crate) dropped: Option<Arc<Mutex<u64>>>,
}

impl LinkSender {
    /// Set this link's backpressure policy (the runner applies the edge's
    /// `LinkPolicy` after building the channel). Only the `std` graph runner
    /// wires per-edge policy today; `no_std` runners use the `Block` default.
    #[cfg(feature = "std")]
    pub(crate) fn set_policy(&mut self, policy: LinkPolicy) {
        self.policy = policy;
    }

    /// Install the shared drop counter so leaky drops are observable.
    #[cfg(feature = "std")]
    pub(crate) fn set_drop_counter(&mut self, counter: Arc<Mutex<u64>>) {
        self.dropped = Some(counter);
    }

    /// Record one dropped frame, if a counter is installed.
    fn record_drop(&self) {
        if let Some(c) = &self.dropped {
            *c.lock() += 1;
        }
    }
}

/// Downstream end of a bidirectional inter-element link. Held by the
/// consuming element (or the runner loop driving it). `request_reconfigure`
/// fires an upstream signal that the producer observes on its next
/// [`OutputSink::push`].
#[derive(Debug)]
pub struct LinkReceiver {
    pub(crate) data: Receiver<PipelinePacket>,
    pub(crate) reconfigure: ReconfigureSlot,
    pub(crate) qos: QosSlot,
}

impl LinkReceiver {
    pub fn recv(&self) -> RecvFuture<'_, PipelinePacket> {
        self.data.recv()
    }

    /// Non-blocking drain of one packet; `None` when the link is empty.
    pub fn try_recv(&self) -> Option<PipelinePacket> {
        self.data.try_recv()
    }

    /// Fill of this link as a percent (0-100), for buffering reports.
    pub fn fill_percent(&self) -> u8 {
        self.data.fill_percent()
    }

    /// Latest-wins: overwrites any pending request that the producer
    /// hasn't yet observed. Reconfigure is a control signal, not a
    /// stream — older proposals are stale by definition.
    pub fn request_reconfigure(&self, r: Reconfigure) {
        self.reconfigure.store(r);
    }

    /// Latest-wins QoS signal (M174): the consuming sink reports it ran behind
    /// the clock; the producer observes it on its next [`OutputSink::push`] as
    /// [`PushOutcome::Qos`] and may skip ahead to shed load.
    pub fn request_qos(&self, q: QosMessage) {
        self.qos.store(q);
    }

    /// A clone of this link's reverse QoS slot (M175). A transform arm hands it
    /// to its *output* [`SenderSink`] as a relay target, so a QoS report seen on
    /// the downstream link is forwarded onto this (upstream) link toward the
    /// source instead of being dropped at the transform.
    pub(crate) fn qos_slot(&self) -> QosSlot {
        self.qos.clone()
    }
}

/// Build a bidirectional inter-element link with `capacity` forward
/// slots and a capacity-1 reverse `Reconfigure` slot.
pub fn link(capacity: usize) -> (LinkSender, LinkReceiver) {
    let (data_tx, data_rx) = bounded::<PipelinePacket>(capacity);
    let slot = ReconfigureSlot::default();
    let qos = QosSlot::default();
    (
        LinkSender {
            data: data_tx,
            reconfigure: slot.clone(),
            qos: qos.clone(),
            policy: LinkPolicy::Block,
            dropped: None,
        },
        LinkReceiver { data: data_rx, reconfigure: slot, qos },
    )
}

/// What a [`LinkInterceptor`] decides for a packet crossing a link.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeAction {
    /// Forward the packet downstream as usual.
    Pass,
    /// Drop the packet; it never reaches the downstream element.
    Drop,
}

/// A probe registered on a link. `on_packet` is called for every packet
/// before it is sent, and returns whether to pass or drop it. The g2g
/// equivalent of a GStreamer pad probe (DESIGN.md §4.9).
pub trait LinkInterceptor {
    fn on_packet(&self, packet: &PipelinePacket) -> ProbeAction;
}

/// Cloneable slot holding the optional [`LinkInterceptor`] of a link's
/// [`SenderSink`]. Same latest-wins shape as [`ReconfigureSlot`]; clones
/// share the inner cell, so the application installs/removes a probe at
/// runtime while the runner drives the link.
#[derive(Clone, Default)]
pub struct ProbeSlot {
    inner: Arc<Mutex<Option<Arc<dyn LinkInterceptor + Send + Sync>>>>,
}

impl core::fmt::Debug for ProbeSlot {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ProbeSlot").finish_non_exhaustive()
    }
}

impl ProbeSlot {
    /// Install (or replace) the probe consulted on every push.
    pub fn install(&self, probe: Arc<dyn LinkInterceptor + Send + Sync>) {
        *self.inner.lock() = Some(probe);
    }

    /// Remove the probe; subsequent packets pass unconditionally.
    pub fn remove(&self) {
        *self.inner.lock() = None;
    }

    fn action(&self, packet: &PipelinePacket) -> ProbeAction {
        match self.inner.lock().as_ref() {
            Some(probe) => probe.on_packet(packet),
            None => ProbeAction::Pass,
        }
    }
}

/// Adapter from a [`LinkSender`] to the async `OutputSink` trait. Push
/// flow per packet:
///
/// 1. A `ProbeSlot` may drop the packet outright.
/// 2. The reverse `Reconfigure` slot is checked **before** send. If
///    downstream already requested reconfigure, the packet is *not*
///    enqueued and the producer sees `PushOutcome::Reconfigure(...)`.
///    The caller is expected to handle the request — typically by
///    calling `reconfigure()`, emitting a fresh `CapsChanged`, and
///    composing the next frame under the agreed caps — before pushing
///    again. The unsent packet is the caller's responsibility: resend
///    it under the new caps, drop it, or skip ahead. This pre-send
///    interception is the in-band ordering fix: rejected packets that
///    the producer had not yet committed never cross the link under
///    stale caps.
/// 3. Otherwise the packet is enqueued. The slot is checked again
///    afterwards: a request that fired *while* the producer was
///    awaiting capacity still surfaces, but the just-enqueued packet
///    has already crossed under old caps. That window is irreducible —
///    the producer was already committed before the request was made.
#[derive(Debug)]
pub struct SenderSink {
    link: LinkSender,
    probe: ProbeSlot,
    /// Relay target for a downstream QoS report (M175). `None` on a source's
    /// output adapter: a QoS seen on the link surfaces as [`PushOutcome::Qos`]
    /// so the source element acts on it. `Some` on a transform's output adapter:
    /// the report is stored into this (the transform's *input* link) reverse
    /// slot instead, forwarding it one hop toward the source. A generic
    /// transform thus relays QoS without having to observe it in `process`.
    upstream_qos: Option<QosSlot>,
}

impl SenderSink {
    pub fn new(link: LinkSender) -> Self {
        Self { link, probe: ProbeSlot::default(), upstream_qos: None }
    }

    /// A handle to this link's probe slot, for installing/removing a
    /// [`LinkInterceptor`] at runtime.
    pub fn probe(&self) -> ProbeSlot {
        self.probe.clone()
    }

    /// Make this adapter relay any downstream QoS report onto `upstream` (the
    /// owning transform's input link) rather than surfacing it (M175). The
    /// runner wires this so QoS propagates source-ward through a transform.
    pub(crate) fn relay_qos_to(&mut self, upstream: QosSlot) {
        self.upstream_qos = Some(upstream);
    }

    /// Outcome to report once a packet has been enqueued: a pending reverse
    /// signal (reconfigure first, then QoS), else `Accepted`. Reconfigure takes
    /// priority because it is negotiation-critical; QoS is advisory. When a
    /// relay target is set (a transform adapter), an observed QoS is forwarded
    /// upstream and the outcome stays `Accepted` rather than surfacing `Qos`.
    fn post_send_outcome(&self) -> PushOutcome {
        if let Some(r) = self.link.reconfigure.take() {
            PushOutcome::Reconfigure(r)
        } else if let Some(q) = self.link.qos.take() {
            match &self.upstream_qos {
                Some(upstream) => {
                    upstream.store(q);
                    PushOutcome::Accepted
                }
                None => PushOutcome::Qos(q),
            }
        } else {
            PushOutcome::Accepted
        }
    }
}

impl OutputSink for SenderSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            // A probe may drop the packet before it ever enters the link.
            if self.probe.action(&packet) == ProbeAction::Drop {
                return Ok(PushOutcome::Accepted);
            }
            // Pre-send check: if downstream already requested a
            // reconfigure, surface it before this packet enters the
            // link. Caller renegotiates and decides what to do with
            // `packet` (resend under agreed caps, drop, etc.).
            if let Some(r) = self.link.reconfigure.take() {
                return Ok(PushOutcome::Reconfigure(r));
            }
            // Leaky links drop *data frames* under a full channel rather than
            // applying backpressure; control packets (caps / segment / flush /
            // eos) are never dropped, they always block so the stream stays
            // correct. A non-leaky link (the default) always blocks.
            let is_data = matches!(packet, PipelinePacket::DataFrame(_));
            if is_data && self.link.policy != LinkPolicy::Block {
                match self.link.policy {
                    LinkPolicy::DropNewest => match self.link.data.try_send(packet) {
                        Ok(()) => {}
                        // Channel full: drop the incoming frame.
                        Err((_dropped, SendError::Full)) => self.link.record_drop(),
                        Err((_v, SendError::Closed)) => return Err(G2gError::Shutdown),
                    },
                    LinkPolicy::DropOldest => match self.link.data.try_send(packet) {
                        Ok(()) => {}
                        Err((returned, SendError::Full)) => {
                            // Evict the oldest queued data frame to make room.
                            // If only control packets are queued, fall back to
                            // blocking rather than dropping a control packet.
                            if self
                                .link
                                .data
                                .evict_front_matching(|p| {
                                    matches!(p, PipelinePacket::DataFrame(_))
                                })
                                .is_some()
                            {
                                self.link.record_drop();
                                match self.link.data.try_send(returned) {
                                    Ok(()) => {}
                                    Err((_v, SendError::Closed)) => {
                                        return Err(G2gError::Shutdown)
                                    }
                                    Err((_v, SendError::Full)) => {
                                        unreachable!("a slot was just freed by eviction")
                                    }
                                }
                            } else {
                                self.link
                                    .data
                                    .send(returned)
                                    .await
                                    .map_err(|_| G2gError::Shutdown)?;
                            }
                        }
                        Err((_v, SendError::Closed)) => return Err(G2gError::Shutdown),
                    },
                    LinkPolicy::Block => unreachable!("guarded by policy != Block"),
                }
                return Ok(self.post_send_outcome());
            }
            match self.link.data.send(packet).await {
                // Post-send check covers the "request fired while we were
                // awaiting capacity" window; the packet is already in the link
                // under old caps.
                Ok(()) => Ok(self.post_send_outcome()),
                Err(SendError::Closed) => Err(G2gError::Shutdown),
                Err(SendError::Full) => unreachable!("send().await never returns Full"),
            }
        })
    }
}

#[cfg(test)]
mod link_tests {
    use super::*;
    use crate::caps::{Caps, Dim, Rate, VideoCodec};
    use crate::frame::{Frame, FrameTiming};
    use crate::memory::{MemoryDomain, SystemSlice};
    use alloc::boxed::Box;
    use alloc::vec::Vec;
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
        match pinned.as_mut().poll(&mut cx) {
            Poll::Ready(v) => v,
            Poll::Pending => panic!("link_tests::run_to_ready saw Pending"),
        }
    }

    fn dummy_frame() -> PipelinePacket {
        PipelinePacket::DataFrame(Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
            timing: FrameTiming::default(),
            sequence: 0,
            meta: Default::default(),
        })
    }

    fn proposed_caps() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
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

        // Pre-send check intercepts: the packet is NOT enqueued, and
        // the producer sees Reconfigure so it can renegotiate before
        // any frame crosses under stale caps. Caller decides whether
        // to resend `packet` under agreed caps, drop it, or skip.
        let outcome = run_to_ready(sink.push(dummy_frame())).expect("push ok");
        match outcome {
            PushOutcome::Reconfigure(Reconfigure::Propose(c)) => {
                assert_eq!(c, proposed_caps());
            }
            other => panic!("expected Reconfigure::Propose, got {other:?}"),
        }

        // Channel is empty — the rejected-caps packet was held back.
        assert!(rx.try_recv().is_none(), "packet must not enqueue when reconfigure pending");
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
    fn request_qos_surfaces_after_the_packet_is_sent() {
        let (tx, rx) = link(2);
        let mut sink = SenderSink::new(tx);

        // Downstream reports it is behind; QoS is advisory, so the packet still
        // crosses and the producer sees Qos on the same push.
        rx.request_qos(QosMessage { jitter_ns: 5_000_000, running_time_ns: 100 });
        let outcome = run_to_ready(sink.push(dummy_frame())).expect("push ok");
        match outcome {
            PushOutcome::Qos(q) => {
                assert_eq!(q.jitter_ns, 5_000_000);
                assert_eq!(q.running_time_ns, 100);
            }
            other => panic!("expected Qos, got {other:?}"),
        }
        // Unlike reconfigure, the packet was enqueued (QoS does not hold it back).
        assert!(rx.try_recv().is_some(), "QoS is advisory; the frame still flowed");
    }

    #[test]
    fn reconfigure_takes_priority_over_qos() {
        let (tx, rx) = link(2);
        let mut sink = SenderSink::new(tx);

        // Both pending: negotiation correctness wins, QoS waits for the next push.
        rx.request_qos(QosMessage { jitter_ns: 1_000, running_time_ns: 0 });
        rx.request_reconfigure(Reconfigure::Renegotiate);
        let first = run_to_ready(sink.push(dummy_frame())).unwrap();
        assert!(matches!(first, PushOutcome::Reconfigure(_)), "reconfigure first");

        let second = run_to_ready(sink.push(dummy_frame())).unwrap();
        assert!(matches!(second, PushOutcome::Qos(_)), "QoS surfaces once reconfigure drained");
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

    fn frame_seq(seq: u64) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
            timing: FrameTiming::default(),
            sequence: seq,
            meta: Default::default(),
        })
    }

    /// Drops `DataFrame`s with an odd sequence number; passes everything else.
    struct DropOdd;
    impl LinkInterceptor for DropOdd {
        fn on_packet(&self, packet: &PipelinePacket) -> ProbeAction {
            match packet {
                PipelinePacket::DataFrame(f) if f.sequence % 2 == 1 => ProbeAction::Drop,
                _ => ProbeAction::Pass,
            }
        }
    }

    #[test]
    fn installed_probe_drops_selected_packets() {
        let (tx, rx) = link(8);
        let mut sink = SenderSink::new(tx);
        sink.probe().install(Arc::new(DropOdd));

        for seq in 0..4 {
            run_to_ready(sink.push(frame_seq(seq))).unwrap();
        }

        let mut got = Vec::new();
        while let Some(PipelinePacket::DataFrame(f)) = rx.try_recv() {
            got.push(f.sequence);
        }
        assert_eq!(got, [0, 2], "odd-sequence frames dropped by the probe");
    }

    #[test]
    fn removed_probe_lets_packets_pass_again() {
        let (tx, rx) = link(8);
        let mut sink = SenderSink::new(tx);
        let probe = sink.probe();

        probe.install(Arc::new(DropOdd));
        run_to_ready(sink.push(frame_seq(1))).unwrap(); // dropped
        probe.remove();
        run_to_ready(sink.push(frame_seq(3))).unwrap(); // passes now

        let mut got = Vec::new();
        while let Some(PipelinePacket::DataFrame(f)) = rx.try_recv() {
            got.push(f.sequence);
        }
        assert_eq!(got, [3], "after remove(), the odd frame passes");
    }

    fn drained_sequences(rx: &LinkReceiver) -> Vec<u64> {
        let mut got = Vec::new();
        while let Some(PipelinePacket::DataFrame(f)) = rx.try_recv() {
            got.push(f.sequence);
        }
        got
    }

    #[test]
    fn drop_newest_discards_incoming_when_full() {
        let (mut tx, rx) = link(2);
        tx.set_policy(LinkPolicy::DropNewest);
        let counter = Arc::new(Mutex::new(0u64));
        tx.set_drop_counter(counter.clone());
        let mut sink = SenderSink::new(tx);

        // Fill capacity, then overflow: the incoming frame is dropped, the
        // queued ones survive.
        for seq in 0..2 {
            assert_eq!(run_to_ready(sink.push(frame_seq(seq))).unwrap(), PushOutcome::Accepted);
        }
        assert_eq!(run_to_ready(sink.push(frame_seq(2))).unwrap(), PushOutcome::Accepted);

        assert_eq!(drained_sequences(&rx), [0, 1], "drop-newest keeps the oldest");
        assert_eq!(*counter.lock(), 1);
    }

    #[test]
    fn drop_oldest_evicts_front_when_full() {
        let (mut tx, rx) = link(2);
        tx.set_policy(LinkPolicy::DropOldest);
        let counter = Arc::new(Mutex::new(0u64));
        tx.set_drop_counter(counter.clone());
        let mut sink = SenderSink::new(tx);

        for seq in 0..2 {
            run_to_ready(sink.push(frame_seq(seq))).unwrap();
        }
        // Overflow evicts the oldest (seq 0) and enqueues the newcomer (seq 2).
        assert_eq!(run_to_ready(sink.push(frame_seq(2))).unwrap(), PushOutcome::Accepted);

        assert_eq!(drained_sequences(&rx), [1, 2], "drop-oldest keeps the newest");
        assert_eq!(*counter.lock(), 1);
    }

    #[test]
    fn fill_percent_tracks_link_occupancy() {
        let (tx, rx) = link(4);
        assert_eq!(rx.fill_percent(), 0, "empty link reads 0%");
        let mut sink = SenderSink::new(tx);
        run_to_ready(sink.push(frame_seq(0))).unwrap();
        run_to_ready(sink.push(frame_seq(1))).unwrap();
        assert_eq!(rx.fill_percent(), 50, "2 of 4 slots = 50%");
        run_to_ready(sink.push(frame_seq(2))).unwrap();
        run_to_ready(sink.push(frame_seq(3))).unwrap();
        assert_eq!(rx.fill_percent(), 100, "full link reads 100%");
        rx.try_recv();
        assert_eq!(rx.fill_percent(), 75, "after one drain, 3 of 4 = 75%");
    }

    #[test]
    fn leaky_links_never_drop_control_packets() {
        // A capacity-1 leaky link, filled with a data frame. A control packet
        // must not be dropped: with the link full it blocks (Pending) instead.
        let (mut tx, rx) = link(1);
        tx.set_policy(LinkPolicy::DropNewest);
        let mut sink = SenderSink::new(tx);
        run_to_ready(sink.push(frame_seq(0))).unwrap();

        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        let mut fut = sink.push(PipelinePacket::CapsChanged(proposed_caps()));
        assert!(
            matches!(fut.as_mut().poll(&mut cx), Poll::Pending),
            "a control packet blocks on a full leaky link, never dropped"
        );
        drop(fut);

        // The queued data frame is untouched.
        assert_eq!(drained_sequences(&rx), [0]);
    }
}
