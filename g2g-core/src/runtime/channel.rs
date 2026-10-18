use alloc::collections::VecDeque;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicBool, Ordering};
use core::task::{Context, Poll, Waker};

use spin::Mutex;

#[cfg(feature = "std")]
use crate::caps::Caps;
use crate::element::{OutputSink, PushOutcome, QosMessage, Reconfigure};
use crate::error::G2gError;
use crate::frame::PipelinePacket;
use crate::link::LinkPolicy;
use crate::runtime::instrument::{EdgeCounters, Probe};
#[cfg(feature = "std")]
use crate::segment::Segment;

pub fn bounded<T>(capacity: usize) -> (Sender<T>, Receiver<T>) {
    assert!(capacity > 0, "channel capacity must be > 0");
    let inner = Arc::new(Mutex::new(Inner {
        queue: VecDeque::with_capacity(capacity),
        capacity,
        send_wakers: Vec::new(),
        recv_waker: None,
        senders: 1,
        receivers: 1,
    }));
    (
        Sender {
            inner: inner.clone(),
        },
        Receiver { inner },
    )
}

#[derive(Debug)]
struct Inner<T> {
    queue: VecDeque<T>,
    capacity: usize,
    // One per parked sender: a cloned sender parks alongside the original.
    send_wakers: Vec<Waker>,
    recv_waker: Option<Waker>,
    senders: usize,
    receivers: usize,
}

impl<T> Inner<T> {
    fn wake_senders(&mut self) {
        for waker in self.send_wakers.drain(..) {
            waker.wake();
        }
    }
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
        Self {
            inner: self.inner.clone(),
        }
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
        // Whatever is still queued can never be received now, so it is released
        // here rather than living on until the last sender drops. A queued value
        // can own the only handle something else is waiting on (a mutation
        // request carries the reply sender its caller is parked on), and that
        // wait would otherwise never end. Taken under the lock and dropped
        // outside it: a value's own `Drop` may lock another channel.
        let orphaned = {
            let mut g = self.inner.lock();
            g.receivers -= 1;
            if g.receivers == 0 {
                g.wake_senders();
                core::mem::take(&mut g.queue)
            } else {
                VecDeque::new()
            }
        };
        drop(orphaned);
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
        SendFuture {
            sender: self,
            value: Some(value),
        }
    }

    /// Poll form of [`send`](Self::send): enqueue `value` once capacity frees,
    /// parking the send waker while full. `value` is taken only on success, so
    /// the caller re-polls with the same slot.
    pub fn poll_send(
        &self,
        cx: &mut Context<'_>,
        value: &mut Option<T>,
    ) -> Poll<Result<(), SendError>> {
        let mut g = self.inner.lock();
        if g.receivers == 0 {
            return Poll::Ready(Err(SendError::Closed));
        }
        if g.queue.len() < g.capacity {
            let v = value.take().expect("poll_send called without a value");
            g.queue.push_back(v);
            if let Some(w) = g.recv_waker.take() {
                w.wake();
            }
            return Poll::Ready(Ok(()));
        }
        if !g.send_wakers.iter().any(|w| w.will_wake(cx.waker())) {
            g.send_wakers.push(cx.waker().clone());
        }
        Poll::Pending
    }

    /// Remove and return the front-most queued value matching `pred`, or
    /// `None` if none match. Used by a leaky `DropOldest` link to evict the
    /// oldest data frame and make room without disturbing queued control
    /// packets. No waker is signalled: a receiver only parks when the queue is
    /// empty, and eviction only runs on a full queue.
    pub fn evict_front_matching(&self, pred: impl Fn(&T) -> bool) -> Option<T> {
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
        this.sender.poll_send(cx, &mut this.value)
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
            g.wake_senders();
        }
        v
    }

    /// True once every sender has dropped. Values already queued still drain.
    pub fn is_closed(&self) -> bool {
        self.inner.lock().senders == 0
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
            g.wake_senders();
            return Poll::Ready(Some(v));
        }
        if g.senders == 0 {
            return Poll::Ready(None);
        }
        g.recv_waker = Some(cx.waker().clone());
        Poll::Pending
    }
}

// Every push takes each reverse signal, so an empty take reads a flag instead of locking.
#[derive(Debug)]
struct LatestValue<T> {
    pending: AtomicBool,
    value: Mutex<Option<T>>,
}

impl<T> Default for LatestValue<T> {
    fn default() -> Self {
        Self {
            pending: AtomicBool::new(false),
            value: Mutex::new(None),
        }
    }
}

impl<T> LatestValue<T> {
    fn store(&self, value: T) {
        let mut slot = self.value.lock();
        *slot = Some(value);
        self.pending.store(true, Ordering::Release);
    }

    fn take(&self) -> Option<T> {
        if !self.pending.load(Ordering::Acquire) {
            return None;
        }
        let mut slot = self.value.lock();
        self.pending.store(false, Ordering::Relaxed);
        slot.take()
    }
}

/// Capacity-1 latest-wins slot carrying the upstream-traveling
/// `Reconfigure` signal of a bidirectional link. Stores overwrite any
/// pending value; takes consume it.
#[derive(Debug, Clone, Default)]
pub struct ReconfigureSlot {
    inner: Arc<LatestValue<Reconfigure>>,
}

impl ReconfigureSlot {
    pub fn store(&self, value: Reconfigure) {
        self.inner.store(value);
    }

    pub fn take(&self) -> Option<Reconfigure> {
        self.inner.take()
    }
}

/// Capacity-1 latest-wins slot carrying the upstream-traveling [`QosMessage`] of
/// a bidirectional link (M174). Same shape as [`ReconfigureSlot`]: a later QoS
/// report supersedes an unobserved earlier one (lateness is a current condition,
/// not a stream).
#[derive(Debug, Clone, Default)]
pub struct QosSlot {
    inner: Arc<LatestValue<QosMessage>>,
}

impl QosSlot {
    pub fn store(&self, value: QosMessage) {
        self.inner.store(value);
    }

    pub fn take(&self) -> Option<QosMessage> {
        self.inner.take()
    }
}

/// Capacity-1 latest-wins slot carrying an upstream-traveling target bitrate
/// (bits/second), the WebRTC congestion-control / BWE signal. Same shape as
/// [`QosSlot`]: a later estimate supersedes an unobserved earlier one (the
/// current target, not a stream).
#[derive(Debug, Clone, Default)]
pub struct BitrateSlot {
    inner: Arc<LatestValue<u32>>,
}

impl BitrateSlot {
    pub fn store(&self, value: u32) {
        self.inner.store(value);
    }

    pub fn take(&self) -> Option<u32> {
        self.inner.take()
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
    /// Reverse bitrate slot: a downstream WebRTC sink stores its BWE estimate
    /// here; the producer (encoder) observes it as [`PushOutcome::Bitrate`].
    pub(crate) bitrate: BitrateSlot,
    /// Backpressure policy for this link. `Block` (the default) awaits
    /// capacity; the leaky variants drop data frames under a full channel.
    pub(crate) policy: LinkPolicy,
    /// Cumulative count of frames this link has dropped, shared with the
    /// runner so the drop total surfaces in `RunStats`. `None` until the
    /// runner installs one (leaky links only).
    pub(crate) dropped: Option<Arc<Mutex<u64>>>,
    /// Per-edge content-inspection slot (dev tooling). The `SenderSink` wrapping
    /// this link shares it, so a tool can install a [`LinkInterceptor`] to sample
    /// packets crossing this edge without touching the arms. Empty (pass-through,
    /// zero cost) unless a subscriber installs one.
    pub(crate) probe: ProbeSlot,
    /// Per-edge transit-time ring (dev tooling): a send-time stamp per queued
    /// `DataFrame`, popped at the consumer to measure queue residency. `None`
    /// (zero cost) unless the runner enabled instrumentation on this edge.
    pub(crate) transit: Option<TransitRing>,
    /// Per-edge packet / byte / drop counters (dev tooling), shared with the
    /// observer tap so a live consumer reads this edge's traffic mid-run.
    /// `None` (zero cost) unless the runner installed them.
    pub(crate) counters: Option<Arc<EdgeCounters>>,
    /// The mutation endpoint the [`SenderSink`] built over this link adopts
    /// (M1115). `None` (one relaxed load per push, no gate) unless the runner
    /// was asked for a [`GraphMutator`](crate::runtime::GraphMutator).
    #[cfg(feature = "std")]
    pub(crate) mutation: Option<Arc<ProducerEndpoint>>,
}

/// The retargetable producing end of one edge (M1115). The arm pushing into the
/// edge holds it through its [`SenderSink`]; the graph mutator holds the same
/// `Arc` and uses it to stop that producer at a packet boundary, take its
/// [`LinkSender`] away, and hand back a different one, so a transform can be
/// spliced onto or lifted off a running edge.
///
/// Per push the producer pays one relaxed load of `pending`; everything else
/// happens only while a mutation is in flight. `caps` is the shape flowing on
/// the edge right now, updated as each `CapsChanged` crosses, because the
/// negotiated solution is stale once a mid-stream re-solve has moved it.
#[cfg(feature = "std")]
#[derive(Debug, Default)]
pub(crate) struct ProducerEndpoint {
    pending: core::sync::atomic::AtomicBool,
    state: Mutex<EndpointState>,
}

#[cfg(feature = "std")]
#[derive(Debug, Default)]
struct EndpointState {
    /// The mutator wants the producer to stop at its next packet boundary.
    park: bool,
    /// The producer has stopped and left its link in `detached`.
    parked: bool,
    /// The link the producer gave up (on parking, or on its arm ending).
    detached: Option<LinkSender>,
    /// The link the producer picks up when it resumes.
    staged: Option<LinkSender>,
    /// The producer's arm has ended; nothing will park again.
    gone: bool,
    /// The mutator wants this link when the producer's arm ends (a remove
    /// waiting for the element it closed to finish draining). Without it an
    /// ending arm drops its link, which is what closes the channel and ends the
    /// consumer below it, so the claim is never taken by default.
    claimed: bool,
    /// A remove is lifting the element pushing here out, so its arm gives it an
    /// `Eos` once its input closes and forwards whatever it was holding (M1132).
    drain: bool,
    /// M1149: a replacement took this producer's link away for good. Its arm is
    /// resuming onto a dead link so it ends by its own downstream-closed path,
    /// which is not a run failure.
    retired: bool,
    producer: Option<Waker>,
    mutator: Option<Waker>,
    caps: Option<Caps>,
    /// M1149: the segment in force on this edge and the last `DataFrame` that
    /// crossed under it (PTS, duration), so a source replacement can continue
    /// the running time its predecessor reached.
    segment: Option<Segment>,
    last_frame: Option<(u64, u64)>,
    // the last DataFrame sequence the consumer below saw on this edge
    last_sequence: Option<u64>,
    // a producer that just took over the edge must number its first frame above this
    continue_after: Option<u64>,
    sequence_shift: u64,
}

#[cfg(feature = "std")]
impl ProducerEndpoint {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Whether the producer must take the slow path on its next packet: the
    /// whole per-push cost of being mutable.
    #[inline]
    fn pending(&self) -> bool {
        self.pending.load(core::sync::atomic::Ordering::Relaxed)
    }

    /// The caps flowing on this edge right now.
    pub(crate) fn caps(&self) -> Option<Caps> {
        self.state.lock().caps.clone()
    }

    pub(crate) fn set_caps(&self, caps: &Caps) {
        self.state.lock().caps = Some(caps.clone());
    }

    /// Follow what crosses this edge: the shape it carries and the timeline it
    /// carries it on, renumbering frames after a takeover so the sequence the
    /// consumer sees keeps rising. Called once per push while a run is mutable, and never
    /// otherwise (the endpoint is `None`), so an ordinary run keeps its single
    /// branch and the zero-alloc proofs are untouched.
    pub(crate) fn observe(&self, packet: &mut PipelinePacket) {
        match packet {
            PipelinePacket::CapsChanged(caps) => self.state.lock().caps = Some(caps.clone()),
            PipelinePacket::Segment(segment) => self.state.lock().segment = Some(*segment),
            PipelinePacket::DataFrame(frame) => {
                let mut g = self.state.lock();
                g.last_frame = Some((frame.timing.pts_ns, frame.timing.duration_ns));
                if let Some(last) = g.continue_after.take() {
                    g.sequence_shift = match frame.sequence <= last {
                        true => last + 1 - frame.sequence,
                        false => 0,
                    };
                }
                frame.sequence = frame.sequence.saturating_add(g.sequence_shift);
                g.last_sequence = Some(frame.sequence);
            }
            _ => {}
        }
    }

    /// The segment the packets on this edge are timed against, and the last
    /// `DataFrame` (PTS, duration) that crossed under it. Both `None` until
    /// something crosses (M1149).
    pub(crate) fn timeline(&self) -> (Option<Segment>, Option<(u64, u64)>) {
        let g = self.state.lock();
        (g.segment, g.last_frame)
    }

    pub(crate) fn last_sequence(&self) -> Option<u64> {
        self.state.lock().last_sequence
    }

    // a consumer that already saw `last` must not see the numbering go backwards
    pub(crate) fn continue_sequence_after(&self, last: Option<u64>) {
        let mut g = self.state.lock();
        g.continue_after = last;
        g.sequence_shift = 0;
    }

    /// Ask for this producer's link when its arm ends, rather than letting the
    /// arm drop it. Called before the mutator closes the element's input.
    pub(crate) fn claim_on_end(&self) {
        self.state.lock().claimed = true;
    }

    /// Ask the arm pushing here to flush its element before it ends (M1132).
    /// Set before the element's input is closed, so the arm sees it by the time
    /// it reaches the end of its input.
    pub(crate) fn begin_drain(&self) {
        self.state.lock().drain = true;
    }

    /// Whether a remove asked this producer's element to be flushed.
    pub(crate) fn draining(&self) -> bool {
        self.state.lock().drain
    }

    /// Ask the producer to stop at its next packet boundary and leave its link
    /// behind. Pairs with [`poll_detached`](Self::poll_detached).
    pub(crate) fn request_park(&self) {
        let mut guard = self.state.lock();
        guard.park = true;
        // Raised under the lock, like every other write to it, so it cannot land
        // after a producer that is concurrently clearing it decides there is
        // nothing to do (see `poll_producer`).
        self.pending
            .store(true, core::sync::atomic::Ordering::Release);
    }

    /// The link the producer gave up, once it has. `Err` once the producer's arm
    /// has ended without leaving one, i.e. the run is over.
    pub(crate) fn poll_detached(&self, cx: &mut Context<'_>) -> Poll<Result<LinkSender, G2gError>> {
        let mut g = self.state.lock();
        if let Some(link) = g.detached.take() {
            return Poll::Ready(Ok(link));
        }
        if g.gone {
            return Poll::Ready(Err(G2gError::Shutdown));
        }
        g.mutator = Some(cx.waker().clone());
        Poll::Pending
    }

    /// Let the producer run again, on `replacement`. Without one it resumes on
    /// the dead link it parked with and fails its arm on the next push, so a
    /// mutation that has already taken a link away passes back what the producer
    /// should use, its own included when it is rolling back.
    pub(crate) fn unpark(&self, replacement: Option<LinkSender>) {
        let waker = {
            let mut g = self.state.lock();
            // Clears only the request this call answers. A park asked for after
            // this point is a later operation's and belongs to whoever set it.
            g.park = false;
            g.staged = replacement;
            self.pending
                .store(true, core::sync::atomic::Ordering::Release);
            g.producer.take()
        };
        if let Some(w) = waker {
            w.wake();
        }
    }

    /// Withdraw a park request the mutation gave up waiting on. A producer that
    /// already stopped resumes on the link it parked with; one that has not
    /// reached its boundary yet never stops. Done under the one lock, so a
    /// producer parking between the two steps cannot leave its link behind.
    pub(crate) fn cancel_park(&self) {
        let waker = {
            let mut g = self.state.lock();
            g.park = false;
            if let Some(own) = g.detached.take() {
                g.staged = Some(own);
            }
            self.pending
                .store(true, core::sync::atomic::Ordering::Release);
            g.producer.take()
        };
        if let Some(w) = waker {
            w.wake();
        }
    }

    /// End the producer parked here for good (M1149): it resumes on the dead
    /// link it parked with, so its next push fails and its element leaves by the
    /// same path it takes when its consumer goes away. The packet it was holding
    /// at the park dies with it; the packets already queued on the link the
    /// mutator took do not.
    pub(crate) fn retire(&self) {
        self.state.lock().retired = true;
        self.unpark(None);
    }

    /// Whether this producer was retired, read by its arm when it ends: an arm
    /// that failed because a replacement took its link is not a run failure.
    pub(crate) fn retired(&self) -> bool {
        self.state.lock().retired
    }

    /// Hand this endpoint to a fresh producer (M1149). Everything the retired
    /// one left behind goes: without this the endpoint still reads as ended, and
    /// the next operation on the edge would be refused as
    /// [`GraphEnded`](crate::runtime::MutationError::GraphEnded). The timeline is
    /// cleared too, so the next stitch measures from the new producer's own
    /// segment.
    pub(crate) fn rehome(&self) {
        let mut g = self.state.lock();
        g.park = false;
        g.parked = false;
        g.gone = false;
        g.claimed = false;
        g.drain = false;
        g.retired = false;
        g.detached = None;
        g.staged = None;
        g.segment = None;
        g.last_frame = None;
        g.continue_after = g.last_sequence;
        g.sequence_shift = 0;
    }

    /// The producer's own side of the gate: parks, resumes, or retargets. `link`
    /// is the [`SenderSink`]'s current link, replaced in place on a retarget.
    ///
    /// Taking the staged link and parking again are separate steps on purpose.
    /// A producer running on its own thread can still be on its way back from a
    /// resume when the next operation asks it to park, so it arrives here with
    /// both a staged link and a standing request; the request has to survive,
    /// since it was never the one `unpark` answered. (Cooperatively the two
    /// cannot cross: the arms are polled before the mutation service, so the
    /// producer has always taken the staged link by the time the next operation
    /// runs.)
    fn poll_producer(&self, cx: &mut Context<'_>, link: &mut LinkSender) -> Poll<()> {
        let mut g = self.state.lock();
        if let Some(staged) = g.staged.take() {
            *link = staged;
            g.parked = false;
        }
        if g.park {
            if !g.parked {
                // The producer must really give the link up: a remove closes the
                // parked element's input by dropping this, the last sender.
                g.detached = Some(core::mem::replace(link, closed_link()));
                g.parked = true;
                if let Some(w) = g.mutator.take() {
                    w.wake();
                }
            }
            g.producer = Some(cx.waker().clone());
            return Poll::Pending;
        }
        self.pending
            .store(false, core::sync::atomic::Ordering::Relaxed);
        Poll::Ready(())
    }

    /// The producing arm has ended. A claimed link is left here (a clone, so the
    /// count of senders on the channel is unchanged) for the remove waiting on
    /// it; an unclaimed one is not, so an arm that ends on its own still closes
    /// the channel behind it.
    fn producer_gone(&self, link: &LinkSender) {
        let waker = {
            let mut g = self.state.lock();
            g.gone = true;
            if g.claimed && g.detached.is_none() {
                g.detached = Some(link.clone());
            }
            g.mutator.take()
        };
        if let Some(w) = waker {
            w.wake();
        }
    }
}

/// A sender whose receiver is already gone, parked in a [`SenderSink`] while its
/// producer is stopped so the real link can be handed to the mutator.
#[cfg(feature = "std")]
fn closed_link() -> LinkSender {
    let (tx, rx) = link(1);
    drop(rx);
    tx
}

/// Send-time stamps for the packets queued on one link, shared between its
/// [`LinkSender`] and [`LinkReceiver`]. FIFO, aligned with the data channel (one
/// stamp pushed per queued `DataFrame`, one popped per received `DataFrame`), so
/// the consumer reads each frame's queue-residency time.
pub(crate) type TransitRing = Arc<Mutex<VecDeque<u64>>>;

#[derive(Default)]
struct LazyNow(Option<u64>);

impl LazyNow {
    fn get(&mut self) -> u64 {
        *self.0.get_or_insert_with(stamp_now_ns)
    }
}

/// Monotonic send stamp for the transit ring; 0 under `no_std` (no clock).
#[inline]
fn stamp_now_ns() -> u64 {
    #[cfg(feature = "std")]
    {
        crate::metrics::monotonic_ns()
    }
    #[cfg(not(feature = "std"))]
    {
        0
    }
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

    /// Install this edge's live traffic counters (dev tooling), so a mid-run
    /// observer snapshot sees the packets / bytes / drops crossing here.
    #[cfg(feature = "std")]
    pub(crate) fn set_counters(&mut self, counters: Arc<EdgeCounters>) {
        self.counters = Some(counters);
    }

    /// The mutation endpoint a [`SenderSink`] built over this link would adopt.
    /// A lent source arm reads it to tell a retirement from a real failure
    /// (M1149).
    #[cfg(feature = "std")]
    pub(crate) fn mutation_endpoint(&self) -> Option<Arc<ProducerEndpoint>> {
        self.mutation.clone()
    }

    /// Give (or take away) the mutation endpoint the [`SenderSink`] built over
    /// this link adopts. Set by the runner per mutable edge, and cleared by the
    /// mutator on a link it re-homes, so an endpoint is never adopted twice.
    #[cfg(feature = "std")]
    pub(crate) fn set_mutation(&mut self, endpoint: Option<Arc<ProducerEndpoint>>) {
        self.mutation = endpoint;
    }

    /// Queue a `CapsChanged` on this link from outside any element's push path:
    /// the mutator announcing the shape a splice changed the edge to, ordered
    /// behind everything already queued and ahead of everything the retargeted
    /// producer sends next.
    #[cfg(feature = "std")]
    pub(crate) async fn send_caps(&self, caps: Caps) -> Result<(), G2gError> {
        self.send_packet(PipelinePacket::CapsChanged(caps)).await
    }

    /// Queue an `Eos` the same way (M1149): the mutator ending a sink it is
    /// replacing, behind everything already queued, so that sink's arm drains
    /// what it was given and then finalizes on its normal end-of-stream path.
    #[cfg(feature = "std")]
    pub(crate) async fn send_eos(&self) -> Result<(), G2gError> {
        self.send_packet(PipelinePacket::Eos).await
    }

    #[cfg(feature = "std")]
    async fn send_packet(&self, packet: PipelinePacket) -> Result<(), G2gError> {
        self.data.send(packet).await.map_err(|_| G2gError::Shutdown)
    }

    /// Record one dropped frame, if a counter is installed.
    fn record_drop(&self) {
        if let Some(c) = &self.dropped {
            *c.lock() += 1;
        }
        if let Some(c) = &self.counters {
            c.record_drop();
        }
    }

    /// Record one packet that entered the link, if counters are installed,
    /// with the time the producer spent awaiting capacity.
    fn record_sent(&self, bytes: u64, blocked_ns: u64) {
        if let Some(c) = &self.counters {
            c.record_packet(bytes, blocked_ns);
        }
    }
}

/// Payload bytes of a packet as they cross a link: the CPU-resident buffer's
/// length. A device-domain frame (a CUDA / texture handle) carries no bytes
/// here, and a control packet none at all, so both count 0.
pub(crate) fn packet_bytes(packet: &PipelinePacket) -> u64 {
    match packet {
        PipelinePacket::DataFrame(f) => match &f.domain {
            crate::memory::MemoryDomain::System(s) => s.as_slice().len() as u64,
            #[cfg(feature = "alloc")]
            crate::memory::MemoryDomain::SystemView(v) => v.backing().len() as u64,
            #[cfg(feature = "alloc")]
            _ => 0,
        },
        _ => 0,
    }
}

/// Downstream end of a bidirectional inter-element link. Held by the
/// consuming element (or the runner loop driving it). `request_reconfigure`
/// fires an upstream signal that the producer observes on its next
/// [`OutputSinkExt::push`](crate::element::OutputSinkExt::push).
#[derive(Debug)]
pub struct LinkReceiver {
    pub(crate) data: Receiver<PipelinePacket>,
    pub(crate) reconfigure: ReconfigureSlot,
    pub(crate) qos: QosSlot,
    pub(crate) bitrate: BitrateSlot,
    /// Shared with the [`LinkSender`] when transit instrumentation is on; see
    /// [`pop_transit_ns`](LinkReceiver::pop_transit_ns).
    pub(crate) transit: Option<TransitRing>,
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

    /// Pop the queue-residency (transit) time in ns of the just-received
    /// `DataFrame`: the wall-clock elapsed since the producer queued it. Call
    /// once per received `DataFrame` to keep the stamp ring aligned with the data
    /// channel. `None` when this edge is not instrumented (or under `no_std`,
    /// where there is no clock so the stamp is 0). Only `DataFrame`s are stamped,
    /// so callers must not pop for control packets.
    pub fn pop_transit_ns(&self) -> Option<u64> {
        #[cfg(feature = "std")]
        {
            self.transit.as_ref()?;
            self.pop_transit_at(crate::metrics::monotonic_ns())
        }
        #[cfg(not(feature = "std"))]
        {
            self.transit.as_ref()?.lock().pop_front()?;
            Some(0)
        }
    }

    // As `pop_transit_ns`, with the caller's clock read.
    #[cfg(feature = "std")]
    pub(crate) fn pop_transit_at(&self, now_ns: u64) -> Option<u64> {
        let sent = self.transit.as_ref()?.lock().pop_front()?;
        Some(now_ns.saturating_sub(sent))
    }

    /// Latest-wins: overwrites any pending request that the producer
    /// hasn't yet observed. Reconfigure is a control signal, not a
    /// stream — older proposals are stale by definition.
    pub fn request_reconfigure(&self, r: Reconfigure) {
        self.reconfigure.store(r);
    }

    /// Latest-wins QoS signal (M174): the consuming sink reports it ran behind
    /// the clock; the producer observes it on its next [`OutputSinkExt::push`](crate::element::OutputSinkExt::push) as
    /// [`PushOutcome::Qos`] and may skip ahead to shed load.
    pub fn request_qos(&self, q: QosMessage) {
        self.qos.store(q);
    }

    /// Latest-wins target bitrate (bits/second): a downstream WebRTC sink reports
    /// its congestion-control / BWE estimate; the producing encoder observes it on
    /// its next [`OutputSinkExt::push`](crate::element::OutputSinkExt::push) as [`PushOutcome::Bitrate`] and retargets.
    pub fn request_bitrate(&self, bps: u32) {
        self.bitrate.store(bps);
    }

    /// A clone of this link's reverse QoS slot (M175). A transform arm hands it
    /// to its *output* [`SenderSink`] as a relay target, so a QoS report seen on
    /// the downstream link is forwarded onto this (upstream) link toward the
    /// source instead of being dropped at the transform.
    #[cfg(feature = "std")]
    pub(crate) fn qos_slot(&self) -> QosSlot {
        self.qos.clone()
    }

    /// A clone of this link's reverse reconfigure slot (M720), the
    /// keyframe-request analog of [`qos_slot`](Self::qos_slot).
    #[cfg(feature = "std")]
    pub(crate) fn reconfigure_slot(&self) -> ReconfigureSlot {
        self.reconfigure.clone()
    }

    /// A clone of this link's reverse bitrate slot (M720).
    #[cfg(feature = "std")]
    pub(crate) fn bitrate_slot(&self) -> BitrateSlot {
        self.bitrate.clone()
    }
}

/// Build a bidirectional inter-element link with `capacity` forward
/// slots and a capacity-1 reverse `Reconfigure` slot.
pub fn link(capacity: usize) -> (LinkSender, LinkReceiver) {
    build_link(capacity, None)
}

/// As [`link`], but with per-edge transit-time instrumentation enabled: the
/// sender stamps each queued `DataFrame`, the receiver pops the stamp to measure
/// queue residency. Used by the graph runner (std) when an observer is attached;
/// gated on `std` so the no_std / runtime-only build doesn't flag it as unused.
#[cfg(feature = "std")]
pub(crate) fn link_with_transit(capacity: usize) -> (LinkSender, LinkReceiver) {
    build_link(capacity, Some(Arc::new(Mutex::new(VecDeque::new()))))
}

fn build_link(capacity: usize, transit: Option<TransitRing>) -> (LinkSender, LinkReceiver) {
    let (data_tx, data_rx) = bounded::<PipelinePacket>(capacity);
    let slot = ReconfigureSlot::default();
    let qos = QosSlot::default();
    let bitrate = BitrateSlot::default();
    (
        LinkSender {
            data: data_tx,
            reconfigure: slot.clone(),
            qos: qos.clone(),
            bitrate: bitrate.clone(),
            policy: LinkPolicy::Block,
            dropped: None,
            probe: ProbeSlot::default(),
            transit: transit.clone(),
            counters: None,
            #[cfg(feature = "std")]
            mutation: None,
        },
        LinkReceiver {
            data: data_rx,
            reconfigure: slot,
            qos,
            bitrate,
            transit,
        },
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
/// equivalent of a GStreamer pad probe (design/runtime.md).
pub trait LinkInterceptor {
    fn on_packet(&self, packet: &PipelinePacket) -> ProbeAction;
}

/// Cloneable slot holding the optional [`LinkInterceptor`] of a link's
/// [`SenderSink`]. Same latest-wins shape as [`ReconfigureSlot`]; clones
/// share the inner cell, so the application installs/removes a probe at
/// runtime while the runner drives the link.
#[derive(Clone, Default)]
pub struct ProbeSlot {
    inner: Arc<ProbeCell>,
}

#[derive(Default)]
struct ProbeCell {
    // Read on every push, so a link with no interceptor never locks.
    installed: AtomicBool,
    interceptor: Mutex<Option<Arc<dyn LinkInterceptor + Send + Sync>>>,
}

impl core::fmt::Debug for ProbeSlot {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ProbeSlot").finish_non_exhaustive()
    }
}

impl ProbeSlot {
    /// Install (or replace) the probe consulted on every push.
    pub fn install(&self, probe: Arc<dyn LinkInterceptor + Send + Sync>) {
        let mut interceptor = self.inner.interceptor.lock();
        *interceptor = Some(probe);
        self.inner.installed.store(true, Ordering::Release);
    }

    /// Remove the probe; subsequent packets pass unconditionally.
    pub fn remove(&self) {
        let mut interceptor = self.inner.interceptor.lock();
        *interceptor = None;
        self.inner.installed.store(false, Ordering::Release);
    }

    /// The verdict for `packet`: `Pass` when no interceptor is installed. Also
    /// consulted by the fan-in session adapter, which shares the tagged channel
    /// and so carries its own per-input slot.
    pub(crate) fn action(&self, packet: &PipelinePacket) -> ProbeAction {
        if !self.inner.installed.load(Ordering::Acquire) {
            return ProbeAction::Pass;
        }
        match self.inner.interceptor.lock().as_ref() {
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
    /// Relay targets for a downstream QoS report (M175). Empty on a source's
    /// output adapter: a QoS seen on the link surfaces as [`PushOutcome::Qos`]
    /// so the source element acts on it. A transform's or fan-out's output
    /// adapter holds its *input* link's reverse slot, a muxer's holds one per
    /// input link (M1242): the report is stored into each instead, forwarding
    /// it one hop toward the source. A generic transform thus relays QoS
    /// without having to observe it in `process`.
    upstream_qos: Vec<QosSlot>,
    /// Whether a QoS report surfaces as [`PushOutcome::Qos`] to this adapter's
    /// producer. `true` on a source's adapter and for a transform that
    /// [`handles_qos`](crate::element::AsyncElement::handles_qos), which with
    /// `upstream_qos` set both observes the report and relays it.
    qos_answered: bool,
    /// As `upstream_qos`, for downstream reverse-channel `Reconfigure`s the
    /// producer does not answer (M720): set on a transform's output adapter.
    upstream_reconfigure: Option<ReconfigureSlot>,
    /// Which `Reconfigure` variants this adapter's own producer answers, i.e.
    /// which ones surface as `PushOutcome::Reconfigure` instead of travelling
    /// on. Per variant because one element answers one signal and passes the
    /// other: `videoflip` takes `AbsorbOrientation` and still relays a
    /// keyframe request.
    reconfigure_answered: ReconfigureAnswered,
    /// As `upstream_qos`, for downstream bitrate targets (M720).
    upstream_bitrate: Option<BitrateSlot>,
    /// M759 auto-propagation: the propagated metadata set the owning transform
    /// arm derived from its most recent input frame (its element declared a
    /// [`meta_transform`](crate::element::AsyncElement::meta_transform)). Attached
    /// to any outgoing `DataFrame` whose own meta is empty, so a transform that
    /// emits fresh frames still carries the survivors forward without touching
    /// its `process`. `None` on every other adapter and when the propagated set
    /// came out empty (a Drop verdict must not leak a stale set).
    #[cfg(feature = "metadata")]
    meta_stash: Option<crate::meta::FrameMetaSet>,
    /// M909: set once an `Eos` has been enqueued through this adapter. A runner
    /// arm that forwards its own `Eos` after `process(Eos)` returns checks this
    /// so an element that already forwarded one (typically via a catch-all
    /// `other => out.push(other)` arm) does not emit a second.
    eos_forwarded: bool,
    /// M947: the probe of the element pushing through this adapter, when the arm
    /// instruments it. Time spent awaiting capacity here is banked on that probe
    /// so the element's `process()` timing separates its own work from
    /// downstream backpressure. `None` on an uninstrumented adapter (no cost:
    /// the blocking send then takes no extra clock read).
    push_wait_probe: Probe,
    /// In-flight push phase, so `poll_push` runs the pre-send steps exactly
    /// once per packet and a blocked send resumes where it left off.
    push_phase: PushPhase,
    /// M1115: the mutation gate this producer pushes through, when the run was
    /// started with a [`GraphMutator`](crate::runtime::GraphMutator). Adopted
    /// from the link this adapter was built over, and kept across a retarget.
    #[cfg(feature = "std")]
    endpoint: Option<Arc<ProducerEndpoint>>,
    /// M1132: an `Eos` pushed through here is swallowed instead of enqueued. Set
    /// for the length of the flush a remove gives an element it is lifting out:
    /// the element's held frames must reach the consumer, its end of stream must
    /// not, since that would end the run.
    #[cfg(feature = "std")]
    strip_terminal: bool,
}

/// Tell the producer feeding `in_rx` that this sink applies an
/// [`OrientationMeta`](crate::meta::OrientationMeta) itself, so a `videoflip`
/// upstream attaches the descriptor instead of remapping pixels.
///
/// Called while the runner is still wiring arms, not from inside the sink arm:
/// the arms are polled source-first, so an advertisement made once the sink arm
/// runs would arrive a linkful of already-rotated frames late.
#[cfg(feature = "std")]
pub(crate) fn advertise_orientation(in_rx: &LinkReceiver, absorbs: bool) {
    if absorbs {
        in_rx.request_reconfigure(crate::element::Reconfigure::AbsorbOrientation);
    }
}

/// Which [`Reconfigure`](crate::element::Reconfigure) variants a
/// [`SenderSink`]'s producer answers itself. A variant it does not answer goes
/// onto the upstream link when one is wired, and is dropped otherwise: the
/// pre-send check never enqueues the packet it intercepts, so surfacing a signal
/// to a producer that ignores it would cost that packet.
///
/// `Propose` / `Renegotiate` are not listed: every producer answers those.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReconfigureAnswered {
    /// [`Reconfigure::ForceKeyframe`](crate::element::Reconfigure::ForceKeyframe).
    pub keyframe: bool,
    /// [`Reconfigure::AbsorbOrientation`](crate::element::Reconfigure::AbsorbOrientation).
    pub orientation: bool,
}

impl Default for ReconfigureAnswered {
    fn default() -> Self {
        // A source adapter's defaults: a keyframe request reaches the source
        // element (an encoder-less live source may act on it), an orientation
        // advertisement never does, since only a flip answers one.
        ReconfigureAnswered {
            keyframe: true,
            orientation: false,
        }
    }
}

/// See [`SenderSink::push_phase`].
#[derive(Debug, Clone, Copy)]
enum PushPhase {
    /// No push in flight: the next poll runs the pre-send steps.
    Idle,
    /// Past the pre-send steps, awaiting queue capacity: only the enqueue and
    /// its accounting remain. `stamped` records whether a transit stamp was
    /// pushed for this packet (Block links only), so a dead link rolls back
    /// exactly what was stamped.
    Sending {
        bytes: u64,
        blocked_since: Option<u64>,
        stamped: bool,
        // A send that never parked waited no time, so it needs no second clock read.
        parked: bool,
    },
}

impl SenderSink {
    pub fn new(link: LinkSender) -> Self {
        // Share the link's per-edge probe slot, so a tool that installed an
        // interceptor on the edge (via the runner/observer) sees this adapter's
        // packets. A bare link has an empty slot: pass-through, no cost.
        let probe = link.probe.clone();
        #[cfg(feature = "std")]
        let endpoint = link.mutation.clone();
        Self {
            link,
            probe,
            upstream_qos: Vec::new(),
            qos_answered: true,
            upstream_reconfigure: None,
            reconfigure_answered: ReconfigureAnswered::default(),
            upstream_bitrate: None,
            #[cfg(feature = "metadata")]
            meta_stash: None,
            eos_forwarded: false,
            push_wait_probe: None,
            push_phase: PushPhase::Idle,
            #[cfg(feature = "std")]
            endpoint,
            #[cfg(feature = "std")]
            strip_terminal: false,
        }
    }

    /// Bank this adapter's push-wait on `probe`, the producing element's (M947).
    /// The arm calls this right after building the adapter, so the element's
    /// `proc` percentiles report compute and its `push_wait` percentiles report
    /// the backpressure it served.
    #[cfg(feature = "std")]
    pub(crate) fn set_push_wait_probe(&mut self, probe: Probe) {
        self.push_wait_probe = probe;
    }

    /// Charge the time this push spent awaiting capacity to the producing
    /// element's probe.
    fn record_push_wait(&self, blocked_ns: u64) {
        if let Some(probe) = &self.push_wait_probe {
            probe.add_push_wait(blocked_ns);
        }
    }

    /// Whether a blocking send through this adapter needs a pre-send stamp,
    /// which the edge counters and the producer's probe each ask for.
    fn wants_blocked_stamp(&self) -> bool {
        self.link.counters.is_some() || self.push_wait_probe.is_some()
    }

    /// Whether an `Eos` has already been enqueued through this adapter (M909).
    #[cfg(feature = "std")]
    pub(crate) fn eos_forwarded(&self) -> bool {
        self.eos_forwarded
    }

    /// Whether a remove asked the element pushing through here to be flushed
    /// (M1132). Read by the arm when its input closes.
    #[cfg(feature = "std")]
    pub(crate) fn drain_requested(&self) -> bool {
        self.endpoint.as_ref().is_some_and(|e| e.draining())
    }

    /// Whether the mutator took this link away for good: a tee port whose
    /// branch was dropped, so a failed push here is the port going, not an error.
    #[cfg(feature = "std")]
    pub(crate) fn retired(&self) -> bool {
        self.endpoint.as_ref().is_some_and(|e| e.retired())
    }

    /// Swallow any `Eos` pushed from here on, for the length of a flush
    /// (M1132). The frames the element releases still cross; its terminal
    /// marker does not, since the consumer's run continues without it.
    #[cfg(feature = "std")]
    pub(crate) fn strip_terminal(&mut self) {
        self.strip_terminal = true;
    }

    /// Stash the propagated metadata set to attach to outgoing meta-empty
    /// `DataFrame`s (M759). The transform arm replaces it on each new input
    /// frame, passing `None` to clear it (a Drop verdict, so no stale set leaks).
    // Only the graph runner's transform arm sets this, and that runner is std,
    // so without std the setter would be dead code (which the workspace denies).
    #[cfg(all(feature = "metadata", feature = "std"))]
    pub(crate) fn set_meta_stash(&mut self, meta: Option<crate::meta::FrameMetaSet>) {
        self.meta_stash = meta;
    }

    /// A handle to this link's probe slot, for installing/removing a
    /// [`LinkInterceptor`] at runtime.
    pub fn probe(&self) -> ProbeSlot {
        self.probe.clone()
    }

    /// Make this adapter relay any downstream QoS report onto `upstream`, an
    /// input link of the owning element (M175), and also surface it to the
    /// element when `answered` (it
    /// [`handles_qos`](crate::element::AsyncElement::handles_qos)). The runner
    /// wires this so QoS propagates source-ward through a transform, a tee or a
    /// demux, and calls it once per input link for a muxer (M1242).
    #[cfg(feature = "std")]
    pub(crate) fn relay_qos_to(&mut self, upstream: QosSlot, answered: bool) {
        self.upstream_qos.push(upstream);
        self.qos_answered = answered;
    }

    /// Relay onto the upstream link (M720) every downstream `Reconfigure` the
    /// owning transform does not answer itself (`answered`), so a PLI or an
    /// orientation advertisement crosses any number of pass-through transforms
    /// to reach the encoder / the flip.
    #[cfg(feature = "std")]
    pub(crate) fn relay_reconfigure_to(
        &mut self,
        upstream: ReconfigureSlot,
        answered: ReconfigureAnswered,
    ) {
        self.upstream_reconfigure = Some(upstream);
        self.reconfigure_answered = answered;
    }

    /// Relay a downstream bitrate target onto the upstream link (M720).
    #[cfg(feature = "std")]
    pub(crate) fn relay_bitrate_to(&mut self, upstream: BitrateSlot) {
        self.upstream_bitrate = Some(upstream);
    }

    /// Outcome to report once a packet has been enqueued: a pending reverse
    /// signal (reconfigure first, then QoS), else `Accepted`. Reconfigure takes
    /// priority because it is negotiation-critical; QoS is advisory. When a
    /// relay target is set (a transform adapter), an observed QoS is forwarded
    /// upstream, and surfaces as `Qos` only when the transform answers it.
    /// Drain a pending downstream reconfigure. A variant this adapter's
    /// producer answers ([`ReconfigureAnswered`]) is returned for it to observe;
    /// anything else goes onto the upstream link when a relay target is set
    /// (M720 for a PLI, M1058 for an orientation advertisement), so it crosses
    /// pass-through elements, and is otherwise dropped. Shared by the pre-send
    /// check and the post-send outcome, which pass `pre_send` accordingly.
    fn take_reconfigure_or_relay(&self, pre_send: bool) -> Option<crate::element::Reconfigure> {
        use crate::element::Reconfigure;
        let r = self.link.reconfigure.take()?;
        let answered = match &r {
            Reconfigure::ForceKeyframe => self.reconfigure_answered.keyframe,
            Reconfigure::AbsorbOrientation => self.reconfigure_answered.orientation,
            Reconfigure::Propose(_) | Reconfigure::Renegotiate => true,
        };
        if answered {
            // A producer answering `AbsorbOrientation` sends the packet again,
            // because the pre-send check holds it back. Surfacing the same
            // signal after a send would have it resend a packet that already
            // crossed, so hold it for the next push's pre-send check instead.
            if !pre_send && matches!(r, Reconfigure::AbsorbOrientation) {
                self.link.reconfigure.store(r);
                return None;
            }
            return Some(r);
        }
        if let Some(upstream) = &self.upstream_reconfigure {
            upstream.store(r);
        }
        None
    }

    fn post_send_outcome(&self) -> PushOutcome {
        if let Some(r) = self.take_reconfigure_or_relay(false) {
            return PushOutcome::Reconfigure(r);
        }
        if let Some(q) = self.link.qos.take() {
            for upstream in &self.upstream_qos {
                upstream.store(q);
            }
            if self.qos_answered {
                return PushOutcome::Qos(q);
            }
        }
        if let Some(bps) = self.link.bitrate.take() {
            // Lowest priority: surfaced to the immediate producer, or relayed
            // upstream past a non-consuming transform (M720).
            match &self.upstream_bitrate {
                Some(upstream) => upstream.store(bps),
                None => return PushOutcome::Bitrate(bps),
            }
        }
        PushOutcome::Accepted
    }
}

impl SenderSink {
    /// The blocking-send tail of a push: enqueue when capacity frees, then the
    /// accounting and the post-send outcome. `stamped` says whether the Block
    /// path pushed a transit stamp for this packet, so a dead link rolls back
    /// exactly that.
    fn poll_blocking_send(
        &mut self,
        cx: &mut core::task::Context<'_>,
        packet: &mut Option<PipelinePacket>,
        bytes: u64,
        blocked_since: Option<u64>,
        stamped: bool,
        parked: bool,
    ) -> Poll<Result<PushOutcome, G2gError>> {
        match self.link.data.poll_send(cx, packet) {
            Poll::Pending => {
                self.push_phase = PushPhase::Sending {
                    bytes,
                    blocked_since,
                    stamped,
                    parked: true,
                };
                Poll::Pending
            }
            // Post-send check covers the "request fired while we were
            // awaiting capacity" window; the packet is already in the link
            // under old caps.
            Poll::Ready(Ok(())) => {
                self.push_phase = PushPhase::Idle;
                let blocked_ns = match blocked_since {
                    Some(t0) if parked => stamp_now_ns().saturating_sub(t0),
                    _ => 0,
                };
                self.link.record_sent(bytes, blocked_ns);
                self.record_push_wait(blocked_ns);
                Poll::Ready(Ok(self.post_send_outcome()))
            }
            Poll::Ready(Err(SendError::Closed)) => {
                self.push_phase = PushPhase::Idle;
                if stamped {
                    if let Some(ring) = &self.link.transit {
                        ring.lock().pop_back();
                    }
                }
                // The old by-value push dropped an unsent packet with its
                // future; taking it here keeps that.
                packet.take();
                Poll::Ready(Err(G2gError::Shutdown))
            }
            Poll::Ready(Err(SendError::Full)) => unreachable!("poll_send never returns Full"),
        }
    }
}

/// M1115: an arm that ends leaves its link on its mutation endpoint, so a
/// remove waiting for the removed element to drain gets the output link it was
/// pushing through and can hand it to the producer that now bypasses it.
#[cfg(feature = "std")]
impl Drop for SenderSink {
    fn drop(&mut self) {
        if let Some(endpoint) = &self.endpoint {
            endpoint.producer_gone(&self.link);
        }
    }
}

impl OutputSink for SenderSink {
    fn begin_push(&mut self) {
        // A cancelled push may have parked mid-send; its packet died with its
        // future, so the phase must not leak into this push.
        self.push_phase = PushPhase::Idle;
    }

    fn poll_push(
        &mut self,
        cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> Poll<Result<PushOutcome, G2gError>> {
        if let PushPhase::Sending {
            bytes,
            blocked_since,
            stamped,
            parked,
        } = self.push_phase
        {
            return self.poll_blocking_send(cx, packet_slot, bytes, blocked_since, stamped, parked);
        }
        // M1115: the mutation gate, between packets. One relaxed load while no
        // mutation is in flight; a parked producer waits here holding the packet
        // it has not sent, and resumes on whichever link the mutator staged.
        #[cfg(feature = "std")]
        if self.endpoint.as_ref().is_some_and(|e| e.pending()) {
            // Cloned only here, so an ordinary push pays the load and nothing
            // else: the gate needs the endpoint while the link it retargets is
            // borrowed mutably.
            let endpoint = self.endpoint.clone().expect("checked above");
            if endpoint.poll_producer(cx, &mut self.link).is_pending() {
                return Poll::Pending;
            }
            // The link may have been swapped (possibly more than once, if
            // operations came back to back), so the probe is re-read here rather
            // than tracked: this runs once per operation, not per packet.
            self.probe = self.link.probe.clone();
        }
        // M1132: during a flush-on-remove the element's end of stream stops
        // here, ahead of every step that would commit it to the link: the
        // frames it released cross, the marker that would end the run does not.
        #[cfg(feature = "std")]
        if self.strip_terminal && matches!(packet_slot.as_ref(), Some(PipelinePacket::Eos)) {
            packet_slot.take();
            return Poll::Ready(Ok(PushOutcome::Accepted));
        }
        let packet = packet_slot
            .as_mut()
            .expect("poll_push called without a packet");
        // M759: attach the arm's stashed propagated metadata to a fresh
        // output frame (one whose own meta is empty), so a transform that
        // emits new frames still carries the survivors forward.
        // Element-authored meta is never overwritten.
        #[cfg(feature = "metadata")]
        if let (Some(stash), PipelinePacket::DataFrame(frame)) = (&self.meta_stash, &mut *packet) {
            if frame.meta.is_empty() {
                frame.meta = stash.clone();
            }
        }
        // A probe may drop the packet before it ever enters the link.
        if self.probe.action(packet) == ProbeAction::Drop {
            packet_slot.take();
            return Poll::Ready(Ok(PushOutcome::Accepted));
        }
        // Pre-send check: if downstream already requested a
        // reconfigure, surface it before this packet enters the
        // link. Caller renegotiates and decides what to do with
        // `packet` (resend under agreed caps, drop, etc.). A relayed
        // ForceKeyframe hops upstream instead (M720).
        // An Eos is exempt: the producer that would resend the held-back packet
        // has already finished, so holding one back loses it and the consumer
        // waits for an end of stream that never comes.
        if !matches!(packet, PipelinePacket::Eos) {
            if let Some(r) = self.take_reconfigure_or_relay(true) {
                packet_slot.take();
                return Poll::Ready(Ok(PushOutcome::Reconfigure(r)));
            }
        }
        // Past the pre-send checks the packet is committed to the link, so
        // an Eos here is one the consumer will see (M909).
        if matches!(packet, PipelinePacket::Eos) {
            self.eos_forwarded = true;
        }
        // M980: keep the caps this link is carrying, so an observer reads the
        // shape data actually flows under, not just the solved one.
        if let PipelinePacket::CapsChanged(caps) = &*packet {
            if let Some(c) = &self.link.counters {
                c.record_caps(caps);
            }
        }
        // M1115 / M1149: the mutation endpoint follows the shape and the
        // timeline the edge is carrying right now, which is what a splice or a
        // replacement is configured against rather than what negotiation
        // settled.
        #[cfg(feature = "std")]
        if let Some(endpoint) = &self.endpoint {
            endpoint.observe(packet);
        }
        // One clock read serves the age, the transit stamp and the blocked stamp.
        let mut now = LazyNow::default();
        // The frame's age as its element emits it, the number that catches an
        // element buffering frames internally. Skipped for unstamped frames.
        #[cfg(feature = "std")]
        if let (Some(probe), PipelinePacket::DataFrame(frame)) = (&self.push_wait_probe, &*packet) {
            if frame.timing.arrival_ns != 0 {
                probe.record_age_at_emit(now.get().saturating_sub(frame.timing.arrival_ns));
            }
        }
        // Leaky links drop *data frames* under a full channel rather than
        // applying backpressure; control packets (caps / segment / flush /
        // eos) are never dropped, they always block so the stream stays
        // correct. A non-leaky link (the default) always blocks.
        let is_data = matches!(packet, PipelinePacket::DataFrame(_));
        // Measured before the send moves the packet into the link.
        let bytes = packet_bytes(packet);
        if is_data && self.link.policy != LinkPolicy::Block {
            let taken = packet_slot.take().expect("packet checked above");
            match self.link.policy {
                LinkPolicy::DropNewest => match self.link.data.try_send(taken) {
                    Ok(()) => self.link.record_sent(bytes, 0),
                    // Channel full: drop the incoming frame.
                    Err((_dropped, SendError::Full)) => self.link.record_drop(),
                    Err((_v, SendError::Closed)) => return Poll::Ready(Err(G2gError::Shutdown)),
                },
                LinkPolicy::DropOldest => match self.link.data.try_send(taken) {
                    Ok(()) => self.link.record_sent(bytes, 0),
                    Err((returned, SendError::Full)) => {
                        // Evict the oldest queued data frame to make room.
                        // If only control packets are queued, fall back to
                        // blocking rather than dropping a control packet.
                        if self
                            .link
                            .data
                            .evict_front_matching(|p| matches!(p, PipelinePacket::DataFrame(_)))
                            .is_some()
                        {
                            self.link.record_drop();
                            match self.link.data.try_send(returned) {
                                Ok(()) => self.link.record_sent(bytes, 0),
                                Err((_v, SendError::Closed)) => {
                                    return Poll::Ready(Err(G2gError::Shutdown))
                                }
                                Err((_v, SendError::Full)) => {
                                    unreachable!("a slot was just freed by eviction")
                                }
                            }
                        } else {
                            *packet_slot = Some(returned);
                            let blocked_since = self.wants_blocked_stamp().then(|| now.get());
                            self.push_phase = PushPhase::Sending {
                                bytes,
                                blocked_since,
                                stamped: false,
                                parked: false,
                            };
                            return self.poll_blocking_send(
                                cx,
                                packet_slot,
                                bytes,
                                blocked_since,
                                false,
                                false,
                            );
                        }
                    }
                    Err((_v, SendError::Closed)) => return Poll::Ready(Err(G2gError::Shutdown)),
                },
                LinkPolicy::Block => unreachable!("guarded by policy != Block"),
            }
            return Poll::Ready(Ok(self.post_send_outcome()));
        }
        // Transit instrumentation (Block links only, where there are no
        // drops so the stamp ring stays aligned): stamp the frame's queue
        // entry before the send, roll back if it never enqueues.
        let stamped = is_data && self.link.transit.is_some();
        if stamped {
            if let Some(ring) = &self.link.transit {
                ring.lock().push_back(now.get());
            }
        }
        // Stamp before the blocking send so the counters carry how long the
        // producer was held up by a full link (M846), and the producing
        // element's probe can take that wait out of its `process()` timing.
        let blocked_since = self.wants_blocked_stamp().then(|| now.get());
        self.push_phase = PushPhase::Sending {
            bytes,
            blocked_since,
            stamped,
            parked: false,
        };
        self.poll_blocking_send(cx, packet_slot, bytes, blocked_since, stamped, false)
    }
}

#[cfg(test)]
mod link_tests {
    use super::*;
    use crate::caps::{Caps, Dim, Rate, VideoCodec};
    use crate::element::OutputSinkExt;
    use crate::frame::{Frame, FrameTiming};
    use crate::memory::{MemoryDomain, SystemSlice};
    use alloc::boxed::Box;
    use alloc::vec::Vec;
    use core::task::{Context, Poll, Waker};

    fn run_to_ready<F: core::future::Future>(fut: F) -> F::Output {
        let mut cx = Context::from_waker(Waker::noop());
        match core::pin::pin!(fut).poll(&mut cx) {
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
            colorimetry: crate::Colorimetry::UNKNOWN,
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
        assert!(
            rx.try_recv().is_none(),
            "packet must not enqueue when reconfigure pending"
        );
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
        rx.request_qos(QosMessage {
            jitter_ns: 5_000_000,
            running_time_ns: 100,
            taken_ns: 0,
        });
        let outcome = run_to_ready(sink.push(dummy_frame())).expect("push ok");
        match outcome {
            PushOutcome::Qos(q) => {
                assert_eq!(q.jitter_ns, 5_000_000);
                assert_eq!(q.running_time_ns, 100);
            }
            other => panic!("expected Qos, got {other:?}"),
        }
        // Unlike reconfigure, the packet was enqueued (QoS does not hold it back).
        assert!(
            rx.try_recv().is_some(),
            "QoS is advisory; the frame still flowed"
        );
    }

    #[test]
    fn reconfigure_takes_priority_over_qos() {
        let (tx, rx) = link(2);
        let mut sink = SenderSink::new(tx);

        // Both pending: negotiation correctness wins, QoS waits for the next push.
        rx.request_qos(QosMessage {
            jitter_ns: 1_000,
            running_time_ns: 0,
            taken_ns: 0,
        });
        rx.request_reconfigure(Reconfigure::Renegotiate);
        let first = run_to_ready(sink.push(dummy_frame())).unwrap();
        assert!(
            matches!(first, PushOutcome::Reconfigure(_)),
            "reconfigure first"
        );

        let second = run_to_ready(sink.push(dummy_frame())).unwrap();
        assert!(
            matches!(second, PushOutcome::Qos(_)),
            "QoS surfaces once reconfigure drained"
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn an_answered_qos_report_surfaces_and_still_travels_upstream() {
        let report = QosMessage {
            jitter_ns: 5_000_000,
            running_time_ns: 100,
            taken_ns: 0,
        };
        for answered in [true, false] {
            let (up_tx, up_rx) = link(2);
            let (down_tx, down_rx) = link(2);
            // Only the upstream link's reverse slot is used.
            drop(up_tx);
            let mut adapter = SenderSink::new(down_tx);
            adapter.relay_qos_to(up_rx.qos_slot(), answered);

            down_rx.request_qos(report);
            let outcome = run_to_ready(adapter.push(dummy_frame())).expect("push ok");
            let expected = if answered {
                PushOutcome::Qos(report)
            } else {
                PushOutcome::Accepted
            };
            assert_eq!(outcome, expected, "answered={answered}");
            assert_eq!(
                up_rx.qos.take(),
                Some(report),
                "the report reaches the upstream link either way (answered={answered})"
            );
        }
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
    fn is_closed_only_after_the_last_sender_drops() {
        let (tx, rx) = bounded::<u32>(2);
        let second = tx.clone();
        drop(tx);
        assert!(!rx.is_closed(), "one sender remains");
        drop(second);
        assert!(rx.is_closed());
    }

    /// The adapter of a transform that answers keyframe requests but not the
    /// orientation advertisement: the advertisement crosses toward the source,
    /// the keyframe request stops at the element.
    #[cfg(feature = "std")]
    #[test]
    fn relay_is_decided_per_variant() {
        let (up_tx, up_rx) = link(2);
        let (down_tx, down_rx) = link(2);
        // The upstream link's sender is never used; only its reverse slot is.
        drop(up_tx);
        let mut adapter = SenderSink::new(down_tx);
        adapter.relay_reconfigure_to(
            up_rx.reconfigure_slot(),
            ReconfigureAnswered {
                keyframe: true,
                orientation: false,
            },
        );

        down_rx.request_reconfigure(Reconfigure::ForceKeyframe);
        let outcome = run_to_ready(adapter.push(dummy_frame())).expect("push ok");
        assert!(
            matches!(
                outcome,
                PushOutcome::Reconfigure(Reconfigure::ForceKeyframe)
            ),
            "an answered variant surfaces to the producer, got {outcome:?}"
        );
        assert!(
            up_rx.reconfigure.take().is_none(),
            "an answered variant must not also travel upstream"
        );

        down_rx.request_reconfigure(Reconfigure::AbsorbOrientation);
        let outcome = run_to_ready(adapter.push(dummy_frame())).expect("push ok");
        assert_eq!(
            outcome,
            PushOutcome::Accepted,
            "an unanswered variant is relayed, not surfaced"
        );
        assert!(
            matches!(
                up_rx.reconfigure.take(),
                Some(Reconfigure::AbsorbOrientation)
            ),
            "the advertisement must reach the upstream link"
        );
        assert!(
            down_rx.try_recv().is_some(),
            "a relayed variant does not hold the packet back"
        );
    }

    /// The mirror case: a `videoflip`'s adapter answers the advertisement and
    /// relays a keyframe request past itself toward the encoder.
    #[cfg(feature = "std")]
    #[test]
    fn an_answered_orientation_surfaces_while_a_keyframe_relays() {
        let (up_tx, up_rx) = link(2);
        let (down_tx, down_rx) = link(2);
        drop(up_tx);
        let mut adapter = SenderSink::new(down_tx);
        adapter.relay_reconfigure_to(
            up_rx.reconfigure_slot(),
            ReconfigureAnswered {
                keyframe: false,
                orientation: true,
            },
        );

        down_rx.request_reconfigure(Reconfigure::AbsorbOrientation);
        let outcome = run_to_ready(adapter.push(dummy_frame())).expect("push ok");
        assert!(
            matches!(
                outcome,
                PushOutcome::Reconfigure(Reconfigure::AbsorbOrientation)
            ),
            "the flip has to see the advertisement, got {outcome:?}"
        );
        assert!(
            down_rx.try_recv().is_none(),
            "the pre-send check holds the packet back for the producer to resend"
        );

        down_rx.request_reconfigure(Reconfigure::ForceKeyframe);
        let outcome = run_to_ready(adapter.push(dummy_frame())).expect("push ok");
        assert_eq!(outcome, PushOutcome::Accepted);
        assert!(matches!(
            up_rx.reconfigure.take(),
            Some(Reconfigure::ForceKeyframe)
        ));
    }

    /// A held-back packet is the producer's to send again, and nothing sends an
    /// end of stream twice: holding one back would leave the consumer waiting
    /// for a stream end that never comes. Eos skips the pre-send hold.
    #[test]
    fn an_eos_crosses_even_with_a_reconfigure_pending() {
        let (tx, rx) = link(2);
        let mut sink = SenderSink::new(tx);
        rx.request_reconfigure(Reconfigure::AbsorbOrientation);
        let outcome = run_to_ready(sink.push(PipelinePacket::Eos)).expect("push ok");
        assert_eq!(outcome, PushOutcome::Accepted);
        assert!(
            matches!(rx.try_recv(), Some(PipelinePacket::Eos)),
            "the end of stream must still reach the consumer"
        );
    }

    /// Without a relay target (a source's adapter) an unanswered variant is
    /// dropped rather than surfaced: the pre-send check does not enqueue the
    /// packet it intercepts, so handing the signal to a producer that ignores it
    /// would cost that frame.
    #[test]
    fn an_unanswered_variant_without_a_relay_target_is_dropped() {
        let (tx, rx) = link(2);
        let mut adapter = SenderSink::new(tx);
        adapter.reconfigure_answered = ReconfigureAnswered {
            keyframe: true,
            orientation: false,
        };

        rx.request_reconfigure(Reconfigure::AbsorbOrientation);
        let outcome = run_to_ready(adapter.push(dummy_frame())).expect("push ok");
        assert_eq!(outcome, PushOutcome::Accepted);
        assert!(rx.try_recv().is_some(), "the frame still crossed");
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

    #[cfg(feature = "std")]
    fn drained_sequences(rx: &LinkReceiver) -> Vec<u64> {
        let mut got = Vec::new();
        while let Some(PipelinePacket::DataFrame(f)) = rx.try_recv() {
            got.push(f.sequence);
        }
        got
    }

    // Per-edge drop policy is wired only by the std graph runner.
    #[cfg(feature = "std")]
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
            assert_eq!(
                run_to_ready(sink.push(frame_seq(seq))).unwrap(),
                PushOutcome::Accepted
            );
        }
        assert_eq!(
            run_to_ready(sink.push(frame_seq(2))).unwrap(),
            PushOutcome::Accepted
        );

        assert_eq!(
            drained_sequences(&rx),
            [0, 1],
            "drop-newest keeps the oldest"
        );
        assert_eq!(*counter.lock(), 1);
    }

    #[cfg(feature = "std")]
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
        assert_eq!(
            run_to_ready(sink.push(frame_seq(2))).unwrap(),
            PushOutcome::Accepted
        );

        assert_eq!(
            drained_sequences(&rx),
            [1, 2],
            "drop-oldest keeps the newest"
        );
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

    #[cfg(feature = "std")]
    #[test]
    fn leaky_links_never_drop_control_packets() {
        // A capacity-1 leaky link, filled with a data frame. A control packet
        // must not be dropped: with the link full it blocks (Pending) instead.
        let (mut tx, rx) = link(1);
        tx.set_policy(LinkPolicy::DropNewest);
        let mut sink = SenderSink::new(tx);
        run_to_ready(sink.push(frame_seq(0))).unwrap();

        let mut cx = Context::from_waker(Waker::noop());
        let mut fut = core::pin::pin!(sink.push(PipelinePacket::CapsChanged(proposed_caps())));
        assert!(
            matches!(fut.as_mut().poll(&mut cx), Poll::Pending),
            "a control packet blocks on a full leaky link, never dropped"
        );

        // The queued data frame is untouched.
        assert_eq!(drained_sequences(&rx), [0]);
    }

    fn frame_bytes(bytes: usize) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(alloc::vec![0u8; bytes].into())),
            FrameTiming::default(),
            0,
        ))
    }

    #[test]
    fn a_downstream_bitrate_target_reaches_the_producer() {
        const TARGET_BPS: u32 = 2_500_000;
        let (tx, rx) = link(2);
        let mut sink = SenderSink::new(tx);
        rx.request_bitrate(TARGET_BPS);
        let outcome = run_to_ready(sink.push(dummy_frame())).expect("push ok");
        assert_eq!(outcome, PushOutcome::Bitrate(TARGET_BPS));
        let outcome = run_to_ready(sink.push(dummy_frame())).expect("push ok");
        assert_eq!(outcome, PushOutcome::Accepted, "the target is taken once");
    }

    #[cfg(feature = "std")]
    #[test]
    fn a_transform_relays_a_bitrate_target_to_its_producer() {
        const TARGET_BPS: u32 = 2_500_000;
        let (up_tx, up_rx) = link(2);
        let (down_tx, down_rx) = link(2);
        let mut transform_out = SenderSink::new(down_tx);
        transform_out.relay_bitrate_to(up_rx.bitrate_slot());
        let mut producer_out = SenderSink::new(up_tx);

        down_rx.request_bitrate(TARGET_BPS);
        let outcome = run_to_ready(transform_out.push(dummy_frame())).expect("push ok");
        assert_eq!(outcome, PushOutcome::Accepted, "the transform passes it on");
        let outcome = run_to_ready(producer_out.push(dummy_frame())).expect("push ok");
        assert_eq!(outcome, PushOutcome::Bitrate(TARGET_BPS));
    }

    #[cfg(feature = "std")]
    #[test]
    fn edge_counters_count_packets_and_payload_bytes() {
        use crate::caps::TensorDType;
        use crate::memory::SystemView;
        use crate::tensor::TensorView;
        const SYSTEM_BYTES: usize = 12;
        const VIEW_BYTES: usize = 20;
        let (mut tx, rx) = link(4);
        let counters = Arc::new(EdgeCounters::default());
        tx.set_counters(counters.clone());
        let mut sink = SenderSink::new(tx);
        let view = PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::SystemView(SystemView::new(
                Arc::from(alloc::vec![0u8; VIEW_BYTES]),
                TensorView::contiguous(TensorDType::U8, &[VIEW_BYTES as u32]),
            )),
            FrameTiming::default(),
            1,
        ));
        for packet in [
            frame_bytes(SYSTEM_BYTES),
            view,
            PipelinePacket::CapsChanged(proposed_caps()),
        ] {
            run_to_ready(sink.push(packet)).expect("push ok");
        }
        let counts = counters.snapshot();
        assert_eq!(counts.packets, 3);
        assert_eq!(counts.bytes, (SYSTEM_BYTES + VIEW_BYTES) as u64);
        drop(rx);
    }

    #[cfg(feature = "std")]
    #[test]
    fn a_sink_that_applies_orientation_advertises_it_upstream() {
        let (tx, rx) = link(2);
        advertise_orientation(&rx, false);
        assert!(tx.reconfigure.take().is_none());
        advertise_orientation(&rx, true);
        assert!(matches!(
            tx.reconfigure.take(),
            Some(Reconfigure::AbsorbOrientation)
        ));
    }

    #[cfg(feature = "std")]
    #[test]
    fn a_push_held_by_a_full_link_charges_the_wait_to_its_producer() {
        extern crate std;
        let pause = std::time::Duration::from_millis(2);
        let (tx, rx) = link(1);
        let mut sink = SenderSink::new(tx);
        let probe = crate::runtime::instrument::ElementProbe::new("producer".into());
        sink.set_push_wait_probe(Some(probe.clone()));

        run_to_ready(sink.push(frame_seq(0))).expect("room for one");
        assert_eq!(
            probe.record_proc_since(None),
            0,
            "an open link holds nothing"
        );

        let mut cx = Context::from_waker(Waker::noop());
        let mut held = core::pin::pin!(sink.push(frame_seq(1)));
        assert!(held.as_mut().poll(&mut cx).is_pending());
        std::thread::sleep(pause);
        rx.try_recv().expect("the first frame");
        assert!(held.as_mut().poll(&mut cx).is_ready());
        assert!(probe.record_proc_since(None) >= pause.as_nanos() as u64);
    }

    #[cfg(feature = "std")]
    #[test]
    fn a_cancelled_push_leaves_no_send_behind_for_the_next() {
        let (tx, rx) = link(1);
        let mut sink = SenderSink::new(tx);
        run_to_ready(sink.push(frame_seq(0))).expect("room for one");
        {
            let mut cx = Context::from_waker(Waker::noop());
            let mut cancelled = core::pin::pin!(sink.push(frame_seq(1)));
            assert!(cancelled.as_mut().poll(&mut cx).is_pending());
        }
        rx.try_recv().expect("the first frame");
        run_to_ready(sink.push(PipelinePacket::Eos)).expect("room again");
        assert!(sink.eos_forwarded());
    }

    #[cfg(feature = "std")]
    #[test]
    fn only_a_frame_stamped_on_arrival_reports_its_age() {
        let (tx, _rx) = link(4);
        let mut sink = SenderSink::new(tx);
        let probe = crate::runtime::instrument::ElementProbe::new("producer".into());
        sink.set_push_wait_probe(Some(probe.clone()));
        run_to_ready(sink.push(frame_seq(0))).expect("push ok");
        assert_eq!(probe.snapshot().age_at_emit.count, 0);
        let mut stamped = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
            FrameTiming::default(),
            1,
        );
        stamped.timing.arrival_ns = crate::metrics::monotonic_ns();
        run_to_ready(sink.push(PipelinePacket::DataFrame(stamped))).expect("push ok");
        assert_eq!(probe.snapshot().age_at_emit.count, 1);
    }

    #[cfg(feature = "std")]
    #[test]
    fn an_adapter_with_no_mutation_endpoint_is_never_retired() {
        let (tx, _rx) = link(1);
        assert!(!SenderSink::new(tx).retired());
    }

    #[cfg(feature = "std")]
    #[test]
    fn a_producer_endpoint_follows_the_shape_and_timeline_on_its_edge() {
        let endpoint = ProducerEndpoint::new();
        let segment = crate::segment::Segment::new();
        endpoint.observe(&mut PipelinePacket::CapsChanged(proposed_caps()));
        endpoint.observe(&mut PipelinePacket::Segment(segment));
        assert_eq!(endpoint.caps(), Some(proposed_caps()));
        assert_eq!(endpoint.timeline().0, Some(segment));
    }

    #[cfg(feature = "std")]
    fn observed_sequence(endpoint: &ProducerEndpoint, sequence: u64) -> u64 {
        let mut packet = frame_seq(sequence);
        endpoint.observe(&mut packet);
        match packet {
            PipelinePacket::DataFrame(f) => f.sequence,
            other => panic!("{other:?}"),
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn a_producer_taking_over_an_edge_numbers_on_from_what_crossed() {
        const SEEN: u64 = 10;
        let endpoint = ProducerEndpoint::new();
        endpoint.continue_sequence_after(Some(SEEN));
        assert_eq!(observed_sequence(&endpoint, 3), SEEN + 1);
        assert_eq!(observed_sequence(&endpoint, 4), SEEN + 2);
    }

    #[cfg(feature = "std")]
    #[test]
    fn a_rehomed_endpoint_is_live_again_and_numbers_on() {
        const LAST: u64 = 5;
        let endpoint = ProducerEndpoint::new();
        endpoint.observe(&mut PipelinePacket::Segment(crate::segment::Segment::new()));
        assert_eq!(observed_sequence(&endpoint, LAST), LAST);
        endpoint.retire();
        assert!(endpoint.retired());
        endpoint.rehome();
        assert!(!endpoint.retired());
        assert_eq!(endpoint.timeline(), (None, None));
        assert_eq!(observed_sequence(&endpoint, 0), LAST + 1);
    }
}
