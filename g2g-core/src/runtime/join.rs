use core::future::Future;
use core::mem;
use core::pin::Pin;
use core::task::{Context, Poll, Waker};

use alloc::sync::Arc;
use alloc::task::Wake;
use alloc::vec::Vec;

use spin::Mutex;

use crate::element::BoxFuture;
#[cfg(feature = "std")]
use crate::runtime::channel::Receiver;

const ARMS_PER_WORD: usize = u64::BITS as usize;

#[derive(Default)]
struct WokenArms {
    bits: Vec<u64>,
    parent: Option<Waker>,
    // Set once a wake reached the task, cleared when the task polls the join.
    parent_woken: bool,
}

impl WokenArms {
    // The task's waker, when this wake is the first since the task last polled.
    fn mark(&mut self, index: usize) -> Option<Waker> {
        self.bits[index / ARMS_PER_WORD] |= 1 << (index % ARMS_PER_WORD);
        if self.parent_woken {
            return None;
        }
        self.parent_woken = true;
        self.parent.clone()
    }

    fn take_next(&mut self, from: usize) -> Option<usize> {
        let mut word_index = from / ARMS_PER_WORD;
        let mut mask = u64::MAX << (from % ARMS_PER_WORD);
        while let Some(word) = self.bits.get_mut(word_index) {
            let candidates = *word & mask;
            if candidates != 0 {
                let bit = candidates.trailing_zeros() as usize;
                *word &= !(1 << bit);
                return Some(word_index * ARMS_PER_WORD + bit);
            }
            word_index += 1;
            mask = u64::MAX;
        }
        None
    }
}

// One per arm, so a push that wakes one arm re-polls only that arm.
struct ArmWake {
    index: usize,
    woken: Arc<Mutex<WokenArms>>,
}

impl Wake for ArmWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        let parent = self.woken.lock().mark(self.index);
        if let Some(parent) = parent {
            parent.wake();
        }
    }
}

// Allocates only in `new` and `push`, so polling stays heap-free.
struct ArmSet<'a, T> {
    arms: Vec<Option<BoxFuture<'a, T>>>,
    wakers: Vec<Waker>,
    woken: Arc<Mutex<WokenArms>>,
    live: usize,
}

impl<'a, T> ArmSet<'a, T> {
    fn new(arms: Vec<BoxFuture<'a, T>>) -> Self {
        let mut set = Self {
            arms: Vec::with_capacity(arms.len()),
            wakers: Vec::with_capacity(arms.len()),
            woken: Arc::new(Mutex::new(WokenArms::default())),
            live: 0,
        };
        for arm in arms {
            set.push(arm);
        }
        set
    }

    // The new arm starts woken, so the next poll reaches it.
    fn push(&mut self, arm: BoxFuture<'a, T>) {
        let index = self.arms.len();
        self.arms.push(Some(arm));
        self.live += 1;
        self.wakers.push(Waker::from(Arc::new(ArmWake {
            index,
            woken: self.woken.clone(),
        })));
        let mut woken = self.woken.lock();
        woken.bits.resize(index / ARMS_PER_WORD + 1, 0);
        woken.mark(index);
    }

    // In index order, and an arm woken by a lower one during the pass is polled
    // in the same pass, so a frame can cross the whole graph in one poll.
    fn poll_woken(&mut self, cx: &mut Context<'_>, mut on_done: impl FnMut(usize, T)) {
        {
            let mut woken = self.woken.lock();
            let registered = woken
                .parent
                .as_ref()
                .is_some_and(|parent| parent.will_wake(cx.waker()));
            if !registered {
                woken.parent = Some(cx.waker().clone());
            }
            woken.parent_woken = false;
        }
        let mut from = 0;
        loop {
            let Some(index) = self.woken.lock().take_next(from) else {
                return;
            };
            from = index + 1;
            let Some(arm) = &mut self.arms[index] else {
                continue;
            };
            let mut arm_cx = Context::from_waker(&self.wakers[index]);
            if let Poll::Ready(output) = arm.as_mut().poll(&mut arm_cx) {
                self.arms[index] = None;
                self.live -= 1;
                on_done(index, output);
            }
        }
    }

    #[cfg(feature = "std")]
    fn len(&self) -> usize {
        self.arms.len()
    }

    fn all_done(&self) -> bool {
        self.live == 0
    }
}

/// Polls a homogeneous set of boxed futures concurrently to completion,
/// returning their outputs in input order. The fan-out runner joins N+2
/// arms (source + router + N sinks), a count unknown at compile time, so it
/// needs this where [`Join2`] does not fit.
///
/// Unlike [`Join2`] this needs no `unsafe`: [`BoxFuture`] is `Pin<Box<..>>`
/// and therefore `Unpin`, so each arm polls through a plain `&mut`.
#[allow(missing_debug_implementations)]
pub struct JoinAll<'a, T> {
    arms: ArmSet<'a, T>,
    outputs: Vec<Option<T>>,
}

/// Build a [`JoinAll`] over the given boxed futures. `T: Unpin` because the
/// completed outputs are buffered in a `Vec<Option<T>>` polled through a
/// plain `&mut`; every output type used here (`Result<u64, G2gError>`) is
/// `Unpin`.
pub fn join_all<T: Unpin>(futs: Vec<BoxFuture<'_, T>>) -> JoinAll<'_, T> {
    let mut outputs = Vec::with_capacity(futs.len());
    outputs.resize_with(futs.len(), || None);
    JoinAll {
        arms: ArmSet::new(futs),
        outputs,
    }
}

impl<'a, T: Unpin> Future for JoinAll<'a, T> {
    type Output = Vec<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // `JoinAll` is `Unpin` (a `Vec` of `Pin<Box<..>>` and `Option<T>`),
        // so a plain `&mut` is sound.
        let this = self.get_mut();
        let outputs = &mut this.outputs;
        this.arms
            .poll_woken(cx, |index, output| outputs[index] = Some(output));
        if this.arms.all_done() {
            Poll::Ready(
                this.outputs
                    .iter_mut()
                    .map(|o| o.take().expect("JoinAll: arm completed without output"))
                    .collect(),
            )
        } else {
            Poll::Pending
        }
    }
}

/// Like [`JoinAll`], but the arm set can grow *while the join is running*: new
/// boxed futures arriving on a control channel are folded into the poll set on
/// the fly. This is the no-spawn primitive behind runtime request pads (M310):
/// a dynamic fan-out adds a branch arm mid-stream by sending its future here.
///
/// Completion = the control channel is closed (every sender dropped) **and**
/// every arm has resolved. So a runner keeps one control sender alive for as
/// long as branches may still be added (typically until the source ends), then
/// drops it; the join then drains the remaining arms and returns. Outputs are
/// returned in completion order (arm identity is carried in `T`, e.g. a tagged
/// enum, since indices are not stable once the set grows).
#[cfg(feature = "std")]
#[allow(missing_debug_implementations)]
pub(crate) struct DynamicJoin<'a, T> {
    arms: ArmSet<'a, T>,
    /// `None` once the control channel has closed; no more arms can arrive.
    new_arms: Option<Receiver<BoxFuture<'a, T>>>,
    /// Resolved outputs, in the order [`JoinMode`] keeps them.
    outputs: Vec<Option<T>>,
    mode: JoinMode,
}

/// When a [`DynamicJoin`] finishes and in what order its outputs come back.
#[cfg(feature = "std")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum JoinMode {
    /// Finish once the control channel has closed and every arm has resolved;
    /// outputs in completion order. What the dynamic fan-out / fan-in runners
    /// want: their control sender lives on the arm that ends the run, and their
    /// arm identity rides in `T`.
    UntilClosed,
    /// Finish as soon as every arm has resolved, whether or not more could
    /// arrive; outputs in arm order. What the graph mutator (M1115) wants: its
    /// control sender belongs to a handle that outlives the run, and its caller
    /// indexes results by arm position.
    UntilIdle,
}

/// Build a [`DynamicJoin`] from an initial arm set plus a control channel that
/// delivers later arms. See [`DynamicJoin`].
#[cfg(feature = "std")]
pub(crate) fn dynamic_join<'a, T: Unpin>(
    initial: Vec<BoxFuture<'a, T>>,
    new_arms: Receiver<BoxFuture<'a, T>>,
) -> DynamicJoin<'a, T> {
    DynamicJoin {
        arms: ArmSet::new(initial),
        new_arms: Some(new_arms),
        outputs: Vec::new(),
        mode: JoinMode::UntilClosed,
    }
}

/// As [`dynamic_join`], but the join ends as soon as every arm has resolved,
/// whether or not more arms could still be sent, and outputs keep arm order (the
/// initial arms first, in the order given, then each later arm in the order it
/// arrived). The graph mutator (M1115) holds its control sender for as long as
/// the caller holds the handle, which is typically past the end of the run, and
/// a graph whose every arm has ended has nothing left to splice onto.
#[cfg(feature = "std")]
pub(crate) fn dynamic_join_open<'a, T: Unpin>(
    initial: Vec<BoxFuture<'a, T>>,
    new_arms: Receiver<BoxFuture<'a, T>>,
) -> DynamicJoin<'a, T> {
    DynamicJoin {
        arms: ArmSet::new(initial),
        new_arms: Some(new_arms),
        outputs: Vec::new(),
        mode: JoinMode::UntilIdle,
    }
}

#[cfg(feature = "std")]
impl<'a, T: Unpin> Future for DynamicJoin<'a, T> {
    type Output = Vec<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // `DynamicJoin` is `Unpin` (Vecs of `Pin<Box<..>>` / `Option<T>` and a
        // `Receiver`), so a plain `&mut` is sound.
        let this = self.get_mut();

        // 1. Fold in any newly-arrived arms (and notice channel closure). A
        // fresh `recv()` future is created and polled each turn; on `Pending`
        // it leaves the recv waker registered, so a later send / sender-drop
        // re-wakes this join.
        if this.new_arms.is_some() {
            loop {
                let polled = {
                    let rx = this.new_arms.as_ref().expect("checked is_some");
                    let mut rf = rx.recv();
                    Pin::new(&mut rf).poll(cx)
                };
                match polled {
                    Poll::Ready(Some(fut)) => this.arms.push(fut),
                    Poll::Ready(None) => {
                        this.new_arms = None;
                        break;
                    }
                    Poll::Pending => break,
                }
            }
        }

        // 2. Poll every woken arm and buffer its output. `UntilIdle` parks each
        // one in its own arm's slot, so the initial arms keep their input order
        // (the graph runner indexes its per-arm bookkeeping by position);
        // `UntilClosed` appends, which is the completion order its callers have
        // always seen.
        if this.mode == JoinMode::UntilIdle {
            this.outputs.resize_with(this.arms.len(), || None);
        }
        let mode = this.mode;
        let outputs = &mut this.outputs;
        this.arms.poll_woken(cx, |index, output| match mode {
            JoinMode::UntilIdle => outputs[index] = Some(output),
            JoinMode::UntilClosed => outputs.push(Some(output)),
        });

        // 3. Done once every arm has resolved, and, for `UntilClosed`, once no
        // more can arrive. Every slot holding an output is filled by then, so
        // flattening drops nothing.
        let more_may_arrive = this.mode == JoinMode::UntilClosed && this.new_arms.is_some();
        if this.arms.all_done() && !more_may_arrive {
            Poll::Ready(mem::take(&mut this.outputs).into_iter().flatten().collect())
        } else {
            Poll::Pending
        }
    }
}

/// Polls two futures concurrently to completion. Returns both outputs once
/// both have resolved. A tiny stand-in for `futures::future::join` to keep
/// `g2g-core` dependency-free.
#[allow(missing_debug_implementations)]
pub struct Join2<A: Future, B: Future> {
    a: MaybeDone<A>,
    b: MaybeDone<B>,
}

impl<A: Future, B: Future> Join2<A, B> {
    pub fn new(a: A, b: B) -> Self {
        Self {
            a: MaybeDone::Future(a),
            b: MaybeDone::Future(b),
        }
    }
}

enum MaybeDone<F: Future> {
    Future(F),
    Done(F::Output),
    Taken,
}

impl<F: Future> MaybeDone<F> {
    fn take_output(&mut self) -> F::Output {
        match mem::replace(self, MaybeDone::Taken) {
            MaybeDone::Done(v) => v,
            _ => panic!("Join2: take_output on incomplete arm"),
        }
    }
}

impl<A: Future, B: Future> Future for Join2<A, B> {
    type Output = (A::Output, B::Output);

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: We structurally pin `a` and `b`; we never move them out of
        // `self`. `take_output` only runs after both arms have transitioned to
        // `Done`, at which point the inner future has been consumed and only
        // the (move-safe) output remains.
        let this = unsafe { self.get_unchecked_mut() };

        poll_arm(&mut this.a, cx);
        poll_arm(&mut this.b, cx);

        let a_done = matches!(this.a, MaybeDone::Done(_));
        let b_done = matches!(this.b, MaybeDone::Done(_));

        if a_done && b_done {
            Poll::Ready((this.a.take_output(), this.b.take_output()))
        } else {
            Poll::Pending
        }
    }
}

fn poll_arm<F: Future>(arm: &mut MaybeDone<F>, cx: &mut Context<'_>) {
    if let MaybeDone::Future(f) = arm {
        // SAFETY: `f` is pinned because its enclosing `MaybeDone` is pinned
        // by our caller; we only ever obtain `&mut F` long enough to poll it,
        // and on completion we replace the variant in place.
        let pinned = unsafe { Pin::new_unchecked(f) };
        if let Poll::Ready(v) = pinned.poll(cx) {
            *arm = MaybeDone::Done(v);
        }
    }
}

/// Which arm of a [`Select2`] resolved first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Either<A, B> {
    Left(A),
    Right(B),
}

/// Polls two futures concurrently, resolving as soon as *either* is ready.
/// Biased to the first arm: when both are ready in the same poll, `a` wins.
///
/// The losing future is **dropped** when the other resolves. This is only
/// sound for futures that hold no state across a drop and are cheap to
/// recreate, e.g. a channel [`recv()`](crate::runtime::Receiver::recv): a
/// pending `recv` has dequeued nothing, so dropping it loses no message.
/// Do not use it where dropping the un-ready arm would discard progress.
///
/// This is the β interruptibility primitive
/// (design/caps.md): a runner arm awaits its
/// data `recv()` and an out-of-band control `recv()` together, so a
/// coordinator directive reaches the arm at the same await point that
/// otherwise blocks on data. Without it the no_std runtime can only `join`
/// (wait for all), never `select` (wait for first), so an arm parked on
/// `recv().await` is uninterruptible. Put the control arm first to bias the
/// directive ahead of a simultaneously-ready data frame.
#[allow(missing_debug_implementations)]
pub struct Select2<A, B> {
    a: A,
    b: B,
}

/// Build a [`Select2`] over two futures. See the type docs for the
/// drop-the-loser contract.
pub fn select2<A: Future, B: Future>(a: A, b: B) -> Select2<A, B> {
    Select2 { a, b }
}

impl<A: Future, B: Future> Future for Select2<A, B> {
    type Output = Either<A::Output, B::Output>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: `a` and `b` are structurally pinned: `self` is pinned, we
        // never move either field out, and each is polled in place through a
        // freshly derived `Pin`. On `Ready` we return immediately and the
        // caller drops `self` (and the losing future) without moving it.
        let this = unsafe { self.get_unchecked_mut() };
        // SAFETY: `a` is pinned because `this` is; the borrow lasts only for
        // the poll.
        let a = unsafe { Pin::new_unchecked(&mut this.a) };
        if let Poll::Ready(v) = a.poll(cx) {
            return Poll::Ready(Either::Left(v));
        }
        // SAFETY: same justification as `a`.
        let b = unsafe { Pin::new_unchecked(&mut this.b) };
        if let Poll::Ready(v) = b.poll(cx) {
            return Poll::Ready(Either::Right(v));
        }
        Poll::Pending
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::channel::bounded;
    // Only the std-gated BoxFuture tests below construct a `Box`.
    #[cfg(feature = "std")]
    use alloc::boxed::Box;
    use core::future::ready;
    use core::task::Waker;

    /// Poll a future exactly once. The drop of `fut` at return is the point
    /// of the drop-safety test below.
    fn poll_once<F: Future>(fut: F) -> Poll<F::Output> {
        let mut cx = Context::from_waker(Waker::noop());
        core::pin::pin!(fut).poll(&mut cx)
    }

    #[test]
    fn biased_to_left_when_both_ready() {
        match poll_once(select2(ready(1u32), ready(2u32))) {
            Poll::Ready(Either::Left(1)) => {}
            other => panic!("expected Left(1), got {other:?}"),
        }
    }

    #[test]
    fn returns_right_when_left_pending() {
        // Empty, still-open channel: its `recv()` is Pending. The ready
        // right arm wins.
        let (_tx, rx) = bounded::<u32>(1);
        match poll_once(select2(rx.recv(), ready(7u32))) {
            Poll::Ready(Either::Right(7)) => {}
            other => panic!("expected Right(7), got {other:?}"),
        }
    }

    #[test]
    fn pending_when_neither_ready() {
        let (_tx_a, rx_a) = bounded::<u32>(1);
        let (_tx_b, rx_b) = bounded::<u32>(1);
        assert!(poll_once(select2(rx_a.recv(), rx_b.recv())).is_pending());
    }

    #[test]
    fn dropping_the_losing_recv_loses_no_message() {
        // The core soundness claim of the drop-the-loser contract: when the
        // right arm wins, the left `recv()` future is dropped, and because a
        // pending recv has dequeued nothing, the left channel's message is
        // still deliverable afterward.
        let (tx_left, rx_left) = bounded::<u32>(1);
        let (tx_right, rx_right) = bounded::<u32>(1);
        tx_right.try_send(9).unwrap();

        match poll_once(select2(rx_left.recv(), rx_right.recv())) {
            Poll::Ready(Either::Right(Some(9))) => {}
            other => panic!("expected Right(Some(9)), got {other:?}"),
        }

        // The left recv future was dropped by the select above. Its channel
        // never had its message consumed, so a fresh recv still gets it.
        tx_left.try_send(5).unwrap();
        match poll_once(rx_left.recv()) {
            Poll::Ready(Some(5)) => {}
            other => panic!("left message lost across select drop: {other:?}"),
        }
    }

    /// Spin a future to completion with a noop waker. Sound here because every
    /// `DynamicJoin` under test makes progress on each poll (ready arms, queued
    /// control messages, channel closure) rather than waiting on an external
    /// wake.
    #[cfg(feature = "std")]
    fn spin<F: Future>(fut: F) -> F::Output {
        let mut cx = Context::from_waker(Waker::noop());
        let mut fut = core::pin::pin!(fut);
        loop {
            if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
                return v;
            }
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn dynamic_join_folds_in_a_late_arm_then_completes_on_close() {
        let (tx, rx) = bounded::<BoxFuture<'static, i32>>(4);
        // One arm up front; a second delivered over the control channel.
        let initial: Vec<BoxFuture<'static, i32>> = alloc::vec![Box::pin(ready(1))];
        // (BoxFuture isn't Debug, so assert on is_ok rather than unwrap.)
        assert!(tx.try_send(Box::pin(ready(2))).is_ok());
        // Closing the control channel is what lets the join finish.
        drop(tx);

        let mut out = spin(dynamic_join(initial, rx));
        out.sort_unstable();
        assert_eq!(
            out,
            alloc::vec![1, 2],
            "both the initial and the late arm resolve"
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn dynamic_join_stays_pending_while_control_channel_is_open() {
        // An open control channel means more arms may still arrive, so even
        // with every current arm resolved the join must not complete.
        let (tx, rx) = bounded::<BoxFuture<'static, i32>>(1);
        let initial: Vec<BoxFuture<'static, i32>> = alloc::vec![Box::pin(ready(1))];
        assert!(
            poll_once(dynamic_join(initial, rx)).is_pending(),
            "open control channel keeps the join alive"
        );
        drop(tx);
    }

    #[cfg(feature = "std")]
    struct CountPolls<'a>(&'a core::cell::Cell<u32>);

    #[cfg(feature = "std")]
    impl Future for CountPolls<'_> {
        type Output = ();
        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
            self.0.set(self.0.get() + 1);
            Poll::Pending
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn a_wake_polls_only_the_woken_arm() {
        let (tx, rx) = bounded::<u32>(1);
        let idle_polls = core::cell::Cell::new(0);
        let woken: BoxFuture<'_, ()> = Box::pin(async {
            rx.recv().await;
        });
        let idle: BoxFuture<'_, ()> = Box::pin(CountPolls(&idle_polls));
        let mut join = join_all(alloc::vec![woken, idle]);
        let mut cx = Context::from_waker(Waker::noop());
        assert!(Pin::new(&mut join).poll(&mut cx).is_pending());
        assert_eq!(idle_polls.get(), 1, "the first poll reaches every arm");

        tx.try_send(1).unwrap();
        assert!(Pin::new(&mut join).poll(&mut cx).is_pending());
        assert_eq!(idle_polls.get(), 1, "the send woke the other arm only");
    }
}
