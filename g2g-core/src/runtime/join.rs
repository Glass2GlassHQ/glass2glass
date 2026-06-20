use core::future::Future;
use core::mem;
use core::pin::Pin;
use core::task::{Context, Poll};

use alloc::vec::Vec;

use crate::element::BoxFuture;

/// Polls a homogeneous set of boxed futures concurrently to completion,
/// returning their outputs in input order. The fan-out runner joins N+2
/// arms (source + router + N sinks), a count unknown at compile time, so it
/// needs this where [`Join2`] does not fit.
///
/// Unlike [`Join2`] this needs no `unsafe`: [`BoxFuture`] is `Pin<Box<..>>`
/// and therefore `Unpin`, so each arm polls through a plain `&mut`.
#[allow(missing_debug_implementations)]
pub struct JoinAll<'a, T> {
    arms: Vec<Option<BoxFuture<'a, T>>>,
    outputs: Vec<Option<T>>,
}

/// Build a [`JoinAll`] over the given boxed futures. `T: Unpin` because the
/// completed outputs are buffered in a `Vec<Option<T>>` polled through a
/// plain `&mut`; every output type used here (`Result<u64, G2gError>`) is
/// `Unpin`.
pub fn join_all<T: Unpin>(futs: Vec<BoxFuture<'_, T>>) -> JoinAll<'_, T> {
    let mut outputs = Vec::with_capacity(futs.len());
    outputs.resize_with(futs.len(), || None);
    JoinAll { arms: futs.into_iter().map(Some).collect(), outputs }
}

impl<'a, T: Unpin> Future for JoinAll<'a, T> {
    type Output = Vec<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // `JoinAll` is `Unpin` (a `Vec` of `Pin<Box<..>>` and `Option<T>`),
        // so a plain `&mut` is sound.
        let this = self.get_mut();
        let mut all_done = true;
        for (arm, slot) in this.arms.iter_mut().zip(this.outputs.iter_mut()) {
            if let Some(fut) = arm {
                match fut.as_mut().poll(cx) {
                    Poll::Ready(v) => {
                        *slot = Some(v);
                        *arm = None;
                    }
                    Poll::Pending => all_done = false,
                }
            }
        }
        if all_done {
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
/// (DESIGN.md §4.13.5): a runner arm awaits its
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
    use core::future::ready;
    use core::task::{RawWaker, RawWakerVTable, Waker};

    // Hand-rolled noop waker: the futures under test resolve (or stay
    // pending) within a single poll, so no real wake is ever needed.
    static NOOP_VTABLE: RawWakerVTable = RawWakerVTable::new(
        |_| RawWaker::new(core::ptr::null(), &NOOP_VTABLE),
        |_| {},
        |_| {},
        |_| {},
    );

    fn noop_waker() -> Waker {
        // SAFETY: every NOOP_VTABLE fn is a no-op that never dereferences the
        // data pointer, so a null data pointer is sound.
        unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &NOOP_VTABLE)) }
    }

    /// Poll a future exactly once. The drop of `fut` at return is the point
    /// of the drop-safety test below.
    fn poll_once<F: Future>(fut: F) -> Poll<F::Output> {
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        let mut fut = fut;
        // SAFETY: `fut` lives on this stack frame and is not moved after
        // pinning; it is dropped when the frame returns.
        let pinned = unsafe { Pin::new_unchecked(&mut fut) };
        pinned.poll(&mut cx)
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
}
