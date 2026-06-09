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
