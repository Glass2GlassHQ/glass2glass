// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

/// Yield once to the executor. Deliberately not `tokio::task::yield_now`: the
/// audio sink features do not pull tokio, and this works on the cooperative
/// runner and a tokio runtime alike.
///
/// The waker fires immediately, so a retry loop built on this polls hot rather
/// than sleeping: waiting on a full queue or on playout costs CPU. That is the
/// price of a cooperative single-threaded executor with no timer primitive, and
/// it buys the thing that matters, that no other arm is starved.
pub(crate) struct YieldNow(bool);

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

pub(crate) fn yield_now() -> YieldNow {
    YieldNow(false)
}
