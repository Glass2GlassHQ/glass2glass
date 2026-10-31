use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicBool, Ordering};
use core::task::{Context, Poll, Waker};

extern crate std;

use alloc::sync::Arc;
use alloc::vec::Vec;

use std::sync::Mutex;

#[derive(Debug, Default)]
struct RunEndInner {
    requested: AtomicBool,
    waiting_sources: Mutex<Vec<Waker>>,
}

// a sink that can present nothing more sets this, and every source then ends its stream so the run drains
#[derive(Debug, Clone, Default)]
pub(crate) struct RunEndRequest {
    inner: Arc<RunEndInner>,
}

impl RunEndRequest {
    pub(crate) fn request(&self) {
        self.inner.requested.store(true, Ordering::SeqCst);
        let waiting = core::mem::take(&mut *self.inner.waiting_sources.lock().unwrap());
        for waker in waiting {
            waker.wake();
        }
    }

    pub(crate) fn requested(&self) -> RunEndRequested<'_> {
        RunEndRequested { request: self }
    }

    fn is_requested(&self) -> bool {
        self.inner.requested.load(Ordering::SeqCst)
    }
}

#[derive(Debug)]
pub(crate) struct RunEndRequested<'r> {
    request: &'r RunEndRequest,
}

impl Future for RunEndRequested<'_> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.request.is_requested() {
            return Poll::Ready(());
        }
        let mut waiting = self.request.inner.waiting_sources.lock().unwrap();
        if !waiting.iter().any(|waker| waker.will_wake(cx.waker())) {
            waiting.push(cx.waker().clone());
        }
        drop(waiting);
        // a request between the first check and the registration would otherwise go unseen
        if self.request.is_requested() {
            return Poll::Ready(());
        }
        Poll::Pending
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::join::{select2, Either};

    #[test]
    fn a_request_wakes_a_waiting_source() {
        let request = RunEndRequest::default();
        let waiter = request.clone();
        let ended = std::thread::spawn(move || {
            crate::runtime::block_on(async { waiter.requested().await });
        });
        request.request();
        ended.join().expect("the waiting source resolves");
    }

    #[test]
    fn an_unrequested_run_keeps_its_sources_running() {
        let request = RunEndRequest::default();
        let first = crate::runtime::block_on(select2(request.requested(), core::future::ready(())));
        assert!(matches!(first, Either::Right(())));
    }
}
