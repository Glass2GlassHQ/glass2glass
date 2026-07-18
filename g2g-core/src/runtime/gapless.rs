//! Gapless playback channel (M383): the app <-> source side-channel a
//! [`GaplessSrc`](../../../g2g_plugins/gaplesssrc/struct.GaplessSrc.html) uses to
//! concatenate a playlist of sources into one continuous stream, the analog of
//! GStreamer playbin's `about-to-finish` signal + next-`uri` enqueue.
//!
//! A gapless source plays its current item, and when nothing is queued behind it
//! posts **about-to-finish** so the app can enqueue the next item *during*
//! playback (so the swap is seamless); on the current item's EOS the source pulls
//! the next from the queue and continues, rebasing timestamps onto the existing
//! timeline (no gap, no flush, the decode chain downstream reused). When the app
//! has no more items it calls [`finish`](GaplessController::finish) and the source
//! emits a single terminal `Eos`.
//!
//! This is the source-swap counterpart of the M358 segment loop: that loops *one*
//! item via a `SEGMENT` seek; this concatenates *different* items. Both are
//! poll-based app <-> source channels with a wakeful idle (the source parks on
//! [`wait_event`](GaplessController::wait_event) between items rather than
//! busy-polling), mirroring [`SeekController`](crate::runtime::SeekController).

use core::fmt;
use core::sync::atomic::{AtomicBool, Ordering};
use core::task::{Context, Poll, Waker};

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::sync::Arc;

use spin::Mutex;

use core::future::Future;
use core::pin::Pin;

use crate::runtime::fanin::DynSourceLoop;

/// Source -> app about-to-finish state. `count` is monotonic so the app can
/// detect a *new* signal (spin until it advances); `fresh` is the take-once flag
/// for [`take_about_to_finish`](GaplessController::take_about_to_finish).
#[derive(Debug, Default, Clone, Copy)]
struct AboutToFinish {
    /// Number of times the source has signaled about-to-finish (monotonic).
    count: u64,
    /// Set by `notify_about_to_finish`, cleared by `take_about_to_finish`.
    fresh: bool,
}

#[derive(Default)]
struct GaplessInner {
    /// App -> source playlist: the next sources to play, in order. The app
    /// enqueues; the source pops the front when its current item ends.
    queue: Mutex<VecDeque<Box<dyn DynSourceLoop>>>,
    /// Source -> app about-to-finish back-channel (the `about-to-finish` analog).
    about: Mutex<AboutToFinish>,
    /// App -> source end request: no more items will be enqueued, so the source
    /// emits `Eos` once the queue drains. Latches once set.
    finished: AtomicBool,
    /// App -> source *instant* switch (M384, the `instant-uri` analog): play this
    /// source **now**, preempting the current item (a flush + timeline reset),
    /// rather than waiting for the current item's EOS like the gapless `queue`.
    /// Latest-wins (a newer switch replaces an untaken one).
    instant: Mutex<Option<Box<dyn DynSourceLoop>>>,
    /// Waker a source parked in [`GaplessController::wait_event`] /
    /// [`wait_instant`](GaplessController::wait_instant) registered, woken by
    /// `enqueue` / `finish` / `switch_now` so an idle or playing source resumes
    /// without busy-polling. `None` when no source is parked.
    waker: Mutex<Option<Waker>>,
}

/// Cloneable gapless-playback channel. Every clone shares one playlist queue, one
/// about-to-finish back-channel, and one finished flag. The app holds one handle;
/// a clone lives in the [`GaplessSrc`] (the same app-holds / source-holds-a-clone
/// shape as [`SeekController`](crate::runtime::SeekController)).
#[derive(Clone, Default)]
pub struct GaplessController {
    inner: Arc<GaplessInner>,
}

impl fmt::Debug for GaplessController {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The queued sources are not `Debug`; report the counts instead.
        f.debug_struct("GaplessController")
            .field("queued", &self.queued())
            .field("about_to_finish_count", &self.about_to_finish_count())
            .field("finished", &self.is_finished())
            .field("instant_pending", &self.has_instant())
            .finish()
    }
}

impl GaplessController {
    /// A controller with an empty playlist.
    pub fn new() -> Self {
        Self::default()
    }

    /// Application side: enqueue the next source to play (a constructed, *not yet*
    /// configured [`DynSourceLoop`]; the [`GaplessSrc`] negotiates and configures
    /// it before playing). Items play in enqueue order.
    pub fn enqueue(&self, source: Box<dyn DynSourceLoop>) {
        self.inner.queue.lock().push_back(source);
        self.wake();
    }

    /// Source side: take the next queued source, or `None` if the playlist is
    /// empty.
    pub fn take_next(&self) -> Option<Box<dyn DynSourceLoop>> {
        self.inner.queue.lock().pop_front()
    }

    /// Whether at least one source is queued (source side: is there an item to
    /// play after the current one).
    pub fn has_next(&self) -> bool {
        !self.inner.queue.lock().is_empty()
    }

    /// Number of sources currently queued.
    pub fn queued(&self) -> usize {
        self.inner.queue.lock().len()
    }

    /// Source side: signal that the current item is about to finish and nothing is
    /// queued behind it (the `about-to-finish` signal). The app reacts by
    /// [`enqueue`](Self::enqueue)ing the next item (or [`finish`](Self::finish)ing).
    pub fn notify_about_to_finish(&self) {
        let mut a = self.inner.about.lock();
        a.count = a.count.saturating_add(1);
        a.fresh = true;
    }

    /// Application side: whether the source posted a *new* about-to-finish since
    /// the last take (take-once). The app enqueues the next item in response.
    pub fn take_about_to_finish(&self) -> bool {
        let mut a = self.inner.about.lock();
        if a.fresh {
            a.fresh = false;
            true
        } else {
            false
        }
    }

    /// Application side: total about-to-finish signals the source has posted
    /// (monotonic), so the app can wait for the next without consuming the
    /// take-once slot.
    pub fn about_to_finish_count(&self) -> u64 {
        self.inner.about.lock().count
    }

    /// Application side: declare the playlist complete. Once the queue drains the
    /// source emits a single terminal `Eos`. Latching (a finished playlist is not
    /// reopened).
    pub fn finish(&self) {
        self.inner.finished.store(true, Ordering::SeqCst);
        self.wake();
    }

    /// Source side: whether the app has declared the playlist complete.
    pub fn is_finished(&self) -> bool {
        self.inner.finished.load(Ordering::SeqCst)
    }

    /// Application side: switch to `source` **now** (M384, the `instant-uri`
    /// analog), preempting the current item with a flush and a timeline reset
    /// rather than waiting for its EOS (the gapless [`enqueue`](Self::enqueue)
    /// path). Latest-wins: a newer `switch_now` replaces an untaken one. Use this
    /// for a user-driven "play this instead", `enqueue` for seamless playlists.
    pub fn switch_now(&self, source: Box<dyn DynSourceLoop>) {
        *self.inner.instant.lock() = Some(source);
        self.wake();
    }

    /// Source side: take the pending instant-switch source, or `None` if none is
    /// pending.
    pub fn take_instant(&self) -> Option<Box<dyn DynSourceLoop>> {
        self.inner.instant.lock().take()
    }

    /// Whether an instant switch is pending (not yet taken).
    pub fn has_instant(&self) -> bool {
        self.inner.instant.lock().is_some()
    }

    /// Source side: park until a source is enqueued, an instant switch is
    /// requested, or the playlist is finished, then resolve. The wakeful idle
    /// between items (the poll-free analog of pausing the source task):
    /// `enqueue` / `finish` / `switch_now` wake the registered waker. The caller
    /// re-checks `take_instant` / `take_next` / `is_finished` after it resolves (a
    /// spurious early resolve is harmless, just a re-poll).
    pub fn wait_event(&self) -> GaplessWait<'_> {
        GaplessWait { ctl: self }
    }

    /// Source side: park until an instant switch is requested, then resolve.
    /// Used *during* an item's playback (selected against the item's `run`) so a
    /// `switch_now` preempts it; a plain `enqueue` does not (that is gapless and
    /// waits for EOS). `switch_now` wakes the registered waker.
    pub fn wait_instant(&self) -> GaplessInstantWait<'_> {
        GaplessInstantWait { ctl: self }
    }

    /// Resolve any parked [`wait_event`](Self::wait_event) /
    /// [`wait_instant`](Self::wait_instant).
    fn wake(&self) {
        if let Some(w) = self.inner.waker.lock().take() {
            w.wake();
        }
    }

    /// Whether an item is queued, an instant switch is pending, or the playlist is
    /// finished (the `wait_event` ready condition).
    fn has_event(&self) -> bool {
        self.has_next() || self.has_instant() || self.is_finished()
    }
}

/// Future returned by [`GaplessController::wait_event`]. Resolves when a source is
/// enqueued, an instant switch is requested, or the playlist is finished.
#[derive(Debug)]
pub struct GaplessWait<'a> {
    ctl: &'a GaplessController,
}

impl Future for GaplessWait<'_> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.ctl.has_event() {
            return Poll::Ready(());
        }
        // Register before the final check so an `enqueue` / `finish` / `switch_now`
        // that lands in the gap still wakes us (it either sees the waker, or we
        // see its state).
        *self.ctl.inner.waker.lock() = Some(cx.waker().clone());
        if self.ctl.has_event() {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

/// Future returned by [`GaplessController::wait_instant`]. Resolves when an
/// instant switch is requested (`switch_now`), so a playing item can be preempted.
#[derive(Debug)]
pub struct GaplessInstantWait<'a> {
    ctl: &'a GaplessController,
}

impl Future for GaplessInstantWait<'_> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.ctl.has_instant() {
            return Poll::Ready(());
        }
        *self.ctl.inner.waker.lock() = Some(cx.waker().clone());
        if self.ctl.has_instant() {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::Caps;
    use crate::element::ConfigureOutcome;
    use crate::element::OutputSink;
    use crate::error::G2gError;
    use crate::runtime::SourceLoop;

    /// A no-op source so the queue has something boxed to hold.
    #[derive(Debug)]
    struct NullSrc;
    impl SourceLoop for NullSrc {
        type RunFuture<'a> = core::future::Ready<Result<u64, G2gError>>;
        type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>;
        fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
            core::future::ready(Err(G2gError::NotConfigured))
        }
        fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
            Ok(ConfigureOutcome::Accepted)
        }
        fn run<'a>(&'a mut self, _out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
            core::future::ready(Ok(0))
        }
    }

    #[test]
    fn enqueue_and_take_are_fifo() {
        let app = GaplessController::new();
        let src = app.clone();
        assert!(!src.has_next());
        app.enqueue(Box::new(NullSrc));
        app.enqueue(Box::new(NullSrc));
        assert_eq!(src.queued(), 2);
        assert!(src.take_next().is_some());
        assert!(src.take_next().is_some());
        assert!(src.take_next().is_none(), "drained");
    }

    #[test]
    fn about_to_finish_counts_and_takes_once() {
        let app = GaplessController::new();
        let src = app.clone();
        assert_eq!(app.about_to_finish_count(), 0);
        assert!(!app.take_about_to_finish());
        src.notify_about_to_finish();
        assert_eq!(app.about_to_finish_count(), 1);
        assert!(app.take_about_to_finish());
        assert!(!app.take_about_to_finish(), "take is once per notify");
    }

    #[test]
    fn finish_latches_and_is_visible_to_the_source() {
        let app = GaplessController::new();
        let src = app.clone();
        assert!(!src.is_finished());
        app.finish();
        assert!(src.is_finished());
        assert!(src.is_finished(), "latches");
    }

    #[test]
    fn switch_now_is_latest_wins_and_taken_once() {
        let app = GaplessController::new();
        let src = app.clone();
        assert!(!src.has_instant());
        app.switch_now(Box::new(NullSrc));
        app.switch_now(Box::new(NullSrc)); // replaces the untaken one
        assert!(src.has_instant());
        assert!(src.take_instant().is_some());
        assert!(!src.has_instant(), "single pending slot");
        assert!(src.take_instant().is_none());
    }

    #[test]
    fn wait_instant_resolves_only_on_switch_now() {
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        let c = GaplessController::new();
        let mut fut = core::pin::pin!(c.wait_instant());
        assert!(fut.as_mut().poll(&mut cx).is_pending(), "no instant yet");
        // A plain enqueue does NOT resolve wait_instant (that is the gapless path).
        c.enqueue(Box::new(NullSrc));
        assert!(
            fut.as_mut().poll(&mut cx).is_pending(),
            "enqueue is not an instant switch"
        );
        // switch_now does.
        c.switch_now(Box::new(NullSrc));
        assert!(
            fut.as_mut().poll(&mut cx).is_ready(),
            "switch_now resolves wait_instant"
        );
    }

    #[test]
    fn wait_event_resolves_on_enqueue_and_finish() {
        use crate::runtime::block_on;
        // Already-ready: an enqueued item resolves immediately.
        let c = GaplessController::new();
        c.enqueue(Box::new(NullSrc));
        block_on(c.wait_event());

        // A parked wait resolves when the playlist is finished.
        let c2 = GaplessController::new();
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        let mut fut = core::pin::pin!(c2.wait_event());
        assert!(fut.as_mut().poll(&mut cx).is_pending(), "no event yet");
        c2.finish();
        assert!(
            fut.as_mut().poll(&mut cx).is_ready(),
            "finish resolves the wait"
        );
    }

    fn noop_waker() -> Waker {
        use core::task::{RawWaker, RawWakerVTable};
        const VT: RawWakerVTable = RawWakerVTable::new(
            |_| RawWaker::new(core::ptr::null(), &VT),
            |_| {},
            |_| {},
            |_| {},
        );
        // SAFETY: the vtable's clone returns a valid RawWaker over the same
        // vtable, and wake/wake_by_ref/drop are no-ops on a null data pointer.
        unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &VT)) }
    }
}
