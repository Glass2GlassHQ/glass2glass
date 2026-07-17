//! Seek request channel (M82) and segment-done back-channel (M358).
//!
//! A cloneable handle carrying a pending [`Seek`] from the application to a
//! seek-aware source. The app calls [`SeekController::seek`]; the source's run
//! loop calls [`SeekController::take_pending`] between frames and, on a flushing
//! seek, emits `Flush`, repositions, emits the post-flush
//! [`Segment`](crate::segment::Segment), and resumes from the new position.
//!
//! Seeks travel upstream to the source in GStreamer; here the app holds the
//! controller and a clone lives in the source, so a seek reaches the producer
//! without a back-reference. The latest request wins: an app scrubbing fast
//! only needs the final target, so a new `seek` replaces any prior unhandled
//! one. Polling (rather than waking) is deliberate — a producing source checks
//! between frames; a parked source isn't producing, so there is nothing to
//! reposition until it resumes.
//!
//! ## Segment seeks (M358)
//!
//! A [`SeekFlags::SEGMENT`](crate::segment::SeekFlags::SEGMENT) seek asks the
//! source to play `[start, stop]` and, on reaching `stop`, report
//! **segment-done** instead of running to `Eos`, so the app can loop seamlessly
//! with a non-flushing accumulating seek (gapless, see
//! [`Segment::accumulate_seek`](crate::segment::Segment::accumulate_seek)). g2g
//! has no `PipelinePacket` for this (the GStreamer `SEGMENT_DONE` event/message);
//! it would force a new control packet through every element's exhaustive match.
//! Instead the controller carries it back on the same app<->source channel that
//! already exists: the source calls [`notify_segment_done`](Self::notify_segment_done)
//! at `stop`, the app observes [`segment_done_count`](Self::segment_done_count) /
//! [`take_segment_done`](Self::take_segment_done) and either re-arms a loop seek
//! or calls [`shutdown`](Self::shutdown) to end the loop. A segment-looping
//! source that is idle between loops polls [`is_shutdown`](Self::is_shutdown) so
//! it can emit `Eos` and terminate (the poll-model analog of pausing the source
//! task; a wakeful wait is a follow-up).

use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicBool, Ordering};
use core::task::{Context, Poll, Waker};

use alloc::sync::Arc;

use spin::Mutex;

use crate::segment::Seek;

/// Source -> app segment-done state. `count` is monotonic so the app can detect
/// a *new* completion (e.g. spin until it advances); `fresh` is the take-once
/// flag for [`take_segment_done`](SeekController::take_segment_done).
#[derive(Debug, Default, Clone, Copy)]
struct SegmentDone {
    /// Number of `SEGMENT` segments the source has completed (monotonic).
    count: u64,
    /// Stream-time position (ns) where the most recent segment ended.
    position_ns: u64,
    /// Set by `notify_segment_done`, cleared by `take_segment_done`.
    fresh: bool,
}

#[derive(Debug, Default)]
struct SeekInner {
    pending: Mutex<Option<Seek>>,
    /// Segment-done back-channel (source -> app), the `SEGMENT_DONE` analog.
    segment_done: Mutex<SegmentDone>,
    /// App -> source stop request: ends a segment-looping source idling between
    /// loop seeks. Latches once set (a run is torn down, not resumed).
    shutdown: AtomicBool,
    /// Waker a source parked in [`SeekController::wait_event`] registered, woken
    /// by `seek` / `shutdown` so an idle segment-looping source resumes without
    /// busy-polling. `None` when no source is parked.
    waker: Mutex<Option<Waker>>,
}

/// Cloneable seek channel. Every clone shares one pending-seek slot, one
/// segment-done back-channel, and one shutdown flag.
#[derive(Debug, Clone, Default)]
pub struct SeekController {
    inner: Arc<SeekInner>,
}

impl SeekController {
    /// A controller with no pending seek.
    pub fn new() -> Self {
        Self::default()
    }

    /// Application side: request a seek. Replaces any prior unhandled request
    /// (latest-wins).
    pub fn seek(&self, seek: Seek) {
        *self.inner.pending.lock() = Some(seek);
        self.wake();
    }

    /// Source side: take and clear the pending seek, or `None` if none is set.
    pub fn take_pending(&self) -> Option<Seek> {
        self.inner.pending.lock().take()
    }

    /// Whether a seek is currently pending (not yet taken).
    pub fn has_pending(&self) -> bool {
        self.inner.pending.lock().is_some()
    }

    /// Source side: report that a `SEGMENT` segment finished at stream-time
    /// `position_ns` (the `SEGMENT_DONE` signal). Bumps the completion count and
    /// arms [`take_segment_done`](Self::take_segment_done).
    pub fn notify_segment_done(&self, position_ns: u64) {
        let mut d = self.inner.segment_done.lock();
        d.count = d.count.saturating_add(1);
        d.position_ns = position_ns;
        d.fresh = true;
    }

    /// Application side: take the stream-time position of the most recent
    /// segment-done, or `None` if none is unconsumed since the last take. The
    /// app reacts by re-arming a (typically non-flushing) loop seek.
    pub fn take_segment_done(&self) -> Option<u64> {
        let mut d = self.inner.segment_done.lock();
        if d.fresh {
            d.fresh = false;
            Some(d.position_ns)
        } else {
            None
        }
    }

    /// Application side: total `SEGMENT` segments the source has completed
    /// (monotonic). Lets the app wait for the next completion without consuming
    /// the take-once slot.
    pub fn segment_done_count(&self) -> u64 {
        self.inner.segment_done.lock().count
    }

    /// Application side: ask a segment-looping source to stop and emit `Eos`.
    /// Latching: a torn-down run is not resumed.
    pub fn shutdown(&self) {
        self.inner.shutdown.store(true, Ordering::SeqCst);
        self.wake();
    }

    /// Source side: whether the app has requested shutdown.
    pub fn is_shutdown(&self) -> bool {
        self.inner.shutdown.load(Ordering::SeqCst)
    }

    /// Source side: park until a seek is pending or shutdown is requested, then
    /// resolve. The wakeful idle for a segment-looping source between loops (the
    /// poll-free analog of pausing the source task): `seek` / `shutdown` wake the
    /// registered waker. The caller re-checks `take_pending` / `is_shutdown`
    /// after it resolves (a spurious early resolve is harmless, just a re-poll).
    pub fn wait_event(&self) -> WaitEvent<'_> {
        WaitEvent { ctl: self }
    }

    /// Resolve any parked [`wait_event`](Self::wait_event).
    fn wake(&self) {
        if let Some(w) = self.inner.waker.lock().take() {
            w.wake();
        }
    }

    /// Whether a seek is pending or shutdown is set (the `wait_event` ready
    /// condition).
    fn has_event(&self) -> bool {
        self.has_pending() || self.is_shutdown()
    }
}

/// Future returned by [`SeekController::wait_event`]. Resolves when a seek is
/// pending or shutdown is requested.
#[derive(Debug)]
pub struct WaitEvent<'a> {
    ctl: &'a SeekController,
}

impl Future for WaitEvent<'_> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.ctl.has_event() {
            return Poll::Ready(());
        }
        // Register before the final check so a `seek` / `shutdown` that lands in
        // the gap still wakes us (it either sees the waker, or we see its state).
        *self.ctl.inner.waker.lock() = Some(cx.waker().clone());
        if self.ctl.has_event() {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment::{Seek, SeekFlags, SeekType};

    #[test]
    fn take_pending_clears_and_returns() {
        let c = SeekController::new();
        assert!(!c.has_pending());
        assert_eq!(c.take_pending(), None);

        c.seek(Seek::flush_to(5_000));
        assert!(c.has_pending());
        let s = c.take_pending().expect("a seek was pending");
        assert_eq!(s.start, 5_000);
        assert!(s.is_flush());
        // Taken: now empty.
        assert!(!c.has_pending());
        assert_eq!(c.take_pending(), None);
    }

    #[test]
    fn latest_seek_wins() {
        let c = SeekController::new();
        c.seek(Seek::flush_to(1_000));
        c.seek(Seek {
            rate: 1.0,
            flags: SeekFlags::FLUSH,
            start_type: SeekType::Set,
            start: 9_000,
            stop_type: SeekType::None,
            stop: 0,
        });
        // Only the final request survives.
        assert_eq!(c.take_pending().map(|s| s.start), Some(9_000));
    }

    #[test]
    fn clones_share_one_slot() {
        let app = SeekController::new();
        let src = app.clone();
        app.seek(Seek::flush_to(42));
        // The source-side clone observes the app-side request.
        assert_eq!(src.take_pending().map(|s| s.start), Some(42));
        assert!(!app.has_pending());
    }

    #[test]
    fn segment_done_counts_and_takes_once() {
        let app = SeekController::new();
        let src = app.clone();
        assert_eq!(app.segment_done_count(), 0);
        assert_eq!(app.take_segment_done(), None);

        // Source reports a completed segment; the app sees the count advance and
        // can take the position exactly once.
        src.notify_segment_done(5_000);
        assert_eq!(app.segment_done_count(), 1);
        assert_eq!(app.take_segment_done(), Some(5_000));
        assert_eq!(app.take_segment_done(), None, "take is once per notify");

        // A second completion bumps the monotonic count and re-arms the take.
        src.notify_segment_done(10_000);
        assert_eq!(app.segment_done_count(), 2);
        assert_eq!(app.take_segment_done(), Some(10_000));
    }

    #[test]
    fn shutdown_latches_and_is_visible_to_the_source() {
        let app = SeekController::new();
        let src = app.clone();
        assert!(!src.is_shutdown());
        app.shutdown();
        assert!(src.is_shutdown());
        // Latching: it stays set (a run is torn down, not resumed).
        assert!(src.is_shutdown());
    }

    // `block_on` parks a thread, so it exists only on the std runtime.
    #[cfg(feature = "std")]
    #[test]
    fn wait_event_resolves_and_wakes() {
        use crate::runtime::block_on;
        // Already-pending: resolves immediately.
        let c = SeekController::new();
        c.seek(Seek::flush_to(1));
        block_on(c.wait_event());

        // A wake from `seek` resolves a previously-pending wait, and a wake from
        // `shutdown` resolves the idle case. Drive a parked wait by polling it
        // once (Pending), then satisfying it and polling again (Ready).
        let c2 = SeekController::new();
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        let mut fut = core::pin::pin!(c2.wait_event());
        assert!(fut.as_mut().poll(&mut cx).is_pending(), "no event yet");
        c2.shutdown();
        assert!(fut.as_mut().poll(&mut cx).is_ready(), "shutdown resolves the wait");
    }

    /// A no-op waker so a `wait_event` future can be polled directly in a test.
    #[cfg(feature = "std")]
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
