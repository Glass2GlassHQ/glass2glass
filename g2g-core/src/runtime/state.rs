//! Shared pipeline state controller (M76 + M77).
//!
//! A cloneable handle over an atomic [`PipelineState`] plus a `Waker`
//! registry, so an application can drive `set_state(Playing)` from one task
//! while a runner's sink arm awaits flow from another. The data plane gates
//! on the sink: when the state is below `Playing`, the sink stops pulling,
//! the bounded link fills, and backpressure stalls the whole chain upstream
//! with no per-element cooperation. `Playing` opens the gate.
//!
//! **Preroll (M77).** A non-live pipeline in `Paused` admits exactly one
//! buffer (the preroll frame) and then holds; `set_state(Paused)` reports
//! `Async` and completes once the sink prerolls ([`BusMessage::AsyncDone`],
//! [`StateController::await_prerolled`]). A live pipeline produces no preroll
//! buffer: `Paused` reports `NoPreroll` and the gate full-holds. Mark a
//! pipeline live with [`StateController::set_live`].
//!
//! This is the additive controller layer: the existing runners are untouched,
//! and `run_simple_pipeline_stateful` opts in by taking a `&StateController`.
//! Rolling the gate into every runner shape (`run_graph` et al.) and graceful
//! mid-stream `Null` teardown are M78.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use core::task::{Context, Poll, Waker};

use spin::Mutex;

use crate::bus::{BusHandle, BusMessage};
use crate::state::{PipelineState, StateChangeReturn};

#[derive(Debug)]
struct Inner {
    state: AtomicU8,
    /// Whether the pipeline has a live source. A live pipeline produces no
    /// preroll buffer in `Paused` (the clock isn't running), so the sink does a
    /// full hold and `set_state(Paused)` reports `NoPreroll`; a non-live
    /// pipeline prerolls one buffer and reports `Async`. Set at build time via
    /// [`StateController::set_live`].
    live: AtomicBool,
    /// Whether the (non-live) sink has taken its preroll buffer. Set by
    /// [`StateController::notify_prerolled`] after the first buffer in `Paused`,
    /// and unconditionally on the `Playing` transition; reset to `false` on a
    /// transition down to `Ready`/`Null` so a later `Paused` re-prerolls.
    prerolled: AtomicBool,
    /// Flow-gate futures parked while data must not cross. Drained and woken on
    /// every state change so they re-evaluate.
    wakers: Mutex<Vec<Waker>>,
    /// [`PrerollGate`] futures awaiting preroll completion. Woken by
    /// `notify_prerolled` and by the `Playing` transition.
    preroll_wakers: Mutex<Vec<Waker>>,
    /// Optional bus; `set_state` posts a [`BusMessage::StateChanged`] and
    /// `notify_prerolled` posts a [`BusMessage::AsyncDone`] when set.
    bus: Option<BusHandle>,
}

/// Cloneable controller for a pipeline's lifecycle state. Every clone shares
/// the same atomic state and waker registry (one `Arc`).
#[derive(Debug, Clone)]
pub struct StateController {
    inner: Arc<Inner>,
}

impl StateController {
    /// New controller starting in `initial`, with no bus. Non-live by default.
    pub fn new(initial: PipelineState) -> Self {
        Self::build(initial, None)
    }

    /// As [`StateController::new`], but `set_state` posts a
    /// [`BusMessage::StateChanged`] to `bus` on every effective transition and
    /// `notify_prerolled` posts a [`BusMessage::AsyncDone`].
    pub fn with_bus(initial: PipelineState, bus: BusHandle) -> Self {
        Self::build(initial, Some(bus))
    }

    fn build(initial: PipelineState, bus: Option<BusHandle>) -> Self {
        Self {
            inner: Arc::new(Inner {
                state: AtomicU8::new(initial.as_u8()),
                live: AtomicBool::new(false),
                prerolled: AtomicBool::new(false),
                wakers: Mutex::new(Vec::new()),
                preroll_wakers: Mutex::new(Vec::new()),
                bus,
            }),
        }
    }

    /// Mark the pipeline live (a live source) or non-live. Set once at build
    /// time, before the run. A live pipeline does not preroll in `Paused`.
    pub fn set_live(&self, live: bool) {
        self.inner.live.store(live, Ordering::Release);
    }

    /// Whether the pipeline is marked live.
    pub fn is_live(&self) -> bool {
        self.inner.live.load(Ordering::Acquire)
    }

    /// Current state.
    pub fn state(&self) -> PipelineState {
        PipelineState::from_u8(self.inner.state.load(Ordering::Acquire))
    }

    /// Whether the pipeline is flowing.
    pub fn is_playing(&self) -> bool {
        self.state() == PipelineState::Playing
    }

    /// Whether the (non-live) sink has taken its preroll buffer, completing an
    /// async `Paused` transition.
    pub fn is_prerolled(&self) -> bool {
        self.inner.prerolled.load(Ordering::Acquire)
    }

    /// Move to `new`. Wakes every parked gate so it re-evaluates, posts
    /// `StateChanged { old, new }` (when a bus is wired), and adjusts the
    /// preroll flag for the target:
    ///
    /// - `Playing` marks prerolled (preroll is trivially satisfied) and wakes
    ///   any [`await_prerolled`](StateController::await_prerolled) waiters.
    /// - `Ready` / `Null` clear prerolled so a later `Paused` re-prerolls.
    ///
    /// Return code (on an effective transition; a no-op returns `Success`):
    /// `Paused` returns [`StateChangeReturn::NoPreroll`] when live (no preroll
    /// buffer is coming) or [`StateChangeReturn::Async`] when non-live (the
    /// change completes when the sink prerolls); every other target returns
    /// [`StateChangeReturn::Success`].
    pub fn set_state(&self, new: PipelineState) -> StateChangeReturn {
        let old = PipelineState::from_u8(self.inner.state.swap(new.as_u8(), Ordering::AcqRel));
        if old == new {
            return StateChangeReturn::Success;
        }

        // Drain under the same lock the gate registers under, so a gate that
        // read the old state and is about to park cannot miss this wake (it
        // holds the lock across its load + push; see `FlowGate`).
        let mut w = self.inner.wakers.lock();
        for waker in w.drain(..) {
            waker.wake();
        }
        drop(w);

        if let Some(bus) = &self.inner.bus {
            bus.try_post(BusMessage::StateChanged { old, new });
        }

        // Preroll bookkeeping for the target state, after `StateChanged` so a
        // `Playing` transition posts `StateChanged` then `AsyncDone`.
        match new {
            PipelineState::Playing => self.mark_prerolled(),
            PipelineState::Ready | PipelineState::Null => {
                self.inner.prerolled.store(false, Ordering::Release);
            }
            PipelineState::Paused => {}
        }

        match new {
            PipelineState::Paused if self.is_live() => StateChangeReturn::NoPreroll,
            PipelineState::Paused => StateChangeReturn::Async,
            _ => StateChangeReturn::Success,
        }
    }

    /// Record that the sink has taken its preroll buffer, completing an async
    /// `Paused` transition. Idempotent: the first call wakes
    /// [`await_prerolled`](StateController::await_prerolled) waiters and posts
    /// [`BusMessage::AsyncDone`]; later calls are no-ops. The
    /// `run_simple_pipeline_stateful` sink arm calls this after the first
    /// buffer it processes in non-live `Paused`.
    pub fn notify_prerolled(&self) {
        self.mark_prerolled();
    }

    fn mark_prerolled(&self) {
        if !self.inner.prerolled.swap(true, Ordering::AcqRel) {
            let mut w = self.inner.preroll_wakers.lock();
            for waker in w.drain(..) {
                waker.wake();
            }
            drop(w);
            if let Some(bus) = &self.inner.bus {
                bus.try_post(BusMessage::AsyncDone);
            }
        }
    }

    /// A future that resolves once the pipeline is flowing or torn down:
    /// [`Flow::Go`] at `Playing` (and the one preroll buffer in non-live
    /// `Paused`), [`Flow::Stop`] at `Null`, and `Pending` otherwise.
    pub fn flow_gate(&self) -> FlowGate {
        FlowGate {
            inner: self.inner.clone(),
        }
    }

    /// A future that resolves once preroll has completed (or the pipeline is
    /// `Playing`). After `set_state(Paused)` returns `Async`, await this before
    /// `set_state(Playing)` to start from a prerolled sink.
    pub fn await_prerolled(&self) -> PrerollGate {
        PrerollGate {
            inner: self.inner.clone(),
        }
    }
}

/// What a sink arm should do after awaiting [`StateController::flow_gate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    /// `Playing`: pull and process the next packet.
    Go,
    /// `Null`: the pipeline is torn down; the arm should stop.
    Stop,
}

/// Future returned by [`StateController::flow_gate`]. Owns an `Arc` clone of
/// the controller's inner state so it carries no lifetime.
#[derive(Debug)]
pub struct FlowGate {
    inner: Arc<Inner>,
}

impl Future for FlowGate {
    type Output = Flow;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Flow> {
        // Hold the waker lock across the state load + park. `set_state` takes
        // the same lock to drain/wake *after* swapping the state, so either we
        // observe the new state here (and return Ready) or we park the waker
        // before `set_state` can drain it. No lost wakeup either way.
        let mut wakers = self.inner.wakers.lock();
        let go_or_park = |wakers: &mut Vec<Waker>, go: bool| {
            if go {
                Poll::Ready(Flow::Go)
            } else {
                if !wakers.iter().any(|w| w.will_wake(cx.waker())) {
                    wakers.push(cx.waker().clone());
                }
                Poll::Pending
            }
        };
        match PipelineState::from_u8(self.inner.state.load(Ordering::Acquire)) {
            PipelineState::Null => Poll::Ready(Flow::Stop),
            PipelineState::Playing => Poll::Ready(Flow::Go),
            PipelineState::Ready => go_or_park(&mut wakers, false),
            // Non-live `Paused` admits exactly one buffer (the preroll frame):
            // `Go` until the sink marks itself prerolled, then park until
            // `Playing`. Live `Paused` admits nothing (no preroll buffer).
            PipelineState::Paused => {
                let prerolled = self.inner.prerolled.load(Ordering::Acquire);
                let live = self.inner.live.load(Ordering::Acquire);
                go_or_park(&mut wakers, !live && !prerolled)
            }
        }
    }
}

/// Future returned by [`StateController::await_prerolled`]. Resolves once the
/// sink has taken its preroll buffer (or the pipeline reached `Playing`).
#[derive(Debug)]
pub struct PrerollGate {
    inner: Arc<Inner>,
}

impl Future for PrerollGate {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        // Same lock discipline as `FlowGate`: register under the lock that
        // `mark_prerolled` drains under, after swapping the flag.
        let mut wakers = self.inner.preroll_wakers.lock();
        if self.inner.prerolled.load(Ordering::Acquire) {
            Poll::Ready(())
        } else {
            if !wakers.iter().any(|w| w.will_wake(cx.waker())) {
                wakers.push(cx.waker().clone());
            }
            Poll::Pending
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::Bus;
    use core::task::{RawWaker, RawWakerVTable, Waker};

    /// A waker that flips a shared flag when woken, so a test can prove the
    /// gate's parked waker is actually fired by `set_state`.
    fn flag_waker(flag: Arc<core::sync::atomic::AtomicBool>) -> Waker {
        // Leak an Arc clone as the waker data; freed in `drop_fn`.
        fn clone_fn(data: *const ()) -> RawWaker {
            // SAFETY: `data` is an `Arc<AtomicBool>` pointer we created below;
            // reconstruct without dropping, bump the count, leak again.
            let arc = unsafe { Arc::from_raw(data as *const core::sync::atomic::AtomicBool) };
            let cloned = arc.clone();
            core::mem::forget(arc);
            RawWaker::new(Arc::into_raw(cloned) as *const (), &VT)
        }
        fn wake_fn(data: *const ()) {
            // SAFETY: takes ownership of one ref (consumes it, as `wake` does).
            let arc = unsafe { Arc::from_raw(data as *const core::sync::atomic::AtomicBool) };
            arc.store(true, Ordering::SeqCst);
        }
        fn wake_by_ref_fn(data: *const ()) {
            // SAFETY: borrows without consuming.
            let arc = unsafe { Arc::from_raw(data as *const core::sync::atomic::AtomicBool) };
            arc.store(true, Ordering::SeqCst);
            core::mem::forget(arc);
        }
        fn drop_fn(data: *const ()) {
            // SAFETY: drops the one ref this waker owned.
            unsafe { drop(Arc::from_raw(data as *const core::sync::atomic::AtomicBool)) };
        }
        static VT: RawWakerVTable = RawWakerVTable::new(clone_fn, wake_fn, wake_by_ref_fn, drop_fn);
        let raw = RawWaker::new(Arc::into_raw(flag) as *const (), &VT);
        // SAFETY: `raw` is built from the VT above whose hooks treat the data
        // pointer as the `Arc<AtomicBool>` it is.
        unsafe { Waker::from_raw(raw) }
    }

    #[test]
    fn set_state_returns_reflect_preroll_model() {
        let sc = StateController::new(PipelineState::Null);
        assert_eq!(sc.state(), PipelineState::Null);
        // Non-live `Paused` is async: it completes when the sink prerolls.
        assert_eq!(
            sc.set_state(PipelineState::Paused),
            StateChangeReturn::Async
        );
        assert_eq!(sc.state(), PipelineState::Paused);
        assert!(!sc.is_playing());
        assert_eq!(
            sc.set_state(PipelineState::Playing),
            StateChangeReturn::Success
        );
        assert!(sc.is_playing());
        // A no-op transition returns Success.
        assert_eq!(
            sc.set_state(PipelineState::Playing),
            StateChangeReturn::Success
        );

        // A live pipeline produces no preroll buffer in `Paused`.
        let live = StateController::new(PipelineState::Null);
        live.set_live(true);
        assert_eq!(
            live.set_state(PipelineState::Paused),
            StateChangeReturn::NoPreroll
        );
    }

    #[test]
    fn nonlive_paused_admits_one_preroll_then_holds() {
        let sc = StateController::new(PipelineState::Paused); // non-live
        let flag = Arc::new(core::sync::atomic::AtomicBool::new(false));
        let waker = flag_waker(flag);
        let mut cx = Context::from_waker(&waker);

        // First poll in non-live `Paused`: the one preroll frame is admitted.
        let mut gate = sc.flow_gate();
        // SAFETY: `gate` pinned to the stack for this poll.
        let pinned = unsafe { Pin::new_unchecked(&mut gate) };
        assert_eq!(
            pinned.poll(&mut cx),
            Poll::Ready(Flow::Go),
            "preroll frame admitted"
        );

        // The sink took its preroll buffer; the gate now holds until Playing.
        sc.notify_prerolled();
        let mut held = sc.flow_gate();
        // SAFETY: pinned to the stack for this poll.
        let pinned = unsafe { Pin::new_unchecked(&mut held) };
        assert_eq!(pinned.poll(&mut cx), Poll::Pending, "holds after preroll");

        sc.set_state(PipelineState::Playing);
        // SAFETY: same pinned gate, re-polled after the transition.
        let pinned = unsafe { Pin::new_unchecked(&mut held) };
        assert_eq!(pinned.poll(&mut cx), Poll::Ready(Flow::Go));
    }

    #[test]
    fn live_paused_holds_fully_and_wakes_on_play() {
        let sc = StateController::new(PipelineState::Paused);
        sc.set_live(true);
        let flag = Arc::new(core::sync::atomic::AtomicBool::new(false));
        let waker = flag_waker(flag.clone());
        let mut cx = Context::from_waker(&waker);

        let mut gate = sc.flow_gate();
        // Live `Paused` admits nothing: the gate parks immediately.
        // SAFETY: `gate` pinned to the stack for the poll.
        let pinned = unsafe { Pin::new_unchecked(&mut gate) };
        assert_eq!(pinned.poll(&mut cx), Poll::Pending, "no preroll when live");
        assert!(!flag.load(Ordering::SeqCst));

        sc.set_state(PipelineState::Playing);
        assert!(
            flag.load(Ordering::SeqCst),
            "transition woke the parked gate"
        );
        // SAFETY: same pinned gate, re-polled after the wake.
        let pinned = unsafe { Pin::new_unchecked(&mut gate) };
        assert_eq!(pinned.poll(&mut cx), Poll::Ready(Flow::Go));
    }

    #[test]
    fn notify_prerolled_resolves_await_and_posts_asyncdone() {
        let (bus, handle) = Bus::new(8);
        let sc = StateController::with_bus(PipelineState::Paused, handle);
        let flag = Arc::new(core::sync::atomic::AtomicBool::new(false));
        let waker = flag_waker(flag.clone());
        let mut cx = Context::from_waker(&waker);

        let mut g = sc.await_prerolled();
        // SAFETY: `g` pinned to the stack for this poll.
        let pinned = unsafe { Pin::new_unchecked(&mut g) };
        assert_eq!(pinned.poll(&mut cx), Poll::Pending, "not prerolled yet");

        sc.notify_prerolled();
        assert!(flag.load(Ordering::SeqCst), "preroll woke the awaiter");
        assert_eq!(bus.try_recv(), Some(BusMessage::AsyncDone));

        // SAFETY: same pinned awaiter, re-polled after preroll.
        let pinned = unsafe { Pin::new_unchecked(&mut g) };
        assert_eq!(pinned.poll(&mut cx), Poll::Ready(()));

        // Idempotent: a second notify posts nothing.
        sc.notify_prerolled();
        assert_eq!(bus.try_recv(), None);
    }

    #[test]
    fn gate_stops_on_null() {
        let sc = StateController::new(PipelineState::Null);
        let flag = Arc::new(core::sync::atomic::AtomicBool::new(false));
        let waker = flag_waker(flag);
        let mut cx = Context::from_waker(&waker);
        let mut gate = sc.flow_gate();
        // SAFETY: `gate` pinned to the stack for this poll.
        let pinned = unsafe { Pin::new_unchecked(&mut gate) };
        assert_eq!(pinned.poll(&mut cx), Poll::Ready(Flow::Stop));
    }

    #[test]
    fn set_state_posts_state_changed_to_bus() {
        let (bus, handle) = Bus::new(8);
        let sc = StateController::with_bus(PipelineState::Null, handle);
        sc.set_state(PipelineState::Ready);
        sc.set_state(PipelineState::Ready); // no-op, posts nothing
        sc.set_state(PipelineState::Playing);
        assert_eq!(
            bus.try_recv(),
            Some(BusMessage::StateChanged {
                old: PipelineState::Null,
                new: PipelineState::Ready
            })
        );
        assert_eq!(
            bus.try_recv(),
            Some(BusMessage::StateChanged {
                old: PipelineState::Ready,
                new: PipelineState::Playing
            })
        );
        // The `Playing` transition also satisfies preroll, so `AsyncDone`
        // follows the `StateChanged` (it never prerolled while Ready).
        assert_eq!(bus.try_recv(), Some(BusMessage::AsyncDone));
        assert_eq!(bus.try_recv(), None, "the no-op transition posted nothing");
    }
}
