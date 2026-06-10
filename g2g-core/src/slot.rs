//! Lock-free atomic-swap container for dynamically replaceable elements.
//!
//! M8 piece 6: the primary mechanism by which the dynamic graph layer
//! responds to mid-stream `Reconfigure` requests, codec switches, branch
//! enable/disable, and any other event that needs to replace one element
//! with another without draining the pipeline.
//!
//! A frame already inside the old element's `process()` call completes
//! against the old element naturally because [`ElementSlot::process`]
//! takes a `load_full()` snapshot of the current contents at the start
//! of each call. The next `process()` invocation sees the new element.
//! See `DESIGN.md` §4.8.2.
//!
//! Inside the slot, the element is wrapped in an `Arc<Mutex<_>>`: `Arc`
//! to share between the snapshot-taking process futures and the swapper,
//! `Mutex` because the underlying `DynAsyncElement` trait takes
//! `&mut self` on the hot path. The mutex is held for the duration of
//! one `process()` call; concurrent swaps complete instantly via
//! `ArcSwap::store`, the swapped-in element is visible to subsequent
//! `process()` calls.

use alloc::boxed::Box;
use alloc::sync::Arc;

use arc_swap::ArcSwap;
use spin::Mutex;

use crate::caps::Caps;
use crate::element::{AsyncElement, BoxFuture, ConfigureOutcome, DynAsyncElement, OutputSink};
use crate::error::G2gError;
use crate::frame::PipelinePacket;

/// Shared cell holding the slot's current element. The outer `Arc` lets a
/// [`SwapHandle`] mutate the contents while the runner owns the
/// [`ElementSlot`] by `&mut`: stores go through a cloned `Arc` and only need
/// `&self`, the swapped-out element lives behind the inner `Arc` until its
/// last reader drops it, so there is no aliasing with the wrapper's `&mut`.
type SlotCell = Arc<ArcSwap<Mutex<Box<dyn DynAsyncElement + Send>>>>;

/// Atomically swappable container for a `Box<dyn DynAsyncElement>`.
/// Implements `AsyncElement` so it can sit inside a typed pipeline runner
/// unchanged — the swap behavior is invisible to the runner.
pub struct ElementSlot {
    inner: SlotCell,
}

impl core::fmt::Debug for ElementSlot {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ElementSlot").finish_non_exhaustive()
    }
}

impl ElementSlot {
    pub fn new(element: Box<dyn DynAsyncElement + Send>) -> Self {
        Self { inner: Arc::new(ArcSwap::new(Arc::new(Mutex::new(element)))) }
    }

    /// Atomically install `element` as the slot's contents. Process calls
    /// already in flight against the previous contents complete naturally;
    /// subsequent calls see `element`. **Caller is responsible** for
    /// having called `configure_pipeline` on `element` against the
    /// pipeline's current fixated caps before installing — the slot will
    /// not re-run negotiation.
    pub fn swap(&self, element: Box<dyn DynAsyncElement + Send>) {
        self.inner.store(Arc::new(Mutex::new(element)));
    }

    /// A cloneable handle that can swap this slot's element from another
    /// task while the runner drives the slot. This is how a mid-stream
    /// `Reconfigure` or codec switch replaces an element without stalling
    /// or rebuilding the pipeline (DESIGN.md §4.8.2).
    pub fn handle(&self) -> SwapHandle {
        SwapHandle { inner: self.inner.clone() }
    }
}

/// Detached handle to an [`ElementSlot`], obtained via
/// [`ElementSlot::handle`]. Holds a clone of the slot's shared cell, so
/// [`SwapHandle::swap`] installs a new element visible to the slot's next
/// `process` call. Cheap to clone; multiple controllers may hold one.
#[derive(Clone)]
pub struct SwapHandle {
    inner: SlotCell,
}

impl core::fmt::Debug for SwapHandle {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SwapHandle").finish_non_exhaustive()
    }
}

impl SwapHandle {
    /// Install `element` as the target slot's contents. Same contract as
    /// [`ElementSlot::swap`]: the caller must have configured `element`
    /// against the pipeline's current fixated caps beforehand.
    pub fn swap(&self, element: Box<dyn DynAsyncElement + Send>) {
        self.inner.store(Arc::new(Mutex::new(element)));
    }
}

impl AsyncElement for ElementSlot {
    type ProcessFuture<'a>
        = BoxFuture<'a, Result<(), G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        self.inner.load().lock().intercept_caps(upstream_caps)
    }

    fn configure_pipeline(
        &mut self,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        self.inner.load().lock().configure_pipeline(absolute_caps)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            // `load_full` captures the current Arc so a concurrent swap
            // can't drop the element while this process call is running.
            // If swap fires during the await, our local Arc keeps the
            // old contents alive until process completes; new pushes go
            // through the new element on their next process call.
            let elem = self.inner.load_full();
            let mut guard = elem.lock();
            guard.process(packet, out).await
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::{Dim, Rate, VideoFormat};
    use crate::element::{DynAsyncElement, OutputSink, PushOutcome};
    use crate::frame::{Frame, FrameTiming};
    use crate::memory::{MemoryDomain, SystemSlice};
    use alloc::sync::Arc as StdArc;
    use core::future::Future;
    use core::pin::Pin;
    use core::sync::atomic::{AtomicU64, Ordering};

    /// Element that increments a shared counter on every `process()`
    /// call so the test can prove which element handled which packet
    /// across an atomic swap.
    struct CountingElement {
        counter: StdArc<AtomicU64>,
        configured: bool,
    }

    impl DynAsyncElement for CountingElement {
        fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
            Ok(upstream_caps.clone())
        }

        fn configure_pipeline(
            &mut self,
            _absolute_caps: &Caps,
        ) -> Result<ConfigureOutcome, G2gError> {
            self.configured = true;
            Ok(ConfigureOutcome::Accepted)
        }

        fn process<'a>(
            &'a mut self,
            _packet: PipelinePacket,
            _out: &'a mut dyn OutputSink,
        ) -> Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>> {
            let counter = self.counter.clone();
            Box::pin(async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
        }

        fn caps_constraint_as_sink(&self) -> crate::format_element::CapsConstraint<'_> {
            crate::format_element::CapsConstraint::AcceptsAny
        }

        fn propose_allocation(
            &self,
            _caps: &Caps,
        ) -> Option<crate::query::AllocationParams> {
            None
        }

        fn configure_allocation(&mut self, _params: &crate::query::AllocationParams) {}
    }

    /// Implements only [`AsyncElement`] (a GAT element, no hand-written
    /// `DynAsyncElement`). Boxing it relies on the blanket impl in
    /// `element.rs`, the path real plugins take into the slot.
    struct AsyncOnlyElement {
        counter: StdArc<AtomicU64>,
    }

    impl AsyncElement for AsyncOnlyElement {
        type ProcessFuture<'a>
            = BoxFuture<'a, Result<(), G2gError>>
        where
            Self: 'a;

        fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
            Ok(upstream_caps.clone())
        }

        fn configure_pipeline(
            &mut self,
            _absolute_caps: &Caps,
        ) -> Result<ConfigureOutcome, G2gError> {
            Ok(ConfigureOutcome::Accepted)
        }

        fn process<'a>(
            &'a mut self,
            _packet: PipelinePacket,
            _out: &'a mut dyn OutputSink,
        ) -> Self::ProcessFuture<'a> {
            let counter = self.counter.clone();
            Box::pin(async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
        }
    }

    struct NoopSink;
    impl OutputSink for NoopSink {
        fn push<'a>(
            &'a mut self,
            _packet: PipelinePacket,
        ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
            Box::pin(async { Ok(PushOutcome::Accepted) })
        }
    }

    fn dummy_frame() -> PipelinePacket {
        PipelinePacket::DataFrame(Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
            caps: Caps::Video {
                format: VideoFormat::H264,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
            },
            timing: FrameTiming::default(),
            sequence: 0,
        })
    }

    /// Block on a future using a noop waker; sufficient because all the
    /// futures in these tests resolve in a single poll.
    fn block_on<F: Future>(mut fut: F) -> F::Output {
        use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
        static VT: RawWakerVTable = RawWakerVTable::new(
            |_| RawWaker::new(core::ptr::null(), &VT),
            |_| {},
            |_| {},
            |_| {},
        );
        // SAFETY: VT's hooks never deref the pointer.
        let waker = unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &VT)) };
        let mut cx = Context::from_waker(&waker);
        // SAFETY: pin to stack for duration of fn.
        let mut pinned = unsafe { Pin::new_unchecked(&mut fut) };
        loop {
            match pinned.as_mut().poll(&mut cx) {
                Poll::Ready(v) => return v,
                Poll::Pending => panic!("slot::tests::block_on saw Pending"),
            }
        }
    }

    #[test]
    fn process_increments_initial_element_counter() {
        let initial_counter = StdArc::new(AtomicU64::new(0));
        let mut slot = ElementSlot::new(Box::new(CountingElement {
            counter: initial_counter.clone(),
            configured: true,
        }));
        let mut sink = NoopSink;

        block_on(AsyncElement::process(&mut slot,dummy_frame(), &mut sink)).unwrap();
        block_on(AsyncElement::process(&mut slot,dummy_frame(), &mut sink)).unwrap();

        assert_eq!(initial_counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn swap_routes_subsequent_pushes_to_new_element() {
        let counter_a = StdArc::new(AtomicU64::new(0));
        let counter_b = StdArc::new(AtomicU64::new(0));

        let mut slot = ElementSlot::new(Box::new(CountingElement {
            counter: counter_a.clone(),
            configured: true,
        }));
        let mut sink = NoopSink;

        // Two pushes to A.
        block_on(AsyncElement::process(&mut slot,dummy_frame(), &mut sink)).unwrap();
        block_on(AsyncElement::process(&mut slot,dummy_frame(), &mut sink)).unwrap();

        // Swap to B.
        slot.swap(Box::new(CountingElement {
            counter: counter_b.clone(),
            configured: true,
        }));

        // Three pushes to B.
        block_on(AsyncElement::process(&mut slot,dummy_frame(), &mut sink)).unwrap();
        block_on(AsyncElement::process(&mut slot,dummy_frame(), &mut sink)).unwrap();
        block_on(AsyncElement::process(&mut slot,dummy_frame(), &mut sink)).unwrap();

        assert_eq!(counter_a.load(Ordering::SeqCst), 2);
        assert_eq!(counter_b.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn blanket_impl_boxes_a_plain_async_element() {
        let counter = StdArc::new(AtomicU64::new(0));
        // A type that implements only AsyncElement still boxes into the slot
        // via the DynAsyncElement blanket impl.
        let boxed: Box<dyn DynAsyncElement + Send> =
            Box::new(AsyncOnlyElement { counter: counter.clone() });
        let mut slot = ElementSlot::new(boxed);
        let mut sink = NoopSink;

        block_on(AsyncElement::process(&mut slot,dummy_frame(), &mut sink)).unwrap();

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn swap_handle_routes_subsequent_pushes_to_new_element() {
        let counter_a = StdArc::new(AtomicU64::new(0));
        let counter_b = StdArc::new(AtomicU64::new(0));

        let mut slot = ElementSlot::new(Box::new(CountingElement {
            counter: counter_a.clone(),
            configured: true,
        }));
        let handle = slot.handle();
        let mut sink = NoopSink;

        block_on(AsyncElement::process(&mut slot,dummy_frame(), &mut sink)).unwrap();

        // Swap via the detached handle rather than the slot itself.
        handle.swap(Box::new(CountingElement { counter: counter_b.clone(), configured: true }));

        block_on(AsyncElement::process(&mut slot,dummy_frame(), &mut sink)).unwrap();

        assert_eq!(counter_a.load(Ordering::SeqCst), 1);
        assert_eq!(counter_b.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn intercept_caps_uses_current_element() {
        let counter = StdArc::new(AtomicU64::new(0));
        let slot = ElementSlot::new(Box::new(CountingElement {
            counter,
            configured: true,
        }));
        let upstream = Caps::Video {
            format: VideoFormat::H264,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
        };
        let caps = AsyncElement::intercept_caps(&slot, &upstream).unwrap();
        assert_eq!(caps, upstream);
    }
}
