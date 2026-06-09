use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;

use crate::caps::Caps;
use crate::error::G2gError;
use crate::frame::PipelinePacket;
use crate::query::{AllocationParams, LatencyReport};

#[cfg(feature = "multi-thread")]
pub trait ElementBound: Send {}
#[cfg(feature = "multi-thread")]
impl<T: Send> ElementBound for T {}

#[cfg(not(feature = "multi-thread"))]
pub trait ElementBound {}
#[cfg(not(feature = "multi-thread"))]
impl<T> ElementBound for T {}

/// Boxed future alias for dyn-safe async methods in element / sink traits.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

/// Downstream-originated request to renegotiate caps. Travels upstream
/// along a link's reverse channel and is surfaced to the producing element
/// as a [`PushOutcome::Reconfigure`].
#[derive(Debug, Clone, PartialEq)]
pub enum Reconfigure {
    /// Downstream proposes specific replacement caps (Phase 3 counter).
    Propose(Caps),
    /// Downstream wants renegotiation but has no specific proposal —
    /// the upstream element picks freely. Equivalent to GStreamer's
    /// bare RECONFIGURE event.
    Renegotiate,
}

/// Outcome of pushing a packet downstream. Sources and transforms must
/// react to `Reconfigure` before pushing any further data; terminal sinks
/// and intermediate adapters that can't renegotiate may ignore it.
#[derive(Debug, Clone, PartialEq)]
pub enum PushOutcome {
    /// Downstream accepted the packet; continue normally.
    Accepted,
    /// Downstream signaled a reconfigure request between this push and
    /// the previous one. The producer should handle the request before
    /// pushing further `DataFrame`s.
    Reconfigure(Reconfigure),
}

/// Downstream output for elements. `push` is async so backpressure-aware
/// implementations can await downstream capacity instead of erroring on a
/// full link. The boxed future keeps the trait dyn-safe.
pub trait OutputSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>>;
}

#[derive(Debug)]
pub enum ConfigureOutcome {
    Accepted,
    ReFixate(Caps),
}

pub trait AsyncElement: ElementBound {
    type ProcessFuture<'a>: Future<Output = Result<(), G2gError>> + 'a
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError>;

    fn configure_pipeline(
        &mut self,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError>;

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a>;

    /// This element's contribution to the pipeline latency query (M12).
    /// Default: zero, non-live. Transforms that buffer (jitter buffers,
    /// reorder queues) and live sources override this; the linear runners
    /// fold the chain into `RunStats::latency`.
    fn latency(&self) -> LatencyReport {
        LatencyReport::ZERO
    }

    /// Answer the upstream peer's allocation query (M12): the buffer size,
    /// count, alignment, and memory domain this element needs allocated so a
    /// pool can be handed over without a copy. Default: no preference
    /// (`None`). A transform that has already received its own downstream
    /// proposal via [`configure_allocation`](Self::configure_allocation) can
    /// fold it in with [`AllocationParams::merge`].
    fn propose_allocation(&self, _caps: &Caps) -> Option<AllocationParams> {
        None
    }

    /// Receive the downstream peer's allocation proposal (M12) so this element
    /// can allocate its output buffers from a compatible pool. Default:
    /// ignore and allocate however the element sees fit.
    fn configure_allocation(&mut self, _params: &AllocationParams) {}
}

/// Dyn-safe variant of [`AsyncElement`] for plugin registries on `std` targets.
/// `no_std` graphs use the monomorphised `AsyncElement` directly.
#[cfg(feature = "std")]
pub trait DynAsyncElement: ElementBound {
    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError>;

    fn configure_pipeline(
        &mut self,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError>;

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> core::pin::Pin<
        alloc::boxed::Box<dyn Future<Output = Result<(), G2gError>> + 'a>,
    >;
}

/// Blanket adapter: every [`AsyncElement`] is usable as a
/// [`DynAsyncElement`] by boxing its `process` future (DESIGN.md §4.3).
/// This is what lets real plugin elements drop into a `Box<dyn
/// DynAsyncElement>` slot without a hand-written impl. Method calls are
/// disambiguated to `AsyncElement::` because the two traits share names.
#[cfg(feature = "std")]
impl<T: AsyncElement> DynAsyncElement for T {
    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        AsyncElement::intercept_caps(self, upstream_caps)
    }

    fn configure_pipeline(
        &mut self,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        AsyncElement::configure_pipeline(self, absolute_caps)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> BoxFuture<'a, Result<(), G2gError>> {
        Box::pin(AsyncElement::process(self, packet, out))
    }
}
