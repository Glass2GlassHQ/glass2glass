use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;

use crate::caps::Caps;
use crate::clock::ClockCandidate;
use crate::error::G2gError;
use crate::format_element::{legacy_sink_constraint, legacy_transform_constraint, CapsConstraint};
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

    /// Offer a clock to the pipeline's clock election (M12). Default: none.
    /// Elements that pace to real hardware (an audio sink to its DAC) override
    /// this; the runner elects the highest-priority offered clock.
    fn provide_clock(&self) -> Option<ClockCandidate> {
        None
    }

    /// Declares that this element changes the caps "domain" between its
    /// input and output: a decoder turns compressed bitstream into raw
    /// pixels, an encoder turns raw pixels into compressed bitstream, a
    /// format converter shifts color space, etc. Default: false.
    ///
    /// Currently informational. The runner uses a single linear caps
    /// cascade and the three workarounds documented in
    /// `architecture_caps_nego_debt` apply on its behalf. The planned
    /// caps redesign (Plan 2) will use this hint to split the pipeline
    /// into per-domain negotiation segments, eliminating the
    /// pass-through `intercept_caps` and deferred-configure dance that
    /// sinks downstream of a boundary currently rely on.
    ///
    /// Declaring this true today changes no behavior — it's a forward
    /// declaration so the redesign can roll out without simultaneously
    /// migrating every decoder.
    fn is_format_boundary(&self) -> bool {
        false
    }

    /// Derive the element's *output* caps from its negotiated input
    /// caps. Only consulted by the runner when `is_format_boundary()`
    /// is true. Default: pass input through (correct for non-boundary
    /// elements that don't change format).
    ///
    /// Boundary elements (decoders, encoders, format converters)
    /// override this to advertise their post-transform caps so the
    /// downstream segment can negotiate honestly. A decoder typically
    /// reads dims from the input caps (already populated from the
    /// stream's SPS / container header) and returns raw video caps at
    /// matching geometry.
    ///
    /// If the output caps genuinely can't be known until first decoded
    /// frame (rare for modern stream containers), return ranged caps
    /// here and emit a fixing `CapsChanged` mid-stream.
    fn propose_output_caps(&self, input: &Caps) -> Caps {
        input.clone()
    }

    /// M16 step 5b: declare this element's negotiation-time constraint
    /// when used as the **sink** of a chain. The default returns the
    /// legacy bridge (`LegacySink` wrapping today's `intercept_caps`).
    /// Migrated sinks override with `Accepts(CapsSet)` (or a more
    /// elaborate native variant) to participate in arc consistency
    /// and skip the dynamic intercept callback.
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        legacy_sink_constraint(self)
    }

    /// M16 step 5b: same as `caps_constraint_as_sink` but for the
    /// **transform** role. Default returns `LegacyTransform` wrapping
    /// today's `intercept_caps` + `propose_output_caps`. Migrated
    /// transforms override with `Identity` / `Mapping` /
    /// `DerivedOutput`.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        legacy_transform_constraint(self)
    }
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

    /// Dyn-safe mirror of [`AsyncElement::caps_constraint_as_sink`], so a
    /// `Box`-erased branch sink (fan-out Phase C FO-2) can be re-solved
    /// against its declared constraint on a mid-stream `CapsChanged`.
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_>;
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

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        AsyncElement::caps_constraint_as_sink(self)
    }
}
