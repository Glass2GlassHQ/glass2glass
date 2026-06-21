use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;

use crate::caps::Caps;
use crate::clock::{ClockCandidate, ClockSync};
use crate::error::G2gError;
use crate::format_element::{legacy_sink_constraint, legacy_transform_constraint, CapsConstraint};
use crate::frame::PipelinePacket;
use crate::property::{PropError, PropValue, PropertySpec};
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

    /// Receive the pipeline's elected clock + base time after election, so a
    /// sink can present each frame at its running-time deadline (PTS mapped
    /// through the active `Segment`) — the "use PTS to decide when to display"
    /// path. The runner calls this once before streaming, only when a clock was
    /// elected. Default: ignore (present as fast as backpressure allows, the
    /// pre-sync behaviour). See [`ClockSync`].
    fn set_clock_sync(&mut self, _sync: ClockSync) {}

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

    /// The runtime properties this element type exposes (M104), the GObject
    /// property-spec analog. Default: none. An element overrides this (and
    /// [`set_property`](Self::set_property) / [`get_property`](Self::get_property))
    /// to be settable by name from a `gst-launch` pipeline or inspectable by a
    /// `gst-inspect` dump. The `with_*` builders remain the zero-cost
    /// construction path; this is the string-keyed runtime face.
    fn properties(&self) -> &'static [PropertySpec] {
        &[]
    }

    /// Set a property by name (M104). Default: every name is
    /// [`PropError::Unknown`] (no properties). An overriding element validates
    /// the value kind against its [`properties`](Self::properties) spec and
    /// applies it.
    fn set_property(&mut self, _name: &str, _value: PropValue) -> Result<(), PropError> {
        Err(PropError::Unknown)
    }

    /// Read a property back by name (M104). Default: `None`. Overriding elements
    /// return the current value for a known property.
    fn get_property(&self, _name: &str) -> Option<PropValue> {
        None
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

    /// Dyn-safe mirror of [`AsyncElement::caps_constraint_as_transform`], so
    /// an interior element of an N-element linear chain (`run_linear_chain`)
    /// declares its transform constraint to the solver while erased.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_>;

    /// Dyn-safe mirror of [`AsyncElement::propose_allocation`], so a
    /// `Box`-erased branch sink can re-derive its own pool on a mid-stream
    /// caps change (fan-out element-local α).
    fn propose_allocation(&self, caps: &Caps) -> Option<AllocationParams>;

    /// Dyn-safe mirror of [`AsyncElement::configure_allocation`].
    fn configure_allocation(&mut self, params: &AllocationParams);

    /// Dyn-safe mirror of [`AsyncElement::latency`], so a buffering interior
    /// element of an N-element chain (`run_linear_chain`) contributes to the
    /// runner's latency fold. Defaults to zero, matching `AsyncElement`.
    fn latency(&self) -> LatencyReport {
        LatencyReport::ZERO
    }

    /// Dyn-safe mirror of [`AsyncElement::provide_clock`], so an interior
    /// element that paces to hardware joins the runner's clock election.
    /// Defaults to none.
    fn provide_clock(&self) -> Option<ClockCandidate> {
        None
    }

    /// Dyn-safe mirror of [`AsyncElement::set_clock_sync`], so an erased sink
    /// receives the elected clock + base time. Defaults to ignore.
    fn set_clock_sync(&mut self, _sync: ClockSync) {}

    /// Dyn-safe mirror of [`AsyncElement::properties`], so a `gst-inspect` dump
    /// and the `gst-launch` parser can introspect / set an erased element.
    fn properties(&self) -> &'static [PropertySpec] {
        &[]
    }

    /// Dyn-safe mirror of [`AsyncElement::set_property`]. Defaults to "no
    /// properties" so a hand-written `DynAsyncElement` need not implement it; the
    /// blanket `impl<T: AsyncElement>` overrides it to forward to the element.
    fn set_property(&mut self, _name: &str, _value: PropValue) -> Result<(), PropError> {
        Err(PropError::Unknown)
    }

    /// Dyn-safe mirror of [`AsyncElement::get_property`]. Defaults to `None`; the
    /// blanket impl forwards to the element.
    fn get_property(&self, _name: &str) -> Option<PropValue> {
        None
    }
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

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        AsyncElement::caps_constraint_as_transform(self)
    }

    fn propose_allocation(&self, caps: &Caps) -> Option<AllocationParams> {
        AsyncElement::propose_allocation(self, caps)
    }

    fn configure_allocation(&mut self, params: &AllocationParams) {
        AsyncElement::configure_allocation(self, params)
    }

    fn latency(&self) -> LatencyReport {
        AsyncElement::latency(self)
    }

    fn provide_clock(&self) -> Option<ClockCandidate> {
        AsyncElement::provide_clock(self)
    }

    fn set_clock_sync(&mut self, sync: ClockSync) {
        AsyncElement::set_clock_sync(self, sync)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        AsyncElement::properties(self)
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        AsyncElement::set_property(self, name, value)
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        AsyncElement::get_property(self, name)
    }
}

/// Forwarding impl so a borrowed `&mut dyn DynAsyncElement` can be boxed into a
/// `Box<dyn DynAsyncElement + 'a>` graph node (the convenience wrappers build a
/// borrowing `Graph` over their `&mut` element references). Disjoint from the
/// `AsyncElement` blanket above: a `&mut dyn DynAsyncElement` does not implement
/// `AsyncElement`.
#[cfg(feature = "std")]
impl<'b> DynAsyncElement for &'b mut (dyn DynAsyncElement + 'b) {
    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        (**self).intercept_caps(upstream_caps)
    }

    fn configure_pipeline(
        &mut self,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        (**self).configure_pipeline(absolute_caps)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> BoxFuture<'a, Result<(), G2gError>> {
        (**self).process(packet, out)
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        (**self).caps_constraint_as_sink()
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        (**self).caps_constraint_as_transform()
    }

    fn propose_allocation(&self, caps: &Caps) -> Option<AllocationParams> {
        (**self).propose_allocation(caps)
    }

    fn configure_allocation(&mut self, params: &AllocationParams) {
        (**self).configure_allocation(params)
    }

    fn latency(&self) -> LatencyReport {
        (**self).latency()
    }

    fn provide_clock(&self) -> Option<ClockCandidate> {
        (**self).provide_clock()
    }

    fn set_clock_sync(&mut self, sync: ClockSync) {
        (**self).set_clock_sync(sync)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        (**self).properties()
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        (**self).set_property(name, value)
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        (**self).get_property(name)
    }
}
