use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;

use crate::caps::Caps;
use crate::clock::{ClockCandidate, ClockSync};
use crate::error::G2gError;
use crate::format_element::{legacy_sink_constraint, legacy_transform_constraint, CapsConstraint};
use crate::frame::PipelinePacket;
use crate::memory::{DomainSet, MemoryDomainKind};
use crate::property::{ElementMetadata, PropError, PropValue, PropertySpec};
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
    /// Downstream needs a keyframe now (no caps change): an encoder should emit
    /// an IDR / key frame on its next output. Originated by a WebRTC egress sink
    /// on a remote PLI (Picture Loss Indication) and carried up the reverse
    /// channel to the encoder. The GStreamer `GstForceKeyUnit` upstream-event
    /// analog.
    ForceKeyframe,
}

/// Downstream-originated quality-of-service signal: a synchronising sink is
/// running behind the pipeline clock and dropped a late frame. Travels upstream
/// along a link's reverse channel and is surfaced to the producing element as a
/// [`PushOutcome::Qos`], so a source / decoder can shed load (skip frames) to let
/// the pipeline catch up. The GStreamer QoS event analog (the
/// [`BusMessage::Qos`](crate::BusMessage::Qos) report is the out-of-band sibling
/// the application observes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QosMessage {
    /// How far past its deadline the dropped frame was, ns. Positive is late
    /// (behind the clock); the producer skips roughly this much stream time to
    /// catch up.
    pub jitter_ns: i64,
    /// Running time (PTS) of the late frame, for reference.
    pub running_time_ns: u64,
}

/// Outcome of pushing a packet downstream. Sources and transforms must
/// react to `Reconfigure` before pushing any further data; terminal sinks
/// and intermediate adapters that can't renegotiate may ignore it. `Qos` is
/// advisory: the packet still flowed, but the producer may shed load.
#[derive(Debug, Clone, PartialEq)]
pub enum PushOutcome {
    /// Downstream accepted the packet; continue normally.
    Accepted,
    /// Downstream signaled a reconfigure request between this push and
    /// the previous one. The producer should handle the request before
    /// pushing further `DataFrame`s.
    Reconfigure(Reconfigure),
    /// Downstream is behind the clock (a sink dropped a late frame). Advisory:
    /// the producer may skip ahead to shed load. Reconfigure takes priority when
    /// both are pending (negotiation correctness over QoS).
    Qos(QosMessage),
    /// Downstream reports a target send bitrate in bits/second (a WebRTC sink
    /// relaying its congestion-control / BWE estimate). Advisory: an encoder
    /// upstream should retarget its bitrate. Lowest priority of the reverse
    /// signals (Reconfigure > Qos > Bitrate); a held estimate surfaces on a
    /// later push, and BWE updates far slower than the frame rate. A target of
    /// `0` is the shed-layer idle hint (M722): the consumer downstream is
    /// discarding this stream (a starved simulcast layer), so the encoder
    /// should mostly stop encoding until a non-zero target resumes it.
    Bitrate(u32),
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

// Closed set: intentionally exhaustive (not #[non_exhaustive]); see STABILITY.md.
#[derive(Debug)]
pub enum ConfigureOutcome {
    Accepted,
    ReFixate(Caps),
}

impl ConfigureOutcome {
    /// Reject a mid-negotiation renegotiation request: at startup the caps handed
    /// to `configure_pipeline` are already fixated, so an element that answers
    /// `ReFixate` cannot be honored and is a hard error. Collapses the
    /// `if let ConfigureOutcome::ReFixate(_) = elem.configure_pipeline(..)? { return
    /// Err(FixationFailed) }` guard repeated across the runner startup paths.
    pub fn reject_refixate(self) -> Result<(), G2gError> {
        match self {
            ConfigureOutcome::Accepted => Ok(()),
            ConfigureOutcome::ReFixate(_) => Err(G2gError::FixationFailed),
        }
    }
}

pub trait AsyncElement: ElementBound {
    type ProcessFuture<'a>: Future<Output = Result<(), G2gError>> + 'a
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError>;

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError>;

    /// Receive this element's negotiated OUTPUT (source-pad) caps after the
    /// solve, alongside the input caps from [`configure_pipeline`] (M185). A
    /// geometry / format / rate-changing transform (videoscale, videoconvert,
    /// audioresample) uses this to take its target from a downstream capsfilter
    /// instead of its own properties, the gst caps-driven idiom. Default: no-op,
    /// so elements that don't need it (and runners that don't yet deliver it)
    /// are unaffected. Called only on transforms, with their single output
    /// link's caps; sources and sinks never receive it.
    fn configure_output(&mut self, _output_caps: &Caps) -> Result<(), G2gError> {
        Ok(())
    }

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

    /// The memory domain of the frames this element emits on its output pad.
    /// Default [`System`](MemoryDomainKind::System); a GPU producer (a hardware
    /// decoder emitting into VRAM, a wgpu/CUDA bridge) overrides it. Surfaced
    /// per edge by the negotiate-only path so the DOT dump can mark the GPU /
    /// zero-copy links (it is not part of `Caps`; see DESIGN.md 4.13.9).
    fn output_memory(&self) -> MemoryDomainKind {
        MemoryDomainKind::System
    }

    /// The full set of memory domains this element *can* emit on its output pad,
    /// not just its preferred one. The producer-capability half of the M351
    /// two-sided allocation-domain negotiation: the runner intersects it with the
    /// downstream consumers' acceptance set and settles on a single domain, so a
    /// decoder that can deliver to System *or* stay resident on the GPU lets the
    /// runner keep the frame copy-free when a downstream wants it. Default: just
    /// [`output_memory`](Self::output_memory), so a single-domain element
    /// negotiates exactly as before. A multi-domain producer overrides this.
    fn output_domains(&self) -> DomainSet {
        DomainSet::only(self.output_memory())
    }

    /// The memory domains this element can accept on its *input* pad (M354), for
    /// the domain-converter auto-plug. Default [`DomainSet::ALL`] (no requirement,
    /// so no converter is forced); a domain-strict element (a CUDA encoder/sink
    /// that needs device-resident input) narrows it, and the auto-plug splices a
    /// converter when the upstream cannot produce a domain in this set. A pure
    /// pass-through element (a memory-domain converter, an aggregator) leaves it
    /// `ALL`. Caps-free so the splice runs before the caps solve.
    fn input_domains(&self) -> DomainSet {
        DomainSet::ALL
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

    /// Take any QoS signal this element wants to send upstream, consuming it. A
    /// synchronising sink that dropped a late frame returns a [`QosMessage`]
    /// here; the runner forwards it onto the element's incoming link, where the
    /// producer observes it as [`PushOutcome::Qos`]. Called by the runner after
    /// each `process`. Default: nothing to send.
    fn take_qos(&mut self) -> Option<QosMessage> {
        None
    }

    /// Take any [`Reconfigure`] this element wants to send upstream, consuming
    /// it. The sink/transform analog of [`Self::take_qos`]: the runner forwards
    /// it onto the element's incoming link, where the producer observes it as
    /// [`PushOutcome::Reconfigure`]. The keyframe-request path uses this: a
    /// WebRTC egress sink that received a remote PLI returns
    /// [`Reconfigure::ForceKeyframe`] so the upstream encoder emits an IDR.
    /// Called by the runner after each `process`. Default: nothing to send.
    fn take_reconfigure(&mut self) -> Option<Reconfigure> {
        None
    }

    /// Take a target send bitrate (bits/second) this element wants to push
    /// upstream, consuming it. A WebRTC egress sink returns its latest
    /// congestion-control / BWE estimate here; the runner forwards it onto the
    /// incoming link, where the encoder observes it as [`PushOutcome::Bitrate`]
    /// and retargets. Called by the runner after each `process`. Default: none.
    fn take_bitrate(&mut self) -> Option<u32> {
        None
    }

    /// Whether this element consumes a downstream keyframe request
    /// (`PushOutcome::Reconfigure(ForceKeyframe)`) itself, i.e. it is an
    /// encoder that forces an IDR. Default `false`: the runner then relays the
    /// request onto the element's input link (M720), so a PLI crosses any
    /// number of pass-through transforms (a parser between the encoder and a
    /// WebRTC sink) to reach the encoder.
    fn handles_keyframe_requests(&self) -> bool {
        false
    }

    /// As [`Self::handles_keyframe_requests`], for a downstream bitrate target
    /// (`PushOutcome::Bitrate`): an encoder that retargets returns `true`.
    fn handles_bitrate_requests(&self) -> bool {
        false
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

    /// The metadata [`Transform`](crate::meta::Transform) this element applies to
    /// per-frame metadata, or `None` to opt out (the default). When `Some(t)` the
    /// runner clones each input frame's metadata,
    /// [`propagate(t)`](crate::meta::FrameMetaSet::propagate)s it, and attaches
    /// the survivors to output frames whose own metadata is empty
    /// (element-authored meta is never overwritten). `None` means the element
    /// carries meta through itself (a pass-through forwarding the same frame) or
    /// produces none, so the runner does nothing. Association is exact for a
    /// 1-in-1-out transform; a pipelined element gets most-recent-input
    /// association. See DESIGN.md 5.4.
    #[cfg(feature = "metadata")]
    fn meta_transform(&self) -> Option<crate::meta::Transform> {
        None
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

    /// Static introspection metadata for this element type (M178): the
    /// `gst-inspect` "Factory Details" (long-name / classification / description
    /// / author). Default: empty, like [`properties`](Self::properties). An
    /// element overrides it with a `const ElementMetadata` to document itself.
    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::default()
    }

    /// Receive this instance's log name (M179), assigned by the runner as
    /// `<category>N`. Default: ignore. An element that logs about itself stores
    /// it and returns it from its [`LogSource`](crate::log::LogSource) so its log
    /// lines carry the instance name.
    fn set_instance_name(&mut self, _name: alloc::string::String) {}

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

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError>;

    /// Dyn-safe mirror of [`AsyncElement::configure_output`] (M185). Defaults to
    /// no-op so unaffected erased elements need not implement it.
    fn configure_output(&mut self, _output_caps: &Caps) -> Result<(), G2gError> {
        Ok(())
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> core::pin::Pin<alloc::boxed::Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

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

    /// Dyn-safe mirror of [`AsyncElement::output_memory`]. Default
    /// [`System`](MemoryDomainKind::System).
    fn output_memory(&self) -> MemoryDomainKind {
        MemoryDomainKind::System
    }

    /// Dyn-safe mirror of [`AsyncElement::output_domains`]. Default
    /// `only(output_memory())`.
    fn output_domains(&self) -> DomainSet {
        DomainSet::only(self.output_memory())
    }

    /// Dyn-safe mirror of [`AsyncElement::input_domains`]. Default
    /// [`DomainSet::ALL`].
    fn input_domains(&self) -> DomainSet {
        DomainSet::ALL
    }

    /// Dyn-safe mirror of [`AsyncElement::meta_transform`]. Default `None`.
    #[cfg(feature = "metadata")]
    fn meta_transform(&self) -> Option<crate::meta::Transform> {
        None
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

    /// Dyn-safe mirror of [`AsyncElement::take_qos`], so an erased sink can send
    /// a QoS signal upstream. Defaults to nothing.
    fn take_qos(&mut self) -> Option<QosMessage> {
        None
    }

    /// Dyn-safe mirror of [`AsyncElement::take_reconfigure`], so an erased sink
    /// can request a keyframe / renegotiation upstream. Defaults to nothing.
    fn take_reconfigure(&mut self) -> Option<Reconfigure> {
        None
    }

    /// Dyn-safe mirror of [`AsyncElement::take_bitrate`], so an erased sink can
    /// push a target bitrate upstream. Defaults to nothing.
    fn take_bitrate(&mut self) -> Option<u32> {
        None
    }

    /// Dyn-safe mirror of [`AsyncElement::handles_keyframe_requests`] (M720).
    fn handles_keyframe_requests(&self) -> bool {
        false
    }

    /// Dyn-safe mirror of [`AsyncElement::handles_bitrate_requests`] (M720).
    fn handles_bitrate_requests(&self) -> bool {
        false
    }

    /// Dyn-safe mirror of [`AsyncElement::properties`], so a `gst-inspect` dump
    /// and the `gst-launch` parser can introspect / set an erased element.
    fn properties(&self) -> &'static [PropertySpec] {
        &[]
    }

    /// Dyn-safe mirror of [`AsyncElement::metadata`], so a `gst-inspect` dump can
    /// read an erased element's "Factory Details". Defaults to empty.
    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::default()
    }

    /// The log category for this erased element (M179): its short type name by
    /// default (the blanket impl fills it from `core::any::type_name`), so the
    /// runner can name (`<category>N`) and log about any element. Filtering key.
    fn log_category(&self) -> &'static str {
        "element"
    }

    /// Dyn-safe mirror of [`AsyncElement::set_instance_name`], so the runner can
    /// name an erased element instance for logging.
    fn set_instance_name(&mut self, _name: alloc::string::String) {}

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

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        AsyncElement::configure_pipeline(self, absolute_caps)
    }

    fn configure_output(&mut self, output_caps: &Caps) -> Result<(), G2gError> {
        AsyncElement::configure_output(self, output_caps)
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

    fn output_memory(&self) -> MemoryDomainKind {
        AsyncElement::output_memory(self)
    }

    fn output_domains(&self) -> DomainSet {
        AsyncElement::output_domains(self)
    }

    fn input_domains(&self) -> DomainSet {
        AsyncElement::input_domains(self)
    }

    #[cfg(feature = "metadata")]
    fn meta_transform(&self) -> Option<crate::meta::Transform> {
        AsyncElement::meta_transform(self)
    }

    fn provide_clock(&self) -> Option<ClockCandidate> {
        AsyncElement::provide_clock(self)
    }

    fn set_clock_sync(&mut self, sync: ClockSync) {
        AsyncElement::set_clock_sync(self, sync)
    }

    fn take_qos(&mut self) -> Option<QosMessage> {
        AsyncElement::take_qos(self)
    }

    fn take_reconfigure(&mut self) -> Option<Reconfigure> {
        AsyncElement::take_reconfigure(self)
    }

    fn take_bitrate(&mut self) -> Option<u32> {
        AsyncElement::take_bitrate(self)
    }

    fn handles_keyframe_requests(&self) -> bool {
        AsyncElement::handles_keyframe_requests(self)
    }

    fn handles_bitrate_requests(&self) -> bool {
        AsyncElement::handles_bitrate_requests(self)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        AsyncElement::properties(self)
    }

    fn metadata(&self) -> ElementMetadata {
        AsyncElement::metadata(self)
    }

    fn log_category(&self) -> &'static str {
        crate::log::short_type_name::<T>()
    }

    fn set_instance_name(&mut self, name: alloc::string::String) {
        AsyncElement::set_instance_name(self, name)
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

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        (**self).configure_pipeline(absolute_caps)
    }

    fn configure_output(&mut self, output_caps: &Caps) -> Result<(), G2gError> {
        (**self).configure_output(output_caps)
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

    fn output_memory(&self) -> MemoryDomainKind {
        (**self).output_memory()
    }

    fn output_domains(&self) -> DomainSet {
        (**self).output_domains()
    }

    fn input_domains(&self) -> DomainSet {
        (**self).input_domains()
    }

    #[cfg(feature = "metadata")]
    fn meta_transform(&self) -> Option<crate::meta::Transform> {
        (**self).meta_transform()
    }

    fn provide_clock(&self) -> Option<ClockCandidate> {
        (**self).provide_clock()
    }

    fn set_clock_sync(&mut self, sync: ClockSync) {
        (**self).set_clock_sync(sync)
    }

    fn take_qos(&mut self) -> Option<QosMessage> {
        (**self).take_qos()
    }

    fn take_reconfigure(&mut self) -> Option<Reconfigure> {
        (**self).take_reconfigure()
    }

    fn take_bitrate(&mut self) -> Option<u32> {
        (**self).take_bitrate()
    }

    fn handles_keyframe_requests(&self) -> bool {
        (**self).handles_keyframe_requests()
    }

    fn handles_bitrate_requests(&self) -> bool {
        (**self).handles_bitrate_requests()
    }

    fn properties(&self) -> &'static [PropertySpec] {
        (**self).properties()
    }

    fn metadata(&self) -> ElementMetadata {
        (**self).metadata()
    }

    fn log_category(&self) -> &'static str {
        (**self).log_category()
    }

    fn set_instance_name(&mut self, name: alloc::string::String) {
        (**self).set_instance_name(name)
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        (**self).set_property(name, value)
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        (**self).get_property(name)
    }
}
