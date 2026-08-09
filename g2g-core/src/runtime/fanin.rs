//! Fan-in runner for the [`Merger`](crate::fanout::Merger) primitive (M9).
//!
//! Drives `N sources → Merger → 1 sink`. Each input is drained by its own
//! forwarder future; the forwarder whose index equals the merger's atomic
//! selection pushes its frames to a single shared output link, the others
//! discard (so no source stalls). The merged stream emits one `Eos` only
//! after **every** input has ended (all-inputs-EOS aggregation), so no
//! upstream branch is stranded.
//!
//! Heterogeneous branches arrive as `Box`-erased `&mut dyn DynSourceLoop`.
//! `DynSourceLoop` is defined here, not in `runner.rs`, so that runner's
//! generic `SourceLoop` calls stay unambiguous (the same reason
//! `DynAsyncElement` lives apart from the runner).

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::future::Future;
use core::sync::atomic::{AtomicUsize, Ordering};

use super::autoplug::PadRequest;
use crate::bus::BusHandle;
use crate::caps::{Caps, CapsSet};
use crate::clock::{ClockCandidate, ClockPriority};
use crate::clock::{DynAsyncClock, PipelineClock};
use crate::element::{
    AsyncElement, BoxFuture, ConfigureOutcome, DynAsyncElement, ElementBound, OutputSink,
    PushOutcome, Reconfigure,
};
use crate::error::G2gError;
use crate::fanout::{
    DuplexInbound, Merger, MultiDuplexSession, MultiInputElement, MultiSenderSink,
};
use crate::format_element::{CapsConstraint, CapsPreferences};
use crate::frame::PipelinePacket;
use crate::graph::Graph;
use crate::memory::{DomainSet, MemoryDomainKind};
use crate::property::{ElementMetadata, PropError, PropValue, PropertySpec};
use crate::query::{AllocationParams, LatencyReport};
use crate::runtime::channel::{
    bounded, packet_bytes, ProbeAction, ProbeSlot, Receiver, SendError, Sender, SenderSink,
};
use crate::runtime::coordinator::log_caps_rejected;
use crate::runtime::graph_runner::{run_graph_inner, GraphNodeRef};
use crate::runtime::instrument::{EdgeCounters, ElementProbe};
use crate::runtime::join::{dynamic_join, join_all, select2, Either};
use crate::runtime::observe::{link_tapped, register_runner_tap, EdgeTap, TapEdge, TapNode};
use crate::runtime::runner::{LinkCapacity, NullSink, RunStats, SourceLoop};
use crate::runtime::{NodeRole, Observer};

/// Dyn-safe mirror of [`SourceLoop`] for heterogeneous fan-in branches, the
/// source-side analog of [`DynAsyncElement`](crate::element::DynAsyncElement).
/// Boxes `run`'s future so a `Vec<&mut dyn DynSourceLoop>` can hold sources
/// of different concrete types.
pub trait DynSourceLoop: ElementBound {
    fn intercept_caps<'a>(&'a mut self) -> BoxFuture<'a, Result<Caps, G2gError>>;

    /// Dyn-safe mirror of the produce set behind
    /// [`SourceLoop::caps_constraint`], so the DAG runner negotiates a source
    /// that offers alternatives (a camera's pixel formats) instead of only its
    /// preferred one. The default is the single-caps answer; the blanket impl
    /// reads the constraint the source declares.
    fn produced_caps<'a>(&'a mut self) -> BoxFuture<'a, Result<CapsSet, G2gError>> {
        Box::pin(async move { Ok(CapsSet::one(self.intercept_caps().await?)) })
    }

    /// Dyn-safe mirror of [`SourceLoop::caps_preferences`], so an erased
    /// source's declared per-alternative costs reach the solver. Defaults to
    /// `None` (cost = alternative index), matching `SourceLoop`.
    fn caps_preferences(&self) -> Option<CapsPreferences> {
        None
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError>;

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> BoxFuture<'a, Result<u64, G2gError>>;

    fn reconfigure(&mut self, request: Reconfigure) -> Result<Caps, G2gError>;

    /// Dyn-safe mirror of [`SourceLoop::latency`], so the DAG runner folds a
    /// source's latency contribution like the linear runner does.
    fn latency(&self) -> LatencyReport;

    /// Dyn-safe mirror of [`SourceLoop::output_memory`]. Default
    /// [`System`](MemoryDomainKind::System).
    fn output_memory(&self) -> MemoryDomainKind {
        MemoryDomainKind::System
    }

    /// Dyn-safe mirror of [`SourceLoop::output_domains`]. Default
    /// `only(output_memory())`.
    fn output_domains(&self) -> DomainSet {
        DomainSet::only(self.output_memory())
    }

    /// Dyn-safe mirror of [`SourceLoop::query_duration`] (M203), so the DAG
    /// runner can publish an erased source's duration on the progress handle.
    fn query_duration(&self) -> Option<u64> {
        None
    }

    /// Dyn-safe mirror of [`SourceLoop::provide_clock`], for the runner's clock
    /// election.
    fn provide_clock(&self) -> Option<ClockCandidate>;

    /// Dyn-safe mirror of [`SourceLoop::configure_allocation`], the upstream end
    /// of the M12 allocation cascade.
    fn configure_allocation(&mut self, params: &AllocationParams);

    /// Dyn-safe mirror of [`SourceLoop::configured_output_caps`] (M195), so the
    /// `decodebin` parser can read an erased source's property-driven caps.
    fn configured_output_caps(&self) -> Option<Caps> {
        None
    }

    /// Dyn-safe mirror of [`SourceLoop::probe_output_caps`] (M480): the parse-time
    /// caps `decodebin` uses to pick a demuxer, allowed to sniff the header.
    fn probe_output_caps(&mut self) -> Option<Caps> {
        self.configured_output_caps()
    }

    /// Dyn-safe mirror of [`SourceLoop::properties`], for `gst-inspect` /
    /// `gst-launch` introspection of an erased source.
    fn properties(&self) -> &'static [PropertySpec] {
        &[]
    }

    /// Dyn-safe mirror of [`SourceLoop::metadata`], for the `gst-inspect`
    /// "Factory Details" of an erased source. Defaults to empty.
    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::default()
    }

    /// The log category for this erased source (M179): its short type name by
    /// default (the blanket impl fills it), so the runner can name and log it.
    fn log_category(&self) -> &'static str {
        "source"
    }

    /// Dyn-safe mirror of [`SourceLoop::set_instance_name`].
    fn set_instance_name(&mut self, _name: alloc::string::String) {}

    /// Dyn-safe mirror of [`SourceLoop::set_log_category`].
    fn set_log_category(&mut self, _category: alloc::string::String) {}

    /// Dyn-safe mirror of [`SourceLoop::set_property`]. Defaults to "no
    /// properties"; the blanket `impl<T: SourceLoop>` overrides it to forward.
    fn set_property(&mut self, _name: &str, _value: PropValue) -> Result<(), PropError> {
        Err(PropError::Unknown)
    }

    /// Dyn-safe mirror of [`SourceLoop::get_property`]. Defaults to `None`; the
    /// blanket impl forwards to the source.
    fn get_property(&self, _name: &str) -> Option<PropValue> {
        None
    }
}

/// Blanket adapter: every [`SourceLoop`] is usable as a [`DynSourceLoop`]
/// by boxing its `run` and `intercept_caps` futures. Calls are
/// disambiguated to `SourceLoop::` because the two traits share method
/// names.
impl<T: SourceLoop> DynSourceLoop for T {
    fn intercept_caps<'a>(&'a mut self) -> BoxFuture<'a, Result<Caps, G2gError>> {
        Box::pin(SourceLoop::intercept_caps(self))
    }

    /// A source shape is `Produces` (native) or `LegacySource` (the migration
    /// bridge). Any other variant means a sink / transform constraint on a source
    /// slot, which is an element bug, so it fails loud rather than negotiating
    /// something the source never offered.
    fn produced_caps<'a>(&'a mut self) -> BoxFuture<'a, Result<CapsSet, G2gError>> {
        Box::pin(async move {
            match SourceLoop::caps_constraint(self).await? {
                CapsConstraint::Produces(set) => Ok(set),
                CapsConstraint::LegacySource(caps) => Ok(CapsSet::one(caps)),
                _ => Err(G2gError::CapsMismatch),
            }
        })
    }

    fn caps_preferences(&self) -> Option<CapsPreferences> {
        SourceLoop::caps_preferences(self)
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        SourceLoop::configure_pipeline(self, absolute_caps)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> BoxFuture<'a, Result<u64, G2gError>> {
        Box::pin(SourceLoop::run(self, out))
    }

    fn reconfigure(&mut self, request: Reconfigure) -> Result<Caps, G2gError> {
        SourceLoop::reconfigure(self, request)
    }

    fn latency(&self) -> LatencyReport {
        SourceLoop::latency(self)
    }

    fn output_memory(&self) -> MemoryDomainKind {
        SourceLoop::output_memory(self)
    }

    fn output_domains(&self) -> DomainSet {
        SourceLoop::output_domains(self)
    }

    fn query_duration(&self) -> Option<u64> {
        SourceLoop::query_duration(self)
    }

    fn provide_clock(&self) -> Option<ClockCandidate> {
        SourceLoop::provide_clock(self)
    }

    fn configure_allocation(&mut self, params: &AllocationParams) {
        SourceLoop::configure_allocation(self, params)
    }

    fn configured_output_caps(&self) -> Option<Caps> {
        SourceLoop::configured_output_caps(self)
    }

    fn probe_output_caps(&mut self) -> Option<Caps> {
        SourceLoop::probe_output_caps(self)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        SourceLoop::properties(self)
    }

    fn metadata(&self) -> ElementMetadata {
        SourceLoop::metadata(self)
    }

    fn log_category(&self) -> &'static str {
        crate::log::short_type_name::<T>()
    }

    fn set_instance_name(&mut self, name: alloc::string::String) {
        SourceLoop::set_instance_name(self, name)
    }

    fn set_log_category(&mut self, category: alloc::string::String) {
        SourceLoop::set_log_category(self, category)
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        SourceLoop::set_property(self, name, value)
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        SourceLoop::get_property(self, name)
    }
}

/// Forwarding impl so a borrowed `&mut dyn DynSourceLoop` can be boxed into a
/// `Box<dyn DynSourceLoop + 'a>` graph node (the muxer/fan-out wrappers build a
/// borrowing `Graph` over their `&mut` source references). Disjoint from the
/// `SourceLoop` blanket above: a `&mut dyn DynSourceLoop` is not a `SourceLoop`.
impl<'b> DynSourceLoop for &'b mut (dyn DynSourceLoop + 'b) {
    fn intercept_caps<'a>(&'a mut self) -> BoxFuture<'a, Result<Caps, G2gError>> {
        (**self).intercept_caps()
    }

    fn produced_caps<'a>(&'a mut self) -> BoxFuture<'a, Result<CapsSet, G2gError>> {
        (**self).produced_caps()
    }

    fn caps_preferences(&self) -> Option<CapsPreferences> {
        (**self).caps_preferences()
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        (**self).configure_pipeline(absolute_caps)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> BoxFuture<'a, Result<u64, G2gError>> {
        (**self).run(out)
    }

    fn reconfigure(&mut self, request: Reconfigure) -> Result<Caps, G2gError> {
        (**self).reconfigure(request)
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

    fn query_duration(&self) -> Option<u64> {
        (**self).query_duration()
    }

    fn provide_clock(&self) -> Option<ClockCandidate> {
        (**self).provide_clock()
    }

    fn configure_allocation(&mut self, params: &AllocationParams) {
        (**self).configure_allocation(params)
    }

    fn configured_output_caps(&self) -> Option<Caps> {
        (**self).configured_output_caps()
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

    fn set_log_category(&mut self, category: alloc::string::String) {
        (**self).set_log_category(category)
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        (**self).set_property(name, value)
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        (**self).get_property(name)
    }
}

/// The caps one fan-in branch runs: the highest-preference alternative of the
/// source's produce set that the pad it feeds accepts (M955). A source offering
/// several formats (a camera's pixel formats) is therefore selected by what the
/// merge point takes, the way the DAG runner selects it by what downstream takes.
///
/// When the pad accepts none of them the preferred alternative is returned
/// anyway, leaving the rejection to that pad's `configure_pipeline`, which is
/// where it was raised before a source had a set to choose from.
fn select_branch_caps(produced: &CapsSet, pad: &CapsConstraint<'_>) -> Result<Caps, G2gError> {
    let accepted = produced
        .alternatives()
        .iter()
        .filter_map(|alt| alt.fixate().ok())
        .find(|fixed| pad.accepts(fixed));
    match accepted {
        Some(caps) => Ok(caps),
        None => produced.fixate().ok_or(G2gError::CapsMismatch),
    }
}

/// Dyn-safe mirror of [`MultiInputElement`] for a fan-in muxer node in the DAG
/// runner (`run_graph`). Boxes `process`'s future and forwards the
/// `Self: Sized` constraint methods, the same shape as [`DynSourceLoop`] /
/// [`DynAsyncElement`](crate::element::DynAsyncElement). Only the methods the
/// runner uses are mirrored (the per-input `intercept_caps` / `output_caps`
/// legacy paths stay on the concrete trait).
pub trait DynMultiInputElement: ElementBound {
    fn input_count(&self) -> usize;
    /// Dyn-safe mirror of [`MultiInputElement::input_pts_ordered`]: whether the
    /// runner delivers inputs in global PTS order rather than arrival order.
    fn input_pts_ordered(&self) -> bool;
    /// Dyn-safe mirror of [`MultiInputElement::output_follows_input`]: the input
    /// pad whose caps the merged output follows (identity-passthrough mux), if any.
    fn output_follows_input(&self) -> Option<usize>;
    /// Dyn-safe mirror of [`MultiInputElement::tick_interval_ns`]: the deadline
    /// tick period this element wants, if any.
    fn tick_interval_ns(&self) -> Option<u64> {
        None
    }
    /// Dyn-safe mirror of [`MultiInputElement::input_pad_index`] (M481): map a
    /// named request pad to the concrete input index, for the launch parser.
    fn input_pad_index(&self, req: &PadRequest, ordinal: usize) -> Option<usize>;
    fn caps_constraint_as_input(&self, input: usize) -> CapsConstraint<'_>;
    fn caps_constraint_for_output(&self) -> Result<CapsConstraint<'_>, G2gError>;
    /// Dyn-safe mirror of [`MultiInputElement::propose_allocation_for_input`].
    fn propose_allocation_for_input(&self, input: usize, caps: &Caps) -> Option<AllocationParams>;
    /// Dyn-safe mirror of [`MultiInputElement::propose_allocation_for_output`].
    fn propose_allocation_for_output(&self, caps: &Caps) -> Option<AllocationParams>;
    /// Dyn-safe mirror of [`MultiInputElement::configure_allocation_for_output`].
    fn configure_allocation_for_output(&mut self, params: &AllocationParams);
    /// Dyn-safe mirror of [`MultiInputElement::output_caps`].
    fn output_caps(&self) -> Result<Caps, G2gError>;
    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError>;
    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> BoxFuture<'a, Result<(), G2gError>>;
    fn properties(&self) -> &'static [PropertySpec];
    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError>;
    fn get_property(&self, name: &str) -> Option<PropValue>;
    /// Dyn-safe mirror of [`MultiInputElement::metadata`], for the `gst-inspect`
    /// "Factory Details" of an erased fan-in muxer.
    fn metadata(&self) -> ElementMetadata;

    /// Dyn-safe mirror of [`MultiInputElement::reverse_channel`], so a terminal
    /// fan-in node's arm can route a per-input reverse signal (WebRTC PLI / BWE)
    /// back to the upstream feeding that pad. Default `None`.
    fn reverse_channel(&self, _input: usize) -> Option<crate::fanout::ReverseChannel> {
        None
    }

    /// Dyn-safe mirror of [`MultiInputElement::is_terminal`].
    fn is_terminal(&self) -> bool {
        false
    }

    /// Dyn-safe mirror of [`MultiInputElement::set_instance_name`], so the runner
    /// can name an erased muxer instance for logging.
    fn set_instance_name(&mut self, _name: alloc::string::String) {}

    /// Dyn-safe mirror of [`MultiInputElement::set_log_category`].
    fn set_log_category(&mut self, _category: alloc::string::String) {}
}

impl<T: MultiInputElement> DynMultiInputElement for T {
    fn input_count(&self) -> usize {
        MultiInputElement::input_count(self)
    }

    fn input_pts_ordered(&self) -> bool {
        MultiInputElement::input_pts_ordered(self)
    }

    fn output_follows_input(&self) -> Option<usize> {
        MultiInputElement::output_follows_input(self)
    }

    fn tick_interval_ns(&self) -> Option<u64> {
        MultiInputElement::tick_interval_ns(self)
    }

    fn input_pad_index(&self, req: &PadRequest, ordinal: usize) -> Option<usize> {
        MultiInputElement::input_pad_index(self, req, ordinal)
    }

    fn caps_constraint_as_input(&self, input: usize) -> CapsConstraint<'_> {
        MultiInputElement::caps_constraint_as_input(self, input)
    }

    fn caps_constraint_for_output(&self) -> Result<CapsConstraint<'_>, G2gError> {
        MultiInputElement::caps_constraint_for_output(self)
    }

    fn propose_allocation_for_input(&self, input: usize, caps: &Caps) -> Option<AllocationParams> {
        MultiInputElement::propose_allocation_for_input(self, input, caps)
    }

    fn propose_allocation_for_output(&self, caps: &Caps) -> Option<AllocationParams> {
        MultiInputElement::propose_allocation_for_output(self, caps)
    }

    fn configure_allocation_for_output(&mut self, params: &AllocationParams) {
        MultiInputElement::configure_allocation_for_output(self, params)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        MultiInputElement::output_caps(self)
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        MultiInputElement::configure_pipeline(self, input, absolute_caps)
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> BoxFuture<'a, Result<(), G2gError>> {
        Box::pin(MultiInputElement::process(self, input, packet, out))
    }

    fn properties(&self) -> &'static [PropertySpec] {
        MultiInputElement::properties(self)
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        MultiInputElement::set_property(self, name, value)
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        MultiInputElement::get_property(self, name)
    }

    fn metadata(&self) -> ElementMetadata {
        MultiInputElement::metadata(self)
    }

    fn reverse_channel(&self, input: usize) -> Option<crate::fanout::ReverseChannel> {
        MultiInputElement::reverse_channel(self, input)
    }

    fn is_terminal(&self) -> bool {
        MultiInputElement::is_terminal(self)
    }

    fn set_instance_name(&mut self, name: alloc::string::String) {
        MultiInputElement::set_instance_name(self, name)
    }

    fn set_log_category(&mut self, category: alloc::string::String) {
        MultiInputElement::set_log_category(self, category)
    }
}

/// Forwarding impl so a borrowed `&mut dyn DynMultiInputElement` can be boxed
/// into a `Box<dyn DynMultiInputElement + 'a>` graph node (the muxer wrapper
/// builds a borrowing `Graph` over its `&mut` muxer reference). Disjoint from
/// the `MultiInputElement` blanket above.
impl<'b> DynMultiInputElement for &'b mut (dyn DynMultiInputElement + 'b) {
    fn input_count(&self) -> usize {
        (**self).input_count()
    }

    fn input_pts_ordered(&self) -> bool {
        (**self).input_pts_ordered()
    }

    fn output_follows_input(&self) -> Option<usize> {
        (**self).output_follows_input()
    }

    fn tick_interval_ns(&self) -> Option<u64> {
        (**self).tick_interval_ns()
    }

    fn input_pad_index(&self, req: &PadRequest, ordinal: usize) -> Option<usize> {
        (**self).input_pad_index(req, ordinal)
    }

    fn caps_constraint_as_input(&self, input: usize) -> CapsConstraint<'_> {
        (**self).caps_constraint_as_input(input)
    }

    fn caps_constraint_for_output(&self) -> Result<CapsConstraint<'_>, G2gError> {
        (**self).caps_constraint_for_output()
    }

    fn propose_allocation_for_input(&self, input: usize, caps: &Caps) -> Option<AllocationParams> {
        (**self).propose_allocation_for_input(input, caps)
    }

    fn propose_allocation_for_output(&self, caps: &Caps) -> Option<AllocationParams> {
        (**self).propose_allocation_for_output(caps)
    }

    fn configure_allocation_for_output(&mut self, params: &AllocationParams) {
        (**self).configure_allocation_for_output(params)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        (**self).output_caps()
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        (**self).configure_pipeline(input, absolute_caps)
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> BoxFuture<'a, Result<(), G2gError>> {
        (**self).process(input, packet, out)
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

    fn metadata(&self) -> ElementMetadata {
        (**self).metadata()
    }

    fn reverse_channel(&self, input: usize) -> Option<crate::fanout::ReverseChannel> {
        (**self).reverse_channel(input)
    }

    fn is_terminal(&self) -> bool {
        (**self).is_terminal()
    }

    fn set_instance_name(&mut self, name: alloc::string::String) {
        (**self).set_instance_name(name)
    }

    fn set_log_category(&mut self, category: alloc::string::String) {
        (**self).set_log_category(category)
    }
}

/// Drives `N sources → Merger → 1 sink` (M9 fan-in). The `Merger` selects
/// which input feeds the sink; the others are drained. The merged stream
/// ends once every input has reached EOS.
///
/// Negotiation fixates each source's proposal independently and configures
/// the sink against input 0's fixated caps (the merged-output caps);
/// per-input caps negotiation is M10, so a `ReFixate` anywhere fails with
/// `FixationFailed`. The slice assumes inputs agree (the A/B case).
pub async fn run_fanin_sink<Snk, Clk>(
    sources: Vec<&mut dyn DynSourceLoop>,
    merger: &mut Merger,
    sink: &mut Snk,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
) -> Result<RunStats, G2gError>
where
    Snk: AsyncElement,
    Clk: PipelineClock,
{
    run_fanin_sink_inner(sources, merger, sink, clock, link_capacity, None).await
}

/// As [`run_fanin_sink`], but taps live telemetry into `observer` (M846), the
/// hand-built analog of [`run_graph_observed`](crate::runtime::run_graph_observed):
/// the topology is the N sources, the merger, and the sink, and a concurrent task
/// reads the sink's measured `process()` latency plus each link's packet / byte /
/// drop counters mid-run via [`Observer::snapshot`].
pub async fn run_fanin_sink_observed<Snk, Clk>(
    sources: Vec<&mut dyn DynSourceLoop>,
    merger: &mut Merger,
    sink: &mut Snk,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    observer: &Observer,
) -> Result<RunStats, G2gError>
where
    Snk: AsyncElement,
    Clk: PipelineClock,
{
    run_fanin_sink_inner(sources, merger, sink, clock, link_capacity, Some(observer)).await
}

async fn run_fanin_sink_inner<Snk, Clk>(
    sources: Vec<&mut dyn DynSourceLoop>,
    merger: &mut Merger,
    sink: &mut Snk,
    _clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    observer: Option<&Observer>,
) -> Result<RunStats, G2gError>
where
    Snk: AsyncElement,
    Clk: PipelineClock,
{
    let link_capacity: usize = link_capacity.into().get();
    let input_count = sources.len();
    assert!(input_count > 0, "fan-in needs at least one source");
    assert!(
        merger.input_count() == input_count,
        "merger input count must match the number of sources"
    );

    let mut sources = sources;
    // M842: instance naming + lifecycle logging, as in `run_graph`. The `Merger`
    // carries no element payload (it selects, like a structural tee), so it is
    // not named, matching the tee case there.
    let mut namer = crate::log::InstanceNamer::new();
    let mut source_names: Vec<alloc::string::String> = Vec::with_capacity(input_count);
    for source in sources.iter_mut() {
        let name = namer.add(source.log_category(), None);
        source.set_instance_name(name.clone());
        source_names.push(name);
    }
    let sink_name = namer.add(crate::log::short_type_name::<Snk>(), None);
    AsyncElement::set_instance_name(sink, sink_name.clone());
    // M846: the sink is the only node here with a `process()`, so it carries the
    // run's measured-latency probe (as in the linear runners).
    let sink_probe = ElementProbe::new(sink_name);

    // Phase 1 + 2: fixate each source's caps and configure it; the sink is
    // configured against input 0's fixated caps (the merged-output caps). This
    // is not routed through `solve_linear` because the branches do not form a
    // chain: each is narrowed against the one peer they share, the sink.
    let mut fixated_caps: Vec<Caps> = Vec::with_capacity(input_count);
    for source in sources.iter_mut() {
        let produced = source.produced_caps().await?;
        let fixated = {
            let sink_constraint = sink.caps_constraint_as_sink();
            select_branch_caps(&produced, &sink_constraint)?
        };
        source.configure_pipeline(&fixated)?.reject_refixate()?;
        fixated_caps.push(fixated);
    }
    let merged_caps = fixated_caps[0].clone();
    sink.configure_pipeline(&merged_caps)?.reject_refixate()?;

    // One input link per source, one shared output link to the sink.
    let tap = observer.is_some();
    let mut input_senders = Vec::with_capacity(input_count);
    let mut input_receivers = Vec::with_capacity(input_count);
    let mut taps = Vec::with_capacity(input_count + 1);
    for _ in 0..input_count {
        let (tx, rx, edge) = link_tapped(link_capacity, tap);
        input_senders.push(tx);
        input_receivers.push(rx);
        taps.push(edge);
    }
    let (out_tx, out_rx, out_tap) = link_tapped(link_capacity, tap);
    taps.push(out_tap);
    let live_inputs = Arc::new(AtomicUsize::new(input_count));

    // Dev-tooling tap: sources 0..N, then the merger (structural, unnamed, like a
    // tee in `run_graph`), then the sink.
    if let Some(obs) = observer {
        let merger_id = input_count;
        let sink_id = input_count + 1;
        let mut nodes: Vec<TapNode> = source_names
            .iter()
            .map(|n| (n.clone(), NodeRole::Source, None))
            .collect();
        nodes.push((alloc::string::String::new(), NodeRole::Muxer, None));
        nodes.push((
            alloc::string::String::from(sink_probe.name()),
            NodeRole::Sink,
            Some(sink_probe.clone()),
        ));
        let mut edges: Vec<TapEdge> = Vec::with_capacity(input_count + 1);
        let mut taps = taps.into_iter();
        for (i, caps) in fixated_caps.iter().enumerate() {
            edges.push((i, merger_id, caps.clone(), taps.next().expect("input tap")));
        }
        edges.push((
            merger_id,
            sink_id,
            merged_caps.clone(),
            taps.next().expect("output tap"),
        ));
        register_runner_tap(obs, nodes, edges);
    }

    let mut source_arms: Vec<BoxFuture<'_, Result<u64, G2gError>>> =
        Vec::with_capacity(input_count);
    for (source, in_tx) in sources.into_iter().zip(input_senders) {
        source_arms.push(Box::pin(async move {
            let mut adapter = SenderSink::new(in_tx);
            source.run(&mut adapter).await
        }));
    }

    let mut forwarder_arms: Vec<BoxFuture<'_, Result<u64, G2gError>>> =
        Vec::with_capacity(input_count);
    for (idx, in_rx) in input_receivers.into_iter().enumerate() {
        let handle = merger.handle();
        let out_tx_i = out_tx.clone();
        let live = live_inputs.clone();
        forwarder_arms.push(Box::pin(async move {
            let mut out = SenderSink::new(out_tx_i);
            loop {
                match in_rx.recv().await {
                    Some(PipelinePacket::Eos) | None => {
                        // Last input to finish emits the single merged EOS.
                        if live.fetch_sub(1, Ordering::SeqCst) == 1 {
                            out.push(PipelinePacket::Eos).await?;
                        }
                        return Ok::<u64, G2gError>(0);
                    }
                    Some(packet) => {
                        if handle.selected() == idx {
                            out.push(packet).await?;
                        }
                        // Non-selected input: drain and discard so its
                        // source never stalls on a full link.
                    }
                }
            }
        }));
    }
    // Drop the runner's own sender clone so only the forwarders keep the
    // output link open.
    drop(out_tx);

    let probe_for_sink = sink_probe.clone();
    let sink_arm: BoxFuture<'_, Result<u64, G2gError>> = Box::pin(async move {
        let mut null = NullSink;
        let mut consumed: u64 = 0;
        loop {
            match out_rx.recv().await {
                Some(PipelinePacket::Eos) => {
                    sink.process(PipelinePacket::Eos, &mut null).await?;
                    return Ok::<u64, G2gError>(consumed);
                }
                Some(PipelinePacket::CapsChanged(new_caps)) => {
                    match log_caps_rejected(
                        Some(probe_for_sink.name()),
                        &new_caps,
                        sink.configure_pipeline(&new_caps),
                    )? {
                        ConfigureOutcome::Accepted => {
                            sink.process(PipelinePacket::CapsChanged(new_caps), &mut null)
                                .await?;
                        }
                        ConfigureOutcome::ReFixate(counter) => {
                            out_rx.request_reconfigure(Reconfigure::Propose(counter));
                        }
                    }
                }
                Some(packet) => {
                    let is_data = matches!(packet, PipelinePacket::DataFrame(_));
                    if is_data {
                        consumed += 1;
                        probe_for_sink.record_fill(out_rx.fill_percent());
                    }
                    let t0 = is_data.then(ElementProbe::mark).flatten();
                    sink.process(packet, &mut null).await?;
                    if is_data {
                        probe_for_sink.record_proc_since(t0);
                    }
                }
                None => return Ok(consumed),
            }
        }
    });

    // Arm order: [source0..N, forwarder0..N, sink].
    let mut arms = Vec::with_capacity(2 * input_count + 1);
    arms.extend(source_arms);
    arms.extend(forwarder_arms);
    arms.push(sink_arm);

    let results = join_all(arms).await;
    let mut counts = Vec::with_capacity(results.len());
    for r in results {
        counts.push(r?);
    }
    let emitted: u64 = counts[0..input_count].iter().copied().sum();
    let consumed = counts[2 * input_count];
    // Fan-in latency / allocation / clock election across N inputs is deferred
    // (M12 covers the linear path); report neutral values rather than a
    // misleading partial one.
    Ok(RunStats {
        frames_emitted: emitted,
        frames_consumed: consumed,
        frames_dropped: 0,
        latency: LatencyReport::ZERO,
        allocation: None,
        clock_priority: ClockPriority::SystemFallback,
        base_time_ns: 0,
        coordinator_events: 0,
        per_element: alloc::vec![sink_probe.snapshot()],
    })
}

/// `OutputSink` that tags each pushed packet with its source's input index and
/// forwards it into the shared session channel. Reverse signals are not routed
/// per-input yet (a follow-up), so push always reports `Accepted`.
struct TaggingSink {
    idx: usize,
    tx: Sender<(usize, PipelinePacket)>,
    /// The session's reverse-signal handle for this input, if any. After queuing
    /// each packet we surface any pending PLI / BWE for this track to the source
    /// as the push outcome, so a reverse signal reaches the matching upstream.
    reverse: Option<crate::fanout::ReverseChannel>,
    /// Live traffic counters for this input's edge (M846). The tagged channel is
    /// shared by every input, so the per-edge count is kept here rather than on
    /// the channel. `None` unless an observer is attached.
    counters: Option<Arc<EdgeCounters>>,
    /// Content-inspection slot for this input's edge (M849), the analog of the
    /// one [`SenderSink`] shares with its link: the tagged channel is shared, so
    /// each input carries its own slot here. `None` unless an observer is
    /// attached, and empty (pass-through) until a tool installs an interceptor.
    probe: Option<ProbeSlot>,
}

impl OutputSink for TaggingSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            // A probe may drop the packet before it enters the channel, as on a
            // `SenderSink` link.
            if let Some(p) = &self.probe {
                if p.action(&packet) == ProbeAction::Drop {
                    return Ok(PushOutcome::Accepted);
                }
            }
            let bytes = packet_bytes(&packet);
            match self.tx.send((self.idx, packet)).await {
                Ok(()) => {
                    if let Some(c) = &self.counters {
                        c.record_packet(bytes, 0);
                    }
                    Ok(self
                        .reverse
                        .as_ref()
                        .and_then(|rc| rc.take())
                        .unwrap_or(PushOutcome::Accepted))
                }
                Err(_) => Err(G2gError::Shutdown),
            }
        })
    }
}

/// Drives `N sources → terminal multi-input element` with **no downstream sink**
/// (the element is the destination, e.g. a WebRTC session that publishes its
/// inputs over one PeerConnection). The fan-in analog of a terminal sink: unlike
/// [`run_muxer_sink`], the [`MultiInputElement`] here produces no merged output,
/// so there is no trailing sink to wire.
///
/// Each source is narrowed against the session input pad it feeds (like
/// [`run_fanin_sink`], which narrows every branch against the merged sink), and
/// the resulting caps configure both the source and that pad. Every source pushes
/// into one shared `(input, packet)` channel; a single session task drains it and
/// calls `session.process(input, ..)` serially, so the session keeps `&mut` state
/// without aliasing. A per-input `Eos` is delivered to the session (so it can
/// flush that track); the run ends once every input has ended.
///
/// `output_caps()` is not consulted (there is no output), and reverse signals
/// (keyframe-request / bitrate / QoS) are not yet routed back per-input through
/// this runner, a documented follow-up.
pub async fn run_fanin_session<Sess, Clk>(
    sources: Vec<&mut dyn DynSourceLoop>,
    session: &mut Sess,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
) -> Result<RunStats, G2gError>
where
    Sess: MultiInputElement,
    Clk: PipelineClock,
{
    run_fanin_session_inner(sources, session, clock, link_capacity, None).await
}

/// As [`run_fanin_session`], but taps live telemetry into `observer` (M846): the
/// topology is the N sources feeding the session, whose measured `process()`
/// latency and per-input packet / byte counts a concurrent task reads mid-run via
/// [`Observer::snapshot`].
pub async fn run_fanin_session_observed<Sess, Clk>(
    sources: Vec<&mut dyn DynSourceLoop>,
    session: &mut Sess,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    observer: &Observer,
) -> Result<RunStats, G2gError>
where
    Sess: MultiInputElement,
    Clk: PipelineClock,
{
    run_fanin_session_inner(sources, session, clock, link_capacity, Some(observer)).await
}

async fn run_fanin_session_inner<Sess, Clk>(
    sources: Vec<&mut dyn DynSourceLoop>,
    session: &mut Sess,
    _clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    observer: Option<&Observer>,
) -> Result<RunStats, G2gError>
where
    Sess: MultiInputElement,
    Clk: PipelineClock,
{
    let link_capacity: usize = link_capacity.into().get();
    let input_count = sources.len();
    assert!(input_count > 0, "fan-in session needs at least one source");
    assert!(
        session.input_count() == input_count,
        "session input count must match the number of sources"
    );

    // M846: instance naming, as in `run_fanin_sink`. The session is the only node
    // with a `process()`, so it carries the run's measured-latency probe.
    let mut sources = sources;
    let mut namer = crate::log::InstanceNamer::new();
    let mut source_names: Vec<alloc::string::String> = Vec::with_capacity(input_count);
    for source in sources.iter_mut() {
        let name = namer.add(source.log_category(), None);
        source.set_instance_name(name.clone());
        source_names.push(name);
    }
    let session_probe = ElementProbe::new(namer.add(crate::log::short_type_name::<Sess>(), None));

    // Phase 1 + 2 per input: each source is narrowed against the session input
    // pad it feeds; the fixated caps configure both the source and that pad (the
    // session decides the track kind, e.g. H.264 video vs Opus audio, from them).
    let mut fixated_caps: Vec<Caps> = Vec::with_capacity(input_count);
    for (i, source) in sources.iter_mut().enumerate() {
        let produced = source.produced_caps().await?;
        let fixated = {
            let pad_constraint = MultiInputElement::caps_constraint_as_input(session, i);
            select_branch_caps(&produced, &pad_constraint)?
        };
        source.configure_pipeline(&fixated)?.reject_refixate()?;
        MultiInputElement::configure_pipeline(session, i, &fixated)?.reject_refixate()?;
        fixated_caps.push(fixated);
    }

    // One shared tagged channel: every source pushes `(its index, packet)`.
    let (tx, rx) = bounded::<(usize, PipelinePacket)>(link_capacity);
    let live_inputs = Arc::new(AtomicUsize::new(input_count));
    // Per-input edge counters and content-inspection slots, kept on each
    // `TaggingSink` (the tagged channel is shared, so it cannot carry per-edge
    // state itself).
    let counters: Vec<Option<Arc<EdgeCounters>>> = (0..input_count)
        .map(|_| observer.map(|_| Arc::new(EdgeCounters::default())))
        .collect();
    let probes: Vec<Option<ProbeSlot>> = (0..input_count)
        .map(|_| observer.map(|_| ProbeSlot::default()))
        .collect();

    if let Some(obs) = observer {
        let session_id = input_count;
        let mut nodes: Vec<TapNode> = source_names
            .iter()
            .map(|n| (n.clone(), NodeRole::Source, None))
            .collect();
        nodes.push((
            alloc::string::String::from(session_probe.name()),
            NodeRole::Muxer,
            Some(session_probe.clone()),
        ));
        let edges: Vec<TapEdge> = fixated_caps
            .iter()
            .zip(counters.iter())
            .zip(probes.iter())
            .enumerate()
            .map(|(i, ((caps, c), p))| {
                (
                    i,
                    session_id,
                    caps.clone(),
                    EdgeTap {
                        probe: p.clone().unwrap_or_default(),
                        counters: c.clone(),
                    },
                )
            })
            .collect();
        register_runner_tap(obs, nodes, edges);
    }

    // Per-input reverse-signal handles (WebRTC PLI / BWE), cloned before the
    // session moves into its arm so a signal for track i reaches source i.
    let reverse: Vec<Option<crate::fanout::ReverseChannel>> = (0..input_count)
        .map(|i| session.reverse_channel(i))
        .collect();

    let mut source_arms: Vec<BoxFuture<'_, Result<u64, G2gError>>> =
        Vec::with_capacity(input_count);
    for (i, source) in sources.into_iter().enumerate() {
        let tx_i = tx.clone();
        let reverse_i = reverse[i].clone();
        let counters_i = counters[i].clone();
        let probe_i = probes[i].clone();
        source_arms.push(Box::pin(async move {
            let mut adapter = TaggingSink {
                idx: i,
                tx: tx_i,
                reverse: reverse_i,
                counters: counters_i,
                probe: probe_i,
            };
            source.run(&mut adapter).await
        }));
    }
    // Drop the runner's own sender so the channel closes once all sources end.
    drop(tx);

    let probe_for_session = session_probe.clone();
    let session_arm: BoxFuture<'_, Result<u64, G2gError>> = Box::pin(async move {
        let mut null = NullSink;
        let mut consumed: u64 = 0;
        loop {
            match rx.recv().await {
                Some((idx, PipelinePacket::Eos)) => {
                    // Per-input end: let the session flush that track, then finish
                    // once every input has ended (the session owns its own EOS to
                    // the network).
                    session.process(idx, PipelinePacket::Eos, &mut null).await?;
                    if live_inputs.fetch_sub(1, Ordering::SeqCst) == 1 {
                        return Ok::<u64, G2gError>(consumed);
                    }
                }
                Some((idx, PipelinePacket::CapsChanged(new_caps))) => {
                    // Mid-stream re-solve (M724): re-configure the changed pad
                    // before the session sees the new caps. A counter-fixation
                    // cannot travel back to the tagged source, so it fails loud.
                    MultiInputElement::configure_pipeline(session, idx, &new_caps)?
                        .reject_refixate()?;
                    session
                        .process(idx, PipelinePacket::CapsChanged(new_caps), &mut null)
                        .await?;
                }
                Some((idx, packet)) => {
                    let is_data = matches!(packet, PipelinePacket::DataFrame(_));
                    if is_data {
                        consumed += 1;
                        probe_for_session.record_fill(rx.fill_percent());
                    }
                    let t0 = is_data.then(ElementProbe::mark).flatten();
                    session.process(idx, packet, &mut null).await?;
                    if is_data {
                        probe_for_session.record_proc_since(t0);
                    }
                }
                None => return Ok(consumed),
            }
        }
    });

    let mut arms = Vec::with_capacity(input_count + 1);
    arms.extend(source_arms);
    arms.push(session_arm);

    let results = join_all(arms).await;
    let mut counts = Vec::with_capacity(results.len());
    for r in results {
        counts.push(r?);
    }
    let emitted: u64 = counts[0..input_count].iter().copied().sum();
    let consumed = counts[input_count];
    Ok(RunStats {
        frames_emitted: emitted,
        frames_consumed: consumed,
        frames_dropped: 0,
        latency: LatencyReport::ZERO,
        allocation: None,
        clock_priority: ClockPriority::SystemFallback,
        base_time_ns: 0,
        coordinator_events: 0,
        per_element: alloc::vec![session_probe.snapshot()],
    })
}

/// [`DuplexInbound`] backed by the runner's shared tagged inbound channel, so a
/// [`MultiDuplexSession`] drains its send-side sources through the same erased
/// interface regardless of how the runner wired them.
struct InboundReceiver {
    rx: Receiver<(usize, PipelinePacket)>,
}

impl DuplexInbound for InboundReceiver {
    fn recv(&mut self) -> BoxFuture<'_, Option<(usize, PipelinePacket)>> {
        Box::pin(async move { self.rx.recv().await })
    }
}

/// Drives a terminal **duplex** session ([`MultiDuplexSession`]): N send-side
/// sources **and** M recv-side sinks over one connection, the union of
/// [`run_fanin_session`] (send) and
/// [`run_fanout_session`](crate::runtime::run_fanout_session) (recv). A
/// `WebRtcBin`-style sendrecv PeerConnection both publishes local tracks and
/// emits the peer's tracks; this is the runner shape that expresses an element
/// that is at once a sink (for its inputs) and a source (for its outputs), which
/// neither the fan-in nor fan-out session runner could.
///
/// Negotiation mirrors both halves: each source is narrowed against the session
/// input pad it feeds and configures it (send side, like [`run_fanin_session`]); each
/// recv-side output's caps configure the matching sink (like
/// [`run_fanout_session`](crate::runtime::run_fanout_session)). At runtime the
/// sources push `(input, packet)` into one shared tagged channel; the single
/// session arm owns `&mut session` and calls `session.run(inbound, out)`, so the
/// send and recv halves share state with no aliasing (no detached task needed);
/// the M sink arms drain the per-output branch links. The run ends when the
/// session's `run` returns (e.g. on peer disconnect), which closes the branch
/// links and lets the sinks finish.
///
/// Per-input reverse signals (a remote PLI / BWE arriving on a track's m-line)
/// are routed back to the matching send source via
/// [`MultiDuplexSession::reverse_channel`], exactly as the fan-in session runner
/// does. Per-branch mid-stream re-solve on the recv side is still a follow-up.
pub async fn run_duplex_session<Sess, Clk>(
    sources: Vec<&mut dyn DynSourceLoop>,
    session: &mut Sess,
    sinks: Vec<&mut dyn DynAsyncElement>,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
) -> Result<RunStats, G2gError>
where
    Sess: MultiDuplexSession,
    Clk: PipelineClock,
{
    run_duplex_session_inner(sources, session, sinks, clock, link_capacity, None).await
}

/// As [`run_duplex_session`], but taps live telemetry into `observer` (M846):
/// the send sources, the session, and the recv sinks, with each sink's measured
/// `process()` latency and every link's packet / byte counts readable mid-run via
/// [`Observer::snapshot`].
pub async fn run_duplex_session_observed<Sess, Clk>(
    sources: Vec<&mut dyn DynSourceLoop>,
    session: &mut Sess,
    sinks: Vec<&mut dyn DynAsyncElement>,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    observer: &Observer,
) -> Result<RunStats, G2gError>
where
    Sess: MultiDuplexSession,
    Clk: PipelineClock,
{
    run_duplex_session_inner(
        sources,
        session,
        sinks,
        clock,
        link_capacity,
        Some(observer),
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn run_duplex_session_inner<Sess, Clk>(
    sources: Vec<&mut dyn DynSourceLoop>,
    session: &mut Sess,
    sinks: Vec<&mut dyn DynAsyncElement>,
    _clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    observer: Option<&Observer>,
) -> Result<RunStats, G2gError>
where
    Sess: MultiDuplexSession,
    Clk: PipelineClock,
{
    let link_capacity: usize = link_capacity.into().get();
    let input_count = sources.len();
    let output_count = sinks.len();
    assert!(
        input_count > 0,
        "duplex session needs at least one send source"
    );
    assert!(
        output_count > 0,
        "duplex session needs at least one recv sink"
    );
    assert!(
        session.input_count() == input_count,
        "session input count must match the number of send sources"
    );
    assert!(
        session.output_count() == output_count,
        "session output count must match the number of recv sinks"
    );

    // M846: instance naming + one measured-latency probe per recv sink (the
    // session drives itself through `run`, so it has no `process()` to time).
    let mut sources = sources;
    let mut sinks = sinks;
    let mut namer = crate::log::InstanceNamer::new();
    let mut source_names: Vec<alloc::string::String> = Vec::with_capacity(input_count);
    for source in sources.iter_mut() {
        let name = namer.add(source.log_category(), None);
        source.set_instance_name(name.clone());
        source_names.push(name);
    }
    let session_name = namer.add(crate::log::short_type_name::<Sess>(), None);
    let mut sink_probes = Vec::with_capacity(output_count);
    for sink in sinks.iter_mut() {
        let name = namer.add(sink.log_category(), None);
        sink.set_instance_name(name.clone());
        sink_probes.push(ElementProbe::new(name));
    }

    // Negotiate the send inputs (like run_fanin_session): each source is narrowed
    // against the send input pad it feeds, and the fixated caps configure both
    // the source and that pad.
    let mut input_caps: Vec<Caps> = Vec::with_capacity(input_count);
    for (i, source) in sources.iter_mut().enumerate() {
        let produced = source.produced_caps().await?;
        let fixated = {
            let pad_constraint = MultiDuplexSession::caps_constraint_as_input(session, i);
            select_branch_caps(&produced, &pad_constraint)?
        };
        source.configure_pipeline(&fixated)?.reject_refixate()?;
        session.configure_input(i, &fixated)?.reject_refixate()?;
        input_caps.push(fixated);
    }
    // Negotiate the recv outputs (like run_fanout_session): the session self-
    // fixates each output's caps and configures the matching sink.
    let mut output_caps: Vec<Caps> = Vec::with_capacity(output_count);
    for (o, sink) in sinks.iter_mut().enumerate() {
        let fixated = session.output_caps(o)?.fixate()?;
        sink.configure_pipeline(&fixated)?.reject_refixate()?;
        output_caps.push(fixated);
    }

    // Inbound: one shared tagged channel; every send source pushes (its index, packet).
    let (in_tx, in_rx) = bounded::<(usize, PipelinePacket)>(link_capacity);
    // Per-input edge counters and content-inspection slots ride the
    // `TaggingSink`s, the shared channel carries no per-edge state itself.
    let in_counters: Vec<Option<Arc<EdgeCounters>>> = (0..input_count)
        .map(|_| observer.map(|_| Arc::new(EdgeCounters::default())))
        .collect();
    let in_probes: Vec<Option<ProbeSlot>> = (0..input_count)
        .map(|_| observer.map(|_| ProbeSlot::default()))
        .collect();
    // Outbound: one branch link per recv output.
    let tap = observer.is_some();
    let mut branch_senders = Vec::with_capacity(output_count);
    let mut branch_receivers = Vec::with_capacity(output_count);
    let mut branch_taps = Vec::with_capacity(output_count);
    for _ in 0..output_count {
        let (tx, rx, edge) = link_tapped(link_capacity, tap);
        branch_senders.push(SenderSink::new(tx));
        branch_receivers.push(rx);
        branch_taps.push(edge);
    }

    if let Some(obs) = observer {
        let session_id = input_count;
        let mut nodes: Vec<TapNode> = source_names
            .iter()
            .map(|n| (n.clone(), NodeRole::Source, None))
            .collect();
        nodes.push((session_name, NodeRole::Muxer, None));
        for probe in &sink_probes {
            nodes.push((
                alloc::string::String::from(probe.name()),
                NodeRole::Sink,
                Some(probe.clone()),
            ));
        }
        let mut edges: Vec<TapEdge> = Vec::with_capacity(input_count + output_count);
        for (i, ((caps, c), p)) in input_caps
            .iter()
            .zip(in_counters.iter())
            .zip(in_probes.iter())
            .enumerate()
        {
            edges.push((
                i,
                session_id,
                caps.clone(),
                EdgeTap {
                    probe: p.clone().unwrap_or_default(),
                    counters: c.clone(),
                },
            ));
        }
        for (o, (caps, edge)) in output_caps
            .iter()
            .zip(core::mem::take(&mut branch_taps))
            .enumerate()
        {
            edges.push((session_id, session_id + 1 + o, caps.clone(), edge));
        }
        register_runner_tap(obs, nodes, edges);
    }

    // Per-input reverse-signal handles (WebRTC PLI / BWE), cloned before the
    // session moves into its arm so a signal for track i (a remote PLI on that
    // m-line, or a BWE estimate) reaches send source i, the duplex analog of the
    // fan-in session's per-input routing.
    let reverse: Vec<Option<crate::fanout::ReverseChannel>> = (0..input_count)
        .map(|i| session.reverse_channel(i))
        .collect();

    let mut source_arms: Vec<BoxFuture<'_, Result<u64, G2gError>>> =
        Vec::with_capacity(input_count);
    for (i, source) in sources.into_iter().enumerate() {
        let tx_i = in_tx.clone();
        let reverse_i = reverse[i].clone();
        let counters_i = in_counters[i].clone();
        let probe_i = in_probes[i].clone();
        source_arms.push(Box::pin(async move {
            let mut adapter = TaggingSink {
                idx: i,
                tx: tx_i,
                reverse: reverse_i,
                counters: counters_i,
                probe: probe_i,
            };
            source.run(&mut adapter).await
        }));
    }
    // Drop the runner's own sender so the inbound channel closes once all sources
    // end (the session sees `recv() == None` and can stop publishing).
    drop(in_tx);

    let session_arm: BoxFuture<'_, Result<u64, G2gError>> = Box::pin(async move {
        let mut inbound = InboundReceiver { rx: in_rx };
        let mut multi = MultiSenderSink::new(branch_senders);
        session.run(&mut inbound, &mut multi).await
    });

    let mut sink_arms: Vec<BoxFuture<'_, Result<u64, G2gError>>> = Vec::with_capacity(output_count);
    for ((sink, rx), probe) in sinks
        .into_iter()
        .zip(branch_receivers)
        .zip(sink_probes.iter().cloned())
    {
        sink_arms.push(Box::pin(async move {
            let mut null = NullSink;
            let mut consumed: u64 = 0;
            loop {
                match rx.recv().await {
                    Some(PipelinePacket::Eos) => {
                        sink.process(PipelinePacket::Eos, &mut null).await?;
                        return Ok::<u64, G2gError>(consumed);
                    }
                    Some(PipelinePacket::CapsChanged(new_caps)) => {
                        match log_caps_rejected(
                            Some(probe.name()),
                            &new_caps,
                            sink.configure_pipeline(&new_caps),
                        )? {
                            ConfigureOutcome::Accepted => {
                                sink.process(PipelinePacket::CapsChanged(new_caps), &mut null)
                                    .await?;
                            }
                            ConfigureOutcome::ReFixate(counter) => {
                                rx.request_reconfigure(Reconfigure::Propose(counter));
                            }
                        }
                    }
                    Some(packet) => {
                        let is_data = matches!(packet, PipelinePacket::DataFrame(_));
                        if is_data {
                            consumed += 1;
                            probe.record_fill(rx.fill_percent());
                        }
                        let t0 = is_data.then(ElementProbe::mark).flatten();
                        sink.process(packet, &mut null).await?;
                        if is_data {
                            probe.record_proc_since(t0);
                        }
                    }
                    None => return Ok(consumed),
                }
            }
        }));
    }

    // Arm order: [source0..N, session, sink0..M].
    let mut arms = Vec::with_capacity(input_count + 1 + output_count);
    arms.extend(source_arms);
    arms.push(session_arm);
    arms.extend(sink_arms);

    let results = join_all(arms).await;
    let mut counts = Vec::with_capacity(results.len());
    for r in results {
        counts.push(r?);
    }
    let emitted: u64 = counts[0..input_count].iter().copied().sum();
    let consumed: u64 = counts[input_count + 1..].iter().copied().sum();
    Ok(RunStats {
        frames_emitted: emitted,
        frames_consumed: consumed,
        frames_dropped: 0,
        latency: LatencyReport::ZERO,
        allocation: None,
        clock_priority: ClockPriority::SystemFallback,
        base_time_ns: 0,
        coordinator_events: 0,
        per_element: crate::runtime::snapshot_all(
            &sink_probes.into_iter().map(Some).collect::<Vec<_>>(),
        ),
    })
}

/// Drives `N sources → muxer → 1 sink` (M10 true fan-in). Unlike
/// [`run_fanin_sink`], a [`MultiInputElement`] muxer combines **all** inputs
/// into the output. Each input's packets are tagged with its index and merged
/// into one channel; a single muxer task drains it and calls
/// `mux.process(input, ..)` serially (so the muxer keeps `&mut` state without
/// aliasing). The output emits one `Eos` after every input has ended.
///
/// Negotiation is per-input: each source ↔ its muxer pad fixate independently;
/// the sink is configured against `mux.output_caps()`. A `ReFixate` anywhere
/// fails with `FixationFailed`.
///
/// When `clock` can sleep on a deadline ([`PipelineClock::as_ticker`], which every
/// [`AsyncClock`](crate::AsyncClock) answers), the muxer arm also gets a **deadline tick** (M880): a
/// muxer declaring a [`MultiInputElement::tick_interval_ns`] receives
/// [`PipelinePacket::Tick`] on that period even while its inputs are silent, so a
/// compositor can keep emitting at its output rate when a pad stalls
/// (zero-order-hold on that pad's last frame) instead of freezing with it.
pub async fn run_muxer_sink<Mux, Snk, Clk>(
    sources: Vec<&mut dyn DynSourceLoop>,
    mux: &mut Mux,
    sink: &mut Snk,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
) -> Result<RunStats, G2gError>
where
    Mux: MultiInputElement,
    Snk: AsyncElement,
    Clk: PipelineClock,
{
    run_muxer_sink_inner(sources, mux, sink, clock, link_capacity, None, None).await
}

/// As [`run_muxer_sink`], but posts a structured
/// [`BusMessage::NegotiationFailed`](crate::BusMessage::NegotiationFailed) to
/// `bus` on a startup or per-input mid-stream negotiation failure (item 7).
pub async fn run_muxer_sink_with_bus<Mux, Snk, Clk>(
    sources: Vec<&mut dyn DynSourceLoop>,
    mux: &mut Mux,
    sink: &mut Snk,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    bus: &BusHandle,
) -> Result<RunStats, G2gError>
where
    Mux: MultiInputElement,
    Snk: AsyncElement,
    Clk: PipelineClock,
{
    run_muxer_sink_inner(sources, mux, sink, clock, link_capacity, Some(bus), None).await
}

#[allow(clippy::too_many_arguments)]
async fn run_muxer_sink_inner<Mux, Snk, Clk>(
    sources: Vec<&mut dyn DynSourceLoop>,
    mux: &mut Mux,
    sink: &mut Snk,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    bus: Option<&BusHandle>,
    ticker: Option<&dyn DynAsyncClock>,
) -> Result<RunStats, G2gError>
where
    Mux: MultiInputElement,
    Snk: AsyncElement,
    Clk: PipelineClock,
{
    let n = sources.len();
    assert!(n > 0, "muxer needs at least one source");
    assert!(
        mux.input_count() == n,
        "muxer input count must match the number of sources"
    );

    // D5: thin builder over the DAG runner. The muxer maps onto the graph's
    // fan-in node; `run_graph` owns negotiation, the per-input forwarders, the
    // single merged Eos, and the MX-1 / MX-2 mid-stream re-solve.
    let mut g: Graph<GraphNodeRef<'_>> = Graph::new();
    let mux_node = g.add_muxer(GraphNodeRef::muxer_ref(mux), n as u8);
    let snk = g.add_sink(GraphNodeRef::element_ref(sink));
    for (i, source) in sources.into_iter().enumerate() {
        let s = g.add_source(GraphNodeRef::source_ref(source));
        g.link(s, mux_node.input(i as u8))
            .map_err(|_| G2gError::CapsMismatch)?;
    }
    g.link(mux_node.output(), snk)
        .map_err(|_| G2gError::CapsMismatch)?;

    run_graph_inner(g, clock, link_capacity, bus, None, None, None, None, ticker).await
}

/// Which arm of a dynamic fan-in ([`run_aggregator_dynamic`]) produced this
/// result. The arm set grows at runtime, so indices are not stable; identity is
/// carried in the variant instead (the [`DynamicJoin`](crate::runtime) contract).
#[derive(Debug, Clone, Copy)]
enum FaninArmOut {
    /// The aggregator arm consumed this many `DataFrame`s.
    Aggregator(u64),
    /// A runtime-attached input source emitted this many `DataFrame`s.
    Source(u64),
    /// The trailing sink arm ([`run_muxer_sink_dynamic`]) consumed this many
    /// merged `DataFrame`s.
    Sink(u64),
}

/// Node id of a dynamic fan-in's aggregator / muxer in its telemetry topology.
/// Runtime-attached inputs append after the fixed stages.
#[cfg(feature = "std")]
const DYN_FANIN_NODE: usize = 0;

/// Node id of the trailing sink in [`run_muxer_sink_dynamic`]'s topology.
#[cfg(feature = "std")]
const DYN_FANIN_SINK_NODE: usize = 1;

/// Telemetry bookkeeping the aggregator / muxer arm of a dynamic fan-in carries,
/// so an input attached mid-run gets named and put in the observer's topology
/// before its first frame (M869). The runtime inputs are sources, so they carry
/// no `process()` probe; their telemetry is the per-input edge counters the
/// [`TaggingSink`] advances.
#[cfg(feature = "std")]
#[derive(Debug)]
struct FaninTap {
    obs: Option<Observer>,
    namer: crate::log::InstanceNamer,
}

/// A handle to add inputs to a *running* dynamic aggregator (M320): the fan-in
/// dual of [`DynamicFanoutHandle`](crate::runtime::DynamicFanoutHandle), the
/// runtime equivalent of GStreamer's aggregator/muxer request **sink** pads. Each
/// [`add_input`](Self::add_input) attaches a new source feeding the next free
/// input pad of the aggregator; the source is fixated and its pad configured on
/// attach, then its frames are tagged with the pad index and aggregated. Cheap to
/// clone (a channel sender plus an atomic), so several controllers can request
/// pads.
///
/// `'a` is the run's lifetime: the handle is used concurrently with the run
/// future and must be dropped no later than it. The aggregator declares a fixed
/// pad capacity ([`MultiInputElement::input_count`]); [`add_input`](Self::add_input)
/// past that capacity, or after the run has finished, is rejected with
/// [`G2gError::Shutdown`].
#[derive(Clone)]
#[allow(missing_debug_implementations)]
pub struct DynamicFaninHandle<'a> {
    new_input_tx: Sender<(usize, Box<dyn DynSourceLoop + 'a>)>,
    /// Next free input pad index, reserved atomically so concurrent callers get
    /// distinct pads. The aggregator's `process(pad, ..)` indexes a fixed pad set,
    /// so a pad is only handed out while `< max_inputs`.
    next_pad: Arc<AtomicUsize>,
    max_inputs: usize,
}

#[cfg(feature = "std")]
impl<'a> DynamicFaninHandle<'a> {
    /// Request a new sink pad: attach `source` as a new input of the running
    /// aggregator. Reserves the next pad index atomically and hands the source to
    /// the aggregator arm, which fixates it and configures the pad before its
    /// first frame. Returns [`G2gError::Shutdown`] if every declared pad is
    /// already in use or the aggregator has already finished, and
    /// [`G2gError::PoolExhausted`] if the add channel is transiently full (the
    /// aggregator has not drained pending adds yet); retry the latter.
    pub fn add_input(&self, source: Box<dyn DynSourceLoop + 'a>) -> Result<(), G2gError> {
        // Reserve a pad. fetch_add can overshoot past capacity under contention,
        // but that only makes later calls also see `>= max_inputs` and fail, which
        // is the intended "no free pad" outcome.
        let pad = self.next_pad.fetch_add(1, Ordering::SeqCst);
        if pad >= self.max_inputs {
            return Err(G2gError::Shutdown);
        }
        match self.new_input_tx.try_send((pad, source)) {
            Ok(()) => Ok(()),
            Err((_, SendError::Closed)) => Err(G2gError::Shutdown),
            Err((_, SendError::Full)) => {
                // Transient backpressure, not a teardown. Roll back the pad we
                // reserved (only when no later add claimed one) so a retry does
                // not permanently shrink the usable pad count.
                let _ = self.next_pad.compare_exchange(
                    pad + 1,
                    pad,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                );
                Err(G2gError::PoolExhausted)
            }
        }
    }
}

/// Drives `N sources -> dynamic aggregator` (M320 fan-in request pads), where
/// inputs can be added at runtime through the returned [`DynamicFaninHandle`].
/// The fan-in dual of
/// [`run_source_tee_dynamic`](crate::runtime::run_source_tee_dynamic): there
/// branches attach to a running source, here sources attach to a running
/// aggregator.
///
/// The aggregator is **terminal** (it consumes its inputs and produces no merged
/// downstream output, like [`run_fanin_session`]): a multi-stream batching sink,
/// a compositor-to-display, a WebRTC publisher. A source attaches via
/// [`DynamicFaninHandle::add_input`]; on attach the source is narrowed against
/// the aggregator input pad it feeds and the resulting caps configure both, so a
/// late input is negotiated without a global re-solve. Each input's packets
/// are tagged with its pad index and drained by a single aggregator arm that owns
/// `&mut aggregator`, so the aggregator keeps its state without aliasing. A
/// per-input `Eos` is delivered to the aggregator (so it can flush that pad's
/// state); the run completes once the handle is dropped **and** every attached
/// input has ended.
///
/// Returns the handle plus the run future; drive them concurrently. A merged
/// downstream output (the [`run_muxer_sink`] shape, with a trailing sink and
/// output-caps coupling) is a follow-up.
#[cfg(feature = "std")]
pub fn run_aggregator_dynamic<'a, Agg>(
    aggregator: &'a mut Agg,
    link_capacity: impl Into<LinkCapacity>,
) -> (
    DynamicFaninHandle<'a>,
    impl Future<Output = Result<RunStats, G2gError>> + 'a,
)
where
    Agg: MultiInputElement + 'a,
{
    run_aggregator_dynamic_inner(aggregator, link_capacity, None)
}

/// As [`run_aggregator_dynamic`], but taps live telemetry into `observer` (M869).
/// The topology starts as the aggregator alone; each input attached through the
/// handle appends its own node and link before its first frame, so a dashboard
/// polling [`Observer::snapshot`] sees runtime inputs appear with their per-input
/// packet / byte counters beside the aggregator's measured `process()` latency.
#[cfg(feature = "std")]
pub fn run_aggregator_dynamic_observed<'a, Agg>(
    aggregator: &'a mut Agg,
    link_capacity: impl Into<LinkCapacity>,
    observer: &Observer,
) -> (
    DynamicFaninHandle<'a>,
    impl Future<Output = Result<RunStats, G2gError>> + 'a,
)
where
    Agg: MultiInputElement + 'a,
{
    run_aggregator_dynamic_inner(aggregator, link_capacity, Some(observer.clone()))
}

#[cfg(feature = "std")]
fn run_aggregator_dynamic_inner<'a, Agg>(
    aggregator: &'a mut Agg,
    link_capacity: impl Into<LinkCapacity>,
    observer: Option<Observer>,
) -> (
    DynamicFaninHandle<'a>,
    impl Future<Output = Result<RunStats, G2gError>> + 'a,
)
where
    Agg: MultiInputElement + 'a,
{
    let link_capacity: usize = link_capacity.into().get();
    let max_inputs = aggregator.input_count();

    // Control channel: handle -> aggregator arm (new (pad, source) inputs).
    let (new_input_tx, new_input_rx) =
        bounded::<(usize, Box<dyn DynSourceLoop + 'a>)>(link_capacity);
    // Arm channel: aggregator arm -> join (the attached source-run futures).
    let (new_arm_tx, new_arm_rx) =
        bounded::<BoxFuture<'a, Result<FaninArmOut, G2gError>>>(link_capacity);

    let handle = DynamicFaninHandle {
        new_input_tx,
        next_pad: Arc::new(AtomicUsize::new(0)),
        max_inputs,
    };

    let run = async move {
        // One shared tagged channel: every attached source pushes `(pad, packet)`.
        let (tagged_tx, tagged_rx) = bounded::<(usize, PipelinePacket)>(link_capacity);

        // M869: instance naming + the aggregator's measured-latency probe, as in
        // `run_fanin_session`. Attached inputs are named off the same namer.
        let mut namer = crate::log::InstanceNamer::new();
        let agg_probe = ElementProbe::new(namer.add(crate::log::short_type_name::<Agg>(), None));
        if let Some(obs) = &observer {
            register_runner_tap(
                obs,
                alloc::vec![(
                    alloc::string::String::from(agg_probe.name()),
                    NodeRole::Muxer,
                    Some(agg_probe.clone()),
                )],
                Vec::new(),
            );
        }
        let mut tap = FaninTap {
            obs: observer,
            namer,
        };
        let probe_for_agg = agg_probe.clone();

        let aggregator_arm: BoxFuture<'a, Result<FaninArmOut, G2gError>> = Box::pin(async move {
            // Coerce once so configure / process go through the boxed-future Dyn
            // surface; `Agg: MultiInputElement` implies `DynMultiInputElement`.
            let aggregator: &mut dyn DynMultiInputElement = aggregator;
            let mut null = NullSink;
            let mut consumed = 0u64;
            let mut accepting = true;
            // Hold one tagged sender open while we still accept inputs, so the
            // tagged channel does not close (and end the run) before any source
            // attaches. Dropped the moment the handle goes away.
            let mut keepalive: Option<Sender<(usize, PipelinePacket)>> = Some(tagged_tx);
            loop {
                // Attach every input queued so far BEFORE draining the next packet,
                // so an input requested before a frame is never missed (select2
                // below is left-biased toward the data channel; mirrors the M310
                // fan-out drain-first gotcha).
                while let Some((pad, source)) = new_input_rx.try_recv() {
                    let tx = keepalive.as_ref().expect("keepalive held while accepting");
                    attach_input(pad, source, aggregator, tx, &new_arm_tx, &mut tap).await?;
                }

                if accepting {
                    match select2(tagged_rx.recv(), new_input_rx.recv()).await {
                        Either::Left(Some((pad, PipelinePacket::Eos))) => {
                            // Per-input end: let the aggregator flush that pad. It
                            // must not forward Eos; the run owns the end.
                            aggregator
                                .process(pad, PipelinePacket::Eos, &mut null)
                                .await?;
                        }
                        Either::Left(Some((pad, packet))) => {
                            let is_data = matches!(packet, PipelinePacket::DataFrame(_));
                            if is_data {
                                consumed += 1;
                                probe_for_agg.record_fill(tagged_rx.fill_percent());
                            }
                            let t0 = is_data.then(ElementProbe::mark).flatten();
                            aggregator.process(pad, packet, &mut null).await?;
                            if is_data {
                                probe_for_agg.record_proc_since(t0);
                            }
                        }
                        // Unreachable while `keepalive` is held (a live sender keeps
                        // the channel open), but folded into the end path for safety.
                        Either::Left(None) => return Ok(FaninArmOut::Aggregator(consumed)),
                        Either::Right(Some((pad, source))) => {
                            let tx = keepalive.as_ref().expect("keepalive held while accepting");
                            attach_input(pad, source, aggregator, tx, &new_arm_tx, &mut tap)
                                .await?;
                        }
                        // Handle dropped: stop accepting and release the keepalive
                        // so the tagged channel can close once every attached input
                        // has ended.
                        Either::Right(None) => {
                            accepting = false;
                            keepalive = None;
                        }
                    }
                } else {
                    match tagged_rx.recv().await {
                        Some((pad, PipelinePacket::Eos)) => {
                            aggregator
                                .process(pad, PipelinePacket::Eos, &mut null)
                                .await?;
                        }
                        Some((pad, packet)) => {
                            let is_data = matches!(packet, PipelinePacket::DataFrame(_));
                            if is_data {
                                consumed += 1;
                                probe_for_agg.record_fill(tagged_rx.fill_percent());
                            }
                            let t0 = is_data.then(ElementProbe::mark).flatten();
                            aggregator.process(pad, packet, &mut null).await?;
                            if is_data {
                                probe_for_agg.record_proc_since(t0);
                            }
                        }
                        None => return Ok(FaninArmOut::Aggregator(consumed)),
                    }
                }
            }
        });

        // The aggregator arm owns `new_arm_tx`; when it returns (run end) the arm
        // channel closes and the dynamic join can finish.
        let arms: Vec<BoxFuture<'a, Result<FaninArmOut, G2gError>>> = alloc::vec![aggregator_arm];
        let results = dynamic_join(arms, new_arm_rx).await;

        let mut emitted = 0u64;
        let mut consumed = 0u64;
        for r in results {
            match r? {
                FaninArmOut::Source(n) => emitted += n,
                FaninArmOut::Aggregator(n) => consumed = n,
                // The terminal aggregator has no trailing sink arm.
                FaninArmOut::Sink(_) => {}
            }
        }
        Ok(RunStats {
            frames_emitted: emitted,
            frames_consumed: consumed,
            frames_dropped: 0,
            latency: LatencyReport::ZERO,
            allocation: None,
            clock_priority: ClockPriority::SystemFallback,
            base_time_ns: 0,
            coordinator_events: 0,
            per_element: alloc::vec![agg_probe.snapshot()],
        })
    };

    (handle, run)
}

/// Like [`run_aggregator_dynamic`], but with a trailing **sink**: the muxer's
/// merged output flows to `sink`, the [`run_muxer_sink`] shape extended to
/// runtime-added inputs (dynamically attach a late audio track to a running
/// `muxer ! filesink`, say). Inputs are added through the returned
/// [`DynamicFaninHandle`] exactly as for the terminal aggregator; the difference
/// is the merged output is not discarded.
///
/// The muxer's output caps are coupled to the sink without a global re-solve:
/// because inputs attach one at a time, the merged output (`output_caps`) only
/// firms up as pads are configured, so the muxer arm emits a `CapsChanged` to the
/// sink whenever the derived output changes (the dynamic analog of the static
/// `run_muxer_sink` MX-2 coupling), and the sink configures against it before the
/// first merged frame. When every input has ended and the handle is dropped, the
/// muxer arm closes the merged link with `Eos`, ending the sink arm.
/// `RunStats::frames_consumed` is the sink's merged-frame count.
#[cfg(feature = "std")]
pub fn run_muxer_sink_dynamic<'a, Mux, Snk>(
    mux: &'a mut Mux,
    sink: &'a mut Snk,
    link_capacity: impl Into<LinkCapacity>,
) -> (
    DynamicFaninHandle<'a>,
    impl Future<Output = Result<RunStats, G2gError>> + 'a,
)
where
    Mux: MultiInputElement + 'a,
    Snk: AsyncElement + 'a,
{
    run_muxer_sink_dynamic_inner(mux, sink, link_capacity, None)
}

/// As [`run_muxer_sink_dynamic`], but taps live telemetry into `observer` (M869).
/// The muxer and the sink are registered up front with their measured-latency
/// probes; each runtime input appends its node and link on attach, and the merged
/// `muxer -> sink` link joins the topology once its caps firm up (they only exist
/// after a pad is configured).
#[cfg(feature = "std")]
pub fn run_muxer_sink_dynamic_observed<'a, Mux, Snk>(
    mux: &'a mut Mux,
    sink: &'a mut Snk,
    link_capacity: impl Into<LinkCapacity>,
    observer: &Observer,
) -> (
    DynamicFaninHandle<'a>,
    impl Future<Output = Result<RunStats, G2gError>> + 'a,
)
where
    Mux: MultiInputElement + 'a,
    Snk: AsyncElement + 'a,
{
    run_muxer_sink_dynamic_inner(mux, sink, link_capacity, Some(observer.clone()))
}

#[cfg(feature = "std")]
fn run_muxer_sink_dynamic_inner<'a, Mux, Snk>(
    mux: &'a mut Mux,
    sink: &'a mut Snk,
    link_capacity: impl Into<LinkCapacity>,
    observer: Option<Observer>,
) -> (
    DynamicFaninHandle<'a>,
    impl Future<Output = Result<RunStats, G2gError>> + 'a,
)
where
    Mux: MultiInputElement + 'a,
    Snk: AsyncElement + 'a,
{
    let link_capacity: usize = link_capacity.into().get();
    let max_inputs = mux.input_count();

    let (new_input_tx, new_input_rx) =
        bounded::<(usize, Box<dyn DynSourceLoop + 'a>)>(link_capacity);
    let (new_arm_tx, new_arm_rx) =
        bounded::<BoxFuture<'a, Result<FaninArmOut, G2gError>>>(link_capacity);

    let handle = DynamicFaninHandle {
        new_input_tx,
        next_pad: Arc::new(AtomicUsize::new(0)),
        max_inputs,
    };

    let run = async move {
        let (tagged_tx, tagged_rx) = bounded::<(usize, PipelinePacket)>(link_capacity);
        // Merged-output link: muxer arm -> sink arm.
        let (out_tx, out_rx, out_tap) = link_tapped(link_capacity, observer.is_some());

        // M869: instance naming + a probe per stage with a `process()`, as in the
        // static `run_muxer_sink`. Attached inputs are named off the same namer.
        let mut namer = crate::log::InstanceNamer::new();
        let mux_probe = ElementProbe::new(namer.add(crate::log::short_type_name::<Mux>(), None));
        let sink_name = namer.add(crate::log::short_type_name::<Snk>(), None);
        AsyncElement::set_instance_name(sink, sink_name.clone());
        let sink_probe = ElementProbe::new(sink_name.clone());
        if let Some(obs) = &observer {
            // The merged link's caps do not exist until a pad is configured, so
            // only the two nodes are registered here; the edge follows.
            register_runner_tap(
                obs,
                alloc::vec![
                    (
                        alloc::string::String::from(mux_probe.name()),
                        NodeRole::Muxer,
                        Some(mux_probe.clone()),
                    ),
                    (sink_name, NodeRole::Sink, Some(sink_probe.clone())),
                ],
                Vec::new(),
            );
        }
        let mut tap = FaninTap {
            obs: observer,
            namer,
        };
        let probe_for_mux = mux_probe.clone();
        let probe_for_sink = sink_probe.clone();

        // Sink arm: configure on the muxer's output `CapsChanged`, then process
        // merged frames until the muxer arm closes the link (Eos / drop).
        let sink_arm: BoxFuture<'a, Result<FaninArmOut, G2gError>> = Box::pin(async move {
            let sink: &mut dyn DynAsyncElement = sink;
            let mut null = NullSink;
            let mut consumed = 0u64;
            while let Some(pkt) = out_rx.recv().await {
                match pkt {
                    PipelinePacket::CapsChanged(caps) => {
                        log_caps_rejected(
                            Some(probe_for_sink.name()),
                            &caps,
                            sink.configure_pipeline(&caps),
                        )?
                        .reject_refixate()?;
                        sink.process(PipelinePacket::CapsChanged(caps), &mut null)
                            .await?;
                    }
                    PipelinePacket::Eos => break,
                    other => {
                        let is_data = matches!(other, PipelinePacket::DataFrame(_));
                        if is_data {
                            consumed += 1;
                            probe_for_sink.record_fill(out_rx.fill_percent());
                        }
                        let t0 = is_data.then(ElementProbe::mark).flatten();
                        sink.process(other, &mut null).await?;
                        if is_data {
                            probe_for_sink.record_proc_since(t0);
                        }
                    }
                }
            }
            Ok(FaninArmOut::Sink(consumed))
        });

        let muxer_arm: BoxFuture<'a, Result<FaninArmOut, G2gError>> = Box::pin(async move {
            let mux: &mut dyn DynMultiInputElement = mux;
            let mut out = SenderSink::new(out_tx);
            // M947: downstream backpressure is push-wait, not muxing work.
            out.set_push_wait_probe(Some(probe_for_mux.clone()));
            let mut merged_tap = Some(out_tap);
            let mut current_output: Option<Caps> = None;
            let mut consumed = 0u64;
            let mut accepting = true;
            let mut keepalive: Option<Sender<(usize, PipelinePacket)>> = Some(tagged_tx);
            loop {
                while let Some((pad, source)) = new_input_rx.try_recv() {
                    let tx = keepalive.as_ref().expect("keepalive held while accepting");
                    attach_input(pad, source, mux, tx, &new_arm_tx, &mut tap).await?;
                }

                let next = if accepting {
                    match select2(tagged_rx.recv(), new_input_rx.recv()).await {
                        Either::Left(packet) => packet,
                        Either::Right(Some((pad, source))) => {
                            let tx = keepalive.as_ref().expect("keepalive held while accepting");
                            attach_input(pad, source, mux, tx, &new_arm_tx, &mut tap).await?;
                            continue;
                        }
                        Either::Right(None) => {
                            // Handle dropped: stop accepting, release the keepalive
                            // so the tagged channel closes once inputs end.
                            accepting = false;
                            keepalive = None;
                            continue;
                        }
                    }
                } else {
                    tagged_rx.recv().await
                };

                match next {
                    Some((pad, PipelinePacket::Eos)) => {
                        // Per-input end: let the muxer flush that pad. It must not
                        // forward Eos; this runner owns the merged end.
                        mux.process(pad, PipelinePacket::Eos, &mut out).await?;
                    }
                    Some((pad, packet)) => {
                        let is_data = matches!(packet, PipelinePacket::DataFrame(_));
                        if is_data {
                            // Couple the merged output to the sink: emit one
                            // `CapsChanged` whenever the derived output firms up or
                            // shifts (a newly attached pad can change it), before
                            // the frame it qualifies.
                            if let Ok(oc) = mux.output_caps() {
                                if current_output.as_ref() != Some(&oc) {
                                    if let Some(obs) = &tap.obs {
                                        // First solved output: the merged edge can
                                        // now be registered with real caps.
                                        if let Some(t) = merged_tap.take() {
                                            obs.add_edge(
                                                DYN_FANIN_NODE,
                                                DYN_FANIN_SINK_NODE,
                                                oc.clone(),
                                                t,
                                            );
                                        }
                                    }
                                    out.push(PipelinePacket::CapsChanged(oc.clone())).await?;
                                    current_output = Some(oc);
                                }
                            }
                            consumed += 1;
                            probe_for_mux.record_fill(tagged_rx.fill_percent());
                        }
                        let t0 = is_data.then(ElementProbe::mark).flatten();
                        mux.process(pad, packet, &mut out).await?;
                        if is_data {
                            probe_for_mux.record_proc_since(t0);
                        }
                    }
                    None => break,
                }
            }
            // Close the merged link so the sink arm ends.
            out.push(PipelinePacket::Eos).await?;
            Ok(FaninArmOut::Aggregator(consumed))
        });

        let arms: Vec<BoxFuture<'a, Result<FaninArmOut, G2gError>>> =
            alloc::vec![muxer_arm, sink_arm];
        let results = dynamic_join(arms, new_arm_rx).await;

        let mut emitted = 0u64;
        let mut consumed = 0u64;
        for r in results {
            match r? {
                FaninArmOut::Source(n) => emitted += n,
                FaninArmOut::Sink(n) => consumed = n,
                FaninArmOut::Aggregator(_) => {}
            }
        }
        Ok(RunStats {
            frames_emitted: emitted,
            frames_consumed: consumed,
            frames_dropped: 0,
            latency: LatencyReport::ZERO,
            allocation: None,
            clock_priority: ClockPriority::SystemFallback,
            base_time_ns: 0,
            coordinator_events: 0,
            per_element: alloc::vec![mux_probe.snapshot(), sink_probe.snapshot()],
        })
    };

    (handle, run)
}

/// Attach a runtime-requested input: fixate the new `source`, configure the
/// aggregator's `pad` against its fixated caps, then hand the source's run loop
/// (feeding a [`TaggingSink`] tagged with `pad`) to the dynamic join. Mirrors
/// [`run_source_router_dynamic`](crate::runtime::run_source_router_dynamic)'s
/// `attach_branch`, transposed to the input side.
#[cfg(feature = "std")]
async fn attach_input<'a>(
    pad: usize,
    mut source: Box<dyn DynSourceLoop + 'a>,
    aggregator: &mut dyn DynMultiInputElement,
    tagged_tx: &Sender<(usize, PipelinePacket)>,
    new_arm_tx: &Sender<BoxFuture<'a, Result<FaninArmOut, G2gError>>>,
    tap: &mut FaninTap,
) -> Result<(), G2gError> {
    // Narrowed against the aggregator pad it feeds (like run_fanin_session); the
    // fixated caps configure both the source and that pad.
    let produced = source.produced_caps().await?;
    let fixated = {
        let pad_constraint = aggregator.caps_constraint_as_input(pad);
        select_branch_caps(&produced, &pad_constraint)?
    };
    source.configure_pipeline(&fixated)?.reject_refixate()?;
    aggregator
        .configure_pipeline(pad, &fixated)?
        .reject_refixate()?;

    // Named like an input the static fan-in was built with, and registered before
    // the arm runs, so the topology holds it before its first frame. The tagged
    // channel is shared by every input, so this input's counters and inspect slot
    // ride its `TaggingSink`.
    let name = tap.namer.add(source.log_category(), None);
    source.set_instance_name(name.clone());
    let (counters, slot) = match &tap.obs {
        Some(obs) => {
            let counters = Arc::new(EdgeCounters::default());
            let slot = ProbeSlot::default();
            let id = obs.add_node(name, NodeRole::Source, None);
            obs.add_edge(
                id,
                DYN_FANIN_NODE,
                fixated,
                EdgeTap {
                    probe: slot.clone(),
                    counters: Some(counters.clone()),
                },
            );
            (Some(counters), Some(slot))
        }
        None => (None, None),
    };

    let tx = tagged_tx.clone();
    let arm: BoxFuture<'a, Result<FaninArmOut, G2gError>> = Box::pin(async move {
        let mut sink = TaggingSink {
            idx: pad,
            tx,
            reverse: None,
            counters,
            probe: slot,
        };
        let mut source = source;
        source.run(&mut sink).await.map(FaninArmOut::Source)
    });
    new_arm_tx.try_send(arm).map_err(|_| G2gError::Shutdown)
}
