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
use core::sync::atomic::{AtomicUsize, Ordering};

use crate::bus::BusHandle;
use crate::caps::Caps;
use crate::clock::PipelineClock;
use crate::element::{
    AsyncElement, BoxFuture, ConfigureOutcome, ElementBound, OutputSink, Reconfigure,
};
use crate::clock::{ClockCandidate, ClockPriority};
use crate::format_element::CapsConstraint;
use crate::error::G2gError;
use crate::fanout::{Merger, MultiInputElement};
use crate::frame::PipelinePacket;
use crate::graph::Graph;
use crate::property::{ElementMetadata, PropError, PropValue, PropertySpec};
use crate::query::{AllocationParams, LatencyReport};
use crate::runtime::channel::{link, SenderSink};
use crate::runtime::graph_runner::{run_graph_inner, GraphNodeRef};
use crate::runtime::join::join_all;
use crate::runtime::runner::{LinkCapacity, NullSink, RunStats, SourceLoop};

/// Dyn-safe mirror of [`SourceLoop`] for heterogeneous fan-in branches, the
/// source-side analog of [`DynAsyncElement`](crate::element::DynAsyncElement).
/// Boxes `run`'s future so a `Vec<&mut dyn DynSourceLoop>` can hold sources
/// of different concrete types.
pub trait DynSourceLoop: ElementBound {
    fn intercept_caps<'a>(
        &'a mut self,
    ) -> BoxFuture<'a, Result<Caps, G2gError>>;

    fn configure_pipeline(
        &mut self,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError>;

    fn run<'a>(
        &'a mut self,
        out: &'a mut dyn OutputSink,
    ) -> BoxFuture<'a, Result<u64, G2gError>>;

    fn reconfigure(&mut self, request: Reconfigure) -> Result<Caps, G2gError>;

    /// Dyn-safe mirror of [`SourceLoop::latency`], so the DAG runner folds a
    /// source's latency contribution like the linear runner does.
    fn latency(&self) -> LatencyReport;

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
    fn intercept_caps<'a>(
        &'a mut self,
    ) -> BoxFuture<'a, Result<Caps, G2gError>> {
        Box::pin(SourceLoop::intercept_caps(self))
    }

    fn configure_pipeline(
        &mut self,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        SourceLoop::configure_pipeline(self, absolute_caps)
    }

    fn run<'a>(
        &'a mut self,
        out: &'a mut dyn OutputSink,
    ) -> BoxFuture<'a, Result<u64, G2gError>> {
        Box::pin(SourceLoop::run(self, out))
    }

    fn reconfigure(&mut self, request: Reconfigure) -> Result<Caps, G2gError> {
        SourceLoop::reconfigure(self, request)
    }

    fn latency(&self) -> LatencyReport {
        SourceLoop::latency(self)
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

    fn configure_pipeline(
        &mut self,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
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

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        (**self).set_property(name, value)
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        (**self).get_property(name)
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
    fn caps_constraint_as_input(&self, input: usize) -> CapsConstraint<'_>;
    fn caps_constraint_for_output(&self) -> Result<CapsConstraint<'_>, G2gError>;
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
}

impl<T: MultiInputElement> DynMultiInputElement for T {
    fn input_count(&self) -> usize {
        MultiInputElement::input_count(self)
    }

    fn caps_constraint_as_input(&self, input: usize) -> CapsConstraint<'_> {
        MultiInputElement::caps_constraint_as_input(self, input)
    }

    fn caps_constraint_for_output(&self) -> Result<CapsConstraint<'_>, G2gError> {
        MultiInputElement::caps_constraint_for_output(self)
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
}

/// Forwarding impl so a borrowed `&mut dyn DynMultiInputElement` can be boxed
/// into a `Box<dyn DynMultiInputElement + 'a>` graph node (the muxer wrapper
/// builds a borrowing `Graph` over its `&mut` muxer reference). Disjoint from
/// the `MultiInputElement` blanket above.
impl<'b> DynMultiInputElement for &'b mut (dyn DynMultiInputElement + 'b) {
    fn input_count(&self) -> usize {
        (**self).input_count()
    }

    fn caps_constraint_as_input(&self, input: usize) -> CapsConstraint<'_> {
        (**self).caps_constraint_as_input(input)
    }

    fn caps_constraint_for_output(&self) -> Result<CapsConstraint<'_>, G2gError> {
        (**self).caps_constraint_for_output()
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
    _clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
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

    // Phase 1 + 2: fixate each source's caps and configure it; the sink is
    // configured against input 0's fixated caps (the merged-output caps).
    // This is not routed through `solve_linear` because each source
    // self-fixates with no peer narrowing — there's no chain to solve.
    let mut sources = sources;
    let mut merged_caps: Option<Caps> = None;
    for (i, source) in sources.iter_mut().enumerate() {
        let proposal = source.intercept_caps().await?;
        let fixated = proposal.fixate()?;
        if let ConfigureOutcome::ReFixate(_) = source.configure_pipeline(&fixated)? {
            return Err(G2gError::FixationFailed);
        }
        if i == 0 {
            merged_caps = Some(fixated);
        }
    }
    let merged_caps = merged_caps.expect("input_count > 0");
    if let ConfigureOutcome::ReFixate(_) = sink.configure_pipeline(&merged_caps)? {
        return Err(G2gError::FixationFailed);
    }

    // One input link per source, one shared output link to the sink.
    let mut input_senders = Vec::with_capacity(input_count);
    let mut input_receivers = Vec::with_capacity(input_count);
    for _ in 0..input_count {
        let (tx, rx) = link(link_capacity);
        input_senders.push(tx);
        input_receivers.push(rx);
    }
    let (out_tx, out_rx) = link(link_capacity);
    let live_inputs = Arc::new(AtomicUsize::new(input_count));

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
                    match sink.configure_pipeline(&new_caps)? {
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
                    if matches!(packet, PipelinePacket::DataFrame(_)) {
                        consumed += 1;
                    }
                    sink.process(packet, &mut null).await?;
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
    run_muxer_sink_inner(sources, mux, sink, clock, link_capacity, None).await
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
    run_muxer_sink_inner(sources, mux, sink, clock, link_capacity, Some(bus)).await
}

async fn run_muxer_sink_inner<Mux, Snk, Clk>(
    sources: Vec<&mut dyn DynSourceLoop>,
    mux: &mut Mux,
    sink: &mut Snk,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    bus: Option<&BusHandle>,
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
        g.link(s, mux_node.input(i as u8)).map_err(|_| G2gError::CapsMismatch)?;
    }
    g.link(mux_node.output(), snk).map_err(|_| G2gError::CapsMismatch)?;

    run_graph_inner(g, clock, link_capacity, bus, None).await
}
