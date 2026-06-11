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

use crate::caps::Caps;
use crate::clock::PipelineClock;
use crate::element::{
    AsyncElement, BoxFuture, ConfigureOutcome, ElementBound, OutputSink, Reconfigure,
};
use crate::clock::ClockPriority;
use crate::format_element::CapsConstraint;
use crate::runtime::solver::solve_linear;
use crate::error::G2gError;
use crate::fanout::{Merger, MultiInputElement};
use crate::frame::PipelinePacket;
use crate::query::LatencyReport;
use crate::runtime::channel::{bounded, link, SenderSink};
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
        latency: LatencyReport::ZERO,
        allocation: None,
        clock_priority: ClockPriority::SystemFallback,
        base_time_ns: 0,
        coordinator_events: 0,
    })
}

/// Solve `source-output ↔ muxer-input[idx]`, returning the fixated input
/// caps. Used by `run_muxer_sink` both at startup and on a mid-stream
/// per-input `CapsChanged` (Phase C MX-1), so the two paths share one
/// solver call.
///
/// `DynSourceLoop` doesn't expose `caps_constraint` (M16 5f added it only
/// to the concrete `SourceLoop`), so muxer sources go through the legacy
/// `LegacySource` bridge here; migrated muxers (e.g. `InterleaveMux` with
/// `AcceptsAny`) still hit the all-native solver path on the muxer side.
/// Expose the method on `DynSourceLoop` when a migrated source needs to
/// feed a muxer.
fn solve_mux_input<Mux>(source_caps: Caps, mux: &Mux, idx: usize) -> Result<Caps, G2gError>
where
    Mux: MultiInputElement,
{
    let src_c = CapsConstraint::LegacySource(source_caps);
    let mux_c = mux.caps_constraint_as_input(idx);
    let links = solve_linear(&[&src_c, &mux_c]).map_err(|_| G2gError::CapsMismatch)?;
    links.last().cloned().ok_or(G2gError::CapsMismatch)
}

/// Solve `muxer-output ↔ downstream sink`, returning the fixated merged
/// output caps. Used at startup and, on a per-input mid-stream change, to
/// re-derive the output (Phase C MX-2). For a static muxer
/// (`InterleaveMux` returns `Produces(set)`) this equals today's direct
/// fixate; for a future input-derived muxer (`DerivedOutput`) the same
/// call re-derives the output from the freshly-reconfigured per-input
/// caps. The sink side is `AcceptsAny`: the muxer's aggregated output is
/// the canonical merged caps and is not narrowed by the sink.
fn solve_mux_output<Mux>(mux: &Mux) -> Result<Caps, G2gError>
where
    Mux: MultiInputElement,
{
    let mux_c = mux.caps_constraint_for_output()?;
    let sink_c = CapsConstraint::AcceptsAny;
    let links = solve_linear(&[&mux_c, &sink_c]).map_err(|_| G2gError::CapsMismatch)?;
    links.last().cloned().ok_or(G2gError::CapsMismatch)
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
    _clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
) -> Result<RunStats, G2gError>
where
    Mux: MultiInputElement,
    Snk: AsyncElement,
    Clk: PipelineClock,
{
    let link_capacity: usize = link_capacity.into().get();
    let input_count = sources.len();
    assert!(input_count > 0, "muxer needs at least one source");
    assert!(
        mux.input_count() == input_count,
        "muxer input count must match the number of sources"
    );

    // M18 step 1: solve each source ↔ muxer-input pair via the shared
    // `solve_mux_input` helper (Phase C MX-1 reuses it mid-stream).
    let mut sources = sources;
    for (i, source) in sources.iter_mut().enumerate() {
        let proposal = source.intercept_caps().await?;
        let fixated = solve_mux_input(proposal, mux, i)?;
        if let ConfigureOutcome::ReFixate(_) = source.configure_pipeline(&fixated)? {
            return Err(G2gError::FixationFailed);
        }
        if let ConfigureOutcome::ReFixate(_) =
            MultiInputElement::configure_pipeline(mux, i, &fixated)?
        {
            return Err(G2gError::FixationFailed);
        }
    }
    // M18 step 1: muxer output negotiation via `solve_mux_output` (Phase C
    // MX-2 reuses it to re-derive the output mid-stream). `output` is the
    // initial merged caps the muxer task tracks for MX-2 change detection.
    let output = solve_mux_output(mux)?;
    if let ConfigureOutcome::ReFixate(_) = sink.configure_pipeline(&output)? {
        return Err(G2gError::FixationFailed);
    }

    // One input link per source, one tagged merge channel, one output link.
    let mut input_senders = Vec::with_capacity(input_count);
    let mut input_receivers = Vec::with_capacity(input_count);
    for _ in 0..input_count {
        let (tx, rx) = link(link_capacity);
        input_senders.push(tx);
        input_receivers.push(rx);
    }
    let (tagged_tx, tagged_rx) = bounded::<(usize, PipelinePacket)>(link_capacity);
    let (out_tx, out_rx) = link(link_capacity);

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
        let tagged = tagged_tx.clone();
        forwarder_arms.push(Box::pin(async move {
            loop {
                match in_rx.recv().await {
                    Some(PipelinePacket::Eos) | None => {
                        // Tag this input's end; the muxer task aggregates.
                        tagged
                            .send((idx, PipelinePacket::Eos))
                            .await
                            .map_err(|_| G2gError::Shutdown)?;
                        return Ok::<u64, G2gError>(0);
                    }
                    Some(packet) => {
                        tagged
                            .send((idx, packet))
                            .await
                            .map_err(|_| G2gError::Shutdown)?;
                    }
                }
            }
        }));
    }
    // Only the forwarders keep the tagged channel open.
    drop(tagged_tx);

    let muxer_arm: BoxFuture<'_, Result<u64, G2gError>> = Box::pin(async move {
        let mut out = SenderSink::new(out_tx);
        let mut ended = 0usize;
        // MX-2 change detection: the merged output caps the muxer last
        // advertised downstream. Seeded with the startup-negotiated output.
        let mut current_output = output;
        loop {
            match tagged_rx.recv().await {
                Some((_, PipelinePacket::Eos)) => {
                    ended += 1;
                    if ended == input_count {
                        out.push(PipelinePacket::Eos).await?;
                        return Ok::<u64, G2gError>(0);
                    }
                }
                Some((i, PipelinePacket::CapsChanged(new_caps))) => {
                    // Phase C MX-1: per-input re-solve, Phase B applied to
                    // input `i`. The per-input `CapsChanged` is input-side
                    // and consumed here (it must NOT leak to the output as
                    // if it were the merged caps). Strict failure: a
                    // per-input caps the muxer can't accept fails loud.
                    // (The β-clean variant signals a structured reverse
                    // `Renegotiate` into source `i` instead; the muxer task
                    // can't reach input `i`'s reconfigure slot pre-β.)
                    let input_caps = solve_mux_input(new_caps, mux, i)?;
                    if let ConfigureOutcome::ReFixate(_) =
                        MultiInputElement::configure_pipeline(mux, i, &input_caps)?
                    {
                        return Err(G2gError::FixationFailed);
                    }
                    // Phase C MX-2: the per-input change may shift the
                    // merged output. Re-derive it from the freshly
                    // reconfigured per-input caps and, only if it actually
                    // changed, eagerly emit one downstream `CapsChanged`
                    // (eager emit recommended in §10.3; the frame-caps
                    // invariant keeps in-flight old-caps frames correct).
                    // Static muxers leave the output unchanged, so nothing
                    // is emitted.
                    let new_output = solve_mux_output(mux)?;
                    if new_output != current_output {
                        current_output = new_output.clone();
                        out.push(PipelinePacket::CapsChanged(new_output)).await?;
                    }
                }
                Some((i, packet)) => {
                    MultiInputElement::process(mux, i, packet, &mut out).await?;
                }
                None => return Ok(0),
            }
        }
    });

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

    // Arm order: [source0..N, forwarder0..N, muxer, sink].
    let mut arms = Vec::with_capacity(2 * input_count + 2);
    arms.extend(source_arms);
    arms.extend(forwarder_arms);
    arms.push(muxer_arm);
    arms.push(sink_arm);

    let results = join_all(arms).await;
    let mut counts = Vec::with_capacity(results.len());
    for r in results {
        counts.push(r?);
    }
    let emitted: u64 = counts[0..input_count].iter().copied().sum();
    let consumed = counts[2 * input_count + 1];
    // Fan-in latency / allocation / clock election across N inputs is deferred
    // (M12 covers the linear path); report neutral values rather than a
    // misleading partial one.
    Ok(RunStats {
        frames_emitted: emitted,
        frames_consumed: consumed,
        latency: LatencyReport::ZERO,
        allocation: None,
        clock_priority: ClockPriority::SystemFallback,
        base_time_ns: 0,
        coordinator_events: 0,
    })
}
