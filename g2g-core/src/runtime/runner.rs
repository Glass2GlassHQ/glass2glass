use core::future::Future;

use alloc::boxed::Box;

use crate::caps::Caps;
use crate::clock::{elect_clock, ClockCandidate, ClockPriority, PipelineClock};
use crate::element::{
    AsyncElement, BoxFuture, ConfigureOutcome, ElementBound, OutputSink, PushOutcome, Reconfigure,
};
use crate::error::G2gError;
use crate::format_element::CapsConstraint;
use crate::frame::PipelinePacket;
use crate::query::{AllocationParams, LatencyReport};
use crate::runtime::channel::{link, SenderSink};
use crate::runtime::coordinator::{
    coordinator, negotiate_source_transform_sink, realloc_local, CoordinatorEvent,
    MAX_FIXATION_ATTEMPTS,
};
use crate::runtime::join::Join2;
use crate::runtime::solver::{solve_linear, NegotiationFailure};

#[cfg(feature = "std")]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use crate::element::DynAsyncElement;
#[cfg(feature = "std")]
use crate::fanout::{MultiOutputElement, MultiOutputSink, MultiSenderSink};
#[cfg(feature = "std")]
use crate::runtime::join::join_all;

/// Source-side element trait. Sources have no input pad, so the packet-in /
/// packet-out shape of [`AsyncElement`] does not fit them. A `SourceLoop`
/// instead receives a single `run` call that iterates internally until EOS
/// and returns the count of `DataFrame` packets pushed.
pub trait SourceLoop: ElementBound {
    type RunFuture<'a>: Future<Output = Result<u64, G2gError>> + 'a
    where
        Self: 'a;

    fn intercept_caps(&self) -> Result<Caps, G2gError>;

    fn configure_pipeline(
        &mut self,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError>;

    /// Runs the source until EOS or error. The implementation MUST emit a
    /// final `PipelinePacket::Eos` before returning `Ok`. Returns the number
    /// of `DataFrame` packets pushed (excluding `Eos`).
    fn run<'a>(
        &'a mut self,
        out: &'a mut dyn OutputSink,
    ) -> Self::RunFuture<'a>;

    /// Handle a downstream-originated `Reconfigure` request observed via
    /// `PushOutcome::Reconfigure` during `run`. Implementations that can
    /// retarget (eg picking a sub-stream over a main stream from an IP
    /// camera, or switching bitrate) return the new caps they will produce
    /// next; the source's `run` loop is then responsible for emitting a
    /// `CapsChanged` packet and resuming under those caps.
    ///
    /// Default: reject — most sources can't change their output shape and
    /// `FixationFailed` propagates as a fatal pipeline error.
    fn reconfigure(&mut self, _request: Reconfigure) -> Result<Caps, G2gError> {
        Err(G2gError::FixationFailed)
    }

    /// This source's latency contribution to the pipeline latency query (M12).
    /// Live capture sources (cameras, RTSP) override this to report `live`
    /// with their capture interval as `min_ns`; the default is zero, non-live
    /// (eg a file or test-pattern source that can produce data on demand).
    fn latency(&self) -> LatencyReport {
        LatencyReport::ZERO
    }

    /// Receive the downstream peer's allocation proposal (M12) so the source
    /// can allocate its output `BufferPool` from compatible parameters
    /// (size, count, alignment, domain). Default: ignore and allocate the
    /// source's own way. The proposal is advisory; a source that cannot honor
    /// it (eg cannot produce the requested domain) falls back silently.
    fn configure_allocation(&mut self, _params: &AllocationParams) {}

    /// Offer a clock to the pipeline's clock election (M12). Default: none.
    /// Live capture sources override this to provide their hardware capture
    /// clock at [`ClockPriority::LiveSource`](crate::ClockPriority::LiveSource)
    /// so the pipeline paces to capture cadence.
    fn provide_clock(&self) -> Option<ClockCandidate> {
        None
    }

    /// M16 step 5f: declare this source's negotiation-time constraint.
    /// Default: eagerly evaluate `intercept_caps()` and wrap as a
    /// `LegacySource(Caps)` for the solver. Migrated sources override
    /// to return `Produces(CapsSet)` (or another native variant) and
    /// the chain takes the native arc-consistency path when every
    /// other element is also native.
    fn caps_constraint(&self) -> Result<CapsConstraint<'_>, G2gError> {
        Ok(CapsConstraint::LegacySource(self.intercept_caps()?))
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RunStats {
    pub frames_emitted: u64,
    pub frames_consumed: u64,
    /// Aggregated source-to-sink latency (M12), computed once after
    /// negotiation. Linear runners fold every element's `latency()`; fan-in /
    /// fan-out runners leave this at `ZERO` (topology aggregation deferred).
    pub latency: LatencyReport,
    /// The allocation proposal (M12) handed to the head producer (the source),
    /// negotiated from downstream. `None` when no downstream element proposed
    /// one; always `None` for fan-in / fan-out runners (deferred).
    pub allocation: Option<AllocationParams>,
    /// Priority of the clock the pipeline elected (M12). `SystemFallback` when
    /// no element provided one (the supplied clock stands); always
    /// `SystemFallback` for fan-in / fan-out runners (deferred).
    pub clock_priority: ClockPriority,
    /// `now_ns()` of the elected clock, read once after election — the
    /// pipeline's base-time origin.
    pub base_time_ns: u64,
    /// M18 β scaffolding: number of `CoordinatorEvent`s the coordinator
    /// task observed over this run's control channel. Today the only
    /// event is a boundary forwarding a mid-stream `CapsChanged` the next
    /// element accepted; β will turn each into a `Recascade`. `0` for
    /// runners that don't yet spawn a coordinator (simple / fan-out /
    /// fan-in / muxer).
    pub coordinator_events: u64,
}

/// M16 workaround #3 Phase B helper: re-solve the downstream subgraph
/// when a forward `CapsChanged` crosses a format boundary mid-stream.
///
/// Today's 3-element runner has a single downstream link (boundary →
/// sink), so the subgraph is one link and the solver's role is
/// structural: it queries the sink's declared `CapsConstraint` (which
/// may reject the boundary's output via `Accepts(set)` cleanly) before
/// the sink ever sees `configure_pipeline`. The returned `Caps` is what
/// the runner then hands to `configure_pipeline`.
///
/// Longer chains (4+ elements, future runner variants) will iterate
/// the solver result to reconfigure every changed downstream link, not
/// just the immediate next element — that's the structural unlock
/// `DESIGN-M16-workaround3-reconfigure.md` §4 calls out.
///
/// Forward × reverse race (§7): an `EmptyLink` here means the sink
/// can't take the boundary's output. The caller drops the forward
/// `CapsChanged` and signals a reverse `Reconfigure` *into the
/// boundary*, not past it to the source — that boundary owns the
/// derivation and is the right place to surface the structured
/// failure. An `Unfixable` (the boundary's caps left a ranged field
/// like `Rate::Any` — common for decoders that don't know framerate at
/// the pixel level) is *not* a failure: it means the sink accepted the
/// shape, the caps just aren't fully fixated. Pass `new_caps` through
/// unchanged.
fn re_solve_downstream_sink<S>(new_caps: &Caps, sink: &S) -> Result<Caps, NegotiationFailure>
where
    S: AsyncElement + ?Sized,
{
    let src_c = CapsConstraint::LegacySource(new_caps.clone());
    let sink_c = sink.caps_constraint_as_sink();
    match solve_linear(&[&src_c, &sink_c]) {
        Ok(links) => links.into_iter().last().ok_or(NegotiationFailure::Degenerate),
        Err(NegotiationFailure::Unfixable { .. }) => Ok(new_caps.clone()),
        Err(other) => Err(other),
    }
}

/// Drives a `source → sink` pipeline over a single bounded link.
/// Initial Phase 1+2 negotiation runs with bounded `ReFixate` backtrack
/// (M8 piece 5): if any element's `configure_pipeline()` returns a
/// counter-proposal, the runner restarts negotiation with that counter
/// as the new starting proposal, up to `MAX_FIXATION_ATTEMPTS` total.
pub async fn run_simple_pipeline<Src, Snk, Clk>(
    source: &mut Src,
    sink: &mut Snk,
    clock: &Clk,
    link_capacity: usize,
) -> Result<RunStats, G2gError>
where
    Src: SourceLoop,
    Snk: AsyncElement,
    Clk: PipelineClock,
{
    // M16 step 5f: startup negotiation honors `SourceLoop::caps_constraint`
    // so migrated native sources (e.g. `VideoTestSrc::Produces(...)`)
    // take the native solver path. `ReFixate` retry falls back to
    // `LegacySource(counter)` because counter-proposals are a legacy
    // model concept and native sources don't accept them.
    let mut refix_counter: Option<Caps> = None;
    let mut attempts = 0u32;
    let negotiated_caps = loop {
        attempts += 1;
        if attempts > MAX_FIXATION_ATTEMPTS {
            return Err(G2gError::FixationFailed);
        }
        let fixated = {
            let src_c = match &refix_counter {
                Some(c) => CapsConstraint::LegacySource(c.clone()),
                None => source.caps_constraint()?,
            };
            let sink_c = sink.caps_constraint_as_sink();
            let links = solve_linear(&[&src_c, &sink_c])
                .map_err(|_| G2gError::CapsMismatch)?;
            links.last().cloned().ok_or(G2gError::CapsMismatch)?
        };
        match source.configure_pipeline(&fixated)? {
            ConfigureOutcome::Accepted => {}
            ConfigureOutcome::ReFixate(counter) => {
                refix_counter = Some(counter);
                continue;
            }
        }
        match sink.configure_pipeline(&fixated)? {
            ConfigureOutcome::Accepted => break fixated,
            ConfigureOutcome::ReFixate(counter) => {
                refix_counter = Some(counter);
                continue;
            }
        }
    };

    // M12 latency query: fold the configured chain source → sink.
    let latency = LatencyReport::aggregate([source.latency(), AsyncElement::latency(sink)]);

    // M12 allocation query: the sink proposes buffers; the source allocates
    // its output pool to match (zero-copy handoff when it can honor them).
    let allocation = sink.propose_allocation(&negotiated_caps);
    if let Some(p) = &allocation {
        source.configure_allocation(p);
    }

    // M12 clock distribution: elect the pipeline clock (source > sink > fallback).
    let elected = elect_clock([source.provide_clock(), AsyncElement::provide_clock(sink)]);
    let (clock_priority, base_time_ns) = match &elected {
        Some(c) => (c.priority, c.clock.now_ns()),
        None => (ClockPriority::SystemFallback, clock.now_ns()),
    };

    let (link_tx, link_rx) = link(link_capacity);

    let source_fut = async move {
        let mut adapter = SenderSink::new(link_tx);
        let emitted = source.run(&mut adapter).await?;
        Ok::<u64, G2gError>(emitted)
    };

    let sink_fut = async move {
        let mut null = NullSink;
        let mut consumed: u64 = 0;
        loop {
            match link_rx.recv().await {
                Some(PipelinePacket::Eos) => {
                    sink.process(PipelinePacket::Eos, &mut null).await?;
                    return Ok::<u64, G2gError>(consumed);
                }
                Some(PipelinePacket::CapsChanged(new_caps)) => {
                    // M16 workaround #3 Phase B: re-solve the downstream
                    // subgraph before applying. For a 2-element chain
                    // the subgraph is one link, so the solver's role is
                    // structural — it checks the sink's declared
                    // `CapsConstraint::caps_constraint_as_sink()`
                    // (which a native sink may use to reject the new
                    // shape cleanly) before any `configure_pipeline`
                    // call. Failure becomes a structured upstream
                    // `Renegotiate` request instead of an opaque
                    // `CapsMismatch`.
                    let sink_caps = match re_solve_downstream_sink(&new_caps, &*sink) {
                        Ok(caps) => caps,
                        Err(_) => {
                            link_rx
                                .request_reconfigure(Reconfigure::Renegotiate);
                            continue;
                        }
                    };
                    // M8 piece 1: runner cascades mid-stream caps changes
                    // through configure_pipeline before the element sees
                    // the notification packet. Guarantees DataFrames with
                    // the new caps never reach a stale element.
                    match sink.configure_pipeline(&sink_caps)? {
                        ConfigureOutcome::Accepted => {
                            // M18 α: element-local re-allocation under the
                            // new caps before the sink sees the packet.
                            realloc_local(sink, &sink_caps);
                            sink.process(
                                PipelinePacket::CapsChanged(sink_caps),
                                &mut null,
                            )
                            .await?;
                        }
                        // M8 piece 5: a sink that rejects new caps fires
                        // its counter-proposal upstream as a Reconfigure
                        // signal. The source observes it on its next push
                        // (piece 4 wires source-side handling). The
                        // CapsChanged packet is dropped — caps were not
                        // accepted — and we keep draining old-caps frames
                        // until the source emits a fresh CapsChanged.
                        ConfigureOutcome::ReFixate(counter) => {
                            link_rx.request_reconfigure(Reconfigure::Propose(counter));
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
    };

    let (src_res, snk_res) = Join2::new(source_fut, sink_fut).await;
    let emitted = src_res?;
    let consumed = snk_res?;

    Ok(RunStats {
        frames_emitted: emitted,
        frames_consumed: consumed,
        latency,
        allocation,
        clock_priority,
        base_time_ns,
        coordinator_events: 0,
    })
}

/// Drives a `source → fan-out element → N sinks` pipeline (M9 fan-out core).
/// The fan-out element (a [`MultiOutputElement`], e.g. `Router`) sends each
/// `DataFrame` to one branch and broadcasts `CapsChanged` to all; the runner
/// broadcasts `Eos` to every branch on shutdown.
///
/// Heterogeneous branches arrive as `Box`-erased `&mut dyn DynAsyncElement`
/// (std only). Negotiation fixates the source proposal once and configures
/// every element with it (DESIGN.md §4.2); per-branch caps negotiation is
/// M10, so a sink returning `ReFixate` here fails with `FixationFailed`.
#[cfg(feature = "std")]
pub async fn run_source_fanout<Src, Tx, Clk>(
    source: &mut Src,
    fanout: &mut Tx,
    sinks: Vec<&mut dyn DynAsyncElement>,
    _clock: &Clk,
    link_capacity: usize,
) -> Result<RunStats, G2gError>
where
    Src: SourceLoop,
    Tx: MultiOutputElement,
    Clk: PipelineClock,
{
    let branch_count = sinks.len();
    assert!(branch_count > 0, "fan-out needs at least one sink");

    // M18 step 1: solve source → fanout via the solver using the new
    // `MultiOutputElement::caps_constraint_as_input()` trait method
    // (M16 step 4c had this constructing `LegacySink` inline). The
    // fan-out acts as the linear "sink" of the negotiation chain;
    // the real sinks downstream of it broadcast-receive the same
    // fixated caps and don't participate in narrowing. Phase C FO-2
    // (per-branch downstream re-solve once a mid-stream `CapsChanged`
    // crosses the fan-out boundary) lands once β (the coordinator
    // restructure) does — it slots in here via per-branch calls to
    // `re_solve_downstream_sink`.
    let fixated = {
        let src_c = source.caps_constraint()?;
        let fanout_c = fanout.caps_constraint_as_input();
        let links = solve_linear(&[&src_c, &fanout_c])
            .map_err(|_| G2gError::CapsMismatch)?;
        links.last().cloned().ok_or(G2gError::CapsMismatch)?
    };

    if let ConfigureOutcome::ReFixate(_) = source.configure_pipeline(&fixated)? {
        return Err(G2gError::FixationFailed);
    }
    if let ConfigureOutcome::ReFixate(_) =
        MultiOutputElement::configure_pipeline(fanout, &fixated)?
    {
        return Err(G2gError::FixationFailed);
    }
    let mut sinks = sinks;
    for sink in sinks.iter_mut() {
        if let ConfigureOutcome::ReFixate(_) = sink.configure_pipeline(&fixated)? {
            return Err(G2gError::FixationFailed);
        }
    }

    let (src_tx, src_rx) = link(link_capacity);
    let mut branch_senders = Vec::with_capacity(branch_count);
    let mut branch_receivers = Vec::with_capacity(branch_count);
    for _ in 0..branch_count {
        let (tx, rx) = link(link_capacity);
        branch_senders.push(SenderSink::new(tx));
        branch_receivers.push(rx);
    }

    let source_fut: BoxFuture<'_, Result<u64, G2gError>> = Box::pin(async move {
        let mut adapter = SenderSink::new(src_tx);
        source.run(&mut adapter).await
    });

    let router_fut: BoxFuture<'_, Result<u64, G2gError>> = Box::pin(async move {
        let mut multi = MultiSenderSink::new(branch_senders);
        loop {
            match src_rx.recv().await {
                Some(PipelinePacket::Eos) => {
                    MultiOutputElement::process(fanout, PipelinePacket::Eos, &mut multi).await?;
                    for port in 0..branch_count {
                        multi.push_to(port, PipelinePacket::Eos).await?;
                    }
                    return Ok::<u64, G2gError>(0);
                }
                Some(packet) => {
                    MultiOutputElement::process(fanout, packet, &mut multi).await?;
                }
                None => return Ok(0),
            }
        }
    });

    let mut arms: Vec<BoxFuture<'_, Result<u64, G2gError>>> =
        Vec::with_capacity(branch_count + 2);
    arms.push(source_fut);
    arms.push(router_fut);

    for (sink, rx) in sinks.into_iter().zip(branch_receivers) {
        let sink_fut: BoxFuture<'_, Result<u64, G2gError>> = Box::pin(async move {
            let mut null = NullSink;
            let mut consumed: u64 = 0;
            loop {
                match rx.recv().await {
                    Some(PipelinePacket::Eos) => {
                        sink.process(PipelinePacket::Eos, &mut null).await?;
                        return Ok::<u64, G2gError>(consumed);
                    }
                    Some(PipelinePacket::CapsChanged(new_caps)) => {
                        match sink.configure_pipeline(&new_caps)? {
                            ConfigureOutcome::Accepted => {
                                sink.process(
                                    PipelinePacket::CapsChanged(new_caps),
                                    &mut null,
                                )
                                .await?;
                            }
                            ConfigureOutcome::ReFixate(counter) => {
                                rx.request_reconfigure(Reconfigure::Propose(counter));
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
        arms.push(sink_fut);
    }

    let results = join_all(arms).await;
    let mut counts = Vec::with_capacity(results.len());
    for r in results {
        counts.push(r?);
    }
    // Arm order: [source, router, sink0, sink1, ...].
    let emitted = counts[0];
    let consumed: u64 = counts[2..].iter().copied().sum();
    // Fan-out latency / allocation / clock election across N branches is
    // deferred (M12 covers the linear path); report neutral values rather than
    // a misleading partial one.
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

/// Sentinel sink for terminal elements (sinks proper): swallows pushes.
/// Process implementations of true sinks should not emit, but the type
/// system still requires an `&mut dyn OutputSink` parameter.
#[derive(Debug)]
pub(crate) struct NullSink;

impl OutputSink for NullSink {
    fn push<'a>(
        &'a mut self,
        _packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async { Ok(PushOutcome::Accepted) })
    }
}

/// Drives a `source → transform → sink` pipeline over two bounded links.
///
/// Transform contract: `process(Eos)` may flush buffered state as
/// `DataFrame` packets but MUST NOT emit `Eos` itself — the runner forwards
/// the EOS sentinel downstream after `process(Eos)` returns.
///
/// `link_capacity` is the primary glass-to-glass latency knob. Under
/// steady-state backpressure each link sits full, so the latency floor is
/// roughly `2 * link_capacity * consumer_period`. For live video pipelines
/// (RTSP -> decode -> display) prefer **2**; for batch / throughput-oriented
/// workloads larger values are fine.
pub async fn run_source_transform_sink<Src, Tx, Snk, Clk>(
    source: &mut Src,
    transform: &mut Tx,
    sink: &mut Snk,
    clock: &Clk,
    link_capacity: usize,
) -> Result<RunStats, G2gError>
where
    Src: SourceLoop,
    Tx: AsyncElement,
    Snk: AsyncElement,
    Clk: PipelineClock,
{
    // M18 Session C: the startup negotiation loop (solver + per-link
    // configure cascade with bounded `ReFixate` retry) is owned by the
    // coordinator module now, since β reuses the same machinery for the
    // mid-stream re-cascade. `sink_link` is the downstream-facing caps
    // (transform output = sink input) that M12 allocation flows along,
    // so it stands in for the loop's former `negotiated_caps`.
    let negotiated_caps = negotiate_source_transform_sink(source, transform, sink)?.sink_link;

    // M12 latency query: fold the configured chain source → transform → sink.
    let latency = LatencyReport::aggregate([
        source.latency(),
        AsyncElement::latency(transform),
        AsyncElement::latency(sink),
    ]);

    // M12 allocation query: each producer asks its downstream peer what
    // buffers to allocate. Resolve sink → transform first so the transform can
    // fold the sink's requirement into the proposal it answers to the source.
    if let Some(p) = sink.propose_allocation(&negotiated_caps) {
        AsyncElement::configure_allocation(transform, &p);
    }
    let allocation = transform.propose_allocation(&negotiated_caps);
    if let Some(p) = &allocation {
        source.configure_allocation(p);
    }

    // M12 clock distribution: elect the pipeline clock from any element that
    // offers one (live source > provider > system fallback) and read its epoch.
    let elected = elect_clock([
        source.provide_clock(),
        AsyncElement::provide_clock(transform),
        AsyncElement::provide_clock(sink),
    ]);
    let (clock_priority, base_time_ns) = match &elected {
        Some(c) => (c.priority, c.clock.now_ns()),
        None => (ClockPriority::SystemFallback, clock.now_ns()),
    };

    let (link1_tx, link1_rx) = link(link_capacity);
    let (link2_tx, link2_rx) = link(link_capacity);

    // M18 β scaffolding: a single coordinator task observes the
    // control channel alongside the data-plane arms. The sink arm is the
    // sole reporter for now (DESIGN-M16-workaround3-reconfigure.md §9.4
    // R3: out-of-band channel, not in-band packets). No reconfiguration
    // logic lives here yet — this validates the channel topology before
    // Session E moves the real `Recascade` cascade onto it. The handle is
    // moved into the sink arm; when that arm finishes, the last handle
    // drops, the channel closes, and the coordinator task resolves.
    let (coord, coord_handle) = coordinator(link_capacity);

    let source_fut = async move {
        let mut adapter = SenderSink::new(link1_tx);
        source.run(&mut adapter).await
    };

    let transform_fut = async move {
        let mut adapter = SenderSink::new(link2_tx);
        loop {
            match link1_rx.recv().await {
                Some(PipelinePacket::Eos) => {
                    transform.process(PipelinePacket::Eos, &mut adapter).await?;
                    adapter.push(PipelinePacket::Eos).await?;
                    return Ok::<(), G2gError>(());
                }
                Some(PipelinePacket::CapsChanged(new_caps)) => {
                    match transform.configure_pipeline(&new_caps)? {
                        ConfigureOutcome::Accepted => {
                            // M18 α: element-local re-allocation under the
                            // new caps before forwarding the notification.
                            realloc_local(transform, &new_caps);
                            transform
                                .process(
                                    PipelinePacket::CapsChanged(new_caps),
                                    &mut adapter,
                                )
                                .await?;
                        }
                        // Mid-stream ReFixate: fire upstream via this
                        // element's input link, drop the rejected
                        // CapsChanged. Piece 4 will source-side react.
                        ConfigureOutcome::ReFixate(counter) => {
                            link1_rx.request_reconfigure(Reconfigure::Propose(counter));
                        }
                    }
                }
                Some(packet) => {
                    transform.process(packet, &mut adapter).await?;
                }
                None => return Ok(()),
            }
        }
    };

    let sink_fut = async move {
        let coord_handle = coord_handle;
        let mut null = NullSink;
        let mut consumed: u64 = 0;
        loop {
            match link2_rx.recv().await {
                Some(PipelinePacket::Eos) => {
                    sink.process(PipelinePacket::Eos, &mut null).await?;
                    return Ok::<u64, G2gError>(consumed);
                }
                Some(PipelinePacket::CapsChanged(new_caps)) => {
                    // M16 workaround #3 Phase B: re-solve the downstream
                    // subgraph (boundary → sink) before applying. The
                    // boundary that emitted this `CapsChanged` is the
                    // transform; the subgraph is the one link feeding
                    // the sink. A `NegotiationFailure` here means the
                    // sink's declared `CapsConstraint` rejects the
                    // boundary's output — surface it as a reverse
                    // Reconfigure into the transform (§7 forward ×
                    // reverse race: terminates at the boundary, does
                    // not propagate past the source).
                    let sink_caps = match re_solve_downstream_sink(&new_caps, &*sink) {
                        Ok(caps) => caps,
                        Err(_) => {
                            link2_rx
                                .request_reconfigure(Reconfigure::Renegotiate);
                            continue;
                        }
                    };
                    match sink.configure_pipeline(&sink_caps)? {
                        ConfigureOutcome::Accepted => {
                            // M18 α: element-local re-allocation under the
                            // new caps before the sink sees the packet.
                            realloc_local(sink, &sink_caps);
                            // M18 β: report the applied mid-stream caps
                            // change to the coordinator before forwarding
                            // it into the sink. Observe-only today.
                            coord_handle
                                .report(CoordinatorEvent::CapsChanged(sink_caps.clone()))
                                .await;
                            sink.process(
                                PipelinePacket::CapsChanged(sink_caps),
                                &mut null,
                            )
                            .await?;
                        }
                        ConfigureOutcome::ReFixate(counter) => {
                            link2_rx.request_reconfigure(Reconfigure::Propose(counter));
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
    };

    // The coordinator task drains the control channel until the sink arm
    // drops its handle. Joined as a fourth arm so it runs concurrently.
    let coordinator_fut = coord.run();

    let (src_res, (tx_res, (snk_res, coordinator_events))) = Join2::new(
        source_fut,
        Join2::new(transform_fut, Join2::new(sink_fut, coordinator_fut)),
    )
    .await;
    let emitted = src_res?;
    tx_res?;
    let consumed = snk_res?;

    Ok(RunStats {
        frames_emitted: emitted,
        frames_consumed: consumed,
        latency,
        allocation,
        clock_priority,
        base_time_ns,
        coordinator_events,
    })
}
