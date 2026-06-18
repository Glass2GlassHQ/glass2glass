use core::future::Future;

use alloc::boxed::Box;

use crate::bus::BusHandle;
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
    coordinator_with_recascade, negotiate_source_transform_sink, realloc_local,
    report_nego_failure, ArmDirective, CoordinatorEvent, MAX_FIXATION_ATTEMPTS,
};
#[cfg(feature = "std")]
use crate::runtime::coordinator::realloc_local_dyn;
use crate::runtime::join::{select2, Either, Join2};
use crate::runtime::solver::{
    resolve_forward_output, solve_linear, ForwardResolve, NegotiationFailure,
};
use crate::runtime::state::{Flow, StateController};

#[cfg(feature = "std")]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use crate::element::DynAsyncElement;
#[cfg(feature = "std")]
use crate::fanout::{MultiOutputElement, MultiOutputSink, MultiSenderSink};
#[cfg(feature = "std")]
use crate::graph::Graph;
#[cfg(feature = "std")]
use crate::runtime::graph_runner::{run_graph_inner, GraphNodeRef};
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

    /// Future returned by [`intercept_caps`]. Async so a source can perform
    /// I/O during negotiation (e.g. RTSP DESCRIBE + SDP parse, hardware
    /// capability probe). Sources that produce caps without I/O can return
    /// [`core::future::Ready`] and the runner pays no cost.
    type CapsFuture<'a>: Future<Output = Result<Caps, G2gError>> + 'a
    where
        Self: 'a;

    /// Negotiation-time caps query. Awaited by the runner during startup
    /// (and on re-fixate retries). `&mut self` because real implementations
    /// (e.g. `RtspSrc`) open a session here and stash the connected state
    /// for `run` to resume from. Synchronous sources just return
    /// `core::future::ready(Ok(caps))`.
    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a>;

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
    /// Default: eagerly await `intercept_caps()` and wrap as a
    /// `LegacySource(Caps)` for the solver. Migrated sources override
    /// to return `Produces(CapsSet)` (or another native variant) and
    /// the chain takes the native arc-consistency path when every
    /// other element is also native.
    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        async move { Ok(CapsConstraint::LegacySource(self.intercept_caps().await?)) }
    }
}

/// Per-link queue depth handed to a runner. Each forward link between
/// elements holds this many in-flight packets before backpressure
/// kicks in; the steady-state glass-to-glass latency floor under
/// backpressure is roughly `2 * link_capacity * consumer_period`.
///
/// Construct via a [`LatencyProfile`] for intent-based selection
/// (`LatencyProfile::Live` for camera-to-display pipelines,
/// `LatencyProfile::Throughput` for batch jobs) or via `From<usize>`
/// for fine-grained tuning. Both forms compose through the runner's
/// `impl Into<LinkCapacity>` parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinkCapacity(usize);

impl LinkCapacity {
    /// `n` clamped to at least 1: a zero-capacity link would deadlock
    /// the producer on its first push.
    pub fn new(n: usize) -> Self {
        Self(n.max(1))
    }

    /// Underlying queue depth. Used by the runner internals; callers
    /// typically pass the `LinkCapacity` (or a `LatencyProfile`)
    /// directly into the runner instead of unpacking.
    pub fn get(self) -> usize {
        self.0
    }
}

impl From<usize> for LinkCapacity {
    fn from(n: usize) -> Self {
        Self::new(n)
    }
}

/// Intent-based selector for [`LinkCapacity`]. Picks the link queue
/// depth from the workload's latency-vs-throughput tradeoff so callers
/// don't have to remember the steady-state floor formula
/// (`2 * cap * consumer_period`).
///
/// At 60 fps:
/// - `Live` (cap=2) -> ~67 ms floor. Right for RTSP -> decode -> display.
/// - `Throughput` (cap=8) -> ~267 ms floor. Right for file ingest /
///   batch where smoothing jitter matters more than time-to-glass.
/// - `Custom(n)` -> caller picks. Useful when a profile bisection or
///   live-edge tuning needs a specific value (a smoke test setting
///   `cap=1` to push the floor below one frame, for example).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatencyProfile {
    /// `link_capacity = 2`. Live camera -> display.
    Live,
    /// `link_capacity = 8`. Batch / throughput.
    Throughput,
    /// Caller-specified depth.
    Custom(usize),
}

impl LatencyProfile {
    /// The `LinkCapacity` this profile maps to.
    pub fn link_capacity(self) -> LinkCapacity {
        match self {
            Self::Live => LinkCapacity::new(2),
            Self::Throughput => LinkCapacity::new(8),
            Self::Custom(n) => LinkCapacity::new(n),
        }
    }
}

impl From<LatencyProfile> for LinkCapacity {
    fn from(p: LatencyProfile) -> Self {
        p.link_capacity()
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
    re_solve_against_sink_constraint(new_caps, &sink.caps_constraint_as_sink())
}

/// Shared core of the downstream re-solve: solve `LegacySource(new_caps)`
/// against the sink's already-evaluated constraint. Factored out so the
/// generic ([`re_solve_downstream_sink`]) and `Box`-erased
/// ([`re_solve_downstream_dyn_sink`]) callers don't duplicate the
/// `Unfixable`-is-not-a-failure handling.
fn re_solve_against_sink_constraint(
    new_caps: &Caps,
    sink_c: &CapsConstraint<'_>,
) -> Result<Caps, NegotiationFailure> {
    let src_c = CapsConstraint::LegacySource(new_caps.clone());
    match solve_linear(&[&src_c, sink_c]) {
        Ok(links) => links.into_iter().last().ok_or(NegotiationFailure::Degenerate),
        Err(NegotiationFailure::Unfixable { .. }) => Ok(new_caps.clone()),
        Err(other) => Err(other),
    }
}

/// Fan-out Phase C FO-2: the [`DynAsyncElement`] counterpart of
/// [`re_solve_downstream_sink`], for `Box`-erased branch sinks.
#[cfg(feature = "std")]
pub(crate) fn re_solve_downstream_dyn_sink(
    new_caps: &Caps,
    sink: &dyn DynAsyncElement,
) -> Result<Caps, NegotiationFailure> {
    re_solve_against_sink_constraint(new_caps, &sink.caps_constraint_as_sink())
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
    link_capacity: impl Into<LinkCapacity>,
) -> Result<RunStats, G2gError>
where
    Src: SourceLoop,
    Snk: AsyncElement,
    Clk: PipelineClock,
{
    run_simple_pipeline_inner(source, sink, clock, link_capacity, None, None).await
}

/// As [`run_simple_pipeline`], but driven by a [`StateController`] (M76).
///
/// The sink arm gates on the controller: while the state is below
/// `Playing` the sink stops pulling, the bounded link fills, and backpressure
/// stalls the source. `set_state(Playing)` (from another task) opens the gate
/// and data flows; `set_state(Null)` stops the sink arm and ends the run.
/// Negotiation still runs eagerly at startup (it is resource acquisition, the
/// `READY` step); only data flow is gated. Pass the controller starting in
/// `Paused` for the common "build prerolled, then play" shape.
pub async fn run_simple_pipeline_stateful<Src, Snk, Clk>(
    source: &mut Src,
    sink: &mut Snk,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    state: &StateController,
) -> Result<RunStats, G2gError>
where
    Src: SourceLoop,
    Snk: AsyncElement,
    Clk: PipelineClock,
{
    run_simple_pipeline_inner(source, sink, clock, link_capacity, None, Some(state.clone())).await
}

/// As [`run_simple_pipeline`], but posts a structured
/// [`BusMessage::NegotiationFailed`](crate::BusMessage::NegotiationFailed) to
/// `bus` on a startup or mid-stream negotiation failure (M18 item 7).
pub async fn run_simple_pipeline_with_bus<Src, Snk, Clk>(
    source: &mut Src,
    sink: &mut Snk,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    bus: &BusHandle,
) -> Result<RunStats, G2gError>
where
    Src: SourceLoop,
    Snk: AsyncElement,
    Clk: PipelineClock,
{
    run_simple_pipeline_inner(source, sink, clock, link_capacity, Some(bus), None).await
}

async fn run_simple_pipeline_inner<Src, Snk, Clk>(
    source: &mut Src,
    sink: &mut Snk,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    bus: Option<&BusHandle>,
    state: Option<StateController>,
) -> Result<RunStats, G2gError>
where
    Src: SourceLoop,
    Snk: AsyncElement,
    Clk: PipelineClock,
{
    let link_capacity: usize = link_capacity.into().get();
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
        // Resolve src_c in its own scope so its borrow of `source`
        // releases before `configure_pipeline(&fixated)` below.
        let fixated = {
            let src_c = match &refix_counter {
                Some(c) => CapsConstraint::LegacySource(c.clone()),
                None => source.caps_constraint().await?,
            };
            let sink_c = sink.caps_constraint_as_sink();
            let links = solve_linear(&[&src_c, &sink_c]).map_err(|f| {
                report_nego_failure(bus, f);
                G2gError::CapsMismatch
            })?;
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

    let bus_for_sink = bus.cloned();
    let state_for_sink = state;
    let sink_fut = async move {
        let bus_for_sink = bus_for_sink;
        let state_for_sink = state_for_sink;
        let mut null = NullSink;
        let mut consumed: u64 = 0;
        let mut prerolled_self = false;
        loop {
            // Flow gate (M76/M77): below `Playing` the sink parks here, so it
            // stops draining the link; the bounded channel fills and
            // backpressure stalls the source. `Playing` opens the gate; `Null`
            // ends the arm. In non-live `Paused` the gate admits exactly one
            // buffer (this sink's preroll frame) before it holds.
            if let Some(sc) = &state_for_sink {
                if sc.flow_gate(prerolled_self).await == Flow::Stop {
                    return Ok::<u64, G2gError>(consumed);
                }
            }
            match link_rx.recv().await {
                Some(PipelinePacket::Eos) => {
                    sink.process(PipelinePacket::Eos, &mut null).await?;
                    // M77: EOS during preroll still completes the async
                    // `Paused` transition (idempotent; no-op once playing).
                    if let Some(sc) = &state_for_sink {
                        sc.notify_prerolled();
                    }
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
                        Err(failure) => {
                            report_nego_failure(bus_for_sink.as_ref(), failure);
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
                    let is_buffer = matches!(packet, PipelinePacket::DataFrame(_));
                    if is_buffer {
                        consumed += 1;
                    }
                    sink.process(packet, &mut null).await?;
                    // M77: the first buffer in non-live `Paused` is the preroll
                    // frame; mark this arm prerolled so the gate flips from
                    // preroll-grant to hold, and report it for aggregation.
                    // Idempotent and a no-op while `Playing`.
                    if is_buffer && !prerolled_self {
                        prerolled_self = true;
                        if let Some(sc) = &state_for_sink {
                            sc.notify_prerolled();
                        }
                    }
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
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
) -> Result<RunStats, G2gError>
where
    Src: SourceLoop,
    Tx: MultiOutputElement,
    Clk: PipelineClock,
{
    run_source_fanout_inner(source, fanout, sinks, clock, link_capacity, None).await
}

/// As [`run_source_fanout`], but posts a structured
/// [`BusMessage::NegotiationFailed`](crate::BusMessage::NegotiationFailed) to
/// `bus` on a startup or per-branch mid-stream negotiation failure (item 7).
#[cfg(feature = "std")]
pub async fn run_source_fanout_with_bus<Src, Tx, Clk>(
    source: &mut Src,
    fanout: &mut Tx,
    sinks: Vec<&mut dyn DynAsyncElement>,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    bus: &BusHandle,
) -> Result<RunStats, G2gError>
where
    Src: SourceLoop,
    Tx: MultiOutputElement,
    Clk: PipelineClock,
{
    run_source_fanout_inner(source, fanout, sinks, clock, link_capacity, Some(bus)).await
}

#[cfg(feature = "std")]
async fn run_source_fanout_inner<Src, Tx, Clk>(
    source: &mut Src,
    fanout: &mut Tx,
    sinks: Vec<&mut dyn DynAsyncElement>,
    _clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    bus: Option<&BusHandle>,
) -> Result<RunStats, G2gError>
where
    Src: SourceLoop,
    Tx: MultiOutputElement,
    Clk: PipelineClock,
{
    let link_capacity: usize = link_capacity.into().get();
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
        let src_c = source.caps_constraint().await?;
        let fanout_c = fanout.caps_constraint_as_input();
        let links = solve_linear(&[&src_c, &fanout_c]).map_err(|f| {
            report_nego_failure(bus, f);
            G2gError::CapsMismatch
        })?;
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
        let bus_for_branch = bus.cloned();
        let sink_fut: BoxFuture<'_, Result<u64, G2gError>> = Box::pin(async move {
            let bus_for_branch = bus_for_branch;
            let mut null = NullSink;
            let mut consumed: u64 = 0;
            loop {
                match rx.recv().await {
                    Some(PipelinePacket::Eos) => {
                        sink.process(PipelinePacket::Eos, &mut null).await?;
                        return Ok::<u64, G2gError>(consumed);
                    }
                    Some(PipelinePacket::CapsChanged(new_caps)) => {
                        // M18 Phase C FO-2: per-branch downstream re-solve
                        // (Phase B applied per branch). Each branch runs in
                        // its own arm, so the broadcast `CapsChanged` is
                        // re-solved on every branch concurrently. FO-1
                        // strict default: a branch whose declared
                        // `caps_constraint_as_sink()` rejects the new caps
                        // fails the fan-out loud (matches GStreamer's
                        // `tee`-with-rejecting-downstream). `AllowBranchDrop`
                        // graceful degradation is a future opt-in.
                        let branch_caps = re_solve_downstream_dyn_sink(&new_caps, &*sink)
                            .map_err(|f| {
                                report_nego_failure(bus_for_branch.as_ref(), f);
                                G2gError::CapsMismatch
                            })?;
                        match sink.configure_pipeline(&branch_caps)? {
                            ConfigureOutcome::Accepted => {
                                // M18 α: element-local re-allocation of this
                                // branch under its re-solved caps.
                                realloc_local_dyn(sink, &branch_caps);
                                sink.process(
                                    PipelinePacket::CapsChanged(branch_caps),
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

/// Drives an arbitrary-length linear pipeline:
/// `source -> transforms[0] -> ... -> transforms[N-1] -> sink`.
///
/// M18 item 4. Generalizes [`run_source_transform_sink`] (one transform) and
/// [`run_simple_pipeline`] (zero) past their fixed arity, lifting the
/// "runner caps at 3 elements" limit so chains like
/// `decoder -> capsfilter -> converter -> sink` are expressible. Interior
/// elements are `&mut dyn DynAsyncElement` (heterogeneous, std-only, the same
/// erasure the fan-out runner uses); source and sink stay statically typed.
///
/// Negotiation runs the solver over all `N + 2` constraints at once and
/// configures each element with its input-side caps (the source with link 0).
/// Data flows over `N + 1` bounded links across `N + 2` concurrently-joined
/// arms. On a mid-stream `CapsChanged` each interior element re-fixates its
/// output against a downstream feasibility snapshot (Caps-α), re-allocates its
/// own pool (α), and the β allocation re-cascade walks the demand back through
/// every interior hop. Clock election and latency aggregation fold the source,
/// every interior element, and the sink (via the dyn-safe `DynAsyncElement`
/// mirrors).
///
/// Owed: Caps-β, a forward coordinator re-solve walk for a downstream
/// `DerivedOutput` element that must re-derive mid-stream (driver-gated,
/// DESIGN-M18-caps-resolve.md §3). ReFixate at startup fails loud
/// (`FixationFailed`), as in `run_source_fanout`.
#[cfg(feature = "std")]
pub async fn run_linear_chain<Src, Snk, Clk>(
    source: &mut Src,
    transforms: Vec<&mut dyn DynAsyncElement>,
    sink: &mut Snk,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
) -> Result<RunStats, G2gError>
where
    Src: SourceLoop,
    Snk: AsyncElement,
    Clk: PipelineClock,
{
    run_linear_chain_inner(source, transforms, sink, clock, link_capacity, None).await
}

/// As [`run_linear_chain`], but posts a structured
/// [`BusMessage::NegotiationFailed`](crate::BusMessage::NegotiationFailed) to
/// `bus` on a startup or mid-stream negotiation failure (M18 item 7).
#[cfg(feature = "std")]
pub async fn run_linear_chain_with_bus<Src, Snk, Clk>(
    source: &mut Src,
    transforms: Vec<&mut dyn DynAsyncElement>,
    sink: &mut Snk,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    bus: &BusHandle,
) -> Result<RunStats, G2gError>
where
    Src: SourceLoop,
    Snk: AsyncElement,
    Clk: PipelineClock,
{
    run_linear_chain_inner(source, transforms, sink, clock, link_capacity, Some(bus)).await
}

#[cfg(feature = "std")]
async fn run_linear_chain_inner<Src, Snk, Clk>(
    source: &mut Src,
    transforms: Vec<&mut dyn DynAsyncElement>,
    sink: &mut Snk,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    bus: Option<&BusHandle>,
) -> Result<RunStats, G2gError>
where
    Src: SourceLoop,
    Snk: AsyncElement,
    Clk: PipelineClock,
{
    // D5: thin builder over the DAG runner. A linear chain maps onto a
    // source -> transform* -> sink path; `run_graph` owns negotiation, the M12
    // stat folds, the β allocation re-cascade, and the Caps-α mid-stream
    // re-solve (graceful on this single-producer chain: no tee upstream).
    let mut g: Graph<GraphNodeRef<'_>> = Graph::new();
    let mut prev = g.add_source(GraphNodeRef::source_ref(source));
    for t in transforms {
        let node = g.add_transform(GraphNodeRef::element_ref(t));
        g.link(prev, node).map_err(|_| G2gError::CapsMismatch)?;
        prev = node;
    }
    let snk = g.add_sink(GraphNodeRef::element_ref(sink));
    g.link(prev, snk).map_err(|_| G2gError::CapsMismatch)?;

    run_graph_inner(g, clock, link_capacity, bus, None).await
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
    link_capacity: impl Into<LinkCapacity>,
) -> Result<RunStats, G2gError>
where
    Src: SourceLoop,
    Tx: AsyncElement,
    Snk: AsyncElement,
    Clk: PipelineClock,
{
    run_source_transform_sink_inner(source, transform, sink, clock, link_capacity, None).await
}

/// As [`run_source_transform_sink`], but posts a structured
/// [`BusMessage::NegotiationFailed`](crate::BusMessage::NegotiationFailed)
/// to `bus` so the application learns *which* link conflicted (the returned
/// error stays the opaque `CapsMismatch`). M18 item 7. Covers both startup
/// negotiation and the mid-stream re-solve sites (the sink's Phase-B re-solve
/// and the transform's Caps-α `Infeasible`). The bus is opt-in to keep the
/// common call site unchanged.
pub async fn run_source_transform_sink_with_bus<Src, Tx, Snk, Clk>(
    source: &mut Src,
    transform: &mut Tx,
    sink: &mut Snk,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    bus: &BusHandle,
) -> Result<RunStats, G2gError>
where
    Src: SourceLoop,
    Tx: AsyncElement,
    Snk: AsyncElement,
    Clk: PipelineClock,
{
    run_source_transform_sink_inner(source, transform, sink, clock, link_capacity, Some(bus)).await
}

async fn run_source_transform_sink_inner<Src, Tx, Snk, Clk>(
    source: &mut Src,
    transform: &mut Tx,
    sink: &mut Snk,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    bus: Option<&BusHandle>,
) -> Result<RunStats, G2gError>
where
    Src: SourceLoop,
    Tx: AsyncElement,
    Snk: AsyncElement,
    Clk: PipelineClock,
{
    let link_capacity: usize = link_capacity.into().get();
    // M18 Session C: the startup negotiation loop (solver + per-link
    // configure cascade with bounded `ReFixate` retry) is owned by the
    // coordinator module now, since β reuses the same machinery for the
    // mid-stream re-cascade. `sink_link` is the downstream-facing caps
    // (transform output = sink input) that M12 allocation flows along,
    // so it stands in for the loop's former `negotiated_caps`.
    // M12 allocation query now runs *inside* negotiation, before the
    // `configure_pipeline` cascade, so a transform (e.g. a hardware decoder)
    // sizes its buffer pool from the downstream `min_buffers` at open time.
    // The folded source-facing proposal comes back on `RunStats`.
    let negotiation = negotiate_source_transform_sink(source, transform, sink, bus).await?;
    let allocation = negotiation.allocation;

    // Caps-α: the transform's downstream subgraph is the single sink link, so
    // its feasibility snapshot is just the sink's accept set (the N-hop sweep
    // in `run_linear_chain` reduces to this for one transform). A wildcard or
    // legacy sink leaves it unconstrained, so the transform keeps forwarding
    // greedily (Defer).
    let downstream_feasible = match sink.caps_constraint_as_sink() {
        CapsConstraint::Accepts(s) => Some(s.clone()),
        _ => None,
    };

    // M12 latency query: fold the configured chain source → transform → sink.
    let latency = LatencyReport::aggregate([
        source.latency(),
        AsyncElement::latency(transform),
        AsyncElement::latency(sink),
    ]);

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

    // M18 β: a single coordinator task owns the cross-element re-cascade.
    // The sink arm reports an applied mid-stream `CapsChanged` (with its
    // re-derived allocation proposal) out-of-band
    // (DESIGN-M16-workaround3-reconfigure.md §9.4 R3); the coordinator
    // forwards the proposal one hop upstream over `transform_ctrl_rx` to the
    // transform's `configure_allocation`. The transform arm selects on that
    // control receiver alongside its data link, so the directive reaches it
    // even while it is parked on `recv().await`. When the sink arm finishes,
    // the handle drops, the coordinator drains and closes `transform_ctrl_rx`,
    // and the transform arm's EOS-drain unblocks.
    let (coord, coord_handle, transform_ctrl_rx) = coordinator_with_recascade(link_capacity);

    let source_fut = async move {
        let mut adapter = SenderSink::new(link1_tx);
        source.run(&mut adapter).await
    };

    let bus_for_transform = bus.cloned();
    let transform_fut = async move {
        let ctrl_rx = transform_ctrl_rx;
        let mut adapter = SenderSink::new(link2_tx);
        // β: while the coordinator is alive, race the data link against the
        // re-cascade control channel so a directive is applied promptly. Once
        // control closes (coordinator gone) we degrade to data-only so the
        // closed arm can't spin.
        let mut control_open = true;
        loop {
            let packet = if control_open {
                match select2(ctrl_rx.recv(), link1_rx.recv()).await {
                    Either::Left(Some(ArmDirective::Recascade(params))) => {
                        // β: apply the sink's downstream-derived proposal to
                        // our own output pool, then keep waiting for data.
                        transform.configure_allocation(&params);
                        continue;
                    }
                    Either::Left(None) => {
                        control_open = false;
                        continue;
                    }
                    Either::Right(packet) => packet,
                }
            } else {
                link1_rx.recv().await
            };
            match packet {
                Some(PipelinePacket::Eos) => {
                    transform.process(PipelinePacket::Eos, &mut adapter).await?;
                    adapter.push(PipelinePacket::Eos).await?;
                    // β: the EOS we just forwarded will, once the sink applies
                    // its final `CapsChanged` and the coordinator forwards the
                    // matching re-cascade, close this control channel. Drain it
                    // first so a tail-end proposal is applied before we exit
                    // (in a live stream these apply inline above; this only
                    // covers the directive still in flight at shutdown).
                    while control_open {
                        match ctrl_rx.recv().await {
                            Some(ArmDirective::Recascade(params)) => {
                                transform.configure_allocation(&params);
                            }
                            None => control_open = false,
                        }
                    }
                    return Ok::<(), G2gError>(());
                }
                Some(PipelinePacket::CapsChanged(new_caps)) => {
                    // Caps-α (D3): derive the forwarded output from the
                    // transform's constraint, steered by the sink's accept set,
                    // instead of forwarding greedily. Mirrors `run_linear_chain`;
                    // here the downstream subgraph is the single sink link.
                    // `Infeasible` means the sink positively rejects every
                    // output the transform can produce: surface it loud as a
                    // reverse reconfigure into this boundary.
                    let forward_caps = {
                        let constraint = transform.caps_constraint_as_transform();
                        match resolve_forward_output(
                            &constraint,
                            &new_caps,
                            downstream_feasible.as_ref(),
                        ) {
                            ForwardResolve::Fixed(caps) => caps,
                            ForwardResolve::Defer => new_caps.clone(),
                            ForwardResolve::Infeasible(failure) => {
                                report_nego_failure(bus_for_transform.as_ref(), failure);
                                link1_rx.request_reconfigure(Reconfigure::Renegotiate);
                                continue;
                            }
                        }
                    };
                    match transform.configure_pipeline(&new_caps)? {
                        ConfigureOutcome::Accepted => {
                            // M18 α: element-local re-allocation under the
                            // re-fixated output caps before forwarding.
                            realloc_local(transform, &forward_caps);
                            transform
                                .process(
                                    PipelinePacket::CapsChanged(forward_caps),
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

    let bus_for_sink = bus.cloned();
    let sink_fut = async move {
        let coord_handle = coord_handle;
        let bus_for_sink = bus_for_sink;
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
                        Err(failure) => {
                            // M18 item 7: the mid-stream re-solve is the case
                            // the bus matters most for, there is no synchronous
                            // return to carry the detail. Post the structured
                            // failure, then drive the reverse Reconfigure.
                            report_nego_failure(bus_for_sink.as_ref(), failure);
                            link2_rx
                                .request_reconfigure(Reconfigure::Renegotiate);
                            continue;
                        }
                    };
                    match sink.configure_pipeline(&sink_caps)? {
                        ConfigureOutcome::Accepted => {
                            // M18 α: element-local re-allocation under the
                            // new caps before the sink sees the packet. The
                            // returned proposal is what the sink now wants
                            // its upstream to allocate.
                            let proposal = realloc_local(sink, &sink_caps);
                            // M18 β: report the applied caps change plus the
                            // sink's re-derived proposal so the coordinator
                            // forwards it one hop upstream to the transform's
                            // `configure_allocation` (the single-hop cascade).
                            coord_handle
                                .report(CoordinatorEvent::CapsChanged {
                                    caps: sink_caps.clone(),
                                    proposal,
                                })
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

#[cfg(test)]
mod profile_tests {
    use super::*;

    #[test]
    fn live_profile_maps_to_capacity_2() {
        assert_eq!(LatencyProfile::Live.link_capacity().get(), 2);
    }

    #[test]
    fn throughput_profile_maps_to_capacity_8() {
        assert_eq!(LatencyProfile::Throughput.link_capacity().get(), 8);
    }

    #[test]
    fn custom_profile_passes_through() {
        assert_eq!(LatencyProfile::Custom(16).link_capacity().get(), 16);
    }

    #[test]
    fn link_capacity_clamps_zero_to_one() {
        // A zero-depth link would deadlock the producer on its first push;
        // the constructor clamps so callers passing `0` (or a misconfigured
        // env var) get a runnable pipeline rather than a hang.
        assert_eq!(LinkCapacity::new(0).get(), 1);
        assert_eq!(LinkCapacity::from(0usize).get(), 1);
        assert_eq!(LatencyProfile::Custom(0).link_capacity().get(), 1);
    }

    #[test]
    fn from_usize_and_from_profile_compose_through_into() {
        // The runner takes `impl Into<LinkCapacity>`; both an integer and
        // a profile must reach the same internal usize without ceremony.
        fn take<C: Into<LinkCapacity>>(c: C) -> usize {
            c.into().get()
        }
        assert_eq!(take(4usize), 4);
        assert_eq!(take(LatencyProfile::Live), 2);
        assert_eq!(take(LatencyProfile::Throughput), 8);
    }
}
