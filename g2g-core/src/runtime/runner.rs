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
use crate::runtime::join::Join2;
use crate::runtime::solver::solve_linear;

#[cfg(feature = "std")]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use crate::element::DynAsyncElement;
#[cfg(feature = "std")]
use crate::fanout::{MultiOutputElement, MultiOutputSink, MultiSenderSink};
#[cfg(feature = "std")]
use crate::runtime::join::join_all;

/// Maximum number of Phase 1 + Phase 2 negotiation passes before a setup
/// gives up with `FixationFailed`. Three is enough for any reasonable
/// `ReFixate` chain (source → sink → source counter) while still being
/// a hard backstop against pathologically-counter-proposing elements.
const MAX_FIXATION_ATTEMPTS: u32 = 3;

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
    // M16 step 4c: startup negotiation via `solve_linear` + legacy bridge.
    let mut proposal = source.intercept_caps()?;
    let mut attempts = 0u32;
    let negotiated_caps = loop {
        attempts += 1;
        if attempts > MAX_FIXATION_ATTEMPTS {
            return Err(G2gError::FixationFailed);
        }
        let fixated = {
            let src_c = CapsConstraint::LegacySource(proposal.clone());
            let sink_c = sink.caps_constraint_as_sink();
            let links = solve_linear(&[&src_c, &sink_c])
                .map_err(|_| G2gError::CapsMismatch)?;
            links.last().cloned().ok_or(G2gError::CapsMismatch)?
        };
        match source.configure_pipeline(&fixated)? {
            ConfigureOutcome::Accepted => {}
            ConfigureOutcome::ReFixate(counter) => {
                proposal = counter;
                continue;
            }
        }
        match sink.configure_pipeline(&fixated)? {
            ConfigureOutcome::Accepted => break fixated,
            ConfigureOutcome::ReFixate(counter) => {
                proposal = counter;
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
                    // M8 piece 1: runner cascades mid-stream caps changes
                    // through configure_pipeline before the element sees
                    // the notification packet. Guarantees DataFrames with
                    // the new caps never reach a stale element.
                    match sink.configure_pipeline(&new_caps)? {
                        ConfigureOutcome::Accepted => {
                            sink.process(
                                PipelinePacket::CapsChanged(new_caps),
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

    // M16 step 4c: solve source → fanout via the solver. The fan-out
    // acts as the linear "sink" of the negotiation chain; the real
    // sinks downstream of it broadcast-receive the same fixated caps
    // and don't participate in narrowing.
    let proposal = source.intercept_caps()?;
    let fixated = {
        let src_c = CapsConstraint::LegacySource(proposal);
        let fanout_ref = &*fanout;
        let fanout_c = CapsConstraint::LegacySink(Box::new(move |c: &Caps| {
            MultiOutputElement::intercept_caps(fanout_ref, c)
        }));
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
    // M16 step 4b: startup negotiation routes through `solve_linear`
    // via the legacy bridge. The bridge wraps today's `intercept_caps`
    // / `propose_output_caps` callbacks as `LegacyTransform` /
    // `LegacySink` constraints; the solver's legacy cascade runs the
    // forward chain identically to the pre-M16 inline cascade and
    // fixates the final caps. `ReFixate` retry stays in the runner
    // (the solver doesn't model counter-proposals) — on each retry the
    // source's `LegacySource` seed is replaced by the counter and the
    // solver is re-run.
    let mut start_proposal = source.intercept_caps()?;
    let mut attempts = 0u32;
    let negotiated_caps = loop {
        attempts += 1;
        if attempts > MAX_FIXATION_ATTEMPTS {
            return Err(G2gError::FixationFailed);
        }
        // Build the constraint chain in a scope so the immutable
        // borrows of `transform` / `sink` are released before the
        // `configure_pipeline` calls below take mutable access.
        let (src_caps, sink_caps) = {
            let src_c = CapsConstraint::LegacySource(start_proposal.clone());
            let tx_c = transform.caps_constraint_as_transform();
            let sink_c = sink.caps_constraint_as_sink();
            let links = solve_linear(&[&src_c, &tx_c, &sink_c])
                .map_err(|_| G2gError::CapsMismatch)?;
            // M16 step 5d: per-link configure. For a 3-element chain
            // links has length 2: [source-output / transform-input,
            // transform-output / sink-input]. The transform's
            // `configure_pipeline` historically receives one caps; we
            // pass its *input* side, which is what existing decoders
            // (e.g. `FfmpegH264Dec`) expect.
            if links.len() != 2 {
                return Err(G2gError::CapsMismatch);
            }
            (links[0].clone(), links[1].clone())
        };

        let mut refixate: Option<Caps> = None;
        for outcome in [
            source.configure_pipeline(&src_caps)?,
            transform.configure_pipeline(&src_caps)?,
            sink.configure_pipeline(&sink_caps)?,
        ] {
            match outcome {
                ConfigureOutcome::Accepted => {}
                ConfigureOutcome::ReFixate(counter) => {
                    refixate = Some(counter);
                    break;
                }
            }
        }
        match refixate {
            Some(counter) => start_proposal = counter,
            // M12 allocation flows along the downstream-facing side
            // (the transform's output = the sink's input), so we
            // break with `sink_caps` for the propose_allocation calls
            // below.
            None => break sink_caps,
        }
    };

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
        let mut null = NullSink;
        let mut consumed: u64 = 0;
        loop {
            match link2_rx.recv().await {
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

    let (src_res, (tx_res, snk_res)) =
        Join2::new(source_fut, Join2::new(transform_fut, sink_fut)).await;
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
    })
}
