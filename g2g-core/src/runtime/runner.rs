use core::future::Future;

use alloc::boxed::Box;

use crate::caps::Caps;
use crate::clock::PipelineClock;
use crate::element::{
    AsyncElement, BoxFuture, ConfigureOutcome, ElementBound, OutputSink, PushOutcome, Reconfigure,
};
use crate::error::G2gError;
use crate::frame::PipelinePacket;
use crate::runtime::channel::{link, SenderSink};
use crate::runtime::join::Join2;

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
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RunStats {
    pub frames_emitted: u64,
    pub frames_consumed: u64,
}

/// Drives a `source → sink` pipeline over a single bounded link.
/// Initial Phase 1+2 negotiation runs with bounded `ReFixate` backtrack
/// (M8 piece 5): if any element's `configure_pipeline()` returns a
/// counter-proposal, the runner restarts negotiation with that counter
/// as the new starting proposal, up to `MAX_FIXATION_ATTEMPTS` total.
pub async fn run_simple_pipeline<Src, Snk, Clk>(
    source: &mut Src,
    sink: &mut Snk,
    _clock: &Clk,
    link_capacity: usize,
) -> Result<RunStats, G2gError>
where
    Src: SourceLoop,
    Snk: AsyncElement,
    Clk: PipelineClock,
{
    let mut proposal = source.intercept_caps()?;
    let mut attempts = 0u32;
    loop {
        attempts += 1;
        if attempts > MAX_FIXATION_ATTEMPTS {
            return Err(G2gError::FixationFailed);
        }
        // Phase 1 narrows; Phase 2 fixates every ranged field to a single
        // value before any element allocates against it (DESIGN.md §4.2).
        let negotiated = sink.intercept_caps(&proposal)?;
        let fixated = negotiated.fixate()?;
        match source.configure_pipeline(&fixated)? {
            ConfigureOutcome::Accepted => {}
            ConfigureOutcome::ReFixate(counter) => {
                proposal = counter;
                continue;
            }
        }
        match sink.configure_pipeline(&fixated)? {
            ConfigureOutcome::Accepted => break,
            ConfigureOutcome::ReFixate(counter) => {
                proposal = counter;
                continue;
            }
        }
    }

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

    Ok(RunStats { frames_emitted: emitted, frames_consumed: consumed })
}

/// Sentinel sink for terminal elements (sinks proper): swallows pushes.
/// Process implementations of true sinks should not emit, but the type
/// system still requires an `&mut dyn OutputSink` parameter.
#[derive(Debug)]
struct NullSink;

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
pub async fn run_source_transform_sink<Src, Tx, Snk, Clk>(
    source: &mut Src,
    transform: &mut Tx,
    sink: &mut Snk,
    _clock: &Clk,
    link_capacity: usize,
) -> Result<RunStats, G2gError>
where
    Src: SourceLoop,
    Tx: AsyncElement,
    Snk: AsyncElement,
    Clk: PipelineClock,
{
    // M8 piece 5: bounded ReFixate retry across all three elements.
    // Any element's counter-proposal restarts Phase 1 from the source's
    // intercept with that counter as the new starting proposal.
    let mut start_proposal = source.intercept_caps()?;
    let mut attempts = 0u32;
    loop {
        attempts += 1;
        if attempts > MAX_FIXATION_ATTEMPTS {
            return Err(G2gError::FixationFailed);
        }
        // Phase 1 narrows through the chain; Phase 2 fixates the result
        // before any element allocates against it (DESIGN.md §4.2).
        let tx_proposal = transform.intercept_caps(&start_proposal)?;
        let negotiated = sink.intercept_caps(&tx_proposal)?;
        let fixated = negotiated.fixate()?;

        let mut refixate: Option<Caps> = None;
        for outcome in [
            source.configure_pipeline(&fixated)?,
            transform.configure_pipeline(&fixated)?,
            sink.configure_pipeline(&fixated)?,
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
            None => break,
        }
    }

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

    Ok(RunStats { frames_emitted: emitted, frames_consumed: consumed })
}
