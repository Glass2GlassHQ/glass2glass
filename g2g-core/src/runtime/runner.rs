use core::future::Future;

use crate::caps::Caps;
use crate::clock::PipelineClock;
use crate::element::{AsyncElement, ConfigureOutcome, ElementBound, OutputSink};
use crate::error::G2gError;
use crate::frame::PipelinePacket;
use crate::runtime::channel::{bounded, SenderSink};
use crate::runtime::join::Join2;

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
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RunStats {
    pub frames_emitted: u64,
    pub frames_consumed: u64,
}

/// M1 runner: drives a single source → sink pipeline over a bounded link.
/// Caps negotiation is a fast Phase 1 + Phase 2; ReFixate is treated as an
/// error. Full graph negotiation lands in M2.
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
    let proposed = source.intercept_caps()?;
    let fixated = sink.intercept_caps(&proposed)?;
    match source.configure_pipeline(&fixated)? {
        ConfigureOutcome::Accepted => {}
        ConfigureOutcome::ReFixate(_) => return Err(G2gError::FixationFailed),
    }
    match sink.configure_pipeline(&fixated)? {
        ConfigureOutcome::Accepted => {}
        ConfigureOutcome::ReFixate(_) => return Err(G2gError::FixationFailed),
    }

    let (tx, rx) = bounded::<PipelinePacket>(link_capacity);

    let source_fut = async move {
        let mut adapter = SenderSink::new(tx);
        let emitted = source.run(&mut adapter).await?;
        Ok::<u64, G2gError>(emitted)
    };

    let sink_fut = async move {
        let mut null = NullSink;
        let mut consumed: u64 = 0;
        loop {
            match rx.recv().await {
                Some(PipelinePacket::Eos) => {
                    sink.process(PipelinePacket::Eos, &mut null).await?;
                    return Ok::<u64, G2gError>(consumed);
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

#[derive(Debug)]
struct NullSink;

impl OutputSink for NullSink {
    fn push(&mut self, _packet: PipelinePacket) -> Result<(), G2gError> {
        Ok(())
    }
}
