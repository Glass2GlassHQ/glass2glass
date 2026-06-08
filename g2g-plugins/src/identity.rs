//! Identity transform: forwards every `DataFrame` downstream unchanged.
//! Useful for validating transform plumbing and as a base for stat-collecting
//! tee elements.
//!
//! Per the transform contract (see `run_source_transform_sink`), this element
//! does NOT emit `Eos` itself — the runner forwards the EOS sentinel after
//! `process(Eos)` returns.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;

use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, G2gError, OutputSink, PipelinePacket,
};

#[derive(Debug, Default)]
pub struct IdentityTransform {
    forwarded: u64,
    configured: bool,
}

impl IdentityTransform {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn forwarded(&self) -> u64 {
        self.forwarded
    }
}

impl AsyncElement for IdentityTransform {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn configure_pipeline(
        &mut self,
        _absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(f) => {
                    self.forwarded += 1;
                    out.push(PipelinePacket::DataFrame(f)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    out.push(PipelinePacket::CapsChanged(c)).await?;
                }
                PipelinePacket::Eos => {}
            }
            Ok(())
        })
    }
}
