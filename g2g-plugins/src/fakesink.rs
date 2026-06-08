//! Inspecting sink. Counts frames, records the last sequence number,
//! and tracks EOS. Used by tests and to validate runner plumbing.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;

use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, G2gError, HardwareError, OutputSink, PipelinePacket,
};

#[derive(Debug, Default)]
pub struct FakeSink {
    received: u64,
    last_sequence: Option<u64>,
    eos_seen: bool,
    configured: bool,
}

impl FakeSink {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn received(&self) -> u64 {
        self.received
    }

    pub fn last_sequence(&self) -> Option<u64> {
        self.last_sequence
    }

    pub fn eos_seen(&self) -> bool {
        self.eos_seen
    }
}

impl AsyncElement for FakeSink {
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
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(f) => {
                    if let Some(prev) = self.last_sequence {
                        if f.sequence <= prev {
                            return Err(G2gError::Hardware(HardwareError::Other));
                        }
                    }
                    self.last_sequence = Some(f.sequence);
                    self.received += 1;
                }
                PipelinePacket::Eos => {
                    self.eos_seen = true;
                }
                PipelinePacket::CapsChanged(_) => {}
            }
            Ok(())
        })
    }
}
