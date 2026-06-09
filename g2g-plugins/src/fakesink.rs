//! Inspecting sink. Counts frames, records the last sequence number,
//! tracks EOS, and records mid-stream `CapsChanged` events alongside
//! the frame count at which they arrived. Used by tests and to
//! validate runner plumbing.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, G2gError, HardwareError, OutputSink, PipelinePacket,
};

/// Position of a recorded `CapsChanged` packet inside the sink's input
/// stream: `frames_before` is the number of `DataFrame` packets that
/// arrived at the sink strictly before this `CapsChanged`.
#[derive(Debug, Clone)]
pub struct CapsChange {
    pub caps: Caps,
    pub frames_before: u64,
}

#[derive(Debug, Default)]
pub struct FakeSink {
    received: u64,
    last_sequence: Option<u64>,
    eos_seen: bool,
    flushes: u64,
    configured: bool,
    caps_changes: Vec<CapsChange>,
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

    /// Number of `Flush` packets seen. A flush resets `last_sequence` so the
    /// stream may resume with a lower sequence after a seek.
    pub fn flushes(&self) -> u64 {
        self.flushes
    }

    pub fn caps_changes(&self) -> &[CapsChange] {
        &self.caps_changes
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
                PipelinePacket::Flush => {
                    // Seek flush: reset position so a lower sequence is
                    // accepted when the stream resumes.
                    self.flushes += 1;
                    self.last_sequence = None;
                }
                PipelinePacket::CapsChanged(caps) => {
                    self.caps_changes.push(CapsChange {
                        caps,
                        frames_before: self.received,
                    });
                }
            }
            Ok(())
        })
    }
}
