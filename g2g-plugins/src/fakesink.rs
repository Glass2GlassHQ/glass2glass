//! Inspecting sink. Counts frames, records the last sequence number,
//! tracks EOS, and records mid-stream `CapsChanged` events alongside
//! the frame count at which they arrived. Used by tests and to
//! validate runner plumbing.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::metrics::{LatencyHistogram, LatencySnapshot};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, G2gError, HardwareError, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket,
};

#[cfg(feature = "std")]
use g2g_core::metrics::monotonic_ns;

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
    /// Glass-to-glass latency distribution recorded as
    /// `monotonic_ns() - arrival_ns` on every received `DataFrame`
    /// whose `FrameTiming::arrival_ns` is non-zero. Frames without a
    /// source-side stamp (synthesized by transforms, or sources that
    /// don't stamp) are skipped silently. Test code can pull a
    /// snapshot via [`FakeSink::latency_snapshot`] and assert bounds.
    latency: LatencyHistogram,
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

    /// Snapshot of the glass-to-glass latency histogram. Count of zero
    /// means no `arrival_ns`-stamped frames have been received (either
    /// the source doesn't stamp, or no frames have arrived yet).
    pub fn latency_snapshot(&self) -> LatencySnapshot {
        self.latency.snapshot()
    }
}

impl AsyncElement for FakeSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    /// M16 step 5c: declare wildcard sink to the solver. `FakeSink`
    /// accepts whatever upstream produces (its `intercept_caps` is a
    /// pass-through), which is exactly what `CapsConstraint::AcceptsAny`
    /// models. Skips the dynamic intercept callback and lets the
    /// solver propagate upstream caps unchanged.
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
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
                    // Record glass-to-glass latency if the source stamped
                    // arrival_ns. Sub-monotonic stamps (zero, or future
                    // values) are skipped silently — they're either
                    // unstamped frames or a clock-domain mismatch the
                    // sink shouldn't paper over.
                    #[cfg(feature = "std")]
                    {
                        let arrival = f.timing.arrival_ns;
                        if arrival != 0 {
                            let now = monotonic_ns();
                            if now >= arrival {
                                self.latency.record(now - arrival);
                            }
                        }
                    }
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

impl PadTemplates for FakeSink {
    /// Wildcard sink: accepts any caps, matching the runtime
    /// `caps_constraint_as_sink` of `AcceptsAny`.
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink_any()])
    }
}
