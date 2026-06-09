//! Presentation sink: waits until each frame's PTS arrives on the pipeline
//! clock before reporting the frame "presented". Records per-frame drift
//! (current clock minus PTS at presentation time) for end-to-end latency
//! analysis.
//!
//! Upstream backpressure naturally paces a free-running source: the source
//! can't push faster than the sink consumes, and the sink consumes no faster
//! than the clock advances toward each frame's PTS.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;

use g2g_core::{
    AsyncClock, AsyncElement, Caps, ConfigureOutcome, ElementBound, G2gError, OutputSink,
    PipelinePacket,
};

#[derive(Debug)]
pub struct SyncSink<C: AsyncClock> {
    clock: C,
    received: u64,
    last_sequence: Option<u64>,
    eos_seen: bool,
    configured: bool,
    max_drift_ns: u64,
    total_drift_ns: u128,
}

impl<C: AsyncClock> SyncSink<C> {
    pub fn new(clock: C) -> Self {
        Self {
            clock,
            received: 0,
            last_sequence: None,
            eos_seen: false,
            configured: false,
            max_drift_ns: 0,
            total_drift_ns: 0,
        }
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

    /// Largest single-frame drift observed: `clock.now_ns() - frame.pts_ns`
    /// at presentation time. Always non-negative because the sink sleeps
    /// until the deadline has passed.
    pub fn max_drift_ns(&self) -> u64 {
        self.max_drift_ns
    }

    pub fn mean_drift_ns(&self) -> u64 {
        if self.received == 0 {
            0
        } else {
            (self.total_drift_ns / u128::from(self.received))
                .try_into()
                .unwrap_or(u64::MAX)
        }
    }
}

impl<C> AsyncElement for SyncSink<C>
where
    C: AsyncClock + ElementBound,
{
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
                    self.clock.sleep_until_ns(f.timing.pts_ns).await;
                    let drift = self.clock.now_ns().saturating_sub(f.timing.pts_ns);
                    self.max_drift_ns = self.max_drift_ns.max(drift);
                    self.total_drift_ns =
                        self.total_drift_ns.saturating_add(u128::from(drift));
                    self.last_sequence = Some(f.sequence);
                    self.received += 1;
                }
                PipelinePacket::Eos => {
                    self.eos_seen = true;
                }
                PipelinePacket::Flush => {
                    // Seek flush: drop position so presentation resumes
                    // cleanly at the post-seek timeline.
                    self.last_sequence = None;
                }
                PipelinePacket::CapsChanged(_) => {}
            }
            Ok(())
        })
    }
}
