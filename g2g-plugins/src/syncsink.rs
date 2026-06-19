//! Presentation sink: waits until each frame's PTS arrives on the pipeline
//! clock before reporting the frame "presented". Records per-frame drift
//! (current clock minus PTS at presentation time) for end-to-end latency
//! analysis.
//!
//! Upstream backpressure naturally paces a free-running source: the source
//! can't push faster than the sink consumes, and the sink consumes no faster
//! than the clock advances toward each frame's PTS.
//!
//! QoS (M85): when given a max-lateness bound, a frame whose deadline is
//! already past by more than that bound is dropped rather than presented late,
//! so the sink catches up instead of compounding the lag. Each drop posts a
//! [`BusMessage::Qos`] to the pipeline bus if one was attached, the GStreamer
//! `GST_MESSAGE_QOS` analog. Default behaviour is unchanged (no bound, no bus):
//! every frame is presented after its deadline.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;

use g2g_core::{
    AsyncClock, AsyncElement, BusHandle, BusMessage, Caps, CapsConstraint, ConfigureOutcome,
    ElementBound, G2gError, OutputSink, PipelinePacket,
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
    /// QoS: drop a frame whose deadline is already past by more than this many
    /// ns. `u64::MAX` (the default) never drops, so the sink presents every
    /// frame however late, preserving the pre-QoS behaviour.
    max_lateness_ns: u64,
    dropped: u64,
    bus: Option<BusHandle>,
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
            max_lateness_ns: u64::MAX,
            dropped: 0,
            bus: None,
        }
    }

    /// Enable QoS dropping: a frame already past its deadline by more than
    /// `ns` is dropped instead of presented late. `0` drops any frame that
    /// arrives after its deadline.
    pub fn with_max_lateness_ns(mut self, ns: u64) -> Self {
        self.max_lateness_ns = ns;
        self
    }

    /// Attach the pipeline bus so QoS drops post a [`BusMessage::Qos`].
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.bus = Some(bus);
        self
    }

    pub fn received(&self) -> u64 {
        self.received
    }

    /// Frames dropped because they arrived too late under the QoS bound.
    pub fn dropped(&self) -> u64 {
        self.dropped
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

    /// M16 step 5c: wildcard sink. Same rationale as `FakeSink`.
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
                    let pts = f.timing.pts_ns;
                    // QoS: a frame already past its deadline by more than the
                    // bound is dropped, not presented late, so the sink catches
                    // up. `now > pts + bound` (saturating, so the u64::MAX
                    // default never fires).
                    let now = self.clock.now_ns();
                    if now > pts.saturating_add(self.max_lateness_ns) {
                        self.dropped += 1;
                        if let Some(bus) = &self.bus {
                            let jitter = i64::try_from(now - pts).unwrap_or(i64::MAX);
                            // Control message: non-blocking, never stalls the
                            // sink (a full bus drops the report).
                            bus.try_post(BusMessage::Qos {
                                running_time_ns: pts,
                                jitter_ns: jitter,
                                processed: self.received,
                                dropped: self.dropped,
                            });
                        }
                        return Ok(());
                    }
                    self.clock.sleep_until_ns(pts).await;
                    let drift = self.clock.now_ns().saturating_sub(pts);
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
                // Segment is control: ignored at sink.
                PipelinePacket::Segment(_) => {}
            }
            Ok(())
        })
    }
}
