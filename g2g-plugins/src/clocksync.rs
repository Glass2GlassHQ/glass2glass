//! Wall-clock pacing transform (`clocksync`, M945): holds each buffer until its
//! PTS comes due on the pipeline clock, so an upstream that produces as fast as
//! the CPU allows feeds downstream at real time.
//!
//! Fully transparent otherwise: any caps in, the same caps out, every control
//! packet forwarded unchanged. It paces on `pts_ns` alone, so a compressed
//! stream (`videotestsrc ! x264enc ! clocksync`) works as well as a raw one.
//!
//! The timing itself is [`PresentationPacer`], the same anchor / segment /
//! deadline logic the display sinks pace on, read through its deadline rather
//! than its QoS verdict. Two differences from a sink: this never drops a buffer
//! (a late or segment-clipped one is forwarded immediately, because a transform
//! dropping data would be a hole in the stream downstream cannot recover), and
//! it supplies its own monotonic clock when the runner elected none, which is
//! how GStreamer's `clocksync` falls back to the pipeline's system clock.

use core::future::Future;
use core::pin::Pin;
use core::time::Duration;

use alloc::boxed::Box;
use alloc::sync::Arc;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ClockSync, ConfigureOutcome, ElementMetadata, G2gError,
    OutputSink, PipelinePacket, PresentationPacer, PropError, PropKind, PropValue, PropertySpec,
};

use crate::clock::WallClock;

#[derive(Debug)]
pub struct ClockSyncTransform {
    /// The PTS -> clock deadline, first-buffer anchor and segment mapping.
    pacer: PresentationPacer,
    /// The clock the pacer measures against, kept here too because the pacer
    /// exposes deadlines but not the clock to sleep on. `None` until the first
    /// paced buffer installs the runner's clock or the local fallback.
    clock: Option<ClockSync>,
    sync: bool,
    ts_offset_ns: i64,
    forwarded: u64,
    configured: bool,
}

impl Default for ClockSyncTransform {
    fn default() -> Self {
        Self::new()
    }
}

impl ClockSyncTransform {
    pub fn new() -> Self {
        Self {
            pacer: PresentationPacer::new(),
            clock: None,
            sync: true,
            ts_offset_ns: 0,
            forwarded: 0,
            configured: false,
        }
    }

    /// `false` makes the element a plain pass-through, forwarding every buffer
    /// as it arrives. Default `true`.
    pub fn with_sync(mut self, sync: bool) -> Self {
        self.sync = sync;
        self
    }

    /// Nanoseconds added to every buffer's deadline: positive holds the stream
    /// back that much further, negative releases it earlier (never before the
    /// buffer arrives).
    pub fn with_ts_offset_ns(mut self, ns: i64) -> Self {
        self.ts_offset_ns = ns;
        self
    }

    /// Buffers forwarded so far.
    pub fn forwarded(&self) -> u64 {
        self.forwarded
    }

    /// The clock to pace on, installing the local monotonic fallback the first
    /// time if the runner elected no clock for us.
    fn clock_sync(&mut self) -> ClockSync {
        if let Some(clock) = &self.clock {
            return clock.clone();
        }
        let sync = ClockSync::new(Arc::new(WallClock::new()), 0);
        self.pacer.set_clock_sync(sync.clone());
        self.clock = Some(sync.clone());
        sync
    }

    /// Hold the buffer until its deadline. A buffer with no deadline (clipped
    /// outside the segment) is forwarded now instead: see the module note.
    async fn hold_until_due(&mut self, pts_ns: u64) {
        let sync = self.clock_sync();
        let Some(deadline) = self.pacer.deadline_ns(pts_ns) else {
            return;
        };
        // Shift the deadline, not the wait: a `ts-offset` larger than the frame
        // period would otherwise be spent again on every buffer instead of once
        // on the schedule.
        let due = shift(deadline, self.ts_offset_ns);
        let now = sync.now_ns();
        if due <= now {
            return;
        }
        // Sleep on the clock's own timer when it has one, so the pacing follows
        // that timeline (and a test clock can skip the wait). An elected clock
        // that cannot sleep (an audio or PTP clock) still yields a deadline, so
        // wait out the interval on the tokio timer instead.
        match sync.clock.as_ticker() {
            Some(ticker) => ticker.sleep_until_ns(due).await,
            None => tokio::time::sleep(Duration::from_nanos(due - now)).await,
        }
    }
}

/// A deadline moved by `ts-offset`, saturating at the start of the timeline.
fn shift(deadline_ns: u64, offset_ns: i64) -> u64 {
    if offset_ns >= 0 {
        deadline_ns.saturating_add(offset_ns as u64)
    } else {
        deadline_ns.saturating_sub(offset_ns.unsigned_abs())
    }
}

impl AsyncElement for ClockSyncTransform {
    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Clock sync",
            "Generic",
            "Delays each buffer until its PTS is due on the pipeline clock",
            "g2g",
        )
    }

    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    /// Pacing says nothing about format: couple the two links without
    /// constraining either, as `identity` does.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    /// Pace against the elected pipeline clock when the runner hands one over,
    /// in place of this element's own monotonic fallback.
    fn set_clock_sync(&mut self, sync: ClockSync) {
        self.pacer.set_clock_sync(sync.clone());
        self.clock = Some(sync);
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        CLOCKSYNC_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "sync" => self.sync = value.as_bool().ok_or(PropError::Type)?,
            "ts-offset" => self.ts_offset_ns = value.as_int().ok_or(PropError::Type)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "sync" => Some(PropValue::Bool(self.sync)),
            "ts-offset" => Some(PropValue::Int(self.ts_offset_ns)),
            _ => None,
        }
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
                PipelinePacket::DataFrame(frame) => {
                    if self.sync {
                        self.hold_until_due(frame.timing.pts_ns).await;
                    }
                    self.forwarded += 1;
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                // Segment and flush drive the PTS -> running-time mapping and the
                // re-anchor after a seek, and are forwarded like any control packet.
                PipelinePacket::Segment(seg) => {
                    self.pacer.set_segment(seg);
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                PipelinePacket::Flush => {
                    self.pacer.flush();
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

static CLOCKSYNC_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "sync",
        PropKind::Bool,
        "hold each buffer until its PTS is due on the clock (default true)",
    ),
    PropertySpec::new(
        "ts-offset",
        PropKind::Int,
        "ns added to each buffer's deadline (positive delays, negative advances)",
    ),
];
