//! A/V offset transform (M385): shifts every `DataFrame`'s PTS/DTS by a signed
//! nanosecond `offset`, the g2g form of GStreamer playbin's `av-offset`. Put one
//! on a branch of a multi-stream playback graph (typically the audio branch) to
//! re-align it against the others: a positive `offset` delays the stream (its
//! frames present later), a negative one advances it (clamped at 0, since a PTS
//! cannot go negative).
//!
//! Pass-through in every other respect (caps, frame data, control packets), so it
//! drops anywhere a 1-in/1-out element fits. Per the transform contract it does
//! NOT emit `Eos`; the runner forwards the sentinel after `process(Eos)` returns.
//!
//! ```text
//! ... ! avoffset offset=40000000 ! audiosink     // delay audio 40 ms vs video
//! ```

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, OutputSink,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

/// Shifts frame timestamps by a signed offset. See the module docs.
#[derive(Debug, Default)]
pub struct AvOffset {
    /// Signed nanosecond shift applied to PTS/DTS. Positive delays, negative
    /// advances (clamped at 0).
    offset_ns: i64,
    configured: bool,
}

impl AvOffset {
    /// An offsetter with `offset_ns` (positive delays the stream, negative
    /// advances it). `0` is a pass-through.
    pub fn new(offset_ns: i64) -> Self {
        Self { offset_ns, configured: false }
    }

    /// Apply the offset to one timestamp, saturating at the `u64` bounds (a
    /// negative result clamps to 0, since a PTS cannot go negative).
    fn shift(&self, ts_ns: u64) -> u64 {
        if self.offset_ns >= 0 {
            ts_ns.saturating_add(self.offset_ns as u64)
        } else {
            ts_ns.saturating_sub(self.offset_ns.unsigned_abs())
        }
    }
}

impl AsyncElement for AvOffset {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    /// Wildcard pass-through (like `identity`): it touches timing, not caps, so it
    /// constrains neither input nor output.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
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
                PipelinePacket::DataFrame(mut f) => {
                    // A zero offset leaves the timeline untouched (a pure
                    // pass-through), so timestamps are only rewritten when set.
                    if self.offset_ns != 0 {
                        f.timing.pts_ns = self.shift(f.timing.pts_ns);
                        f.timing.dts_ns = self.shift(f.timing.dts_ns);
                    }
                    out.push(PipelinePacket::DataFrame(f)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    out.push(PipelinePacket::CapsChanged(c)).await?;
                }
                PipelinePacket::Flush => {
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        AVOFFSET_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "AvOffset",
            "Filter/Effect",
            "Shifts frame PTS/DTS by a signed offset (the av-offset A/V sync knob)",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "offset" => self.offset_ns = value.as_int().ok_or(PropError::Type)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "offset" => Some(PropValue::Int(self.offset_ns)),
            _ => None,
        }
    }
}

/// `AvOffset`'s settable properties (M104).
static AVOFFSET_PROPS: &[PropertySpec] =
    &[PropertySpec::new("offset", PropKind::Int, "PTS/DTS shift in ns (positive delays, negative advances)")];
