//! Decoded-GOP reverser (M897): the presentation half of reverse playback.
//!
//! A decoder only runs forward, so a reverse-rate segment (`rate < 0`) is played
//! by feeding it whole GOPs in decode order, newest GOP first
//! ([`Mp4Src`](crate::mp4src)), and reversing each decoded GOP afterwards. This
//! element is that second half: it sits after the decoder, buffers the frames of
//! one GOP, and re-emits them in descending PTS, so the frames reaching the sink
//! are in reverse presentation order and the reverse `Segment` maps them to
//! ascending running time.
//!
//! The GOP boundary needs no extra signalling: within a GOP the decoder emits
//! ascending PTS, and the next (earlier) GOP restarts lower, so a PTS that jumps
//! backward closes the batch. `Eos` closes the last one.
//!
//! Forward segments pass straight through, so the element is harmless anywhere in
//! a graph that may later seek backward.

use core::future::Future;
use core::mem;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, OutputSink,
    PipelinePacket,
};

#[derive(Debug, Default)]
pub struct GopReverse {
    /// Frames of the GOP being collected, in the order the decoder emitted them.
    /// Empty unless a reverse segment is in force.
    gop: Vec<Frame>,
    /// Whether the active segment is reverse (`rate < 0`).
    reverse: bool,
    /// PTS of the previous frame, so a backward jump can close the GOP.
    last_pts: Option<u64>,
    configured: bool,
}

impl GopReverse {
    pub fn new() -> Self {
        Self::default()
    }

    /// Emit the buffered GOP newest frame first. The decoder emits a GOP in
    /// ascending presentation order, so this is a reversal; sorting (rather than
    /// reversing) also covers a decoder that emits a GOP slightly out of order.
    async fn emit_gop(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let mut gop = mem::take(&mut self.gop);
        gop.sort_by_key(|f| core::cmp::Reverse(f.timing.pts_ns));
        for f in gop {
            out.push(PipelinePacket::DataFrame(f)).await?;
        }
        Ok(())
    }
}

impl AsyncElement for GopReverse {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    /// Timing-only transform: whatever the decoder emits is what the sink gets.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "GOP reverser",
            "Filter/Video",
            "Re-emits each decoded GOP in descending PTS for reverse playback",
            "g2g",
        )
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
                    if !self.reverse {
                        out.push(PipelinePacket::DataFrame(f)).await?;
                        return Ok(());
                    }
                    if self.last_pts.is_some_and(|prev| f.timing.pts_ns < prev) {
                        self.emit_gop(out).await?;
                    }
                    self.last_pts = Some(f.timing.pts_ns);
                    self.gop.push(f);
                }
                // A new segment ends the old one: its frames go out first.
                PipelinePacket::Segment(seg) => {
                    self.emit_gop(out).await?;
                    self.reverse = seg.rate < 0.0;
                    self.last_pts = None;
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                // A flush discards what was buffered; it is not presented.
                PipelinePacket::Flush => {
                    self.gop.clear();
                    self.last_pts = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                // The last GOP has no following frame to close it. The runner
                // forwards the EOS sentinel itself.
                PipelinePacket::Eos => self.emit_gop(out).await?,
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}
