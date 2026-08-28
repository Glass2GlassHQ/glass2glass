//! Machinery shared by the subtitle writers, [`crate::srtenc`] and
//! [`crate::webvttenc`] (the two elements of GStreamer's `subenc` plugin). Both
//! take timed `Caps::Text{Utf8}` cues and emit one document block per cue; they
//! differ only in the block dialect ([`write_cue_block`]), the output caps, and
//! WebVTT's file header.

use alloc::string::String;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::{
    Caps, G2gError, MemoryDomain, OutputSink, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec, TextFormat,
};

use crate::subparse::write_cue_block;

/// The offset knobs both writers expose: they shift the cue window written into
/// the document without moving the frame the block rides on.
pub static SUBTITLE_ENC_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "timestamp",
        PropKind::Int,
        "nanoseconds added to each written cue's start time",
    )
    .with_default("0"),
    PropertySpec::new(
        "duration",
        PropKind::Int,
        "nanoseconds added to each written cue's duration",
    )
    .with_default("0"),
];

/// Shift a cue time by a signed nanosecond offset, clamping at zero: the offsets
/// are signed, and a cue cannot start before the start of the stream.
fn shift_ns(time_ns: u64, offset_ns: i64) -> u64 {
    if offset_ns >= 0 {
        time_ns.saturating_add(offset_ns as u64)
    } else {
        time_ns.saturating_sub(offset_ns.unsigned_abs())
    }
}

/// The cue-to-block state both subtitle writers keep: the two timing offsets,
/// the SubRip cue counter, and whether the output caps have been announced.
#[derive(Debug)]
pub struct CueWriter {
    timestamp_offset_ns: i64,
    duration_offset_ns: i64,
    /// The number the next written cue gets; SubRip counts from 1 (WebVTT cues
    /// carry no number, so the count is unused there).
    next_number: u64,
    caps_emitted: bool,
    configured: bool,
}

impl Default for CueWriter {
    fn default() -> Self {
        Self {
            timestamp_offset_ns: 0,
            duration_offset_ns: 0,
            next_number: 1,
            caps_emitted: false,
            configured: false,
        }
    }
}

impl CueWriter {
    pub fn set_timestamp_offset(&mut self, offset_ns: i64) {
        self.timestamp_offset_ns = offset_ns;
    }

    pub fn set_duration_offset(&mut self, offset_ns: i64) {
        self.duration_offset_ns = offset_ns;
    }

    pub fn configure(&mut self) {
        self.configured = true;
    }

    pub fn is_configured(&self) -> bool {
        self.configured
    }

    pub fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match (name, value) {
            ("timestamp", PropValue::Int(v)) => {
                self.timestamp_offset_ns = v;
                Ok(())
            }
            ("duration", PropValue::Int(v)) => {
                self.duration_offset_ns = v;
                Ok(())
            }
            ("timestamp" | "duration", _) => Err(PropError::Type),
            _ => Err(PropError::Unknown),
        }
    }

    pub fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "timestamp" => Some(PropValue::Int(self.timestamp_offset_ns)),
            "duration" => Some(PropValue::Int(self.duration_offset_ns)),
            _ => None,
        }
    }

    /// The `format` block this cue frame writes, or `None` when the frame holds
    /// no host memory or its text is blank (an empty block would end the
    /// preceding cue early when the document is read back). A written SubRip
    /// block consumes the next cue number.
    pub fn cue_block(&mut self, frame: &Frame, format: TextFormat) -> Option<String> {
        let text = String::from_utf8_lossy(frame.domain.as_system_slice()?);
        let start = shift_ns(frame.timing.pts_ns, self.timestamp_offset_ns);
        let duration = shift_ns(frame.timing.duration_ns, self.duration_offset_ns);
        let block = write_cue_block(
            self.next_number,
            start,
            start.saturating_add(duration),
            &text,
            format,
        );
        if block.is_empty() {
            return None;
        }
        self.next_number += 1;
        Some(block)
    }

    /// Emit `text` as one output frame carrying `timing`, announcing `caps`
    /// before the first one.
    pub async fn push_text(
        &mut self,
        out: &mut dyn OutputSink,
        text: String,
        timing: FrameTiming,
        sequence: u64,
        caps: Caps,
    ) -> Result<(), G2gError> {
        if !self.caps_emitted {
            out.push(PipelinePacket::CapsChanged(caps)).await?;
            self.caps_emitted = true;
        }
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(
                text.into_bytes().into_boxed_slice(),
            )),
            timing,
            sequence,
        );
        out.push(PipelinePacket::DataFrame(frame)).await.map(|_| ())
    }

    /// The runner hands each element its own pre-fixed output caps as a
    /// `CapsChanged` packet, which must be forwarded (and suppresses the
    /// element's own emission before the first block). Any other caps is an
    /// upstream echo with no effect on the document, so it is absorbed.
    pub async fn forward_output_caps(
        &mut self,
        out: &mut dyn OutputSink,
        caps: Caps,
        output_caps: &Caps,
    ) -> Result<(), G2gError> {
        if caps != *output_caps {
            return Ok(());
        }
        self.caps_emitted = true;
        out.push(PipelinePacket::CapsChanged(caps))
            .await
            .map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_negative_offset_clamps_at_the_start_of_the_stream() {
        assert_eq!(shift_ns(1_000, -400), 600);
        assert_eq!(shift_ns(1_000, -4_000), 0);
        assert_eq!(shift_ns(1_000, 400), 1_400);
    }
}
