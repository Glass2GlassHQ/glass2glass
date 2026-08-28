//! SubRip (`.srt`) subtitle writer (M1096, `no_std`): the inverse of
//! [`SubParse`](crate::subparse::SubParse). Timed `Caps::Text{Utf8}` cues in, a
//! `Caps::Text{Srt}` byte stream out, one frame per cue holding the numbered
//! block a SubRip file is made of (sequence number, `HH:MM:SS,mmm -->
//! HH:MM:SS,mmm`, the text, a blank line).
//!
//! The block itself is written by
//! [`write_cue_block`](crate::subparse::write_cue_block), which the SubRip parser
//! reads back, and the cue bookkeeping is the [`CueWriter`] shared with
//! [`crate::webvttenc`].

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, G2gError,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropValue, PropertySpec,
    TextFormat,
};

use crate::subenc::{CueWriter, SUBTITLE_ENC_PROPS};

/// Write timed text cues as a SubRip (`.srt`) document, the `srtenc` analog: one
/// `Caps::Text{Utf8}` frame per cue in (its window the frame's PTS + duration),
/// one `Caps::Text{Srt}` frame per cue out holding that cue's numbered block. The
/// blocks concatenate into a `.srt` file, so `... ! srtenc ! filesink
/// location=out.srt` records one.
///
/// Cues are numbered from 1 in arrival order. A cue whose text is blank writes
/// nothing and takes no number: an empty block would end the preceding cue early
/// when the document is read back.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::srtenc::SrtEnc;
///
/// let encoder = SrtEnc::new().with_timestamp_offset(-500_000_000);
/// ```
#[derive(Debug, Default)]
pub struct SrtEnc {
    writer: CueWriter,
}

impl SrtEnc {
    pub fn new() -> Self {
        Self::default()
    }

    /// Shift every written cue start by `offset_ns` (the `timestamp` property).
    pub fn with_timestamp_offset(mut self, offset_ns: i64) -> Self {
        self.writer.set_timestamp_offset(offset_ns);
        self
    }

    /// Shift every written cue duration by `offset_ns` (the `duration` property).
    pub fn with_duration_offset(mut self, offset_ns: i64) -> Self {
        self.writer.set_duration_offset(offset_ns);
        self
    }

    pub(crate) fn input_caps() -> Caps {
        Caps::Text {
            format: TextFormat::Utf8,
        }
    }

    pub(crate) fn output_caps() -> Caps {
        Caps::Text {
            format: TextFormat::Srt,
        }
    }
}

impl AsyncElement for SrtEnc {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Reads the cue text out of host memory, so it takes system frames only.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        if *upstream_caps == Self::input_caps() {
            Ok(upstream_caps.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    /// Encoder-style: plain UTF-8 cues in, a SubRip document out, so the solver
    /// negotiates `Text{Srt}` downstream whatever the cue source was.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| {
            if *input == SrtEnc::input_caps() {
                CapsSet::one(SrtEnc::output_caps())
            } else {
                CapsSet::from_alternatives(Vec::new())
            }
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if *absolute_caps != Self::input_caps() {
            return Err(G2gError::CapsMismatch);
        }
        self.writer.configure();
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "SubRip subtitle encoder",
            "Codec/Encoder/Subtitle",
            "Writes timed UTF-8 cues as a SubRip (.srt) document",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        SUBTITLE_ENC_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        self.writer.set_property(name, value)
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        self.writer.get_property(name)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.writer.is_configured() {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let Some(block) = self.writer.cue_block(&frame, TextFormat::Srt) else {
                        return Ok(());
                    };
                    self.writer
                        .push_text(
                            out,
                            block,
                            frame.timing,
                            frame.sequence,
                            Self::output_caps(),
                        )
                        .await
                }
                PipelinePacket::CapsChanged(caps) => {
                    self.writer
                        .forward_output_caps(out, caps, &Self::output_caps())
                        .await
                }
                // A flush is a discontinuity in the stream, not a new file: the
                // numbering keeps counting, or the document downstream would
                // repeat numbers a reader orders its cues by.
                // The runner arm forwards the trailing Eos; SubRip has no trailer.
                PipelinePacket::Eos => Ok(()),
                other => out.push(other).await.map(|_| ()),
            }
        })
    }
}

impl PadTemplates for SrtEnc {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::one(Self::input_caps())),
            PadTemplate::source(CapsSet::one(Self::output_caps())),
        ])
    }
}
