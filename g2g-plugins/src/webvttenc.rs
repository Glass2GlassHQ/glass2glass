//! WebVTT (`.vtt`) subtitle writer (M1096, `no_std`): the WebVTT half of the
//! pair [`crate::srtenc`] opens, and the inverse of the WebVTT side of
//! [`SubParse`](crate::subparse::SubParse). Timed `Caps::Text{Utf8}` cues in, a
//! `Caps::Text{WebVtt}` byte stream out.
//!
//! A WebVTT document opens with the `WEBVTT` signature, so the first emitted
//! frame is that header; the cue blocks that follow are the SubRip ones without
//! the sequence number and with `.` separating the milliseconds. The header is
//! written even for a stream that carries no cue, since a `.vtt` file without it
//! is not a WebVTT document at all.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, FrameTiming,
    G2gError, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropValue,
    PropertySpec, TextFormat,
};

use crate::subenc::{CueWriter, SUBTITLE_ENC_PROPS};
use crate::subparse::WEBVTT_HEADER;

/// Write timed text cues as a WebVTT (`.vtt`) document, the `webvttenc` analog:
/// one `Caps::Text{Utf8}` frame per cue in (its window the frame's PTS +
/// duration), a `WEBVTT` header frame then one `Caps::Text{WebVtt}` frame per cue
/// out. The frames concatenate into a `.vtt` file, so `... ! webvttenc ! filesink
/// location=out.vtt` records one.
///
/// A cue whose text is blank writes nothing: an empty block would end the
/// preceding cue early when the document is read back.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::webvttenc::WebVttEnc;
///
/// let encoder = WebVttEnc::new().with_duration_offset(500_000_000);
/// ```
#[derive(Debug, Default)]
pub struct WebVttEnc {
    writer: CueWriter,
    /// Whether the `WEBVTT` signature has been emitted.
    header_written: bool,
}

impl WebVttEnc {
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
            format: TextFormat::WebVtt,
        }
    }

    /// Emit the `WEBVTT` signature and the blank line ending its block, once,
    /// stamped with `timing` so the header rides the timeline of the cue it
    /// precedes (a lone header at `Eos` carries the stream's last timing).
    async fn write_header(
        &mut self,
        out: &mut dyn OutputSink,
        timing: FrameTiming,
        sequence: u64,
    ) -> Result<(), G2gError> {
        if self.header_written {
            return Ok(());
        }
        self.header_written = true;
        let header = alloc::format!("{WEBVTT_HEADER}\n\n");
        self.writer
            .push_text(out, header, timing, sequence, Self::output_caps())
            .await
    }
}

impl AsyncElement for WebVttEnc {
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

    /// Encoder-style: plain UTF-8 cues in, a WebVTT document out, so the solver
    /// negotiates `Text{WebVtt}` downstream whatever the cue source was.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| {
            if *input == WebVttEnc::input_caps() {
                CapsSet::one(WebVttEnc::output_caps())
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
            "WebVTT subtitle encoder",
            "Codec/Encoder/Subtitle",
            "Writes timed UTF-8 cues as a WebVTT (.vtt) document",
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
                    let Some(block) = self.writer.cue_block(&frame, TextFormat::WebVtt) else {
                        return Ok(());
                    };
                    self.write_header(out, frame.timing, frame.sequence).await?;
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
                // A stream that carried no cue still owes a header, or the file
                // downstream is not a WebVTT document. The runner arm forwards
                // the trailing Eos.
                PipelinePacket::Eos => self.write_header(out, FrameTiming::default(), 0).await,
                other => out.push(other).await.map(|_| ()),
            }
        })
    }
}

impl PadTemplates for WebVttEnc {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::one(Self::input_caps())),
            PadTemplate::source(CapsSet::one(Self::output_caps())),
        ])
    }
}
