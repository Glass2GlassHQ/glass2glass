//! Closed-caption combiner (M1096, `no_std`): the `cccombiner` analog. A video
//! stream on input 0 and a closed-caption stream on input 1 go in, the same video
//! comes out with each frame carrying the caption triples that belong with it as
//! [`CaptionMeta`](g2g_core::meta::CaptionMeta). The pixels are untouched, so this
//! is the point where captions rejoin a decoded stream: `cccombiner ! h264enc !
//! ccinsert` (meta-sourced) puts them back in the bitstream's SEI, and a
//! caption-aware sink reads them off the frame.
//!
//! The two pads are merged by PTS, so a caption payload lands on the video frame
//! whose window it opens in. The caption pad reads any of the layouts
//! [`ClosedCaptionFormat`] names, through [`crate::cea`]'s transport codec, so an
//! MP4 caption track, an ancillary CDP and a line-21 byte-pair stream all combine
//! the same way.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::meta::CaptionMeta;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ClosedCaptionFormat, ConfigureOutcome, ElementMetadata,
    G2gError, MultiInputElement, OutputSink, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec,
};

use crate::cea::{
    parse_caption_transport, CcTriple, TransportParams, CEA608_FIELD_1, CEA608_FIELD_2,
};

/// What to do with caption meta the incoming video frame already carries, the
/// `input-meta-processing` property. A decoded stream can arrive with the
/// captions its bitstream carried, so a combiner has to say which set wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputMetaProcessing {
    /// Keep both: the frame's own triples then the combined ones.
    Append,
    /// Discard the frame's own triples and keep the combined ones.
    Drop,
    /// Keep the frame's own triples when it has any, else the combined ones.
    Favor,
    /// Keep the frame's own triples and never add the combined ones.
    Force,
}

const NICK_APPEND: &str = "append";
const NICK_DROP: &str = "drop";
const NICK_FAVOR: &str = "favor";
const NICK_FORCE: &str = "force";
const INPUT_META_PROCESSING_NICKS: &str = "append | drop | favor | force";

impl InputMetaProcessing {
    fn from_nick(nick: &str) -> Option<Self> {
        match nick {
            NICK_APPEND => Some(Self::Append),
            NICK_DROP => Some(Self::Drop),
            NICK_FAVOR => Some(Self::Favor),
            NICK_FORCE => Some(Self::Force),
            _ => None,
        }
    }

    fn nick(self) -> &'static str {
        match self {
            Self::Append => NICK_APPEND,
            Self::Drop => NICK_DROP,
            Self::Favor => NICK_FAVOR,
            Self::Force => NICK_FORCE,
        }
    }
}

/// Queued caption payloads before the oldest is dropped, matching GStreamer's
/// default. A video stream that stalls must not grow the queue without bound.
const DEFAULT_MAX_SCHEDULED: u64 = 30;

static CCCOMBINER_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "max-scheduled",
        PropKind::Uint,
        "caption payloads queued for the next video frame before the oldest is dropped (0 = no limit)",
    )
    .with_default("30"),
    PropertySpec::new(
        "input-meta-processing",
        PropKind::Str,
        "what to do with caption meta the incoming video frame already carries",
    )
    .with_default(NICK_APPEND)
    .with_enum_values(INPUT_META_PROCESSING_NICKS),
    PropertySpec::new(
        "field",
        PropKind::Uint,
        "line-21 field bare CEA-608 byte pairs belong to (the `raw` caption layout only)",
    )
    .with_range("0", "1")
    .with_default("0"),
];

/// Attach a closed-caption stream to the video stream it belongs with, the
/// `cccombiner` analog. Input 0 is the video (any caps, and the merged output
/// follows it), input 1 the captions; the pads merge by PTS and each video frame
/// leaves carrying the caption triples queued for it.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::cccombiner::CcCombiner;
///
/// let combiner = CcCombiner::new();
/// ```
#[derive(Debug)]
pub struct CcCombiner {
    /// The caption layout fixed at `configure(CAPTION)`.
    caption_format: Option<ClosedCaptionFormat>,
    /// The negotiated video caps, the merged output (it follows the video pad).
    video_caps: Option<Caps>,
    /// Caption triples waiting for the next video frame, oldest first, one entry
    /// per caption payload so `max-scheduled` counts what GStreamer counts.
    scheduled: VecDeque<Vec<CcTriple>>,
    max_scheduled: u64,
    input_meta_processing: InputMetaProcessing,
    field: u8,
}

impl Default for CcCombiner {
    fn default() -> Self {
        Self {
            caption_format: None,
            video_caps: None,
            scheduled: VecDeque::new(),
            max_scheduled: DEFAULT_MAX_SCHEDULED,
            input_meta_processing: InputMetaProcessing::Append,
            field: CEA608_FIELD_1,
        }
    }
}

impl CcCombiner {
    /// Input pad indices: video on 0, the caption stream on 1.
    pub const VIDEO: usize = 0;
    pub const CAPTION: usize = 1;

    pub fn new() -> Self {
        Self::default()
    }

    /// The caption layouts the caption pad reads.
    fn caption_alternatives() -> CapsSet {
        CapsSet::from_alternatives(Vec::from([
            Caps::ClosedCaption {
                format: ClosedCaptionFormat::Cea708,
            },
            Caps::ClosedCaption {
                format: ClosedCaptionFormat::Cea708Cdp,
            },
            Caps::ClosedCaption {
                format: ClosedCaptionFormat::Cea608,
            },
            Caps::ClosedCaption {
                format: ClosedCaptionFormat::Cea608S334,
            },
            Caps::ClosedCaption {
                format: ClosedCaptionFormat::Cea608Raw,
            },
        ]))
    }

    /// Queue one caption payload's triples, dropping the oldest once
    /// `max-scheduled` payloads are waiting.
    fn schedule(&mut self, payload: &[u8]) {
        let Some(format) = self.caption_format else {
            return;
        };
        let params = TransportParams {
            field: self.field,
            ..TransportParams::default()
        };
        let triples = parse_caption_transport(payload, format, params);
        if triples.is_empty() {
            return;
        }
        self.scheduled.push_back(triples);
        while self.max_scheduled > 0 && self.scheduled.len() as u64 > self.max_scheduled {
            self.scheduled.pop_front();
        }
    }

    /// The triples this video frame leaves carrying: the queued ones combined
    /// with whatever the frame already had, per `input-meta-processing`. Drains
    /// the queue whichever way the two combine, so a discarded payload is not
    /// re-offered to the next frame.
    fn combine(&mut self, existing: Vec<CcTriple>) -> Vec<CcTriple> {
        let mut queued: Vec<CcTriple> = Vec::new();
        for payload in self.scheduled.drain(..) {
            queued.extend(payload);
        }
        match self.input_meta_processing {
            InputMetaProcessing::Append => {
                let mut out = existing;
                out.extend(queued);
                out
            }
            InputMetaProcessing::Drop => queued,
            InputMetaProcessing::Favor => {
                if existing.is_empty() {
                    queued
                } else {
                    existing
                }
            }
            InputMetaProcessing::Force => existing,
        }
    }
}

impl MultiInputElement for CcCombiner {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    // The video frame's pixels are never read, only its metadata written, so the
    // default `DomainSet::ALL` stands and a GPU frame is not downloaded to pass
    // through here. A caption payload is always host memory (nothing produces
    // `Caps::ClosedCaption` anywhere else); one that is not carries no captions.

    fn input_count(&self) -> usize {
        2
    }

    /// Merge the two pads by PTS, so a caption payload lands on the video frame
    /// whose window it opens in.
    fn input_pts_ordered(&self) -> bool {
        true
    }

    /// The merged output is the video pad's stream (the same pixels, now carrying
    /// caption meta), so the solver derives the output caps from pad 0.
    fn output_follows_input(&self) -> Option<usize> {
        Some(Self::VIDEO)
    }

    /// Named request pads: `video` -> the video pad (0), `caption` / `text` ->
    /// the caption pad (1), so a launch line can wire the branches in either
    /// order.
    fn input_pad_index(
        &self,
        req: &g2g_core::runtime::PadRequest,
        _ordinal: usize,
    ) -> Option<usize> {
        match req.kind {
            g2g_core::runtime::PadKind::Video => Some(Self::VIDEO),
            g2g_core::runtime::PadKind::Text => Some(Self::CAPTION),
            _ => None,
        }
    }

    fn intercept_caps(&self, input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match input {
            // The pixels are untouched, so the video pad takes any caps, as
            // GStreamer's does.
            Self::VIDEO => Ok(upstream_caps.clone()),
            Self::CAPTION
                if Self::caption_alternatives()
                    .alternatives()
                    .contains(upstream_caps) =>
            {
                Ok(upstream_caps.clone())
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn caps_constraint_as_input(&self, input: usize) -> CapsConstraint<'_> {
        match input {
            Self::CAPTION => CapsConstraint::Accepts(Self::caption_alternatives()),
            _ => CapsConstraint::AcceptsAny,
        }
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        match (input, absolute_caps) {
            (Self::VIDEO, _) => {
                self.video_caps = Some(absolute_caps.clone());
                Ok(ConfigureOutcome::Accepted)
            }
            (Self::CAPTION, Caps::ClosedCaption { format }) => {
                self.caption_format = Some(*format);
                Ok(ConfigureOutcome::Accepted)
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        self.video_caps.clone().ok_or(G2gError::NotConfigured)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Closed-caption combiner",
            "Filter/Video/ClosedCaption",
            "Attaches a closed-caption stream to the video frames it belongs with",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        CCCOMBINER_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match (name, value) {
            ("max-scheduled", PropValue::Uint(v)) => {
                self.max_scheduled = v;
                Ok(())
            }
            ("input-meta-processing", PropValue::Str(nick)) => {
                self.input_meta_processing =
                    InputMetaProcessing::from_nick(&nick).ok_or(PropError::Value)?;
                Ok(())
            }
            ("field", PropValue::Uint(v)) => {
                if v > u64::from(CEA608_FIELD_2) {
                    return Err(PropError::Value);
                }
                self.field = v as u8;
                Ok(())
            }
            ("max-scheduled" | "input-meta-processing" | "field", _) => Err(PropError::Type),
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "max-scheduled" => Some(PropValue::Uint(self.max_scheduled)),
            "input-meta-processing" => Some(PropValue::Str(String::from(
                self.input_meta_processing.nick(),
            ))),
            "field" => Some(PropValue::Uint(u64::from(self.field))),
            _ => None,
        }
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match input {
                Self::VIDEO => match packet {
                    PipelinePacket::DataFrame(mut frame) => {
                        let existing: Vec<CcTriple> = frame
                            .meta
                            .get::<CaptionMeta>()
                            .map(|m| m.iter().map(|t| CcTriple::from(*t)).collect())
                            .unwrap_or_default();
                        let had_captions = !existing.is_empty();
                        let combined = self.combine(existing);
                        // Attaching replaces, so a frame whose own captions the
                        // merge discarded has to be written even when what
                        // replaces them is nothing.
                        if !combined.is_empty() || had_captions {
                            let mut meta = CaptionMeta::new();
                            for triple in combined {
                                meta.push(triple.into());
                            }
                            frame.meta.attach(meta);
                        }
                        out.push(PipelinePacket::DataFrame(frame)).await.map(|_| ())
                    }
                    // The runner aggregates the per-pad Eos, so the element does
                    // not forward it.
                    PipelinePacket::Eos => Ok(()),
                    other => out.push(other).await.map(|_| ()),
                },
                // The caption pad (and any other pad, defensively, though there
                // are two): its packets feed the queue, never the output.
                _ => {
                    if let PipelinePacket::DataFrame(frame) = packet {
                        if let Some(payload) = frame.domain.as_system_slice() {
                            self.schedule(payload);
                        }
                    }
                    Ok(())
                }
            }
        })
    }
}
