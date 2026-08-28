//! Closed-caption transport converter (M1096, `no_std`): the `ccconverter`
//! analog. The same `(cc_type, cc_data_1, cc_data_2)` triples travel in four byte
//! layouts, and each carrier picks a different one: an H.264 SEI and an MP4
//! caption track hold packed ATSC `cc_data`, an ST 2110-40 ancillary packet holds
//! a caption distribution packet, an SDI ancillary path holds SMPTE ST 334-1
//! Annex A triplets, and a line-21 tool holds bare byte pairs. This element reads
//! one layout and writes another.
//!
//! The layouts, and the parse / write pair for each, are
//! [`crate::cea`]'s ([`parse_caption_transport`] / [`write_caption_transport`]);
//! the layout itself rides in the caps as a
//! [`ClosedCaptionFormat`], so the solver negotiates it like any other media type.
//!
//! A conversion is lossy exactly where the standards are: bare CEA-608 pairs
//! carry one line-21 field, and ST 334-1 carries no DTVCC, so triples those
//! layouts cannot hold are dropped rather than mistyped.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ClosedCaptionFormat, ConfigureOutcome,
    ElementMetadata, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
    PropError, PropKind, PropValue, PropertySpec,
};

use crate::cea::{
    cdp_frame_rate_code, parse_caption_transport, write_caption_transport, TransportParams,
    CEA608_FIELD_1, CEA608_FIELD_2,
};

/// The layout nicks, spelled as GStreamer spells the `format` field of a
/// `closedcaption/x-cea-608` / `x-cea-708` caps.
const NICK_RAW: &str = "raw";
const NICK_S334_1A: &str = "s334-1a";
const NICK_CC_DATA: &str = "cc_data";
const NICK_CDP: &str = "cdp";
/// The nicks as one `|` separated list, the closed set the properties accept.
const LAYOUT_NICKS: &str = "raw | s334-1a | cc_data | cdp";

/// The frame rate a CDP header declares when nothing sets one: 29.97, the
/// North-American caption norm.
const DEFAULT_FRAMERATE: (u32, u32) = (30_000, 1_001);

/// The layout a nick names, or `None` for a nick outside the set.
fn layout_from_nick(nick: &str) -> Option<ClosedCaptionFormat> {
    match nick {
        NICK_RAW => Some(ClosedCaptionFormat::Cea608Raw),
        NICK_S334_1A => Some(ClosedCaptionFormat::Cea608S334),
        NICK_CC_DATA => Some(ClosedCaptionFormat::Cea708),
        NICK_CDP => Some(ClosedCaptionFormat::Cea708Cdp),
        _ => None,
    }
}

/// The nick naming a layout. `Cea608` and `Cea708` are the same packed `cc_data`
/// bytes and share the one nick.
fn nick_of_layout(format: ClosedCaptionFormat) -> &'static str {
    match format {
        ClosedCaptionFormat::Cea608Raw => NICK_RAW,
        ClosedCaptionFormat::Cea608S334 => NICK_S334_1A,
        ClosedCaptionFormat::Cea708Cdp => NICK_CDP,
        _ => NICK_CC_DATA,
    }
}

static CCCONVERTER_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "in-format",
        PropKind::Str,
        "byte layout the incoming caption payloads are in",
    )
    .with_default(NICK_CC_DATA)
    .with_enum_values(LAYOUT_NICKS),
    PropertySpec::new(
        "out-format",
        PropKind::Str,
        "byte layout the outgoing caption payloads are written in",
    )
    .with_default(NICK_CDP)
    .with_enum_values(LAYOUT_NICKS),
    PropertySpec::new(
        "field",
        PropKind::Uint,
        "line-21 field bare CEA-608 byte pairs belong to (the `raw` layout only)",
    )
    .with_range("0", "1")
    .with_default("0"),
    PropertySpec::new(
        "framerate",
        PropKind::Fraction,
        "frame rate a written CDP header declares (the `cdp` layout only)",
    )
    .with_default("30000/1001"),
];

/// Convert closed captions between the four transport layouts, the `ccconverter`
/// analog: `in-format` in, `out-format` out, the triples unchanged in between.
/// One input frame becomes one output frame with the same timing, so a caption
/// track keeps its pacing across the conversion.
///
/// The layout is part of the caps, so `ccextract`, an MP4 caption track and an
/// ancillary-data packetizer each link only to the layout they read.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::ccconverter::CcConverter;
///
/// // An MP4 caption track's packed `cc_data` out as ancillary CDPs.
/// let converter = CcConverter::new();
/// ```
#[derive(Debug)]
pub struct CcConverter {
    in_format: ClosedCaptionFormat,
    out_format: ClosedCaptionFormat,
    /// The line-21 field bare CEA-608 pairs carry, on either side.
    field: u8,
    /// Numerator / denominator of the frame rate a written CDP declares.
    framerate: (u32, u32),
    /// The CDP sequence counter, rising once per written packet.
    sequence: u16,
    configured: bool,
    caps_emitted: bool,
}

impl Default for CcConverter {
    fn default() -> Self {
        Self {
            in_format: ClosedCaptionFormat::Cea708,
            out_format: ClosedCaptionFormat::Cea708Cdp,
            field: CEA608_FIELD_1,
            framerate: DEFAULT_FRAMERATE,
            sequence: 0,
            configured: false,
            caps_emitted: false,
        }
    }
}

impl CcConverter {
    /// Packed `cc_data` in, a caption distribution packet out.
    pub fn new() -> Self {
        Self::default()
    }

    /// Read `in_format` and write `out_format`.
    pub fn between(in_format: ClosedCaptionFormat, out_format: ClosedCaptionFormat) -> Self {
        Self {
            in_format,
            out_format,
            ..Self::default()
        }
    }

    /// The line-21 field bare CEA-608 byte pairs belong to (the `field` property).
    pub fn with_field(mut self, field: u8) -> Self {
        self.field = field;
        self
    }

    /// The frame rate a written CDP header declares (the `framerate` property).
    pub fn with_framerate(mut self, numerator: u32, denominator: u32) -> Self {
        self.framerate = (numerator, denominator);
        self
    }

    /// The caps this element's sink pad takes. Packed `cc_data` is one layout with
    /// two caps spellings (a 608 caption track and a 708 one carry the same
    /// bytes), so both link when `in-format` is `cc_data`.
    fn input_alternatives(&self) -> CapsSet {
        let mut alternatives = Vec::from([Caps::ClosedCaption {
            format: self.in_format,
        }]);
        if self.in_format == ClosedCaptionFormat::Cea708 {
            alternatives.push(Caps::ClosedCaption {
                format: ClosedCaptionFormat::Cea608,
            });
        }
        CapsSet::from_alternatives(alternatives)
    }

    fn output_caps(&self) -> Caps {
        Caps::ClosedCaption {
            format: self.out_format,
        }
    }

    /// The layout knobs the parse / write pair needs beyond the layout itself.
    fn transport_params(&self) -> TransportParams {
        TransportParams {
            field: self.field,
            frame_rate_code: cdp_frame_rate_code(self.framerate.0, self.framerate.1),
            sequence: self.sequence,
        }
    }

    /// Re-lay one caption payload from the input layout into the output one, or
    /// `None` when nothing survives (the target layout cannot carry these
    /// triples). Advances the CDP sequence counter for a written packet.
    fn convert(&mut self, payload: &[u8]) -> Option<Vec<u8>> {
        let triples = parse_caption_transport(payload, self.in_format, self.transport_params());
        let out = write_caption_transport(&triples, self.out_format, self.transport_params());
        if out.is_empty() {
            return None;
        }
        self.sequence = self.sequence.wrapping_add(1);
        Some(out)
    }
}

impl AsyncElement for CcConverter {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Re-lays the caption bytes in host memory, so it takes system frames only.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        if self
            .input_alternatives()
            .alternatives()
            .contains(upstream_caps)
        {
            Ok(upstream_caps.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    /// One layout in, another out, both fixed by the properties, so the legal
    /// pair is stated outright.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Mapping(Vec::from([(
            self.input_alternatives(),
            CapsSet::one(self.output_caps()),
        )]))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !self
            .input_alternatives()
            .alternatives()
            .contains(absolute_caps)
        {
            return Err(G2gError::CapsMismatch);
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Closed-caption converter",
            "Filter/ClosedCaption",
            "Converts closed captions between the cc_data, CDP, S334-1A and raw CEA-608 layouts",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        CCCONVERTER_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match (name, value) {
            ("in-format", PropValue::Str(nick)) => {
                self.in_format = layout_from_nick(&nick).ok_or(PropError::Value)?;
                Ok(())
            }
            ("out-format", PropValue::Str(nick)) => {
                self.out_format = layout_from_nick(&nick).ok_or(PropError::Value)?;
                Ok(())
            }
            ("field", PropValue::Uint(v)) => {
                if v > u64::from(CEA608_FIELD_2) {
                    return Err(PropError::Value);
                }
                self.field = v as u8;
                Ok(())
            }
            ("framerate", PropValue::Fraction(numerator, denominator)) => {
                if numerator <= 0 || denominator <= 0 {
                    return Err(PropError::Value);
                }
                self.framerate = (numerator as u32, denominator as u32);
                Ok(())
            }
            ("in-format" | "out-format" | "field" | "framerate", _) => Err(PropError::Type),
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "in-format" => Some(PropValue::Str(String::from(nick_of_layout(self.in_format)))),
            "out-format" => Some(PropValue::Str(String::from(nick_of_layout(
                self.out_format,
            )))),
            "field" => Some(PropValue::Uint(u64::from(self.field))),
            "framerate" => Some(PropValue::Fraction(
                self.framerate.0 as i32,
                self.framerate.1 as i32,
            )),
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
                    let Some(payload) = frame.domain.as_system_slice() else {
                        return Ok(());
                    };
                    let Some(converted) = self.convert(payload) else {
                        return Ok(());
                    };
                    if !self.caps_emitted {
                        out.push(PipelinePacket::CapsChanged(self.output_caps()))
                            .await?;
                        self.caps_emitted = true;
                    }
                    let new = Frame::new(
                        MemoryDomain::System(SystemSlice::from_boxed(converted.into_boxed_slice())),
                        frame.timing,
                        frame.sequence,
                    );
                    out.push(PipelinePacket::DataFrame(new)).await.map(|_| ())
                }
                // The runner hands each element its own pre-fixed output caps as a
                // CapsChanged packet: forward it and suppress the emission at the
                // first converted frame. An upstream echo is absorbed.
                PipelinePacket::CapsChanged(caps) => {
                    if caps != self.output_caps() {
                        return Ok(());
                    }
                    self.caps_emitted = true;
                    out.push(PipelinePacket::CapsChanged(caps))
                        .await
                        .map(|_| ())
                }
                // The runner arm forwards the trailing Eos.
                PipelinePacket::Eos => Ok(()),
                other => out.push(other).await.map(|_| ()),
            }
        })
    }
}

impl PadTemplates for CcConverter {
    fn pad_templates() -> Vec<PadTemplate> {
        // Every layout is a legal pad caps; the properties pick the pair, so the
        // template advertises the whole set on both sides (as gst's does).
        let every_layout = CapsSet::from_alternatives(Vec::from([
            Caps::ClosedCaption {
                format: ClosedCaptionFormat::Cea608,
            },
            Caps::ClosedCaption {
                format: ClosedCaptionFormat::Cea608Raw,
            },
            Caps::ClosedCaption {
                format: ClosedCaptionFormat::Cea608S334,
            },
            Caps::ClosedCaption {
                format: ClosedCaptionFormat::Cea708,
            },
            Caps::ClosedCaption {
                format: ClosedCaptionFormat::Cea708Cdp,
            },
        ]));
        Vec::from([
            PadTemplate::sink(every_layout.clone()),
            PadTemplate::source(every_layout),
        ])
    }
}
