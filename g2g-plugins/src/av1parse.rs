//! AV1 frame parser that refines source-side `Caps` from the sequence header.
//!
//! The AV1 sibling of `vp9parse`: it walks the OBUs of each `DataFrame` (the
//! low-overhead bitstream format, size-delimited), and when a sequence-header
//! OBU is present parses `max_frame_width` / `max_frame_height`, emitting a
//! `CapsChanged` with `Dim::Fixed` geometry before forwarding. A demuxer
//! (mkvdemux) can take geometry from the container Tracks; this recovers it from
//! the bitstream, where the sequence header rides the keyframe temporal unit.
//!
//! Two layers: an OBU walk (1-byte `obu_header`, optional extension byte, a
//! LEB128 size) selects the `OBU_SEQUENCE_HEADER`, then the sequence header is
//! read MSB-first via the shared `annexb::BitReader`. The header is the fiddliest
//! of the parser sprint: past `seq_profile` it branches on
//! `reduced_still_picture_header`, then walks the operating-points loop (with the
//! optional `timing_info` / `decoder_model_info` / `initial_display_delay`
//! sub-structures) to reach the `frame_width_bits` / `frame_height_bits` sizing
//! and the variable-width size fields. AV1 carries no framerate relevant to caps
//! geometry here, so refined caps report `Rate::Any` (matching mkvdemux).
//! Frames without a sequence-header OBU forward without a caps change.
//!
//! The same walk reads the header's `color_config`, so the refined caps also
//! carry the stream's colorimetry (`Colorimetry::UNKNOWN` where the header codes
//! no colour description).
//!
//! In re-framing mode ([`Av1Parse::reframing`], what `gst-launch` and the
//! auto-plugged decode chains build) a byte stream that carries temporal
//! delimiters, a raw `.obu` file read in arbitrary chunks, is split into one
//! temporal unit per `DataFrame`. Input without delimiters (the container
//! mappings strip them) already arrives one unit per buffer and passes through.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata,
    FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
    Rate, VideoCodec,
};

use crate::annexb::BitReader;

/// `OBU_SEQUENCE_HEADER` type, the only OBU carrying frame geometry.
const OBU_SEQUENCE_HEADER: u8 = 1;

/// What every CICP colour field spells "unspecified", the value `color_config`
/// leaves its fields at when it codes no colour description.
const CICP_UNSPECIFIED: u32 = 2;

/// # Example
///
/// ```no_run
/// use g2g_plugins::av1parse::Av1Parse;
///
/// let parser = Av1Parse::new();
/// assert_eq!(parser.caps_changes_emitted(), 0);
/// ```
#[derive(Debug, Default)]
pub struct Av1Parse {
    configured: bool,
    last_emitted_caps: Option<Caps>,
    headers_emitted: u64,
    reframe: bool,
    framer: TemporalUnitFramer,
    reframed_sequence: u64,
}

impl Av1Parse {
    pub fn new() -> Self {
        Self::default()
    }

    // splits a delimited byte stream into one temporal unit per frame
    pub fn reframing() -> Self {
        Self {
            reframe: true,
            ..Self::default()
        }
    }

    /// Count of `CapsChanged` packets pushed downstream, for tests asserting
    /// re-emission is suppressed when the dimensions are unchanged.
    pub fn caps_changes_emitted(&self) -> u64 {
        self.headers_emitted
    }

    async fn refine_caps(
        &mut self,
        temporal_unit: &[u8],
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let Some(info) = extract_seq_header(temporal_unit) else {
            return Ok(());
        };
        let new_caps = Caps::CompressedVideo {
            codec: VideoCodec::Av1,
            width: Dim::Fixed(info.width),
            height: Dim::Fixed(info.height),
            framerate: Rate::Any,
            colorimetry: info.colorimetry,
        };
        if self.last_emitted_caps.as_ref() != Some(&new_caps) {
            out.push(PipelinePacket::CapsChanged(new_caps.clone()))
                .await?;
            self.last_emitted_caps = Some(new_caps);
            self.headers_emitted += 1;
        }
        Ok(())
    }

    async fn emit_temporal_unit(
        &mut self,
        temporal_unit: Vec<u8>,
        timing: FrameTiming,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        self.refine_caps(&temporal_unit, out).await?;
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(temporal_unit.into_boxed_slice())),
            timing,
            self.reframed_sequence,
        );
        self.reframed_sequence += 1;
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }
}

impl AsyncElement for Av1Parse {
    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "AV1 parser",
            "Codec/Parser/Video",
            "Frames an AV1 OBU stream and reads its sequence header",
            "g2g",
        )
    }
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        let supported = Caps::CompressedVideo {
            codec: VideoCodec::Av1,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        upstream_caps.intersect(&supported)
    }

    /// Pass-through identity over AV1 of any geometry (the parser refines
    /// geometry mid-stream from the sequence header but never changes media type).
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Identity(CapsSet::one(Caps::CompressedVideo {
            codec: VideoCodec::Av1,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::CompressedVideo {
                codec: VideoCodec::Av1,
                ..
            } => {
                self.configured = true;
                Ok(ConfigureOutcome::Accepted)
            }
            _ => Err(G2gError::CapsMismatch),
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
                    let Some(bytes) = frame.domain.as_system_slice() else {
                        out.push(PipelinePacket::DataFrame(frame)).await?;
                        return Ok(());
                    };
                    let delimited_input =
                        self.reframe && (!self.framer.is_empty() || opens_temporal_unit(bytes));
                    if !delimited_input {
                        self.refine_caps(bytes, out).await?;
                        out.push(PipelinePacket::DataFrame(frame)).await?;
                        return Ok(());
                    }
                    for (temporal_unit, timing) in self.framer.push(bytes, frame.timing)? {
                        self.emit_temporal_unit(temporal_unit, timing, out).await?;
                    }
                }
                PipelinePacket::CapsChanged(c) => {
                    out.push(PipelinePacket::CapsChanged(c)).await?;
                }
                PipelinePacket::Flush => {
                    self.last_emitted_caps = None;
                    self.framer = TemporalUnitFramer::default();
                    out.push(PipelinePacket::Flush).await?;
                }
                // Segment is control: forward unchanged.
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                PipelinePacket::Eos => {
                    if let Some((temporal_unit, timing)) = self.framer.finish()? {
                        self.emit_temporal_unit(temporal_unit, timing, out).await?;
                    }
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for Av1Parse {
    fn pad_templates() -> Vec<PadTemplate> {
        let av1 = Caps::CompressedVideo {
            codec: VideoCodec::Av1,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        Vec::from([
            PadTemplate::sink(CapsSet::one(av1.clone())),
            PadTemplate::source(CapsSet::one(av1)),
        ])
    }
}

/// The fields decoded from an AV1 sequence header: the geometry the caps need
/// plus the first operating point's level/tier and the `color_config`, which
/// together are exactly the `AV1CodecConfigurationRecord` (`av1C`) header a
/// muxer writes (M773).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Av1SeqHeader {
    pub width: u32,
    pub height: u32,
    /// `seq_profile` (0..=2): bit depth + chroma support.
    pub profile: u8,
    /// `seq_level_idx[0]` (the first operating point, what `av1C` carries).
    pub level: u8,
    /// `seq_tier[0]` (0 unless the level has a tier bit).
    pub tier: u8,
    pub high_bitdepth: bool,
    pub twelve_bit: bool,
    pub monochrome: bool,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    /// `chroma_sample_position` (2 bits; 0 = unknown).
    pub chroma_sample_position: u8,
    /// The `color_config` colour description as caps colorimetry, `UNKNOWN` when
    /// the header codes none (or its tail was truncated).
    pub colorimetry: g2g_core::Colorimetry,
    /// Every other sequence-header field, which a decoder needs. `None` when the
    /// header ends before `film_grain_params_present`.
    pub detail: Option<Av1SeqHeaderDetail>,
}

/// The sequence-header fields beyond what caps and `av1C` carry (spec 5.5.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Av1SeqHeaderDetail {
    pub still_picture: bool,
    pub reduced_still_picture_header: bool,
    pub timing_info_present_flag: bool,
    pub decoder_model_info_present_flag: bool,
    pub initial_display_delay_present_flag: bool,
    pub operating_points_cnt_minus_1: u8,
    pub frame_width_bits_minus_1: u8,
    pub frame_height_bits_minus_1: u8,
    pub tools: Av1SequenceTools,
    pub color_description_present_flag: bool,
    pub color_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
    pub color_range: bool,
    pub separate_uv_delta_q: bool,
    pub film_grain_params_present: bool,
}

/// The flag block between the geometry and `color_config` (spec 5.5.1).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Av1SequenceTools {
    pub frame_id_numbers_present_flag: bool,
    pub delta_frame_id_length_minus_2: u8,
    pub additional_frame_id_length_minus_1: u8,
    pub use_128x128_superblock: bool,
    pub enable_filter_intra: bool,
    pub enable_intra_edge_filter: bool,
    pub enable_interintra_compound: bool,
    pub enable_masked_compound: bool,
    pub enable_warped_motion: bool,
    pub enable_dual_filter: bool,
    pub enable_order_hint: bool,
    pub enable_jnt_comp: bool,
    pub enable_ref_frame_mvs: bool,
    /// [`SELECT_SCREEN_CONTENT_TOOLS`] when chosen per frame.
    pub seq_force_screen_content_tools: u8,
    /// [`SELECT_INTEGER_MV`] when chosen per frame.
    pub seq_force_integer_mv: u8,
    /// `OrderHintBits` (0 when `enable_order_hint` is clear).
    pub order_hint_bits: u8,
    pub enable_superres: bool,
    pub enable_cdef: bool,
    pub enable_restoration: bool,
}

/// `seq_force_screen_content_tools` / `seq_force_integer_mv` value for "chosen
/// per frame" (spec 6.4.2).
pub const SELECT_SCREEN_CONTENT_TOOLS: u8 = 2;
pub const SELECT_INTEGER_MV: u8 = 2;

/// LEB128 (unsigned, little-endian 7-bit groups) read at `*pos`, advancing it.
/// `None` on truncation or an over-long (> 8 byte) encoding.
pub(crate) fn read_leb128(data: &[u8], pos: &mut usize) -> Option<u64> {
    let mut value = 0u64;
    for i in 0..8 {
        let byte = *data.get(*pos)?;
        *pos += 1;
        value |= u64::from(byte & 0x7f) << (7 * i);
        if byte & 0x80 == 0 {
            return Some(value);
        }
    }
    None
}

/// One OBU located in a temporal unit: its type, the whole-OBU byte range
/// (header through payload), and the payload's start offset.
struct ObuRange {
    obu_type: u8,
    // whole-OBU start: only the std-gated muxer helpers slice from it.
    #[cfg_attr(not(feature = "std"), allow(dead_code))]
    start: usize,
    payload: usize,
    end: usize,
}

/// Walk the OBUs of one temporal unit (low-overhead format: size-delimited).
/// `None` on malformed framing (a forbidden bit, a truncated size). Shared by
/// the caps parse, the muxers' `av1C` extraction, and temporal-delimiter
/// stripping.
fn obu_ranges(data: &[u8]) -> Option<Vec<ObuRange>> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        let start = pos;
        let header = data[pos];
        pos += 1;
        if header & 0x80 != 0 {
            return None; // obu_forbidden_bit must be 0
        }
        let obu_type = (header >> 3) & 0x0F;
        let has_extension = (header >> 2) & 1 == 1;
        let has_size = (header >> 1) & 1 == 1;
        if has_extension {
            pos += 1; // skip the 1-byte obu_extension_header
            if pos > data.len() {
                return None;
            }
        }
        let size = if has_size {
            read_leb128(data, &mut pos)? as usize
        } else {
            data.len().checked_sub(pos)? // no size: OBU runs to the end
        };
        let payload = pos;
        let end = pos.checked_add(size)?;
        if end > data.len() {
            return None;
        }
        out.push(ObuRange {
            obu_type,
            start,
            payload,
            end,
        });
        pos = end;
    }
    Some(out)
}

/// One OBU of a temporal unit: its type and payload (past header and size).
// only the Vulkan decoder reads every field
#[cfg_attr(not(feature = "vulkan-video"), allow(dead_code))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Obu<'a> {
    pub obu_type: u8,
    pub payload: &'a [u8],
}

/// The OBUs of one temporal unit, empty when its framing is malformed.
#[cfg_attr(not(feature = "vulkan-video"), allow(dead_code))]
pub(crate) fn obus(data: &[u8]) -> Vec<Obu<'_>> {
    obu_ranges(data)
        .unwrap_or_default()
        .iter()
        .map(|o| Obu {
            obu_type: o.obu_type,
            payload: &data[o.payload..o.end],
        })
        .collect()
}

/// Whether a temporal unit carries a sequence-header OBU. The MP4 demuxers use
/// it to decide if the `av1C` configOBUs need prepending to the first sample
/// (M779).
pub(crate) fn has_sequence_header(data: &[u8]) -> bool {
    obu_ranges(data).is_some_and(|obus| obus.iter().any(|o| o.obu_type == OBU_SEQUENCE_HEADER))
}

// far above any real temporal unit
const MAX_TEMPORAL_UNIT_BYTES: usize = 16 * 1024 * 1024;

// a byte-stream chunk boundary can land on a delimiter by chance
const TEMPORAL_UNIT_ALIGNED_CONFIRMATIONS: u8 = 2;

// spec 4.10.5: `leb128()` codes at most 8 bytes
const LEB128_MAX_BYTES: usize = 8;

// seq_profile 3..=7 are reserved
const MAX_SEQ_PROFILE: u8 = 2;

fn obu_type(header: u8) -> u8 {
    (header >> 3) & 0x0F
}

fn opens_temporal_unit(data: &[u8]) -> bool {
    data.first()
        .is_some_and(|&header| obu_type(header) == OBU_TEMPORAL_DELIMITER)
}

// Ok(None) until all of the OBU's bytes have arrived
fn delimited_obu_at(data: &[u8], start: usize) -> Result<Option<ObuRange>, G2gError> {
    let Some(&header) = data.get(start) else {
        return Ok(None);
    };
    let has_extension = (header >> 2) & 1 == 1;
    let has_size = (header >> 1) & 1 == 1;
    // the low-overhead format codes every OBU's size
    if header & 0x80 != 0 || !has_size {
        return Err(G2gError::CapsMismatch);
    }
    let size_start = start + 1 + usize::from(has_extension);
    let mut payload = size_start;
    let Some(size) = read_leb128(data, &mut payload) else {
        let size_truncated = data.len().saturating_sub(size_start) < LEB128_MAX_BYTES;
        return if size_truncated {
            Ok(None)
        } else {
            Err(G2gError::CapsMismatch)
        };
    };
    let size = usize::try_from(size)
        .ok()
        .filter(|&size| size <= MAX_TEMPORAL_UNIT_BYTES)
        .ok_or(G2gError::CapsMismatch)?;
    let end = payload.checked_add(size).ok_or(G2gError::CapsMismatch)?;
    Ok((end <= data.len()).then_some(ObuRange {
        obu_type: obu_type(header),
        start,
        payload,
        end,
    }))
}

// a raw .obu file opens on an empty temporal delimiter, then a sequence header
pub(crate) fn opens_obu_stream(data: &[u8]) -> bool {
    let Ok(Some(delimiter)) = delimited_obu_at(data, 0) else {
        return false;
    };
    let Ok(Some(sequence_header)) = delimited_obu_at(data, delimiter.end) else {
        return false;
    };
    let reserved_bits_clear =
        data[delimiter.start] & 1 == 0 && data[sequence_header.start] & 1 == 0;
    delimiter.obu_type == OBU_TEMPORAL_DELIMITER
        && delimiter.payload == delimiter.end
        && sequence_header.obu_type == OBU_SEQUENCE_HEADER
        && reserved_bits_clear
        && parse_complete_sequence_header(&data[sequence_header.payload..sequence_header.end])
            .is_some_and(|header| header.profile <= MAX_SEQ_PROFILE)
}

// the last unit waits for the next delimiter unless input proved one unit per buffer
#[derive(Debug, Default)]
struct TemporalUnitFramer {
    // starts on a temporal delimiter whenever it is not empty
    pending: Vec<u8>,
    // offset of the first OBU in `pending` not yet complete
    walked: usize,
    // each delimiter in `pending` and the timing of the buffer it completed in
    unit_starts: Vec<(usize, FrameTiming)>,
    // consecutive buffers that began on a delimiter
    aligned_runs: u8,
}

type FramedTemporalUnit = (Vec<u8>, FrameTiming);

impl TemporalUnitFramer {
    fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    fn push(
        &mut self,
        bytes: &[u8],
        timing: FrameTiming,
    ) -> Result<Vec<FramedTemporalUnit>, G2gError> {
        let boundary = self.pending.len();
        self.pending.extend_from_slice(bytes);
        while let Some(obu) = delimited_obu_at(&self.pending, self.walked)? {
            if obu.obu_type == OBU_TEMPORAL_DELIMITER {
                self.unit_starts.push((obu.start, timing));
            }
            self.walked = obu.end;
        }
        // the first buffer opens a unit by construction
        if boundary > 0 {
            let opened_a_unit = self.unit_starts.iter().any(|&(start, _)| start == boundary);
            self.aligned_runs = if opened_a_unit {
                self.aligned_runs.saturating_add(1)
            } else {
                0
            };
        }
        let aligned = self.aligned_runs >= TEMPORAL_UNIT_ALIGNED_CONFIRMATIONS
            && self.walked == self.pending.len();
        let complete_through = if aligned {
            self.pending.len()
        } else {
            self.unit_starts.last().map_or(0, |&(start, _)| start)
        };
        let temporal_units = self.split_off(complete_through);
        if self.pending.len() > MAX_TEMPORAL_UNIT_BYTES {
            return Err(G2gError::CapsMismatch);
        }
        Ok(temporal_units)
    }

    fn finish(&mut self) -> Result<Option<FramedTemporalUnit>, G2gError> {
        if self.walked != self.pending.len() {
            return Err(G2gError::CapsMismatch);
        }
        Ok(self.split_off(self.pending.len()).pop())
    }

    fn split_off(&mut self, complete_through: usize) -> Vec<FramedTemporalUnit> {
        let retained_from = self
            .unit_starts
            .partition_point(|&(start, _)| start < complete_through);
        let complete_starts: Vec<(usize, FrameTiming)> =
            self.unit_starts.drain(..retained_from).collect();
        let retained = self.pending.split_off(complete_through);
        let complete = core::mem::replace(&mut self.pending, retained);
        for (start, _) in &mut self.unit_starts {
            *start -= complete_through;
        }
        self.walked -= complete_through;
        let unit_ends = complete_starts
            .iter()
            .skip(1)
            .map(|&(start, _)| start)
            .chain(core::iter::once(complete.len()));
        complete_starts
            .iter()
            .zip(unit_ends)
            .map(|(&(start, timing), end)| (complete[start..end].to_vec(), timing))
            .collect()
    }
}

/// Walk the OBUs of one temporal unit and parse the first sequence-header OBU.
/// `None` if none is present or the framing is malformed.
fn extract_seq_header(data: &[u8]) -> Option<Av1SeqHeader> {
    let obus = obu_ranges(data)?;
    obus.iter()
        .filter(|o| o.obu_type == OBU_SEQUENCE_HEADER)
        .find_map(|o| parse_sequence_header(&data[o.payload..o.end]))
}

/// The first sequence-header OBU of a temporal unit, whole (header byte through
/// payload), as a muxer's `av1C` `configOBUs` carries it verbatim. Paired with
/// its parse, so a caller gets consistent record fields and config bytes.
#[cfg(feature = "std")]
pub(crate) fn seq_header_obu(data: &[u8]) -> Option<(Av1SeqHeader, &[u8])> {
    let obus = obu_ranges(data)?;
    obus.iter()
        .filter(|o| o.obu_type == OBU_SEQUENCE_HEADER)
        .find_map(|o| {
            // Strict: the muxer's av1C must carry real color_config fields.
            parse_sequence_header_at(&data[o.payload..o.end], true)
                .map(|h| (h, &data[o.start..o.end]))
        })
}

/// `OBU_TEMPORAL_DELIMITER` type.
const OBU_TEMPORAL_DELIMITER: u8 = 2;
/// `OBU_FRAME_HEADER` / `OBU_FRAME` types (the ones opening a coded frame).
const OBU_FRAME_HEADER: u8 = 3;
const OBU_FRAME: u8 = 6;

/// A temporal unit with its temporal-delimiter OBUs stripped, as the ISOBMFF /
/// Matroska AV1 mappings store samples (the container frames the units, so the
/// delimiter is redundant). Unparseable framing passes through unchanged.
#[cfg(feature = "std")]
pub(crate) fn strip_temporal_delimiters(data: &[u8]) -> Vec<u8> {
    let Some(obus) = obu_ranges(data) else {
        return data.to_vec();
    };
    let mut out = Vec::with_capacity(data.len());
    for o in &obus {
        if o.obu_type != OBU_TEMPORAL_DELIMITER {
            out.extend_from_slice(&data[o.start..o.end]);
        }
    }
    out
}

/// Whether a temporal unit is a random-access (sync) sample: it carries a
/// sequence header (encoders re-send it on key frames), or its first frame OBU
/// codes `frame_type == KEY` with `show_existing_frame == 0`. Malformed framing
/// is not a sync point. Used for a muxer's keyframe flag and for the MPEG-TS
/// demuxer's seek resume point.
pub(crate) fn av1_keyframe(data: &[u8]) -> bool {
    let Some(obus) = obu_ranges(data) else {
        return false;
    };
    if obus.iter().any(|o| o.obu_type == OBU_SEQUENCE_HEADER) {
        return true;
    }
    for o in &obus {
        if o.obu_type == OBU_FRAME || o.obu_type == OBU_FRAME_HEADER {
            // show_existing_frame(1), frame_type(2): KEY = 0. (A
            // reduced-still-picture stream omits these bits, but such a stream
            // is a single still whose unit carries the sequence header above.)
            let Some(&b0) = data.get(o.payload) else {
                return false;
            };
            let show_existing = b0 >> 7 == 1;
            let frame_type = (b0 >> 5) & 3;
            return !show_existing && frame_type == 0;
        }
    }
    false
}

/// Parse a sequence-header OBU payload. The geometry / level prefix is
/// required; the tail through `film_grain_params_present` is best-effort for
/// the caps parse (a truncated header keeps the 8-bit 4:2:0 defaults) and
/// required when `strict` (the muxer's `av1C` and a decoder need real fields).
fn parse_sequence_header(payload: &[u8]) -> Option<Av1SeqHeader> {
    parse_sequence_header_at(payload, false)
}

/// [`parse_sequence_header`] requiring the whole header, so `detail` is set.
pub(crate) fn parse_complete_sequence_header(payload: &[u8]) -> Option<Av1SeqHeader> {
    parse_sequence_header_at(payload, true)
}

fn parse_sequence_header_at(payload: &[u8], strict: bool) -> Option<Av1SeqHeader> {
    let mut br = BitReader::new(payload);
    let seq_profile = br.read_bits(3)? as u8;
    let still_picture = br.read_bit()? == 1;
    let reduced_still_picture_header = br.read_bit()?;

    let mut level = 0u8;
    let mut tier = 0u8;
    let mut timing_info_present = 0u32;
    let mut decoder_model_info_present = 0u32;
    let mut initial_display_delay_present = 0u32;
    let mut operating_points_cnt_minus_1 = 0u32;
    if reduced_still_picture_header == 1 {
        level = br.read_bits(5)? as u8;
    } else {
        timing_info_present = br.read_bit()?;
        let mut buffer_delay_length = 0u32;
        if timing_info_present == 1 {
            br.read_bits(32)?; // num_units_in_display_tick
            br.read_bits(32)?; // time_scale
            if br.read_bit()? == 1 {
                read_uvlc(&mut br)?; // num_ticks_per_picture_minus_1
            }
            decoder_model_info_present = br.read_bit()?;
            if decoder_model_info_present == 1 {
                buffer_delay_length = br.read_bits(5)? + 1; // buffer_delay_length_minus_1
                br.read_bits(32)?; // num_units_in_decoding_tick
                br.read_bits(5)?; // buffer_removal_time_length_minus_1
                br.read_bits(5)?; // frame_presentation_time_length_minus_1
            }
        }
        initial_display_delay_present = br.read_bit()?;
        operating_points_cnt_minus_1 = br.read_bits(5)?;
        for i in 0..=operating_points_cnt_minus_1 {
            br.read_bits(12)?; // operating_point_idc[i]
            let seq_level_idx = br.read_bits(5)?;
            let seq_tier = if seq_level_idx > 7 {
                br.read_bit()? // seq_tier[i]
            } else {
                0
            };
            if i == 0 {
                // The first operating point is what `av1C` describes.
                level = seq_level_idx as u8;
                tier = seq_tier as u8;
            }
            if decoder_model_info_present == 1 && br.read_bit()? == 1 {
                // operating_parameters_info(i): two f(n) delays + a flag.
                br.read_bits(buffer_delay_length)?; // decoder_buffer_delay
                br.read_bits(buffer_delay_length)?; // encoder_buffer_delay
                br.read_bit()?; // low_delay_mode_flag
            }
            if initial_display_delay_present == 1 && br.read_bit()? == 1 {
                br.read_bits(4)?; // initial_display_delay_minus_1[i]
            }
        }
    }

    let frame_width_bits_minus_1 = br.read_bits(4)?;
    let frame_height_bits_minus_1 = br.read_bits(4)?;
    let max_frame_width = br.read_bits(frame_width_bits_minus_1 + 1)? + 1;
    let max_frame_height = br.read_bits(frame_height_bits_minus_1 + 1)? + 1;

    let mut header = Av1SeqHeader {
        width: max_frame_width,
        height: max_frame_height,
        profile: seq_profile,
        level,
        tier,
        // 8-bit 4:2:0 defaults; overwritten from color_config below.
        high_bitdepth: false,
        twelve_bit: false,
        monochrome: false,
        subsampling_x: true,
        subsampling_y: true,
        chroma_sample_position: 0,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        detail: None,
    };
    let Some((tools, color)) =
        parse_color_config(&mut br, reduced_still_picture_header, seq_profile)
    else {
        // Truncated tail: the caps parse only needs the geometry.
        return (!strict).then_some(header);
    };
    header.high_bitdepth = color.high_bitdepth;
    header.twelve_bit = color.twelve_bit;
    header.monochrome = color.monochrome;
    header.subsampling_x = color.subsampling_x;
    header.subsampling_y = color.subsampling_y;
    header.chroma_sample_position = color.chroma_sample_position;
    header.colorimetry = color.colorimetry();
    let mut read_trailer = || {
        // the monochrome branch of color_config codes no separate_uv_delta_q
        let separate_uv_delta_q = !color.monochrome && br.read_bit()? == 1;
        let film_grain_params_present = br.read_bit()? == 1;
        Some((separate_uv_delta_q, film_grain_params_present))
    };
    let Some((separate_uv_delta_q, film_grain_params_present)) = read_trailer() else {
        return (!strict).then_some(header);
    };
    header.detail = Some(Av1SeqHeaderDetail {
        still_picture,
        reduced_still_picture_header: reduced_still_picture_header == 1,
        timing_info_present_flag: timing_info_present == 1,
        decoder_model_info_present_flag: decoder_model_info_present == 1,
        initial_display_delay_present_flag: initial_display_delay_present == 1,
        operating_points_cnt_minus_1: operating_points_cnt_minus_1 as u8,
        frame_width_bits_minus_1: frame_width_bits_minus_1 as u8,
        frame_height_bits_minus_1: frame_height_bits_minus_1 as u8,
        tools,
        color_description_present_flag: color.color_description_present_flag,
        color_primaries: color.color_primaries,
        transfer_characteristics: color.transfer_characteristics,
        matrix_coefficients: color.matrix_coefficients,
        color_range: color.color_range,
        separate_uv_delta_q,
        film_grain_params_present,
    });
    Some(header)
}

/// What `color_config` (spec 5.5.2) codes: the sample layout the geometry parse
/// folds into [`Av1SeqHeader`], and the colour description.
struct ColorConfig {
    high_bitdepth: bool,
    twelve_bit: bool,
    monochrome: bool,
    subsampling_x: bool,
    subsampling_y: bool,
    chroma_sample_position: u8,
    color_description_present_flag: bool,
    color_primaries: u8,
    transfer_characteristics: u8,
    matrix_coefficients: u8,
    color_range: bool,
}

impl ColorConfig {
    fn colorimetry(&self) -> g2g_core::Colorimetry {
        g2g_core::Colorimetry::from_cicp(
            self.color_primaries,
            self.transfer_characteristics,
            self.matrix_coefficients,
            self.color_range,
        )
    }
}

/// Parse the flag block between the geometry and `color_config` (spec 5.5.1)
/// and `color_config` itself (5.5.2) up to `separate_uv_delta_q`. `None` on
/// truncation.
fn parse_color_config(
    br: &mut BitReader,
    reduced_still_picture_header: u32,
    seq_profile: u8,
) -> Option<(Av1SequenceTools, ColorConfig)> {
    let mut tools = Av1SequenceTools {
        seq_force_screen_content_tools: SELECT_SCREEN_CONTENT_TOOLS,
        seq_force_integer_mv: SELECT_INTEGER_MV,
        ..Av1SequenceTools::default()
    };
    if reduced_still_picture_header == 0 && br.read_bit()? == 1 {
        tools.frame_id_numbers_present_flag = true;
        tools.delta_frame_id_length_minus_2 = br.read_bits(4)? as u8;
        tools.additional_frame_id_length_minus_1 = br.read_bits(3)? as u8;
    }
    tools.use_128x128_superblock = br.read_bit()? == 1;
    tools.enable_filter_intra = br.read_bit()? == 1;
    tools.enable_intra_edge_filter = br.read_bit()? == 1;
    if reduced_still_picture_header == 0 {
        tools.enable_interintra_compound = br.read_bit()? == 1;
        tools.enable_masked_compound = br.read_bit()? == 1;
        tools.enable_warped_motion = br.read_bit()? == 1;
        tools.enable_dual_filter = br.read_bit()? == 1;
        tools.enable_order_hint = br.read_bit()? == 1;
        if tools.enable_order_hint {
            tools.enable_jnt_comp = br.read_bit()? == 1;
            tools.enable_ref_frame_mvs = br.read_bit()? == 1;
        }
        // seq_choose_screen_content_tools -> SELECT, else an explicit bit.
        if br.read_bit()? == 0 {
            tools.seq_force_screen_content_tools = br.read_bit()? as u8;
        }
        // seq_choose_integer_mv -> SELECT, else an explicit bit.
        if tools.seq_force_screen_content_tools > 0 && br.read_bit()? == 0 {
            tools.seq_force_integer_mv = br.read_bit()? as u8;
        }
        if tools.enable_order_hint {
            tools.order_hint_bits = br.read_bits(3)? as u8 + 1; // order_hint_bits_minus_1
        }
    }
    tools.enable_superres = br.read_bit()? == 1;
    tools.enable_cdef = br.read_bit()? == 1;
    tools.enable_restoration = br.read_bit()? == 1;

    // color_config (spec 5.5.2).
    let high_bitdepth = br.read_bit()? == 1;
    let twelve_bit = if seq_profile == 2 && high_bitdepth {
        br.read_bit()? == 1
    } else {
        false
    };
    let monochrome = if seq_profile == 1 {
        false
    } else {
        br.read_bit()? == 1
    };
    // Absent a color_description the three fields are CICP unspecified.
    let color_description_present_flag = br.read_bit()? == 1;
    let (mut primaries, mut transfer, mut matrix) =
        (CICP_UNSPECIFIED, CICP_UNSPECIFIED, CICP_UNSPECIFIED);
    if color_description_present_flag {
        primaries = br.read_bits(8)?;
        transfer = br.read_bits(8)?;
        matrix = br.read_bits(8)?;
    }
    let mut color = ColorConfig {
        high_bitdepth,
        twelve_bit,
        monochrome,
        subsampling_x: true,
        subsampling_y: true,
        chroma_sample_position: 0,
        color_description_present_flag,
        color_primaries: primaries as u8,
        transfer_characteristics: transfer as u8,
        matrix_coefficients: matrix as u8,
        color_range: false,
    };
    // CP_BT_709 / TC_SRGB / MC_IDENTITY: the RGB case with no subsampling.
    let identity_matrix = matrix == 0 && primaries == 1 && transfer == 13;
    if monochrome {
        color.color_range = br.read_bit()? == 1;
        return Some((tools, color));
    }
    if identity_matrix {
        // The GBR case codes no range bit: it is full by definition.
        color.color_range = true;
        color.subsampling_x = false;
        color.subsampling_y = false;
        return Some((tools, color));
    }
    color.color_range = br.read_bit()? == 1;
    let (ssx, ssy) = match seq_profile {
        0 => (true, true),
        1 => (false, false),
        _ => {
            if twelve_bit {
                let ssx = br.read_bit()? == 1;
                let ssy = if ssx { br.read_bit()? == 1 } else { false };
                (ssx, ssy)
            } else {
                (true, false)
            }
        }
    };
    color.subsampling_x = ssx;
    color.subsampling_y = ssy;
    if ssx && ssy {
        color.chroma_sample_position = br.read_bits(2)? as u8;
    }
    Some((tools, color))
}

/// AV1 unsigned variable-length code (spec `uvlc()`): count leading zeros, then
/// read that many value bits. `>= 32` leading zeros saturates with no value read.
pub(crate) fn read_uvlc(br: &mut BitReader) -> Option<u32> {
    let mut leading_zeros = 0u32;
    loop {
        if br.read_bit()? == 1 {
            break;
        }
        leading_zeros += 1;
        if leading_zeros >= 32 {
            return Some(u32::MAX);
        }
    }
    let value = br.read_bits(leading_zeros)?;
    Some(value + (1u32 << leading_zeros) - 1)
}

/// Fuzzing entry: walk an AV1 temporal unit's OBUs and parse the sequence
/// header (the hand-written LEB128 / bit-reader path). Exposed only under
/// `--cfg fuzzing` (set by cargo-fuzz) so the normal public API is unchanged.
#[cfg(fuzzing)]
pub fn fuzz_parse(data: &[u8]) {
    let _ = extract_seq_header(data);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annexb::BitWriter;
    use crate::testutil::{frame_with_bytes, CollectSink};
    use alloc::vec;

    fn leb128(mut v: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let mut byte = (v & 0x7f) as u8;
            v >>= 7;
            if v != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if v == 0 {
                break;
            }
        }
        out
    }

    /// Bits needed to represent `v` (at least 1).
    fn bits_for(v: u32) -> u32 {
        if v == 0 {
            1
        } else {
            32 - v.leading_zeros()
        }
    }

    /// A non-reduced sequence-header OBU payload for `width` x `height` at
    /// `profile`: one operating point, no timing / decoder-model / display-delay.
    fn seq_header_payload(width: u32, height: u32, profile: u8) -> Vec<u8> {
        let mut w = BitWriter::default();
        w.write_bits(profile as u32, 3); // seq_profile
        w.write_bit(0); // still_picture
        w.write_bit(0); // reduced_still_picture_header
        w.write_bit(0); // timing_info_present_flag
        w.write_bit(0); // initial_display_delay_present_flag
        w.write_bits(0, 5); // operating_points_cnt_minus_1
        w.write_bits(0, 12); // operating_point_idc[0]
        w.write_bits(0, 5); // seq_level_idx[0] (<= 7, no tier)
        let (wm1, hm1) = (width - 1, height - 1);
        let (nw, nh) = (bits_for(wm1), bits_for(hm1));
        w.write_bits(nw - 1, 4); // frame_width_bits_minus_1
        w.write_bits(nh - 1, 4); // frame_height_bits_minus_1
        w.write_bits(wm1, nw); // max_frame_width_minus_1
        w.write_bits(hm1, nh); // max_frame_height_minus_1
        w.align_to_byte();
        w.into_bytes()
    }

    /// A reduced-still-picture sequence-header OBU payload.
    fn reduced_seq_header_payload(width: u32, height: u32, profile: u8) -> Vec<u8> {
        let mut w = BitWriter::default();
        w.write_bits(profile as u32, 3); // seq_profile
        w.write_bit(1); // still_picture
        w.write_bit(1); // reduced_still_picture_header
        w.write_bits(0, 5); // seq_level_idx[0]
        let (wm1, hm1) = (width - 1, height - 1);
        let (nw, nh) = (bits_for(wm1), bits_for(hm1));
        w.write_bits(nw - 1, 4);
        w.write_bits(nh - 1, 4);
        w.write_bits(wm1, nw);
        w.write_bits(hm1, nh);
        w.align_to_byte();
        w.into_bytes()
    }

    /// Wrap `payload` as a size-delimited OBU of `obu_type`.
    fn obu(obu_type: u8, payload: &[u8]) -> Vec<u8> {
        let header = (obu_type << 3) | 0x02; // ext_flag=0, has_size_field=1
        let mut out = vec![header];
        out.extend_from_slice(&leb128(payload.len() as u64));
        out.extend_from_slice(payload);
        out
    }

    /// A temporal unit: a temporal-delimiter OBU then a sequence-header OBU.
    fn temporal_unit(width: u32, height: u32, profile: u8) -> Vec<u8> {
        let mut tu = obu(2, &[]); // OBU_TEMPORAL_DELIMITER
        tu.extend_from_slice(&obu(
            OBU_SEQUENCE_HEADER,
            &seq_header_payload(width, height, profile),
        ));
        tu
    }

    #[test]
    fn recovers_1920x1080_after_temporal_delimiter() {
        let info =
            extract_seq_header(&temporal_unit(1920, 1080, 0)).expect("seq header must parse");
        assert_eq!((info.width, info.height), (1920, 1080));
        assert_eq!(info.profile, 0);
    }

    #[test]
    fn recovers_non_power_of_two_dimensions() {
        let info = extract_seq_header(&temporal_unit(1280, 720, 2)).expect("seq header must parse");
        assert_eq!((info.width, info.height), (1280, 720));
        assert_eq!(info.profile, 2);
    }

    #[test]
    fn parses_reduced_still_picture_header() {
        let obu = obu(
            OBU_SEQUENCE_HEADER,
            &reduced_seq_header_payload(800, 600, 1),
        );
        let info = extract_seq_header(&obu).expect("reduced seq header must parse");
        assert_eq!((info.width, info.height), (800, 600));
        assert_eq!(info.profile, 1);
    }

    #[test]
    fn ignores_a_frame_obu_without_a_sequence_header() {
        // A temporal delimiter + a frame OBU (type 6) carries no sequence header.
        let mut tu = obu(2, &[]);
        tu.extend_from_slice(&obu(6, &[0xAA, 0xBB, 0xCC]));
        assert!(extract_seq_header(&tu).is_none());
    }

    #[test]
    fn rejects_forbidden_bit_and_empty() {
        assert!(
            extract_seq_header(&[0x80]).is_none(),
            "obu_forbidden_bit set"
        );
        assert!(extract_seq_header(&[]).is_none());
    }

    #[test]
    fn read_leb128_decodes_multibyte() {
        // 300 = 0xAC 0x02 in LEB128.
        let mut pos = 0;
        assert_eq!(read_leb128(&[0xAC, 0x02], &mut pos), Some(300));
        assert_eq!(pos, 2);
    }

    // -- Element-level tests (drive Av1Parse::process directly) -------------

    fn av1_caps() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::Av1,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        }
    }

    #[tokio::test]
    async fn emits_caps_changed_before_first_data_frame() {
        let mut parse = Av1Parse::new();
        parse.configure_pipeline(&av1_caps()).unwrap();
        let mut sink = CollectSink::default();

        let frame = frame_with_bytes(0, temporal_unit(1920, 1080, 0));
        parse
            .process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();

        assert_eq!(sink.packets.len(), 2, "expected CapsChanged then DataFrame");
        match &sink.packets[0] {
            PipelinePacket::CapsChanged(Caps::CompressedVideo { width, height, .. }) => {
                assert_eq!(*width, Dim::Fixed(1920));
                assert_eq!(*height, Dim::Fixed(1080));
            }
            other => panic!("expected CapsChanged first, got {other:?}"),
        }
        assert!(matches!(sink.packets[1], PipelinePacket::DataFrame(_)));
        assert_eq!(parse.caps_changes_emitted(), 1);
    }

    #[tokio::test]
    async fn does_not_re_emit_caps_when_unchanged() {
        let mut parse = Av1Parse::new();
        parse.configure_pipeline(&av1_caps()).unwrap();
        let mut sink = CollectSink::default();

        for seq in 0..3 {
            let frame = frame_with_bytes(seq, temporal_unit(1280, 720, 0));
            parse
                .process(PipelinePacket::DataFrame(frame), &mut sink)
                .await
                .unwrap();
        }

        let caps_count = sink
            .packets
            .iter()
            .filter(|p| matches!(p, PipelinePacket::CapsChanged(_)))
            .count();
        assert_eq!(
            caps_count, 1,
            "CapsChanged fires once for identical dimensions"
        );
    }

    #[tokio::test]
    async fn re_emits_caps_on_resolution_change() {
        let mut parse = Av1Parse::new();
        parse.configure_pipeline(&av1_caps()).unwrap();
        let mut sink = CollectSink::default();

        parse
            .process(
                PipelinePacket::DataFrame(frame_with_bytes(0, temporal_unit(1280, 720, 0))),
                &mut sink,
            )
            .await
            .unwrap();
        parse
            .process(
                PipelinePacket::DataFrame(frame_with_bytes(1, temporal_unit(1920, 1080, 0))),
                &mut sink,
            )
            .await
            .unwrap();

        let widths: Vec<Dim> = sink
            .packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::CapsChanged(Caps::CompressedVideo { width, .. }) => {
                    Some(width.clone())
                }
                _ => None,
            })
            .collect();
        assert_eq!(widths, vec![Dim::Fixed(1280), Dim::Fixed(1920)]);
        assert_eq!(parse.caps_changes_emitted(), 2);
    }

    fn delimited_unit(tag: u8, with_sequence_header: bool) -> Vec<u8> {
        let mut unit = obu(OBU_TEMPORAL_DELIMITER, &[]);
        if with_sequence_header {
            unit.extend(obu(OBU_SEQUENCE_HEADER, &seq_header_payload(64, 48, 0)));
        }
        unit.extend(obu(OBU_FRAME, &[tag; 7]));
        unit
    }

    fn delimited_units(count: u8) -> Vec<Vec<u8>> {
        (0..count)
            .map(|tag| delimited_unit(tag, tag == 0))
            .collect()
    }

    fn data_payloads(sink: &CollectSink) -> Vec<Vec<u8>> {
        sink.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => f.domain.as_system_slice().map(<[u8]>::to_vec),
                _ => None,
            })
            .collect()
    }

    async fn push_each(parse: &mut Av1Parse, buffers: &[Vec<u8>], sink: &mut CollectSink) {
        for (sequence, buffer) in buffers.iter().enumerate() {
            parse
                .process(
                    PipelinePacket::DataFrame(frame_with_bytes(sequence as u64, buffer.clone())),
                    sink,
                )
                .await
                .unwrap();
        }
    }

    #[tokio::test]
    async fn reframing_splits_a_chunked_obu_stream_into_temporal_units() {
        let units = delimited_units(4);
        let stream = units.concat();
        // lands on no unit start
        let chunk_len = 9;
        let chunks: Vec<Vec<u8>> = stream.chunks(chunk_len).map(<[u8]>::to_vec).collect();
        let mut parse = Av1Parse::reframing();
        parse.configure_pipeline(&av1_caps()).unwrap();
        let mut sink = CollectSink::default();

        push_each(&mut parse, &chunks, &mut sink).await;
        assert_eq!(
            data_payloads(&sink),
            units[..units.len() - 1],
            "every unit but the last is out, whole"
        );
        parse.process(PipelinePacket::Eos, &mut sink).await.unwrap();
        assert_eq!(data_payloads(&sink), units, "Eos flushes the last one");
        assert!(matches!(
            sink.packets[0],
            PipelinePacket::CapsChanged(Caps::CompressedVideo {
                width: Dim::Fixed(64),
                height: Dim::Fixed(48),
                ..
            })
        ));
    }

    #[tokio::test]
    async fn reframing_stops_holding_once_the_input_proves_unit_aligned() {
        let units = delimited_units(4);
        let mut parse = Av1Parse::reframing();
        parse.configure_pipeline(&av1_caps()).unwrap();
        let mut sink = CollectSink::default();

        push_each(&mut parse, &units, &mut sink).await;
        assert_eq!(
            data_payloads(&sink),
            units,
            "a unit-per-buffer upstream is not held back a frame"
        );
    }

    #[tokio::test]
    async fn reframing_passes_undelimited_units_straight_through() {
        // the container mappings strip the delimiters
        let units: Vec<Vec<u8>> = (0..3).map(|tag| obu(OBU_FRAME, &[tag; 7])).collect();
        let mut parse = Av1Parse::reframing();
        parse.configure_pipeline(&av1_caps()).unwrap();
        let mut sink = CollectSink::default();

        push_each(&mut parse, &units, &mut sink).await;
        assert_eq!(data_payloads(&sink), units);
    }

    #[tokio::test]
    async fn reframing_refuses_an_obu_size_past_the_cap() {
        let mut stream = obu(OBU_TEMPORAL_DELIMITER, &[]);
        stream.push((OBU_FRAME << 3) | 0x02);
        stream.extend(leb128(MAX_TEMPORAL_UNIT_BYTES as u64 + 1));
        let mut parse = Av1Parse::reframing();
        parse.configure_pipeline(&av1_caps()).unwrap();
        let mut sink = CollectSink::default();

        let result = parse
            .process(
                PipelinePacket::DataFrame(frame_with_bytes(0, stream)),
                &mut sink,
            )
            .await;
        assert_eq!(result, Err(G2gError::CapsMismatch));
    }

    #[tokio::test]
    async fn rejects_non_av1_caps_in_intercept() {
        let parse = Av1Parse::new();
        let vp9 = Caps::CompressedVideo {
            codec: VideoCodec::Vp9,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        assert_eq!(parse.intercept_caps(&vp9), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn caps_constraint_is_identity_av1_any() {
        let parse = Av1Parse::new();
        let c = parse.caps_constraint_as_transform();
        match c {
            CapsConstraint::Identity(set) => {
                assert_eq!(
                    set.alternatives(),
                    &[Caps::CompressedVideo {
                        codec: VideoCodec::Av1,
                        width: Dim::Any,
                        height: Dim::Any,
                        framerate: Rate::Any,
                        colorimetry: g2g_core::Colorimetry::UNKNOWN
                    }]
                );
            }
            _ => panic!("expected Identity"),
        }
    }
}
