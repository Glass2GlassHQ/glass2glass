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

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket, Rate, VideoCodec,
};

use crate::annexb::BitReader;

/// `OBU_SEQUENCE_HEADER` type, the only OBU carrying frame geometry.
const OBU_SEQUENCE_HEADER: u8 = 1;

#[derive(Debug, Default)]
pub struct Av1Parse {
    configured: bool,
    last_emitted_caps: Option<Caps>,
    headers_emitted: u64,
}

impl Av1Parse {
    pub fn new() -> Self {
        Self::default()
    }

    /// Count of `CapsChanged` packets pushed downstream, for tests asserting
    /// re-emission is suppressed when the dimensions are unchanged.
    pub fn caps_changes_emitted(&self) -> u64 {
        self.headers_emitted
    }
}

impl AsyncElement for Av1Parse {
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
                    if let g2g_core::MemoryDomain::System(slice) = &frame.domain {
                        if let Some(info) = extract_seq_header(slice.as_slice()) {
                            let new_caps = Caps::CompressedVideo {
                                codec: VideoCodec::Av1,
                                width: Dim::Fixed(info.width),
                                height: Dim::Fixed(info.height),
                                framerate: Rate::Any,
                            };
                            if self.last_emitted_caps.as_ref() != Some(&new_caps) {
                                out.push(PipelinePacket::CapsChanged(new_caps.clone()))
                                    .await?;
                                self.last_emitted_caps = Some(new_caps);
                                self.headers_emitted += 1;
                            }
                        }
                    }
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    out.push(PipelinePacket::CapsChanged(c)).await?;
                }
                PipelinePacket::Flush => {
                    self.last_emitted_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                // Segment is control: forward unchanged.
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
}

impl PadTemplates for Av1Parse {
    fn pad_templates() -> Vec<PadTemplate> {
        let av1 = Caps::CompressedVideo {
            codec: VideoCodec::Av1,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
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
}

/// LEB128 (unsigned, little-endian 7-bit groups) read at `*pos`, advancing it.
/// `None` on truncation or an over-long (> 8 byte) encoding.
fn read_leb128(data: &[u8], pos: &mut usize) -> Option<u64> {
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
#[cfg(feature = "std")]
const OBU_TEMPORAL_DELIMITER: u8 = 2;
/// `OBU_FRAME_HEADER` / `OBU_FRAME` types (the ones opening a coded frame).
#[cfg(feature = "std")]
const OBU_FRAME_HEADER: u8 = 3;
#[cfg(feature = "std")]
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

/// Whether a temporal unit is a random-access (sync) sample for a muxer's
/// keyframe flag: it carries a sequence header (encoders re-send it on key
/// frames), or its first frame OBU codes `frame_type == KEY` with
/// `show_existing_frame == 0`. Malformed framing is not a sync point.
#[cfg(feature = "std")]
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
/// required; the tail through `color_config` is best-effort for the caps parse
/// (a truncated header keeps the 8-bit 4:2:0 defaults) and required when
/// `strict` (the muxer's `av1C` needs real fields).
fn parse_sequence_header(payload: &[u8]) -> Option<Av1SeqHeader> {
    parse_sequence_header_at(payload, false)
}

fn parse_sequence_header_at(payload: &[u8], strict: bool) -> Option<Av1SeqHeader> {
    let mut br = BitReader::new(payload);
    let seq_profile = br.read_bits(3)? as u8;
    let _still_picture = br.read_bit()?;
    let reduced_still_picture_header = br.read_bit()?;

    let mut level = 0u8;
    let mut tier = 0u8;
    if reduced_still_picture_header == 1 {
        level = br.read_bits(5)? as u8;
    } else {
        let timing_info_present = br.read_bit()?;
        let mut decoder_model_info_present = 0u32;
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
        let initial_display_delay_present = br.read_bit()?;
        let operating_points_cnt_minus_1 = br.read_bits(5)?;
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

    let frame_width_bits = br.read_bits(4)? + 1; // frame_width_bits_minus_1 + 1
    let frame_height_bits = br.read_bits(4)? + 1;
    let max_frame_width = br.read_bits(frame_width_bits)? + 1;
    let max_frame_height = br.read_bits(frame_height_bits)? + 1;

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
    };
    match parse_color_config(&mut br, reduced_still_picture_header, seq_profile) {
        Some(color) => {
            (
                header.high_bitdepth,
                header.twelve_bit,
                header.monochrome,
                header.subsampling_x,
                header.subsampling_y,
                header.chroma_sample_position,
            ) = color;
        }
        None if strict => return None,
        // Truncated tail: the caps parse only needs the geometry.
        None => {}
    }
    Some(header)
}

/// Skip the flag block between the geometry and `color_config` (spec 5.5.1) and
/// parse `color_config` (5.5.2): `(high_bitdepth, twelve_bit, monochrome,
/// subsampling_x, subsampling_y, chroma_sample_position)`. `None` on truncation.
#[allow(clippy::type_complexity)]
fn parse_color_config(
    br: &mut BitReader,
    reduced_still_picture_header: u32,
    seq_profile: u8,
) -> Option<(bool, bool, bool, bool, bool, u8)> {
    if reduced_still_picture_header == 0 && br.read_bit()? == 1 {
        // frame_id_numbers_present_flag
        br.read_bits(4)?; // delta_frame_id_length_minus_2
        br.read_bits(3)?; // additional_frame_id_length_minus_1
    }
    br.read_bit()?; // use_128x128_superblock
    br.read_bit()?; // enable_filter_intra
    br.read_bit()?; // enable_intra_edge_filter
    if reduced_still_picture_header == 0 {
        br.read_bit()?; // enable_interintra_compound
        br.read_bit()?; // enable_masked_compound
        br.read_bit()?; // enable_warped_motion
        br.read_bit()?; // enable_dual_filter
        let enable_order_hint = br.read_bit()?;
        if enable_order_hint == 1 {
            br.read_bit()?; // enable_jnt_comp
            br.read_bit()?; // enable_ref_frame_mvs
        }
        // seq_choose_screen_content_tools -> SELECT (2), else an explicit bit.
        let force_sct = if br.read_bit()? == 1 {
            2
        } else {
            br.read_bit()?
        };
        if force_sct > 0 {
            // seq_choose_integer_mv -> SELECT, else an explicit bit.
            if br.read_bit()? == 0 {
                br.read_bit()?; // seq_force_integer_mv
            }
        }
        if enable_order_hint == 1 {
            br.read_bits(3)?; // order_hint_bits_minus_1
        }
    }
    br.read_bit()?; // enable_superres
    br.read_bit()?; // enable_cdef
    br.read_bit()?; // enable_restoration

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
    let mut identity_matrix = false;
    if br.read_bit()? == 1 {
        // color_description_present: primaries / transfer / matrix.
        let primaries = br.read_bits(8)?;
        let transfer = br.read_bits(8)?;
        let matrix = br.read_bits(8)?;
        // CP_BT_709 / TC_SRGB / MC_IDENTITY: the RGB case with no subsampling.
        identity_matrix = matrix == 0 && primaries == 1 && transfer == 13;
    }
    if monochrome {
        br.read_bit()?; // color_range
        return Some((high_bitdepth, twelve_bit, monochrome, true, true, 0));
    }
    if identity_matrix {
        return Some((high_bitdepth, twelve_bit, monochrome, false, false, 0));
    }
    br.read_bit()?; // color_range
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
    let csp = if ssx && ssy {
        br.read_bits(2)? as u8
    } else {
        0
    };
    Some((high_bitdepth, twelve_bit, monochrome, ssx, ssy, csp))
}

/// AV1 unsigned variable-length code (spec `uvlc()`): count leading zeros, then
/// read that many value bits. `>= 32` leading zeros saturates with no value read.
fn read_uvlc(br: &mut BitReader) -> Option<u32> {
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

    use g2g_core::frame::Frame;
    use g2g_core::memory::SystemSlice;
    use g2g_core::{FrameTiming, MemoryDomain, PushOutcome};

    #[derive(Default)]
    struct RecordingSink {
        packets: Vec<PipelinePacket>,
    }

    impl OutputSink for RecordingSink {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async move {
                self.packets.push(packet);
                Ok(PushOutcome::Accepted)
            })
        }
    }

    fn frame_with_bytes(seq: u64, bytes: Vec<u8>) -> Frame {
        Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            timing: FrameTiming::default(),
            sequence: seq,
            meta: Default::default(),
        }
    }

    fn av1_caps() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::Av1,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }
    }

    #[tokio::test]
    async fn emits_caps_changed_before_first_data_frame() {
        let mut parse = Av1Parse::new();
        parse.configure_pipeline(&av1_caps()).unwrap();
        let mut sink = RecordingSink::default();

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
        let mut sink = RecordingSink::default();

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
        let mut sink = RecordingSink::default();

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

    #[tokio::test]
    async fn rejects_non_av1_caps_in_intercept() {
        let parse = Av1Parse::new();
        let vp9 = Caps::CompressedVideo {
            codec: VideoCodec::Vp9,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
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
                    }]
                );
            }
            _ => panic!("expected Identity"),
        }
    }
}
