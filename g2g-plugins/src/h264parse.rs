//! H.264 access-unit parser that refines source-side `Caps`.
//!
//! M6: scans each `DataFrame`'s bitstream for an SPS NAL unit, parses
//! width/height, and emits a `CapsChanged` packet with `Dim::Fixed` values
//! before forwarding the frame. This is the first element that refines
//! caps mid-stream — `RtspSrc` advertises `Dim::Any` at negotiation time
//! because the SPS only lands once bytes flow.
//!
//! Bitstream framing: both Annex-B (00 00 01 / 00 00 00 01 start codes) and
//! AVCC (4-byte length-prefixed NALs, what retina emits by default) are
//! accepted, detected per access unit via `annexb::nal_units_any`. The element
//! refines caps only; it does not rewrite the bitstream between framings.
//!
//! Framerate is recovered from the SPS VUI `timing_info` when present
//! (`time_scale / (2 * num_units_in_tick)`, emitted as a Q16 `Rate::Fixed`);
//! caps carry `Rate::Any` when the VUI omits it.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, Rate, VideoCodec,
};

use crate::annexb::{strip_emulation_prevention, BitReader};

#[derive(Debug, Default)]
pub struct H264Parse {
    configured: bool,
    last_emitted_caps: Option<Caps>,
    sps_emitted: u64,
}

impl H264Parse {
    pub fn new() -> Self {
        Self::default()
    }

    /// Count of `CapsChanged` packets this element has pushed downstream.
    /// Useful for tests asserting that re-emission is suppressed when the
    /// SPS dimensions are unchanged.
    pub fn caps_changes_emitted(&self) -> u64 {
        self.sps_emitted
    }
}

impl AsyncElement for H264Parse {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // H264Parse consumes H.264 at any geometry; intersecting against
        // that narrows the proposal and rejects non-H.264 inputs.
        let supported = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        upstream_caps.intersect(&supported)
    }

    /// M16 step 5g: pass-through identity over H.264 of any geometry.
    /// `Identity(CapsSet::one(...))` is the native shape for transforms
    /// that accept and emit the same caps. With a fully-native chain
    /// the solver couples input and output links and rejects non-H.264
    /// upstream at negotiation time instead of via the dynamic
    /// `intercept_caps` callback.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Identity(CapsSet::one(Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }))
    }

    fn configure_pipeline(
        &mut self,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::CompressedVideo {
                codec: VideoCodec::H264,
                ..
            } => {
                self.configured = true;
                Ok(ConfigureOutcome::Accepted)
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "H.264 parser",
            "Codec/Parser/Video",
            "Parses an H.264 Annex-B stream and refines caps from SPS/PPS",
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
                PipelinePacket::DataFrame(frame) => {
                    if let g2g_core::MemoryDomain::System(slice) = &frame.domain {
                        if let Some(info) = extract_sps_info(slice.as_slice()) {
                            let new_caps = Caps::CompressedVideo {
                                codec: VideoCodec::H264,
                                width: Dim::Fixed(info.width),
                                height: Dim::Fixed(info.height),
                                framerate: info.framerate.map_or(Rate::Any, Rate::Fixed),
                            };
                            if self.last_emitted_caps.as_ref() != Some(&new_caps) {
                                out.push(PipelinePacket::CapsChanged(new_caps.clone()))
                                    .await?;
                                self.last_emitted_caps = Some(new_caps);
                                self.sps_emitted += 1;
                            }
                        }
                    }
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    out.push(PipelinePacket::CapsChanged(c)).await?;
                }
                PipelinePacket::Flush => {
                    // Reset SPS tracking so caps re-emit after the seek.
                    self.last_emitted_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                // Segment is control: forward unchanged.
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                PipelinePacket::Eos => {}
            }
            Ok(())
        })
    }
}

impl PadTemplates for H264Parse {
    /// Consumes and produces H.264 at any geometry (the parser refines
    /// geometry mid-stream from the SPS but never changes media type).
    fn pad_templates() -> Vec<PadTemplate> {
        let h264 = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        Vec::from([
            PadTemplate::sink(CapsSet::one(h264.clone())),
            PadTemplate::source(CapsSet::one(h264)),
        ])
    }
}

/// Geometry and optional framerate recovered from an SPS.
struct SpsInfo {
    width: u32,
    height: u32,
    /// Framerate as Q16 fixed-point from the VUI `timing_info`, `None` when
    /// the SPS carries no timing.
    framerate: Option<u32>,
}

/// Walk the NALs of `au` (Annex-B or AVCC, auto-detected), returning the info
/// from the first SPS NAL (nal_unit_type == 7) we can fully parse.
fn extract_sps_info(au: &[u8]) -> Option<SpsInfo> {
    for nal in crate::annexb::nal_units_any(au) {
        if nal.is_empty() {
            continue;
        }
        let nal_unit_type = nal[0] & 0x1F;
        if nal_unit_type != 7 {
            continue;
        }
        let rbsp = strip_emulation_prevention(&nal[1..]);
        if let Some(info) = parse_sps(&rbsp) {
            return Some(info);
        }
    }
    None
}

/// Parse the SPS RBSP (post NAL-header byte) for the coded picture dimensions
/// and, when the VUI carries `timing_info`, the framerate. Returns `None` on a
/// parse failure up to the dimensions; a failure past them leaves only the
/// framerate unknown.
fn parse_sps(rbsp: &[u8]) -> Option<SpsInfo> {
    if rbsp.len() < 3 {
        return None;
    }
    let profile_idc = rbsp[0];
    // rbsp[1] = constraint_set flags + reserved zero bits (skipped)
    // rbsp[2] = level_idc (skipped)
    let mut br = BitReader::new(&rbsp[3..]);

    let _sps_id = br.read_ue()?;

    // 4:2:0 unless a high profile signals a different chroma format.
    let mut chroma_format_idc = 1u32;
    let mut separate_colour_plane_flag = 0u32;
    // Profiles that include chroma/scaling/etc. extra header fields.
    if matches!(
        profile_idc,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
    ) {
        chroma_format_idc = br.read_ue()?;
        if chroma_format_idc == 3 {
            separate_colour_plane_flag = br.read_bit()?;
        }
        let _bit_depth_luma_minus8 = br.read_ue()?;
        let _bit_depth_chroma_minus8 = br.read_ue()?;
        let _qpprime_y_zero_transform_bypass_flag = br.read_bit()?;
        let seq_scaling_matrix_present_flag = br.read_bit()?;
        if seq_scaling_matrix_present_flag == 1 {
            // We don't decode the (optional) scaling lists. M6 test fixtures
            // never set this flag; live streams that do will leave dims
            // unknown until M7 lands a full SPS parser.
            return None;
        }
    }

    let _log2_max_frame_num_minus4 = br.read_ue()?;
    let pic_order_cnt_type = br.read_ue()?;
    if pic_order_cnt_type == 0 {
        let _log2_max_pic_order_cnt_lsb_minus4 = br.read_ue()?;
    } else if pic_order_cnt_type == 1 {
        let _delta_pic_order_always_zero_flag = br.read_bit()?;
        let _offset_for_non_ref_pic = br.read_se()?;
        let _offset_for_top_to_bottom_field = br.read_se()?;
        let num_ref_frames_in_pic_order_cnt_cycle = br.read_ue()?;
        for _ in 0..num_ref_frames_in_pic_order_cnt_cycle {
            let _offset = br.read_se()?;
        }
    }
    let _max_num_ref_frames = br.read_ue()?;
    let _gaps_in_frame_num_value_allowed_flag = br.read_bit()?;

    let pic_width_in_mbs_minus1 = br.read_ue()?;
    let pic_height_in_map_units_minus1 = br.read_ue()?;
    let frame_mbs_only_flag = br.read_bit()?;
    if frame_mbs_only_flag == 0 {
        let _mb_adaptive_frame_field_flag = br.read_bit()?;
    }
    let _direct_8x8_inference_flag = br.read_bit()?;

    let frame_cropping_flag = br.read_bit()?;
    let (crop_left, crop_right, crop_top, crop_bottom) = if frame_cropping_flag == 1 {
        let l = br.read_ue()?;
        let r = br.read_ue()?;
        let t = br.read_ue()?;
        let b = br.read_ue()?;
        (l, r, t, b)
    } else {
        (0, 0, 0, 0)
    };

    // Crop units in luma samples (H.264 7.4.2.1.1). ChromaArrayType 0
    // (monochrome, or 4:4:4 with separate colour planes) crops 1 x (2-fmof);
    // otherwise SubWidthC x SubHeightC*(2-fmof).
    let chroma_array_type = if separate_colour_plane_flag == 1 { 0 } else { chroma_format_idc };
    let (sub_width_c, sub_height_c) = match chroma_array_type {
        1 => (2u32, 2u32), // 4:2:0
        2 => (2, 1),       // 4:2:2
        _ => (1, 1),       // 4:4:4 / monochrome
    };
    let crop_x = (crop_left + crop_right).saturating_mul(sub_width_c);
    let crop_y = (crop_top + crop_bottom)
        .saturating_mul(sub_height_c.saturating_mul(2u32.saturating_sub(frame_mbs_only_flag)));

    let width = ((pic_width_in_mbs_minus1 + 1) * 16).saturating_sub(crop_x);
    let height = ((2 - frame_mbs_only_flag) * (pic_height_in_map_units_minus1 + 1) * 16)
        .saturating_sub(crop_y);

    // vui_parameters_present_flag follows frame cropping. Read it without `?`
    // so a stream truncated right here still yields the dimensions.
    let framerate = match br.read_bit() {
        Some(1) => parse_vui_framerate(&mut br),
        _ => None,
    };

    Some(SpsInfo {
        width,
        height,
        framerate,
    })
}

/// Walk the VUI parameters up to `timing_info`, returning the framerate as Q16
/// fixed-point (`time_scale / (2 * num_units_in_tick)`). `None` if the VUI
/// omits timing or ends early; the caller already has the dimensions.
fn parse_vui_framerate(br: &mut BitReader) -> Option<u32> {
    // aspect_ratio_info_present_flag
    if br.read_bit()? == 1 {
        let aspect_ratio_idc = br.read_bits(8)?;
        // 255 = Extended_SAR: explicit sar_width / sar_height follow.
        if aspect_ratio_idc == 255 {
            br.read_bits(16)?; // sar_width
            br.read_bits(16)?; // sar_height
        }
    }
    // overscan_info_present_flag
    if br.read_bit()? == 1 {
        br.read_bit()?; // overscan_appropriate_flag
    }
    // video_signal_type_present_flag
    if br.read_bit()? == 1 {
        br.read_bits(3)?; // video_format
        br.read_bit()?; // video_full_range_flag
        if br.read_bit()? == 1 {
            // colour_description_present_flag
            br.read_bits(8)?; // colour_primaries
            br.read_bits(8)?; // transfer_characteristics
            br.read_bits(8)?; // matrix_coefficients
        }
    }
    // chroma_loc_info_present_flag
    if br.read_bit()? == 1 {
        br.read_ue()?; // chroma_sample_loc_type_top_field
        br.read_ue()?; // chroma_sample_loc_type_bottom_field
    }
    // timing_info_present_flag
    if br.read_bit()? == 1 {
        let num_units_in_tick = br.read_bits(32)?;
        let time_scale = br.read_bits(32)?;
        let _fixed_frame_rate_flag = br.read_bit()?;
        if num_units_in_tick == 0 {
            return None;
        }
        // fps = time_scale / (2 * num_units_in_tick); carry to Q16 in u64.
        let q16 = ((time_scale as u64) << 16) / (2 * num_units_in_tick as u64);
        return u32::try_from(q16).ok();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Build a minimal baseline-profile SPS RBSP for `width` x `height`
    /// (both multiples of 16), then frame it in Annex-B. Returns the
    /// full byte stream including a 4-byte start code and NAL header.
    fn build_test_annexb_sps(width: u32, height: u32) -> Vec<u8> {
        assert!(width % 16 == 0 && height % 16 == 0);
        let mut w = BitWriter::default();
        // Post NAL-header SPS fields:
        // seq_parameter_set_id = 0
        w.write_ue(0);
        // log2_max_frame_num_minus4 = 0
        w.write_ue(0);
        // pic_order_cnt_type = 0
        w.write_ue(0);
        // log2_max_pic_order_cnt_lsb_minus4 = 0
        w.write_ue(0);
        // max_num_ref_frames = 1
        w.write_ue(1);
        // gaps_in_frame_num_value_allowed_flag = 0
        w.write_bit(0);
        // pic_width_in_mbs_minus1
        w.write_ue(width / 16 - 1);
        // pic_height_in_map_units_minus1
        w.write_ue(height / 16 - 1);
        // frame_mbs_only_flag = 1
        w.write_bit(1);
        // direct_8x8_inference_flag = 0
        w.write_bit(0);
        // frame_cropping_flag = 0
        w.write_bit(0);
        // vui_parameters_present_flag = 0
        w.write_bit(0);
        // rbsp_trailing_bits: 1 then zero-pad
        w.write_bit(1);
        w.align_to_byte();

        let rbsp = w.into_bytes();

        // Annex-B framing: 00 00 00 01 | nal_header | profile/level bytes | rbsp
        // nal_header for SPS: forbidden_zero_bit=0, nal_ref_idc=3, nal_unit_type=7
        // → (3 << 5) | 7 = 0x67
        // Then the SPS's byte-aligned prefix: profile_idc=66 (baseline),
        // constraint flags + reserved = 0, level_idc=30 — chosen so the
        // parser takes the simple (non-high-profile) branch.
        let mut out = vec![0u8, 0, 0, 1, 0x67, 66, 0, 30];
        out.extend_from_slice(&rbsp);
        out
    }

    /// Build an Annex-B SPS for `width` x `height` carrying a VUI `timing_info`
    /// block. Emulation-prevention bytes are inserted (as a real encoder would
    /// for the 32-bit fields' zero runs) so the parser's de-emulation
    /// round-trips them exactly.
    fn build_annexb_sps_with_timing(
        width: u32,
        height: u32,
        num_units_in_tick: u32,
        time_scale: u32,
    ) -> Vec<u8> {
        assert!(width % 16 == 0 && height % 16 == 0);
        let mut w = BitWriter::default();
        w.write_ue(0); // seq_parameter_set_id
        w.write_ue(0); // log2_max_frame_num_minus4
        w.write_ue(0); // pic_order_cnt_type
        w.write_ue(0); // log2_max_pic_order_cnt_lsb_minus4
        w.write_ue(1); // max_num_ref_frames
        w.write_bit(0); // gaps_in_frame_num_value_allowed_flag
        w.write_ue(width / 16 - 1); // pic_width_in_mbs_minus1
        w.write_ue(height / 16 - 1); // pic_height_in_map_units_minus1
        w.write_bit(1); // frame_mbs_only_flag
        w.write_bit(0); // direct_8x8_inference_flag
        w.write_bit(0); // frame_cropping_flag
        w.write_bit(1); // vui_parameters_present_flag
        w.write_bit(0); // aspect_ratio_info_present_flag
        w.write_bit(0); // overscan_info_present_flag
        w.write_bit(0); // video_signal_type_present_flag
        w.write_bit(0); // chroma_loc_info_present_flag
        w.write_bit(1); // timing_info_present_flag
        w.write_bits(num_units_in_tick, 32);
        w.write_bits(time_scale, 32);
        w.write_bit(0); // fixed_frame_rate_flag
        w.write_bit(1); // rbsp_trailing_bits
        w.align_to_byte();
        let rbsp = w.into_bytes();

        // NAL payload after the 0x67 header: profile/constraint/level + RBSP,
        // emulation-prevented like an encoder's output.
        let mut payload = vec![66u8, 0, 30];
        payload.extend_from_slice(&rbsp);
        let payload = crate::annexb::add_emulation_prevention(&payload);

        let mut out = vec![0u8, 0, 0, 1, 0x67];
        out.extend_from_slice(&payload);
        out
    }

    #[derive(Default)]
    struct BitWriter {
        buf: Vec<u8>,
        bit_pos: usize,
    }

    impl BitWriter {
        fn write_bit(&mut self, b: u32) {
            let byte_idx = self.bit_pos / 8;
            if byte_idx >= self.buf.len() {
                self.buf.push(0);
            }
            let bit_off = 7 - (self.bit_pos % 8);
            self.buf[byte_idx] |= ((b & 1) as u8) << bit_off;
            self.bit_pos += 1;
        }

        fn write_bits(&mut self, value: u32, n: u32) {
            for i in (0..n).rev() {
                self.write_bit((value >> i) & 1);
            }
        }

        fn write_ue(&mut self, v: u32) {
            let v1 = v + 1;
            let n = 31 - v1.leading_zeros();
            for _ in 0..n {
                self.write_bit(0);
            }
            self.write_bits(v1, n + 1);
        }

        fn align_to_byte(&mut self) {
            while self.bit_pos % 8 != 0 {
                self.write_bit(0);
            }
        }

        fn into_bytes(self) -> Vec<u8> {
            self.buf
        }
    }

    #[test]
    fn round_trips_a_1280x720_sps() {
        let stream = build_test_annexb_sps(1280, 720);
        let info = extract_sps_info(&stream).expect("SPS must parse");
        assert_eq!((info.width, info.height), (1280, 720));
        assert_eq!(info.framerate, None, "no VUI timing in the baseline fixture");
    }

    #[test]
    fn round_trips_a_1920x1080_sps() {
        let stream = build_test_annexb_sps(1920, 1088);
        // height 1088 because 1080 is not a multiple of 16; the test
        // builder asserts on alignment. Real 1080p streams use cropping.
        let info = extract_sps_info(&stream).expect("SPS must parse");
        assert_eq!((info.width, info.height), (1920, 1088));
    }

    #[test]
    fn parses_an_avcc_framed_sps() {
        // Re-frame the same SPS NAL as AVCC (4-byte length prefix instead of
        // the Annex-B start code) and confirm the dimensions still resolve.
        let annexb = build_test_annexb_sps(1280, 720);
        let nal = &annexb[4..]; // drop the 00 00 00 01 start code
        let mut avcc = (nal.len() as u32).to_be_bytes().to_vec();
        avcc.extend_from_slice(nal);
        let info = extract_sps_info(&avcc).expect("AVCC SPS must parse");
        assert_eq!((info.width, info.height), (1280, 720));
    }

    #[test]
    fn recovers_framerate_from_vui_timing() {
        // num_units_in_tick = 1, time_scale = 60 -> 30 fps.
        let stream = build_annexb_sps_with_timing(1280, 720, 1, 60);
        let info = extract_sps_info(&stream).expect("SPS with VUI must parse");
        assert_eq!((info.width, info.height), (1280, 720));
        assert_eq!(info.framerate, Some(30 << 16), "30 fps in Q16");
    }

    #[test]
    fn ignores_non_sps_nals() {
        // A stream containing only a slice NAL (type 5 = IDR) returns None.
        let stream = [0u8, 0, 0, 1, 0x65, 0xAA, 0xBB, 0xCC];
        assert!(extract_sps_info(&stream).is_none());
    }

    #[test]
    fn returns_none_on_empty_input() {
        assert!(extract_sps_info(&[]).is_none());
    }

    #[test]
    fn strips_emulation_prevention_bytes() {
        // Input "00 00 03 01" should decode to "00 00 01".
        let ebsp = [0u8, 0, 3, 1, 2, 0, 0, 3, 0xFF];
        let rbsp = strip_emulation_prevention(&ebsp);
        assert_eq!(rbsp, [0u8, 0, 1, 2, 0, 0, 0xFF]);
    }

    #[test]
    // the binary grouping aligns to the Exp-Golomb code words, not byte nibbles.
    #[allow(clippy::unusual_byte_groupings)]
    fn bit_reader_decodes_known_ue_codes() {
        // Bits: 1 010 011 00100 → ue values 0, 1, 2, 3
        let bytes = [0b1_010_011_0, 0b0100_0000];
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_ue(), Some(0));
        assert_eq!(r.read_ue(), Some(1));
        assert_eq!(r.read_ue(), Some(2));
        assert_eq!(r.read_ue(), Some(3));
    }

    #[test]
    fn parse_vui_framerate_reads_timing_info() {
        // num_units_in_tick = 1001, time_scale = 60000 -> 29.97 fps, exercising
        // the Q16 conversion on a non-integer rate.
        let mut w = BitWriter::default();
        w.write_bit(0); // aspect_ratio_info_present_flag
        w.write_bit(0); // overscan_info_present_flag
        w.write_bit(0); // video_signal_type_present_flag
        w.write_bit(0); // chroma_loc_info_present_flag
        w.write_bit(1); // timing_info_present_flag
        w.write_bits(1001, 32);
        w.write_bits(60000, 32);
        w.write_bit(0); // fixed_frame_rate_flag
        w.align_to_byte();
        let bytes = w.into_bytes();
        let mut br = BitReader::new(&bytes);
        let fps = parse_vui_framerate(&mut br).expect("timing info present");
        let expected = ((60000u64 << 16) / (2 * 1001)) as u32;
        assert_eq!(fps, expected);
        assert_eq!(fps >> 16, 29, "~29.97 fps");
    }

    #[test]
    fn parse_vui_framerate_absent_timing_is_none() {
        let mut w = BitWriter::default();
        w.write_bit(0); // aspect_ratio_info_present_flag
        w.write_bit(0); // overscan_info_present_flag
        w.write_bit(0); // video_signal_type_present_flag
        w.write_bit(0); // chroma_loc_info_present_flag
        w.write_bit(0); // timing_info_present_flag = 0
        w.align_to_byte();
        let bytes = w.into_bytes();
        let mut br = BitReader::new(&bytes);
        assert_eq!(parse_vui_framerate(&mut br), None);
    }

    // -- Element-level tests (drive H264Parse::process directly) -----------

    use core::pin::Pin;
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

    fn h264_parse_caps() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }
    }

    #[tokio::test]
    async fn emits_caps_changed_before_first_data_frame() {
        let mut parse = H264Parse::new();
        parse.configure_pipeline(&h264_parse_caps()).unwrap();
        let mut sink = RecordingSink::default();

        let stream = build_test_annexb_sps(1280, 720);
        let frame = frame_with_bytes(0, stream);
        parse
            .process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();

        assert_eq!(sink.packets.len(), 2, "expected CapsChanged then DataFrame");
        match &sink.packets[0] {
            PipelinePacket::CapsChanged(Caps::CompressedVideo { width, height, .. }) => {
                assert_eq!(*width, Dim::Fixed(1280));
                assert_eq!(*height, Dim::Fixed(720));
            }
            other => panic!("expected CapsChanged first, got {other:?}"),
        }
        assert!(matches!(sink.packets[1], PipelinePacket::DataFrame(_)));
        assert_eq!(parse.caps_changes_emitted(), 1);
    }

    #[tokio::test]
    async fn emits_caps_from_an_avcc_access_unit() {
        let mut parse = H264Parse::new();
        parse.configure_pipeline(&h264_parse_caps()).unwrap();
        let mut sink = RecordingSink::default();

        // Re-frame the SPS NAL as AVCC (4-byte length prefix).
        let annexb = build_test_annexb_sps(1280, 720);
        let nal = &annexb[4..];
        let mut avcc = (nal.len() as u32).to_be_bytes().to_vec();
        avcc.extend_from_slice(nal);

        parse
            .process(PipelinePacket::DataFrame(frame_with_bytes(0, avcc)), &mut sink)
            .await
            .unwrap();

        match &sink.packets[0] {
            PipelinePacket::CapsChanged(Caps::CompressedVideo { width, height, .. }) => {
                assert_eq!(*width, Dim::Fixed(1280));
                assert_eq!(*height, Dim::Fixed(720));
            }
            other => panic!("expected CapsChanged from the AVCC AU, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn does_not_re_emit_caps_when_unchanged() {
        let mut parse = H264Parse::new();
        parse.configure_pipeline(&h264_parse_caps()).unwrap();
        let mut sink = RecordingSink::default();

        for seq in 0..3 {
            let stream = build_test_annexb_sps(1280, 720);
            let frame = frame_with_bytes(seq, stream);
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
        assert_eq!(caps_count, 1, "CapsChanged must fire once for identical SPS");
        assert_eq!(parse.caps_changes_emitted(), 1);
    }

    #[tokio::test]
    async fn re_emits_caps_on_resolution_change() {
        let mut parse = H264Parse::new();
        parse.configure_pipeline(&h264_parse_caps()).unwrap();
        let mut sink = RecordingSink::default();

        let frame_720 = frame_with_bytes(0, build_test_annexb_sps(1280, 720));
        parse
            .process(PipelinePacket::DataFrame(frame_720), &mut sink)
            .await
            .unwrap();

        let frame_1080 = frame_with_bytes(1, build_test_annexb_sps(1920, 1088));
        parse
            .process(PipelinePacket::DataFrame(frame_1080), &mut sink)
            .await
            .unwrap();

        let widths: Vec<Dim> = sink
            .packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::CapsChanged(Caps::CompressedVideo { width, .. }) => Some(width.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(widths, vec![Dim::Fixed(1280), Dim::Fixed(1920)]);
        assert_eq!(parse.caps_changes_emitted(), 2);
    }

    #[tokio::test]
    async fn rejects_non_h264_caps_in_intercept() {
        let parse = H264Parse::new();
        let vp9 = Caps::CompressedVideo {
            codec: VideoCodec::Vp9,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        assert_eq!(parse.intercept_caps(&vp9), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn caps_constraint_is_identity_h264_any() {
        // M16 step 5g: native shape is `Identity` over H.264 with any
        // geometry. With a fully-native chain the solver enforces the
        // input/output coupling and the format requirement during
        // arc-consistency, not via the dynamic `intercept_caps`.
        let parse = H264Parse::new();
        let c = parse.caps_constraint_as_transform();
        match c {
            CapsConstraint::Identity(set) => {
                assert_eq!(
                    set.alternatives(),
                    &[Caps::CompressedVideo {
                        codec: VideoCodec::H264,
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
