//! H.265 (HEVC) access-unit parser that refines source-side `Caps`.
//!
//! The H.265 sibling of `h264parse`: it scans each `DataFrame` for an SPS NAL
//! (`nal_unit_type == 33`), recovers the coded picture dimensions, and emits a
//! `CapsChanged` with `Dim::Fixed` values before forwarding the frame. This
//! lets a raw H.265 elementary stream (which advertises `Dim::Any` at
//! negotiation, since the SPS only lands once bytes flow) be restreamed or
//! recorded with concrete geometry.
//!
//! H.265's NAL header is two bytes (type is bits `[1..7]` of the first), and
//! the SPS carries a variable-size `profile_tier_level` before the dimensions;
//! for a single-layer stream (`sps_max_sub_layers_minus1 == 0`) that block is a
//! fixed 96 bits. The Annex-B / AVCC framing, the RBSP de-emulation, and the
//! exp-Golomb bit reader are shared with `h264parse` via the `annexb` module.
//!
//! Framerate from the VUI `timing_info` is not recovered yet: in H.265 the VUI
//! sits past the PCM, short-term-ref-pic-set, and long-term-ref loops, too deep
//! to reach safely without a real-stream reference, so caps carry `Rate::Any`.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket, Rate, VideoCodec,
};

use crate::annexb::{strip_emulation_prevention, BitReader};

/// H.265 NAL unit type for a sequence parameter set (SPS_NUT).
const SPS_NUT: u8 = 33;

#[derive(Debug, Default)]
pub struct H265Parse {
    configured: bool,
    last_emitted_caps: Option<Caps>,
    sps_emitted: u64,
}

impl H265Parse {
    pub fn new() -> Self {
        Self::default()
    }

    /// Count of `CapsChanged` packets pushed downstream, for tests asserting
    /// re-emission is suppressed when the SPS dimensions are unchanged.
    pub fn caps_changes_emitted(&self) -> u64 {
        self.sps_emitted
    }
}

impl AsyncElement for H265Parse {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        let supported = Caps::CompressedVideo {
            codec: VideoCodec::H265,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        upstream_caps.intersect(&supported)
    }

    /// Pass-through identity over H.265 of any geometry (the parser refines
    /// geometry mid-stream from the SPS but never changes media type).
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Identity(CapsSet::one(Caps::CompressedVideo {
            codec: VideoCodec::H265,
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
                codec: VideoCodec::H265,
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
                        if let Some(info) = extract_sps_info(slice.as_slice()) {
                            let new_caps = Caps::CompressedVideo {
                                codec: VideoCodec::H265,
                                width: Dim::Fixed(info.width),
                                height: Dim::Fixed(info.height),
                                framerate: Rate::Any,
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
                    self.last_emitted_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Eos => {}
            }
            Ok(())
        })
    }
}

impl PadTemplates for H265Parse {
    fn pad_templates() -> Vec<PadTemplate> {
        let h265 = Caps::CompressedVideo {
            codec: VideoCodec::H265,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        Vec::from([
            PadTemplate::sink(CapsSet::one(h265.clone())),
            PadTemplate::source(CapsSet::one(h265)),
        ])
    }
}

/// Coded picture dimensions recovered from an SPS (post conformance-window
/// cropping).
struct SpsInfo {
    width: u32,
    height: u32,
}

/// Walk the NALs of `au` (Annex-B or AVCC, auto-detected), returning the info
/// from the first SPS NAL we can parse. H.265 NAL type is bits `[1..7]` of the
/// first header byte.
fn extract_sps_info(au: &[u8]) -> Option<SpsInfo> {
    for nal in crate::annexb::nal_units_any(au) {
        if nal.len() < 2 {
            continue;
        }
        let nal_unit_type = (nal[0] >> 1) & 0x3F;
        if nal_unit_type != SPS_NUT {
            continue;
        }
        // Strip the 2-byte NAL header, then de-emulate the RBSP.
        let rbsp = strip_emulation_prevention(&nal[2..]);
        if let Some(info) = parse_sps(&rbsp) {
            return Some(info);
        }
    }
    None
}

/// Parse the SPS RBSP (H.265 7.3.2.2) up to the conformance window, returning
/// the cropped picture dimensions. `None` on a parse failure before the
/// dimensions resolve.
fn parse_sps(rbsp: &[u8]) -> Option<SpsInfo> {
    let mut br = BitReader::new(rbsp);
    let _sps_video_parameter_set_id = br.read_bits(4)?;
    let sps_max_sub_layers_minus1 = br.read_bits(3)?;
    let _sps_temporal_id_nesting_flag = br.read_bit()?;
    skip_profile_tier_level(&mut br, sps_max_sub_layers_minus1)?;

    let _sps_seq_parameter_set_id = br.read_ue()?;
    let chroma_format_idc = br.read_ue()?;
    let separate_colour_plane_flag = if chroma_format_idc == 3 { br.read_bit()? } else { 0 };
    let pic_width = br.read_ue()?;
    let pic_height = br.read_ue()?;

    let conformance_window_flag = br.read_bit()?;
    let (left, right, top, bottom) = if conformance_window_flag == 1 {
        (br.read_ue()?, br.read_ue()?, br.read_ue()?, br.read_ue()?)
    } else {
        (0, 0, 0, 0)
    };

    // Crop offsets are in chroma sample units, scaled to luma by SubWidthC /
    // SubHeightC (H.265 7.4.3.2.1). ChromaArrayType 0 (monochrome or 4:4:4 with
    // separate colour planes) and 4:4:4 use 1x1.
    let chroma_array_type = if separate_colour_plane_flag == 1 { 0 } else { chroma_format_idc };
    let (sub_width_c, sub_height_c) = match chroma_array_type {
        1 => (2u32, 2u32), // 4:2:0
        2 => (2, 1),       // 4:2:2
        _ => (1, 1),       // 4:4:4 / monochrome
    };
    let width = pic_width.saturating_sub((left + right).saturating_mul(sub_width_c));
    let height = pic_height.saturating_sub((top + bottom).saturating_mul(sub_height_c));
    Some(SpsInfo { width, height })
}

/// Skip `profile_tier_level(1, max_sub_layers_minus1)` (H.265 7.3.3). The
/// general block is a fixed 96 bits (88-bit profile/tier/constraints + 8-bit
/// level); per-sub-layer blocks follow only when `max_sub_layers_minus1 > 0`.
fn skip_profile_tier_level(br: &mut BitReader, max_sub_layers_minus1: u32) -> Option<()> {
    br.skip_bits(88)?; // general profile/tier + constraint/reserved/inbld
    br.skip_bits(8)?; // general_level_idc

    let mut sub_profile_present = [false; 8];
    let mut sub_level_present = [false; 8];
    for i in 0..max_sub_layers_minus1 as usize {
        sub_profile_present[i] = br.read_bit()? == 1;
        sub_level_present[i] = br.read_bit()? == 1;
    }
    if max_sub_layers_minus1 > 0 {
        for _ in max_sub_layers_minus1..8 {
            br.read_bits(2)?; // reserved_zero_2bits
        }
    }
    for i in 0..max_sub_layers_minus1 as usize {
        if sub_profile_present[i] {
            br.skip_bits(88)?; // sub_layer profile/tier block
        }
        if sub_level_present[i] {
            br.skip_bits(8)?; // sub_layer_level_idc
        }
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Build an Annex-B H.265 SPS for `pic_w` x `pic_h` luma samples at
    /// `chroma_format_idc`, optionally with a conformance window
    /// `(left, right, top, bottom)`. The profile_tier_level is written as 96
    /// zero bits (its content is skipped); the RBSP is emulation-prevented like
    /// a real encoder's output so the parser's de-emulation round-trips it.
    fn build_annexb_sps(
        pic_w: u32,
        pic_h: u32,
        chroma_format_idc: u32,
        conf: Option<(u32, u32, u32, u32)>,
    ) -> Vec<u8> {
        let mut w = BitWriter::default();
        w.write_bits(0, 4); // sps_video_parameter_set_id
        w.write_bits(0, 3); // sps_max_sub_layers_minus1
        w.write_bit(1); // sps_temporal_id_nesting_flag
        for _ in 0..96 {
            w.write_bit(0); // profile_tier_level (single layer = 96 bits)
        }
        w.write_ue(0); // sps_seq_parameter_set_id
        w.write_ue(chroma_format_idc);
        if chroma_format_idc == 3 {
            w.write_bit(0); // separate_colour_plane_flag
        }
        w.write_ue(pic_w);
        w.write_ue(pic_h);
        match conf {
            Some((l, r, t, b)) => {
                w.write_bit(1); // conformance_window_flag
                w.write_ue(l);
                w.write_ue(r);
                w.write_ue(t);
                w.write_ue(b);
            }
            None => w.write_bit(0),
        }
        w.write_bit(1); // rbsp_stop_one_bit
        w.align_to_byte();
        let rbsp = w.into_bytes();
        let ebsp = add_emulation_prevention(&rbsp);

        // 00 00 00 01 | NAL header (type 33, layer 0, tid+1 = 1) | EBSP
        let mut out = vec![0u8, 0, 0, 1, 0x42, 0x01];
        out.extend_from_slice(&ebsp);
        out
    }

    /// Inverse of `annexb::strip_emulation_prevention`: insert `0x03` after each
    /// `00 00` run preceding a byte <= 0x03.
    fn add_emulation_prevention(rbsp: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(rbsp.len());
        let mut zeros = 0usize;
        for &b in rbsp {
            if zeros >= 2 && b <= 0x03 {
                out.push(0x03);
                zeros = 0;
            }
            out.push(b);
            zeros = if b == 0 { zeros + 1 } else { 0 };
        }
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
    fn recovers_dimensions_from_sps() {
        let stream = build_annexb_sps(1920, 1080, 1, None);
        let info = extract_sps_info(&stream).expect("SPS must parse");
        assert_eq!((info.width, info.height), (1920, 1080));
    }

    #[test]
    fn applies_conformance_window_cropping() {
        // 1920x1088 coded, 4:2:0 (SubHeightC = 2), crop 4 chroma rows off the
        // bottom -> 1088 - 2*4 = 1080.
        let stream = build_annexb_sps(1920, 1088, 1, Some((0, 0, 0, 4)));
        let info = extract_sps_info(&stream).expect("SPS with conf window must parse");
        assert_eq!((info.width, info.height), (1920, 1080));
    }

    #[test]
    fn parses_an_avcc_framed_sps() {
        // Re-frame the SPS NAL as length-prefixed (HVCC-style) and confirm the
        // dimensions still resolve.
        let annexb = build_annexb_sps(1280, 720, 1, None);
        let nal = &annexb[4..]; // drop the 00 00 00 01 start code
        let mut hvcc = (nal.len() as u32).to_be_bytes().to_vec();
        hvcc.extend_from_slice(nal);
        let info = extract_sps_info(&hvcc).expect("length-prefixed SPS must parse");
        assert_eq!((info.width, info.height), (1280, 720));
    }

    #[test]
    fn ignores_non_sps_nals() {
        // A TRAIL_R slice NAL (type 1 -> first byte 0x02) carries no SPS.
        let stream = [0u8, 0, 0, 1, 0x02, 0x01, 0xAA, 0xBB];
        assert!(extract_sps_info(&stream).is_none());
    }

    #[test]
    fn returns_none_on_empty_input() {
        assert!(extract_sps_info(&[]).is_none());
    }

    #[test]
    fn skip_profile_tier_level_advances_96_bits_for_single_layer() {
        // 12 bytes of PTL then a ue(0) = single '1' bit. After the skip the
        // reader must sit exactly on that bit.
        let mut w = BitWriter::default();
        for _ in 0..96 {
            w.write_bit(0);
        }
        w.write_bit(1); // a marker the reader should land on (ue value 0)
        w.align_to_byte();
        let bytes = w.into_bytes();
        let mut br = BitReader::new(&bytes);
        skip_profile_tier_level(&mut br, 0).expect("skips the fixed 96-bit block");
        assert_eq!(br.read_ue(), Some(0), "reader landed just past the PTL");
    }

    // -- Element-level tests (drive H265Parse::process directly) -----------

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
        }
    }

    fn h265_caps() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H265,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }
    }

    #[tokio::test]
    async fn emits_caps_changed_before_first_data_frame() {
        let mut parse = H265Parse::new();
        parse.configure_pipeline(&h265_caps()).unwrap();
        let mut sink = RecordingSink::default();

        let frame = frame_with_bytes(0, build_annexb_sps(1920, 1080, 1, None));
        parse
            .process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();

        assert_eq!(sink.packets.len(), 2, "expected CapsChanged then DataFrame");
        match &sink.packets[0] {
            PipelinePacket::CapsChanged(Caps::CompressedVideo {
                codec: VideoCodec::H265,
                width,
                height,
                ..
            }) => {
                assert_eq!(*width, Dim::Fixed(1920));
                assert_eq!(*height, Dim::Fixed(1080));
            }
            other => panic!("expected H.265 CapsChanged first, got {other:?}"),
        }
        assert!(matches!(sink.packets[1], PipelinePacket::DataFrame(_)));
        assert_eq!(parse.caps_changes_emitted(), 1);
    }

    #[tokio::test]
    async fn does_not_re_emit_caps_when_unchanged() {
        let mut parse = H265Parse::new();
        parse.configure_pipeline(&h265_caps()).unwrap();
        let mut sink = RecordingSink::default();

        for seq in 0..3 {
            let frame = frame_with_bytes(seq, build_annexb_sps(1280, 720, 1, None));
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
        assert_eq!(caps_count, 1, "CapsChanged fires once for identical SPS");
        assert_eq!(parse.caps_changes_emitted(), 1);
    }

    #[tokio::test]
    async fn re_emits_caps_on_resolution_change() {
        let mut parse = H265Parse::new();
        parse.configure_pipeline(&h265_caps()).unwrap();
        let mut sink = RecordingSink::default();

        parse
            .process(
                PipelinePacket::DataFrame(frame_with_bytes(0, build_annexb_sps(1280, 720, 1, None))),
                &mut sink,
            )
            .await
            .unwrap();
        parse
            .process(
                PipelinePacket::DataFrame(frame_with_bytes(
                    1,
                    build_annexb_sps(1920, 1080, 1, None),
                )),
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
    async fn rejects_non_h265_caps_in_intercept() {
        let parse = H265Parse::new();
        let h264 = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        assert_eq!(parse.intercept_caps(&h264), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn caps_constraint_is_identity_h265_any() {
        let parse = H265Parse::new();
        let c = parse.caps_constraint_as_transform();
        match c {
            CapsConstraint::Identity(set) => {
                assert_eq!(
                    set.alternatives(),
                    &[Caps::CompressedVideo {
                        codec: VideoCodec::H265,
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
