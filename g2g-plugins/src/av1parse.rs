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

/// The dimensions (and profile) decoded from an AV1 sequence header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Av1SeqHeader {
    pub width: u32,
    pub height: u32,
    /// `seq_profile` (0..=2): bit depth + chroma support. Not in `Caps`,
    /// surfaced for completeness.
    pub profile: u8,
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

/// Walk the OBUs of one temporal unit (low-overhead format: size-delimited) and
/// parse the first sequence-header OBU. `None` if none is present or the framing
/// is malformed.
fn extract_seq_header(data: &[u8]) -> Option<Av1SeqHeader> {
    let mut pos = 0;
    while pos < data.len() {
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
        let end = pos.checked_add(size)?;
        if end > data.len() {
            return None;
        }
        if obu_type == OBU_SEQUENCE_HEADER {
            if let Some(info) = parse_sequence_header(&data[pos..end]) {
                return Some(info);
            }
        }
        pos = end;
    }
    None
}

/// Parse a sequence-header OBU payload up to `max_frame_width/height`. `None` on
/// truncation.
fn parse_sequence_header(payload: &[u8]) -> Option<Av1SeqHeader> {
    let mut br = BitReader::new(payload);
    let seq_profile = br.read_bits(3)? as u8;
    let _still_picture = br.read_bit()?;
    let reduced_still_picture_header = br.read_bit()?;

    if reduced_still_picture_header == 1 {
        let _seq_level_idx = br.read_bits(5)?;
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
        for _ in 0..=operating_points_cnt_minus_1 {
            br.read_bits(12)?; // operating_point_idc[i]
            let seq_level_idx = br.read_bits(5)?;
            if seq_level_idx > 7 {
                br.read_bit()?; // seq_tier[i]
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
    Some(Av1SeqHeader {
        width: max_frame_width,
        height: max_frame_height,
        profile: seq_profile,
    })
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
