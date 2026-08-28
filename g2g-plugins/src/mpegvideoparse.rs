//! MPEG-1 / MPEG-2 video elementary-stream parser.
//!
//! Frames the stream into one access unit per coded picture (the picture header
//! plus the sequence / GOP headers, extensions and user data that lead it, plus
//! its slices) and refines caps from the sequence header (`00 00 01 B3`,
//! ISO/IEC 13818-2 6.2.2.1) and the sequence extension that may follow it.
//!
//! A stream with no sequence extension is MPEG-1 (ISO/IEC 11172-2): the 12-bit
//! sizes stand alone and `aspect_ratio_information` indexes the pel aspect
//! ratio table. With an extension it is MPEG-2: the two 2-bit size extensions
//! widen the sizes to 14 bits, the framerate extension scales the frame rate,
//! and `aspect_ratio_information` indexes display aspect ratios instead, so the
//! sample aspect is derived from the coded size. Both decode with the same
//! `VideoCodec::Mpeg2` link.

use g2g_core::{PropertySpec, VideoCodec};

use crate::startcodeparse::{
    StartCodeCodec, StartCodeParse, StartCodeRole, VideoGeometry, PIXEL_ASPECT_PROPERTY,
};

/// `sequence_header_code`: the sequence header that carries the geometry.
const SEQUENCE_HEADER_CODE: u8 = 0xB3;
/// `group_start_code`: the GOP header, which only ever leads an I-picture.
const GROUP_START_CODE: u8 = 0xB8;
/// `picture_start_code`: the coded picture itself.
const PICTURE_START_CODE: u8 = 0x00;

/// MPEG-1 / MPEG-2 video parser.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::mpegvideoparse::MpegVideoParse;
///
/// let parse = MpegVideoParse::new();
/// ```
pub type MpegVideoParse = StartCodeParse<MpegVideoCodec>;

/// MPEG-1 / MPEG-2 hooks for [`StartCodeParse`].
#[derive(Debug)]
pub struct MpegVideoCodec;

impl StartCodeCodec for MpegVideoCodec {
    const CODEC: VideoCodec = VideoCodec::Mpeg2;
    const NAME: &'static str = "MPEG-1/2 video parser";
    const DESCRIPTION: &'static str =
        "Frames an MPEG-1 / MPEG-2 video stream into access units and reads its sequence header";
    const PROPERTIES: &'static [PropertySpec] = &[PIXEL_ASPECT_PROPERTY];

    fn start_code_role(code: u8) -> StartCodeRole {
        match code {
            PICTURE_START_CODE => StartCodeRole::Picture,
            SEQUENCE_HEADER_CODE | GROUP_START_CODE => StartCodeRole::Leads,
            // Slices (0x01..=0xAF), extensions, user data and the sequence end
            // all belong to the picture in progress.
            _ => StartCodeRole::Continues,
        }
    }

    /// The sequence header is shared with the program- and transport-stream
    /// demuxers, which read the same fields to stamp sparsely timestamped
    /// pictures.
    fn geometry(au: &[u8]) -> Option<VideoGeometry> {
        let header = crate::mpeg2video::parse_sequence_header(au)?;
        Some(VideoGeometry {
            width: header.width,
            height: header.height,
            framerate: Some(header.framerate_q16),
            pixel_aspect: header.pixel_aspect,
        })
    }

    fn au_is_keyframe(au: &[u8]) -> bool {
        crate::annexb::au_is_keyframe(VideoCodec::Mpeg2, au)
    }
}

/// Fuzzing entry: parse an MPEG-1 / MPEG-2 sequence header and its extension out
/// of an access unit. Exposed only under `--cfg fuzzing` (cargo-fuzz).
#[cfg(fuzzing)]
pub fn fuzz_parse(data: &[u8]) {
    let _ = MpegVideoCodec::geometry(data);
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    use g2g_core::frame::Frame;
    use g2g_core::memory::SystemSlice;
    use g2g_core::{
        AsyncElement, Caps, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket,
        PropValue, PushOutcome, Rate,
    };

    use crate::annexb::BitWriter;

    /// `extension_start_code`, which the fixtures write by hand.
    const EXTENSION_START_CODE: u8 = 0xB5;
    /// `extension_start_code_identifier` of the sequence extension.
    const SEQUENCE_EXTENSION_ID: u32 = 1;

    /// `frame_rate_code` for 25 fps, the PAL / DVD rate the fixtures use.
    const FRAME_RATE_CODE_25: u32 = 3;
    /// `frame_rate_code` for 29.97 fps (30000/1001).
    const FRAME_RATE_CODE_29_97: u32 = 4;
    /// `aspect_ratio_information` 2: a 4:3 display aspect under MPEG-2, and the
    /// 0.6735 pel aspect ratio under MPEG-1.
    const ASPECT_CODE_2: u32 = 2;
    /// `aspect_ratio_information` 1: square samples.
    const ASPECT_CODE_SQUARE: u32 = 1;

    /// A sequence header unit for `width` x `height`, start code included. Only
    /// the fields this parser reads are given real values; the rest are the
    /// spec's fixed widths, zero filled.
    fn sequence_header(width: u32, height: u32, aspect: u32, frame_rate_code: u32) -> Vec<u8> {
        let mut w = BitWriter::default();
        w.write_bits(width & 0xFFF, 12); // horizontal_size_value
        w.write_bits(height & 0xFFF, 12); // vertical_size_value
        w.write_bits(aspect, 4); // aspect_ratio_information
        w.write_bits(frame_rate_code, 4); // frame_rate_code
        w.write_bits(0x3FFFF, 18); // bit_rate_value
        w.write_bit(1); // marker_bit
        w.write_bits(0, 10); // vbv_buffer_size_value
        w.write_bit(0); // constrained_parameters_flag
        w.write_bit(0); // load_intra_quantiser_matrix
        w.write_bit(0); // load_non_intra_quantiser_matrix
        w.align_to_byte();
        let mut unit = vec![0u8, 0, 1, SEQUENCE_HEADER_CODE];
        unit.extend_from_slice(&w.into_bytes());
        unit
    }

    /// A sequence extension unit, start code included.
    fn sequence_extension(
        horizontal_extension: u32,
        vertical_extension: u32,
        frame_rate_extension_n: u32,
        frame_rate_extension_d: u32,
    ) -> Vec<u8> {
        let mut w = BitWriter::default();
        w.write_bits(SEQUENCE_EXTENSION_ID, 4); // extension_start_code_identifier
        w.write_bits(0x48, 8); // profile_and_level_indication (main@main)
        w.write_bit(0); // progressive_sequence
        w.write_bits(1, 2); // chroma_format (4:2:0)
        w.write_bits(horizontal_extension, 2);
        w.write_bits(vertical_extension, 2);
        w.write_bits(0, 12); // bit_rate_extension
        w.write_bit(1); // marker_bit
        w.write_bits(0, 8); // vbv_buffer_size_extension
        w.write_bit(0); // low_delay
        w.write_bits(frame_rate_extension_n, 2);
        w.write_bits(frame_rate_extension_d, 5);
        w.align_to_byte();
        let mut unit = vec![0u8, 0, 1, EXTENSION_START_CODE];
        unit.extend_from_slice(&w.into_bytes());
        unit
    }

    /// A picture header unit whose `picture_coding_type` is `coding_type`
    /// (1 = I, 2 = P, 3 = B), start code included.
    fn picture_header(temporal_reference: u32, coding_type: u32) -> Vec<u8> {
        let mut w = BitWriter::default();
        w.write_bits(temporal_reference, 10);
        w.write_bits(coding_type, 3);
        w.write_bits(0xFFFF, 16); // vbv_delay
        w.align_to_byte();
        let mut unit = vec![0u8, 0, 1, PICTURE_START_CODE];
        unit.extend_from_slice(&w.into_bytes());
        unit
    }

    /// A slice unit (`slice_start_code` 0x01), start code included.
    fn slice(payload: u8) -> Vec<u8> {
        vec![0, 0, 1, 0x01, payload, 0x11]
    }

    #[test]
    fn reads_mpeg2_geometry_from_header_plus_extension() {
        // 1920x1080 needs the extension bits: 1920 & 0xFFF = 1920 fits, but
        // 1080 also fits, so use a size that needs them: 4096x2160.
        let mut au = sequence_header(
            4096 & 0xFFF,
            2160 & 0xFFF,
            ASPECT_CODE_2,
            FRAME_RATE_CODE_25,
        );
        au.extend_from_slice(&sequence_extension(1, 0, 0, 0));
        let info = MpegVideoCodec::geometry(&au).expect("sequence header must parse");
        assert_eq!((info.width, info.height), (4096, 2160));
        assert_eq!(info.framerate, Some(25 << 16));
    }

    #[test]
    fn reads_mpeg2_dvd_geometry_and_sample_aspect() {
        // 720x576 with a 4:3 display aspect: sample aspect = 4*576 : 3*720 = 16:15.
        let mut au = sequence_header(720, 576, ASPECT_CODE_2, FRAME_RATE_CODE_25);
        au.extend_from_slice(&sequence_extension(0, 0, 0, 0));
        let info = MpegVideoCodec::geometry(&au).expect("sequence header must parse");
        assert_eq!((info.width, info.height), (720, 576));
        assert_eq!(info.framerate, Some(25 << 16));
        assert_eq!(info.pixel_aspect, Some((16, 15)));
    }

    #[test]
    fn applies_the_frame_rate_extension() {
        // 29.97 base scaled by (n+1)/(d+1) = 2/1 -> 59.94 fps.
        let mut au = sequence_header(720, 480, ASPECT_CODE_SQUARE, FRAME_RATE_CODE_29_97);
        au.extend_from_slice(&sequence_extension(0, 0, 1, 0));
        let info = MpegVideoCodec::geometry(&au).expect("sequence header must parse");
        let expected = u32::try_from(((30000u64 << 16) * 2) / 1001).unwrap();
        assert_eq!(info.framerate, Some(expected));
        assert_eq!(info.framerate.unwrap() >> 16, 59, "~59.94 fps");
        assert_eq!(info.pixel_aspect, Some((1, 1)));
    }

    #[test]
    fn reads_mpeg1_geometry_and_pel_aspect_without_an_extension() {
        // A PAL VCD: 352x288, pel aspect code 8 (0.9157), 25 fps.
        const PAL_VCD_ASPECT_CODE: u32 = 8;
        let mut au = sequence_header(352, 288, PAL_VCD_ASPECT_CODE, FRAME_RATE_CODE_25);
        au.extend_from_slice(&picture_header(0, 1));
        let info = MpegVideoCodec::geometry(&au).expect("sequence header must parse");
        assert_eq!((info.width, info.height), (352, 288));
        assert_eq!(info.framerate, Some(25 << 16));
        // Sample aspect is the reciprocal of the 0.9157 pel aspect: 10000:9157.
        assert_eq!(info.pixel_aspect, Some((10000, 9157)));
    }

    #[test]
    fn rejects_a_zero_dimension_and_a_truncated_header() {
        let zero_height = sequence_header(720, 0, ASPECT_CODE_2, FRAME_RATE_CODE_25);
        assert!(
            MpegVideoCodec::geometry(&zero_height).is_none(),
            "a zero coded height is not a geometry"
        );
        // The header cut off inside the 12-bit sizes: the bit reader runs out
        // rather than reading past the buffer.
        let truncated = vec![0u8, 0, 1, SEQUENCE_HEADER_CODE, 0x2D];
        assert!(MpegVideoCodec::geometry(&truncated).is_none());
        assert!(MpegVideoCodec::geometry(&[]).is_none());
    }

    #[test]
    fn a_reserved_frame_rate_code_is_rejected() {
        const RESERVED_FRAME_RATE_CODE: u32 = 9;
        let au = sequence_header(720, 576, ASPECT_CODE_2, RESERVED_FRAME_RATE_CODE);
        assert!(
            MpegVideoCodec::geometry(&au).is_none(),
            "a rate the table does not define cannot fixate caps"
        );
    }

    #[test]
    fn access_units_split_at_each_picture() {
        // sequence header + I-picture + slice | GOP + P-picture + slice.
        let mut stream = sequence_header(720, 576, ASPECT_CODE_2, FRAME_RATE_CODE_25);
        stream.extend_from_slice(&picture_header(0, 1));
        stream.extend_from_slice(&slice(0xAA));
        let second = stream.len();
        stream.extend_from_slice(&[0, 0, 1, GROUP_START_CODE, 0x00, 0x00, 0x00, 0x00]);
        stream.extend_from_slice(&picture_header(1, 2));
        stream.extend_from_slice(&slice(0xBB));
        let starts = crate::startcodeparse::au_starts_by(&stream, MpegVideoCodec::start_code_role);
        assert_eq!(starts, vec![0, second]);
    }

    #[test]
    fn keyframe_flag_follows_the_picture_coding_type() {
        let mut intra = sequence_header(720, 576, ASPECT_CODE_2, FRAME_RATE_CODE_25);
        intra.extend_from_slice(&picture_header(0, 1));
        assert!(MpegVideoCodec::au_is_keyframe(&intra));

        let mut predicted = picture_header(1, 2);
        predicted.extend_from_slice(&slice(0xBB));
        assert!(!MpegVideoCodec::au_is_keyframe(&predicted));
    }

    // -- Element-level tests (drive MpegVideoParse::process directly) --------

    #[derive(Default)]
    struct RecordingSink {
        packets: Vec<PipelinePacket>,
    }

    impl OutputSink for RecordingSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");
            self.packets.push(packet);
            core::task::Poll::Ready(Ok(PushOutcome::Accepted))
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

    fn mpeg2_caps() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::Mpeg2,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }
    }

    fn data_payloads(sink: &RecordingSink) -> Vec<Vec<u8>> {
        sink.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => f.domain.as_system_slice().map(<[u8]>::to_vec),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn emits_caps_changed_then_one_frame_per_picture() {
        let mut parse = MpegVideoParse::new();
        parse.configure_pipeline(&mpeg2_caps()).unwrap();
        let mut sink = RecordingSink::default();

        let mut first = sequence_header(720, 576, ASPECT_CODE_2, FRAME_RATE_CODE_25);
        first.extend_from_slice(&sequence_extension(0, 0, 0, 0));
        first.extend_from_slice(&picture_header(0, 1));
        first.extend_from_slice(&slice(0xAA));
        let mut second = picture_header(1, 2);
        second.extend_from_slice(&slice(0xBB));

        let mut buffer = first.clone();
        buffer.extend_from_slice(&second);
        parse
            .process(
                PipelinePacket::DataFrame(frame_with_bytes(0, buffer)),
                &mut sink,
            )
            .await
            .unwrap();
        parse.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        match &sink.packets[0] {
            PipelinePacket::CapsChanged(Caps::CompressedVideo {
                width,
                height,
                framerate,
                ..
            }) => {
                assert_eq!(*width, Dim::Fixed(720));
                assert_eq!(*height, Dim::Fixed(576));
                assert_eq!(*framerate, Rate::Fixed(25 << 16));
            }
            other => panic!("expected CapsChanged first, got {other:?}"),
        }
        assert_eq!(data_payloads(&sink), vec![first, second]);
        assert_eq!(parse.caps_changes_emitted(), 1);
        assert_eq!(
            parse.get_property("pixel-aspect-ratio"),
            Some(PropValue::Fraction(16, 15))
        );
    }

    #[tokio::test]
    async fn reassembles_a_picture_split_across_buffers() {
        let mut parse = MpegVideoParse::new();
        parse.configure_pipeline(&mpeg2_caps()).unwrap();
        let mut sink = RecordingSink::default();

        let mut au = picture_header(0, 1);
        au.extend_from_slice(&slice(0xAA));
        let split = 5;
        parse
            .process(
                PipelinePacket::DataFrame(frame_with_bytes(0, au[..split].to_vec())),
                &mut sink,
            )
            .await
            .unwrap();
        parse
            .process(
                PipelinePacket::DataFrame(frame_with_bytes(1, au[split..].to_vec())),
                &mut sink,
            )
            .await
            .unwrap();
        parse.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        assert_eq!(data_payloads(&sink), vec![au]);
    }

    #[tokio::test]
    async fn does_not_re_emit_unchanged_caps() {
        let mut parse = MpegVideoParse::new();
        parse.configure_pipeline(&mpeg2_caps()).unwrap();
        let mut sink = RecordingSink::default();

        for seq in 0..3 {
            let mut au = sequence_header(720, 576, ASPECT_CODE_2, FRAME_RATE_CODE_25);
            au.extend_from_slice(&sequence_extension(0, 0, 0, 0));
            au.extend_from_slice(&picture_header(seq as u32, 1));
            parse
                .process(
                    PipelinePacket::DataFrame(frame_with_bytes(seq, au)),
                    &mut sink,
                )
                .await
                .unwrap();
        }
        parse.process(PipelinePacket::Eos, &mut sink).await.unwrap();
        assert_eq!(parse.caps_changes_emitted(), 1);
    }

    #[tokio::test]
    async fn rejects_non_mpeg_caps_in_intercept() {
        let parse = MpegVideoParse::new();
        let h264 = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        assert_eq!(parse.intercept_caps(&h264), Err(G2gError::CapsMismatch));
    }
}
