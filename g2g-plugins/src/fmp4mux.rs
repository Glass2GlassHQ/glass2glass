// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! The NAL / `avcC` / `hvcC` / sample-entry helpers shared by the MP4 muxer
//! [`Mp4MuxN`](crate::mp4muxn) and the Matroska muxer
//! [`MkvMuxN`](crate::mkvmuxn). H.264 carries `avc1`/`avcC` with SPS+PPS, H.265
//! `hvc1`/`hvcC` with VPS+SPS+PPS, and samples are AVCC-style 4-byte
//! length-prefixed NALUs. Pure `no_std + alloc`.

use alloc::vec::Vec;

use g2g_core::{Caps, Dim, PixelAspect, VideoCodec};

use crate::annexb::au_is_keyframe;
// The Annex-B split / NAL-type / parameter-set / AVCC helpers moved to the
// ungated `annexb` module (M662, the no_std FLV muxer shares them);
// re-exported so this module's users keep their import path.
pub(crate) use crate::annexb::{avcc_record, avcc_sample, parameter_sets, split_annexb};

use crate::mp4box::mp4_box;

// --- box writers ----------------------------------------------------------

// the size of a video track's pictures and the aspect of their pixels, from its caps
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct VideoGeometry {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) pixel_aspect: PixelAspect,
}

impl VideoGeometry {
    // `None` for anything but compressed video of fixed size
    pub(crate) fn from_caps(caps: &Caps) -> Option<Self> {
        let Caps::CompressedVideo {
            width: Dim::Fixed(width),
            height: Dim::Fixed(height),
            pixel_aspect,
            ..
        } = caps
        else {
            return None;
        };
        Some(Self {
            width: *width,
            height: *height,
            pixel_aspect: *pixel_aspect,
        })
    }
}

/// The `VisualSampleEntry` box `fourcc` holding the codec's `config` box, and a
/// `pasp` box when the pixel aspect is known.
pub(crate) fn visual_sample_entry(
    fourcc: &[u8; 4],
    geometry: &VideoGeometry,
    config: &[u8],
) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&[0u8; 6]); // reserved
    p.extend_from_slice(&1u16.to_be_bytes()); // data reference index
    p.extend_from_slice(&[0u8; 16]); // pre_defined/reserved
    p.extend_from_slice(&(geometry.width as u16).to_be_bytes());
    p.extend_from_slice(&(geometry.height as u16).to_be_bytes());
    p.extend_from_slice(&0x00480000u32.to_be_bytes()); // 72 dpi horiz
    p.extend_from_slice(&0x00480000u32.to_be_bytes()); // 72 dpi vert
    p.extend_from_slice(&[0u8; 4]); // reserved
    p.extend_from_slice(&1u16.to_be_bytes()); // frame count
    p.extend_from_slice(&[0u8; 32]); // compressor name
    p.extend_from_slice(&0x0018u16.to_be_bytes()); // depth 24
    p.extend_from_slice(&0xFFFFu16.to_be_bytes()); // pre_defined -1
    p.extend_from_slice(config);
    if let PixelAspect::Exact(ratio) = geometry.pixel_aspect {
        let spacing = [
            ratio.numerator().to_be_bytes(),
            ratio.denominator().to_be_bytes(),
        ];
        p.extend_from_slice(&mp4_box(b"pasp", &spacing.concat()));
    }
    mp4_box(fourcc, &p)
}

/// The sample entry fourcc and config box for H.26x: `avc1`+`avcC` for H.264,
/// `hvc1`+`hvcC` for H.265. `param_sets` is the ordered set the moov needs
/// (SPS,PPS for H.264; VPS,SPS,PPS for H.265).
pub(crate) fn h26x_sample_config(
    codec: VideoCodec,
    param_sets: &[&[u8]],
) -> (&'static [u8; 4], Vec<u8>) {
    match codec {
        VideoCodec::H265 => (b"hvc1", hvcc(param_sets)),
        _ => (b"avc1", avcc(param_sets)),
    }
}

/// `avcC` decoder configuration record box. `param_sets` is [SPS, PPS].
fn avcc(param_sets: &[&[u8]]) -> Vec<u8> {
    mp4_box(b"avcC", &avcc_record(param_sets))
}

/// One Annex-B video access unit as the MP4 and Matroska muxers store it, and
/// whether it is a sync sample: H.26x as AVCC length-prefixed NALUs, AV1 without
/// temporal delimiters, VP8 / VP9 / MPEG-4 Part 2 verbatim.
pub(crate) fn video_sample(codec: VideoCodec, au: &[u8]) -> (Vec<u8>, bool) {
    match codec {
        VideoCodec::H264 | VideoCodec::H265 => {
            (avcc_sample(&split_annexb(au)), au_is_keyframe(codec, au))
        }
        VideoCodec::Mpeg4Part2 => (au.to_vec(), au_is_keyframe(codec, au)),
        VideoCodec::Vp8 => (au.to_vec(), vp8_keyframe(au)),
        // JPEG has no inter prediction
        VideoCodec::Mjpeg => (au.to_vec(), true),
        VideoCodec::Av1 => (
            crate::av1parse::strip_temporal_delimiters(au),
            crate::av1parse::av1_keyframe(au),
        ),
        _ => (au.to_vec(), vp9_keyframe(au)),
    }
}

/// VP8 keyframe flag: the frame tag's bit 0 (`0` = key frame). Shared by the
/// container muxers that store VP8 frames verbatim (Matroska, MP4).
pub(crate) fn vp8_keyframe(frame: &[u8]) -> bool {
    frame.first().is_some_and(|b| b & 1 == 0)
}

/// VP9 keyframe flag from the uncompressed frame header: frame_marker(2)=0b10,
/// profile(2) (+1 reserved bit for profile 3), show_existing_frame(1), then
/// frame_type(1) where `0` = key frame. Superframes are not unpacked (the vpx
/// encoder emits a single frame per buffer). Shared by the VP9-carrying muxers.
pub(crate) fn vp9_keyframe(frame: &[u8]) -> bool {
    let Some(&b0) = frame.first() else {
        return false;
    };
    let bit = |i: u32| (b0 >> (7 - i)) & 1;
    if ((bit(0) << 1) | bit(1)) != 0b10 {
        return false; // not a valid VP9 frame marker
    }
    let profile = (bit(3) << 1) | bit(2);
    let mut cursor: u32 = 4;
    if profile == 3 {
        cursor += 1; // reserved_zero
    }
    if bit(cursor) == 1 {
        return false; // show_existing_frame: a repeat, not a key frame
    }
    cursor += 1;
    bit(cursor) == 0 // frame_type: 0 = key frame
}

/// `hvcC` decoder configuration record. `param_sets` is [VPS, SPS, PPS]. The
/// general profile_tier_level, chroma format, bit depths and temporal layers
/// are read from the SPS with its emulation prevention bytes removed. Parameter
/// sets stay in-band in each sample regardless.
fn hvcc(param_sets: &[&[u8]]) -> Vec<u8> {
    mp4_box(b"hvcC", &hvcc_record(param_sets))
}

/// The `hvcC` HEVCDecoderConfigurationRecord body (no box header), shared as the
/// Matroska `CodecPrivate` for `V_MPEGH/ISO/HEVC`. See [`hvcc`] for the field
/// layout; `param_sets` is [VPS, SPS, PPS].
pub(crate) fn hvcc_record(param_sets: &[&[u8]]) -> Vec<u8> {
    let vps = param_sets[0];
    let sps = param_sets[1];
    let pps = param_sets[2];
    let fields = crate::annexb::hvcc_sps_fields(sps)
        .expect("parameter_sets refuses an SPS hvcC cannot describe");

    let mut p = Vec::new();
    p.push(1); // configuration version
    p.extend_from_slice(&fields.profile_tier_level);
    p.extend_from_slice(&0xF000u16.to_be_bytes()); // reserved + min_spatial_segmentation_idc 0
    p.push(0xFC); // reserved + parallelismType 0
    p.push(0xFC | fields.chroma_format_idc);
    p.push(0xF8 | fields.bit_depth_luma_minus8);
    p.push(0xF8 | fields.bit_depth_chroma_minus8);
    p.extend_from_slice(&0u16.to_be_bytes()); // avgFrameRate (unspecified)
                                              // constantFrameRate(2)=0 | numTemporalLayers(3) | temporalIdNested(1) | lengthSizeMinusOne(2)=3
    p.push(
        ((fields.max_sub_layers_minus1 + 1) << 3) | (u8::from(fields.temporal_id_nesting) << 2) | 3,
    );
    p.push(3); // numOfArrays: VPS, SPS, PPS

    for (ty, nalu) in [(32u8, vps), (33u8, sps), (34u8, pps)] {
        p.push(0x80 | ty); // array_completeness=1, NAL_unit_type
        p.extend_from_slice(&1u16.to_be_bytes()); // numNalus
        p.extend_from_slice(&(nalu.len() as u16).to_be_bytes());
        p.extend_from_slice(nalu);
    }
    p
}

/// The `AV1CodecConfigurationRecord` body (no box header): the `av1C` box
/// payload and the Matroska `CodecPrivate` for `V_AV1`. Header fields come
/// from the parsed sequence header; `seq_obu` (the whole sequence-header OBU)
/// rides as `configOBUs` verbatim, so a player re-parses authoritative values.
pub(crate) fn av1c_record(seq: &crate::av1parse::Av1SeqHeader, seq_obu: &[u8]) -> Vec<u8> {
    let mut p = Vec::with_capacity(4 + seq_obu.len());
    p.push(0x81); // marker(1)=1 | version(7)=1
    p.push((seq.profile << 5) | (seq.level & 0x1F));
    p.push(
        ((seq.tier & 1) << 7)
            | ((seq.high_bitdepth as u8) << 6)
            | ((seq.twelve_bit as u8) << 5)
            | ((seq.monochrome as u8) << 4)
            | ((seq.subsampling_x as u8) << 3)
            | ((seq.subsampling_y as u8) << 2)
            | (seq.chroma_sample_position & 3),
    );
    p.push(0); // reserved + no initial_presentation_delay
    p.extend_from_slice(seq_obu);
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annexb::nalu_type;
    use alloc::vec;

    #[test]
    fn annexb_splitter_handles_both_start_codes() {
        // 4-byte code, 3-byte code, trailing NALU
        let data = [
            0, 0, 0, 1, 0x67, 0xAA, // SPS
            0, 0, 1, 0x68, 0xBB, // PPS
            0, 0, 0, 1, 0x65, 0xCC, 0xDD, // IDR
        ];
        let nalus = split_annexb(&data);
        assert_eq!(nalus.len(), 3);
        assert_eq!(nalu_type(VideoCodec::H264, nalus[0]), 7);
        assert_eq!(nalu_type(VideoCodec::H264, nalus[1]), 8);
        assert_eq!(nalu_type(VideoCodec::H264, nalus[2]), 5);
        assert_eq!(nalus[2], &[0x65, 0xCC, 0xDD]);
    }

    #[test]
    fn hevc_nal_type_reads_header_bits() {
        // H.265 NAL type is bits 1..6 of byte 0. VPS=32 -> 0x40, IDR_W_RADL=19
        // -> 0x26.
        assert_eq!(nalu_type(VideoCodec::H265, &[0x40, 0x01]), 32);
        assert_eq!(nalu_type(VideoCodec::H265, &[0x26, 0x01]), 19);
    }

    #[test]
    fn hvcc_describes_a_main10_sps_from_its_unescaped_bytes() {
        let stream = include_bytes!("../tests/fixtures/h265_320x192_main10.hevc");
        let nalus = split_annexb(stream);
        let sets = parameter_sets(VideoCodec::H265, &nalus).expect("VPS, SPS and PPS");
        let (vps, sps, pps) = (sets[0], sets[1], sets[2]);
        let escaped_ptl = &sps[3..15];
        let rbsp = crate::annexb::strip_emulation_prevention(&sps[2..]);
        let ptl = &rbsp[1..13];
        assert_ne!(
            escaped_ptl, ptl,
            "the fixture's SPS escapes a byte in its PTL"
        );

        let cfg = hvcc(&sets);
        assert_eq!(&cfg[4..8], b"hvcC");
        let payload = &cfg[8..];
        assert_eq!(payload[0], 1, "configuration version");
        assert_eq!(&payload[1..13], ptl, "general PTL without the escape byte");
        assert_eq!(payload[16] & 0x03, 1, "chroma 4:2:0");
        assert_eq!(payload[17] & 0x07, 2, "luma bit depth 10");
        assert_eq!(payload[18] & 0x07, 2, "chroma bit depth 10");
        assert_eq!(payload[22], 3, "numOfArrays = VPS, SPS, PPS");
        for set in [vps, sps, pps] {
            assert!(cfg.windows(set.len()).any(|w| w == set));
        }
    }

    #[test]
    fn avcc_sample_length_prefixes_every_nalu() {
        let nalus: Vec<&[u8]> = vec![&[0x67, 1, 2], &[0x65, 3]];
        let s = avcc_sample(&nalus);
        assert_eq!(
            s,
            vec![0, 0, 0, 3, 0x67, 1, 2, 0, 0, 0, 2, 0x65, 3],
            "4-byte BE length before each NALU"
        );
    }

    #[test]
    fn avcc_sample_drops_empty_nalus() {
        let nalus: Vec<&[u8]> = vec![&[0x67, 1], &[], &[0x65, 3]];
        assert_eq!(
            avcc_sample(&nalus),
            vec![0, 0, 0, 2, 0x67, 1, 0, 0, 0, 2, 0x65, 3]
        );
    }

    #[test]
    fn boxes_carry_size_and_type() {
        let b = mp4_box(b"mdat", &[1, 2, 3]);
        assert_eq!(&b[..4], &11u32.to_be_bytes());
        assert_eq!(&b[4..8], b"mdat");
        assert_eq!(&b[8..], &[1, 2, 3]);
    }
}
