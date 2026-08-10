//! M952: per-frame video timestamp synthesis in the TS demuxer for the streams
//! M948 left alone. MPEG-2 access units take their display slot from the picture
//! header's `temporal_reference`, and an H.264 / H.265 stream that reorders
//! takes it from the picture order count, anchored on the real PES stamps.

use g2g_plugins::mpegts::{
    TsDemuxer, TsMuxer, STREAM_TYPE_H264, STREAM_TYPE_H265, STREAM_TYPE_MPEG2_VIDEO, TS_PACKET_LEN,
};

/// 25 fps frame period in 90 kHz units.
const PERIOD: u64 = 3600;
/// VUI `timing_info` for 25 fps: fps = time_scale / (2 * num_units_in_tick).
const NUM_UNITS_IN_TICK: u32 = 1;
const TIME_SCALE: u32 = 50;

/// Minimal MSB-first bit writer for the hand-built parameter sets and slices.
#[derive(Default)]
struct Bits {
    out: Vec<u8>,
    written: u32,
}

impl Bits {
    fn bit(&mut self, value: u32) {
        if self.written.is_multiple_of(8) {
            self.out.push(0);
        }
        if value & 1 == 1 {
            let last = self.out.len() - 1;
            self.out[last] |= 0x80 >> (self.written % 8);
        }
        self.written += 1;
    }

    fn bits(&mut self, value: u32, count: u32) {
        for shift in (0..count).rev() {
            self.bit(value >> shift);
        }
    }

    /// Unsigned exp-Golomb: `n - 1` leading zeros then the `n`-bit code `v + 1`.
    fn ue(&mut self, value: u32) {
        let code = value + 1;
        let n = 32 - code.leading_zeros();
        self.bits(code, 2 * n - 1);
    }

    /// Signed exp-Golomb.
    fn se(&mut self, value: i32) {
        let mapped = if value <= 0 {
            (-value as u32) * 2
        } else {
            (value as u32) * 2 - 1
        };
        self.ue(mapped);
    }

    /// Close the RBSP with `rbsp_trailing_bits()`.
    fn finish(mut self) -> Vec<u8> {
        self.bit(1);
        while !self.written.is_multiple_of(8) {
            self.bit(0);
        }
        self.out
    }
}

/// Insert emulation-prevention bytes, so a parameter set whose payload contains
/// a `00 00 01` run (the 32-bit VUI timing fields do) stays one NAL.
fn escape(nal: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut zeros = 0;
    for &byte in nal {
        if zeros >= 2 && byte <= 3 {
            out.push(0x03);
            zeros = 0;
        }
        out.push(byte);
        zeros = if byte == 0 { zeros + 1 } else { 0 };
    }
    out
}

fn annexb_nal(body: &[u8]) -> Vec<u8> {
    let mut out = Vec::from([0x00u8, 0x00, 0x00, 0x01]);
    out.extend_from_slice(&escape(body));
    out
}

// ============================================================================
// MPEG-2
// ============================================================================

/// An MPEG-2 sequence header: 720x576 at 25 fps (`frame_rate_code` 3).
fn mpeg2_sequence_header() -> Vec<u8> {
    let width = 720u32;
    let height = 576u32;
    Vec::from([
        0x00,
        0x00,
        0x01,
        0xB3,
        (width >> 4) as u8,
        (((width & 0x0F) << 4) | (height >> 8)) as u8,
        (height & 0xFF) as u8,
        0x13, // aspect_ratio_information 1, frame_rate_code 3 (25 fps)
    ])
}

/// An MPEG-2 GOP header (its fields do not matter here, only its presence: it
/// opens a new display-order base).
fn mpeg2_gop_header() -> Vec<u8> {
    Vec::from([0x00, 0x00, 0x01, 0xB8, 0x00, 0x08, 0x00, 0x40])
}

/// An MPEG-2 picture header with the given `temporal_reference` (the picture's
/// display index in its GOP) and `picture_coding_type` (1 = I, 2 = P, 3 = B).
fn mpeg2_picture_header(temporal_reference: u16, coding_type: u8) -> Vec<u8> {
    Vec::from([
        0x00,
        0x00,
        0x01,
        0x00,
        (temporal_reference >> 2) as u8,
        (((temporal_reference & 0x3) as u8) << 6) | (coding_type << 3),
        0xFF,
        0xF8,
    ])
}

/// One MPEG-2 access unit: the optional headers that open it, its picture
/// header, and a stand-in slice.
fn mpeg2_access_unit(
    sequence: bool,
    gop: bool,
    temporal_reference: u16,
    coding_type: u8,
) -> Vec<u8> {
    let mut au = Vec::new();
    if sequence {
        au.extend(mpeg2_sequence_header());
    }
    if gop {
        au.extend(mpeg2_gop_header());
    }
    au.extend(mpeg2_picture_header(temporal_reference, coding_type));
    au.extend_from_slice(&[0x00, 0x00, 0x01, 0x01, 0x0A, 0x0B, 0x0C]); // slice
    au
}

/// Two GOPs of `I P B B` in coded order, the first opening with a sequence
/// header. Display order runs 0..8, so an exact synthesis puts each picture on
/// its own slot.
fn mpeg2_stream() -> Vec<Vec<u8>> {
    let mut units = Vec::new();
    for gop in 0..2 {
        units.push(mpeg2_access_unit(gop == 0, true, 0, 1));
        units.push(mpeg2_access_unit(false, false, 3, 2));
        units.push(mpeg2_access_unit(false, false, 1, 3));
        units.push(mpeg2_access_unit(false, false, 2, 3));
    }
    units
}

/// Mux `units` with the given per-unit stamps, demux, and report the
/// `(pts, dts)` of each unit in coded order.
fn demux_timestamps(
    stream_type: u8,
    units: &[Vec<u8>],
    stamps: &[(Option<u64>, Option<u64>)],
) -> Vec<(Option<u64>, Option<u64>)> {
    let mut mux = TsMuxer::new(stream_type);
    let mut ts = Vec::new();
    for (au, (pts, dts)) in units.iter().zip(stamps) {
        ts.extend(mux.push_au(au, *pts, *dts));
    }
    let mut demux = TsDemuxer::new();
    for packet in ts.chunks(TS_PACKET_LEN) {
        demux.push_packet(packet);
    }
    demux.flush();
    demux
        .take_units()
        .into_iter()
        .filter(|u| u.stream_type == stream_type)
        .map(|u| (u.pts_90khz, u.dts_90khz))
        .collect()
}

/// Presentation timestamps only, for the cases that carry no DTS.
fn demux_pts(stream_type: u8, units: &[Vec<u8>], stamps: &[Option<u64>]) -> Vec<Option<u64>> {
    let with_dts: Vec<(Option<u64>, Option<u64>)> = stamps.iter().map(|p| (*p, None)).collect();
    demux_timestamps(stream_type, units, &with_dts)
        .into_iter()
        .map(|(pts, _)| pts)
        .collect()
}

/// An MPEG-2 transport stream stamped once per GOP: every picture in between
/// lands on its own display slot, B-frames included, and each GOP's base
/// advances by the pictures the one before it displayed.
#[test]
fn mpeg2_units_take_their_slot_from_temporal_reference() {
    let base = 90_000u64;
    let units = mpeg2_stream();
    let mut stamps = vec![None; units.len()];
    stamps[0] = Some(base);
    let pts = demux_pts(STREAM_TYPE_MPEG2_VIDEO, &units, &stamps);
    let expected: Vec<Option<u64>> = [0u64, 3, 1, 2, 4, 7, 5, 6]
        .iter()
        .map(|slot| Some(base + slot * PERIOD))
        .collect();
    assert_eq!(
        pts, expected,
        "each picture presents at its GOP base plus temporal_reference periods"
    );
}

/// A real stamp on a later GOP re-anchors it: synthesis follows the mux's own
/// timeline rather than the arithmetic one.
#[test]
fn mpeg2_a_real_stamp_reanchors_the_gop() {
    let base = 90_000u64;
    let drift = 90u64;
    let anchor = base + 4 * PERIOD + drift;
    let units = mpeg2_stream();
    let mut stamps = vec![None; units.len()];
    stamps[0] = Some(base);
    stamps[4] = Some(anchor);
    let pts = demux_pts(STREAM_TYPE_MPEG2_VIDEO, &units, &stamps);
    assert_eq!(
        pts,
        vec![
            Some(base),
            Some(base + 3 * PERIOD),
            Some(base + PERIOD),
            Some(base + 2 * PERIOD),
            Some(anchor),
            Some(anchor + 3 * PERIOD),
            Some(anchor + PERIOD),
            Some(anchor + 2 * PERIOD),
        ],
    );
}

/// A stamped MPEG-2 unit also seeds the decode timeline: the units after it get
/// a DTS one frame period apart, in coded order.
#[test]
fn mpeg2_decode_timestamps_advance_in_coded_order() {
    let base = 90_000u64;
    let decode = base - 2 * PERIOD;
    let units = mpeg2_stream();
    let mut stamps = vec![(None, None); units.len()];
    stamps[0] = (Some(base), Some(decode));
    let dts: Vec<Option<u64>> = demux_timestamps(STREAM_TYPE_MPEG2_VIDEO, &units, &stamps)
        .into_iter()
        .map(|(_, dts)| dts)
        .collect();
    assert_eq!(
        dts,
        (0..8)
            .map(|i| Some(decode + i * PERIOD))
            .collect::<Vec<_>>(),
    );
}

/// An MPEG-2 stream with no sequence header has no frame period, so its units
/// stay unstamped rather than land on a guessed grid.
#[test]
fn mpeg2_without_a_sequence_header_is_left_unstamped() {
    let base = 90_000u64;
    let units: Vec<Vec<u8>> = (0..4)
        .map(|i| mpeg2_access_unit(false, i == 0, i as u16, 1))
        .collect();
    let mut stamps = vec![None; units.len()];
    stamps[0] = Some(base);
    let pts = demux_pts(STREAM_TYPE_MPEG2_VIDEO, &units, &stamps);
    assert_eq!(pts, vec![Some(base), None, None, None]);
}

// ============================================================================
// H.264 with B-frame reordering
// ============================================================================

/// An H.264 SPS NAL: baseline geometry, POC type 0 (what any B-frame stream
/// codes), and VUI `timing_info` at 25 fps.
fn h264_sps() -> Vec<u8> {
    let mut b = Bits::default();
    b.ue(0); // seq_parameter_set_id
    b.ue(0); // log2_max_frame_num_minus4
    b.ue(0); // pic_order_cnt_type
    b.ue(0); // log2_max_pic_order_cnt_lsb_minus4 -> 16 counts before a wrap
    b.ue(2); // max_num_ref_frames
    b.bit(0); // gaps_in_frame_num_value_allowed_flag
    b.ue(79); // pic_width_in_mbs_minus1 -> 1280
    b.ue(44); // pic_height_in_map_units_minus1 -> 720
    b.bit(1); // frame_mbs_only_flag
    b.bit(1); // direct_8x8_inference_flag
    b.bit(0); // frame_cropping_flag
    b.bit(1); // vui_parameters_present_flag
    b.bit(0); // aspect_ratio_info_present_flag
    b.bit(0); // overscan_info_present_flag
    b.bit(0); // video_signal_type_present_flag
    b.bit(0); // chroma_loc_info_present_flag
    b.bit(1); // timing_info_present_flag
    b.bits(NUM_UNITS_IN_TICK, 32);
    b.bits(TIME_SCALE, 32);
    b.bit(1); // fixed_frame_rate_flag
    b.bit(0); // nal_hrd_parameters_present_flag
    b.bit(0); // vcl_hrd_parameters_present_flag
    b.bit(0); // pic_struct_present_flag
    b.bit(0); // bitstream_restriction_flag
    let mut nal = Vec::from([0x67u8, 66, 0x00, 30]);
    nal.extend(b.finish());
    annexb_nal(&nal)
}

/// An H.264 PPS NAL. The order-count synthesis reads it for
/// `bottom_field_pic_order_in_frame_present_flag`, so a stream that carries no
/// PPS in band gets no synthesis at all.
fn h264_pps() -> Vec<u8> {
    let mut b = Bits::default();
    b.ue(0); // pic_parameter_set_id
    b.ue(0); // seq_parameter_set_id
    b.bit(1); // entropy_coding_mode_flag
    b.bit(0); // bottom_field_pic_order_in_frame_present_flag
    b.ue(0); // num_slice_groups_minus1
    b.ue(1); // num_ref_idx_l0_default_active_minus1
    b.ue(0); // num_ref_idx_l1_default_active_minus1
    b.bit(0); // weighted_pred_flag
    b.bits(0, 2); // weighted_bipred_idc
    b.se(0); // pic_init_qp_minus26
    b.se(0); // pic_init_qs_minus26
    b.se(0); // chroma_qp_index_offset
    b.bit(1); // deblocking_filter_control_present_flag
    b.bit(0); // constrained_intra_pred_flag
    b.bit(0); // redundant_pic_cnt_present_flag
    let mut nal = Vec::from([0x68u8]);
    nal.extend(b.finish());
    annexb_nal(&nal)
}

/// One coded picture: an IDR, a reference P, or a non-reference B, carrying the
/// order-count lsb that fixes its display slot.
fn h264_slice(kind: PictureKind, frame_num: u32, pic_order_cnt_lsb: u32) -> Vec<u8> {
    let (header, slice_type) = match kind {
        PictureKind::Idr => (0x65u8, 7),     // nal_ref_idc 3, IDR, I slice
        PictureKind::Reference => (0x41, 5), // nal_ref_idc 2, non-IDR, P slice
        PictureKind::NonReference => (0x01, 6), // nal_ref_idc 0, non-IDR, B slice
    };
    let mut b = Bits::default();
    b.ue(0); // first_mb_in_slice
    b.ue(slice_type);
    b.ue(0); // pic_parameter_set_id
    b.bits(frame_num, 4);
    if kind == PictureKind::Idr {
        b.ue(0); // idr_pic_id
    }
    b.bits(pic_order_cnt_lsb, 4);
    let mut nal = Vec::from([header]);
    nal.extend(b.finish());
    annexb_nal(&nal)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PictureKind {
    Idr,
    Reference,
    NonReference,
}

/// Seven pictures in coded order `I P B B P B B`, whose display order is
/// `0 1 2 3 4 5 6`. H.264 counts fields, so the order count steps 2 per frame.
const H264_CODED_PICTURES: [(PictureKind, u32, u32); 7] = [
    (PictureKind::Idr, 0, 0),           // display 0
    (PictureKind::Reference, 1, 6),     // display 3
    (PictureKind::NonReference, 2, 2),  // display 1
    (PictureKind::NonReference, 2, 4),  // display 2
    (PictureKind::Reference, 2, 12),    // display 6
    (PictureKind::NonReference, 3, 8),  // display 4
    (PictureKind::NonReference, 3, 10), // display 5
];

fn h264_stream() -> Vec<Vec<u8>> {
    H264_CODED_PICTURES
        .iter()
        .enumerate()
        .map(|(index, &(kind, frame_num, poc_lsb))| {
            let mut au = Vec::new();
            if index == 0 {
                au.extend(h264_sps());
                au.extend(h264_pps());
            }
            au.extend(h264_slice(kind, frame_num, poc_lsb));
            au
        })
        .collect()
}

/// A reordering H.264 stream stamped twice: the pair fixes how far apart the
/// display slots are, and every unstamped unit after them lands on its own,
/// which for a B frame is behind the anchor it was coded after.
#[test]
fn h264_reordered_units_land_in_display_order() {
    let base = 90_000u64;
    let anchor = base + 6 * PERIOD;
    let stamps = [Some(base), None, None, None, Some(anchor), None, None];
    let pts = demux_pts(STREAM_TYPE_H264, &h264_stream(), &stamps);
    assert_eq!(
        pts,
        vec![
            Some(base),
            None, // no scale yet: one stamp cannot say how long a count is
            None,
            None,
            Some(anchor),
            Some(base + 4 * PERIOD),
            Some(base + 5 * PERIOD),
        ],
        "the B frames after the second stamp present before it, on their own slots"
    );
}

/// A fully stamped reordering stream keeps every timestamp it carries, the
/// out-of-order ones included.
#[test]
fn h264_a_fully_stamped_reordering_stream_is_unchanged() {
    let base = 45_000u64;
    let stamps: Vec<Option<u64>> = [0u64, 3, 1, 2, 6, 4, 5]
        .iter()
        .map(|slot| Some(base + slot * PERIOD))
        .collect();
    let pts = demux_pts(STREAM_TYPE_H264, &h264_stream(), &stamps);
    assert_eq!(pts, stamps);
}

/// The synthesis needs the picture parameter set to read a slice header, so a
/// stream that never sends one is left alone rather than stamped on a guess.
#[test]
fn h264_without_a_picture_parameter_set_is_left_unstamped() {
    let base = 90_000u64;
    let units: Vec<Vec<u8>> = H264_CODED_PICTURES
        .iter()
        .enumerate()
        .map(|(index, &(kind, frame_num, poc_lsb))| {
            let mut au = Vec::new();
            if index == 0 {
                au.extend(h264_sps());
            }
            au.extend(h264_slice(kind, frame_num, poc_lsb));
            au
        })
        .collect();
    let stamps = [
        Some(base),
        None,
        None,
        None,
        Some(base + 6 * PERIOD),
        None,
        None,
    ];
    let pts = demux_pts(STREAM_TYPE_H264, &units, &stamps);
    assert_eq!(pts[5..], [None, None]);
}

// ============================================================================
// H.265 with B-frame reordering
// ============================================================================

/// An H.265 SPS NAL: 1280x720, `sps_max_num_reorder_pics` 2 (so the stream may
/// reorder), no VUI, so the display-slot spacing has to be measured from the
/// real PES stamps.
fn h265_sps() -> Vec<u8> {
    let mut b = Bits::default();
    b.bits(0, 4); // sps_video_parameter_set_id
    b.bits(0, 3); // sps_max_sub_layers_minus1
    b.bit(1); // sps_temporal_id_nesting_flag
              // profile_tier_level(1, 0): 96 fixed bits for a single sub-layer.
    b.bits(0, 2); // general_profile_space
    b.bit(0); // general_tier_flag
    b.bits(1, 5); // general_profile_idc (Main)
    b.bits(1 << 30, 32); // general_profile_compatibility_flag[32]
    b.bits(1, 16); // general_progressive_source_flag + 15 constraint bits
    b.bits(0, 32); // remaining constraint / reserved bits
    b.bits(120, 8); // general_level_idc (4.0)
    b.ue(0); // sps_seq_parameter_set_id
    b.ue(1); // chroma_format_idc (4:2:0)
    b.ue(1280); // pic_width_in_luma_samples
    b.ue(720); // pic_height_in_luma_samples
    b.bit(0); // conformance_window_flag
    b.ue(0); // bit_depth_luma_minus8
    b.ue(0); // bit_depth_chroma_minus8
    b.ue(0); // log2_max_pic_order_cnt_lsb_minus4
    b.bit(1); // sps_sub_layer_ordering_info_present_flag
    b.ue(3); // sps_max_dec_pic_buffering_minus1
    b.ue(2); // sps_max_num_reorder_pics
    b.ue(0); // sps_max_latency_increase_plus1
    b.ue(0); // log2_min_luma_coding_block_size_minus3
    b.ue(2); // log2_diff_max_min_luma_coding_block_size
    b.ue(0); // log2_min_luma_transform_block_size_minus2
    b.ue(3); // log2_diff_max_min_luma_transform_block_size
    b.ue(0); // max_transform_hierarchy_depth_inter
    b.ue(0); // max_transform_hierarchy_depth_intra
    b.bit(0); // scaling_list_enabled_flag
    b.bit(0); // amp_enabled_flag
    b.bit(0); // sample_adaptive_offset_enabled_flag
    b.bit(0); // pcm_enabled_flag
    b.ue(0); // num_short_term_ref_pic_sets
    b.bit(0); // long_term_ref_pics_present_flag
    b.bit(0); // sps_temporal_mvp_enabled_flag
    b.bit(0); // strong_intra_smoothing_enabled_flag
    b.bit(0); // vui_parameters_present_flag
    b.bit(0); // sps_extension_present_flag
    let mut nal = Vec::from([0x42u8, 0x01]);
    nal.extend(b.finish());
    annexb_nal(&nal)
}

/// An H.265 PPS NAL, read for the slice-header fields that precede the order
/// count.
fn h265_pps() -> Vec<u8> {
    let mut b = Bits::default();
    b.ue(0); // pps_pic_parameter_set_id
    b.ue(0); // pps_seq_parameter_set_id
    b.bit(0); // dependent_slice_segments_enabled_flag
    b.bit(0); // output_flag_present_flag
    b.bits(0, 3); // num_extra_slice_header_bits
    b.bit(0); // sign_data_hiding_enabled_flag
    b.bit(0); // cabac_init_present_flag
    b.ue(0); // num_ref_idx_l0_default_active_minus1
    b.ue(0); // num_ref_idx_l1_default_active_minus1
    b.se(0); // init_qp_minus26
    b.bit(0); // constrained_intra_pred_flag
    b.bit(0); // transform_skip_enabled_flag
    b.bit(0); // cu_qp_delta_enabled_flag
    let mut nal = Vec::from([0x44u8, 0x01]);
    nal.extend(b.finish());
    annexb_nal(&nal)
}

/// One H.265 coded picture: IDR_W_RADL, or a TRAIL_R carrying its order-count
/// lsb (H.265 counts pictures, so the count is the display index).
fn h265_slice(is_idr: bool, slice_pic_order_cnt_lsb: u32) -> Vec<u8> {
    let mut b = Bits::default();
    b.bit(1); // first_slice_segment_in_pic_flag
    if is_idr {
        b.bit(0); // no_output_of_prior_pics_flag
    }
    b.ue(0); // slice_pic_parameter_set_id
    b.ue(if is_idr { 2 } else { 0 }); // slice_type (I / B)
    if !is_idr {
        b.bits(slice_pic_order_cnt_lsb, 4);
        b.bit(1); // short_term_ref_pic_set_sps_flag (unread here)
    }
    let header: &[u8] = if is_idr { &[0x26, 0x01] } else { &[0x02, 0x01] };
    let mut nal = Vec::from(header);
    nal.extend(b.finish());
    annexb_nal(&nal)
}

/// Seven pictures in coded order `I P B B P B B`, display order `0 1 2 3 4 5 6`.
const H265_CODED_COUNTS: [u32; 7] = [0, 3, 1, 2, 6, 4, 5];

fn h265_stream() -> Vec<Vec<u8>> {
    H265_CODED_COUNTS
        .iter()
        .enumerate()
        .map(|(index, &count)| {
            let mut au = Vec::new();
            if index == 0 {
                au.extend(h265_sps());
                au.extend(h265_pps());
            }
            au.extend(h265_slice(index == 0, count));
            au
        })
        .collect()
}

/// A reordering H.265 stream with no VUI timing: the two real stamps measure how
/// long one order count spans, and the units after them land in display order.
#[test]
fn h265_reordered_units_land_in_display_order() {
    let base = 90_000u64;
    let anchor = base + 6 * PERIOD;
    let stamps = [Some(base), None, None, None, Some(anchor), None, None];
    let pts = demux_pts(STREAM_TYPE_H265, &h265_stream(), &stamps);
    assert_eq!(
        pts,
        vec![
            Some(base),
            None,
            None,
            None,
            Some(anchor),
            Some(base + 4 * PERIOD),
            Some(base + 5 * PERIOD),
        ],
    );
}
