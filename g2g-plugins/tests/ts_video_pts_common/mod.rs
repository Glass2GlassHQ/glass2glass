//! Hand-built H.264 / H.265 streams shared by the TS video-timestamp tests
//! (`m952_ts_video_pts`, `m1156_ts_single_stamp_reorder`): parameter sets and
//! slice headers written bit by bit, and the mux / demux round trip that reports
//! what the demuxer stamped each access unit with. Nothing here reuses the
//! demuxer's own parses, so a broken parse cannot pass by agreeing with itself.
//! One definition, included per test binary via `mod ts_video_pts_common;`.
#![allow(dead_code)] // no one test file uses every builder here

use g2g_plugins::mpegts::{TsDemuxer, TsMuxer, TS_PACKET_LEN};

/// 25 fps frame period in 90 kHz units.
pub(crate) const PERIOD: u64 = 3600;
/// VUI `timing_info` for 25 fps. An H.264 tick is a field, so its time scale is
/// twice the frame rate, and an H.265 tick is a picture.
pub(crate) const NUM_UNITS_IN_TICK: u32 = 1;
pub(crate) const H264_TIME_SCALE: u32 = 50;
pub(crate) const H265_TIME_SCALE: u32 = 25;

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

/// Mux `units` with the given per-unit stamps, demux, and report the
/// `(pts, dts)` of each unit in coded order.
pub(crate) fn demux_timestamps(
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
pub(crate) fn demux_pts(
    stream_type: u8,
    units: &[Vec<u8>],
    stamps: &[Option<u64>],
) -> Vec<Option<u64>> {
    let with_dts: Vec<(Option<u64>, Option<u64>)> = stamps.iter().map(|p| (*p, None)).collect();
    demux_timestamps(stream_type, units, &with_dts)
        .into_iter()
        .map(|(pts, _)| pts)
        .collect()
}

/// An H.264 SPS NAL: baseline geometry, POC type 0 (what any B-frame stream
/// codes), frames only, and VUI `timing_info` at 25 fps when `with_timing`.
pub(crate) fn h264_sps(with_timing: bool) -> Vec<u8> {
    h264_sps_bits(with_timing, true)
}

/// The same with `frame_mbs_only_flag` under the caller's control, so a stream
/// whose access units are single fields can be built.
fn h264_sps_bits(with_timing: bool, frame_mbs_only: bool) -> Vec<u8> {
    let mut b = Bits::default();
    b.ue(0); // seq_parameter_set_id
    b.ue(0); // log2_max_frame_num_minus4
    b.ue(0); // pic_order_cnt_type
    b.ue(0); // log2_max_pic_order_cnt_lsb_minus4 -> 16 counts before a wrap
    b.ue(2); // max_num_ref_frames
    b.bit(0); // gaps_in_frame_num_value_allowed_flag
    b.ue(79); // pic_width_in_mbs_minus1 -> 1280
    b.ue(44); // pic_height_in_map_units_minus1 -> 720
    b.bit(u32::from(frame_mbs_only)); // frame_mbs_only_flag
    if !frame_mbs_only {
        b.bit(0); // mb_adaptive_frame_field_flag
    }
    b.bit(1); // direct_8x8_inference_flag
    b.bit(0); // frame_cropping_flag
    b.bit(1); // vui_parameters_present_flag
    b.bit(0); // aspect_ratio_info_present_flag
    b.bit(0); // overscan_info_present_flag
    b.bit(0); // video_signal_type_present_flag
    b.bit(0); // chroma_loc_info_present_flag
    b.bit(u32::from(with_timing)); // timing_info_present_flag
    if with_timing {
        b.bits(NUM_UNITS_IN_TICK, 32);
        b.bits(H264_TIME_SCALE, 32);
        b.bit(1); // fixed_frame_rate_flag
    }
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
pub(crate) fn h264_pps() -> Vec<u8> {
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
pub(crate) fn h264_slice(kind: PictureKind, frame_num: u32, pic_order_cnt_lsb: u32) -> Vec<u8> {
    h264_slice_bits(kind, frame_num, pic_order_cnt_lsb, None)
}

/// The same, `bottom_field` naming which field this access unit codes. Only a
/// stream whose SPS allows fields carries those flags, so `None` is the
/// frames-only case.
fn h264_slice_bits(
    kind: PictureKind,
    frame_num: u32,
    pic_order_cnt_lsb: u32,
    bottom_field: Option<bool>,
) -> Vec<u8> {
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
    if let Some(bottom) = bottom_field {
        b.bit(1); // field_pic_flag
        b.bit(u32::from(bottom));
    }
    if kind == PictureKind::Idr {
        b.ue(0); // idr_pic_id
    }
    b.bits(pic_order_cnt_lsb, 4);
    let mut nal = Vec::from([header]);
    nal.extend(b.finish());
    annexb_nal(&nal)
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PictureKind {
    Idr,
    Reference,
    NonReference,
}

/// Seven pictures in coded order `I P B B P B B`, whose display order is
/// `0 1 2 3 4 5 6`. H.264 counts fields, so these order counts step 2 per frame.
pub(crate) const H264_FIELD_COUNTED_PICTURES: [(PictureKind, u32, u32); 7] = [
    (PictureKind::Idr, 0, 0),           // display 0
    (PictureKind::Reference, 1, 6),     // display 3
    (PictureKind::NonReference, 2, 2),  // display 1
    (PictureKind::NonReference, 2, 4),  // display 2
    (PictureKind::Reference, 2, 12),    // display 6
    (PictureKind::NonReference, 3, 8),  // display 4
    (PictureKind::NonReference, 3, 10), // display 5
];

/// One access unit per `(kind, frame_num, pic_order_cnt_lsb)`, the first
/// carrying the parameter sets.
pub(crate) fn h264_stream(pictures: &[(PictureKind, u32, u32)], with_timing: bool) -> Vec<Vec<u8>> {
    pictures
        .iter()
        .enumerate()
        .map(|(index, &(kind, frame_num, poc_lsb))| {
            let mut au = Vec::new();
            if index == 0 {
                au.extend(h264_sps(with_timing));
                au.extend(h264_pps());
            }
            au.extend(h264_slice(kind, frame_num, poc_lsb));
            au
        })
        .collect()
}

/// A field-coded H.264 stream: one access unit per field, carrying the given
/// order-count lsbs in coded order, the odd ones as bottom fields.
pub(crate) fn h264_field_coded_stream(field_counts: &[u32], with_timing: bool) -> Vec<Vec<u8>> {
    field_counts
        .iter()
        .enumerate()
        .map(|(index, &count)| {
            let mut au = Vec::new();
            if index == 0 {
                au.extend(h264_sps_bits(with_timing, false));
                au.extend(h264_pps());
            }
            let kind = if index == 0 {
                PictureKind::Idr
            } else {
                PictureKind::Reference
            };
            let frame_num = (index as u32 / 2) & 0xF;
            au.extend(h264_slice_bits(
                kind,
                frame_num,
                count,
                Some(!count.is_multiple_of(2)),
            ));
            au
        })
        .collect()
}

/// An H.265 SPS NAL: 1280x720, `sps_max_num_reorder_pics` 2 (so the stream may
/// reorder), and VUI `timing_info` at 25 fps only when `with_timing`. Without
/// it the display-slot spacing has to be measured from the real PES stamps.
pub(crate) fn h265_sps(with_timing: bool) -> Vec<u8> {
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
    b.bit(u32::from(with_timing)); // vui_parameters_present_flag
    if with_timing {
        b.bit(0); // aspect_ratio_info_present_flag
        b.bit(0); // overscan_info_present_flag
        b.bit(0); // video_signal_type_present_flag
        b.bit(0); // chroma_loc_info_present_flag
        b.bit(0); // neutral_chroma_indication_flag
        b.bit(0); // field_seq_flag
        b.bit(0); // frame_field_info_present_flag
        b.bit(0); // default_display_window_flag
        b.bit(1); // vui_timing_info_present_flag
        b.bits(NUM_UNITS_IN_TICK, 32);
        b.bits(H265_TIME_SCALE, 32);
        b.bit(0); // vui_poc_proportional_to_timing_flag
        b.bit(0); // vui_hrd_parameters_present_flag
        b.bit(0); // bitstream_restriction_flag
    }
    b.bit(0); // sps_extension_present_flag
    let mut nal = Vec::from([0x42u8, 0x01]);
    nal.extend(b.finish());
    annexb_nal(&nal)
}

/// An H.265 PPS NAL, read for the slice-header fields that precede the order
/// count.
pub(crate) fn h265_pps() -> Vec<u8> {
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
pub(crate) fn h265_slice(is_idr: bool, slice_pic_order_cnt_lsb: u32) -> Vec<u8> {
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

/// The same coded order in H.265, which counts pictures: each count is the
/// picture's display index.
pub(crate) const H265_CODED_COUNTS: [u32; 7] = [0, 3, 1, 2, 6, 4, 5];

/// One access unit per order count, the first an IDR carrying the parameter sets.
pub(crate) fn h265_stream(counts: &[u32], with_timing: bool) -> Vec<Vec<u8>> {
    counts
        .iter()
        .enumerate()
        .map(|(index, &count)| {
            let mut au = Vec::new();
            if index == 0 {
                au.extend(h265_sps(with_timing));
                au.extend(h265_pps());
            }
            au.extend(h265_slice(index == 0, count));
            au
        })
        .collect()
}
