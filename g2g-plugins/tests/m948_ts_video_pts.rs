//! M948: per-frame video PTS synthesis in the TS demuxer. A transport stream
//! need only carry a PES timestamp every 700 ms, so the H.264 / H.265 access
//! units in between arrive unstamped and used to land at PTS 0. They now get
//! `last + frame_period`, but only for a stream whose SPS proves it presents
//! pictures in coded order (H.264 POC type 2, H.265 `sps_max_num_reorder_pics`
//! 0); a stream that may reorder is left alone rather than stamped wrong.

use g2g_plugins::mpegts::{TsDemuxer, TsMuxer, STREAM_TYPE_H264, STREAM_TYPE_H265, TS_PACKET_LEN};

/// 25 fps frame period in 90 kHz units.
const PERIOD: u64 = 3600;
/// VUI `timing_info` for 25 fps: fps = time_scale / (2 * num_units_in_tick).
const NUM_UNITS_IN_TICK: u32 = 1;
const TIME_SCALE: u32 = 50;

/// Minimal MSB-first bit writer for the hand-built parameter sets.
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

/// An H.264 SPS NAL: baseline profile, 1280x720, the given `pic_order_cnt_type`,
/// and VUI `timing_info` only when `with_timing`.
fn h264_sps(pic_order_cnt_type: u32, with_timing: bool) -> Vec<u8> {
    let mut b = Bits::default();
    b.ue(0); // seq_parameter_set_id
    b.ue(0); // log2_max_frame_num_minus4
    b.ue(pic_order_cnt_type);
    if pic_order_cnt_type == 0 {
        b.ue(0); // log2_max_pic_order_cnt_lsb_minus4
    }
    b.ue(1); // max_num_ref_frames
    b.bit(0); // gaps_in_frame_num_value_allowed_flag
    b.ue(79); // pic_width_in_mbs_minus1 -> 1280
    b.ue(44); // pic_height_in_map_units_minus1 -> 720
    b.bit(1); // frame_mbs_only_flag
    b.bit(1); // direct_8x8_inference_flag
    b.bit(0); // frame_cropping_flag
    b.bit(u32::from(with_timing)); // vui_parameters_present_flag
    if with_timing {
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
    }
    // nal_ref_idc 3 + type 7, then profile_idc / constraint flags / level_idc.
    let mut nal = Vec::from([0x67u8, 66, 0x00, 30]);
    nal.extend(b.finish());
    annexb_nal(&nal)
}

/// An H.265 SPS NAL: 1280x720, the given `sps_max_num_reorder_pics`, no VUI (so
/// the demuxer has to measure the frame period from the real PES stamps).
fn h265_sps(max_num_reorder_pics: u32) -> Vec<u8> {
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
    b.ue(1); // sps_max_dec_pic_buffering_minus1
    b.ue(max_num_reorder_pics);
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
              // SPS_NUT (33), nuh_layer_id 0, nuh_temporal_id_plus1 1.
    let mut nal = Vec::from([0x42u8, 0x01]);
    nal.extend(b.finish());
    annexb_nal(&nal)
}

/// A slice NAL standing in for coded picture data.
fn slice_nal(stream_type: u8, keyframe: bool) -> Vec<u8> {
    let header: &[u8] = match (stream_type, keyframe) {
        (STREAM_TYPE_H265, true) => &[0x26, 0x01],  // IDR_W_RADL
        (STREAM_TYPE_H265, false) => &[0x02, 0x01], // TRAIL_R
        (_, true) => &[0x65],                       // IDR
        (_, false) => &[0x41],                      // non-IDR
    };
    let mut nal = Vec::from(header);
    nal.extend_from_slice(&[0x88u8; 24]);
    annexb_nal(&nal)
}

/// One access unit: the parameter set (keyframes only) plus a slice.
fn access_unit(stream_type: u8, sps: &[u8], keyframe: bool) -> Vec<u8> {
    let mut au = Vec::new();
    if keyframe {
        au.extend_from_slice(sps);
    }
    au.extend(slice_nal(stream_type, keyframe));
    au
}

/// Mux `stamps` access units (the first a keyframe carrying the SPS) into a
/// transport stream, then demux it and report each unit's PTS in order.
fn demux_pts(stream_type: u8, sps: &[u8], stamps: &[Option<u64>]) -> Vec<Option<u64>> {
    let mut mux = TsMuxer::new(stream_type);
    let mut ts = Vec::new();
    for (index, pts) in stamps.iter().enumerate() {
        let au = access_unit(stream_type, sps, index == 0);
        ts.extend(mux.push_au(&au, *pts, None));
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
        .map(|u| u.pts_90khz)
        .collect()
}

/// A 25 fps H.264 stream stamped only on its first PES: every later access unit
/// gets its own display slot from the VUI frame period.
#[test]
fn unstamped_h264_units_get_the_vui_frame_period() {
    let base = 90_000u64;
    let mut stamps = vec![None; 8];
    stamps[0] = Some(base);
    let pts = demux_pts(STREAM_TYPE_H264, &h264_sps(2, true), &stamps);
    assert_eq!(
        pts,
        (0..8).map(|i| Some(base + i * PERIOD)).collect::<Vec<_>>(),
        "each unstamped unit advances one 25 fps frame period"
    );
}

/// A later real stamp re-anchors the run, so synthesis follows the real
/// timeline (mux drift) rather than the arithmetic one.
#[test]
fn a_real_stamp_reanchors_the_run() {
    let base = 90_000u64;
    let drift = 90u64;
    let anchor = base + 3 * PERIOD + drift;
    let stamps = [Some(base), None, None, Some(anchor), None, None];
    let pts = demux_pts(STREAM_TYPE_H264, &h264_sps(2, true), &stamps);
    assert_eq!(
        pts,
        vec![
            Some(base),
            Some(base + PERIOD),
            Some(base + 2 * PERIOD),
            Some(anchor),
            Some(anchor + PERIOD),
            Some(anchor + 2 * PERIOD),
        ],
    );
}

/// A stream that stamps every PES keeps its own timestamps, off-grid ones
/// included: synthesis must never "correct" a real stamp.
#[test]
fn a_fully_stamped_stream_is_unchanged() {
    let base = 45_000u64;
    let stamps: Vec<Option<u64>> = [0u64, 11_111, 22_222, 33_333]
        .iter()
        .map(|off| Some(base + off))
        .collect();
    let pts = demux_pts(STREAM_TYPE_H264, &h264_sps(2, true), &stamps);
    assert_eq!(pts, stamps);
}

/// An H.264 stream whose SPS leaves reordering open (POC type 0, what any
/// B-frame stream codes) is left unstamped: coded order is not display order
/// there, so interpolating would put the pictures on the wrong slots.
#[test]
fn a_stream_that_may_reorder_is_left_unstamped() {
    let base = 90_000u64;
    let stamps = [Some(base), None, None, None];
    let pts = demux_pts(STREAM_TYPE_H264, &h264_sps(0, true), &stamps);
    assert_eq!(pts, vec![Some(base), None, None, None]);
}

/// H.265 with `sps_max_num_reorder_pics` 0 and no VUI timing: the frame period
/// comes from the span between the two real PES stamps and the units between
/// them. The units before the second stamp have no period yet, so they stay
/// unstamped rather than guess.
#[test]
fn h265_measures_the_period_between_real_stamps() {
    let base = 90_000u64;
    let second = base + 3 * PERIOD;
    let stamps = [Some(base), None, None, Some(second), None, None];
    let pts = demux_pts(STREAM_TYPE_H265, &h265_sps(0), &stamps);
    assert_eq!(
        pts,
        vec![
            Some(base),
            None,
            None,
            Some(second),
            Some(second + PERIOD),
            Some(second + 2 * PERIOD),
        ],
    );
}

/// The same H.265 stream declaring a reorder depth gets no synthesis.
#[test]
fn h265_with_a_reorder_depth_is_left_unstamped() {
    let base = 90_000u64;
    let second = base + 3 * PERIOD;
    let stamps = [Some(base), None, None, Some(second), None, None];
    let pts = demux_pts(STREAM_TYPE_H265, &h265_sps(2), &stamps);
    assert_eq!(pts, vec![Some(base), None, None, Some(second), None, None],);
}
