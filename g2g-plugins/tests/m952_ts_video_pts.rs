//! M952: per-frame video timestamp synthesis in the TS demuxer for the streams
//! M948 left alone. MPEG-2 access units take their display slot from the picture
//! header's `temporal_reference`, and an H.264 / H.265 stream that reorders
//! takes it from the picture order count, anchored on the real PES stamps.

use g2g_plugins::mpegts::{STREAM_TYPE_H264, STREAM_TYPE_H265, STREAM_TYPE_MPEG2_VIDEO};

mod ts_video_pts_common;
use ts_video_pts_common::{
    demux_pts, demux_timestamps, h264_slice, h264_sps, h264_stream, h265_stream,
    H264_FIELD_COUNTED_PICTURES, H265_CODED_COUNTS, PERIOD,
};

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

/// A reordering H.264 stream stamped twice: the pair fixes how far apart the
/// display slots are, and every unstamped unit after them lands on its own,
/// which for a B frame is behind the anchor it was coded after. The units before
/// the second stamp are on the slope the VUI frame period declares (M1156).
#[test]
fn h264_reordered_units_land_in_display_order() {
    let base = 90_000u64;
    let anchor = base + 6 * PERIOD;
    let stamps = [Some(base), None, None, None, Some(anchor), None, None];
    let pts = demux_pts(
        STREAM_TYPE_H264,
        &h264_stream(&H264_FIELD_COUNTED_PICTURES, true),
        &stamps,
    );
    assert_eq!(
        pts,
        vec![
            Some(base),
            Some(base + 3 * PERIOD),
            Some(base + PERIOD),
            Some(base + 2 * PERIOD),
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
    let pts = demux_pts(
        STREAM_TYPE_H264,
        &h264_stream(&H264_FIELD_COUNTED_PICTURES, true),
        &stamps,
    );
    assert_eq!(pts, stamps);
}

/// The synthesis needs the picture parameter set to read a slice header, so a
/// stream that never sends one is left alone rather than stamped on a guess.
#[test]
fn h264_without_a_picture_parameter_set_is_left_unstamped() {
    let base = 90_000u64;
    let units: Vec<Vec<u8>> = H264_FIELD_COUNTED_PICTURES
        .iter()
        .enumerate()
        .map(|(index, &(kind, frame_num, poc_lsb))| {
            let mut au = Vec::new();
            if index == 0 {
                au.extend(h264_sps(true));
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

/// A reordering H.265 stream with no VUI timing: the two real stamps measure how
/// long one order count spans, and the units after them land in display order.
#[test]
fn h265_reordered_units_land_in_display_order() {
    let base = 90_000u64;
    let anchor = base + 6 * PERIOD;
    let stamps = [Some(base), None, None, None, Some(anchor), None, None];
    let pts = demux_pts(
        STREAM_TYPE_H265,
        &h265_stream(&H265_CODED_COUNTS, false),
        &stamps,
    );
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
