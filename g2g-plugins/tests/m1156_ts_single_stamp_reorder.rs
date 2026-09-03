//! M1156: a reordering H.264 / H.265 stream in a transport stream gets its
//! display timeline from the first PES stamp alone, as long as the SPS declares
//! the frame period. The count step per frame follows from the codec (H.265
//! counts pictures, H.264 counts fields), so the slope is known before a second
//! stamp measures it, and an odd H.264 count in a frame-coded stream proves the
//! encoder steps once per frame after all.

use g2g_plugins::mpegts::{STREAM_TYPE_H264, STREAM_TYPE_H265};

mod ts_video_pts_common;
use ts_video_pts_common::{
    demux_pts, h264_field_coded_stream, h264_stream, h265_stream, PictureKind,
    H264_FIELD_COUNTED_PICTURES, H265_CODED_COUNTS, PERIOD,
};

/// The order count of the first access unit of every stream here, an IDR.
const ANCHOR_POC: u64 = 0;
/// Counts one coded frame advances: an H.264 stream that counts fields steps
/// both of them, and one that counts pictures steps once, as H.265 always does.
const H264_FIELD_POC_STEP: u64 = 2;
const PICTURE_POC_STEP: u64 = 1;

/// Where a count sits on the display timeline: its distance from the anchor's
/// count, one frame period per `poc_step_per_frame` counts.
fn slot(anchor_pts: u64, poc: u64, poc_step_per_frame: u64) -> Option<u64> {
    Some(anchor_pts + (poc - ANCHOR_POC) * PERIOD / poc_step_per_frame)
}

/// A field-counted H.264 stream stamped only on its first access unit: the
/// declared 25 fps and the field step put every later unit on its display slot,
/// with no second stamp to measure against.
#[test]
fn one_stamp_times_a_field_counted_h264_stream() {
    let base = 90_000u64;
    let mut stamps = vec![None; H264_FIELD_COUNTED_PICTURES.len()];
    stamps[0] = Some(base);
    let pts = demux_pts(
        STREAM_TYPE_H264,
        &h264_stream(&H264_FIELD_COUNTED_PICTURES, true),
        &stamps,
    );
    // No count here wraps, so each picture's order count is the lsb it codes.
    let expected: Vec<Option<u64>> = H264_FIELD_COUNTED_PICTURES
        .iter()
        .map(|&(_, _, poc)| slot(base, u64::from(poc), H264_FIELD_POC_STEP))
        .collect();
    assert_eq!(
        pts, expected,
        "every unit is stamped, on its own display slot, from the one real stamp"
    );
}

/// Coded order `I P B P B` for an encoder that steps the order count once per
/// frame rather than once per field: display order `0 2 1 4 3`.
const PICTURES_COUNTING_BY_ONE: [(PictureKind, u32, u32); 5] = [
    (PictureKind::Idr, 0, 0),          // display 0
    (PictureKind::Reference, 1, 2),    // display 2
    (PictureKind::NonReference, 2, 1), // display 1
    (PictureKind::Reference, 2, 4),    // display 4
    (PictureKind::NonReference, 3, 3), // display 3
];

/// The first odd count of a frame-coded stream proves the step is one, and the
/// units after it are spaced accordingly. The units already emitted keep the
/// value the field step gave them: the demuxer does not retime what it has
/// pushed downstream.
#[test]
fn an_odd_h264_count_switches_the_step_to_one() {
    let base = 90_000u64;
    let mut stamps = vec![None; PICTURES_COUNTING_BY_ONE.len()];
    stamps[0] = Some(base);
    let pts = demux_pts(
        STREAM_TYPE_H264,
        &h264_stream(&PICTURES_COUNTING_BY_ONE, true),
        &stamps,
    );
    assert_eq!(pts[0], Some(base), "the real stamp is kept");
    assert_eq!(
        pts[1],
        slot(base, 2, H264_FIELD_POC_STEP),
        "the count 2 arrived before any odd one, so it took the field step"
    );
    let after_the_odd_count: Vec<Option<u64>> = PICTURES_COUNTING_BY_ONE[2..]
        .iter()
        .map(|&(_, _, poc)| slot(base, u64::from(poc), PICTURE_POC_STEP))
        .collect();
    assert_eq!(
        pts[2..],
        after_the_odd_count[..],
        "from the odd count on, one count is one frame period"
    );
}

/// The same for H.265, which counts pictures: one stamp and the declared 25 fps
/// put each unit one period per count from the anchor.
#[test]
fn one_stamp_times_an_h265_stream() {
    let base = 90_000u64;
    let mut stamps = vec![None; H265_CODED_COUNTS.len()];
    stamps[0] = Some(base);
    let pts = demux_pts(
        STREAM_TYPE_H265,
        &h265_stream(&H265_CODED_COUNTS, true),
        &stamps,
    );
    let expected: Vec<Option<u64>> = H265_CODED_COUNTS
        .iter()
        .map(|&poc| slot(base, u64::from(poc), PICTURE_POC_STEP))
        .collect();
    assert_eq!(pts, expected);
}

/// A field-coded stream codes one access unit per field, so its bottom fields
/// carry odd counts and the parity rule must not fire: each field lands half a
/// frame period after the one before it.
#[test]
fn a_field_coded_h264_stream_keeps_the_field_step() {
    let base = 90_000u64;
    let field_counts: [u32; 6] = [0, 1, 2, 3, 4, 5];
    let mut stamps = vec![None; field_counts.len()];
    stamps[0] = Some(base);
    let pts = demux_pts(
        STREAM_TYPE_H264,
        &h264_field_coded_stream(&field_counts, true),
        &stamps,
    );
    let expected: Vec<Option<u64>> = field_counts
        .iter()
        .map(|&poc| slot(base, u64::from(poc), H264_FIELD_POC_STEP))
        .collect();
    assert_eq!(pts, expected);
}

/// With no VUI timing there is no period to stand in, so the units before the
/// second real stamp stay unstamped, and the pair measures the slope as before.
#[test]
fn without_declared_timing_the_second_stamp_is_still_needed() {
    let base = 90_000u64;
    let second = base + 6 * PERIOD;
    let stamps = [Some(base), None, None, None, Some(second), None, None];
    let pts = demux_pts(
        STREAM_TYPE_H264,
        &h264_stream(&H264_FIELD_COUNTED_PICTURES, false),
        &stamps,
    );
    assert_eq!(pts[..4], [Some(base), None, None, None]);
    assert_eq!(
        pts[4..],
        [
            Some(second),
            slot(base, 8, H264_FIELD_POC_STEP),
            slot(base, 10, H264_FIELD_POC_STEP),
        ],
    );
}

/// Twice the declared frame period: what the second stamp says the stream really
/// runs at, against a VUI that declares 25 fps.
const MEASURED_PERIOD: u64 = 2 * PERIOD;

/// The second real stamp measures the slope and replaces the one the declared
/// period stood in with, so the units after it follow the mux's own spacing.
#[test]
fn a_second_stamp_replaces_the_declared_slope() {
    let base = 90_000u64;
    let second = base + 6 * MEASURED_PERIOD;
    let stamps = [Some(base), None, None, None, Some(second), None, None];
    let pts = demux_pts(
        STREAM_TYPE_H264,
        &h264_stream(&H264_FIELD_COUNTED_PICTURES, true),
        &stamps,
    );
    assert_eq!(
        pts[1],
        slot(base, 6, H264_FIELD_POC_STEP),
        "this unit precedes the second stamp, so it is on the declared grid"
    );
    assert_eq!(
        pts[5..],
        [
            Some(base + 4 * MEASURED_PERIOD),
            Some(base + 5 * MEASURED_PERIOD),
        ],
        "displays 4 and 5 at the measured spacing, not the declared one"
    );
}

/// Coded order `I P B B B`, field-counted, whose last access unit carries a
/// stray odd count. The two stamps have already measured the slope, so the
/// parity rule leaves it alone and that unit lands mid-frame.
const PICTURES_WITH_A_STRAY_ODD_COUNT: [(PictureKind, u32, u32); 5] = [
    (PictureKind::Idr, 0, 0),
    (PictureKind::Reference, 1, 6),
    (PictureKind::NonReference, 2, 2),
    (PictureKind::NonReference, 2, 4),
    (PictureKind::NonReference, 2, 3),
];

#[test]
fn a_measured_slope_survives_an_odd_count() {
    let base = 90_000u64;
    let second = base + 3 * PERIOD;
    let stamps = [Some(base), Some(second), None, None, None];
    let pts = demux_pts(
        STREAM_TYPE_H264,
        &h264_stream(&PICTURES_WITH_A_STRAY_ODD_COUNT, true),
        &stamps,
    );
    let expected: Vec<Option<u64>> = PICTURES_WITH_A_STRAY_ODD_COUNT
        .iter()
        .enumerate()
        .map(|(index, &(_, _, poc))| match index {
            0 => Some(base),
            1 => Some(second),
            _ => slot(base, u64::from(poc), H264_FIELD_POC_STEP),
        })
        .collect();
    assert_eq!(pts, expected);
}
