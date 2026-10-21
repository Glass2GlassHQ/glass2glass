use alloc::vec::Vec;

use g2g_core::log::Target;
use g2g_core::{g2g_warn, Caps, Dim, Interlace, Rate, RawVideoFormat};

use crate::pixel::{frame_byte_size, row_bytes, tight_plane};
use crate::typefind::PLACEHOLDER_FRAMERATE;

// AVI biCompression, Matroska ColourSpace and QuickTime sample entry codes
const RAW_VIDEO_FOURCCS: &[([u8; 4], RawVideoFormat)] = &[
    (*b"I420", RawVideoFormat::I420),
    (*b"IYUV", RawVideoFormat::I420),
    (*b"YUY2", RawVideoFormat::Yuyv),
    (*b"YUYV", RawVideoFormat::Yuyv),
    (*b"YUNV", RawVideoFormat::Yuyv),
    (*b"YUVS", RawVideoFormat::Yuyv),
    (*b"UYVY", RawVideoFormat::Uyvy),
    (*b"2VUY", RawVideoFormat::Uyvy),
    (*b"HDYC", RawVideoFormat::Uyvy),
    (*b"UYNV", RawVideoFormat::Uyvy),
    (*b"v210", RawVideoFormat::V210),
];

const RAW_VIDEO_STREAM_NAMES: &[(&str, RawVideoFormat)] = &[
    ("raw-i420", RawVideoFormat::I420),
    ("raw-yuy2", RawVideoFormat::Yuyv),
    ("raw-uyvy", RawVideoFormat::Uyvy),
    ("raw-v210", RawVideoFormat::V210),
    ("raw-gray8", RawVideoFormat::Gray8),
    ("raw-bgr", RawVideoFormat::Bgr8),
    ("raw-bgrx", RawVideoFormat::Bgrx8),
    ("raw-rgb", RawVideoFormat::Rgb8),
    ("raw-argb", RawVideoFormat::Argb8),
];

const MAX_DIM: u32 = 65_535;

pub(crate) fn raw_video_format(fourcc: [u8; 4]) -> Option<RawVideoFormat> {
    RAW_VIDEO_FOURCCS
        .iter()
        .find(|(code, _)| code.eq_ignore_ascii_case(&fourcc))
        .map(|(_, format)| *format)
}

// how a container stores a frame the graph carries tight and top-down
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RawVideoStorage {
    Tight,
    // packed rows four bytes apart when the frame holds them, else tight, the way libavcodec reads an AVI
    AviRows { bottom_up: bool },
    // packed rows as far apart as the sample size spreads them, since QuickTime writers pad rows differently
    QuickTimeRows,
    // I420 with the V plane stored ahead of the U plane
    Yv12,
}

const AVI_ROW_ALIGN: usize = 4;

// the AVI layouts outside the shared fourcc table: BI_RGB at 24 and 32 bits, YV12 and the 8-bit gray fourccs
pub(crate) fn avi_raw_video(
    compression: [u8; 4],
    bit_count: u16,
    bottom_up: bool,
) -> Option<(RawVideoFormat, RawVideoStorage)> {
    let rows = RawVideoStorage::AviRows { bottom_up };
    match (&compression.map(|byte| byte.to_ascii_uppercase()), bit_count) {
        // biCompression BI_RGB
        (&[0, 0, 0, 0], 24) => Some((RawVideoFormat::Bgr8, rows)),
        (&[0, 0, 0, 0], 32) => Some((RawVideoFormat::Bgrx8, rows)),
        (b"YV12", _) => Some((RawVideoFormat::I420, RawVideoStorage::Yv12)),
        // only BI_RGB rows run bottom-up
        (b"Y800" | b"GREY", _) => Some((
            RawVideoFormat::Gray8,
            RawVideoStorage::AviRows { bottom_up: false },
        )),
        _ => None,
    }
}

// QuickTime `raw ` video, whose depth names its pixel layout
pub(crate) fn quicktime_raw_video(depth: u16) -> Option<(RawVideoFormat, RawVideoStorage)> {
    match depth {
        24 => Some((RawVideoFormat::Rgb8, RawVideoStorage::QuickTimeRows)),
        32 => Some((RawVideoFormat::Argb8, RawVideoStorage::QuickTimeRows)),
        _ => None,
    }
}

// the tight top-down frame a stored one holds, none when it is short of its geometry
pub(crate) fn copy_raw_video_frame(
    format: RawVideoFormat,
    storage: RawVideoStorage,
    width: u32,
    height: u32,
    stored: &[u8],
) -> Option<Vec<u8>> {
    let (columns, rows) = (width as usize, height as usize);
    match storage {
        RawVideoStorage::Tight => {
            let pixels = raw_video_frame_bytes(format, width, height, stored.len())?;
            Some(stored[..pixels].to_vec())
        }
        RawVideoStorage::Yv12 => {
            let pixels = raw_video_frame_bytes(format, width, height, stored.len())?;
            let (u_at, chroma_row, chroma_rows) = tight_plane(format, 1, columns, rows)?;
            let v_at = u_at + chroma_row * chroma_rows;
            let frame = &stored[..pixels];
            Some([&frame[..u_at], &frame[v_at..], &frame[u_at..v_at]].concat())
        }
        RawVideoStorage::AviRows { bottom_up } => {
            let row = row_bytes(format, columns);
            let aligned = row.checked_next_multiple_of(AVI_ROW_ALIGN)?;
            let stride = match aligned.checked_mul(rows)? <= stored.len() {
                true => aligned,
                false => row,
            };
            copy_rows(stored, row, stride, rows, bottom_up)
        }
        RawVideoStorage::QuickTimeRows => {
            let stride = stored.len().checked_div(rows)?;
            copy_rows(stored, row_bytes(format, columns), stride, rows, false)
        }
    }
}

// `rows` rows of `row` bytes, read `stride` bytes apart, the bottom one first when `bottom_up`
fn copy_rows(
    stored: &[u8],
    row: usize,
    stride: usize,
    rows: usize,
    bottom_up: bool,
) -> Option<Vec<u8>> {
    let last_row_end = stride.checked_mul(rows.checked_sub(1)?)?.checked_add(row)?;
    if row == 0 || stride < row || last_row_end > stored.len() {
        return None;
    }
    let mut frame = Vec::with_capacity(row * rows);
    for index in 0..rows {
        let stored_row = if bottom_up { rows - 1 - index } else { index };
        let start = stored_row * stride;
        frame.extend_from_slice(&stored[start..start + row]);
    }
    Some(frame)
}

pub(crate) fn raw_video_stream_str(format: RawVideoFormat) -> Option<&'static str> {
    RAW_VIDEO_STREAM_NAMES
        .iter()
        .find(|(_, f)| *f == format)
        .map(|(name, _)| *name)
}

pub(crate) fn raw_video_stream_from_str(name: &str) -> Option<RawVideoFormat> {
    RAW_VIDEO_STREAM_NAMES
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, format)| *format)
}

pub(crate) fn raw_video_formats() -> impl Iterator<Item = RawVideoFormat> {
    RAW_VIDEO_STREAM_NAMES.iter().map(|(_, format)| *format)
}

pub(crate) fn raw_video_caps(
    format: RawVideoFormat,
    width: Dim,
    height: Dim,
    framerate: Rate,
) -> Caps {
    Caps::RawVideo {
        format,
        width,
        height,
        framerate,
        interlace: Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

pub(crate) fn raw_video_placeholder_caps(format: RawVideoFormat) -> Caps {
    let span = || Dim::Range {
        min: 1,
        max: MAX_DIM,
    };
    raw_video_caps(format, span(), span(), PLACEHOLDER_FRAMERATE)
}

// bytes past the frame's rows are ignored, as libavcodec does
pub(crate) fn raw_video_frame_bytes(
    format: RawVideoFormat,
    width: u32,
    height: u32,
    stored: usize,
) -> Option<usize> {
    let pixels = frame_byte_size(format, width, height);
    // a track that declared no geometry has no frame size to check against
    (pixels != 0 && stored >= pixels).then_some(pixels)
}

pub(crate) fn warn_short_raw_video_frame(
    element: &'static str,
    format: RawVideoFormat,
    width: u32,
    height: u32,
    stored: usize,
) {
    g2g_warn!(
        Target::category(element),
        "dropped a {stored}-byte {} frame, short of its {width}x{height} pixels",
        format.gst_name()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    // a 1x3 BGR frame, whose three-byte rows an AVI pads to four
    const BGR_ROWS: [[u8; 3]; 3] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    const PADDING: u8 = 0xEE;

    fn bottom_up_padded_rows() -> Vec<u8> {
        BGR_ROWS
            .iter()
            .rev()
            .flat_map(|row| [row[0], row[1], row[2], PADDING])
            .collect()
    }

    #[test]
    fn a_bottom_up_bi_rgb_frame_comes_out_tight_and_top_down() {
        let (format, storage) = avi_raw_video([0; 4], 24, true).expect("BI_RGB 24");
        let frame = copy_raw_video_frame(format, storage, 1, 3, &bottom_up_padded_rows());
        assert_eq!(frame, Some(BGR_ROWS.concat()));
    }

    #[test]
    fn avi_rows_too_short_for_padding_are_read_tight() {
        let (format, storage) = avi_raw_video(*b"Y800", 8, false).expect("Y800");
        let tight: Vec<u8> = (1..=3).collect();
        assert_eq!(
            copy_raw_video_frame(format, storage, 1, 3, &tight),
            Some(tight.clone())
        );
        assert_eq!(copy_raw_video_frame(format, storage, 1, 3, &tight[..2]), None);
    }

    #[test]
    fn quicktime_rows_take_their_stride_from_the_sample_size() {
        let (format, storage) = quicktime_raw_video(24).expect("depth 24");
        let mut padded = bottom_up_padded_rows();
        padded.reverse();
        let frame = copy_raw_video_frame(format, storage, 1, 3, &padded).expect("whole rows");
        let rows: Vec<u8> = padded
            .chunks(4)
            .flat_map(|row| row[..3].to_vec())
            .collect();
        assert_eq!(frame, rows);
    }

    #[test]
    fn a_yv12_frame_comes_out_in_i420_plane_order() {
        let (format, storage) = avi_raw_video(*b"YV12", 12, false).expect("YV12");
        // 3x3: nine luma bytes, then the 2x2 V plane, then the 2x2 U plane
        let luma = [0u8; 9];
        let (v, u) = ([1u8; 4], [2u8; 4]);
        let stored = [&luma[..], &v[..], &u[..]].concat();
        let frame = copy_raw_video_frame(format, storage, 3, 3, &stored);
        assert_eq!(frame, Some([&luma[..], &u[..], &v[..]].concat()));
    }

    #[test]
    fn other_depths_of_the_raw_layouts_are_not_read() {
        assert_eq!(avi_raw_video([0; 4], 16, true), None);
        assert_eq!(quicktime_raw_video(16), None);
    }
}
