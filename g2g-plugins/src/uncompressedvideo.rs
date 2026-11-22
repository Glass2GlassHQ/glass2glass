// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::vec::Vec;

use g2g_core::log::Target;
use g2g_core::meta::MAX_PLANES;
use g2g_core::{g2g_warn, Caps, Dim, Interlace, Rate, RawVideoFormat};

use crate::annexb::BitReader;
use crate::pixel::{frame_byte_size, row_bytes, tight_plane};
use crate::typefind::{PLACEHOLDER_FRAMERATE, PLACEHOLDER_MIN_DIMENSION};

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

// the fourcc a writer names a format by, the first `RAW_VIDEO_FOURCCS` entry for it
pub(crate) fn raw_video_fourcc(format: RawVideoFormat) -> Option<[u8; 4]> {
    RAW_VIDEO_FOURCCS
        .iter()
        .find(|(_, candidate)| *candidate == format)
        .map(|(code, _)| *code)
}

// how a container stores a frame the graph carries tight and top-down
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RawVideoStorage {
    Tight,
    // packed rows four bytes apart when the frame holds them, else tight, the way libavcodec reads an AVI
    AviRows {
        bottom_up: bool,
    },
    // packed rows as far apart as the sample size spreads them, since QuickTime writers pad rows differently
    #[cfg(feature = "std")]
    QuickTimeRows,
    // I420 with the V plane stored ahead of the U plane
    Yv12,
    // planar rows padded to four bytes per plane when the frame is exactly that size, as GStreamer's avimux writes them, else tight
    AviPlaneRows,
    // ISO/IEC 23001-17 rows, each plane's rows padded to a multiple of `row_align` bytes, none when it is zero
    #[cfg(feature = "std")]
    AlignedRows {
        row_align: u32,
        // For reordered three-plane formats, the output plane at each stored position.
        plane_order: Option<[u8; 3]>,
        // Unsigned MSB-first samples, byte-aligned at the start of each row.
        bit_packed: bool,
        // One row from every plane in file order before the next row group.
        row_interleaved: bool,
    },
    #[cfg(feature = "std")]
    MixedChromaRows(MixedChromaLayout),
    #[cfg(feature = "std")]
    PackedYuv422Rows {
        row_align: u32,
        group_bytes: u32,
        component_order: [usize; 4],
    },
}

#[cfg(feature = "std")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MixedChromaLayout {
    pub row_align: u32,
    pub chroma_first: bool,
    pub vu: bool,
    pub packing: MixedSamplePacking,
}

#[cfg(feature = "std")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MixedSamplePacking {
    Bytes,
    PackedBits,
    Words { little_endian: bool },
}

const AVI_ROW_ALIGN: usize = 4;

// the AVI layouts the shared fourcc table does not cover: BI_RGB at 24 and 32 bits, I420, YV12 and the 8-bit gray fourccs
pub(crate) fn avi_raw_video(
    compression: [u8; 4],
    bit_count: u16,
    bottom_up: bool,
) -> Option<(RawVideoFormat, RawVideoStorage)> {
    let rows = RawVideoStorage::AviRows { bottom_up };
    match (
        &compression.map(|byte| byte.to_ascii_uppercase()),
        bit_count,
    ) {
        // biCompression BI_RGB
        (&[0, 0, 0, 0], 24) => Some((RawVideoFormat::Bgr8, rows)),
        (&[0, 0, 0, 0], 32) => Some((RawVideoFormat::Bgrx8, rows)),
        (b"I420" | b"IYUV", _) => Some((RawVideoFormat::I420, RawVideoStorage::AviPlaneRows)),
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
#[cfg(feature = "std")]
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
            copy_rows(stored, row, row, stride, rows, bottom_up)
        }
        #[cfg(feature = "std")]
        RawVideoStorage::QuickTimeRows => {
            let stride = stored.len().checked_div(rows)?;
            let row = row_bytes(format, columns);
            copy_rows(stored, row, row, stride, rows, false)
        }
        RawVideoStorage::AviPlaneRows => {
            let planes = aligned_planes(format, columns, rows, AVI_ROW_ALIGN, false)?;
            if stored.len() != stored_planes_len(&planes)? {
                return copy_raw_video_frame(format, RawVideoStorage::Tight, width, height, stored);
            }
            copy_planes(format, width, height, &planes, stored, None, false)
        }
        #[cfg(feature = "std")]
        RawVideoStorage::AlignedRows {
            row_align,
            plane_order,
            bit_packed,
            row_interleaved,
        } => {
            let align = usize::try_from(row_align).ok()?;
            let mut planes = aligned_planes(format, columns, rows, align, bit_packed)?;
            if bit_packed && align != 0 && matches!(format.chroma_shift(), Some((1, _))) {
                match_chroma_alignment(&mut planes, align, stored.len())?;
            }
            copy_planes(
                format,
                width,
                height,
                &planes,
                stored,
                plane_order,
                row_interleaved,
            )
        }
        #[cfg(feature = "std")]
        RawVideoStorage::MixedChromaRows(layout) => {
            copy_mixed_chroma_rows(format, columns, rows, stored, layout)
        }
        #[cfg(feature = "std")]
        RawVideoStorage::PackedYuv422Rows {
            row_align,
            group_bytes,
            component_order,
        } => copy_packed_yuv422_rows(
            columns,
            rows,
            row_align,
            group_bytes,
            component_order,
            stored,
        ),
    }
}

#[derive(Debug, Clone, Copy)]
struct AlignedPlane {
    input_row: usize,
    frame_row: usize,
    stride: usize,
    rows: usize,
    packed_samples: Option<usize>,
}

fn aligned_planes(
    format: RawVideoFormat,
    columns: usize,
    rows: usize,
    align: usize,
    bit_packed: bool,
) -> Option<Vec<AlignedPlane>> {
    (0..format.plane_count())
        .map(|plane| {
            let (_, row, rows) = tight_plane(format, plane, columns, rows)?;
            let pixel_bytes = match format {
                // GStreamer pads Bayer rows to four pixels, which containers need not store
                RawVideoFormat::Bayer(_) => columns.checked_mul(format.pixel_stride()?)?,
                _ => row,
            };
            aligned_plane(format, pixel_bytes, row, rows, align, bit_packed)
        })
        .collect()
}

fn aligned_plane(
    format: RawVideoFormat,
    pixel_bytes: usize,
    frame_row: usize,
    rows: usize,
    align: usize,
    bit_packed: bool,
) -> Option<AlignedPlane> {
    let packed_samples = bit_packed.then(|| pixel_bytes / format.bytes_per_sample());
    let input_row = match packed_samples {
        Some(samples) => samples
            .checked_mul(usize::from(format.bit_depth()))?
            .div_ceil(8),
        None => pixel_bytes,
    };
    Some(AlignedPlane {
        input_row,
        frame_row,
        stride: aligned_stride(input_row, align)?,
        rows,
        packed_samples,
    })
}

fn aligned_stride(row: usize, align: usize) -> Option<usize> {
    match align {
        0 => Some(row),
        align => row.checked_next_multiple_of(align),
    }
}

#[cfg(feature = "std")]
fn copy_packed_yuv422_rows(
    columns: usize,
    rows: usize,
    row_align: u32,
    group_bytes: u32,
    component_order: [usize; 4],
    stored: &[u8],
) -> Option<Vec<u8>> {
    if columns == 0 || !columns.is_multiple_of(2) {
        return None;
    }
    let group_bytes = usize::try_from(group_bytes).ok()?;
    let input_row = (columns / 2).checked_mul(group_bytes)?;
    let stride = aligned_stride(input_row, usize::try_from(row_align).ok()?)?;
    let stored = stored.get(..stride.checked_mul(rows)?)?;
    let mut frame = Vec::new();
    frame
        .try_reserve_exact(columns.checked_mul(rows)?.checked_mul(2)?)
        .ok()?;
    for row in stored.chunks_exact(stride) {
        for group in row[..input_row].chunks_exact(group_bytes) {
            frame.extend_from_slice(&component_order.map(|index| group[index]));
        }
    }
    Some(frame)
}

#[cfg(feature = "std")]
fn copy_mixed_chroma_rows(
    format: RawVideoFormat,
    columns: usize,
    rows: usize,
    stored: &[u8],
    layout: MixedChromaLayout,
) -> Option<Vec<u8>> {
    let align = usize::try_from(layout.row_align).ok()?;
    let (_, y_row, y_rows) = tight_plane(format, 0, columns, rows)?;
    let (_, chroma_row, chroma_rows) = tight_plane(format, 1, columns, rows)?;
    let pair_row = match format {
        RawVideoFormat::Nv12 => chroma_row,
        RawVideoFormat::I422
        | RawVideoFormat::I420p10
        | RawVideoFormat::I422p10
        | RawVideoFormat::I420p12
        | RawVideoFormat::I422p12 => chroma_row.checked_mul(2)?,
        _ => return None,
    };
    let bits = format.bit_depth();
    let bit_packed = layout.packing == MixedSamplePacking::PackedBits;
    let luma_plane = aligned_plane(format, y_row, y_row, y_rows, align, bit_packed)?;
    let chroma_plane = aligned_plane(format, pair_row, pair_row, chroma_rows, align, bit_packed)?;
    let y_len = luma_plane.stride.checked_mul(y_rows)?;
    let pair_len = chroma_plane.stride.checked_mul(chroma_rows)?;
    let stored = stored.get(..y_len.checked_add(pair_len)?)?;
    let (y, pairs) = match layout.chroma_first {
        true => (&stored[pair_len..], &stored[..pair_len]),
        false => stored.split_at(y_len),
    };
    let mut frame = copy_mixed_plane(y, &luma_plane, bits, layout.packing)?;
    let mut pairs = copy_mixed_plane(pairs, &chroma_plane, bits, layout.packing)?;
    frame.try_reserve_exact(pairs.len()).ok()?;
    if format == RawVideoFormat::Nv12 {
        if layout.vu {
            for pair in pairs.as_chunks_mut::<2>().0 {
                pair.swap(0, 1);
            }
        }
        frame.extend_from_slice(&pairs);
    } else {
        let y_len = frame.len();
        frame.resize(y_len.checked_add(pairs.len())?, 0);
        let chroma = &mut frame[y_len..];
        if bits > 8 {
            crate::pixel::deinterleave_chroma(
                pairs.as_chunks::<2>().0,
                chroma.as_chunks_mut::<2>().0,
                layout.vu,
            );
        } else {
            crate::pixel::deinterleave_chroma(&pairs, chroma, layout.vu);
        }
    }
    Some(frame)
}

#[cfg(feature = "std")]
fn copy_mixed_plane(
    input: &[u8],
    plane: &AlignedPlane,
    bits: u8,
    packing: MixedSamplePacking,
) -> Option<Vec<u8>> {
    if let Some(samples) = plane.packed_samples {
        let mut plane_data = Vec::new();
        plane_data
            .try_reserve_exact(plane.frame_row.checked_mul(plane.rows)?)
            .ok()?;
        unpack_plane(&mut plane_data, input, plane, plane.stride, samples, bits)?;
        return Some(plane_data);
    }
    let mut data = copy_rows(
        input,
        plane.input_row,
        plane.frame_row,
        plane.stride,
        plane.rows,
        false,
    )?;
    if let MixedSamplePacking::Words { little_endian } = packing {
        if !little_endian {
            crate::pcmendian::swap_endian(&mut data, 2);
        }
        // ISO/IEC 23001-17 stores the value in the low bits and requires zero padding.
        let limit = 1u16.checked_shl(u32::from(bits))?;
        if data
            .as_chunks::<2>()
            .0
            .iter()
            .any(|&word| u16::from_le_bytes(word) >= limit)
        {
            return None;
        }
    }
    Some(data)
}

fn stored_planes_len(planes: &[AlignedPlane]) -> Option<usize> {
    planes.iter().try_fold(0usize, |len, plane| {
        len.checked_add(plane.stride.checked_mul(plane.rows)?)
    })
}

#[cfg(feature = "std")]
fn match_chroma_alignment(
    planes: &mut [AlignedPlane],
    align: usize,
    stored_len: usize,
) -> Option<()> {
    // libheif applies the full alignment to every plane. The standard halves it
    // for subsampled chroma. With no tile padding, the sample size distinguishes them.
    if stored_planes_len(planes) == Some(stored_len) {
        return Some(());
    }
    for chroma in planes.get_mut(1..=2)? {
        chroma.stride = chroma.input_row.checked_next_multiple_of(align / 2)?;
    }
    (stored_planes_len(planes) == Some(stored_len)).then_some(())
}

#[cfg(feature = "std")]
pub(crate) fn aligned_rows_are_tight(
    format: RawVideoFormat,
    width: u32,
    height: u32,
    row_align: u32,
) -> Option<bool> {
    let planes = aligned_planes(
        format,
        width as usize,
        height as usize,
        usize::try_from(row_align).ok()?,
        false,
    )?;
    Some(planes.iter().all(|plane| plane.stride == plane.frame_row))
}

fn copy_planes(
    format: RawVideoFormat,
    width: u32,
    height: u32,
    planes: &[AlignedPlane],
    mut stored: &[u8],
    plane_order: Option<[u8; 3]>,
    row_interleaved: bool,
) -> Option<Vec<u8>> {
    let group_stride = if row_interleaved {
        if stored.len() < stored_planes_len(planes)? {
            return None;
        }
        Some(
            planes
                .iter()
                .try_fold(0usize, |stride, plane| stride.checked_add(plane.stride))?,
        )
    } else {
        None
    };
    let mut input_planes = [&[][..]; MAX_PLANES];
    for position in 0..planes.len() {
        let index = match &plane_order {
            Some(order) => usize::from(*order.get(position)?),
            None => position,
        };
        let plane = planes.get(index)?;
        let (input, rest) = if row_interleaved {
            (stored, stored.get(plane.stride..)?)
        } else {
            let len = plane.stride.checked_mul(plane.rows)?;
            stored.split_at_checked(len)?
        };
        *input_planes.get_mut(index)? = input;
        stored = rest;
    }
    let mut frame = Vec::new();
    frame
        .try_reserve_exact(frame_byte_size(format, width, height))
        .ok()?;
    for (plane, input) in planes.iter().zip(input_planes) {
        let stride = group_stride.unwrap_or(plane.stride);
        if let Some(samples) = plane.packed_samples {
            unpack_plane(
                &mut frame,
                input,
                plane,
                stride,
                samples,
                format.bit_depth(),
            )?;
            continue;
        }
        let plane_rows = copy_rows(
            input,
            plane.input_row,
            plane.frame_row,
            stride,
            plane.rows,
            false,
        )?;
        frame.extend_from_slice(&plane_rows);
    }
    Some(frame)
}

fn unpack_plane(
    frame: &mut Vec<u8>,
    input: &[u8],
    plane: &AlignedPlane,
    stride: usize,
    samples: usize,
    bits: u8,
) -> Option<()> {
    if stride == 0 {
        return None;
    }
    let padding = plane
        .frame_row
        .checked_sub(samples.checked_mul(size_of::<u16>())?)?;
    let mut rows = input.chunks(stride);
    for _ in 0..plane.rows {
        let mut reader = BitReader::new(rows.next()?.get(..plane.input_row)?);
        for _ in 0..samples {
            let sample = reader.read_bits(u32::from(bits))? as u16;
            frame.extend_from_slice(&sample.to_le_bytes());
        }
        frame.resize(frame.len().checked_add(padding)?, 0);
    }
    Some(())
}

// `rows` rows of `pixel_bytes` bytes, read `stride` bytes apart and zero-padded to `frame_row` bytes, the bottom one first when `bottom_up`
fn copy_rows(
    stored: &[u8],
    pixel_bytes: usize,
    frame_row: usize,
    stride: usize,
    rows: usize,
    bottom_up: bool,
) -> Option<Vec<u8>> {
    let last_row_end = stride
        .checked_mul(rows.checked_sub(1)?)?
        .checked_add(pixel_bytes)?;
    if pixel_bytes == 0
        || stride < pixel_bytes
        || frame_row < pixel_bytes
        || last_row_end > stored.len()
    {
        return None;
    }
    let mut frame = Vec::with_capacity(frame_row * rows);
    for index in 0..rows {
        let source_row = if bottom_up { rows - 1 - index } else { index };
        let start = source_row * stride;
        frame.extend_from_slice(&stored[start..start + pixel_bytes]);
        frame.resize(frame.len() + frame_row - pixel_bytes, 0);
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
    pixel_aspect: g2g_core::PixelAspect,
) -> Caps {
    Caps::RawVideo {
        format,
        width,
        height,
        framerate,
        interlace: Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        pixel_aspect,
    }
}

pub(crate) fn raw_video_placeholder_caps(format: RawVideoFormat) -> Caps {
    let span = || Dim::Range {
        min: PLACEHOLDER_MIN_DIMENSION,
        max: MAX_DIM,
    };
    raw_video_caps(
        format,
        span(),
        span(),
        PLACEHOLDER_FRAMERATE,
        g2g_core::PixelAspect::Any,
    )
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
        assert_eq!(
            copy_raw_video_frame(format, storage, 1, 3, &tight[..2]),
            None
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn quicktime_rows_take_their_stride_from_the_sample_size() {
        let (format, storage) = quicktime_raw_video(24).expect("depth 24");
        let mut padded = bottom_up_padded_rows();
        padded.reverse();
        let frame = copy_raw_video_frame(format, storage, 1, 3, &padded).expect("whole rows");
        let rows: Vec<u8> = padded.chunks(4).flat_map(|row| row[..3].to_vec()).collect();
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

    #[cfg(feature = "std")]
    #[test]
    fn bit_packed_bayer_rows_skip_input_padding_and_zero_output_padding() {
        let format = RawVideoFormat::Bayer(g2g_core::BayerFormat {
            pattern: g2g_core::BayerPattern::Rggb,
            sample: g2g_core::BayerSampleFormat::Bits10Le,
        });
        let storage = RawVideoStorage::AlignedRows {
            row_align: 8,
            plane_order: None,
            bit_packed: true,
            row_interleaved: false,
        };
        let stored = [
            0x00, 0x3f, 0xf5, 0x54, 0xee, 0xee, 0xee, 0xee, 0xaa, 0x80, 0x18, 0x00, 0xee, 0xee,
            0xee, 0xee,
        ];
        let expected: Vec<u8> = [0u16, 1023, 341, 0, 682, 1, 512, 0]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect();
        assert_eq!(
            copy_raw_video_frame(format, storage, 3, 2, &stored),
            Some(expected)
        );
        assert!(copy_raw_video_frame(format, storage, 3, 2, &stored[..11]).is_none());
    }

    #[cfg(feature = "std")]
    #[test]
    fn padded_bit_packed_chroma_rejects_sizes_between_alignments() {
        let storage = RawVideoStorage::AlignedRows {
            row_align: 16,
            plane_order: Some([2, 0, 1]),
            bit_packed: true,
            row_interleaved: false,
        };
        // At 6x4 the recognized sample sizes are 96 and 128 bytes.
        let format = RawVideoFormat::I420p10;
        assert!(copy_raw_video_frame(format, storage, 6, 4, &[0; 127]).is_none());
    }

    #[cfg(feature = "std")]
    #[test]
    fn row_interleaved_frames_require_complete_row_groups() {
        let storage = RawVideoStorage::AlignedRows {
            row_align: 4,
            plane_order: None,
            bit_packed: false,
            row_interleaved: true,
        };
        // Two groups of three four-byte rows. The truncated sample still
        // holds more bytes than the graph's 18-byte RGBP frame.
        assert!(copy_raw_video_frame(RawVideoFormat::Rgbp8, storage, 3, 2, &[0; 23]).is_none());
    }

    #[cfg(feature = "std")]
    #[test]
    fn mixed_chroma_rows_require_a_complete_padded_frame() {
        let storage = RawVideoStorage::MixedChromaRows(MixedChromaLayout {
            row_align: 8,
            chroma_first: true,
            vu: true,
            packing: MixedSamplePacking::Bytes,
        });
        assert!(copy_raw_video_frame(RawVideoFormat::Nv12, storage, 6, 4, &[0; 47]).is_none());
    }

    #[cfg(feature = "std")]
    #[test]
    fn mixed_chroma_words_require_zero_padding_bits() {
        let storage = RawVideoStorage::MixedChromaRows(MixedChromaLayout {
            row_align: 0,
            chroma_first: false,
            vu: false,
            packing: MixedSamplePacking::Words {
                little_endian: false,
            },
        });
        let mut stored = [0; 12];
        assert!(copy_raw_video_frame(RawVideoFormat::I420p10, storage, 2, 2, &stored).is_some());
        stored[0] = 4;
        assert!(copy_raw_video_frame(RawVideoFormat::I420p10, storage, 2, 2, &stored).is_none());
    }

    #[test]
    fn other_depths_of_the_raw_layouts_are_not_read() {
        assert_eq!(avi_raw_video([0; 4], 16, true), None);
        #[cfg(feature = "std")]
        assert_eq!(quicktime_raw_video(16), None);
    }
}
