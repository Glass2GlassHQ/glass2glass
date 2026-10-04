//! Shared pixel-format helpers for the packed-RGBA element family.

use g2g_core::RawVideoFormat;

/// Byte offsets of the red and blue channels in a packed 4-byte pixel (green is
/// always index 1, alpha index 3). RGBA is `[R, G, B, A]`, BGRA `[B, G, R, A]`.
/// Only the two packed formats are admitted by the callers' negotiation.
pub(crate) fn rgba_rb_offsets(format: RawVideoFormat) -> (usize, usize) {
    match format {
        RawVideoFormat::Rgba8 => (0, 2),
        RawVideoFormat::Bgra8 => (2, 0),
        _ => unreachable!("packed RGBA / BGRA only"),
    }
}

/// Luma of pixel `(x, y)` in a tightly packed `w x h` frame of `format`.
/// Packed RGBA / BGRA use BT.709; I420 reads the Y plane. Other formats are
/// not admitted by the callers.
pub(crate) fn luma_at(format: RawVideoFormat, w: u32, src: &[u8], x: u32, y: u32) -> u8 {
    let (w, x, y) = (w as usize, x as usize, y as usize);
    match format {
        RawVideoFormat::I420 => src[y * w + x],
        RawVideoFormat::Rgba8 | RawVideoFormat::Bgra8 => {
            let i = (y * w + x) * 4;
            let (r_idx, b_idx) = rgba_rb_offsets(format);
            bt709_luma(src[i + r_idx], src[i + 1], src[i + b_idx])
        }
        _ => unreachable!("luma_at: I420 / packed RGBA only"),
    }
}

/// BT.709 luma of an 8-bit RGB triple: 0.2126 R + 0.7152 G + 0.0722 B in 16-bit
/// fixed point. The grey a packed-RGBA element writes when it drops colour, and
/// the brightness it tests a pixel by.
pub(crate) fn bt709_luma(r: u8, g: u8, b: u8) -> u8 {
    const LUMA_R: u32 = 13938;
    const LUMA_G: u32 = 46869;
    const LUMA_B: u32 = 4730;
    const LUMA_SHIFT: u32 = 16;
    let luma = (LUMA_R * r as u32 + LUMA_G * g as u32 + LUMA_B * b as u32) >> LUMA_SHIFT;
    luma.min(u8::MAX as u32) as u8
}

/// Whether a format's samples are YUV, so its caps colorimetry names the matrix
/// a converter has to apply or undo. The packed RGB formats are the only ones
/// that are not.
pub(crate) fn carries_yuv(format: RawVideoFormat) -> bool {
    !matches!(
        format,
        RawVideoFormat::Rgba8 | RawVideoFormat::Bgra8 | RawVideoFormat::Rgb8
    )
}

/// Whether a format's chroma subsampling forces an even (width, height): a
/// horizontally-subsampled format needs even width, a vertically-subsampled one
/// needs even height, so a crop / scale stays on chroma-sample boundaries. NV12
/// and YUYV are handled explicitly (NV12 is 4:2:0, YUYV packed 4:2:2); the fully
/// planar family follows its [`RawVideoFormat::chroma_shift`]; RGBA needs neither.
pub(crate) fn even_dims_required(format: RawVideoFormat) -> (bool, bool) {
    match format {
        RawVideoFormat::Nv12 | RawVideoFormat::P010 => (true, true),
        RawVideoFormat::Yuyv => (true, false),
        _ => match format.chroma_shift() {
            Some((hs, vs)) => (hs > 0, vs > 0),
            None => (false, false),
        },
    }
}

/// Byte layout of a fully-planar YUV `format` at `w x h`: `(byte offset, plane
/// width in samples, plane height)` for the Y, U, and V planes in turn. Chroma
/// plane dimensions follow the format's subsampling; the sample byte width is
/// [`RawVideoFormat::bytes_per_sample`]. Panics if `format` is not fully planar.
pub(crate) fn planar_planes(
    format: RawVideoFormat,
    w: usize,
    h: usize,
) -> [(usize, usize, usize); 3] {
    assert!(format.is_planar_yuv(), "fully-planar format");
    let bytes_per_sample = format.bytes_per_sample();
    [0, 1, 2].map(|plane| {
        let (offset, row_bytes, rows) = tight_plane(format, plane, w, h).expect("frame size fits");
        (offset, row_bytes / bytes_per_sample, rows)
    })
}

// (byte offset, row bytes, rows) of one plane of a tight frame, as g2g-core defines it
pub(crate) fn tight_plane(
    format: RawVideoFormat,
    plane: usize,
    w: usize,
    h: usize,
) -> Option<(usize, usize, usize)> {
    let (w, h) = (u32::try_from(w).ok()?, u32::try_from(h).ok()?);
    let offset = usize::try_from(format.plane_offset(plane, w, h)?).ok()?;
    let row_bytes = usize::try_from(format.plane_stride(plane, w)?).ok()?;
    let rows = usize::try_from(format.plane_rows(plane, h)?).ok()?;
    Some((offset, row_bytes, rows))
}

/// Per-plane `(row bytes, rows)` of one `w x h` frame in `format`, in plane
/// order: the shape a [`PlaneLayout`](g2g_core::meta::PlaneLayout) puts offsets
/// and strides on. A tightly-packed frame is exactly these rows back to back;
/// a padded one differs only in where each row starts. `None` on overflow.
#[cfg(feature = "metadata")]
pub(crate) fn plane_shapes(
    format: RawVideoFormat,
    w: usize,
    h: usize,
) -> Option<alloc::vec::Vec<(usize, usize)>> {
    let shapes = crate::paddedrows::plane_shapes_with_stride_shift(format, w, h)?;
    Some(
        shapes
            .into_iter()
            .map(|(row_bytes, rows, _)| (row_bytes, rows))
            .collect(),
    )
}

/// Copy a frame whose planes sit where `layout` says into its tightly-packed
/// form. `None` when the layout does not describe this format's planes, or the
/// buffer does not hold what it claims: a layout can come from any producer, so
/// a bad one fails the frame instead of reading out of bounds.
#[cfg(feature = "metadata")]
pub(crate) fn pack_planes(
    src: &[u8],
    format: RawVideoFormat,
    w: usize,
    h: usize,
    layout: &g2g_core::meta::PlaneLayout,
) -> Option<alloc::boxed::Box<[u8]>> {
    let shapes = plane_shapes(format, w, h)?;
    if layout.count() != shapes.len() {
        return None;
    }
    let mut out = alloc::vec::Vec::with_capacity(frame_byte_size(format, w as u32, h as u32));
    for (plane, &(row_bytes, rows)) in shapes.iter().enumerate() {
        for row in 0..rows {
            out.extend_from_slice(src.get(layout.row_range(plane, row, row_bytes)?)?);
        }
    }
    Some(out.into_boxed_slice())
}

/// Byte width of one row of `format`'s **first** plane at `w` pixels: the row
/// pitch of a tightly-packed frame, [`RawVideoFormat::plane_stride`] as a
/// `usize`. Saturates at `usize::MAX` on overflow.
pub(crate) fn row_bytes(format: RawVideoFormat, w: usize) -> usize {
    u32::try_from(w)
        .ok()
        .and_then(|w| format.plane_stride(0, w))
        .and_then(|stride| usize::try_from(stride).ok())
        .unwrap_or(usize::MAX)
}

/// Tightly-packed byte size of one `w x h` frame in `format` (no row padding),
/// [`RawVideoFormat::unpadded_frame_bytes`] as a `usize`. Saturates at
/// `usize::MAX` on overflow, so a length check against it fails.
pub(crate) fn frame_byte_size(format: RawVideoFormat, w: u32, h: u32) -> usize {
    format
        .unpadded_frame_bytes(w, h)
        .and_then(|bytes| usize::try_from(bytes).ok())
        .unwrap_or(usize::MAX)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) const EVERY_FORMAT: [RawVideoFormat; 15] = [
        RawVideoFormat::Nv12,
        RawVideoFormat::I420,
        RawVideoFormat::Rgba8,
        RawVideoFormat::Bgra8,
        RawVideoFormat::Rgb8,
        RawVideoFormat::Yuyv,
        RawVideoFormat::I420p10,
        RawVideoFormat::I420p12,
        RawVideoFormat::I422,
        RawVideoFormat::I422p10,
        RawVideoFormat::I422p12,
        RawVideoFormat::I444,
        RawVideoFormat::I444p10,
        RawVideoFormat::I444p12,
        RawVideoFormat::P010,
    ];

    // odd and even on each axis, so a subsampled chroma plane has to round up
    pub(crate) const GEOMETRIES: [(u32, u32); 5] = [(37, 5), (37, 4), (38, 5), (38, 4), (1, 1)];

    #[test]
    fn frame_size_is_the_core_definition() {
        for format in EVERY_FORMAT {
            for (w, h) in GEOMETRIES {
                let core = format.unpadded_frame_bytes(w, h).unwrap() as usize;
                assert_eq!(frame_byte_size(format, w, h), core, "{format:?} {w}x{h}");
            }
        }
    }

    #[test]
    fn planes_are_the_core_definition() {
        for format in EVERY_FORMAT {
            for (w, h) in GEOMETRIES {
                let (wu, hu) = (w as usize, h as usize);
                let core_plane = |plane: usize| {
                    (
                        format.plane_offset(plane, w, h).unwrap() as usize,
                        format.plane_stride(plane, w).unwrap() as usize,
                        format.plane_rows(plane, h).unwrap() as usize,
                    )
                };
                if format.is_planar_yuv() {
                    let bytes_per_sample = format.bytes_per_sample();
                    for (plane, (offset, samples, rows)) in
                        planar_planes(format, wu, hu).into_iter().enumerate()
                    {
                        assert_eq!(
                            (offset, samples * bytes_per_sample, rows),
                            core_plane(plane),
                            "{format:?} {w}x{h} plane {plane}"
                        );
                    }
                }
                #[cfg(feature = "metadata")]
                assert_eq!(
                    plane_shapes(format, wu, hu).unwrap(),
                    (0..format.plane_count())
                        .map(|plane| {
                            let (_, row_bytes, rows) = core_plane(plane);
                            (row_bytes, rows)
                        })
                        .collect::<alloc::vec::Vec<_>>(),
                    "{format:?} {w}x{h}"
                );
            }
        }
    }

    #[test]
    fn an_odd_width_semi_planar_chroma_row_holds_a_whole_last_pair() {
        let (w, h) = (GEOMETRIES[0].0 as usize, GEOMETRIES[0].1 as usize);
        for format in [RawVideoFormat::Nv12, RawVideoFormat::P010] {
            let (_, luma_row, _) = tight_plane(format, 0, w, h).unwrap();
            let (_, chroma_row, chroma_rows) = tight_plane(format, 1, w, h).unwrap();
            assert_eq!(
                chroma_row,
                luma_row + format.bytes_per_sample(),
                "{format:?}"
            );
            assert_eq!(chroma_rows, h.div_ceil(2), "{format:?}");
        }
    }

    #[test]
    fn an_overflowing_frame_size_saturates() {
        assert_eq!(
            frame_byte_size(RawVideoFormat::Rgba8, u32::MAX, u32::MAX),
            usize::MAX
        );
    }
}
