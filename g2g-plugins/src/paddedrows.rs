//! Where a raw frame's rows sit when a producer pads them, and the copy that
//! packs them tight again.
//!
//! A V4L2 `v4l2_pix_format`, a PipeWire buffer chunk and a mapped VAAPI image
//! all report one row stride, plane 0's, and lay every later plane out at the
//! stride its own row width implies: the same stride for NV12's interleaved
//! chroma, half of it for I420's. A producer either declares that with a
//! [`PlaneLayout`](g2g_core::meta::PlaneLayout), when a consumer asked for one,
//! or copies the rows into the tightly packed shape everything downstream
//! assumes by default.

use alloc::vec::Vec;

#[cfg(any(
    all(target_os = "linux", feature = "v4l2"),
    all(target_os = "linux", feature = "pipewire")
))]
use g2g_core::frame::Frame;
use g2g_core::RawVideoFormat;

/// Per-plane `(row bytes, rows, stride shift)` of one `w x h` frame in
/// `format`, in plane order. The shift derives a plane's row stride from plane
/// 0's, the only stride these producers report: a horizontally-subsampled
/// planar format's chroma rows sit at half the luma stride, NV12's interleaved
/// chroma plane at the full one. `None` on overflow.
pub(crate) fn plane_shapes_with_stride_shift(
    format: RawVideoFormat,
    w: usize,
    h: usize,
) -> Option<Vec<(usize, usize, u32)>> {
    let chroma_stride_shift = format
        .chroma_shift()
        .map_or(0, |(horizontal, _)| horizontal);
    (0..format.plane_count())
        .map(|plane| {
            let (_, row_bytes, rows) = crate::pixel::tight_plane(format, plane, w, h)?;
            let shift = if plane == 0 { 0 } else { chroma_stride_shift };
            Some((row_bytes, rows, shift))
        })
        .collect()
}

/// One plane of a padded frame: where its rows start, how far apart they are,
/// how wide each one is and how many there are.
#[cfg(any(
    all(target_os = "linux", feature = "v4l2"),
    all(target_os = "linux", feature = "pipewire"),
    all(target_os = "linux", feature = "dmabuf-wgpu")
))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PaddedPlane {
    pub offset: usize,
    pub stride: usize,
    pub row_bytes: usize,
    pub rows: usize,
}

/// Every plane of a `w x h` frame whose plane 0 starts at `plane0_offset` with
/// its rows `first_stride` bytes apart (0 means tightly packed). `None` when a
/// stride cannot hold its own row or the arithmetic overflows: the stride comes
/// from a driver or a daemon, so it is checked once here and every caller
/// slices with what it gets back.
#[cfg(any(
    all(target_os = "linux", feature = "v4l2"),
    all(target_os = "linux", feature = "pipewire"),
    all(target_os = "linux", feature = "dmabuf-wgpu")
))]
pub(crate) fn padded_planes(
    format: RawVideoFormat,
    w: usize,
    h: usize,
    plane0_offset: usize,
    first_stride: usize,
) -> Option<Vec<PaddedPlane>> {
    let mut offset = plane0_offset;
    let mut planes = Vec::new();
    for (row_bytes, rows, shift) in plane_shapes_with_stride_shift(format, w, h)? {
        let stride = match first_stride {
            0 => row_bytes,
            s => s >> shift,
        };
        if stride < row_bytes {
            return None;
        }
        planes.push(PaddedPlane {
            offset,
            stride,
            row_bytes,
            rows,
        });
        offset = offset.checked_add(stride.checked_mul(rows)?)?;
    }
    Some(planes)
}

/// How many bytes a padded frame occupies: the end of its last plane's last
/// row, `plane0_offset` included.
#[cfg(any(
    all(target_os = "linux", feature = "v4l2"),
    all(target_os = "linux", feature = "pipewire")
))]
pub(crate) fn padded_frame_bytes(
    format: RawVideoFormat,
    w: usize,
    h: usize,
    plane0_offset: usize,
    first_stride: usize,
) -> Option<usize> {
    let planes = padded_planes(format, w, h, plane0_offset, first_stride)?;
    let last = planes.last()?;
    last.offset
        .checked_add(last.stride.checked_mul(last.rows.saturating_sub(1))?)?
        .checked_add(last.row_bytes)
}

/// Append the frame in `src` to `dst`, tightly packed, reading plane 0's rows
/// `first_stride` bytes apart (0 means already tight). `None` when `src` does
/// not hold what the geometry claims, in which case `dst` may hold a partial
/// frame and has to be discarded.
#[cfg(any(
    all(target_os = "linux", feature = "v4l2"),
    all(target_os = "linux", feature = "pipewire")
))]
pub(crate) fn pack_tight(
    src: &[u8],
    format: RawVideoFormat,
    w: usize,
    h: usize,
    first_stride: usize,
    dst: &mut Vec<u8>,
) -> Option<()> {
    for plane in padded_planes(format, w, h, 0, first_stride)? {
        for row in 0..plane.rows {
            let start = plane.offset.checked_add(plane.stride.checked_mul(row)?)?;
            let end = start.checked_add(plane.row_bytes)?;
            if end > src.len() {
                return None;
            }
            dst.extend_from_slice(&src[start..end]);
        }
    }
    Some(())
}

/// The meta describing a padded frame's rows, `None` when the geometry and the
/// stride do not fit together.
#[cfg(all(
    feature = "metadata",
    any(
        all(target_os = "linux", feature = "v4l2"),
        all(target_os = "linux", feature = "pipewire")
    )
))]
pub(crate) fn padded_plane_layout(
    format: RawVideoFormat,
    w: usize,
    h: usize,
    plane0_offset: usize,
    first_stride: usize,
) -> Option<g2g_core::meta::PlaneLayout> {
    let planes: Vec<g2g_core::meta::Plane> =
        padded_planes(format, w, h, plane0_offset, first_stride)?
            .into_iter()
            .map(|p| g2g_core::meta::Plane {
                offset: p.offset,
                stride: p.stride,
            })
            .collect();
    g2g_core::meta::PlaneLayout::new(&planes)
}

/// Declare on `frame` where its padded rows sit. `first_stride` of 0 means the
/// producer packed them tight, so nothing is declared. Compiles away without
/// the `metadata` feature, where there is no meta to attach.
#[cfg(any(
    all(target_os = "linux", feature = "v4l2"),
    all(target_os = "linux", feature = "pipewire")
))]
pub(crate) fn declare_padded_rows(
    frame: &mut Frame,
    format: RawVideoFormat,
    w: usize,
    h: usize,
    plane0_offset: usize,
    first_stride: usize,
) {
    #[cfg(feature = "metadata")]
    if first_stride != 0 {
        if let Some(layout) = padded_plane_layout(format, w, h, plane0_offset, first_stride) {
            frame.meta.attach(layout);
        }
    }
    #[cfg(not(feature = "metadata"))]
    let _ = (frame, format, w, h, plane0_offset, first_stride);
}

#[cfg(all(
    test,
    any(
        all(target_os = "linux", feature = "v4l2"),
        all(target_os = "linux", feature = "pipewire")
    )
))]
mod tests {
    use super::*;

    #[test]
    fn a_tight_stride_lays_the_planes_back_to_back() {
        let planes = padded_planes(RawVideoFormat::I420, 4, 4, 0, 0).unwrap();
        assert_eq!(
            planes,
            [
                PaddedPlane {
                    offset: 0,
                    stride: 4,
                    row_bytes: 4,
                    rows: 4
                },
                PaddedPlane {
                    offset: 16,
                    stride: 2,
                    row_bytes: 2,
                    rows: 2
                },
                PaddedPlane {
                    offset: 20,
                    stride: 2,
                    row_bytes: 2,
                    rows: 2
                },
            ]
        );
        assert_eq!(
            padded_frame_bytes(RawVideoFormat::I420, 4, 4, 0, 0),
            Some(24)
        );
    }

    #[test]
    fn chroma_strides_follow_the_first_plane() {
        // I420 halves the luma stride, NV12's interleaved plane keeps it.
        let i420 = padded_planes(RawVideoFormat::I420, 4, 4, 0, 8).unwrap();
        assert_eq!(i420[0].stride, 8);
        assert_eq!(i420[1].stride, 4);
        assert_eq!(i420[1].offset, 32);
        assert_eq!(i420[2].offset, 32 + 8);

        let nv12 = padded_planes(RawVideoFormat::Nv12, 4, 4, 0, 8).unwrap();
        assert_eq!(nv12.len(), 2);
        assert_eq!(nv12[1].stride, 8);
        assert_eq!(nv12[1].offset, 32);
        assert_eq!(
            padded_frame_bytes(RawVideoFormat::Nv12, 4, 4, 0, 8),
            Some(44)
        );
    }

    #[test]
    fn shapes_are_the_core_planes() {
        use crate::pixel::tests::{EVERY_FORMAT, GEOMETRIES};
        for format in EVERY_FORMAT {
            for (w, h) in GEOMETRIES {
                let shapes = plane_shapes_with_stride_shift(format, w as usize, h as usize)
                    .expect("small frame");
                let core: Vec<(usize, usize)> = (0..format.plane_count())
                    .map(|plane| {
                        (
                            format.plane_stride(plane, w).unwrap() as usize,
                            format.plane_rows(plane, h).unwrap() as usize,
                        )
                    })
                    .collect();
                let rows: Vec<(usize, usize)> = shapes
                    .iter()
                    .map(|&(row_bytes, rows, _)| (row_bytes, rows))
                    .collect();
                assert_eq!(rows, core, "{format:?} {w}x{h}");
                let tight = padded_frame_bytes(format, w as usize, h as usize, 0, 0);
                assert_eq!(
                    tight,
                    Some(format.unpadded_frame_bytes(w, h).unwrap() as usize),
                    "{format:?} {w}x{h}"
                );
            }
        }
    }

    #[test]
    fn an_odd_width_nv12_stride_has_to_hold_the_wider_chroma_row() {
        let (w, h) = crate::pixel::tests::GEOMETRIES[0];
        let (w, h) = (w as usize, h as usize);
        let format = RawVideoFormat::Nv12;
        let luma_row = format.plane_stride(0, w as u32).unwrap() as usize;
        let chroma_row = format.plane_stride(1, w as u32).unwrap() as usize;
        assert!(chroma_row > luma_row);
        assert_eq!(padded_planes(format, w, h, 0, luma_row), None);
        let planes = padded_planes(format, w, h, 0, chroma_row).expect("holds both rows");
        assert_eq!(planes[1].row_bytes, chroma_row);
        assert_eq!(planes[1].stride, chroma_row);
    }

    #[test]
    fn packing_an_odd_width_nv12_frame_keeps_the_last_chroma_pair() {
        let (w, h) = crate::pixel::tests::GEOMETRIES[0];
        let (w, h) = (w as usize, h as usize);
        let format = RawVideoFormat::Nv12;
        let chroma_row = format.plane_stride(1, w as u32).unwrap() as usize;
        let stride = chroma_row + 2;
        let rows = h + h.div_ceil(2);
        let src: Vec<u8> = (0..stride * rows).map(|i| i as u8).collect();
        let mut dst = Vec::new();
        pack_tight(&src, format, w, h, stride, &mut dst).expect("the buffer fits");
        assert_eq!(
            dst.len(),
            format.unpadded_frame_bytes(w as u32, h as u32).unwrap() as usize
        );
        let last_chroma_row = &src[(rows - 1) * stride..][..chroma_row];
        assert_eq!(&dst[dst.len() - chroma_row..], last_chroma_row);
    }

    #[test]
    fn a_stride_narrower_than_a_row_is_rejected() {
        assert_eq!(padded_planes(RawVideoFormat::Yuyv, 4, 4, 0, 7), None);
        assert_eq!(
            padded_planes(RawVideoFormat::I420, 4, 4, 0, usize::MAX),
            None,
            "the plane offsets overflow"
        );
    }

    #[test]
    fn packing_de_strides_every_plane() {
        // 2x2 I420 with 4-byte luma rows: chroma rows are 2 bytes apart.
        let src = [
            1, 2, 0xff, 0xff, // Y row 0 + pad
            3, 4, 0xff, 0xff, // Y row 1 + pad
            5, 0xff, // U row 0 + pad
            6, 0xff, // V row 0 + pad
        ];
        let mut dst = Vec::new();
        pack_tight(&src, RawVideoFormat::I420, 2, 2, 4, &mut dst).expect("the buffer fits");
        assert_eq!(dst, [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn packing_a_short_buffer_fails() {
        // The V plane's row ends at byte 11, so 10 bytes cannot hold the frame.
        let mut dst = Vec::new();
        assert_eq!(
            pack_tight(&[0u8; 10], RawVideoFormat::I420, 2, 2, 4, &mut dst),
            None
        );
        assert_eq!(
            pack_tight(&[0u8; 11], RawVideoFormat::I420, 2, 2, 4, &mut dst),
            Some(())
        );
    }

    #[cfg(feature = "metadata")]
    #[test]
    fn the_layout_addresses_the_padded_rows() {
        use g2g_core::meta::Plane;
        let layout = padded_plane_layout(RawVideoFormat::Nv12, 4, 4, 0, 8).unwrap();
        assert_eq!(layout.count(), 2);
        assert_eq!(
            layout.plane(1),
            Some(Plane {
                offset: 32,
                stride: 8
            })
        );
        // the last chroma row of a 4x4 NV12 frame at stride 8
        assert_eq!(layout.row_range(1, 1, 4), Some(40..44));

        // A chunk offset shifts every plane.
        let shifted = padded_plane_layout(RawVideoFormat::Nv12, 4, 4, 16, 8).unwrap();
        assert_eq!(
            shifted.plane(0),
            Some(Plane {
                offset: 16,
                stride: 8
            })
        );
        assert_eq!(
            shifted.plane(1),
            Some(Plane {
                offset: 48,
                stride: 8
            })
        );
    }
}
