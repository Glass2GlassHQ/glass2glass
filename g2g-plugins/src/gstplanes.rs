use core::ffi::{c_int, c_uint};

use g2g_core::meta::{Plane, PlaneLayout, MAX_PLANES};
use g2g_core::{Caps, Dim, RawVideoFormat};

// `GST_VIDEO_MAX_PLANES`, which the C sides assert at compile time.
pub const GST_VIDEO_MAX_PLANES: usize = 4;

const _: () = assert!(MAX_PLANES == GST_VIDEO_MAX_PLANES);

// mirrors `G2gGstWrapPlanes` in the gstwrap helper and `G2gBridgePlanes` in the bridge shell
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GstVideoPlanes {
    count: c_uint,
    offsets: [usize; GST_VIDEO_MAX_PLANES],
    strides: [c_int; GST_VIDEO_MAX_PLANES],
    size: usize,
}

impl GstVideoPlanes {
    pub fn tight(format: RawVideoFormat, width: u32, height: u32) -> Option<Self> {
        Self::of_layout(&tight_layout(format, width, height)?, format, width, height)
    }

    pub fn tight_for_caps(caps: &Caps) -> Option<Self> {
        match caps {
            Caps::RawVideo {
                format,
                width: Dim::Fixed(width),
                height: Dim::Fixed(height),
                ..
            } => Self::tight(*format, *width, *height),
            _ => None,
        }
    }

    // `None` when `layout` does not have the format's planes or a row overflows
    pub fn of_layout(
        layout: &PlaneLayout,
        format: RawVideoFormat,
        width: u32,
        height: u32,
    ) -> Option<Self> {
        let count = format.plane_count();
        if layout.count() != count || count > GST_VIDEO_MAX_PLANES {
            return None;
        }
        let mut planes = Self {
            count: c_uint::try_from(count).ok()?,
            offsets: [0; GST_VIDEO_MAX_PLANES],
            strides: [0; GST_VIDEO_MAX_PLANES],
            size: 0,
        };
        for index in 0..count {
            let plane = layout.plane(index)?;
            let row_bytes = usize::try_from(format.plane_stride(index, width)?).ok()?;
            let last_row =
                usize::try_from(format.plane_rows(index, height)?.checked_sub(1)?).ok()?;
            let end = layout.row_range(index, last_row, row_bytes)?.end;
            planes.offsets[index] = plane.offset;
            planes.strides[index] = c_int::try_from(plane.stride).ok()?;
            planes.size = planes.size.max(end);
        }
        Some(planes)
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

fn tight_layout(format: RawVideoFormat, width: u32, height: u32) -> Option<PlaneLayout> {
    let count = format.plane_count();
    let mut planes = [Plane {
        offset: 0,
        stride: 0,
    }; MAX_PLANES];
    for (index, plane) in planes.iter_mut().enumerate().take(count) {
        *plane = Plane {
            offset: usize::try_from(format.plane_offset(index, width, height)?).ok()?,
            stride: usize::try_from(format.plane_stride(index, width)?).ok()?,
        };
    }
    PlaneLayout::new(planes.get(..count)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tight_frame_spans_its_unpadded_bytes() {
        const WIDTH: u32 = 37;
        const HEIGHT: u32 = 5;
        for format in [
            RawVideoFormat::Rgb8,
            RawVideoFormat::I420,
            RawVideoFormat::Nv12,
        ] {
            let planes = GstVideoPlanes::tight(format, WIDTH, HEIGHT).expect("planes");
            let frame_bytes = format.unpadded_frame_bytes(WIDTH, HEIGHT).unwrap();
            assert_eq!(planes.size() as u64, frame_bytes, "{format:?}");
        }
    }
}
