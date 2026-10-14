//! The shared pixel-format contract between g2g and the gst-python-ml backend.
//!
//! gst-python-ml's `tasks/frame_format.py` and its `FrameIO.read_frame(s)`
//! speak GStreamer-style format strings (`"RGBA"`, `"NV12"`, ...). g2g speaks
//! [`RawVideoFormat`]. Caps negotiation (which format the link carries) and the
//! per-frame call (which `fmt` string Python is handed) must agree, so both
//! sides use [`RawVideoFormat::gst_name`] and [`g2g_core::caps_parse::parse_raw_format`].
//!
//! An unmapped string from the Python side is a negotiation error, not a
//! silent guess.

use g2g_core::RawVideoFormat;

/// Bytes one `width` x `height` frame of `fmt` occupies, for allocating a blank
/// source buffer: [`RawVideoFormat::unpadded_frame_bytes`] as a `usize`,
/// saturating at `usize::MAX` on overflow so the allocation fails loud.
pub fn frame_bytes(fmt: RawVideoFormat, width: u32, height: u32) -> usize {
    fmt.unpadded_frame_bytes(width, height)
        .and_then(|bytes| usize::try_from(bytes).ok())
        .unwrap_or(usize::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_bytes_match_planar_geometry() {
        // 4x4: I420 8-bit = 16 + 2*4 = 24; 10-bit doubles to 48.
        assert_eq!(frame_bytes(RawVideoFormat::I420, 4, 4), 24);
        assert_eq!(frame_bytes(RawVideoFormat::I420p10, 4, 4), 48);
        // 4:2:2 keeps full height: 16 + 2*(2*4) = 32 (8-bit), 64 (12-bit).
        assert_eq!(frame_bytes(RawVideoFormat::I422, 4, 4), 32);
        assert_eq!(frame_bytes(RawVideoFormat::I422p12, 4, 4), 64);
        // 4:4:4 full chroma: 16 + 2*16 = 48 (8-bit), 96 (10-bit).
        assert_eq!(frame_bytes(RawVideoFormat::I444, 4, 4), 48);
        assert_eq!(frame_bytes(RawVideoFormat::I444p10, 4, 4), 96);
    }

    #[test]
    fn frame_bytes_per_format() {
        assert_eq!(frame_bytes(RawVideoFormat::Rgba8, 4, 2), 32);
        assert_eq!(frame_bytes(RawVideoFormat::Yuyv, 4, 2), 16);
        assert_eq!(frame_bytes(RawVideoFormat::Nv12, 4, 2), 12);
    }

    #[test]
    fn an_odd_size_semi_planar_frame_holds_its_rounded_up_chroma() {
        const ODD_WIDTH: u32 = 37;
        const ODD_HEIGHT: u32 = 5;
        for fmt in [RawVideoFormat::Nv12, RawVideoFormat::P010] {
            assert_eq!(
                Some(frame_bytes(fmt, ODD_WIDTH, ODD_HEIGHT) as u64),
                fmt.unpadded_frame_bytes(ODD_WIDTH, ODD_HEIGHT),
                "{fmt:?}"
            );
        }
    }
}
