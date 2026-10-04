//! The shared pixel-format contract between g2g and the gst-python-ml backend.
//!
//! gst-python-ml's `tasks/frame_format.py` and its `FrameIO.read_frame(s)`
//! speak GStreamer-style format strings (`"RGBA"`, `"NV12"`, ...). g2g speaks
//! [`RawVideoFormat`]. Caps negotiation (which format the link carries) and the
//! per-frame call (which `fmt` string Python is handed) must agree, so the
//! mapping lives in one place here rather than being re-derived on each side.
//!
//! Only the formats g2g currently models are mapped; an unmapped string from
//! the Python side is a negotiation error, not a silent guess.

use g2g_core::RawVideoFormat;

/// The GStreamer-style format string for a g2g raw-video format, as the
/// gst-python-ml `FrameIO` / `frame_format` code expects it.
pub fn format_to_py(fmt: RawVideoFormat) -> &'static str {
    match fmt {
        RawVideoFormat::Rgba8 => "RGBA",
        RawVideoFormat::Bgra8 => "BGRA",
        RawVideoFormat::Rgb8 => "RGB",
        RawVideoFormat::Nv12 => "NV12",
        RawVideoFormat::I420 => "I420",
        RawVideoFormat::Yuyv => "YUY2",
        RawVideoFormat::I420p10 => "I420_10LE",
        RawVideoFormat::I420p12 => "I420_12LE",
        RawVideoFormat::I422 => "Y42B",
        RawVideoFormat::I422p10 => "I422_10LE",
        RawVideoFormat::I422p12 => "I422_12LE",
        RawVideoFormat::I444 => "Y444",
        RawVideoFormat::I444p10 => "Y444_10LE",
        RawVideoFormat::I444p12 => "Y444_12LE",
        RawVideoFormat::P010 => "P010_10LE",
        // A g2g format this binding does not model (or one added since): return
        // a marker the gst-python-ml `FrameIO` will reject, rather than guess.
        _ => "UNKNOWN",
    }
}

/// Parse a gst-python-ml format string back into a g2g [`RawVideoFormat`].
/// Returns `None` for a format g2g does not model. `YUYV` is accepted as an
/// alias for `YUY2` (V4L2 vs GStreamer spelling of the same packed 4:2:2).
pub fn format_from_py(s: &str) -> Option<RawVideoFormat> {
    Some(match s {
        "RGBA" => RawVideoFormat::Rgba8,
        "BGRA" => RawVideoFormat::Bgra8,
        "RGB" => RawVideoFormat::Rgb8,
        "NV12" => RawVideoFormat::Nv12,
        "I420" => RawVideoFormat::I420,
        "YUY2" | "YUYV" => RawVideoFormat::Yuyv,
        "I420_10LE" => RawVideoFormat::I420p10,
        "I420_12LE" => RawVideoFormat::I420p12,
        "Y42B" => RawVideoFormat::I422,
        "I422_10LE" => RawVideoFormat::I422p10,
        "I422_12LE" => RawVideoFormat::I422p12,
        "Y444" => RawVideoFormat::I444,
        "Y444_10LE" => RawVideoFormat::I444p10,
        "Y444_12LE" => RawVideoFormat::I444p12,
        "P010_10LE" => RawVideoFormat::P010,
        _ => return None,
    })
}

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
    fn round_trips_every_modeled_format() {
        for fmt in [
            RawVideoFormat::Rgba8,
            RawVideoFormat::Bgra8,
            RawVideoFormat::Nv12,
            RawVideoFormat::I420,
            RawVideoFormat::Yuyv,
            RawVideoFormat::I420p10,
            RawVideoFormat::I420p12,
            RawVideoFormat::I422,
            RawVideoFormat::I422p10,
            RawVideoFormat::I422p12,
            RawVideoFormat::I444,
            RawVideoFormat::I444p10,
            RawVideoFormat::I444p12,
        ] {
            assert_eq!(format_from_py(format_to_py(fmt)), Some(fmt));
        }
    }

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
    fn yuyv_is_an_alias_for_yuy2() {
        assert_eq!(format_from_py("YUYV"), Some(RawVideoFormat::Yuyv));
        assert_eq!(format_from_py("YUY2"), Some(RawVideoFormat::Yuyv));
    }

    #[test]
    fn unmodeled_format_is_none() {
        assert_eq!(format_from_py("GRAY8"), None);
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
