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
        RawVideoFormat::Nv12 => "NV12",
        RawVideoFormat::I420 => "I420",
        RawVideoFormat::Yuyv => "YUY2",
    }
}

/// Parse a gst-python-ml format string back into a g2g [`RawVideoFormat`].
/// Returns `None` for a format g2g does not model. `YUYV` is accepted as an
/// alias for `YUY2` (V4L2 vs GStreamer spelling of the same packed 4:2:2).
pub fn format_from_py(s: &str) -> Option<RawVideoFormat> {
    Some(match s {
        "RGBA" => RawVideoFormat::Rgba8,
        "BGRA" => RawVideoFormat::Bgra8,
        "NV12" => RawVideoFormat::Nv12,
        "I420" => RawVideoFormat::I420,
        "YUY2" | "YUYV" => RawVideoFormat::Yuyv,
        _ => return None,
    })
}

/// Bytes one `width` x `height` frame of `fmt` occupies, for allocating a blank
/// source buffer. Packed formats are exact; planar 4:2:0 (`Nv12` / `I420`) is
/// the standard `w*h*3/2`.
pub fn frame_bytes(fmt: RawVideoFormat, width: u32, height: u32) -> usize {
    let (w, h) = (width as usize, height as usize);
    match fmt {
        RawVideoFormat::Rgba8 | RawVideoFormat::Bgra8 => w * h * 4,
        RawVideoFormat::Yuyv => w * h * 2,
        RawVideoFormat::Nv12 | RawVideoFormat::I420 => w * h * 3 / 2,
    }
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
        ] {
            assert_eq!(format_from_py(format_to_py(fmt)), Some(fmt));
        }
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
}
