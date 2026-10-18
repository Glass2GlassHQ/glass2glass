use g2g_core::log::Target;
use g2g_core::{g2g_warn, Caps, Dim, Interlace, Rate, RawVideoFormat};

use crate::pixel::frame_byte_size;
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
];

const RAW_VIDEO_STREAM_NAMES: &[(&str, RawVideoFormat)] = &[
    ("raw-i420", RawVideoFormat::I420),
    ("raw-yuy2", RawVideoFormat::Yuyv),
    ("raw-uyvy", RawVideoFormat::Uyvy),
];

const MAX_DIM: u32 = 65_535;

pub(crate) fn raw_video_format(fourcc: [u8; 4]) -> Option<RawVideoFormat> {
    RAW_VIDEO_FOURCCS
        .iter()
        .find(|(code, _)| code.eq_ignore_ascii_case(&fourcc))
        .map(|(_, format)| *format)
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

// none of these layouts pads its rows, so bytes past the pixels are ignored, as libavcodec does
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
