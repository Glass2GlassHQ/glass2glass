//! The pixel formats a video capture source hands downstream, and the `Caps`
//! each one produces. Shared by the capture sources (`V4l2Src`,
//! `LibCameraSrc`), which sit on different fourcc registries (V4L2 pixel
//! formats vs libcamera's DRM-based ones) but agree on what a format means once
//! it is on a link.

use g2g_core::{Caps, Dim, Interlace, Rate, RawVideoFormat, VideoCodec};

/// One output format a capture source can negotiate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CapturePixelFormat {
    /// Planar 4:2:0, Y plane then interleaved UV. Decoder / ML / GPU friendly.
    Nv12,
    /// Packed 4:2:2, the near-universal raw UVC output.
    Yuyv,
    /// Planar 4:2:0 with separate U and V planes.
    I420,
    /// Baseline JPEG per frame, compressed on the camera. Fits resolutions and
    /// frame rates over USB that uncompressed raw cannot; `MjpegDec` decodes it
    /// downstream.
    Mjpeg,
}

impl CapturePixelFormat {
    /// Byte size of one frame at this geometry, for the formats whose size is
    /// fixed. `None` for MJPEG, whose length varies per frame: use the driver's
    /// per-buffer byte count there. The geometry comes from the device, so the
    /// arithmetic is checked.
    pub fn frame_bytes(self, width: u32, height: u32) -> Option<usize> {
        self.raw_format()?
            .unpadded_frame_bytes(width, height)?
            .try_into()
            .ok()
    }

    /// The caps this format produces at a negotiated geometry and rate. Raw
    /// layouts map to [`Caps::RawVideo`], MJPEG to `CompressedVideo{Mjpeg}`.
    pub fn caps(self, width: u32, height: u32, fps: u32) -> Caps {
        self.caps_with_dims(
            Dim::Fixed(width),
            Dim::Fixed(height),
            Rate::Fixed(fps << 16),
        )
    }

    /// The same mapping over caps-level geometry, so a device listing can carry
    /// the ranges a driver reports instead of one fixed mode.
    pub fn caps_with_dims(self, width: Dim, height: Dim, framerate: Rate) -> Caps {
        match self.raw_format() {
            Some(format) => Caps::RawVideo {
                format,
                width,
                height,
                framerate,
                interlace: Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            },
            None => Caps::CompressedVideo {
                codec: VideoCodec::Mjpeg,
                width,
                height,
                framerate,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            },
        }
    }

    /// The raw-video format on the link, `None` for a compressed output.
    pub fn raw_format(self) -> Option<RawVideoFormat> {
        match self {
            CapturePixelFormat::Nv12 => Some(RawVideoFormat::Nv12),
            CapturePixelFormat::Yuyv => Some(RawVideoFormat::Yuyv),
            CapturePixelFormat::I420 => Some(RawVideoFormat::I420),
            CapturePixelFormat::Mjpeg => None,
        }
    }

    /// The format that produces `caps`, so a source can read back which of its
    /// advertised alternatives the solver settled on. `None` for caps no
    /// capture format produces.
    pub fn from_caps(caps: &Caps) -> Option<Self> {
        match caps {
            Caps::RawVideo { format, .. } => match format {
                RawVideoFormat::Nv12 => Some(CapturePixelFormat::Nv12),
                RawVideoFormat::Yuyv => Some(CapturePixelFormat::Yuyv),
                RawVideoFormat::I420 => Some(CapturePixelFormat::I420),
                _ => None,
            },
            Caps::CompressedVideo {
                codec: VideoCodec::Mjpeg,
                ..
            } => Some(CapturePixelFormat::Mjpeg),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_bytes_match_the_packed_layouts() {
        assert_eq!(
            CapturePixelFormat::Yuyv.frame_bytes(640, 480),
            Some(614_400)
        );
        assert_eq!(
            CapturePixelFormat::Nv12.frame_bytes(640, 480),
            Some(460_800)
        );
        assert_eq!(
            CapturePixelFormat::I420.frame_bytes(640, 480),
            Some(460_800)
        );
        // odd geometry rounds the chroma planes up, never down (a truncated
        // plane would make the frame look short and be dropped).
        assert_eq!(CapturePixelFormat::Nv12.frame_bytes(3, 3), Some(9 + 8));
        // MJPEG has no fixed size.
        assert_eq!(CapturePixelFormat::Mjpeg.frame_bytes(640, 480), None);
        // a geometry that would overflow the byte count fails instead of wrapping.
        assert_eq!(
            CapturePixelFormat::Yuyv.frame_bytes(u32::MAX, u32::MAX),
            None
        );
    }

    #[test]
    fn caps_round_trip_through_from_caps() {
        for format in [
            CapturePixelFormat::Nv12,
            CapturePixelFormat::Yuyv,
            CapturePixelFormat::I420,
            CapturePixelFormat::Mjpeg,
        ] {
            let caps = format.caps(1280, 720, 30);
            assert_eq!(CapturePixelFormat::from_caps(&caps), Some(format));
        }
        assert_eq!(
            CapturePixelFormat::Mjpeg.caps(1280, 720, 30),
            Caps::CompressedVideo {
                codec: VideoCodec::Mjpeg,
                width: Dim::Fixed(1280),
                height: Dim::Fixed(720),
                framerate: Rate::Fixed(30 << 16),
                colorimetry: g2g_core::Colorimetry::UNKNOWN
            }
        );
    }
}
