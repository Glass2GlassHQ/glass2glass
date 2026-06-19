//! URI-scheme handlers for [`Registry::build_uridecodebin`], the `uridecodebin`
//! front door (M92). Each maps a URI scheme to one of the concrete g2g sources
//! and is gated to that source's feature, so an app registers only the schemes
//! its build supports:
//!
//! ```ignore
//! use g2g_core::runtime::{is_raw_video, Registry, ElementFactory};
//! use g2g_plugins::uridecodebin;
//!
//! let mut reg = Registry::new();
//! reg.register_uri(uridecodebin::udp_handler())          // udp://host:port
//!    .register_uri(uridecodebin::file_handler())         // file:///clip.mp4
//!    .register(ElementFactory::of::<FfmpegH264Dec>("h264dec", |_| Box::new(FfmpegH264Dec::new())));
//! let graph = reg.build_uridecodebin("udp://0.0.0.0:5004", sink, &is_raw_video, 4)?;
//! ```
//!
//! The handler builds the source *from the URI* (parsing host:port / path),
//! reports the media type it produces, and the registry auto-plugs the decode
//! chain down to the target. Geometry is resolved at runtime negotiation, so a
//! handler's declared caps only name the media type the decoder is plugged for.

use alloc::boxed::Box;

use g2g_core::runtime::{DynSourceLoop, Uri, UriError, UriSourceFactory};
use g2g_core::{Caps, Dim, Rate, VideoCodec};

/// H.264 at any geometry: the media type the H.264 sources declare. Real
/// dimensions ride in-band in the SPS and are resolved at negotiation.
#[cfg(any(feature = "udp-ingress", feature = "rtsp", feature = "std"))]
fn h264_any() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

/// `udp://host:port` -> [`UdpSrc`](crate::udpsrc::UdpSrc): raw RTP H.264 ingest.
#[cfg(feature = "udp-ingress")]
pub fn udp_handler() -> UriSourceFactory {
    UriSourceFactory::new("udp", |uri: &Uri| {
        // `rest` is the bare authority `host:port` for udp://.
        let addr = uri.rest.parse().map_err(|_| UriError::Malformed)?;
        let src = crate::udpsrc::UdpSrc::new(addr);
        Ok((Box::new(src) as Box<dyn DynSourceLoop>, h264_any()))
    })
}

/// `rtsp://...` -> [`RtspSrc`](crate::rtspsrc::RtspSrc): the full URI is handed
/// to retina, which parses it.
#[cfg(feature = "rtsp")]
pub fn rtsp_handler() -> UriSourceFactory {
    UriSourceFactory::new("rtsp", |uri: &Uri| {
        let src = crate::rtspsrc::RtspSrc::new(uri.raw);
        Ok((Box::new(src) as Box<dyn DynSourceLoop>, h264_any()))
    })
}

/// `file:///path.mp4` -> [`Mp4Src`](crate::mp4src::Mp4Src): demuxes an MP4
/// file's H.264 track. `rest` is the absolute path (the `file://` authority is
/// empty, so `file:///a/b` leaves `/a/b`).
#[cfg(feature = "std")]
pub fn file_handler() -> UriSourceFactory {
    UriSourceFactory::new("file", |uri: &Uri| {
        if uri.rest.is_empty() {
            return Err(UriError::Malformed);
        }
        let src = crate::mp4src::Mp4Src::new(uri.rest);
        Ok((Box::new(src) as Box<dyn DynSourceLoop>, h264_any()))
    })
}

/// `v4l2:///dev/videoN` -> [`V4l2Src`](crate::v4l2src::V4l2Src): YUYV capture.
/// `rest` is the device path.
#[cfg(all(target_os = "linux", feature = "v4l2"))]
pub fn v4l2_handler() -> UriSourceFactory {
    UriSourceFactory::new("v4l2", |uri: &Uri| {
        if uri.rest.is_empty() {
            return Err(UriError::Malformed);
        }
        let src = crate::v4l2src::V4l2Src::new(uri.rest);
        Ok((
            Box::new(src) as Box<dyn DynSourceLoop>,
            Caps::RawVideo {
                format: g2g_core::RawVideoFormat::Yuyv,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
            },
        ))
    })
}
