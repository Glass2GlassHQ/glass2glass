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

/// Bytes read to probe the container's `Tracks` for the `playbin` auto-fan-out.
/// `Tracks` sits near the front of a Matroska Segment, so a few MiB is ample.
#[cfg(feature = "std")]
const PLAYBIN_PROBE_BYTES: u64 = 4 * 1024 * 1024;

/// Per-port decode-chain search depth, matching the `decodebin` / `playbin` macro.
#[cfg(feature = "std")]
const PLAYBIN_MAX_DEPTH: usize = 6;

/// The `playbin uri=X` auto-fan-out hook (M382): probe a `file://` Matroska
/// container, then assemble `FileSrc -> MkvDemuxN -> {decode -> auto sink}` with
/// one branch per forwardable stream, the multi-stream form of `playbin` (vs the
/// M196 single-stream expansion). Register it on a [`Registry`] with
/// [`Registry::register_playbin`] (done by [`default_registry`]).
///
/// Declines (returns `Ok(None)`, so the parser falls back to single-stream
/// `playbin`) for a non-`file://` URI, an unreadable file, or a file whose
/// `Tracks` carries no Matroska stream g2g forwards (e.g. an MP4). A probed
/// Matroska whose per-port decode chain cannot be plugged (no decoder feature
/// compiled in) is an error, not a decline.
///
/// [`default_registry`]: crate::registry::default_registry
#[cfg(feature = "std")]
pub fn mkv_playbin(
    reg: &g2g_core::runtime::Registry,
    uri: &str,
) -> Result<Option<g2g_core::Graph<g2g_core::runtime::GraphNode>>, g2g_core::runtime::ParseError> {
    use std::io::Read;

    use alloc::string::ToString;
    use alloc::vec::Vec;

    use g2g_core::runtime::{
        is_raw_audio, is_raw_video, ParseError, PlaybinGraphError, PlaybinPort,
    };
    use g2g_core::{ByteStreamEncoding, Caps};

    use crate::filesrc::FileSrc;
    use crate::matroska::MatroskaDemuxer;
    use crate::mkvdemux::{forwardable_streams, MkvDemuxN};

    // Only file:// is probed; other schemes fall through to single-stream playbin.
    let parsed = match Uri::parse(uri) {
        Some(u) if u.scheme == "file" && !u.rest.is_empty() => u,
        _ => return Ok(None),
    };

    // Read a bounded prefix to parse the Tracks element. An unreadable file
    // declines (the single-stream path reports its own error).
    let mut prefix = Vec::new();
    match std::fs::File::open(parsed.rest) {
        Ok(f) => {
            if f.take(PLAYBIN_PROBE_BYTES).read_to_end(&mut prefix).is_err() {
                return Ok(None);
            }
        }
        Err(_) => return Ok(None),
    }

    let mut demux = MatroskaDemuxer::new();
    demux.push_data(&prefix);
    let infos = forwardable_streams(&demux);
    if infos.is_empty() {
        // Not a Matroska container (or no forwardable track): decline so the
        // single-stream playbin (its own scheme handler) takes over.
        return Ok(None);
    }

    let streams = infos.iter().map(|i| i.stream).collect::<Vec<_>>();
    let mut ports = Vec::with_capacity(infos.len());
    for info in &infos {
        let sink_name = if info.video { "autovideosink" } else { "autoaudiosink" };
        let target: Box<dyn Fn(&Caps) -> bool> =
            if info.video { Box::new(is_raw_video) } else { Box::new(is_raw_audio) };
        let sink = reg
            .make_element(sink_name)
            .ok_or_else(|| ParseError::UnknownElement(sink_name.to_string()))?;
        ports.push(PlaybinPort { input_caps: info.caps.clone(), target, sink });
    }

    // The URI handler's file:// source self-demuxes MP4, so build the matching
    // Matroska byte source directly and feed it to the multi-output demuxer.
    let source = FileSrc::new(parsed.rest, Caps::ByteStream { encoding: ByteStreamEncoding::Matroska });
    let demux_n = MkvDemuxN::new(streams);
    reg.build_playbin_graph_with_source(Box::new(source), demux_n, ports, PLAYBIN_MAX_DEPTH)
        .map(Some)
        .map_err(|e| match e {
            PlaybinGraphError::NoPorts => {
                ParseError::NoDecodeChain("playbin: no forwardable streams".to_string())
            }
            PlaybinGraphError::Uri(e) => ParseError::Uri(alloc::format!("{uri}: {e:?}")),
            PlaybinGraphError::Graph(e) => ParseError::Graph(e),
            PlaybinGraphError::Decode(e) => ParseError::NoDecodeChain(alloc::format!("{e:?}")),
        })
}
