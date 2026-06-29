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

#[cfg(feature = "std")]
use alloc::string::ToString;
#[cfg(feature = "std")]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use g2g_core::runtime::{
    is_raw_audio, is_raw_video, GraphNode, ParseError, PlaybinGraphError, PlaybinPort, Registry,
};
#[cfg(feature = "std")]
use g2g_core::{ByteStreamEncoding, Graph};

/// file:// probe: read a bounded prefix of the URI's file for container sniffing.
/// `None` for a non-`file://` URI or an unreadable file (the hook then declines),
/// else the absolute path (for the byte source) plus the prefix bytes.
#[cfg(feature = "std")]
fn open_file_prefix(uri: &str) -> Option<(alloc::string::String, Vec<u8>)> {
    use std::io::Read;
    let parsed = Uri::parse(uri)?;
    if parsed.scheme != "file" || parsed.rest.is_empty() {
        return None;
    }
    let mut prefix = Vec::new();
    let f = std::fs::File::open(parsed.rest).ok()?;
    f.take(PLAYBIN_PROBE_BYTES).read_to_end(&mut prefix).ok()?;
    Some((parsed.rest.to_string(), prefix))
}

/// Build one [`PlaybinPort`] per `(elementary caps, is_video)`: an `autovideosink`
/// for a video stream, an `autoaudiosink` for audio, each with the matching
/// raw-shape `target`. Shared by every container's playbin hook.
#[cfg(feature = "std")]
fn playbin_ports(
    reg: &Registry,
    infos: impl Iterator<Item = (Caps, bool)>,
) -> Result<Vec<PlaybinPort>, ParseError> {
    let mut ports = Vec::new();
    for (caps, video) in infos {
        let sink_name = if video { "autovideosink" } else { "autoaudiosink" };
        let target: Box<dyn Fn(&Caps) -> bool> =
            if video { Box::new(is_raw_video) } else { Box::new(is_raw_audio) };
        let sink = reg
            .make_element(sink_name)
            .ok_or_else(|| ParseError::UnknownElement(sink_name.to_string()))?;
        ports.push(PlaybinPort { input_caps: caps, target, sink });
    }
    Ok(ports)
}

/// Map a multi-stream graph-build failure to the text-parser error.
#[cfg(feature = "std")]
fn map_playbin_err(uri: &str, e: PlaybinGraphError) -> ParseError {
    match e {
        PlaybinGraphError::NoPorts => {
            ParseError::NoDecodeChain("playbin: no forwardable streams".to_string())
        }
        PlaybinGraphError::Uri(e) => ParseError::Uri(alloc::format!("{uri}: {e:?}")),
        PlaybinGraphError::Graph(e) => ParseError::Graph(e),
        PlaybinGraphError::Decode(e) => ParseError::NoDecodeChain(alloc::format!("{e:?}")),
    }
}

/// The `playbin uri=X` auto-fan-out hook for Matroska / WebM (M382): probe a
/// `file://` MKV container, then assemble `FileSrc -> MkvDemuxN -> {decode -> auto
/// sink}` with one branch per forwardable stream, the multi-stream form of
/// `playbin`. Register it with [`Registry::register_playbin`] (done by
/// [`default_registry`]).
///
/// Declines (`Ok(None)`, so the parser tries the next hook / falls back to
/// single-stream `playbin`) for a non-`file://` URI, an unreadable file, or a
/// non-Matroska container. A probed MKV whose per-port decode chain cannot be
/// plugged (no decoder feature compiled in) is an error, not a decline.
///
/// [`default_registry`]: crate::registry::default_registry
#[cfg(feature = "std")]
pub fn mkv_playbin(reg: &Registry, uri: &str) -> Result<Option<Graph<GraphNode>>, ParseError> {
    let Some((path, prefix)) = open_file_prefix(uri) else { return Ok(None) };
    let mut demux = crate::matroska::MatroskaDemuxer::new();
    demux.push_data(&prefix);
    let infos = crate::mkvdemux::forwardable_streams(&demux);
    if infos.is_empty() {
        return Ok(None); // not Matroska (or no forwardable track): decline
    }
    let streams: Vec<_> = infos.iter().map(|i| i.stream).collect();
    let ports = playbin_ports(reg, infos.iter().map(|i| (i.caps.clone(), i.video)))?;
    // The file:// URI handler self-demuxes MP4, so build the matching Matroska
    // byte source directly and feed it to the multi-output demuxer.
    let source = crate::filesrc::FileSrc::new(&path, Caps::ByteStream { encoding: ByteStreamEncoding::Matroska });
    reg.build_playbin_graph_with_source(
        Box::new(source),
        crate::mkvdemux::MkvDemuxN::new(streams),
        ports,
        PLAYBIN_MAX_DEPTH,
    )
    .map(Some)
    .map_err(|e| map_playbin_err(uri, e))
}

/// The `playbin uri=X` auto-fan-out hook for MPEG-TS (M389): probe a `file://`
/// transport stream's PMT, then assemble `FileSrc -> TsDemuxN -> {decode -> auto
/// sink}` with one branch per forwardable stream, the MPEG-TS sibling of
/// [`mkv_playbin`]. Declines (`Ok(None)`) for a non-`file://` URI, an unreadable
/// file, or a non-MPEG-TS container (no PMT in the probed prefix).
#[cfg(feature = "std")]
pub fn ts_playbin(reg: &Registry, uri: &str) -> Result<Option<Graph<GraphNode>>, ParseError> {
    let Some((path, prefix)) = open_file_prefix(uri) else { return Ok(None) };
    // Resync to the TS sync byte and feed whole 188-byte packets to parse the PMT.
    let mut demux = crate::mpegts::TsDemuxer::new();
    let mut off = 0;
    while off + 188 <= prefix.len() {
        if prefix[off] != 0x47 {
            off += 1;
            continue;
        }
        demux.push_packet(&prefix[off..off + 188]);
        off += 188;
    }
    let infos = crate::tsdemux::forwardable_streams(&demux);
    if infos.is_empty() {
        return Ok(None); // not MPEG-TS (or no PMT yet): decline
    }
    let streams: Vec<_> = infos.iter().map(|i| i.stream).collect();
    let ports = playbin_ports(reg, infos.iter().map(|i| (i.caps.clone(), i.video)))?;
    let source = crate::filesrc::FileSrc::new(&path, Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs });
    reg.build_playbin_graph_with_source(
        Box::new(source),
        crate::tsdemux::TsDemuxN::new(streams),
        ports,
        PLAYBIN_MAX_DEPTH,
    )
    .map(Some)
    .map_err(|e| map_playbin_err(uri, e))
}
