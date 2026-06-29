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

/// `hls://host/path.m3u8` -> [`HlsSrc`](crate::hlssrc::HlsSrc): single-stream
/// HLS playback. The `hls` scheme maps to an `https` origin for fetching; the
/// declared media type is `ByteStream{MpegTs}` (the dominant HLS packaging), so
/// the registry auto-plugs `tsdemux -> decode`. This is the fallback the
/// [`hls_playbin`] fan-out hook defers to (a media-only playlist, an fMP4 variant,
/// or a network failure during the master probe).
#[cfg(feature = "hls")]
pub fn hls_handler() -> UriSourceFactory {
    UriSourceFactory::new("hls", |uri: &Uri| {
        if uri.rest.is_empty() {
            return Err(UriError::Malformed);
        }
        let url = alloc::format!("https://{}", uri.rest);
        let src = crate::hlssrc::HlsSrc::new(url);
        Ok((
            Box::new(src) as Box<dyn DynSourceLoop>,
            Caps::ByteStream { encoding: g2g_core::ByteStreamEncoding::MpegTs },
        ))
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
#[cfg(feature = "hls")]
use g2g_core::AudioFormat;

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

/// The `playbin uri=X` auto-fan-out hook for fragmented MP4 / CMAF (M392): probe
/// a `file://` ISO-BMFF container's `moov`, then assemble `FileSrc -> Mp4DemuxN ->
/// {decode -> auto sink}` with one branch per forwardable track, the MP4 sibling
/// of [`mkv_playbin`] / [`ts_playbin`]. Declines (`Ok(None)`) for a non-`file://`
/// URI, an unreadable file, or a container whose `moov` is not in the probed
/// prefix (a non-MP4, or a progressive file whose `moov` trails the data, which
/// stays on the single-stream `file://` -> `Mp4Src` path).
#[cfg(feature = "std")]
pub fn mp4_playbin(reg: &Registry, uri: &str) -> Result<Option<Graph<GraphNode>>, ParseError> {
    let Some((path, prefix)) = open_file_prefix(uri) else { return Ok(None) };
    let infos = crate::mp4demuxn::forwardable_streams(&prefix);
    if infos.is_empty() {
        return Ok(None); // not MP4 (or moov not in the prefix): decline
    }
    let ports = playbin_ports(reg, infos.iter().map(|i| (i.caps.clone(), i.video)))?;
    let demux_ports: Vec<_> = infos
        .iter()
        .map(|i| crate::mp4demuxn::Mp4Port { track_id: i.track_id, caps: i.caps.clone() })
        .collect();
    let source =
        crate::filesrc::FileSrc::new(&path, Caps::ByteStream { encoding: ByteStreamEncoding::IsoBmff });
    reg.build_playbin_graph_with_source(
        Box::new(source),
        crate::mp4demuxn::Mp4DemuxN::new(demux_ports),
        ports,
        PLAYBIN_MAX_DEPTH,
    )
    .map(Some)
    .map_err(|e| map_playbin_err(uri, e))
}

/// Fetch a text resource synchronously on a throwaway current-thread runtime, for
/// the `playbin uri=hls://...` probe (M395). Returns `None` on any failure, so the
/// hook declines to the single-stream `hls` handler. Network-coupled: validated
/// against a live server, not in CI (like the RTSP / WHIP paths).
#[cfg(feature = "hls")]
fn blocking_get_text(url: &str) -> Option<alloc::string::String> {
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().ok()?;
    let client = reqwest::Client::new();
    rt.block_on(crate::fetch::get_text(&client, url, crate::fetch::MAX_MANIFEST_BYTES)).ok()
}

/// The MPEG-TS elementary-stream selector for an HLS discovered stream, or `None`
/// for a codec `TsDemuxN` cannot route (only H.264 / H.265 video + AAC audio).
#[cfg(feature = "hls")]
fn hls_ts_stream(info: &crate::hlssrc::HlsStreamInfo) -> Option<crate::tsdemux::TsStream> {
    use crate::tsdemux::TsStream;
    match &info.caps {
        Caps::CompressedVideo { codec: VideoCodec::H264, .. } => Some(TsStream::H264),
        Caps::CompressedVideo { codec: VideoCodec::H265, .. } => Some(TsStream::H265),
        Caps::Audio { format: AudioFormat::Aac, .. } => Some(TsStream::Aac),
        _ => None,
    }
}

/// Assemble the `HlsSrc -> TsDemuxN -> {decode -> auto sink}` fan-out from a
/// master variant's discovered streams (M395). The network-free core of
/// [`hls_playbin`], so it is unit-testable: only the *muxed* streams (carried in
/// the variant's own TS segments, `uri == None`) fan out through one demuxer, one
/// `TsStream` port per routable codec. Returns `Ok(None)` (decline, the
/// single-stream handler takes over) when fewer than two routable muxed streams
/// remain, e.g. a single-rendition or separate-audio variant.
#[cfg(feature = "hls")]
pub fn build_hls_ts_fanout(
    reg: &Registry,
    source_url: &str,
    streams: &[crate::hlssrc::HlsStreamInfo],
) -> Result<Option<Graph<GraphNode>>, ParseError> {
    let mut ts_streams = Vec::new();
    let mut port_infos = Vec::new();
    for s in streams.iter().filter(|s| s.uri.is_none()) {
        if let Some(ts) = hls_ts_stream(s) {
            ts_streams.push(ts);
            port_infos.push((s.caps.clone(), s.video));
        }
    }
    if ts_streams.len() < 2 {
        return Ok(None); // not a multi-stream muxed TS variant
    }
    let ports = playbin_ports(reg, port_infos.into_iter())?;
    let source = crate::hlssrc::HlsSrc::new(source_url);
    reg.build_playbin_graph_with_source(
        Box::new(source),
        crate::tsdemux::TsDemuxN::new(ts_streams),
        ports,
        PLAYBIN_MAX_DEPTH,
    )
    .map(Some)
    .map_err(|e| map_playbin_err(source_url, e))
}

/// The `playbin uri=X` auto-fan-out hook for HLS (M395): probe a `hls://` master
/// playlist, discover its selected variant's renditions, and assemble `HlsSrc ->
/// TsDemuxN -> {decode -> auto sink}`, one branch per muxed elementary stream, the
/// HLS sibling of [`mkv_playbin`] / [`ts_playbin`] / [`mp4_playbin`]. The `hls`
/// scheme maps to an `https` origin.
///
/// Declines (`Ok(None)`, so the single-stream [`hls_handler`] takes over) for a
/// non-`hls` URI, a master-probe network failure, a media-only playlist, an fMP4
/// (non-TS) variant, or a variant without at least two routable muxed streams.
/// Network-coupled: the probe is validated against a live server, not in CI.
#[cfg(feature = "hls")]
pub fn hls_playbin(reg: &Registry, uri: &str) -> Result<Option<Graph<GraphNode>>, ParseError> {
    let Some(parsed) = Uri::parse(uri) else { return Ok(None) };
    if parsed.scheme != "hls" || parsed.rest.is_empty() {
        return Ok(None);
    }
    let source_url = alloc::format!("https://{}", parsed.rest);
    // Fetch + parse the master; decline (single-stream fallback) on any failure.
    let Some(master_text) = blocking_get_text(&source_url) else { return Ok(None) };
    let Ok(crate::hls::Playlist::Master(master)) = crate::hls::parse(&master_text) else {
        return Ok(None); // a media playlist or parse error: single-stream handles it
    };
    let Some(variant) = master.select(None) else { return Ok(None) };
    // The variant's media playlist tells the container; only muxed TS fans out
    // here (an fMP4 init segment's track ids are not known without fetching it).
    let media_url = crate::fetch::resolve_url(&source_url, &variant.uri);
    let Some(media_text) = blocking_get_text(&media_url) else { return Ok(None) };
    match crate::hls::parse(&media_text) {
        Ok(crate::hls::Playlist::Media(m)) if m.map_uri.is_none() => {}
        _ => return Ok(None), // fMP4 (EXT-X-MAP) or unexpected: decline
    }
    let streams = crate::hlssrc::variant_streams(&master, variant);
    build_hls_ts_fanout(reg, &source_url, &streams)
}
