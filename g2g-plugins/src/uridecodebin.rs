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
/// [`hls_playbin`] fan-out hook defers to (a media-only playlist, a single-stream
/// variant, or a network failure during the master probe).
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
    is_raw_audio, is_raw_video, DecodebinError, GraphNode, GraphNodeRef, ParseError,
    PlaybinGraphError, PlaybinPort, Registry,
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
    // A subtitle track + a video track: overlay the subtitles onto the video (the
    // MP4 M412 sibling, M415).
    let subs = crate::mkvdemux::subtitle_streams(&demux);
    if infos.iter().any(|i| i.video) {
        if let Some(text) = subs.first() {
            return build_mkv_subtitle_overlay(reg, &path, &infos, text).map(Some);
        }
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

/// Map a per-branch decode-chain failure to the text-parser error (the
/// [`decodebin`](Registry::decodebin) form, vs [`map_playbin_err`]'s whole-graph
/// form): no chain quotes the unplugged input caps, a link error wraps the graph
/// error.
#[cfg(feature = "std")]
fn map_decode_err(input: &Caps, e: DecodebinError) -> ParseError {
    match e {
        DecodebinError::NoChain => ParseError::NoDecodeChain(alloc::format!("{input:?}")),
        DecodebinError::Graph(e) => ParseError::Graph(e),
    }
}

/// Splice the subtitle-overlay subgraph onto a built demux, shared by every
/// container's `playbin` hook (MP4 M412, MKV / TS M415). Given a `demux` whose
/// leading ports are the A/V tracks `av` (`(elementary caps, is_video)` in port
/// order, video at `video_idx`) and whose trailing `text_port` carries the chosen
/// subtitle track (caps `text_caps`): the video track decodes, converts to RGBA8,
/// and feeds a [`TextOverlayN`](crate::textoverlay::TextOverlayN) whose text pad is
/// fed by the subtitle track (linked straight in for a plain-UTF8 cue stream, via
/// `SubParse` for a structured cue format); the overlay output converts back to
/// NV12 for the auto video sink, and the other A/V tracks (audio) fan out to their
/// own auto sinks. The decoder is auto-plugged (codec-agnostic), but the
/// `videoconvert`s around the overlay are wired explicitly: they are caps-driven
/// `register_launch` elements outside the auto-plug pool, and the overlay requires
/// RGBA8 in / out while the display sink requires NV12.
#[cfg(feature = "std")]
fn wire_subtitle_overlay(
    reg: &Registry,
    graph: &mut Graph<GraphNode>,
    demux: g2g_core::graph::Demux,
    av: &[(Caps, bool)],
    video_idx: usize,
    text_port: u8,
    text_caps: &Caps,
) -> Result<(), ParseError> {
    use g2g_core::RawVideoFormat;

    // The overlay and the RGBA8 / NV12 converts bracketing it.
    let overlay = graph.add_muxer(GraphNodeRef::muxer(crate::textoverlay::TextOverlayN::new()), 2);
    let to_rgba =
        graph.add_transform(GraphNodeRef::element(crate::videoconvert::VideoConvert::new(RawVideoFormat::Rgba8)));
    let to_nv12 =
        graph.add_transform(GraphNodeRef::element(crate::videoconvert::VideoConvert::new(RawVideoFormat::Nv12)));
    graph.link(to_rgba, overlay.input(0)).map_err(ParseError::Graph)?;
    graph.link(overlay.output(), to_nv12).map_err(ParseError::Graph)?;
    let vsink = reg
        .make_element("autovideosink")
        .ok_or_else(|| ParseError::UnknownElement("autovideosink".to_string()))?;
    let vsnk = graph.add_sink(GraphNodeRef::Element(vsink));
    graph.link(to_nv12, vsnk).map_err(ParseError::Graph)?;

    // The subtitle track feeds the overlay's text pad. A plain-UTF8 cue stream
    // (MP4 `tx3g`, MKV `S_TEXT/UTF8`) links straight in; a structured format
    // (SRT / WebVTT / SSA / TTML) parses to timed UTF-8 cues via SubParse first.
    let text_in: g2g_core::graph::PadId = match text_caps {
        Caps::Text { format: g2g_core::TextFormat::Utf8 } => overlay.input(1),
        _ => {
            let sub = graph.add_transform(GraphNodeRef::element(crate::subparse::SubParse::new()));
            graph.link(sub, overlay.input(1)).map_err(ParseError::Graph)?;
            sub.into()
        }
    };
    graph.link(demux.out(text_port), text_in).map_err(ParseError::Graph)?;

    // Each A/V track: the video one decodes into the overlay's RGBA8 convert, the
    // rest fan out to their own auto sinks.
    for (i, (caps, _video)) in av.iter().enumerate() {
        if i == video_idx {
            reg.decodebin(graph, demux.out(i as u8), to_rgba, caps, &is_raw_video, PLAYBIN_MAX_DEPTH)
                .map_err(|e| map_decode_err(caps, e))?;
        } else {
            let sink = reg
                .make_element("autoaudiosink")
                .ok_or_else(|| ParseError::UnknownElement("autoaudiosink".to_string()))?;
            let snk = graph.add_sink(GraphNodeRef::Element(sink));
            reg.decodebin(graph, demux.out(i as u8), snk, caps, &is_raw_audio, PLAYBIN_MAX_DEPTH)
                .map_err(|e| map_decode_err(caps, e))?;
        }
    }
    Ok(())
}

/// Assemble the subtitle-overlay MP4 graph (M412): `FileSrc -> Mp4DemuxN -> {video:
/// decode -> overlay; text: -> overlay} -> sink`. `av` carries the A/V tracks (in
/// `moov` order, the demux's leading ports) and `text` the chosen subtitle track
/// (the trailing port); `av` must contain a video track. The overlay wiring is the
/// shared [`wire_subtitle_overlay`].
#[cfg(feature = "std")]
fn build_mp4_subtitle_overlay(
    reg: &Registry,
    path: &str,
    av: &[crate::mp4demuxn::Mp4StreamInfo],
    text: &crate::mp4demuxn::Mp4StreamInfo,
) -> Result<Graph<GraphNode>, ParseError> {
    use crate::mp4demuxn::{Mp4DemuxN, Mp4Port};

    let video_idx = av.iter().position(|i| i.video).ok_or_else(|| {
        ParseError::NoDecodeChain("subtitle overlay needs a video track".into())
    })?;

    // Demux ports: every A/V track (in moov order) then the subtitle track.
    let mut demux_ports: Vec<Mp4Port> =
        av.iter().map(|i| Mp4Port { track_id: i.track_id, caps: i.caps.clone() }).collect();
    demux_ports.push(Mp4Port { track_id: text.track_id, caps: text.caps.clone() });
    let text_port = (demux_ports.len() - 1) as u8;
    let outputs = demux_ports.len() as u8;

    let mut graph: Graph<GraphNode> = Graph::new();
    let source =
        crate::filesrc::FileSrc::new(path, Caps::ByteStream { encoding: ByteStreamEncoding::IsoBmff });
    let src = graph.add_source(GraphNodeRef::source(source));
    let demux = graph.add_demux(GraphNodeRef::demux(Mp4DemuxN::new(demux_ports)), outputs);
    graph.link(src, demux.input()).map_err(ParseError::Graph)?;

    let av_caps: Vec<(Caps, bool)> = av.iter().map(|i| (i.caps.clone(), i.video)).collect();
    wire_subtitle_overlay(reg, &mut graph, demux, &av_caps, video_idx, text_port, &text.caps)?;
    Ok(graph)
}

/// Assemble the subtitle-overlay Matroska graph (M415): the MKV sibling of
/// [`build_mp4_subtitle_overlay`]. `FileSrc -> MkvDemuxN -> {video: decode ->
/// overlay; text: -> overlay} -> sink`, with one trailing demux port for the chosen
/// subtitle track. `av` must contain a video track. Shares [`wire_subtitle_overlay`].
#[cfg(feature = "std")]
fn build_mkv_subtitle_overlay(
    reg: &Registry,
    path: &str,
    av: &[crate::mkvdemux::MkvStreamInfo],
    text: &crate::mkvdemux::MkvStreamInfo,
) -> Result<Graph<GraphNode>, ParseError> {
    use crate::mkvdemux::MkvDemuxN;

    let video_idx = av.iter().position(|i| i.video).ok_or_else(|| {
        ParseError::NoDecodeChain("subtitle overlay needs a video track".into())
    })?;

    // Demux ports (selected streams): every A/V stream (in track order) then the
    // subtitle stream.
    let mut stream_ports: Vec<_> = av.iter().map(|i| i.stream).collect();
    stream_ports.push(text.stream);
    let text_port = (stream_ports.len() - 1) as u8;
    let outputs = stream_ports.len() as u8;

    let mut graph: Graph<GraphNode> = Graph::new();
    let source =
        crate::filesrc::FileSrc::new(path, Caps::ByteStream { encoding: ByteStreamEncoding::Matroska });
    let src = graph.add_source(GraphNodeRef::source(source));
    let demux = graph.add_demux(GraphNodeRef::demux(MkvDemuxN::new(stream_ports)), outputs);
    graph.link(src, demux.input()).map_err(ParseError::Graph)?;

    let av_caps: Vec<(Caps, bool)> = av.iter().map(|i| (i.caps.clone(), i.video)).collect();
    wire_subtitle_overlay(reg, &mut graph, demux, &av_caps, video_idx, text_port, &text.caps)?;
    Ok(graph)
}

/// The `playbin uri=X` auto-fan-out hook for fragmented MP4 / CMAF (M392): probe
/// a `file://` ISO-BMFF container's `moov`, then assemble `FileSrc -> Mp4DemuxN ->
/// {decode -> auto sink}` with one branch per forwardable track, the MP4 sibling
/// of [`mkv_playbin`] / [`ts_playbin`]. When the file also carries a subtitle
/// track and a video track, the video branch routes through a `TextOverlayN` fed
/// by the subtitle track (M412, [`build_mp4_subtitle_overlay`]) so the subtitles
/// render on screen. Declines (`Ok(None)`) for a non-`file://` URI, an unreadable
/// file, or a container whose `moov` is not in the probed prefix (a non-MP4, or a
/// progressive file whose `moov` trails the data, which stays on the single-stream
/// `file://` -> `Mp4Src` path).
#[cfg(feature = "std")]
pub fn mp4_playbin(reg: &Registry, uri: &str) -> Result<Option<Graph<GraphNode>>, ParseError> {
    let Some((path, prefix)) = open_file_prefix(uri) else { return Ok(None) };
    let infos = crate::mp4demuxn::forwardable_streams(&prefix);
    if infos.is_empty() {
        return Ok(None); // not MP4 (or moov not in the prefix): decline
    }
    // A subtitle track + a video track: overlay the subtitles onto the video.
    let subs = crate::mp4demuxn::subtitle_streams(&prefix);
    if infos.iter().any(|i| i.video) {
        if let Some(text) = subs.first() {
            return build_mp4_subtitle_overlay(reg, &path, &infos, text).map(Some);
        }
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

/// Fetch a binary resource synchronously (the fMP4 `#EXT-X-MAP` init segment),
/// the byte sibling of [`blocking_get_text`]. `None` on any failure.
#[cfg(feature = "hls")]
fn blocking_get_bytes(url: &str) -> Option<Vec<u8>> {
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().ok()?;
    let client = reqwest::Client::new();
    rt.block_on(crate::fetch::get_bytes(&client, url, crate::fetch::MAX_SEGMENT_BYTES)).ok()
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

/// Assemble the `HlsSrc -> Mp4DemuxN -> {decode -> auto sink}` fan-out for an
/// fMP4 / CMAF HLS variant (M396), discovering its tracks from the `#EXT-X-MAP`
/// init segment's `moov` ([`mp4demuxn::forwardable_streams`]). The network-free
/// core of the fMP4 branch of [`hls_playbin`], unit-testable on an init segment.
/// `Ok(None)` (decline) when the init carries no forwardable track.
///
/// Note: `Mp4DemuxN` buffers the whole byte stream and parses on EOS, so this
/// fans out a VOD playlist after it ends rather than progressively, the batch
/// limitation of the file-shaped demuxer applied to the segment stream.
#[cfg(feature = "hls")]
pub fn build_hls_fmp4_fanout(
    reg: &Registry,
    source_url: &str,
    init: &[u8],
) -> Result<Option<Graph<GraphNode>>, ParseError> {
    let infos = crate::mp4demuxn::forwardable_streams(init);
    if infos.is_empty() {
        return Ok(None); // init segment carries no A/V track
    }
    let ports = playbin_ports(reg, infos.iter().map(|i| (i.caps.clone(), i.video)))?;
    let demux_ports: Vec<_> = infos
        .iter()
        .map(|i| crate::mp4demuxn::Mp4Port { track_id: i.track_id, caps: i.caps.clone() })
        .collect();
    let source = crate::hlssrc::HlsSrc::new(source_url);
    reg.build_playbin_graph_with_source(
        Box::new(source),
        crate::mp4demuxn::Mp4DemuxN::new(demux_ports),
        ports,
        PLAYBIN_MAX_DEPTH,
    )
    .map(Some)
    .map_err(|e| map_playbin_err(source_url, e))
}

/// Assemble a *multi-source* HLS fan-out for a variant with a separate audio
/// rendition (M397): the video rides the variant's own (video-only) TS segments
/// while the audio is a distinct `#EXT-X-MEDIA` rendition playlist, so two
/// independent `HlsSrc -> TsDemuxN -> decode -> sink` chains are built and merged
/// into one graph (each rendition is its own source, unlike the single-demuxer
/// muxed fan-out). `streams` supplies the muxed video stream; `audio_url` is the
/// resolved rendition playlist. `Ok(None)` (decline) when there is no routable
/// muxed video stream. Network-free assembly (the two sources probe at run).
#[cfg(feature = "hls")]
pub fn build_hls_separate_fanout(
    reg: &Registry,
    master_url: &str,
    streams: &[crate::hlssrc::HlsStreamInfo],
    audio_url: &str,
) -> Result<Option<Graph<GraphNode>>, ParseError> {
    use crate::tsdemux::{TsDemuxN, TsStream};
    // The variant's own (video-only) TS stream.
    let Some(video) = streams.iter().find(|s| s.video && s.uri.is_none()) else {
        return Ok(None);
    };
    let Some(vts) = hls_ts_stream(video) else { return Ok(None) };
    let video_ports = playbin_ports(reg, core::iter::once((video.caps.clone(), true)))?;
    let mut graph = reg
        .build_playbin_graph_with_source(
            Box::new(crate::hlssrc::HlsSrc::new(master_url)),
            TsDemuxN::new(Vec::from([vts])),
            video_ports,
            PLAYBIN_MAX_DEPTH,
        )
        .map_err(|e| map_playbin_err(master_url, e))?;
    // The separate audio rendition playlist, its own source -> demux -> sink chain.
    let audio_caps = Caps::Audio { format: AudioFormat::Aac, channels: 0, sample_rate: 0 };
    let audio_ports = playbin_ports(reg, core::iter::once((audio_caps, false)))?;
    let audio_graph = reg
        .build_playbin_graph_with_source(
            Box::new(crate::hlssrc::HlsSrc::new(audio_url)),
            TsDemuxN::new(Vec::from([TsStream::Aac])),
            audio_ports,
            PLAYBIN_MAX_DEPTH,
        )
        .map_err(|e| map_playbin_err(audio_url, e))?;
    graph.merge(audio_graph);
    Ok(Some(graph))
}

/// The `playbin uri=X` auto-fan-out hook for HLS (M395 TS, M396 fMP4): probe a
/// `hls://` master playlist, discover its selected variant's renditions, and
/// assemble `HlsSrc -> {TsDemuxN | Mp4DemuxN} -> {decode -> auto sink}`, one branch
/// per elementary stream, the HLS sibling of [`mkv_playbin`] / [`ts_playbin`] /
/// [`mp4_playbin`]. The variant's media playlist picks the demuxer: an
/// `#EXT-X-MAP` init segment is fMP4 (Mp4DemuxN, tracks from the init's moov),
/// otherwise muxed MPEG-TS (TsDemuxN, ports from CODECS). The `hls` scheme maps to
/// an `https` origin.
///
/// Declines (`Ok(None)`, so the single-stream [`hls_handler`] takes over) for a
/// non-`hls` URI, a master-probe network failure, a media-only playlist, or a TS
/// variant without at least two routable muxed streams. Network-coupled: the probe
/// is validated against a live server, not in CI.
/// Parse the `playbin uri=hls://...` rendition-language hints from the URI fragment
/// (M418): `#audio-lang=fr&subtitle-lang=en`. Returns the playlist `rest` with the
/// fragment stripped, plus the requested audio and subtitle languages (`None` when
/// absent). `subtitle-lang` / `text-lang` are accepted aliases. An HLS URL has no
/// other place to carry a preference, so the fragment (never sent to the server)
/// holds it.
#[cfg(feature = "hls")]
fn hls_lang_hints(rest: &str) -> (&str, Option<alloc::string::String>, Option<alloc::string::String>) {
    let (url, frag) = rest.split_once('#').unwrap_or((rest, ""));
    let (mut audio, mut subtitle) = (None, None);
    for kv in frag.split('&').filter(|s| !s.is_empty()) {
        if let Some((k, v)) = kv.split_once('=') {
            match k {
                "audio-lang" => audio = Some(v.to_string()),
                "subtitle-lang" | "text-lang" => subtitle = Some(v.to_string()),
                _ => {}
            }
        }
    }
    (url, audio, subtitle)
}

#[cfg(feature = "hls")]
pub fn hls_playbin(reg: &Registry, uri: &str) -> Result<Option<Graph<GraphNode>>, ParseError> {
    let Some(parsed) = Uri::parse(uri) else { return Ok(None) };
    if parsed.scheme != "hls" || parsed.rest.is_empty() {
        return Ok(None);
    }
    // A `#audio-lang=` / `#subtitle-lang=` URI fragment carries the rendition
    // language preference (HLS URIs have no other place for it); strip it off the
    // playlist URL (M418).
    let (url_rest, audio_lang, _subtitle_lang) = hls_lang_hints(parsed.rest);
    let source_url = alloc::format!("https://{url_rest}");
    // Fetch + parse the master; decline (single-stream fallback) on any failure.
    let Some(master_text) = blocking_get_text(&source_url) else { return Ok(None) };
    let Ok(crate::hls::Playlist::Master(master)) = crate::hls::parse(&master_text) else {
        return Ok(None); // a media playlist or parse error: single-stream handles it
    };
    let Some(variant) = master.select(None) else { return Ok(None) };
    // The variant's media playlist tells the container: an `#EXT-X-MAP` init
    // segment means fMP4 / CMAF (fan out via Mp4DemuxN, tracks from the init's
    // moov), otherwise muxed MPEG-TS (fan out via TsDemuxN, ports from CODECS).
    let media_url = crate::fetch::resolve_url(&source_url, &variant.uri);
    let Some(media_text) = blocking_get_text(&media_url) else { return Ok(None) };
    let media = match crate::hls::parse(&media_text) {
        Ok(crate::hls::Playlist::Media(m)) => m,
        _ => return Ok(None), // a master pointing at a master, or a parse error
    };
    if let Some(map) = &media.map_uri {
        let init_url = crate::fetch::resolve_url(&media_url, map);
        let Some(init) = blocking_get_bytes(&init_url) else { return Ok(None) };
        return build_hls_fmp4_fanout(reg, &source_url, &init);
    }
    let streams = crate::hlssrc::variant_streams(&master, variant);
    // A separate audio rendition (its own playlist) means a multi-source graph: the
    // variant carries video, the audio is a distinct rendition. Pick the rendition
    // the `#audio-lang=` hint asks for (else DEFAULT, else the first), M418.
    if let Some(group) = &variant.audio_group {
        if let Some(audio) = master.pick_rendition(crate::hls::MediaType::Audio, group, audio_lang.as_deref()) {
            if let Some(audio_uri) = &audio.uri {
                let audio_url = crate::fetch::resolve_url(&source_url, audio_uri);
                return build_hls_separate_fanout(reg, &source_url, &streams, &audio_url);
            }
        }
    }
    build_hls_ts_fanout(reg, &source_url, &streams)
}

#[cfg(all(test, feature = "hls"))]
mod hls_hint_tests {
    use super::hls_lang_hints;

    #[test]
    fn parses_language_hints_from_the_uri_fragment() {
        // Both hints present: the URL is returned fragment-free.
        let (url, a, s) = hls_lang_hints("host/master.m3u8#audio-lang=fr&subtitle-lang=en");
        assert_eq!(url, "host/master.m3u8");
        assert_eq!(a.as_deref(), Some("fr"));
        assert_eq!(s.as_deref(), Some("en"));

        // `text-lang` is an accepted alias for the subtitle hint; order-independent.
        let (_, a, s) = hls_lang_hints("h/x.m3u8#text-lang=de");
        assert_eq!(a, None);
        assert_eq!(s.as_deref(), Some("de"));

        // No fragment: no preferences, URL unchanged.
        let (url, a, s) = hls_lang_hints("h/x.m3u8");
        assert_eq!(url, "h/x.m3u8");
        assert!(a.is_none() && s.is_none());
    }
}
