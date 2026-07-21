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

#[cfg(feature = "std")]
use g2g_core::runtime::{DynMultiOutputElement, PadKind, PadRequest};
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
            Caps::ByteStream {
                encoding: g2g_core::ByteStreamEncoding::MpegTs,
            },
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
    is_raw_audio, is_raw_video, DecodebinError, GraphNode, GraphNodeRef, ParseError, PrimaryStream,
    Registry,
};
#[cfg(feature = "hls")]
use g2g_core::AudioFormat;
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

/// Parse a closed-caption request from a `playbin` URI fragment, returning the URI
/// with the fragment stripped (so the file path opens) plus the selected caption
/// service. Closed captions are not discoverable up front (they ride in the video
/// SEI, not a container track), so an explicit `#closed-captions=` (alias `#cc=`)
/// fragment opts them in, the file-container analog of the HLS `#subtitle-lang=`
/// hint. Accepts `cc1`..`cc4` (CEA-608 channels), `service-N` / `708-N` (CEA-708
/// services), or a bare `on` / `1` (= CC1); an unrecognised value yields `None`.
#[cfg(feature = "std")]
fn cc_request(uri: &str) -> (alloc::string::String, Option<crate::ccextract::CcSource>) {
    let (base, frag) = uri.split_once('#').unwrap_or((uri, ""));
    let mut cc = None;
    for kv in frag.split('&').filter(|s| !s.is_empty()) {
        if let Some((k, v)) = kv.split_once('=') {
            if k == "closed-captions" || k == "cc" {
                cc = parse_cc_source(v);
            }
        }
    }
    (base.to_string(), cc)
}

/// Map a `#closed-captions=` value to a [`CcSource`](crate::ccextract::CcSource).
#[cfg(feature = "std")]
fn parse_cc_source(v: &str) -> Option<crate::ccextract::CcSource> {
    crate::ccextract::CcSource::parse(v)
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
    let (uri, cc) = cc_request(uri);
    let Some((path, prefix)) = open_file_prefix(&uri) else {
        return Ok(None);
    };
    let mut demux = crate::matroska::MatroskaDemuxer::new();
    demux.push_data(&prefix);
    let infos = crate::mkvdemux::forwardable_streams(&demux);
    if infos.is_empty() {
        return Ok(None); // not Matroska (or no forwardable track): decline
    }
    let streams: Vec<_> = infos.iter().map(|i| i.stream).collect();
    let av: Vec<(Caps, bool)> = infos.iter().map(|i| (i.caps.clone(), i.video)).collect();
    if let Some(video_idx) = infos.iter().position(|i| i.video) {
        // An explicit `#closed-captions=` request overlays the in-SEI captions
        // (M430); else a subtitle track overlays its cues (the MP4 M412 sibling,
        // M415). Only one text pad, so the explicit caption request wins.
        if let Some(cc) = cc {
            let source = crate::filesrc::FileSrc::new(
                &path,
                Caps::ByteStream {
                    encoding: ByteStreamEncoding::Matroska,
                },
            );
            return build_cc_overlay(
                reg,
                Box::new(source),
                crate::mkvdemux::MkvDemuxN::new(streams),
                &av,
                video_idx,
                cc,
            )
            .map(Some);
        }
        if let Some(text) = crate::mkvdemux::subtitle_streams(&demux).first() {
            return build_mkv_subtitle_overlay(reg, &path, &infos, text).map(Some);
        }
    }
    // The file:// URI handler self-demuxes MP4, so build the matching Matroska
    // byte source directly and feed it to the multi-output demuxer. Audio tracks
    // go through the convert/resample branch (not decoder -> sink direct).
    let source = crate::filesrc::FileSrc::new(
        &path,
        Caps::ByteStream {
            encoding: ByteStreamEncoding::Matroska,
        },
    );
    build_av_fanout(
        reg,
        Box::new(source),
        crate::mkvdemux::MkvDemuxN::new(streams),
        &av,
    )
    .map(Some)
}

/// The `playbin uri=X` auto-fan-out hook for MPEG-TS (M389): probe a `file://`
/// transport stream's PMT, then assemble `FileSrc -> TsDemuxN -> {decode -> auto
/// sink}` with one branch per forwardable stream, the MPEG-TS sibling of
/// [`mkv_playbin`]. Declines (`Ok(None)`) for a non-`file://` URI, an unreadable
/// file, or a non-MPEG-TS container (no PMT in the probed prefix).
#[cfg(feature = "std")]
pub fn ts_playbin(reg: &Registry, uri: &str) -> Result<Option<Graph<GraphNode>>, ParseError> {
    let (uri, cc) = cc_request(uri);
    let Some((path, prefix)) = open_file_prefix(&uri) else {
        return Ok(None);
    };
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
    let av: Vec<(Caps, bool)> = infos.iter().map(|i| (i.caps.clone(), i.video)).collect();
    let source = crate::filesrc::FileSrc::new(
        &path,
        Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        },
    );
    // An explicit `#closed-captions=` request overlays the in-SEI captions (M430);
    // MPEG-TS broadcast is the most common CEA-608 / 708 carrier.
    if let (Some(cc), Some(video_idx)) = (cc, infos.iter().position(|i| i.video)) {
        return build_cc_overlay(
            reg,
            Box::new(source),
            crate::tsdemux::TsDemuxN::new(streams),
            &av,
            video_idx,
            cc,
        )
        .map(Some);
    }
    build_av_fanout(
        reg,
        Box::new(source),
        crate::tsdemux::TsDemuxN::new(streams),
        &av,
    )
    .map(Some)
}

/// Map a per-branch decode-chain failure to the text-parser error (the
/// [`decodebin`](Registry::decodebin) form): no chain quotes the unplugged input
/// caps, a link error wraps the graph error.
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
    let overlay = wire_overlay_av(reg, graph, demux, av, video_idx)?;
    // The text pad is fed from a port of the *same* demux (single-container case).
    link_text_into_overlay(graph, overlay, demux.out(text_port), text_caps)
}

/// Build the video half of a subtitle overlay onto `demux`'s A/V ports and return
/// the [`TextOverlayN`] muxer (its `input(1)` is the still-open text pad). The
/// video track (`av[video_idx]`) decodes and converts to RGBA8 into `overlay.video`;
/// the overlay output converts back to NV12 for the auto video sink; the other A/V
/// tracks fan out to their own auto sinks. The decoder is auto-plugged; the
/// `videoconvert`s are wired explicitly (caps-driven `register_launch` elements
/// outside the auto-plug pool, RGBA8 in/out vs the display sink's NV12). Shared by
/// the single-container [`wire_subtitle_overlay`] and the multi-source HLS builder,
/// which differ only in where the text pad is fed from.
/// Decode a compressed-audio demux port to an auto audio sink (M422): the audio
/// analog of the video overlay's explicit `VideoConvert`s. The decoded PCM's
/// channel count and sample rate are only known once a frame decodes, but the
/// sink needs a fixed format at configure, so the branch is
/// `decode -> audioconvert(stereo) -> audioresample(48 kHz) -> autoaudiosink`: the
/// converters absorb the stream's real params (learned via `CapsChanged`) and
/// always present 2ch / 48000 to the sink. The `audioconvert` / `audioresample`
/// are `register_launch` elements outside the auto-plug pool, so they are wired
/// explicitly (like the overlay's converters), while the decoder is auto-plugged.
#[cfg(feature = "std")]
fn wire_audio_branch(
    reg: &Registry,
    graph: &mut Graph<GraphNode>,
    src: impl Into<g2g_core::graph::PadId>,
    caps: &Caps,
) -> Result<(), ParseError> {
    let sink = reg
        .make_element("autoaudiosink")
        .ok_or_else(|| ParseError::UnknownElement("autoaudiosink".to_string()))?;
    let snk = graph.add_sink(GraphNodeRef::Element(sink));
    let resample = graph.add_transform(GraphNodeRef::element(
        crate::audioresample::AudioResample::new(48_000),
    ));
    let convert = graph.add_transform(GraphNodeRef::element(
        crate::audioconvert::AudioConvert::new(g2g_core::AudioFormat::PcmS16Le, 2),
    ));
    graph.link(convert, resample).map_err(ParseError::Graph)?;
    graph.link(resample, snk).map_err(ParseError::Graph)?;
    reg.decodebin(graph, src, convert, caps, &is_raw_audio, PLAYBIN_MAX_DEPTH)
        .map_err(|e| map_decode_err(caps, e))?;
    Ok(())
}

/// Build a plain (no-subtitle) A/V fan-out graph: `source -> demux -> { per-stream
/// branch }`, the non-overlay sibling of [`build_mkv_subtitle_overlay`]. Each video
/// track decodes straight to an auto video sink (the decoder is auto-plugged to the
/// sink's format, as `build_playbin_graph` does); each audio track goes through the
/// [`wire_audio_branch`] decode -> `audioconvert` -> `audioresample` chain so the
/// sink always sees a fixed PCM format while the converters absorb the stream's real
/// channels / rate. `av` lists each demux port's `(elementary caps, is_video)` in
/// port order. This replaces a `build_playbin_graph_with_source` call for the
/// audio-bearing containers: that g2g-core builder links a decoder straight to the
/// audio sink (no converter chain), since the plugin-side `audioconvert` /
/// `audioresample` are outside its element pool.
#[cfg(feature = "std")]
fn build_av_fanout<D>(
    reg: &Registry,
    source: Box<dyn DynSourceLoop>,
    demux: D,
    av: &[(Caps, bool)],
) -> Result<Graph<GraphNode>, ParseError>
where
    D: g2g_core::MultiOutputElement + 'static,
{
    let mut graph: Graph<GraphNode> = Graph::new();
    let src = graph.add_source(GraphNodeRef::Source(source));
    let demux = graph.add_demux(GraphNodeRef::demux(demux), av.len() as u8);
    graph.link(src, demux.input()).map_err(ParseError::Graph)?;
    wire_av_fanout(reg, &mut graph, demux, av)?;
    Ok(graph)
}

/// Wire each demux port of a plain A/V fan-out: video -> auto video sink
/// (auto-plugged decoder), audio -> [`wire_audio_branch`]. Shared by every
/// container's no-subtitle `playbin` hook (MKV / TS / MP4).
#[cfg(feature = "std")]
fn wire_av_fanout(
    reg: &Registry,
    graph: &mut Graph<GraphNode>,
    demux: g2g_core::graph::Demux,
    av: &[(Caps, bool)],
) -> Result<(), ParseError> {
    for (i, (caps, video)) in av.iter().enumerate() {
        if *video {
            let sink = reg
                .make_element("autovideosink")
                .ok_or_else(|| ParseError::UnknownElement("autovideosink".to_string()))?;
            let vsnk = graph.add_sink(GraphNodeRef::Element(sink));
            reg.decodebin(
                graph,
                demux.out(i as u8),
                vsnk,
                caps,
                &is_raw_video,
                PLAYBIN_MAX_DEPTH,
            )
            .map_err(|e| map_decode_err(caps, e))?;
        } else {
            wire_audio_branch(reg, graph, demux.out(i as u8), caps)?;
        }
    }
    Ok(())
}

#[cfg(feature = "std")]
fn wire_overlay_av(
    reg: &Registry,
    graph: &mut Graph<GraphNode>,
    demux: g2g_core::graph::Demux,
    av: &[(Caps, bool)],
    video_idx: usize,
) -> Result<g2g_core::graph::Muxer, ParseError> {
    use g2g_core::RawVideoFormat;

    let overlay = graph.add_muxer(
        GraphNodeRef::muxer(crate::textoverlay::TextOverlayN::new()),
        2,
    );
    let to_rgba = graph.add_transform(GraphNodeRef::element(
        crate::videoconvert::VideoConvert::new(RawVideoFormat::Rgba8),
    ));
    let to_nv12 = graph.add_transform(GraphNodeRef::element(
        crate::videoconvert::VideoConvert::new(RawVideoFormat::Nv12),
    ));
    graph
        .link(to_rgba, overlay.input(0))
        .map_err(ParseError::Graph)?;
    graph
        .link(overlay.output(), to_nv12)
        .map_err(ParseError::Graph)?;
    let vsink = reg
        .make_element("autovideosink")
        .ok_or_else(|| ParseError::UnknownElement("autovideosink".to_string()))?;
    let vsnk = graph.add_sink(GraphNodeRef::Element(vsink));
    graph.link(to_nv12, vsnk).map_err(ParseError::Graph)?;

    // Each A/V track: the video one decodes into the overlay's RGBA8 convert, the
    // rest fan out to their own auto sinks through the audio decode/convert chain.
    for (i, (caps, _video)) in av.iter().enumerate() {
        if i == video_idx {
            reg.decodebin(
                graph,
                demux.out(i as u8),
                to_rgba,
                caps,
                &is_raw_video,
                PLAYBIN_MAX_DEPTH,
            )
            .map_err(|e| map_decode_err(caps, e))?;
        } else {
            wire_audio_branch(reg, graph, demux.out(i as u8), caps)?;
        }
    }
    Ok(overlay)
}

/// Build a closed-caption overlay graph (M430): `source -> demux -> {video tee}`,
/// where the compressed video is teed so one copy decodes for display and the
/// other feeds a [`CcExtract`](crate::ccextract::CcExtract). The CcExtract output
/// (timed `Text{Utf8}` cues) drives the same `TextOverlayN` text pad a subtitle
/// track would, so embedded CEA-608 / CEA-708 captions render onto the video.
/// `av` lists each demux port's `(elementary caps, is_video)`; the video track is
/// at `video_idx`. Generic over the demux element, like [`build_av_fanout`].
#[cfg(feature = "std")]
fn build_cc_overlay<D>(
    reg: &Registry,
    source: Box<dyn DynSourceLoop>,
    demux: D,
    av: &[(Caps, bool)],
    video_idx: usize,
    cc: crate::ccextract::CcSource,
) -> Result<Graph<GraphNode>, ParseError>
where
    D: g2g_core::MultiOutputElement + 'static,
{
    let mut graph: Graph<GraphNode> = Graph::new();
    let src = graph.add_source(GraphNodeRef::Source(source));
    let demux = graph.add_demux(GraphNodeRef::demux(demux), av.len() as u8);
    graph.link(src, demux.input()).map_err(ParseError::Graph)?;
    wire_cc_overlay(reg, &mut graph, demux, av, video_idx, cc)?;
    Ok(graph)
}

/// Wire the closed-caption overlay onto a built demux (the shared core of
/// [`build_cc_overlay`]). The video port feeds a [`Tee`](g2g_core::graph::Tee):
/// branch 0 is the display chain (`decode -> RGBA8 convert -> overlay -> NV12
/// convert -> auto video sink`, the [`wire_overlay_av`] shape), branch 1 reframes
/// the compressed video to access units (so a TS PES does not split an SEI NAL)
/// and runs `CcExtract` into the overlay's text pad. Audio tracks fan out through
/// the [`wire_audio_branch`] decode chain.
#[cfg(feature = "std")]
fn wire_cc_overlay(
    reg: &Registry,
    graph: &mut Graph<GraphNode>,
    demux: g2g_core::graph::Demux,
    av: &[(Caps, bool)],
    video_idx: usize,
    cc: crate::ccextract::CcSource,
) -> Result<(), ParseError> {
    use g2g_core::RawVideoFormat;

    let overlay = graph.add_muxer(
        GraphNodeRef::muxer(crate::textoverlay::TextOverlayN::new()),
        2,
    );
    let to_rgba = graph.add_transform(GraphNodeRef::element(
        crate::videoconvert::VideoConvert::new(RawVideoFormat::Rgba8),
    ));
    let to_nv12 = graph.add_transform(GraphNodeRef::element(
        crate::videoconvert::VideoConvert::new(RawVideoFormat::Nv12),
    ));
    graph
        .link(to_rgba, overlay.input(0))
        .map_err(ParseError::Graph)?;
    graph
        .link(overlay.output(), to_nv12)
        .map_err(ParseError::Graph)?;
    let vsink = reg
        .make_element("autovideosink")
        .ok_or_else(|| ParseError::UnknownElement("autovideosink".to_string()))?;
    let vsnk = graph.add_sink(GraphNodeRef::Element(vsink));
    graph.link(to_nv12, vsnk).map_err(ParseError::Graph)?;

    let video_caps = &av[video_idx].0;
    // Tee the compressed video: branch 0 decodes for display, branch 1 captions.
    let tee = graph.add_tee(2);
    graph
        .link(demux.out(video_idx as u8), tee.input())
        .map_err(ParseError::Graph)?;
    reg.decodebin(
        graph,
        tee.out(0),
        to_rgba,
        video_caps,
        &is_raw_video,
        PLAYBIN_MAX_DEPTH,
    )
    .map_err(|e| map_decode_err(video_caps, e))?;

    // Caption branch: CcExtract (compressed video in, Text{Utf8} cues out) feeds
    // the overlay text pad, fed in turn by a re-framing parser when the codec has
    // one (the same parser `decodebin` auto-plugs before the decoder).
    let cc_in = graph.add_transform(GraphNodeRef::element(
        crate::ccextract::CcExtract::for_source(cc),
    ));
    graph
        .link(cc_in, overlay.input(1))
        .map_err(ParseError::Graph)?;
    let cc_head = match video_caps {
        Caps::CompressedVideo {
            codec: g2g_core::VideoCodec::H264,
            ..
        } => {
            let p = graph.add_transform(GraphNodeRef::element(
                crate::h264parse::H264Parse::reframing(),
            ));
            graph.link(p, cc_in).map_err(ParseError::Graph)?;
            p
        }
        Caps::CompressedVideo {
            codec: g2g_core::VideoCodec::H265,
            ..
        } => {
            let p = graph.add_transform(GraphNodeRef::element(
                crate::h265parse::H265Parse::reframing(),
            ));
            graph.link(p, cc_in).map_err(ParseError::Graph)?;
            p
        }
        _ => cc_in,
    };
    graph.link(tee.out(1), cc_head).map_err(ParseError::Graph)?;

    // Audio / other tracks fan out to their own sinks.
    for (i, (caps, _video)) in av.iter().enumerate() {
        if i != video_idx {
            wire_audio_branch(reg, graph, demux.out(i as u8), caps)?;
        }
    }
    Ok(())
}

/// Feed a subtitle stream into the overlay's text pad (`overlay.input(1)`). A
/// plain-UTF8 cue stream (MP4 `tx3g`, MKV `S_TEXT/UTF8`) links straight in; a
/// structured format (SRT / WebVTT / SSA / TTML, e.g. an HLS WebVTT rendition)
/// parses to timed UTF-8 cues via `SubParse` first. `text_src` is the pad producing
/// `text_caps` (a demux port, or an HLS subtitle source's output).
#[cfg(feature = "std")]
fn link_text_into_overlay(
    graph: &mut Graph<GraphNode>,
    overlay: g2g_core::graph::Muxer,
    text_src: g2g_core::graph::PadId,
    text_caps: &Caps,
) -> Result<(), ParseError> {
    let text_in: g2g_core::graph::PadId = match text_caps {
        Caps::Text {
            format: g2g_core::TextFormat::Utf8,
        } => overlay.input(1),
        _ => {
            let sub = graph.add_transform(GraphNodeRef::element(crate::subparse::SubParse::new()));
            graph
                .link(sub, overlay.input(1))
                .map_err(ParseError::Graph)?;
            sub.into()
        }
    };
    graph.link(text_src, text_in).map_err(ParseError::Graph)?;
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

    let video_idx = av
        .iter()
        .position(|i| i.video)
        .ok_or_else(|| ParseError::NoDecodeChain("subtitle overlay needs a video track".into()))?;

    // Demux ports: every A/V track (in moov order) then the subtitle track.
    let mut demux_ports: Vec<Mp4Port> = av
        .iter()
        .map(|i| Mp4Port {
            track_id: i.track_id,
            caps: i.caps.clone(),
        })
        .collect();
    demux_ports.push(Mp4Port {
        track_id: text.track_id,
        caps: text.caps.clone(),
    });
    let text_port = (demux_ports.len() - 1) as u8;
    let outputs = demux_ports.len() as u8;

    let mut graph: Graph<GraphNode> = Graph::new();
    let source = crate::filesrc::FileSrc::new(
        path,
        Caps::ByteStream {
            encoding: ByteStreamEncoding::IsoBmff,
        },
    );
    let src = graph.add_source(GraphNodeRef::source(source));
    let demux = graph.add_demux(GraphNodeRef::demux(Mp4DemuxN::new(demux_ports)), outputs);
    graph.link(src, demux.input()).map_err(ParseError::Graph)?;

    let av_caps: Vec<(Caps, bool)> = av.iter().map(|i| (i.caps.clone(), i.video)).collect();
    wire_subtitle_overlay(
        reg, &mut graph, demux, &av_caps, video_idx, text_port, &text.caps,
    )?;
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

    let video_idx = av
        .iter()
        .position(|i| i.video)
        .ok_or_else(|| ParseError::NoDecodeChain("subtitle overlay needs a video track".into()))?;

    // Demux ports (selected streams): every A/V stream (in track order) then the
    // subtitle stream.
    let mut stream_ports: Vec<_> = av.iter().map(|i| i.stream).collect();
    stream_ports.push(text.stream);
    let text_port = (stream_ports.len() - 1) as u8;
    let outputs = stream_ports.len() as u8;

    let mut graph: Graph<GraphNode> = Graph::new();
    let source = crate::filesrc::FileSrc::new(
        path,
        Caps::ByteStream {
            encoding: ByteStreamEncoding::Matroska,
        },
    );
    let src = graph.add_source(GraphNodeRef::source(source));
    let demux = graph.add_demux(GraphNodeRef::demux(MkvDemuxN::new(stream_ports)), outputs);
    graph.link(src, demux.input()).map_err(ParseError::Graph)?;

    let av_caps: Vec<(Caps, bool)> = av.iter().map(|i| (i.caps.clone(), i.video)).collect();
    wire_subtitle_overlay(
        reg, &mut graph, demux, &av_caps, video_idx, text_port, &text.caps,
    )?;
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
    let (uri, cc) = cc_request(uri);
    let Some((path, prefix)) = open_file_prefix(&uri) else {
        return Ok(None);
    };
    let infos = crate::mp4demuxn::forwardable_streams(&prefix);
    if infos.is_empty() {
        return Ok(None); // not MP4 (or moov not in the prefix): decline
    }
    let av: Vec<(Caps, bool)> = infos.iter().map(|i| (i.caps.clone(), i.video)).collect();
    let demux_ports: Vec<_> = infos
        .iter()
        .map(|i| crate::mp4demuxn::Mp4Port {
            track_id: i.track_id,
            caps: i.caps.clone(),
        })
        .collect();
    let source = crate::filesrc::FileSrc::new(
        &path,
        Caps::ByteStream {
            encoding: ByteStreamEncoding::IsoBmff,
        },
    );
    if let Some(video_idx) = infos.iter().position(|i| i.video) {
        // An explicit `#closed-captions=` request overlays the in-SEI captions
        // (M430); else a subtitle track overlays its cues (M412). One text pad, so
        // the explicit caption request wins.
        if let Some(cc) = cc {
            return build_cc_overlay(
                reg,
                Box::new(source),
                crate::mp4demuxn::Mp4DemuxN::new(demux_ports),
                &av,
                video_idx,
                cc,
            )
            .map(Some);
        }
        if let Some(text) = crate::mp4demuxn::subtitle_streams(&prefix).first() {
            return build_mp4_subtitle_overlay(reg, &path, &infos, text).map(Some);
        }
    }
    build_av_fanout(
        reg,
        Box::new(source),
        crate::mp4demuxn::Mp4DemuxN::new(demux_ports),
        &av,
    )
    .map(Some)
}

// ---- Explicit-demux fan-out hooks (M476) --------------------------------
//
// The `matroskademux name=d  d.video_0 ! ...  d.audio_0 ! ...` path: unlike the
// lone `playbin`, the demux sits inside a user-authored line, so the hook does not
// build a whole graph. It probes the upstream file, maps each pad request to a
// forwardable stream, and returns the multi-output demuxer with one port per pad
// (in reference order); the core parser wires the user's branches to the ports.

/// Read up to the probe limit from a plain file path (the demux hook receives a
/// `filesrc location=`, a bare path, not a `file://` URI).
#[cfg(feature = "std")]
fn read_prefix(path: &str) -> Option<Vec<u8>> {
    use std::io::Read;
    let mut prefix = Vec::new();
    let f = std::fs::File::open(path).ok()?;
    f.take(PLAYBIN_PROBE_BYTES).read_to_end(&mut prefix).ok()?;
    Some(prefix)
}

/// Resolve each output-pad request to a selectable-stream index (M476, text M477),
/// given the per-stream kind in selection order (A/V streams first, then subtitle
/// streams). `video_k` picks the k-th video stream, `audio_k` the k-th audio,
/// `text_k` / `subtitle_k` the k-th subtitle, `src_k` / a bare `d.` the k-th stream
/// overall. `None` if any request is unsatisfiable (out of range, or a text pad on
/// a container with no subtitle track), declining the whole hook.
#[cfg(feature = "std")]
fn resolve_pads(kinds: &[PadKind], pads: &[PadRequest]) -> Option<Vec<usize>> {
    pads.iter()
        .map(|req| match req.kind {
            PadKind::Any => (req.index < kinds.len()).then_some(req.index),
            want => kinds
                .iter()
                .enumerate()
                .filter(|(_, k)| **k == want)
                .nth(req.index)
                .map(|(i, _)| i),
        })
        .collect()
}

/// Probe a Matroska file and build a [`MkvDemuxN`](crate::mkvdemux::MkvDemuxN) with
/// one port per pad request, returning the demuxer plus each selected port's
/// elementary caps (M482). Shared by `matroskademux` demux-select (M476, drops the
/// caps) and `decodebin` fan-out (M482, keeps them to auto-plug a decoder per port).
/// Declines a non-Matroska file or an unsatisfiable request.
#[cfg(feature = "std")]
fn mkv_select(
    location: &str,
    pads: &[PadRequest],
) -> Option<(Box<dyn DynMultiOutputElement>, Vec<Caps>)> {
    let prefix = read_prefix(location)?;
    let mut demux = crate::matroska::MatroskaDemuxer::new();
    demux.push_data(&prefix);
    // Selection order: A/V streams first, then subtitle streams (M477), so a
    // `d.text_0` request maps to the container's subtitle track. `MkvDemuxN`
    // de-frames a selected subtitle block to `Text{Utf8}` on its port.
    let mut infos = crate::mkvdemux::forwardable_streams(&demux);
    let mut kinds: Vec<PadKind> = infos
        .iter()
        .map(|i| {
            if i.video {
                PadKind::Video
            } else {
                PadKind::Audio
            }
        })
        .collect();
    for text in crate::mkvdemux::subtitle_streams(&demux) {
        kinds.push(PadKind::Text);
        infos.push(text);
    }
    if infos.is_empty() {
        return None;
    }
    let sel = resolve_pads(&kinds, pads)?;
    let streams: Vec<_> = sel.iter().map(|&i| infos[i].stream).collect();
    let caps: Vec<Caps> = sel.iter().map(|&i| infos[i].caps.clone()).collect();
    Some((Box::new(crate::mkvdemux::MkvDemuxN::new(streams)), caps))
}

/// `matroskademux` explicit fan-out (M476): build the multi-output demuxer,
/// dropping the per-port caps (each branch names its own decode chain).
#[cfg(feature = "std")]
pub fn mkv_demux_select(
    name: &str,
    location: &str,
    pads: &[PadRequest],
) -> Option<Box<dyn DynMultiOutputElement>> {
    (name == "matroskademux")
        .then(|| mkv_select(location, pads).map(|(d, _)| d))
        .flatten()
}

/// `decodebin` fan-out over a Matroska file (M482): the demuxer + per-port caps.
#[cfg(feature = "std")]
pub fn mkv_decodebin_select(
    location: &str,
    pads: &[PadRequest],
) -> Option<(Box<dyn DynMultiOutputElement>, Vec<Caps>)> {
    mkv_select(location, pads)
}

/// Probe an MPEG-TS file and build a [`TsDemuxN`](crate::tsdemux::TsDemuxN) with
/// the requested streams + their caps (M482). Shared by `tsdemux` demux-select and
/// `decodebin` fan-out. Declines a non-MPEG-TS file.
#[cfg(feature = "std")]
fn ts_select(
    location: &str,
    pads: &[PadRequest],
) -> Option<(Box<dyn DynMultiOutputElement>, Vec<Caps>)> {
    let prefix = read_prefix(location)?;
    let mut demux = crate::mpegts::TsDemuxer::new();
    let mut off = 0;
    while off + 188 <= prefix.len() {
        if prefix[off] == 0x47 {
            demux.push_packet(&prefix[off..off + 188]);
        }
        off += 188;
    }
    let infos = crate::tsdemux::forwardable_streams(&demux);
    if infos.is_empty() {
        return None;
    }
    // MPEG-TS carries no subtitle track in the demuxer, so a `d.text_0` request
    // finds no `Text` kind and declines (M477).
    let kinds: Vec<PadKind> = infos
        .iter()
        .map(|i| {
            if i.video {
                PadKind::Video
            } else {
                PadKind::Audio
            }
        })
        .collect();
    let sel = resolve_pads(&kinds, pads)?;
    let streams: Vec<_> = sel.iter().map(|&i| infos[i].stream).collect();
    let caps: Vec<Caps> = sel.iter().map(|&i| infos[i].caps.clone()).collect();
    Some((Box::new(crate::tsdemux::TsDemuxN::new(streams)), caps))
}

/// `tsdemux` explicit fan-out (M476).
#[cfg(feature = "std")]
pub fn ts_demux_select(
    name: &str,
    location: &str,
    pads: &[PadRequest],
) -> Option<Box<dyn DynMultiOutputElement>> {
    (name == "tsdemux")
        .then(|| ts_select(location, pads).map(|(d, _)| d))
        .flatten()
}

/// `decodebin` fan-out over an MPEG-TS file (M482): the demuxer + per-port caps.
#[cfg(feature = "std")]
pub fn ts_decodebin_select(
    location: &str,
    pads: &[PadRequest],
) -> Option<(Box<dyn DynMultiOutputElement>, Vec<Caps>)> {
    ts_select(location, pads)
}

/// Bare-`decodebin` primary-stream hook for MPEG-TS (M746): a `filesrc
/// location=X.ts ! decodebin` on an audio-only transport stream needs the
/// single-stream [`TsDemux`](crate::tsdemux::TsDemux) to select its audio stream
/// (the default is a video port, so the auto-plug would pick a video decoder).
/// Sniff the PMT; decline (`None`) a non-TS file, an empty PMT, or one that carries
/// a video track (the default video path is correct), else return `tsdemux` with the
/// `stream=<codec>` selection and the audio elementary caps for the decoder search.
#[cfg(feature = "std")]
pub fn ts_primary_stream(location: &str, caps: &Caps) -> Option<PrimaryStream> {
    if !matches!(
        caps,
        Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs
        }
    ) {
        return None;
    }
    let prefix = read_prefix(location)?;
    let mut demux = crate::mpegts::TsDemuxer::new();
    let mut off = 0;
    while off + 188 <= prefix.len() {
        if prefix[off] == 0x47 {
            demux.push_packet(&prefix[off..off + 188]);
        }
        off += 188;
    }
    let infos = crate::tsdemux::forwardable_streams(&demux);
    // A video track present: the demux's default video port is right, decline.
    if infos.is_empty() || infos.iter().any(|i| i.video) {
        return None;
    }
    let audio = infos.into_iter().find(|i| !i.video)?;
    Some(PrimaryStream {
        demux: "tsdemux",
        props: alloc::vec![(
            "stream".to_string(),
            crate::tsdemux::ts_stream_to_str(audio.stream).to_string(),
        )],
        caps: audio.caps,
    })
}

/// Bare-`decodebin` primary-stream hook for MP4 (M748): the MP4 sibling of
/// [`ts_primary_stream`]. A `filesrc location=X.m4a ! decodebin` on an audio-only
/// MP4 needs the single-stream [`Mp4Demux`](crate::mp4demux::Mp4Demux) to select
/// its audio track (the default is the video port, so the auto-plug would pick a
/// video decoder and fail "no caps overlap"). Sniff the `moov`; decline (`None`) a
/// non-MP4 file, a `moov` past the probe window, an empty track set, or a file that
/// carries a video track (the default video path is right), else return `qtdemux`
/// with `stream=aac` and the audio elementary caps for the decoder search.
#[cfg(feature = "std")]
pub fn mp4_primary_stream(location: &str, caps: &Caps) -> Option<PrimaryStream> {
    if !matches!(
        caps,
        Caps::ByteStream {
            encoding: ByteStreamEncoding::Mp4
        }
    ) {
        return None;
    }
    let prefix = read_prefix(location)?;
    let infos = crate::mp4demuxn::forwardable_streams(&prefix);
    // A video track present: the demux's default video port is right, decline.
    if infos.is_empty() || infos.iter().any(|i| i.video) {
        return None;
    }
    let audio = infos.into_iter().find(|i| !i.video)?;
    Some(PrimaryStream {
        demux: "qtdemux",
        props: alloc::vec![("stream".to_string(), "aac".to_string())],
        caps: audio.caps,
    })
}

/// Probe an MP4 file and build an [`Mp4DemuxN`](crate::mp4demuxn::Mp4DemuxN) with
/// the requested tracks + their caps (M482). Shared by `qtdemux` demux-select and
/// `decodebin` fan-out. Declines a non-MP4 file (or a `moov` past the probe window).
#[cfg(feature = "std")]
fn mp4_select(
    location: &str,
    pads: &[PadRequest],
) -> Option<(Box<dyn DynMultiOutputElement>, Vec<Caps>)> {
    let prefix = read_prefix(location)?;
    // Selection order: A/V tracks first, then subtitle (`tx3g`/`wvtt`/`stpp`)
    // tracks (M477), so a `d.text_0` request maps to the container's text track.
    // `Mp4DemuxN` forwards a `tx3g`/`wvtt` cue as `Text{Utf8}` (a `stpp` track as
    // `Text{Ttml}`, which a following `subparse` reduces to `Text{Utf8}`).
    let mut infos = crate::mp4demuxn::forwardable_streams(&prefix);
    let mut kinds: Vec<PadKind> = infos
        .iter()
        .map(|i| {
            if i.video {
                PadKind::Video
            } else {
                PadKind::Audio
            }
        })
        .collect();
    for text in crate::mp4demuxn::subtitle_streams(&prefix) {
        kinds.push(PadKind::Text);
        infos.push(text);
    }
    if infos.is_empty() {
        return None;
    }
    let sel = resolve_pads(&kinds, pads)?;
    let caps: Vec<Caps> = sel.iter().map(|&i| infos[i].caps.clone()).collect();
    let ports: Vec<_> = sel
        .iter()
        .map(|&i| crate::mp4demuxn::Mp4Port {
            track_id: infos[i].track_id,
            caps: infos[i].caps.clone(),
        })
        .collect();
    Some((Box::new(crate::mp4demuxn::Mp4DemuxN::new(ports)), caps))
}

/// `qtdemux` explicit fan-out (M476).
#[cfg(feature = "std")]
pub fn mp4_demux_select(
    name: &str,
    location: &str,
    pads: &[PadRequest],
) -> Option<Box<dyn DynMultiOutputElement>> {
    (name == "qtdemux")
        .then(|| mp4_select(location, pads).map(|(d, _)| d))
        .flatten()
}

/// `decodebin` fan-out over an MP4 file (M482): the demuxer + per-port caps.
#[cfg(feature = "std")]
pub fn mp4_decodebin_select(
    location: &str,
    pads: &[PadRequest],
) -> Option<(Box<dyn DynMultiOutputElement>, Vec<Caps>)> {
    mp4_select(location, pads)
}

/// Fetch a text resource synchronously on a throwaway current-thread runtime, for
/// the `playbin uri=hls://...` probe (M395). Returns `None` on any failure, so the
/// hook declines to the single-stream `hls` handler. Network-coupled: validated
/// against a live server, not in CI (like the RTSP / WHIP paths).
#[cfg(feature = "hls")]
fn blocking_get_text(url: &str) -> Option<alloc::string::String> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .ok()?;
    let client = reqwest::Client::new();
    rt.block_on(crate::fetch::get_text(
        &client,
        url,
        crate::fetch::MAX_MANIFEST_BYTES,
    ))
    .ok()
}

/// Fetch a binary resource synchronously (the fMP4 `#EXT-X-MAP` init segment),
/// the byte sibling of [`blocking_get_text`]. `None` on any failure.
#[cfg(feature = "hls")]
fn blocking_get_bytes(url: &str) -> Option<Vec<u8>> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .ok()?;
    let client = reqwest::Client::new();
    rt.block_on(crate::fetch::get_bytes(
        &client,
        url,
        crate::fetch::MAX_SEGMENT_BYTES,
    ))
    .ok()
}

/// The MPEG-TS elementary-stream selector for an HLS discovered stream, or `None`
/// for a codec `TsDemuxN` cannot route (only H.264 / H.265 video + AAC audio).
#[cfg(feature = "hls")]
fn hls_ts_stream(info: &crate::hlssrc::HlsStreamInfo) -> Option<crate::tsdemux::TsStream> {
    use crate::tsdemux::TsStream;
    match &info.caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            ..
        } => Some(TsStream::H264),
        Caps::CompressedVideo {
            codec: VideoCodec::H265,
            ..
        } => Some(TsStream::H265),
        Caps::Audio {
            format: AudioFormat::Aac,
            ..
        } => Some(TsStream::Aac),
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
    let source = crate::hlssrc::HlsSrc::new(source_url);
    build_av_fanout(
        reg,
        Box::new(source),
        crate::tsdemux::TsDemuxN::new(ts_streams),
        &port_infos,
    )
    .map(Some)
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
    let av: Vec<(Caps, bool)> = infos.iter().map(|i| (i.caps.clone(), i.video)).collect();
    let demux_ports: Vec<_> = infos
        .iter()
        .map(|i| crate::mp4demuxn::Mp4Port {
            track_id: i.track_id,
            caps: i.caps.clone(),
        })
        .collect();
    let source = crate::hlssrc::HlsSrc::new(source_url);
    build_av_fanout(
        reg,
        Box::new(source),
        crate::mp4demuxn::Mp4DemuxN::new(demux_ports),
        &av,
    )
    .map(Some)
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
    let Some(vts) = hls_ts_stream(video) else {
        return Ok(None);
    };
    let mut graph = build_av_fanout(
        reg,
        Box::new(crate::hlssrc::HlsSrc::new(master_url)),
        TsDemuxN::new(Vec::from([vts])),
        &[(video.caps.clone(), true)],
    )?;
    // The separate audio rendition playlist, its own source -> demux -> audio
    // branch (decode -> convert -> resample -> sink), merged into the graph.
    let audio_caps = Caps::Audio {
        format: AudioFormat::Aac,
        channels: 0,
        sample_rate: 0,
    };
    let audio_graph = build_av_fanout(
        reg,
        Box::new(crate::hlssrc::HlsSrc::new(audio_url)),
        TsDemuxN::new(Vec::from([TsStream::Aac])),
        &[(audio_caps, false)],
    )?;
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
/// Assemble the HLS subtitle-overlay graph (M419): the video (and any muxed audio)
/// rides the variant's own MPEG-TS segments through a `TsDemuxN`, the video branch
/// feeding a [`TextOverlayN`](crate::textoverlay::TextOverlayN); the subtitle is a
/// *separate* `#EXT-X-MEDIA:TYPE=SUBTITLES` WebVTT rendition, its own
/// `HlsSrc(text) -> SubParse` source chain in the same graph, linked into the
/// overlay's text pad (the cross-source join, vs the single-demuxer MP4 / MKV
/// overlay). `streams` carries the variant's muxed streams (video required);
/// `subtitle_url` is the resolved rendition playlist. `Ok(None)` (decline) when
/// there is no routable muxed video. Raw `.vtt` segment renditions only; an fMP4
/// `wvtt` subtitle rendition (the `IsoBmff` + `Mp4DemuxN` path) is a follow-up.
#[cfg(feature = "hls")]
pub fn build_hls_subtitle_overlay(
    reg: &Registry,
    master_url: &str,
    streams: &[crate::hlssrc::HlsStreamInfo],
    subtitle_url: &str,
) -> Result<Option<Graph<GraphNode>>, ParseError> {
    use crate::tsdemux::TsDemuxN;

    // The variant's muxed A/V streams (video + any muxed audio), in order.
    let mut ts_streams = Vec::new();
    let mut av: Vec<(Caps, bool)> = Vec::new();
    for s in streams.iter().filter(|s| s.uri.is_none()) {
        if let Some(ts) = hls_ts_stream(s) {
            ts_streams.push(ts);
            av.push((s.caps.clone(), s.video));
        }
    }
    let Some(video_idx) = av.iter().position(|(_, v)| *v) else {
        return Ok(None); // no routable muxed video to overlay onto
    };
    let outputs = ts_streams.len() as u8;

    let mut graph: Graph<GraphNode> = Graph::new();
    let src = graph.add_source(GraphNodeRef::source(crate::hlssrc::HlsSrc::new(master_url)));
    let demux = graph.add_demux(GraphNodeRef::demux(TsDemuxN::new(ts_streams)), outputs);
    graph.link(src, demux.input()).map_err(ParseError::Graph)?;

    let overlay = wire_overlay_av(reg, &mut graph, demux, &av, video_idx)?;

    // The subtitle rendition is a separate source: its WebVTT text flows through
    // SubParse into the overlay's text pad.
    let sub_src = graph.add_source(GraphNodeRef::source(
        crate::hlssrc::HlsSrc::new(subtitle_url).with_text(),
    ));
    link_text_into_overlay(
        &mut graph,
        overlay,
        sub_src.into(),
        &Caps::Text {
            format: g2g_core::TextFormat::WebVtt,
        },
    )?;
    Ok(Some(graph))
}

/// Assemble the *three-source* HLS subtitle-overlay graph (M420): a variant that
/// carries video in its own TS segments but pairs it with BOTH a separate audio
/// rendition and a SUBTITLES rendition. The video rides the variant's (video-only)
/// MPEG-TS segments through a `TsDemuxN` into a
/// [`TextOverlayN`](crate::textoverlay::TextOverlayN); the audio is a distinct
/// `#EXT-X-MEDIA` rendition (`HlsSrc -> TsDemuxN -> decode -> auto audio sink`); the
/// subtitle is another distinct WebVTT rendition (`HlsSrc(text) -> SubParse ->
/// overlay text pad`). Three independent sources in one graph, the union of
/// [`build_hls_separate_fanout`] (separate audio) and [`build_hls_subtitle_overlay`]
/// (cross-source text join). `streams` supplies the muxed video stream; `audio_url`
/// / `subtitle_url` are the resolved rendition playlists. `Ok(None)` (decline) when
/// there is no routable muxed video to overlay onto.
#[cfg(feature = "hls")]
pub fn build_hls_separate_subtitle_overlay(
    reg: &Registry,
    master_url: &str,
    streams: &[crate::hlssrc::HlsStreamInfo],
    audio_url: &str,
    subtitle_url: &str,
) -> Result<Option<Graph<GraphNode>>, ParseError> {
    use crate::tsdemux::{TsDemuxN, TsStream};

    // The variant's own (video-only) TS stream feeds the overlay's video pad.
    let Some(video) = streams.iter().find(|s| s.video && s.uri.is_none()) else {
        return Ok(None); // no routable muxed video to overlay onto
    };
    let Some(vts) = hls_ts_stream(video) else {
        return Ok(None);
    };

    let mut graph: Graph<GraphNode> = Graph::new();
    let src = graph.add_source(GraphNodeRef::source(crate::hlssrc::HlsSrc::new(master_url)));
    let demux = graph.add_demux(GraphNodeRef::demux(TsDemuxN::new(Vec::from([vts]))), 1);
    graph.link(src, demux.input()).map_err(ParseError::Graph)?;

    // A video-only variant: one A/V entry, the video, at index 0.
    let av = [(video.caps.clone(), true)];
    let overlay = wire_overlay_av(reg, &mut graph, demux, &av, 0)?;

    // The separate audio rendition: its own source -> demux -> decode -> auto sink.
    let audio_src = graph.add_source(GraphNodeRef::source(crate::hlssrc::HlsSrc::new(audio_url)));
    let audio_demux = graph.add_demux(
        GraphNodeRef::demux(TsDemuxN::new(Vec::from([TsStream::Aac]))),
        1,
    );
    graph
        .link(audio_src, audio_demux.input())
        .map_err(ParseError::Graph)?;
    let asink = reg
        .make_element("autoaudiosink")
        .ok_or_else(|| ParseError::UnknownElement("autoaudiosink".to_string()))?;
    let asnk = graph.add_sink(GraphNodeRef::Element(asink));
    let audio_caps = Caps::Audio {
        format: AudioFormat::Aac,
        channels: 0,
        sample_rate: 0,
    };
    reg.decodebin(
        &mut graph,
        audio_demux.out(0),
        asnk,
        &audio_caps,
        &is_raw_audio,
        PLAYBIN_MAX_DEPTH,
    )
    .map_err(|e| map_decode_err(&audio_caps, e))?;

    // The separate subtitle rendition: WebVTT text -> SubParse -> overlay text pad.
    let sub_src = graph.add_source(GraphNodeRef::source(
        crate::hlssrc::HlsSrc::new(subtitle_url).with_text(),
    ));
    link_text_into_overlay(
        &mut graph,
        overlay,
        sub_src.into(),
        &Caps::Text {
            format: g2g_core::TextFormat::WebVtt,
        },
    )?;
    Ok(Some(graph))
}

/// Assemble the muxed-TS HLS closed-caption overlay (M436): the CEA-608 / 708
/// sibling of [`build_hls_ts_fanout`]. The variant's muxed MPEG-TS streams fan out
/// through a `TsDemuxN`, the video port teed so one copy decodes for display and the
/// other feeds a [`CcExtract`](crate::ccextract::CcExtract) into a `TextOverlayN`, so
/// the in-SEI CEA-608 / 708 captions render on screen, the HLS analog of
/// [`ts_playbin`]'s `#closed-captions=`. Any muxed audio fans out to its own sink.
/// `Ok(None)` (decline) when no routable muxed video carries the captions.
#[cfg(feature = "hls")]
pub fn build_hls_ts_cc_overlay(
    reg: &Registry,
    source_url: &str,
    streams: &[crate::hlssrc::HlsStreamInfo],
    cc: crate::ccextract::CcSource,
) -> Result<Option<Graph<GraphNode>>, ParseError> {
    let mut ts_streams = Vec::new();
    let mut av: Vec<(Caps, bool)> = Vec::new();
    for s in streams.iter().filter(|s| s.uri.is_none()) {
        if let Some(ts) = hls_ts_stream(s) {
            ts_streams.push(ts);
            av.push((s.caps.clone(), s.video));
        }
    }
    let Some(video_idx) = av.iter().position(|(_, v)| *v) else {
        return Ok(None); // no routable muxed video to caption
    };
    let source = crate::hlssrc::HlsSrc::new(source_url);
    build_cc_overlay(
        reg,
        Box::new(source),
        crate::tsdemux::TsDemuxN::new(ts_streams),
        &av,
        video_idx,
        cc,
    )
    .map(Some)
}

/// Assemble the fMP4 / CMAF HLS closed-caption overlay (M436): the CEA-608 / 708
/// sibling of [`build_hls_fmp4_fanout`], tracks discovered from the `#EXT-X-MAP`
/// init segment's `moov`. The video track is teed for a display decode and a
/// `CcExtract` caption branch into a `TextOverlayN`; any audio track fans out to its
/// own sink. `Ok(None)` (decline) when the init carries no video track.
#[cfg(feature = "hls")]
pub fn build_hls_fmp4_cc_overlay(
    reg: &Registry,
    source_url: &str,
    init: &[u8],
    cc: crate::ccextract::CcSource,
) -> Result<Option<Graph<GraphNode>>, ParseError> {
    let infos = crate::mp4demuxn::forwardable_streams(init);
    let av: Vec<(Caps, bool)> = infos.iter().map(|i| (i.caps.clone(), i.video)).collect();
    let Some(video_idx) = av.iter().position(|(_, v)| *v) else {
        return Ok(None); // init segment carries no video track to caption
    };
    let demux_ports: Vec<_> = infos
        .iter()
        .map(|i| crate::mp4demuxn::Mp4Port {
            track_id: i.track_id,
            caps: i.caps.clone(),
        })
        .collect();
    let source = crate::hlssrc::HlsSrc::new(source_url);
    build_cc_overlay(
        reg,
        Box::new(source),
        crate::mp4demuxn::Mp4DemuxN::new(demux_ports),
        &av,
        video_idx,
        cc,
    )
    .map(Some)
}

/// Assemble the *multi-source* HLS closed-caption overlay (M436) for a variant whose
/// video rides its own TS segments while the audio is a separate `#EXT-X-MEDIA`
/// rendition: the video-only TS feeds the caption overlay (tee -> {decode for
/// display, `CcExtract` for the text pad} -> `TextOverlayN`) and the audio rendition
/// is its own `HlsSrc -> TsDemuxN -> decode -> auto sink` chain merged in, the
/// caption analog of [`build_hls_separate_fanout`]. `Ok(None)` (decline) when there
/// is no routable muxed video.
#[cfg(feature = "hls")]
pub fn build_hls_separate_cc_overlay(
    reg: &Registry,
    master_url: &str,
    streams: &[crate::hlssrc::HlsStreamInfo],
    audio_url: &str,
    cc: crate::ccextract::CcSource,
) -> Result<Option<Graph<GraphNode>>, ParseError> {
    use crate::tsdemux::{TsDemuxN, TsStream};

    // The variant's own (video-only) TS stream feeds the caption overlay.
    let Some(video) = streams.iter().find(|s| s.video && s.uri.is_none()) else {
        return Ok(None); // no routable muxed video to caption
    };
    let Some(vts) = hls_ts_stream(video) else {
        return Ok(None);
    };

    let mut graph: Graph<GraphNode> = Graph::new();
    let src = graph.add_source(GraphNodeRef::source(crate::hlssrc::HlsSrc::new(master_url)));
    let demux = graph.add_demux(GraphNodeRef::demux(TsDemuxN::new(Vec::from([vts]))), 1);
    graph.link(src, demux.input()).map_err(ParseError::Graph)?;
    let av = [(video.caps.clone(), true)];
    wire_cc_overlay(reg, &mut graph, demux, &av, 0, cc)?;

    // The separate audio rendition: its own source -> demux -> audio branch, merged.
    let audio_caps = Caps::Audio {
        format: AudioFormat::Aac,
        channels: 0,
        sample_rate: 0,
    };
    let audio_graph = build_av_fanout(
        reg,
        Box::new(crate::hlssrc::HlsSrc::new(audio_url)),
        TsDemuxN::new(Vec::from([TsStream::Aac])),
        &[(audio_caps, false)],
    )?;
    graph.merge(audio_graph);
    Ok(Some(graph))
}

/// Resolve the playlist URL of the chosen `#EXT-X-MEDIA` rendition in `group`
/// (M418 language pick: `language` -> DEFAULT -> first), absolute against
/// `base_url`. `None` when there is no group, no matching rendition, or the chosen
/// rendition has no `URI` (a track muxed into the variant rather than its own
/// playlist).
#[cfg(feature = "hls")]
fn resolve_rendition(
    master: &crate::hls::MasterPlaylist,
    base_url: &str,
    group: Option<&str>,
    media_type: crate::hls::MediaType,
    language: Option<&str>,
) -> Option<alloc::string::String> {
    let r = master.pick_rendition(media_type, group?, language)?;
    Some(crate::fetch::resolve_url(base_url, r.uri.as_ref()?))
}

/// Parse the `playbin uri=hls://...` rendition-language hints from the URI fragment
/// (M418): `#audio-lang=fr&subtitle-lang=en`. Returns the playlist `rest` with the
/// fragment stripped, plus the requested audio and subtitle languages (`None` when
/// absent). `subtitle-lang` / `text-lang` are accepted aliases. An HLS URL has no
/// other place to carry a preference, so the fragment (never sent to the server)
/// holds it.
#[cfg(feature = "hls")]
type HlsHints = (
    Option<alloc::string::String>,
    Option<alloc::string::String>,
    Option<crate::ccextract::CcSource>,
);

#[cfg(feature = "hls")]
fn hls_lang_hints(rest: &str) -> (&str, HlsHints) {
    let (url, frag) = rest.split_once('#').unwrap_or((rest, ""));
    let (mut audio, mut subtitle, mut cc) = (None, None, None);
    for kv in frag.split('&').filter(|s| !s.is_empty()) {
        if let Some((k, v)) = kv.split_once('=') {
            match k {
                "audio-lang" => audio = Some(v.to_string()),
                "subtitle-lang" | "text-lang" => subtitle = Some(v.to_string()),
                // The file hooks' `#closed-captions=` (alias `#cc=`) carries the same
                // in-SEI caption opt-in for HLS: captions are not discoverable from the
                // playlist (they ride the video SEI, not a rendition), so the fragment
                // selects the CEA-608 channel / CEA-708 service (M436).
                "closed-captions" | "cc" => cc = parse_cc_source(v),
                _ => {}
            }
        }
    }
    (url, (audio, subtitle, cc))
}

#[cfg(feature = "hls")]
pub fn hls_playbin(reg: &Registry, uri: &str) -> Result<Option<Graph<GraphNode>>, ParseError> {
    let Some(parsed) = Uri::parse(uri) else {
        return Ok(None);
    };
    if parsed.scheme != "hls" || parsed.rest.is_empty() {
        return Ok(None);
    }
    // A `#audio-lang=` / `#subtitle-lang=` URI fragment carries the rendition
    // language preference (HLS URIs have no other place for it); strip it off the
    // playlist URL (M418).
    let (url_rest, (audio_lang, subtitle_lang, cc)) = hls_lang_hints(parsed.rest);
    let source_url = alloc::format!("https://{url_rest}");
    // Fetch + parse the master; decline (single-stream fallback) on any failure.
    let Some(master_text) = blocking_get_text(&source_url) else {
        return Ok(None);
    };
    let Ok(crate::hls::Playlist::Master(master)) = crate::hls::parse(&master_text) else {
        return Ok(None); // a media playlist or parse error: single-stream handles it
    };
    let Some(variant) = master.select(None) else {
        return Ok(None);
    };
    // The variant's media playlist tells the container: an `#EXT-X-MAP` init
    // segment means fMP4 / CMAF (fan out via Mp4DemuxN, tracks from the init's
    // moov), otherwise muxed MPEG-TS (fan out via TsDemuxN, ports from CODECS).
    let media_url = crate::fetch::resolve_url(&source_url, &variant.uri);
    let Some(media_text) = blocking_get_text(&media_url) else {
        return Ok(None);
    };
    let media = match crate::hls::parse(&media_text) {
        Ok(crate::hls::Playlist::Media(m)) => m,
        _ => return Ok(None), // a master pointing at a master, or a parse error
    };
    if let Some(map) = &media.map_uri {
        let init_url = crate::fetch::resolve_url(&media_url, map);
        let Some(init) = blocking_get_bytes(&init_url) else {
            return Ok(None);
        };
        // An explicit `#closed-captions=` overlays the in-SEI captions onto the
        // fMP4 video (M436), the HLS analog of the file hooks' caption auto-plug.
        if let Some(cc) = cc {
            if let Some(g) = build_hls_fmp4_cc_overlay(reg, &source_url, &init, cc)? {
                return Ok(Some(g));
            }
        }
        return build_hls_fmp4_fanout(reg, &source_url, &init);
    }
    let streams = crate::hlssrc::variant_streams(&master, variant);
    // Resolve the chosen renditions (M418 language pick): a SUBTITLES WebVTT
    // rendition and/or a separate audio rendition, each absolute and `None` when the
    // variant binds no such group (or the rendition is muxed-in, no own `URI`). The
    // pair decides the graph shape below.
    let subtitle_url = resolve_rendition(
        &master,
        &source_url,
        variant.subtitles_group.as_deref(),
        crate::hls::MediaType::Subtitles,
        subtitle_lang.as_deref(),
    );
    let audio_url = resolve_rendition(
        &master,
        &source_url,
        variant.audio_group.as_deref(),
        crate::hls::MediaType::Audio,
        audio_lang.as_deref(),
    );

    // An explicit `#closed-captions=` request overlays the in-SEI captions (M436),
    // the HLS analog of the file hooks' caption auto-plug. Captions and subtitles
    // share the single overlay text pad, so an explicit caption request wins over a
    // subtitle rendition (matching the MKV / TS / MP4 hooks). The fMP4 variant is
    // handled above; here the variant is muxed TS, with an optional separate audio
    // rendition.
    if let Some(cc) = cc {
        if let Some(a) = &audio_url {
            if let Some(g) = build_hls_separate_cc_overlay(reg, &source_url, &streams, a, cc)? {
                return Ok(Some(g));
            }
        } else if let Some(g) = build_hls_ts_cc_overlay(reg, &source_url, &streams, cc)? {
            return Ok(Some(g));
        }
    }
    // Separate audio + SUBTITLES renditions: the three-source overlay, video from
    // the variant's TS, audio + subtitle each its own rendition source (M420).
    if let (Some(a), Some(s)) = (&audio_url, &subtitle_url) {
        if let Some(g) = build_hls_separate_subtitle_overlay(reg, &source_url, &streams, a, s)? {
            return Ok(Some(g));
        }
    }
    // A SUBTITLES rendition with muxed A/V (no separate audio): overlay the chosen
    // rendition onto the video (M419, the cross-source join).
    if audio_url.is_none() {
        if let Some(s) = &subtitle_url {
            if let Some(g) = build_hls_subtitle_overlay(reg, &source_url, &streams, s)? {
                return Ok(Some(g));
            }
        }
    }
    // A separate audio rendition (no subtitle): the two-source A/V fan-out, the
    // variant carries video, the audio is a distinct rendition (M397).
    if let Some(a) = &audio_url {
        return build_hls_separate_fanout(reg, &source_url, &streams, a);
    }
    build_hls_ts_fanout(reg, &source_url, &streams)
}

#[cfg(all(test, feature = "hls"))]
mod hls_hint_tests {
    use super::hls_lang_hints;

    #[test]
    fn parses_language_hints_from_the_uri_fragment() {
        use super::parse_cc_source;

        // Both hints present: the URL is returned fragment-free.
        let (url, (a, s, cc)) = hls_lang_hints("host/master.m3u8#audio-lang=fr&subtitle-lang=en");
        assert_eq!(url, "host/master.m3u8");
        assert_eq!(a.as_deref(), Some("fr"));
        assert_eq!(s.as_deref(), Some("en"));
        assert!(cc.is_none());

        // `text-lang` is an accepted alias for the subtitle hint; order-independent.
        let (_, (a, s, cc)) = hls_lang_hints("h/x.m3u8#text-lang=de");
        assert_eq!(a, None);
        assert_eq!(s.as_deref(), Some("de"));
        assert!(cc.is_none());

        // A `#closed-captions=` (alias `#cc=`) opt-in selects the in-SEI caption
        // channel / service, alongside the language hints (M436).
        let (url, (a, _s, cc)) = hls_lang_hints("h/x.m3u8#audio-lang=fr&closed-captions=cc1");
        assert_eq!(url, "h/x.m3u8");
        assert_eq!(a.as_deref(), Some("fr"));
        assert_eq!(cc, parse_cc_source("cc1"));
        let (_, (_, _, cc)) = hls_lang_hints("h/x.m3u8#cc=service-2");
        assert_eq!(cc, parse_cc_source("service-2"));

        // No fragment: no preferences, URL unchanged.
        let (url, (a, s, cc)) = hls_lang_hints("h/x.m3u8");
        assert_eq!(url, "h/x.m3u8");
        assert!(a.is_none() && s.is_none() && cc.is_none());
    }
}
