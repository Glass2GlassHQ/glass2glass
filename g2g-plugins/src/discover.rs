// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! Media discovery: what a file or URI contains, without hand-writing a
//! graph. The `gst-discoverer-1.0` analog behind the `g2g-discover` binary.
//!
//! [`discover`] sniffs the container with [`crate::typefind`], builds a headless
//! probe graph (`<source> ! <demuxer> ! probe sink`) with a [`Bus`] and a
//! [`PipelineProgress`] handle attached, and runs it only until the first
//! payload frame reaches the sink. Every demuxer parses its header before
//! forwarding a payload, so by that point it has posted its
//! [`StreamCollection`](g2g_core::stream::StreamCollection) and metadata, and
//! the runner has published the source's duration. Nothing is decoded: the
//! answer comes from container headers and tags. MP4 reads the movie box
//! directly, and MP3 also checks the tail for an ID3v1 fallback.
//!
//! Which demuxer to plug, and which stream it should forward, comes from the
//! registry's primary-stream hooks
//! ([`Registry::container_streams`]), the same probe `decodebin` uses.

use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;
use core::future::Future;
use core::pin::Pin;
use std::io::{Read, Seek, SeekFrom};

use g2g_core::element::DynAsyncElement;
use g2g_core::runtime::{
    apply_props, run_graph_with_progress, DynMultiOutputElement, DynSourceLoop, GraphNodeRef,
    PipelineProgress, Registry,
};
use g2g_core::stream::StreamType;
use g2g_core::{
    AsyncElement, AudioFormat, Bus, BusMessage, Caps, CapsConstraint, ConfigureOutcome,
    ElementMetadata, G2gError, Graph, OutputSink, PipelinePacket, TagList,
};

use crate::clock::WallClock;
use crate::typefind::sniff_caps;

/// Depth of every probe link. One packet in flight is all the probe needs, and a
/// deeper link only buys frames nobody reads.
const PROBE_LINK_CAPACITY: usize = 1;
/// Bus backlog. A demuxer posts a collection, a global tag list and one tag list
/// per stream, and a full bus silently drops messages, so leave room for a
/// container with many tracks.
const PROBE_BUS_CAPACITY: usize = 64;
/// Id given to the one stream of a container that announces no track list, so a
/// report always names its streams even without a demuxer's collection.
const LONE_STREAM_ID: &str = "stream-0";

/// One elementary stream of the probed media.
#[derive(Debug, Clone, PartialEq)]
pub struct DiscoveredStream {
    /// The demuxer's stable id for this stream (`matroska-track-1`), the name a
    /// stream selection takes.
    pub id: String,
    pub stream_type: StreamType,
    /// What the stream carries: codec plus geometry / framerate for video,
    /// format plus channels / sample rate for audio.
    pub caps: Caps,
    /// Metadata scoped to this stream, empty when the container has none.
    pub tags: TagList,
}

/// Everything a probe learned about one file or URI.
#[derive(Debug, Clone, PartialEq)]
pub struct Discovery {
    /// The URI as given.
    pub uri: String,
    /// The media type the content sniff decided, in GStreamer's spelling
    /// (`video/x-matroska`, `audio/x-wav`).
    pub container: String,
    /// Total duration in nanoseconds, when the source reported one. `None` for a
    /// container whose header does not carry a length, or whose g2g source does
    /// not answer a duration query.
    pub duration_ns: Option<u64>,
    pub streams: Vec<DiscoveredStream>,
    /// Metadata describing the whole file.
    pub tags: TagList,
}

/// Why a probe could not answer.
#[derive(Debug, Clone, PartialEq)]
pub enum DiscoverError {
    /// A URI whose scheme has no local-file probe. Carries the scheme.
    UnsupportedScheme(String),
    /// The file could not be opened or read.
    Io(String),
    /// Nothing in the header matched a known media type.
    UnknownType,
    /// The type is known but this build has no element that parses it. Carries
    /// the sniffed media type.
    NoParser(String),
    /// The probe graph could not be built, or failed before it learned anything.
    Probe(String),
}

impl fmt::Display for DiscoverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiscoverError::UnsupportedScheme(scheme) => write!(
                f,
                "{scheme}: g2g-discover reads local files only (a path or a file:// URI)"
            ),
            DiscoverError::Io(msg) => write!(f, "{msg}"),
            DiscoverError::UnknownType => {
                write!(f, "no media type matched the file header")
            }
            DiscoverError::NoParser(media_type) => {
                write!(f, "{media_type}: this build has no element that parses it")
            }
            DiscoverError::Probe(msg) => write!(f, "probe pipeline failed: {msg}"),
        }
    }
}

/// The local path a discover argument names: a bare filesystem path, or a
/// `file://` URI. Any other scheme is refused by name rather than guessed at.
fn local_path(uri: &str) -> Result<String, DiscoverError> {
    if let Some(rest) = uri.strip_prefix("file://") {
        // `file:///clip.mp4` (empty authority) and `file://clip.mp4` both reach
        // the path after the last leading slash pair.
        return Ok(crate::uridecodebin::file_uri_rest_to_path(rest).to_string());
    }
    match uri.split_once("://") {
        Some((scheme, _)) => Err(DiscoverError::UnsupportedScheme(scheme.to_string())),
        None => Ok(uri.to_string()),
    }
}

/// The concrete element behind a demux name, with the pipeline bus attached so
/// it announces its stream collection and tags. The names are the ones the
/// registry's primary-stream hooks and `demux_name_for` return.
fn demux_with_bus(name: &str, bus: g2g_core::BusHandle) -> Option<Box<dyn DynAsyncElement>> {
    Some(match name {
        "matroskademux" => Box::new(crate::mkvdemux::MkvDemux::new().with_bus(bus)),
        "tsdemux" => Box::new(crate::tsdemux::TsDemux::new().with_bus(bus)),
        "oggdemux" => Box::new(crate::oggdemux::OggDemux::new().with_bus(bus)),
        "avidemux" => Box::new(crate::avidemux::AviDemux::new().with_bus(bus)),
        "mpegpsdemux" => Box::new(crate::psdemux::PsDemux::new().with_bus(bus)),
        "flvdemux" => Box::new(crate::flvdemux::FlvDemux::new().with_bus(bus)),
        "wavparse" => Box::new(crate::wavparse::WavParse::new().with_bus(bus)),
        _ => return None,
    })
}

/// The parser for a container with a single elementary stream, which the
/// primary-stream hooks do not cover (they exist to pick one stream of several).
/// `None` for content that needs no parser at all: a bare elementary stream
/// already types itself in the sniff.
fn demux_name_for(caps: &Caps) -> Option<&'static str> {
    let Caps::ByteStream { encoding } = caps else {
        return None;
    };
    match encoding {
        g2g_core::ByteStreamEncoding::Wav => Some("wavparse"),
        g2g_core::ByteStreamEncoding::Flv => Some("flvdemux"),
        _ => None,
    }
}

enum ProbeDemux {
    Single(Box<dyn DynAsyncElement>),
    Fanout {
        element: Box<dyn DynMultiOutputElement>,
        outputs: usize,
    },
}

/// Terminal of the probe graph. Accepts whatever the demuxer emits, keeps the
/// caps it was configured with (refined by any `CapsChanged`), and ends the run
/// at the first payload frame: the header is parsed by then, so reading further
/// only costs I/O.
#[derive(Debug, Default)]
struct ProbeSink {
    caps: Option<Caps>,
}

impl AsyncElement for ProbeSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.caps = Some(absolute_caps.clone());
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Discovery probe sink",
            "Sink",
            "Records negotiated caps and stops the probe at the first frame",
            crate::AARON_BOXER,
        )
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::CapsChanged(caps) => {
                    self.caps = Some(caps);
                    Ok(())
                }
                // The probe is finished: everything the header carries has
                // already crossed the bus. `discover` reads this as the stop.
                PipelinePacket::DataFrame(_) => Err(G2gError::Shutdown),
                _ => Ok(()),
            }
        })
    }
}

/// Probe `uri` and report what it holds: media type, elementary streams with
/// their caps, duration when the source knows one, and container metadata.
///
/// Local files only. A URI with any other scheme is refused
/// ([`DiscoverError::UnsupportedScheme`]) rather than fetched, so a probe never
/// opens a socket.
pub async fn discover(reg: &Registry, uri: &str) -> Result<Discovery, DiscoverError> {
    let path = local_path(uri)?;
    let header = read_header(&path)?;
    let sniffed = sniff_caps(&header).ok_or(DiscoverError::UnknownType)?;
    let container = sniffed.to_gst_string();
    if is_mp4(&sniffed) {
        return discover_mp4(uri, &path, container);
    }
    let audio_format = match sniffed {
        Caps::Audio { format, .. } => Some(format),
        _ => None,
    };

    // Which demuxer parses this container, and which of its streams it should
    // forward: the same probe `decodebin` runs, so an audio-only file plugs an
    // audio port instead of failing on the default video one.
    let primary = reg
        .container_streams(&path, &sniffed)
        .and_then(|streams| streams.streams.into_iter().next());
    let demux_name = primary
        .as_ref()
        .map(|p| p.demux)
        .or_else(|| demux_name_for(&sniffed));

    let (bus, bus_handle) = Bus::new(PROBE_BUS_CAPACITY);

    let source: Box<dyn DynSourceLoop> =
        Box::new(crate::filesrc::FileSrc::new(&path, sniffed.clone()));

    // FLAC and MPEG audio carry tags the sniff does not read. Other bare
    // elementary streams use the caps from the sniff alone.
    let demux = if is_mxf(&sniffed) {
        let prefix = crate::uridecodebin::read_prefix(&path)
            .ok_or_else(|| DiscoverError::Io(format!("{path}: cannot read the MXF header")))?;
        let tracks = crate::mxfdemux::forwardable_tracks(&prefix);
        if tracks.is_empty() {
            return Err(DiscoverError::NoParser(container));
        }
        Some(ProbeDemux::Fanout {
            outputs: tracks.len(),
            element: Box::new(crate::mxfdemux::MxfDemuxN::new(tracks)),
        })
    } else if matches!(sniffed, Caps::ByteStream { .. }) {
        let name = demux_name.ok_or_else(|| DiscoverError::NoParser(container.clone()))?;
        let mut element = demux_with_bus(name, bus_handle.clone())
            .ok_or_else(|| DiscoverError::NoParser(container.clone()))?;
        if let Some(primary) = &primary {
            apply_props(&mut element, name, &primary.props).map_err(|e| {
                DiscoverError::Probe(format!("{name} rejected its stream selection: {e}"))
            })?;
        }
        Some(ProbeDemux::Single(element))
    } else {
        let element: Option<Box<dyn DynAsyncElement>> = match audio_format {
            Some(AudioFormat::Flac) => Some(Box::new(
                crate::flacparse::FlacParse::new().with_bus(bus_handle.clone()),
            )),
            Some(AudioFormat::Mp3 | AudioFormat::Mp2) => Some(Box::new(
                crate::mpegaudioparse::MpegAudioParse::new().with_bus(bus_handle.clone()),
            )),
            _ => None,
        };
        element.map(ProbeDemux::Single)
    };
    let progress = PipelineProgress::new();
    let sink_count = match &demux {
        Some(ProbeDemux::Fanout { outputs, .. }) => *outputs,
        Some(ProbeDemux::Single(_)) | None => 1,
    };
    let mut sinks: Vec<ProbeSink> = (0..sink_count).map(|_| ProbeSink::default()).collect();
    let mut graph: Graph<GraphNodeRef<'_>> = Graph::new();
    let source_node = graph.add_source(GraphNodeRef::Source(source));
    let sink_nodes: Vec<g2g_core::NodeId> = sinks
        .iter_mut()
        .map(|sink| graph.add_sink(GraphNodeRef::element_ref(sink)))
        .collect();
    match demux {
        Some(ProbeDemux::Fanout { element, outputs }) => {
            let demux = graph.add_demux(GraphNodeRef::Demux(element), outputs as u8);
            link(&mut graph, source_node, demux.input())?;
            for (port, sink_node) in sink_nodes.iter().enumerate() {
                link(&mut graph, demux.out(port as u8), *sink_node)?;
            }
        }
        Some(ProbeDemux::Single(element)) => {
            let demux_node = graph.add_transform(GraphNodeRef::Element(element));
            link(&mut graph, source_node, demux_node)?;
            link(&mut graph, demux_node, sink_nodes[0])?;
        }
        None => link(&mut graph, source_node, sink_nodes[0])?,
    }

    let clock = WallClock::new();
    let outcome = run_graph_with_progress(
        graph,
        &clock,
        PROBE_LINK_CAPACITY,
        &progress,
        Some(&bus_handle),
    )
    .await;
    drop(bus_handle);

    let mut info = harvest(uri, container, &bus, &progress);
    if audio_format == Some(AudioFormat::Mp3) && info.tags.is_empty() {
        if let Some(tags) = trailing_id3_tags(&path) {
            info.tags = tags;
        }
    }
    if info.streams.is_empty() {
        // No demuxer announced a collection (a single-stream container like WAV,
        // or a bare elementary stream). The caps the probe negotiated describe
        // that one stream.
        if let Some(caps) = sinks[0].caps.take() {
            info.streams.push(DiscoveredStream {
                id: LONE_STREAM_ID.to_string(),
                stream_type: StreamType::of(&caps),
                caps,
                tags: TagList::new(),
            });
        }
    }
    // The probe sink ends the run with `Shutdown` once the first frame arrives;
    // a short file reaches EOS first. Any other error may still have let the
    // header cross the bus, so it is only fatal when nothing was learned.
    match outcome {
        Ok(_) | Err(G2gError::Shutdown) => Ok(info),
        Err(err) if info.streams.is_empty() => Err(DiscoverError::Probe(format!("{err:?}"))),
        Err(_) => Ok(info),
    }
}

fn trailing_id3_tags(path: &str) -> Option<TagList> {
    let mut file = std::fs::File::open(path).ok()?;
    let mut block = [0; crate::id3::ID3V1_LEN];
    file.seek(SeekFrom::End(-(block.len() as i64))).ok()?;
    file.read_exact(&mut block).ok()?;
    crate::id3::parse_id3v1(&block)
}

fn discover_mp4(uri: &str, path: &str, container: String) -> Result<Discovery, DiscoverError> {
    let data = crate::uridecodebin::read_mp4_top_level_box(path, b"moov")
        .ok_or_else(|| DiscoverError::Probe(format!("{path}: cannot read the MP4 movie box")))?;
    let tracks = crate::fmp4::parse_all_tracks(&data)
        .map_err(|error| DiscoverError::Probe(format!("{path}: invalid MP4 tracks: {error:?}")))?;
    let moov = crate::mp4box::find_box(&data, b"moov").expect("read_mp4_top_level_box found moov");
    let streams = tracks
        .into_iter()
        .map(|track| {
            let caps = crate::mp4demuxn::real_caps(&track.kind);
            DiscoveredStream {
                id: crate::mp4demuxn::stream_id(track.track_id),
                stream_type: StreamType::of(&caps),
                caps,
                tags: track.tags,
            }
        })
        .collect();
    let duration_ns = crate::fmp4::movie_duration_ns(moov)
        .or_else(|| crate::fmp4::parse_header(&data).ok()?.duration_ns);
    Ok(Discovery {
        uri: uri.to_string(),
        container,
        duration_ns,
        streams,
        tags: crate::mp4box::parse_ilst_tags(moov),
    })
}

/// The ISO base media family.
fn is_mp4(caps: &Caps) -> bool {
    matches!(
        caps,
        Caps::ByteStream {
            encoding: g2g_core::ByteStreamEncoding::Mp4 | g2g_core::ByteStreamEncoding::IsoBmff,
        }
    )
}

fn is_mxf(caps: &Caps) -> bool {
    matches!(
        caps,
        Caps::ByteStream {
            encoding: g2g_core::ByteStreamEncoding::Mxf,
        }
    )
}

fn link(
    graph: &mut Graph<GraphNodeRef<'_>>,
    from: impl Into<g2g_core::graph::PadId>,
    to: impl Into<g2g_core::graph::PadId>,
) -> Result<(), DiscoverError> {
    graph
        .link(from, to)
        .map_err(|e| DiscoverError::Probe(format!("{e:?}")))
}

/// Read the header bytes the content sniff needs. A file shorter than that is
/// read whole.
fn read_header(path: &str) -> Result<Vec<u8>, DiscoverError> {
    let mut file =
        std::fs::File::open(path).map_err(|e| DiscoverError::Io(format!("{path}: {e}")))?;
    crate::typefind::read_sniff_window(&mut file)
        .map_err(|e| DiscoverError::Io(format!("{path}: {e}")))
}

/// Fold every message the probe posted into the report: the stream collection,
/// the file's tags, and each stream's own tags.
fn harvest(uri: &str, container: String, bus: &Bus, progress: &PipelineProgress) -> Discovery {
    let mut streams: Vec<DiscoveredStream> = Vec::new();
    let mut tags = TagList::new();
    let mut duration_ns = progress.duration();
    while let Some(message) = bus.try_recv() {
        match message {
            // The first collection is the container's own; a demuxer posts it
            // once, and a multi-program transport stream posts one per program.
            BusMessage::StreamCollection(collection) if streams.is_empty() => {
                streams = collection
                    .streams
                    .into_iter()
                    .map(|s| DiscoveredStream {
                        id: s.id,
                        stream_type: s.stream_type,
                        caps: s.caps,
                        tags: TagList::new(),
                    })
                    .collect();
            }
            BusMessage::Tag { tags: posted, .. } => extend(&mut tags, posted),
            BusMessage::StreamTag {
                stream_id,
                tags: posted,
            } => {
                if let Some(stream) = streams.iter_mut().find(|s| s.id == stream_id) {
                    extend(&mut stream.tags, posted);
                }
            }
            BusMessage::DurationChanged { duration_ns: ns } => duration_ns = Some(ns),
            _ => {}
        }
    }
    Discovery {
        uri: uri.to_string(),
        container,
        duration_ns,
        streams,
        tags,
    }
}

fn extend(target: &mut TagList, posted: TagList) {
    for tag in posted.tags() {
        target.push(tag.clone());
    }
}

/// Render a report the way `gst-discoverer-1.0` does: the file, its duration and
/// container, then one block per stream.
pub fn to_text(info: &Discovery) -> String {
    let mut out = format!("Analyzing {}\n", info.uri);
    out.push_str(&format!("  container: {}\n", info.container));
    match info.duration_ns {
        Some(ns) => out.push_str(&format!("  duration: {}\n", format_duration(ns))),
        None => out.push_str("  duration: unknown\n"),
    }
    for tag in info.tags.tags() {
        out.push_str(&format!("  {}: {}\n", tag.key(), tag.value_string()));
    }
    out.push_str(&format!("  streams: {}\n", info.streams.len()));
    for stream in &info.streams {
        out.push_str(&format!(
            "    {} ({}): {}\n",
            stream.id,
            stream.stream_type.name(),
            stream.caps.to_gst_string()
        ));
        for tag in stream.tags.tags() {
            out.push_str(&format!("      {}: {}\n", tag.key(), tag.value_string()));
        }
    }
    out
}

/// `H:MM:SS.mmm`, the form `gst-discoverer-1.0` prints a duration in.
fn format_duration(ns: u64) -> String {
    let total_ms = ns / 1_000_000;
    let hours = total_ms / 3_600_000;
    let minutes = (total_ms / 60_000) % 60;
    let seconds = (total_ms / 1_000) % 60;
    let millis = total_ms % 1_000;
    format!("{hours}:{minutes:02}:{seconds:02}.{millis:03}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_network_uri_is_refused_by_scheme() {
        assert_eq!(
            local_path("rtsp://camera/stream"),
            Err(DiscoverError::UnsupportedScheme("rtsp".to_string()))
        );
    }

    #[test]
    fn a_file_uri_and_a_bare_path_both_resolve() {
        assert_eq!(local_path("file:///tmp/clip.mp4").unwrap(), "/tmp/clip.mp4");
        assert_eq!(local_path("/tmp/clip.mp4").unwrap(), "/tmp/clip.mp4");
        let drive_path = if cfg!(windows) {
            "C:/clip.mp4"
        } else {
            "/C:/clip.mp4"
        };
        assert_eq!(local_path("file:///C:/clip.mp4").unwrap(), drive_path);
        assert_eq!(local_path("file://C:/clip.mp4").unwrap(), "C:/clip.mp4");
    }

    #[test]
    fn duration_renders_as_hours_minutes_seconds() {
        assert_eq!(format_duration(0), "0:00:00.000");
        assert_eq!(format_duration(1_500_000_000), "0:00:01.500");
        assert_eq!(format_duration(3_723_400_000_000), "1:02:03.400");
    }
}
