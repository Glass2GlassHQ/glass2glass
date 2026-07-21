//! Matroska / WebM demuxer element (M110): `Caps::ByteStream{Matroska}` in, one
//! selected elementary stream out. Video leaves as `Caps::CompressedVideo`
//! (H.264 / H.265 / VP8 / VP9 / AV1), audio as `Caps::Audio` (AAC / Opus).
//!
//! Wraps the pure [`crate::matroska::MatroskaDemuxer`] parser, the MKV sibling of
//! [`crate::tsdemux::TsDemux`]. The parser reassembles every track the Segment's
//! Tracks element names; this element has one output pad, so a [`MkvStream`]
//! selection picks which to emit, and a second `mkvdemux` selecting another
//! stream demuxes the rest of the file. CPU, `no_std` baseline.
//!
//! ```text
//! filesrc(location=x.webm, caps=ByteStream{Matroska}) ! mkvdemux stream=vp9 ! <decoder> ! <sink>
//! mkvdemux stream=opus ! <audio>
//! ```
//!
//! The selection is by codec, like `TsDemux`, because the output pad's media type
//! is fixed at negotiation before any byte is parsed. Unlike `TsDemux`, the
//! Tracks element carries concrete geometry / audio parameters, so once parsed
//! the demuxer refines the caps itself via `CapsChanged` (no downstream parser
//! needed for the dimensions). The default is `Vp9`, WebM's video codec; set the
//! `stream` property to match the file.
//!
//! Seeking uses the `Cues` index: with it parsed (M373) the demuxer byte-seeks
//! straight to the target Cluster, keeping its parser state. When `Cues` sit at
//! the end of the file (the common layout) and a `SeekHead` at the start locates
//! them, a seek first prefetches them (a byte-seek to `Cues`, parse, then a
//! byte-seek to the target, M374) so the first seek is index-fast without reading
//! the whole file. With neither, it falls back to the M364 re-scan from offset 0.
//!
//! Scope (v1): the first track of the selected codec; multi-track-of-one-codec
//! selection and lacing are follow-ups (DESIGN.md §4.17).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{SeekController, StreamSelectController};
use g2g_core::{
    AsyncElement, AudioFormat, BusHandle, BusMessage, ByteStreamEncoding, Caps, CapsConstraint,
    CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError, MemoryDomain, MultiOutputElement,
    MultiOutputSink, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec, Rate, Seek, Segment, Stream, StreamCollection, StreamType, TagList,
    TextFormat, VideoCodec,
};

use crate::demuxseek::{Admit, DemuxSeek};
use crate::matroska::{MatroskaDemuxer, MkvCodec};

/// Which elementary stream a [`MkvDemux`] instance forwards. One output pad
/// carries one stream; the choice is by codec because the output caps are fixed
/// at negotiation before any byte is parsed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MkvStream {
    H264,
    H265,
    Vp8,
    /// The first VP9 video stream. The default (WebM's video codec).
    #[default]
    Vp9,
    Av1,
    Aac,
    Opus,
    /// The first AC-3 audio stream (`A_AC3`).
    Ac3,
    /// The first FLAC audio stream (`A_FLAC`); its `CodecPrivate` `fLaC` header
    /// is forwarded in-band before the first frame (decoder extradata).
    Flac,
    /// A timed-text subtitle stream (`S_TEXT/UTF8` -> `Caps::Text { format }`).
    Subtitle(TextFormat),
}

/// A Matroska `Cues` prefetch in flight (M374): an end-of-file `Cues` index
/// located by a `SeekHead` is fetched via a byte-seek *before* the real seek, so
/// the demuxer can then jump straight to the target Cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CuePrefetch {
    Idle,
    /// Byte-seek to the `Cues` element issued; `flushed` flips on the upstream
    /// flush, after which incoming bytes are the `Cues` element to parse.
    Fetching {
        target_ns: u64,
        flushed: bool,
    },
}

/// Demuxes a Matroska / WebM byte stream into one selected elementary stream.
#[derive(Debug)]
pub struct MkvDemux {
    demux: MatroskaDemuxer,
    stream: MkvStream,
    configured: bool,
    emitted: u64,
    last_caps: Option<Caps>,
    bus: Option<BusHandle>,
    /// Count of tags already posted, so newly parsed tags (the `Tags` element
    /// can trail the `Info` `Title`) post once each.
    tags_posted: usize,
    /// Set once the `StreamCollection` has been announced (M376), so the demuxer
    /// posts the available-streams list once, when the `Tracks` element parses.
    collection_posted: bool,
    /// App-driven stream selection (M377): the app names the stream id to forward
    /// (from the announced collection); the demuxer switches its single output to
    /// that track. Inert unless `with_stream_select` wired it.
    stream_select: Option<StreamSelectController>,
    /// Seek support (M362): app time seeks drive an upstream byte-seek and a
    /// re-sync. Inert unless `with_seek` wired the controllers.
    seek: DemuxSeek,
    /// FLAC only: whether the track's `CodecPrivate` (the native `fLaC`
    /// STREAMINFO header) has been forwarded in-band, once, before the first
    /// frame (the decoder takes it as extradata). Re-armed on a flush.
    flac_header_sent: bool,
    /// Clones of the seek controllers (M374): the `Cues` prefetch consumes the app
    /// seek and drives the two-hop upstream byte-seek directly, so it needs the
    /// same channels `seek` holds. `None` unless `with_seek` wired them.
    app: Option<SeekController>,
    upstream: Option<SeekController>,
    /// `Cues`-prefetch state (M374). `Idle` outside a prefetch.
    prefetch: CuePrefetch,
}

impl Default for MkvDemux {
    fn default() -> Self {
        Self::new()
    }
}

impl MkvDemux {
    pub fn new() -> Self {
        Self {
            demux: MatroskaDemuxer::new(),
            stream: MkvStream::default(),
            configured: false,
            emitted: 0,
            last_caps: None,
            bus: None,
            tags_posted: 0,
            collection_posted: false,
            stream_select: None,
            seek: DemuxSeek::default(),
            flac_header_sent: false,
            app: None,
            upstream: None,
            prefetch: CuePrefetch::Idle,
        }
    }

    /// Make the demuxer seekable (M362): `app` carries app time seeks; `upstream`
    /// is the byte source's ([`FileSrc`](crate::filesrc)) byte-seek controller.
    /// On a time seek the demuxer rewinds the source and re-syncs from the
    /// keyframe at/after the target; with a `Cues` index (parsed, or located via a
    /// `SeekHead` and prefetched, M373/M374) it jumps straight to the Cluster.
    pub fn with_seek(mut self, app: SeekController, upstream: SeekController) -> Self {
        self.app = Some(app.clone());
        self.upstream = Some(upstream.clone());
        self.seek.with(app, upstream);
        self
    }

    /// Service a pending app seek (M373/M374). Three paths, cheapest first:
    /// `Cues` already parsed -> seek straight to the target Cluster; only a
    /// `SeekHead` locating `Cues` -> prefetch them (byte-seek to `Cues`, then to
    /// the target); neither -> re-scan from offset 0. A no-op while a seek or
    /// prefetch is already in flight, or with no pending app seek.
    fn poll_seek(&mut self) {
        if self.prefetch != CuePrefetch::Idle || self.seek.is_seeking() {
            return;
        }
        match &self.app {
            Some(app) if app.has_pending() => {}
            _ => return,
        }
        if !self.demux.cues().is_empty() {
            // Index known: seek directly to the target Cluster.
            let demux = &self.demux;
            self.seek.poll_request_indexed(|t| demux.cue_seek_offset(t));
        } else if let Some(cues_off) = self.demux.cue_index_offset() {
            // Index located but not parsed: prefetch it before the real seek.
            if let (Some(app), Some(upstream)) = (&self.app, &self.upstream) {
                if let Some(seek) = app.take_pending() {
                    upstream.seek(Seek::flush_to(cues_off));
                    self.prefetch = CuePrefetch::Fetching {
                        target_ns: seek.start,
                        flushed: false,
                    };
                }
            }
        } else {
            // No index at all: re-scan from the start (M364).
            self.seek.poll_request_indexed(|_| None);
        }
    }

    /// Reset the parser for a discontinuity (a `Flush` / seek): drop the
    /// demuxer's EBML/Cluster state, which the re-read stream re-establishes from
    /// its EBML header. The codec / caps and posted tags are unchanged (same
    /// file), so `last_caps` / `tags_posted` are kept (no redundant re-emit).
    fn reset_parser(&mut self) {
        self.demux = MatroskaDemuxer::new();
    }

    /// Select which elementary stream to forward (default [`MkvStream::Vp9`]).
    pub fn with_stream(mut self, stream: MkvStream) -> Self {
        self.stream = stream;
        self
    }

    /// Attach the pipeline bus so the Segment's `Tags` / `Info` `Title` metadata
    /// posts as a [`BusMessage::Tag`] once parsed.
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.bus = Some(bus);
        self
    }

    /// Make the demuxer's output stream app-selectable (M377): the controller
    /// carries a selection of stream ids (from the announced collection); the
    /// demuxer switches its single output to the named track and confirms the
    /// active id on the bus ([`BusMessage::StreamsSelected`]). Pair with
    /// [`with_bus`](Self::with_bus) so the app can discover the ids first.
    pub fn with_stream_select(mut self, select: StreamSelectController) -> Self {
        self.stream_select = Some(select);
        self
    }

    /// The elementary stream this instance forwards.
    pub fn stream(&self) -> MkvStream {
        self.stream
    }

    /// Count of frames forwarded.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    /// The input this element accepts: a Matroska / WebM byte stream.
    fn input_caps() -> Caps {
        Caps::ByteStream {
            encoding: ByteStreamEncoding::Matroska,
        }
    }

    /// The placeholder output caps for the selected stream at negotiation. Video
    /// geometry is unknown until Tracks is parsed, so it advertises a fixatable
    /// `Range` refined via `CapsChanged`; audio advertises a sentinel
    /// channels/rate, likewise refined once Tracks lands.
    fn output_caps(stream: MkvStream) -> Caps {
        match stream {
            MkvStream::H264 => Self::compressed_video(VideoCodec::H264),
            MkvStream::H265 => Self::compressed_video(VideoCodec::H265),
            MkvStream::Vp8 => Self::compressed_video(VideoCodec::Vp8),
            MkvStream::Vp9 => Self::compressed_video(VideoCodec::Vp9),
            MkvStream::Av1 => Self::compressed_video(VideoCodec::Av1),
            MkvStream::Aac => Caps::Audio {
                format: AudioFormat::Aac,
                channels: 0,
                sample_rate: 0,
            },
            MkvStream::Ac3 => Caps::Audio {
                format: AudioFormat::Ac3,
                channels: 0,
                sample_rate: 0,
            },
            MkvStream::Flac => Caps::Audio {
                format: AudioFormat::Flac,
                channels: 0,
                sample_rate: 0,
            },
            MkvStream::Opus => Caps::Audio {
                format: AudioFormat::Opus,
                channels: 0,
                sample_rate: 0,
            },
            // Every subtitle stream is de-framed to plain UTF-8 cue text at emit
            // (the source syntax, `S_TEXT/UTF8` / `ASS` / `WEBVTT`, only selects the
            // de-framing), so the forwarded caps is always `Text { Utf8 }`.
            MkvStream::Subtitle(_) => Caps::Text {
                format: TextFormat::Utf8,
            },
        }
    }

    fn compressed_video(codec: VideoCodec) -> Caps {
        Caps::CompressedVideo {
            codec,
            width: Dim::Range {
                min: 16,
                max: 65_535,
            },
            height: Dim::Range {
                min: 16,
                max: 65_535,
            },
            framerate: Rate::Range {
                min_q16: 1 << 16,
                max_q16: 240 << 16,
            },
        }
    }

    /// The parser codec matching the selected stream.
    fn selected_codec(stream: MkvStream) -> MkvCodec {
        match stream {
            MkvStream::H264 => MkvCodec::H264,
            MkvStream::H265 => MkvCodec::H265,
            MkvStream::Vp8 => MkvCodec::Vp8,
            MkvStream::Vp9 => MkvCodec::Vp9,
            MkvStream::Av1 => MkvCodec::Av1,
            MkvStream::Aac => MkvCodec::Aac,
            MkvStream::Opus => MkvCodec::Opus,
            MkvStream::Ac3 => MkvCodec::Ac3,
            MkvStream::Flac => MkvCodec::Flac,
            MkvStream::Subtitle(format) => MkvCodec::Subtitle(format),
        }
    }

    /// The concrete caps for the selected track once Tracks has been parsed,
    /// with real geometry / audio parameters. `None` until the track is known.
    fn concrete_caps(&self) -> Option<Caps> {
        Self::concrete_caps_of(&self.demux, self.stream)
    }

    /// The concrete caps for `stream` in `demux` once Tracks has been parsed, with
    /// real geometry / audio parameters (`None` until that track is known). The
    /// stream-parameterized form [`concrete_caps`](Self::concrete_caps) and the
    /// multi-output [`MkvDemuxN`] share, so a per-port announce reuses it.
    fn concrete_caps_of(demux: &MatroskaDemuxer, stream: MkvStream) -> Option<Caps> {
        let want = Self::selected_codec(stream);
        let track = demux.tracks().iter().find(|t| t.codec == want)?;
        match Self::output_caps(stream) {
            Caps::CompressedVideo { codec, .. } if track.width > 0 && track.height > 0 => {
                Some(Caps::CompressedVideo {
                    codec,
                    width: Dim::Fixed(track.width),
                    height: Dim::Fixed(track.height),
                    framerate: Rate::Any,
                })
            }
            Caps::Audio { format, .. } if track.sample_rate > 0 => Some(Caps::Audio {
                format,
                channels: track.channels.max(1),
                sample_rate: track.sample_rate,
            }),
            // A text track carries no geometry / rate to refine; its caps is final.
            text @ Caps::Text { .. } => Some(text),
            _ => None,
        }
    }

    /// Autoplug support: WebM / Matroska carries a single video elementary stream
    /// whose codec the caller cannot know up front (the default selection is Vp9),
    /// so a VP8 / AV1 / H.264 file would otherwise forward nothing. Once Tracks is
    /// parsed, if the selected stream's codec is absent, fall back to the file's
    /// first video track.
    fn auto_select_video_track(&mut self) {
        let want = Self::selected_codec(self.stream);
        let tracks = self.demux.tracks();
        if tracks.is_empty() || tracks.iter().any(|t| t.codec == want) {
            return;
        }
        if let Some(stream) = tracks.iter().find_map(|t| {
            matches!(
                t.codec,
                MkvCodec::H264 | MkvCodec::H265 | MkvCodec::Vp8 | MkvCodec::Vp9 | MkvCodec::Av1
            )
            .then(|| codec_to_stream(t.codec))
            .flatten()
        }) {
            self.stream = stream;
        }
    }

    /// Post any tags parsed since the last call as a [`BusMessage::Tag`]. A
    /// no-op without a bus attached or when nothing new has been parsed.
    fn post_tags(&mut self) {
        let total = self.demux.tags().len();
        if total <= self.tags_posted {
            return;
        }
        let fresh: TagList = self.demux.tags().tags()[self.tags_posted..]
            .iter()
            .cloned()
            .collect();
        self.tags_posted = total;
        if let Some(bus) = &self.bus {
            bus.try_post(BusMessage::Tag(fresh));
        }
    }

    /// Announce every elementary stream the container declares as a
    /// [`BusMessage::StreamCollection`] (M376), once, when the `Tracks` element
    /// has parsed. Lists all tracks regardless of which one this instance
    /// forwards: the discovery half of the playbin model. A no-op without a bus,
    /// before Tracks is parsed, or once already posted (kept across a mid-segment
    /// seek's `reset_keeping_tracks`, so an unchanged collection re-posts).
    fn post_stream_collection(&mut self) {
        if self.collection_posted {
            return;
        }
        let tracks = self.demux.tracks();
        if tracks.is_empty() {
            return;
        }
        let streams: alloc::vec::Vec<Stream> =
            tracks.iter().filter_map(Self::track_to_stream).collect();
        if streams.is_empty() {
            return;
        }
        self.collection_posted = true;
        if let Some(bus) = &self.bus {
            bus.try_post(BusMessage::StreamCollection(StreamCollection::new(
                "matroska-0",
                streams,
            )));
        }
    }

    /// Map one parsed Matroska track to a [`Stream`] for the collection: its kind
    /// (video / audio) and the [`Caps`] it carries, with concrete geometry / audio
    /// parameters when the track declared them. `None` for an unmappable codec.
    fn track_to_stream(track: &crate::matroska::MkvTrack) -> Option<Stream> {
        let id = alloc::format!("matroska-track-{}", track.number);
        let video = |codec| Caps::CompressedVideo {
            codec,
            width: if track.width > 0 {
                Dim::Fixed(track.width)
            } else {
                Dim::Any
            },
            height: if track.height > 0 {
                Dim::Fixed(track.height)
            } else {
                Dim::Any
            },
            framerate: Rate::Any,
        };
        let audio = |format| Caps::Audio {
            format,
            channels: track.channels.max(1),
            sample_rate: track.sample_rate,
        };
        let (stream_type, caps) = match track.codec {
            MkvCodec::H264 => (StreamType::Video, video(VideoCodec::H264)),
            MkvCodec::H265 => (StreamType::Video, video(VideoCodec::H265)),
            MkvCodec::Vp8 => (StreamType::Video, video(VideoCodec::Vp8)),
            MkvCodec::Vp9 => (StreamType::Video, video(VideoCodec::Vp9)),
            MkvCodec::Av1 => (StreamType::Video, video(VideoCodec::Av1)),
            MkvCodec::Aac => (StreamType::Audio, audio(AudioFormat::Aac)),
            MkvCodec::Opus => (StreamType::Audio, audio(AudioFormat::Opus)),
            MkvCodec::Ac3 => (StreamType::Audio, audio(AudioFormat::Ac3)),
            MkvCodec::Flac => (StreamType::Audio, audio(AudioFormat::Flac)),
            // Forwarded as plain UTF-8 text (de-framed at emit), whatever the source.
            MkvCodec::Subtitle(_) => (
                StreamType::Text,
                Caps::Text {
                    format: TextFormat::Utf8,
                },
            ),
            MkvCodec::Other => return None,
        };
        Some(Stream::new(id, stream_type, caps))
    }

    /// Apply any pending app stream selection (M377): switch the single output to
    /// the first selected id that names a forwardable track, force a `CapsChanged`
    /// for the new stream, and confirm the active id on the bus
    /// ([`BusMessage::StreamsSelected`]). A no-op without a controller, with no
    /// pending selection, or when no id resolves (the current stream stays).
    fn apply_stream_selection(&mut self) {
        let Some(ctrl) = &self.stream_select else {
            return;
        };
        let Some(ids) = ctrl.take_pending() else {
            return;
        };
        for id in &ids {
            let Some(stream) = self.resolve_stream_id(id) else {
                continue;
            };
            if stream != self.stream {
                self.stream = stream;
                // Re-emit caps for the newly selected stream on the next frame.
                self.last_caps = None;
            }
            if let Some(bus) = &self.bus {
                bus.try_post(BusMessage::StreamsSelected {
                    ids: alloc::vec![id.clone()],
                });
            }
            return; // single output: the first resolvable id wins
        }
    }

    /// Resolve a published stream id (`matroska-track-N`) to the [`MkvStream`] this
    /// demuxer would forward for it, or `None` if the id is unknown / its codec is
    /// not forwardable. Needs `Tracks` parsed (the collection's ids come from it).
    fn resolve_stream_id(&self, id: &str) -> Option<MkvStream> {
        resolve_stream_id(&self.demux, id)
    }

    /// Emit a `CapsChanged` once the selected track's concrete caps are known,
    /// then forward each demuxed frame of that stream.
    async fn emit_ready(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        if self.bus.is_some() {
            self.post_tags();
            self.post_stream_collection();
        }
        // Honor an app stream selection before computing this batch's caps, so the
        // switch (and its CapsChanged) takes effect for the frames forwarded now.
        self.apply_stream_selection();
        // Fall back to the file's actual video track when the default (Vp9)
        // selection is absent, so autoplug of a VP8 / AV1 / H.264 WebM works.
        self.auto_select_video_track();
        if let Some(caps) = self.concrete_caps() {
            if self.last_caps.as_ref() != Some(&caps) {
                out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                self.last_caps = Some(caps);
            }
        }
        let want = Self::selected_codec(self.stream);
        for f in self.demux.take_frames() {
            if f.codec != want {
                continue; // a stream other than the selected one
            }
            // M362 seek: drop frames until the keyframe at/after the target (the
            // Matroska block keyframe flag); the resuming frame emits a segment.
            match self.seek.admit(f.pts_ns, f.keyframe) {
                Admit::Drop => continue,
                Admit::Resume(start) => {
                    let seg = Segment::for_flush_seek(&Seek::flush_to(start), None);
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                Admit::Emit => {}
            }
            // FLAC: the decoder needs the track's `fLaC` STREAMINFO (CodecPrivate)
            // as extradata; forward it in-band once, ahead of the first frame.
            if self.stream == MkvStream::Flac && !self.flac_header_sent {
                self.flac_header_sent = true;
                if let Some(private) = self.demux.codec_private(f.track) {
                    let header = Frame::new(
                        MemoryDomain::System(SystemSlice::from_boxed(
                            private.to_vec().into_boxed_slice(),
                        )),
                        FrameTiming {
                            pts_ns: f.pts_ns,
                            dts_ns: f.pts_ns,
                            ..FrameTiming::default()
                        },
                        self.emitted,
                    );
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(header)).await?;
                }
            }
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(
                    deframe_block(f.codec, f.data).into_boxed_slice(),
                )),
                FrameTiming {
                    pts_ns: f.pts_ns,
                    dts_ns: f.pts_ns,
                    duration_ns: f.duration_ns,
                    ..FrameTiming::default()
                },
                self.emitted,
            );
            self.emitted += 1;
            out.push(PipelinePacket::DataFrame(frame)).await?;
        }
        Ok(())
    }
}

impl AsyncElement for MkvDemux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&Self::input_caps())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        // ByteStream{Matroska} in -> the selected elementary stream out. The
        // demuxer refines geometry / audio params from Tracks via CapsChanged.
        let stream = self.stream;
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match input {
            Caps::ByteStream {
                encoding: ByteStreamEncoding::Matroska,
            } => CapsSet::one(Self::output_caps(stream)),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !matches!(
            absolute_caps,
            Caps::ByteStream {
                encoding: ByteStreamEncoding::Matroska
            }
        ) {
            return Err(G2gError::CapsMismatch);
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            // M362/M373/M374: service a pending app seek (direct index hit,
            // SeekHead-located Cues prefetch, or re-scan from 0). Until the
            // resulting flush returns, input is dropped so no stale frames emit.
            self.poll_seek();
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    match self.prefetch {
                        // Awaiting the prefetch byte-seek's flush: drop in-flight
                        // pre-seek bytes (the same as `dropping_input`).
                        CuePrefetch::Fetching { flushed: false, .. } => return Ok(()),
                        // Reading the Cues element: parse it (emit nothing); once
                        // the index is populated, seek to the real target Cluster.
                        CuePrefetch::Fetching {
                            flushed: true,
                            target_ns,
                        } => {
                            self.demux.push_data(slice.as_slice());
                            if !self.demux.cues().is_empty() {
                                let off = self.demux.cue_seek_offset(target_ns).unwrap_or(0);
                                self.seek.begin_indexed_seek(target_ns, off);
                                self.prefetch = CuePrefetch::Idle;
                            }
                            return Ok(());
                        }
                        CuePrefetch::Idle => {}
                    }
                    if self.seek.dropping_input() {
                        return Ok(());
                    }
                    self.demux.push_data(slice.as_slice());
                    self.emit_ready(out).await?;
                }
                // The upstream byte-seek's flush. The internal Cues-prefetch flush
                // is consumed (not forwarded): downstream sees a flush only on the
                // real seek. The real seek's flush resets the parser, keeping the
                // Tracks / TimestampScale / Cues a mid-segment (indexed) landing
                // does not re-send (a from-start re-scan fully resets, re-reading
                // the EBML header), then forwards the flush.
                PipelinePacket::Flush => {
                    if let CuePrefetch::Fetching {
                        target_ns,
                        flushed: false,
                    } = self.prefetch
                    {
                        self.prefetch = CuePrefetch::Fetching {
                            target_ns,
                            flushed: true,
                        };
                        self.demux.reset_keeping_tracks();
                        return Ok(());
                    }
                    self.seek.on_flush();
                    if self.seek.keeps_state() {
                        self.demux.reset_keeping_tracks();
                    } else {
                        self.reset_parser();
                    }
                    // a fresh decoder after the flush needs the header again.
                    self.flac_header_sent = false;
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Eos => {
                    // Emit any final frames; the runner's transform arm forwards EOS.
                    self.emit_ready(out).await?;
                }
                // ByteStream caps carry no geometry; nothing to forward.
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        MKVDEMUX_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "stream" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.stream = mkv_stream_from_str(s).ok_or(PropError::Value)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "stream" => Some(PropValue::Str(mkv_stream_to_str(self.stream).into())),
            _ => None,
        }
    }
}

/// `MkvDemux`'s settable properties (M110).
static MKVDEMUX_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "stream",
    PropKind::Str,
    "elementary stream to emit: h264 | h265 | vp8 | vp9 | av1 | aac | opus",
)];

/// The [`MkvStream`] a demuxer forwards for a parsed track's codec, or `None` for
/// an unmappable one. Shared by the single-output and multi-output selection paths.
fn codec_to_stream(codec: MkvCodec) -> Option<MkvStream> {
    match codec {
        MkvCodec::H264 => Some(MkvStream::H264),
        MkvCodec::H265 => Some(MkvStream::H265),
        MkvCodec::Vp8 => Some(MkvStream::Vp8),
        MkvCodec::Vp9 => Some(MkvStream::Vp9),
        MkvCodec::Av1 => Some(MkvStream::Av1),
        MkvCodec::Aac => Some(MkvStream::Aac),
        MkvCodec::Opus => Some(MkvStream::Opus),
        MkvCodec::Ac3 => Some(MkvStream::Ac3),
        MkvCodec::Flac => Some(MkvStream::Flac),
        MkvCodec::Subtitle(format) => Some(MkvStream::Subtitle(format)),
        MkvCodec::Other => None,
    }
}

/// De-frame a demuxed frame's bytes for forwarding: a subtitle block is reduced to
/// its plain UTF-8 cue text (the source-format framing stripped, see
/// [`crate::subparse::deframe_subtitle_block`]) so every subtitle stream forwards
/// as `Text { Utf8 }`; every other codec passes through unchanged. The cue timing
/// rides the frame (block PTS + `BlockDuration`), so only the text is extracted.
fn deframe_block(codec: MkvCodec, data: Vec<u8>) -> Vec<u8> {
    match codec {
        MkvCodec::Subtitle(format) => {
            let text = alloc::string::String::from_utf8_lossy(&data);
            crate::subparse::deframe_subtitle_block(&text, format).into_bytes()
        }
        _ => data,
    }
}

/// One forwardable elementary stream discovered in a parsed Matroska container
/// (M382): which [`MkvStream`] a demux port would carry, the elementary [`Caps`]
/// a decode branch plugs from (the demux's per-port output caps), and whether it
/// is video (vs audio). The `playbin uri=` auto-fan-out builds one decode branch
/// per entry.
#[derive(Debug, Clone)]
pub struct MkvStreamInfo {
    /// The stream a demux port forwards for this track.
    pub stream: MkvStream,
    /// The elementary-stream caps the decode chain plugs from.
    pub caps: Caps,
    /// `true` for a video stream, `false` for audio (picks the auto sink: an
    /// `autovideosink` vs an `autoaudiosink`).
    pub video: bool,
}

/// The forwardable elementary streams a parsed Matroska container carries, in
/// track order (M382): one [`MkvStreamInfo`] per track whose codec maps to an
/// [`MkvStream`] (an unmappable track is dropped). `demux` must have parsed its
/// `Tracks` element (feed a file prefix first); returns empty for a
/// non-Matroska or not-yet-parsed input, which the `playbin` hook reads as
/// "decline, fall through to single-stream playbin".
pub fn forwardable_streams(demux: &MatroskaDemuxer) -> Vec<MkvStreamInfo> {
    demux
        .tracks()
        .iter()
        .filter_map(|t| {
            let stream = codec_to_stream(t.codec)?;
            // Subtitle tracks are discovered (and forwardable by an explicit
            // MkvDemuxN port) but not yet auto-plugged by playbin (no MKV
            // text-branch overlay wiring), so omit them from the A/V fan-out.
            if matches!(stream, MkvStream::Subtitle(_)) {
                return None;
            }
            let video = matches!(
                stream,
                MkvStream::H264
                    | MkvStream::H265
                    | MkvStream::Vp8
                    | MkvStream::Vp9
                    | MkvStream::Av1
            );
            // Fill the concrete channel count from the track header for an audio
            // stream: `output_caps` is the generic per-stream placeholder (channels
            // 0), but an Opus decoder must be created with the real channel count
            // (libopus is per-channel-count, unlike AAC where libavcodec discovers
            // it from the bitstream), so the playbin audio branch needs it up front.
            // The sample rate stays the "unknown until parsed" placeholder (0):
            // compressed-audio caps intersect the rate strictly (only PCM has the
            // wildcard), so a concrete rate would not match a `rate: 0` decoder pad.
            let caps = match MkvDemux::output_caps(stream) {
                Caps::Audio {
                    format,
                    sample_rate,
                    ..
                } => Caps::Audio {
                    format,
                    channels: t.channels.max(1),
                    sample_rate,
                },
                other => other,
            };
            Some(MkvStreamInfo {
                stream,
                caps,
                video,
            })
        })
        .collect()
}

/// The subtitle (text) tracks a parsed Matroska container carries, in track order
/// (M415): one [`MkvStreamInfo`] per track whose codec maps to an
/// [`MkvStream::Subtitle`]. The read-side complement of [`forwardable_streams`]
/// (which is A/V-only): the subtitle-overlay `playbin` builder pairs these with the
/// A/V streams to plug a `TextOverlayN` onto the video branch (the MKV sibling of
/// `mp4demuxn::subtitle_streams`). `video` is always `false` for a text track.
pub fn subtitle_streams(demux: &MatroskaDemuxer) -> Vec<MkvStreamInfo> {
    demux
        .tracks()
        .iter()
        .filter_map(|t| {
            let stream = codec_to_stream(t.codec)?;
            if !matches!(stream, MkvStream::Subtitle(_)) {
                return None;
            }
            Some(MkvStreamInfo {
                stream,
                caps: MkvDemux::output_caps(stream),
                video: false,
            })
        })
        .collect()
}

/// Resolve a published stream id (`matroska-track-N`) to the [`MkvStream`] the
/// demuxer forwards for it, given the parsed tracks (`None` if the id is unknown or
/// its codec is unforwardable). Needs `Tracks` parsed (the ids come from it).
fn resolve_stream_id(demux: &MatroskaDemuxer, id: &str) -> Option<MkvStream> {
    let num: u64 = id.strip_prefix("matroska-track-")?.parse().ok()?;
    let track = demux.tracks().iter().find(|t| t.number == num)?;
    codec_to_stream(track.codec)
}

fn mkv_stream_from_str(s: &str) -> Option<MkvStream> {
    match s {
        "h264" => Some(MkvStream::H264),
        "h265" => Some(MkvStream::H265),
        "vp8" => Some(MkvStream::Vp8),
        "vp9" => Some(MkvStream::Vp9),
        "av1" => Some(MkvStream::Av1),
        "aac" => Some(MkvStream::Aac),
        "opus" => Some(MkvStream::Opus),
        "ac3" => Some(MkvStream::Ac3),
        "flac" => Some(MkvStream::Flac),
        _ => None,
    }
}

/// The `stream` property value naming an [`MkvStream`] (the hook and launch use it).
pub fn mkv_stream_str(stream: MkvStream) -> &'static str {
    mkv_stream_to_str(stream)
}

fn mkv_stream_to_str(stream: MkvStream) -> &'static str {
    match stream {
        MkvStream::H264 => "h264",
        MkvStream::H265 => "h265",
        MkvStream::Vp8 => "vp8",
        MkvStream::Vp9 => "vp9",
        MkvStream::Av1 => "av1",
        MkvStream::Aac => "aac",
        MkvStream::Opus => "opus",
        MkvStream::Ac3 => "ac3",
        MkvStream::Flac => "flac",
        MkvStream::Subtitle(_) => "text",
    }
}

impl PadTemplates for MkvDemux {
    fn pad_templates() -> Vec<PadTemplate> {
        // One sink (the Matroska byte stream); the source pad can carry any of
        // the selectable elementary streams (an instance pins one via with_stream).
        let source = CapsSet::from_alternatives(Vec::from([
            Self::output_caps(MkvStream::H264),
            Self::output_caps(MkvStream::H265),
            Self::output_caps(MkvStream::Vp8),
            Self::output_caps(MkvStream::Vp9),
            Self::output_caps(MkvStream::Av1),
            Self::output_caps(MkvStream::Aac),
            Self::output_caps(MkvStream::Opus),
            Self::output_caps(MkvStream::Ac3),
            Self::output_caps(MkvStream::Flac),
            Self::output_caps(MkvStream::Subtitle(TextFormat::Utf8)),
        ]));
        Vec::from([
            PadTemplate::sink(CapsSet::one(Self::input_caps())),
            PadTemplate::source(source),
        ])
    }
}

/// Multi-output Matroska / WebM demuxer (M378): one Matroska byte stream in, N
/// elementary streams out, one per output port (the decodebin core). The
/// multi-output counterpart of [`MkvDemux`] (which forwards a single selected
/// stream); the read-side analog of [`crate::mkvmuxn::MkvMuxN`].
///
/// A [`MultiOutputElement`] driven by
/// [`run_source_fanout`](g2g_core::runtime::run_source_fanout): each port carries
/// one selected [`MkvStream`] (the "dark slots", §4.9.3), so one demuxer feeds
/// several decode branches in one pipeline rather than instantiating a
/// single-output demuxer per stream. The demuxer parses the container once and
/// routes each track's access units to its port by codec; a parsed track with no
/// matching port is dropped. Port `i` emits its concrete [`Caps`]
/// ([`PipelinePacket::CapsChanged`]) before its first frame, so the branch retypes
/// from the (byte-stream) input caps to the elementary stream, exactly as the
/// single-output demuxer announces its output. A port whose stream the container
/// does not carry simply stays dark.
///
/// The app picks the per-port streams from the announced
/// [`StreamCollection`](BusMessage::StreamCollection) (M376); wiring a `demux`
/// node into the `gst-launch` text DSL and a `playbin` element are the follow-ups.
#[derive(Debug)]
pub struct MkvDemuxN {
    demux: MatroskaDemuxer,
    /// Port `i` emits this elementary stream (one selected stream per output pad).
    ports: Vec<MkvStream>,
    /// Whether port `i` has emitted its opening `CapsChanged` yet.
    announced: Vec<bool>,
    bus: Option<BusHandle>,
    /// Set once the `StreamCollection` has been announced (M376), so it posts once.
    collection_posted: bool,
    /// App-driven stream selection (M381): the app names the stream id each port
    /// should carry (port `i` <- selection id `i`); the demuxer re-maps its ports.
    /// Inert unless `with_stream_select` wired it.
    stream_select: Option<StreamSelectController>,
    emitted: u64,
}

impl MkvDemuxN {
    /// A demuxer with one output port per entry of `ports` (the selected streams),
    /// in port order. Panics if `ports` is empty (a fan-out needs a port).
    pub fn new(ports: Vec<MkvStream>) -> Self {
        assert!(
            !ports.is_empty(),
            "MkvDemuxN needs at least one output port"
        );
        let announced = alloc::vec![false; ports.len()];
        Self {
            demux: MatroskaDemuxer::new(),
            ports,
            announced,
            bus: None,
            collection_posted: false,
            stream_select: None,
            emitted: 0,
        }
    }

    /// Attach the pipeline bus so the container's `StreamCollection` (M376) and
    /// `Tags` post once parsed, the way [`MkvDemux::with_bus`] does.
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.bus = Some(bus);
        self
    }

    /// Make the per-port stream assignment app-selectable (M381): the controller
    /// carries stream ids (from the announced collection); port `i` re-maps to the
    /// stream named by selection id `i`. The demuxer re-emits that port's
    /// `CapsChanged` for the new stream and confirms the active ids on the bus
    /// ([`BusMessage::StreamsSelected`]). Switching a port to a different *codec*
    /// needs its downstream decode branch re-plugged (a follow-up); a same-codec
    /// switch (e.g. between two AAC language tracks) takes effect directly.
    pub fn with_stream_select(mut self, select: StreamSelectController) -> Self {
        self.stream_select = Some(select);
        self
    }

    /// Apply any pending app selection (M381): re-map port `i` to the stream named
    /// by the `i`-th selected id, re-arming that port's `CapsChanged` when its
    /// stream changes, and confirm the active ids on the bus. A no-op without a
    /// controller, with no pending selection, or before `Tracks` is parsed.
    fn apply_stream_selection(&mut self) {
        let Some(ctrl) = &self.stream_select else {
            return;
        };
        let Some(ids) = ctrl.take_pending() else {
            return;
        };
        let mut active = Vec::new();
        for (port, id) in ids.iter().enumerate().take(self.ports.len()) {
            let Some(stream) = resolve_stream_id(&self.demux, id) else {
                continue;
            };
            if self.ports[port] != stream {
                self.ports[port] = stream;
                self.announced[port] = false; // re-emit caps for the new stream
            }
            active.push(id.clone());
        }
        if !active.is_empty() {
            if let Some(bus) = &self.bus {
                bus.try_post(BusMessage::StreamsSelected { ids: active });
            }
        }
    }

    /// Number of output ports (the selected-stream count).
    pub fn port_count(&self) -> usize {
        self.ports.len()
    }

    /// Count of frames forwarded across all ports.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    /// The output port that carries the elementary stream of `codec`, or `None`
    /// when no selected port matches (the track is dropped).
    fn port_for_codec(&self, codec: MkvCodec) -> Option<usize> {
        self.ports
            .iter()
            .position(|&s| MkvDemux::selected_codec(s) == codec)
    }

    /// Announce every container track as a [`BusMessage::StreamCollection`] (M376),
    /// once, when `Tracks` has parsed. Mirrors [`MkvDemux::post_stream_collection`]
    /// (lists all tracks, not just the ported ones, so the app can re-select).
    fn post_stream_collection(&mut self) {
        if self.collection_posted {
            return;
        }
        let streams: Vec<Stream> = self
            .demux
            .tracks()
            .iter()
            .filter_map(MkvDemux::track_to_stream)
            .collect();
        if streams.is_empty() {
            return;
        }
        self.collection_posted = true;
        if let Some(bus) = &self.bus {
            bus.try_post(BusMessage::StreamCollection(StreamCollection::new(
                "matroska-0",
                streams,
            )));
        }
    }
}

impl MultiOutputElement for MkvDemuxN {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&MkvDemux::input_caps())
    }

    /// Declare each port's elementary-stream caps (M380), so the solver negotiates
    /// each branch against its codec at startup and a real decoder downstream of
    /// the port configures against it (geometry is a placeholder `Range`, refined
    /// at runtime by the port's `CapsChanged`). `None` for an out-of-range port.
    fn port_output_caps(&self, port: usize) -> Option<Caps> {
        self.ports
            .get(port)
            .map(|&stream| MkvDemux::output_caps(stream))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        // Accept the negotiated byte-stream input; per-port output caps are
        // announced from `process` as each stream first routes.
        absolute_caps
            .intersect(&MkvDemux::input_caps())
            .map(|_| ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.demux.push_data(slice.as_slice());
                    if self.bus.is_some() {
                        self.post_stream_collection();
                    }
                    // Honor an app selection before routing this batch, so a re-map
                    // (and its re-armed CapsChanged) takes effect for the frames now.
                    self.apply_stream_selection();
                    for f in self.demux.take_frames() {
                        let Some(port) = self.port_for_codec(f.codec) else {
                            continue; // a track no selected port carries
                        };
                        if !self.announced[port] {
                            let caps = MkvDemux::concrete_caps_of(&self.demux, self.ports[port])
                                .unwrap_or_else(|| MkvDemux::output_caps(self.ports[port]));
                            out.push_to(port, PipelinePacket::CapsChanged(caps)).await?;
                            self.announced[port] = true;
                        }
                        let out_frame = Frame::new(
                            MemoryDomain::System(SystemSlice::from_boxed(
                                deframe_block(f.codec, f.data).into_boxed_slice(),
                            )),
                            FrameTiming {
                                pts_ns: f.pts_ns,
                                dts_ns: f.pts_ns,
                                duration_ns: f.duration_ns,
                                ..FrameTiming::default()
                            },
                            self.emitted,
                        );
                        self.emitted += 1;
                        out.push_to(port, PipelinePacket::DataFrame(out_frame))
                            .await?;
                    }
                }
                // Flush / Segment apply to every branch (the parser resets on a
                // flush, as the single-output demuxer does).
                PipelinePacket::Flush => {
                    self.demux.reset_keeping_tracks();
                    for port in 0..self.ports.len() {
                        out.push_to(port, PipelinePacket::Flush).await?;
                    }
                }
                PipelinePacket::Segment(seg) => {
                    for port in 0..self.ports.len() {
                        out.push_to(port, PipelinePacket::Segment(seg)).await?;
                    }
                }
                // The input's own (byte-stream) CapsChanged is consumed: each port
                // defines its own caps, announced per port above.
                PipelinePacket::CapsChanged(_) => {}
                // The runner broadcasts the single merged Eos to every port.
                PipelinePacket::Eos => {}
                // future PipelinePacket variants: no-op.
                _ => {}
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::{PushOutcome, RawVideoFormat};

    // Synthetic WebM builders (mirror the matroska parser unit tests).
    fn vint(value: u64) -> Vec<u8> {
        // Grow until the value fits and isn't the all-ones (unknown-size) pattern.
        let mut len = 1usize;
        while len < 8 && value >= (1u64 << (7 * len)) - 1 {
            len += 1;
        }
        let mut out = alloc::vec![0u8; len];
        let mut v = value;
        for i in (0..len).rev() {
            out[i] = (v & 0xFF) as u8;
            v >>= 8;
        }
        out[0] |= 1 << (8 - len);
        out
    }

    fn elem(id: &[u8], body: &[u8]) -> Vec<u8> {
        let mut out = id.to_vec();
        out.extend_from_slice(&vint(body.len() as u64));
        out.extend_from_slice(body);
        out
    }

    fn uint_body(v: u64) -> Vec<u8> {
        if v == 0 {
            return alloc::vec![0];
        }
        let mut bytes = v.to_be_bytes().to_vec();
        while bytes.len() > 1 && bytes[0] == 0 {
            bytes.remove(0);
        }
        bytes
    }

    fn block(track: u64, rel: i16, frame: &[u8]) -> Vec<u8> {
        let mut b = vint(track);
        b.extend_from_slice(&rel.to_be_bytes());
        b.push(0x80); // keyframe, no lacing
        b.extend_from_slice(frame);
        b
    }

    fn video_track(num: u64, codec: &[u8], w: u32, h: u32) -> Vec<u8> {
        let v = [
            elem(&[0xB0], &uint_body(w as u64)),
            elem(&[0xBA], &uint_body(h as u64)),
        ]
        .concat();
        let body = [
            elem(&[0xD7], &uint_body(num)),
            elem(&[0x86], codec),
            elem(&[0xE0], &v),
        ]
        .concat();
        elem(&[0xAE], &body)
    }

    fn audio_track(num: u64, codec: &[u8], ch: u8, sr: u32) -> Vec<u8> {
        let mut a = elem(&[0x9F], &uint_body(ch as u64));
        a.extend_from_slice(&elem(&[0xB5], &(sr as f32).to_be_bytes()));
        let body = [
            elem(&[0xD7], &uint_body(num)),
            elem(&[0x86], codec),
            elem(&[0xE1], &a),
        ]
        .concat();
        elem(&[0xAE], &body)
    }

    fn webm() -> Vec<u8> {
        let tracks = elem(
            &[0x16, 0x54, 0xAE, 0x6B],
            &[
                video_track(1, b"V_VP9", 320, 240),
                audio_track(2, b"A_OPUS", 2, 48_000),
            ]
            .concat(),
        );
        let cluster = elem(
            &[0x1F, 0x43, 0xB6, 0x75],
            &[
                elem(&[0xE7], &uint_body(0)),
                elem(&[0xA3], &block(1, 0, &[0x11, 0x22])),
                elem(&[0xA3], &block(2, 0, &[0x33, 0x44])),
                elem(&[0xA3], &block(1, 40, &[0x55, 0x66])),
            ]
            .concat(),
        );
        let segment = elem(&[0x18, 0x53, 0x80, 0x67], &[tracks, cluster].concat());
        [elem(&[0x1A, 0x45, 0xDF, 0xA3], &[]), segment].concat()
    }

    #[derive(Default)]
    struct CaptureSink {
        caps: Vec<Caps>,
        frames: Vec<Vec<u8>>,
        eos: bool,
    }
    impl OutputSink for CaptureSink {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async move {
                match packet {
                    PipelinePacket::CapsChanged(c) => self.caps.push(c),
                    PipelinePacket::DataFrame(f) => {
                        if let MemoryDomain::System(s) = &f.domain {
                            self.frames.push(s.as_slice().to_vec());
                        }
                    }
                    PipelinePacket::Eos => self.eos = true,
                    _ => {}
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }

    async fn run(stream: MkvStream, bytes: &[u8], sink: &mut CaptureSink) {
        let mut d = MkvDemux::new().with_stream(stream);
        d.configure_pipeline(&MkvDemux::input_caps()).unwrap();
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.to_vec().into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        d.process(PipelinePacket::DataFrame(frame), sink)
            .await
            .unwrap();
        d.process(PipelinePacket::Eos, sink).await.unwrap();
    }

    #[test]
    fn forwardable_streams_carry_concrete_audio_channels() {
        // An Opus decoder is created per channel count, so the playbin audio branch
        // needs the real channels from the container up front (not the channels-0
        // placeholder): forwardable_streams must fill them from the track header.
        let mut demux = MatroskaDemuxer::new();
        demux.push_data(&webm());
        let infos = forwardable_streams(&demux);
        let opus = infos
            .iter()
            .find(|i| {
                matches!(
                    i.caps,
                    Caps::Audio {
                        format: AudioFormat::Opus,
                        ..
                    }
                )
            })
            .expect("WebM has an Opus track");
        assert_eq!(
            opus.caps,
            Caps::Audio {
                format: AudioFormat::Opus,
                channels: 2,
                sample_rate: 0
            },
            "Opus forwardable caps carry the track's concrete channel count (rate stays the \
             unknown-until-parsed placeholder, since compressed rate intersects strictly)"
        );
        assert!(!opus.video);
    }

    #[test]
    fn caps_byte_stream_in_compressed_out() {
        let d = MkvDemux::new();
        assert!(d.intercept_caps(&MkvDemux::input_caps()).is_ok());
        let raw = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(2),
            height: Dim::Fixed(2),
            framerate: Rate::Any,
        };
        assert!(d.intercept_caps(&raw).is_err());
        // A TS byte stream is the wrong container.
        let ts = Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        };
        assert!(d.intercept_caps(&ts).is_err());
    }

    #[tokio::test]
    async fn selects_video_with_refined_geometry() {
        let mut sink = CaptureSink::default();
        run(MkvStream::Vp9, &webm(), &mut sink).await;

        // The demuxer refines geometry from Tracks before the frames.
        assert_eq!(
            sink.caps,
            alloc::vec![Caps::CompressedVideo {
                codec: VideoCodec::Vp9,
                width: Dim::Fixed(320),
                height: Dim::Fixed(240),
                framerate: Rate::Any,
            }]
        );
        assert_eq!(
            sink.frames,
            alloc::vec![alloc::vec![0x11, 0x22], alloc::vec![0x55, 0x66]]
        );
        assert!(
            !sink.eos,
            "EOS is forwarded by the runner's arm, not the element"
        );
    }

    #[tokio::test]
    async fn selects_audio_with_refined_params() {
        let mut sink = CaptureSink::default();
        run(MkvStream::Opus, &webm(), &mut sink).await;

        assert_eq!(
            sink.caps,
            alloc::vec![Caps::Audio {
                format: AudioFormat::Opus,
                channels: 2,
                sample_rate: 48_000
            }]
        );
        assert_eq!(sink.frames, alloc::vec![alloc::vec![0x33, 0x44]]);
    }

    #[test]
    fn output_caps_track_the_selection() {
        assert!(matches!(
            MkvDemux::output_caps(MkvStream::Vp8),
            Caps::CompressedVideo {
                codec: VideoCodec::Vp8,
                ..
            }
        ));
        assert!(matches!(
            MkvDemux::output_caps(MkvStream::Opus),
            Caps::Audio {
                format: AudioFormat::Opus,
                ..
            }
        ));
    }

    /// A WebM with a Segment `Title`, a VP9 track, a `Tags` element, then one
    /// Cluster frame.
    fn webm_with_tags() -> Vec<u8> {
        let info = elem(&[0x15, 0x49, 0xA9, 0x66], &elem(&[0x7B, 0xA9], b"My Clip")); // Info/Title
        let tracks = elem(
            &[0x16, 0x54, 0xAE, 0x6B],
            &video_track(1, b"V_VP9", 320, 240),
        );
        let simple = [elem(&[0x45, 0xA3], b"ARTIST"), elem(&[0x44, 0x87], b"Band")].concat();
        let tag = [elem(&[0x63, 0xC0], &[]), elem(&[0x67, 0xC8], &simple)].concat();
        let tags = elem(&[0x12, 0x54, 0xC3, 0x67], &elem(&[0x73, 0x73], &tag));
        let cluster = elem(
            &[0x1F, 0x43, 0xB6, 0x75],
            &[
                elem(&[0xE7], &uint_body(0)),
                elem(&[0xA3], &block(1, 0, &[0x11, 0x22])),
            ]
            .concat(),
        );
        let segment = elem(
            &[0x18, 0x53, 0x80, 0x67],
            &[info, tracks, tags, cluster].concat(),
        );
        [elem(&[0x1A, 0x45, 0xDF, 0xA3], &[]), segment].concat()
    }

    #[tokio::test]
    async fn posts_tags_on_the_bus() {
        use g2g_core::{Bus, BusMessage, Tag};
        let (bus, handle) = Bus::new(8);
        let mut d = MkvDemux::new().with_stream(MkvStream::Vp9).with_bus(handle);
        d.configure_pipeline(&MkvDemux::input_caps()).unwrap();
        let mut sink = CaptureSink::default();
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(webm_with_tags().into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        d.process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();

        let mut posted = TagList::new();
        while let Some(m) = bus.try_recv() {
            if let BusMessage::Tag(t) = m {
                for tag in t.tags() {
                    posted.push(tag.clone());
                }
            }
        }
        assert_eq!(
            posted.tags(),
            &[Tag::Title("My Clip".into()), Tag::Artist("Band".into())]
        );
        // The selected video frame still flows while the tags go out of band.
        assert_eq!(sink.frames, alloc::vec![alloc::vec![0x11, 0x22]]);
    }

    #[test]
    fn stream_property_round_trips() {
        let mut d = MkvDemux::new();
        assert_eq!(d.get_property("stream"), Some(PropValue::Str("vp9".into())));
        d.set_property("stream", PropValue::Str("opus".into()))
            .unwrap();
        assert_eq!(d.stream(), MkvStream::Opus);
        assert_eq!(
            d.set_property("stream", PropValue::Str("theora".into())),
            Err(PropError::Value)
        );
    }

    fn subtitle_track(num: u64, codec: &[u8]) -> Vec<u8> {
        // TrackNumber, TrackType(subtitle=0x11), CodecID.
        let body = [
            elem(&[0xD7], &uint_body(num)),
            elem(&[0x83], &uint_body(0x11)),
            elem(&[0x86], codec),
        ]
        .concat();
        elem(&[0xAE], &body)
    }

    fn mkv_subtitle(codec: &[u8], payload: &[u8]) -> Vec<u8> {
        let tracks = elem(&[0x16, 0x54, 0xAE, 0x6B], &subtitle_track(1, codec));
        let cluster = elem(
            &[0x1F, 0x43, 0xB6, 0x75],
            &[
                elem(&[0xE7], &uint_body(0)),
                elem(&[0xA3], &block(1, 0, payload)),
            ]
            .concat(),
        );
        let segment = elem(&[0x18, 0x53, 0x80, 0x67], &[tracks, cluster].concat());
        [elem(&[0x1A, 0x45, 0xDF, 0xA3], &[]), segment].concat()
    }

    #[tokio::test]
    async fn s_text_ass_deframes_to_plain_utf8() {
        // A Matroska ASS block is `ReadOrder,Layer,Style,Name,MarginL,MarginR,
        // MarginV,Effect,Text`; the demuxer extracts the Text field (tags / `\N`
        // resolved) and forwards it as `Text { Utf8 }`.
        let bytes = mkv_subtitle(
            b"S_TEXT/ASS",
            b"0,0,Default,,0,0,0,,{\\i1}Hello{\\i0}\\Nthere",
        );
        let mut sink = CaptureSink::default();
        run(MkvStream::Subtitle(TextFormat::Ssa), &bytes, &mut sink).await;
        assert_eq!(
            sink.caps,
            alloc::vec![Caps::Text {
                format: TextFormat::Utf8
            }]
        );
        assert_eq!(sink.frames, alloc::vec![b"Hello\nthere".to_vec()]);
    }

    #[tokio::test]
    async fn s_text_webvtt_strips_inline_tags() {
        let bytes = mkv_subtitle(b"S_TEXT/WEBVTT", b"<c.yellow>Hi</c> there");
        let mut sink = CaptureSink::default();
        run(MkvStream::Subtitle(TextFormat::WebVtt), &bytes, &mut sink).await;
        assert_eq!(
            sink.caps,
            alloc::vec![Caps::Text {
                format: TextFormat::Utf8
            }]
        );
        assert_eq!(sink.frames, alloc::vec![b"Hi there".to_vec()]);
    }
}
