//! MPEG-TS demuxer element (M108): `Caps::ByteStream{MpegTs}` in, one selected
//! elementary stream out. H.264 / H.265 video leave as `Caps::CompressedVideo`
//! (Annex-B); AAC audio leaves as `Caps::Audio` (ADTS).
//!
//! Wraps the pure [`crate::mpegts::TsDemuxer`] parser. Incoming byte frames are
//! resynchronized to 188-byte TS packets and fed to the demuxer; the reassembled
//! PES access units of the selected stream ([`TsStream`], default H.264) are
//! forwarded with their PTS, ready for the matching parser. CPU, `no_std`
//! baseline.
//!
//! ```text
//! filesrc(location=x.ts, caps=ByteStream{MpegTs}) ! tsdemux ! h264parse ! <decoder> ! <sink>
//! tsdemux stream=aac ! aacparse ! <audio>
//! ```
//!
//! One output pad carries one elementary stream: the parser reassembles every
//! stream the PMT names, and the [`TsStream`] selection picks which to emit, so a
//! second `tsdemux` selecting another stream demuxes the rest of the multiplex.
//! The choice is by codec, not a runtime-discovered "first video", because the
//! output caps are fixed at negotiation before any packet is parsed (M109).
//! Scope (v1): the first stream of the selected codec; multi-program selection
//! and a muxer are follow-ups (DESIGN.md §4.17).

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
    PropValue, PropertySpec, Rate, Seek, Segment, Stream, StreamCollection, StreamType, VideoCodec,
};

use crate::demuxseek::{Admit, DemuxSeek};
use crate::mpegts::{
    EsUnit, TsDemuxer, STREAM_TYPE_AAC, STREAM_TYPE_H264, STREAM_TYPE_H265, STREAM_TYPE_MPEG4P2,
    TS_PACKET_LEN,
};

const TS_SYNC: u8 = 0x47;

/// Which elementary stream a [`TsDemux`] instance forwards. A TS multiplex
/// carries several (video + audio); this element has one output pad, so it emits
/// exactly one, chosen here. The choice is by codec because the output caps are
/// fixed at negotiation, before any packet is parsed: H.264 and H.265 are
/// distinct downstream decoders, not a geometry refinement of one another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TsStream {
    /// The first H.264 (AVC) video elementary stream. The default.
    #[default]
    H264,
    /// The first H.265 (HEVC) video elementary stream.
    H265,
    /// The first MPEG-4 Part 2 (Visual) video elementary stream.
    Mpeg4Part2,
    /// The first AAC (ADTS) audio elementary stream.
    Aac,
}

/// Demuxes an MPEG-TS byte stream into one selected elementary stream.
#[derive(Debug)]
pub struct TsDemux {
    demux: TsDemuxer,
    /// The elementary stream this instance forwards (the single output pad).
    stream: TsStream,
    /// Bytes not yet consumed as whole TS packets (packet realignment across
    /// input frames).
    buf: Vec<u8>,
    configured: bool,
    emitted: u64,
    /// Pipeline bus, for announcing the program's `StreamCollection` (M386).
    /// Inert unless `with_bus` wired it.
    bus: Option<BusHandle>,
    /// Set once the `StreamCollection` has been announced, so it posts once.
    collection_posted: bool,
    /// Seek support (M362): app time seeks drive an upstream byte-seek and a
    /// re-sync. Inert unless `with_seek` wired the controllers.
    seek: DemuxSeek,
}

impl Default for TsDemux {
    fn default() -> Self {
        Self::new()
    }
}

impl TsDemux {
    pub fn new() -> Self {
        Self {
            demux: TsDemuxer::new(),
            stream: TsStream::H264,
            buf: Vec::new(),
            configured: false,
            emitted: 0,
            bus: None,
            collection_posted: false,
            seek: DemuxSeek::default(),
        }
    }

    /// Select which elementary stream to forward (default [`TsStream::H264`]).
    pub fn with_stream(mut self, stream: TsStream) -> Self {
        self.stream = stream;
        self
    }

    /// Attach the pipeline bus so the program's `StreamCollection` (M386) is
    /// announced once the PMT is parsed, the MPEG-TS sibling of
    /// [`MkvDemux::with_bus`](crate::mkvdemux::MkvDemux::with_bus).
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.bus = Some(bus);
        self
    }

    /// Announce every elementary stream the PMT declares as a
    /// [`BusMessage::StreamCollection`] (M386), once, when the PMT has parsed.
    /// Lists all programs' streams regardless of which one this instance forwards
    /// (the discovery half of the multi-stream model). A no-op without a bus,
    /// before the PMT, or once already posted.
    fn post_stream_collection(&mut self) {
        if self.collection_posted {
            return;
        }
        let streams: alloc::vec::Vec<Stream> = self
            .demux
            .streams()
            .iter()
            .filter_map(Self::es_to_stream)
            .collect();
        if streams.is_empty() {
            return;
        }
        self.collection_posted = true;
        if let Some(bus) = &self.bus {
            bus.try_post(BusMessage::StreamCollection(StreamCollection::new(
                "mpegts-0", streams,
            )));
        }
    }

    /// Map one PMT elementary stream to a [`Stream`] for the collection: its kind
    /// (video / audio) and the media-type [`Caps`] it carries (geometry is
    /// unknown from the PMT, so `Any`, refined later by `CapsChanged`). `None` for
    /// a `stream_type` g2g does not forward.
    fn es_to_stream(es: &crate::mpegts::ElementaryStream) -> Option<Stream> {
        let id = alloc::format!("mpegts-pid-{}", es.pid);
        let video = |codec| Caps::CompressedVideo {
            codec,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        let (stream_type, caps) = match es.stream_type {
            STREAM_TYPE_H264 => (StreamType::Video, video(VideoCodec::H264)),
            STREAM_TYPE_H265 => (StreamType::Video, video(VideoCodec::H265)),
            STREAM_TYPE_MPEG4P2 => (StreamType::Video, video(VideoCodec::Mpeg4Part2)),
            STREAM_TYPE_AAC => (
                StreamType::Audio,
                Caps::Audio {
                    format: AudioFormat::Aac,
                    channels: 0,
                    sample_rate: 0,
                },
            ),
            _ => return None,
        };
        Some(Stream::new(id, stream_type, caps))
    }

    /// Make the demuxer seekable (M362): `app` carries app time seeks; `upstream`
    /// is the byte source's ([`FileSrc`](crate::filesrc)) byte-seek controller.
    /// On a time seek the demuxer rewinds the source and re-syncs from the
    /// keyframe at/after the target.
    pub fn with_seek(mut self, app: SeekController, upstream: SeekController) -> Self {
        self.seek.with(app, upstream);
        self
    }

    /// Reset the parser for a discontinuity (a `Flush` / seek): drop buffered
    /// bytes and the demuxer's PAT/PMT/PES state, which the re-read stream
    /// re-establishes from its start.
    fn reset_parser(&mut self) {
        self.buf.clear();
        self.demux = TsDemuxer::new();
    }

    /// The elementary stream this instance forwards.
    pub fn stream(&self) -> TsStream {
        self.stream
    }

    /// Count of frames forwarded.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    /// The input this element accepts: an MPEG-TS byte stream.
    fn input_caps() -> Caps {
        Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        }
    }

    /// The output caps for the selected elementary stream. Video geometry is
    /// unknown until the bitstream parser reads the SPS, so H.264 / H.265
    /// advertise a fixatable placeholder `Range` (`Dim::Any` would fail Phase-2
    /// fixate) refined downstream via `CapsChanged`, the pattern `RtspSrc` uses
    /// for async-discovered dims. AAC has no open `Caps` field, so it advertises
    /// the sentinel channels/rate 0 that `aacparse` accepts pre-header and
    /// refines from the ADTS header.
    fn output_caps(stream: TsStream) -> Caps {
        match stream {
            TsStream::H264 => Self::compressed_video(VideoCodec::H264),
            TsStream::H265 => Self::compressed_video(VideoCodec::H265),
            TsStream::Mpeg4Part2 => Self::compressed_video(VideoCodec::Mpeg4Part2),
            TsStream::Aac => Caps::Audio {
                format: AudioFormat::Aac,
                channels: 0,
                sample_rate: 0,
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

    /// The PMT `stream_type` the selected output corresponds to.
    fn selected_stream_type(stream: TsStream) -> u8 {
        match stream {
            TsStream::H264 => STREAM_TYPE_H264,
            TsStream::H265 => STREAM_TYPE_H265,
            TsStream::Mpeg4Part2 => STREAM_TYPE_MPEG4P2,
            TsStream::Aac => STREAM_TYPE_AAC,
        }
    }

    /// Consume whole 188-byte TS packets from `buf`, resyncing to the sync byte,
    /// feeding each to the demuxer. Leaves any trailing partial packet in `buf`.
    fn drain_packets(&mut self) {
        loop {
            // Resync: drop bytes before the next sync byte.
            if self.buf.first() != Some(&TS_SYNC) {
                match self.buf.iter().position(|&b| b == TS_SYNC) {
                    Some(pos) => {
                        self.buf.drain(..pos);
                    }
                    None => {
                        self.buf.clear();
                        return;
                    }
                }
            }
            if self.buf.len() < TS_PACKET_LEN {
                return;
            }
            // Feed one packet. (A copy keeps the borrow off `self.buf` so the
            // drain below is clean.)
            let mut pkt = [0u8; TS_PACKET_LEN];
            pkt.copy_from_slice(&self.buf[..TS_PACKET_LEN]);
            self.demux.push_packet(&pkt);
            self.buf.drain(..TS_PACKET_LEN);
        }
    }

    /// Emit each completed access unit of the selected elementary stream as a
    /// frame (Annex-B for H.264 / H.265, ADTS for AAC), carrying its PTS.
    async fn emit_units(
        &mut self,
        units: Vec<EsUnit>,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let want = Self::selected_stream_type(self.stream);
        for u in units {
            if u.stream_type != want {
                continue; // a stream other than the selected one
            }
            let pts_ns = u
                .pts_90khz
                .map(|p| (p as u128 * 1_000_000_000 / 90_000) as u64)
                .unwrap_or(0);
            // M362 seek: an audio frame is always a resync point; a video AU is
            // one only if it carries an IDR/IRAP. Drop until the target keyframe.
            let keyframe = match self.stream {
                TsStream::H264 => crate::annexb::au_is_keyframe(VideoCodec::H264, &u.data),
                TsStream::H265 => crate::annexb::au_is_keyframe(VideoCodec::H265, &u.data),
                TsStream::Mpeg4Part2 => {
                    crate::annexb::au_is_keyframe(VideoCodec::Mpeg4Part2, &u.data)
                }
                TsStream::Aac => true,
            };
            match self.seek.admit(pts_ns, keyframe) {
                Admit::Drop => continue,
                Admit::Resume(start) => {
                    let seg = Segment::for_flush_seek(&Seek::flush_to(start), None);
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                Admit::Emit => {}
            }
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(u.data.into_boxed_slice())),
                FrameTiming {
                    pts_ns,
                    dts_ns: pts_ns,
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

impl AsyncElement for TsDemux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&Self::input_caps())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        // ByteStream{MpegTs} in -> the selected elementary stream out. The solver
        // hands downstream the chosen caps; the bitstream parser refines video
        // geometry / audio params from the stream via CapsChanged.
        let stream = self.stream;
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match input {
            Caps::ByteStream {
                encoding: ByteStreamEncoding::MpegTs,
            } => CapsSet::one(Self::output_caps(stream)),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !matches!(
            absolute_caps,
            Caps::ByteStream {
                encoding: ByteStreamEncoding::MpegTs
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
            // M362: a pending app seek triggers an upstream byte-seek; until its
            // `Flush` returns, drop input so no stale pre-seek units are emitted.
            self.seek.poll_request();
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    if self.seek.dropping_input() {
                        return Ok(());
                    }
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.buf.extend_from_slice(slice.as_slice());
                    self.drain_packets();
                    if self.bus.is_some() {
                        self.post_stream_collection();
                    }
                    let units = self.demux.take_units();
                    self.emit_units(units, out).await?;
                }
                // The upstream byte-seek's flush: reset the parser, then re-sync
                // from the re-read stream. Forward it downstream.
                PipelinePacket::Flush => {
                    self.seek.on_flush();
                    self.reset_parser();
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Eos => {
                    // Flush the final in-flight PES and emit it; the runner's
                    // transform arm forwards the EOS itself.
                    self.demux.flush();
                    let units = self.demux.take_units();
                    self.emit_units(units, out).await?;
                }
                // ByteStream caps don't carry geometry; nothing to forward, and
                // a Segment passes through.
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        TSDEMUX_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "stream" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.stream = ts_stream_from_str(s).ok_or(PropError::Value)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "stream" => Some(PropValue::Str(ts_stream_to_str(self.stream).into())),
            _ => None,
        }
    }
}

/// `TsDemux`'s settable properties (M109).
static TSDEMUX_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "stream",
    PropKind::Str,
    "elementary stream to emit: h264 | h265 | aac",
)];

/// Parse a `stream` property string to a [`TsStream`].
fn ts_stream_from_str(s: &str) -> Option<TsStream> {
    match s {
        "h264" => Some(TsStream::H264),
        "h265" => Some(TsStream::H265),
        "mpeg4part2" => Some(TsStream::Mpeg4Part2),
        "aac" => Some(TsStream::Aac),
        _ => None,
    }
}

/// The `stream` property string for a [`TsStream`].
fn ts_stream_to_str(stream: TsStream) -> &'static str {
    match stream {
        TsStream::H264 => "h264",
        TsStream::H265 => "h265",
        TsStream::Mpeg4Part2 => "mpeg4part2",
        TsStream::Aac => "aac",
    }
}

/// The [`TsStream`] a demuxer forwards for a PMT `stream_type`, or `None` for one
/// g2g does not forward.
fn stream_type_to_ts(stream_type: u8) -> Option<TsStream> {
    match stream_type {
        STREAM_TYPE_H264 => Some(TsStream::H264),
        STREAM_TYPE_H265 => Some(TsStream::H265),
        STREAM_TYPE_MPEG4P2 => Some(TsStream::Mpeg4Part2),
        STREAM_TYPE_AAC => Some(TsStream::Aac),
        _ => None,
    }
}

/// One forwardable elementary stream discovered in a parsed transport stream
/// (M389): which [`TsStream`] a demux port would carry, the elementary [`Caps`] a
/// decode branch plugs from, and whether it is video (vs audio). The
/// `playbin uri=*.ts` auto-fan-out builds one decode branch per entry. The
/// MPEG-TS analog of [`crate::mkvdemux::MkvStreamInfo`].
#[derive(Debug, Clone)]
pub struct TsStreamInfo {
    /// The stream a demux port forwards for this PMT entry.
    pub stream: TsStream,
    /// The elementary-stream caps the decode chain plugs from.
    pub caps: Caps,
    /// `true` for a video stream, `false` for audio (picks the auto sink).
    pub video: bool,
}

/// The forwardable elementary streams a parsed transport stream carries, in PMT
/// order (M389): one [`TsStreamInfo`] per PMT entry whose `stream_type` maps to a
/// [`TsStream`]. `demux` must have parsed its PMT (feed a file prefix first);
/// returns empty for a non-MPEG-TS or not-yet-parsed input, which the `playbin`
/// hook reads as "decline, fall through".
pub fn forwardable_streams(demux: &TsDemuxer) -> Vec<TsStreamInfo> {
    demux
        .streams()
        .iter()
        .filter_map(|es| {
            let stream = stream_type_to_ts(es.stream_type)?;
            let video = matches!(stream, TsStream::H264 | TsStream::H265);
            Some(TsStreamInfo {
                stream,
                caps: TsDemux::output_caps(stream),
                video,
            })
        })
        .collect()
}

impl PadTemplates for TsDemux {
    fn pad_templates() -> Vec<PadTemplate> {
        // One sink (the TS byte stream); the source pad can carry any of the
        // selectable elementary streams (an instance pins one via with_stream).
        let source = CapsSet::from_alternatives(Vec::from([
            Self::output_caps(TsStream::H264),
            Self::output_caps(TsStream::H265),
            Self::output_caps(TsStream::Aac),
        ]));
        Vec::from([
            PadTemplate::sink(CapsSet::one(Self::input_caps())),
            PadTemplate::source(source),
        ])
    }
}

/// Multi-output MPEG-TS demuxer (M388): one TS byte stream in, N elementary
/// streams out, one selected [`TsStream`] per output port. The MPEG-TS sibling of
/// [`MkvDemuxN`](crate::mkvdemux::MkvDemuxN), and the multi-output counterpart of
/// [`TsDemux`] (which forwards a single selected stream).
///
/// A [`MultiOutputElement`] driven by
/// [`run_source_fanout`](g2g_core::runtime::run_source_fanout): it parses the
/// transport stream once and routes each PES access unit to the port whose
/// selected codec matches the unit's PMT `stream_type`, so one demuxer feeds
/// several decode branches in one pipeline (audio + video together). Port `i`
/// emits its elementary [`Caps`] ([`PipelinePacket::CapsChanged`]) before its
/// first frame (the branch retypes from the byte-stream input caps); a parsed
/// stream no port carries is dropped, and a port whose stream the multiplex lacks
/// stays dark. With a bus, announces the same `StreamCollection` (M386) as
/// [`TsDemux`]. The `playbin uri=*.ts` fan-out (M389) builds this.
#[derive(Debug)]
pub struct TsDemuxN {
    demux: TsDemuxer,
    /// Bytes not yet consumed as whole TS packets (realignment across frames).
    buf: Vec<u8>,
    /// Port `i` emits this elementary stream (one selected stream per output pad).
    ports: Vec<TsStream>,
    /// Whether port `i` has emitted its opening `CapsChanged` yet.
    announced: Vec<bool>,
    bus: Option<BusHandle>,
    /// Set once the `StreamCollection` has been announced (M386), so it posts once.
    collection_posted: bool,
    /// App-driven stream selection (M475): the app names the stream id each port
    /// should carry (port `i` <- selection id `i`); the demuxer re-maps its ports.
    /// Inert unless `with_stream_select` wired it. The MPEG-TS sibling of
    /// [`MkvDemuxN::with_stream_select`](crate::mkvdemux::MkvDemuxN::with_stream_select).
    stream_select: Option<StreamSelectController>,
    emitted: u64,
}

impl TsDemuxN {
    /// A demuxer with one output port per entry of `ports` (the selected streams),
    /// in port order. Panics if `ports` is empty (a fan-out needs a port).
    pub fn new(ports: Vec<TsStream>) -> Self {
        assert!(!ports.is_empty(), "TsDemuxN needs at least one output port");
        let announced = alloc::vec![false; ports.len()];
        Self {
            demux: TsDemuxer::new(),
            buf: Vec::new(),
            ports,
            announced,
            bus: None,
            collection_posted: false,
            stream_select: None,
            emitted: 0,
        }
    }

    /// Attach the pipeline bus so the program's `StreamCollection` (M386) posts
    /// once the PMT is parsed, the way [`TsDemux::with_bus`] does.
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.bus = Some(bus);
        self
    }

    /// Make the per-port stream assignment app-selectable (M475): the controller
    /// carries stream ids (from the announced collection, e.g. `"mpegts-pid-256"`);
    /// port `i` re-maps to the stream named by selection id `i`. The demuxer
    /// re-emits that port's `CapsChanged` for the new stream and confirms the active
    /// ids on the bus ([`BusMessage::StreamsSelected`]). Because TS ports route by
    /// PMT `stream_type`, a re-map to a different *codec* takes effect at routing;
    /// two streams of the same codec share a port (as in [`MkvDemuxN`]). The
    /// MPEG-TS sibling of [`MkvDemuxN::with_stream_select`](crate::mkvdemux::MkvDemuxN::with_stream_select).
    pub fn with_stream_select(mut self, select: StreamSelectController) -> Self {
        self.stream_select = Some(select);
        self
    }

    /// Apply any pending app selection (M475): re-map port `i` to the stream named
    /// by the `i`-th selected id, re-arming that port's `CapsChanged` when its
    /// stream changes, and confirm the active ids on the bus. A no-op without a
    /// controller, with no pending selection, or before the PMT is parsed (an id
    /// resolves against the PMT's declared streams).
    fn apply_stream_selection(&mut self) {
        let Some(ctrl) = &self.stream_select else {
            return;
        };
        let Some(ids) = ctrl.take_pending() else {
            return;
        };
        let mut active = Vec::new();
        for (port, id) in ids.iter().enumerate().take(self.ports.len()) {
            let Some(stream) = resolve_ts_stream_id(&self.demux, id) else {
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

    /// The port carrying the elementary stream of PMT `stream_type`, or `None` if
    /// no selected port carries it. The first matching port wins.
    fn port_for_stream_type(&self, stream_type: u8) -> Option<usize> {
        self.ports
            .iter()
            .position(|&s| TsDemux::selected_stream_type(s) == stream_type)
    }

    /// Consume whole 188-byte TS packets from `buf`, resyncing to the sync byte,
    /// feeding each to the demuxer. Leaves any trailing partial packet in `buf`.
    /// (The realignment is identical to [`TsDemux::drain_packets`].)
    fn drain_packets(&mut self) {
        loop {
            if self.buf.first() != Some(&TS_SYNC) {
                match self.buf.iter().position(|&b| b == TS_SYNC) {
                    Some(pos) => {
                        self.buf.drain(..pos);
                    }
                    None => {
                        self.buf.clear();
                        return;
                    }
                }
            }
            if self.buf.len() < TS_PACKET_LEN {
                return;
            }
            let mut pkt = [0u8; TS_PACKET_LEN];
            pkt.copy_from_slice(&self.buf[..TS_PACKET_LEN]);
            self.demux.push_packet(&pkt);
            self.buf.drain(..TS_PACKET_LEN);
        }
    }

    /// Announce every PMT elementary stream as a `StreamCollection` (M386), once.
    /// Reuses [`TsDemux::es_to_stream`]. A no-op without a bus, before the PMT, or
    /// once posted.
    fn post_stream_collection(&mut self) {
        if self.collection_posted {
            return;
        }
        let streams: Vec<Stream> = self
            .demux
            .streams()
            .iter()
            .filter_map(TsDemux::es_to_stream)
            .collect();
        if streams.is_empty() {
            return;
        }
        self.collection_posted = true;
        if let Some(bus) = &self.bus {
            bus.try_post(BusMessage::StreamCollection(StreamCollection::new(
                "mpegts-0", streams,
            )));
        }
    }

    /// Route each completed access unit to the port carrying its `stream_type`,
    /// emitting that port's opening `CapsChanged` before its first frame.
    async fn route_units(&mut self, out: &mut dyn MultiOutputSink) -> Result<(), G2gError> {
        for u in self.demux.take_units() {
            let Some(port) = self.port_for_stream_type(u.stream_type) else {
                continue; // a stream no selected port carries
            };
            if !self.announced[port] {
                out.push_to(
                    port,
                    PipelinePacket::CapsChanged(TsDemux::output_caps(self.ports[port])),
                )
                .await?;
                self.announced[port] = true;
            }
            let pts_ns = u
                .pts_90khz
                .map(|p| (p as u128 * 1_000_000_000 / 90_000) as u64)
                .unwrap_or(0);
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(u.data.into_boxed_slice())),
                FrameTiming {
                    pts_ns,
                    dts_ns: pts_ns,
                    ..FrameTiming::default()
                },
                self.emitted,
            );
            self.emitted += 1;
            out.push_to(port, PipelinePacket::DataFrame(frame)).await?;
        }
        Ok(())
    }
}

/// Resolve a collection stream id (`"mpegts-pid-{pid}"`) to the [`TsStream`] that
/// selects it, by looking the PID up in the parsed PMT and mapping its
/// `stream_type`. `None` for an unparseable id, an unknown PID, or a stream_type
/// g2g does not forward (M475).
fn resolve_ts_stream_id(demux: &TsDemuxer, id: &str) -> Option<TsStream> {
    let pid: u16 = id.strip_prefix("mpegts-pid-")?.parse().ok()?;
    let es = demux.streams().iter().find(|e| e.pid == pid)?;
    match es.stream_type {
        STREAM_TYPE_H264 => Some(TsStream::H264),
        STREAM_TYPE_H265 => Some(TsStream::H265),
        STREAM_TYPE_MPEG4P2 => Some(TsStream::Mpeg4Part2),
        STREAM_TYPE_AAC => Some(TsStream::Aac),
        _ => None,
    }
}

impl MultiOutputElement for TsDemuxN {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&TsDemux::input_caps())
    }

    /// Declare each port's elementary-stream caps (M380), so the solver negotiates
    /// each branch against its codec at startup. `None` for an out-of-range port.
    fn port_output_caps(&self, port: usize) -> Option<Caps> {
        self.ports
            .get(port)
            .map(|&stream| TsDemux::output_caps(stream))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        absolute_caps
            .intersect(&TsDemux::input_caps())
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
                    self.buf.extend_from_slice(slice.as_slice());
                    self.drain_packets();
                    if self.bus.is_some() {
                        self.post_stream_collection();
                    }
                    // Honor an app selection before routing this batch, so a re-map
                    // (and its re-armed CapsChanged) takes effect for the frames now.
                    self.apply_stream_selection();
                    self.route_units(out).await?;
                }
                // A flush resets the parser; the re-read stream re-establishes
                // PAT/PMT/PES. Broadcast it to every branch.
                PipelinePacket::Flush => {
                    self.buf.clear();
                    self.demux = TsDemuxer::new();
                    for port in 0..self.ports.len() {
                        out.push_to(port, PipelinePacket::Flush).await?;
                    }
                }
                PipelinePacket::Segment(seg) => {
                    for port in 0..self.ports.len() {
                        out.push_to(port, PipelinePacket::Segment(seg)).await?;
                    }
                }
                PipelinePacket::Eos => {
                    // Flush the final in-flight PES, route it; the runner
                    // broadcasts the merged Eos to every port.
                    self.demux.flush();
                    self.route_units(out).await?;
                }
                // The input's (byte-stream) CapsChanged is consumed: each port
                // defines its own caps, announced per port above.
                PipelinePacket::CapsChanged(_) => {}
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

    /// PMT stream_type 0x10 maps to an MPEG-4 Part 2 video stream, and the
    /// selection round-trips through the `stream` property string and the
    /// stream_type <-> TsStream tables.
    #[test]
    fn stream_type_0x10_is_mpeg4_part2() {
        let es = crate::mpegts::ElementaryStream {
            pid: 0x100,
            stream_type: STREAM_TYPE_MPEG4P2,
        };
        let stream = TsDemux::es_to_stream(&es).expect("0x10 is forwarded");
        assert_eq!(stream.stream_type, StreamType::Video);
        assert!(
            matches!(
                stream.caps,
                Caps::CompressedVideo {
                    codec: VideoCodec::Mpeg4Part2,
                    ..
                }
            ),
            "0x10 -> MPEG-4 Part 2 caps"
        );

        assert_eq!(
            stream_type_to_ts(STREAM_TYPE_MPEG4P2),
            Some(TsStream::Mpeg4Part2)
        );
        assert_eq!(
            TsDemux::selected_stream_type(TsStream::Mpeg4Part2),
            STREAM_TYPE_MPEG4P2
        );
        assert_eq!(ts_stream_from_str("mpeg4part2"), Some(TsStream::Mpeg4Part2));
        assert_eq!(ts_stream_to_str(TsStream::Mpeg4Part2), "mpeg4part2");
    }

    // Re-use the synthetic TS builders by constructing equivalent packets here.
    fn ts_packet(pid: u16, pusi: bool, payload: &[u8]) -> Vec<u8> {
        const ROOM: usize = TS_PACKET_LEN - 4;
        let mut p = alloc::vec![0u8; TS_PACKET_LEN];
        p[0] = TS_SYNC;
        p[1] = if pusi { 0x40 } else { 0x00 } | ((pid >> 8) as u8 & 0x1F);
        p[2] = (pid & 0xFF) as u8;
        let l = payload.len();
        if l == ROOM {
            p[3] = 0x10;
            p[4..].copy_from_slice(payload);
        } else {
            p[3] = 0x30;
            let af_len = ROOM - 1 - l;
            p[4] = af_len as u8;
            if af_len >= 1 {
                p[5] = 0x00;
                for b in p.iter_mut().take(6 + (af_len - 1)).skip(6) {
                    *b = 0xFF;
                }
            }
            p[5 + af_len..].copy_from_slice(payload);
        }
        p
    }

    fn psi(pid: u16, table_id: u8, body: &[u8]) -> Vec<u8> {
        let section_length = body.len() + 4;
        let mut s = alloc::vec![
            table_id,
            0xB0 | ((section_length >> 8) as u8 & 0x0F),
            (section_length & 0xFF) as u8
        ];
        s.extend_from_slice(body);
        s.extend_from_slice(&[0, 0, 0, 0]);
        let mut payload = alloc::vec![0u8];
        payload.extend_from_slice(&s);
        ts_packet(pid, true, &payload)
    }

    fn pat(pmt_pid: u16) -> Vec<u8> {
        psi(
            0x0000,
            0x00,
            &[
                0,
                1,
                0xC1,
                0,
                0,
                0,
                1,
                0xE0 | (pmt_pid >> 8) as u8 & 0x1F,
                pmt_pid as u8,
            ],
        )
    }

    fn pmt(es_pid: u16) -> Vec<u8> {
        psi(
            0x1000,
            0x02,
            &[
                0x00,
                0x01,
                0xC1,
                0x00,
                0x00,
                0xE0 | (es_pid >> 8) as u8 & 0x1F,
                es_pid as u8,
                0xF0,
                0x00,
                STREAM_TYPE_H264,
                0xE0 | (es_pid >> 8) as u8 & 0x1F,
                es_pid as u8,
                0xF0,
                0x00,
            ],
        )
    }

    fn pes(es: &[u8]) -> Vec<u8> {
        pes_id(0xE0, es)
    }

    /// A PES with an explicit `stream_id` (video 0xE0, audio 0xC0), no PTS.
    fn pes_id(stream_id: u8, es: &[u8]) -> Vec<u8> {
        let mut p = alloc::vec![0x00, 0x00, 0x01, stream_id];
        let header = [0x80u8, 0x00, 0x00];
        let len = header.len() + es.len();
        p.push((len >> 8) as u8);
        p.push((len & 0xFF) as u8);
        p.extend_from_slice(&header);
        p.extend_from_slice(es);
        p
    }

    /// A two-stream PMT (one video, one audio), the common A/V multiplex shape.
    fn pmt2(v_pid: u16, v_type: u8, a_pid: u16, a_type: u8) -> Vec<u8> {
        psi(
            0x1000,
            0x02,
            &[
                0x00,
                0x01,
                0xC1,
                0x00,
                0x00,
                0xE0 | (v_pid >> 8) as u8 & 0x1F,
                v_pid as u8, // PCR_PID
                0xF0,
                0x00,
                v_type,
                0xE0 | (v_pid >> 8) as u8 & 0x1F,
                v_pid as u8,
                0xF0,
                0x00,
                a_type,
                0xE0 | (a_pid >> 8) as u8 & 0x1F,
                a_pid as u8,
                0xF0,
                0x00,
            ],
        )
    }

    #[derive(Default)]
    struct CaptureSink {
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

    #[test]
    fn caps_byte_stream_in_h264_out() {
        let d = TsDemux::new();
        assert!(d.intercept_caps(&TsDemux::input_caps()).is_ok());
        // A non-TS byte stream / other caps is rejected.
        let raw = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(2),
            height: Dim::Fixed(2),
            framerate: Rate::Any,
        };
        assert!(d.intercept_caps(&raw).is_err());
    }

    #[tokio::test]
    async fn demuxes_h264_frames_from_ts_bytes() {
        let es_pid = 0x0100;
        let mut d = TsDemux::new();
        d.configure_pipeline(&TsDemux::input_caps()).unwrap();

        // Build a TS byte stream: PAT, PMT, then two H.264 access units, each its
        // own PES (PUSI). The first flushes when the second's PES starts; the
        // second flushes on EOS.
        let au0 = [0u8, 0, 0, 1, 0x65, 0xAA];
        let au1 = [0u8, 0, 0, 1, 0x41, 0xBB, 0xCC];
        let mut stream = Vec::new();
        stream.extend_from_slice(&pat(0x1000));
        stream.extend_from_slice(&pmt(es_pid));
        stream.extend_from_slice(&ts_packet(es_pid, true, &pes(&au0)));
        stream.extend_from_slice(&ts_packet(es_pid, true, &pes(&au1)));

        let mut sink = CaptureSink::default();
        // Feed the whole stream as one System frame.
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(stream.into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        d.process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();
        d.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        assert_eq!(sink.frames.len(), 2, "two H.264 access units demuxed");
        assert_eq!(
            sink.frames[0], au0,
            "first AU bytes intact (PES header stripped)"
        );
        assert_eq!(sink.frames[1], au1);
        assert!(
            !sink.eos,
            "EOS is forwarded by the runner's arm, not the element"
        );
        assert_eq!(d.emitted(), 2);
    }

    /// Feed the whole stream as one frame, then EOS, capturing the output.
    async fn run_demux(d: &mut TsDemux, stream: &[u8], sink: &mut CaptureSink) {
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(stream.to_vec().into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        d.process(PipelinePacket::DataFrame(frame), sink)
            .await
            .unwrap();
        d.process(PipelinePacket::Eos, sink).await.unwrap();
    }

    #[tokio::test]
    async fn selects_audio_or_video_from_a_multiplex() {
        let v_pid = 0x0100u16;
        let a_pid = 0x0101u16;
        // PAT, a 2-stream PMT (H.264 video + AAC audio), then interleaved access
        // units, each its own PES. The same bytes feed two demuxers that pick
        // apart the multiplex: one the video, one the audio.
        let v0 = [0u8, 0, 0, 1, 0x65, 0x11];
        let v1 = [0u8, 0, 0, 1, 0x41, 0x22];
        let a0 = [0xFFu8, 0xF1, 0x50, 0x80, 0x01, 0x23];
        let a1 = [0xFFu8, 0xF1, 0x50, 0x80, 0x02, 0x45];
        let mut stream = Vec::new();
        stream.extend_from_slice(&pat(0x1000));
        stream.extend_from_slice(&pmt2(v_pid, STREAM_TYPE_H264, a_pid, STREAM_TYPE_AAC));
        stream.extend_from_slice(&ts_packet(v_pid, true, &pes_id(0xE0, &v0)));
        stream.extend_from_slice(&ts_packet(a_pid, true, &pes_id(0xC0, &a0)));
        stream.extend_from_slice(&ts_packet(v_pid, true, &pes_id(0xE0, &v1)));
        stream.extend_from_slice(&ts_packet(a_pid, true, &pes_id(0xC0, &a1)));

        // Default selects H.264: only the two video AUs come out.
        let mut video = TsDemux::new();
        video.configure_pipeline(&TsDemux::input_caps()).unwrap();
        let mut vsink = CaptureSink::default();
        run_demux(&mut video, &stream, &mut vsink).await;
        assert_eq!(
            vsink.frames,
            alloc::vec![v0.to_vec(), v1.to_vec()],
            "video only"
        );

        // stream=aac selects AAC: only the two audio AUs come out (ADTS payload).
        let mut audio = TsDemux::new().with_stream(TsStream::Aac);
        audio.configure_pipeline(&TsDemux::input_caps()).unwrap();
        let mut asink = CaptureSink::default();
        run_demux(&mut audio, &stream, &mut asink).await;
        assert_eq!(
            asink.frames,
            alloc::vec![a0.to_vec(), a1.to_vec()],
            "audio only"
        );
    }

    #[test]
    fn output_caps_track_the_selection() {
        assert!(matches!(
            TsDemux::output_caps(TsStream::H264),
            Caps::CompressedVideo {
                codec: VideoCodec::H264,
                ..
            }
        ));
        assert!(matches!(
            TsDemux::output_caps(TsStream::H265),
            Caps::CompressedVideo {
                codec: VideoCodec::H265,
                ..
            }
        ));
        assert!(matches!(
            TsDemux::output_caps(TsStream::Aac),
            Caps::Audio {
                format: AudioFormat::Aac,
                ..
            }
        ));
    }

    #[test]
    fn stream_property_round_trips_and_drives_output() {
        let mut d = TsDemux::new();
        assert_eq!(
            d.get_property("stream"),
            Some(PropValue::Str("h264".into()))
        );

        d.set_property("stream", PropValue::Str("aac".into()))
            .unwrap();
        assert_eq!(d.stream(), TsStream::Aac);
        assert_eq!(d.get_property("stream"), Some(PropValue::Str("aac".into())));

        // An unsupported codec name is rejected (leaving the selection unchanged).
        assert_eq!(
            d.set_property("stream", PropValue::Str("vp9".into())),
            Err(PropError::Value)
        );
        assert_eq!(d.stream(), TsStream::Aac);

        // DerivedOutput now maps the TS byte stream to AAC audio.
        let CapsConstraint::DerivedOutput(f) = d.caps_constraint_as_transform() else {
            panic!("expected DerivedOutput");
        };
        let out = f(&TsDemux::input_caps());
        assert!(matches!(
            out.alternatives(),
            [Caps::Audio {
                format: AudioFormat::Aac,
                ..
            }]
        ));
    }
}
