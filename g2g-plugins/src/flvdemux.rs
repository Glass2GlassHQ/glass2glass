//! FLV demuxer element (M119): `Caps::ByteStream{Flv}` in, one selected
//! elementary stream out. H.264 video leaves as `Caps::CompressedVideo` (AVCC);
//! AAC audio leaves as `Caps::Audio`.
//!
//! Wraps the pure [`crate::flv::FlvDemuxer`], the FLV sibling of
//! [`crate::tsdemux::TsDemux`]: incoming byte frames are fed to the parser, and
//! the access units of the selected stream ([`FlvStream`], default H.264) are
//! forwarded with their PTS, ready for the matching parser / decoder. CPU,
//! `no_std` baseline.
//!
//! ```text
//! filesrc(location=x.flv, caps=ByteStream{Flv}) ! flvdemux ! h264parse ! <decoder>
//! flvdemux stream=aac ! aacparse ! <audio>
//! ```
//!
//! One output pad carries one elementary stream; the [`FlvStream`] selection picks
//! which, so a second `flvdemux stream=aac` demuxes the audio. The choice is by
//! codec because the output caps are fixed at negotiation, before any tag is
//! parsed. Scope (v1): the H.264 video and AAC audio tracks (DESIGN.md §4.17).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use alloc::string::String;

use g2g_core::runtime::SeekController;
use g2g_core::{
    AsyncElement, AudioFormat, BusHandle, BusMessage, ByteStreamEncoding, Caps, CapsConstraint,
    CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate,
    PadTemplates, PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate, Seek, Segment,
    Tag, TagList, VideoCodec,
};

use crate::demuxseek::{Admit, DemuxSeek};
use crate::flv::{FlvDemuxer, FlvTrack, FlvUnit};

/// Which elementary stream an [`FlvDemux`] instance forwards. An FLV stream
/// interleaves one video and one audio track; this element has one output pad, so
/// it emits exactly one, chosen by codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FlvStream {
    /// The H.264 (AVC) video track. The default.
    #[default]
    H264,
    /// The AAC audio track.
    Aac,
}

/// Demuxes an FLV byte stream into one selected elementary stream.
#[derive(Debug)]
pub struct FlvDemux {
    demux: FlvDemuxer,
    /// The elementary stream this instance forwards (the single output pad).
    stream: FlvStream,
    configured: bool,
    emitted: u64,
    bus: Option<BusHandle>,
    tags_posted: bool,
    /// Seek support (M362): app time seeks drive an upstream byte-seek and a
    /// re-sync. Inert unless `with_seek` wired the controllers.
    seek: DemuxSeek,
}

impl Default for FlvDemux {
    fn default() -> Self {
        Self::new()
    }
}

impl FlvDemux {
    pub fn new() -> Self {
        Self {
            demux: FlvDemuxer::new(),
            stream: FlvStream::H264,
            configured: false,
            emitted: 0,
            bus: None,
            tags_posted: false,
            seek: DemuxSeek::default(),
        }
    }

    /// Select which elementary stream to forward (default [`FlvStream::H264`]).
    pub fn with_stream(mut self, stream: FlvStream) -> Self {
        self.stream = stream;
        self
    }

    /// Make the demuxer seekable (M362): `app` carries app time seeks; `upstream`
    /// is the byte source's ([`FileSrc`](crate::filesrc)) byte-seek controller.
    /// On a time seek the demuxer rewinds the source and re-syncs from the
    /// keyframe at/after the target.
    pub fn with_seek(mut self, app: SeekController, upstream: SeekController) -> Self {
        self.seek.with(app, upstream);
        self
    }

    /// Reset the parser for a discontinuity (a `Flush` / seek): drop the FLV
    /// demuxer's tag-stream state, which the re-read stream re-establishes from
    /// its FLV header.
    fn reset_parser(&mut self) {
        self.demux = FlvDemuxer::new();
    }

    /// Attach the pipeline bus so the FLV `onMetaData` metadata posts as a
    /// [`BusMessage::Tag`] once the script tag is parsed.
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.bus = Some(bus);
        self
    }

    /// Parse the `onMetaData` body into tags and post them once a bus is attached.
    fn maybe_post_tags(&mut self) {
        if self.tags_posted || self.bus.is_none() {
            return;
        }
        let tags = match self.demux.metadata() {
            Some(meta) => parse_on_metadata(meta),
            None => return,
        };
        self.tags_posted = true;
        if !tags.is_empty() {
            if let Some(bus) = &self.bus {
                bus.try_post(BusMessage::Tag(tags));
            }
        }
    }

    /// The elementary stream this instance forwards.
    pub fn stream(&self) -> FlvStream {
        self.stream
    }

    /// Count of frames forwarded.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    /// The input this element accepts: an FLV byte stream.
    fn input_caps() -> Caps {
        Caps::ByteStream { encoding: ByteStreamEncoding::Flv }
    }

    /// The output caps for the selected elementary stream. Video geometry is
    /// unknown until the bitstream parser reads the SPS, so H.264 advertises a
    /// fixatable placeholder `Range` refined downstream via `CapsChanged`. AAC has
    /// no open `Caps` field, so it advertises the sentinel channels/rate 0 that
    /// `aacparse` accepts pre-header.
    fn output_caps(stream: FlvStream) -> Caps {
        match stream {
            FlvStream::H264 => Caps::CompressedVideo {
                codec: VideoCodec::H264,
                width: Dim::Range { min: 16, max: 65_535 },
                height: Dim::Range { min: 16, max: 65_535 },
                framerate: Rate::Range { min_q16: 1 << 16, max_q16: 240 << 16 },
            },
            FlvStream::Aac => Caps::Audio { format: AudioFormat::Aac, channels: 0, sample_rate: 0 },
        }
    }

    /// The track this instance's selected stream corresponds to.
    fn selected_track(stream: FlvStream) -> FlvTrack {
        match stream {
            FlvStream::H264 => FlvTrack::Video,
            FlvStream::Aac => FlvTrack::Audio,
        }
    }

    /// Emit each access unit of the selected track as a frame, carrying its PTS
    /// (the FLV millisecond timestamp converted to nanoseconds).
    async fn emit_units(
        &mut self,
        units: Vec<FlvUnit>,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let want = Self::selected_track(self.stream);
        for u in units {
            if u.track != want {
                continue;
            }
            let pts_ns = u.pts_ms as u64 * 1_000_000;
            // M362 seek: drop units until the keyframe at/after the target (the
            // FLV frame-type flag); the resuming unit emits a fresh segment.
            match self.seek.admit(pts_ns, u.keyframe) {
                Admit::Drop => continue,
                Admit::Resume(start) => {
                    let seg = Segment::for_flush_seek(&Seek::flush_to(start), None);
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                Admit::Emit => {}
            }
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(u.data.into_boxed_slice())),
                FrameTiming { pts_ns, dts_ns: pts_ns, ..FrameTiming::default() },
                self.emitted,
            );
            self.emitted += 1;
            out.push(PipelinePacket::DataFrame(frame)).await?;
        }
        Ok(())
    }
}

impl AsyncElement for FlvDemux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&Self::input_caps())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let stream = self.stream;
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match input {
            Caps::ByteStream { encoding: ByteStreamEncoding::Flv } => {
                CapsSet::one(Self::output_caps(stream))
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !matches!(absolute_caps, Caps::ByteStream { encoding: ByteStreamEncoding::Flv }) {
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
                    self.demux.push_data(slice.as_slice());
                    self.maybe_post_tags();
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
                    // Emit any final access units. The runner's transform arm
                    // forwards the EOS itself, so pushing it here would double it
                    // (the second hits a closed sink under a full link).
                    self.maybe_post_tags();
                    let units = self.demux.take_units();
                    self.emit_units(units, out).await?;
                }
                // ByteStream caps don't carry geometry; nothing to forward.
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        FLVDEMUX_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "stream" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.stream = flv_stream_from_str(s).ok_or(PropError::Value)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "stream" => Some(PropValue::Str(flv_stream_to_str(self.stream).into())),
            _ => None,
        }
    }
}

/// `FlvDemux`'s settable properties.
static FLVDEMUX_PROPS: &[PropertySpec] =
    &[PropertySpec::new("stream", PropKind::Str, "elementary stream to emit: h264 | aac")];

/// Parse a `stream` property string to an [`FlvStream`].
fn flv_stream_from_str(s: &str) -> Option<FlvStream> {
    match s {
        "h264" => Some(FlvStream::H264),
        "aac" => Some(FlvStream::Aac),
        _ => None,
    }
}

/// The `stream` property string for an [`FlvStream`].
fn flv_stream_to_str(stream: FlvStream) -> &'static str {
    match stream {
        FlvStream::H264 => "h264",
        FlvStream::Aac => "aac",
    }
}

impl PadTemplates for FlvDemux {
    fn pad_templates() -> Vec<PadTemplate> {
        let source = CapsSet::from_alternatives(Vec::from([
            Self::output_caps(FlvStream::H264),
            Self::output_caps(FlvStream::Aac),
        ]));
        Vec::from([PadTemplate::sink(CapsSet::one(Self::input_caps())), PadTemplate::source(source)])
    }
}

// AMF0 markers (the FLV `onMetaData` serialization uses this subset).
mod amf0 {
    pub(super) const NUMBER: u8 = 0x00;
    pub(super) const BOOLEAN: u8 = 0x01;
    pub(super) const STRING: u8 = 0x02;
    pub(super) const OBJECT: u8 = 0x03;
    pub(super) const NULL: u8 = 0x05;
    pub(super) const UNDEFINED: u8 = 0x06;
    pub(super) const ECMA_ARRAY: u8 = 0x08;
    pub(super) const OBJECT_END: u8 = 0x09;
    pub(super) const STRICT_ARRAY: u8 = 0x0A;
    pub(super) const DATE: u8 = 0x0B;
    pub(super) const LONG_STRING: u8 = 0x0C;
}

/// Parse an FLV `onMetaData` script body (AMF0) into a [`TagList`]. The body is
/// the event-name string (`onMetaData`) followed by an ECMA array / object of
/// properties; its string-valued entries become tags (numbers, booleans, nested
/// objects are walked to stay aligned but not turned into tags). A malformed /
/// truncated body yields whatever parsed before the error.
fn parse_on_metadata(body: &[u8]) -> TagList {
    let mut list = TagList::new();
    let mut pos = 0usize;
    // The first value is the event name; bail unless it is "onMetaData".
    if read_amf0_value(body, &mut pos, 0) != Some(Some(String::from("onMetaData"))) {
        return list;
    }
    // The second value holds the properties: an ECMA array or an anonymous object.
    let Some(marker) = body.get(pos).copied() else { return list };
    pos += 1;
    let _ = match marker {
        amf0::ECMA_ARRAY => {
            let _count = read_u32_be(body, &mut pos);
            read_amf0_object(body, &mut pos, Some(&mut list), 0)
        }
        amf0::OBJECT => read_amf0_object(body, &mut pos, Some(&mut list), 0),
        _ => Some(()),
    };
    list
}

/// Cap AMF0 nesting so a crafted onMetaData (each level costs ~4 bytes, the body
/// is up to 16 MB) cannot recurse deep enough to overflow the stack. Real
/// metadata is only a level or two deep.
const MAX_AMF0_DEPTH: u32 = 32;

/// Read an AMF0 value at `*pos`, advancing past it. Returns `Some(Some(s))` for a
/// string value, `Some(None)` for any other (correctly skipped) value, or `None`
/// on a parse error / unknown marker / excessive nesting.
fn read_amf0_value(b: &[u8], pos: &mut usize, depth: u32) -> Option<Option<String>> {
    if depth >= MAX_AMF0_DEPTH {
        return None;
    }
    let marker = *b.get(*pos)?;
    *pos += 1;
    match marker {
        amf0::NUMBER => advance(b, pos, 8).map(|_| None),
        amf0::BOOLEAN => advance(b, pos, 1).map(|_| None),
        amf0::STRING => {
            let len = read_u16_be(b, pos)? as usize;
            Some(Some(read_amf0_str(b, pos, len)?))
        }
        amf0::OBJECT => read_amf0_object(b, pos, None, depth + 1).map(|_| None),
        amf0::ECMA_ARRAY => {
            let _count = read_u32_be(b, pos)?;
            read_amf0_object(b, pos, None, depth + 1).map(|_| None)
        }
        amf0::NULL | amf0::UNDEFINED => Some(None),
        amf0::STRICT_ARRAY => {
            let count = read_u32_be(b, pos)?;
            for _ in 0..count {
                read_amf0_value(b, pos, depth + 1)?;
            }
            Some(None)
        }
        amf0::DATE => advance(b, pos, 10).map(|_| None), // f64 + s16 timezone
        amf0::LONG_STRING => {
            let len = read_u32_be(b, pos)? as usize;
            advance(b, pos, len).map(|_| None)
        }
        _ => None,
    }
}

/// Read AMF0 `(key, value)` property pairs until the object-end marker. When
/// `collect` is set, each string-valued property is added as a [`Tag`].
fn read_amf0_object(
    b: &[u8],
    pos: &mut usize,
    mut collect: Option<&mut TagList>,
    depth: u32,
) -> Option<()> {
    if depth >= MAX_AMF0_DEPTH {
        return None;
    }
    loop {
        let key_len = read_u16_be(b, pos)? as usize;
        if key_len == 0 {
            // An empty key precedes the object-end marker.
            return if *b.get(*pos)? == amf0::OBJECT_END {
                *pos += 1;
                Some(())
            } else {
                None
            };
        }
        let key = read_amf0_str(b, pos, key_len)?;
        let value = read_amf0_value(b, pos, depth + 1)?;
        if let (Some(list), Some(s)) = (collect.as_deref_mut(), value) {
            list.push(Tag::from_key_value(&key, &s));
        }
    }
}

fn advance(b: &[u8], pos: &mut usize, n: usize) -> Option<()> {
    let new = pos.checked_add(n)?;
    if new > b.len() {
        return None;
    }
    *pos = new;
    Some(())
}

fn read_u16_be(b: &[u8], pos: &mut usize) -> Option<u16> {
    let s = b.get(*pos..*pos + 2)?;
    *pos += 2;
    Some(u16::from_be_bytes([s[0], s[1]]))
}

fn read_u32_be(b: &[u8], pos: &mut usize) -> Option<u32> {
    let s = b.get(*pos..*pos + 4)?;
    *pos += 4;
    Some(u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
}

/// Read `len` bytes as a UTF-8 string (AMF0 object keys / string values are raw,
/// not marker-prefixed).
fn read_amf0_str(b: &[u8], pos: &mut usize, len: usize) -> Option<String> {
    let s = b.get(*pos..*pos + len)?;
    *pos += len;
    core::str::from_utf8(s).ok().map(String::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::{Dim, PushOutcome, Rate, RawVideoFormat};

    fn push_u24(out: &mut Vec<u8>, v: u32) {
        out.push((v >> 16) as u8);
        out.push((v >> 8) as u8);
        out.push(v as u8);
    }

    fn tag(tag_type: u8, timestamp: u32, body: &[u8]) -> Vec<u8> {
        let mut t = alloc::vec![tag_type];
        push_u24(&mut t, body.len() as u32);
        push_u24(&mut t, timestamp & 0x00FF_FFFF);
        t.push((timestamp >> 24) as u8);
        push_u24(&mut t, 0);
        t.extend_from_slice(body);
        t
    }

    fn avc_nalu(au: &[u8]) -> Vec<u8> {
        let mut b = alloc::vec![0x17, 0x01, 0x00, 0x00, 0x00];
        b.extend_from_slice(au);
        b
    }

    fn aac_raw(frame: &[u8]) -> Vec<u8> {
        let mut b = alloc::vec![0xAF, 0x01];
        b.extend_from_slice(frame);
        b
    }

    fn flv_stream(tags: &[Vec<u8>]) -> Vec<u8> {
        let mut s = b"FLV".to_vec();
        s.push(1);
        s.push(0x05);
        s.extend_from_slice(&9u32.to_be_bytes());
        let mut prev = 0u32;
        for t in tags {
            s.extend_from_slice(&prev.to_be_bytes());
            s.extend_from_slice(t);
            prev = t.len() as u32;
        }
        s
    }

    #[derive(Default)]
    struct CaptureSink {
        frames: Vec<Vec<u8>>,
        pts: Vec<u64>,
    }
    impl OutputSink for CaptureSink {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async move {
                if let PipelinePacket::DataFrame(f) = packet {
                    if let MemoryDomain::System(s) = &f.domain {
                        self.frames.push(s.as_slice().to_vec());
                    }
                    self.pts.push(f.timing.pts_ns);
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }

    async fn run_demux(d: &mut FlvDemux, stream: &[u8], sink: &mut CaptureSink) {
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(stream.to_vec().into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        d.process(PipelinePacket::DataFrame(frame), sink).await.unwrap();
        d.process(PipelinePacket::Eos, sink).await.unwrap();
    }

    #[test]
    fn caps_byte_stream_in_h264_out() {
        let d = FlvDemux::new();
        assert!(d.intercept_caps(&FlvDemux::input_caps()).is_ok());
        let raw = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(2),
            height: Dim::Fixed(2),
            framerate: Rate::Any,
        };
        assert!(d.intercept_caps(&raw).is_err());
        // The Matroska byte stream is the wrong container.
        let mkv = Caps::ByteStream { encoding: ByteStreamEncoding::Matroska };
        assert!(d.intercept_caps(&mkv).is_err());
    }

    #[tokio::test]
    async fn selects_video_or_audio_from_a_stream() {
        let v0 = [0u8, 0, 0, 5, 0x65, 0x11];
        let v1 = [0u8, 0, 0, 5, 0x41, 0x22];
        let a0 = [0x33u8, 0x44];
        let a1 = [0x55u8, 0x66];
        let stream = flv_stream(&[
            tag(9, 0, &avc_nalu(&v0)),
            tag(8, 0, &aac_raw(&a0)),
            tag(9, 40, &avc_nalu(&v1)),
            tag(8, 40, &aac_raw(&a1)),
        ]);

        // Default selects H.264: only the two video AUs come out, PTS in ns.
        let mut video = FlvDemux::new();
        video.configure_pipeline(&FlvDemux::input_caps()).unwrap();
        let mut vsink = CaptureSink::default();
        run_demux(&mut video, &stream, &mut vsink).await;
        assert_eq!(vsink.frames, alloc::vec![v0.to_vec(), v1.to_vec()], "video only");
        assert_eq!(vsink.pts, alloc::vec![0, 40_000_000], "ms timestamps to ns");
        assert_eq!(video.emitted(), 2);

        // stream=aac selects AAC: only the two audio frames come out.
        let mut audio = FlvDemux::new().with_stream(FlvStream::Aac);
        audio.configure_pipeline(&FlvDemux::input_caps()).unwrap();
        let mut asink = CaptureSink::default();
        run_demux(&mut audio, &stream, &mut asink).await;
        assert_eq!(asink.frames, alloc::vec![a0.to_vec(), a1.to_vec()], "audio only");
    }

    fn amf0_string(s: &str) -> Vec<u8> {
        let mut v = alloc::vec![0x02u8]; // STRING marker
        v.extend_from_slice(&(s.len() as u16).to_be_bytes());
        v.extend_from_slice(s.as_bytes());
        v
    }

    fn amf0_number(n: f64) -> Vec<u8> {
        let mut v = alloc::vec![0x00u8]; // NUMBER marker
        v.extend_from_slice(&n.to_be_bytes());
        v
    }

    /// An `onMetaData` script body: the event name + an ECMA array of `props`
    /// (each value already AMF0-encoded with its marker).
    fn on_metadata(props: &[(&str, Vec<u8>)]) -> Vec<u8> {
        let mut b = amf0_string("onMetaData");
        b.push(0x08); // ECMA_ARRAY
        b.extend_from_slice(&(props.len() as u32).to_be_bytes());
        for (k, v) in props {
            b.extend_from_slice(&(k.len() as u16).to_be_bytes());
            b.extend_from_slice(k.as_bytes());
            b.extend_from_slice(v);
        }
        b.extend_from_slice(&0u16.to_be_bytes()); // empty key
        b.push(0x09); // OBJECT_END
        b
    }

    #[test]
    fn parse_on_metadata_extracts_string_tags() {
        let body = on_metadata(&[
            ("width", amf0_number(1280.0)),
            ("encoder", amf0_string("Lavf58.76.100")),
            ("title", amf0_string("Clip")),
        ]);
        let tags = parse_on_metadata(&body);
        // The number is walked past; the two string fields become typed tags.
        assert_eq!(
            tags.tags(),
            &[Tag::Encoder("Lavf58.76.100".into()), Tag::Title("Clip".into())]
        );
        // A body that is not onMetaData yields nothing.
        assert!(parse_on_metadata(&amf0_string("onCuePoint")).is_empty());
    }

    #[test]
    fn parse_on_metadata_bounds_nesting_depth() {
        // A pathologically nested object must not overflow the stack: parsing
        // bails at the depth cap and returns gracefully.
        let mut body = amf0_string("onMetaData");
        body.push(0x03); // OBJECT
        // Many levels of `{"a": {` opened and never closed.
        for _ in 0..10_000 {
            body.extend_from_slice(&(1u16).to_be_bytes());
            body.push(b'a');
            body.push(0x03); // nested OBJECT value
        }
        assert!(parse_on_metadata(&body).is_empty());
    }

    #[tokio::test]
    async fn posts_on_metadata_tags_on_the_bus() {
        use g2g_core::Bus;
        let (bus, handle) = Bus::new(8);
        let meta = on_metadata(&[("width", amf0_number(640.0)), ("encoder", amf0_string("g2g"))]);
        let stream = flv_stream(&[tag(18, 0, &meta), tag(9, 0, &avc_nalu(&[0x65, 0xAA]))]);

        let mut d = FlvDemux::new().with_bus(handle);
        d.configure_pipeline(&FlvDemux::input_caps()).unwrap();
        let mut sink = CaptureSink::default();
        run_demux(&mut d, &stream, &mut sink).await;

        assert_eq!(sink.frames, alloc::vec![alloc::vec![0x65, 0xAA]], "the video AU still flows");
        let mut posted = None;
        while let Some(m) = bus.try_recv() {
            if let BusMessage::Tag(t) = m {
                posted = Some(t);
            }
        }
        assert_eq!(posted.expect("a Tag message was posted").tags(), &[Tag::Encoder("g2g".into())]);
    }

    #[test]
    fn output_caps_track_the_selection() {
        assert!(matches!(
            FlvDemux::output_caps(FlvStream::H264),
            Caps::CompressedVideo { codec: VideoCodec::H264, .. }
        ));
        assert!(matches!(
            FlvDemux::output_caps(FlvStream::Aac),
            Caps::Audio { format: AudioFormat::Aac, .. }
        ));
    }

    #[test]
    fn stream_property_round_trips_and_drives_output() {
        let mut d = FlvDemux::new();
        assert_eq!(d.get_property("stream"), Some(PropValue::Str("h264".into())));

        d.set_property("stream", PropValue::Str("aac".into())).unwrap();
        assert_eq!(d.stream(), FlvStream::Aac);

        // An unsupported codec name is rejected, leaving the selection unchanged.
        assert_eq!(d.set_property("stream", PropValue::Str("vp9".into())), Err(PropError::Value));
        assert_eq!(d.stream(), FlvStream::Aac);

        let CapsConstraint::DerivedOutput(f) = d.caps_constraint_as_transform() else {
            panic!("expected DerivedOutput");
        };
        let out = f(&FlvDemux::input_caps());
        assert!(matches!(out.alternatives(), [Caps::Audio { format: AudioFormat::Aac, .. }]));
    }
}
