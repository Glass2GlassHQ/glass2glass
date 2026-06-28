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
//! Seeking uses the `Cues` index when it has been parsed (M373): the demuxer
//! byte-seeks straight to the target Cluster, keeping its parser state, instead
//! of the M364 re-scan from offset 0 (the fallback when `Cues` are not yet known).
//!
//! Scope (v1): the first track of the selected codec; multi-track-of-one-codec
//! selection and lacing are follow-ups (DESIGN.md §4.17). `Cues` placed at the
//! end of the file (the common layout) are only available for the fast path once
//! read past during playback; a `SeekHead`-driven prefetch is a follow-up (the
//! re-scan fallback handles the not-yet-known case correctly meanwhile).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SeekController;
use g2g_core::{
    AsyncElement, AudioFormat, BusHandle, BusMessage, ByteStreamEncoding, Caps, CapsConstraint,
    CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate,
    PadTemplates, PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate, Seek, Segment,
    TagList, VideoCodec,
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
    /// Seek support (M362): app time seeks drive an upstream byte-seek and a
    /// re-sync. Inert unless `with_seek` wired the controllers.
    seek: DemuxSeek,
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
            seek: DemuxSeek::default(),
        }
    }

    /// Make the demuxer seekable (M362): `app` carries app time seeks; `upstream`
    /// is the byte source's ([`FileSrc`](crate::filesrc)) byte-seek controller.
    /// On a time seek the demuxer rewinds the source and re-syncs from the
    /// keyframe at/after the target.
    pub fn with_seek(mut self, app: SeekController, upstream: SeekController) -> Self {
        self.seek.with(app, upstream);
        self
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
        Caps::ByteStream { encoding: ByteStreamEncoding::Matroska }
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
            MkvStream::Aac => Caps::Audio { format: AudioFormat::Aac, channels: 0, sample_rate: 0 },
            MkvStream::Opus => Caps::Audio { format: AudioFormat::Opus, channels: 0, sample_rate: 0 },
        }
    }

    fn compressed_video(codec: VideoCodec) -> Caps {
        Caps::CompressedVideo {
            codec,
            width: Dim::Range { min: 16, max: 65_535 },
            height: Dim::Range { min: 16, max: 65_535 },
            framerate: Rate::Range { min_q16: 1 << 16, max_q16: 240 << 16 },
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
        }
    }

    /// The concrete caps for the selected track once Tracks has been parsed,
    /// with real geometry / audio parameters. `None` until the track is known.
    fn concrete_caps(&self) -> Option<Caps> {
        let want = Self::selected_codec(self.stream);
        let track = self.demux.tracks().iter().find(|t| t.codec == want)?;
        match Self::output_caps(self.stream) {
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
            _ => None,
        }
    }

    /// Post any tags parsed since the last call as a [`BusMessage::Tag`]. A
    /// no-op without a bus attached or when nothing new has been parsed.
    fn post_tags(&mut self) {
        let total = self.demux.tags().len();
        if total <= self.tags_posted {
            return;
        }
        let fresh: TagList = self.demux.tags().tags()[self.tags_posted..].iter().cloned().collect();
        self.tags_posted = total;
        if let Some(bus) = &self.bus {
            bus.try_post(BusMessage::Tag(fresh));
        }
    }

    /// Emit a `CapsChanged` once the selected track's concrete caps are known,
    /// then forward each demuxed frame of that stream.
    async fn emit_ready(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        if self.bus.is_some() {
            self.post_tags();
        }
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
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(f.data.into_boxed_slice())),
                FrameTiming { pts_ns: f.pts_ns, dts_ns: f.pts_ns, ..FrameTiming::default() },
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
            Caps::ByteStream { encoding: ByteStreamEncoding::Matroska } => {
                CapsSet::one(Self::output_caps(stream))
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !matches!(absolute_caps, Caps::ByteStream { encoding: ByteStreamEncoding::Matroska }) {
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
            // M362/M373: a pending app seek triggers an upstream byte-seek; until
            // its `Flush` returns, drop input so no stale pre-seek frames are
            // emitted. If the `Cues` index is already parsed, seek straight to the
            // target Cluster (mid-segment, parser state kept); else re-scan from 0.
            {
                let demux = &self.demux;
                self.seek.poll_request_indexed(|target| demux.cue_seek_offset(target));
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    if self.seek.dropping_input() {
                        return Ok(());
                    }
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.demux.push_data(slice.as_slice());
                    self.emit_ready(out).await?;
                }
                // The upstream byte-seek's flush: reset the parser, then re-sync
                // from the re-read stream. A `Cues` (indexed) seek landed inside
                // the Segment, so keep the Tracks / TimestampScale / Cues the
                // landing point does not re-send; a from-start re-scan fully resets
                // (it re-reads the EBML header). Forward the flush downstream.
                PipelinePacket::Flush => {
                    self.seek.on_flush();
                    if self.seek.keeps_state() {
                        self.demux.reset_keeping_tracks();
                    } else {
                        self.reset_parser();
                    }
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

fn mkv_stream_from_str(s: &str) -> Option<MkvStream> {
    match s {
        "h264" => Some(MkvStream::H264),
        "h265" => Some(MkvStream::H265),
        "vp8" => Some(MkvStream::Vp8),
        "vp9" => Some(MkvStream::Vp9),
        "av1" => Some(MkvStream::Av1),
        "aac" => Some(MkvStream::Aac),
        "opus" => Some(MkvStream::Opus),
        _ => None,
    }
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
        ]));
        Vec::from([PadTemplate::sink(CapsSet::one(Self::input_caps())), PadTemplate::source(source)])
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
        let v = [elem(&[0xB0], &uint_body(w as u64)), elem(&[0xBA], &uint_body(h as u64))].concat();
        let body = [elem(&[0xD7], &uint_body(num)), elem(&[0x86], codec), elem(&[0xE0], &v)].concat();
        elem(&[0xAE], &body)
    }

    fn audio_track(num: u64, codec: &[u8], ch: u8, sr: u32) -> Vec<u8> {
        let mut a = elem(&[0x9F], &uint_body(ch as u64));
        a.extend_from_slice(&elem(&[0xB5], &(sr as f32).to_be_bytes()));
        let body = [elem(&[0xD7], &uint_body(num)), elem(&[0x86], codec), elem(&[0xE1], &a)].concat();
        elem(&[0xAE], &body)
    }

    fn webm() -> Vec<u8> {
        let tracks = elem(
            &[0x16, 0x54, 0xAE, 0x6B],
            &[video_track(1, b"V_VP9", 320, 240), audio_track(2, b"A_OPUS", 2, 48_000)].concat(),
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
        d.process(PipelinePacket::DataFrame(frame), sink).await.unwrap();
        d.process(PipelinePacket::Eos, sink).await.unwrap();
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
        let ts = Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs };
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
        assert_eq!(sink.frames, alloc::vec![alloc::vec![0x11, 0x22], alloc::vec![0x55, 0x66]]);
        assert!(!sink.eos, "EOS is forwarded by the runner's arm, not the element");
    }

    #[tokio::test]
    async fn selects_audio_with_refined_params() {
        let mut sink = CaptureSink::default();
        run(MkvStream::Opus, &webm(), &mut sink).await;

        assert_eq!(
            sink.caps,
            alloc::vec![Caps::Audio { format: AudioFormat::Opus, channels: 2, sample_rate: 48_000 }]
        );
        assert_eq!(sink.frames, alloc::vec![alloc::vec![0x33, 0x44]]);
    }

    #[test]
    fn output_caps_track_the_selection() {
        assert!(matches!(
            MkvDemux::output_caps(MkvStream::Vp8),
            Caps::CompressedVideo { codec: VideoCodec::Vp8, .. }
        ));
        assert!(matches!(
            MkvDemux::output_caps(MkvStream::Opus),
            Caps::Audio { format: AudioFormat::Opus, .. }
        ));
    }

    /// A WebM with a Segment `Title`, a VP9 track, a `Tags` element, then one
    /// Cluster frame.
    fn webm_with_tags() -> Vec<u8> {
        let info = elem(&[0x15, 0x49, 0xA9, 0x66], &elem(&[0x7B, 0xA9], b"My Clip")); // Info/Title
        let tracks = elem(&[0x16, 0x54, 0xAE, 0x6B], &video_track(1, b"V_VP9", 320, 240));
        let simple = [elem(&[0x45, 0xA3], b"ARTIST"), elem(&[0x44, 0x87], b"Band")].concat();
        let tag = [elem(&[0x63, 0xC0], &[]), elem(&[0x67, 0xC8], &simple)].concat();
        let tags = elem(&[0x12, 0x54, 0xC3, 0x67], &elem(&[0x73, 0x73], &tag));
        let cluster = elem(
            &[0x1F, 0x43, 0xB6, 0x75],
            &[elem(&[0xE7], &uint_body(0)), elem(&[0xA3], &block(1, 0, &[0x11, 0x22]))].concat(),
        );
        let segment = elem(&[0x18, 0x53, 0x80, 0x67], &[info, tracks, tags, cluster].concat());
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
        d.process(PipelinePacket::DataFrame(frame), &mut sink).await.unwrap();

        let mut posted = TagList::new();
        while let Some(m) = bus.try_recv() {
            if let BusMessage::Tag(t) = m {
                for tag in t.tags() {
                    posted.push(tag.clone());
                }
            }
        }
        assert_eq!(posted.tags(), &[Tag::Title("My Clip".into()), Tag::Artist("Band".into())]);
        // The selected video frame still flows while the tags go out of band.
        assert_eq!(sink.frames, alloc::vec![alloc::vec![0x11, 0x22]]);
    }

    #[test]
    fn stream_property_round_trips() {
        let mut d = MkvDemux::new();
        assert_eq!(d.get_property("stream"), Some(PropValue::Str("vp9".into())));
        d.set_property("stream", PropValue::Str("opus".into())).unwrap();
        assert_eq!(d.stream(), MkvStream::Opus);
        assert_eq!(d.set_property("stream", PropValue::Str("theora".into())), Err(PropError::Value));
    }
}
