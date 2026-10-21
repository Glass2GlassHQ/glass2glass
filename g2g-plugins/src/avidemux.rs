//! AVI demuxer elements (M1071): `ByteStream{Avi}` in, the elementary streams
//! the `movi` chunks carry out. [`AviDemux`] emits one stream (the `stream=`
//! selection, `filesrc location=x.avi ! avidemux ! ...`), [`AviDemuxN`] emits
//! one per output port (`avidemux name=d  d.video_0 ! ...  d.audio_0 ! ...`),
//! the same pair as `qtdemux` / [`Mp4DemuxN`](crate::mp4demuxn::Mp4DemuxN).
//!
//! AVI keeps its stream headers at the front but its `idx1` at the end, and a
//! chunk's keyframe flag lives only in that index, so the whole file is
//! buffered and parsed at `Eos` (what [`Mp4Demux`](crate::mp4demux) does for a
//! progressive `.mp4`). A file with no `idx1` still plays: `movi` is walked in
//! order and only an all-intra stream claims keyframes.
//!
//! Payloads are re-framed to what the rest of the pipeline expects: an H.264
//! stream muxed as AVCC converts to Annex-B and takes the `strf` parameter sets
//! ahead of its first access unit, and an AAC stream is ADTS-framed from its
//! `AudioSpecificConfig` the way `qtdemux` does, so `aacparse` and the ffmpeg
//! audio decoder read it unchanged. Everything else is forwarded byte for byte.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, AudioFormat, BusHandle, BusMessage, ByteStreamEncoding, Caps, CapsConstraint,
    CapsSet, ConfigureOutcome, Dim, ElementMetadata, FrameMetaSet, G2gError, MultiOutputElement,
    MultiOutputSink, OutputSink, PadTemplate, PadTemplates, PropError, PropKind, PropValue,
    PropertySpec, Rate, RawVideoFormat, Stream, StreamCollection, StreamType, VideoCodec,
    ANY_CHANNELS, ANY_SAMPLE_RATE,
};

use crate::aacparse::{adts_from_asc, is_raw_aac_config};
use crate::annexb::{
    is_annex_b, length_prefixed_to_annexb, parse_avcc, prepend_param_sets, starts_with_param_set,
};
use crate::avi::{palette_change, parse, parse_streams, AviFile, AviStream, AviStreamKind};
use crate::codectag::{
    audio_from_stream_name, audio_stream_name, video_from_stream_name, video_stream_name,
};
use crate::uncompressedvideo::{
    raw_video_caps, raw_video_formats, raw_video_frame_bytes, raw_video_placeholder_caps,
    raw_video_stream_from_str, raw_video_stream_str, warn_short_raw_video_frame,
};

/// The `stream=` names for a codec-named video selection, so a decode chain
/// negotiates the file's real codec instead of the nominal default.
const VIDEO_STREAM_NAMES: &[(&str, VideoCodec)] = &[
    ("mjpeg", VideoCodec::Mjpeg),
    ("h264", VideoCodec::H264),
    ("mpeg4part2", VideoCodec::Mpeg4Part2),
    ("vp6", VideoCodec::Vp6Plain),
];

/// The `stream=` names for a format-named audio selection.
const AUDIO_STREAM_NAMES: &[(&str, AudioFormat)] = &[
    ("pcm-u8", AudioFormat::PcmU8),
    ("pcm-s16le", AudioFormat::PcmS16Le),
    ("pcm-s24le", AudioFormat::PcmS24Le),
    ("pcm-s32le", AudioFormat::PcmS32Le),
    ("mulaw", AudioFormat::Mulaw),
    ("alaw", AudioFormat::Alaw),
    ("mp3", AudioFormat::Mp3),
    ("ac3", AudioFormat::Ac3),
    ("aac", AudioFormat::Aac),
];

/// A coded video stream is at least one macroblock, and no wider than a
/// `BITMAPINFOHEADER` dimension a decoder would accept.
const MIN_CODED_DIM: u32 = 16;
const MAX_CODED_DIM: u32 = 65_535;
/// The framerate window the negotiation placeholder spans. Never `Rate::Any`,
/// which cannot fixate; the real timing rides each frame's pts.
const MIN_RATE_Q16: u32 = 1 << 16;
const MAX_RATE_Q16: u32 = 240 << 16;

/// Which stream [`AviDemux`] forwards. The named forms additionally pin the
/// startup caps to that codec / format, which the bare-`decodebin` primary
/// stream hook uses so a decoder negotiates the truth up front.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum AviStreamSelect {
    /// The first video stream, its codec refined from the file at `Eos`.
    #[default]
    Video,
    VideoNamed(VideoCodec),
    RawVideo(RawVideoFormat),
    /// The first audio stream, its format refined from the file at `Eos`.
    Audio,
    AudioNamed(AudioFormat),
}

pub(crate) fn video_stream_str(codec: VideoCodec) -> Option<String> {
    video_stream_name(VIDEO_STREAM_NAMES, codec)
}

pub(crate) fn audio_stream_str(format: AudioFormat) -> Option<String> {
    audio_stream_name(AUDIO_STREAM_NAMES, format)
}

fn select_from_str(name: &str) -> Option<AviStreamSelect> {
    match name {
        "video" => Some(AviStreamSelect::Video),
        "audio" => Some(AviStreamSelect::Audio),
        _ => video_from_stream_name(VIDEO_STREAM_NAMES, name)
            .map(AviStreamSelect::VideoNamed)
            .or_else(|| raw_video_stream_from_str(name).map(AviStreamSelect::RawVideo))
            .or_else(|| {
                audio_from_stream_name(AUDIO_STREAM_NAMES, name).map(AviStreamSelect::AudioNamed)
            }),
    }
}

fn input_caps() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::Avi,
    }
}

/// The negotiation caps of a video codec: a `Fixed` geometry once the file is
/// probed, a `Range` framerate because AVI's per-chunk timing is what carries
/// the real rate.
fn video_caps(codec: VideoCodec, width: Dim, height: Dim) -> Caps {
    Caps::CompressedVideo {
        codec,
        width,
        height,
        framerate: framerate_span(),
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn framerate_span() -> Rate {
    Rate::Range {
        min_q16: MIN_RATE_Q16,
        max_q16: MAX_RATE_Q16,
    }
}

/// The startup placeholder for a stream nothing has probed yet.
fn video_placeholder_caps(codec: VideoCodec) -> Caps {
    let span = || Dim::Range {
        min: MIN_CODED_DIM,
        max: MAX_CODED_DIM,
    };
    video_caps(codec, span(), span())
}

/// Compressed audio negotiates on the `0/0` "unknown until parsed" layout, the
/// convention `tsdemux` / `qtdemux` share; PCM has no parser downstream to
/// refine it, so it negotiates and runs on the layout the `strf` declared.
fn audio_caps(format: AudioFormat, channels: u8, sample_rate: u32) -> Caps {
    Caps::Audio {
        format,
        channels,
        sample_rate,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    }
}

/// The layout a PCM selection negotiates on before the `strf` is read: a
/// concrete default first so the link can fixate, the wildcard second so a
/// downstream `rate=` pin intersects to it (a lone wildcard cannot fixate, M754).
/// The real layout arrives with the `CapsChanged` the parsed header emits, the
/// way [`WavParse`](crate::wavparse::WavParse) does it.
const DEFAULT_PCM_SAMPLE_RATE: u32 = 48_000;
const DEFAULT_PCM_CHANNELS: u8 = 2;

pub(crate) fn pcm_alternatives(format: AudioFormat) -> Vec<Caps> {
    Vec::from([
        audio_caps(format, DEFAULT_PCM_CHANNELS, DEFAULT_PCM_SAMPLE_RATE),
        audio_caps(format, ANY_CHANNELS, ANY_SAMPLE_RATE),
    ])
}

pub(crate) fn is_pcm(format: AudioFormat) -> bool {
    matches!(
        format,
        AudioFormat::PcmU8
            | AudioFormat::PcmS16Le
            | AudioFormat::PcmS24Le
            | AudioFormat::PcmS32Le
            | AudioFormat::PcmF32Le
    )
}

/// A stream's negotiation caps, and whether it is video.
fn nego_caps(kind: &AviStreamKind) -> Option<(Caps, bool)> {
    match kind {
        AviStreamKind::Video {
            codec,
            width,
            height,
        } => Some((
            video_caps(*codec, Dim::Fixed(*width), Dim::Fixed(*height)),
            true,
        )),
        AviStreamKind::RawVideo {
            format,
            width,
            height,
        } => Some((
            raw_video_caps(
                *format,
                Dim::Fixed(*width),
                Dim::Fixed(*height),
                framerate_span(),
            ),
            true,
        )),
        AviStreamKind::Audio {
            format,
            channels,
            sample_rate,
        } if is_pcm(*format) => Some((audio_caps(*format, *channels, *sample_rate), false)),
        AviStreamKind::Audio { format, .. } => Some((audio_caps(*format, 0, 0), false)),
        AviStreamKind::Unsupported { .. } => None,
    }
}

/// A stream's concrete caps: the runtime `CapsChanged` refinement, and what the
/// `StreamCollection` publishes.
fn real_caps(kind: &AviStreamKind) -> Option<Caps> {
    match kind {
        AviStreamKind::Audio {
            format,
            channels,
            sample_rate,
        } => Some(audio_caps(*format, *channels, *sample_rate)),
        _ => nego_caps(kind).map(|(caps, _)| caps),
    }
}

/// One forwardable stream of a probed AVI: its stream number, the caps a decode
/// branch plugs from, and whether it is video. The `playbin` / `decodebin`
/// fan-out builds one branch per entry.
#[derive(Debug, Clone)]
pub struct AviStreamInfo {
    pub stream: usize,
    pub caps: Caps,
    pub video: bool,
}

/// The forwardable streams an AVI file carries, in `hdrl` order. `data` needs
/// to hold the `hdrl` list (a file prefix is enough); returns empty for a
/// non-AVI or unparseable input, which a hook reads as "decline".
///
/// A prefix that stops anywhere past `hdrl` still yields the streams: only the
/// header is read.
pub fn forwardable_streams(data: &[u8]) -> Vec<AviStreamInfo> {
    let Ok(streams) = parse_streams(data) else {
        return Vec::new();
    };
    streams
        .iter()
        .enumerate()
        .filter_map(|(stream, s)| {
            let (caps, video) = nego_caps(&s.kind)?;
            Some(AviStreamInfo {
                stream,
                caps,
                video,
            })
        })
        .collect()
}

pub fn unsupported_codecs(data: &[u8]) -> Vec<alloc::string::String> {
    let Ok(streams) = parse_streams(data) else {
        return Vec::new();
    };
    streams
        .into_iter()
        .filter_map(|stream| match stream.kind {
            AviStreamKind::Unsupported { codec } => codec,
            _ => None,
        })
        .collect()
}

/// The `LIST INFO` metadata an AVI file carries, the tags both demuxers post on
/// the bus. `None` for a non-AVI or unparseable input.
pub fn probe_tags(data: &[u8]) -> Option<g2g_core::TagList> {
    parse(data).ok().map(|file| file.tags)
}

/// The id a stream carries in the `StreamCollection`.
fn stream_id(stream: usize) -> String {
    alloc::format!("avi-stream-{stream}")
}

// an H.264 stream's out-of-band parameter sets, and the NAL length width when its chunks are length-prefixed
#[derive(Debug, Default)]
struct H264Config {
    param_sets: Vec<Vec<u8>>,
    nal_length_size: Option<usize>,
}

/// Read a video stream's `strf` extradata. An H.264 writer stores either the
/// raw NALs there, and then writes Annex-B chunks, or an `avcC` record, and
/// then writes chunks length-prefixed to the record's width. Anything else (an
/// `strf` with no extradata, an unparseable record) means Annex-B chunks that
/// carry their config in band.
fn h264_config(stream: &AviStream) -> H264Config {
    let AviStreamKind::Video {
        codec: VideoCodec::H264,
        ..
    } = stream.kind
    else {
        return H264Config::default();
    };
    let config = &stream.codec_config;
    if is_annex_b(config) {
        return H264Config {
            param_sets: crate::annexb::split_annexb(config)
                .into_iter()
                .map(Vec::from)
                .collect(),
            nal_length_size: None,
        };
    }
    parse_avcc(config).map_or_else(
        |_| H264Config::default(),
        |record| H264Config {
            param_sets: record.parameter_sets,
            nal_length_size: Some(record.nal_length_size),
        },
    )
}

/// Re-frame one chunk's payload for the pipeline: AVCC H.264 to Annex-B with
/// the out-of-band parameter sets ahead of the first access unit, raw AAC to
/// ADTS where ADTS can carry its config, uncompressed video cut to its pixels,
/// everything else untouched. `first` is set only for the stream's opening
/// chunk. `None` drops an uncompressed frame too short for its geometry.
fn deframe(stream: &AviStream, h264: &H264Config, data: &[u8], first: bool) -> Option<Vec<u8>> {
    Some(match &stream.kind {
        AviStreamKind::Video {
            codec: VideoCodec::H264,
            ..
        } => {
            let mut annexb = match h264.nal_length_size {
                // The lengths come from the file, so a walk that does not consume
                // the chunk exactly leaves the original bytes rather than
                // emitting a mis-framed stream.
                Some(len_size) => {
                    length_prefixed_to_annexb(data, len_size).unwrap_or_else(|| Vec::from(data))
                }
                None => Vec::from(data),
            };
            if first
                && !h264.param_sets.is_empty()
                && !starts_with_param_set(&annexb, VideoCodec::H264)
            {
                annexb = prepend_param_sets(&annexb, &h264.param_sets, VideoCodec::H264);
            }
            annexb
        }
        AviStreamKind::Audio {
            format: AudioFormat::Aac,
            ..
        } => adts_from_asc(&stream.codec_config, data).unwrap_or_else(|| Vec::from(data)),
        AviStreamKind::RawVideo {
            format,
            width,
            height,
        } => {
            let Some(pixels) = raw_video_frame_bytes(*format, *width, *height, data.len()) else {
                warn_short_raw_video_frame("avidemux", *format, *width, *height, data.len());
                return None;
            };
            Vec::from(&data[..pixels])
        }
        _ => Vec::from(data),
    })
}

/// One chunk ready to leave the demuxer: which stream, the re-framed payload,
/// and its reconstructed timing.
#[derive(Debug)]
struct Emitted {
    stream: usize,
    data: Vec<u8>,
    timing: FrameTiming,
    meta: FrameMetaSet,
}

impl Emitted {
    fn into_frame(self, sequence: u64) -> Frame {
        Frame {
            meta: self.meta,
            ..frame_of(self.data, self.timing, sequence)
        }
    }
}

#[cfg(feature = "metadata")]
fn palette_meta(colors: Vec<(u8, [u8; 3])>) -> FrameMetaSet {
    let mut meta = FrameMetaSet::new();
    if !colors.is_empty() {
        meta.attach(g2g_core::meta::PaletteChange { colors });
    }
    meta
}

#[cfg(not(feature = "metadata"))]
fn palette_meta(_colors: Vec<(u8, [u8; 3])>) -> FrameMetaSet {
    FrameMetaSet::new()
}

// The config to send as a frame of its own ahead of a stream's units: an AAC
// AudioSpecificConfig ADTS cannot carry (ER AAC, ELD, USAC), or a tagged codec's
// whole `strf` header.
fn lead_config(stream: &AviStream) -> Option<&[u8]> {
    match stream.kind {
        AviStreamKind::Audio {
            format: AudioFormat::Aac,
            ..
        } => Some(stream.codec_config.as_slice()).filter(|config| is_raw_aac_config(config)),
        AviStreamKind::Video {
            codec: VideoCodec::Tagged(_),
            ..
        }
        | AviStreamKind::Audio {
            format: AudioFormat::Tagged(_),
            ..
        } => Some(stream.codec_config.as_slice()),
        _ => None,
    }
}

/// Walk a parsed file's chunks of the `selected` streams into emittable frames,
/// reconstructing each stream's timing from its `strh`. AVI stores chunks in
/// decode order with no presentation timestamp, so dts equals pts and a stream
/// with B-frames is reordered by the decoder downstream. A raw AAC stream's
/// config, or a tagged codec's header, goes out as a frame of its own ahead of
/// its first chunk.
fn emittable<'a>(
    data: &'a [u8],
    file: &'a AviFile,
    selected: &'a [usize],
) -> impl Iterator<Item = Emitted> + 'a {
    let mut positions = alloc::vec![0u64; file.streams.len()];
    let mut first = alloc::vec![true; file.streams.len()];
    let mut palette_changes: Vec<Vec<(u8, [u8; 3])>> = alloc::vec![Vec::new(); file.streams.len()];
    let h264_configs: Vec<H264Config> = file.streams.iter().map(h264_config).collect();
    file.chunks
        .iter()
        .filter(move |chunk| {
            selected.contains(&chunk.stream)
                && !matches!(
                    file.streams[chunk.stream].kind,
                    AviStreamKind::Unsupported { .. }
                )
        })
        .flat_map(move |chunk| {
            let stream = &file.streams[chunk.stream];
            let bytes = data.get(chunk.body.clone()).unwrap_or_default();
            if chunk.palette_change {
                palette_changes[chunk.stream].extend(palette_change(bytes));
                return None.into_iter().chain(None);
            }
            let samples = stream.samples_in(bytes.len());
            let (pts_ns, duration_ns) = stream.timing(positions[chunk.stream], samples);
            positions[chunk.stream] += samples;
            let payload = deframe(
                stream,
                &h264_configs[chunk.stream],
                bytes,
                first[chunk.stream],
            );
            let timing = FrameTiming {
                pts_ns,
                dts_ns: pts_ns,
                duration_ns,
                capture_ns: pts_ns,
                keyframe: chunk.keyframe,
                ..FrameTiming::default()
            };
            let lead_config = lead_config(stream)
                .filter(|_| first[chunk.stream])
                .map(|config| Emitted {
                    stream: chunk.stream,
                    data: Vec::from(config),
                    timing: FrameTiming::default(),
                    meta: FrameMetaSet::new(),
                });
            first[chunk.stream] = false;
            let meta = palette_meta(core::mem::take(&mut palette_changes[chunk.stream]));
            lead_config.into_iter().chain(payload.map(|data| Emitted {
                stream: chunk.stream,
                data,
                timing,
                meta,
            }))
        })
}

fn frame_of(data: Vec<u8>, timing: FrameTiming, sequence: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        timing,
        sequence,
        meta: Default::default(),
    }
}

/// The `StreamCollection` a parsed file publishes, or `None` when it carries no
/// stream g2g forwards.
fn stream_collection(file: &AviFile) -> Option<StreamCollection> {
    let streams: Vec<Stream> = file
        .streams
        .iter()
        .enumerate()
        .filter_map(|(index, s)| {
            let ty = match s.kind {
                AviStreamKind::Video { .. } | AviStreamKind::RawVideo { .. } => StreamType::Video,
                AviStreamKind::Audio { .. } => StreamType::Audio,
                AviStreamKind::Unsupported { .. } => return None,
            };
            Some(Stream::new(stream_id(index), ty, real_caps(&s.kind)?))
        })
        .collect();
    (!streams.is_empty()).then(|| StreamCollection::new("avi-0", streams))
}

/// Demuxes an AVI byte stream into one of its elementary streams.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::avidemux::AviDemux;
///
/// // gst-launch equivalent: filesrc location=clip.avi ! avidemux ! decodebin
/// let demux = AviDemux::new();
/// ```
#[derive(Debug, Default)]
pub struct AviDemux {
    /// The whole file: the `idx1` keyframe flags sit at the end, so nothing is
    /// emitted before the last byte arrives.
    buffer: Vec<u8>,
    select: AviStreamSelect,
    configured: bool,
    drained: bool,
    sequence: u64,
    bus: Option<BusHandle>,
}

impl AviDemux {
    pub fn new() -> Self {
        Self::default()
    }

    /// Forward this stream instead of the default first video one.
    pub fn with_stream(mut self, select: AviStreamSelect) -> Self {
        self.select = select;
        self
    }

    /// Attach the pipeline bus so the file's streams post as a
    /// `StreamCollection` and its `LIST INFO` metadata as tags.
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.bus = Some(bus);
        self
    }

    /// Count of frames forwarded.
    pub fn emitted(&self) -> u64 {
        self.sequence
    }

    /// The caps the current selection negotiates on, before any byte is read.
    fn nego_alternatives(&self) -> Vec<Caps> {
        match self.select {
            AviStreamSelect::Video => Vec::from([video_placeholder_caps(VideoCodec::H264)]),
            AviStreamSelect::VideoNamed(codec) => Vec::from([video_placeholder_caps(codec)]),
            AviStreamSelect::RawVideo(format) => Vec::from([raw_video_placeholder_caps(format)]),
            AviStreamSelect::Audio => pcm_alternatives(AudioFormat::PcmS16Le),
            AviStreamSelect::AudioNamed(format) if is_pcm(format) => pcm_alternatives(format),
            AviStreamSelect::AudioNamed(format) => Vec::from([audio_caps(format, 0, 0)]),
        }
    }

    /// The stream number the selection picks in a parsed file.
    fn selected(&self, file: &AviFile) -> Option<usize> {
        let want_video = matches!(
            self.select,
            AviStreamSelect::Video | AviStreamSelect::VideoNamed(_) | AviStreamSelect::RawVideo(_)
        );
        let of_wanted_kind = |kind: &AviStreamKind| match kind {
            AviStreamKind::Video { .. } | AviStreamKind::RawVideo { .. } => want_video,
            AviStreamKind::Audio { .. } => !want_video,
            AviStreamKind::Unsupported { .. } => false,
        };
        // A named selection points at the stream carrying that codec / format,
        // so a file with two audio streams of different formats picks the one
        // the caller negotiated against. It falls back to the first stream of
        // the kind, which is what a bare `video` / `audio` selection takes.
        let named = |kind: &AviStreamKind| match (self.select, kind) {
            (AviStreamSelect::VideoNamed(want), AviStreamKind::Video { codec, .. }) => {
                *codec == want
            }
            (AviStreamSelect::RawVideo(want), AviStreamKind::RawVideo { format, .. }) => {
                *format == want
            }
            (AviStreamSelect::AudioNamed(want), AviStreamKind::Audio { format, .. }) => {
                *format == want
            }
            _ => false,
        };
        file.streams
            .iter()
            .position(|s| named(&s.kind))
            .or_else(|| file.streams.iter().position(|s| of_wanted_kind(&s.kind)))
    }

    /// Parse the buffered file and emit the selected stream. Runs once, at
    /// `Eos`: the concrete caps first, then every chunk in file order.
    async fn drain(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        if self.drained {
            return Ok(());
        }
        self.drained = true;
        let file = parse(&self.buffer)?;
        let selected = self.selected(&file).ok_or(G2gError::CapsMismatch)?;
        if let Some(bus) = &self.bus {
            if let Some(collection) = stream_collection(&file) {
                bus.try_post(BusMessage::StreamCollection(collection));
            }
            if !file.tags.is_empty() {
                bus.try_post(BusMessage::Tag {
                    tags: file.tags.clone(),
                    program: None,
                });
            }
        }
        let caps = real_caps(&file.streams[selected].kind).ok_or(G2gError::CapsMismatch)?;
        out.push(PipelinePacket::CapsChanged(caps)).await?;
        for emitted in emittable(&self.buffer, &file, &[selected]) {
            let frame = emitted.into_frame(self.sequence);
            self.sequence += 1;
            out.push(PipelinePacket::DataFrame(frame)).await?;
        }
        Ok(())
    }
}

/// `AviDemux`'s settable properties. `stream` picks the emitted stream, the
/// single-output analog of `qtdemux`'s `stream`.
static AVIDEMUX_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "stream",
    PropKind::Str,
    "stream to emit: video (the default) | mjpeg | h264 | mpeg4part2 | vp6 (the video stream by codec) | raw-i420 | raw-yuy2 | raw-uyvy | raw-v210 (the uncompressed video stream by layout) | audio | pcm-u8 | pcm-s16le | pcm-s24le | pcm-s32le | mulaw | alaw | mp3 | ac3 | aac (the audio stream by format) | video-riff-<fourcc> | audio-riff-0x<tag> (a codec known only by its tag, with ffmpeg-long-tail)",
)
.with_default("video")];

impl AsyncElement for AviDemux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "AVI demuxer",
            "Codec/Demuxer",
            "Demuxes an AVI byte stream to one of its elementary streams",
            "g2g",
        )
    }

    /// Reads host memory, so it takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&input_caps())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let nego = self.nego_alternatives();
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match input {
            Caps::ByteStream {
                encoding: ByteStreamEncoding::Avi,
            } => CapsSet::from_alternatives(nego.clone()),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !matches!(
            absolute_caps,
            Caps::ByteStream {
                encoding: ByteStreamEncoding::Avi
            }
        ) {
            return Err(G2gError::CapsMismatch);
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        AVIDEMUX_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "stream" => {
                self.select = select_from_str(value.as_str().ok_or(PropError::Type)?)
                    .ok_or(PropError::Value)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "stream" => Some(PropValue::Str(match self.select {
                AviStreamSelect::Video => String::from("video"),
                AviStreamSelect::Audio => String::from("audio"),
                // Both come from the tables the names were parsed from.
                AviStreamSelect::VideoNamed(codec) => video_stream_str(codec)?,
                AviStreamSelect::RawVideo(format) => String::from(raw_video_stream_str(format)?),
                AviStreamSelect::AudioNamed(format) => audio_stream_str(format)?,
            })),
            _ => None,
        }
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
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let slice = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    self.buffer.extend_from_slice(slice);
                }
                // The whole file is in hand: parse and emit, then the runner's
                // transform arm forwards the EOS.
                PipelinePacket::Eos => self.drain(out).await?,
                // A flushing seek re-reads the file from the start.
                PipelinePacket::Flush => {
                    self.buffer.clear();
                    self.drained = false;
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for AviDemux {
    fn pad_templates() -> Vec<PadTemplate> {
        let mut outputs: Vec<Caps> = VIDEO_STREAM_NAMES
            .iter()
            .map(|(_, codec)| video_placeholder_caps(*codec))
            .chain(raw_video_formats().map(raw_video_placeholder_caps))
            .collect();
        for (_, format) in AUDIO_STREAM_NAMES {
            if is_pcm(*format) {
                outputs.extend(pcm_alternatives(*format));
            } else {
                outputs.push(audio_caps(*format, 0, 0));
            }
        }
        Vec::from([
            PadTemplate::sink(CapsSet::one(input_caps())),
            PadTemplate::source(CapsSet::from_alternatives(outputs)),
        ])
    }
}

/// Multi-output AVI demuxer: one byte stream in, one elementary stream per
/// output port (`avidemux name=d  d.video_0 ! ...  d.audio_0 ! ...`).
#[derive(Debug)]
pub struct AviDemuxN {
    buffer: Vec<u8>,
    /// Port `i` forwards `ports[i]`, an AVI stream number.
    ports: Vec<usize>,
    /// Port `i`'s startup caps, from the probe that built the fan-out.
    port_caps: Vec<Caps>,
    bus: Option<BusHandle>,
    drained: bool,
    emitted: u64,
}

impl AviDemuxN {
    /// A demuxer with one output port per entry of `ports`, in port order.
    /// Panics if `ports` is empty (a fan-out needs a port).
    pub fn new(ports: Vec<AviStreamInfo>) -> Self {
        assert!(
            !ports.is_empty(),
            "AviDemuxN needs at least one output port"
        );
        Self {
            buffer: Vec::new(),
            port_caps: ports.iter().map(|p| p.caps.clone()).collect(),
            ports: ports.iter().map(|p| p.stream).collect(),
            bus: None,
            drained: false,
            emitted: 0,
        }
    }

    /// Attach the pipeline bus so the file's streams post as a
    /// `StreamCollection` and its `LIST INFO` metadata as tags.
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.bus = Some(bus);
        self
    }

    pub fn port_count(&self) -> usize {
        self.ports.len()
    }

    /// Count of frames forwarded across all ports.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    /// Parse the buffered file at `Eos` and route every chunk to its port, each
    /// port's concrete `CapsChanged` first.
    async fn drain(&mut self, out: &mut dyn MultiOutputSink) -> Result<(), G2gError> {
        if self.drained {
            return Ok(());
        }
        self.drained = true;
        let file = parse(&self.buffer)?;
        if let Some(bus) = &self.bus {
            if let Some(collection) = stream_collection(&file) {
                bus.try_post(BusMessage::StreamCollection(collection));
            }
            if !file.tags.is_empty() {
                bus.try_post(BusMessage::Tag {
                    tags: file.tags.clone(),
                    program: None,
                });
            }
        }
        for (port, stream) in self.ports.iter().enumerate() {
            let caps = file
                .streams
                .get(*stream)
                .and_then(|s| real_caps(&s.kind))
                .ok_or(G2gError::CapsMismatch)?;
            out.push_to(port, PipelinePacket::CapsChanged(caps)).await?;
        }
        for emitted in emittable(&self.buffer, &file, &self.ports) {
            let Some(port) = self.ports.iter().position(|s| *s == emitted.stream) else {
                continue;
            };
            let frame = emitted.into_frame(self.emitted);
            self.emitted += 1;
            out.push_to(port, PipelinePacket::DataFrame(frame)).await?;
        }
        Ok(())
    }
}

impl MultiOutputElement for AviDemuxN {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Reads host memory, so it takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&input_caps())
    }

    fn caps_constraint_as_input(&self) -> CapsConstraint<'_> {
        CapsConstraint::Produces(CapsSet::one(input_caps()))
    }

    fn port_output_caps(&self, port: usize) -> Option<Caps> {
        self.port_caps.get(port).cloned()
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !matches!(
            absolute_caps,
            Caps::ByteStream {
                encoding: ByteStreamEncoding::Avi
            }
        ) {
            return Err(G2gError::CapsMismatch);
        }
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let slice = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    self.buffer.extend_from_slice(slice);
                }
                PipelinePacket::Eos => self.drain(out).await?,
                PipelinePacket::Flush => {
                    self.buffer.clear();
                    self.drained = false;
                    for port in 0..self.ports.len() {
                        out.push_to(port, PipelinePacket::Flush).await?;
                    }
                }
                PipelinePacket::Segment(segment) => {
                    for port in 0..self.ports.len() {
                        out.push_to(port, PipelinePacket::Segment(segment)).await?;
                    }
                }
                // The input's byte-stream caps are consumed: each port declares
                // its own, announced in `drain`.
                PipelinePacket::CapsChanged(_) => {}
                _ => {}
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avi::{AviChunk, AviWriteStream, AviWriter};
    use alloc::vec;
    use core::task::{Context, Poll};
    use g2g_core::runtime::block_on;
    use g2g_core::PushOutcome;

    /// An AVI holding one MJPEG stream and one 16-bit PCM stream, four chunks
    /// each, written by the muxer's own writer.
    fn av_file() -> Vec<u8> {
        const FRAME_PERIOD_NS: u64 = 40_000_000;
        let mut writer = AviWriter::new(vec![
            AviWriteStream::Video {
                codec: VideoCodec::Mjpeg,
                width: 64,
                height: 48,
            },
            AviWriteStream::Audio {
                format: AudioFormat::PcmS16Le,
                channels: 1,
                sample_rate: 8_000,
            },
        ]);
        for i in 0..4u64 {
            writer
                .push(0, vec![i as u8; 3], i * FRAME_PERIOD_NS, true)
                .expect("video chunk");
            writer
                .push(1, vec![1u8; 640], i * FRAME_PERIOD_NS, true)
                .expect("audio chunk");
        }
        writer.finish().expect("the file serializes")
    }

    /// Records, per port, the caps and the frames it received.
    #[derive(Debug, Default)]
    struct PortCapture {
        caps: Vec<Option<Caps>>,
        payloads: Vec<Vec<Vec<u8>>>,
        timings: Vec<Vec<FrameTiming>>,
    }

    impl PortCapture {
        fn new(ports: usize) -> Self {
            Self {
                caps: alloc::vec![None; ports],
                payloads: alloc::vec![Vec::new(); ports],
                timings: alloc::vec![Vec::new(); ports],
            }
        }
    }

    impl MultiOutputSink for PortCapture {
        fn port_count(&self) -> usize {
            self.payloads.len()
        }

        fn poll_push_to(
            &mut self,
            _cx: &mut Context<'_>,
            port: usize,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> Poll<Result<PushOutcome, G2gError>> {
            match packet_slot.take().expect("poll_push without a packet") {
                PipelinePacket::CapsChanged(c) => self.caps[port] = Some(c),
                PipelinePacket::DataFrame(f) => {
                    self.timings[port].push(f.timing);
                    self.payloads[port].push(f.domain.as_system_slice().unwrap_or_default().into());
                }
                _ => {}
            }
            Poll::Ready(Ok(PushOutcome::Accepted))
        }
    }

    /// Captures a single-output demuxer's frames.
    #[derive(Debug, Default)]
    struct Capture {
        caps: Option<Caps>,
        payloads: Vec<Vec<u8>>,
    }

    impl OutputSink for Capture {
        fn poll_push(
            &mut self,
            _cx: &mut Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> Poll<Result<PushOutcome, G2gError>> {
            match packet_slot.take().expect("poll_push without a packet") {
                PipelinePacket::CapsChanged(c) => self.caps = Some(c),
                PipelinePacket::DataFrame(f) => self
                    .payloads
                    .push(f.domain.as_system_slice().unwrap_or_default().into()),
                _ => {}
            }
            Poll::Ready(Ok(PushOutcome::Accepted))
        }
    }

    #[test]
    fn fans_an_av_file_out_to_two_ports() {
        let bytes = av_file();
        let infos = forwardable_streams(&bytes);
        assert_eq!(infos.len(), 2, "one video and one audio stream");
        let mut demux = AviDemuxN::new(infos);
        let mut sink = PortCapture::new(2);
        demux
            .configure_pipeline(&input_caps())
            .expect("the byte stream is accepted");
        block_on(async {
            let frame = frame_of(bytes.clone(), FrameTiming::default(), 0);
            demux
                .process(PipelinePacket::DataFrame(frame), &mut sink)
                .await
                .expect("the bytes buffer");
            demux
                .process(PipelinePacket::Eos, &mut sink)
                .await
                .expect("the file demuxes");
        });

        assert_eq!(sink.payloads[0].len(), 4);
        for (i, payload) in sink.payloads[0].iter().enumerate() {
            assert_eq!(payload.as_slice(), &[i as u8; 3][..]);
            assert_eq!(sink.timings[0][i].pts_ns, i as u64 * 40_000_000);
            assert!(
                sink.timings[0][i].keyframe,
                "idx1 marks every MJPEG chunk key"
            );
        }
        // 640 bytes of mono 16-bit at 8 kHz is 320 sample frames, 40 ms each.
        assert_eq!(sink.payloads[1].len(), 4);
        for (i, timing) in sink.timings[1].iter().enumerate() {
            assert_eq!(timing.pts_ns, i as u64 * 40_000_000);
        }
        assert!(matches!(
            sink.caps[0],
            Some(Caps::CompressedVideo {
                codec: VideoCodec::Mjpeg,
                width: Dim::Fixed(64),
                height: Dim::Fixed(48),
                ..
            })
        ));
        assert_eq!(
            sink.caps[1],
            Some(audio_caps(AudioFormat::PcmS16Le, 1, 8_000))
        );
    }

    #[test]
    fn the_single_output_demux_picks_the_selected_stream() {
        let bytes = av_file();
        for (select, want_caps) in [
            (
                AviStreamSelect::Video,
                video_caps(VideoCodec::Mjpeg, Dim::Fixed(64), Dim::Fixed(48)),
            ),
            (
                AviStreamSelect::AudioNamed(AudioFormat::PcmS16Le),
                audio_caps(AudioFormat::PcmS16Le, 1, 8_000),
            ),
        ] {
            let mut demux = AviDemux::new().with_stream(select);
            demux
                .configure_pipeline(&input_caps())
                .expect("the byte stream is accepted");
            let mut sink = Capture::default();
            block_on(async {
                let frame = frame_of(bytes.clone(), FrameTiming::default(), 0);
                demux
                    .process(PipelinePacket::DataFrame(frame), &mut sink)
                    .await
                    .expect("the bytes buffer");
                demux
                    .process(PipelinePacket::Eos, &mut sink)
                    .await
                    .expect("the file demuxes");
            });
            assert_eq!(sink.payloads.len(), 4, "{select:?}");
            assert_eq!(sink.caps, Some(want_caps), "{select:?}");
        }
    }

    #[cfg(feature = "metadata")]
    #[test]
    fn a_palette_change_rides_the_next_frame_of_its_stream() {
        let first_frame = [1u8, 2];
        let change = [7u8, 1, 0, 0, 10, 20, 30, 0];
        let second_frame = [3u8, 4];
        let data = [first_frame.as_slice(), &change, &second_frame].concat();
        let chunk = |body: core::ops::Range<usize>, palette_change| AviChunk {
            stream: 0,
            body,
            keyframe: true,
            palette_change,
        };
        let change_at = first_frame.len();
        let second_at = change_at + change.len();
        let file = AviFile {
            streams: vec![AviStream {
                kind: AviStreamKind::Video {
                    codec: VideoCodec::Mjpeg,
                    width: 16,
                    height: 16,
                },
                scale: 1,
                rate: 25,
                start: 0,
                sample_size: 0,
                codec_config: Vec::new(),
            }],
            chunks: vec![
                chunk(0..change_at, false),
                chunk(change_at..second_at, true),
                chunk(second_at..data.len(), false),
            ],
            ..AviFile::default()
        };
        let emitted: Vec<Emitted> = emittable(&data, &file, &[0]).collect();
        let payloads: Vec<&[u8]> = emitted.iter().map(|e| e.data.as_slice()).collect();
        assert_eq!(payloads, [first_frame.as_slice(), &second_frame]);
        let changes: Vec<_> = emitted
            .iter()
            .map(|e| e.meta.get::<g2g_core::meta::PaletteChange>().cloned())
            .collect();
        let expected = g2g_core::meta::PaletteChange {
            colors: vec![(7, [10, 20, 30])],
        };
        assert_eq!(changes, [None, Some(expected)]);
        // the change chunk takes no frame slot on the stream's clock
        let second_pts = emitted[1].timing.pts_ns;
        assert_eq!(second_pts, file.streams[0].timing(1, 1).0);
    }

    #[test]
    fn converts_avcc_chunks_whose_length_prefix_looks_like_a_start_code() {
        use crate::annexb::{avcc_record, ANNEX_B_START_CODE, AVCC_LENGTH_SIZE, START_CODE_PREFIX};
        let sps = [0x67u8, 0x42, 0x00, 0x1E];
        let pps = [0x68u8, 0xCE, 0x3C, 0x80];
        let idr = [0x65u8, 0x88];
        // the shortest NAL whose 4-byte length prefix begins `00 00 01`
        let mut slice_prefix = [0u8; AVCC_LENGTH_SIZE];
        slice_prefix[..START_CODE_PREFIX.len()].copy_from_slice(&START_CODE_PREFIX);
        let mut slice = vec![0x22u8; u32::from_be_bytes(slice_prefix) as usize];
        slice[0] = 0x41;
        let mut idr_chunk = (idr.len() as u32).to_be_bytes().to_vec();
        idr_chunk.extend_from_slice(&idr);
        let mut slice_chunk = slice_prefix.to_vec();
        slice_chunk.extend_from_slice(&slice);
        let data = [idr_chunk.clone(), slice_chunk].concat();
        let file = AviFile {
            streams: vec![AviStream {
                kind: AviStreamKind::Video {
                    codec: VideoCodec::H264,
                    width: 16,
                    height: 16,
                },
                scale: 1,
                rate: 25,
                start: 0,
                sample_size: 0,
                codec_config: avcc_record(&[&sps, &pps]),
            }],
            chunks: vec![
                AviChunk {
                    stream: 0,
                    body: 0..idr_chunk.len(),
                    keyframe: true,
                    palette_change: false,
                },
                AviChunk {
                    stream: 0,
                    body: idr_chunk.len()..data.len(),
                    keyframe: false,
                    palette_change: false,
                },
            ],
            ..AviFile::default()
        };
        let payloads: Vec<Vec<u8>> = emittable(&data, &file, &[0])
            .map(|emitted| emitted.data)
            .collect();
        let annexb = |nals: &[&[u8]]| -> Vec<u8> {
            nals.iter()
                .flat_map(|nal| ANNEX_B_START_CODE.iter().chain(nal.iter()).copied())
                .collect()
        };
        assert_eq!(
            payloads,
            vec![annexb(&[&sps, &pps, &idr]), annexb(&[&slice])],
            "every chunk is converted with the avcC length width"
        );
    }

    #[test]
    fn sends_an_eld_config_ahead_of_its_raw_chunks() {
        // AAC-ELD 48 kHz stereo, an object type ADTS has no profile for
        let eld_config = vec![0xF8u8, 0xE6, 0x40, 0x00];
        let units = [vec![0x7Eu8, 0x89, 0xB4], vec![0x56, 0x78, 0x9A]];
        let data = units.concat();
        let file = AviFile {
            streams: vec![AviStream {
                kind: AviStreamKind::Audio {
                    format: AudioFormat::Aac,
                    channels: 2,
                    sample_rate: 48_000,
                },
                scale: 512,
                rate: 48_000,
                start: 0,
                sample_size: 0,
                codec_config: eld_config.clone(),
            }],
            chunks: vec![
                AviChunk {
                    stream: 0,
                    body: 0..units[0].len(),
                    keyframe: true,
                    palette_change: false,
                },
                AviChunk {
                    stream: 0,
                    body: units[0].len()..data.len(),
                    keyframe: true,
                    palette_change: false,
                },
            ],
            ..AviFile::default()
        };
        let payloads: Vec<Vec<u8>> = emittable(&data, &file, &[0])
            .map(|emitted| emitted.data)
            .collect();
        assert_eq!(
            payloads,
            vec![eld_config, units[0].clone(), units[1].clone()],
            "the config leads, then each chunk unframed"
        );
    }

    #[test]
    fn an_uncompressed_stream_leaves_as_raw_video_cut_to_its_pixels() {
        const WIDTH: u32 = 4;
        const HEIGHT: u32 = 2;
        const FRAME_PERIOD_NS: u64 = 40_000_000;
        let pixels = crate::pixel::frame_byte_size(RawVideoFormat::Uyvy, WIDTH, HEIGHT);
        let stored = [
            [vec![1u8; pixels], vec![0xEE; 2]].concat(),
            vec![2u8; pixels - 1],
            vec![3u8; pixels],
        ];
        let mut writer = AviWriter::new(vec![AviWriteStream::Video {
            codec: VideoCodec::Mjpeg,
            width: WIDTH,
            height: HEIGHT,
        }]);
        for (i, frame) in stored.iter().enumerate() {
            writer
                .push(0, frame.clone(), i as u64 * FRAME_PERIOD_NS, true)
                .expect("video chunk");
        }
        let mut bytes = writer.finish().expect("the file serializes");
        while let Some(at) = bytes.windows(4).position(|w| w == b"MJPG") {
            bytes[at..at + 4].copy_from_slice(b"UYVY");
        }

        let want_caps = raw_video_caps(
            RawVideoFormat::Uyvy,
            Dim::Fixed(WIDTH),
            Dim::Fixed(HEIGHT),
            framerate_span(),
        );
        let infos = forwardable_streams(&bytes);
        assert_eq!(infos[0].caps, want_caps);
        let mut demux = AviDemux::new();
        demux
            .set_property("stream", PropValue::Str("raw-uyvy".into()))
            .expect("raw-uyvy names a stream");
        demux
            .configure_pipeline(&input_caps())
            .expect("the byte stream is accepted");
        let mut sink = Capture::default();
        block_on(async {
            let frame = frame_of(bytes, FrameTiming::default(), 0);
            demux
                .process(PipelinePacket::DataFrame(frame), &mut sink)
                .await
                .expect("the bytes buffer");
            demux
                .process(PipelinePacket::Eos, &mut sink)
                .await
                .expect("the file demuxes");
        });
        assert_eq!(sink.caps, Some(want_caps));
        assert_eq!(
            sink.payloads,
            vec![vec![1u8; pixels], vec![3u8; pixels]],
            "the short frame is dropped and the long one cut"
        );
    }

    #[test]
    fn the_stream_property_round_trips_every_name() {
        let mut demux = AviDemux::new();
        for spec in AVIDEMUX_PROPS {
            assert_eq!(spec.name, "stream");
        }
        for name in [
            "video",
            "audio",
            "mjpeg",
            "h264",
            "mpeg4part2",
            "mp3",
            "aac",
        ] {
            demux
                .set_property("stream", PropValue::Str(name.into()))
                .unwrap_or_else(|e| panic!("{name} is a stream name: {e:?}"));
            assert_eq!(
                demux.get_property("stream"),
                Some(PropValue::Str(name.into()))
            );
        }
        assert!(demux
            .set_property("stream", PropValue::Str("vorbis".into()))
            .is_err());
    }
}
