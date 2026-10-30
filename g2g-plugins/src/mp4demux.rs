//! Progressive / whole-file MP4 demuxer (Mp4Demux): `ByteStream{Mp4}` in,
//! one track's access units out, a one-port [`Mp4DemuxN`](crate::mp4demuxn) run
//! as a single-output element. The whole-file sibling of
//! [`Fmp4Demux`](crate::fmp4demux) (which streams a live fragmented
//! `ByteStream{IsoBmff}`): a progressive `.mp4` / `.mov` keeps its `moov` sample
//! tables (`stbl`) and an `mdat`, and the `moov` may sit at the *end* of the file
//! with absolute `stco` chunk offsets, so such a file is buffered whole before the
//! `fmp4::parse_progressive_multi` pass runs at `Eos`.
//!
//! This is what a bare `filesrc location=movie.mp4 ! decodebin` auto-plugs: a file
//! source types itself `ByteStream{Mp4}`, the auto-plugger picks this
//! whole-file demuxer for it, and the fragmented `fmp4demux` still serves the
//! streaming `IsoBmff` that HLS / DASH produce. Multi-track fan-out (video, audio,
//! text) stays on [`Mp4DemuxN`](crate::mp4demuxn) via `qtdemux name=d ...`; this
//! element emits one stream, the video track by default (`stream=av1` etc. names
//! its codec for truthful startup caps), or the audio track when
//! `stream=aac`, the one stream a linear `decodebin` chain decodes.

use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::{
    AudioFormat, ByteStreamEncoding, Caps, CapsSet, Dim, PadTemplate, PadTemplates, ProResVariant,
    PropKind, PropertySpec, RawVideoFormat, VideoCodec,
};

use crate::avidemux::{is_pcm, pcm_alternatives};
use crate::codectag::{
    audio_from_stream_name, audio_stream_name, video_from_stream_name, video_stream_name,
};
use crate::fmp4::TrackKind;
use crate::mkvdemux::{prores_stream_from_str, prores_stream_str};
use crate::mp4demuxn::Mp4DemuxN;
use crate::oneport::OnePortElement;
use crate::typefind::PLACEHOLDER_FRAMERATE;
use crate::uncompressedvideo::{
    raw_video_formats, raw_video_placeholder_caps, raw_video_stream_from_str, raw_video_stream_str,
};

/// Which track the single-output demux emits: the video track (the
/// default, so `filesrc ! qtdemux ! ...` is unchanged), the video track with its
/// codec named (`stream=av1` etc., so a codec-specific decoder negotiates
/// truthful startup caps instead of the nominal H.264), or the audio track,
/// which a bare `decodebin` on an audio-only file selects via the primary-stream
/// hook (`stream=aac`, `stream=pcm-s16le`, ...), named by its format.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum StreamSelect {
    Video,
    VideoNamed(VideoCodec),
    RawVideo(RawVideoFormat),
    Audio(AudioFormat),
}

impl StreamSelect {
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        if name == "video" {
            return Some(Self::Video);
        }
        audio_from_stream_name(AUDIO_STREAM_NAMES, name)
            .map(Self::Audio)
            .or_else(|| mp4_video_stream_from_str(name).map(Self::VideoNamed))
            .or_else(|| raw_video_stream_from_str(name).map(Self::RawVideo))
    }

    pub(crate) fn name(self) -> String {
        match self {
            Self::Video => String::from("video"),
            // constructed from the same table, so the name is always there
            Self::VideoNamed(codec) => {
                mp4_video_stream_str(codec).unwrap_or_else(|| String::from("video"))
            }
            Self::RawVideo(format) => String::from(raw_video_stream_str(format).unwrap_or("video")),
            Self::Audio(format) => {
                mp4_audio_stream_str(format).unwrap_or_else(|| String::from("aac"))
            }
        }
    }

    pub(crate) fn matches(self, kind: &TrackKind) -> bool {
        match (self, kind) {
            (Self::Video, TrackKind::Video { .. }) => true,
            (Self::VideoNamed(named), TrackKind::Video { codec, .. }) => named == *codec,
            (Self::RawVideo(selected), TrackKind::RawVideo { format, .. }) => selected == *format,
            (Self::Audio(selected), TrackKind::Audio { format, .. }) => selected == *format,
            _ => false,
        }
    }

    // Video defaults to H.264, refined from the moov via CapsChanged, but a
    // downstream that names another codec (an `h265parse`) can pick it, so the
    // other codecs follow the default as alternatives. A codec-named selection
    // (M961) negotiates that codec alone; audio negotiates its selected format
    // with a wildcard layout, refined the same way.
    pub(crate) fn nego_caps(self) -> CapsSet {
        match self {
            Self::Video => CapsSet::from_alternatives(
                VIDEO_STREAM_NAMES
                    .iter()
                    .map(|(_, codec)| nego_caps(*codec))
                    .collect(),
            ),
            Self::VideoNamed(codec) => CapsSet::one(nego_caps(codec)),
            Self::RawVideo(format) => CapsSet::one(raw_video_placeholder_caps(format)),
            Self::Audio(format) => CapsSet::from_alternatives(audio_nego_caps(format)),
        }
    }
}

/// The `stream=` names for a codec-named video selection, the set
/// [`parse_all_tracks`](crate::fmp4) can read from a `moov` sample entry. Shared
/// with the bare-`decodebin` primary-stream hook, which names the file's actual
/// codec through this table. H.264 leads, as the default video selection's
/// negotiation caps.
const VIDEO_STREAM_NAMES: &[(&str, VideoCodec)] = &[
    ("h264", VideoCodec::H264),
    ("h265", VideoCodec::H265),
    ("av1", VideoCodec::Av1),
    ("mpeg4part2", VideoCodec::Mpeg4Part2),
    ("mpeg2", VideoCodec::Mpeg2),
    ("vp8", VideoCodec::Vp8),
    ("mjpeg", VideoCodec::Mjpeg),
    ("dnxhd", VideoCodec::DnxHd),
];

/// The `stream=` names for the audio formats [`parse_all_tracks`](crate::fmp4)
/// reads from a `moov` sample entry.
const AUDIO_STREAM_NAMES: &[(&str, AudioFormat)] = &[
    ("aac", AudioFormat::Aac),
    ("opus", AudioFormat::Opus),
    ("mp3", AudioFormat::Mp3),
    ("pcm-u8", AudioFormat::PcmU8),
    ("pcm-s16le", AudioFormat::PcmS16Le),
    ("pcm-s24le", AudioFormat::PcmS24Le),
    ("pcm-s32le", AudioFormat::PcmS32Le),
    ("pcm-f32le", AudioFormat::PcmF32Le),
    ("alaw", AudioFormat::Alaw),
    ("mulaw", AudioFormat::Mulaw),
    ("vorbis", AudioFormat::Vorbis),
];

pub(crate) fn stream_for_consumer(accepted: &CapsSet) -> Option<String> {
    let named = VIDEO_STREAM_NAMES
        .iter()
        .map(|(_, codec)| StreamSelect::VideoNamed(*codec))
        .chain(
            AUDIO_STREAM_NAMES
                .iter()
                .map(|(_, format)| StreamSelect::Audio(*format)),
        );
    crate::demuxstream::stream_for_consumer(
        &StreamSelect::Video.nego_caps(),
        named.flat_map(|select| {
            select
                .nego_caps()
                .alternatives()
                .iter()
                .map(|caps| (select, caps.clone()))
                .collect::<Vec<_>>()
        }),
        accepted,
    )
    .map(StreamSelect::name)
}

pub(crate) fn mp4_audio_stream_str(format: AudioFormat) -> Option<String> {
    audio_stream_name(AUDIO_STREAM_NAMES, format)
}

pub(crate) fn mp4_video_stream_str(codec: VideoCodec) -> Option<String> {
    if let VideoCodec::ProRes { variant } = codec {
        return Some(String::from(prores_stream_str(variant)));
    }
    video_stream_name(VIDEO_STREAM_NAMES, codec)
}

fn mp4_video_stream_from_str(name: &str) -> Option<VideoCodec> {
    video_from_stream_name(VIDEO_STREAM_NAMES, name)
        .or_else(|| prores_stream_from_str(name).map(|variant| VideoCodec::ProRes { variant }))
}

/// The audio negotiation placeholder: the format with a channels /
/// sample-rate wildcard (`0`), refined to the concrete `moov` layout via
/// `CapsChanged`. PCM leads with a concrete layout, as `AviDemux` does, since
/// a lone wildcard cannot fixate.
fn audio_nego_caps(format: AudioFormat) -> Vec<Caps> {
    if is_pcm(format) {
        return pcm_alternatives(format);
    }
    Vec::from([Caps::Audio {
        format,
        channels: 0,
        sample_rate: 0,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    }])
}

/// The negotiation placeholder: a fixatable `Range` (not `Dim::Any`, which
/// fails Phase-2 fixate, e.g. a downstream `h264parse`), refined to the concrete
/// moov geometry via `CapsChanged`. Mirrors `tsdemux` / `matroskademux`.
fn nego_caps(codec: VideoCodec) -> Caps {
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
        framerate: PLACEHOLDER_FRAMERATE,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        alignment: crate::annexb::sample_alignment(codec),
    }
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::mp4demux::Mp4Demux;
///
/// // gst-launch equivalent: filesrc location=clip.mp4 ! mp4demux ! decodebin
/// let demux = Mp4Demux::new();
/// ```
pub type Mp4Demux = OnePortElement<Mp4DemuxN>;

impl Default for Mp4Demux {
    fn default() -> Self {
        Self::new()
    }
}

impl Mp4Demux {
    pub fn new() -> Self {
        Self::wrap(Mp4DemuxN::whole_file(StreamSelect::Video), "Mp4Demux")
    }
}

/// `Mp4Demux`'s settable properties. `stream` picks the emitted track, the
/// single-stream analog of [`TsDemux`](crate::tsdemux::TsDemux)'s `stream`.
pub(crate) static MP4DEMUX_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "stream",
        PropKind::Str,
        "track to emit: video (the default) | h264 | h265 | av1 | mpeg4part2 | mpeg2 | vp8 | mjpeg | dnxhd | prores-proxy | prores-lt | prores-standard | prores-hq | prores-4444 | prores-4444xq (the video track by codec) | raw-i420 | raw-yuy2 | raw-uyvy | raw-v210 (the uncompressed video track by layout) | aac | opus | mp3 | pcm-u8 | pcm-s16le | pcm-s24le | pcm-s32le | pcm-f32le | alaw | mulaw | vorbis (the audio track by format) | video-mov-<fourcc> | audio-mov-<fourcc> (a codec known only by its sample entry, with ffmpeg-long-tail)",
    ),
    PropertySpec::new(
        "track-id",
        PropKind::Uint,
        "tkhd track_ID of the track to emit among those `stream` matches (0 = the first of them)",
    ),
];

impl PadTemplates for Mp4Demux {
    fn pad_templates() -> Vec<PadTemplate> {
        let mut source = Vec::from([
            nego_caps(VideoCodec::H264),
            nego_caps(VideoCodec::H265),
            nego_caps(VideoCodec::Av1),
            nego_caps(VideoCodec::Mpeg4Part2),
            nego_caps(VideoCodec::Mpeg2),
            nego_caps(VideoCodec::Vp8),
            nego_caps(VideoCodec::Mjpeg),
            nego_caps(VideoCodec::DnxHd),
        ]);
        source.extend(ProResVariant::ALL.map(|variant| nego_caps(VideoCodec::ProRes { variant })));
        source.extend(raw_video_formats().map(raw_video_placeholder_caps));
        for (_, format) in AUDIO_STREAM_NAMES {
            source.extend(audio_nego_caps(*format));
        }
        Vec::from([
            PadTemplate::sink(CapsSet::one(Caps::ByteStream {
                encoding: ByteStreamEncoding::Mp4,
            })),
            PadTemplate::source(CapsSet::from_alternatives(source)),
        ])
    }
}
