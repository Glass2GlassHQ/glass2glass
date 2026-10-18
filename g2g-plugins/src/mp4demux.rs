//! Progressive / whole-file MP4 demuxer (Mp4Demux, M479): `ByteStream{Mp4}` in,
//! the video track's Annex-B access units out. The single-output, whole-file
//! sibling of [`Fmp4Demux`](crate::fmp4demux) (which streams a live fragmented
//! `ByteStream{IsoBmff}`): a progressive `.mp4` / `.mov` keeps its `moov` sample
//! tables (`stbl`) and an `mdat`, and the `moov` may sit at the *end* of the file
//! with absolute `stco` chunk offsets, so the whole file is buffered before the
//! `fmp4::parse_progressive_multi` pass runs at `Eos`.
//!
//! This is what a bare `filesrc location=movie.mp4 ! decodebin` auto-plugs: a file
//! source types itself `ByteStream{Mp4}` (M478), the auto-plugger picks this
//! whole-file demuxer for it, and the fragmented `fmp4demux` still serves the
//! streaming `IsoBmff` that HLS / DASH produce. Multi-track fan-out (video, audio,
//! text) stays on [`Mp4DemuxN`](crate::mp4demuxn) via `qtdemux name=d ...`; this
//! element emits one stream, the video track by default (`stream=av1` etc. names
//! its codec for truthful startup caps, M961), or the audio track when
//! `stream=aac` (M748), the one stream a linear `decodebin` chain decodes.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::future::Future;
use core::pin::Pin;

use g2g_core::element::AsyncElement;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::log::short_type_name;
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AudioFormat, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim,
    ElementMetadata, G2gError, OutputSink, PadTemplate, PadTemplates, PropError, PropKind,
    PropValue, PropertySpec, Rate, Segment, VideoCodec,
};

use crate::aacparse::{is_raw_aac_config, require_adts};
use crate::avidemux::{is_pcm, pcm_alternatives};
use crate::fmp4::{
    parse_all_tracks, parse_fragments_multi, parse_progressive_multi, prepend_param_sets,
    starts_with_param_set, track_framerate_q16, Sample, TrackHeader, TrackKind,
};
use crate::mp4box::find_box;
use crate::typefind::{framerate_within_placeholder, PLACEHOLDER_FRAMERATE};

/// Which track the single-output demux emits (M748): the video track (the
/// default, so `filesrc ! qtdemux ! ...` is unchanged), the video track with its
/// codec named (M961, `stream=av1` etc., so a codec-specific decoder negotiates
/// truthful startup caps instead of the nominal H.264), or the audio track,
/// which a bare `decodebin` on an audio-only file selects via the primary-stream
/// hook (`stream=aac`, `stream=pcm-s16le`, ...), named by its format.
#[derive(Debug, Clone, Copy, PartialEq)]
enum StreamSelect {
    Video,
    VideoNamed(VideoCodec),
    Audio(AudioFormat),
}

/// The `stream=` names for a codec-named video selection (M961), the set
/// [`parse_all_tracks`](crate::fmp4) can read from a `moov` sample entry. Shared
/// with the bare-`decodebin` primary-stream hook, which names the file's actual
/// codec through this table.
const VIDEO_STREAM_NAMES: &[(&str, VideoCodec)] = &[
    ("h264", VideoCodec::H264),
    ("h265", VideoCodec::H265),
    ("av1", VideoCodec::Av1),
    ("mpeg4part2", VideoCodec::Mpeg4Part2),
    ("vp8", VideoCodec::Vp8),
    ("mjpeg", VideoCodec::Mjpeg),
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
    ("alaw", AudioFormat::Alaw),
    ("mulaw", AudioFormat::Mulaw),
];

pub(crate) fn mp4_audio_stream_str(format: AudioFormat) -> Option<&'static str> {
    AUDIO_STREAM_NAMES
        .iter()
        .find(|(_, f)| *f == format)
        .map(|(name, _)| *name)
}

pub(crate) fn mp4_video_stream_str(codec: VideoCodec) -> Option<&'static str> {
    VIDEO_STREAM_NAMES
        .iter()
        .find(|(_, c)| *c == codec)
        .map(|(name, _)| *name)
}

fn mp4_video_stream_from_str(name: &str) -> Option<VideoCodec> {
    VIDEO_STREAM_NAMES
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, c)| *c)
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::mp4demux::Mp4Demux;
///
/// // gst-launch equivalent: filesrc location=clip.mp4 ! mp4demux ! decodebin
/// let demux = Mp4Demux::new();
/// ```
#[derive(Debug)]
pub struct Mp4Demux {
    /// The whole file, accumulated across chunks; parsed once at `Eos` (the `moov`
    /// may be at the end, so no sample can be emitted before the file is complete).
    buffer: Vec<u8>,
    /// Negotiation-time output codec, refined from the `moov` via `CapsChanged`.
    out_codec: VideoCodec,
    /// Which track to emit; set by the `stream` property before negotiation.
    select: StreamSelect,
    caps_sent: bool,
    sequence: u64,
    configured: bool,
    drained: bool,
}

impl Default for Mp4Demux {
    fn default() -> Self {
        Self::new()
    }
}

impl Mp4Demux {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            out_codec: VideoCodec::H264,
            select: StreamSelect::Video,
            caps_sent: false,
            sequence: 0,
            configured: false,
            drained: false,
        }
    }

    fn input_caps() -> Caps {
        Caps::ByteStream {
            encoding: ByteStreamEncoding::Mp4,
        }
    }

    fn output_caps(codec: VideoCodec, width: Dim, height: Dim, framerate: Rate) -> Caps {
        Caps::CompressedVideo {
            codec,
            width,
            height,
            framerate,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        }
    }

    /// The audio negotiation placeholder: the format with a channels /
    /// sample-rate wildcard (`0`), refined to the concrete `moov` layout via
    /// `CapsChanged` at `Eos`. Mirrors [`Mp4DemuxN`](crate::mp4demuxn)'s audio
    /// port so the same `aacparse -> decoder` chain plugs. PCM leads with a
    /// concrete layout, as `AviDemux` does, since a lone wildcard cannot fixate.
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
    /// moov geometry via `CapsChanged` at `Eos`. Mirrors `tsdemux` / `matroskademux`.
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
        }
    }

    /// Emit the selected track once the whole file is in hand. Idempotent; runs
    /// once, at `Eos`, dispatching to the video or audio drain.
    async fn drain(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        if self.drained {
            return Ok(());
        }
        self.drained = true;
        match self.select {
            StreamSelect::Video | StreamSelect::VideoNamed(_) => self.drain_video(out).await,
            StreamSelect::Audio(format) => self.drain_audio(format, out).await,
        }
    }

    /// Parse the buffered file's first audio track of the `selected` format and
    /// emit it (M748): the concrete caps (channels / rate from the `moov`) via
    /// `CapsChanged`, then every access unit, AAC ADTS-framed from the track's
    /// AudioSpecificConfig so the elementary stream is self-describing (the shape
    /// [`Mp4DemuxN`](crate::mp4demuxn)'s audio port emits). Runs once, at `Eos`.
    /// Fails loud if there is no such track.
    async fn drain_audio(
        &mut self,
        selected: AudioFormat,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let tracks = parse_all_tracks(&self.buffer)?;
        let audio = tracks
            .iter()
            .find(|t| matches!(t.kind, TrackKind::Audio { format, .. } if format == selected))
            .ok_or(G2gError::CapsMismatch)?;
        let TrackKind::Audio {
            format,
            channels,
            sample_rate,
            config,
            ..
        } = &audio.kind
        else {
            return Err(G2gError::CapsMismatch);
        };
        if !self.caps_sent {
            out.push(PipelinePacket::CapsChanged(Caps::Audio {
                format: *format,
                channels: *channels,
                sample_rate: *sample_rate,
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
            }))
            .await?;
            self.caps_sent = true;
        }
        let segment = audio.presentation_segment();
        if segment != Segment::new() {
            out.push(PipelinePacket::Segment(segment)).await?;
        }
        // Opus config leads the audio in-band (M791), the `OggDemux` convention,
        // so the decoder trims the pre-skip; AAC is self-describing once ADTS-framed,
        // except the object types ADTS cannot carry, which go raw behind their config.
        let raw_aac = *format == AudioFormat::Aac && is_raw_aac_config(config);
        if (*format == AudioFormat::Opus && !config.is_empty()) || raw_aac {
            out.push(PipelinePacket::DataFrame(Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(config.clone().into_boxed_slice())),
                FrameTiming::default(),
                self.sequence,
            )))
            .await?;
            self.sequence += 1;
        }
        for sample in track_samples(&self.buffer, audio)? {
            // ADTS-frame the raw AAC access unit from the track's ASC, so a decoder
            // starts without the out-of-band config (like the in-band video param
            // sets); raw AAC and every other format go as they are.
            let data = match format {
                AudioFormat::Aac if !raw_aac => {
                    require_adts(config, &sample.annexb, short_type_name::<Self>())?
                }
                _ => sample.annexb,
            };
            let frame = Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
                timing: FrameTiming {
                    pts_ns: sample.pts_ns,
                    dts_ns: sample.pts_ns,
                    duration_ns: sample.duration_ns,
                    capture_ns: sample.pts_ns,
                    arrival_ns: g2g_core::metrics::monotonic_ns(),
                    keyframe: sample.keyframe,
                },
                sequence: self.sequence,
                meta: Default::default(),
            };
            self.sequence += 1;
            out.push(PipelinePacket::DataFrame(frame)).await?;
        }
        Ok(())
    }

    /// Parse the buffered file and emit the video track: announce the concrete
    /// caps from the `moov`, then every access unit (parameter sets prepended to
    /// the first, matching `fmp4demux`). The track is the first video track, or
    /// the first of the named codec, read from the sample tables or, in a
    /// fragmented file, from the `moof`s. Idempotent; runs once, at `Eos`.
    async fn drain_video(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let named_codec = match self.select {
            StreamSelect::VideoNamed(codec) => Some(codec),
            StreamSelect::Video | StreamSelect::Audio(_) => None,
        };
        let tracks = parse_all_tracks(&self.buffer)?;
        let video = tracks
            .iter()
            .find(|t| match t.kind {
                TrackKind::Video { codec, .. } => named_codec.is_none_or(|named| named == codec),
                _ => false,
            })
            .ok_or(G2gError::CapsMismatch)?;
        let TrackKind::Video {
            codec,
            width,
            height,
            param_sets,
        } = &video.kind
        else {
            return Err(G2gError::CapsMismatch);
        };
        let codec = *codec;
        if !self.caps_sent {
            let framerate = framerate_within_placeholder(track_framerate_q16(&self.buffer, video));
            let caps = Self::output_caps(codec, Dim::Fixed(*width), Dim::Fixed(*height), framerate);
            out.push(PipelinePacket::CapsChanged(caps)).await?;
            self.out_codec = codec;
            self.caps_sent = true;
        }
        // the decoder drops the pictures the edit list cuts
        let edit = video.presentation_segment();
        if edit != Segment::new() {
            out.push(PipelinePacket::Segment(edit)).await?;
        }
        let mut need_param_sets = true;
        for s in track_samples(&self.buffer, video)? {
            let mut annexb = s.annexb;
            // The moov's config-record parameter sets are out-of-band, so prepend
            // them to the first access unit if it does not already carry them.
            if need_param_sets && !starts_with_param_set(&annexb, codec) {
                annexb = prepend_param_sets(&annexb, param_sets, codec);
            }
            need_param_sets = false;
            let frame = Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(annexb.into_boxed_slice())),
                timing: FrameTiming {
                    pts_ns: s.pts_ns,
                    dts_ns: s.pts_ns,
                    duration_ns: s.duration_ns,
                    capture_ns: s.pts_ns,
                    arrival_ns: g2g_core::metrics::monotonic_ns(),
                    keyframe: s.keyframe,
                },
                sequence: self.sequence,
                meta: Default::default(),
            };
            self.sequence += 1;
            out.push(PipelinePacket::DataFrame(frame)).await?;
        }
        Ok(())
    }
}

// a fragmented file's samples are in its moofs, a progressive file's in the sample tables
fn track_samples(data: &[u8], track: &TrackHeader) -> Result<Vec<Sample>, G2gError> {
    let selected = core::slice::from_ref(track);
    let samples = if find_box(data, b"moof").is_some() {
        parse_fragments_multi(data, selected, 0, None)?
    } else {
        parse_progressive_multi(data, selected)?
    };
    Ok(samples
        .into_iter()
        .map(|(_, sample)| track.edited_sample(sample))
        .collect())
}

impl AsyncElement for Mp4Demux {
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
        upstream_caps.intersect(&Self::input_caps())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        // ByteStream{Mp4} in -> the selected track out. Video defaults to H.264,
        // refined from the moov via CapsChanged at Eos (like fmp4demux), but a
        // downstream that names another codec (an `h265parse`) can pick it, so
        // the other codecs follow the default as alternatives. A codec-named
        // selection (M961) negotiates that codec alone; audio negotiates its
        // selected format with a wildcard layout, refined the same way.
        let nego = match self.select {
            StreamSelect::Video => {
                let mut codecs = Vec::from([self.out_codec]);
                codecs.extend(
                    VIDEO_STREAM_NAMES
                        .iter()
                        .map(|(_, codec)| *codec)
                        .filter(|codec| *codec != self.out_codec),
                );
                CapsSet::from_alternatives(codecs.into_iter().map(Self::nego_caps).collect())
            }
            StreamSelect::VideoNamed(codec) => CapsSet::one(Self::nego_caps(codec)),
            StreamSelect::Audio(format) => {
                CapsSet::from_alternatives(Self::audio_nego_caps(format))
            }
        };
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match input {
            Caps::ByteStream {
                encoding: ByteStreamEncoding::Mp4,
            } => nego.clone(),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !matches!(
            absolute_caps,
            Caps::ByteStream {
                encoding: ByteStreamEncoding::Mp4
            }
        ) {
            return Err(G2gError::CapsMismatch);
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Progressive MP4 demuxer",
            "Codec/Demuxer",
            "Demuxes a whole-file (progressive) MP4 / QuickTime byte stream to its video (or audio) track",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        MP4DEMUX_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "stream" => {
                let name = value.as_str().ok_or(PropError::Type)?;
                let audio = AUDIO_STREAM_NAMES
                    .iter()
                    .find(|(n, _)| *n == name)
                    .map(|(_, format)| StreamSelect::Audio(*format));
                self.select = match name {
                    "video" => StreamSelect::Video,
                    _ => audio
                        .or_else(|| mp4_video_stream_from_str(name).map(StreamSelect::VideoNamed))
                        .ok_or(PropError::Value)?,
                };
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "stream" => Some(PropValue::Str(
                match self.select {
                    StreamSelect::Video => "video",
                    // constructed from the same table, so the name is always there
                    StreamSelect::VideoNamed(codec) => {
                        mp4_video_stream_str(codec).unwrap_or("video")
                    }
                    StreamSelect::Audio(format) => mp4_audio_stream_str(format).unwrap_or("aac"),
                }
                .into(),
            )),
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
                    // Accumulate; the moov may be at the end, so parse only at Eos.
                    self.buffer.extend_from_slice(slice);
                }
                // The whole file is in hand: parse and emit, then the runner's
                // transform arm forwards the EOS.
                PipelinePacket::Eos => self.drain(out).await?,
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

/// `Mp4Demux`'s settable properties (M748). `stream` picks the emitted track, the
/// single-stream analog of [`TsDemux`](crate::tsdemux::TsDemux)'s `stream`.
static MP4DEMUX_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "stream",
    PropKind::Str,
    "track to emit: video (the default) | h264 | h265 | av1 | mpeg4part2 | vp8 | mjpeg (the video track by codec) | aac | opus | mp3 | pcm-u8 | pcm-s16le | pcm-s24le | pcm-s32le | alaw | mulaw (the audio track by format)",
)];

impl PadTemplates for Mp4Demux {
    fn pad_templates() -> Vec<PadTemplate> {
        let video = Self::nego_caps;
        let mut source = Vec::from([
            video(VideoCodec::H264),
            video(VideoCodec::H265),
            video(VideoCodec::Av1),
            video(VideoCodec::Mpeg4Part2),
            video(VideoCodec::Vp8),
            video(VideoCodec::Mjpeg),
        ]);
        for (_, format) in AUDIO_STREAM_NAMES {
            source.extend(Self::audio_nego_caps(*format));
        }
        Vec::from([
            PadTemplate::sink(CapsSet::one(Self::input_caps())),
            PadTemplate::source(CapsSet::from_alternatives(source)),
        ])
    }
}
