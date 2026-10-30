//! MPEG-TS demuxers: `Caps::ByteStream{MpegTs}` in, selected
//! elementary streams out. H.264 / H.265 video leave as `Caps::CompressedVideo`
//! (Annex-B); AAC audio leaves as `Caps::Audio` (ADTS).
//!
//! Wraps the pure [`crate::mpegts::TsDemuxer`] parser. Incoming byte frames are
//! resynchronized to 188-byte TS packets and fed to the demuxer; the reassembled
//! PES access units of the selected streams ([`TsStream`]) are forwarded with
//! their PTS, ready for the matching parser. CPU, `no_std` baseline.
//!
//! ```text
//! filesrc(location=x.ts, caps=ByteStream{MpegTs}) ! tsdemux ! h264parse ! <decoder> ! <sink>
//! tsdemux stream=aac ! aacparse ! <audio>
//! ```
//!
//! [`TsDemuxN`] routes one selected stream to each of its output ports, and
//! [`TsDemux`] is a one-port `TsDemuxN` run as a single-output element. The
//! parser reassembles every stream the PMT names, and the [`TsStream`] selection
//! picks which to emit, so a second `tsdemux` selecting another stream demuxes
//! the rest of the multiplex. The choice is by codec, not a runtime-discovered
//! "first video", because the output caps are fixed at negotiation before any
//! packet is parsed. Scope: the first stream of the selected codec within
//! the active PAT program (`program-number`, default the first program).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::log::Target;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{ElementSeekRoute, SeekController, StreamSelectController};
use g2g_core::{
    g2g_error, AsyncElement, AudioFormat, BusHandle, BusMessage, ByteStreamEncoding, Caps,
    CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, FrameTiming, G2gError,
    LatencyReport, MemoryDomain, MultiOutputElement, MultiOutputSink, PadTemplate, PadTemplates,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate, Segment, Stream,
    StreamCollection, StreamType, SubPictureFormat, Tag, TagList, VideoCodec,
};

use crate::demuxseek::DemuxSeek;
use crate::mpegts::{
    unwrap_metadata_au_cells, EitSlot, EsUnit, TsDemuxer, TsPacketSplitter, STREAM_TYPE_AAC,
    STREAM_TYPE_AAC_LATM, STREAM_TYPE_AC3, STREAM_TYPE_H264, STREAM_TYPE_H265,
    STREAM_TYPE_METADATA_PES, STREAM_TYPE_MPEG1_AUDIO, STREAM_TYPE_MPEG1_VIDEO,
    STREAM_TYPE_MPEG2_AUDIO, STREAM_TYPE_MPEG2_VIDEO, STREAM_TYPE_MPEG4P2, STREAM_TYPE_PRIVATE_PES,
    TAG_KEY_EVENT_DURATION, TAG_KEY_EVENT_NAME, TAG_KEY_EVENT_START, TAG_KEY_EVENT_TEXT,
    TAG_KEY_NEXT_EVENT_DURATION, TAG_KEY_NEXT_EVENT_NAME, TAG_KEY_NEXT_EVENT_START,
    TAG_KEY_NEXT_EVENT_TEXT, TAG_KEY_SCHEDULE_EVENT_DURATION, TAG_KEY_SCHEDULE_EVENT_ID,
    TAG_KEY_SCHEDULE_EVENT_NAME, TAG_KEY_SCHEDULE_EVENT_START, TAG_KEY_SCHEDULE_EVENT_TEXT,
    TAG_KEY_SERVICE_PROVIDER,
};
use crate::oneport::OnePortElement;
use crate::typefind::{elementary_video_caps, sized_by_sps};

const LOG_CATEGORY: &str = "tsdemux";

/// The published stream id of an elementary stream: the id it takes in the
/// `StreamCollection`, and the id its per-stream tags post on.
fn stream_id(pid: u16) -> alloc::string::String {
    alloc::format!("mpegts-pid-{pid}")
}

/// Posts a transport stream's tags on the bus once each: each SDT service's
/// text as a [`BusMessage::Tag`] scoped to its program (the service name as
/// [`Tag::Title`], the provider as a [`Tag::Other`] under ffprobe's
/// `service_provider` key), and each PMT stream's ISO 639 language as a
/// [`BusMessage::StreamTag`] on that stream's collection id. The MPEG-TS sibling of
/// the Matroska `MetadataPoster`.
///
/// The SDT describes the whole multiplex, so every service it names posts under its
/// own `program_number`, not only the one this element routes: an
/// application learns what the transport stream carries from one demuxer. TS
/// carries nothing else, the SDT and the language descriptor being its only
/// standard tag slots. State survives a seek's parser reset, so re-reading the
/// tables of the same stream does not re-post.
#[derive(Debug, Default)]
struct TagPoster {
    service_posted: bool,
    languages_posted: bool,
    /// The parser's EIT generation this poster last posted at. Unlike the
    /// SDT and the language descriptors, present/following events change during
    /// the stream, so they re-post whenever a new table version arrives and never
    /// on a table that only repeats itself.
    eit_generation_posted: u64,
}

impl TagPoster {
    /// Post whatever the demuxer has parsed since the last call; a no-op until the
    /// SDT / PMT parse, and once each thereafter. Takes the demuxer by `&mut`
    /// because the EIT schedule events are drained rather than kept.
    fn post(&mut self, demux: &mut TsDemuxer, bus: Option<&BusHandle>) {
        if !self.service_posted {
            for (program, service) in demux.services() {
                let mut list = TagList::new();
                if !service.name.is_empty() {
                    list.push(Tag::Title(service.name.clone()));
                }
                if !service.provider.is_empty() {
                    list.push(Tag::Other {
                        key: TAG_KEY_SERVICE_PROVIDER.into(),
                        value: service.provider.clone(),
                    });
                }
                if !list.is_empty() {
                    self.service_posted = true;
                    if let Some(bus) = bus {
                        bus.try_post(BusMessage::Tag {
                            tags: list,
                            program: Some(program),
                        });
                    }
                }
            }
        }
        self.post_events(demux, bus);
        if self.languages_posted || demux.streams().is_empty() {
            return;
        }
        self.languages_posted = true;
        for es in demux.streams() {
            let Some(code) = es.language_code() else {
                continue;
            };
            let tags: TagList = [Tag::Language(code.into())].into_iter().collect();
            if let Some(bus) = bus {
                bus.try_post(BusMessage::StreamTag {
                    stream_id: stream_id(es.pid),
                    tags,
                });
            }
        }
    }

    /// Post each service's EIT present/following text, start time and duration as
    /// a [`BusMessage::Tag`] scoped to its program, the way the SDT
    /// service name posts: a `service_id` is a `program_number`, so the two
    /// describe the same program. The event on air and the one after it share one
    /// tag list per service, under their own keys, since [`Tag::Title`] there is
    /// already the service name.
    ///
    /// A schedule table names days of events rather than those two, so each of its
    /// events posts as a message of its own: one tag list holds one event's
    /// fields, which is what lets a consumer tell whose start time is whose.
    fn post_events(&mut self, demux: &mut TsDemuxer, bus: Option<&BusHandle>) {
        for event in demux.take_eit_schedule() {
            let mut list = TagList::new();
            list.push(Tag::Number {
                key: TAG_KEY_SCHEDULE_EVENT_ID.into(),
                value: event.event_id.into(),
            });
            if !event.name.is_empty() {
                list.push(Tag::Other {
                    key: TAG_KEY_SCHEDULE_EVENT_NAME.into(),
                    value: event.name,
                });
            }
            if !event.text.is_empty() {
                list.push(Tag::Other {
                    key: TAG_KEY_SCHEDULE_EVENT_TEXT.into(),
                    value: event.text,
                });
            }
            push_event_timing(
                &mut list,
                TAG_KEY_SCHEDULE_EVENT_START,
                TAG_KEY_SCHEDULE_EVENT_DURATION,
                event.start_unix_secs,
                event.duration_secs,
            );
            if let Some(bus) = bus {
                bus.try_post(BusMessage::Tag {
                    tags: list,
                    program: Some(event.service_id),
                });
            }
        }

        let generation = demux.eit_generation();
        if generation == self.eit_generation_posted {
            return;
        }
        self.eit_generation_posted = generation;
        let mut posted: Vec<u16> = Vec::new();
        for service in demux.eit_events().iter().map(|e| e.service_id) {
            if posted.contains(&service) {
                continue;
            }
            posted.push(service);
            let mut list = TagList::new();
            for event in demux
                .eit_events()
                .iter()
                .filter(|e| e.service_id == service)
            {
                let keys = if event.slot == EitSlot::Following {
                    [
                        TAG_KEY_NEXT_EVENT_NAME,
                        TAG_KEY_NEXT_EVENT_TEXT,
                        TAG_KEY_NEXT_EVENT_START,
                        TAG_KEY_NEXT_EVENT_DURATION,
                    ]
                } else {
                    [
                        TAG_KEY_EVENT_NAME,
                        TAG_KEY_EVENT_TEXT,
                        TAG_KEY_EVENT_START,
                        TAG_KEY_EVENT_DURATION,
                    ]
                };
                if !event.name.is_empty() {
                    list.push(Tag::Other {
                        key: keys[0].into(),
                        value: event.name.clone(),
                    });
                }
                if !event.text.is_empty() {
                    list.push(Tag::Other {
                        key: keys[1].into(),
                        value: event.text.clone(),
                    });
                }
                push_event_timing(
                    &mut list,
                    keys[2],
                    keys[3],
                    event.start_unix_secs,
                    event.duration_secs,
                );
            }
            if list.is_empty() {
                continue;
            }
            if let Some(bus) = bus {
                bus.try_post(BusMessage::Tag {
                    tags: list,
                    program: Some(service),
                });
            }
        }
    }
}

/// Append an EIT event's start time (Unix seconds, UTC) and duration (seconds) to
/// a tag list. A stream that declared either field undefined or encoded it
/// invalidly posts no tag for it, rather than a zero a consumer would read as a
/// real time.
fn push_event_timing(
    list: &mut TagList,
    start_key: &str,
    duration_key: &str,
    start_unix_secs: Option<u64>,
    duration_secs: u32,
) {
    if let Some(start) = start_unix_secs {
        list.push(Tag::Number {
            key: start_key.into(),
            value: start,
        });
    }
    if duration_secs != 0 {
        list.push(Tag::Number {
            key: duration_key.into(),
            value: duration_secs.into(),
        });
    }
}

fn ticks_90khz_to_ns(ticks: u64) -> u64 {
    (ticks as u128 * 1_000_000_000 / 90_000) as u64
}

/// The bytes to forward for one access unit: a metadata-in-PES (0x15) payload
/// has its ISO 13818-1 AU cells unwrapped when it carries them (a spec-strict
/// synchronous KLV writer), and is forwarded unchanged otherwise. Every other
/// stream type passes through.
fn unwrap_sync_klv(stream_type: u8, data: Vec<u8>) -> Vec<u8> {
    if stream_type != STREAM_TYPE_METADATA_PES {
        return data;
    }
    unwrap_metadata_au_cells(&data).unwrap_or(data)
}

/// Which elementary stream a [`TsDemuxN`] port forwards. A TS multiplex
/// carries several (video + audio); a port emits exactly one, chosen here. The
/// choice is by codec because the output caps are fixed at negotiation, before
/// any packet is parsed: H.264 and H.265 are distinct downstream decoders, not a
/// geometry refinement of one another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TsStream {
    /// The first H.264 (AVC) video elementary stream. The default.
    #[default]
    H264,
    /// The first H.265 (HEVC) video elementary stream.
    H265,
    /// The first MPEG-4 Part 2 (Visual) video elementary stream.
    Mpeg4Part2,
    /// The first AV1 video elementary stream: a private PES (0x06)
    /// carrying an AV1 registration descriptor. Each PES payload is one temporal
    /// unit in the low-overhead OBU format, which `av1parse` and the AV1 decoders
    /// read as-is.
    Av1,
    /// The first MPEG-1 / MPEG-2 video elementary stream (stream_type 0x01 /
    /// 0x02). The broadcast and DVD video codec; access units are forwarded as
    /// the PES payload carries them (start-code framed, no parse element).
    Mpeg2,
    /// The first AAC audio elementary stream: ADTS (stream_type 0x0F), or
    /// LOAS/LATM (0x11, or 0x0F where a broadcaster mislabels it), whose frames
    /// go out as the PES carries them for `aacparse` to read.
    Aac,
    /// The first MPEG-1/2 Audio (Layer II, `mp2`) elementary stream (stream_type
    /// 0x03 / 0x04).
    Mp2,
    /// The first Opus audio elementary stream (private PES 0x06 with an 'Opus'
    /// registration descriptor); the control-header framing is unwrapped to raw
    /// Opus packets before forwarding.
    Opus,
    /// The first AC-3 audio elementary stream: ATSC `stream_type` 0x81, or a DVB
    /// private PES (0x06) carrying an AC-3 descriptor. Self-syncing frames are
    /// forwarded raw.
    Ac3,
    /// The first KLV metadata elementary stream (STANAG 4609 / MISB ST 1402):
    /// a private PES (0x06) carrying a 'KLVA' registration descriptor
    /// (asynchronous KLV), or a metadata-in-PES (0x15) stream (synchronous KLV).
    /// Each PES payload (one or more SMPTE ST 336 KLV packets) is forwarded raw
    /// as one `Caps::Klv` frame.
    Klv,
    /// The first DVB subtitle elementary stream (ETSI EN 300 743): a private PES
    /// (0x06) carrying a `subtitling_descriptor` (tag 0x59). Each PES payload is
    /// one display set, forwarded with its PES data-field header intact, and the
    /// descriptor's composition / ancillary page ids go out in band ahead of the
    /// first one so the decoder knows which page to compose.
    DvbSub,
    /// The first EBU teletext elementary stream (ETSI EN 300 472): a private PES
    /// (0x06) carrying a `teletext_descriptor` (tag 0x56 / 0x46). Each PES
    /// payload is a run of teletext lines, forwarded whole, and the descriptor's
    /// page selection goes out in band ahead of the first one so the decoder
    /// knows which page to assemble.
    Teletext,
}

/// Demuxes an MPEG-TS byte stream into one selected elementary stream.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::tsdemux::{TsDemux, TsStream};
///
/// let element = TsDemux::new()
///     .with_stream(TsStream::Aac)
///     .with_program_number(Some(1));
/// ```
pub type TsDemux = OnePortElement<TsDemuxN>;

impl Default for TsDemux {
    fn default() -> Self {
        Self::new()
    }
}

const DEFAULT_STREAM: TsStream = TsStream::H264;

impl TsDemux {
    pub fn new() -> Self {
        Self::wrap(TsDemuxN::new(Vec::from([DEFAULT_STREAM])), "TsDemux")
    }

    /// Select which elementary stream to forward (default [`TsStream::H264`]).
    pub fn with_stream(self, stream: TsStream) -> Self {
        self.map(|mut demux| {
            demux.ports[0] = stream;
            demux
        })
    }

    /// Select which PAT program to demux by `program_number` (`None` = the first
    /// program in PAT order, the default). Matches GStreamer `tsdemux`'s
    /// `program-number` (its `-1` is this `None`).
    pub fn with_program_number(self, n: Option<u16>) -> Self {
        self.map(|demux| demux.with_program_number(n))
    }

    /// Attach the pipeline bus so the program's `StreamCollection` is
    /// announced once the PMT is parsed, and its SDT service text / per-stream
    /// languages post as tags, the MPEG-TS sibling of
    /// [`MkvDemux::with_bus`](crate::mkvdemux::MkvDemux::with_bus).
    pub fn with_bus(self, bus: BusHandle) -> Self {
        self.map(|demux| demux.with_bus(bus))
    }

    /// Make the demuxer seekable: `app` carries app time seeks; `upstream`
    /// is the byte source's ([`FileSrc`](crate::filesrc)) byte-seek controller.
    /// On a time seek the demuxer rewinds the source and re-syncs from the
    /// keyframe at/before the target.
    pub fn with_seek(mut self, app: SeekController, upstream: SeekController) -> Self {
        AsyncElement::set_seek(&mut self, app, upstream);
        self
    }
}

/// The input a TS demuxer accepts: an MPEG-TS byte stream.
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
        TsStream::H264 => elementary_video_caps(VideoCodec::H264),
        TsStream::H265 => elementary_video_caps(VideoCodec::H265),
        TsStream::Mpeg4Part2 => elementary_video_caps(VideoCodec::Mpeg4Part2),
        TsStream::Av1 => elementary_video_caps(VideoCodec::Av1),
        TsStream::Mpeg2 => elementary_video_caps(VideoCodec::Mpeg2),
        TsStream::Aac => compressed_audio(AudioFormat::Aac),
        TsStream::Mp2 => compressed_audio(AudioFormat::Mp2),
        TsStream::Opus => compressed_audio(AudioFormat::Opus),
        TsStream::Ac3 => compressed_audio(AudioFormat::Ac3),
        TsStream::Klv => Caps::Klv,
        TsStream::Teletext => Caps::Text {
            format: g2g_core::TextFormat::Teletext,
        },
        TsStream::DvbSub => Caps::SubPicture {
            format: SubPictureFormat::DvbSub,
        },
    }
}

fn compressed_audio(format: AudioFormat) -> Caps {
    Caps::Audio {
        format,
        channels: 0,
        sample_rate: 0,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    }
}

/// The canonical PMT `stream_type` for a selection (the first, for Mp2 which
/// spans 0x03 / 0x04). Prefer [`accepts_stream_type`]
/// for routing; this is the representative value the PMT round-trip test uses.
fn selected_stream_type(stream: TsStream) -> u8 {
    match stream {
        TsStream::H264 => STREAM_TYPE_H264,
        TsStream::H265 => STREAM_TYPE_H265,
        TsStream::Mpeg4Part2 => STREAM_TYPE_MPEG4P2,
        TsStream::Av1 => STREAM_TYPE_PRIVATE_PES,
        TsStream::Mpeg2 => STREAM_TYPE_MPEG2_VIDEO,
        TsStream::Aac => STREAM_TYPE_AAC,
        TsStream::Mp2 => STREAM_TYPE_MPEG1_AUDIO,
        TsStream::Opus => STREAM_TYPE_PRIVATE_PES,
        TsStream::Ac3 => STREAM_TYPE_AC3,
        TsStream::Klv => STREAM_TYPE_PRIVATE_PES,
        TsStream::DvbSub => STREAM_TYPE_PRIVATE_PES,
        TsStream::Teletext => STREAM_TYPE_PRIVATE_PES,
    }
}

/// Whether a selection carries a PMT `stream_type`. Mp2 accepts both MPEG-1
/// (0x03) and MPEG-2 (0x04) audio; Opus accepts the private PES 0x06 (the
/// 'Opus' registration is checked separately, at PMT parse); Klv accepts the
/// private PES 0x06 (the 'KLVA' registration likewise) and metadata-in-PES
/// 0x15; Av1 accepts the private PES 0x06 (its registration likewise); Ac3
/// accepts ATSC 0x81 and the DVB private PES 0x06 (its AC-3 descriptor likewise);
/// Aac accepts ADTS 0x0F and LOAS/LATM 0x11.
fn accepts_stream_type(stream: TsStream, stream_type: u8) -> bool {
    match stream {
        TsStream::Aac => stream_type == STREAM_TYPE_AAC || stream_type == STREAM_TYPE_AAC_LATM,
        TsStream::Mp2 => {
            stream_type == STREAM_TYPE_MPEG1_AUDIO || stream_type == STREAM_TYPE_MPEG2_AUDIO
        }
        TsStream::Ac3 => stream_type == STREAM_TYPE_AC3 || stream_type == STREAM_TYPE_PRIVATE_PES,
        TsStream::Klv => {
            stream_type == STREAM_TYPE_PRIVATE_PES || stream_type == STREAM_TYPE_METADATA_PES
        }
        TsStream::Mpeg2 => {
            stream_type == STREAM_TYPE_MPEG1_VIDEO || stream_type == STREAM_TYPE_MPEG2_VIDEO
        }
        other => selected_stream_type(other) == stream_type,
    }
}

/// Map one PMT elementary stream to a [`Stream`] for the collection: its kind
/// (video / audio) and the media-type [`Caps`] it carries (geometry is
/// unknown from the PMT, so `Any`, refined later by `CapsChanged`). `None` for
/// a `stream_type` g2g does not forward.
fn es_to_stream(es: &crate::mpegts::ElementaryStream) -> Option<Stream> {
    let id = stream_id(es.pid);
    let video = |codec| Caps::CompressedVideo {
        codec,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        alignment: g2g_core::BitstreamAlignment::Unspecified,
    };
    let audio = |format| {
        (
            StreamType::Audio,
            Caps::Audio {
                format,
                channels: 0,
                sample_rate: 0,
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
            },
        )
    };
    let (stream_type, caps) = match es.stream_type {
        STREAM_TYPE_H264 => (StreamType::Video, video(VideoCodec::H264)),
        STREAM_TYPE_H265 => (StreamType::Video, video(VideoCodec::H265)),
        STREAM_TYPE_MPEG4P2 => (StreamType::Video, video(VideoCodec::Mpeg4Part2)),
        STREAM_TYPE_MPEG1_VIDEO | STREAM_TYPE_MPEG2_VIDEO => {
            (StreamType::Video, video(VideoCodec::Mpeg2))
        }
        STREAM_TYPE_AAC | STREAM_TYPE_AAC_LATM => audio(AudioFormat::Aac),
        STREAM_TYPE_MPEG1_AUDIO | STREAM_TYPE_MPEG2_AUDIO => audio(AudioFormat::Mp2),
        // 0x06 is a generic private PES; only an 'Opus' registration marks it
        // as Opus (else g2g does not forward it).
        STREAM_TYPE_PRIVATE_PES if es.opus_channels.is_some() => audio(AudioFormat::Opus),
        STREAM_TYPE_PRIVATE_PES if es.av1 => (StreamType::Video, video(VideoCodec::Av1)),
        STREAM_TYPE_AC3 => audio(AudioFormat::Ac3),
        STREAM_TYPE_PRIVATE_PES if es.ac3 => audio(AudioFormat::Ac3),
        STREAM_TYPE_PRIVATE_PES if es.subtitling.is_some() => (
            StreamType::Text,
            Caps::SubPicture {
                format: SubPictureFormat::DvbSub,
            },
        ),
        STREAM_TYPE_PRIVATE_PES if es.teletext.is_some() => (
            StreamType::Text,
            Caps::Text {
                format: g2g_core::TextFormat::Teletext,
            },
        ),
        // KLV metadata: no dedicated StreamType kind, the caps classify it.
        _ if es.klv => (StreamType::Unknown, Caps::Klv),
        _ => return None,
    };
    Some(Stream::new(id, stream_type, caps))
}

// an audio frame is always a seek resync point, a video AU only when it carries an IDR or IRAP
fn is_keyframe(stream: TsStream, au: &[u8]) -> bool {
    match stream {
        TsStream::H264 => crate::annexb::au_is_keyframe(VideoCodec::H264, au),
        TsStream::H265 => crate::annexb::au_is_keyframe(VideoCodec::H265, au),
        TsStream::Mpeg4Part2 => crate::annexb::au_is_keyframe(VideoCodec::Mpeg4Part2, au),
        TsStream::Mpeg2 => crate::annexb::au_is_keyframe(VideoCodec::Mpeg2, au),
        // AV1 is OBU-framed, not Annex-B: its resume points read from the
        // frame header instead.
        TsStream::Av1 => crate::av1parse::av1_keyframe(au),
        TsStream::Aac
        | TsStream::Mp2
        | TsStream::Opus
        | TsStream::Ac3
        | TsStream::Klv
        | TsStream::DvbSub
        | TsStream::Teletext => true,
    }
}

fn no_pinned_stream(pid: u16, stream: TsStream) -> G2gError {
    g2g_error!(
        Target::category(LOG_CATEGORY),
        "pid={pid} carries no {} stream",
        ts_stream_to_str(stream)
    );
    G2gError::CapsMismatch
}

const PROGRAM_NUMBER_PROP: PropertySpec = PropertySpec::new(
    "program-number",
    PropKind::Int,
    "PAT program to demux (-1 = the first program)",
);

/// A one-port `TsDemuxN`'s settable properties.
static ONE_PORT_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "stream",
        PropKind::Str,
        "elementary stream to emit: h264 | h265 | av1 | mpeg2 | mpeg4part2 | aac | mp2 (or mp3) | opus | ac3 | klv | dvbsub | teletext",
    ),
    PROGRAM_NUMBER_PROP,
    PropertySpec::new(
        "pid",
        PropKind::Uint,
        "PID of the elementary stream to emit among those carrying `stream` (0 = the first of them)",
    ),
];

/// `TsDemuxN`'s settable properties with several ports.
static MULTI_PORT_PROPS: &[PropertySpec] = &[PROGRAM_NUMBER_PROP];

/// Map the `program-number` property (`-1` = first program) to the demuxer's
/// selection.
fn program_number_to_int(n: Option<u16>) -> i64 {
    n.map_or(-1, i64::from)
}

/// Parse the `program-number` property: `-1` is "first program" (`None`), any
/// other non-negative value in `u16` range is that program; other negatives (and
/// out-of-range values) are rejected.
fn program_number_from_int(v: i64) -> Result<Option<u16>, PropError> {
    match v {
        -1 => Ok(None),
        0..=0xFFFF => Ok(Some(v as u16)),
        _ => Err(PropError::Value),
    }
}

/// Parse a `stream` property string to a [`TsStream`].
fn ts_stream_from_str(s: &str) -> Option<TsStream> {
    match s {
        "h264" => Some(TsStream::H264),
        "h265" => Some(TsStream::H265),
        "mpeg4part2" => Some(TsStream::Mpeg4Part2),
        "av1" => Some(TsStream::Av1),
        "mpeg2" => Some(TsStream::Mpeg2),
        "aac" => Some(TsStream::Aac),
        // one PMT stream type carries every MPEG audio layer
        "mp2" | "mp3" => Some(TsStream::Mp2),
        "opus" => Some(TsStream::Opus),
        "ac3" => Some(TsStream::Ac3),
        "klv" => Some(TsStream::Klv),
        "dvbsub" => Some(TsStream::DvbSub),
        "teletext" => Some(TsStream::Teletext),
        _ => None,
    }
}

#[cfg(feature = "std")]
const TS_STREAMS: [TsStream; 12] = [
    TsStream::H264,
    TsStream::H265,
    TsStream::Mpeg4Part2,
    TsStream::Av1,
    TsStream::Mpeg2,
    TsStream::Aac,
    TsStream::Mp2,
    TsStream::Opus,
    TsStream::Ac3,
    TsStream::Klv,
    TsStream::DvbSub,
    TsStream::Teletext,
];

// the first stream a consumer with sink caps `accepted` takes
#[cfg(feature = "std")]
pub(crate) fn stream_carrying(accepted: &CapsSet) -> Option<TsStream> {
    TS_STREAMS
        .into_iter()
        .find(|&stream| crate::demuxstream::takes(accepted, &output_caps(stream)))
}

#[cfg(feature = "std")]
pub(crate) fn stream_for_consumer(accepted: &CapsSet) -> Option<alloc::string::String> {
    crate::demuxstream::stream_for_consumer(
        &CapsSet::one(output_caps(DEFAULT_STREAM)),
        TS_STREAMS.map(|stream| (stream, output_caps(stream))),
        accepted,
    )
    .map(|stream| alloc::string::String::from(ts_stream_to_str(stream)))
}

/// The `stream` property string for a [`TsStream`].
pub(crate) fn ts_stream_to_str(stream: TsStream) -> &'static str {
    match stream {
        TsStream::H264 => "h264",
        TsStream::H265 => "h265",
        TsStream::Mpeg4Part2 => "mpeg4part2",
        TsStream::Av1 => "av1",
        TsStream::Mpeg2 => "mpeg2",
        TsStream::Aac => "aac",
        TsStream::Mp2 => "mp2",
        TsStream::Opus => "opus",
        TsStream::Ac3 => "ac3",
        TsStream::Klv => "klv",
        TsStream::DvbSub => "dvbsub",
        TsStream::Teletext => "teletext",
    }
}

/// The [`TsStream`] a demuxer forwards for a PMT elementary stream, or `None` for
/// one g2g does not forward. Takes the whole [`ElementaryStream`] because a private
/// (0x06) stream is only Opus when its descriptors carried the 'Opus' registration.
fn es_to_ts_stream(es: &crate::mpegts::ElementaryStream) -> Option<TsStream> {
    match es.stream_type {
        STREAM_TYPE_H264 => Some(TsStream::H264),
        STREAM_TYPE_H265 => Some(TsStream::H265),
        STREAM_TYPE_MPEG4P2 => Some(TsStream::Mpeg4Part2),
        STREAM_TYPE_MPEG1_VIDEO | STREAM_TYPE_MPEG2_VIDEO => Some(TsStream::Mpeg2),
        STREAM_TYPE_AAC | STREAM_TYPE_AAC_LATM => Some(TsStream::Aac),
        STREAM_TYPE_MPEG1_AUDIO | STREAM_TYPE_MPEG2_AUDIO => Some(TsStream::Mp2),
        STREAM_TYPE_PRIVATE_PES if es.opus_channels.is_some() => Some(TsStream::Opus),
        STREAM_TYPE_PRIVATE_PES if es.av1 => Some(TsStream::Av1),
        STREAM_TYPE_AC3 => Some(TsStream::Ac3),
        STREAM_TYPE_PRIVATE_PES if es.ac3 => Some(TsStream::Ac3),
        // KLV and DVB subtitles stay out: this feeds the playbin fan-out, which
        // builds A/V decode branches only (their ports are selected explicitly
        // on a TsDemuxN port).
        _ => None,
    }
}

/// One forwardable elementary stream discovered in a parsed transport stream:
/// which [`TsStream`] a demux port would carry, the elementary [`Caps`] a
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
    pub pid: u16,
}

/// The forwardable elementary streams a parsed transport stream carries, in PMT
/// order within the active program: one [`TsStreamInfo`] per PMT
/// entry whose `stream_type` maps to a [`TsStream`]. `demux` must have parsed its
/// PMT (feed a file prefix first);
/// returns empty for a non-MPEG-TS or not-yet-parsed input, which the `playbin`
/// hook reads as "decline, fall through". It drains the PES units `demux`
/// completed, and a video stream takes its picture size from the first SPS
/// among them.
pub fn forwardable_streams(demux: &mut TsDemuxer) -> Vec<TsStreamInfo> {
    let units = demux.take_units();
    demux
        .streams()
        .iter()
        .filter_map(|es| Some(stream_info(es_to_ts_stream(es)?, es.pid, &units)))
        .collect()
}

// the first SPS in the stream's PES units lets auto-plug weigh decoders by cost
fn stream_info(stream: TsStream, pid: u16, units: &[EsUnit]) -> TsStreamInfo {
    let video = matches!(
        stream,
        TsStream::H264 | TsStream::H265 | TsStream::Mpeg2 | TsStream::Mpeg4Part2 | TsStream::Av1
    );
    let placeholder = output_caps(stream);
    let caps = units
        .iter()
        .filter(|unit| unit.pid == pid)
        .find_map(|unit| sized_by_sps(&placeholder, &unit.data))
        .unwrap_or(placeholder);
    TsStreamInfo {
        stream,
        caps,
        video,
        pid,
    }
}

// a probed transport stream's audio and video streams, split by whether a demux port forwards them
#[derive(Debug, Default)]
pub struct TsProbedStreams {
    pub forwardable: Vec<TsStreamInfo>,
    pub unsupported: Vec<&'static str>,
}

// reads the PES units `demux` completed so far, to size video and tell apart ambiguous stream types
pub fn probe_streams(demux: &mut TsDemuxer) -> TsProbedStreams {
    let units = demux.take_units();
    let first_payload = |pid: u16| {
        units
            .iter()
            .find(|unit| unit.pid == pid)
            .map(|unit| unit.data.as_slice())
    };
    let mut probed = TsProbedStreams::default();
    for es in demux.streams() {
        if let Some(codec) = unsupported_codec(es, first_payload(es.pid)) {
            probed.unsupported.push(codec);
        } else if let Some(stream) = es_to_ts_stream(es) {
            let info = stream_info(stream, es.pid, &units);
            probed.forwardable.push(info);
        }
    }
    probed
}

// ISO/IEC 13818-1 audio and video stream types no `TsStream` carries
const UNSUPPORTED_STREAM_TYPES: [(u8, &str); 3] =
    [(0x21, "jpeg2000"), (0x33, "vvc"), (0x42, "cavs")];
const DTS_CORE_SYNC: [u8; 4] = [0x7F, 0xFE, 0x80, 0x01];

fn unsupported_codec(
    es: &crate::mpegts::ElementaryStream,
    first_payload: Option<&[u8]>,
) -> Option<&'static str> {
    match es.stream_type {
        STREAM_TYPE_PRIVATE_PES if es_to_ts_stream(es).is_none() => first_payload
            .is_some_and(|payload| payload.starts_with(&DTS_CORE_SYNC))
            .then_some("dts"),
        stream_type => UNSUPPORTED_STREAM_TYPES
            .iter()
            .find(|(unsupported, _)| *unsupported == stream_type)
            .map(|(_, codec)| *codec),
    }
}

impl PadTemplates for TsDemux {
    fn pad_templates() -> Vec<PadTemplate> {
        // One sink (the TS byte stream); the source pad can carry any of the
        // selectable elementary streams (an instance pins one via with_stream).
        let source = CapsSet::from_alternatives(Vec::from([
            output_caps(TsStream::H264),
            output_caps(TsStream::H265),
            output_caps(TsStream::Av1),
            output_caps(TsStream::Mpeg2),
            output_caps(TsStream::Aac),
            output_caps(TsStream::Mp2),
            output_caps(TsStream::Opus),
            output_caps(TsStream::Ac3),
            output_caps(TsStream::Klv),
            output_caps(TsStream::DvbSub),
            output_caps(TsStream::Teletext),
        ]));
        Vec::from([
            PadTemplate::sink(CapsSet::one(input_caps())),
            PadTemplate::source(source),
        ])
    }
}

/// Multi-output MPEG-TS demuxer: one TS byte stream in, N elementary
/// streams out, one selected [`TsStream`] per output port. The MPEG-TS sibling of
/// [`MkvDemuxN`](crate::mkvdemux::MkvDemuxN). [`TsDemux`] is this element with
/// one port.
///
/// A [`MultiOutputElement`] driven by
/// [`run_source_fanout`](g2g_core::runtime::run_source_fanout): it parses the
/// transport stream once and routes each PES access unit to the port whose
/// selected codec matches the unit's PMT `stream_type` (and whose PID matches, for
/// ports built with `from_stream_infos`), so one demuxer feeds
/// several decode branches in one pipeline (audio + video together). Port `i`
/// emits its elementary [`Caps`] ([`PipelinePacket::CapsChanged`]) before its
/// first frame (the branch retypes from the byte-stream input caps); a parsed
/// stream no port carries is dropped, and a port whose stream the multiplex lacks
/// stays dark. With a bus, announces the program's `StreamCollection`.
/// The `playbin uri=*.ts` fan-out builds this.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::tsdemux::{TsDemuxN, TsStream};
///
/// let element = TsDemuxN::new(vec![TsStream::H264, TsStream::Aac])
///     .with_program_number(Some(1));
/// ```
#[derive(Debug)]
pub struct TsDemuxN {
    demux: TsDemuxer,
    /// Bytes not yet consumed as whole TS packets (realignment and stride
    /// detection across frames).
    packets: TsPacketSplitter,
    /// Port `i` emits this elementary stream (one selected stream per output pad).
    ports: Vec<TsStream>,
    /// Whether port `i` has emitted its opening `CapsChanged` and the in-band
    /// config its PMT descriptor implies. Re-armed on a flush.
    announced: Vec<bool>,
    port_pids: Vec<PortPid>,
    bus: Option<BusHandle>,
    /// Set once the `StreamCollection` has been announced, so it posts once.
    collection_posted: bool,
    /// Posts the SDT service text and per-stream languages as tags.
    tags: TagPoster,
    /// App-driven stream selection: the app names the stream id each port
    /// should carry (port `i` <- selection id `i`); the demuxer re-maps its ports.
    /// Inert unless `with_stream_select` wired it. The MPEG-TS sibling of
    /// [`MkvDemuxN::with_stream_select`](crate::mkvdemux::MkvDemuxN::with_stream_select).
    stream_select: Option<StreamSelectController>,
    /// The PAT program to demux (`None` = the first program in PAT order),
    /// preserved across a parser reset so a seek keeps the selection.
    program_number: Option<u16>,
    emitted: u64,
    /// Seek support: app time seeks drive an upstream byte-seek and a
    /// re-sync. Inert unless `set_seek` wired the controllers.
    seek: DemuxSeek<EsUnit>,
    // units of every PID held while a seek waits for the earliest stream's first pts
    start_probe: Vec<EsUnit>,
    start_probe_bytes: usize,
    // every port opens with this segment, so the ports share one running-time origin
    segment: Option<Segment>,
    /// Whether port `i` has emitted its opening `Segment` yet. Re-armed on a flush.
    segment_sent: Vec<bool>,
    /// MPEG-1/2 video only: whether port `i` has seen a sequence header. Those
    /// pictures carry no geometry of their own, so a mid-GOP join hands
    /// libavcodec a stream it cannot size and it fails the lot ("invalid frame
    /// dimensions"). Dropping to the first sequence header is the tune-in
    /// convention `RtspSrc` follows for the same reason. The other video codecs
    /// keep their parameter sets in band and their decoders resynchronize on
    /// their own, so they are never dropped. Re-armed on a flush.
    mpeg2_tuned_in: Vec<bool>,
    mpeg2_skipped_before_tune_in: Vec<u64>,
    last_unit_dts_ns: Vec<Option<u64>>,
    held_unit_ns: Vec<Option<u64>>,
    timeline: RestartTimeline,
    // a Flush this demuxer did not seek for, sent on only if a Segment follows it
    withheld_flush: bool,
}

// the PID a port carries among those carrying its stream
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PortPid {
    FirstInPmtOrder,
    // a probed or app-selected track, routed whatever a later PMT lists it as
    Track(u16),
    // set through the `pid` property, which fails the run on a PID that carries another codec or never appears
    Required(u16),
}

// ISO/IEC 13818-1 caps the gap between coded PTS at 0.7 s, so a longer one is a discontinuity
const MAX_PTS_INTERVAL_NS: u64 = 700_000_000;

// output timestamps continue across a source restart, a Flush with no seek behind it, as gst's tsdemux does
#[derive(Debug, Default)]
struct RestartTimeline {
    // added to every timestamp the restarted stream carries
    offset_ns: i128,
    // the latest output dts, and the last gap between two output dts
    last_dts_ns: Option<u64>,
    step_ns: u64,
    // where the next unit's dts lands after a restart
    resume_at_ns: Option<u64>,
}

impl RestartTimeline {
    fn restart(&mut self) {
        self.resume_at_ns = self
            .last_dts_ns
            .map(|last| last.saturating_add(self.step_ns));
    }

    // a seek moves the timeline on purpose
    fn seek(&mut self) {
        *self = Self::default();
    }

    fn resuming(&self) -> bool {
        self.resume_at_ns.is_some()
    }

    // the output pts and dts of a unit
    fn map(&mut self, pts_ns: u64, dts_ns: u64) -> (u64, u64) {
        if let Some(resume_at_ns) = self.resume_at_ns.take() {
            self.offset_ns = i128::from(resume_at_ns) - i128::from(dts_ns);
        }
        let shift =
            |ns: u64| u64::try_from((i128::from(ns) + self.offset_ns).max(0)).unwrap_or(u64::MAX);
        let (pts_ns, dts_ns) = (shift(pts_ns), shift(dts_ns));
        if let Some(step) = self
            .last_dts_ns
            .and_then(|last| dts_ns.checked_sub(last))
            .filter(|&step| step > 0 && step <= MAX_PTS_INTERVAL_NS)
        {
            self.step_ns = step;
        }
        self.last_dts_ns = Some(self.last_dts_ns.map_or(dts_ns, |last| last.max(dts_ns)));
        (pts_ns, dts_ns)
    }
}

// a sparse stream (subtitles, KLV) holds a unit until the next one arrives, however long that is
fn holds_timed_units(stream: TsStream) -> bool {
    matches!(
        output_caps(stream),
        Caps::CompressedVideo { .. } | Caps::Audio { .. }
    )
}

impl TsDemuxN {
    /// A demuxer with one output port per entry of `ports` (the selected streams),
    /// in port order. Panics if `ports` is empty (a fan-out needs a port).
    pub fn new(ports: Vec<TsStream>) -> Self {
        assert!(!ports.is_empty(), "TsDemuxN needs at least one output port");
        let announced = alloc::vec![false; ports.len()];
        let port_pids = alloc::vec![PortPid::FirstInPmtOrder; ports.len()];
        let segment_sent = alloc::vec![false; ports.len()];
        let mpeg2_tuned_in = alloc::vec![false; ports.len()];
        let mpeg2_skipped_before_tune_in = alloc::vec![0; ports.len()];
        let last_unit_dts_ns = alloc::vec![None; ports.len()];
        let held_unit_ns = alloc::vec![None; ports.len()];
        Self {
            demux: TsDemuxer::new(),
            packets: TsPacketSplitter::new(),
            ports,
            announced,
            port_pids,
            seek: DemuxSeek::default(),
            start_probe: Vec::new(),
            start_probe_bytes: 0,
            segment: None,
            segment_sent,
            mpeg2_tuned_in,
            mpeg2_skipped_before_tune_in,
            last_unit_dts_ns,
            held_unit_ns,
            timeline: RestartTimeline::default(),
            withheld_flush: false,
            bus: None,
            collection_posted: false,
            tags: TagPoster::default(),
            stream_select: None,
            program_number: None,
            emitted: 0,
        }
    }

    // one port per probed stream, so two streams of one codec each reach their own port
    pub fn from_stream_infos(infos: &[TsStreamInfo]) -> Self {
        let mut demux = Self::new(infos.iter().map(|info| info.stream).collect());
        demux.port_pids = infos.iter().map(|info| PortPid::Track(info.pid)).collect();
        demux
    }

    // a PES stays in the parser until the next one on its PID starts
    fn learn_held_unit(&mut self, port: usize, dts_ns: Option<u64>) {
        let previous = core::mem::replace(&mut self.last_unit_dts_ns[port], dts_ns);
        if self.held_unit_ns[port].is_some() {
            return;
        }
        self.held_unit_ns[port] = previous
            .zip(dts_ns)
            .and_then(|(previous, now)| now.checked_sub(previous))
            .filter(|&interval| interval > 0 && interval <= MAX_PTS_INTERVAL_NS);
    }

    /// Attach the pipeline bus so the program's `StreamCollection` posts
    /// once the PMT is parsed, and its SDT service text / per-stream languages
    /// post as tags.
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.bus = Some(bus);
        self
    }

    /// Select which PAT program to demux by `program_number` (`None` = the first
    /// program in PAT order, the default).
    pub fn with_program_number(mut self, n: Option<u16>) -> Self {
        self.program_number = n;
        self.demux.set_program_number(n);
        self
    }

    /// Make the per-port stream assignment app-selectable: the controller
    /// carries stream ids (from the announced collection, e.g. `"mpegts-pid-256"`);
    /// port `i` re-maps to the stream named by selection id `i`. The demuxer
    /// re-emits that port's `CapsChanged` for the new stream and confirms the active
    /// ids on the bus ([`BusMessage::StreamsSelected`]). A port carries the PID its
    /// id names, so a re-map can pick the second of two tracks of one codec. The
    /// MPEG-TS sibling of [`MkvDemuxN::with_stream_select`](crate::mkvdemux::MkvDemuxN::with_stream_select).
    pub fn with_stream_select(mut self, select: StreamSelectController) -> Self {
        self.stream_select = Some(select);
        self
    }

    /// Apply any pending app selection: re-map port `i` to the stream named
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
            let Some((pid, stream)) = resolve_ts_stream_id(&self.demux, id) else {
                continue;
            };
            if self.ports[port] != stream || self.port_pids[port] != PortPid::Track(pid) {
                self.ports[port] = stream;
                self.port_pids[port] = PortPid::Track(pid);
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

    // called once units arrive, so the first PMT has parsed
    fn port_pid(&self, port: usize) -> Result<Option<u16>, G2gError> {
        let stream = self.ports[port];
        let pid = match self.port_pids[port] {
            PortPid::FirstInPmtOrder => return Ok(first_pid_carrying(&self.demux, stream)),
            PortPid::Track(pid) => return Ok(Some(pid)),
            PortPid::Required(pid) => pid,
        };
        let Some(listed) = self.demux.streams().iter().find(|es| es.pid == pid) else {
            // a later PMT version can still add the PID
            return Ok(None);
        };
        if pid_carries(&self.demux, stream, pid, listed.stream_type) {
            return Ok(Some(pid));
        }
        Err(no_pinned_stream(pid, stream))
    }

    fn port_for_pid(&self, pid: u16) -> Result<Option<usize>, G2gError> {
        for port in 0..self.ports.len() {
            if self.port_pid(port)? == Some(pid) {
                return Ok(Some(port));
            }
        }
        Ok(None)
    }

    /// Announce the active program's elementary streams as a
    /// [`BusMessage::StreamCollection`], once, when that program's PMT has
    /// parsed. The collection is the selected program's streams (GStreamer's
    /// collection is per-program too). A no-op without a bus, before the PMT, or
    /// once already posted.
    fn post_stream_collection(&mut self) {
        if self.collection_posted {
            return;
        }
        let streams: Vec<Stream> = self
            .demux
            .streams()
            .iter()
            .filter_map(es_to_stream)
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

    /// Reset the parser for a discontinuity (a `Flush` / seek): drop buffered
    /// bytes and the demuxer's PAT/PMT/PES state, which the re-read stream
    /// re-establishes from its start, and re-arm what each port sends ahead of
    /// its first unit.
    fn reset_parser(&mut self) {
        self.packets.clear();
        self.demux = TsDemuxer::new();
        self.demux.set_program_number(self.program_number);
        self.start_probe.clear();
        self.start_probe_bytes = 0;
        self.announced.fill(false);
        self.segment_sent.fill(false);
        self.mpeg2_tuned_in.fill(false);
        self.mpeg2_skipped_before_tune_in.fill(0);
        self.last_unit_dts_ns.fill(None);
    }

    // ffmpeg -ss counts from the earliest stream's start, and a file can carry that stream last
    fn past_stream_start_probe(&mut self, units: Vec<EsUnit>, ended: bool) -> Vec<EsUnit> {
        // ffmpeg's default probesize
        const STREAM_START_PROBE_BYTES: usize = 5_000_000;
        if !self.seek.awaits_stream_start() {
            return units;
        }
        self.start_probe_bytes = units.iter().fold(self.start_probe_bytes, |bytes, u| {
            bytes.saturating_add(u.data.len())
        });
        self.start_probe.extend(units);
        let probe = &self.start_probe;
        let every_stream_started = self
            .demux
            .streams()
            .iter()
            .filter(|es| es_to_ts_stream(es).is_some())
            .all(|es| {
                probe
                    .iter()
                    .any(|u| u.pid == es.pid && u.pts_90khz.is_some())
            });
        if !every_stream_started && !ended && self.start_probe_bytes <= STREAM_START_PROBE_BYTES {
            return Vec::new();
        }
        // a stream starts at its first stamped unit in decode order, as ffmpeg's start_time does
        let mut started_pids: Vec<u16> = Vec::new();
        let mut start_90khz: Option<u64> = None;
        for u in probe {
            let Some(pts) = u.pts_90khz.filter(|_| !started_pids.contains(&u.pid)) else {
                continue;
            };
            started_pids.push(u.pid);
            start_90khz = Some(start_90khz.map_or(pts, |start| start.min(pts)));
        }
        if let Some(start) = start_90khz {
            self.seek.note_stream_start(ticks_90khz_to_ns(start));
        }
        self.start_probe_bytes = 0;
        core::mem::take(&mut self.start_probe)
    }

    /// Route each completed access unit to the port carrying its PID. While a
    /// seek re-syncs, hold from the keyframe at or before the target until the
    /// target.
    async fn route_units(
        &mut self,
        units: Vec<EsUnit>,
        out: &mut dyn MultiOutputSink,
    ) -> Result<(), G2gError> {
        for u in units {
            let pts_ns = u.pts_90khz.map(ticks_90khz_to_ns);
            if let Some(pts_ns) = pts_ns {
                self.seek.note_stream_start(pts_ns);
            }
            let Some(port) = self.port_for_pid(u.pid)? else {
                continue; // a stream no selected port carries
            };
            let stream = self.ports[port];
            // MPEG-2 tune-in: nothing before the first sequence header can be
            // sized, let alone decoded.
            if stream == TsStream::Mpeg2 && !self.mpeg2_tuned_in[port] {
                if !mpeg2_can_open_here(&u.data) {
                    self.mpeg2_skipped_before_tune_in[port] += 1;
                    continue;
                }
                self.mpeg2_tuned_in[port] = true;
                g2g_core::g2g_debug!(
                    Target::category(LOG_CATEGORY),
                    "port {port} tuned in at the first sequence header after dropping {} units",
                    self.mpeg2_skipped_before_tune_in[port]
                );
                self.mpeg2_skipped_before_tune_in[port] = 0;
            }
            let keyframe = is_keyframe(stream, &u.data);
            let admitted = self.seek.admit(pts_ns.unwrap_or(0), keyframe, u);
            if admitted.segment.is_some() {
                self.segment = admitted.segment;
            }
            for u in admitted {
                self.emit_unit(port, u, out).await?;
            }
        }
        Ok(())
    }

    /// Emit one access unit on `port` as frames (Annex-B for H.264 / H.265, ADTS
    /// for AAC, raw MPEG-audio frames for Mp2, and control-header-unwrapped raw
    /// Opus packets for Opus), preceded by the port's opening `Segment`,
    /// `CapsChanged` and in-band config when it has not sent them yet.
    async fn emit_unit(
        &mut self,
        port: usize,
        u: EsUnit,
        out: &mut dyn MultiOutputSink,
    ) -> Result<(), G2gError> {
        let stream = self.ports[port];
        let pts_ns = u.pts_90khz.map_or(0, ticks_90khz_to_ns);
        // Reordered video carries a separate DTS; fall back to the PTS when
        // absent (the no-DTS convention shared with the fmp4 / mkv demux).
        let dts_ns = u.dts_90khz.map_or(pts_ns, ticks_90khz_to_ns);
        let resuming = self.timeline.resuming();
        let (pts_ns, dts_ns) = self.timeline.map(pts_ns, dts_ns);
        // Stream start: map the first emitted DTS to running time 0, since
        // open-GOP B frames present before the first I frame. A seek's resume
        // already did that for its own case.
        if resuming && self.segment.is_some() {
            self.seek.keep_segment();
        } else if let Some(opening) = self.seek.opening_segment(dts_ns) {
            self.segment = Some(opening);
        }
        // another stream can start before the first unit out, gst's tsdemux opens at the lowest first pts
        let lowered = u.pts_90khz.and_then(|_| self.seek.lowered_segment(pts_ns));
        if let Some(lowered) = lowered {
            self.segment = Some(lowered);
            for opened_port in 0..self.ports.len() {
                if self.segment_sent[opened_port] {
                    out.push_to(opened_port, PipelinePacket::Segment(lowered))
                        .await?;
                }
            }
        }
        if !self.segment_sent[port] {
            self.segment_sent[port] = true;
            let segment = self.segment.expect("the stream opened with a segment");
            out.push_to(port, PipelinePacket::Segment(segment)).await?;
        }
        if !self.announced[port] {
            let caps = match stream {
                TsStream::Opus => opus_stream_caps(&self.demux, u.pid),
                stream => output_caps(stream),
            };
            out.push_to(port, PipelinePacket::CapsChanged(caps)).await?;
            self.announced[port] = true;
            if let Some(blob) = in_band_config(&self.demux, u.pid, stream) {
                let frame = config_frame(blob, pts_ns, self.emitted);
                self.emitted += 1;
                out.push_to(port, PipelinePacket::DataFrame(frame)).await?;
            }
        }
        if holds_timed_units(stream) {
            self.learn_held_unit(port, u.pts_90khz.map(|_| dts_ns));
        }
        let data = unwrap_sync_klv(u.stream_type, u.data);
        let units = match stream {
            TsStream::Opus => crate::mpegts::opus_ts_packets(&data)
                .into_iter()
                .map(<[u8]>::to_vec)
                .collect(),
            _ => alloc::vec![data],
        };
        for unit in units {
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(unit.into_boxed_slice())),
                FrameTiming {
                    pts_ns,
                    dts_ns,
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

// an Opus stream's caps carry its descriptor's channel count, since no OpusHead rides a transport stream
fn opus_stream_caps(demux: &TsDemuxer, pid: u16) -> Caps {
    Caps::Audio {
        format: AudioFormat::Opus,
        channels: demux.opus_channels(pid).unwrap_or(2),
        sample_rate: 0,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    }
}

/// Whether an MPEG-1/2 video access unit can open a decode: it must carry the
/// sequence header, the only thing that states the picture geometry. Shared with
/// the program stream demuxer's tune-in, which drops on the same rule.
fn mpeg2_can_open_here(au: &[u8]) -> bool {
    crate::psdemux::parse_sequence_header(au).is_some()
}

/// The in-band config a stream's PMT descriptor implies: the DVB
/// `subtitling_descriptor`'s page ids (the same bytes a Matroska `S_DVBSUB`
/// `CodecPrivate` carries), the `teletext_descriptor`'s page and language, or
/// an `OpusHead` for the Opus descriptor's channel count. The PMT is the only
/// place a transport stream states these, so they have to reach the decoder
/// some way, and this is the one a Matroska source already uses.
fn in_band_config(demux: &TsDemuxer, pid: u16, stream: TsStream) -> Option<Vec<u8>> {
    match stream {
        // a transport stream states no pre-skip, which is what ffmpeg's demuxer assumes too
        TsStream::Opus => Some(crate::opusparse::synth_opus_head(
            demux.opus_channels(pid)?,
            crate::opusparse::OPUS_RATE_HZ,
            0,
        )),
        TsStream::DvbSub => {
            let (subtitling_type, composition, ancillary) = demux.subtitling(pid)?;
            Some(
                crate::dvbsub::page_id_blob(
                    crate::dvbsub::PageIds {
                        composition,
                        ancillary,
                    },
                    subtitling_type,
                )
                .to_vec(),
            )
        }
        TsStream::Teletext => {
            let service = demux.teletext(pid)?;
            Some(
                crate::teletext::page_config_blob(service.page_number(), service.language).to_vec(),
            )
        }
        _ => None,
    }
}

fn config_frame(blob: Vec<u8>, pts_ns: u64, seq: u64) -> Frame {
    Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(blob.into_boxed_slice())),
        FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            ..FrameTiming::default()
        },
        seq,
    )
}

/// Resolve a collection stream id (`"mpegts-pid-{pid}"`) to its PID and the [`TsStream`] that
/// selects it, by looking the PID up in the parsed PMT and mapping its
/// `stream_type`. `None` for an unparseable id, an unknown PID, or a stream_type
/// g2g does not forward.
fn resolve_ts_stream_id(demux: &TsDemuxer, id: &str) -> Option<(u16, TsStream)> {
    let pid: u16 = id.strip_prefix("mpegts-pid-")?.parse().ok()?;
    let es = demux.streams().iter().find(|e| e.pid == pid)?;
    let stream = match es.stream_type {
        STREAM_TYPE_H264 => TsStream::H264,
        STREAM_TYPE_H265 => TsStream::H265,
        STREAM_TYPE_MPEG4P2 => TsStream::Mpeg4Part2,
        STREAM_TYPE_MPEG1_VIDEO | STREAM_TYPE_MPEG2_VIDEO => TsStream::Mpeg2,
        STREAM_TYPE_PRIVATE_PES if es.av1 => TsStream::Av1,
        STREAM_TYPE_AAC | STREAM_TYPE_AAC_LATM => TsStream::Aac,
        _ => return None,
    };
    Some((pid, stream))
}

// a private PES (0x06) carries a codec only when its PMT descriptors name it
fn pid_carries(demux: &TsDemuxer, stream: TsStream, pid: u16, stream_type: u8) -> bool {
    if !accepts_stream_type(stream, stream_type) {
        return false;
    }
    let private = stream_type == STREAM_TYPE_PRIVATE_PES;
    match stream {
        TsStream::Opus => demux.opus_channels(pid).is_some(),
        TsStream::Ac3 => !private || demux.is_dvb_ac3(pid),
        TsStream::Klv => !private || demux.is_klv(pid),
        TsStream::Av1 => demux.is_av1(pid),
        TsStream::DvbSub => demux.subtitling(pid).is_some(),
        TsStream::Teletext => demux.teletext(pid).is_some(),
        _ => true,
    }
}

// in PMT order, so several tracks of one codec (languages) never interleave on one output
fn first_pid_carrying(demux: &TsDemuxer, stream: TsStream) -> Option<u16> {
    demux
        .streams()
        .iter()
        .find(|es| pid_carries(demux, stream, es.pid, es.stream_type))
        .map(|es| es.pid)
}

impl MultiOutputElement for TsDemuxN {
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
        CapsConstraint::Accepts(CapsSet::one(input_caps()))
    }

    /// Declare each port's elementary-stream caps, so the solver negotiates
    /// each branch against its codec at startup. `None` for an out-of-range port.
    fn port_output_caps(&self, port: usize) -> Option<Caps> {
        self.ports.get(port).map(|&stream| output_caps(stream))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        absolute_caps
            .intersect(&input_caps())
            .map(|_| ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            // M362: a pending app seek triggers an upstream byte-seek; until its
            // `Flush` returns, drop input so no stale pre-seek units are emitted.
            self.seek.poll_request();
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    self.withheld_flush = false;
                    if self.seek.dropping_input() {
                        return Ok(());
                    }
                    let slice = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    self.packets.push(slice, &mut self.demux);
                    if self.bus.is_some() {
                        self.post_stream_collection();
                        self.tags.post(&mut self.demux, self.bus.as_ref());
                    }
                    // Honor an app selection before routing this batch, so a re-map
                    // (and its re-armed CapsChanged) takes effect for the frames now.
                    self.apply_stream_selection();
                    let units = self.demux.take_units();
                    let units = self.past_stream_start_probe(units, false);
                    self.route_units(units, out).await?;
                }
                // The upstream byte-seek's flush or a source restart: reset the
                // parser, then re-sync from the re-read stream. Broadcast it to
                // every branch.
                PipelinePacket::Flush => {
                    let seeking = self.seek.dropping_input();
                    if seeking {
                        self.timeline.seek();
                    } else {
                        self.timeline.restart();
                    }
                    self.seek.on_flush();
                    self.reset_parser();
                    // a restarted source's next stream carries on the timeline, so the
                    // decoders keep the frames they hold, as gst's do
                    if !seeking {
                        self.withheld_flush = true;
                        return Ok(());
                    }
                    for port in 0..self.ports.len() {
                        out.push_to(port, PipelinePacket::Flush).await?;
                    }
                }
                PipelinePacket::Segment(seg) => {
                    // an upstream seek
                    if core::mem::take(&mut self.withheld_flush) {
                        for port in 0..self.ports.len() {
                            out.push_to(port, PipelinePacket::Flush).await?;
                        }
                    }
                    self.timeline.seek();
                    self.segment = Some(seg);
                    for port in 0..self.ports.len() {
                        out.push_to(port, PipelinePacket::Segment(seg)).await?;
                    }
                }
                PipelinePacket::Eos => {
                    self.withheld_flush = false;
                    if self.seek.eos_before_flush(self.seek.dropping_input())? {
                        return Ok(());
                    }
                    // Flush the final in-flight PES, route it; the runner
                    // broadcasts the merged Eos to every port.
                    self.packets.finish(&mut self.demux);
                    self.demux.flush();
                    let units = self.demux.take_units();
                    let units = self.past_stream_start_probe(units, true);
                    self.route_units(units, out).await?;
                    for port in 0..self.ports.len() {
                        if let (PortPid::Required(pid), None) =
                            (self.port_pids[port], self.port_pid(port)?)
                        {
                            return Err(no_pinned_stream(pid, self.ports[port]));
                        }
                    }
                    if self.seek.takes_seeks_after_end() {
                        for port in 0..self.ports.len() {
                            out.push_to(port, PipelinePacket::Eos).await?;
                        }
                        self.seek.wait_for_seek_or_shutdown().await;
                        self.seek.poll_request();
                    }
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

    fn seek_route(&self) -> Option<ElementSeekRoute> {
        // a seek re-syncs on one stream's keyframes
        (self.ports.len() == 1).then_some(ElementSeekRoute::UpstreamBytes)
    }

    fn set_seek(&mut self, app: SeekController, upstream: SeekController) {
        self.seek.with(app, upstream);
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "MPEG-TS demuxer",
            "Codec/Demuxer",
            "Demuxes an MPEG transport stream into the elementary streams its ports select",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        if self.ports.len() == 1 {
            ONE_PORT_PROPS
        } else {
            MULTI_PORT_PROPS
        }
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        let one_port = self.ports.len() == 1;
        match name {
            "program-number" => {
                let n = program_number_from_int(value.as_int().ok_or(PropError::Type)?)?;
                self.program_number = n;
                self.demux.set_program_number(n);
                Ok(())
            }
            "stream" if one_port => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.ports[0] = ts_stream_from_str(s).ok_or(PropError::Value)?;
                Ok(())
            }
            "pid" if one_port => {
                let pid = value.as_uint().ok_or(PropError::Type)?;
                self.port_pids[0] = match (pid, self.port_pids[0]) {
                    (0, _) => PortPid::FirstInPmtOrder,
                    // the PID a probed port reads back, so setting it again changes nothing
                    (pid, PortPid::Track(track)) if pid == u64::from(track) => {
                        PortPid::Track(track)
                    }
                    (pid, _) => {
                        PortPid::Required(u16::try_from(pid).map_err(|_| PropError::Value)?)
                    }
                };
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        let one_port = self.ports.len() == 1;
        match name {
            "program-number" => Some(PropValue::Int(program_number_to_int(self.program_number))),
            "stream" if one_port => Some(PropValue::Str(ts_stream_to_str(self.ports[0]).into())),
            "pid" if one_port => Some(PropValue::Uint(match self.port_pids[0] {
                PortPid::FirstInPmtOrder => 0,
                PortPid::Track(pid) | PortPid::Required(pid) => pid.into(),
            })),
            _ => None,
        }
    }

    fn latency(&self) -> LatencyReport {
        let held_ns = self
            .held_unit_ns
            .iter()
            .flatten()
            .copied()
            .max()
            .unwrap_or(0);
        LatencyReport::buffered(held_ns, Some(held_ns))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mpegts::{SYNC_BYTE, TS_PACKET_LEN};
    use g2g_core::{OutputSink, PushOutcome, RawVideoFormat};

    /// PMT stream_type 0x10 maps to an MPEG-4 Part 2 video stream, and the
    /// selection round-trips through the `stream` property string and the
    /// stream_type <-> TsStream tables.
    #[test]
    fn stream_type_0x10_is_mpeg4_part2() {
        let es = crate::mpegts::ElementaryStream {
            pid: 0x100,
            stream_type: STREAM_TYPE_MPEG4P2,
            opus_channels: None,
            ac3: false,
            klv: false,
            av1: false,
            subtitling: None,
            teletext: None,
            language: None,
        };
        let stream = es_to_stream(&es).expect("0x10 is forwarded");
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
            es_to_ts_stream(&crate::mpegts::ElementaryStream {
                pid: 0x100,
                stream_type: STREAM_TYPE_MPEG4P2,
                opus_channels: None,
                ac3: false,
                klv: false,
                av1: false,
                subtitling: None,
                teletext: None,
                language: None,
            }),
            Some(TsStream::Mpeg4Part2)
        );
        assert_eq!(
            selected_stream_type(TsStream::Mpeg4Part2),
            STREAM_TYPE_MPEG4P2
        );
        assert_eq!(ts_stream_from_str("mpeg4part2"), Some(TsStream::Mpeg4Part2));
        assert_eq!(ts_stream_to_str(TsStream::Mpeg4Part2), "mpeg4part2");
    }

    /// The playbin fan-out must put an MPEG-4 Part 2 stream in a video branch:
    /// it once fell through the H.264/H.265-only `video` match and was routed
    /// as audio.
    #[test]
    fn forwardable_streams_marks_mpeg4part2_video() {
        let mut d = TsDemuxer::new();
        d.push_packet(&pat(0x1000));
        d.push_packet(&pmt2(0x100, STREAM_TYPE_MPEG4P2, 0x101, STREAM_TYPE_AAC));
        let infos = forwardable_streams(&mut d);
        let v = infos
            .iter()
            .find(|i| i.stream == TsStream::Mpeg4Part2)
            .expect("mpeg4part2 is forwardable");
        assert!(v.video, "mpeg4part2 routes to a video branch");
        let a = infos
            .iter()
            .find(|i| i.stream == TsStream::Aac)
            .expect("aac is forwardable");
        assert!(!a.video);
    }

    // Re-use the synthetic TS builders by constructing equivalent packets here.
    fn ts_packet(pid: u16, pusi: bool, payload: &[u8]) -> Vec<u8> {
        const ROOM: usize = TS_PACKET_LEN - 4;
        let mut p = alloc::vec![0u8; TS_PACKET_LEN];
        p[0] = SYNC_BYTE;
        p[1] = if pusi { 0x40 } else { 0x00 } | ((pid >> 8) as u8 & 0x1F);
        p[2] = (pid & 0xFF) as u8;
        let continuity_counter = crate::mpegts::tests::next_continuity_counter(pid);
        let l = payload.len();
        if l == ROOM {
            p[3] = 0x10 | continuity_counter;
            p[4..].copy_from_slice(payload);
        } else {
            p[3] = 0x30 | continuity_counter;
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
        pts_ns: Vec<u64>,
        segments: Vec<Segment>,
        flushes: usize,
        eos: bool,
    }
    impl OutputSink for CaptureSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");
            core::task::Poll::Ready({
                match packet {
                    PipelinePacket::DataFrame(f) => {
                        if let Some(s) = f.domain.as_system_slice() {
                            self.frames.push(s.to_vec());
                        }
                        self.pts_ns.push(f.timing.pts_ns);
                    }
                    PipelinePacket::Segment(seg) => self.segments.push(seg),
                    PipelinePacket::Flush => self.flushes += 1,
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
        assert!(d.intercept_caps(&input_caps()).is_ok());
        // A non-TS byte stream / other caps is rejected.
        let raw = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(2),
            height: Dim::Fixed(2),
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        assert!(d.intercept_caps(&raw).is_err());
    }

    #[tokio::test]
    async fn demuxes_h264_frames_from_ts_bytes() {
        let es_pid = 0x0100;
        let mut d = TsDemux::new();
        d.configure_pipeline(&input_caps()).unwrap();

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

    const RESTART_FRAME_TICKS: u64 = 3_000;
    const FIRST_SESSION_TICKS: u64 = 900_000;
    // a restarted origin starts its clock again, below where the first session stopped
    const SECOND_SESSION_TICKS: u64 = 90_000;

    // one access unit per frame period from `start_ticks`
    fn session(
        mut muxer: crate::mpegts::TsMuxer,
        start_ticks: u64,
        aus: &[Vec<u8>],
    ) -> PipelinePacket {
        let mut stream = Vec::new();
        for (frame, au) in (0..).zip(aus) {
            let pts = start_ticks + frame * RESTART_FRAME_TICKS;
            stream.extend(muxer.push_au(au, Some(pts), None, true));
        }
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(stream.into_boxed_slice())),
            FrameTiming::default(),
            0,
        ))
    }

    fn h264_session(start_ticks: u64, frames: u64) -> PipelinePacket {
        let aus: Vec<Vec<u8>> = (0..frames)
            .map(|frame| alloc::vec![0, 0, 0, 1, 0x65, frame as u8])
            .collect();
        session(
            crate::mpegts::TsMuxer::with_streams(&[STREAM_TYPE_H264]),
            start_ticks,
            &aus,
        )
    }

    // the pts a demuxer reads back from `h264_session`, which adds the muxer's decoder delay
    fn session_pts_ns(start_ticks: u64, frames: u64) -> Vec<u64> {
        let delay = crate::mpegts::TsMuxer::with_streams(&[STREAM_TYPE_H264]).decoder_delay_90khz();
        (0..frames)
            .map(|frame| ticks_90khz_to_ns(start_ticks + delay + frame * RESTART_FRAME_TICKS))
            .collect()
    }

    // the first session's two emitted frames, then the second session one frame gap after them
    fn continued_pts_ns() -> Vec<u64> {
        let first = session_pts_ns(FIRST_SESSION_TICKS, 2);
        let second = session_pts_ns(SECOND_SESSION_TICKS, 3);
        let resume_ns = first[1] + (first[1] - first[0]);
        let continued = second.iter().map(|pts| pts - second[0] + resume_ns);
        first.iter().copied().chain(continued).collect()
    }

    // the flush drops the first session's last unit, still waiting for the PES after it
    #[tokio::test]
    async fn a_restarted_source_continues_the_timeline() {
        let mut d = TsDemux::new();
        d.configure_pipeline(&input_caps()).unwrap();
        let mut sink = CaptureSink::default();
        for packet in [
            h264_session(FIRST_SESSION_TICKS, 3),
            PipelinePacket::Flush,
            h264_session(SECOND_SESSION_TICKS, 3),
            PipelinePacket::Eos,
        ] {
            d.process(packet, &mut sink).await.unwrap();
        }

        assert_eq!(sink.pts_ns, continued_pts_ns());
        let [opening, continued] = sink.segments.as_slice() else {
            panic!("one segment per session, got {:?}", sink.segments);
        };
        assert_eq!(continued, opening);
        assert_eq!(sink.flushes, 0, "the decoders keep their frames");
    }

    #[tokio::test]
    async fn an_upstream_seek_keeps_the_new_timestamps() {
        let mut d = TsDemux::new();
        d.configure_pipeline(&input_caps()).unwrap();
        let mut sink = CaptureSink::default();
        for packet in [
            h264_session(FIRST_SESSION_TICKS, 3),
            PipelinePacket::Flush,
            PipelinePacket::Segment(Segment::new()),
            h264_session(SECOND_SESSION_TICKS, 3),
            PipelinePacket::Eos,
        ] {
            d.process(packet, &mut sink).await.unwrap();
        }

        let mut expected = session_pts_ns(FIRST_SESSION_TICKS, 2);
        expected.extend(session_pts_ns(SECOND_SESSION_TICKS, 3));
        assert_eq!(sink.pts_ns, expected);
        assert_eq!(sink.flushes, 1);
    }

    #[derive(Default)]
    struct PortCapture {
        frames: Vec<Vec<u8>>,
        pts_ns: Vec<u64>,
    }

    impl MultiOutputSink for PortCapture {
        fn port_count(&self) -> usize {
            1
        }

        fn poll_push_to(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            _port: usize,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            if let Some(PipelinePacket::DataFrame(frame)) = packet_slot.take() {
                if let Some(bytes) = frame.domain.as_system_slice() {
                    self.frames.push(bytes.to_vec());
                }
                self.pts_ns.push(frame.timing.pts_ns);
            }
            core::task::Poll::Ready(Ok(PushOutcome::Accepted))
        }
    }

    async fn run_ports(
        demux: &mut TsDemuxN,
        packets: impl IntoIterator<Item = PipelinePacket>,
    ) -> Result<PortCapture, G2gError> {
        let mut sink = PortCapture::default();
        for packet in packets {
            MultiOutputElement::process(demux, packet, &mut sink).await?;
        }
        Ok(sink)
    }

    #[tokio::test]
    async fn a_restarted_source_continues_every_port_timeline() {
        let mut d = TsDemuxN::new(Vec::from([TsStream::H264]));
        let sink = run_ports(
            &mut d,
            [
                h264_session(FIRST_SESSION_TICKS, 3),
                PipelinePacket::Flush,
                h264_session(SECOND_SESSION_TICKS, 3),
                PipelinePacket::Eos,
            ],
        )
        .await
        .unwrap();

        assert_eq!(sink.pts_ns, continued_pts_ns());
    }

    // a 16x16, 25 fps sequence header, then a picture start code
    const MPEG2_SEQUENCE: [u8; 13] = [0, 0, 1, 0xB3, 0x01, 0x00, 0x10, 0x13, 0, 0, 1, 0x00, 0xAA];
    const MPEG2_PICTURE: [u8; 5] = [0, 0, 1, 0x00, 0xBB];

    // the flush drops each session's last unit, still waiting for the PES after it
    #[tokio::test]
    async fn a_restarted_mpeg2_port_tunes_in_again() {
        let mpeg2 = || crate::mpegts::TsMuxer::with_streams(&[STREAM_TYPE_MPEG2_VIDEO]);
        let (sequence, picture) = (MPEG2_SEQUENCE.to_vec(), MPEG2_PICTURE.to_vec());
        let mut d = TsDemuxN::new(Vec::from([TsStream::Mpeg2]));
        let sink = run_ports(
            &mut d,
            [
                session(
                    mpeg2(),
                    FIRST_SESSION_TICKS,
                    &[sequence.clone(), picture.clone(), picture.clone()],
                ),
                PipelinePacket::Flush,
                session(
                    mpeg2(),
                    SECOND_SESSION_TICKS,
                    &[picture.clone(), sequence.clone(), picture.clone()],
                ),
                PipelinePacket::Eos,
            ],
        )
        .await
        .unwrap();

        assert_eq!(
            sink.frames,
            [sequence.clone(), picture.clone(), sequence, picture]
        );
    }

    #[tokio::test]
    async fn a_restarted_dvb_subtitle_port_sends_its_page_ids_again() {
        const SUBTITLING_TYPE: u8 = 0x10;
        let ids = crate::dvbsub::PageIds {
            composition: 1,
            ancillary: 2,
        };
        let dvbsub = || {
            let mut muxer = crate::mpegts::TsMuxer::with_streams(&[STREAM_TYPE_PRIVATE_PES]);
            muxer.set_stream_subtitling(0, "eng", SUBTITLING_TYPE, ids);
            muxer
        };
        let display_sets = [
            alloc::vec![0x0F, 0x10, 0x00, 0x01],
            alloc::vec![0x0F, 0x10, 0x00, 0x02],
        ];
        let mut d = TsDemuxN::new(Vec::from([TsStream::DvbSub]));
        let sink = run_ports(
            &mut d,
            [
                session(dvbsub(), FIRST_SESSION_TICKS, &display_sets),
                PipelinePacket::Flush,
                session(dvbsub(), SECOND_SESSION_TICKS, &display_sets),
                PipelinePacket::Eos,
            ],
        )
        .await
        .unwrap();

        let page_ids = crate::dvbsub::page_id_blob(ids, SUBTITLING_TYPE).to_vec();
        assert_eq!(
            sink.frames.len(),
            5,
            "page ids and one set, page ids and two sets"
        );
        assert_eq!(sink.frames[0], page_ids);
        assert_eq!(sink.frames[2], page_ids);
    }

    #[tokio::test]
    async fn a_pid_carrying_another_codec_fails_the_run() {
        const VIDEO_PID: u16 = 0x0100;
        const AUDIO_PID: u16 = 0x0101;
        let mut stream = Vec::new();
        stream.extend_from_slice(&pat(0x1000));
        stream.extend_from_slice(&pmt2(
            VIDEO_PID,
            STREAM_TYPE_H264,
            AUDIO_PID,
            STREAM_TYPE_AAC,
        ));
        stream.extend_from_slice(&ts_packet(
            VIDEO_PID,
            true,
            &pes_id(0xE0, &[0, 0, 0, 1, 0x65]),
        ));

        let mut d = TsDemuxN::new(Vec::from([TsStream::Aac]));
        d.set_property("pid", PropValue::Uint(VIDEO_PID.into()))
            .unwrap();
        let ran = run_ports(
            &mut d,
            [
                PipelinePacket::DataFrame(Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(stream.into_boxed_slice())),
                    FrameTiming::default(),
                    0,
                )),
                PipelinePacket::Eos,
            ],
        )
        .await;
        assert!(matches!(ran, Err(G2gError::CapsMismatch)));
    }

    // the PID `h264_session` muxes its track on
    const TRACK_PID: u16 = crate::mpegts::MUX_ES_PID;
    const MOVED_TRACK_PID: u16 = TRACK_PID + 1;

    // the port a playbin fan-out builds for a probed H.264 track
    fn probed_track_port() -> TsDemuxN {
        TsDemuxN::from_stream_infos(&[TsStreamInfo {
            stream: TsStream::H264,
            caps: output_caps(TsStream::H264),
            video: true,
            pid: TRACK_PID,
        }])
    }

    fn two_units(pmt: Vec<u8>, pid: u16, stream_id: u8, units: [&[u8]; 2]) -> PipelinePacket {
        let mut stream = pat(0x1000);
        stream.extend_from_slice(&pmt);
        for unit in units {
            stream.extend_from_slice(&ts_packet(pid, true, &pes_id(stream_id, unit)));
        }
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(stream.into_boxed_slice())),
            FrameTiming::default(),
            0,
        ))
    }

    const H264_AU: [u8; 5] = [0, 0, 0, 1, 0x65];

    #[tokio::test]
    async fn a_probed_track_port_ends_cleanly_without_data() {
        let ran = run_ports(&mut probed_track_port(), [PipelinePacket::Eos]).await;
        assert!(ran.is_ok(), "{:?}", ran.err());
    }

    // the flush drops the first session's last unit, still waiting for the PES after it
    #[tokio::test]
    async fn a_probed_track_port_stays_dark_when_a_restart_moves_its_pid() {
        let sink = run_ports(
            &mut probed_track_port(),
            [
                h264_session(FIRST_SESSION_TICKS, 3),
                PipelinePacket::Flush,
                two_units(
                    pmt(MOVED_TRACK_PID),
                    MOVED_TRACK_PID,
                    0xE0,
                    [&H264_AU, &H264_AU],
                ),
                PipelinePacket::Eos,
            ],
        )
        .await
        .unwrap();
        assert_eq!(sink.pts_ns, session_pts_ns(FIRST_SESSION_TICKS, 2));
    }

    async fn run_a_restart_that_lists_the_track_pid_as_aac(
        demux: &mut TsDemuxN,
    ) -> Result<PortCapture, G2gError> {
        const ADTS: [u8; 4] = [0xFF, 0xF1, 0x50, 0x80];
        run_ports(
            demux,
            [
                h264_session(FIRST_SESSION_TICKS, 3),
                PipelinePacket::Flush,
                two_units(
                    pmt2(
                        MOVED_TRACK_PID,
                        STREAM_TYPE_H264,
                        TRACK_PID,
                        STREAM_TYPE_AAC,
                    ),
                    TRACK_PID,
                    0xC0,
                    [&ADTS, &ADTS],
                ),
                PipelinePacket::Eos,
            ],
        )
        .await
    }

    #[tokio::test]
    async fn a_probed_track_port_survives_a_restart_that_lists_its_pid_as_another_codec() {
        let ran = run_a_restart_that_lists_the_track_pid_as_aac(&mut probed_track_port()).await;
        assert!(ran.is_ok(), "{:?}", ran.err());
    }

    #[tokio::test]
    async fn setting_a_probed_port_the_pid_it_reads_back_keeps_it_routing() {
        let mut demux = probed_track_port();
        let pid = demux
            .get_property("pid")
            .expect("a one-port demuxer has a pid");
        demux.set_property("pid", pid).unwrap();
        let ran = run_a_restart_that_lists_the_track_pid_as_aac(&mut demux).await;
        assert!(ran.is_ok(), "{:?}", ran.err());
    }

    #[test]
    #[should_panic(expected = "a one-port element is a demuxer with caps for port 0 alone")]
    fn a_one_port_demuxer_rejects_a_second_port() {
        let _ = TsDemux::new().map(|_| TsDemuxN::new(Vec::from([TsStream::H264, TsStream::Aac])));
    }

    #[tokio::test]
    async fn open_gop_b_frames_before_the_first_i_frame_stay_in_the_segment() {
        const FRAME_TICKS: u64 = 3_600;
        const START_TICKS: u64 = 900_000;
        let mut m = crate::mpegts::TsMuxer::with_streams(&[STREAM_TYPE_H264]);
        let mut stream = m.push_au(
            &[0, 0, 0, 1, 0x65, 0x11],
            Some(START_TICKS + 3 * FRAME_TICKS),
            Some(START_TICKS + FRAME_TICKS),
            true,
        );
        for frame in [1, 2] {
            let pts = START_TICKS + frame * FRAME_TICKS;
            stream.extend(m.push_au(&[0, 0, 0, 1, 0x01, 0x22], Some(pts), None, false));
        }

        let mut d = TsDemux::new();
        d.configure_pipeline(&input_caps()).unwrap();
        let mut sink = CaptureSink::default();
        run_demux(&mut d, &stream, &mut sink).await;
        let [seg] = sink.segments.as_slice() else {
            panic!("one stream-start segment, got {:?}", sink.segments);
        };
        assert_eq!(sink.pts_ns.len(), 3);
        assert!(
            sink.pts_ns.iter().all(|&pts| seg.presents(pts)),
            "every picture presents inside {seg:?}: {:?}",
            sink.pts_ns
        );
    }

    const FIRST_AAC_PID: u16 = 0x0101;
    const SECOND_AAC_PID: u16 = 0x0102;
    const FIRST_AAC: [u8; 6] = [0xFF, 0xF1, 0x50, 0x80, 0x01, 0x23];
    const SECOND_AAC: [u8; 6] = [0xFF, 0xF1, 0x50, 0x80, 0x02, 0x45];

    // two AAC tracks, the second PID's units around the first's
    fn two_aac_tracks() -> Vec<u8> {
        let mut stream = Vec::new();
        stream.extend_from_slice(&pat(0x1000));
        stream.extend_from_slice(&pmt2(
            FIRST_AAC_PID,
            STREAM_TYPE_AAC,
            SECOND_AAC_PID,
            STREAM_TYPE_AAC,
        ));
        for (pid, unit) in [
            (SECOND_AAC_PID, SECOND_AAC),
            (FIRST_AAC_PID, FIRST_AAC),
            (SECOND_AAC_PID, SECOND_AAC),
        ] {
            stream.extend_from_slice(&ts_packet(pid, true, &pes_id(0xC0, &unit)));
        }
        stream
    }

    #[tokio::test]
    async fn forwards_only_the_first_pmt_stream_of_the_selected_codec() {
        let mut d = TsDemux::new().with_stream(TsStream::Aac);
        d.configure_pipeline(&input_caps()).unwrap();
        let mut sink = CaptureSink::default();
        run_demux(&mut d, &two_aac_tracks(), &mut sink).await;
        assert_eq!(sink.frames, alloc::vec![FIRST_AAC.to_vec()]);
    }

    #[tokio::test]
    async fn pid_picks_a_track_among_those_of_the_selected_codec() {
        let mut d = TsDemux::new().with_stream(TsStream::Aac);
        d.set_property("pid", PropValue::Uint(SECOND_AAC_PID.into()))
            .unwrap();
        d.configure_pipeline(&input_caps()).unwrap();
        let mut sink = CaptureSink::default();
        run_demux(&mut d, &two_aac_tracks(), &mut sink).await;
        assert_eq!(sink.frames, alloc::vec![SECOND_AAC.to_vec(); 2]);
    }

    // a PMT naming each (stream_type, pid, descriptors) entry in order, PCR on the first
    fn pmt_entries(entries: &[(u8, u16, &[u8])]) -> Vec<u8> {
        let pcr_pid = entries[0].1;
        let mut body = alloc::vec![
            0x00,
            0x01,
            0xC1,
            0x00,
            0x00,
            0xE0 | (pcr_pid >> 8) as u8 & 0x1F,
            pcr_pid as u8,
            0xF0,
            0x00,
        ];
        for (stream_type, pid, descriptors) in entries {
            body.extend_from_slice(&[
                *stream_type,
                0xE0 | (pid >> 8) as u8 & 0x1F,
                *pid as u8,
                0xF0 | (descriptors.len() >> 8) as u8 & 0x0F,
                descriptors.len() as u8,
            ]);
            body.extend_from_slice(descriptors);
        }
        psi(0x1000, 0x02, &body)
    }

    #[tokio::test]
    async fn a_pid_listed_twice_in_the_pmt_takes_its_last_entry() {
        const AC3_PID: u16 = 0x0134;
        const PRIVATE_STREAM_1: u8 = 0xBD;
        let ac3_descriptor = [crate::mpegts::DESC_TAG_AC3, 1, 0x00];
        let syncframe = [0x0Bu8, 0x77, 0x12, 0x34];
        let mut stream = Vec::new();
        stream.extend_from_slice(&pat(0x1000));
        stream.extend_from_slice(&pmt_entries(&[
            (STREAM_TYPE_MPEG2_AUDIO, AC3_PID, &[]),
            (STREAM_TYPE_PRIVATE_PES, AC3_PID, &ac3_descriptor),
        ]));
        stream.extend_from_slice(&ts_packet(
            AC3_PID,
            true,
            &pes_id(PRIVATE_STREAM_1, &syncframe),
        ));

        let mut probe = TsDemuxer::new();
        for packet in stream.chunks(TS_PACKET_LEN) {
            probe.push_packet(packet);
        }
        let streams: Vec<TsStream> = forwardable_streams(&mut probe)
            .iter()
            .map(|info| info.stream)
            .collect();
        assert_eq!(streams, alloc::vec![TsStream::Ac3]);

        let mut d = TsDemux::new().with_stream(TsStream::Ac3);
        d.configure_pipeline(&input_caps()).unwrap();
        let mut sink = CaptureSink::default();
        run_demux(&mut d, &stream, &mut sink).await;
        assert_eq!(sink.frames, alloc::vec![syncframe.to_vec()]);
    }

    /// The one-port demuxer posts the tags a transport stream carries:
    /// the SDT service text once as a `Tag`, each PMT language as a `StreamTag` on
    /// its stream's collection id. Re-feeding the same stream posts nothing more.
    #[tokio::test]
    async fn posts_service_and_language_tags_on_the_bus() {
        use g2g_core::Bus;
        let mut m = crate::mpegts::TsMuxer::with_streams(&[STREAM_TYPE_H264]);
        m.set_service("News One", "G2G Broadcasting");
        m.set_stream_language(0, "eng");
        let mut stream = m.push_au(&[0, 0, 0, 1, 0x65, 0x11], Some(900_000), None, false);
        stream.extend(m.push_au(&[0, 0, 0, 1, 0x41, 0x22], Some(903_000), None, false));

        let (bus, handle) = Bus::new(16);
        let mut d = TsDemux::new().with_bus(handle);
        d.configure_pipeline(&input_caps()).unwrap();
        let mut sink = CaptureSink::default();
        run_demux(&mut d, &stream, &mut sink).await;
        run_demux(&mut d, &stream, &mut sink).await; // the tables repeat, the tags do not
        assert_eq!(sink.frames.len(), 4, "both feeds' AUs still come out");

        let (mut global, mut per_stream) = (Vec::new(), Vec::new());
        while let Some(msg) = bus.try_recv() {
            match msg {
                BusMessage::Tag { tags, program } => global.push((program, tags)),
                BusMessage::StreamTag { stream_id, tags } => per_stream.push((stream_id, tags)),
                _ => {}
            }
        }
        assert_eq!(global.len(), 1, "the service text posts once");
        assert_eq!(global[0].0, Some(1), "scoped to the program it names");
        assert_eq!(
            global[0].1.tags(),
            &[
                Tag::Title("News One".into()),
                Tag::Other {
                    key: TAG_KEY_SERVICE_PROVIDER.into(),
                    value: "G2G Broadcasting".into()
                }
            ]
        );
        assert_eq!(per_stream.len(), 1, "one language, posted once");
        assert_eq!(per_stream[0].0, "mpegts-pid-256");
        assert_eq!(per_stream[0].1.tags(), &[Tag::Language("eng".into())]);
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
        video.configure_pipeline(&input_caps()).unwrap();
        let mut vsink = CaptureSink::default();
        run_demux(&mut video, &stream, &mut vsink).await;
        assert_eq!(
            vsink.frames,
            alloc::vec![v0.to_vec(), v1.to_vec()],
            "video only"
        );

        // stream=aac selects AAC: only the two audio AUs come out (ADTS payload).
        let mut audio = TsDemux::new().with_stream(TsStream::Aac);
        audio.configure_pipeline(&input_caps()).unwrap();
        let mut asink = CaptureSink::default();
        run_demux(&mut audio, &stream, &mut asink).await;
        assert_eq!(
            asink.frames,
            alloc::vec![a0.to_vec(), a1.to_vec()],
            "audio only"
        );
    }

    #[test]
    fn program_number_property_round_trips() {
        let mut d = TsDemux::new();
        // The default is -1: the first program in PAT order.
        assert_eq!(d.get_property("program-number"), Some(PropValue::Int(-1)));
        d.set_property("program-number", PropValue::Int(2)).unwrap();
        assert_eq!(d.get_property("program-number"), Some(PropValue::Int(2)));
        // -1 resets to the first-program default.
        d.set_property("program-number", PropValue::Int(-1))
            .unwrap();
        assert_eq!(d.get_property("program-number"), Some(PropValue::Int(-1)));
        // A negative other than -1 is rejected (leaving the value unchanged).
        assert_eq!(
            d.set_property("program-number", PropValue::Int(-2)),
            Err(PropError::Value)
        );
        assert_eq!(d.get_property("program-number"), Some(PropValue::Int(-1)));
        // The builder round-trips too.
        let d2 = TsDemux::new().with_program_number(Some(5));
        assert_eq!(d2.get_property("program-number"), Some(PropValue::Int(5)));
    }

    #[test]
    fn output_caps_track_the_selection() {
        assert!(matches!(
            output_caps(TsStream::H264),
            Caps::CompressedVideo {
                codec: VideoCodec::H264,
                ..
            }
        ));
        assert!(matches!(
            output_caps(TsStream::H265),
            Caps::CompressedVideo {
                codec: VideoCodec::H265,
                ..
            }
        ));
        assert!(matches!(
            output_caps(TsStream::Aac),
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

        d.set_property("stream", PropValue::Str("mp3".into()))
            .unwrap();
        assert_eq!(d.get_property("stream"), Some(PropValue::Str("mp2".into())));

        d.set_property("stream", PropValue::Str("aac".into()))
            .unwrap();
        assert_eq!(d.get_property("stream"), Some(PropValue::Str("aac".into())));

        // An unsupported codec name is rejected (leaving the selection unchanged).
        assert_eq!(
            d.set_property("stream", PropValue::Str("vp9".into())),
            Err(PropError::Value)
        );
        assert_eq!(d.get_property("stream"), Some(PropValue::Str("aac".into())));

        // DerivedOutput now maps the TS byte stream to AAC audio.
        let CapsConstraint::DerivedOutput(f) = d.caps_constraint_as_transform() else {
            panic!("expected DerivedOutput");
        };
        let out = f(&input_caps());
        assert!(matches!(
            out.alternatives(),
            [Caps::Audio {
                format: AudioFormat::Aac,
                ..
            }]
        ));
    }

    fn probed_video_caps(stream: &[u8]) -> Caps {
        let mut demux = TsDemuxer::new();
        TsPacketSplitter::new().push(stream, &mut demux);
        probe_streams(&mut demux)
            .forwardable
            .into_iter()
            .find(|info| info.video)
            .expect("a video stream")
            .caps
    }

    #[test]
    fn probing_reads_the_video_picture_size_from_its_sps() {
        const H264_FIXTURE_SIZE: (u32, u32) = (160, 120);
        const H265_FIXTURE_SIZE: (u32, u32) = (3840, 2160);
        for (stream, (width, height)) in [
            (
                include_bytes!("../tests/fixtures/av_h264_aac44100.ts").as_slice(),
                H264_FIXTURE_SIZE,
            ),
            (
                include_bytes!("../tests/fixtures/h265_3840x2160.ts").as_slice(),
                H265_FIXTURE_SIZE,
            ),
        ] {
            let caps = probed_video_caps(stream);
            let (probed_width, probed_height, _) = caps.dims().expect("video caps");
            assert_eq!(
                (probed_width, probed_height),
                (&Dim::Fixed(width), &Dim::Fixed(height))
            );
        }
    }

    #[test]
    fn probing_video_without_an_sps_keeps_the_placeholder_size() {
        let mut muxer = crate::mpegts::TsMuxer::with_streams(&[STREAM_TYPE_H264]);
        let mut stream = muxer.push_au(&[0, 0, 0, 1, 0x65, 0x11], None, None, true);
        stream.extend(muxer.push_au(&[0, 0, 0, 1, 0x01, 0x22], None, None, false));
        assert_eq!(probed_video_caps(&stream), output_caps(TsStream::H264));
    }
}
