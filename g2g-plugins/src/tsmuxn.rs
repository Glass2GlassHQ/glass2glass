//! MPEG-TS multiplexer element: N elementary streams in (e.g. H.264
//! video + AAC audio, or one of them alone), one MPEG-TS byte stream out, the
//! everyday live-streaming container case. Wraps the pure [`crate::mpegts::TsMuxer`],
//! the inverse of [`crate::tsdemux::TsDemux`].
//!
//! A [`MultiInputElement`]: each input pad accepts one elementary stream
//! (`Caps::CompressedVideo{H264|H265|Av1}`, `Caps::Audio{Aac|Opus}`, `Caps::Klv`,
//! or the `Caps::SubPicture{DvbSub}` subtitle pad), and the access
//! units are interleaved into one multiplex by decode time (the DTS when the
//! frame carries one, else the PTS) before being written to their per-stream
//! PIDs. Time-ordering reuses the
//! [`InputAggregator::take_earliest_by`](g2g_core::InputAggregator::take_earliest_by)
//! merge (release the globally earliest AU once every contributing input has one
//! queued), so the muxed TS is in decode order across streams the way a decoder
//! expects, and no stream's data trails a PCR taken from another. The PMT (built once all inputs are configured) names every stream of
//! its program.
//!
//! An AAC stream that opens on a raw AudioSpecificConfig (ER AAC, ELD, USAC,
//! which ADTS cannot signal) goes out as LOAS/LATM under stream type 0x11, with
//! that config in every frame. A DVB subtitle pad is a private PES whose PMT
//! entry gets the `subtitling_descriptor`, declaring the pages from the page-id
//! config blob the stream leads with, else from the `dvbsub-page-id` property.
//!
//! Scope: one program by default, several via `prog-map`, which assigns a
//! program number per input pad; each program gets its own PMT and its own PCR on
//! its first stream's PID, on the `pcr-interval` cadence (see
//! [`crate::mpegts::TsMuxer`]). Reachable from
//! the `gst-launch` fan-in syntax: registered only as the `mpegtsmux` muxer
//! in [`default_registry`](crate::registry::default_registry), so
//! `... ! h264parse ! mpegtsmux ! filesink` and `v.! m.  a.! m.  mpegtsmux name=m ! sink`
//! both build this element. Also runs programmatically through `run_muxer_sink`
//! / a `run_graph` muxer node.

use core::future::Future;
use core::pin::Pin;

use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::time::ns_to_ticks;
use g2g_core::{
    g2g_error, resolve_tags, AudioFormat, ByteStreamEncoding, Caps, CapsConstraint, CapsSet,
    ConfigureOutcome, ElementMetadata, FrameTiming, G2gError, InputAggregator, MemoryDomain,
    MultiInputElement, OutputSink, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
    SubPictureFormat, Tag, TagList, VideoCodec,
};

use crate::aacparse::{
    is_leading_aac_config, is_raw_aac_config, is_usac_config, loas_from_asc,
    muxer_audio_specific_config, opens_loas_frame,
};
use crate::dvbsub::{parse_page_ids, PageIds, DEFAULT_PAGE_ID, DEFAULT_SUBTITLING_TYPE};
use crate::mpegts::{
    TsMuxer, DEFAULT_ORIGINAL_NETWORK_ID, DEFAULT_PCR_INTERVAL_90KHZ, DEFAULT_TABLE_INTERVAL_90KHZ,
    DEFAULT_TRANSPORT_STREAM_ID, MAX_TABLES_VERSION, MUX_ES_PID, MUX_PID_RANGE, MUX_PMT_PID,
    PES_CLOCK_HZ, STREAM_TYPE_AAC, STREAM_TYPE_H264, STREAM_TYPE_H265, STREAM_TYPE_METADATA_PES,
    STREAM_TYPE_PRIVATE_PES, TAG_KEY_SERVICE_NAME, TAG_KEY_SERVICE_PROVIDER,
};
use crate::mux::{audio_track_params, refused_input};
use crate::opusparse::{is_opus_config, parse_opus_head, OPUS_RATE_HZ};
use g2g_core::log::{short_type_name, Target};

/// The language a `subtitling_descriptor` declares when the stream's tags name
/// none, the ISO 639 code for "undetermined" (what ffmpeg writes).
const UNDETERMINED_LANGUAGE: &str = "und";

/// What a DVB subtitle pad's PMT `subtitling_descriptor` declares: the
/// page ids and subtitling type from the config blob the stream carries in band
/// ahead of its first display set, falling back to the `dvbsub-page-id` property
/// for a stream that sends none.
#[derive(Debug, Clone, Copy)]
struct DvbSubDecl {
    /// The `dvbsub-page-id` property: the composition and ancillary page a
    /// stream with no config blob is declared on.
    page_id: u16,
    /// The ids and `subtitling_type` a config blob named, once one has arrived.
    from_stream: Option<(PageIds, u8)>,
}

impl DvbSubDecl {
    /// A pad declared on `page_id` until its stream says otherwise.
    fn on_page(page_id: u16) -> Self {
        Self {
            page_id,
            from_stream: None,
        }
    }

    /// Adopt a page-id config blob. `false` when the bytes are not one, so they
    /// are a display set and belong in the multiplex.
    fn adopt(&mut self, data: &[u8]) -> bool {
        let Some(ids) = parse_page_ids(data) else {
            return false;
        };
        let subtitling_type = data.get(4).copied().unwrap_or(DEFAULT_SUBTITLING_TYPE);
        self.from_stream = Some((ids, subtitling_type));
        true
    }

    fn ids(&self) -> PageIds {
        self.from_stream.map(|(ids, _)| ids).unwrap_or(PageIds {
            composition: self.page_id,
            ancillary: self.page_id,
        })
    }

    fn subtitling_type(&self) -> u8 {
        self.from_stream
            .map(|(_, t)| t)
            .unwrap_or(DEFAULT_SUBTITLING_TYPE)
    }
}

const PROPERTIES: [PropertySpec; 13] = [
    PropertySpec::new(
        "pat-interval",
        PropKind::Uint,
        "PAT/PMT re-emission interval, ticks of the 90kHz clock (0 = once)",
    )
    .with_default("9000"),
    // The PAT and PMT are emitted as a pair, so this shares the cadence.
    PropertySpec::new(
        "pmt-interval",
        PropKind::Uint,
        "alias of pat-interval (the tables are emitted together)",
    )
    .with_default("9000"),
    PropertySpec::new(
        "si-interval",
        PropKind::Uint,
        "SDT re-emission interval, ticks of the 90kHz clock (0 = once)",
    )
    .with_default("9000"),
    PropertySpec::new(
        "pcr-interval",
        PropKind::Uint,
        "PCR insertion interval, ticks of the 90kHz clock",
    )
    .with_default("3600"),
    PropertySpec::new(
        "bitrate",
        PropKind::Uint,
        "constant output bitrate in bits per second, padded with null packets (0 = variable bitrate)",
    )
    .with_default("0"),
    PropertySpec::new(
        "transport-stream-id",
        PropKind::Uint,
        "transport_stream_id written in the PAT and SDT",
    )
    .with_default("1")
    .with_range("0", "65535"),
    PropertySpec::new(
        "original-network-id",
        PropKind::Uint,
        "original_network_id written in the SDT",
    )
    .with_default("1")
    .with_range("0", "65535"),
    PropertySpec::new(
        "start-pid",
        PropKind::Uint,
        "PID of the first elementary stream, the rest follow in order",
    )
    .with_default("256")
    .with_range("16", "8190"),
    PropertySpec::new(
        "pmt-start-pid",
        PropKind::Uint,
        "PID of the first program's PMT, the rest follow in order",
    )
    .with_default("4096")
    .with_range("16", "8190"),
    PropertySpec::new(
        "tables-version",
        PropKind::Uint,
        "version_number written in the PAT, PMT and SDT",
    )
    .with_default("0")
    .with_range("0", "31"),
    PropertySpec::new(
        "klv-sync",
        PropKind::Bool,
        "carry KLV metadata as synchronous metadata-in-PES (stream_type 0x15) instead of asynchronous private PES",
    )
    .with_default("false"),
    PropertySpec::new(
        "dvbsub-page-id",
        PropKind::Uint,
        "composition and ancillary page a DVB subtitle input is declared on when the stream carries no page-id config",
    )
    .with_default("1"),
    // GStreamer mpegtsmux takes a pad-name -> program structure for prog-map, g2g properties are scalar
    PropertySpec::new(
        "prog-map",
        PropKind::Str,
        "program number per input, comma separated in pad order (e.g. 1,1,2; default all in program 1)",
    )
    .with_default("1"),
];

#[derive(Debug, Clone, Copy)]
struct TsMuxSettings {
    table_interval_90khz: u64,
    si_interval_90khz: u64,
    pcr_interval_90khz: u64,
    bitrate: u64,
    transport_stream_id: u16,
    original_network_id: u16,
    start_pid: u16,
    pmt_start_pid: u16,
    tables_version: u8,
}

impl Default for TsMuxSettings {
    fn default() -> Self {
        Self {
            table_interval_90khz: DEFAULT_TABLE_INTERVAL_90KHZ,
            si_interval_90khz: DEFAULT_TABLE_INTERVAL_90KHZ,
            pcr_interval_90khz: DEFAULT_PCR_INTERVAL_90KHZ,
            bitrate: 0,
            transport_stream_id: DEFAULT_TRANSPORT_STREAM_ID,
            original_network_id: DEFAULT_ORIGINAL_NETWORK_ID,
            start_pid: MUX_ES_PID,
            pmt_start_pid: MUX_PMT_PID,
            tables_version: 0,
        }
    }
}

impl TsMuxSettings {
    // None when the name is not one of these
    fn set_property(
        &mut self,
        name: &str,
        value: &PropValue,
        mux: Option<&mut TsMuxer>,
    ) -> Option<Result<(), PropError>> {
        let result = match name {
            "pat-interval" | "pmt-interval" | "si-interval" | "pcr-interval" => {
                self.set_cadence(name, value, mux)
            }
            "bitrate"
            | "transport-stream-id"
            | "original-network-id"
            | "start-pid"
            | "pmt-start-pid"
            | "tables-version" => {
                // baked into the stream when the muxer is built
                if mux.is_some() {
                    return Some(Err(PropError::ReadOnly));
                }
                self.set_identity(name, value)
            }
            _ => return None,
        };
        Some(result)
    }

    fn set_cadence(
        &mut self,
        name: &str,
        value: &PropValue,
        mux: Option<&mut TsMuxer>,
    ) -> Result<(), PropError> {
        let ticks = value.as_uint().ok_or(PropError::Type)?;
        match name {
            "si-interval" => self.si_interval_90khz = ticks,
            "pcr-interval" => self.pcr_interval_90khz = ticks,
            _ => self.table_interval_90khz = ticks,
        }
        if let Some(mux) = mux {
            self.apply_cadences(mux);
        }
        Ok(())
    }

    fn set_identity(&mut self, name: &str, value: &PropValue) -> Result<(), PropError> {
        let number = value.as_uint().ok_or(PropError::Type)?;
        if name == "bitrate" {
            self.bitrate = number;
            return Ok(());
        }
        let field = u16::try_from(number).map_err(|_| PropError::Value)?;
        let pid_ok = MUX_PID_RANGE.contains(&field);
        match name {
            "transport-stream-id" => self.transport_stream_id = field,
            "original-network-id" => self.original_network_id = field,
            "start-pid" if pid_ok => self.start_pid = field,
            "pmt-start-pid" if pid_ok => self.pmt_start_pid = field,
            "tables-version" if number <= u64::from(MAX_TABLES_VERSION) => {
                self.tables_version = field as u8
            }
            _ => return Err(PropError::Value),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        let value = match name {
            "pat-interval" | "pmt-interval" => self.table_interval_90khz,
            "si-interval" => self.si_interval_90khz,
            "pcr-interval" => self.pcr_interval_90khz,
            "bitrate" => self.bitrate,
            "transport-stream-id" => u64::from(self.transport_stream_id),
            "original-network-id" => u64::from(self.original_network_id),
            "start-pid" => u64::from(self.start_pid),
            "pmt-start-pid" => u64::from(self.pmt_start_pid),
            "tables-version" => u64::from(self.tables_version),
            _ => return None,
        };
        Some(PropValue::Uint(value))
    }

    fn apply_cadences(&self, mux: &mut TsMuxer) {
        mux.set_table_interval_90khz(self.table_interval_90khz);
        mux.set_si_interval_90khz(self.si_interval_90khz);
        mux.set_pcr_interval_90khz(self.pcr_interval_90khz);
    }

    fn apply(&self, mux: &mut TsMuxer) -> Result<(), G2gError> {
        self.apply_cadences(mux);
        mux.set_bitrate(self.bitrate);
        mux.set_transport_stream_id(self.transport_stream_id);
        mux.set_original_network_id(self.original_network_id);
        mux.set_tables_version(self.tables_version)?;
        mux.set_pid_bases(self.start_pid, self.pmt_start_pid)
    }
}

// how an AAC pad's frames enter the multiplex, fixed by the pad's first frame
#[derive(Debug, Clone, Default)]
enum AacCarriage {
    #[default]
    Undecided,
    // ADTS frames, multiplexed as they arrive
    Adts,
    // raw access units behind this config, which ADTS cannot signal, multiplexed as LOAS
    Loas(Vec<u8>),
    // LOAS frames, multiplexed as they arrive
    SourceLoas,
}

impl AacCarriage {
    // true for a raw stream's config frame, which rides every LOAS frame instead of the multiplex
    fn consume_config(&mut self, frame: &[u8], muxer: &'static str) -> Result<bool, G2gError> {
        match self {
            Self::Loas(config) => Ok(is_leading_aac_config(config.as_slice(), frame)),
            Self::Adts | Self::SourceLoas => Ok(false),
            Self::Undecided if opens_loas_frame(frame) => {
                *self = Self::SourceLoas;
                Ok(false)
            }
            Self::Undecided => {
                let raw_config =
                    muxer_audio_specific_config(frame, muxer)?.filter(|_| is_raw_aac_config(frame));
                // ADTS has no profile for USAC, and LATM decoders (ffmpeg's included) do not read it
                if raw_config.as_deref().is_some_and(is_usac_config) {
                    g2g_error!(Target::category(muxer), "cannot carry USAC");
                    return Err(G2gError::CapsMismatch);
                }
                let leads_raw_stream = raw_config.is_some();
                *self = raw_config.map_or(Self::Adts, Self::Loas);
                Ok(leads_raw_stream)
            }
        }
    }

    fn is_loas(&self) -> bool {
        matches!(self, Self::Loas(_) | Self::SourceLoas)
    }

    // the bytes one frame puts on the pad's PID
    fn access_unit<'a>(
        &self,
        frame: &'a [u8],
        muxer: &'static str,
    ) -> Result<Cow<'a, [u8]>, G2gError> {
        let Self::Loas(config) = self else {
            return Ok(Cow::Borrowed(frame));
        };
        loas_from_asc(config, frame).map(Cow::Owned).ok_or_else(|| {
            g2g_error!(
                Target::category(muxer),
                "AAC access unit of {} bytes exceeds the LOAS frame length",
                frame.len()
            );
            G2gError::CapsMismatch
        })
    }
}

/// The PMT `stream_type` for an input caps, or `None` if unsupported. KLV metadata
/// rides a private PES (0x06) with the 'KLVA' registration descriptor
/// (asynchronous KLV, MISB ST 1402 / STANAG 4609), or, with `klv_sync`,
/// metadata-in-PES (0x15) with a metadata descriptor (synchronous KLV).
fn stream_type_for(caps: &Caps, klv_sync: bool) -> Option<u8> {
    match caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            ..
        } => Some(STREAM_TYPE_H264),
        Caps::CompressedVideo {
            codec: VideoCodec::H265,
            ..
        } => Some(STREAM_TYPE_H265),
        // AV1 has no stream_type of its own: it rides a private PES told apart by
        // the AV1 registration descriptor the muxer writes for it (M1049).
        Caps::CompressedVideo {
            codec: VideoCodec::Av1,
            ..
        } => Some(STREAM_TYPE_PRIVATE_PES),
        Caps::Audio {
            format: AudioFormat::Aac,
            ..
        } => Some(STREAM_TYPE_AAC),
        // Opus has no stream_type of its own: it rides a private PES told apart
        // by the 'Opus' registration + extension descriptors the muxer writes.
        Caps::Audio {
            format: AudioFormat::Opus,
            ..
        } => Some(STREAM_TYPE_PRIVATE_PES),
        Caps::Klv => Some(if klv_sync {
            STREAM_TYPE_METADATA_PES
        } else {
            STREAM_TYPE_PRIVATE_PES
        }),
        // DVB subtitles ride a private PES too, told apart by the
        // `subtitling_descriptor` the muxer writes for them. VobSub has no TS
        // carriage (it is a DVD program-stream codec), so only DVB is muxable.
        Caps::SubPicture {
            format: SubPictureFormat::DvbSub,
        } => Some(STREAM_TYPE_PRIVATE_PES),
        _ => None,
    }
}

/// Whether an input caps is the AV1 pad, whose PMT entry needs the AV1
/// registration descriptor to be anything other than an unidentified private PES.
fn is_av1(caps: &Caps) -> bool {
    matches!(
        caps,
        Caps::CompressedVideo {
            codec: VideoCodec::Av1,
            ..
        }
    )
}

/// The channel count of an Opus input pad, whose PMT entry needs the Opus
/// descriptors and whose access units need the control header. `None` for
/// any other caps.
fn opus_channels(caps: &Caps) -> Option<u8> {
    match caps {
        Caps::Audio {
            format: AudioFormat::Opus,
            channels,
            ..
        } => Some(*channels),
        _ => None,
    }
}

/// Whether an input caps is the DVB subtitle pad, whose PMT entry needs the
/// `subtitling_descriptor` and whose access units are display sets.
fn is_dvbsub(caps: &Caps) -> bool {
    matches!(
        caps,
        Caps::SubPicture {
            format: SubPictureFormat::DvbSub
        }
    )
}

/// The SDT service text a tag list names: the service name from
/// [`Tag::Title`] or a `service_name` key, the provider from a `service_provider`
/// key (ffprobe's names for the two `service_descriptor` fields). `None` when the
/// list names neither, in which case no SDT is written.
///
/// TS standardizes only these two whole-service strings and a per-stream language,
/// with no free-form tag carrier, so every other tag is dropped rather than
/// smuggled into a private descriptor.
fn service_from_tags(tags: &TagList) -> Option<(String, String)> {
    let mut name: Option<&str> = None;
    let mut provider: Option<&str> = None;
    for tag in tags.tags() {
        match tag {
            Tag::Title(v) => name = Some(v),
            Tag::Other { key, value } if key.eq_ignore_ascii_case(TAG_KEY_SERVICE_NAME) => {
                name = Some(value)
            }
            Tag::Other { key, value } if key.eq_ignore_ascii_case(TAG_KEY_SERVICE_PROVIDER) => {
                provider = Some(value)
            }
            _ => {}
        }
    }
    (name.is_some() || provider.is_some()).then(|| {
        (
            String::from(name.unwrap_or_default()),
            String::from(provider.unwrap_or_default()),
        )
    })
}

/// The language a tag list declares, for a stream's
/// `ISO_639_language_descriptor`. The only per-stream tag TS carries.
fn language_from_tags(tags: &TagList) -> Option<&str> {
    tags.tags().iter().find_map(|tag| match tag {
        Tag::Language(v) => Some(v.as_str()),
        _ => None,
    })
}

/// Muxes N elementary streams into one MPEG-TS byte stream, in decode order.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::tsmuxn::TsMux;
///
/// // two elementary streams in one program, PAT/PMT every 100 ms.
/// let mux = TsMux::new(2)
///     .with_program_numbers(&[1, 1])
///     .with_table_interval_90khz(9000);
/// assert_eq!(mux.emitted(), 0);
/// ```
#[derive(Debug)]
pub struct TsMux {
    inputs: usize,
    /// PMT stream type per input pad, learned at configure; the muxer is built
    /// once all are known.
    stream_types: Vec<Option<u8>>,
    /// Built lazily once every input is configured (the PMT needs all streams).
    mux: Option<TsMuxer>,
    /// Per-input AU buffer; releases the globally earliest-decode-time AU.
    agg: InputAggregator<Frame>,
    emitted: u64,
    /// The cadences, bitrate and identity knobs, applied to the inner `TsMuxer`
    /// when it is built. The PAT and PMT are emitted together, so `pat-interval`
    /// and `pmt-interval` share one cadence.
    settings: TsMuxSettings,
    /// Program number per input pad, in pad order (default: every input in
    /// program 1). Zipped with the learned stream types to build the muxer.
    program_numbers: Vec<u16>,
    /// Carry a `Caps::Klv` input as synchronous KLV (metadata-in-PES 0x15)
    /// instead of the default asynchronous private PES (0x06 + 'KLVA').
    klv_sync: bool,
    /// Whole-service metadata, written to the SDT: the service every
    /// program without its own is named with.
    tags: TagList,
    /// Per-program service metadata, keyed by program number: that
    /// program's own SDT entry text.
    program_tags: Vec<(u16, TagList)>,
    /// Per-input metadata: its `Tag::Language` becomes that stream's
    /// `ISO_639_language_descriptor`. One (possibly empty) list per input pad.
    track_tags: Vec<TagList>,
    /// Per-pad DVB subtitle declaration, `None` for a pad that carries
    /// something else. Its page ids become that stream's `subtitling_descriptor`.
    dvbsub: Vec<Option<DvbSubDecl>>,
    /// The `dvbsub-page-id` property, the page every DVB subtitle pad that
    /// carries no page-id config blob is declared on.
    dvbsub_page_id: u16,
    /// Which pads carry AV1, so their PMT entries get the AV1
    /// registration descriptor rather than the private PES default of KLV.
    av1: Vec<bool>,
    /// Per input: the channel count of an Opus pad, which declares the Opus
    /// descriptors and wraps its access units in the control header.
    opus: Vec<Option<u8>>,
    // per input, how an AAC pad's frames are multiplexed, `None` for any other pad
    aac: Vec<Option<AacCarriage>>,
}

impl TsMux {
    /// A muxer with `inputs` input pads. Each pad's stream type is determined
    /// from its negotiated caps; the order of inputs is the order of streams in
    /// the PMT (and their PID assignment).
    pub fn new(inputs: usize) -> Self {
        assert!(inputs > 0, "TsMux needs at least one input");
        Self {
            inputs,
            stream_types: alloc::vec![None; inputs],
            mux: None,
            agg: InputAggregator::new(inputs),
            emitted: 0,
            settings: TsMuxSettings::default(),
            program_numbers: alloc::vec![1; inputs],
            klv_sync: false,
            tags: TagList::new(),
            program_tags: Vec::new(),
            track_tags: alloc::vec![TagList::new(); inputs],
            dvbsub: alloc::vec![None; inputs],
            dvbsub_page_id: DEFAULT_PAGE_ID,
            av1: alloc::vec![false; inputs],
            opus: alloc::vec![None; inputs],
            aac: alloc::vec![None; inputs],
        }
    }

    /// Attach whole-service metadata: the service name ([`g2g_core::Tag::Title`]
    /// or a `service_name` key) and provider (a `service_provider` key) written to
    /// the SDT, naming every program that has no service of its own
    /// ([`with_program_tags`](Self::with_program_tags)). TS defines no free-form tag
    /// carrier, so any other tag here is dropped.
    pub fn with_tags(mut self, tags: TagList) -> Self {
        self.tags = tags;
        self
    }

    /// Attach metadata scoped to one program of a multi-program mux: its
    /// service name / provider become that program's own SDT entry, so each program
    /// carries distinct service text instead of repeating the
    /// [`with_tags`](Self::with_tags) default. The program is named by its number in
    /// the [`with_program_numbers`](Self::with_program_numbers) / `prog-map` layout.
    /// A [`g2g_core::Tag::Language`] here is the default for that program's streams,
    /// between the global list and a stream's own
    /// [`with_track_tags`](Self::with_track_tags); TS carries nothing else, so the
    /// rest of the list is dropped.
    ///
    /// Panics when no input is in `program`: the SDT would silently omit the service
    /// otherwise. Set the program map first (every input is in program 1 by
    /// default).
    pub fn with_program_tags(mut self, program: u16, tags: TagList) -> Self {
        assert!(
            self.program_numbers.contains(&program),
            "TsMux has no input in program {program}"
        );
        self.program_tags.retain(|(n, _)| *n != program);
        self.program_tags.push((program, tags));
        self
    }

    /// Attach metadata scoped to one input pad's elementary stream: its
    /// [`g2g_core::Tag::Language`] becomes an `ISO_639_language_descriptor` in that
    /// stream's PMT entry, so a reader reports the language on that stream. Nothing
    /// else rides a TS elementary stream, so the rest of the list is dropped.
    /// Out-of-range inputs are ignored.
    ///
    /// A language set on the stream's program ([`with_program_tags`](Self::with_program_tags))
    /// or globally by [`with_tags`](Self::with_tags) applies to every stream that
    /// does not name its own (`g2g_core::resolve_tags`).
    pub fn with_track_tags(mut self, input: usize, tags: TagList) -> Self {
        if input < self.inputs {
            self.track_tags[input] = tags;
        }
        self
    }

    /// Assign a program number to each input pad, in pad order: inputs
    /// sharing a number land in one program, and each program gets its own PMT and
    /// PCR. `numbers` must have one entry per input.
    pub fn with_program_numbers(mut self, numbers: &[u16]) -> Self {
        assert!(
            numbers.len() == self.inputs,
            "TsMux needs one program number per input"
        );
        assert!(
            self.program_tags.iter().all(|(n, _)| numbers.contains(n)),
            "a program with per-program tags is missing from the new program map"
        );
        self.program_numbers = numbers.to_vec();
        self
    }

    /// Set the PAT/PMT re-emission interval in 90 kHz ticks (`0` = once up
    /// front, default 9000).
    pub fn with_table_interval_90khz(mut self, ticks: u64) -> Self {
        self.settings.table_interval_90khz = ticks;
        self
    }

    /// Set the PCR insertion interval in 90 kHz ticks (default 3600).
    pub fn with_pcr_interval(mut self, ticks: u64) -> Self {
        self.settings.pcr_interval_90khz = ticks;
        self
    }

    /// Carry a `Caps::Klv` input as synchronous KLV: metadata-in-PES
    /// (`stream_type` 0x15) rather than the default asynchronous private PES.
    pub fn with_klv_sync(mut self, sync: bool) -> Self {
        self.klv_sync = sync;
        self
    }

    /// The composition and ancillary page a DVB subtitle input is declared on in
    /// the PMT `subtitling_descriptor`, for a stream that carries no page-id
    /// config blob of its own. A blob wins over this.
    pub fn with_dvbsub_page_id(mut self, page_id: u16) -> Self {
        self.dvbsub_page_id = page_id;
        for decl in self.dvbsub.iter_mut().flatten() {
            decl.page_id = page_id;
        }
        self
    }

    /// Count of TS byte frames emitted.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    /// The tags attached to input `i`'s program, if that program has any.
    fn tags_of_program(&self, i: usize) -> Option<&TagList> {
        self.program_tags
            .iter()
            .find(|(n, _)| *n == self.program_numbers[i])
            .map(|(_, t)| t)
    }

    /// The canonical `prog-map` value: the per-input program numbers joined by
    /// commas in pad order.
    fn prog_map_string(&self) -> String {
        let mut s = String::new();
        for (i, n) in self.program_numbers.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&alloc::format!("{n}"));
        }
        s
    }

    fn output_caps_value() -> Caps {
        Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        }
    }

    fn is_video(&self, input: usize) -> bool {
        self.av1[input]
            || matches!(
                self.stream_types[input],
                Some(STREAM_TYPE_H264) | Some(STREAM_TYPE_H265)
            )
    }

    /// The metadata that applies to input `i`: its own tags over its program's
    /// over the muxer's.
    fn effective_tags(&self, i: usize) -> TagList {
        let above = match self.tags_of_program(i) {
            Some(program) => resolve_tags(&self.tags, program),
            None => self.tags.clone(),
        };
        resolve_tags(&above, &self.track_tags[i])
    }

    /// Rewrite a DVB subtitle stream's `subtitling_descriptor` after its page ids
    /// changed. Only reaches the PMT while the muxer is still unbuilt or has yet
    /// to emit its tables, which is the case as long as the config blob leads the
    /// stream the way both demuxers send it.
    fn declare_subtitling(&mut self, input: usize) {
        let Some(decl) = self.dvbsub[input] else {
            return;
        };
        let tags = self.effective_tags(input);
        let language = String::from(language_from_tags(&tags).unwrap_or(UNDETERMINED_LANGUAGE));
        if let Some(mux) = self.mux.as_mut() {
            mux.set_stream_subtitling(input, &language, decl.subtitling_type(), decl.ids());
        }
    }
}

// 0 is the unset DTS sentinel
fn decode_time_ns(frame: &Frame) -> u64 {
    if frame.timing.dts_ns != 0 {
        frame.timing.dts_ns
    } else {
        frame.timing.pts_ns
    }
}

/// Parse a `prog-map` value: comma-separated program numbers, one per input in
/// pad order. `None` on an empty list or an entry that is not a `u16`.
fn parse_prog_map(s: &str) -> Option<Vec<u16>> {
    let numbers: Option<Vec<u16>> = s.split(',').map(|n| n.trim().parse::<u16>().ok()).collect();
    numbers.filter(|n| !n.is_empty())
}

impl MultiInputElement for TsMux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Reads host memory, so every pad takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn input_count(&self) -> usize {
        self.inputs
    }

    /// Named request pads: a container mux's inputs are caps-typed slots, so
    /// `video_%u` / `audio_%u` / `sink_%u` each claim the next positional slot (the
    /// track type is read from the input's caps, not its index), so a launch line
    /// can name the pads (`m.video_0` / `m.audio_0`) in any order.
    fn input_pad_index(
        &self,
        _req: &g2g_core::runtime::PadRequest,
        ordinal: usize,
    ) -> Option<usize> {
        (ordinal < self.inputs).then_some(ordinal)
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        if stream_type_for(upstream_caps, self.klv_sync).is_some() {
            Ok(upstream_caps.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        // Each pad forwards its stream verbatim (frames carry their own caps);
        // the per-pad stream type is pinned at `configure_pipeline`, which rejects
        // an unsupported caps. `AcceptsAny` is the native muxer-input shape (as in
        // `InterleaveMux`); the legacy intercept-narrowing path is skipped.
        CapsConstraint::AcceptsAny
    }

    fn caps_constraint_for_output(&self) -> Result<CapsConstraint<'_>, G2gError> {
        Ok(CapsConstraint::Produces(CapsSet::one(
            Self::output_caps_value(),
        )))
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        let stream_type = stream_type_for(absolute_caps, self.klv_sync)
            .ok_or_else(|| refused_input(short_type_name::<Self>(), absolute_caps))?;
        // a mid-stream caps change of the same stream keeps how its AAC frames are carried
        if self.stream_types[input] != Some(stream_type) {
            self.aac[input] = (stream_type == STREAM_TYPE_AAC).then(AacCarriage::default);
        }
        self.stream_types[input] = Some(stream_type);
        if is_dvbsub(absolute_caps) {
            self.dvbsub[input] = Some(DvbSubDecl::on_page(self.dvbsub_page_id));
        }
        self.av1[input] = is_av1(absolute_caps);
        self.opus[input] = opus_channels(absolute_caps);
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(Self::output_caps_value())
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "MPEG-TS muxer",
            "Codec/Muxer",
            "Muxes elementary streams into an MPEG transport stream",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        &PROPERTIES
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if let Some(result) = self.settings.set_property(name, &value, self.mux.as_mut()) {
            return result;
        }
        match name {
            "prog-map" => {
                // The program layout is baked into the PSI when the muxer is built,
                // so it can only be set before the first frame.
                if self.mux.is_some() {
                    return Err(PropError::ReadOnly);
                }
                self.program_numbers = parse_prog_map(value.as_str().ok_or(PropError::Type)?)
                    .filter(|n| n.len() == self.inputs)
                    // A map that drops a program whose service was set would write
                    // an SDT without that service; refuse it instead.
                    .filter(|n| self.program_tags.iter().all(|(p, _)| n.contains(p)))
                    .ok_or(PropError::Value)?;
                Ok(())
            }
            "klv-sync" => {
                // Like prog-map: the stream types are baked into the PMT when the
                // muxer is built, so this only takes effect before the first frame.
                if self.mux.is_some() {
                    return Err(PropError::ReadOnly);
                }
                self.klv_sync = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "dvbsub-page-id" => {
                let v = u16::try_from(value.as_uint().ok_or(PropError::Type)?)
                    .map_err(|_| PropError::Value)?;
                self.dvbsub_page_id = v;
                for decl in self.dvbsub.iter_mut().flatten() {
                    decl.page_id = v;
                }
                for input in 0..self.inputs {
                    self.declare_subtitling(input);
                }
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        if let Some(value) = self.settings.get_property(name) {
            return Some(value);
        }
        match name {
            "prog-map" => Some(PropValue::Str(self.prog_map_string())),
            "klv-sync" => Some(PropValue::Bool(self.klv_sync)),
            "dvbsub-page-id" => Some(PropValue::Uint(self.dvbsub_page_id as u64)),
            _ => None,
        }
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                // M204: buffer the AU; release the globally earliest below.
                PipelinePacket::DataFrame(frame) => {
                    // A DVB subtitle pad opens on its page-id config blob, which
                    // is what the `subtitling_descriptor` declares and never a
                    // display set to multiplex.
                    let config = frame
                        .domain
                        .as_system_slice()
                        .is_some_and(|s| self.dvbsub[input].as_mut().is_some_and(|d| d.adopt(s)));
                    if config {
                        self.declare_subtitling(input);
                        return Ok(());
                    }
                    // an Opus pad's in-band headers are config, the PMT descriptor states the count
                    if let Some(s) = frame
                        .domain
                        .as_system_slice()
                        .filter(|s| self.opus[input].is_some() && is_opus_config(s))
                    {
                        if let Some((channels, _)) = parse_opus_head(s) {
                            self.opus[input] = Some(channels);
                            if let Some(mux) = self.mux.as_mut() {
                                mux.set_stream_opus(input, channels);
                            }
                        }
                        return Ok(());
                    }
                    // a raw AAC pad opens on its config, which rides every LOAS frame instead
                    if let (Some(aac), Some(s)) =
                        (self.aac[input].as_mut(), frame.domain.as_system_slice())
                    {
                        let config = aac.consume_config(s, short_type_name::<Self>())?;
                        if let Some(mux) = self.mux.as_mut().filter(|_| aac.is_loas()) {
                            mux.set_stream_latm(input);
                        }
                        if config {
                            return Ok(());
                        }
                    }
                    // the PMT's Opus descriptor states the channel count
                    if let Some(channels) = self.opus[input] {
                        audio_track_params(
                            short_type_name::<Self>(),
                            AudioFormat::Opus,
                            channels,
                            OPUS_RATE_HZ,
                            &[],
                        )?;
                    }
                    self.agg.push(input, frame);
                }
                // M22: a per-input Eos lets the merge release AUs held waiting on
                // this input, and flush its tail; the runner emits the merged Eos.
                PipelinePacket::Eos => self.agg.mark_ended(input),
                // CapsChanged is consumed by the runner's muxer arm; geometry /
                // params do not change the TS framing.
                PipelinePacket::CapsChanged(_) => return Ok(()),
                // A per-input `Segment` maps that stream to running time; a muxed
                // container carries its own timestamps, so it is consumed rather
                // than forwarded into the byte stream.
                PipelinePacket::Segment(_) => return Ok(()),
                other => {
                    out.push(other).await?;
                    return Ok(());
                }
            }

            // The PMT needs every stream type, so build the muxer only once all
            // inputs are configured (always true by the first frame: the runner
            // negotiates every input pad before any process call).
            if self.mux.is_none() {
                if self.stream_types.iter().any(|s| s.is_none()) {
                    return Ok(());
                }
                let streams: Vec<(u16, u8)> = self
                    .program_numbers
                    .iter()
                    .zip(&self.stream_types)
                    .map(|(&program, s)| (program, s.expect("all set")))
                    .collect();
                let mut mux = TsMuxer::with_programs(&streams);
                self.settings.apply(&mut mux)?;
                if let Some((name, provider)) = service_from_tags(&self.tags) {
                    mux.set_service(&name, &provider);
                }
                for (program, tags) in &self.program_tags {
                    if let Some((name, provider)) = service_from_tags(tags) {
                        // The builder / property setter keep the map and these
                        // programs in step, so the program is always found.
                        let applied = mux.set_program_service(*program, &name, &provider);
                        debug_assert!(applied, "program {program} is in the program map");
                    }
                }
                for i in 0..self.inputs {
                    if self.av1[i] {
                        mux.set_stream_av1(i);
                    }
                    if let Some(channels) = self.opus[i] {
                        mux.set_stream_opus(i, channels);
                    }
                    if self.aac[i].as_ref().is_some_and(AacCarriage::is_loas) {
                        mux.set_stream_latm(i);
                    }
                    let effective = self.effective_tags(i);
                    let language = language_from_tags(&effective);
                    // A DVB subtitle stream's language rides its
                    // `subtitling_descriptor`, the way ffmpeg writes it, so it
                    // gets no separate language descriptor.
                    match self.dvbsub[i] {
                        Some(decl) => mux.set_stream_subtitling(
                            i,
                            language.unwrap_or(UNDETERMINED_LANGUAGE),
                            decl.subtitling_type(),
                            decl.ids(),
                        ),
                        None => {
                            if let Some(language) = language {
                                mux.set_stream_language(i, language);
                            }
                        }
                    }
                }
                self.mux = Some(mux);
            }

            let carries_video = (0..self.inputs).any(|i| self.is_video(i));
            // Drain every AU now safe to emit, in global decode order, writing each
            // to its stream's PID.
            while let Some((stream, frame)) = self.agg.take_earliest_by(decode_time_ns) {
                let muxer = g2g_core::log::short_type_name::<Self>();
                let slice = frame.domain.require_system_slice(muxer)?;
                let access_unit = match &self.aac[stream] {
                    Some(aac) => aac.access_unit(slice, muxer)?,
                    None => Cow::Borrowed(slice),
                };
                let pts_90khz = ns_to_ticks(frame.timing.pts_ns, PES_CLOCK_HZ);
                // A DTS rides the PES only for reordered video (dts_ns set and
                // distinct from the PTS); 0 is the unset sentinel, equal adds nothing.
                let dts_90khz = (frame.timing.dts_ns != 0
                    && frame.timing.dts_ns != frame.timing.pts_ns)
                    .then(|| ns_to_ticks(frame.timing.dts_ns, PES_CLOCK_HZ));
                let ts = self.mux.as_mut().expect("built above").push_au_on(
                    stream,
                    &access_unit,
                    Some(pts_90khz),
                    dts_90khz,
                    frame.timing.keyframe,
                );
                let out_frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(ts.into_boxed_slice())),
                    FrameTiming {
                        pts_ns: frame.timing.pts_ns,
                        // one output frame is one access unit's packets, so the
                        // AU's sync flag still describes it and a downstream
                        // segmenter cuts on it. Beside a video pad only the video
                        // counts: every audio AU is a sync sample, which would cut
                        // everywhere.
                        keyframe: frame.timing.keyframe
                            && (self.is_video(stream) || !carries_video),
                        ..FrameTiming::default()
                    },
                    self.emitted,
                );
                self.emitted += 1;
                out.push(PipelinePacket::DataFrame(out_frame)).await?;
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::{AudioFormat, Dim, PropValue, PushOutcome, Rate, VideoCodec};

    #[derive(Default)]
    struct CaptureSink {
        bytes: Vec<u8>,
        /// Per output frame, its PTS and sync flag.
        timings: Vec<(u64, bool)>,
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
                        self.timings.push((f.timing.pts_ns, f.timing.keyframe));
                        if let Some(s) = f.domain.as_system_slice() {
                            self.bytes.extend_from_slice(s);
                        }
                    }
                    PipelinePacket::Eos => self.eos = true,
                    _ => {}
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }

    // the access units a `TsDemuxer` recovers from `ts`
    fn demuxed_units(ts: &[u8]) -> Vec<Vec<u8>> {
        demuxer_of(ts)
            .take_units()
            .into_iter()
            .map(|u| u.data)
            .collect()
    }

    fn demuxer_of(ts: &[u8]) -> crate::mpegts::TsDemuxer {
        let mut demuxer = crate::mpegts::TsDemuxer::new();
        for packet in ts.chunks(crate::mpegts::TS_PACKET_LEN) {
            demuxer.push_packet(packet);
        }
        demuxer.flush();
        demuxer
    }

    fn h264_caps() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(320),
            height: Dim::Fixed(240),
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        }
    }

    fn aac_caps() -> Caps {
        Caps::Audio {
            format: AudioFormat::Aac,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }
    }

    fn au_frame(au: Vec<u8>, pts_ns: u64, keyframe: bool) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(au.into_boxed_slice())),
            FrameTiming {
                pts_ns,
                keyframe,
                ..FrameTiming::default()
            },
            0,
        ))
    }

    fn h264_frame(au: Vec<u8>, pts_ns: u64) -> PipelinePacket {
        au_frame(au, pts_ns, false)
    }

    /// Count TS packets carrying PID 0 (the PAT) across all output bytes. A TS
    /// packet is 188 bytes, sync 0x47, PID = ((b1 & 0x1F) << 8) | b2.
    fn pat_packet_count(bytes: &[u8]) -> usize {
        bytes
            .chunks(188)
            .filter(|p| {
                p.len() == 188 && p[0] == 0x47 && (((p[1] & 0x1F) as u16) << 8 | p[2] as u16) == 0
            })
            .count()
    }

    /// The `pat-interval` knob re-emits the PAT/PMT at the cadence instead of only
    /// once. Set via `set_property`, the path `parse_launch` uses; `pmt-interval`
    /// shares the cadence.
    #[tokio::test]
    async fn pat_interval_property_re_emits_tables_on_the_fan_in_muxer() {
        let au = |b: u8| alloc::vec![0u8, 0, 0, 1, b, 0xAA, 0xBB];

        // 10 ms cadence, AUs at 0/20/40/60 ms: each AU past the first re-emits the
        // tables, so the PAT appears more than once.
        const TEN_MS_90KHZ: u64 = 900;
        let mut mux = TsMux::new(1);
        mux.set_property("pat-interval", PropValue::Uint(TEN_MS_90KHZ))
            .unwrap();
        assert_eq!(
            mux.get_property("pmt-interval"),
            Some(PropValue::Uint(TEN_MS_90KHZ))
        );
        mux.configure_pipeline(0, &h264_caps()).unwrap();
        let mut sink = CaptureSink::default();
        for i in 0..4u64 {
            mux.process(0, h264_frame(au(0x65), i * 20_000_000), &mut sink)
                .await
                .unwrap();
        }
        mux.process(0, PipelinePacket::Eos, &mut sink)
            .await
            .unwrap();
        assert!(
            pat_packet_count(&sink.bytes) > 1,
            "PAT re-emitted at the interval"
        );

        // Default (100 ms): AUs inside 60 ms see the PAT once.
        let mut once = TsMux::new(1);
        once.configure_pipeline(0, &h264_caps()).unwrap();
        let mut sink0 = CaptureSink::default();
        for i in 0..4u64 {
            once.process(0, h264_frame(au(0x65), i * 20_000_000), &mut sink0)
                .await
                .unwrap();
        }
        once.process(0, PipelinePacket::Eos, &mut sink0)
            .await
            .unwrap();
        assert_eq!(
            pat_packet_count(&sink0.bytes),
            1,
            "PAT emitted once inside the default cadence"
        );
    }

    // video DTS trails its PTS by more than the VBR decoder delay
    #[tokio::test]
    async fn written_decode_times_never_trail_the_pcr() {
        const FRAME_NS: u64 = 40_000_000;
        const AUDIO_NS: u64 = 20_000_000;
        const REORDER_LAG_NS: u64 = 200_000_000;
        const TICKS_PER_90KHZ: u64 = 300;
        const ADAPTATION_FIELD_PRESENT: u8 = 0x20;
        const PCR_FLAG: u8 = 0x10;
        const PAYLOAD_UNIT_START: u8 = 0x40;
        const FIRST_PMT_PID: u16 = 0x1000;
        let idr = alloc::vec![0u8, 0, 0, 1, 0x65, 0xAA];
        let inter = alloc::vec![0u8, 0, 0, 1, 0x41, 0xBB];
        let adts = alloc::vec![0xFFu8, 0xF1, 0x4C, 0x80, 0x01, 0x00, 0xFC, 0x00];

        let mut mux = TsMux::new(2);
        mux.configure_pipeline(0, &h264_caps()).unwrap();
        mux.configure_pipeline(1, &aac_caps()).unwrap();
        let mut sink = CaptureSink::default();
        for i in 0..20u64 {
            let key = i % 6 == 0;
            let video = PipelinePacket::DataFrame(Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(
                    if key { idr.clone() } else { inter.clone() }.into_boxed_slice(),
                )),
                FrameTiming {
                    pts_ns: i * FRAME_NS + REORDER_LAG_NS,
                    // +1 keeps frame 0's DTS off the unset sentinel
                    dts_ns: i * FRAME_NS + 1,
                    keyframe: key,
                    ..FrameTiming::default()
                },
                0,
            ));
            mux.process(0, video, &mut sink).await.unwrap();
            for half in 0..2 {
                let pts = i * FRAME_NS + half * AUDIO_NS;
                mux.process(1, au_frame(adts.clone(), pts, true), &mut sink)
                    .await
                    .unwrap();
            }
        }
        mux.process(0, PipelinePacket::Eos, &mut sink)
            .await
            .unwrap();
        mux.process(1, PipelinePacket::Eos, &mut sink)
            .await
            .unwrap();

        let mut last_pcr: Option<u64> = None;
        let mut checked = 0;
        for p in sink.bytes.chunks(crate::mpegts::TS_PACKET_LEN) {
            let pid = ((p[1] & 0x1F) as u16) << 8 | p[2] as u16;
            let has_adaptation = p[3] & ADAPTATION_FIELD_PRESENT != 0;
            let has_pcr = has_adaptation && p[4] > 0 && p[5] & PCR_FLAG != 0;
            if has_pcr {
                let base = (p[6] as u64) << 25
                    | (p[7] as u64) << 17
                    | (p[8] as u64) << 9
                    | (p[9] as u64) << 1
                    | (p[10] as u64) >> 7;
                let pcr = base * TICKS_PER_90KHZ + ((((p[10] & 1) as u64) << 8) | p[11] as u64);
                assert!(
                    last_pcr.is_none_or(|last| pcr >= last),
                    "PCR went backwards"
                );
                last_pcr = Some(pcr);
            }
            let starts_pes = p[1] & PAYLOAD_UNIT_START != 0 && pid != 0 && pid < FIRST_PMT_PID;
            if !starts_pes {
                continue;
            }
            let offset = if has_adaptation { 5 + p[4] as usize } else { 4 };
            let (pts, dts, _) = crate::mpegts::parse_pes_header(&p[offset..]);
            let decode_time = dts.or(pts).expect("every AU is stamped") * TICKS_PER_90KHZ;
            let pcr = last_pcr.expect("a PCR leads every PES");
            assert!(
                decode_time >= pcr,
                "PID {pid:#x} decode time {decode_time} < PCR {pcr}"
            );
            checked += 1;
        }
        assert_eq!(checked, 60, "every AU checked");
    }

    #[tokio::test]
    async fn identity_knobs_fix_at_build_and_a_pid_collision_fails_it() {
        let au = alloc::vec![0u8, 0, 0, 1, 0x65, 0xAA];
        let mut mux = TsMux::new(1);
        mux.configure_pipeline(0, &h264_caps()).unwrap();
        let mut sink = CaptureSink::default();
        mux.process(0, h264_frame(au.clone(), 0), &mut sink)
            .await
            .unwrap();
        assert_eq!(
            mux.set_property("start-pid", PropValue::Uint(0x0200)),
            Err(PropError::ReadOnly)
        );
        assert_eq!(
            mux.set_property("bitrate", PropValue::Uint(1_000_000)),
            Err(PropError::ReadOnly)
        );

        const SHARED_PID: u64 = 0x0200;
        let mut colliding = TsMux::new(1);
        colliding
            .set_property("start-pid", PropValue::Uint(SHARED_PID))
            .unwrap();
        colliding
            .set_property("pmt-start-pid", PropValue::Uint(SHARED_PID))
            .unwrap();
        colliding.configure_pipeline(0, &h264_caps()).unwrap();
        assert!(colliding
            .process(0, h264_frame(au, 0), &mut sink)
            .await
            .is_err());
    }

    /// An output frame carries the sync flag of the video AU it packetizes,
    /// so a downstream segmenter (`tsmuxn ! hlssink`) cuts on it. Audio never sets
    /// it: every AAC AU is a sync sample, which would cut everywhere.
    #[tokio::test]
    async fn output_keyframe_flag_comes_from_the_video_pad_only() {
        const FRAME_NS: u64 = 40_000_000;
        let idr = alloc::vec![0u8, 0, 0, 1, 0x65, 0xAA];
        let inter = alloc::vec![0u8, 0, 0, 1, 0x41, 0xBB];
        let adts = alloc::vec![0xFFu8, 0xF1, 0x4C, 0x80, 0x01, 0x00, 0xFC, 0x00];

        let mut mux = TsMux::new(2);
        mux.configure_pipeline(0, &h264_caps()).unwrap();
        mux.configure_pipeline(1, &aac_caps()).unwrap();
        let mut sink = CaptureSink::default();
        // Video AUs every 40 ms with an IDR every third; an audio AU 20 ms after
        // each, flagged a sync sample the way an AAC parser flags them.
        for i in 0..6u64 {
            let key = i % 3 == 0;
            let au = if key { idr.clone() } else { inter.clone() };
            mux.process(0, au_frame(au, i * FRAME_NS, key), &mut sink)
                .await
                .unwrap();
            mux.process(
                1,
                au_frame(adts.clone(), i * FRAME_NS + FRAME_NS / 2, true),
                &mut sink,
            )
            .await
            .unwrap();
        }
        mux.process(0, PipelinePacket::Eos, &mut sink)
            .await
            .unwrap();
        mux.process(1, PipelinePacket::Eos, &mut sink)
            .await
            .unwrap();

        assert_eq!(sink.timings.len(), 12, "one output frame per AU");
        let flagged: Vec<u64> = sink
            .timings
            .iter()
            .filter(|(_, key)| *key)
            .map(|(pts, _)| *pts)
            .collect();
        assert_eq!(
            flagged,
            alloc::vec![0, 3 * FRAME_NS],
            "only the video IDRs are flagged, never an audio AU"
        );
    }

    // the video pad builds the muxer before the AAC pad's config arrives
    #[tokio::test]
    async fn a_raw_aac_pad_is_declared_latm_after_the_muxer_is_built() {
        use crate::mpegts::{TsDemuxer, STREAM_TYPE_AAC_LATM, TS_PACKET_LEN};
        // AAC-ELD, 48 kHz, stereo
        const ELD_CONFIG: [u8; 4] = [0xF8, 0xE6, 0x40, 0x00];
        const FRAME_NS: u64 = 20_000_000;
        const AUDIO_UNITS: u64 = 3;
        let idr = alloc::vec![0u8, 0, 0, 1, 0x65, 0xAA];

        let mut mux = TsMux::new(2);
        mux.configure_pipeline(0, &h264_caps()).unwrap();
        mux.configure_pipeline(1, &aac_caps()).unwrap();
        let mut sink = CaptureSink::default();
        mux.process(0, au_frame(idr.clone(), 0, true), &mut sink)
            .await
            .unwrap();
        mux.process(1, au_frame(ELD_CONFIG.to_vec(), 0, false), &mut sink)
            .await
            .unwrap();
        for i in 0..AUDIO_UNITS {
            let unit = alloc::vec![0x7Eu8, 0x89, i as u8];
            mux.process(1, au_frame(unit, i * FRAME_NS, true), &mut sink)
                .await
                .unwrap();
        }
        mux.process(0, au_frame(idr, AUDIO_UNITS * FRAME_NS, true), &mut sink)
            .await
            .unwrap();
        mux.process(0, PipelinePacket::Eos, &mut sink)
            .await
            .unwrap();
        mux.process(1, PipelinePacket::Eos, &mut sink)
            .await
            .unwrap();

        let mut demuxer = TsDemuxer::new();
        for packet in sink.bytes.chunks(TS_PACKET_LEN) {
            demuxer.push_packet(packet);
        }
        demuxer.flush();
        let stream_types: Vec<u8> = demuxer.streams().iter().map(|s| s.stream_type).collect();
        assert_eq!(stream_types, [STREAM_TYPE_H264, STREAM_TYPE_AAC_LATM]);
        let audio_pid = demuxer.streams()[1].pid;
        let audio_units = demuxer
            .take_units()
            .into_iter()
            .filter(|unit| unit.pid == audio_pid)
            .count();
        assert_eq!(
            audio_units as u64, AUDIO_UNITS,
            "the config frame is not multiplexed"
        );
    }

    #[tokio::test]
    async fn a_caps_change_keeps_a_raw_aac_pad_on_loas() {
        use crate::aacparse::loas_from_asc;
        use crate::mpegts::{TsDemuxer, TS_PACKET_LEN};
        // AAC-ELD, 48 kHz, stereo
        const ELD_CONFIG: [u8; 4] = [0xF8, 0xE6, 0x40, 0x00];
        const FRAME_NS: u64 = 20_000_000;
        let units: Vec<Vec<u8>> = (0..2u8).map(|n| alloc::vec![0x7E, 0x89, n]).collect();

        let mut mux = TsMux::new(1);
        mux.configure_pipeline(0, &aac_caps()).unwrap();
        let mut sink = CaptureSink::default();
        mux.process(0, au_frame(ELD_CONFIG.to_vec(), 0, false), &mut sink)
            .await
            .unwrap();
        for (index, unit) in units.iter().enumerate() {
            if index == 1 {
                mux.configure_pipeline(0, &aac_caps()).unwrap();
            }
            let pts_ns = index as u64 * FRAME_NS;
            mux.process(0, au_frame(unit.clone(), pts_ns, true), &mut sink)
                .await
                .unwrap();
        }
        mux.process(0, PipelinePacket::Eos, &mut sink)
            .await
            .unwrap();

        let mut demuxer = TsDemuxer::new();
        for packet in sink.bytes.chunks(TS_PACKET_LEN) {
            demuxer.push_packet(packet);
        }
        demuxer.flush();
        let carried: Vec<Vec<u8>> = demuxer.take_units().into_iter().map(|u| u.data).collect();
        let expected: Vec<Vec<u8>> = units
            .iter()
            .map(|unit| loas_from_asc(&ELD_CONFIG, unit).unwrap())
            .collect();
        assert_eq!(carried, expected);
    }

    #[test]
    fn caps_codec_in_byte_stream_out() {
        let m = TsMux::new(1);
        assert!(m.intercept_caps(0, &h264_caps()).is_ok());
        // Raw video / an existing byte stream have nothing to mux.
        let raw = Caps::RawVideo {
            format: g2g_core::RawVideoFormat::Rgba8,
            width: Dim::Fixed(2),
            height: Dim::Fixed(2),
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        assert!(m.intercept_caps(0, &raw).is_err());

        let Ok(CapsConstraint::Produces(set)) = m.caps_constraint_for_output() else {
            panic!("expected Produces");
        };
        assert!(matches!(
            set.alternatives(),
            [Caps::ByteStream {
                encoding: ByteStreamEncoding::MpegTs
            }]
        ));
    }

    #[tokio::test]
    async fn one_pad_round_trips_through_tsdemux() {
        use crate::tsdemux::TsDemux;
        use g2g_core::AsyncElement;

        let au0 = alloc::vec![0u8, 0, 0, 1, 0x65, 0xAA];
        let au1 = alloc::vec![0u8, 0, 0, 1, 0x41, 0xBB, 0xCC];

        let mut mux = TsMux::new(1);
        mux.configure_pipeline(0, &h264_caps()).unwrap();
        let mut ts_sink = CaptureSink::default();
        mux.process(0, h264_frame(au0.clone(), 10_000_000), &mut ts_sink)
            .await
            .unwrap();
        mux.process(0, h264_frame(au1.clone(), 20_000_000), &mut ts_sink)
            .await
            .unwrap();
        mux.process(0, PipelinePacket::Eos, &mut ts_sink)
            .await
            .unwrap();
        assert!(
            !ts_sink.eos,
            "EOS is forwarded by the runner's arm, not the element"
        );

        // Feed the muxed TS bytes back through the demuxer.
        let mut demux = TsDemux::new();
        demux
            .configure_pipeline(&TsMux::output_caps_value())
            .unwrap();
        let mut au_sink = AuSink::default();
        let ts_frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(ts_sink.bytes.into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        demux
            .process(PipelinePacket::DataFrame(ts_frame), &mut au_sink)
            .await
            .unwrap();
        demux
            .process(PipelinePacket::Eos, &mut au_sink)
            .await
            .unwrap();

        assert_eq!(
            au_sink.frames,
            alloc::vec![au0, au1],
            "AUs recovered through mux + demux"
        );
        assert_eq!(mux.emitted(), 2);
    }

    #[derive(Default)]
    struct AuSink {
        frames: Vec<Vec<u8>>,
    }
    impl OutputSink for AuSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.frames.push(s.to_vec());
                }
            }
            core::task::Poll::Ready(Ok(PushOutcome::Accepted))
        }
    }

    // an audio-only multiplex has no video sync point to cut on, so every sync sample is one
    #[tokio::test]
    async fn an_audio_only_mux_keeps_every_sync_flag() {
        const FRAME_NS: u64 = 20_000_000;
        let adts = alloc::vec![0xFFu8, 0xF1, 0x4C, 0x80, 0x01, 0x00, 0xFC, 0x00];
        let mut mux = TsMux::new(1);
        mux.configure_pipeline(0, &aac_caps()).unwrap();
        let mut sink = CaptureSink::default();
        for i in 0..3u64 {
            mux.process(0, au_frame(adts.clone(), i * FRAME_NS, true), &mut sink)
                .await
                .unwrap();
        }
        assert_eq!(
            sink.timings,
            alloc::vec![(0, true), (FRAME_NS, true), (2 * FRAME_NS, true)]
        );
    }

    #[tokio::test]
    async fn raw_aac_goes_out_as_loas_under_the_latm_stream_type() {
        use crate::aacparse::loas_from_asc;
        use crate::mpegts::STREAM_TYPE_AAC_LATM;
        // AAC-ELD, 48 kHz, stereo
        const ELD_CONFIG: [u8; 4] = [0xF8, 0xE6, 0x40, 0x00];
        let units = alloc::vec![
            alloc::vec![0x7Eu8, 0x89, 0xB4],
            alloc::vec![0x56, 0x78, 0x9A]
        ];

        let mut mux = TsMux::new(1);
        mux.configure_pipeline(0, &aac_caps()).unwrap();
        let mut ts_sink = CaptureSink::default();
        mux.process(0, au_frame(ELD_CONFIG.to_vec(), 0, false), &mut ts_sink)
            .await
            .unwrap();
        for (index, unit) in units.iter().enumerate() {
            let pts_ns = (index as u64 + 1) * 10_000_000;
            mux.process(0, au_frame(unit.clone(), pts_ns, false), &mut ts_sink)
                .await
                .unwrap();
        }
        assert_eq!(
            mux.emitted(),
            units.len() as u64,
            "the config is not multiplexed"
        );

        let mut demuxer = demuxer_of(&ts_sink.bytes);
        assert_eq!(demuxer.streams()[0].stream_type, STREAM_TYPE_AAC_LATM);
        let carried: Vec<Vec<u8>> = demuxer.take_units().into_iter().map(|u| u.data).collect();
        let expected: Vec<Vec<u8>> = units
            .iter()
            .map(|unit| loas_from_asc(&ELD_CONFIG, unit).unwrap())
            .collect();
        assert_eq!(carried, expected, "each unit rides its own LOAS frame");
    }

    #[tokio::test]
    async fn refuses_usac() {
        // the AudioSpecificConfig of fate-suite aac/usac/Fd_1_c1_0x03.mp4, USAC 48 kHz mono
        const USAC_CONFIG: [u8; 7] = [0xF9, 0x46, 0x23, 0x21, 0x10, 0xC0, 0x00];
        let mut mux = TsMux::new(1);
        mux.configure_pipeline(
            0,
            &Caps::Audio {
                format: AudioFormat::Aac,
                channels: 1,
                sample_rate: 48_000,
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
            },
        )
        .unwrap();
        let mut ts_sink = CaptureSink::default();
        let result = mux
            .process(0, au_frame(USAC_CONFIG.to_vec(), 0, false), &mut ts_sink)
            .await;
        assert_eq!(result, Err(G2gError::CapsMismatch));
    }

    #[tokio::test]
    async fn an_in_band_opus_head_sets_the_descriptor_and_stays_out_of_the_multiplex() {
        use crate::opusparse::synth_opus_head;
        const CHANNELS: u8 = 6;
        let packets: Vec<Vec<u8>> = (0..2u8).map(|n| alloc::vec![0xFC, n, n]).collect();
        let mut mux = TsMux::new(1);
        mux.configure_pipeline(
            0,
            &Caps::Audio {
                format: AudioFormat::Opus,
                channels: 0,
                sample_rate: OPUS_RATE_HZ,
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
            },
        )
        .unwrap();
        let mut ts_sink = CaptureSink::default();
        let head = synth_opus_head(CHANNELS, OPUS_RATE_HZ, 0);
        mux.process(0, au_frame(head, 0, false), &mut ts_sink)
            .await
            .unwrap();
        for (index, packet) in packets.iter().enumerate() {
            let pts_ns = index as u64 * 20_000_000;
            mux.process(0, au_frame(packet.clone(), pts_ns, false), &mut ts_sink)
                .await
                .unwrap();
        }

        let mut demuxer = demuxer_of(&ts_sink.bytes);
        let pid = demuxer.streams()[0].pid;
        assert_eq!(demuxer.opus_channels(pid), Some(CHANNELS));
        let carried: Vec<Vec<u8>> = demuxer
            .take_units()
            .into_iter()
            .flat_map(|u| {
                crate::mpegts::opus_ts_packets(&u.data)
                    .into_iter()
                    .map(<[u8]>::to_vec)
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(carried, packets);
    }

    #[tokio::test]
    async fn loas_frames_go_out_unchanged_under_the_latm_stream_type() {
        use crate::aacparse::loas_from_asc;
        use crate::mpegts::STREAM_TYPE_AAC_LATM;
        // AAC LC, 48 kHz, stereo
        const LC_CONFIG: [u8; 2] = [0x11, 0x90];
        let frames: Vec<Vec<u8>> = [[0x21u8, 0x10, 0x04], [0x21, 0x10, 0x05]]
            .iter()
            .map(|unit| loas_from_asc(&LC_CONFIG, unit).unwrap())
            .collect();

        let mut mux = TsMux::new(1);
        mux.configure_pipeline(0, &aac_caps()).unwrap();
        let mut ts_sink = CaptureSink::default();
        for (index, frame) in frames.iter().enumerate() {
            let pts_ns = index as u64 * 10_000_000;
            mux.process(0, au_frame(frame.clone(), pts_ns, false), &mut ts_sink)
                .await
                .unwrap();
        }

        let demuxer = demuxer_of(&ts_sink.bytes);
        assert_eq!(demuxer.streams()[0].stream_type, STREAM_TYPE_AAC_LATM);
        assert_eq!(demuxed_units(&ts_sink.bytes), frames);
    }

    #[tokio::test]
    async fn a_caps_change_keeps_the_continuity_counters_running() {
        use crate::mpegts::TS_PACKET_LEN;
        // more than one TS packet each, so a unit is in flight when the next one starts
        const UNIT_LEN: usize = 3 * TS_PACKET_LEN;
        let caps = |channels| Caps::Audio {
            format: AudioFormat::Aac,
            channels,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        let units: Vec<Vec<u8>> = (0..3u8).map(|n| alloc::vec![n; UNIT_LEN]).collect();

        let mut mux = TsMux::new(1);
        mux.configure_pipeline(0, &caps(2)).unwrap();
        let mut ts_sink = CaptureSink::default();
        for (index, unit) in units.iter().enumerate() {
            if index == 1 {
                mux.configure_pipeline(0, &caps(1)).unwrap();
            }
            let pts_ns = index as u64 * 10_000_000;
            mux.process(0, au_frame(unit.clone(), pts_ns, false), &mut ts_sink)
                .await
                .unwrap();
        }

        assert_eq!(demuxed_units(&ts_sink.bytes), units);
    }
}
