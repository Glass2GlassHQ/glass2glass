//! MPEG-2 Transport Stream demuxer (M108): parse a TS byte stream into the
//! elementary-stream access units it carries (ISO/IEC 13818-1).
//!
//! Pure `no_std + alloc` parsing, like the `mp4box` precedent and `annexb`: this
//! module is just the state machine (sync to 188-byte packets, read the PAT to
//! find the PMT, read the PMT to find the elementary streams, reassemble PES
//! packets per PID and strip their headers). The [`crate::tsdemux::TsDemux`]
//! element wraps it; the split keeps the bit-twiddling testable without a runner.
//!
//! Scope: every program the PAT names (select one via
//! [`TsDemuxer::set_program_number`], default the first in PAT order); first PAT
//! and first PMT per program win (no version / update handling). PSI sections
//! (PAT / PMT) are assumed to fit in one TS packet (true for the small tables in
//! practice); PES payloads reassemble across packets. The carried elementary
//! stream for H.264 / H.265 is already Annex-B, so a unit feeds `h264parse`
//! directly.

use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::VideoCodec;

use crate::mpeg2video::Mpeg2TimestampSynth;
use crate::poc::AccessUnitPoc;

/// MPEG-TS packet size in bytes (the standard 188; M2TS 192 with a 4-byte
/// timestamp prefix is not handled).
pub const TS_PACKET_LEN: usize = 188;

const SYNC_BYTE: u8 = 0x47;
const PID_PAT: u16 = 0x0000;
/// The DVB SI PID the SDT rides (ETSI EN 300 468); shared with the BAT and the
/// other-TS SDT, which the `table_id` filter drops.
const PID_SDT: u16 = 0x0011;
/// SDT `table_id` for this transport stream (0x46 is the other-TS variant, which
/// describes services carried elsewhere and is ignored).
const TABLE_ID_SDT: u8 = 0x42;
/// The DVB SI PID the EIT rides (ETSI EN 300 468). It carries the present/following
/// tables and the much larger schedule tables.
const PID_EIT: u16 = 0x0012;
/// EIT `table_id` for the present/following events of this transport stream. 0x4F
/// is the other-TS variant (events of a service carried elsewhere) and is ignored.
const TABLE_ID_EIT_PRESENT_FOLLOWING: u8 = 0x4E;
/// EIT `table_id` range for the schedule events of this transport stream (M1056):
/// what a service shows over the coming days, segmented across many sections.
/// 0x60..=0x6F is the other-TS variant and is ignored like 0x4F.
const TABLE_ID_EIT_SCHEDULE: core::ops::RangeInclusive<u8> = 0x50..=0x5F;
/// DVB `short_event_descriptor` tag: a language code then the length-prefixed
/// event name and short description.
const DESC_TAG_SHORT_EVENT: u8 = 0x4D;
/// `registration_descriptor` tag: a 4-byte format_identifier naming what a
/// stream_type on its own does not say (which is every use of a private PES).
const DESC_TAG_REGISTRATION: u8 = 0x05;
/// DVB `service_descriptor` tag: service_type, then the length-prefixed provider
/// and service name.
const DESC_TAG_SERVICE: u8 = 0x48;
/// `ISO_639_language_descriptor` tag: 4 bytes per language (a 3-letter code plus
/// an audio_type byte).
const DESC_TAG_ISO639: u8 = 0x0A;
/// DVB `subtitling_descriptor` tag (ETSI EN 300 468): 8 bytes per subtitle
/// stream, a 3-letter language code, a subtitling_type, then the composition and
/// ancillary page ids.
const DESC_TAG_SUBTITLING: u8 = 0x59;
/// DVB `teletext_descriptor` tag (ETSI EN 300 468): 5 bytes per teletext
/// service, a 3-letter language code then a packed teletext_type (5 bits) +
/// magazine number (3 bits) and the BCD page number. `VBI_teletext_descriptor`
/// (0x46) has the identical body and marks the same carriage, so both are read.
const DESC_TAG_TELETEXT: u8 = 0x56;
const DESC_TAG_VBI_TELETEXT: u8 = 0x46;
/// SDT `service_type` for digital television, and for digital radio (a program
/// with no video stream).
const SERVICE_TYPE_TV: u8 = 0x01;
const SERVICE_TYPE_RADIO: u8 = 0x02;

/// The [`g2g_core::Tag::Other`] key the SDT `service_provider_name` rides under.
/// `Tag` has no typed provider variant, so the key is ffprobe's own
/// (`service_provider`, what it reports for this field): a tag list built from one
/// tool's output means the same thing to the other. The service name has a typed
/// home, [`g2g_core::Tag::Title`].
pub const TAG_KEY_SERVICE_PROVIDER: &str = "service_provider";
/// The alternative key for the service name, accepted on the mux side next to
/// [`g2g_core::Tag::Title`] because it is what ffmpeg's `-metadata` takes and
/// ffprobe reports.
pub const TAG_KEY_SERVICE_NAME: &str = "service_name";

/// The [`g2g_core::Tag::Other`] keys the DVB EIT present/following event text
/// rides under (M1049), and the [`g2g_core::Tag::Number`] keys its start time
/// (Unix seconds, UTC) and duration (seconds) ride under (M1056). `Tag` has no
/// typed event variant, and the SDT service name already owns
/// [`g2g_core::Tag::Title`] for the same program, so the fields keep their own
/// keys: the event on air now, and the one after it.
pub const TAG_KEY_EVENT_NAME: &str = "event_name";
pub const TAG_KEY_EVENT_TEXT: &str = "event_text";
pub const TAG_KEY_EVENT_START: &str = "event_start";
pub const TAG_KEY_EVENT_DURATION: &str = "event_duration";
pub const TAG_KEY_NEXT_EVENT_NAME: &str = "next_event_name";
pub const TAG_KEY_NEXT_EVENT_TEXT: &str = "next_event_text";
pub const TAG_KEY_NEXT_EVENT_START: &str = "next_event_start";
pub const TAG_KEY_NEXT_EVENT_DURATION: &str = "next_event_duration";

/// The keys one EIT schedule event posts under (M1056). A schedule table names
/// far more than the two events present/following holds, so each event posts as
/// its own tag list (one [`g2g_core::BusMessage::Tag`] per event) and the keys do
/// not distinguish a slot: the event id, the name and short description, the
/// start time in Unix seconds (UTC) and the duration in seconds.
pub const TAG_KEY_SCHEDULE_EVENT_ID: &str = "schedule_event_id";
pub const TAG_KEY_SCHEDULE_EVENT_NAME: &str = "schedule_event_name";
pub const TAG_KEY_SCHEDULE_EVENT_TEXT: &str = "schedule_event_text";
pub const TAG_KEY_SCHEDULE_EVENT_START: &str = "schedule_event_start";
pub const TAG_KEY_SCHEDULE_EVENT_DURATION: &str = "schedule_event_duration";

/// Cap on a single reassembled PES payload. A video PES carries no declared
/// length and is delimited only by the next payload-unit-start, so a stream that
/// opens a PES and then sends an endless run of continuation packets (never
/// another start) would grow the buffer without bound. 16 MiB comfortably holds
/// a large intra access unit while bounding the memory an untrusted stream costs.
const MAX_PES_BYTES: usize = 16 * 1024 * 1024;

/// PMT `stream_type` for H.264 (AVC) video.
pub const STREAM_TYPE_H264: u8 = 0x1B;
/// PMT `stream_type` for H.265 (HEVC) video.
pub const STREAM_TYPE_H265: u8 = 0x24;
/// PMT `stream_type` for MPEG-4 Part 2 (Visual) video.
pub const STREAM_TYPE_MPEG4P2: u8 = 0x10;
/// PMT `stream_type` for MPEG-1 Video (ISO/IEC 11172-2).
pub const STREAM_TYPE_MPEG1_VIDEO: u8 = 0x01;
/// PMT `stream_type` for MPEG-2 Video (ISO/IEC 13818-2). One `VideoCodec::Mpeg2`
/// covers both this and 0x01, the MPEG2VIDEO decoder playing either.
pub const STREAM_TYPE_MPEG2_VIDEO: u8 = 0x02;
/// PMT `stream_type` for ADTS AAC audio.
pub const STREAM_TYPE_AAC: u8 = 0x0F;
/// PMT `stream_type` for MPEG-1 Audio (Layer I/II/III, e.g. `mp2`).
pub const STREAM_TYPE_MPEG1_AUDIO: u8 = 0x03;
/// PMT `stream_type` for MPEG-2 Audio (the low-sample-rate extension of the above).
pub const STREAM_TYPE_MPEG2_AUDIO: u8 = 0x04;
/// PMT `stream_type` for a private PES stream (0x06). Opus, DVB AC-3 and DVB
/// subtitles all ride this, identified by their ES descriptors (an 'Opus'
/// registration for Opus, an AC-3 descriptor (tag 0x6A) for AC-3, a
/// subtitling_descriptor (tag 0x59) for subtitles).
pub const STREAM_TYPE_PRIVATE_PES: u8 = 0x06;
/// PMT `stream_type` for ATSC AC-3 audio (A/52). The ATSC carriage of Dolby
/// Digital; DVB instead uses a private PES (0x06) with an AC-3 descriptor.
pub const STREAM_TYPE_AC3: u8 = 0x81;
/// PMT `stream_type` for metadata carried in PES packets (0x15), the synchronous
/// KLV carriage of MISB ST 1402 / STANAG 4609. Asynchronous KLV instead rides a
/// private PES (0x06) with a 'KLVA' registration descriptor.
///
/// The mux wraps each KLV access unit in one ISO 13818-1 metadata AU cell, which
/// is both what ST 1402 calls for and what ffmpeg's demuxer assumes: it skips 5
/// bytes off every 0x15 PES payload on the metadata stream_id. The demux accepts
/// a bare payload too (see `unwrap_metadata_au_cells`).
pub const STREAM_TYPE_METADATA_PES: u8 = 0x15;

/// PMT ES-info for a synchronous-KLV (0x15) stream: a `metadata_descriptor`
/// (tag 0x26) naming 'KLVA' as both the application format and the metadata
/// format, then metadata_service_id 0 and a decoder_config_flags / DSM-CC_flag
/// byte of zero (reserved bits set). This is what identifies the stream as KLV:
/// a bare 0x15 with no descriptor reads as an unknown data stream (ffmpeg maps
/// 0x15 to KLV only through this descriptor's format identifier).
const KLV_METADATA_DESCRIPTOR: &[u8] = &[
    0x26, 13, // tag, length
    0xFF, 0xFF, // metadata_application_format = 0xFFFF (identifier follows)
    b'K', b'L', b'V', b'A', // metadata_application_format_identifier
    0xFF, // metadata_format = 0xFF (identifier follows)
    b'K', b'L', b'V', b'A', // metadata_format_identifier
    0x00, // metadata_service_id
    0x0F, // decoder_config_flags '000', DSM-CC_flag 0, reserved
];

/// The MISB ST 1402 `registration_descriptor` marking a private PES (0x06) as
/// asynchronous KLV, which is how this muxer writes an unqualified private stream.
const KLVA_REGISTRATION: &[u8] = &[DESC_TAG_REGISTRATION, 4, b'K', b'L', b'V', b'A'];

/// The `registration_descriptor` format_identifier the AOM "Carriage of AV1 in
/// MPEG-2 TS" spec assigns AV1 on a private PES (0x06). AV1 has no `stream_type`
/// of its own, so this is what tells it apart from every other 0x06 use.
const AV1_REGISTRATION_ID: &[u8; 4] = b"AV01";
/// The format_identifier GStreamer's `tsdemux` / `mpegtsmux` actually read and
/// write for the same carriage. It predates the AOM spec (the muxer still calls
/// the mapping "custom", refusing it without `enable-custom-mappings`), and no
/// shipping demuxer accepts 'AV01': GStreamer's own `tsdemux` activates no program
/// for a stream carrying it, and ffmpeg's muxer writes AV1 with no descriptor at
/// all, a bare 0x06 nothing (its own demuxer included) can identify. Both are
/// accepted on the demux side; the mux writes this one so a real receiver plays
/// the result.
const AV1_REGISTRATION_ID_GSTREAMER: &[u8; 4] = b"AV1G";

/// The first 4 bytes of every SMPTE ST 336 KLV key (the SMPTE UL designator).
const KLV_UL_PREFIX: [u8; 4] = [0x06, 0x0E, 0x2B, 0x34];

/// One elementary stream announced by the PMT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElementaryStream {
    pub pid: u16,
    pub stream_type: u8,
    /// Opus channel count for a private (0x06) stream whose ES descriptors carry
    /// the 'Opus' registration + DVB extension descriptor. `Some` marks the stream
    /// as Opus (disambiguating the generic 0x06); `None` for any other 0x06 use.
    pub opus_channels: Option<u8>,
    /// True for a private (0x06) stream carrying an AC-3 descriptor (tag 0x6A), the
    /// DVB carriage of AC-3 (disambiguating the generic 0x06). ATSC AC-3 rides its
    /// own `stream_type` 0x81 and does not set this.
    pub ac3: bool,
    /// True for a KLV metadata stream (STANAG 4609 / MISB ST 1402): a private
    /// (0x06) stream carrying a 'KLVA' registration descriptor (asynchronous
    /// KLV), or any metadata-in-PES (0x15) stream (synchronous KLV).
    pub klv: bool,
    /// True for a private (0x06) stream carrying an AV1 `registration_descriptor`
    /// (M1049), the AOM carriage of AV1 (disambiguating the generic 0x06). The PES
    /// payload is then one AV1 temporal unit in the low-overhead OBU format, which
    /// is what [`crate::av1parse::Av1Parse`] and the AV1 decoders read.
    pub av1: bool,
    /// The 3-letter code of the ES-info `ISO_639_language_descriptor` (tag 0x0A),
    /// if it carried one. Read it as text with
    /// [`language_code`](Self::language_code).
    pub language: Option<[u8; 3]>,
    /// The first entry of a private (0x06) stream's DVB `subtitling_descriptor`
    /// (tag 0x59): `(subtitling_type, composition_page_id, ancillary_page_id)`.
    /// `Some` marks the stream as DVB subtitles (disambiguating the generic
    /// 0x06) and carries the page ids the decoder composes under.
    pub subtitling: Option<(u8, u16, u16)>,
    /// The teletext service a private (0x06) stream's DVB `teletext_descriptor`
    /// (tag 0x56 / 0x46) names. `Some` marks the stream as EBU teletext
    /// (disambiguating the generic 0x06) and carries the page a decoder should
    /// follow. The subtitle entry wins when the descriptor lists several.
    pub teletext: Option<TeletextService>,
}

/// One entry of a DVB `teletext_descriptor`: which teletext page a stream
/// carries, in what language, and what it holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TeletextService {
    /// 3-letter ISO 639 language code.
    pub language: [u8; 3],
    /// EN 300 468 `teletext_type`: 0x01 initial page, 0x02 subtitle page, 0x03
    /// additional information, 0x04 programme schedule, 0x05 subtitle page for
    /// the hearing impaired.
    pub teletext_type: u8,
    /// Magazine 1..8 (the wire's 0 means 8), the hundreds digit of the page.
    pub magazine: u8,
    /// The two BCD digits of the page within its magazine.
    pub page: u8,
}

impl TeletextService {
    /// The page as a viewer reads it: magazine hundreds plus the two BCD digits
    /// (magazine 8, page 0x88 is page 888). A page byte whose nibbles are not
    /// decimal digits still yields a number, since the wire field is nominally
    /// BCD but nothing enforces it.
    pub fn page_number(&self) -> u16 {
        let mag = if self.magazine == 0 { 8 } else { self.magazine };
        let tens = (self.page >> 4) as u16;
        let units = (self.page & 0x0f) as u16;
        mag as u16 * 100 + tens * 10 + units
    }

    /// Whether this entry is a subtitle page (the two subtitle `teletext_type`s),
    /// the only kind a subtitle decoder composes.
    pub fn is_subtitle(&self) -> bool {
        matches!(self.teletext_type, 0x02 | 0x05)
    }
}

impl ElementaryStream {
    /// The PMT-declared language of this stream, if it carried an
    /// `ISO_639_language_descriptor`.
    pub fn language_code(&self) -> Option<&str> {
        self.language
            .as_ref()
            .and_then(|c| core::str::from_utf8(c).ok())
    }
}

/// The DVB `service_descriptor` text of one service (SDT): the two names a
/// transport stream carries for the program, which the demuxer surfaces as tags
/// and the muxer writes. Empty when the stream declared the field empty, or when
/// its DVB character table was one this parser will not decode (see `dvb_text`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ServiceInfo {
    /// `service_name`, the program's own name.
    pub name: String,
    /// `service_provider_name`, the broadcaster.
    pub provider: String,
}

/// Which EIT table an event came from (M1056).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EitSlot {
    /// The event on air now: present/following, section 0.
    Present,
    /// The event after it: present/following, section 1.
    Following,
    /// An event of a schedule table (`table_id` 0x50..=0x5F), which announces
    /// what a service shows over the coming days.
    Schedule,
}

/// One DVB EIT event (M1049): what a service is showing now, what it shows next,
/// or an entry of its schedule. The text comes from the `short_event_descriptor`,
/// so an event announced without one is not reported at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EitEvent {
    /// The service the event belongs to. A `service_id` is a `program_number`,
    /// the same join the SDT uses.
    pub service_id: u16,
    pub event_id: u16,
    /// Which table the event came from.
    pub slot: EitSlot,
    /// `event_name_char`, empty when the field was empty or its DVB character
    /// table was one this parser will not decode (see `dvb_text`).
    pub name: String,
    /// `text_char`, the short description under the name; empty on the same terms.
    pub text: String,
    /// `start_time` as seconds since the Unix epoch, UTC (M1056). `None` when the
    /// stream declared the field undefined (all ones) or encoded it invalidly.
    pub start_unix_secs: Option<u64>,
    /// `duration` in seconds, 0 when the stream encoded it invalidly.
    pub duration_secs: u32,
}

/// Cap on the EIT present/following events held (M1049). Present/following is two
/// per service, so this holds a large multiplex; past it, later services go
/// unreported rather than growing the table on a stream that churns `service_id`s.
const MAX_EIT_EVENTS: usize = 64;

/// Cap on the queued EIT schedule events (M1056). A whole multiplex's 8-day EPG
/// runs to a few thousand events, so a consumer that drains between feeds sees a
/// full one; past it, later events are dropped rather than growing the queue
/// without bound on a stream that never stops announcing new ones.
pub const MAX_EIT_SCHEDULE_EVENTS: usize = 4096;

/// Cap on the `(service_id, table_id, section_number)` version slots tracked
/// (M1056). Present/following is two sections per service, but a schedule table
/// is segmented into up to 256 sections across 16 `table_id`s, so the version
/// bookkeeping needs far more room than the present/following table itself. Past
/// this a further section is ignored rather than growing the list.
const MAX_EIT_VERSION_SLOTS: usize = 4096;

/// MJD 40587 is 1970-01-01, so it is the Modified Julian Date of the Unix epoch
/// (EN 300 468 Annex C).
const MJD_UNIX_EPOCH: u16 = 40587;
const SECONDS_PER_DAY: u64 = 86_400;
const SECONDS_PER_HOUR: u64 = 3600;
const SECONDS_PER_MINUTE: u64 = 60;

/// Decode an EIT 5-byte `start_time` (EN 300 468 §5.2.4 and Annex C): a 16-bit
/// Modified Julian Date, then three BCD bytes of UTC hh:mm:ss. `None` for the
/// all-ones value the spec defines as undefined, for a date before the Unix
/// epoch, and for a byte that is not BCD.
fn mjd_bcd_to_unix_secs(field: [u8; 5]) -> Option<u64> {
    if field == [0xFF; 5] {
        return None;
    }
    let mjd = u16::from_be_bytes([field[0], field[1]]);
    let days = u64::from(mjd.checked_sub(MJD_UNIX_EPOCH)?);
    let time = bcd_hms_secs([field[2], field[3], field[4]])?;
    days.checked_mul(SECONDS_PER_DAY)?.checked_add(time)
}

/// Decode three BCD bytes of hh:mm:ss into seconds. `None` when a nibble is above
/// 9, which is not BCD.
fn bcd_hms_secs(field: [u8; 3]) -> Option<u64> {
    let hours = u64::from(bcd_byte(field[0])?);
    let minutes = u64::from(bcd_byte(field[1])?);
    let seconds = u64::from(bcd_byte(field[2])?);
    hours
        .checked_mul(SECONDS_PER_HOUR)?
        .checked_add(minutes.checked_mul(SECONDS_PER_MINUTE)?)?
        .checked_add(seconds)
}

/// One packed BCD byte as its decimal value, `None` when either nibble is above 9.
fn bcd_byte(byte: u8) -> Option<u8> {
    let tens = byte >> 4;
    let units = byte & 0x0F;
    (tens <= 9 && units <= 9).then_some(tens * 10 + units)
}

/// Cap on the EIT section being reassembled, the limit EN 300 468 sets on an EIT
/// section. The declared `section_length` cannot be trusted before the section is
/// whole, so the buffer is capped rather than sized from it.
const MAX_PSI_SECTION_BYTES: usize = 4096;

/// A reassembled PES payload: one access unit of an elementary stream.
#[derive(Debug, Clone, PartialEq)]
pub struct EsUnit {
    pub pid: u16,
    pub stream_type: u8,
    /// Presentation timestamp in 90 kHz units, if the PES carried one.
    pub pts_90khz: Option<u64>,
    /// Decode timestamp in 90 kHz units, if the PES carried a separate DTS
    /// (`PTS_DTS_flags == '11'`); `None` when the PES was PTS-only.
    pub dts_90khz: Option<u64>,
    /// The elementary stream bytes (for H.264/H.265, Annex-B).
    pub data: Vec<u8>,
}

/// Cap on the per-PID [`VideoPtsSynth`] table. PIDs come from the PMT, so a real
/// multiplex has a handful; the cap keeps a stream that churns PMT video PIDs
/// from growing the table without bound. Past it, later PIDs go unsynthesized.
const MAX_VIDEO_PTS_STREAMS: usize = 16;

/// Per-frame video PTS synthesis for one elementary stream (M948). A transport
/// stream need only carry a PES timestamp every 700 ms, so a conforming mux may
/// leave most access units unstamped, and forwarding those unstamped lands them
/// all at 0: a pacing sink plays that as a burst then a freeze.
///
/// An unstamped unit gets `last + frame_period` when the stream presents
/// pictures in the order it codes them, which an SPS proves (H.264 POC type 2,
/// H.265 `sps_max_num_reorder_pics` 0). Every real stamp re-anchors, so a period
/// that is slightly off never drifts past one stamping interval.
///
/// A stream that may reorder takes the display order from the picture order
/// count instead (M952), and MPEG-2 from the picture header's
/// `temporal_reference`. Both anchor on the real stamps the same way.
#[derive(Debug, Default)]
struct VideoPtsSynth {
    /// Whether the SPS proved coded order is display order. `None` until one
    /// parses; the per-unit period path runs only on `Some(true)`.
    presents_in_decode_order: Option<bool>,
    /// Frame period in 90 kHz ticks.
    period_90: Option<u64>,
    /// Whether `period_90` came from the SPS VUI timing info. A measured period
    /// keeps being refined; a declared one is not overridden.
    period_from_sps: bool,
    /// The last PTS this stream carried, real or synthesized.
    last_pts_90: Option<u64>,
    /// Units seen since `last_pts_90` was set, so a unit that could not be
    /// stamped still costs its display slot.
    units_since_last_pts: u64,
    /// The last real PES stamp, and the units seen since, for measuring the
    /// period from the two nearest real stamps when the SPS declares none.
    last_real_pts_90: Option<u64>,
    units_since_last_real: u64,
    /// Picture order count per access unit, for a stream that may reorder.
    poc: AccessUnitPoc,
    /// Where those counts sit on the presentation timeline.
    reorder: ReorderPts,
    /// MPEG-2 display-order synthesis, shared with the program-stream demuxer.
    mpeg2: Mpeg2TimestampSynth,
}

impl VideoPtsSynth {
    /// Stamp one access unit, filling `pts_90khz` (and, for MPEG-2, `dts_90khz`)
    /// in place when it is unstamped and this stream qualifies. On the
    /// coded-order path the decode timestamp equals the presentation one and
    /// needs no synthesis (the element falls back to the PTS for a missing DTS).
    fn stamp(
        &mut self,
        codec: VideoCodec,
        au: &[u8],
        pts_90khz: &mut Option<u64>,
        dts_90khz: &mut Option<u64>,
    ) {
        if codec == VideoCodec::Mpeg2 {
            self.stamp_mpeg2(au, pts_90khz, dts_90khz);
            return;
        }
        self.units_since_last_pts = self.units_since_last_pts.saturating_add(1);
        self.units_since_last_real = self.units_since_last_real.saturating_add(1);
        self.read_sps(codec, au);
        if self.presents_in_decode_order == Some(false) {
            self.stamp_by_poc(codec, au, pts_90khz);
            return;
        }

        if let Some(real) = *pts_90khz {
            self.measure_period(real);
            self.last_pts_90 = Some(real);
            self.units_since_last_pts = 0;
            return;
        }
        if self.presents_in_decode_order != Some(true) {
            return;
        }
        let (Some(last), Some(period)) = (self.last_pts_90, self.period_90) else {
            return;
        };
        let synthesized = last.saturating_add(period.saturating_mul(self.units_since_last_pts));
        *pts_90khz = Some(synthesized);
        self.last_pts_90 = Some(synthesized);
        self.units_since_last_pts = 0;
    }

    /// Take the frame period from the span between this real stamp and the
    /// previous one, divided by the units in between. Only for a stream whose
    /// SPS declared no timing info; a backwards or repeated stamp yields nothing.
    fn measure_period(&mut self, real: u64) {
        if !self.period_from_sps {
            if let Some(previous) = self.last_real_pts_90 {
                let span = real.saturating_sub(previous);
                if let Some(measured) = span
                    .checked_div(self.units_since_last_real)
                    .filter(|period| *period > 0)
                {
                    self.period_90 = Some(measured);
                }
            }
        }
        self.last_real_pts_90 = Some(real);
        self.units_since_last_real = 0;
    }

    /// Re-read the SPS this access unit carries, if any. Every unit is scanned,
    /// since parameter sets travel in band and may change mid-stream.
    fn read_sps(&mut self, codec: VideoCodec, au: &[u8]) {
        use crate::nalparse::NalCodec;
        let info = match codec {
            VideoCodec::H265 => crate::h265parse::H265Codec::extract_sps_info(au),
            _ => crate::h264parse::H264Codec::extract_sps_info(au),
        };
        let Some(info) = info else {
            return;
        };
        self.presents_in_decode_order = Some(info.presents_in_decode_order);
        if let Some(poc) = info.poc {
            self.poc.set_sps(poc);
        }
        let Some(period) = info.framerate.and_then(frame_period_90khz) else {
            return;
        };
        self.period_90 = Some(period);
        self.period_from_sps = true;
    }

    /// Stamp an access unit of a stream that may reorder, by its picture order
    /// count. A real stamp re-anchors the timeline (and, with a second one,
    /// measures how many ticks a count unit spans); an unstamped unit lands at
    /// its own distance from that anchor, which is behind it for a picture the
    /// stream coded ahead of its display slot.
    ///
    /// The first stamp alone carries the timeline when the SPS declares the
    /// frame period (M1156): the count step per frame follows from the codec,
    /// so the slope is known before a second stamp measures it.
    fn stamp_by_poc(&mut self, codec: VideoCodec, au: &[u8], pts_90khz: &mut Option<u64>) {
        let Some(poc) = self.poc.push_access_unit(codec, au).map(i64::from) else {
            return;
        };
        let declared_period = self.period_90.filter(|_| self.period_from_sps);
        if let Some(real) = *pts_90khz {
            self.reorder.anchor(poc, real, declared_period, codec);
        }
        let frame_coded_h264 = codec != VideoCodec::H265 && self.poc.frame_mbs_only() == Some(true);
        if frame_coded_h264 {
            self.reorder.refine_h264_step_from_parity(poc);
        }
        if pts_90khz.is_none() {
            *pts_90khz = self.reorder.synthesize(poc);
        }
    }

    /// Stamp an MPEG-2 access unit from its picture header, on the frame period
    /// the sequence header in effect declares.
    fn stamp_mpeg2(&mut self, au: &[u8], pts_90khz: &mut Option<u64>, dts_90khz: &mut Option<u64>) {
        if let Some(period) = crate::mpeg2video::parse_sequence_header(au)
            .and_then(|seq| frame_period_90khz(seq.framerate_q16))
        {
            self.period_90 = Some(period);
        }
        let Some(period) = self.period_90 else {
            return;
        };
        self.mpeg2.stamp(au, pts_90khz, dts_90khz, period);
    }
}

/// Largest picture-order-count step per coded frame [`ReorderPts`] will snap a
/// declared frame period onto: H.265 counts pictures (1), H.264 counts fields
/// (2, or 4 across a field pair). A measurement implying more than that is a
/// count this synthesis cannot read as a frame grid, so the measured ratio
/// stands instead.
const MAX_POC_STEP_PER_FRAME: u64 = 4;

/// Picture order counts one coded frame advances in H.265, which counts pictures.
const H265_POC_STEP_PER_FRAME: u64 = 1;
/// The same for H.264, which counts fields: a frame-coded picture advances both,
/// and a field-coded stream codes one access unit per field, half a period apart.
const H264_POC_STEP_PER_FRAME: u64 = 2;
/// The H.264 step when the encoder counts pictures instead of fields, which an
/// odd count in a frame-coded stream proves.
const H264_PICTURE_COUNTING_POC_STEP: u64 = 1;

/// Where a reordering stream's picture order counts sit on the presentation
/// timeline (M952). Presentation time is linear in the count, so two real PES
/// stamps fix the line exactly, whatever reorder depth the stream codes at: the
/// later of the two anchors it and the pair measures its slope.
///
/// One stamp is enough when the SPS declares the frame period (M1156): the
/// codec's count step per frame turns that period into a provisional slope, and
/// the second stamp replaces it with the measured one.
#[derive(Debug, Default)]
struct ReorderPts {
    /// The last real stamp and the count it named.
    anchor: Option<(i64, u64)>,
    /// Ticks per count, as the exact ratio `(ticks, counts)`.
    scale: Option<(u64, u64)>,
    /// Whether `scale` is the declared period over an assumed count step rather
    /// than a span measured between two stamps.
    scale_is_provisional: bool,
}

impl ReorderPts {
    /// Take a real stamp: re-anchor on it, and measure the slope against the
    /// previous anchor. A declared frame period replaces the measured slope
    /// when it divides into it a whole number of counts per frame, so the grid
    /// follows the rate the stream declares rather than a rounded span. The
    /// first stamp has nothing to measure against, so a declared period stands
    /// in over the codec's count step until the second one arrives.
    fn anchor(
        &mut self,
        poc: i64,
        pts_90khz: u64,
        declared_period_90: Option<u64>,
        codec: VideoCodec,
    ) {
        match self.anchor {
            Some((previous_poc, previous_pts)) => {
                let counts = poc.abs_diff(previous_poc);
                let ticks = pts_90khz.abs_diff(previous_pts);
                if counts > 0 && ticks > 0 {
                    self.scale = Some(snap_to_declared_period(ticks, counts, declared_period_90));
                    self.scale_is_provisional = false;
                }
            }
            None => {
                if let Some(period) = declared_period_90 {
                    self.scale = Some((period, poc_step_per_frame(codec)));
                    self.scale_is_provisional = true;
                }
            }
        }
        self.anchor = Some((poc, pts_90khz));
    }

    /// An H.264 encoder may step the order count once per frame rather than once
    /// per field. Every count of a frame-coded stream is then even under the
    /// field step, so the first odd count proves the step is one. A measured
    /// slope already spans whatever the stream does, so only a provisional one
    /// is rewritten.
    fn refine_h264_step_from_parity(&mut self, poc: i64) {
        if !self.scale_is_provisional || poc % 2 == 0 {
            return;
        }
        if let Some((ticks, _)) = self.scale {
            self.scale = Some((ticks, H264_PICTURE_COUNTING_POC_STEP));
        }
    }

    /// The presentation time of an unstamped unit whose count is `poc`, once an
    /// anchor and a slope are both known.
    fn synthesize(&self, poc: i64) -> Option<u64> {
        let (anchor_poc, anchor_pts) = self.anchor?;
        let (ticks, counts) = self.scale?;
        let distance = poc - anchor_poc;
        let offset = distance.unsigned_abs().checked_mul(ticks)? / counts;
        Some(if distance >= 0 {
            anchor_pts.saturating_add(offset)
        } else {
            anchor_pts.saturating_sub(offset)
        })
    }
}

/// How many picture order counts one coded frame of `codec` advances.
fn poc_step_per_frame(codec: VideoCodec) -> u64 {
    match codec {
        VideoCodec::H265 => H265_POC_STEP_PER_FRAME,
        _ => H264_POC_STEP_PER_FRAME,
    }
}

/// Express a measured `ticks` per `counts` slope as the declared frame period
/// over a whole count step per frame, when the measurement implies a plausible
/// one. Otherwise the measurement stands.
fn snap_to_declared_period(ticks: u64, counts: u64, declared_period_90: Option<u64>) -> (u64, u64) {
    let Some(period) = declared_period_90 else {
        return (ticks, counts);
    };
    let step = (period.saturating_mul(counts).saturating_add(ticks / 2)) / ticks;
    if (1..=MAX_POC_STEP_PER_FRAME).contains(&step) {
        (period, step)
    } else {
        (ticks, counts)
    }
}

/// The frame period in 90 kHz units for a Q16 fixed-point frame rate, rounded
/// to the nearest tick. `None` for a rate of zero, or one so high the period
/// rounds away.
pub(crate) fn frame_period_90khz(framerate_q16: u32) -> Option<u64> {
    let q16 = u64::from(framerate_q16);
    if q16 == 0 {
        return None;
    }
    Some(((90_000u64 << 16) + q16 / 2) / q16).filter(|period| *period > 0)
}

/// A PES packet being reassembled across TS packets for one PID.
#[derive(Debug)]
struct PendingPes {
    pid: u16,
    stream_type: u8,
    pts_90khz: Option<u64>,
    dts_90khz: Option<u64>,
    data: Vec<u8>,
}

/// One program declared by the PAT: its `program_number`, the PID its PMT rides
/// on, and the elementary streams that PMT names (empty until the PMT parses).
#[derive(Debug, Clone)]
struct TsProgram {
    number: u16,
    pmt_pid: u16,
    streams: Vec<ElementaryStream>,
}

/// Incremental MPEG-TS demuxer: feed 188-byte packets, drain [`EsUnit`]s.
#[derive(Debug, Default)]
pub struct TsDemuxer {
    programs: Vec<TsProgram>,
    /// Program to route (`None` = the first in PAT order). A set number with no
    /// matching program routes nothing (strict, no fallback to the first).
    selected_program: Option<u16>,
    /// SDT services as `(service_id, text)`, in section order (M872). Kept apart
    /// from `programs` because the SDT may parse before the PAT; a service_id is a
    /// `program_number`, so [`service`](Self::service) joins the two.
    services: Vec<(u16, ServiceInfo)>,
    /// EIT present/following events (M1049), at most one per
    /// `(service_id, slot)`: a later section of the same slot replaces the
    /// earlier one, so this always holds what the services are showing.
    eit_events: Vec<EitEvent>,
    /// EIT schedule events (M1056), queued in arrival order for
    /// [`take_eit_schedule`](Self::take_eit_schedule) to drain. A schedule table
    /// names days of events rather than the two present/following holds, so they
    /// are handed over once instead of kept in a replace-in-place table.
    eit_schedule: Vec<EitEvent>,
    /// The `version_number` last accepted per
    /// `(service_id, table_id, section_number)`. A section repeating its version
    /// carries the events already reported, so it is dropped rather than
    /// re-reported.
    eit_versions: Vec<(u16, u8, u8, u8)>,
    /// Bumped whenever [`eit_events`](Self::eit_events) changes, so the element
    /// layer can re-post without diffing.
    eit_generation: u64,
    /// The EIT section being reassembled across TS packets. Unlike the PAT / PMT /
    /// SDT, an EIT section routinely outgrows one packet.
    eit_section: Vec<u8>,
    pending: Vec<PendingPes>,
    completed: Vec<EsUnit>,
    /// Video PTS synthesis state, one entry per video PID (M948).
    video_pts: Vec<(u16, VideoPtsSynth)>,
}

impl TsDemuxer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Select which PAT program to demux by `program_number` (`None` = the first
    /// in PAT order, the default). A number with no matching program routes
    /// nothing until such a program appears.
    pub fn set_program_number(&mut self, n: Option<u16>) {
        self.selected_program = n;
    }

    /// The active program: the one whose `program_number` matches the selection,
    /// or the first in PAT order when none is selected. `None` before the PAT
    /// parses, or when a selected number matches no program.
    fn active_program(&self) -> Option<&TsProgram> {
        match self.selected_program {
            Some(n) => self.programs.iter().find(|p| p.number == n),
            None => self.programs.first(),
        }
    }

    /// The elementary streams the active program's PMT announced (empty until that
    /// PMT is seen, or when the selected program has no match).
    pub fn streams(&self) -> &[ElementaryStream] {
        self.active_program().map_or(&[], |p| &p.streams)
    }

    /// Every parsed program as `(program_number, streams)`, in PAT order; each
    /// stream list is empty until that program's PMT parses. The multi-program
    /// introspection point for the element layer.
    pub fn programs(&self) -> impl Iterator<Item = (u16, &[ElementaryStream])> + '_ {
        self.programs
            .iter()
            .map(|p| (p.number, p.streams.as_slice()))
    }

    /// The active program's SDT service text (M872), if an SDT named a service
    /// whose `service_id` matches that program's number. `None` for a stream with
    /// no SDT, or before both tables parse.
    pub fn service(&self) -> Option<&ServiceInfo> {
        let number = self.active_program()?.number;
        self.services
            .iter()
            .find(|(id, _)| *id == number)
            .map(|(_, s)| s)
    }

    /// Every service the SDT named, as `(program_number, text)` in section order
    /// (M878): the SDT describes the whole transport stream, so a multi-program
    /// multiplex reports one entry per program regardless of which one is selected.
    /// Empty for a stream with no SDT.
    pub fn services(&self) -> impl Iterator<Item = (u16, &ServiceInfo)> + '_ {
        self.services.iter().map(|(id, s)| (*id, s))
    }

    /// Every EIT present/following event parsed so far (M1049), in arrival order.
    /// Empty for a stream with no EIT, or before the table parses.
    pub fn eit_events(&self) -> &[EitEvent] {
        &self.eit_events
    }

    /// Take every EIT schedule event parsed since the last call, in arrival order
    /// (M1056), leaving the queue empty. Empty for a stream carrying no schedule
    /// table, and for one whose schedule sections have all been read already.
    pub fn take_eit_schedule(&mut self) -> Vec<EitEvent> {
        core::mem::take(&mut self.eit_schedule)
    }

    /// A counter bumped every time [`eit_events`](Self::eit_events) changes. A
    /// caller that posts the events compares it with the value it last posted at,
    /// so a table repeating its `version_number` costs nothing.
    pub fn eit_generation(&self) -> u64 {
        self.eit_generation
    }

    /// Opus channel count for `pid`, if its PMT entry is a private (0x06) stream
    /// carrying the 'Opus' registration descriptor; `None` for any other stream.
    pub fn opus_channels(&self, pid: u16) -> Option<u8> {
        self.streams()
            .iter()
            .find(|s| s.pid == pid)
            .and_then(|s| s.opus_channels)
    }

    /// Whether `pid` is a private (0x06) stream carrying an AC-3 descriptor (the
    /// DVB AC-3 carriage); `false` for any other stream.
    pub fn is_dvb_ac3(&self, pid: u16) -> bool {
        self.streams()
            .iter()
            .find(|s| s.pid == pid)
            .is_some_and(|s| s.ac3)
    }

    /// Whether `pid` is a KLV metadata stream (a private 0x06 stream with a 'KLVA'
    /// registration, or a metadata-in-PES 0x15 stream); `false` for any other.
    pub fn is_klv(&self, pid: u16) -> bool {
        self.streams()
            .iter()
            .find(|s| s.pid == pid)
            .is_some_and(|s| s.klv)
    }

    /// Whether `pid` is an AV1 video stream (a private 0x06 stream with an AV1
    /// registration descriptor, M1049); `false` for any other stream.
    pub fn is_av1(&self, pid: u16) -> bool {
        self.streams()
            .iter()
            .find(|s| s.pid == pid)
            .is_some_and(|s| s.av1)
    }

    /// The DVB `subtitling_descriptor` entry of `pid`, if its PMT entry is a
    /// private (0x06) stream carrying one; `None` for any other stream.
    pub fn subtitling(&self, pid: u16) -> Option<(u8, u16, u16)> {
        self.streams()
            .iter()
            .find(|s| s.pid == pid)
            .and_then(|s| s.subtitling)
    }

    /// The teletext service of `pid`, if its PMT entry is a private (0x06) stream
    /// carrying a `teletext_descriptor`; `None` for any other stream.
    pub fn teletext(&self, pid: u16) -> Option<TeletextService> {
        self.streams()
            .iter()
            .find(|s| s.pid == pid)
            .and_then(|s| s.teletext)
    }

    /// The PID of the first video elementary stream (H.264 or H.265), if any.
    pub fn video_pid(&self) -> Option<u16> {
        self.streams()
            .iter()
            .find(|s| s.stream_type == STREAM_TYPE_H264 || s.stream_type == STREAM_TYPE_H265)
            .map(|s| s.pid)
    }

    /// Feed one TS packet (must be [`TS_PACKET_LEN`] bytes starting at the sync
    /// byte). Malformed or short packets are ignored. Completed access units
    /// accumulate; drain them with [`take_units`](Self::take_units).
    pub fn push_packet(&mut self, pkt: &[u8]) {
        if pkt.len() < TS_PACKET_LEN || pkt[0] != SYNC_BYTE {
            return;
        }
        let pusi = pkt[1] & 0x40 != 0;
        let pid = (((pkt[1] & 0x1F) as u16) << 8) | pkt[2] as u16;
        let afc = (pkt[3] >> 4) & 0x03;

        // Locate the payload after any adaptation field.
        let mut off = 4;
        if afc & 0x02 != 0 {
            // adaptation field present: its length byte, then that many bytes.
            let af_len = pkt[4] as usize;
            off = 5 + af_len;
        }
        if afc & 0x01 == 0 || off >= TS_PACKET_LEN {
            return; // no payload
        }
        let payload = &pkt[off..TS_PACKET_LEN];

        if pid == PID_PAT {
            self.parse_pat(payload, pusi);
        } else if pid == PID_SDT {
            self.parse_sdt(payload, pusi);
        } else if pid == PID_EIT {
            self.parse_eit(payload, pusi);
        } else if let Some(idx) = self.programs.iter().position(|p| p.pmt_pid == pid) {
            self.parse_pmt(idx, payload, pusi);
        } else if let Some(stream_type) = self.stream_type_of(pid) {
            self.accumulate_pes(pid, stream_type, payload, pusi);
        }
    }

    /// Drain the access units completed so far, synthesizing a PTS for each
    /// unstamped video unit that qualifies (see `VideoPtsSynth`).
    pub fn take_units(&mut self) -> Vec<EsUnit> {
        let mut units = core::mem::take(&mut self.completed);
        for unit in units.iter_mut() {
            self.synthesize_video_pts(unit);
        }
        units
    }

    /// Run one unit through its stream's [`VideoPtsSynth`]. A non-video unit,
    /// and any video PID past [`MAX_VIDEO_PTS_STREAMS`], passes through.
    fn synthesize_video_pts(&mut self, unit: &mut EsUnit) {
        let codec = match unit.stream_type {
            STREAM_TYPE_H264 => VideoCodec::H264,
            STREAM_TYPE_H265 => VideoCodec::H265,
            STREAM_TYPE_MPEG1_VIDEO | STREAM_TYPE_MPEG2_VIDEO => VideoCodec::Mpeg2,
            _ => return,
        };
        let slot = match self.video_pts.iter().position(|(pid, _)| *pid == unit.pid) {
            Some(slot) => slot,
            None if self.video_pts.len() < MAX_VIDEO_PTS_STREAMS => {
                self.video_pts.push((unit.pid, VideoPtsSynth::default()));
                self.video_pts.len() - 1
            }
            None => return,
        };
        self.video_pts[slot]
            .1
            .stamp(codec, &unit.data, &mut unit.pts_90khz, &mut unit.dts_90khz);
    }

    /// Finalize any PES still being reassembled (call at end of stream). The
    /// units land in the queue drained by [`take_units`](Self::take_units).
    pub fn flush(&mut self) {
        for p in core::mem::take(&mut self.pending) {
            Self::finish(&mut self.completed, p);
        }
    }

    fn stream_type_of(&self, pid: u16) -> Option<u8> {
        // Route PES through the active program only (PIDs may recur across
        // programs; only the selected one's streams accumulate).
        self.streams()
            .iter()
            .find(|s| s.pid == pid)
            .map(|s| s.stream_type)
    }

    /// Skip the PSI `pointer_field` and return the section bytes, or `None` if
    /// this packet does not start a section (a non-PUSI continuation, which v1
    /// does not reassemble).
    fn section(payload: &[u8], pusi: bool) -> Option<&[u8]> {
        if !pusi || payload.is_empty() {
            return None;
        }
        let pointer = payload[0] as usize;
        payload.get(1 + pointer..)
    }

    /// The PSI section payload bounds: bytes `[3 .. 3 + section_length - 4]`
    /// (after the table-id + length header, before the trailing 4-byte CRC).
    fn section_body(section: &[u8]) -> Option<&[u8]> {
        if section.len() < 3 {
            return None;
        }
        let section_length = (((section[1] & 0x0F) as usize) << 8) | section[2] as usize;
        let total = 3 + section_length;
        if section_length < 4 + 5 || total > section.len() {
            return None;
        }
        // Body excludes the 8-byte common header start we index from section[3],
        // and the 4-byte CRC at the end.
        section.get(..total - 4)
    }

    /// The section body of a section whose trailing MPEG-2 CRC-32 checks out (the
    /// CRC over a section including its own 4 CRC bytes is 0 when it matches),
    /// else `None`. Used for the SDT and the EIT: their PIDs carry several tables, a
    /// section can start mid-stream, and unlike a PAT/PMT PID nothing later
    /// cross-checks the text they carry, so a bad section must not become a tag.
    fn checked_section_body(section: &[u8]) -> Option<&[u8]> {
        let body = Self::section_body(section)?;
        let whole = section.get(..body.len() + 4)?;
        (mpeg_crc32(whole) == 0).then_some(body)
    }

    fn parse_pat(&mut self, payload: &[u8], pusi: bool) {
        if !self.programs.is_empty() {
            return; // first PAT wins (no version / update handling)
        }
        let Some(section) = Self::section(payload, pusi) else {
            return;
        };
        if section.first() != Some(&0x00) {
            return; // table_id 0x00 = PAT
        }
        let Some(body) = Self::section_body(section) else {
            return;
        };
        // Program loop starts at section[8] (after the 8-byte PSI header). The
        // body is already CRC-bounded by section_body, so the walk is bounded.
        let mut i = 8;
        while i + 4 <= body.len() {
            let program_number = ((body[i] as u16) << 8) | body[i + 1] as u16;
            let pid = (((body[i + 2] & 0x1F) as u16) << 8) | body[i + 3] as u16;
            // program_number 0 is the NIT pointer, not a program: skip it.
            if program_number != 0 {
                self.programs.push(TsProgram {
                    number: program_number,
                    pmt_pid: pid,
                    streams: Vec::new(),
                });
            }
            i += 4;
        }
    }

    /// Parse an SDT section (PID 0x11, `table_id` 0x42) into the per-service
    /// name / provider text (M872): the service loop follows the 8-byte PSI header,
    /// `original_network_id` and a reserved byte, and each entry's descriptor loop
    /// carries the DVB `service_descriptor`. First SDT wins, the PAT / PMT
    /// discipline (no version handling). The CRC is verified
    /// ([`checked_section_body`](Self::checked_section_body)) and every length is
    /// bounds-checked, so a malformed or unrelated section on this PID is ignored
    /// rather than yielding garbled text.
    fn parse_sdt(&mut self, payload: &[u8], pusi: bool) {
        if !self.services.is_empty() {
            return;
        }
        let Some(section) = Self::section(payload, pusi) else {
            return;
        };
        if section.first() != Some(&TABLE_ID_SDT) {
            return; // an other-TS SDT (0x46), a BAT or an EIT sharing the PID
        }
        let Some(body) = Self::checked_section_body(section) else {
            return;
        };
        let mut i = 11usize;
        while i + 5 <= body.len() {
            let service_id = ((body[i] as u16) << 8) | body[i + 1] as u16;
            let loop_len = (((body[i + 3] & 0x0F) as usize) << 8) | body[i + 4] as usize;
            // A descriptor loop declared past the section end abandons the walk:
            // the count is attacker-controlled.
            let Some(desc) = body.get(i + 5..i + 5 + loop_len) else {
                return;
            };
            if let Some(info) = parse_service_descriptor(desc) {
                self.services.push((service_id, info));
            }
            i = i.saturating_add(5).saturating_add(loop_len);
        }
    }

    /// Collect one EIT section across TS packets (M1049), returning it once whole.
    /// The PAT / PMT / SDT are read from a single packet, which holds for those
    /// small tables, but an EIT event carries free text and routinely spans
    /// several. A section is only buffered once its `table_id` is the
    /// present/following one or a schedule one of this transport stream, which
    /// keeps the other-TS tables sharing this PID out of the buffer entirely.
    ///
    /// The declared `section_length` is attacker-controlled, so the buffer is
    /// capped and a section that overruns is abandoned; the next
    /// payload-unit-start resyncs.
    fn collect_eit_section(&mut self, payload: &[u8], pusi: bool) -> Option<Vec<u8>> {
        if pusi {
            self.eit_section.clear();
            let pointer = *payload.first()? as usize;
            let start = payload.get(1 + pointer..)?;
            let table_id = *start.first()?;
            if table_id != TABLE_ID_EIT_PRESENT_FOLLOWING
                && !TABLE_ID_EIT_SCHEDULE.contains(&table_id)
            {
                return None; // an other-TS EIT or stuffing
            }
            self.eit_section.extend_from_slice(start);
        } else if self.eit_section.is_empty() {
            return None; // no section open on this PID
        } else {
            self.eit_section.extend_from_slice(payload);
        }
        if self.eit_section.len() > MAX_PSI_SECTION_BYTES {
            self.eit_section.clear();
            return None;
        }
        if self.eit_section.len() < 3 {
            return None;
        }
        let total =
            3 + ((((self.eit_section[1] & 0x0F) as usize) << 8) | self.eit_section[2] as usize);
        if total > MAX_PSI_SECTION_BYTES {
            self.eit_section.clear();
            return None;
        }
        if self.eit_section.len() < total {
            return None; // more packets to come
        }
        let mut section = core::mem::take(&mut self.eit_section);
        section.truncate(total); // drop the 0xFF stuffing after the section
        Some(section)
    }

    /// Parse a DVB EIT section (PID 0x12) into the per-service event text, start
    /// time and duration (M1049, M1056). The event loop follows the 14-byte
    /// header, and each entry's descriptor loop carries the
    /// `short_event_descriptor`. `table_id` 0x4E is present/following, whose
    /// section 0 is the event on air and section 1 the one after it; 0x50..=0x5F
    /// are the schedule tables, which carry many events per section and are
    /// segmented, so any `section_number` is valid there.
    ///
    /// Unlike the PAT / PMT / SDT these tables update mid-stream, so rather than
    /// "first wins" a section is read when its `version_number` differs from the
    /// one last accepted for the same `(service_id, table_id, section_number)`.
    /// The CRC is verified ([`checked_section_body`](Self::checked_section_body))
    /// and every length is bounds-checked, so a malformed section is ignored
    /// rather than yielding garbled text.
    fn parse_eit(&mut self, payload: &[u8], pusi: bool) {
        let Some(section) = self.collect_eit_section(payload, pusi) else {
            return;
        };
        let Some(body) = Self::checked_section_body(&section) else {
            return;
        };
        // The 14-byte EIT header: the 8-byte PSI common header, then
        // transport_stream_id, original_network_id, segment_last_section_number
        // and last_table_id.
        if body.len() < 14 {
            return;
        }
        let table_id = body[0];
        let service_id = ((body[3] as u16) << 8) | body[4] as u16;
        // current_next_indicator 0 marks a version not yet in force: its events
        // are not what the service is showing.
        if body[5] & 0x01 == 0 {
            return;
        }
        let version = (body[5] >> 1) & 0x1F;
        let section_number = body[6];
        let slot = if TABLE_ID_EIT_SCHEDULE.contains(&table_id) {
            EitSlot::Schedule
        } else if section_number == 0 {
            EitSlot::Present
        } else if section_number == 1 {
            EitSlot::Following
        } else {
            return; // present/following is sections 0 and 1 only
        };
        if !self.accept_eit_version(service_id, table_id, section_number, version) {
            return;
        }
        let mut i = 14usize;
        // Each event: event_id, a 5-byte start_time, a 3-byte duration, then
        // running_status / free_CA_mode / descriptors_loop_length.
        while i + 12 <= body.len() {
            let event_id = ((body[i] as u16) << 8) | body[i + 1] as u16;
            let start_unix_secs = mjd_bcd_to_unix_secs([
                body[i + 2],
                body[i + 3],
                body[i + 4],
                body[i + 5],
                body[i + 6],
            ]);
            let duration_secs =
                bcd_hms_secs([body[i + 7], body[i + 8], body[i + 9]]).unwrap_or(0) as u32;
            let loop_len = (((body[i + 10] & 0x0F) as usize) << 8) | body[i + 11] as usize;
            // A descriptor loop declared past the section end abandons the walk:
            // the count is attacker-controlled.
            let Some(desc) = body.get(i + 12..i + 12 + loop_len) else {
                return;
            };
            if let Some((name, text)) = parse_short_event_descriptor(desc) {
                self.record_eit_event(EitEvent {
                    service_id,
                    event_id,
                    slot,
                    name,
                    text,
                    start_unix_secs,
                    duration_secs,
                });
            }
            i = i.saturating_add(12).saturating_add(loop_len);
        }
    }

    /// Whether an EIT section carries a version not yet read, recording it when so.
    /// Past [`MAX_EIT_VERSION_SLOTS`] the list stops growing, so a stream churning
    /// `service_id`s cannot cost unbounded memory.
    fn accept_eit_version(
        &mut self,
        service_id: u16,
        table_id: u8,
        section_number: u8,
        version: u8,
    ) -> bool {
        if let Some(slot) = self
            .eit_versions
            .iter_mut()
            .find(|(id, table, section, _)| {
                *id == service_id && *table == table_id && *section == section_number
            })
        {
            if slot.3 == version {
                return false;
            }
            slot.3 = version;
            return true;
        }
        if self.eit_versions.len() >= MAX_EIT_VERSION_SLOTS {
            return false;
        }
        self.eit_versions
            .push((service_id, table_id, section_number, version));
        true
    }

    /// Record one event. A present/following event replaces whatever the same
    /// `(service_id, slot)` held and bumps the generation so the element layer
    /// re-posts; a schedule event joins the drain queue, and is dropped once that
    /// queue is full so the older events survive.
    fn record_eit_event(&mut self, event: EitEvent) {
        if event.slot == EitSlot::Schedule {
            if self.eit_schedule.len() < MAX_EIT_SCHEDULE_EVENTS {
                self.eit_schedule.push(event);
            }
            return;
        }
        let held = self
            .eit_events
            .iter()
            .position(|e| e.service_id == event.service_id && e.slot == event.slot);
        match held {
            Some(i) => self.eit_events[i] = event,
            None if self.eit_events.len() < MAX_EIT_EVENTS => self.eit_events.push(event),
            None => return,
        }
        self.eit_generation = self.eit_generation.saturating_add(1);
    }

    fn parse_pmt(&mut self, prog_idx: usize, payload: &[u8], pusi: bool) {
        if !self.programs[prog_idx].streams.is_empty() {
            return; // first PMT per program wins
        }
        let Some(section) = Self::section(payload, pusi) else {
            return;
        };
        if section.first() != Some(&0x02) {
            return; // table_id 0x02 = PMT
        }
        let Some(body) = Self::section_body(section) else {
            return;
        };
        if body.len() < 12 {
            return;
        }
        let program_info_length = (((body[10] & 0x0F) as usize) << 8) | body[11] as usize;
        let mut i = 12usize.saturating_add(program_info_length);
        while i + 5 <= body.len() {
            let stream_type = body[i];
            let pid = (((body[i + 1] & 0x1F) as u16) << 8) | body[i + 2] as u16;
            let es_info_length = (((body[i + 3] & 0x0F) as usize) << 8) | body[i + 4] as usize;
            // Bounds-check the descriptor slice: a bogus es_info_length must not
            // read past the section body (the count is attacker-controlled).
            let descriptors = body.get(i + 5..i + 5 + es_info_length);
            // Only a private (0x06) stream needs its descriptors inspected to tell
            // Opus / AC-3 / KLV apart from an unidentified private stream; the
            // language descriptor rides any stream type.
            let private = descriptors.filter(|_| stream_type == STREAM_TYPE_PRIVATE_PES);
            let opus_channels = private.and_then(parse_opus_descriptors);
            let ac3 = private.is_some_and(has_ac3_descriptor);
            // A metadata-in-PES (0x15) stream is KLV without needing a descriptor
            // (the ffmpeg convention); a private 0x06 needs the 'KLVA' registration.
            let klv = stream_type == STREAM_TYPE_METADATA_PES
                || private.is_some_and(has_klv_registration);
            self.programs[prog_idx].streams.push(ElementaryStream {
                pid,
                stream_type,
                opus_channels,
                ac3,
                klv,
                av1: private.is_some_and(has_av1_registration),
                subtitling: private.and_then(parse_subtitling_descriptor),
                teletext: private.and_then(parse_teletext_descriptor),
                language: descriptors.and_then(parse_iso639_language),
            });
            i = i.saturating_add(5).saturating_add(es_info_length);
        }
    }

    fn accumulate_pes(&mut self, pid: u16, stream_type: u8, payload: &[u8], pusi: bool) {
        if pusi {
            // A new PES starts: finalize the previous one for this PID.
            if let Some(idx) = self.pending.iter().position(|p| p.pid == pid) {
                let prev = self.pending.swap_remove(idx);
                Self::finish(&mut self.completed, prev);
            }
            let (pts, dts, es) = parse_pes_header(payload);
            self.pending.push(PendingPes {
                pid,
                stream_type,
                pts_90khz: pts,
                dts_90khz: dts,
                data: es.to_vec(),
            });
        } else if let Some(idx) = self.pending.iter().position(|p| p.pid == pid) {
            // Continuation of the current PES. Drop a PES that overruns the cap
            // rather than growing the buffer on an endless continuation run; the
            // next payload-unit-start resyncs a fresh PES on this PID.
            if self.pending[idx].data.len().saturating_add(payload.len()) > MAX_PES_BYTES {
                self.pending.swap_remove(idx);
            } else {
                self.pending[idx].data.extend_from_slice(payload);
            }
        }
    }

    fn finish(completed: &mut Vec<EsUnit>, p: PendingPes) {
        if p.data.is_empty() {
            return;
        }
        completed.push(EsUnit {
            pid: p.pid,
            stream_type: p.stream_type,
            pts_90khz: p.pts_90khz,
            dts_90khz: p.dts_90khz,
            data: p.data,
        });
    }
}

/// Parse a PES packet header at the start of `payload`, returning the PTS and
/// DTS (each if present) and the elementary-stream bytes after the header. A DTS
/// only rides a PES that also carries a PTS (`PTS_DTS_flags == '11'`). If the
/// start code or optional header is malformed, returns the whole payload with no
/// timestamps (so a best-effort stream still flows).
pub(crate) fn parse_pes_header(payload: &[u8]) -> (Option<u64>, Option<u64>, &[u8]) {
    // PES: 00 00 01, stream_id, PES_packet_length(2), then for media stream_ids
    // an optional header: flags(2) + PES_header_data_length(1) + that many bytes.
    if payload.len() < 9 || payload[0] != 0x00 || payload[1] != 0x00 || payload[2] != 0x01 {
        return (None, None, payload);
    }
    // byte 6 must have the '10' marker bits for an optional PES header.
    if payload[6] & 0xC0 != 0x80 {
        return (None, None, &payload[6..]);
    }
    let pts_dts_flags = (payload[7] >> 6) & 0x03;
    let header_data_len = payload[8] as usize;
    let es_start = 9 + header_data_len;
    if es_start > payload.len() {
        return (None, None, payload);
    }
    let pts = if pts_dts_flags & 0x02 != 0 && payload.len() >= 14 {
        Some(decode_timestamp(&payload[9..14]))
    } else {
        None
    };
    // A DTS follows the PTS (bytes 14..19) only when both flags are set. Malformed
    // (too short) keeps the PTS-only best-effort behavior.
    let dts = if pts_dts_flags == 0b11 && payload.len() >= 19 {
        Some(decode_timestamp(&payload[14..19]))
    } else {
        None
    };
    (pts, dts, &payload[es_start..])
}

/// Decode a 33-bit MPEG PTS/DTS from its 5-byte field (90 kHz units).
pub(crate) fn decode_timestamp(b: &[u8]) -> u64 {
    (((b[0] >> 1) & 0x07) as u64) << 30
        | (b[1] as u64) << 22
        | (((b[2] >> 1) & 0x7F) as u64) << 15
        | (b[3] as u64) << 7
        | ((b[4] >> 1) & 0x7F) as u64
}

/// Walk a PMT ES-info descriptor list for the Opus carriage (DVB/ETSI): an
/// `registration_descriptor` (tag 0x05) with format_identifier "Opus", plus a
/// DVB `extension_descriptor` (tag 0x7F) whose extension tag is 0x80 carrying the
/// `channel_config_code`. Returns the Opus channel count when the registration is
/// present (`code == 0` is dual-mono, mapped to 2 channels; `1..=8` is the count;
/// a missing/unknown extension defaults to stereo), else `None`. Every field is
/// bounds-checked so a malformed descriptor loop fails to `None`, never panics.
fn parse_opus_descriptors(mut desc: &[u8]) -> Option<u8> {
    let mut is_opus = false;
    let mut channels: Option<u8> = None;
    while desc.len() >= 2 {
        let tag = desc[0];
        let len = desc[1] as usize;
        let body = desc.get(2..2 + len)?;
        match tag {
            // registration_descriptor: format_identifier is the first 4 bytes.
            DESC_TAG_REGISTRATION if body.len() >= 4 && &body[..4] == b"Opus" => is_opus = true,
            // DVB extension_descriptor: ext tag 0x80 (provisional Opus) + code.
            0x7F if body.len() >= 2 && body[0] == 0x80 => {
                let code = body[1];
                channels = Some(if code == 0 { 2 } else { code });
            }
            _ => {}
        }
        desc = &desc[2 + len..];
    }
    is_opus.then(|| channels.unwrap_or(2))
}

/// Whether a PMT ES-info descriptor list carries an AC-3 descriptor (tag 0x6A,
/// the DVB carriage of AC-3) or an 'AC-3' registration descriptor (tag 0x05).
/// Every field is bounds-checked so a malformed loop returns `false`, never
/// panics.
fn has_ac3_descriptor(mut desc: &[u8]) -> bool {
    while desc.len() >= 2 {
        let tag = desc[0];
        let len = desc[1] as usize;
        let Some(body) = desc.get(2..2 + len) else {
            return false;
        };
        match tag {
            0x6A => return true, // DVB AC-3_descriptor
            DESC_TAG_REGISTRATION if body.len() >= 4 && &body[..4] == b"AC-3" => return true,
            _ => {}
        }
        desc = &desc[2 + len..];
    }
    false
}

/// Whether a PMT ES-info descriptor list carries a 'KLVA' registration descriptor
/// (tag 0x05), the MISB ST 1402 marker for asynchronous KLV metadata on a private
/// (0x06) stream. Every field is bounds-checked so a malformed loop returns
/// `false`, never panics.
fn has_klv_registration(mut desc: &[u8]) -> bool {
    while desc.len() >= 2 {
        let tag = desc[0];
        let len = desc[1] as usize;
        let Some(body) = desc.get(2..2 + len) else {
            return false;
        };
        if tag == DESC_TAG_REGISTRATION && body.len() >= 4 && &body[..4] == b"KLVA" {
            return true;
        }
        desc = &desc[2 + len..];
    }
    false
}

/// Whether a PMT ES-info descriptor list carries an AV1 `registration_descriptor`
/// (tag 0x05), the AOM marker for AV1 video on a private (0x06) stream (M1049).
/// Both the spec's 'AV01' and GStreamer's 'AV1G' are accepted, since only the
/// latter has a live reader. The AV1_video_descriptor (tag 0x80) the spec pairs
/// with the registration is not required: it carries profile / bit-depth fields
/// the bitstream repeats in its sequence header, which is where `av1parse` reads
/// them, and GStreamer's demuxer likewise keys on the registration alone. Every
/// field is bounds-checked so a malformed loop returns `false`, never panics.
fn has_av1_registration(mut desc: &[u8]) -> bool {
    while desc.len() >= 2 {
        let tag = desc[0];
        let len = desc[1] as usize;
        let Some(body) = desc.get(2..2 + len) else {
            return false;
        };
        if tag == DESC_TAG_REGISTRATION
            && (body.starts_with(AV1_REGISTRATION_ID)
                || body.starts_with(AV1_REGISTRATION_ID_GSTREAMER))
        {
            return true;
        }
        desc = &desc[2 + len..];
    }
    false
}

/// The name and short description of an EIT event's DVB `short_event_descriptor`
/// (tag 0x4D, M1049): a 3-letter language code, then the length-prefixed
/// `event_name_char` and `text_char`. `None` when the loop names no such
/// descriptor or a field runs past it; a field this parser will not decode comes
/// back empty rather than garbled (see [`dvb_text`]). Bounds-checked throughout.
fn parse_short_event_descriptor(mut desc: &[u8]) -> Option<(String, String)> {
    while desc.len() >= 2 {
        let tag = desc[0];
        let len = desc[1] as usize;
        let body = desc.get(2..2 + len)?;
        if tag == DESC_TAG_SHORT_EVENT {
            let name_len = *body.get(3)? as usize;
            let name = body.get(4..4 + name_len)?;
            let text_len = *body.get(4 + name_len)? as usize;
            let text = body.get(5 + name_len..5 + name_len + text_len)?;
            return Some((
                dvb_text(name).unwrap_or_default(),
                dvb_text(text).unwrap_or_default(),
            ));
        }
        desc = &desc[2 + len..];
    }
    None
}

/// The first entry of a PMT ES-info DVB `subtitling_descriptor` (tag 0x59), the
/// marker for DVB subtitles on a private (0x06) stream: 8 bytes per subtitle
/// stream, a 3-letter language code, a subtitling_type, then the composition and
/// ancillary page ids. Only the first entry is kept, since one decoder composes
/// one page. Every field is bounds-checked so a malformed loop returns `None`,
/// never panics.
fn parse_subtitling_descriptor(mut desc: &[u8]) -> Option<(u8, u16, u16)> {
    while desc.len() >= 2 {
        let tag = desc[0];
        let len = desc[1] as usize;
        let body = desc.get(2..2 + len)?;
        if tag == DESC_TAG_SUBTITLING && body.len() >= 8 {
            return Some((
                body[3],
                u16::from_be_bytes([body[4], body[5]]),
                u16::from_be_bytes([body[6], body[7]]),
            ));
        }
        desc = &desc[2 + len..];
    }
    None
}

/// The teletext service a PMT ES-info DVB `teletext_descriptor` (tag 0x56, or the
/// identical `VBI_teletext_descriptor` 0x46) names, the marker for EBU teletext on
/// a private (0x06) stream: 5 bytes per service, a 3-letter language code then a
/// packed teletext_type / magazine byte and the BCD page number. A descriptor
/// listing several services yields its first subtitle page, else its first entry,
/// since one decoder composes one page. Every field is bounds-checked so a
/// malformed loop returns `None`, never panics.
fn parse_teletext_descriptor(mut desc: &[u8]) -> Option<TeletextService> {
    while desc.len() >= 2 {
        let tag = desc[0];
        let len = desc[1] as usize;
        let body = desc.get(2..2 + len)?;
        if tag == DESC_TAG_TELETEXT || tag == DESC_TAG_VBI_TELETEXT {
            let mut first = None;
            for entry in body.as_chunks::<5>().0 {
                let service = TeletextService {
                    language: [entry[0], entry[1], entry[2]],
                    teletext_type: entry[3] >> 3,
                    magazine: entry[3] & 0x07,
                    page: entry[4],
                };
                if service.is_subtitle() {
                    return Some(service);
                }
                first.get_or_insert(service);
            }
            if let Some(service) = first {
                return Some(service);
            }
        }
        desc = &desc[2 + len..];
    }
    None
}

/// The first language of a PMT ES-info `ISO_639_language_descriptor` (tag 0x0A):
/// 4 bytes per language, a 3-letter code then an audio_type byte. Only the first
/// is kept, since a [`g2g_core::Tag::Language`] holds one. `None` when the
/// descriptor is absent, truncated, or its code is not three ASCII letters. Every
/// field is bounds-checked so a malformed loop returns `None`, never panics.
fn parse_iso639_language(mut desc: &[u8]) -> Option<[u8; 3]> {
    while desc.len() >= 2 {
        let tag = desc[0];
        let len = desc[1] as usize;
        let body = desc.get(2..2 + len)?;
        if tag == DESC_TAG_ISO639 && body.len() >= 3 {
            let code = [body[0], body[1], body[2]];
            if code.iter().all(u8::is_ascii_alphabetic) {
                return Some(code);
            }
        }
        desc = &desc[2 + len..];
    }
    None
}

/// The service text of an SDT service entry's descriptor loop: the DVB
/// `service_descriptor` (tag 0x48), a service_type byte then two length-prefixed
/// DVB text fields, provider first. `None` when the loop names no service
/// descriptor or a field runs past it; a field this parser will not decode comes
/// back empty rather than garbled (see [`dvb_text`]). Bounds-checked throughout.
fn parse_service_descriptor(mut desc: &[u8]) -> Option<ServiceInfo> {
    while desc.len() >= 2 {
        let tag = desc[0];
        let len = desc[1] as usize;
        let body = desc.get(2..2 + len)?;
        if tag == DESC_TAG_SERVICE {
            let provider_len = *body.get(1)? as usize;
            let provider = body.get(2..2 + provider_len)?;
            let name_len = *body.get(2 + provider_len)? as usize;
            let name = body.get(3 + provider_len..3 + provider_len + name_len)?;
            return Some(ServiceInfo {
                name: dvb_text(name).unwrap_or_default(),
                provider: dvb_text(provider).unwrap_or_default(),
            });
        }
        desc = &desc[2 + len..];
    }
    None
}

/// The character-table selection byte selecting UTF-8 for the rest of a DVB text
/// field (ETSI EN 300 468 table A.3).
const DVB_CHAR_TABLE_UTF8: u8 = 0x15;

/// Decode one DVB text field (ETSI EN 300 468 annex A). A field opening with the
/// UTF-8 selection byte is decoded as UTF-8, and one opening with any other
/// character-table selection byte (0x01..=0x1F) is rejected with `None`: those
/// tables are non-Latin single- or multi-byte, and reporting no name beats
/// reporting a mis-decoded one. The default table (ISO 6937) is read as Latin-1,
/// which agrees for the ASCII real service and event names use and leaves its
/// accent escapes uncomposed; control codes inside the text are dropped.
fn dvb_text(raw: &[u8]) -> Option<String> {
    if raw.first() == Some(&DVB_CHAR_TABLE_UTF8) {
        let text = core::str::from_utf8(raw.get(1..)?).ok()?;
        return Some(text.chars().filter(|c| !is_dvb_control(*c)).collect());
    }
    if raw.first().is_some_and(|&b| b < 0x20) {
        return None;
    }
    Some(
        raw.iter()
            .map(|&b| b as char)
            .filter(|c| !is_dvb_control(*c))
            .collect(),
    )
}

/// Whether a character is one of the DVB text control codes: C0 (below 0x20) or
/// C1 (0x80..=0x9F), both of which select formatting rather than carrying text.
fn is_dvb_control(c: char) -> bool {
    let code = c as u32;
    code < 0x20 || (0x80..=0x9F).contains(&code)
}

/// Unwrap the ISO 13818-1 metadata access-unit cells of one metadata-in-PES
/// (0x15) payload, returning the concatenated cell payloads, or `None` to say
/// "forward the payload unchanged". A cell is a 5-byte header
/// (metadata_service_id, sequence_number, flags, then a big-endian 16-bit
/// AU_cell_data_length) followed by that many bytes.
///
/// The unwrap is validation-gated because the flags-field semantics (which bits
/// mark a cell that starts / ends an AU) are unverified here: a payload that
/// already starts with a KLV key is forwarded raw (ffmpeg-authored streams look
/// like this), and a cell walk is only believed when the cells tile the payload
/// exactly and every cell payload starts with a KLV key. Anything else forwards
/// raw, so a wrong guess cannot corrupt the output. Every length is
/// bounds-checked against the (attacker-controlled) payload before slicing.
pub(crate) fn unwrap_metadata_au_cells(payload: &[u8]) -> Option<Vec<u8>> {
    if payload.starts_with(&KLV_UL_PREFIX) {
        return None;
    }
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < payload.len() {
        let header = payload.get(pos..pos.checked_add(5)?)?;
        let len = ((header[3] as usize) << 8) | header[4] as usize;
        let start = pos.checked_add(5)?;
        let end = start.checked_add(len)?;
        // A cell running past the payload end fails the walk (no exact tiling).
        let cell = payload.get(start..end)?;
        if !cell.starts_with(&KLV_UL_PREFIX) {
            return None;
        }
        out.extend_from_slice(cell);
        pos = end;
    }
    (!out.is_empty()).then_some(out)
}

/// Unwrap the Opus-in-MPEG-TS control-header access units in one PES payload into
/// the raw Opus packets (Opus-in-TS spec / ETSI TS 103 420): each is prefixed by
/// an 11-bit `0x3FF` sync (`hdr & 0xFFE0 == 0x7FE0`), a flags byte
/// (start_trim / end_trim / control_extension), a variable-length `au_size` (a run
/// of `0xFF` plus a final byte), then optional 2-byte trim fields and a
/// control-extension blob, and finally `au_size` bytes of Opus packet. The trim
/// values are read past but not applied (the no-`OpusHead` path decodes untrimmed,
/// RTP-like). Walking stops at the first malformed / truncated header, so a partial
/// tail is dropped rather than over-read; `au_size` is bounds-checked against the
/// payload before slicing.
pub(crate) fn opus_ts_packets(buf: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos + 2 <= buf.len() {
        let hdr = ((buf[pos] as u16) << 8) | buf[pos + 1] as u16;
        if hdr & 0xFFE0 != 0x7FE0 {
            break; // not a control-header prefix
        }
        let flags = buf[pos + 1];
        let start_trim = (flags >> 4) & 1 == 1;
        let end_trim = (flags >> 3) & 1 == 1;
        let control_ext = (flags >> 2) & 1 == 1;
        let mut i = pos + 2;
        // au_size: sum of a 0xFF run then the final (< 0xFF) byte.
        let mut au_size: usize = 0;
        loop {
            let Some(&b) = buf.get(i) else { return out };
            i += 1;
            au_size = au_size.saturating_add(b as usize);
            if b != 0xFF {
                break;
            }
        }
        if start_trim {
            i = i.saturating_add(2);
        }
        if end_trim {
            i = i.saturating_add(2);
        }
        if control_ext {
            let Some(&ext_len) = buf.get(i) else {
                return out;
            };
            i = i.saturating_add(1).saturating_add(ext_len as usize);
        }
        match i.checked_add(au_size) {
            Some(end) if au_size > 0 && end <= buf.len() => {
                out.push(&buf[i..end]);
                pos = end;
            }
            _ => break, // truncated / overrun / empty au: drop the tail
        }
    }
    out
}

// --- Muxing (M114): the inverse of the demuxer above. ---

/// Base PID layout for the mux: program `p`'s PMT rides `MUX_PMT_PID + p`, and
/// elementary stream `i` (global index) rides `MUX_ES_PID + i`. (The demuxer
/// discovers these from the tables, so any values pair.)
const MUX_PMT_PID: u16 = 0x1000;
const MUX_ES_PID: u16 = 0x0100;

/// How far (90 kHz ticks) the PCR is placed ahead of the PTS it rides with:
/// 100 ms. PCR must precede the PTS so the decoder has buffer lead time, and the
/// PTS-PCR distance must stay under the T-STD ~700 ms bound.
const PCR_LEAD_90KHZ: u64 = 9_000;

/// Adaptation-field bytes a PCR costs a packet: the AF length byte, the AF flags
/// byte, and the 6-byte PCR field. The first packet of a PCR-carrying PES loses
/// this much payload room.
const PCR_AF_OVERHEAD: usize = 8;

/// One elementary stream in a [`TsMuxer`]: its PMT stream type, the TS PID it is
/// carried on, its PES `stream_id`, its running continuity counter, and the index
/// of the program whose PMT names it.
#[derive(Debug)]
struct MuxStream {
    stream_type: u8,
    pid: u16,
    stream_id: u8,
    es_cc: u8,
    program: usize,
    /// Running metadata AU cell sequence_number (metadata-in-PES streams only).
    meta_seq: u8,
    /// ES-info descriptor bytes for this stream's PMT entry (the 'KLVA'
    /// registration for asynchronous KLV, the metadata descriptor for
    /// synchronous KLV, an `ISO_639_language_descriptor` when
    /// [`TsMuxer::set_stream_language`] added one; empty otherwise).
    es_info: Vec<u8>,
    /// How many leading bytes of `es_info` are the private stream's identifying
    /// descriptor, so re-declaring the identity replaces exactly those and leaves
    /// any descriptor added separately (a language) in place.
    identity_len: usize,
    /// Set by [`TsMuxer::set_stream_subtitling`]: each access unit is a DVB
    /// display set, so it goes out wrapped in the PES data field EN 300 743
    /// carries it in rather than bare.
    dvb_subtitle: bool,
}

/// One program in a [`TsMuxer`]: its `program_number`, the PID its PMT rides on
/// with that PID's continuity counter, the stream its PCR rides (the PMT's
/// PCR_PID), and the decode clock the last PCR went out at.
#[derive(Debug)]
struct MuxProgram {
    number: u16,
    pmt_pid: u16,
    pmt_cc: u8,
    /// Global stream index carrying this program's PCR (its first stream).
    pcr_stream: usize,
    last_pcr_90khz: Option<u64>,
    /// This program's own SDT service text (M878), overriding the muxer-wide
    /// default set by [`TsMuxer::set_service`].
    service: Option<ServiceInfo>,
}

/// MPEG-TS multiplexer (M114, multi-stream since M207, multi-program since M783):
/// wraps access units in PES packets and 188-byte TS packets, emitting PAT + PMT
/// once up front. The inverse of [`TsDemuxer`]; the [`crate::tsmux::TsMux`]
/// element wraps it. Each elementary stream (e.g. H.264 video + AAC audio) rides
/// its own PID, named by the PMT of the program it belongs to.
///
/// Scope: one or more programs ([`with_programs`](Self::with_programs); the other
/// constructors put every stream in program 1). The PAT names every program, each
/// with its own PMT, and a PCR rides each program's first stream's PID (that PMT's
/// PCR_PID) in the adaptation field of a PES's first TS packet, on the
/// [`pcr_interval_90khz`] cadence. The caller is expected to interleave access
/// units in timestamp order, which [`crate::tsmux::TsMux`] does. The PSI carries a
/// real MPEG-2 CRC-32, so the output is a valid TS.
///
/// [`pcr_interval_90khz`]: Self::set_pcr_interval_90khz
#[derive(Debug)]
pub struct TsMuxer {
    streams: Vec<MuxStream>,
    programs: Vec<MuxProgram>,
    pat_cc: u8,
    /// Continuity counter of the SDT PID, used only with a service set.
    sdt_cc: u8,
    /// The service text the SDT announces for every program that does not name
    /// its own (M872). With this and every program's own service `None`, no SDT
    /// is written at all.
    default_service: Option<ServiceInfo>,
    tables_written: bool,
    /// PAT/PMT re-emission cadence in 90 kHz ticks (`0` = emit once up front, the
    /// default). When set, the table pair is re-emitted before the first access
    /// unit whose PTS is at least this far past the last emission, so a decoder
    /// that joins mid-stream (a tuned-in multicast, an HLS/DASH segment boundary)
    /// finds the PSI without waiting for the start of the stream.
    table_interval_90khz: u64,
    /// PTS (90 kHz) the tables were last emitted at, for the cadence above.
    last_tables_pts: Option<u64>,
    /// PCR insertion cadence in 90 kHz ticks (default 3600, matching GStreamer
    /// mpegtsmux). A PCR is emitted on a program's PCR_PID when the decode clock
    /// (DTS, else PTS) is at least this far past that program's last PCR (and
    /// always on its first clocked AU).
    pcr_interval_90khz: u64,
}

impl TsMuxer {
    /// A single-stream muxer for `stream_type` (e.g. [`STREAM_TYPE_H264`]).
    pub fn new(stream_type: u8) -> Self {
        Self::with_streams(&[stream_type])
    }

    /// A multi-stream muxer: one elementary stream per entry of `stream_types`,
    /// in input order, all in one program numbered 1. Stream `i` is carried on PID
    /// `MUX_ES_PID + i`; the PES `stream_id` is assigned per media kind (video
    /// `0xE0..`, audio `0xC0..`), distinct within each kind so several video or
    /// audio streams stay addressable. [`push_au_on`](Self::push_au_on) selects the
    /// stream by index.
    pub fn with_streams(stream_types: &[u8]) -> Self {
        let one_program: Vec<(u16, u8)> = stream_types.iter().map(|&t| (1, t)).collect();
        Self::with_programs(&one_program)
    }

    /// A multi-program muxer (M783): entry `i` of `streams` is
    /// `(program_number, stream_type)` for elementary stream `i`. Streams keep
    /// their global index ([`push_au_on`](Self::push_au_on) still selects by it)
    /// and their PID `MUX_ES_PID + i`, so several programs can share the numbering.
    /// Programs enter the PAT in first-appearance order of their numbers; program
    /// `p` gets its own PMT on PID `MUX_PMT_PID + p`, naming only its own streams,
    /// with its first stream as PCR_PID. PES `stream_id`s restart per program.
    pub fn with_programs(streams: &[(u16, u8)]) -> Self {
        let mut programs: Vec<MuxProgram> = Vec::new();
        // per-program (video, audio) stream_id counters, indexed like `programs`.
        let mut ids: Vec<(u8, u8)> = Vec::new();
        let mut mux_streams = Vec::with_capacity(streams.len());
        for (i, &(number, stream_type)) in streams.iter().enumerate() {
            let program = match programs.iter().position(|p| p.number == number) {
                Some(p) => p,
                None => {
                    programs.push(MuxProgram {
                        number,
                        pmt_pid: MUX_PMT_PID + programs.len() as u16,
                        pmt_cc: 0,
                        pcr_stream: i,
                        last_pcr_90khz: None,
                        service: None,
                    });
                    ids.push((0, 0));
                    programs.len() - 1
                }
            };
            let (video_n, audio_n) = &mut ids[program];
            // The mux's only private-PES (0x06) use is asynchronous KLV metadata
            // (MISB ST 1402): it rides private_stream_1 (0xBD) and its PMT entry
            // carries the 'KLVA' registration descriptor the demux side keys on.
            // Synchronous KLV rides metadata-in-PES (0x15) with the 13818-1
            // metadata stream_id (0xFC) and a metadata_descriptor instead.
            let stream_id = if stream_type == STREAM_TYPE_PRIVATE_PES {
                0xBD
            } else if stream_type == STREAM_TYPE_METADATA_PES {
                0xFC
            } else if stream_type == STREAM_TYPE_AAC {
                let id = 0xC0 + *audio_n;
                *audio_n += 1;
                id
            } else {
                let id = 0xE0 + *video_n;
                *video_n += 1;
                id
            };
            let es_info: &[u8] = match stream_type {
                STREAM_TYPE_PRIVATE_PES => KLVA_REGISTRATION,
                STREAM_TYPE_METADATA_PES => KLV_METADATA_DESCRIPTOR,
                _ => &[],
            };
            mux_streams.push(MuxStream {
                stream_type,
                pid: MUX_ES_PID + i as u16,
                stream_id,
                es_cc: 0,
                program,
                meta_seq: 0,
                es_info: es_info.to_vec(),
                identity_len: es_info.len(),
                dvb_subtitle: false,
            });
        }
        Self {
            streams: mux_streams,
            programs,
            pat_cc: 0,
            sdt_cc: 0,
            default_service: None,
            tables_written: false,
            table_interval_90khz: 0,
            last_tables_pts: None,
            pcr_interval_90khz: 3600,
        }
    }

    /// Name the service the SDT announces (M872): the `service_name` and
    /// `service_provider_name` of a DVB `service_descriptor`. With this set the mux
    /// writes an SDT (PID 0x11, `table_id` 0x42) alongside the PAT/PMT, on the same
    /// cadence, so a reader (ffprobe, [`TsDemuxer::service`]) reports the program's
    /// text. This text names every program that has no service of its own
    /// ([`set_program_service`](Self::set_program_service)), whichever order the two
    /// are called in. A field longer than the single-byte DVB length allows is
    /// truncated at a char boundary. Call before the first access unit, since that
    /// is when the tables go out.
    pub fn set_service(&mut self, name: &str, provider: &str) {
        self.default_service = Some(ServiceInfo {
            name: String::from(name),
            provider: String::from(provider),
        });
    }

    /// Name one program's service (M878), overriding
    /// [`set_service`](Self::set_service) for it: a multi-program multiplex gets
    /// one SDT entry per program, each with its own name / provider. `false` when
    /// no program carries `program_number` (nothing was recorded), so a caller
    /// never sets a service the SDT then omits.
    pub fn set_program_service(&mut self, program_number: u16, name: &str, provider: &str) -> bool {
        let Some(program) = self
            .programs
            .iter_mut()
            .find(|p| p.number == program_number)
        else {
            return false;
        };
        program.service = Some(ServiceInfo {
            name: String::from(name),
            provider: String::from(provider),
        });
        true
    }

    /// The service text of program `index` (into `programs`): its own, else the
    /// muxer-wide default.
    fn program_service(&self, index: usize) -> Option<&ServiceInfo> {
        self.programs[index]
            .service
            .as_ref()
            .or(self.default_service.as_ref())
    }

    /// Declare elementary stream `index`'s language in its PMT entry, as an
    /// `ISO_639_language_descriptor` (M872, what ffmpeg writes for
    /// `-metadata:s:0 language=deu`). A code that is not three ASCII letters is
    /// ignored, since the descriptor's field is exactly three bytes, as is an
    /// out-of-range index. Call before the first access unit (the PMT goes out then).
    pub fn set_stream_language(&mut self, index: usize, code: &str) {
        let c = code.as_bytes();
        if c.len() != 3 || !c.iter().all(u8::is_ascii_alphabetic) {
            return;
        }
        if let Some(s) = self.streams.get_mut(index) {
            // audio_type 0 = undefined, what ffmpeg writes.
            s.es_info
                .extend_from_slice(&[DESC_TAG_ISO639, 4, c[0], c[1], c[2], 0x00]);
        }
    }

    /// Declare elementary stream `index` as EBU teletext in its PMT entry, as a
    /// DVB `teletext_descriptor` naming one service (M924). Without this the
    /// stream is asynchronous KLV, the muxer's default reading of a private PES,
    /// which no receiver routes to a teletext decoder. An out-of-range index or a
    /// stream that is not a private PES is ignored. Call before the first access
    /// unit (the PMT goes out then).
    pub fn set_stream_teletext(&mut self, index: usize, service: TeletextService) {
        self.set_private_identity(
            index,
            &[
                DESC_TAG_TELETEXT,
                5,
                service.language[0],
                service.language[1],
                service.language[2],
                (service.teletext_type << 3) | (service.magazine & 0x07),
                service.page,
            ],
        );
    }

    /// Declare elementary stream `index` as DVB subtitles in its PMT entry, as a
    /// `subtitling_descriptor` naming one subtitle stream (M927): the language,
    /// the `subtitling_type`, and the composition / ancillary page ids a decoder
    /// follows. This is also what marks the stream's access units as display
    /// sets, so each goes out in the PES data field EN 300 743 wraps them in.
    /// Without it the stream is asynchronous KLV, the muxer's default reading of
    /// a private PES. An out-of-range index or a stream that is not a private PES
    /// is ignored, as is a language code that is not three ASCII letters (the
    /// descriptor's field is exactly three bytes). Call before the first access
    /// unit (the PMT goes out then).
    pub fn set_stream_subtitling(
        &mut self,
        index: usize,
        language: &str,
        subtitling_type: u8,
        ids: crate::dvbsub::PageIds,
    ) {
        let l = language.as_bytes();
        if l.len() != 3 || !l.iter().all(u8::is_ascii_alphabetic) {
            return;
        }
        let c = ids.composition.to_be_bytes();
        let a = ids.ancillary.to_be_bytes();
        self.set_private_identity(
            index,
            &[
                DESC_TAG_SUBTITLING,
                8,
                l[0],
                l[1],
                l[2],
                subtitling_type,
                c[0],
                c[1],
                a[0],
                a[1],
            ],
        );
        if let Some(s) = self
            .streams
            .get_mut(index)
            .filter(|s| s.stream_type == STREAM_TYPE_PRIVATE_PES)
        {
            s.dvb_subtitle = true;
        }
    }

    /// Declare elementary stream `index` as AV1 video in its PMT entry (M1049),
    /// as the AOM carriage's `registration_descriptor`, with the format_identifier
    /// a live receiver reads (see `AV1_REGISTRATION_ID_GSTREAMER`). Without this
    /// the stream is asynchronous KLV, the muxer's default reading of a private
    /// PES. An out-of-range index or a stream that is not a private PES is
    /// ignored. Call before the first access unit (the PMT goes out then).
    pub fn set_stream_av1(&mut self, index: usize) {
        let mut descriptor = Vec::from([
            DESC_TAG_REGISTRATION,
            AV1_REGISTRATION_ID_GSTREAMER.len() as u8,
        ]);
        descriptor.extend_from_slice(AV1_REGISTRATION_ID_GSTREAMER);
        self.set_private_identity(index, &descriptor);
    }

    /// Replace a private-PES stream's identifying descriptor. A bare 0x06 means
    /// nothing on its own, so this muxer writes the 'KLVA' registration by
    /// default; a stream that is teletext, DVB subtitles or AV1 instead says so
    /// with its own descriptor, which leads the ES-info loop. Descriptors added
    /// separately (a language) are kept whichever order the calls come in, and
    /// setting the identity twice replaces it rather than writing both.
    fn set_private_identity(&mut self, index: usize, descriptor: &[u8]) {
        let Some(s) = self.streams.get_mut(index) else {
            return;
        };
        if s.stream_type != STREAM_TYPE_PRIVATE_PES {
            return;
        }
        s.es_info.drain(..s.identity_len);
        s.es_info.splice(0..0, descriptor.iter().copied());
        s.identity_len = descriptor.len();
    }

    /// Set the PAT/PMT re-emission cadence in 90 kHz ticks (`0` = once up front).
    /// See `table_interval_90khz`.
    pub fn set_table_interval_90khz(&mut self, ticks: u64) {
        self.table_interval_90khz = ticks;
    }

    /// Set the PCR insertion cadence in 90 kHz ticks (default 3600).
    pub fn set_pcr_interval_90khz(&mut self, ticks: u64) {
        self.pcr_interval_90khz = ticks;
    }

    /// Mux one access unit of stream 0 (the single-stream convenience). See
    /// [`push_au_on`](Self::push_au_on).
    pub fn push_au(
        &mut self,
        au: &[u8],
        pts_90khz: Option<u64>,
        dts_90khz: Option<u64>,
    ) -> Vec<u8> {
        self.push_au_on(0, au, pts_90khz, dts_90khz)
    }

    /// Mux one access unit of elementary stream `stream_index` into TS bytes,
    /// preceded by the PAT + every program's PMT (+ the SDT, with a service set)
    /// on the very first call (any stream). `pts_90khz`,
    /// when present, is written into the PES header; a `dts_90khz` that differs
    /// from it adds a second (DTS) timestamp for reordered (B-frame) video.
    pub fn push_au_on(
        &mut self,
        stream_index: usize,
        au: &[u8],
        pts_90khz: Option<u64>,
        dts_90khz: Option<u64>,
    ) -> Vec<u8> {
        let mut out = Vec::new();
        // Emit the PAT and every PMT up front, then again on the configured cadence
        // so a mid-stream joiner finds the tables. A `None` PTS can't be time-gated,
        // so it only ever triggers the initial emission.
        let due = if !self.tables_written {
            true
        } else if self.table_interval_90khz > 0 {
            match (pts_90khz, self.last_tables_pts) {
                (Some(now), Some(last)) => now.saturating_sub(last) >= self.table_interval_90khz,
                (Some(_), None) => true,
                _ => false,
            }
        } else {
            false
        };
        if due {
            self.pat_packet(&mut out);
            for p in 0..self.programs.len() {
                self.pmt_packet(p, &mut out);
            }
            // The SDT joins the pair on the same cadence, so a mid-stream joiner
            // learns the service name as soon as it learns the program layout.
            self.sdt_packet(&mut out);
            self.tables_written = true;
            if let Some(now) = pts_90khz {
                self.last_tables_pts = Some(now);
            }
        }
        // PCR rides its program's first stream's PID (that PMT's PCR_PID) and is
        // clocked on the decode timeline (DTS, falling back to PTS when no DTS):
        // DTS is monotonic in decode order, so the cadence bound holds even for
        // reordered (B-frame) streams whose PTS is non-monotonic. Emit one in the
        // first TS packet of this PES when the program's cadence is due (or no PCR
        // has gone out for it yet). Other streams and clock-less AUs never carry
        // PCR.
        let clock = dts_90khz.or(pts_90khz);
        let interval = self.pcr_interval_90khz;
        let program = &mut self.programs[self.streams[stream_index].program];
        let pcr = if program.pcr_stream == stream_index {
            clock.and_then(|now| {
                let due = match program.last_pcr_90khz {
                    None => true,
                    Some(last) => now.saturating_sub(last) >= interval,
                };
                due.then(|| {
                    program.last_pcr_90khz = Some(now);
                    now.saturating_sub(PCR_LEAD_90KHZ)
                })
            })
        } else {
            None
        };
        let s = &mut self.streams[stream_index];
        // Synchronous KLV rides one metadata AU cell per access unit. An AU too
        // big for the cell's 16-bit length goes out bare (no ST 0601 local set
        // comes near 64 KiB, and a wrong length would be worse than a bare one).
        let cell = (s.stream_type == STREAM_TYPE_METADATA_PES && au.len() <= 0xFFFF).then(|| {
            let c = metadata_au_cell(s.meta_seq, au);
            s.meta_seq = s.meta_seq.wrapping_add(1);
            c
        });
        // A DVB subtitle access unit is a display set, which a transport stream
        // carries in a data field: the data_identifier ahead of the segments and
        // the end marker behind (EN 300 743 clause 7.1).
        let field = s
            .dvb_subtitle
            .then(|| crate::dvbsub::pes_data_field(au))
            .or(cell);
        let pes = build_pes(
            s.stream_id,
            field.as_deref().unwrap_or(au),
            pts_90khz,
            dts_90khz,
        );
        let mut off = 0;
        let mut pusi = true;
        let mut first = true;
        while off < pes.len() {
            // Only the first packet of the PES carries the PCR; it shrinks the
            // payload room by the adaptation-field overhead.
            let pkt_pcr = if first { pcr } else { None };
            let room = TS_PACKET_LEN
                - 4
                - if pkt_pcr.is_some() {
                    PCR_AF_OVERHEAD
                } else {
                    0
                };
            let take = (pes.len() - off).min(room);
            ts_packet(
                s.pid,
                pusi,
                s.es_cc,
                pkt_pcr,
                &pes[off..off + take],
                &mut out,
            );
            s.es_cc = (s.es_cc + 1) & 0x0F;
            pusi = false;
            first = false;
            off += take;
        }
        out
    }

    fn pat_packet(&mut self, out: &mut Vec<u8>) {
        let mut body = Vec::with_capacity(5 + self.programs.len() * 4);
        body.extend_from_slice(&[
            0x00, 0x01, // transport_stream_id
            0xC1, 0x00, 0x00, // version/current, section_number, last_section_number
        ]);
        // One entry per program: program_number then its PMT PID.
        for p in &self.programs {
            body.extend_from_slice(&[
                (p.number >> 8) as u8,
                p.number as u8,
                0xE0 | (p.pmt_pid >> 8) as u8 & 0x1F,
                p.pmt_pid as u8,
            ]);
        }
        self.pat_cc = psi_packet(PID_PAT, 0x00, &body, self.pat_cc, out);
    }

    /// Emit program `prog_idx`'s PMT: its own streams only, with its first stream
    /// as PCR_PID (no separate PCR stream).
    fn pmt_packet(&mut self, prog_idx: usize, out: &mut Vec<u8>) {
        let program_number = self.programs[prog_idx].number;
        let pmt_pid = self.programs[prog_idx].pmt_pid;
        let pcr_pid = self.streams[self.programs[prog_idx].pcr_stream].pid;
        let mut body = Vec::with_capacity(9 + self.streams.len() * 5);
        body.extend_from_slice(&[
            (program_number >> 8) as u8,
            program_number as u8,
            0xC1,
            0x00,
            0x00, // version, section/last
            0xE0 | (pcr_pid >> 8) as u8 & 0x1F,
            pcr_pid as u8, // PCR_PID
            0xF0,
            0x00, // program_info_length = 0
        ]);
        // One ES loop entry per stream of this program: stream_type,
        // elementary_PID, ES_info_len, then any ES-info descriptors.
        for s in self.streams.iter().filter(|s| s.program == prog_idx) {
            body.extend_from_slice(&[
                s.stream_type,
                0xE0 | (s.pid >> 8) as u8 & 0x1F,
                s.pid as u8,
                0xF0 | (s.es_info.len() >> 8) as u8 & 0x0F,
                s.es_info.len() as u8,
            ]);
            body.extend_from_slice(&s.es_info);
        }
        self.programs[prog_idx].pmt_cc =
            psi_packet(pmt_pid, 0x02, &body, self.programs[prog_idx].pmt_cc, out);
    }

    /// Emit the SDT (PID 0x11, `table_id` 0x42), one service entry per program
    /// that names a service. A no-op when none does.
    fn sdt_packet(&mut self, out: &mut Vec<u8>) {
        let body = self.sdt_body();
        if body.is_empty() {
            return;
        }
        self.sdt_cc = psi_packet(PID_SDT, TABLE_ID_SDT, &body, self.sdt_cc, out);
    }

    /// The SDT section body (from section[3]): the PSI header tail,
    /// `original_network_id` and a reserved byte, then one service entry per
    /// program that names a service, carrying that program's own text (M878).
    /// `service_type` says digital television for a program carrying video and
    /// digital radio for one that does not. Empty when no program names a service,
    /// which writes no SDT.
    fn sdt_body(&self) -> Vec<u8> {
        let mut body = Vec::new();
        for (idx, p) in self.programs.iter().enumerate() {
            let Some(service) = self.program_service(idx) else {
                continue;
            };
            if body.is_empty() {
                body.extend_from_slice(&[
                    0x00, 0x01, // transport_stream_id (matches the PAT's)
                    0xC1, 0x00, 0x00, // version/current, section_number, last_section_number
                    0x00, 0x01, // original_network_id
                    0xFF, // reserved_future_use
                ]);
            }
            let name = dvb_field(&service.name);
            let provider = dvb_field(&service.provider);
            // service_descriptor: tag, length, service_type, then the two fields.
            let desc_len = 3 + provider.len() + name.len();
            let loop_len = 2 + desc_len;
            let video = self
                .streams
                .iter()
                .any(|s| s.program == idx && mux_stream_type_is_video(s.stream_type));
            body.extend_from_slice(&[
                (p.number >> 8) as u8,
                p.number as u8,
                0xFC, // reserved, EIT_schedule / EIT_present_following = 0
                // running_status '100' (running), free_CA 0, loop length.
                0x80 | ((loop_len >> 8) as u8 & 0x0F),
                loop_len as u8,
                DESC_TAG_SERVICE,
                desc_len as u8,
                if video {
                    SERVICE_TYPE_TV
                } else {
                    SERVICE_TYPE_RADIO
                },
                provider.len() as u8,
            ]);
            body.extend_from_slice(provider);
            body.push(name.len() as u8);
            body.extend_from_slice(name);
        }
        body
    }
}

/// Whether a `stream_type` the mux writes is video, for the SDT `service_type`.
fn mux_stream_type_is_video(stream_type: u8) -> bool {
    matches!(
        stream_type,
        STREAM_TYPE_H264 | STREAM_TYPE_H265 | STREAM_TYPE_MPEG4P2
    )
}

/// The bytes of one DVB text field: the string truncated at a char boundary so
/// both fields plus the `service_descriptor` header fit its single length byte.
/// ASCII, which real service names are, rides unchanged.
fn dvb_field(s: &str) -> &[u8] {
    /// Room for two fields and the 3-byte descriptor header inside 255 bytes.
    const MAX: usize = 124;
    let end = s
        .char_indices()
        .map(|(i, c)| i + c.len_utf8())
        .take_while(|&e| e <= MAX)
        .last()
        .unwrap_or(0);
    &s.as_bytes()[..end]
}

/// Wrap one metadata access unit in a single ISO 13818-1 metadata AU cell
/// (section 2.12.4): metadata_service_id, sequence_number, a flags byte, then a
/// big-endian 16-bit AU_cell_data_length. The flags say the cell holds a complete
/// AU (cell_fragmentation_indication '11'), carries no decoder config and is a
/// random access point; both known readers (ffmpeg, [`unwrap_metadata_au_cells`])
/// skip the byte rather than interpret it.
fn metadata_au_cell(sequence_number: u8, au: &[u8]) -> Vec<u8> {
    let mut cell = Vec::with_capacity(5 + au.len());
    cell.push(0x00); // metadata_service_id
    cell.push(sequence_number);
    cell.push(0xDF); // '11' complete AU, config 0, random access 1, reserved
    cell.push((au.len() >> 8) as u8);
    cell.push(au.len() as u8);
    cell.extend_from_slice(au);
    cell
}

/// Build a PES packet for one access unit (start code + stream_id + length + an
/// optional header carrying the PTS, plus a DTS for reordered video), matching
/// what [`parse_pes_header`] reads. A DTS is written (flags '11', a 10-byte field
/// pair) only when it is present and differs from the PTS; otherwise PTS-only.
fn build_pes(stream_id: u8, au: &[u8], pts_90khz: Option<u64>, dts_90khz: Option<u64>) -> Vec<u8> {
    let mut header = Vec::new();
    header.push(0x80); // marker '10'
    match (pts_90khz, dts_90khz) {
        (Some(pts), Some(dts)) if dts != pts => {
            header.push(0xC0); // PTS_DTS_flags = '11'
            header.push(10); // PES_header_data_length: two 5-byte fields
            encode_timestamp(0x3, pts, &mut header); // '0011' prefix for PTS (with DTS)
            encode_timestamp(0x1, dts, &mut header); // '0001' prefix for DTS
        }
        (Some(pts), _) => {
            header.push(0x80); // PTS_DTS_flags = '10'
            header.push(5); // PES_header_data_length
            encode_timestamp(0x2, pts, &mut header); // '0010' prefix for PTS-only
        }
        (None, _) => {
            header.push(0x00); // no PTS (and so no DTS)
            header.push(0);
        }
    }
    let pes_payload_len = header.len() + au.len();
    let mut pes = alloc::vec![0x00, 0x00, 0x01, stream_id];
    // PES_packet_length: the real length when it fits, else 0 (unbounded, the
    // standard video case). The demuxer delimits by TS packet boundaries anyway.
    let len_field = u16::try_from(pes_payload_len).unwrap_or(0);
    pes.push((len_field >> 8) as u8);
    pes.push(len_field as u8);
    pes.extend_from_slice(&header);
    pes.extend_from_slice(au);
    pes
}

/// Append a 5-byte PTS/DTS field in 90 kHz units, the inverse of
/// [`decode_timestamp`]. `prefix` is the 4-bit marker: `0010` for a lone PTS,
/// `0011` for a PTS paired with a DTS, `0001` for that DTS.
fn encode_timestamp(prefix: u8, ts: u64, out: &mut Vec<u8>) {
    out.push((prefix << 4) | (((ts >> 30) & 0x07) as u8) << 1 | 0x01);
    out.push(((ts >> 22) & 0xFF) as u8);
    out.push((((ts >> 15) & 0x7F) as u8) << 1 | 0x01);
    out.push(((ts >> 7) & 0xFF) as u8);
    out.push(((ts & 0x7F) as u8) << 1 | 0x01);
}

/// Write one 188-byte TS packet to `out`: a payload of up to 184 bytes, padded
/// with an adaptation-field stuffing run when shorter (the last packet of a PES).
/// With `pcr` set (the 90 kHz base) the adaptation field always exists, carrying
/// the PCR_flag and 6-byte PCR ahead of any stuffing, so the payload room drops
/// by [`PCR_AF_OVERHEAD`].
fn ts_packet(pid: u16, pusi: bool, cc: u8, pcr: Option<u64>, payload: &[u8], out: &mut Vec<u8>) {
    const PAYLOAD_MAX: usize = TS_PACKET_LEN - 4;
    let payload_max = PAYLOAD_MAX - if pcr.is_some() { PCR_AF_OVERHEAD } else { 0 };
    debug_assert!(payload.len() <= payload_max);
    out.push(SYNC_BYTE);
    out.push((if pusi { 0x40 } else { 0 }) | ((pid >> 8) as u8 & 0x1F));
    out.push(pid as u8);
    let l = payload.len();
    if pcr.is_none() && l == PAYLOAD_MAX {
        out.push(0x10 | (cc & 0x0F)); // payload only
        out.extend_from_slice(payload);
        return;
    }
    out.push(0x30 | (cc & 0x0F)); // adaptation field + payload
    let af_len = PAYLOAD_MAX - 1 - l; // bytes after the AF length byte
    out.push(af_len as u8);
    if let Some(base) = pcr {
        out.push(0x10); // AF flags: PCR_flag
        write_pcr(base, out);
        out.resize(out.len() + (af_len - 1 - 6), 0xFF); // stuffing after flags + PCR
    } else if af_len >= 1 {
        out.push(0x00); // AF flags (no PCR / no options)
        out.resize(out.len() + (af_len - 1), 0xFF); // stuffing
    }
    out.extend_from_slice(payload);
}

/// Encode the 48-bit PCR field: 33-bit `base` (90 kHz), 6 reserved bits set to 1,
/// 9-bit extension = 0 (no 27 MHz phase to encode).
fn write_pcr(base: u64, out: &mut Vec<u8>) {
    let base = base & 0x1_FFFF_FFFF;
    out.push((base >> 25) as u8);
    out.push((base >> 17) as u8);
    out.push((base >> 9) as u8);
    out.push((base >> 1) as u8);
    // low base bit, then the 6 reserved 1s and the top extension bit (0).
    out.push(((base as u8 & 0x1) << 7) | 0x7E);
    out.push(0x00); // extension low bits = 0
}

/// Write a PSI section (pointer field + table + MPEG-2 CRC-32), spanning more
/// than one TS packet when the section exceeds a single 184-byte payload (e.g. a
/// PMT with more than ~33 streams). The first packet carries PUSI + the pointer
/// field; continuations carry PUSI=0 with no new pointer. Returns the continuity
/// counter for the next packet on this PID.
fn psi_packet(pid: u16, table_id: u8, body: &[u8], mut cc: u8, out: &mut Vec<u8>) -> u8 {
    let section_length = body.len() + 4; // body + 4-byte CRC
    let mut section = Vec::with_capacity(3 + section_length);
    section.push(table_id);
    section.push(0xB0 | ((section_length >> 8) as u8 & 0x0F)); // syntax=1, reserved, len hi
    section.push((section_length & 0xFF) as u8);
    section.extend_from_slice(body);
    let crc = mpeg_crc32(&section); // over table_id .. end of body
    section.extend_from_slice(&crc.to_be_bytes());
    let mut payload = alloc::vec![0u8]; // pointer_field = 0 (first packet only)
    payload.extend_from_slice(&section);

    const ROOM: usize = TS_PACKET_LEN - 4;
    let mut rest = &payload[..];
    let mut pusi = true;
    loop {
        let n = rest.len().min(ROOM);
        ts_packet(pid, pusi, cc, None, &rest[..n], out);
        cc = (cc + 1) & 0x0F;
        rest = &rest[n..];
        pusi = false;
        if rest.is_empty() {
            break;
        }
    }
    cc
}

/// MPEG-2 systems CRC-32 (poly 0x04C11DB7, init all-ones, no final xor, MSB
/// first), as the PSI section trailer.
fn mpeg_crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= (b as u32) << 24;
        for _ in 0..8 {
            crc = if crc & 0x8000_0000 != 0 {
                (crc << 1) ^ 0x04C1_1DB7
            } else {
                crc << 1
            };
        }
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 188-byte TS packet with the given PID / PUSI / payload. A short
    /// payload is padded with adaptation-field stuffing (as real muxers do), so
    /// the carried payload is exactly `payload` with no trailing junk.
    fn ts_packet(pid: u16, pusi: bool, payload: &[u8]) -> Vec<u8> {
        const ROOM: usize = TS_PACKET_LEN - 4;
        assert!(payload.len() <= ROOM, "payload too big for one packet");
        let mut p = alloc::vec![0u8; TS_PACKET_LEN];
        p[0] = SYNC_BYTE;
        p[1] = if pusi { 0x40 } else { 0x00 } | ((pid >> 8) as u8 & 0x1F);
        p[2] = (pid & 0xFF) as u8;
        let l = payload.len();
        if l == ROOM {
            p[3] = 0x10; // payload only
            p[4..].copy_from_slice(payload);
        } else {
            p[3] = 0x30; // adaptation field + payload
            let af_len = ROOM - 1 - l; // bytes after the length byte
            p[4] = af_len as u8;
            if af_len >= 1 {
                p[5] = 0x00; // adaptation flags (none)
                for b in p.iter_mut().take(6 + (af_len - 1)).skip(6) {
                    *b = 0xFF; // stuffing
                }
            }
            p[5 + af_len..].copy_from_slice(payload);
        }
        p
    }

    /// A PSI section with a leading pointer_field (0), the given table_id and
    /// body, and a dummy 4-byte CRC. `body` is everything from section[3].
    fn psi_packet(pid: u16, table_id: u8, body: &[u8]) -> Vec<u8> {
        let section_length = body.len() + 4; // body + CRC
        let mut section = Vec::new();
        section.push(table_id);
        section.push(0xB0 | ((section_length >> 8) as u8 & 0x0F));
        section.push((section_length & 0xFF) as u8);
        section.extend_from_slice(body);
        section.extend_from_slice(&[0, 0, 0, 0]); // dummy CRC
        let mut payload = alloc::vec![0u8]; // pointer_field = 0
        payload.extend_from_slice(&section);
        ts_packet(pid, true, &payload)
    }

    /// PAT body (from section[3]) mapping one program to a PMT PID.
    fn pat_body(program: u16, pmt_pid: u16) -> Vec<u8> {
        alloc::vec![
            (program >> 8) as u8,
            program as u8, // transport_stream_id (reuse)
            0xC1,
            0x00,
            0x00, // version/current, section_number, last_section_number
            (program >> 8) as u8,
            program as u8,
            0xE0 | ((pmt_pid >> 8) as u8 & 0x1F),
            pmt_pid as u8,
        ]
    }

    /// PAT body (from section[3]) mapping several `(program_number, pmt_pid)`
    /// pairs, in order.
    fn pat_body_multi(programs: &[(u16, u16)]) -> Vec<u8> {
        let mut b = alloc::vec![
            0x00, 0x01, // transport_stream_id
            0xC1, 0x00, 0x00, // version/current, section_number, last_section_number
        ];
        for &(program, pmt_pid) in programs {
            b.push((program >> 8) as u8);
            b.push(program as u8);
            b.push(0xE0 | ((pmt_pid >> 8) as u8 & 0x1F));
            b.push(pmt_pid as u8);
        }
        b
    }

    /// PMT body (from section[3]) announcing one elementary stream.
    fn pmt_body(es_pid: u16, stream_type: u8) -> Vec<u8> {
        alloc::vec![
            0x00,
            0x01, // program_number
            0xC1,
            0x00,
            0x00, // version, section/last
            0xE0 | ((es_pid >> 8) as u8 & 0x1F),
            es_pid as u8, // PCR_PID
            0xF0,
            0x00, // program_info_length = 0
            stream_type,
            0xE0 | ((es_pid >> 8) as u8 & 0x1F),
            es_pid as u8, // elementary_PID
            0xF0,
            0x00, // ES_info_length = 0
        ]
    }

    /// A PES packet carrying `es` with an optional PTS.
    fn pes(pts_90khz: Option<u64>, es: &[u8]) -> Vec<u8> {
        let mut p = alloc::vec![0x00, 0x00, 0x01, 0xE0]; // start code + stream_id (video)
        let mut header = Vec::new();
        if let Some(pts) = pts_90khz {
            header.push(0x80); // marker '10'
            header.push(0x80); // PTS_DTS_flags = '10'
            header.push(5); // header_data_length
                            // 5-byte PTS field with '0010' prefix.
            header.push(0x21 | (((pts >> 30) & 0x07) as u8) << 1);
            header.push(((pts >> 22) & 0xFF) as u8);
            header.push(0x01 | (((pts >> 15) & 0x7F) as u8) << 1);
            header.push(((pts >> 7) & 0xFF) as u8);
            header.push(0x01 | ((pts & 0x7F) as u8) << 1);
        } else {
            header.push(0x80);
            header.push(0x00);
            header.push(0);
        }
        let pes_len = header.len() + es.len();
        p.push((pes_len >> 8) as u8);
        p.push((pes_len & 0xFF) as u8);
        p.extend_from_slice(&header);
        p.extend_from_slice(es);
        p
    }

    #[test]
    fn demuxes_pat_pmt_and_one_pes() {
        let pmt_pid = 0x1000;
        let es_pid = 0x0100;
        let mut d = TsDemuxer::new();
        d.push_packet(&psi_packet(PID_PAT, 0x00, &pat_body(1, pmt_pid)));
        assert_eq!(d.video_pid(), None, "no PMT yet");
        d.push_packet(&psi_packet(
            pmt_pid,
            0x02,
            &pmt_body(es_pid, STREAM_TYPE_H264),
        ));
        assert_eq!(
            d.streams(),
            &[ElementaryStream {
                pid: es_pid,
                stream_type: STREAM_TYPE_H264,
                opus_channels: None,
                ac3: false,
                klv: false,
                av1: false,
                subtitling: None,
                teletext: None,
                language: None
            }]
        );
        assert_eq!(d.video_pid(), Some(es_pid));

        // One PES (Annex-B-ish payload) with a PTS, then a second PES start to
        // flush the first.
        let au = [0x00, 0x00, 0x00, 0x01, 0x65, 0xAA, 0xBB];
        d.push_packet(&ts_packet(es_pid, true, &pes(Some(900_000), &au)));
        assert!(
            d.take_units().is_empty(),
            "first PES not flushed until next PES start"
        );
        d.push_packet(&ts_packet(
            es_pid,
            true,
            &pes(Some(901_000), &[0x00, 0x00, 0x01, 0x41]),
        ));
        let units = d.take_units();
        assert_eq!(units.len(), 1, "first PES completed by the second's start");
        assert_eq!(units[0].pid, es_pid);
        assert_eq!(units[0].pts_90khz, Some(900_000));
        assert_eq!(units[0].data, au, "PES header stripped, ES bytes intact");
    }

    #[test]
    fn pes_reassembles_across_packets() {
        let pmt_pid = 0x1000;
        let es_pid = 0x0100;
        let mut d = TsDemuxer::new();
        d.push_packet(&psi_packet(PID_PAT, 0x00, &pat_body(1, pmt_pid)));
        d.push_packet(&psi_packet(
            pmt_pid,
            0x02,
            &pmt_body(es_pid, STREAM_TYPE_H264),
        ));

        // A PES whose ES payload spans two TS packets.
        let part1: Vec<u8> = (0..150u8).collect();
        let part2: Vec<u8> = (0..150u8).map(|x| x ^ 0x55).collect();
        let mut whole = part1.clone();
        whole.extend_from_slice(&part2);
        let pes_bytes = pes(Some(12_345), &whole);
        // Split the PES across two TS packets: first carries the header + part1.
        let split = pes_bytes.len() - part2.len();
        d.push_packet(&ts_packet(es_pid, true, &pes_bytes[..split]));
        d.push_packet(&ts_packet(es_pid, false, &pes_bytes[split..]));
        d.flush();

        let units = d.take_units();
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].data, whole, "ES reassembled across TS packets");
        assert_eq!(units[0].pts_90khz, Some(12_345));
    }

    #[test]
    fn oversized_pes_continuation_is_dropped_not_unbounded() {
        // A video PES has no declared length and is delimited only by the next
        // payload-unit-start, so an endless continuation run must be bounded: a
        // continuation that would exceed the cap drops the pending PES rather
        // than growing its buffer without limit.
        let mut d = TsDemuxer::new();
        d.accumulate_pes(0x0100, STREAM_TYPE_H264, &[0u8; 8], true); // open a PES
        assert_eq!(d.pending.len(), 1, "the PES is open");
        let huge = alloc::vec![0u8; MAX_PES_BYTES + 1];
        d.accumulate_pes(0x0100, STREAM_TYPE_H264, &huge, false);
        assert!(
            d.pending.is_empty(),
            "the oversized PES is dropped, not buffered"
        );
    }

    #[test]
    fn ignores_non_sync_and_other_pids() {
        let mut d = TsDemuxer::new();
        d.push_packet(&[0u8; TS_PACKET_LEN]); // bad sync
        d.push_packet(&ts_packet(0x0123, true, &[1, 2, 3])); // unknown PID, no PMT
        assert!(d.take_units().is_empty());
        assert!(d.streams().is_empty());
    }

    #[test]
    fn multi_program_selection_routes_the_active_program() {
        let pmt1 = 0x1000u16;
        let pmt2 = 0x1001u16;
        let es1 = 0x0100u16; // program 1, H.264
        let es2 = 0x0200u16; // program 2, AAC
                             // PAT: a NIT pointer (program 0, skipped) plus programs 1 and 2 on
                             // distinct PMT PIDs, each PMT naming a distinct ES / codec.
        let pat = pat_body_multi(&[(0, 0x0010), (1, pmt1), (2, pmt2)]);
        let build = || {
            let mut d = TsDemuxer::new();
            d.push_packet(&psi_packet(PID_PAT, 0x00, &pat));
            d.push_packet(&psi_packet(pmt1, 0x02, &pmt_body(es1, STREAM_TYPE_H264)));
            d.push_packet(&psi_packet(pmt2, 0x02, &pmt_body(es2, STREAM_TYPE_AAC)));
            d
        };

        // Default: program 1 is active; only its ES shows, only its PES routes.
        let mut d = build();
        assert_eq!(d.streams().len(), 1, "active program's streams only");
        assert_eq!(d.streams()[0].pid, es1);
        assert_eq!(d.streams()[0].stream_type, STREAM_TYPE_H264);
        d.push_packet(&ts_packet(es2, true, &pes(Some(1), &[1, 2, 3]))); // program 2: ignored
        d.push_packet(&ts_packet(es1, true, &pes(Some(2), &[4, 5, 6]))); // program 1: routes
        d.flush();
        let units = d.take_units();
        assert_eq!(units.len(), 1, "only the active program's PES routes");
        assert_eq!(units[0].pid, es1);

        // Select program 2: its AAC ES becomes active and its PES routes.
        let mut d = build();
        d.set_program_number(Some(2));
        assert_eq!(d.streams().len(), 1);
        assert_eq!(d.streams()[0].pid, es2);
        assert_eq!(d.streams()[0].stream_type, STREAM_TYPE_AAC);
        d.push_packet(&ts_packet(es2, true, &pes(Some(1), &[7, 8])));
        d.flush();
        let units = d.take_units();
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].pid, es2);

        // An unknown program number routes nothing.
        let mut d = build();
        d.set_program_number(Some(99));
        assert!(d.streams().is_empty(), "no match = no active program");
        d.push_packet(&ts_packet(es1, true, &pes(Some(1), &[1])));
        d.push_packet(&ts_packet(es2, true, &pes(Some(1), &[2])));
        d.flush();
        assert!(d.take_units().is_empty());

        // programs() lists both real programs; the NIT (program 0) is skipped.
        let progs: Vec<u16> = build().programs().map(|(n, _)| n).collect();
        assert_eq!(progs, alloc::vec![1, 2]);
    }

    #[test]
    fn mux_demux_round_trip() {
        // Mux two H.264 access units with PTS, then demux the TS back to them.
        let au0 = [0u8, 0, 0, 1, 0x65, 0xAA, 0xBB];
        let au1 = [0u8, 0, 0, 1, 0x41, 0xCC];
        let mut mux = TsMuxer::new(STREAM_TYPE_H264);
        let mut bytes = mux.push_au(&au0, Some(900_000), None);
        bytes.extend(mux.push_au(&au1, Some(903_000), None));
        assert_eq!(bytes.len() % TS_PACKET_LEN, 0, "output is whole TS packets");

        let mut d = TsDemuxer::new();
        for pkt in bytes.chunks(TS_PACKET_LEN) {
            d.push_packet(pkt);
        }
        d.flush();
        let units = d.take_units();
        assert_eq!(units.len(), 2, "both AUs survive the round trip");
        assert_eq!(units[0].stream_type, STREAM_TYPE_H264);
        assert_eq!(units[0].data, au0, "AU bytes intact");
        assert_eq!(units[0].pts_90khz, Some(900_000));
        assert_eq!(units[1].data, au1);
        assert_eq!(units[1].pts_90khz, Some(903_000));
    }

    /// A DVB `service_descriptor` over the given raw provider / name field bytes.
    fn service_desc(provider: &[u8], name: &[u8]) -> Vec<u8> {
        let mut d = alloc::vec![
            DESC_TAG_SERVICE,
            (3 + provider.len() + name.len()) as u8,
            SERVICE_TYPE_TV,
            provider.len() as u8,
        ];
        d.extend_from_slice(provider);
        d.push(name.len() as u8);
        d.extend_from_slice(name);
        d
    }

    /// A one-service SDT body (from section[3]) carrying `desc` as that service's
    /// descriptor loop. `loop_fudge` perturbs the declared loop length, so a test
    /// can declare it past the section end.
    fn sdt_body(desc: &[u8], loop_fudge: i32) -> Vec<u8> {
        let loop_len = (desc.len() as i32 + loop_fudge).max(0) as usize;
        let mut b = alloc::vec![
            0x00,
            0x01, // transport_stream_id
            0xC1,
            0x00,
            0x00, // version/current, section/last
            0x00,
            0x01, // original_network_id
            0xFF, // reserved
            0x00,
            0x01, // service_id = program 1
            0xFC, // reserved + EIT flags
            0x80 | ((loop_len >> 8) as u8 & 0x0F),
            loop_len as u8,
        ];
        b.extend_from_slice(desc);
        b
    }

    /// The service a demuxer reports after a PAT naming program 1 and `sdt`, which
    /// carries a real CRC (the mux's own section writer).
    fn service_of(body: &[u8], table_id: u8, corrupt_crc: bool) -> Option<ServiceInfo> {
        let mut sdt = Vec::new();
        super::psi_packet(PID_SDT, table_id, body, 0, &mut sdt);
        if corrupt_crc {
            // A short PSI payload sits at the tail of its packet (stuffing first),
            // so the section's last CRC byte is the packet's last byte.
            let last = sdt.len() - 1;
            sdt[last] ^= 0xFF;
        }
        let mut d = TsDemuxer::new();
        d.push_packet(&psi_packet(PID_PAT, 0x00, &pat_body(1, 0x1000)));
        for pkt in sdt.chunks(TS_PACKET_LEN) {
            d.push_packet(pkt);
        }
        d.service().cloned()
    }

    /// The mux writes the SDT service text and per-stream language descriptors, and
    /// the demux reads both back on the streams they belong to.
    #[test]
    fn sdt_service_and_language_descriptors_round_trip() {
        let mut m = TsMuxer::with_streams(&[STREAM_TYPE_H264, STREAM_TYPE_AAC]);
        m.set_service("News One", "G2G Broadcasting");
        m.set_stream_language(1, "deu");
        m.set_stream_language(0, "en"); // not three letters: no descriptor
        m.set_stream_language(9, "eng"); // out of range: ignored
        let mut ts = m.push_au_on(0, &[0, 0, 0, 1, 0x65, 0x11], Some(900_000), None);
        ts.extend(m.push_au_on(1, &[0xFF, 0xF1, 0x22], Some(901_000), None));

        let mut d = TsDemuxer::new();
        for pkt in ts.chunks(TS_PACKET_LEN) {
            d.push_packet(pkt);
        }
        let service = d.service().expect("the SDT named program 1");
        assert_eq!(service.name, "News One");
        assert_eq!(service.provider, "G2G Broadcasting");
        assert_eq!(d.streams()[1].language_code(), Some("deu"));
        assert_eq!(
            d.streams()[0].language,
            None,
            "a two-letter code writes no descriptor"
        );

        // The extra descriptor bytes leave the PMT (and the PES flow) intact.
        d.flush();
        assert_eq!(d.take_units().len(), 2, "both AUs still demux");

        // A program carrying video is a television service; audio-only is radio.
        let sdt = psi_body_of(&ts, PID_SDT).expect("SDT emitted");
        assert_eq!(sdt[15], SERVICE_TYPE_TV, "service_type at the descriptor");
        let mut audio_only = TsMuxer::new(STREAM_TYPE_AAC);
        audio_only.set_service("Radio Two", "");
        let radio = audio_only.push_au(&[0xFF, 0xF1, 0x33], Some(0), None);
        let sdt = psi_body_of(&radio, PID_SDT).expect("SDT emitted");
        assert_eq!(sdt[15], SERVICE_TYPE_RADIO);
    }

    /// Each program of a multi-program mux carries its own SDT entry (M878): the
    /// per-program text where it is set, the muxer-wide default elsewhere, and the
    /// demuxer reads every service the SDT names whichever program it routes.
    #[test]
    fn per_program_service_entries_round_trip() {
        let mut m = TsMuxer::with_programs(&[(1, STREAM_TYPE_H264), (2, STREAM_TYPE_AAC)]);
        m.set_service("Network", "G2G");
        assert!(m.set_program_service(2, "Radio Two", "G2G Radio"));
        assert!(
            !m.set_program_service(7, "Nobody", ""),
            "no program 7: the caller learns the service went nowhere"
        );
        let mut ts = m.push_au_on(0, &[0, 0, 0, 1, 0x65, 0x11], Some(900_000), None);
        ts.extend(m.push_au_on(1, &[0xFF, 0xF1, 0x22], Some(901_000), None));

        let mut d = TsDemuxer::new();
        for pkt in ts.chunks(TS_PACKET_LEN) {
            d.push_packet(pkt);
        }
        let services: Vec<(u16, String, String)> = d
            .services()
            .map(|(n, s)| (n, s.name.clone(), s.provider.clone()))
            .collect();
        assert_eq!(
            services,
            alloc::vec![
                (1, String::from("Network"), String::from("G2G")),
                (2, String::from("Radio Two"), String::from("G2G Radio")),
            ],
            "program 1 takes the default, program 2 its own"
        );
        // Selecting program 2 changes which service is "the" active one, not the
        // set the SDT announced.
        d.set_program_number(Some(2));
        assert_eq!(d.service().map(|s| s.name.as_str()), Some("Radio Two"));
        assert_eq!(d.services().count(), 2);
    }

    /// Without a service the mux writes no SDT at all (nothing to say).
    #[test]
    fn no_service_writes_no_sdt() {
        let mut m = TsMuxer::new(STREAM_TYPE_H264);
        let ts = m.push_au(&[0, 0, 0, 1, 0x65, 0x88], Some(0), None);
        assert!(psi_body_of(&ts, PID_SDT).is_none());
    }

    /// Malformed or unrelated sections on the SDT PID stay quiet: a section whose
    /// CRC fails, one whose descriptor loop is declared past the section end, one
    /// whose text field runs past its descriptor, and the other-transport-stream
    /// table that shares the PID all report no service rather than garbled text or
    /// a panic. Each case carries a real CRC except the one that must not.
    #[test]
    fn malformed_sdt_sections_are_ignored() {
        let good = sdt_body(&service_desc(b"Prov", b"Name"), 0);
        assert_eq!(
            service_of(&good, TABLE_ID_SDT, false),
            Some(ServiceInfo {
                name: "Name".into(),
                provider: "Prov".into()
            }),
            "the well-formed section parses"
        );

        assert_eq!(
            service_of(&good, TABLE_ID_SDT, true),
            None,
            "a section whose CRC fails is dropped"
        );
        assert_eq!(
            service_of(&good, 0x46, false),
            None,
            "the other-TS SDT is not this stream's service"
        );
        assert_eq!(
            service_of(
                &sdt_body(&service_desc(b"Prov", b"Name"), 8),
                TABLE_ID_SDT,
                false
            ),
            None,
            "a descriptor loop declared past the section end abandons the walk"
        );

        // A provider length running past the descriptor body.
        let mut overrun = service_desc(b"Prov", b"Name");
        overrun[3] = 200;
        assert_eq!(
            service_of(&sdt_body(&overrun, 0), TABLE_ID_SDT, false),
            None
        );

        // A DVB character-table selection byte this parser will not decode: that
        // field comes back empty rather than mis-decoded, the other still decodes.
        let charset = service_of(
            &sdt_body(&service_desc(b"Prov", &[0x05, b'X', b'Y']), 0),
            TABLE_ID_SDT,
            false,
        )
        .expect("the section itself is well formed");
        assert_eq!(charset.provider, "Prov");
        assert!(charset.name.is_empty(), "the non-default table is skipped");

        // Degenerate descriptor loops return None rather than panicking.
        assert_eq!(parse_service_descriptor(&[]), None);
        assert_eq!(
            parse_service_descriptor(&[DESC_TAG_SERVICE, 40, 0x01]),
            None
        );
        assert_eq!(parse_iso639_language(&[DESC_TAG_ISO639, 40, b'e']), None);
        assert_eq!(
            parse_iso639_language(&[DESC_TAG_ISO639, 4, b'1', b'2', b'3', 0]),
            None,
            "a code that is not letters is not a language"
        );
    }

    #[test]
    fn mpeg_crc32_matches_known_vector() {
        // The documented CRC-32/MPEG-2 check value for ASCII "123456789".
        assert_eq!(mpeg_crc32(b"123456789"), 0x0376_E6E7);
    }

    #[test]
    fn large_psi_section_spans_multiple_packets() {
        // A section too big for one TS payload (e.g. a PMT with many streams)
        // must span whole packets, PUSI only on the first, rather than
        // underflowing the adaptation-field length.
        let body = alloc::vec![0xABu8; 400];
        let mut out = Vec::new();
        let next_cc = super::psi_packet(MUX_PMT_PID, 0x02, &body, 5, &mut out);
        assert_eq!(out.len() % TS_PACKET_LEN, 0, "emits whole packets");
        let packets = out.len() / TS_PACKET_LEN;
        assert!(
            packets >= 3,
            "400-byte body spans 3+ packets, got {packets}"
        );
        assert_eq!(out[1] & 0x40, 0x40, "first packet carries PUSI");
        assert_eq!(
            out[TS_PACKET_LEN + 1] & 0x40,
            0x00,
            "continuation clears PUSI"
        );
        assert_eq!(
            next_cc,
            (5 + packets as u8) & 0x0F,
            "cc advances per packet"
        );
    }

    #[test]
    fn pat_pmt_reemitted_on_interval() {
        // PAT TS packets in a byte stream (sync 0x47, PID == PID_PAT).
        fn pat_count(ts: &[u8]) -> usize {
            ts.chunks(TS_PACKET_LEN)
                .filter(|p| {
                    p.len() == TS_PACKET_LEN
                        && p[0] == 0x47
                        && (((p[1] as u16 & 0x1F) << 8) | p[2] as u16) == PID_PAT
                })
                .count()
        }
        let mut m = TsMuxer::new(STREAM_TYPE_H264);
        m.set_table_interval_90khz(90 * 100); // 100 ms cadence
        let au = alloc::vec![0u8, 0, 0, 1, 0x65, 0x88]; // a minimal IDR-ish AU

        let out0 = m.push_au(&au, Some(0), None);
        assert_eq!(pat_count(&out0), 1, "PAT emitted up front");
        let out1 = m.push_au(&au, Some(90 * 50), None); // +50 ms: under the interval
        assert_eq!(pat_count(&out1), 0, "no PAT before the interval elapses");
        let out2 = m.push_au(&au, Some(90 * 150), None); // 150 ms since last emit: due
        assert_eq!(pat_count(&out2), 1, "PAT re-emitted after the interval");
    }

    #[test]
    fn table_interval_zero_emits_tables_once() {
        let mut m = TsMuxer::new(STREAM_TYPE_H264);
        let au = alloc::vec![0u8, 0, 0, 1, 0x65, 0x88];
        let _ = m.push_au(&au, Some(0), None);
        let later = m.push_au(&au, Some(90 * 10_000), None); // 10 s later
        let pats = later
            .chunks(TS_PACKET_LEN)
            .filter(|p| {
                p.len() == TS_PACKET_LEN && (((p[1] as u16 & 0x1F) << 8) | p[2] as u16) == PID_PAT
            })
            .count();
        assert_eq!(pats, 0, "default cadence emits the tables only once");
    }

    #[test]
    fn opus_descriptors_parse_and_reject_malformed() {
        // registration 'Opus' + extension channel code 1 (mono).
        let desc = [0x05, 4, b'O', b'p', b'u', b's', 0x7F, 2, 0x80, 1];
        assert_eq!(parse_opus_descriptors(&desc), Some(1));
        // registration alone defaults to stereo; dual-mono code 0 maps to 2.
        assert_eq!(
            parse_opus_descriptors(&[0x05, 4, b'O', b'p', b'u', b's']),
            Some(2)
        );
        let dual = [0x05, 4, b'O', b'p', b'u', b's', 0x7F, 2, 0x80, 0];
        assert_eq!(parse_opus_descriptors(&dual), Some(2));
        // no 'Opus' registration: not Opus. Truncated descriptor: fails to None.
        assert_eq!(parse_opus_descriptors(&[0x7F, 2, 0x80, 2]), None);
        assert_eq!(parse_opus_descriptors(&[0x05, 40, b'O']), None);
    }

    #[test]
    fn opus_ts_control_headers_unwrap_and_bound() {
        // Two AUs: sizes 3 and 2, no trim, no extension (header 0x7FE0).
        let pes = [0x7F, 0xE0, 3, 9, 9, 9, 0x7F, 0xE0, 2, 8, 8];
        let pkts = opus_ts_packets(&pes);
        assert_eq!(pkts.len(), 2);
        assert_eq!(pkts[0], &[9, 9, 9]);
        assert_eq!(pkts[1], &[8, 8]);
        // au_size overrunning the payload drops the tail, keeps the first AU.
        let overrun = [0x7F, 0xE0, 3, 9, 9, 9, 0x7F, 0xE0, 200, 1];
        assert_eq!(opus_ts_packets(&overrun).len(), 1);
        // a 0xFF size-run that never terminates must not loop or panic.
        assert!(opus_ts_packets(&[0x7F, 0xE0, 0xFF, 0xFF, 0xFF]).is_empty());
        // start/end trim fields are skipped to reach the AU.
        let trimmed = [0x7F, 0xF8, 2, 0, 10, 0, 20, 5, 5];
        let p = opus_ts_packets(&trimmed);
        assert_eq!(p.len(), 1);
        assert_eq!(p[0], &[5, 5]);
    }

    /// Decode the PCR base (90 kHz) from the first TS packet on `pid` that carries
    /// one in its adaptation field, or `None` if none does.
    fn find_pcr(ts: &[u8], pid: u16) -> Option<u64> {
        for p in ts.chunks(TS_PACKET_LEN) {
            if p.len() != TS_PACKET_LEN || p[0] != SYNC_BYTE {
                continue;
            }
            let ppid = (((p[1] & 0x1F) as u16) << 8) | p[2] as u16;
            let afc = (p[3] >> 4) & 0x03;
            if ppid != pid || afc & 0x02 == 0 || p[4] == 0 || p[5] & 0x10 == 0 {
                continue;
            }
            return Some(
                ((p[6] as u64) << 25)
                    | ((p[7] as u64) << 17)
                    | ((p[8] as u64) << 9)
                    | ((p[9] as u64) << 1)
                    | ((p[10] as u64) >> 7),
            );
        }
        None
    }

    /// The PSI section body (from section[3], minus the trailing CRC) of the first
    /// section-start packet on `pid` in a muxed byte stream.
    fn psi_body_of(ts: &[u8], pid: u16) -> Option<Vec<u8>> {
        for p in ts.chunks(TS_PACKET_LEN) {
            if p.len() != TS_PACKET_LEN || p[0] != SYNC_BYTE || p[1] & 0x40 == 0 {
                continue;
            }
            if ((((p[1] & 0x1F) as u16) << 8) | p[2] as u16) != pid {
                continue;
            }
            let off = if (p[3] >> 4) & 0x02 != 0 {
                5 + p[4] as usize
            } else {
                4
            };
            let payload = &p[off..];
            let section = &payload[1 + payload[0] as usize..]; // skip pointer_field
            let section_length = (((section[1] & 0x0F) as usize) << 8) | section[2] as usize;
            return Some(section[3..3 + section_length - 4].to_vec());
        }
        None
    }

    /// The `(program_number, pmt_pid)` pairs a PAT body names.
    fn pat_programs(body: &[u8]) -> Vec<(u16, u16)> {
        body[5..]
            .chunks(4)
            .map(|c| {
                (
                    ((c[0] as u16) << 8) | c[1] as u16,
                    (((c[2] & 0x1F) as u16) << 8) | c[3] as u16,
                )
            })
            .collect()
    }

    /// A PMT body's program number, PCR_PID and `(stream_type, elementary_PID)`
    /// list. Assumes program_info_length 0 (what the muxer writes).
    fn pmt_entries(body: &[u8]) -> (u16, u16, Vec<(u8, u16)>) {
        let number = ((body[0] as u16) << 8) | body[1] as u16;
        let pcr_pid = (((body[5] & 0x1F) as u16) << 8) | body[6] as u16;
        let streams = body[9..]
            .chunks(5)
            .map(|c| (c[0], (((c[1] & 0x1F) as u16) << 8) | c[2] as u16))
            .collect();
        (number, pcr_pid, streams)
    }

    /// Every PID in a muxed byte stream paired with its continuity-counter run.
    fn cc_runs(ts: &[u8]) -> Vec<(u16, Vec<u8>)> {
        let mut runs: Vec<(u16, Vec<u8>)> = Vec::new();
        for p in ts.chunks(TS_PACKET_LEN) {
            let pid = (((p[1] & 0x1F) as u16) << 8) | p[2] as u16;
            let cc = p[3] & 0x0F;
            match runs.iter_mut().find(|(q, _)| *q == pid) {
                Some((_, v)) => v.push(cc),
                None => runs.push((pid, alloc::vec![cc])),
            }
        }
        runs
    }

    /// A two-program muxer: program 1 carries H.264 + AAC, program 2 a second
    /// H.264 stream. Access units are pushed by global stream index.
    fn two_program_ts() -> Vec<u8> {
        let mut m = TsMuxer::with_programs(&[
            (1, STREAM_TYPE_H264),
            (1, STREAM_TYPE_AAC),
            (2, STREAM_TYPE_H264),
        ]);
        let mut ts = m.push_au_on(0, &[0, 0, 0, 1, 0x65, 0x11], Some(900_000), None);
        ts.extend(m.push_au_on(1, &[0xFF, 0xF1, 0x22], Some(901_000), None));
        ts.extend(m.push_au_on(2, &[0, 0, 0, 1, 0x65, 0x33], Some(902_000), None));
        ts.extend(m.push_au_on(0, &[0, 0, 0, 1, 0x41, 0x44], Some(903_000), None));
        ts.extend(m.push_au_on(2, &[0, 0, 0, 1, 0x41, 0x55], Some(904_000), None));
        ts
    }

    #[test]
    fn multi_program_tables_name_each_program_separately() {
        let ts = two_program_ts();

        // The PAT names both programs, each on its own PMT PID.
        let pat = psi_body_of(&ts, PID_PAT).expect("PAT emitted");
        assert_eq!(
            pat_programs(&pat),
            alloc::vec![(1, MUX_PMT_PID), (2, MUX_PMT_PID + 1)]
        );

        // Each PMT lists only its own streams, with that program's first stream
        // as PCR_PID.
        let pmt1 = psi_body_of(&ts, MUX_PMT_PID).expect("program 1 PMT");
        assert_eq!(
            pmt_entries(&pmt1),
            (
                1,
                MUX_ES_PID,
                alloc::vec![
                    (STREAM_TYPE_H264, MUX_ES_PID),
                    (STREAM_TYPE_AAC, MUX_ES_PID + 1)
                ]
            )
        );
        let pmt2 = psi_body_of(&ts, MUX_PMT_PID + 1).expect("program 2 PMT");
        assert_eq!(
            pmt_entries(&pmt2),
            (
                2,
                MUX_ES_PID + 2,
                alloc::vec![(STREAM_TYPE_H264, MUX_ES_PID + 2)]
            )
        );

        // A PCR rides each program's PCR_PID, and no other stream.
        assert!(find_pcr(&ts, MUX_ES_PID).is_some(), "program 1 PCR");
        assert!(find_pcr(&ts, MUX_ES_PID + 2).is_some(), "program 2 PCR");
        assert!(
            find_pcr(&ts, MUX_ES_PID + 1).is_none(),
            "the audio stream is not a PCR_PID"
        );

        // Continuity counters run per PID, each starting at 0.
        for (pid, run) in cc_runs(&ts) {
            let want: Vec<u8> = (0..run.len() as u8).map(|i| i & 0x0F).collect();
            assert_eq!(run, want, "cc sequential on pid {pid:#x}");
        }
    }

    #[test]
    fn multi_program_mux_round_trips_per_program() {
        let ts = two_program_ts();
        let demux_program = |n: u16| {
            let mut d = TsDemuxer::new();
            d.set_program_number(Some(n));
            for pkt in ts.chunks(TS_PACKET_LEN) {
                d.push_packet(pkt);
            }
            d.flush();
            d.take_units()
        };

        // The AUs recovered on one PID, in stream order (units complete across
        // PIDs in reassembly order, so compare per PID).
        let on = |units: &[EsUnit], pid: u16| -> Vec<Vec<u8>> {
            units
                .iter()
                .filter(|u| u.pid == pid)
                .map(|u| u.data.clone())
                .collect()
        };

        // Program 1: both its streams, program 2's AUs filtered out.
        let p1 = demux_program(1);
        assert_eq!(p1.len(), 3, "program 1 carries three AUs");
        assert_eq!(
            on(&p1, MUX_ES_PID),
            alloc::vec![
                alloc::vec![0, 0, 0, 1, 0x65, 0x11],
                alloc::vec![0, 0, 0, 1, 0x41, 0x44]
            ]
        );
        assert_eq!(
            on(&p1, MUX_ES_PID + 1),
            alloc::vec![alloc::vec![0xFF, 0xF1, 0x22]]
        );

        // Program 2: only its own H.264 stream.
        let p2 = demux_program(2);
        assert_eq!(p2.len(), 2, "program 2 carries two AUs");
        assert_eq!(
            on(&p2, MUX_ES_PID + 2),
            alloc::vec![
                alloc::vec![0, 0, 0, 1, 0x65, 0x33],
                alloc::vec![0, 0, 0, 1, 0x41, 0x55]
            ]
        );
    }

    #[test]
    fn pcr_on_first_stream0_packet_carries_pts_minus_lead() {
        let au = alloc::vec![0u8, 0, 0, 1, 0x65, 0x88];
        let mut m = TsMuxer::new(STREAM_TYPE_H264);
        let out = m.push_au(&au, Some(900_000), None);
        assert_eq!(
            out.len() % TS_PACKET_LEN,
            0,
            "PCR packet is still 188 bytes"
        );
        assert_eq!(
            find_pcr(&out, MUX_ES_PID),
            Some(900_000 - PCR_LEAD_90KHZ),
            "PCR base is pts minus the 100 ms lead"
        );
        // A PTS under the lead saturates the base to 0 rather than wrapping.
        let mut m2 = TsMuxer::new(STREAM_TYPE_H264);
        let out2 = m2.push_au(&au, Some(100), None);
        assert_eq!(find_pcr(&out2, MUX_ES_PID), Some(0));
    }

    #[test]
    fn pcr_cadence_follows_the_interval() {
        let au = alloc::vec![0u8, 0, 0, 1, 0x65, 0x88];
        // Default interval (3600): AUs exactly 3600 ticks apart each carry a PCR.
        let mut m = TsMuxer::new(STREAM_TYPE_H264);
        assert!(find_pcr(&m.push_au(&au, Some(3600), None), MUX_ES_PID).is_some());
        assert!(find_pcr(&m.push_au(&au, Some(7200), None), MUX_ES_PID).is_some());

        // A huge interval: only the first PTS'd AU carries a PCR.
        let mut big = TsMuxer::new(STREAM_TYPE_H264);
        big.set_pcr_interval_90khz(u64::MAX);
        assert!(find_pcr(&big.push_au(&au, Some(0), None), MUX_ES_PID).is_some());
        assert!(find_pcr(&big.push_au(&au, Some(1_000_000), None), MUX_ES_PID).is_none());
    }

    #[test]
    fn audio_stream_never_carries_pcr() {
        // Stream 1 (audio) is not the PCR_PID, so its packets carry no PCR.
        let mut m = TsMuxer::with_streams(&[STREAM_TYPE_H264, STREAM_TYPE_AAC]);
        let adts = alloc::vec![0xFFu8, 0xF1, 0x00, 0x00];
        let out = m.push_au_on(1, &adts, Some(900_000), None);
        assert!(find_pcr(&out, MUX_ES_PID + 1).is_none());
    }

    #[test]
    fn build_pes_pts_only_and_pts_dts_headers() {
        // PTS-only: optional-header marker '10', PTS_DTS_flags '10', one 5-byte field.
        let pts_only = build_pes(0xE0, &[0xAA], Some(6000), None);
        assert_eq!(pts_only[6] & 0xC0, 0x80, "optional-header marker '10'");
        assert_eq!(pts_only[7] >> 6, 0b10, "PTS_DTS_flags = '10'");
        assert_eq!(pts_only[8], 5, "one 5-byte timestamp field");
        // A DTS equal to the PTS adds nothing: still PTS-only.
        let equal = build_pes(0xE0, &[0xAA], Some(6000), Some(6000));
        assert_eq!(equal[7] >> 6, 0b10);
        assert_eq!(equal[8], 5);
        // A distinct DTS: flags '11', two 5-byte fields decoding to PTS then DTS.
        let both = build_pes(0xE0, &[0xAA], Some(6000), Some(3600));
        assert_eq!(both[7] >> 6, 0b11, "PTS_DTS_flags = '11'");
        assert_eq!(both[8], 10, "two 5-byte timestamp fields");
        assert_eq!(decode_timestamp(&both[9..14]), 6000);
        assert_eq!(decode_timestamp(&both[14..19]), 3600);
    }

    #[test]
    fn pes_with_dts_round_trips_both_timestamps() {
        // A reordered (B-frame) AU: PTS ahead of DTS. Both survive mux -> demux.
        let au = alloc::vec![0u8, 0, 0, 1, 0x65, 0x11];
        let mut mux = TsMuxer::new(STREAM_TYPE_H264);
        let bytes = mux.push_au(&au, Some(6000), Some(3600));
        let mut d = TsDemuxer::new();
        for pkt in bytes.chunks(TS_PACKET_LEN) {
            d.push_packet(pkt);
        }
        d.flush();
        let units = d.take_units();
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].pts_90khz, Some(6000));
        assert_eq!(units[0].dts_90khz, Some(3600), "DTS recovered");

        // A PTS-only PES demuxes with no DTS.
        let mut mux2 = TsMuxer::new(STREAM_TYPE_H264);
        let bytes2 = mux2.push_au(&au, Some(9000), None);
        let mut d2 = TsDemuxer::new();
        for pkt in bytes2.chunks(TS_PACKET_LEN) {
            d2.push_packet(pkt);
        }
        d2.flush();
        let units2 = d2.take_units();
        assert_eq!(units2[0].pts_90khz, Some(9000));
        assert_eq!(units2[0].dts_90khz, None, "PTS-only stays PTS-only");
    }

    #[test]
    fn pcr_clocked_on_dts_holds_cadence_for_reordered_input() {
        // Decode order: DTS is monotonic (step = interval), PTS is reordered by
        // B-frames so it is non-monotonic. Clocking PCR on DTS keeps the cadence.
        let au = alloc::vec![0u8, 0, 0, 1, 0x65, 0x88];
        let interval = 3600u64;
        let frame_period = 3600u64;
        let base = 900_000u64; // keep DTS above the PCR lead (no saturation to 0)
        let pts_order = [3u64, 1, 2, 6, 4, 5];
        let mut m = TsMuxer::new(STREAM_TYPE_H264);
        let mut pcrs = Vec::new();
        for (i, &p) in pts_order.iter().enumerate() {
            let dts = base + i as u64 * frame_period;
            let pts = base + p * frame_period;
            let out = m.push_au(&au, Some(pts), Some(dts));
            if let Some(pcr) = find_pcr(&out, MUX_ES_PID) {
                pcrs.push(pcr);
            }
        }
        assert_eq!(pcrs.len(), pts_order.len(), "every AU carries a PCR");
        assert!(
            pcrs.windows(2).all(|w| w[1] > w[0]),
            "PCR strictly increases with DTS"
        );
        assert!(
            pcrs.windows(2)
                .all(|w| w[1] - w[0] <= interval + frame_period),
            "consecutive PCR gap stays within interval + one frame period"
        );
    }

    /// A synthetic KLV packet: a 16-byte SMPTE UL key, a 1-byte BER length and
    /// `n` body bytes.
    fn klv_packet(n: u8, fill: u8) -> Vec<u8> {
        let mut p = alloc::vec![
            0x06, 0x0E, 0x2B, 0x34, 0x02, 0x0B, 0x01, 0x01, 0x0E, 0x01, 0x03, 0x01, 0x01, 0x00,
            0x00, 0x00,
        ];
        p.push(n);
        p.extend(core::iter::repeat_n(fill, n as usize));
        p
    }

    /// Wrap each packet in a metadata AU cell (5-byte header, big-endian length).
    fn au_cell(seq: u8, payload: &[u8]) -> Vec<u8> {
        let mut c = alloc::vec![0x01, seq, 0x00];
        c.push((payload.len() >> 8) as u8);
        c.push(payload.len() as u8);
        c.extend_from_slice(payload);
        c
    }

    #[test]
    fn metadata_cells_unwrap_only_when_they_tile_and_carry_klv() {
        let a = klv_packet(4, 0xAA);
        let b = klv_packet(3, 0xBB);

        // Bare KLV (what ffmpeg writes) is forwarded unchanged.
        assert_eq!(unwrap_metadata_au_cells(&a), None);

        // Two cells tiling the payload unwrap to the packets they carry.
        let mut wrapped = au_cell(0, &a);
        wrapped.extend_from_slice(&au_cell(1, &b));
        let mut want = a.clone();
        want.extend_from_slice(&b);
        assert_eq!(unwrap_metadata_au_cells(&wrapped), Some(want));

        // A trailing byte past the last cell breaks the tiling: forward raw.
        let mut ragged = wrapped.clone();
        ragged.push(0x00);
        assert_eq!(unwrap_metadata_au_cells(&ragged), None);

        // A cell whose declared length runs past the payload: forward raw.
        let mut truncated = au_cell(0, &a);
        truncated.truncate(truncated.len() - 1);
        assert_eq!(unwrap_metadata_au_cells(&truncated), None);

        // Cells that tile but do not carry KLV: forward raw.
        let junk = au_cell(0, &[0x11, 0x22, 0x33]);
        assert_eq!(unwrap_metadata_au_cells(&junk), None);

        // Degenerate inputs stay raw rather than panicking.
        assert_eq!(unwrap_metadata_au_cells(&[]), None);
        assert_eq!(unwrap_metadata_au_cells(&[0xFF; 3]), None);
        assert_eq!(
            unwrap_metadata_au_cells(&[0x01, 0x00, 0x00, 0xFF, 0xFF]),
            None
        );
    }

    #[test]
    fn sync_klv_stream_gets_the_metadata_stream_id_and_descriptor() {
        let mut m = TsMuxer::with_streams(&[STREAM_TYPE_METADATA_PES]);
        let klv = klv_packet(2, 0xCD);
        let bytes = m.push_au(&klv, Some(9000), None);
        // PES start code + stream_id in the first payload byte run of the ES PID.
        assert!(
            bytes.windows(4).any(|w| w == [0x00, 0x00, 0x01, 0xFC]),
            "metadata PES uses stream_id 0xFC"
        );
        assert!(
            !bytes
                .windows(6)
                .any(|w| w == [0x05, 4, b'K', b'L', b'V', b'A']),
            "the KLVA registration descriptor stays on the async 0x06 path"
        );
        assert!(
            bytes
                .windows(KLV_METADATA_DESCRIPTOR.len())
                .any(|w| w == KLV_METADATA_DESCRIPTOR),
            "the PMT entry carries the metadata_descriptor"
        );
        let mut d = TsDemuxer::new();
        for pkt in bytes.chunks(TS_PACKET_LEN) {
            d.push_packet(pkt);
        }
        d.flush();
        let units = d.take_units();
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].stream_type, STREAM_TYPE_METADATA_PES);
        assert_eq!(
            units[0].pts_90khz,
            Some(9000),
            "sync KLV always carries a PTS"
        );
        assert_eq!(
            &units[0].data[..5],
            &[0x00, 0x00, 0xDF, 0x00, klv.len() as u8],
            "the KLV rides one metadata AU cell"
        );
        assert_eq!(
            unwrap_metadata_au_cells(&units[0].data),
            Some(klv),
            "and unwraps back to the packet"
        );
    }

    /// EN 300 468 Annex C's worked example: MJD 45218 is 1982-09-06, so with the
    /// BCD time 12:45:00 the field is 1982-09-06 12:45:00 UTC. The expected value
    /// is what `date -u -d @400164300` prints.
    #[test]
    fn eit_start_time_decodes_the_annex_c_example() {
        assert_eq!(
            mjd_bcd_to_unix_secs([0xB0, 0xA2, 0x12, 0x45, 0x00]),
            Some(400_164_300)
        );
    }

    /// The all-ones start_time the spec defines as undefined, a date before the
    /// Unix epoch, and a byte that is not BCD all report no time rather than a
    /// wrapped or garbled one.
    #[test]
    fn eit_start_time_declines_undefined_and_invalid_fields() {
        assert_eq!(mjd_bcd_to_unix_secs([0xFF; 5]), None, "undefined");
        let before_epoch = (MJD_UNIX_EPOCH - 1).to_be_bytes();
        assert_eq!(
            mjd_bcd_to_unix_secs([before_epoch[0], before_epoch[1], 0x00, 0x00, 0x00]),
            None,
            "1969 has no Unix seconds"
        );
        assert_eq!(
            mjd_bcd_to_unix_secs([0xB0, 0xA2, 0x1A, 0x45, 0x00]),
            None,
            "0xA is not a BCD nibble"
        );
    }

    /// The 3-byte BCD duration in seconds, and the same nibble check.
    #[test]
    fn eit_duration_decodes_bcd_hours_minutes_seconds() {
        assert_eq!(bcd_hms_secs([0x01, 0x30, 0x00]), Some(5400));
        assert_eq!(bcd_hms_secs([0x00, 0x00, 0x45]), Some(45));
        assert_eq!(bcd_hms_secs([0x0F, 0x00, 0x00]), None);
    }
}
