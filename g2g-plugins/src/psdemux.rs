//! MPEG program stream demuxer element (M929): `Caps::ByteStream{MpegPs}` in,
//! one selected elementary stream out. The `.mpg` / `.vob` sibling of
//! [`crate::tsdemux::TsDemux`], covering VCD-era MPEG-1 program streams and DVD
//! MPEG-2 ones alike.
//!
//! A program stream is a run of packs, each a `00 00 01 BA` header followed by
//! the PES packets belonging to it. There is no PAT/PMT: a stream is identified
//! by its PES `stream_id` (0xE0..=0xEF video, 0xC0..=0xDF audio) and, for the
//! `private_stream_1` (0xBD) DVD carries AC-3 and subpictures on, by the
//! substream id byte that opens its payload. Streams are therefore discovered by
//! observing packets rather than read from a table, so a probe has to see some
//! data before it knows what a file holds.
//!
//! ```text
//! filesrc location=x.vob ! mpegpsdemux ! ffmpegdec ! <sink>
//! mpegpsdemux stream=subpicture ! vobsubdec ! compositor.
//! ```
//!
//! Video geometry comes from the MPEG sequence header (`00 00 01 B3`), which the
//! demuxer parses to refine the video pad's caps via `CapsChanged` before the
//! first access unit; there is no separate MPEG-2 parse element. Scope: video,
//! MPEG audio, AC-3 and DVD subpictures. LPCM (0xA0..=0xA7) and DTS
//! (0x88..=0x8F) substreams, the program stream map (0xBC) and seeking are not
//! handled. CPU, `no_std` baseline.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::StreamSelectController;
use g2g_core::{
    AsyncElement, AudioFormat, BusHandle, BusMessage, ByteStreamEncoding, Caps, CapsConstraint,
    CapsSet, ConfigureOutcome, Dim, ElementMetadata, FrameTiming, G2gError, MemoryDomain,
    MultiOutputElement, MultiOutputSink, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
    PropError, PropKind, PropValue, PropertySpec, Rate, Seek, Segment, Stream, StreamCollection,
    StreamType, SubPictureFormat, VideoCodec,
};

use crate::mpeg2video::Mpeg2TimestampSynth;
pub use crate::mpeg2video::{parse_sequence_header, SequenceHeader};
use crate::mpegts::{decode_timestamp, parse_pes_header};
use crate::vobsub::MAX_SPU_BYTES;

/// Pack header start code id: `00 00 01 BA` opens every pack.
const PS_PACK: u8 = 0xBA;
/// Program end code.
const PS_END: u8 = 0xB9;
/// `private_stream_1`: AC-3, DTS, LPCM and DVD subpictures, told apart by the
/// substream id byte that opens the PES payload.
const PS_PRIVATE_1: u8 = 0xBD;

/// Largest number of distinct elementary streams recorded from one file. A
/// program stream declares nothing up front, so the set grows as packets are
/// seen; the bound keeps a crafted file cycling stream / substream ids from
/// growing the list without limit. Far above the 16 video + 32 audio + 32
/// subpicture streams the formats allow in practice.
const MAX_STREAMS: usize = 64;

/// Cap on one reassembled access unit. A unit is delimited by the next
/// timestamped packet of the same stream, so a stream that stamps nothing after
/// its first packet would otherwise grow the buffer without bound. 4 MiB is far
/// above any DVD-rate intra picture while bounding what an untrusted file costs.
const MAX_UNIT_BYTES: usize = 4 * 1024 * 1024;

/// One elementary stream observed in a program stream. There is no table to read
/// it from, so this is what the parser has actually seen a packet of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PsElementaryStream {
    /// The PES `stream_id`.
    pub stream_id: u8,
    /// The `private_stream_1` substream id, for a 0xBD stream only.
    pub substream_id: Option<u8>,
}

impl PsElementaryStream {
    /// Which selection forwards this stream, or `None` for one g2g does not
    /// carry (LPCM / DTS substreams, a private stream with no known substream).
    pub fn kind(&self) -> Option<PsStream> {
        match (self.stream_id, self.substream_id) {
            (0xE0..=0xEF, _) => Some(PsStream::Mpeg2),
            (0xC0..=0xDF, _) => Some(PsStream::Mp2),
            (PS_PRIVATE_1, Some(0x80..=0x87)) => Some(PsStream::Ac3),
            (PS_PRIVATE_1, Some(0x20..=0x3F)) => Some(PsStream::SubPicture),
            _ => None,
        }
    }
}

/// One reassembled access unit: a PES payload, or a whole subpicture unit for a
/// subtitle substream (those span several PES packets).
#[derive(Debug, Clone, PartialEq)]
pub struct PsUnit {
    pub stream_id: u8,
    pub substream_id: Option<u8>,
    /// Presentation timestamp in 90 kHz units, if the PES carried one.
    pub pts_90khz: Option<u64>,
    /// Decode timestamp in 90 kHz units, if the PES carried a separate one.
    pub dts_90khz: Option<u64>,
    /// The video geometry in effect for this access unit: the sequence header it
    /// opens with, or the last one seen before it. `None` for a non-video unit,
    /// and for video before any sequence header has parsed. Per unit rather than
    /// per file because a program stream can splice clips of different geometry.
    pub sequence: Option<SequenceHeader>,
    pub data: Vec<u8>,
}

/// A subpicture unit under reassembly: its first two bytes declare the total
/// size, and the packets after the first carry no timestamp of their own.
#[derive(Debug)]
struct PendingSpu {
    substream_id: u8,
    pts_90khz: Option<u64>,
    data: Vec<u8>,
}

/// Split a PES packet into its timestamps and elementary-stream bytes, for
/// either PES flavour. MPEG-2 packets (the `10` marker bits at byte 6) go
/// through the shared [`parse_pes_header`]; an MPEG-1 program stream instead
/// writes 0xFF stuffing, an optional STD buffer bound, and the timestamps under
/// their own `0010` / `0011` prefixes, which no MPEG-2 header can collide with
/// (its byte 6 is always 0x80..=0xBF).
fn ps_pes_payload(packet: &[u8]) -> (Option<u64>, Option<u64>, &[u8]) {
    match packet.get(6) {
        Some(b) if b & 0xC0 == 0x80 => parse_pes_header(packet),
        Some(_) => mpeg1_pes_payload(packet),
        None => (None, None, &[]),
    }
}

/// The MPEG-1 (ISO 11172-1) packet header: up to 16 stuffing bytes, an optional
/// 2-byte STD buffer bound, then a PTS (`0010`), a PTS + DTS (`0011`), or the
/// lone 0x0F that says neither is present. Every field is bounds-checked, so a
/// truncated or malformed header yields the payload with no timestamps rather
/// than panicking.
fn mpeg1_pes_payload(packet: &[u8]) -> (Option<u64>, Option<u64>, &[u8]) {
    let best_effort = || (None, None, packet.get(6..).unwrap_or(&[]));
    let mut at = 6usize;
    // ISO 11172-1 allows at most 16 stuffing bytes; more means this is not a
    // packet header at all.
    for _ in 0..16 {
        if packet.get(at) != Some(&0xFF) {
            break;
        }
        at += 1;
    }
    let Some(&b) = packet.get(at) else {
        return best_effort();
    };
    if b & 0xC0 == 0x40 {
        at += 2;
    }
    let Some(&b) = packet.get(at) else {
        return best_effort();
    };
    let (pts, dts, next) = match b & 0xF0 {
        0x20 => {
            let Some(f) = packet.get(at..at + 5) else {
                return best_effort();
            };
            (Some(decode_timestamp(f)), None, at + 5)
        }
        0x30 => {
            let (Some(p), Some(d)) = (packet.get(at..at + 5), packet.get(at + 5..at + 10)) else {
                return best_effort();
            };
            (
                Some(decode_timestamp(p)),
                Some(decode_timestamp(d)),
                at + 10,
            )
        }
        _ if b == 0x0F => (None, None, at + 1),
        // Not a header this parser understands: forward the bytes untouched.
        _ => return best_effort(),
    };
    (pts, dts, packet.get(next..).unwrap_or(&[]))
}

/// Cap on one audio stream's carry. An AC-3 syncframe is at most 3840 bytes and
/// an MPEG audio frame about 2 KiB, so a carry this large means the stream has
/// stopped parsing as frames at all: it is dropped and re-synced rather than
/// grown.
const MAX_AUDIO_CARRY: usize = 64 * 1024;

/// Which self-syncing audio bitstream an aligner is framing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AudioKind {
    Ac3,
    Mpa,
}

impl AudioKind {
    /// Length of the frame at the front of `buf`, or `None` when `buf` does not
    /// open on a usable frame header.
    fn frame_len(self, buf: &[u8]) -> Option<usize> {
        match self {
            Self::Ac3 => crate::audioframe::ac3_frame_len(buf),
            Self::Mpa => crate::audioframe::mpa_frame_len(buf),
        }
    }

    /// Bytes of header the length calculation needs, so "not a frame" can be
    /// told from "not enough bytes yet".
    fn header_len(self) -> usize {
        match self {
            Self::Ac3 => crate::audioframe::AC3_HEADER_LEN,
            Self::Mpa => crate::audioframe::MPA_HEADER_LEN,
        }
    }

    /// Offset of the next parseable frame header in `buf`, skipping `from`
    /// bytes. Used to acquire sync and to re-acquire it after a break.
    fn find_sync(self, buf: &[u8], from: usize) -> Option<usize> {
        (from..buf.len()).find(|&at| self.frame_len(&buf[at..]).is_some())
    }
}

/// Audio frame realignment for one elementary stream.
///
/// A program stream cuts its PES packets on 2 KiB sector boundaries with no
/// regard for audio frame boundaries, so a payload routinely begins and ends
/// mid-frame. Downstream decoding assumes the container delivers frame-aligned
/// access units (MPEG-TS aligns syncframes to PES packets, so that holds there),
/// and feeding it sector fragments drops all but the occasional frame that
/// happens to start a packet. This restores the alignment: carry the partial
/// tail across packets and emit only whole frames.
#[derive(Debug)]
struct AudioAligner {
    stream_id: u8,
    substream_id: Option<u8>,
    kind: AudioKind,
    /// Bytes of a frame that has not fully arrived, in front of the next ones.
    carry: Vec<u8>,
    /// Whether the front of `carry` is known to be a frame boundary.
    synced: bool,
    /// Timestamps for the unit being assembled: those of the packet whose first
    /// access unit opened it.
    pts_90khz: Option<u64>,
    dts_90khz: Option<u64>,
    /// Timestamps held for the unit after this one, when a packet arrived while
    /// a frame was still straddling the boundary.
    next_ts: Option<(Option<u64>, Option<u64>)>,
}

impl AudioAligner {
    fn new(stream_id: u8, substream_id: Option<u8>, kind: AudioKind) -> Self {
        Self {
            stream_id,
            substream_id,
            kind,
            carry: Vec::new(),
            synced: false,
            pts_90khz: None,
            dts_90khz: None,
            next_ts: None,
        }
    }

    /// Append one PES payload. `first_au` is the offset the container says the
    /// first frame starting in this packet lies at (the DVD substream header's
    /// pointer); it is verified against the frame header there before being
    /// trusted, and a scan takes over when it does not hold or is absent.
    fn push(
        &mut self,
        pts: Option<u64>,
        dts: Option<u64>,
        mut data: &[u8],
        first_au: Option<usize>,
        out: &mut Vec<PsUnit>,
    ) {
        if !self.synced {
            let at = first_au
                .filter(|&at| data.len() > at && self.kind.frame_len(&data[at..]).is_some())
                .or_else(|| self.kind.find_sync(data, 0));
            let Some(at) = at else {
                return; // no frame starts here: wait for one that does
            };
            self.carry.clear();
            self.synced = true;
            self.pts_90khz = pts;
            self.dts_90khz = dts;
            data = &data[at..];
        } else if pts.is_some() {
            if self.carry.is_empty() {
                // This packet opens the next unit outright.
                self.pts_90khz = pts;
                self.dts_90khz = dts;
            } else if self.next_ts.is_none() {
                // A frame is still straddling: this stamp applies once it drains.
                self.next_ts = Some((pts, dts));
            }
        }
        if self.carry.len().saturating_add(data.len()) > MAX_AUDIO_CARRY {
            self.carry.clear();
            self.synced = false;
            return;
        }
        self.carry.extend_from_slice(data);
        self.drain_frames(out);
    }

    /// Emit every whole frame at the front of the carry, re-acquiring sync if the
    /// front stops parsing as a frame header.
    fn drain_frames(&mut self, out: &mut Vec<PsUnit>) {
        loop {
            let mut end = 0;
            while let Some(len) = self.kind.frame_len(&self.carry[end..]) {
                if end + len > self.carry.len() {
                    break;
                }
                end += len;
            }
            if end > 0 {
                let unit: Vec<u8> = self.carry.drain(..end).collect();
                out.push(PsUnit {
                    stream_id: self.stream_id,
                    substream_id: self.substream_id,
                    pts_90khz: self.pts_90khz,
                    dts_90khz: self.dts_90khz,
                    sequence: None,
                    data: unit,
                });
                if let Some((pts, dts)) = self.next_ts.take() {
                    self.pts_90khz = pts;
                    self.dts_90khz = dts;
                }
            }
            // The front is not a whole frame. Either the rest has yet to arrive,
            // or the stream broke mid-frame and sync has to be re-acquired.
            let header = self.kind.header_len();
            if self.carry.len() < header || self.kind.frame_len(&self.carry).is_some() {
                return; // waiting for more bytes of a frame we can already size
            }
            match self.kind.find_sync(&self.carry, 1) {
                Some(at) => {
                    self.carry.drain(..at);
                }
                None => {
                    // Nothing parseable left: keep only what could still open a
                    // header once more bytes land.
                    let keep = self.carry.len().saturating_sub(header - 1);
                    self.carry.drain(..keep);
                    self.synced = false;
                    return;
                }
            }
        }
    }
}

/// Cap on the timestamp marks kept for one video stream. Each mark costs at
/// least one buffered byte, so the count is already bounded, but a stream of
/// tiny timestamped fragments would still make the list large; far more than
/// the handful of PES packets a picture really spans.
const MAX_MARKS: usize = 256;

/// Video access-unit framing. A program stream splits its PES packets on sector
/// boundaries with no regard for picture boundaries, so one packet can hold the
/// tail of a picture and the head of the next: unlike MPEG-TS, a PES payload is
/// not an access unit. The demuxer therefore reframes the video on its own start
/// codes, the job an elementary-stream parser does for the other codecs.
///
/// A unit runs from one picture (with any sequence / GOP header that opens it)
/// up to the next, and takes the timestamp of the PES packet its first byte fell
/// in. Everything is bounded: a stream that never shows a second picture is
/// dropped at [`MAX_UNIT_BYTES`] rather than buffered forever.
#[derive(Debug, Default)]
struct VideoReframer {
    buf: Vec<u8>,
    /// `(offset into buf, pts, dts)` in increasing offset order. The first entry
    /// is always at offset 0: the timestamp of the unit being assembled.
    marks: Vec<(usize, Option<u64>, Option<u64>)>,
    /// How far the start-code scan has reached.
    scanned: usize,
    /// Whether the unit being assembled already holds a picture header.
    have_picture: bool,
    /// Offset of the first sequence / GOP header seen after that picture: it
    /// opens the next unit, not this one.
    cut: Option<usize>,
}

impl VideoReframer {
    /// Append one PES fragment and emit every access unit it completes.
    fn push(
        &mut self,
        stream_id: u8,
        pts: Option<u64>,
        dts: Option<u64>,
        data: &[u8],
        out: &mut Vec<PsUnit>,
    ) {
        if data.is_empty() {
            return;
        }
        if self.buf.len().saturating_add(data.len()) > MAX_UNIT_BYTES {
            // Not a stream this parser can frame: drop it and start over at the
            // next picture rather than growing without bound.
            self.reset();
            return;
        }
        if self.marks.is_empty() {
            self.marks.push((0, pts, dts));
        } else if pts.is_some() && self.marks.len() < MAX_MARKS {
            let at = self.buf.len();
            match self.marks.last_mut() {
                // A real stamp may fill the placeholder a cut left at this
                // offset; a second real one cannot refine the first.
                Some(m) if m.0 == at => {
                    if m.1.is_none() {
                        (m.1, m.2) = (pts, dts);
                    }
                }
                _ => self.marks.push((at, pts, dts)),
            }
        }
        self.buf.extend_from_slice(data);
        self.scan(stream_id, out);
    }

    fn reset(&mut self) {
        self.buf.clear();
        self.marks.clear();
        self.scanned = 0;
        self.have_picture = false;
        self.cut = None;
    }

    /// Emit whatever is buffered as a final unit (end of stream).
    fn flush(&mut self, stream_id: u8, out: &mut Vec<PsUnit>) {
        if !self.buf.is_empty() {
            let (_, pts, dts) = self.marks.first().copied().unwrap_or((0, None, None));
            out.push(PsUnit {
                stream_id,
                substream_id: None,
                pts_90khz: pts,
                dts_90khz: dts,
                sequence: None,
                data: core::mem::take(&mut self.buf),
            });
        }
        self.reset();
    }

    fn scan(&mut self, stream_id: u8, out: &mut Vec<PsUnit>) {
        // Only a start code whose id byte has arrived can be classified; back the
        // scan up three bytes so a prefix split across fragments is still seen.
        while let Some(rel) = self.buf[self.scanned..]
            .windows(4)
            .position(|w| w[0] == 0 && w[1] == 0 && w[2] == 1)
        {
            let at = self.scanned + rel;
            let id = self.buf[at + 3];
            self.scanned = at + 4;
            match id {
                // Picture header: the second one in a unit ends the first.
                0x00 => {
                    if !self.have_picture {
                        self.have_picture = true;
                        self.cut = None;
                        continue;
                    }
                    let cut_at = self.cut.take().unwrap_or(at);
                    let (_, pts, dts) = self.marks.first().copied().unwrap_or((0, None, None));
                    let rest = self.buf.split_off(cut_at);
                    let unit = core::mem::replace(&mut self.buf, rest);
                    // A PES timestamp names the first access unit commencing in
                    // its packet, so a stamp whose packet began inside the
                    // emitted unit's bytes belongs to the next unit; the one at
                    // offset 0 was the emitted unit's own and is consumed with
                    // it. Units between stamps stay unstamped (`None`), and the
                    // demuxer synthesizes their PTS from temporal_reference:
                    // carrying the old stamp forward here made a whole GOP share
                    // one PTS, which a pacing sink plays as burst-and-freeze.
                    let carried = self
                        .marks
                        .iter()
                        .rev()
                        .find(|&&(off, _, _)| off > 0 && off <= cut_at)
                        .copied()
                        .map_or((0, None, None), |(_, p, d)| (0, p, d));
                    self.marks.retain(|&(off, _, _)| off > cut_at);
                    for m in self.marks.iter_mut() {
                        m.0 -= cut_at;
                    }
                    self.marks.insert(0, carried);
                    self.scanned -= cut_at;
                    self.have_picture = true;
                    if !unit.is_empty() {
                        out.push(PsUnit {
                            stream_id,
                            substream_id: None,
                            pts_90khz: pts,
                            dts_90khz: dts,
                            sequence: None,
                            data: unit,
                        });
                    }
                }
                // A sequence or GOP header after a picture opens the next unit.
                0xB3 | 0xB8 if self.have_picture && self.cut.is_none() => {
                    self.cut = Some(at);
                }
                _ => {}
            }
        }
        // Everything up to the last three bytes has been examined: a start code
        // needs a 3-byte prefix, so only those can still open one once the next
        // fragment lands. Parking the scan there keeps a stream with no start
        // codes at all from re-scanning the whole buffer on every fragment.
        self.scanned = self.scanned.max(self.buf.len().saturating_sub(3));
    }
}

/// Pure MPEG program stream parser: bytes in, reassembled access units out. The
/// program stream sibling of [`TsDemuxer`](crate::mpegts::TsDemuxer).
///
/// Every count, length and offset here comes off the wire, so all of them are
/// bounds-checked and the arithmetic is checked or saturating: a malformed pack,
/// an impossible packet length or a subpicture unit claiming a size it never
/// delivers drops that packet and resynchronizes to the next start code rather
/// than panicking or allocating on the claim.
#[derive(Debug, Default)]
pub struct PsDemuxer {
    /// Bytes not yet consumed as whole packs / packets.
    buf: Vec<u8>,
    completed: Vec<PsUnit>,
    /// Streams seen so far, in discovery order.
    seen: Vec<PsElementaryStream>,
    /// The video geometry, once a sequence header has been parsed.
    sequence: Option<SequenceHeader>,
    /// Video access-unit framing, one per video stream_id.
    video: Vec<(u8, VideoReframer)>,
    /// Audio frame realignment, one per audio stream / substream.
    audio: Vec<AudioAligner>,
    /// How many entries of `completed` the sequence-header scan has examined, so
    /// each unit is read once however often the scan runs.
    seq_scanned: usize,
    /// Subpicture units under reassembly, one per substream.
    spu: Vec<PendingSpu>,
    /// Video timestamp synthesis state, one per video stream_id.
    video_ts: Vec<(u8, Mpeg2TimestampSynth)>,
}

impl PsDemuxer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a chunk of the byte stream, parsing every whole pack / packet it
    /// completes. A trailing partial packet stays buffered for the next call.
    pub fn push_data(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
        self.parse();
    }

    /// Take the access units reassembled so far.
    ///
    /// Video units seen before the first sequence header are dropped. A byte
    /// range cut from the middle of a VOB joins mid-GOP, so the first pictures
    /// arrive with no geometry ahead of them and no decoder can accept them
    /// (libavcodec's MPEG-2 decoder fails the whole stream on "invalid frame
    /// dimensions"). Dropping to the first sync point is the tune-in convention
    /// the rest of the tree follows, `RtspSrc` dropping until the first IDR.
    /// A unit's geometry stamp is exactly that test: it is set from the first
    /// unit carrying a sequence header on.
    pub fn take_units(&mut self) -> Vec<PsUnit> {
        self.seq_scanned = 0;
        let mut units = core::mem::take(&mut self.completed);
        units.retain(|u| !(0xE0..=0xEF).contains(&u.stream_id) || u.sequence.is_some());
        units
    }

    /// The elementary streams seen so far, in discovery order.
    pub fn streams(&self) -> &[PsElementaryStream] {
        &self.seen
    }

    /// The video geometry the sequence header declared, if one has been parsed.
    pub fn sequence(&self) -> Option<SequenceHeader> {
        self.sequence
    }

    /// End of stream: emit whatever access units are still being assembled, and
    /// drop any half-assembled subpicture unit (its declared size never arrived,
    /// so it can never complete). An audio carry is dropped too: it is a partial
    /// frame by definition, and downstream expects whole ones.
    pub fn flush(&mut self) {
        for (stream_id, r) in self.video.iter_mut() {
            r.flush(*stream_id, &mut self.completed);
        }
        self.audio.clear();
        self.spu.clear();
        self.sequence_from_completed();
    }

    fn record(&mut self, stream_id: u8, substream_id: Option<u8>) {
        let es = PsElementaryStream {
            stream_id,
            substream_id,
        };
        if self.seen.contains(&es) || self.seen.len() >= MAX_STREAMS {
            return;
        }
        self.seen.push(es);
    }

    /// Consume every whole pack / packet in `buf`, resynchronizing to the next
    /// start code whenever a header does not parse.
    fn parse(&mut self) {
        loop {
            // Resync: drop everything before the next start code prefix, keeping
            // the last two bytes in case a prefix straddles the chunk boundary.
            match find_start_code(&self.buf) {
                Some(0) => {}
                Some(pos) => {
                    self.buf.drain(..pos);
                }
                None => {
                    let keep = self.buf.len().saturating_sub(2);
                    self.buf.drain(..keep);
                    return;
                }
            }
            let Some(&id) = self.buf.get(3) else {
                return;
            };
            match id {
                PS_PACK => match pack_header_len(&self.buf) {
                    // Not enough bytes yet to know the pack header's length.
                    Some(None) => return,
                    Some(Some(len)) => {
                        self.buf.drain(..len);
                    }
                    // Neither an MPEG-1 nor an MPEG-2 pack: skip the start code.
                    None => {
                        self.buf.drain(..4);
                    }
                },
                PS_END => {
                    self.buf.drain(..4);
                }
                // Every other PS start code is length prefixed.
                _ => {
                    let Some(len) = self.buf.get(4..6) else {
                        return;
                    };
                    let len = u16::from_be_bytes([len[0], len[1]]) as usize;
                    // A program stream always declares a packet length; a zero
                    // one would leave the packet unbounded, so treat it as a bad
                    // header and resync.
                    if len == 0 {
                        self.buf.drain(..4);
                        continue;
                    }
                    let total = 6 + len;
                    if self.buf.len() < total {
                        return;
                    }
                    if matches!(id, PS_PRIVATE_1 | 0xC0..=0xEF) {
                        // Copied out so the payload parse does not borrow `buf`
                        // while the reassembly below mutates `self`.
                        let packet: Vec<u8> = self.buf[..total].to_vec();
                        self.handle_pes(id, &packet);
                    }
                    self.buf.drain(..total);
                }
            }
        }
    }

    fn handle_pes(&mut self, stream_id: u8, packet: &[u8]) {
        let (pts, dts, es) = ps_pes_payload(packet);
        if stream_id != PS_PRIVATE_1 {
            self.record(stream_id, None);
            if (0xE0..=0xEF).contains(&stream_id) {
                // Video is reframed on picture start codes: a program stream's
                // PES packets do not align to access units.
                let at = match self.video.iter().position(|(id, _)| *id == stream_id) {
                    Some(i) => i,
                    None if self.video.len() < MAX_STREAMS => {
                        self.video.push((stream_id, VideoReframer::default()));
                        self.video.len() - 1
                    }
                    None => return,
                };
                self.video[at]
                    .1
                    .push(stream_id, pts, dts, es, &mut self.completed);
                self.sequence_from_completed();
                return;
            }
            // MPEG audio is sector-cut the same way video is, so it is realigned
            // onto frame boundaries too. There is no first-access-unit pointer
            // outside the DVD private stream, so sync comes from a header scan.
            if (0xC0..=0xDF).contains(&stream_id) {
                self.push_audio(stream_id, None, AudioKind::Mpa, pts, dts, es, None);
            }
            return;
        }
        // DVD private_stream_1: a substream id byte opens the payload.
        let Some((&sub, rest)) = es.split_first() else {
            return;
        };
        match sub {
            // AC-3: a frame-header count byte and a 2-byte pointer to the first
            // access unit follow the substream id, then the AC-3 frames. The
            // pointer is 1-based from the byte after it (verified against a
            // retail NTSC VOB), and 0 says no frame starts in this packet.
            0x80..=0x87 => {
                let Some(head) = rest.get(..3) else {
                    return;
                };
                let first_au = u16::from_be_bytes([head[1], head[2]])
                    .checked_sub(1)
                    .map(usize::from);
                let Some(data) = rest.get(3..) else {
                    return;
                };
                self.record(stream_id, Some(sub));
                self.push_audio(
                    stream_id,
                    Some(sub),
                    AudioKind::Ac3,
                    pts,
                    dts,
                    data,
                    first_au,
                );
            }
            0x20..=0x3F => {
                self.record(stream_id, Some(sub));
                self.push_spu(sub, pts, rest);
            }
            // LPCM and DTS substreams are out of scope; anything else is not a
            // substream this parser knows.
            _ => {}
        }
    }

    /// Feed one PES payload to its stream's frame aligner, creating one on first
    /// sight of the stream.
    #[allow(clippy::too_many_arguments)]
    fn push_audio(
        &mut self,
        stream_id: u8,
        substream_id: Option<u8>,
        kind: AudioKind,
        pts: Option<u64>,
        dts: Option<u64>,
        data: &[u8],
        first_au: Option<usize>,
    ) {
        let at = match self
            .audio
            .iter()
            .position(|a| a.stream_id == stream_id && a.substream_id == substream_id)
        {
            Some(i) => i,
            None if self.audio.len() < MAX_STREAMS => {
                self.audio
                    .push(AudioAligner::new(stream_id, substream_id, kind));
                self.audio.len() - 1
            }
            None => return,
        };
        self.audio[at].push(pts, dts, data, first_au, &mut self.completed);
    }

    /// Read the video geometry off each newly completed video unit that carries a
    /// sequence header. A header can straddle two PES packets, so it is read from
    /// the reframed unit rather than from a fragment.
    ///
    /// The latest header wins, not the first: a program stream can splice clips
    /// of different geometry (concatenated DVD titles), and the caps have to
    /// follow. Units are examined once each, so this stays linear however many
    /// times the scan runs before `take_units` drains the list.
    fn sequence_from_completed(&mut self) {
        while self.seq_scanned < self.completed.len() {
            let at = self.seq_scanned;
            self.seq_scanned += 1;
            let stream_id = self.completed[at].stream_id;
            if !(0xE0..=0xEF).contains(&stream_id) {
                continue;
            }
            if let Some(seq) = parse_sequence_header(&self.completed[at].data) {
                self.sequence = Some(seq);
            }
            self.completed[at].sequence = self.sequence;
            // Synthesize the unstamped pictures' timestamps (M934). Needs the
            // frame period, so units before the first sequence header stay
            // untouched; `take_units` drops those anyway (mid-GOP tune-in).
            let Some(period_90) = self.completed[at]
                .sequence
                .and_then(|s| crate::mpegts::frame_period_90khz(s.framerate_q16))
            else {
                continue;
            };
            let ts = match self.video_ts.iter().position(|(id, _)| *id == stream_id) {
                Some(i) => i,
                None => {
                    self.video_ts
                        .push((stream_id, Mpeg2TimestampSynth::default()));
                    self.video_ts.len() - 1
                }
            };
            let unit = &mut self.completed[at];
            self.video_ts[ts].1.stamp(
                &unit.data,
                &mut unit.pts_90khz,
                &mut unit.dts_90khz,
                period_90,
            );
        }
    }

    /// Append one subpicture fragment, emitting the unit once its declared size
    /// has arrived. The size is the unit's own first two bytes, so it is checked
    /// against the 16-bit maximum a subpicture can declare before anything is
    /// kept: a unit that overruns it is a stream whose packets do not belong to
    /// one cue, and the reassembly is dropped.
    fn push_spu(&mut self, substream_id: u8, pts_90khz: Option<u64>, data: &[u8]) {
        let idx = match self.spu.iter().position(|p| p.substream_id == substream_id) {
            Some(i) => i,
            None => {
                // A continuation with nothing to continue: wait for a packet
                // that opens a unit (only the first one carries a PTS).
                if pts_90khz.is_none() {
                    return;
                }
                if self.spu.len() >= MAX_STREAMS {
                    return;
                }
                self.spu.push(PendingSpu {
                    substream_id,
                    pts_90khz,
                    data: Vec::new(),
                });
                self.spu.len() - 1
            }
        };
        // A timestamped packet opens a new unit, so anything left over from an
        // unfinished one is stale.
        if pts_90khz.is_some() && !self.spu[idx].data.is_empty() {
            self.spu[idx].data.clear();
            self.spu[idx].pts_90khz = pts_90khz;
        }
        if self.spu[idx].data.len().saturating_add(data.len()) > MAX_SPU_BYTES {
            self.spu.swap_remove(idx);
            return;
        }
        self.spu[idx].data.extend_from_slice(data);
        let size = match self.spu[idx].data.get(..2) {
            Some(s) => u16::from_be_bytes([s[0], s[1]]) as usize,
            None => return,
        };
        // The size covers the two bytes that declare it plus the control-sequence
        // offset, so anything under 4 is not a subpicture unit.
        if size < 4 {
            self.spu.swap_remove(idx);
            return;
        }
        if self.spu[idx].data.len() < size {
            return;
        }
        let mut pending = self.spu.swap_remove(idx);
        pending.data.truncate(size);
        self.completed.push(PsUnit {
            stream_id: PS_PRIVATE_1,
            substream_id: Some(substream_id),
            pts_90khz: pending.pts_90khz,
            dts_90khz: None,
            sequence: None,
            data: pending.data,
        });
    }
}

/// Offset of the next `00 00 01` start code prefix in `buf`.
fn find_start_code(buf: &[u8]) -> Option<usize> {
    buf.windows(3).position(|w| w == [0x00, 0x00, 0x01])
}

/// Total length of the pack header at the start of `buf` (start code included).
/// `None` when the marker bits name neither pack layout; `Some(None)` when the
/// pack is not yet fully buffered. MPEG-2 packs open with the bits `01` and
/// carry a 10-byte field plus up to 7 stuffing bytes; MPEG-1 packs open with
/// `0010` and are a flat 8 bytes.
///
/// The length is only reported once every byte it covers is present: the
/// stuffing count is read from the header itself, so a truncated pack would
/// otherwise name a length past the end of the buffer.
fn pack_header_len(buf: &[u8]) -> Option<Option<usize>> {
    let Some(&b) = buf.get(4) else {
        return Some(None);
    };
    let len = if b & 0xC0 == 0x40 {
        let Some(&stuffing) = buf.get(13) else {
            return Some(None);
        };
        4 + 10 + (stuffing & 0x07) as usize
    } else if b & 0xF0 == 0x20 {
        4 + 8
    } else {
        return None;
    };
    Some((buf.len() >= len).then_some(len))
}

/// Which elementary stream a [`PsDemux`] instance forwards. A program stream
/// carries several; this element has one output pad, so the selection picks one,
/// by codec as in [`TsStream`](crate::tsdemux::TsStream) and for the same reason:
/// the output caps are fixed at negotiation, before any pack is parsed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PsStream {
    /// The first MPEG-1 / MPEG-2 video stream (stream_id 0xE0..=0xEF). The
    /// default.
    #[default]
    Mpeg2,
    /// The first MPEG audio stream (stream_id 0xC0..=0xDF), Layer II on a VCD or
    /// DVD.
    Mp2,
    /// The first AC-3 stream: a `private_stream_1` substream 0x80..=0x87, the DVD
    /// carriage of Dolby Digital.
    Ac3,
    /// The first DVD subpicture stream: a `private_stream_1` substream
    /// 0x20..=0x3F. Each cue is one subpicture unit, reassembled across the PES
    /// packets that carry it.
    SubPicture,
}

/// Demuxes an MPEG program stream into one selected elementary stream.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::psdemux::{PsDemux, PsStream};
///
/// let demux = PsDemux::new().with_stream(PsStream::Ac3);
/// assert_eq!(demux.stream(), PsStream::Ac3);
/// ```
#[derive(Debug)]
pub struct PsDemux {
    demux: PsDemuxer,
    /// The elementary stream this instance forwards (the single output pad).
    stream: PsStream,
    configured: bool,
    emitted: u64,
    /// Pipeline bus, for announcing the file's `StreamCollection`. Inert unless
    /// `with_bus` wired it.
    bus: Option<BusHandle>,
    /// Set once the `StreamCollection` has been announced, so it posts once.
    collection_posted: bool,
    /// The output caps last announced, so the sequence header's geometry emits
    /// one `CapsChanged` and no more.
    last_caps: Option<Caps>,
    /// Subpicture only: whether the synthesized `.idx` config has gone out in
    /// band ahead of the first cue.
    config_sent: bool,
    /// Set once the stream-start `Segment` has gone out. A DVD title's PTS
    /// starts wherever the disc puts it (a mid-title slice can open at hundreds
    /// of seconds), so the first frame's PTS must be mapped to running time 0
    /// or a paced sink holds every frame until that wall-clock offset passes.
    segment_sent: bool,
}

impl Default for PsDemux {
    fn default() -> Self {
        Self::new()
    }
}

impl PsDemux {
    pub fn new() -> Self {
        Self {
            demux: PsDemuxer::new(),
            stream: PsStream::Mpeg2,
            configured: false,
            emitted: 0,
            bus: None,
            collection_posted: false,
            last_caps: None,
            config_sent: false,
            segment_sent: false,
        }
    }

    /// Select which elementary stream to forward (default [`PsStream::Mpeg2`]).
    pub fn with_stream(mut self, stream: PsStream) -> Self {
        self.stream = stream;
        self
    }

    /// Attach the pipeline bus so the file's `StreamCollection` is announced once
    /// the first packets of each stream have been seen, the program stream
    /// sibling of [`TsDemux::with_bus`](crate::tsdemux::TsDemux::with_bus).
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.bus = Some(bus);
        self
    }

    /// The elementary stream this instance forwards.
    pub fn stream(&self) -> PsStream {
        self.stream
    }

    /// Count of frames forwarded.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    /// The input this element accepts: an MPEG program stream.
    fn input_caps() -> Caps {
        Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegPs,
        }
    }

    /// The output caps for a selection, at the placeholder geometry the sequence
    /// header refines via `CapsChanged` (never `Dim::Any`, which cannot fixate).
    fn output_caps(stream: PsStream) -> Caps {
        match stream {
            PsStream::Mpeg2 => Caps::CompressedVideo {
                codec: VideoCodec::Mpeg2,
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
            },
            PsStream::Mp2 => Caps::Audio {
                format: AudioFormat::Mp2,
                channels: 0,
                sample_rate: 0,
            },
            PsStream::Ac3 => Caps::Audio {
                format: AudioFormat::Ac3,
                channels: 0,
                sample_rate: 0,
            },
            PsStream::SubPicture => Caps::SubPicture {
                format: SubPictureFormat::VobSub,
            },
        }
    }

    /// The video caps a sequence header fixes, or `None` for a non-video
    /// selection or a unit seen before any header parsed.
    fn refined_video_caps(stream: PsStream, seq: Option<SequenceHeader>) -> Option<Caps> {
        if stream != PsStream::Mpeg2 {
            return None;
        }
        let seq = seq?;
        Some(Caps::CompressedVideo {
            codec: VideoCodec::Mpeg2,
            width: Dim::Fixed(seq.width),
            height: Dim::Fixed(seq.height),
            framerate: Rate::Fixed(seq.framerate_q16),
        })
    }

    /// Announce the observed elementary streams as a
    /// [`BusMessage::StreamCollection`], once. A program stream has no table to
    /// read, so the collection is what the parser has seen packets of by now.
    fn post_stream_collection(&mut self) {
        if self.collection_posted {
            return;
        }
        let streams: Vec<Stream> = self
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
                "mpegps-0", streams,
            )));
        }
    }

    /// Map one observed elementary stream to a [`Stream`] for the collection.
    fn es_to_stream(es: &PsElementaryStream) -> Option<Stream> {
        let kind = es.kind()?;
        let stream_type = match kind {
            PsStream::Mpeg2 => StreamType::Video,
            PsStream::Mp2 | PsStream::Ac3 => StreamType::Audio,
            PsStream::SubPicture => StreamType::Text,
        };
        Some(Stream::new(
            stream_id(es),
            stream_type,
            Self::output_caps(kind),
        ))
    }

    /// Emit each completed access unit of the selected stream as a frame.
    async fn emit_units(
        &mut self,
        units: Vec<PsUnit>,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        for u in units {
            let es = PsElementaryStream {
                stream_id: u.stream_id,
                substream_id: u.substream_id,
            };
            if es.kind() != Some(self.stream) {
                continue; // a stream other than the selected one
            }
            let pts_ns = ns_from_90khz(u.pts_90khz).unwrap_or(0);
            // Reordered video carries a separate DTS; fall back to the PTS when
            // the packet had none (the convention the TS / mkv demuxers share).
            let dts_ns = ns_from_90khz(u.dts_90khz).unwrap_or(pts_ns);
            if !self.segment_sent {
                self.segment_sent = true;
                let seg = Segment::for_flush_seek(&Seek::flush_to(pts_ns), None);
                out.push(PipelinePacket::Segment(seg)).await?;
            }
            // The sequence header is only known once a video packet has been
            // parsed, so the refinement rides ahead of the frame it describes,
            // and follows a mid-stream splice that changes it.
            if let Some(caps) = Self::refined_video_caps(self.stream, u.sequence) {
                if self.last_caps.as_ref() != Some(&caps) {
                    self.last_caps = Some(caps.clone());
                    out.push(PipelinePacket::CapsChanged(caps)).await?;
                }
            }
            // The `.idx` needs the video's geometry, which is only known once a
            // video access unit has completed. A cue can arrive first (a stream
            // joined mid-title), so the send is only marked done once it really
            // went out: a later cue then still gets the geometry rather than the
            // decoder keeping its default forever.
            if self.stream == PsStream::SubPicture && !self.config_sent {
                if let Some(blob) = idx_config_blob(&self.demux) {
                    self.config_sent = true;
                    let frame = config_frame(blob, pts_ns, self.emitted);
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
            }
            let duration_ns = cue_duration_ns(self.stream, &u.data);
            let keyframe = unit_is_keyframe(self.stream, &u.data);
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(u.data.into_boxed_slice())),
                FrameTiming {
                    pts_ns,
                    dts_ns,
                    duration_ns,
                    keyframe,
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

/// The `.idx` text a subpicture pad opens with: just the display size, the
/// video's own geometry, so an NTSC disc's cues land on a 720x480 canvas rather
/// than the decoder's PAL default. A program stream carries no palette, so none
/// is written and the decoder's default one renders the cues. `None` before any
/// sequence header has parsed. Shared by both demuxers so a subtitle branch
/// renders the same whichever one built it.
fn idx_config_blob(demux: &PsDemuxer) -> Option<Vec<u8>> {
    let seq = demux.sequence()?;
    let cfg = crate::vobsub::VobSubConfig {
        size: Some((seq.width, seq.height)),
        palette: None,
    };
    Some(crate::vobsub::idx_config_text(&cfg).into_bytes())
}

/// The published stream id of an elementary stream: the id it takes in the
/// `StreamCollection`.
fn stream_id(es: &PsElementaryStream) -> alloc::string::String {
    match es.substream_id {
        Some(sub) => alloc::format!("mpegps-{:02x}-{:02x}", es.stream_id, sub),
        None => alloc::format!("mpegps-{:02x}", es.stream_id),
    }
}

/// Resolve a collection stream id back to the selection that carries it.
fn resolve_ps_stream_id(demux: &PsDemuxer, id: &str) -> Option<PsStream> {
    demux
        .streams()
        .iter()
        .find(|es| stream_id(es) == id)
        .and_then(|es| es.kind())
}

/// Whether a unit begins an independently-decodable point. An MPEG video access
/// unit is one when it opens the sequence or a GOP, or carries an I-picture;
/// every audio frame and every subpicture cue is one by construction.
fn unit_is_keyframe(stream: PsStream, data: &[u8]) -> bool {
    match stream {
        PsStream::Mpeg2 => crate::annexb::au_is_keyframe(VideoCodec::Mpeg2, data),
        PsStream::Mp2 | PsStream::Ac3 | PsStream::SubPicture => true,
    }
}

/// A subpicture cue's own display duration: its control sequence carries the
/// hide time, so the duration is exact rather than "until the next cue". Zero
/// for any other stream, and for a unit whose control sequence sets no stop
/// date (the decoder then holds the cue until the next one).
fn cue_duration_ns(stream: PsStream, data: &[u8]) -> u64 {
    if stream != PsStream::SubPicture {
        return 0;
    }
    crate::vobsub::spu_timing(data)
        .and_then(|(start, stop)| stop.map(|stop| stop.saturating_sub(start)))
        .unwrap_or(0)
}

fn ns_from_90khz(t: Option<u64>) -> Option<u64> {
    t.map(|t| (t as u128 * 1_000_000_000 / 90_000) as u64)
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

impl AsyncElement for PsDemux {
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
            Caps::ByteStream {
                encoding: ByteStreamEncoding::MpegPs,
            } => CapsSet::one(Self::output_caps(stream)),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !matches!(
            absolute_caps,
            Caps::ByteStream {
                encoding: ByteStreamEncoding::MpegPs
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
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let Some(slice) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.demux.push_data(slice);
                    if self.bus.is_some() {
                        self.post_stream_collection();
                    }
                    let units = self.demux.take_units();
                    self.emit_units(units, out).await?;
                }
                PipelinePacket::Flush => {
                    self.demux = PsDemuxer::new();
                    self.config_sent = false;
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Eos => {
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

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "MPEG program stream demuxer",
            "Codec/Demuxer",
            "Demuxes an MPEG-1 / MPEG-2 program stream (.mpg / .vob): MPEG video, \
             MPEG audio, AC-3, and DVD subpictures",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        PSDEMUX_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "stream" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.stream = ps_stream_from_str(s).ok_or(PropError::Value)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "stream" => Some(PropValue::Str(ps_stream_to_str(self.stream).into())),
            _ => None,
        }
    }
}

/// `PsDemux`'s settable properties (M929).
static PSDEMUX_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "stream",
    PropKind::Str,
    "elementary stream to emit: mpeg2 | mp2 | ac3 | subpicture",
)];

/// Parse a `stream` property string to a [`PsStream`].
fn ps_stream_from_str(s: &str) -> Option<PsStream> {
    match s {
        "mpeg2" => Some(PsStream::Mpeg2),
        "mp2" => Some(PsStream::Mp2),
        "ac3" => Some(PsStream::Ac3),
        "subpicture" => Some(PsStream::SubPicture),
        _ => None,
    }
}

/// The `stream` property string for a [`PsStream`].
pub(crate) fn ps_stream_to_str(stream: PsStream) -> &'static str {
    match stream {
        PsStream::Mpeg2 => "mpeg2",
        PsStream::Mp2 => "mp2",
        PsStream::Ac3 => "ac3",
        PsStream::SubPicture => "subpicture",
    }
}

impl PadTemplates for PsDemux {
    fn pad_templates() -> Vec<PadTemplate> {
        let source = CapsSet::from_alternatives(Vec::from([
            Self::output_caps(PsStream::Mpeg2),
            Self::output_caps(PsStream::Mp2),
            Self::output_caps(PsStream::Ac3),
            Self::output_caps(PsStream::SubPicture),
        ]));
        Vec::from([
            PadTemplate::sink(CapsSet::one(Self::input_caps())),
            PadTemplate::source(source),
        ])
    }
}

/// One forwardable elementary stream discovered in a probed program stream:
/// which [`PsStream`] a demux port would carry, the elementary [`Caps`] a decode
/// branch plugs from, and whether it is video. The program stream analog of
/// [`TsStreamInfo`](crate::tsdemux::TsStreamInfo).
#[derive(Debug, Clone)]
pub struct PsStreamInfo {
    pub stream: PsStream,
    pub caps: Caps,
    pub video: bool,
}

/// The forwardable elementary streams a probed program stream carries, in
/// discovery order: one entry per distinct selection seen. Subpictures stay out,
/// as DVB subtitles do on the MPEG-TS side: this feeds the `playbin` fan-out,
/// which builds A/V decode branches only (a subtitle port is selected explicitly
/// on `PsDemux` / `PsDemuxN`). Returns empty for a non-program-stream or a prefix
/// too short to have shown a packet, which the `playbin` hook reads as "decline".
pub fn forwardable_streams(demux: &PsDemuxer) -> Vec<PsStreamInfo> {
    let mut out: Vec<PsStreamInfo> = Vec::new();
    for es in demux.streams() {
        let Some(stream) = es.kind() else {
            continue;
        };
        if stream == PsStream::SubPicture || out.iter().any(|i| i.stream == stream) {
            continue;
        }
        out.push(PsStreamInfo {
            stream,
            caps: PsDemux::output_caps(stream),
            video: stream == PsStream::Mpeg2,
        });
    }
    out
}

/// The subpicture streams a probed program stream carries, in discovery order.
/// Separate from [`forwardable_streams`] because the A/V fan-out does not plug
/// them; a caller that wants the subtitle track asks for it.
pub fn subpicture_streams(demux: &PsDemuxer) -> Vec<PsElementaryStream> {
    demux
        .streams()
        .iter()
        .filter(|es| es.kind() == Some(PsStream::SubPicture))
        .copied()
        .collect()
}

/// Multi-output MPEG program stream demuxer: one PS byte stream in, N elementary
/// streams out, one selected [`PsStream`] per output port. The program stream
/// sibling of [`TsDemuxN`](crate::tsdemux::TsDemuxN), driven by
/// [`run_source_fanout`](g2g_core::runtime::run_source_fanout).
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::psdemux::{PsDemuxN, PsStream};
///
/// let demux = PsDemuxN::new(vec![PsStream::Mpeg2, PsStream::Ac3]);
/// assert_eq!(demux.port_count(), 2);
/// ```
#[derive(Debug)]
pub struct PsDemuxN {
    demux: PsDemuxer,
    /// Port `i` emits this elementary stream.
    ports: Vec<PsStream>,
    /// Whether port `i` has emitted its opening `CapsChanged` yet.
    announced: Vec<bool>,
    /// The geometry port `i` last announced, so a mid-stream change re-announces
    /// and an unchanged one does not.
    refined: Vec<Option<SequenceHeader>>,
    bus: Option<BusHandle>,
    collection_posted: bool,
    /// App-driven stream selection: the app names the stream id each port should
    /// carry (port `i` <- selection id `i`). Inert unless `with_stream_select`
    /// wired it.
    stream_select: Option<StreamSelectController>,
    emitted: u64,
    /// Stream-start PTS (ns) latched from the first routed unit; every port's
    /// opening `Segment` maps it to running time 0 so A/V stay aligned and a
    /// mid-title slice's large PTS does not stall a paced sink.
    segment_base: Option<u64>,
    /// Whether port `i` has emitted its opening `Segment` yet.
    segment_sent: Vec<bool>,
    /// Whether port `i` has forwarded the synthesized `.idx` geometry in band
    /// ahead of its first cue (subpicture ports only), the per-port form of
    /// [`PsDemux`]'s `config_sent`.
    config_sent: Vec<bool>,
    /// Video geometry known before the run, from a probe that already read the
    /// sequence header. A video port otherwise advertises a fixatable `Range`
    /// (the size is unknown until a unit parses) and the solver fixates a range
    /// at its minimum, so the branch would negotiate 16x16 and only reach the
    /// real size later by `CapsChanged`. That is fine on its own, but a fan-in
    /// downstream configures its pads from the solved caps and cannot accept two
    /// inputs of different geometry, so a builder that knows the size states it.
    seed_geometry: Option<SequenceHeader>,
}

impl PsDemuxN {
    /// A demuxer with one output port per entry of `ports`, in port order.
    /// Panics if `ports` is empty (a fan-out needs a port).
    pub fn new(ports: Vec<PsStream>) -> Self {
        assert!(!ports.is_empty(), "PsDemuxN needs at least one output port");
        let announced = alloc::vec![false; ports.len()];
        let refined = alloc::vec![None; ports.len()];
        let segment_sent = alloc::vec![false; ports.len()];
        let config_sent = alloc::vec![false; ports.len()];
        Self {
            demux: PsDemuxer::new(),
            ports,
            announced,
            refined,
            bus: None,
            collection_posted: false,
            stream_select: None,
            emitted: 0,
            segment_base: None,
            segment_sent,
            config_sent,
            seed_geometry: None,
        }
    }

    /// Declare the video geometry a probe already read, so the video port
    /// negotiates at the real size instead of a `Range` placeholder's minimum.
    pub fn with_video_geometry(mut self, geometry: SequenceHeader) -> Self {
        self.seed_geometry = Some(geometry);
        self
    }

    /// Attach the pipeline bus so the file's `StreamCollection` posts once, the
    /// way [`PsDemux::with_bus`] does.
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.bus = Some(bus);
        self
    }

    /// Make the per-port stream assignment app-selectable, the program stream
    /// sibling of
    /// [`TsDemuxN::with_stream_select`](crate::tsdemux::TsDemuxN::with_stream_select).
    pub fn with_stream_select(mut self, select: StreamSelectController) -> Self {
        self.stream_select = Some(select);
        self
    }

    /// Number of output ports.
    pub fn port_count(&self) -> usize {
        self.ports.len()
    }

    /// Count of frames forwarded across all ports.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    fn apply_stream_selection(&mut self) {
        let Some(ctrl) = &self.stream_select else {
            return;
        };
        let Some(ids) = ctrl.take_pending() else {
            return;
        };
        let mut active = Vec::new();
        for (port, id) in ids.iter().enumerate().take(self.ports.len()) {
            let Some(stream) = resolve_ps_stream_id(&self.demux, id) else {
                continue;
            };
            if self.ports[port] != stream {
                self.ports[port] = stream;
                self.announced[port] = false; // re-emit caps for the new stream
                self.refined[port] = None;
            }
            active.push(id.clone());
        }
        if !active.is_empty() {
            if let Some(bus) = &self.bus {
                bus.try_post(BusMessage::StreamsSelected { ids: active });
            }
        }
    }

    fn post_stream_collection(&mut self) {
        if self.collection_posted {
            return;
        }
        let streams: Vec<Stream> = self
            .demux
            .streams()
            .iter()
            .filter_map(PsDemux::es_to_stream)
            .collect();
        if streams.is_empty() {
            return;
        }
        self.collection_posted = true;
        if let Some(bus) = &self.bus {
            bus.try_post(BusMessage::StreamCollection(StreamCollection::new(
                "mpegps-0", streams,
            )));
        }
    }

    /// Route each completed access unit to the port carrying its stream,
    /// emitting that port's opening `CapsChanged` before its first frame and the
    /// sequence header's geometry once it is known.
    async fn route_units(&mut self, out: &mut dyn MultiOutputSink) -> Result<(), G2gError> {
        for u in self.demux.take_units() {
            let es = PsElementaryStream {
                stream_id: u.stream_id,
                substream_id: u.substream_id,
            };
            let Some(kind) = es.kind() else {
                continue;
            };
            let Some(port) = self.ports.iter().position(|&s| s == kind) else {
                continue; // a stream no selected port carries
            };
            let pts_ns = ns_from_90khz(u.pts_90khz).unwrap_or(0);
            let dts_ns = ns_from_90khz(u.dts_90khz).unwrap_or(pts_ns);
            if !self.announced[port] {
                // Announce what `port_output_caps` declared: a seeded video
                // port solved at its Fixed geometry, and the runner's
                // mid-stream check rejects a Range placeholder on that link.
                let caps = match (kind, self.seed_geometry) {
                    (PsStream::Mpeg2, Some(seq)) => Caps::CompressedVideo {
                        codec: VideoCodec::Mpeg2,
                        width: Dim::Fixed(seq.width),
                        height: Dim::Fixed(seq.height),
                        framerate: Rate::Fixed(seq.framerate_q16),
                    },
                    _ => PsDemux::output_caps(kind),
                };
                out.push_to(port, PipelinePacket::CapsChanged(caps)).await?;
                self.announced[port] = true;
                // A seeded announce already names the geometry; the parsed
                // sequence header then only re-announces a real change.
                if let (PsStream::Mpeg2, Some(seq)) = (kind, self.seed_geometry) {
                    self.refined[port] = Some(seq);
                }
            }
            if !self.segment_sent[port] {
                self.segment_sent[port] = true;
                let base = *self.segment_base.get_or_insert(pts_ns);
                let seg = Segment::for_flush_seek(&Seek::flush_to(base), None);
                out.push_to(port, PipelinePacket::Segment(seg)).await?;
            }
            // A subpicture port opens on the same synthesized `.idx` the single
            // output demuxer sends, so a fan-out subtitle branch renders on the
            // video's geometry rather than the decoder's default.
            if kind == PsStream::SubPicture && !self.config_sent[port] {
                if let Some(blob) = idx_config_blob(&self.demux) {
                    self.config_sent[port] = true;
                    let frame = config_frame(blob, pts_ns, self.emitted);
                    self.emitted += 1;
                    out.push_to(port, PipelinePacket::DataFrame(frame)).await?;
                }
            }
            if kind == PsStream::Mpeg2 {
                if let Some(seq) = u.sequence {
                    if self.refined[port] != Some(seq) {
                        self.refined[port] = Some(seq);
                        out.push_to(
                            port,
                            PipelinePacket::CapsChanged(Caps::CompressedVideo {
                                codec: VideoCodec::Mpeg2,
                                width: Dim::Fixed(seq.width),
                                height: Dim::Fixed(seq.height),
                                framerate: Rate::Fixed(seq.framerate_q16),
                            }),
                        )
                        .await?;
                    }
                }
            }
            let duration_ns = cue_duration_ns(kind, &u.data);
            let keyframe = unit_is_keyframe(kind, &u.data);
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(u.data.into_boxed_slice())),
                FrameTiming {
                    pts_ns,
                    dts_ns,
                    duration_ns,
                    keyframe,
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

impl MultiOutputElement for PsDemuxN {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&PsDemux::input_caps())
    }

    /// Declare each port's elementary-stream caps, so the solver negotiates each
    /// branch against its codec at startup. `None` for an out-of-range port.
    fn port_output_caps(&self, port: usize) -> Option<Caps> {
        let stream = *self.ports.get(port)?;
        match (stream, self.seed_geometry) {
            (PsStream::Mpeg2, Some(seq)) => Some(Caps::CompressedVideo {
                codec: VideoCodec::Mpeg2,
                width: Dim::Fixed(seq.width),
                height: Dim::Fixed(seq.height),
                framerate: Rate::Fixed(seq.framerate_q16),
            }),
            _ => Some(PsDemux::output_caps(stream)),
        }
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        absolute_caps
            .intersect(&PsDemux::input_caps())
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
                    let Some(slice) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.demux.push_data(slice);
                    if self.bus.is_some() {
                        self.post_stream_collection();
                    }
                    self.apply_stream_selection();
                    self.route_units(out).await?;
                }
                PipelinePacket::Flush => {
                    self.demux = PsDemuxer::new();
                    for port in 0..self.ports.len() {
                        self.config_sent[port] = false;
                        out.push_to(port, PipelinePacket::Flush).await?;
                    }
                }
                PipelinePacket::Segment(seg) => {
                    for port in 0..self.ports.len() {
                        out.push_to(port, PipelinePacket::Segment(seg)).await?;
                    }
                }
                PipelinePacket::Eos => {
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

    fn properties(&self) -> &'static [PropertySpec] {
        &[]
    }

    fn set_property(&mut self, _name: &str, _value: PropValue) -> Result<(), PropError> {
        Err(PropError::Unknown)
    }

    fn get_property(&self, _name: &str) -> Option<PropValue> {
        None
    }
}
