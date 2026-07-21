//! MPEG-2 Transport Stream demuxer (M108): parse a TS byte stream into the
//! elementary-stream access units it carries (ISO/IEC 13818-1).
//!
//! Pure `no_std + alloc` parsing, like the `mp4box` precedent and `annexb`: this
//! module is just the state machine (sync to 188-byte packets, read the PAT to
//! find the PMT, read the PMT to find the elementary streams, reassemble PES
//! packets per PID and strip their headers). The [`crate::tsdemux::TsDemux`]
//! element wraps it; the split keeps the bit-twiddling testable without a runner.
//!
//! Scope (v1): a single program; PSI sections (PAT / PMT) are assumed to fit in
//! one TS packet (true for the small tables in practice); PES payloads reassemble
//! across packets. The carried elementary stream for H.264 / H.265 is already
//! Annex-B, so a unit feeds `h264parse` directly.

use alloc::vec::Vec;

/// MPEG-TS packet size in bytes (the standard 188; M2TS 192 with a 4-byte
/// timestamp prefix is not handled).
pub const TS_PACKET_LEN: usize = 188;

const SYNC_BYTE: u8 = 0x47;
const PID_PAT: u16 = 0x0000;

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
/// PMT `stream_type` for ADTS AAC audio.
pub const STREAM_TYPE_AAC: u8 = 0x0F;
/// PMT `stream_type` for MPEG-1 Audio (Layer I/II/III, e.g. `mp2`).
pub const STREAM_TYPE_MPEG1_AUDIO: u8 = 0x03;
/// PMT `stream_type` for MPEG-2 Audio (the low-sample-rate extension of the above).
pub const STREAM_TYPE_MPEG2_AUDIO: u8 = 0x04;
/// PMT `stream_type` for a private PES stream (0x06). Opus in MPEG-TS rides this,
/// identified by an 'Opus' registration descriptor (DVB/ETSI carriage).
pub const STREAM_TYPE_PRIVATE_PES: u8 = 0x06;

/// One elementary stream announced by the PMT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElementaryStream {
    pub pid: u16,
    pub stream_type: u8,
    /// Opus channel count for a private (0x06) stream whose ES descriptors carry
    /// the 'Opus' registration + DVB extension descriptor. `Some` marks the stream
    /// as Opus (disambiguating the generic 0x06); `None` for any other 0x06 use.
    pub opus_channels: Option<u8>,
}

/// A reassembled PES payload: one access unit of an elementary stream.
#[derive(Debug, Clone, PartialEq)]
pub struct EsUnit {
    pub pid: u16,
    pub stream_type: u8,
    /// Presentation timestamp in 90 kHz units, if the PES carried one.
    pub pts_90khz: Option<u64>,
    /// The elementary stream bytes (for H.264/H.265, Annex-B).
    pub data: Vec<u8>,
}

/// A PES packet being reassembled across TS packets for one PID.
#[derive(Debug)]
struct PendingPes {
    pid: u16,
    stream_type: u8,
    pts_90khz: Option<u64>,
    data: Vec<u8>,
}

/// Incremental MPEG-TS demuxer: feed 188-byte packets, drain [`EsUnit`]s.
#[derive(Debug, Default)]
pub struct TsDemuxer {
    pmt_pid: Option<u16>,
    streams: Vec<ElementaryStream>,
    pending: Vec<PendingPes>,
    completed: Vec<EsUnit>,
}

impl TsDemuxer {
    pub fn new() -> Self {
        Self::default()
    }

    /// The elementary streams announced by the PMT (empty until a PMT is seen).
    pub fn streams(&self) -> &[ElementaryStream] {
        &self.streams
    }

    /// Opus channel count for `pid`, if its PMT entry is a private (0x06) stream
    /// carrying the 'Opus' registration descriptor; `None` for any other stream.
    pub fn opus_channels(&self, pid: u16) -> Option<u8> {
        self.streams
            .iter()
            .find(|s| s.pid == pid)
            .and_then(|s| s.opus_channels)
    }

    /// The PID of the first video elementary stream (H.264 or H.265), if any.
    pub fn video_pid(&self) -> Option<u16> {
        self.streams
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
        } else if Some(pid) == self.pmt_pid {
            self.parse_pmt(payload, pusi);
        } else if let Some(stream_type) = self.stream_type_of(pid) {
            self.accumulate_pes(pid, stream_type, payload, pusi);
        }
    }

    /// Drain the access units completed so far.
    pub fn take_units(&mut self) -> Vec<EsUnit> {
        core::mem::take(&mut self.completed)
    }

    /// Finalize any PES still being reassembled (call at end of stream). The
    /// units land in the queue drained by [`take_units`](Self::take_units).
    pub fn flush(&mut self) {
        for p in core::mem::take(&mut self.pending) {
            Self::finish(&mut self.completed, p);
        }
    }

    fn stream_type_of(&self, pid: u16) -> Option<u8> {
        self.streams
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

    fn parse_pat(&mut self, payload: &[u8], pusi: bool) {
        if self.pmt_pid.is_some() {
            return; // first PAT wins (single program)
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
        // Program loop starts at section[8] (after the 8-byte PSI header).
        let mut i = 8;
        while i + 4 <= body.len() {
            let program_number = ((body[i] as u16) << 8) | body[i + 1] as u16;
            let pid = (((body[i + 2] & 0x1F) as u16) << 8) | body[i + 3] as u16;
            if program_number != 0 {
                self.pmt_pid = Some(pid);
                return;
            }
            i += 4;
        }
    }

    fn parse_pmt(&mut self, payload: &[u8], pusi: bool) {
        if !self.streams.is_empty() {
            return; // first PMT wins
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
            let opus_channels = body
                .get(i + 5..i + 5 + es_info_length)
                .filter(|_| stream_type == STREAM_TYPE_PRIVATE_PES)
                .and_then(parse_opus_descriptors);
            self.streams.push(ElementaryStream {
                pid,
                stream_type,
                opus_channels,
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
            let (pts, es) = parse_pes_header(payload);
            self.pending.push(PendingPes {
                pid,
                stream_type,
                pts_90khz: pts,
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
            data: p.data,
        });
    }
}

/// Parse a PES packet header at the start of `payload`, returning the PTS (if
/// present) and the elementary-stream bytes after the header. If the start code
/// or optional header is malformed, returns the whole payload with no PTS (so a
/// best-effort stream still flows).
fn parse_pes_header(payload: &[u8]) -> (Option<u64>, &[u8]) {
    // PES: 00 00 01, stream_id, PES_packet_length(2), then for media stream_ids
    // an optional header: flags(2) + PES_header_data_length(1) + that many bytes.
    if payload.len() < 9 || payload[0] != 0x00 || payload[1] != 0x00 || payload[2] != 0x01 {
        return (None, payload);
    }
    // byte 6 must have the '10' marker bits for an optional PES header.
    if payload[6] & 0xC0 != 0x80 {
        return (None, &payload[6..]);
    }
    let pts_dts_flags = (payload[7] >> 6) & 0x03;
    let header_data_len = payload[8] as usize;
    let es_start = 9 + header_data_len;
    if es_start > payload.len() {
        return (None, payload);
    }
    let pts = if pts_dts_flags & 0x02 != 0 && payload.len() >= 14 {
        Some(decode_timestamp(&payload[9..14]))
    } else {
        None
    };
    (pts, &payload[es_start..])
}

/// Decode a 33-bit MPEG PTS/DTS from its 5-byte field (90 kHz units).
fn decode_timestamp(b: &[u8]) -> u64 {
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
            0x05 if body.len() >= 4 && &body[..4] == b"Opus" => is_opus = true,
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

/// Fixed PID layout for the single-program mux: the PMT and the one elementary
/// stream. (The demuxer discovers these from the tables, so any values pair.)
const MUX_PMT_PID: u16 = 0x1000;
const MUX_ES_PID: u16 = 0x0100;

/// One elementary stream in a [`TsMuxer`]: its PMT stream type, the TS PID it is
/// carried on, its PES `stream_id`, and its running continuity counter.
#[derive(Debug)]
struct MuxStream {
    stream_type: u8,
    pid: u16,
    stream_id: u8,
    es_cc: u8,
}

/// MPEG-TS multiplexer (M114, multi-stream since M207): wraps access units in PES
/// packets and 188-byte TS packets, emitting PAT + PMT once up front. The inverse
/// of [`TsDemuxer`]; the [`crate::tsmux::TsMux`] element wraps it. One program
/// carrying one or more elementary streams (e.g. H.264 video + AAC audio), each
/// on its own PID, named together in a single PMT.
///
/// Scope: one program, no PCR (a PCR in the adaptation field is a follow-up;
/// lenient decoders and the demuxer here do not need it; the caller is expected
/// to interleave access units in timestamp order, which [`crate::tsmux::TsMux`]
/// does). The PSI carries a real MPEG-2 CRC-32, so the output is a valid TS.
#[derive(Debug)]
pub struct TsMuxer {
    streams: Vec<MuxStream>,
    pat_cc: u8,
    pmt_cc: u8,
    tables_written: bool,
    /// PAT/PMT re-emission cadence in 90 kHz ticks (`0` = emit once up front, the
    /// default). When set, the table pair is re-emitted before the first access
    /// unit whose PTS is at least this far past the last emission, so a decoder
    /// that joins mid-stream (a tuned-in multicast, an HLS/DASH segment boundary)
    /// finds the PSI without waiting for the start of the stream.
    table_interval_90khz: u64,
    /// PTS (90 kHz) the tables were last emitted at, for the cadence above.
    last_tables_pts: Option<u64>,
}

impl TsMuxer {
    /// A single-stream muxer for `stream_type` (e.g. [`STREAM_TYPE_H264`]).
    pub fn new(stream_type: u8) -> Self {
        Self::with_streams(&[stream_type])
    }

    /// A multi-stream muxer: one elementary stream per entry of `stream_types`,
    /// in input order. Stream `i` is carried on PID `MUX_ES_PID + i`; the PES
    /// `stream_id` is assigned per media kind (video `0xE0..`, audio `0xC0..`),
    /// distinct within each kind so several video or audio streams stay
    /// addressable. [`push_au_on`](Self::push_au_on) selects the stream by index.
    pub fn with_streams(stream_types: &[u8]) -> Self {
        let mut video_n = 0u8;
        let mut audio_n = 0u8;
        let streams = stream_types
            .iter()
            .enumerate()
            .map(|(i, &stream_type)| {
                let stream_id = if stream_type == STREAM_TYPE_AAC {
                    let id = 0xC0 + audio_n;
                    audio_n += 1;
                    id
                } else {
                    let id = 0xE0 + video_n;
                    video_n += 1;
                    id
                };
                MuxStream {
                    stream_type,
                    pid: MUX_ES_PID + i as u16,
                    stream_id,
                    es_cc: 0,
                }
            })
            .collect();
        Self {
            streams,
            pat_cc: 0,
            pmt_cc: 0,
            tables_written: false,
            table_interval_90khz: 0,
            last_tables_pts: None,
        }
    }

    /// Set the PAT/PMT re-emission cadence in 90 kHz ticks (`0` = once up front).
    /// See [`table_interval_90khz`](Self::table_interval_90khz).
    pub fn set_table_interval_90khz(&mut self, ticks: u64) {
        self.table_interval_90khz = ticks;
    }

    /// Mux one access unit of stream 0 (the single-stream convenience). See
    /// [`push_au_on`](Self::push_au_on).
    pub fn push_au(&mut self, au: &[u8], pts_90khz: Option<u64>) -> Vec<u8> {
        self.push_au_on(0, au, pts_90khz)
    }

    /// Mux one access unit of elementary stream `stream_index` into TS bytes,
    /// preceded by PAT + PMT on the very first call (any stream). `pts_90khz`,
    /// when present, is written into the PES header.
    pub fn push_au_on(
        &mut self,
        stream_index: usize,
        au: &[u8],
        pts_90khz: Option<u64>,
    ) -> Vec<u8> {
        let mut out = Vec::new();
        // Emit the PAT/PMT pair up front, then again on the configured cadence so a
        // mid-stream joiner finds the tables. A `None` PTS can't be time-gated, so
        // it only ever triggers the initial emission.
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
            self.pmt_packet(&mut out);
            self.tables_written = true;
            if let Some(now) = pts_90khz {
                self.last_tables_pts = Some(now);
            }
        }
        let s = &mut self.streams[stream_index];
        let pes = build_pes(s.stream_id, au, pts_90khz);
        let mut off = 0;
        let mut pusi = true;
        while off < pes.len() {
            let take = (pes.len() - off).min(TS_PACKET_LEN - 4);
            ts_packet(s.pid, pusi, s.es_cc, &pes[off..off + take], &mut out);
            s.es_cc = (s.es_cc + 1) & 0x0F;
            pusi = false;
            off += take;
        }
        out
    }

    fn pat_packet(&mut self, out: &mut Vec<u8>) {
        let body = [
            0x00,
            0x01, // transport_stream_id
            0xC1,
            0x00,
            0x00, // version/current, section_number, last_section_number
            0x00,
            0x01, // program_number 1
            0xE0 | (MUX_PMT_PID >> 8) as u8 & 0x1F,
            MUX_PMT_PID as u8,
        ];
        self.pat_cc = psi_packet(PID_PAT, 0x00, &body, self.pat_cc, out);
    }

    fn pmt_packet(&mut self, out: &mut Vec<u8>) {
        // PCR_PID = the first stream's PID (no separate PCR stream).
        let pcr_pid = self.streams[0].pid;
        let mut body = Vec::with_capacity(9 + self.streams.len() * 5);
        body.extend_from_slice(&[
            0x00,
            0x01, // program_number
            0xC1,
            0x00,
            0x00, // version, section/last
            0xE0 | (pcr_pid >> 8) as u8 & 0x1F,
            pcr_pid as u8, // PCR_PID
            0xF0,
            0x00, // program_info_length = 0
        ]);
        // One ES loop entry per stream: stream_type, elementary_PID, ES_info_len.
        for s in &self.streams {
            body.extend_from_slice(&[
                s.stream_type,
                0xE0 | (s.pid >> 8) as u8 & 0x1F,
                s.pid as u8,
                0xF0,
                0x00, // ES_info_length = 0
            ]);
        }
        self.pmt_cc = psi_packet(MUX_PMT_PID, 0x02, &body, self.pmt_cc, out);
    }
}

/// Build a PES packet for one access unit (start code + stream_id + length + an
/// optional header carrying the PTS), matching what [`parse_pes_header`] reads.
fn build_pes(stream_id: u8, au: &[u8], pts_90khz: Option<u64>) -> Vec<u8> {
    let mut header = Vec::new();
    header.push(0x80); // marker '10'
    header.push(if pts_90khz.is_some() { 0x80 } else { 0x00 }); // PTS_DTS_flags
    if let Some(pts) = pts_90khz {
        header.push(5); // PES_header_data_length
        encode_timestamp(0x2, pts, &mut header); // '0010' prefix for PTS-only
    } else {
        header.push(0);
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

/// Append a 5-byte PTS/DTS field (`prefix` is `0010` for PTS-only) in 90 kHz
/// units, the inverse of [`decode_timestamp`].
fn encode_timestamp(prefix: u8, ts: u64, out: &mut Vec<u8>) {
    out.push((prefix << 4) | (((ts >> 30) & 0x07) as u8) << 1 | 0x01);
    out.push(((ts >> 22) & 0xFF) as u8);
    out.push((((ts >> 15) & 0x7F) as u8) << 1 | 0x01);
    out.push(((ts >> 7) & 0xFF) as u8);
    out.push(((ts & 0x7F) as u8) << 1 | 0x01);
}

/// Write one 188-byte TS packet to `out`: a payload of up to 184 bytes, padded
/// with an adaptation-field stuffing run when shorter (the last packet of a PES).
fn ts_packet(pid: u16, pusi: bool, cc: u8, payload: &[u8], out: &mut Vec<u8>) {
    const PAYLOAD_MAX: usize = TS_PACKET_LEN - 4;
    debug_assert!(payload.len() <= PAYLOAD_MAX);
    out.push(SYNC_BYTE);
    out.push((if pusi { 0x40 } else { 0 }) | ((pid >> 8) as u8 & 0x1F));
    out.push(pid as u8);
    let l = payload.len();
    if l == PAYLOAD_MAX {
        out.push(0x10 | (cc & 0x0F)); // payload only
        out.extend_from_slice(payload);
    } else {
        out.push(0x30 | (cc & 0x0F)); // adaptation field + payload
        let af_len = PAYLOAD_MAX - 1 - l; // bytes after the AF length byte
        out.push(af_len as u8);
        if af_len >= 1 {
            out.push(0x00); // AF flags (no PCR / no options)
            out.resize(out.len() + (af_len - 1), 0xFF); // stuffing
        }
        out.extend_from_slice(payload);
    }
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
        ts_packet(pid, pusi, cc, &rest[..n], out);
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
                opus_channels: None
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
    fn mux_demux_round_trip() {
        // Mux two H.264 access units with PTS, then demux the TS back to them.
        let au0 = [0u8, 0, 0, 1, 0x65, 0xAA, 0xBB];
        let au1 = [0u8, 0, 0, 1, 0x41, 0xCC];
        let mut mux = TsMuxer::new(STREAM_TYPE_H264);
        let mut bytes = mux.push_au(&au0, Some(900_000));
        bytes.extend(mux.push_au(&au1, Some(903_000)));
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

        let out0 = m.push_au(&au, Some(0));
        assert_eq!(pat_count(&out0), 1, "PAT emitted up front");
        let out1 = m.push_au(&au, Some(90 * 50)); // +50 ms: under the interval
        assert_eq!(pat_count(&out1), 0, "no PAT before the interval elapses");
        let out2 = m.push_au(&au, Some(90 * 150)); // 150 ms since last emit: due
        assert_eq!(pat_count(&out2), 1, "PAT re-emitted after the interval");
    }

    #[test]
    fn table_interval_zero_emits_tables_once() {
        let mut m = TsMuxer::new(STREAM_TYPE_H264);
        let au = alloc::vec![0u8, 0, 0, 1, 0x65, 0x88];
        let _ = m.push_au(&au, Some(0));
        let later = m.push_au(&au, Some(90 * 10_000)); // 10 s later
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
}
