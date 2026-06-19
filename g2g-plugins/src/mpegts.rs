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

/// PMT `stream_type` for H.264 (AVC) video.
pub const STREAM_TYPE_H264: u8 = 0x1B;
/// PMT `stream_type` for H.265 (HEVC) video.
pub const STREAM_TYPE_H265: u8 = 0x24;
/// PMT `stream_type` for ADTS AAC audio.
pub const STREAM_TYPE_AAC: u8 = 0x0F;

/// One elementary stream announced by the PMT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElementaryStream {
    pub pid: u16,
    pub stream_type: u8,
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
        self.streams.iter().find(|s| s.pid == pid).map(|s| s.stream_type)
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
        let Some(section) = Self::section(payload, pusi) else { return };
        if section.first() != Some(&0x00) {
            return; // table_id 0x00 = PAT
        }
        let Some(body) = Self::section_body(section) else { return };
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
        let Some(section) = Self::section(payload, pusi) else { return };
        if section.first() != Some(&0x02) {
            return; // table_id 0x02 = PMT
        }
        let Some(body) = Self::section_body(section) else { return };
        if body.len() < 12 {
            return;
        }
        let program_info_length = (((body[10] & 0x0F) as usize) << 8) | body[11] as usize;
        let mut i = 12 + program_info_length;
        while i + 5 <= body.len() {
            let stream_type = body[i];
            let pid = (((body[i + 1] & 0x1F) as u16) << 8) | body[i + 2] as u16;
            let es_info_length = (((body[i + 3] & 0x0F) as usize) << 8) | body[i + 4] as usize;
            self.streams.push(ElementaryStream { pid, stream_type });
            i += 5 + es_info_length;
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
        } else if let Some(p) = self.pending.iter_mut().find(|p| p.pid == pid) {
            // Continuation of the current PES.
            p.data.extend_from_slice(payload);
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
            (program >> 8) as u8, program as u8, // transport_stream_id (reuse)
            0xC1, 0x00, 0x00, // version/current, section_number, last_section_number
            (program >> 8) as u8, program as u8,
            0xE0 | ((pmt_pid >> 8) as u8 & 0x1F), pmt_pid as u8,
        ]
    }

    /// PMT body (from section[3]) announcing one elementary stream.
    fn pmt_body(es_pid: u16, stream_type: u8) -> Vec<u8> {
        alloc::vec![
            0x00, 0x01, // program_number
            0xC1, 0x00, 0x00, // version, section/last
            0xE0 | ((es_pid >> 8) as u8 & 0x1F), es_pid as u8, // PCR_PID
            0xF0, 0x00, // program_info_length = 0
            stream_type,
            0xE0 | ((es_pid >> 8) as u8 & 0x1F), es_pid as u8, // elementary_PID
            0xF0, 0x00, // ES_info_length = 0
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
        d.push_packet(&psi_packet(pmt_pid, 0x02, &pmt_body(es_pid, STREAM_TYPE_H264)));
        assert_eq!(d.streams(), &[ElementaryStream { pid: es_pid, stream_type: STREAM_TYPE_H264 }]);
        assert_eq!(d.video_pid(), Some(es_pid));

        // One PES (Annex-B-ish payload) with a PTS, then a second PES start to
        // flush the first.
        let au = [0x00, 0x00, 0x00, 0x01, 0x65, 0xAA, 0xBB];
        d.push_packet(&ts_packet(es_pid, true, &pes(Some(900_000), &au)));
        assert!(d.take_units().is_empty(), "first PES not flushed until next PES start");
        d.push_packet(&ts_packet(es_pid, true, &pes(Some(901_000), &[0x00, 0x00, 0x01, 0x41])));
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
        d.push_packet(&psi_packet(pmt_pid, 0x02, &pmt_body(es_pid, STREAM_TYPE_H264)));

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
    fn ignores_non_sync_and_other_pids() {
        let mut d = TsDemuxer::new();
        d.push_packet(&[0u8; TS_PACKET_LEN]); // bad sync
        d.push_packet(&ts_packet(0x0123, true, &[1, 2, 3])); // unknown PID, no PMT
        assert!(d.take_units().is_empty());
        assert!(d.streams().is_empty());
    }
}
