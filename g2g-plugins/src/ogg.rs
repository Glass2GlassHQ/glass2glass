//! Ogg demuxer (M116): parse an Ogg byte stream into the packets of its logical
//! bitstream (RFC 3533), the Opus / Vorbis carrier.
//!
//! Pure `no_std + alloc` parsing, the [`crate::mpegts`] / [`crate::matroska`]
//! precedent for Ogg: sync to "OggS" pages, read the segment-table lacing to
//! frame packets (a packet runs across 255-valued segments and ends on a value
//! 0..254), reassemble packets that span pages, and skip the codec setup headers.
//! The [`crate::oggdemux::OggDemux`] element wraps it.
//!
//! Scope (v1): one logical bitstream (the first serial); Opus fully (codec +
//! channel count from `OpusHead`, the two setup headers skipped), other codecs
//! best-effort (tagged, all packets emitted). Granule-position timing and
//! multi-stream Ogg are follow-ups (packets carry no PTS yet).

use alloc::vec::Vec;

const CAPTURE_PATTERN: [u8; 4] = *b"OggS";
const HEADER_LEN: usize = 27; // fixed header before the segment table

/// The codec of an Ogg logical bitstream, sniffed from its first packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OggCodec {
    Opus,
    Vorbis,
    /// A first packet this demuxer does not recognize.
    Other,
}

/// Stream parameters recovered from the first (identification) packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OggStreamInfo {
    pub codec: OggCodec,
    pub channels: u8,
    pub sample_rate: u32,
}

/// Incremental Ogg demuxer: feed bytes, drain elementary-stream packets.
#[derive(Debug, Default)]
pub struct OggDemuxer {
    buf: Vec<u8>,
    serial: Option<u32>,
    /// Bytes of a packet still being reassembled across pages.
    partial: Vec<u8>,
    info: Option<OggStreamInfo>,
    /// Count of packets finalized so far, to skip the codec setup headers.
    packets_seen: u32,
    completed: Vec<Vec<u8>>,
}

impl OggDemuxer {
    pub fn new() -> Self {
        Self::default()
    }

    /// The logical stream's parameters (set once the first packet is parsed).
    pub fn info(&self) -> Option<OggStreamInfo> {
        self.info
    }

    /// Drain the elementary-stream packets demuxed so far.
    pub fn take_packets(&mut self) -> Vec<Vec<u8>> {
        core::mem::take(&mut self.completed)
    }

    /// Feed Ogg bytes. Complete pages are parsed as they arrive; a partial
    /// trailing page waits for the next call.
    pub fn push_data(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
        self.drain_pages();
    }

    fn drain_pages(&mut self) {
        loop {
            if !self.resync() {
                return;
            }
            if self.buf.len() < HEADER_LEN {
                return;
            }
            let num_segments = self.buf[26] as usize;
            let table_end = HEADER_LEN + num_segments;
            if self.buf.len() < table_end {
                return;
            }
            let body_len: usize =
                self.buf[HEADER_LEN..table_end].iter().map(|&s| s as usize).sum();
            let total = table_end + body_len;
            if self.buf.len() < total {
                return;
            }
            // Own the page so the borrow ends before draining / mutating self.
            let page: Vec<u8> = self.buf[..total].to_vec();
            self.buf.drain(..total);
            self.parse_page(&page, table_end);
        }
    }

    /// Drop bytes before the next "OggS" capture pattern. Returns false (and
    /// keeps a short tail that might be a split pattern) when none is buffered.
    fn resync(&mut self) -> bool {
        if self.buf.starts_with(&CAPTURE_PATTERN) {
            return true;
        }
        match self.buf.windows(4).position(|w| w == CAPTURE_PATTERN) {
            Some(pos) => {
                self.buf.drain(..pos);
                true
            }
            None => {
                let keep = self.buf.len().saturating_sub(3);
                self.buf.drain(..keep);
                false
            }
        }
    }

    fn parse_page(&mut self, page: &[u8], table_end: usize) {
        let header_type = page[5];
        let serial = u32::from_le_bytes([page[14], page[15], page[16], page[17]]);
        match self.serial {
            Some(s) if s != serial => return, // a different logical stream (v1: first only)
            None => self.serial = Some(serial),
            _ => {}
        }
        // A page not flagged "continued" abandons any half-built packet (a lost
        // page upstream); otherwise the first packet continues `partial`.
        let mut acc = if header_type & 0x01 != 0 {
            core::mem::take(&mut self.partial)
        } else {
            self.partial.clear();
            Vec::new()
        };
        let mut pos = table_end;
        for &seg in &page[HEADER_LEN..table_end] {
            let seg = seg as usize;
            acc.extend_from_slice(&page[pos..pos + seg]);
            pos += seg;
            if seg < 255 {
                self.finalize(core::mem::take(&mut acc));
            }
        }
        // A trailing 255-segment leaves an incomplete packet for the next page.
        self.partial = acc;
    }

    fn finalize(&mut self, packet: Vec<u8>) {
        if packet.is_empty() {
            return;
        }
        if self.info.is_none() {
            self.info = Some(detect(&packet));
        }
        let header_count = match self.info.map(|i| i.codec) {
            Some(OggCodec::Opus) => 2,   // OpusHead + OpusTags
            Some(OggCodec::Vorbis) => 3, // id + comment + setup
            _ => 0,
        };
        if self.packets_seen < header_count {
            self.packets_seen += 1;
            return;
        }
        self.packets_seen += 1;
        self.completed.push(packet);
    }
}

/// Identify the logical stream from its first packet's magic.
fn detect(packet: &[u8]) -> OggStreamInfo {
    if packet.starts_with(b"OpusHead") && packet.len() >= 10 {
        // OpusHead: magic(8), version(1), channel_count(1) at offset 9. Opus
        // always decodes at 48 kHz regardless of the original input rate.
        OggStreamInfo { codec: OggCodec::Opus, channels: packet[9], sample_rate: 48_000 }
    } else if packet.starts_with(b"\x01vorbis") {
        OggStreamInfo { codec: OggCodec::Vorbis, channels: 0, sample_rate: 0 }
    } else {
        OggStreamInfo { codec: OggCodec::Other, channels: 0, sample_rate: 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Build one Ogg page: header_type, serial, and a list of packets (each laced
    /// into 255-byte segments; a packet that is a multiple of 255 gets a trailing
    /// 0 segment so it terminates on this page).
    fn page(header_type: u8, serial: u32, seq: u32, packets: &[&[u8]]) -> Vec<u8> {
        let mut table = Vec::new();
        let mut body = Vec::new();
        for p in packets {
            let mut n = p.len();
            loop {
                let seg = n.min(255);
                table.push(seg as u8);
                n -= seg;
                if seg < 255 {
                    break;
                }
            }
            body.extend_from_slice(p);
        }
        let mut out = Vec::new();
        out.extend_from_slice(&CAPTURE_PATTERN);
        out.push(0); // version
        out.push(header_type);
        out.extend_from_slice(&0u64.to_le_bytes()); // granule
        out.extend_from_slice(&serial.to_le_bytes());
        out.extend_from_slice(&seq.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // CRC (ignored on read)
        out.push(table.len() as u8);
        out.extend_from_slice(&table);
        out.extend_from_slice(&body);
        out
    }

    fn opus_head(channels: u8) -> Vec<u8> {
        let mut h = b"OpusHead".to_vec();
        h.push(1); // version
        h.push(channels);
        h.extend_from_slice(&[0, 0]); // pre-skip
        h.extend_from_slice(&48_000u32.to_le_bytes()); // input sample rate
        h.extend_from_slice(&[0, 0, 0]); // output gain + mapping family
        h
    }

    #[test]
    fn demuxes_opus_packets_skipping_headers() {
        let serial = 0xDEAD_BEEF;
        let mut d = OggDemuxer::new();
        d.push_data(&page(0x02, serial, 0, &[&opus_head(2)])); // BOS: OpusHead
        d.push_data(&page(0x00, serial, 1, &[b"OpusTags...."])); // setup header
        d.push_data(&page(0x00, serial, 2, &[&[0xAA, 0xBB], &[0xCC, 0xDD, 0xEE]]));

        assert_eq!(d.info(), Some(OggStreamInfo { codec: OggCodec::Opus, channels: 2, sample_rate: 48_000 }));
        let packets = d.take_packets();
        assert_eq!(packets, vec![vec![0xAA, 0xBB], vec![0xCC, 0xDD, 0xEE]], "audio packets only");
    }

    #[test]
    fn reassembles_packet_across_pages() {
        let serial = 1;
        // A 300-byte audio packet (> 255) spans two pages: page 1 ends on a
        // 255-segment (continued), page 2 carries the rest with the continued flag.
        let big: Vec<u8> = (0..300u32).map(|x| x as u8).collect();
        let mut d = OggDemuxer::new();
        d.push_data(&page(0x02, serial, 0, &[&opus_head(1)]));
        d.push_data(&page(0x00, serial, 1, &[b"OpusTags"]));

        // Hand-build the split page pair for the big packet: page 1 ends on a
        // lone 255-segment (no terminator, so the packet continues).
        let mut page1 = Vec::new();
        page1.extend_from_slice(&CAPTURE_PATTERN);
        page1.extend_from_slice(&[0, 0x00]); // version, header_type
        page1.extend_from_slice(&0u64.to_le_bytes());
        page1.extend_from_slice(&serial.to_le_bytes());
        page1.extend_from_slice(&2u32.to_le_bytes());
        page1.extend_from_slice(&0u32.to_le_bytes());
        page1.push(1); // one segment
        page1.push(255); // 255 bytes, packet continues
        page1.extend_from_slice(&big[..255]);

        let mut page2 = Vec::new();
        page2.extend_from_slice(&CAPTURE_PATTERN);
        page2.extend_from_slice(&[0, 0x01]); // continued flag
        page2.extend_from_slice(&0u64.to_le_bytes());
        page2.extend_from_slice(&serial.to_le_bytes());
        page2.extend_from_slice(&3u32.to_le_bytes());
        page2.extend_from_slice(&0u32.to_le_bytes());
        page2.push(1);
        page2.push((300 - 255) as u8); // 45 bytes, terminates
        page2.extend_from_slice(&big[255..]);

        d.push_data(&page1);
        d.push_data(&page2);
        assert_eq!(d.take_packets(), vec![big], "packet reassembled across the page boundary");
    }

    #[test]
    fn ignores_a_second_logical_stream() {
        let mut d = OggDemuxer::new();
        d.push_data(&page(0x02, 1, 0, &[&opus_head(2)]));
        d.push_data(&page(0x00, 1, 1, &[b"OpusTags"]));
        d.push_data(&page(0x00, 2, 0, &[b"other-stream-packet"])); // different serial
        d.push_data(&page(0x00, 1, 2, &[&[0x01, 0x02]]));
        assert_eq!(d.take_packets(), vec![vec![0x01, 0x02]], "only the first serial");
    }
}
