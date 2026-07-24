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
                              // Cap cross-page packet reassembly. No real codec packet approaches this; the
                              // bound just stops a never-terminating run of continued pages from growing the
                              // partial packet without limit.
const MAX_PACKET_BYTES: usize = 8 * 1024 * 1024;

/// The codec of an Ogg logical bitstream, sniffed from its first packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OggCodec {
    Opus,
    Vorbis,
    /// The Ogg-FLAC mapping (`\x7fFLAC` first packet embedding the native
    /// `fLaC` + STREAMINFO header).
    Flac,
    /// A first packet this demuxer does not recognize.
    Other,
}

/// Stream parameters recovered from the first (identification) packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OggStreamInfo {
    pub codec: OggCodec,
    pub channels: u8,
    pub sample_rate: u32,
    /// Opus encoder lookahead: the count of leading 48 kHz output samples the
    /// decoder must discard (RFC 7845 `OpusHead` offset 10, LE u16). `0` for
    /// non-Opus streams.
    pub pre_skip: u16,
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
    /// The comment header (packet index 1: `OpusTags` / Vorbis comment), kept so
    /// the element can surface its VorbisComment tags. `None` until parsed.
    comment_header: Option<Vec<u8>>,
    /// The identification header (packet index 0: `OpusHead`), kept so the
    /// element can forward it in-band to the decoder (which reads its pre-skip).
    /// `None` until parsed.
    head_header: Option<Vec<u8>>,
    /// The Vorbis setup header (packet index 2: `\x05vorbis`, the codebooks),
    /// kept so the element can forward it in-band to the decoder. `None` until
    /// parsed / for other codecs.
    setup_header: Option<Vec<u8>>,
    /// Granule position of the stream's final page (the one flagged end-of-stream,
    /// header bit `0x04`). For Opus this is the total 48 kHz sample count including
    /// pre-skip; samples decoded beyond it are encoder padding. `None` until the
    /// EOS page is parsed, or if it carried the -1 "no packet completed" sentinel.
    end_granulepos: Option<u64>,
    /// The first audio-bearing page's granule position, the count of audio
    /// packets completed through it, and whether that page is also the EOS
    /// page (M778). Anchors the Vorbis timeline: the granule names the
    /// position after the page's last packet, so any excess of the natural
    /// packet durations over it is initial priming.
    first_data: Option<(u64, u32, bool)>,
    /// Running count of audio (non-header) packets finalized.
    audio_finalized: u32,
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

    /// The codec comment header (`OpusTags` for Opus), once parsed. Carries the
    /// stream's VorbisComment metadata.
    pub fn comment_header(&self) -> Option<&[u8]> {
        self.comment_header.as_deref()
    }

    /// The identification header (`OpusHead`), once parsed. The decoder reads its
    /// pre-skip from it.
    pub fn head_header(&self) -> Option<&[u8]> {
        self.head_header.as_deref()
    }

    /// The Vorbis setup header (`\x05vorbis`), once parsed. The decoder builds
    /// its codebooks from it.
    pub fn setup_header(&self) -> Option<&[u8]> {
        self.setup_header.as_deref()
    }

    /// The final page's granule position (total 48 kHz samples incl. pre-skip),
    /// once the end-of-stream page is parsed. Drives the end-of-stream padding
    /// trim: decoded samples beyond it are encoder padding.
    pub fn end_granule(&self) -> Option<u64> {
        self.end_granulepos
    }

    /// The first audio-bearing page's `(granulepos, audio packets completed
    /// through it, is-EOS-page)`, once parsed (M778). See `first_data`.
    pub fn first_data_granule(&self) -> Option<(u64, u32, bool)> {
        self.first_data
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
            let body_len: usize = self.buf[HEADER_LEN..table_end]
                .iter()
                .map(|&s| s as usize)
                .sum();
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
        // Page granule position (offset 6, LE u64; -1 = no packet completed).
        // On the EOS page (bit 0x04) it is the stream's total sample count.
        // Attacker-controlled, so only stored, bounded when used against the
        // running decoded count.
        let gp = u64::from_le_bytes([
            page[6], page[7], page[8], page[9], page[10], page[11], page[12], page[13],
        ]);
        if header_type & 0x04 != 0 && gp != u64::MAX {
            self.end_granulepos = Some(gp);
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
        // Drop and resync if it grew past the cap (malformed or abusive stream).
        if acc.len() > MAX_PACKET_BYTES {
            acc.clear();
        }
        self.partial = acc;
        // The first page that completed audio packets anchors the timeline.
        if self.first_data.is_none() && self.audio_finalized > 0 && gp != u64::MAX {
            self.first_data = Some((gp, self.audio_finalized, header_type & 0x04 != 0));
        }
    }

    fn finalize(&mut self, packet: Vec<u8>) {
        if packet.is_empty() {
            return;
        }
        if self.info.is_none() {
            self.info = Some(detect(&packet));
        }
        // Ogg-FLAC: the first packet declares how many header packets follow,
        // but that count is attacker-controlled, so classify instead: metadata
        // blocks lead with a block-type byte (never 0xFF, an invalid type),
        // audio frames with the 0xFF sync. VorbisComment is block type 4.
        if self.info.map(|i| i.codec) == Some(OggCodec::Flac) {
            if self.packets_seen == 0 {
                self.head_header = Some(packet);
            } else if packet[0] != 0xFF {
                if packet[0] & 0x7F == 4 {
                    self.comment_header = Some(packet);
                }
            } else {
                self.packets_seen += 1;
                self.audio_finalized += 1;
                self.completed.push(packet);
                return;
            }
            self.packets_seen += 1;
            return;
        }
        let header_count = match self.info.map(|i| i.codec) {
            Some(OggCodec::Opus) => 2,   // OpusHead + OpusTags
            Some(OggCodec::Vorbis) => 3, // id + comment + setup
            _ => 0,
        };
        if self.packets_seen < header_count {
            // Packet index 0 is the identification header (OpusHead), index 1 the
            // comment header (OpusTags / Vorbis comment), index 2 the Vorbis
            // setup header (codebooks).
            if self.packets_seen == 0 {
                self.head_header = Some(packet);
            } else if self.packets_seen == 1 {
                self.comment_header = Some(packet);
            } else if self.packets_seen == 2 {
                self.setup_header = Some(packet);
            }
            self.packets_seen += 1;
            return;
        }
        self.packets_seen += 1;
        self.audio_finalized += 1;
        self.completed.push(packet);
    }
}

/// Vorbis per-packet timing tables (M778), recovered from the identification
/// and setup headers without a codebook parse: the two block sizes (ident
/// byte 28) and each mode's blockflag, located by a validated backward scan
/// of the setup header's mode section (the ffmpeg `vorbis_parser` technique).
/// Drives demux-side packet durations; see [`Self::packet_samples`].
#[derive(Debug, Clone)]
pub struct VorbisTiming {
    bs0: u32,
    bs1: u32,
    /// Per-mode blockflag: `false` = short (`bs0`), `true` = long (`bs1`).
    mode_blockflag: Vec<bool>,
}

impl VorbisTiming {
    /// Recover the tables from the `\x01vorbis` ident and `\x05vorbis` setup
    /// headers. `None` on any layout mismatch (timing then stays unknown).
    pub fn parse(ident: &[u8], setup: &[u8]) -> Option<Self> {
        if !ident.starts_with(b"\x01vorbis") || ident.len() < 30 {
            return None;
        }
        // Byte 28: blocksize_0 exponent in the low nibble, blocksize_1 high.
        let bs0 = 1u32.checked_shl(u32::from(ident[28] & 0x0F))?;
        let bs1 = 1u32.checked_shl(u32::from(ident[28] >> 4))?;
        if !(64..=8192).contains(&bs0) || !(64..=8192).contains(&bs1) || bs0 > bs1 {
            return None;
        }
        Some(Self {
            bs0,
            bs1,
            mode_blockflag: mode_blockflags(setup)?,
        })
    }

    /// The block size of an audio packet (from the mode number in its first
    /// byte), or `None` for a header packet. Packet `n`'s true PCM output is
    /// the lapped `(blocksize(n-1) + blocksize(n)) / 4` (the first packet
    /// counts `blocksize / 2` on the timeline while decoding to nothing);
    /// ffmpeg's pts follow the same lapped model, though its reported
    /// per-packet duration field approximates short blocks as `bs0 / 2`.
    pub fn packet_blocksize(&self, packet: &[u8]) -> Option<u32> {
        let b0 = *packet.first()?;
        if b0 & 1 != 0 {
            return None; // header packet (audio packets have bit 0 clear)
        }
        // The mode number is the ilog(mode_count - 1) bits after the type bit;
        // modes are capped at 64, so it always fits the first byte.
        let bits = 32 - (self.mode_blockflag.len() as u32 - 1).leading_zeros();
        let mode = (u32::from(b0) >> 1) & ((1u32 << bits) - 1);
        let long = *self.mode_blockflag.get(mode as usize)?;
        Some(if long { self.bs1 } else { self.bs0 })
    }
}

/// Extract each mode's blockflag from a `\x05vorbis` setup header by scanning
/// the mode section backwards from the framing bit, without parsing the
/// variable-length codebooks before it (ffmpeg's `vorbis_parser` technique).
/// Vorbis packs bits LSB-first and pads the final byte's high bits with zeros,
/// so the framing bit (always 1) is the highest set bit of the last non-zero
/// byte. Each mode entry is 41 bits: blockflag, then 16-bit window / transform
/// types (reserved zero in Vorbis I) and an 8-bit mapping number (<= 63),
/// preceded by a 6-bit count. Walk entries backwards while they validate and
/// keep the LARGEST count whose 6-bit field matches: a mode's zero mapping
/// field can mimic the count field, so the first match may be short (the
/// false positive ffmpeg's walk also defends against). A malformed header
/// yields `None`.
fn mode_blockflags(setup: &[u8]) -> Option<Vec<bool>> {
    if !setup.starts_with(b"\x05vorbis") {
        return None;
    }
    let last = setup.iter().rposition(|&b| b != 0)?;
    let framing = last as u64 * 8 + u64::from(7 - setup[last].leading_zeros() as u8);
    let bit = |i: u64| (setup[(i / 8) as usize] >> (i % 8)) & 1;
    // A k-bit LSB-first field whose final bit sits at index `end`.
    let field = |end: u64, k: u64| -> u64 {
        (0..k).fold(0u64, |v, j| v | (u64::from(bit(end - k + 1 + j)) << j))
    };
    let mut best: Option<u64> = None;
    for m in 1..=64u64 {
        // The candidate count field must still sit past the 7-byte magic.
        if framing < m * 41 + 6 + 7 * 8 + 1 {
            break;
        }
        let e = framing - m * 41; // start bit of the m-th entry from the end
        if field(e + 40, 8) > 63 || field(e + 16, 16) != 0 || field(e + 32, 16) != 0 {
            break; // ran into codebook bits: no more mode entries
        }
        if field(e - 1, 6) == m - 1 {
            best = Some(m);
        }
    }
    let start = framing - best? * 41;
    Some((0..best?).map(|i| bit(start + i * 41) == 1).collect())
}

/// Identify the logical stream from its first packet's magic.
fn detect(packet: &[u8]) -> OggStreamInfo {
    if packet.starts_with(b"OpusHead") && packet.len() >= 12 {
        // OpusHead: magic(8), version(1), channel_count(1) at offset 9, pre-skip
        // (LE u16) at offset 10. Opus always decodes at 48 kHz regardless of the
        // original input rate.
        OggStreamInfo {
            codec: OggCodec::Opus,
            channels: packet[9],
            sample_rate: 48_000,
            pre_skip: u16::from_le_bytes([packet[10], packet[11]]),
        }
    } else if packet.starts_with(b"\x7fFLAC") && packet.len() >= 13 && &packet[9..13] == b"fLaC" {
        // Ogg-FLAC mapping: 0x7F "FLAC" major(1) minor(1) header-count(2 BE),
        // then the native "fLaC" marker + STREAMINFO block at offset 9.
        // A first packet whose STREAMINFO does not parse stays Other.
        match crate::flacparse::parse_streaminfo(&packet[9..]) {
            Some(si) => OggStreamInfo {
                codec: OggCodec::Flac,
                channels: si.channels,
                sample_rate: si.sample_rate,
                pre_skip: 0,
            },
            None => OggStreamInfo {
                codec: OggCodec::Other,
                channels: 0,
                sample_rate: 0,
                pre_skip: 0,
            },
        }
    } else if packet.starts_with(b"\x01vorbis") && packet.len() >= 16 {
        // Vorbis identification header: magic(7), version(4), channels at
        // offset 11, sample rate (LE u32) at offset 12.
        OggStreamInfo {
            codec: OggCodec::Vorbis,
            channels: packet[11],
            sample_rate: u32::from_le_bytes([packet[12], packet[13], packet[14], packet[15]]),
            pre_skip: 0,
        }
    } else {
        OggStreamInfo {
            codec: OggCodec::Other,
            channels: 0,
            sample_rate: 0,
            pre_skip: 0,
        }
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

    /// Like `page`, but with an explicit granule position.
    fn page_g(header_type: u8, serial: u32, seq: u32, granule: u64, packets: &[&[u8]]) -> Vec<u8> {
        let mut p = page(header_type, serial, seq, packets);
        p[6..14].copy_from_slice(&granule.to_le_bytes());
        p
    }

    #[test]
    fn parses_pre_skip_and_end_granule() {
        let serial = 5;
        let mut head = b"OpusHead".to_vec();
        head.push(1);
        head.push(2);
        head.extend_from_slice(&312u16.to_le_bytes()); // pre-skip
        head.extend_from_slice(&48_000u32.to_le_bytes());
        head.extend_from_slice(&[0, 0, 0]);

        let mut d = OggDemuxer::new();
        d.push_data(&page(0x02, serial, 0, &[&head]));
        d.push_data(&page(0x00, serial, 1, &[b"OpusTags"]));
        // End-of-stream page (bit 0x04) with a real granule position.
        d.push_data(&page_g(0x04, serial, 2, 96_312, &[&[0xAA, 0xBB]]));

        assert_eq!(d.info().unwrap().pre_skip, 312);
        assert_eq!(d.end_granule(), Some(96_312));
        assert!(d.head_header().unwrap().starts_with(b"OpusHead"));
    }

    #[test]
    fn end_granule_ignores_minus_one_sentinel() {
        let serial = 6;
        let mut d = OggDemuxer::new();
        d.push_data(&page(0x02, serial, 0, &[&opus_head(1)]));
        d.push_data(&page(0x00, serial, 1, &[b"OpusTags"]));
        // A -1 granule (no packet completed on the page) must not be recorded.
        d.push_data(&page_g(0x04, serial, 2, u64::MAX, &[&[0xAA]]));
        assert_eq!(d.end_granule(), None);
    }

    #[test]
    fn demuxes_opus_packets_skipping_headers() {
        let serial = 0xDEAD_BEEF;
        let mut d = OggDemuxer::new();
        d.push_data(&page(0x02, serial, 0, &[&opus_head(2)])); // BOS: OpusHead
        d.push_data(&page(0x00, serial, 1, &[b"OpusTags...."])); // setup header
        d.push_data(&page(
            0x00,
            serial,
            2,
            &[&[0xAA, 0xBB], &[0xCC, 0xDD, 0xEE]],
        ));

        assert_eq!(
            d.info(),
            Some(OggStreamInfo {
                codec: OggCodec::Opus,
                channels: 2,
                sample_rate: 48_000,
                pre_skip: 0
            })
        );
        let packets = d.take_packets();
        assert_eq!(
            packets,
            vec![vec![0xAA, 0xBB], vec![0xCC, 0xDD, 0xEE]],
            "audio packets only"
        );
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
        assert_eq!(
            d.take_packets(),
            vec![big],
            "packet reassembled across the page boundary"
        );
    }

    /// A page filled with 255-byte segments (no terminator), so its whole body
    /// continues the current packet into the next page.
    fn full_continued_page(header_type: u8, serial: u32, seq: u32) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&CAPTURE_PATTERN);
        out.extend_from_slice(&[0, header_type]);
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&serial.to_le_bytes());
        out.extend_from_slice(&seq.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.push(255); // 255 segments...
        out.extend_from_slice(&[255u8; 255]); // ...all max, so the packet never terminates
        out.extend_from_slice(&vec![0u8; 255 * 255]);
        out
    }

    #[test]
    fn unbounded_continuation_run_is_capped() {
        let serial = 7;
        let mut d = OggDemuxer::new();
        d.push_data(&full_continued_page(0x00, serial, 0));
        let pages = MAX_PACKET_BYTES / (255 * 255) + 2;
        for seq in 1..=pages as u32 {
            d.push_data(&full_continued_page(0x01, serial, seq));
        }
        assert!(
            d.partial.len() <= MAX_PACKET_BYTES,
            "reassembly buffer stays bounded, got {}",
            d.partial.len()
        );
    }

    /// The Ogg-FLAC mapping's first packet: `\x7fFLAC`, version 1.0, a BE u16
    /// count of following header packets, then the native `fLaC` marker +
    /// STREAMINFO block carrying `channels` / `sample_rate`.
    fn flac_first_packet(channels: u8, sample_rate: u32, headers: u16) -> Vec<u8> {
        let mut p = alloc::vec![0x7F];
        p.extend_from_slice(b"FLAC");
        p.extend_from_slice(&[1, 0]);
        p.extend_from_slice(&headers.to_be_bytes());
        p.extend_from_slice(b"fLaC");
        p.extend_from_slice(&[0x00, 0, 0, 34]);
        let mut body = [0u8; 34];
        body[10] = (sample_rate >> 12) as u8;
        body[11] = (sample_rate >> 4) as u8;
        body[12] = (((sample_rate & 0xF) as u8) << 4) | ((channels - 1) << 1);
        p.extend_from_slice(&body);
        p
    }

    #[test]
    fn detects_ogg_flac_and_classifies_headers() {
        let serial = 3;
        let mut d = OggDemuxer::new();
        d.push_data(&page(0x02, serial, 0, &[&flac_first_packet(2, 44_100, 1)]));
        // A VorbisComment metadata block (type 4, last-flag set) is a header.
        let comment = [&[0x84u8, 0, 0, 4][..], &[0u8; 4]].concat();
        d.push_data(&page(0x00, serial, 1, &[&comment]));
        // An audio frame leads with the 0xFF sync byte.
        let audio = [0xFFu8, 0xF8, 0x69, 0x18, 0x00, 0xBF];
        d.push_data(&page(0x00, serial, 2, &[&audio]));

        let info = d.info().unwrap();
        assert_eq!(info.codec, OggCodec::Flac);
        assert_eq!(info.channels, 2);
        assert_eq!(info.sample_rate, 44_100);
        assert!(d.head_header().unwrap().starts_with(b"\x7fFLAC"));
        assert_eq!(d.comment_header(), Some(comment.as_slice()));
        assert_eq!(d.take_packets(), vec![audio.to_vec()], "audio packets only");
    }

    #[test]
    fn malformed_flac_first_packet_is_other() {
        // Right magic, but the embedded native header is absent.
        let mut d = OggDemuxer::new();
        d.push_data(&page(0x02, 4, 0, &[b"\x7fFLAC\x01\x00\x00\x01fLa_"]));
        assert_eq!(d.info().unwrap().codec, OggCodec::Other);
        // Truncated STREAMINFO: detected magic but unparseable parameters.
        let mut d = OggDemuxer::new();
        d.push_data(&page(0x02, 5, 0, &[b"\x7fFLAC\x01\x00\x00\x01fLaC\x00"]));
        assert_eq!(d.info().unwrap().codec, OggCodec::Other);
    }

    #[test]
    fn ignores_a_second_logical_stream() {
        let mut d = OggDemuxer::new();
        d.push_data(&page(0x02, 1, 0, &[&opus_head(2)]));
        d.push_data(&page(0x00, 1, 1, &[b"OpusTags"]));
        d.push_data(&page(0x00, 2, 0, &[b"other-stream-packet"])); // different serial
        d.push_data(&page(0x00, 1, 2, &[&[0x01, 0x02]]));
        assert_eq!(
            d.take_packets(),
            vec![vec![0x01, 0x02]],
            "only the first serial"
        );
    }
}
