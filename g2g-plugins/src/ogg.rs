//! Ogg demuxer (M116) + page writer (M789): parse an Ogg byte stream into the
//! packets of its logical bitstream (RFC 3533), the Opus / Vorbis carrier, and
//! frame packets back into pages.
//!
//! Pure `no_std + alloc` parsing, the [`crate::mpegts`] / [`crate::matroska`]
//! precedent for Ogg: sync to "OggS" pages, read the segment-table lacing to
//! frame packets (a packet runs across 255-valued segments and ends on a value
//! 0..254), reassemble packets that span pages, and skip the codec setup headers.
//! The [`crate::oggdemux::OggDemux`] element wraps it.
//!
//! Grouped multi-stream Ogg (M790) is handled: a file opens with one BOS page
//! per logical bitstream before any other page (RFC 3533 §4), and each serial's
//! codec mapping, headers, packets and granule timing are tracked independently
//! ([`OggLogicalStream`]). Codec support per stream: Opus fully (codec + channel
//! count from `OpusHead`, the two setup headers skipped), Vorbis and Ogg-FLAC
//! likewise, other codecs best-effort (tagged, all packets emitted).
//!
//! Not handled: **chained** Ogg, where a second physical stream (a fresh BOS
//! page) follows the first one's end-of-stream page. A BOS page arriving after
//! the opening group is ignored rather than misparsed, so a chained file
//! demuxes as its first physical stream.
//!
//! The [`OggPageWriter`] is the inverse framing side, wrapped by the
//! [`crate::oggmux::OggMux`] and [`crate::oggmuxn::OggMuxN`] elements.

use alloc::vec::Vec;

const CAPTURE_PATTERN: [u8; 4] = *b"OggS";
const HEADER_LEN: usize = 27; // fixed header before the segment table
                              // Cap cross-page packet reassembly. No real codec packet approaches this; the
                              // bound just stops a never-terminating run of continued pages from growing the
                              // partial packet without limit.
const MAX_PACKET_BYTES: usize = 8 * 1024 * 1024;
/// Cap on concurrent logical bitstreams. Serial numbers come from the file, so
/// a crafted opening block could otherwise name unboundedly many streams and
/// make the demuxer allocate per-serial state for each. Real grouped files carry
/// a handful.
const MAX_STREAMS: usize = 16;

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

/// One logical bitstream of an Ogg file: its serial number, codec mapping,
/// headers, granule anchors and demuxed packets. A grouped file carries several,
/// interleaved page by page; [`OggDemuxer`] keeps one of these per serial.
#[derive(Debug)]
pub struct OggLogicalStream {
    serial: u32,
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

impl OggLogicalStream {
    fn new(serial: u32) -> Self {
        Self {
            serial,
            partial: Vec::new(),
            info: None,
            packets_seen: 0,
            comment_header: None,
            head_header: None,
            setup_header: None,
            end_granulepos: None,
            first_data: None,
            audio_finalized: 0,
            completed: Vec::new(),
        }
    }

    /// The serial number identifying this bitstream in the file.
    pub fn serial(&self) -> u32 {
        self.serial
    }

    /// The stream's parameters (set once its first packet is parsed).
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
}

/// Incremental Ogg demuxer: feed bytes, drain elementary-stream packets. Tracks
/// every logical bitstream of a grouped file ([`OggLogicalStream`] per serial);
/// the single-stream accessors below read the first one.
#[derive(Debug, Default)]
pub struct OggDemuxer {
    buf: Vec<u8>,
    streams: Vec<OggLogicalStream>,
    /// Set by the first page that is not flagged beginning-of-stream. Grouped
    /// Ogg puts every stream's BOS page in the opening block, so after that a
    /// BOS page belongs to a chained physical stream, which is not handled.
    grouping_done: bool,
}

impl OggDemuxer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Every logical bitstream seen so far, in the order their BOS pages arrived.
    pub fn streams(&self) -> &[OggLogicalStream] {
        &self.streams
    }

    /// Mutable access to the `index`-th logical bitstream, to drain its packets.
    pub fn stream_mut(&mut self, index: usize) -> Option<&mut OggLogicalStream> {
        self.streams.get_mut(index)
    }

    /// Whether the opening beginning-of-stream block is over, so every logical
    /// bitstream of the file is now known.
    pub fn grouping_done(&self) -> bool {
        self.grouping_done
    }

    /// The index of the first logical bitstream of `codec`, or `None`.
    pub fn stream_of(&self, codec: OggCodec) -> Option<usize> {
        self.streams
            .iter()
            .position(|s| s.info.map(|i| i.codec) == Some(codec))
    }

    /// The first logical bitstream's parameters (set once its first packet is
    /// parsed).
    pub fn info(&self) -> Option<OggStreamInfo> {
        self.streams.first()?.info()
    }

    /// Drain the first logical bitstream's elementary-stream packets.
    pub fn take_packets(&mut self) -> Vec<Vec<u8>> {
        match self.streams.first_mut() {
            Some(s) => s.take_packets(),
            None => Vec::new(),
        }
    }

    /// The first stream's codec comment header (`OpusTags` for Opus), once
    /// parsed. Carries its VorbisComment metadata.
    pub fn comment_header(&self) -> Option<&[u8]> {
        self.streams.first()?.comment_header()
    }

    /// The first stream's identification header (`OpusHead`), once parsed. The
    /// decoder reads its pre-skip from it.
    pub fn head_header(&self) -> Option<&[u8]> {
        self.streams.first()?.head_header()
    }

    /// The first stream's Vorbis setup header (`\x05vorbis`), once parsed. The
    /// decoder builds its codebooks from it.
    pub fn setup_header(&self) -> Option<&[u8]> {
        self.streams.first()?.setup_header()
    }

    /// The first stream's final page granule position (total 48 kHz samples incl.
    /// pre-skip), once its end-of-stream page is parsed. Drives the
    /// end-of-stream padding trim: decoded samples beyond it are encoder padding.
    pub fn end_granule(&self) -> Option<u64> {
        self.streams.first()?.end_granule()
    }

    /// The first stream's first audio-bearing page granule (M778). See
    /// [`OggLogicalStream::first_data_granule`].
    pub fn first_data_granule(&self) -> Option<(u64, u32, bool)> {
        self.streams.first()?.first_data_granule()
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

    /// Route a complete page to its logical bitstream, starting one for a serial
    /// first seen in the opening beginning-of-stream block.
    fn parse_page(&mut self, page: &[u8], table_end: usize) {
        let header_type = page[5];
        let serial = u32::from_le_bytes([page[14], page[15], page[16], page[17]]);
        let bos = header_type & 0x02 != 0;
        if !bos {
            self.grouping_done = true;
        }
        let index = match self.streams.iter().position(|s| s.serial == serial) {
            Some(i) => i,
            None => {
                // A serial joins either from a BOS page in the opening group, or
                // as the very first stream seen when the file was joined
                // mid-stream (a byte-seek / network tune-in, which has no BOS
                // page to read). A BOS page after the group is a chained
                // physical stream and a later headerless serial is a stray page:
                // ignore both rather than misparse. The count cap keeps a
                // crafted opening block from allocating without end.
                let opens_group = bos && !self.grouping_done;
                if !(opens_group || self.streams.is_empty()) || self.streams.len() >= MAX_STREAMS {
                    return;
                }
                self.streams.push(OggLogicalStream::new(serial));
                self.streams.len() - 1
            }
        };
        self.streams[index].parse_page(page, table_end, header_type);
    }
}

impl OggLogicalStream {
    fn parse_page(&mut self, page: &[u8], table_end: usize, header_type: u8) {
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

/// The Ogg page checksum table: CRC-32 with polynomial `0x04c11db7`, zero
/// initial value, no reflection and no final inversion (RFC 3533 §6, *not* the
/// zlib CRC). Built at compile time.
const CRC_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut r = (i as u32) << 24;
        let mut bit = 0;
        while bit < 8 {
            r = if r & 0x8000_0000 != 0 {
                (r << 1) ^ 0x04c1_1db7
            } else {
                r << 1
            };
            bit += 1;
        }
        table[i] = r;
        i += 1;
    }
    table
};

/// The Ogg page checksum of `page`, whose CRC field must already be zeroed.
fn page_crc(page: &[u8]) -> u32 {
    page.iter().fold(0u32, |crc, &b| {
        (crc << 8) ^ CRC_TABLE[((crc >> 24) as u8 ^ b) as usize]
    })
}

/// Body bytes buffered before a page is flushed. Ogg allows 255 segments of up
/// to 255 bytes per page; libogg flushes around 4 kB, which keeps the 27-byte
/// header overhead under a percent without adding latency worth noticing.
const PAGE_FLUSH_BYTES: usize = 4096;

/// Frames elementary-stream packets into Ogg pages (RFC 3533), the inverse of
/// [`OggDemuxer`]. Packets are buffered and flushed a page at a time, so the
/// per-page header cost is amortized; the caller supplies each packet's granule
/// position (the codec mapping's unit, e.g. 48 kHz samples for Opus).
///
/// One logical bitstream: the first page emitted carries the beginning-of-stream
/// flag, the last carries end-of-stream. A packet longer than one page's worth
/// of segments spills onto continuation pages, whose granule is the -1 "no
/// packet completed here" sentinel.
#[derive(Debug)]
pub struct OggPageWriter {
    serial: u32,
    seq: u32,
    /// Queued packets with the granule position reached after each.
    pending: Vec<(Vec<u8>, u64)>,
    pending_bytes: usize,
    bos_done: bool,
    /// Granule of the last packet flushed, so a zero-packet end-of-stream page
    /// still names the stream's length.
    last_granule: u64,
}

impl OggPageWriter {
    pub fn new(serial: u32) -> Self {
        Self {
            serial,
            seq: 0,
            pending: Vec::new(),
            pending_bytes: 0,
            bos_done: false,
            last_granule: 0,
        }
    }

    /// The logical bitstream's serial number.
    pub fn serial(&self) -> u32 {
        self.serial
    }

    /// Queue `packet`, reaching `granule` once it is decoded. Returns whatever
    /// pages that completed (empty until a page's worth has accumulated). The
    /// most recent packet is always held back so the end-of-stream flag has a
    /// page to ride on.
    pub fn push_packet(&mut self, packet: Vec<u8>, granule: u64) -> Vec<u8> {
        self.pending_bytes = self.pending_bytes.saturating_add(packet.len());
        self.pending.push((packet, granule));
        if self.pending_bytes < PAGE_FLUSH_BYTES || self.pending.len() < 2 {
            return Vec::new();
        }
        let held = self.pending.pop().expect("pending is non-empty");
        let pages = self.emit(false);
        self.pending_bytes = held.0.len();
        self.pending.push(held);
        pages
    }

    /// Flush every queued packet. `eos` flags the final page end-of-stream,
    /// which closes the logical bitstream.
    pub fn flush(&mut self, eos: bool) -> Vec<u8> {
        self.emit(eos)
    }

    /// Build pages for the queued packets, splitting on the 255-segment page
    /// limit. `eos` flags the last page of this batch.
    fn emit(&mut self, eos: bool) -> Vec<u8> {
        // Lacing: each packet is `len / 255` full segments plus a terminator of
        // `len % 255` (so a packet whose length is a multiple of 255 ends on an
        // explicit 0 segment). Only the terminator completes the packet.
        let mut lacing: Vec<(u8, Option<u64>)> = Vec::new();
        for (packet, granule) in &self.pending {
            let mut left = packet.len();
            while left >= 255 {
                lacing.push((255, None));
                left -= 255;
            }
            lacing.push((left as u8, Some(*granule)));
        }
        let mut body = Vec::with_capacity(self.pending_bytes);
        for (packet, _) in &self.pending {
            body.extend_from_slice(packet);
        }
        self.pending.clear();
        self.pending_bytes = 0;

        let mut out = Vec::new();
        if lacing.is_empty() {
            // Nothing queued: an end-of-stream flush still needs a page to carry
            // the flag, so emit a zero-segment one at the last known granule.
            if eos {
                self.write_page(&mut out, &[], &[], self.last_granule, false, true);
            }
            return out;
        }
        let mut at = 0usize; // body offset of the current page
        let mut continued = false;
        let mut first = 0usize; // index of this page's first lacing entry
        while first < lacing.len() {
            let last = (first + 255).min(lacing.len());
            let group = &lacing[first..last];
            let len: usize = group.iter().map(|(s, _)| *s as usize).sum();
            let segments: Vec<u8> = group.iter().map(|(s, _)| *s).collect();
            // The page's granule names the last packet completing on it; a page
            // that completes none carries the -1 sentinel.
            let granule = group.iter().rev().find_map(|(_, g)| *g).unwrap_or(u64::MAX);
            if granule != u64::MAX {
                self.last_granule = granule;
            }
            let is_last = last == lacing.len();
            self.write_page(
                &mut out,
                &segments,
                &body[at..at + len],
                granule,
                continued,
                eos && is_last,
            );
            at += len;
            // A page ending on a 255 segment leaves its packet unterminated.
            continued = group.last().map(|(_, g)| g.is_none()).unwrap_or(false);
            first = last;
        }
        out
    }

    fn write_page(
        &mut self,
        out: &mut Vec<u8>,
        segments: &[u8],
        body: &[u8],
        granule: u64,
        continued: bool,
        eos: bool,
    ) {
        let start = out.len();
        let mut header_type = 0u8;
        if continued {
            header_type |= 0x01;
        }
        if !self.bos_done {
            header_type |= 0x02;
            self.bos_done = true;
        }
        if eos {
            header_type |= 0x04;
        }
        out.extend_from_slice(&CAPTURE_PATTERN);
        out.push(0); // stream structure version
        out.push(header_type);
        out.extend_from_slice(&granule.to_le_bytes());
        out.extend_from_slice(&self.serial.to_le_bytes());
        out.extend_from_slice(&self.seq.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // CRC, filled in below
        out.push(segments.len() as u8);
        out.extend_from_slice(segments);
        out.extend_from_slice(body);
        self.seq = self.seq.wrapping_add(1);
        let crc = page_crc(&out[start..]);
        out[start + 22..start + 26].copy_from_slice(&crc.to_le_bytes());
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
        let partial = &d.streams[0].partial;
        assert!(
            partial.len() <= MAX_PACKET_BYTES,
            "reassembly buffer stays bounded, got {}",
            partial.len()
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

    /// A real beginning-of-stream page from an ffmpeg-authored `.opus` file
    /// (`ffmpeg -f lavfi -i sine -c:a libopus`): 27-byte header, one 19-byte
    /// segment carrying `OpusHead`. Its stored checksum is the CRC oracle.
    const FFMPEG_BOS_PAGE: [u8; 47] = [
        0x4F, 0x67, 0x67, 0x53, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x75,
        0x9A, 0xD7, 0x87, 0x00, 0x00, 0x00, 0x00, 0x70, 0x0B, 0x30, 0x2A, 0x01, 0x13, 0x4F, 0x70,
        0x75, 0x73, 0x48, 0x65, 0x61, 0x64, 0x01, 0x02, 0x38, 0x01, 0x80, 0xBB, 0x00, 0x00, 0x00,
        0x00, 0x00,
    ];

    #[test]
    fn page_crc_matches_an_ffmpeg_authored_page() {
        let mut page = FFMPEG_BOS_PAGE;
        let stored = u32::from_le_bytes([page[22], page[23], page[24], page[25]]);
        page[22..26].copy_from_slice(&0u32.to_le_bytes());
        assert_eq!(
            page_crc(&page),
            stored,
            "Ogg CRC-32 (poly 0x04c11db7, init 0)"
        );
        // The writer reproduces the same page byte for byte from the same
        // packet, serial and flags.
        let serial = u32::from_le_bytes([
            FFMPEG_BOS_PAGE[14],
            FFMPEG_BOS_PAGE[15],
            FFMPEG_BOS_PAGE[16],
            FFMPEG_BOS_PAGE[17],
        ]);
        let mut w = OggPageWriter::new(serial);
        w.push_packet(FFMPEG_BOS_PAGE[28..].to_vec(), 0);
        assert_eq!(w.flush(false), FFMPEG_BOS_PAGE.to_vec());
    }

    /// Read a page's `(header_type, granule, sequence, segment table, body)`.
    fn read_page(data: &[u8], at: usize) -> (u8, u64, u32, Vec<u8>, Vec<u8>, usize) {
        assert_eq!(&data[at..at + 4], &CAPTURE_PATTERN, "page at {at}");
        let n = data[at + 26] as usize;
        let table = data[at + HEADER_LEN..at + HEADER_LEN + n].to_vec();
        let body_len: usize = table.iter().map(|&s| s as usize).sum();
        let start = at + HEADER_LEN + n;
        let gp = u64::from_le_bytes(data[at + 6..at + 14].try_into().unwrap());
        let seq = u32::from_le_bytes(data[at + 18..at + 22].try_into().unwrap());
        (
            data[at + 5],
            gp,
            seq,
            table,
            data[start..start + body_len].to_vec(),
            start + body_len,
        )
    }

    /// Every page in `data` has a valid checksum.
    fn assert_crcs_valid(data: &[u8]) {
        let mut at = 0;
        while at < data.len() {
            let (_, _, _, table, body, end) = read_page(data, at);
            let mut page = data[at..end].to_vec();
            let stored = u32::from_le_bytes([page[22], page[23], page[24], page[25]]);
            page[22..26].copy_from_slice(&0u32.to_le_bytes());
            assert_eq!(page_crc(&page), stored, "page at {at}");
            assert_eq!(table.len(), page[26] as usize);
            assert_eq!(body.len(), end - at - HEADER_LEN - table.len());
            at = end;
        }
    }

    #[test]
    fn laces_a_255_multiple_with_a_terminating_zero_segment() {
        let mut w = OggPageWriter::new(1);
        w.push_packet(vec![7u8; 510], 100);
        let pages = w.flush(true);
        let (ht, gp, seq, table, body, end) = read_page(&pages, 0);
        assert_eq!(end, pages.len(), "one page");
        assert_eq!(ht, 0x02 | 0x04, "first and last page of the stream");
        assert_eq!((gp, seq), (100, 0));
        assert_eq!(
            table,
            vec![255, 255, 0],
            "a 255-multiple needs an explicit terminator"
        );
        assert_eq!(body.len(), 510);
        assert_crcs_valid(&pages);
        // The demuxer recovers exactly one packet.
        let mut d = OggDemuxer::new();
        d.push_data(&pages);
        assert_eq!(d.take_packets(), vec![vec![7u8; 510]]);
    }

    #[test]
    fn a_packet_past_64k_spills_onto_continuation_pages() {
        // 255 * 255 = 65025 bytes fill one page's segment table exactly, so a
        // longer packet must continue onto a second page.
        let big: Vec<u8> = (0..70_000u32).map(|x| x as u8).collect();
        let mut w = OggPageWriter::new(9);
        w.push_packet(big.clone(), 4242);
        let pages = w.flush(true);
        assert_crcs_valid(&pages);

        let (ht0, gp0, seq0, table0, _, end0) = read_page(&pages, 0);
        assert_eq!(ht0, 0x02, "beginning of stream, not yet the end");
        assert_eq!(gp0, u64::MAX, "no packet completes on the first page");
        assert_eq!((seq0, table0.len()), (0, 255));
        let (ht1, gp1, seq1, _, _, end1) = read_page(&pages, end0);
        assert_eq!(ht1, 0x01 | 0x04, "continued, and the end of the stream");
        assert_eq!((gp1, seq1), (4242, 1));
        assert_eq!(end1, pages.len(), "two pages");

        let mut d = OggDemuxer::new();
        d.push_data(&pages);
        assert_eq!(d.take_packets(), vec![big], "reassembled across the split");
    }

    #[test]
    fn header_pages_are_flagged_and_sequenced() {
        let mut w = OggPageWriter::new(3);
        w.push_packet(b"ident".to_vec(), 0);
        let mut out = w.flush(false); // beginning-of-stream page, ident alone
        w.push_packet(b"comment".to_vec(), 0);
        w.push_packet(b"setup".to_vec(), 0);
        out.extend_from_slice(&w.flush(false));
        w.push_packet(b"audio".to_vec(), 1024);
        out.extend_from_slice(&w.flush(true));
        assert_crcs_valid(&out);

        let (ht0, gp0, _, t0, b0, e0) = read_page(&out, 0);
        assert_eq!((ht0, gp0, t0.len()), (0x02, 0, 1));
        assert_eq!(b0, b"ident".to_vec());
        let (ht1, gp1, seq1, t1, _, e1) = read_page(&out, e0);
        assert_eq!(
            (ht1, gp1, seq1, t1.len()),
            (0x00, 0, 1, 2),
            "comment + setup share the second page at granule 0"
        );
        let (ht2, gp2, seq2, _, b2, e2) = read_page(&out, e1);
        assert_eq!((ht2, gp2, seq2), (0x04, 1024, 2));
        assert_eq!(b2, b"audio".to_vec());
        assert_eq!(e2, out.len());
    }

    #[test]
    fn ignores_a_serial_that_never_opened_a_stream() {
        let mut d = OggDemuxer::new();
        d.push_data(&page(0x02, 1, 0, &[&opus_head(2)]));
        d.push_data(&page(0x00, 1, 1, &[b"OpusTags"]));
        // A serial with no BOS page in the opening group is a stray page.
        d.push_data(&page(0x00, 2, 0, &[b"other-stream-packet"]));
        d.push_data(&page(0x00, 1, 2, &[&[0x01, 0x02]]));
        assert_eq!(d.streams().len(), 1);
        assert_eq!(
            d.take_packets(),
            vec![vec![0x01, 0x02]],
            "only the opened serial"
        );
    }

    #[test]
    fn grouped_streams_demux_independently() {
        let (a, b) = (0x1111u32, 0x2222u32);
        let mut d = OggDemuxer::new();
        // RFC 3533 grouping: every stream's BOS page first, then the rest.
        d.push_data(&page(0x02, a, 0, &[&opus_head(2)]));
        d.push_data(&page(0x02, b, 0, &[&flac_first_packet(1, 44_100, 1)]));
        d.push_data(&page(0x00, a, 1, &[b"OpusTags"]));
        let comment = [&[0x84u8, 0, 0, 4][..], &[0u8; 4]].concat();
        d.push_data(&page(0x00, b, 1, &[&comment]));
        // Interleaved data pages, each ending its own stream.
        d.push_data(&page_g(0x00, a, 2, 960, &[&[0xAA, 0xBB]]));
        let flac_frame = [0xFFu8, 0xF8, 0x69, 0x18, 0x00, 0x8A];
        d.push_data(&page_g(0x04, b, 2, 4096, &[&flac_frame]));
        d.push_data(&page_g(0x04, a, 3, 1920, &[&[0xCC]]));

        assert_eq!(d.streams().len(), 2, "both logical bitstreams tracked");
        assert_eq!(d.streams()[0].serial(), a);
        assert_eq!(d.streams()[1].serial(), b);
        assert_eq!(d.streams()[0].info().unwrap().codec, OggCodec::Opus);
        assert_eq!(d.streams()[1].info().unwrap().codec, OggCodec::Flac);
        assert_eq!(d.streams()[1].info().unwrap().sample_rate, 44_100);
        assert_eq!(d.stream_of(OggCodec::Flac), Some(1));
        // Per-stream granules, not a shared one.
        assert_eq!(d.streams()[0].end_granule(), Some(1920));
        assert_eq!(d.streams()[1].end_granule(), Some(4096));
        assert_eq!(
            d.stream_mut(0).unwrap().take_packets(),
            vec![vec![0xAA, 0xBB], vec![0xCC]]
        );
        assert_eq!(
            d.stream_mut(1).unwrap().take_packets(),
            vec![flac_frame.to_vec()]
        );
    }

    #[test]
    fn a_chained_beginning_of_stream_page_is_ignored() {
        let mut d = OggDemuxer::new();
        d.push_data(&page(0x02, 1, 0, &[&opus_head(2)]));
        d.push_data(&page(0x00, 1, 1, &[b"OpusTags"]));
        d.push_data(&page_g(0x04, 1, 2, 960, &[&[0x01]]));
        // A fresh BOS after the first stream ended: a chained physical stream.
        d.push_data(&page(0x02, 9, 0, &[&opus_head(1)]));
        d.push_data(&page(0x00, 9, 1, &[b"OpusTags"]));
        d.push_data(&page(0x00, 9, 2, &[&[0x02]]));
        assert_eq!(d.streams().len(), 1, "chained streams are not demuxed");
        assert_eq!(d.take_packets(), vec![vec![0x01]]);
    }

    #[test]
    fn the_logical_stream_count_is_capped() {
        let mut d = OggDemuxer::new();
        for serial in 0..(MAX_STREAMS as u32 + 8) {
            d.push_data(&page(0x02, serial, 0, &[&opus_head(1)]));
        }
        assert_eq!(d.streams().len(), MAX_STREAMS);
    }
}
