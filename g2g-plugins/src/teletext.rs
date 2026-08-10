//! EBU teletext (ETSI EN 300 706) subtitle-page decoding: the pure half behind
//! [`TeletextDec`](crate::teletextdec::TeletextDec).
//!
//! DVB carries teletext in a private PES (EN 300 472): a data_identifier byte
//! then fixed 46-byte data units, each one broadcast line. A subtitle unit holds
//! a framing code, a hamming 8/4 coded magazine / packet address, and 40
//! odd-parity bytes. Packet X/0 is the page header (page number, control bits,
//! and the national option subset the G0 set is read under); packets X/1..X/23
//! are the display rows.
//!
//! Bit order: a data unit's address and data bytes are the on-air bytes, which
//! EN 300 706 transmits LSB first, so each is bit-reversed before its hamming or
//! parity code means anything. The two bytes ahead of them (field parity / line
//! offset, and the framing code) are ordinary MSB-first fields and are not.
//!
//! Scope is subtitle pages: rows are decoded to plain text, colour and mosaic
//! control codes render as spaces, and a double-height row's blanked bottom half
//! is dropped so the cue carries the line once. Enhancement packets (X/26,
//! X/28, M/29) are not read, so the national option comes from the header bits.
//!
//! Every length, address and code here comes off the wire: a failed hamming or
//! parity check drops that byte, row or packet rather than propagating a
//! corrupt page, and no field is used before it is range-checked.

use alloc::string::String;
use alloc::vec::Vec;

/// The page a DVB teletext subtitle service conventionally rides, and what a
/// decoder follows when neither the PMT nor the caller names one.
pub const DEFAULT_PAGE: u16 = 888;

/// EN 300 472 data_identifier range for DVB teletext (0x10..=0x1F); anything
/// else on the pad is not a teletext PES payload.
const DATA_ID_MIN: u8 = 0x10;
const DATA_ID_MAX: u8 = 0x1F;

/// `data_unit_id` for an EBU teletext non-subtitle line, and for a subtitle line.
/// Both carry the same 44-byte body; a subtitle page is normally sent on 0x03,
/// but broadcasters do use 0x02, so both are decoded.
const UNIT_ID_NON_SUBTITLE: u8 = 0x02;
const UNIT_ID_SUBTITLE: u8 = 0x03;
/// `data_unit_id` for the stuffing that pads a payload to its required length.
const UNIT_ID_STUFFING: u8 = 0xFF;

/// Data units per 184-byte transport block, and the count a payload's unit total
/// must be congruent to (see [`encode_payload`]).
const UNITS_PER_BLOCK: usize = 4;
const UNITS_TARGET: usize = 3;

/// Every EBU teletext data unit is exactly this long: one line-offset byte, the
/// framing code, the two address bytes, and 40 data bytes.
const UNIT_BODY_LEN: usize = 44;

/// EN 300 706 framing code (`11100100`), stored verbatim: unlike the address and
/// data bytes it is not one of the bit-reversed on-air bytes.
const FRAMING_CODE: u8 = 0xE4;

/// `reserved_future_use` ('11') and `field_parity` (first field) of a data unit's
/// first byte, an ordinary MSB-first bit field. The low five bits are the VBI
/// line the unit was carried on; 7 is the first line EN 300 472 permits.
const FIELD_FLAGS: u8 = 0xE0;
const DEFAULT_LINE_OFFSET: u8 = 0x07;

/// Teletext rows: the header (0) plus 23 display rows.
const ROWS: usize = 24;
/// Display columns, and therefore the length of a packet's data block.
const COLUMNS: usize = 40;

/// Hamming 8/4 code words (EN 300 706 clause 8.2), indexed by the 4-bit value
/// they protect, in the data unit's bit order.
const HAM84_FWD: [u8; 16] = [
    0x15, 0x02, 0x49, 0x5E, 0x64, 0x73, 0x38, 0x2F, 0xD0, 0xC7, 0x8C, 0x9B, 0xA1, 0xB6, 0xEA, 0xFD,
];

/// Decode table for [`HAM84_FWD`]: a received byte maps to its 4-bit value, with
/// the single-bit errors the code corrects folded in. `INVALID` marks a byte at
/// distance 2 from a code word, which is a detected but uncorrectable error.
const HAM84_INV: [u8; 256] = build_ham84_inv();
const INVALID: u8 = 0xFF;

const fn build_ham84_inv() -> [u8; 256] {
    let mut table = [INVALID; 256];
    let mut v = 0usize;
    while v < 16 {
        table[HAM84_FWD[v] as usize] = v as u8;
        v += 1;
    }
    // One flipped bit is still uniquely attributable: the code has minimum
    // distance 4, so no byte is one bit from two different code words.
    let mut v = 0usize;
    while v < 16 {
        let mut bit = 0u32;
        while bit < 8 {
            table[(HAM84_FWD[v] ^ (1 << bit)) as usize] = v as u8;
            bit += 1;
        }
        v += 1;
    }
    table
}

/// Decode a hamming 8/4 byte to its 4-bit value, correcting a single-bit error.
/// `None` when the byte is too far from any code word.
pub fn unham84(byte: u8) -> Option<u8> {
    match HAM84_INV[byte as usize] {
        INVALID => None,
        v => Some(v),
    }
}

/// Encode a 4-bit value as its hamming 8/4 code word (the inverse of
/// [`unham84`]), for a writer authoring a teletext line.
pub fn ham84(value: u8) -> u8 {
    HAM84_FWD[(value & 0x0f) as usize]
}

/// Check a data byte's odd parity and return its seven data bits. `None` when the
/// byte's parity is even, which is a transmission error.
pub fn unparity(byte: u8) -> Option<u8> {
    (byte.count_ones() % 2 == 1).then_some(byte & 0x7f)
}

/// Set the parity bit of a 7-bit value so the byte has odd parity, the inverse of
/// [`unparity`].
pub fn parity(value: u8) -> u8 {
    let v = value & 0x7f;
    if v.count_ones().is_multiple_of(2) {
        v | 0x80
    } else {
        v
    }
}

/// Reverse a byte's bits, the transform EN 300 472's storage order needs before
/// any teletext code word can be read.
pub const fn reverse8(b: u8) -> u8 {
    b.reverse_bits()
}

/// One teletext line recovered from a PES data unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataUnit {
    /// Magazine 1..8 (the wire's 0 means 8).
    pub magazine: u8,
    /// Packet number: 0 is the page header, 1..=23 the display rows, and higher
    /// numbers are enhancement / service packets this decoder does not read.
    pub packet: u8,
    /// The 40 data bytes, still parity-coded.
    pub data: [u8; COLUMNS],
}

impl DataUnit {
    /// Parse one data unit body (the 44 bytes after `data_unit_id` and
    /// `data_unit_length`). `None` when the framing code or the hamming-coded
    /// address does not decode, which drops the line.
    pub fn parse(body: &[u8]) -> Option<Self> {
        if body.len() < UNIT_BODY_LEN {
            return None;
        }
        // body[0] is field parity + line offset, which a subtitle decoder that
        // does not reconstruct VBI timing has no use for. It and the framing code
        // are plain MSB-first bytes; only the address and data blocks below carry
        // the on-air bit order.
        if body[1] != FRAMING_CODE {
            return None;
        }
        let address = (unham84(reverse8(body[3]))? << 4) | unham84(reverse8(body[2]))?;
        let mut data = [0u8; COLUMNS];
        for (dst, src) in data.iter_mut().zip(&body[4..UNIT_BODY_LEN]) {
            *dst = reverse8(*src);
        }
        Some(DataUnit {
            magazine: address & 0x07,
            packet: (address >> 3) & 0x1f,
            data,
        })
    }

    /// The magazine as a page number's hundreds digit (the wire's 0 means 8).
    pub fn magazine_number(&self) -> u16 {
        if self.magazine == 0 {
            8
        } else {
            self.magazine as u16
        }
    }

    /// Write this line as a whole PES data unit (`data_unit_id`,
    /// `data_unit_length`, then the 44-byte body), the inverse of
    /// [`parse`](Self::parse). Subtitle lines go out on `data_unit_id` 0x03.
    pub fn encode(&self) -> [u8; UNIT_BODY_LEN + 2] {
        let address = (self.magazine & 0x07) | ((self.packet & 0x1f) << 3);
        let mut out = [0u8; UNIT_BODY_LEN + 2];
        out[0] = UNIT_ID_SUBTITLE;
        out[1] = UNIT_BODY_LEN as u8;
        out[2] = FIELD_FLAGS | DEFAULT_LINE_OFFSET;
        out[3] = FRAMING_CODE;
        out[4] = reverse8(ham84(address & 0x0f));
        out[5] = reverse8(ham84(address >> 4));
        for (dst, &src) in out[6..].iter_mut().zip(self.data.iter()) {
            *dst = reverse8(src);
        }
        out
    }

    /// A page header line (packet X/0) for `page`, marked as a subtitle page and
    /// read under national option subset `national_option`. `erase` sets C4, the
    /// bit that tells a receiver to clear the page before the rows arrive.
    pub fn page_header(page: u16, national_option: u8, erase: bool) -> Self {
        let mut data = [parity(b' '); COLUMNS];
        let within = page % 100;
        data[0] = ham84((within % 10) as u8);
        data[1] = ham84((within / 10) as u8);
        data[2] = ham84(0);
        data[3] = ham84(if erase { 0x08 } else { 0 });
        data[4] = ham84(0);
        data[5] = ham84(0x08); // C6: subtitle page
        data[6] = ham84(0);
        data[7] = ham84((national_option & 0x07) << 1);
        DataUnit {
            magazine: (page / 100) as u8 & 0x07,
            packet: 0,
            data,
        }
    }

    /// A display row (packet X/1..X/23) of `magazine` carrying `text`, padded with
    /// spaces to the full 40 columns.
    pub fn text_row(magazine: u8, packet: u8, text: &str) -> Self {
        let mut data = [parity(b' '); COLUMNS];
        for (slot, b) in data.iter_mut().zip(text.bytes()) {
            *slot = parity(b);
        }
        DataUnit {
            magazine: magazine & 0x07,
            packet,
            data,
        }
    }
}

/// Wrap teletext lines as one DVB teletext PES payload: the `data_identifier`
/// byte, each line's data unit, then stuffing units to the length EN 300 472
/// requires.
///
/// That length rule is why the stuffing is here rather than left to a caller: a
/// teletext PES packet must be a multiple of 184 bytes, and its header is fixed
/// at 45, so the payload has to come to 139 more than a multiple of 184. Since a
/// data unit is 46 bytes and 184 is four of them, that means padding the unit
/// count to three more than a multiple of four. A reference decoder rejects the
/// whole packet when this does not hold.
pub fn encode_payload(units: &[DataUnit]) -> Vec<u8> {
    let stuffing =
        (UNITS_PER_BLOCK + UNITS_TARGET - units.len() % UNITS_PER_BLOCK) % UNITS_PER_BLOCK;
    let total = units.len() + stuffing;
    let mut out = Vec::with_capacity(1 + total * (UNIT_BODY_LEN + 2));
    out.push(DATA_ID_MIN);
    for unit in units {
        out.extend_from_slice(&unit.encode());
    }
    for _ in 0..stuffing {
        out.push(UNIT_ID_STUFFING);
        out.push(UNIT_BODY_LEN as u8);
        out.extend(core::iter::repeat_n(0xFFu8, UNIT_BODY_LEN));
    }
    out
}

/// The page header (packet X/0) of a magazine: which page starts here and the
/// control bits that say how to read it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageHeader {
    /// The full page number as a viewer reads it (magazine hundreds + the two
    /// BCD digits of the page).
    pub page: u16,
    /// C6: this page is a subtitle page.
    pub subtitle: bool,
    /// C12..C14, the national option subset the G0 set is read under.
    pub national_option: u8,
}

impl PageHeader {
    /// Parse a page header from packet X/0's data block. `None` when a hamming
    /// code fails, which drops the header (and so the page it would have opened).
    fn parse(unit: &DataUnit) -> Option<Self> {
        let units = unham84(unit.data[0])?;
        let tens = unham84(unit.data[1])?;
        // Bytes 2..8 are the subcode nibbles with the control bits interleaved:
        // C6 (subtitle) is bit 3 of byte 5, C12..C14 bits 1..3 of byte 7.
        let c5c6 = unham84(unit.data[5])?;
        let c12c14 = unham84(unit.data[7])?;
        Some(PageHeader {
            page: unit.magazine_number() * 100 + tens as u16 * 10 + units as u16,
            subtitle: c5c6 & 0x08 != 0,
            // Bit 0 of byte 7's nibble is C11 (magazine serial); C12 is the low
            // bit of the subset index and C14 the high one.
            national_option: (c12c14 >> 1) & 0x07,
        })
    }
}

/// One decoded subtitle cue: the page's visible rows and when it went up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TeletextCue {
    /// Presentation timestamp of the page header that opened the cue.
    pub pts_ns: u64,
    /// How long the cue stands: until the page was replaced or erased. `0` for
    /// the cue still open when the stream ends.
    pub duration_ns: u64,
    /// The visible rows, newline-separated. Empty for the blank page that ends a
    /// subtitle (a consumer clears on it).
    pub text: String,
}

/// The out-of-band configuration the demuxer forwards ahead of the first data
/// unit: which page the PMT's `teletext_descriptor` named.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TeletextConfig {
    pub page: u16,
    pub language: [u8; 3],
}

/// Marker leading the page-selection blob a demuxer sends in band. A teletext PES
/// payload starts with a data_identifier in 0x10..=0x1F, so this byte cannot
/// begin one and the decoder tells the two apart without a side channel.
const CONFIG_MAGIC: [u8; 3] = [0xFF, b'T', b'X'];

/// The page-selection blob for `page` in `language`, the form
/// [`parse_page_config`] reads.
pub fn page_config_blob(page: u16, language: [u8; 3]) -> [u8; 8] {
    let p = page.to_be_bytes();
    [
        CONFIG_MAGIC[0],
        CONFIG_MAGIC[1],
        CONFIG_MAGIC[2],
        p[0],
        p[1],
        language[0],
        language[1],
        language[2],
    ]
}

/// Read a page-selection blob, or `None` when the bytes are not one (so they are
/// a teletext PES payload).
pub fn parse_page_config(bytes: &[u8]) -> Option<TeletextConfig> {
    if bytes.len() != 8 || bytes[..3] != CONFIG_MAGIC {
        return None;
    }
    Some(TeletextConfig {
        page: u16::from_be_bytes([bytes[3], bytes[4]]),
        language: [bytes[5], bytes[6], bytes[7]],
    })
}

/// Assembles teletext data units into subtitle cues for one page.
///
/// A cue's duration is only known when the page is replaced, so the decoder
/// holds the open page and releases it on the next header for the followed page
/// (or on [`flush`](Self::flush) at end of stream).
#[derive(Debug)]
pub struct TeletextDecoder {
    /// The page followed, or `None` to adopt the first subtitle page seen.
    page: Option<u16>,
    /// Rows 1..=23 of the page being received; `None` for a row not sent.
    rows: [Option<[u8; COLUMNS]>; ROWS],
    /// The national option subset of the open page's header.
    national_option: u8,
    /// PTS of the header that opened the page currently being received.
    open_pts: Option<u64>,
}

impl Default for TeletextDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl TeletextDecoder {
    pub fn new() -> Self {
        Self {
            page: None,
            rows: [None; ROWS],
            national_option: 0,
            open_pts: None,
        }
    }

    /// Follow `page` rather than the first subtitle page the stream carries.
    pub fn select_page(&mut self, page: u16) {
        self.page = Some(page);
    }

    /// The page being followed, once one is known.
    pub fn page(&self) -> Option<u16> {
        self.page
    }

    /// Feed one PES payload and take the cues it completed. A payload that is not
    /// a teletext one, or whose units all fail their codes, yields nothing.
    pub fn push(&mut self, payload: &[u8], pts_ns: u64) -> Vec<TeletextCue> {
        let mut cues = Vec::new();
        let Some((&data_id, mut rest)) = payload.split_first() else {
            return cues;
        };
        if !(DATA_ID_MIN..=DATA_ID_MAX).contains(&data_id) {
            return cues;
        }
        while rest.len() >= 2 {
            let unit_id = rest[0];
            let unit_len = rest[1] as usize;
            let Some(body) = rest.get(2..2 + unit_len) else {
                // A unit longer than what is left abandons the payload: the rest
                // cannot be resynchronized without guessing.
                break;
            };
            rest = &rest[2 + unit_len..];
            if unit_id != UNIT_ID_SUBTITLE && unit_id != UNIT_ID_NON_SUBTITLE {
                continue;
            }
            let Some(unit) = DataUnit::parse(body) else {
                continue;
            };
            if let Some(cue) = self.feed_unit(&unit, pts_ns) {
                cues.push(cue);
            }
        }
        cues
    }

    /// Release the page still open at end of stream, ending it at `end_ns` (the
    /// last timestamp the stream carried). A page that was never on screen for
    /// any span is dropped rather than emitted with a zero-length one, which an
    /// overlay would never paint.
    pub fn flush(&mut self, end_ns: u64) -> Option<TeletextCue> {
        self.close(end_ns)
    }

    /// Drop the page being received, for a seek: the rows behind the last header
    /// belong to the stream position being left.
    pub fn reset(&mut self) {
        self.rows = [None; ROWS];
        self.open_pts = None;
    }

    /// Route one line into the page being received, returning the cue it closed.
    fn feed_unit(&mut self, unit: &DataUnit, pts_ns: u64) -> Option<TeletextCue> {
        if unit.packet == 0 {
            let header = PageHeader::parse(unit)?;
            // Adopt the first subtitle page when no page was selected: a stream
            // whose PMT named none still decodes.
            if self.page.is_none() && header.subtitle {
                self.page = Some(header.page);
            }
            let followed = self.page?;
            if header.page != followed {
                // A header for another page of our magazine still ends ours: the
                // rows behind it are that page's, so row capture has to stop or
                // they leak into the page on screen.
                if unit.magazine_number() == followed / 100 {
                    return self.close(pts_ns);
                }
                return None;
            }
            let closed = self.close(pts_ns);
            self.national_option = header.national_option;
            self.open_pts = Some(pts_ns);
            return closed;
        }
        // Rows only land while a header for the followed page has opened one,
        // so a row from another magazine's page cannot leak in.
        if self.open_pts.is_some() && unit.magazine_number() == self.page? / 100 {
            if let Some(slot) = self.rows.get_mut(unit.packet as usize) {
                *slot = Some(unit.data);
            }
        }
        None
    }

    /// End the open page at `end_ns` and turn it into a cue.
    fn close(&mut self, end_ns: u64) -> Option<TeletextCue> {
        let start = self.open_pts.take()?;
        let text = self.render();
        self.rows = [None; ROWS];
        let duration_ns = end_ns.saturating_sub(start);
        // The blank page that erases a subtitle carries no text, and its own
        // start time is what ends the cue before it, so it is not a cue itself.
        // Nor is a page replaced within the payload that opened it: it never
        // stood on screen.
        (duration_ns > 0 && !text.is_empty()).then_some(TeletextCue {
            pts_ns: start,
            duration_ns,
            text,
        })
    }

    /// The visible rows of the page being received, as newline-separated text.
    /// Row 0 is the header line (clock and channel name), not subtitle text.
    fn render(&self) -> String {
        let mut out = String::new();
        let mut skip_next = false;
        for row in self.rows.iter().skip(1) {
            let Some(row) = row else {
                skip_next = false;
                continue;
            };
            if skip_next {
                skip_next = false;
                continue;
            }
            // A double-height row's bottom half is transmitted blanked; dropping
            // it keeps the line out of the cue once rather than twice.
            skip_next = row.iter().filter_map(|&b| unparity(b)).any(|c| c == 0x0d);
            let line = decode_row(row, self.national_option);
            let line = line.trim_end();
            if line.is_empty() {
                continue;
            }
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(line.trim_start());
        }
        out
    }
}

/// Decode one 40-byte display row to text under `national_option`. A byte whose
/// parity fails, and every spacing control code, renders as a space so the row
/// keeps its column positions.
fn decode_row(row: &[u8; COLUMNS], national_option: u8) -> String {
    let mut out = String::new();
    for &byte in row {
        match unparity(byte) {
            // Control codes are spacing attributes: they occupy a column and
            // display as a space.
            Some(c) if c < 0x20 => out.push(' '),
            Some(c) => out.push(g0_char(c, national_option)),
            None => out.push(' '),
        }
    }
    out
}

/// The character a G0 code point maps to under a national option subset. Thirteen
/// positions of the Latin G0 set are national; the rest are ASCII.
fn g0_char(code: u8, national_option: u8) -> char {
    let subset = &G0_NATIONAL_SUBSETS[(national_option & 0x07) as usize];
    match G0_NATIONAL_POSITIONS.iter().position(|&p| p == code) {
        Some(i) => subset[i],
        None => code as char,
    }
}

/// The thirteen G0 code points EN 300 706 leaves to the national option subset.
const G0_NATIONAL_POSITIONS: [u8; 13] = [
    0x23, 0x24, 0x40, 0x5B, 0x5C, 0x5D, 0x5E, 0x5F, 0x60, 0x7B, 0x7C, 0x7D, 0x7E,
];

/// The Latin G0 national option subsets EN 300 706 addresses with the three
/// national option bits of the page header, each giving the characters for
/// [`G0_NATIONAL_POSITIONS`] in order. The wider seven-bit selection (the four
/// extra bits of a packet X/28/0 or M/29/0) reaches Turkish, Estonian and the
/// other sets; without those packets the code is just these eight.
const G0_NATIONAL_SUBSETS: [[char; 13]; 8] = [
    // 0: English
    [
        '£', '$', '@', '←', '½', '→', '↑', '#', '\u{2013}', '¼', '‖', '¾', '÷',
    ],
    // 1: French
    [
        'é', 'ï', 'à', 'ë', 'ê', 'ù', 'î', '#', 'è', 'â', 'ô', 'û', 'ç',
    ],
    // 2: Swedish / Finnish / Hungarian
    [
        '#', '¤', 'É', 'Ä', 'Ö', 'Å', 'Ü', '_', 'é', 'ä', 'ö', 'å', 'ü',
    ],
    // 3: Czech / Slovak
    [
        '#', 'ů', 'č', 'ť', 'ž', 'ý', 'í', 'ř', 'é', 'á', 'ě', 'ú', 'š',
    ],
    // 4: German
    [
        '#', '$', '§', 'Ä', 'Ö', 'Ü', '^', '_', '°', 'ä', 'ö', 'ü', 'ß',
    ],
    // 5: Portuguese / Spanish
    [
        'ç', '$', '¡', 'á', 'é', 'í', 'ó', 'ú', '¿', 'ü', 'ñ', 'è', 'à',
    ],
    // 6: Italian
    [
        '£', '$', 'é', '°', 'ç', '→', '↑', '#', 'ù', 'à', 'ò', 'è', 'ì',
    ],
    // 7: Rumanian
    [
        '#', '¤', 'Ţ', 'Â', 'Ş', 'Ă', 'Î', 'ı', 'ţ', 'â', 'ş', 'ă', 'î',
    ],
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hamming_84_is_a_distance_four_code() {
        for a in 0..16u8 {
            for b in (a + 1)..16 {
                let d = (HAM84_FWD[a as usize] ^ HAM84_FWD[b as usize]).count_ones();
                assert!(d >= 4, "code words {a} and {b} are only distance {d} apart");
            }
        }
    }

    #[test]
    fn hamming_84_round_trips_and_corrects_one_flipped_bit() {
        for v in 0..16u8 {
            let word = ham84(v);
            assert_eq!(unham84(word), Some(v));
            for bit in 0..8 {
                assert_eq!(unham84(word ^ (1 << bit)), Some(v), "value {v} bit {bit}");
            }
        }
    }

    #[test]
    fn hamming_84_rejects_two_flipped_bits() {
        // Two flips land at distance 2 from the sent word and distance 2 from
        // some other one, so the code detects but cannot correct them.
        let word = ham84(5);
        assert_eq!(unham84(word ^ 0b11), None);
    }

    #[test]
    fn odd_parity_round_trips_and_rejects_a_flipped_bit() {
        for v in 0..0x80u8 {
            let byte = parity(v);
            assert_eq!(byte.count_ones() % 2, 1);
            assert_eq!(unparity(byte), Some(v));
            assert_eq!(unparity(byte ^ 0x01), None);
        }
    }

    #[test]
    fn a_payload_is_stuffed_to_the_length_a_teletext_pes_must_have() {
        for real in 1..12usize {
            let units: Vec<DataUnit> = (0..real)
                .map(|i| DataUnit::text_row(8, 1 + i as u8, "X"))
                .collect();
            let payload = encode_payload(&units);
            assert_eq!(
                (payload.len() + 45) % 184,
                0,
                "{real} units: a teletext PES packet (45-byte header) must be a multiple of 184"
            );
            // The stuffing must not add cues: only the real lines decode.
            let mut dec = TeletextDecoder::new();
            dec.select_page(888);
            dec.push(&encode_payload(&[DataUnit::page_header(888, 0, true)]), 0);
            dec.push(&payload, 0);
            assert_eq!(dec.flush(1_000).map(|c| c.text.lines().count()), Some(real));
        }
    }

    #[test]
    fn a_data_unit_round_trips_through_the_wire_bit_order() {
        let unit = DataUnit::text_row(8, 20, "HELLO");
        let wire = unit.encode();
        assert_eq!(wire[0], UNIT_ID_SUBTITLE);
        assert_eq!(wire[1] as usize, UNIT_BODY_LEN);
        assert_eq!(DataUnit::parse(&wire[2..]), Some(unit));
    }

    #[test]
    fn a_page_decodes_to_its_visible_rows_and_ends_on_the_next_header() {
        let mut dec = TeletextDecoder::new();
        dec.select_page(888);
        let payload = encode_payload(&[
            DataUnit::page_header(888, 0, true),
            DataUnit::text_row(8, 20, "HELLO WORLD"),
            DataUnit::text_row(8, 21, "SECOND LINE"),
        ]);
        assert!(dec.push(&payload, 1_000_000_000).is_empty(), "page is open");

        // A header with no rows behind it erases the subtitle.
        let clear = encode_payload(&[DataUnit::page_header(888, 0, true)]);
        let cues = dec.push(&clear, 3_000_000_000);
        assert_eq!(
            cues,
            Vec::from([TeletextCue {
                pts_ns: 1_000_000_000,
                duration_ns: 2_000_000_000,
                text: String::from("HELLO WORLD\nSECOND LINE"),
            }])
        );
        assert_eq!(
            dec.flush(4_000_000_000),
            None,
            "the erasing page carries no text"
        );
    }

    #[test]
    fn another_page_on_the_same_magazine_is_ignored() {
        let mut dec = TeletextDecoder::new();
        dec.select_page(888);
        let payload = encode_payload(&[
            DataUnit::page_header(801, 0, true),
            DataUnit::text_row(8, 20, "NOT SUBTITLES"),
        ]);
        assert!(dec.push(&payload, 0).is_empty());
        assert_eq!(dec.flush(1_000), None, "no page was opened");
    }

    #[test]
    fn a_header_for_another_page_of_the_magazine_ends_the_open_page() {
        let mut dec = TeletextDecoder::new();
        dec.select_page(888);
        dec.push(
            &encode_payload(&[
                DataUnit::page_header(888, 0, true),
                DataUnit::text_row(8, 20, "SUBTITLE"),
            ]),
            1_000_000_000,
        );
        // Page 801 shares magazine 8, so its rows follow its own header and must
        // not land on the subtitle page still on screen.
        let cues = dec.push(
            &encode_payload(&[
                DataUnit::page_header(801, 0, false),
                DataUnit::text_row(8, 21, "NEWS TICKER"),
            ]),
            2_000_000_000,
        );
        assert_eq!(
            cues,
            Vec::from([TeletextCue {
                pts_ns: 1_000_000_000,
                duration_ns: 1_000_000_000,
                text: String::from("SUBTITLE"),
            }])
        );
        assert_eq!(dec.flush(3_000_000_000), None, "no page is open any more");
    }

    #[test]
    fn a_page_replaced_inside_the_payload_that_opened_it_is_not_a_cue() {
        let mut dec = TeletextDecoder::new();
        dec.select_page(888);
        // Both headers carry the same PTS, so the first page never stood on
        // screen for any span an overlay could paint.
        let cues = dec.push(
            &encode_payload(&[
                DataUnit::page_header(888, 0, true),
                DataUnit::text_row(8, 20, "FLASHED"),
                DataUnit::page_header(888, 0, true),
                DataUnit::text_row(8, 20, "SHOWN"),
            ]),
            1_000_000_000,
        );
        assert!(cues.is_empty());
        assert_eq!(
            dec.flush(2_000_000_000).map(|c| c.text),
            Some(String::from("SHOWN"))
        );
    }

    #[test]
    fn a_reset_drops_the_page_being_received() {
        let mut dec = TeletextDecoder::new();
        dec.select_page(888);
        dec.push(
            &encode_payload(&[
                DataUnit::page_header(888, 0, true),
                DataUnit::text_row(8, 20, "BEFORE THE SEEK"),
            ]),
            1_000_000_000,
        );
        dec.reset();
        assert_eq!(dec.flush(2_000_000_000), None);
    }

    #[test]
    fn the_first_subtitle_page_is_adopted_when_none_was_selected() {
        let mut dec = TeletextDecoder::new();
        dec.push(
            &encode_payload(&[
                DataUnit::page_header(150, 0, true),
                DataUnit::text_row(1, 22, "AUTO PAGE"),
            ]),
            0,
        );
        assert_eq!(dec.page(), Some(150));
        assert_eq!(
            dec.flush(1_000).map(|c| c.text),
            Some(String::from("AUTO PAGE"))
        );
    }

    #[test]
    fn the_national_option_selects_the_g0_subset() {
        let mut dec = TeletextDecoder::new();
        dec.select_page(888);
        // Subset 4 (German) reads 0x5B as the character its subset names there,
        // not as the ASCII bracket subset 0 leaves in place.
        dec.push(
            &encode_payload(&[
                DataUnit::page_header(888, 4, true),
                DataUnit::text_row(8, 20, "A[B"),
            ]),
            0,
        );
        let expected = String::from_iter(['A', G0_NATIONAL_SUBSETS[4][3], 'B']);
        assert_ne!(G0_NATIONAL_SUBSETS[4][3], '[');
        assert_eq!(dec.flush(1_000).map(|c| c.text), Some(expected));
    }

    #[test]
    fn a_corrupt_framing_code_drops_the_line() {
        let mut dec = TeletextDecoder::new();
        dec.select_page(888);
        let mut payload = encode_payload(&[
            DataUnit::page_header(888, 0, true),
            DataUnit::text_row(8, 20, "DROPPED"),
        ]);
        // The framing code of the second unit: past the data_identifier, the
        // first unit, and the second unit's id / length bytes.
        payload[1 + (UNIT_BODY_LEN + 2) + 3] ^= 0xff;
        assert!(dec.push(&payload, 0).is_empty());
        assert_eq!(dec.flush(1_000), None, "the only row was dropped");
    }

    #[test]
    fn a_data_unit_length_past_the_payload_ends_the_walk_cleanly() {
        let mut dec = TeletextDecoder::new();
        dec.select_page(888);
        let mut payload = encode_payload(&[DataUnit::page_header(888, 0, true)]);
        payload[2] = 0xff; // data_unit_length longer than what follows
        assert!(dec.push(&payload, 0).is_empty());
        assert_eq!(dec.flush(1_000), None);
    }

    #[test]
    fn a_page_config_blob_round_trips_and_is_not_mistaken_for_a_payload() {
        let blob = page_config_blob(888, *b"eng");
        assert_eq!(
            parse_page_config(&blob),
            Some(TeletextConfig {
                page: 888,
                language: *b"eng",
            })
        );
        assert_eq!(
            parse_page_config(&encode_payload(&[DataUnit::page_header(888, 0, true)])),
            None
        );
    }
}
