//! CEA-608 / CEA-708 closed-caption decoding (M426, `no_std`): mine the
//! `cc_data` byte triples from H.264 / H.265 SEI `user_data_registered_itu_t_t35`
//! messages, then decode them into timed [`Cue`]s the existing overlay path
//! renders (`crate::textoverlay`). Pure byte / bit work with no OS dependency, so
//! it sits on the `no_std + alloc` baseline alongside `crate::subparse`; the
//! `crate::ccextract` element wraps it as a pipeline node.
//!
//! Closed captions ride *inside* the compressed video bitstream, not in a
//! container text track: each coded picture may carry an SEI message whose payload
//! (ATSC A/53 / SCTE-128) is a `cc_data` block of `(cc_type, cc_data_1,
//! cc_data_2)` triples. `cc_type` 0/1 are the two fields of legacy CEA-608
//! line-21 captions; 2/3 are CEA-708 DTVCC packet bytes. This module decodes the
//! CEA-608 field-1 path (M426: pop-on captions, the basic North-American
//! character set, channel CC1); CEA-608 positioning / channels / roll-up and
//! CEA-708 are layered on in later milestones.
//!
//! **Never trust the stream.** Counts, lengths, and offsets come from an
//! attacker-controlled bitstream, so every read is bounds-checked and a malformed
//! SEI yields no triples rather than panicking.

use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::VideoCodec;

use crate::annexb::{h264_nal_type, h265_nal_type, nal_units_any, strip_emulation_prevention};
use crate::subparse::{Cue, CueSettings, TextAlign};

/// One closed-caption byte triple extracted from a `cc_data` block: a two-bit
/// `cc_type` and the two caption data bytes. `cc_type` 0/1 select CEA-608 line-21
/// field 1/2; 2 is a CEA-708 DTVCC packet continuation and 3 a packet start. Only
/// triples whose `cc_valid` bit was set are surfaced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CcTriple {
    /// The two-bit data-channel type (0..=3).
    pub cc_type: u8,
    /// First caption data byte (`cc_data_1`), parity bit still present.
    pub b0: u8,
    /// Second caption data byte (`cc_data_2`), parity bit still present.
    pub b1: u8,
}

/// ATSC A/53 user-identifier `GA94` marking a `cc_data` ATSC1 user-data SEI.
const USER_IDENTIFIER_GA94: u32 = 0x4741_3934;
/// `user_data_type_code` selecting `cc_data` within an ATSC1 user-data block.
const USER_DATA_TYPE_CC: u8 = 0x03;

/// Extract every valid `cc_data` triple carried in the SEI messages of one access
/// unit (`au`, either Annex-B or AVCC framed) for `codec`. Walks the NAL units,
/// parses each SEI NAL's messages, and decodes the ATSC1 `cc_data` payload of any
/// `user_data_registered_itu_t_t35` (payload type 4) message tagged `GA94`.
/// Returns triples in transmission order; a malformed message is skipped.
pub fn extract_cc_data(au: &[u8], codec: VideoCodec) -> Vec<CcTriple> {
    let mut out = Vec::new();
    for nal in nal_units_any(au) {
        // SEI NAL header + RBSP offset differs by codec: H.264 SEI is NAL type 6
        // with a 1-byte header; H.265 prefix-SEI (39) / suffix-SEI (40) carry a
        // 2-byte header.
        let rbsp_off = match codec {
            VideoCodec::H265 => match h265_nal_type(nal) {
                Some(39) | Some(40) => 2,
                _ => continue,
            },
            _ => match h264_nal_type(nal) {
                Some(6) => 1,
                _ => continue,
            },
        };
        if nal.len() <= rbsp_off {
            continue;
        }
        let rbsp = strip_emulation_prevention(&nal[rbsp_off..]);
        parse_sei_messages(&rbsp, &mut out);
    }
    out
}

/// Walk the SEI messages of one SEI RBSP, appending the `cc_data` triples of any
/// ATSC1 caption message. Each message is `payloadType` then `payloadSize` (both
/// little-endian-extended by `0xFF` run bytes) then `payloadSize` payload bytes.
fn parse_sei_messages(rbsp: &[u8], out: &mut Vec<CcTriple>) {
    let mut i = 0usize;
    // Stop once only the rbsp_trailing_bits (a lone 0x80) remain.
    while i + 1 < rbsp.len() {
        let Some((payload_type, n)) = read_ff_extended(rbsp, i) else { break };
        i = n;
        let Some((payload_size, n)) = read_ff_extended(rbsp, i) else { break };
        i = n;
        let end = match i.checked_add(payload_size) {
            Some(e) if e <= rbsp.len() => e,
            _ => break,
        };
        if payload_type == 4 {
            parse_user_data_registered(&rbsp[i..end], out);
        }
        i = end;
    }
}

/// Read an SEI `0xFF`-extended value (`payloadType` / `payloadSize`): a run of
/// `0xFF` bytes each adding 255, then a final byte. Returns the value and the
/// index past it, or `None` if the buffer ends mid-value.
fn read_ff_extended(data: &[u8], mut i: usize) -> Option<(usize, usize)> {
    let mut value: usize = 0;
    loop {
        let b = *data.get(i)?;
        i += 1;
        value = value.checked_add(b as usize)?;
        if b != 0xFF {
            return Some((value, i));
        }
    }
}

/// Parse a `user_data_registered_itu_t_t35` payload, appending `cc_data` triples
/// when it is an ATSC1 `GA94` caption block. Layout: `itu_t_t35_country_code`
/// (0xB5 USA, with a 0xFF escape), `provider_code` (16 bits), `user_identifier`
/// (32 bits, `GA94`), `user_data_type_code` (8 bits, 0x03), a flags byte holding
/// `cc_count`, an `em_data` byte, then `cc_count` triples.
fn parse_user_data_registered(p: &[u8], out: &mut Vec<CcTriple>) {
    let mut i = 0usize;
    let country = *p.first().unwrap_or(&0);
    i += 1;
    if country == 0xFF {
        // A 0xFF country code is followed by an extension byte (T.35 escape).
        i += 1;
    }
    // provider_code (16) + user_identifier (32) = 6 bytes after the country code.
    let Some(window) = p.get(i..i + 6) else { return };
    let user_identifier = u32::from_be_bytes([window[2], window[3], window[4], window[5]]);
    if user_identifier != USER_IDENTIFIER_GA94 {
        return;
    }
    i += 6;
    let Some(&type_code) = p.get(i) else { return };
    i += 1;
    if type_code != USER_DATA_TYPE_CC {
        return;
    }
    let Some(&flags) = p.get(i) else { return };
    i += 1;
    // process_cc_data_flag (bit 6) must be set; cc_count is the low 5 bits.
    if flags & 0x40 == 0 {
        return;
    }
    let cc_count = (flags & 0x1F) as usize;
    i += 1; // em_data
    for _ in 0..cc_count {
        let Some(triple) = p.get(i..i + 3) else { break };
        i += 3;
        let marker = triple[0];
        let cc_valid = marker & 0x04 != 0;
        let cc_type = marker & 0x03;
        if cc_valid {
            out.push(CcTriple { cc_type, b0: triple[1], b1: triple[2] });
        }
    }
}

/// The visible row count of a CEA-608 caption grid (rows 1..=15).
const ROWS: usize = 15;
/// The visible column count of a CEA-608 caption row.
const COLS: usize = 32;

/// Which of the four CEA-608 caption services to decode. Each line-21 field
/// carries two interleaved channels selected by the channel bit of the control
/// codes: CC1 / CC2 ride field 1 (`cc_type` 0), CC3 / CC4 field 2 (`cc_type` 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Cea608Channel {
    /// Field 1, channel 1 (the primary captions).
    #[default]
    Cc1,
    /// Field 1, channel 2.
    Cc2,
    /// Field 2, channel 1.
    Cc3,
    /// Field 2, channel 2.
    Cc4,
}

impl Cea608Channel {
    /// The line-21 field (`cc_type`) this channel rides: 0 for CC1/CC2, 1 for CC3/CC4.
    pub fn field(self) -> u8 {
        match self {
            Cea608Channel::Cc1 | Cea608Channel::Cc2 => 0,
            Cea608Channel::Cc3 | Cea608Channel::Cc4 => 1,
        }
    }

    /// The in-field channel number (1 or 2) the control-code channel bit selects.
    fn channel(self) -> u8 {
        match self {
            Cea608Channel::Cc1 | Cea608Channel::Cc3 => 1,
            Cea608Channel::Cc2 | Cea608Channel::Cc4 => 2,
        }
    }
}

/// Caption presentation mode, set by the misc-control commands: pop-on loads a
/// hidden back buffer flipped on by EOC; roll-up types into the displayed window
/// and scrolls on CR; paint-on writes directly to the displayed buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    PopOn,
    RollUp,
    PaintOn,
}

/// A 15x32 character grid (the CEA-608 safe-caption area); empty cells are spaces.
#[derive(Debug, Clone)]
struct Screen {
    rows: [[char; COLS]; ROWS],
}

impl Screen {
    fn new() -> Self {
        Self { rows: [[' '; COLS]; ROWS] }
    }

    fn clear(&mut self) {
        self.rows = [[' '; COLS]; ROWS];
    }

    fn is_empty(&self) -> bool {
        self.rows.iter().all(|r| r.iter().all(|&c| c == ' '))
    }

    /// Join the non-blank rows top-to-bottom into cue text (each row's leading and
    /// trailing padding trimmed; block placement is carried in [`Screen::settings`]).
    fn text(&self) -> String {
        let mut out = String::new();
        for r in &self.rows {
            let line: String = r.iter().collect();
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(trimmed);
        }
        out
    }

    /// Derive cue placement: vertical from the topmost non-blank row, horizontal
    /// from the smallest leading-space count, alignment left (CEA-608 is left-set).
    fn settings(&self, color: Option<[u8; 4]>) -> CueSettings {
        let mut first_row = None;
        let mut min_indent = COLS;
        for (i, r) in self.rows.iter().enumerate() {
            let line: String = r.iter().collect();
            if line.trim().is_empty() {
                continue;
            }
            if first_row.is_none() {
                first_row = Some(i);
            }
            let indent = line.len() - line.trim_start().len();
            min_indent = min_indent.min(indent);
        }
        let line = first_row.map(|r| ((r as u32 * 100) / (ROWS as u32 - 1)) as u8);
        let position = (min_indent < COLS).then(|| ((min_indent as u32 * 100) / COLS as u32) as u8);
        CueSettings { position, line, align: TextAlign::Start, color, ..CueSettings::default() }
    }
}

/// A CEA-608 line-21 caption decoder. Feed it the `(cc_data_1, cc_data_2)` byte
/// pairs of its channel's field, in presentation order, via [`Cea608::push_pair`];
/// finished cues accumulate and are drained with [`Cea608::take_cues`]. It decodes
/// the chosen channel of the field (the other channel's codes are tracked but not
/// rendered), pop-on / roll-up / paint-on modes, PAC row+indent positioning, the
/// basic / special / extended-Western-European character sets, mid-row style
/// codes, and colour.
#[derive(Debug)]
pub struct Cea608 {
    /// The channel this decoder renders (CC1..CC4).
    selected: Cea608Channel,
    /// The channel the most recent control code addressed (1 or 2).
    cur_channel: u8,
    mode: Mode,
    /// The displayed (on-air) grid: roll-up / paint-on write here, EOC flips into it.
    disp: Screen,
    /// The non-displayed (back) grid pop-on captions load into.
    back: Screen,
    /// Running time the current displayed content began, for the next cue's start.
    disp_start: Option<u64>,
    /// Roll-up window height (2, 3, or 4 rows).
    roll_rows: usize,
    /// Roll-up base (bottom) row, 1-based.
    base_row: usize,
    /// Current write row (1..=15) and column (0..=31).
    row: usize,
    col: usize,
    /// Current text colour (from a PAC / mid-row code), applied to finished cues.
    color: Option<[u8; 4]>,
    /// PTS of the pair currently being decoded (so writes can stamp `disp_start`).
    cur_pts: u64,
    /// Last control pair, to drop the immediate doubled retransmission.
    last_ctrl: Option<(u8, u8)>,
    out: Vec<Cue>,
}

impl Default for Cea608 {
    fn default() -> Self {
        Self::new()
    }
}

impl Cea608 {
    /// A fresh CC1 decoder in pop-on mode with empty buffers.
    pub fn new() -> Self {
        Self::for_channel(Cea608Channel::Cc1)
    }

    /// A fresh decoder rendering `channel` (CC1..CC4).
    pub fn for_channel(channel: Cea608Channel) -> Self {
        Self {
            selected: channel,
            cur_channel: channel.channel(),
            mode: Mode::PopOn,
            disp: Screen::new(),
            back: Screen::new(),
            disp_start: None,
            roll_rows: 4,
            base_row: 15,
            row: 15,
            col: 0,
            color: None,
            cur_pts: 0,
            last_ctrl: None,
            out: Vec::new(),
        }
    }

    /// Take the cues finished so far, leaving the decoder ready for more pairs.
    pub fn take_cues(&mut self) -> Vec<Cue> {
        core::mem::take(&mut self.out)
    }

    /// Finalize any on-screen caption at running time `end_ns` (call at EOS).
    pub fn flush(&mut self, end_ns: u64) {
        self.snapshot(end_ns);
        self.disp_start = None;
    }

    /// Decode one `(cc_data_1, cc_data_2)` pair seen at running time `pts_ns`. The
    /// caller routes only this channel's field to it. Parity bits are stripped;
    /// control codes set the channel context and are interpreted for the selected
    /// channel, while printable bytes write to the active grid at the cursor.
    pub fn push_pair(&mut self, raw0: u8, raw1: u8, pts_ns: u64) {
        self.cur_pts = pts_ns;
        let b0 = raw0 & 0x7F;
        let b1 = raw1 & 0x7F;
        if b0 == 0 {
            // A null first byte is padding.
            return;
        }
        if (0x10..=0x1F).contains(&b0) {
            // Control codes are transmitted twice; act on the first, drop the
            // immediate identical repeat.
            if self.last_ctrl == Some((b0, b1)) {
                self.last_ctrl = None;
                return;
            }
            self.last_ctrl = Some((b0, b1));
            // The 0x08 bit of the base byte selects channel 2; normalize to the
            // channel-1 form for the command tables.
            self.cur_channel = if b0 <= 0x17 { 1 } else { 2 };
            let nb0 = if b0 >= 0x18 { b0 - 8 } else { b0 };
            if self.cur_channel == self.selected.channel() {
                self.handle_control(nb0, b1, pts_ns);
            }
        } else if b0 >= 0x20 && self.cur_channel == self.selected.channel() {
            self.last_ctrl = None;
            self.put_char(basic_char(b0));
            if b1 >= 0x20 {
                self.put_char(basic_char(b1));
            }
        } else {
            self.last_ctrl = None;
        }
    }

    /// Interpret a channel-normalized control pair: PAC (row + indent / style),
    /// mid-row style, special / extended characters, tab offset, and the
    /// misc-control commands.
    fn handle_control(&mut self, b0: u8, b1: u8, pts_ns: u64) {
        match b0 {
            0x11 => match b1 {
                0x20..=0x2F => self.mid_row(b1),
                0x30..=0x3F => self.put_char(special_char(b1)),
                0x40..=0x7F => self.pac(b0, b1),
                _ => {}
            },
            0x12 => match b1 {
                0x20..=0x3F => self.extended_char(extended_char_1(b1)),
                0x40..=0x7F => self.pac(b0, b1),
                _ => {}
            },
            0x13 => match b1 {
                0x20..=0x3F => self.extended_char(extended_char_2(b1)),
                0x40..=0x7F => self.pac(b0, b1),
                _ => {}
            },
            0x14 => match b1 {
                0x20..=0x2F => self.misc_control(b1, pts_ns),
                0x40..=0x7F => self.pac(b0, b1),
                _ => {}
            },
            0x17 => match b1 {
                0x21..=0x23 => self.col = (self.col + (b1 - 0x20) as usize).min(COLS),
                0x40..=0x7F => self.pac(b0, b1),
                _ => {}
            },
            0x10 | 0x15 | 0x16 if (0x40..=0x7F).contains(&b1) => self.pac(b0, b1),
            _ => {}
        }
    }

    /// A Preamble Address Code: set the write row, then the column / colour from the
    /// indent (`0x10` bit set) or style form of the second byte.
    fn pac(&mut self, b0: u8, b1: u8) {
        let Some(row) = pac_row(b0, b1) else { return };
        self.row = row as usize;
        if b1 & 0x10 != 0 {
            // Indent form: columns in groups of four.
            self.col = (((b1 & 0x0E) >> 1) as usize) * 4;
        } else {
            // Style form: the second byte selects the colour.
            self.col = 0;
            self.color = pac_color((b1 & 0x0E) >> 1);
        }
    }

    /// A mid-row style code sets the colour and occupies one cell (a space).
    fn mid_row(&mut self, b1: u8) {
        self.color = pac_color((b1 & 0x0E) >> 1);
        self.put_char(' ');
    }

    /// Misc-control command (`0x14`, second byte `0x20..=0x2F`).
    fn misc_control(&mut self, b1: u8, pts_ns: u64) {
        match b1 {
            0x20 => self.mode = Mode::PopOn,                  // RCL: resume caption loading
            0x21 => self.backspace(),                         // BS
            0x25..=0x27 => {
                // RU2/RU3/RU4: enter roll-up with a 2/3/4-row window at the base row.
                self.mode = Mode::RollUp;
                self.roll_rows = (b1 - 0x23) as usize;
                self.base_row = ROWS;
                self.row = ROWS;
                self.col = 0;
            }
            0x29 => self.mode = Mode::PaintOn,                // RDC: resume direct captioning
            0x2C => {
                // EDM: erase displayed memory.
                self.snapshot(pts_ns);
                self.disp.clear();
                self.disp_start = None;
            }
            0x2D => self.carriage_return(pts_ns),             // CR
            0x2E => self.back.clear(),                        // ENM: erase non-displayed memory
            0x2F => self.end_of_caption(pts_ns),             // EOC
            _ => {}
        }
    }

    /// EOC: end the on-screen caption and flip the loaded back buffer into view.
    fn end_of_caption(&mut self, now: u64) {
        self.snapshot(now);
        core::mem::swap(&mut self.disp, &mut self.back);
        self.back.clear();
        self.disp_start = (!self.disp.is_empty()).then_some(now);
    }

    /// CR (roll-up): emit the current window, scroll it up one row, and home the
    /// cursor to the base row.
    fn carriage_return(&mut self, now: u64) {
        if self.mode != Mode::RollUp {
            return;
        }
        self.snapshot(now);
        for r in 1..ROWS {
            self.disp.rows[r - 1] = self.disp.rows[r];
        }
        self.disp.rows[ROWS - 1] = [' '; COLS];
        self.col = 0;
        self.row = self.base_row;
        self.disp_start = (!self.disp.is_empty()).then_some(now);
    }

    /// Erase one cell to the left of the cursor.
    fn backspace(&mut self) {
        if self.col == 0 {
            return;
        }
        self.col -= 1;
        let r = self.row.clamp(1, ROWS) - 1;
        match self.mode {
            Mode::PopOn => self.back.rows[r][self.col] = ' ',
            _ => self.disp.rows[r][self.col] = ' ',
        }
    }

    /// An extended-character code overwrites the standard fallback glyph the encoder
    /// sent just before it, so step the cursor back one cell before writing.
    fn extended_char(&mut self, c: char) {
        if self.col > 0 {
            self.col -= 1;
        }
        self.put_char(c);
    }

    /// Write a glyph at the cursor of the active grid (back buffer in pop-on mode,
    /// displayed buffer otherwise) and advance the column.
    fn put_char(&mut self, c: char) {
        let r = self.row.clamp(1, ROWS) - 1;
        let col = self.col.min(COLS - 1);
        match self.mode {
            Mode::PopOn => self.back.rows[r][col] = c,
            _ => {
                if self.disp_start.is_none() {
                    self.disp_start = Some(self.cur_pts);
                }
                self.disp.rows[r][col] = c;
            }
        }
        self.col = (self.col + 1).min(COLS);
    }

    /// Push the current displayed content as a finished cue ending at `end_ns`, if
    /// it is non-empty and has a known start time. Does not mutate the grid.
    fn snapshot(&mut self, end_ns: u64) {
        let Some(start) = self.disp_start else { return };
        if end_ns <= start || self.disp.is_empty() {
            return;
        }
        self.out.push(Cue {
            start_ns: start,
            end_ns,
            text: self.disp.text(),
            settings: self.disp.settings(self.color),
        });
    }
}

/// Map a CEA-608 Preamble Address Code to its 1-based row (1..=15) for data
/// channel 1, or `None` if `(b0, b1)` is not a channel-1 PAC. The base byte
/// selects a row pair and the second byte's `0x20` bit picks the lower row; base
/// `0x10` addresses row 11 alone.
fn pac_row(b0: u8, b1: u8) -> Option<u8> {
    let base = match b0 {
        0x11 => 1,
        0x12 => 3,
        0x15 => 5,
        0x16 => 7,
        0x17 => 9,
        0x10 => return Some(11),
        0x13 => 12,
        0x14 => 14,
        _ => return None,
    };
    let second = if b1 & 0x20 != 0 { 1 } else { 0 };
    Some(base + second)
}

/// Map a CEA-608 colour index (PAC / mid-row style bits) to an opaque RGBA, or
/// `None` for white / italics (no override; the overlay uses its default).
fn pac_color(idx: u8) -> Option<[u8; 4]> {
    Some(match idx {
        1 => [0, 255, 0, 255],     // green
        2 => [0, 0, 255, 255],     // blue
        3 => [0, 255, 255, 255],   // cyan
        4 => [255, 0, 0, 255],     // red
        5 => [255, 255, 0, 255],   // yellow
        6 => [255, 0, 255, 255],   // magenta
        _ => return None,          // 0 = white, 7 = italics
    })
}

/// Map a CEA-608 basic North-American character byte (0x20..=0x7F, parity
/// stripped) to its Unicode glyph. Most are ASCII; a handful of code points carry
/// accented Latin letters and symbols per the standard.
fn basic_char(c: u8) -> char {
    match c {
        0x2A => 'á',
        0x5C => 'é',
        0x5E => 'í',
        0x5F => 'ó',
        0x60 => 'ú',
        0x7B => 'ç',
        0x7C => '÷',
        0x7D => 'Ñ',
        0x7E => 'ñ',
        0x7F => '█',
        // Printable ASCII passes through; anything else renders as a space.
        0x20..=0x7E => c as char,
        _ => ' ',
    }
}

/// Map a CEA-608 special North-American character (`0x11`, second byte
/// `0x30..=0x3F`) to its glyph.
fn special_char(b1: u8) -> char {
    match b1 {
        0x30 => '®',
        0x31 => '°',
        0x32 => '½',
        0x33 => '¿',
        0x34 => '™',
        0x35 => '¢',
        0x36 => '£',
        0x37 => '♪',
        0x38 => 'à',
        0x39 => ' ', // transparent space
        0x3A => 'è',
        0x3B => 'â',
        0x3C => 'ê',
        0x3D => 'î',
        0x3E => 'ô',
        0x3F => 'û',
        _ => ' ',
    }
}

/// Map a CEA-608 extended Western-European character, set 1 (`0x12`, Spanish /
/// French), second byte `0x20..=0x3F`, to its glyph.
fn extended_char_1(b1: u8) -> char {
    match b1 {
        0x20 => 'Á',
        0x21 => 'É',
        0x22 => 'Ó',
        0x23 => 'Ú',
        0x24 => 'Ü',
        0x25 => 'ü',
        0x26 => '´',
        0x27 => '¡',
        0x28 => '*',
        0x29 => '\'',
        0x2A => '—',
        0x2B => '©',
        0x2C => '℠',
        0x2D => '•',
        0x2E => '“',
        0x2F => '”',
        0x30 => 'À',
        0x31 => 'Â',
        0x32 => 'Ç',
        0x33 => 'È',
        0x34 => 'Ê',
        0x35 => 'Ë',
        0x36 => 'ë',
        0x37 => 'Î',
        0x38 => 'Ï',
        0x39 => 'ï',
        0x3A => 'Ô',
        0x3B => 'Ù',
        0x3C => 'ù',
        0x3D => 'Û',
        0x3E => '«',
        0x3F => '»',
        _ => ' ',
    }
}

/// Map a CEA-608 extended Western-European character, set 2 (`0x13`, Portuguese /
/// German / Danish), second byte `0x20..=0x3F`, to its glyph.
fn extended_char_2(b1: u8) -> char {
    match b1 {
        0x20 => 'Ã',
        0x21 => 'ã',
        0x22 => 'Í',
        0x23 => 'Ì',
        0x24 => 'ì',
        0x25 => 'Ò',
        0x26 => 'ò',
        0x27 => 'Õ',
        0x28 => 'õ',
        0x29 => '{',
        0x2A => '}',
        0x2B => '\\',
        0x2C => '^',
        0x2D => '_',
        0x2E => '|',
        0x2F => '~',
        0x30 => 'Ä',
        0x31 => 'ä',
        0x32 => 'Ö',
        0x33 => 'ö',
        0x34 => 'ß',
        0x35 => '¥',
        0x36 => '¤',
        0x37 => '│',
        0x38 => 'Å',
        0x39 => 'å',
        0x3A => 'Ø',
        0x3B => 'ø',
        0x3C => '┌',
        0x3D => '┐',
        0x3E => '└',
        0x3F => '┘',
        _ => ' ',
    }
}

// ---------------------------------------------------------------------------
// CEA-708 (DTVCC)
// ---------------------------------------------------------------------------

/// The maximum number of CEA-708 caption windows.
const NUM_WINDOWS: usize = 8;

/// One CEA-708 caption window: a defined region with its own anchor / size, pen
/// position, and a character grid. Text is written into the current window at the
/// pen; the window is shown / hidden by the DisplayWindows family of commands.
#[derive(Debug, Clone)]
struct Window {
    defined: bool,
    visible: bool,
    priority: u8,
    /// Vertical / horizontal placement as a percent (0..=100), resolved at define
    /// time from the anchor (absolute or relative).
    line_pct: u8,
    pos_pct: u8,
    pen_row: usize,
    pen_col: usize,
    /// `row_count` x `col_count` character cells; empty cells are spaces.
    grid: Vec<Vec<char>>,
}

impl Window {
    fn new() -> Self {
        Self {
            defined: false,
            visible: false,
            priority: 0,
            line_pct: 0,
            pos_pct: 0,
            pen_row: 0,
            pen_col: 0,
            grid: Vec::new(),
        }
    }

    fn is_empty(&self) -> bool {
        self.grid.iter().all(|r| r.iter().all(|&c| c == ' '))
    }

    /// The window's visible text, blank rows dropped and each row trimmed.
    fn text(&self) -> String {
        let mut out = String::new();
        for row in &self.grid {
            let line: String = row.iter().collect();
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(trimmed);
        }
        out
    }
}

/// A CEA-708 DTVCC caption decoder for one service (service 1 by default). Feed it
/// the `cc_type` 2 / 3 caption triples in order via [`Cea708::push_triple`]: it
/// reassembles the DTVCC packets, splits them into service blocks, runs the
/// chosen service's window / pen command stream, and emits a [`Cue`] each time the
/// displayed window set changes. The window command set (DefineWindow,
/// SetCurrentWindow, the DisplayWindows family, SetPenLocation, pen / window
/// attributes) drives positioning; G0 / G1 bytes are the text.
#[derive(Debug)]
pub struct Cea708 {
    /// The service number to render (1 = primary caption service).
    service: u8,
    /// The DTVCC packet being reassembled (header byte first).
    buf: Vec<u8>,
    /// PTS of the packet's start triple, the time its commands take effect.
    pts: u64,
    windows: [Window; NUM_WINDOWS],
    current_window: usize,
    /// The text currently on screen (joined visible windows) with its start time
    /// and placement, awaiting the change that ends it.
    shown: Option<(String, u64, CueSettings)>,
    out: Vec<Cue>,
}

impl Default for Cea708 {
    fn default() -> Self {
        Self::new()
    }
}

impl Cea708 {
    /// A fresh decoder rendering service 1.
    pub fn new() -> Self {
        Self::for_service(1)
    }

    /// A fresh decoder rendering `service` (1 = primary, 2 = secondary, ...).
    pub fn for_service(service: u8) -> Self {
        Self {
            service,
            buf: Vec::new(),
            pts: 0,
            windows: core::array::from_fn(|_| Window::new()),
            current_window: 0,
            shown: None,
            out: Vec::new(),
        }
    }

    /// Take the cues finished so far, leaving the decoder ready for more triples.
    pub fn take_cues(&mut self) -> Vec<Cue> {
        core::mem::take(&mut self.out)
    }

    /// Finalize any on-screen caption at running time `end_ns` (call at EOS).
    pub fn flush(&mut self, end_ns: u64) {
        if let Some((text, start, settings)) = self.shown.take() {
            if end_ns > start && !text.is_empty() {
                self.out.push(Cue { start_ns: start, end_ns, text, settings });
            }
        }
    }

    /// Feed one caption triple. `cc_type` 3 starts a new DTVCC packet and 2
    /// continues it; 0 / 1 (CEA-608) are ignored. A completed packet is decoded.
    pub fn push_triple(&mut self, cc_type: u8, b0: u8, b1: u8, pts_ns: u64) {
        match cc_type {
            3 => {
                // A new packet starts; abandon any incomplete one.
                self.buf.clear();
                self.pts = pts_ns;
                self.buf.push(b0);
                self.buf.push(b1);
            }
            2 => {
                if self.buf.is_empty() {
                    return;
                }
                self.buf.push(b0);
                self.buf.push(b1);
            }
            _ => return,
        }
        self.try_decode_packet();
    }

    /// Decode the reassembled packet once enough bytes have arrived. The header
    /// byte's `packet_size_code` gives the data length; the data is a sequence of
    /// service blocks.
    fn try_decode_packet(&mut self) {
        let Some(&header) = self.buf.first() else { return };
        let size_code = header & 0x3F;
        let data_size = if size_code == 0 { 127 } else { size_code as usize * 2 - 1 };
        let total = 1 + data_size;
        if self.buf.len() < total {
            return;
        }
        let now = self.pts;
        let data: Vec<u8> = self.buf[1..total].to_vec();
        self.buf.clear();
        self.decode_service_blocks(&data, now);
        self.update_display(now);
    }

    /// Split a packet's data into service blocks and run the selected service's.
    fn decode_service_blocks(&mut self, data: &[u8], now: u64) {
        let mut i = 0;
        while i < data.len() {
            let hdr = data[i];
            i += 1;
            let mut service = (hdr >> 5) & 0x07;
            let block_size = (hdr & 0x1F) as usize;
            if service == 0 {
                // NULL service block: end of the packet's blocks.
                break;
            }
            if service == 7 {
                // Extended service number in the next byte.
                let Some(&ext) = data.get(i) else { break };
                i += 1;
                service = ext & 0x3F;
            }
            let end = (i + block_size).min(data.len());
            if service == self.service {
                self.run_service(&data[i..end], now);
            }
            i = end;
        }
    }

    /// Run a service block's command stream against the window model.
    fn run_service(&mut self, block: &[u8], now: u64) {
        let mut i = 0;
        while i < block.len() {
            let c = block[i];
            match c {
                // C0 control codes (1 / 2 / 3 bytes by range).
                0x00..=0x1F => {
                    self.handle_c0(c);
                    i += match c {
                        0x00..=0x0F => 1,
                        0x10..=0x17 => 2,
                        _ => 3,
                    };
                }
                // G0: ASCII, with 0x7F the music note.
                0x20..=0x7F => {
                    self.put_char(if c == 0x7F { '♪' } else { c as char });
                    i += 1;
                }
                // C1 caption commands (length by code).
                0x80..=0x9F => {
                    let len = c1_len(c);
                    let params = &block[(i + 1).min(block.len())..(i + len).min(block.len())];
                    self.handle_c1(c, params, now);
                    i += len;
                }
                // G1: ISO 8859-1 Latin-1.
                0xA0..=0xFF => {
                    self.put_char(c as char);
                    i += 1;
                }
            }
        }
    }

    /// Handle a C0 control code (only the layout-affecting ones matter here).
    fn handle_c0(&mut self, c: u8) {
        let w = &mut self.windows[self.current_window];
        match c {
            0x08 => w.pen_col = w.pen_col.saturating_sub(1), // BS
            0x0C => {
                // FF: clear the window and home the pen.
                for row in &mut w.grid {
                    row.iter_mut().for_each(|cell| *cell = ' ');
                }
                w.pen_row = 0;
                w.pen_col = 0;
            }
            0x0D => {
                // CR: next row, first column.
                w.pen_row += 1;
                w.pen_col = 0;
            }
            0x0E => w.pen_col = 0, // HCR: home the pen on the current row
            _ => {}
        }
    }

    /// Handle a C1 caption command.
    fn handle_c1(&mut self, c: u8, params: &[u8], _now: u64) {
        match c {
            0x80..=0x87 => self.current_window = (c & 0x07) as usize, // CWx
            0x88 => self.for_each_window(params, |w| w.grid.iter_mut().for_each(|r| r.iter_mut().for_each(|c| *c = ' '))), // CLW
            0x89 => self.for_each_window(params, |w| w.visible = true),  // DSW
            0x8A => self.for_each_window(params, |w| w.visible = false), // HDW
            0x8B => self.for_each_window(params, |w| w.visible = !w.visible), // TGW
            0x8C => self.for_each_window(params, |w| {
                *w = Window::new();
            }), // DLW
            0x8F => self.windows = core::array::from_fn(|_| Window::new()), // RST
            0x92 => self.set_pen_location(params),                       // SPL
            0x98..=0x9F => self.define_window((c & 0x07) as usize, params), // DFx
            // CLW/DSW/... handled above; DLY/DLC/SPA/SPC/SWA carry no text effect here.
            _ => {}
        }
    }

    /// Apply `f` to each window whose bit is set in the 1-byte window bitmap.
    fn for_each_window(&mut self, params: &[u8], f: impl Fn(&mut Window)) {
        let Some(&bitmap) = params.first() else { return };
        for (i, w) in self.windows.iter_mut().enumerate() {
            if bitmap & (1 << i) != 0 {
                f(w);
            }
        }
    }

    /// SetPenLocation: move the current window's pen to `(row, column)`.
    fn set_pen_location(&mut self, params: &[u8]) {
        if params.len() < 2 {
            return;
        }
        let w = &mut self.windows[self.current_window];
        w.pen_row = (params[0] & 0x0F) as usize;
        w.pen_col = (params[1] & 0x3F) as usize;
    }

    /// DefineWindow: set the window's visibility, anchor (resolved to percent),
    /// size, priority, and allocate its grid; make it the current window.
    fn define_window(&mut self, id: usize, p: &[u8]) {
        if p.len() < 6 {
            return;
        }
        let visible = p[0] & 0x20 != 0;
        let priority = p[0] & 0x07;
        let relative = p[1] & 0x80 != 0;
        let anchor_v = p[1] & 0x7F;
        let anchor_h = p[2];
        let row_count = ((p[3] & 0x0F) as usize) + 1;
        let col_count = ((p[4] & 0x3F) as usize) + 1;
        // Resolve the anchor to a percent of the safe area (absolute ranges are
        // 0..=74 vertical, 0..=209 horizontal; relative anchors are already 0..=99).
        let line_pct = if relative {
            anchor_v.min(100)
        } else {
            ((anchor_v as u32 * 100) / 74).min(100) as u8
        };
        let pos_pct = if relative {
            anchor_h.min(100)
        } else {
            ((anchor_h as u32 * 100) / 209).min(100) as u8
        };
        self.windows[id] = Window {
            defined: true,
            visible,
            priority,
            line_pct,
            pos_pct,
            pen_row: 0,
            pen_col: 0,
            grid: alloc::vec![alloc::vec![' '; col_count]; row_count],
        };
        self.current_window = id;
    }

    /// Write a glyph into the current window at the pen and advance the column.
    fn put_char(&mut self, ch: char) {
        let w = &mut self.windows[self.current_window];
        if !w.defined {
            return;
        }
        if let Some(row) = w.grid.get_mut(w.pen_row) {
            if let Some(cell) = row.get_mut(w.pen_col) {
                *cell = ch;
            }
            w.pen_col += 1;
        }
    }

    /// Recompute the displayed text from the visible windows and, if it changed,
    /// finalize the previous caption (ending now) and start the new one.
    fn update_display(&mut self, now: u64) {
        let new = self.visible_text();
        let changed = new.as_ref().map(|(t, _)| t) != self.shown.as_ref().map(|(t, _, _)| t);
        if !changed {
            return;
        }
        if let Some((text, start, settings)) = self.shown.take() {
            if now > start && !text.is_empty() {
                self.out.push(Cue { start_ns: start, end_ns: now, text, settings });
            }
        }
        self.shown = new.map(|(text, settings)| (text, now, settings));
    }

    /// Join the text of the visible, non-empty windows (top-to-bottom by anchor,
    /// then priority) and derive placement from the topmost window.
    fn visible_text(&self) -> Option<(String, CueSettings)> {
        let mut wins: Vec<&Window> = self
            .windows
            .iter()
            .filter(|w| w.defined && w.visible && !w.is_empty())
            .collect();
        if wins.is_empty() {
            return None;
        }
        wins.sort_by_key(|w| (w.line_pct, w.priority));
        let mut text = String::new();
        for w in &wins {
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(&w.text());
        }
        let top = wins[0];
        let settings = CueSettings {
            line: Some(top.line_pct),
            position: Some(top.pos_pct),
            align: TextAlign::Start,
            ..CueSettings::default()
        };
        Some((text, settings))
    }
}

/// Total length in bytes (command + parameters) of a CEA-708 C1 caption command.
fn c1_len(code: u8) -> usize {
    match code {
        0x80..=0x87 => 1, // CW0-CW7
        0x88..=0x8D => 2, // CLW / DSW / HDW / TGW / DLW / DLY (1 param)
        0x8E | 0x8F => 1, // DLC / RST
        0x90 => 3,        // SPA (2 params)
        0x91 => 4,        // SPC (3 params)
        0x92 => 3,        // SPL (2 params)
        0x93..=0x96 => 1, // reserved
        0x97 => 5,        // SWA (4 params)
        0x98..=0x9F => 7, // DF0-DF7 (6 params)
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Odd-parity-encode a CEA-608 byte (bit 7 makes the total bit count odd).
    fn parity(mut b: u8) -> u8 {
        let ones = b.count_ones();
        if ones % 2 == 0 {
            b |= 0x80;
        }
        b
    }

    /// Build an H.264 Annex-B SEI NAL carrying a GA94 `cc_data` block from the
    /// given `(cc_type, b0, b1)` triples.
    fn h264_cc_sei(triples: &[(u8, u8, u8)]) -> Vec<u8> {
        // user_data_registered_itu_t_t35 payload.
        let mut payload = vec![0xB5u8]; // country: USA
        payload.extend_from_slice(&[0x00, 0x31]); // provider_code (ATSC)
        payload.extend_from_slice(&[0x47, 0x41, 0x39, 0x34]); // user_identifier GA94
        payload.push(0x03); // user_data_type_code: cc_data
        // flags: process_cc_data_flag (0x40) | reserved (0x80) | cc_count.
        payload.push(0xC0 | (triples.len() as u8 & 0x1F));
        payload.push(0xFF); // em_data
        for &(t, b0, b1) in triples {
            payload.push(0xF8 | (t & 0x03) | 0x04); // marker | cc_valid | cc_type
            payload.push(b0);
            payload.push(b1);
        }
        payload.push(0xFF); // marker_bits trailer

        // SEI message: payloadType=4, payloadSize, payload.
        let mut sei = vec![0x04u8];
        sei.push(payload.len() as u8);
        sei.extend_from_slice(&payload);
        sei.push(0x80); // rbsp_trailing_bits

        // NAL: start code + header (type 6) + SEI RBSP.
        let mut nal = vec![0x00, 0x00, 0x00, 0x01, 0x06];
        nal.extend_from_slice(&sei);
        nal
    }

    #[test]
    fn extracts_valid_triples_from_h264_sei() {
        let au = h264_cc_sei(&[(0, 0x12, 0x34), (0, 0x56, 0x78)]);
        let triples = extract_cc_data(&au, VideoCodec::H264);
        assert_eq!(
            triples,
            vec![
                CcTriple { cc_type: 0, b0: 0x12, b1: 0x34 },
                CcTriple { cc_type: 0, b0: 0x56, b1: 0x78 },
            ]
        );
    }

    #[test]
    fn ignores_non_ga94_user_data() {
        // A user-data SEI with a different identifier yields nothing.
        let mut au = vec![0x00, 0x00, 0x00, 0x01, 0x06, 0x04, 0x08];
        au.extend_from_slice(&[0xB5, 0x00, 0x31, b'D', b'T', b'G', b'1', 0x80]);
        assert!(extract_cc_data(&au, VideoCodec::H264).is_empty());
    }

    #[test]
    fn truncated_cc_block_does_not_panic() {
        // cc_count claims 4 triples but the payload ends early.
        let mut au = h264_cc_sei(&[(0, 0x41, 0x42)]);
        au.truncate(au.len() - 2);
        let _ = extract_cc_data(&au, VideoCodec::H264);
    }

    #[test]
    fn decodes_a_pop_on_caption() {
        let mut dec = Cea608::new();
        // RCL (load), PAC row 15, "HI", EOC (display at t=1000), then EDM at 5000.
        dec.push_pair(parity(0x14), parity(0x20), 1000); // RCL
        dec.push_pair(parity(0x14), parity(0x70), 1000); // PAC -> row 15
        dec.push_pair(parity(b'H'), parity(b'I'), 1000); // "HI"
        dec.push_pair(parity(0x14), parity(0x2F), 1000); // EOC: show now
        dec.push_pair(parity(0x14), parity(0x2C), 5000); // EDM: erase now
        let cues = dec.take_cues();
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text, "HI");
        assert_eq!(cues[0].start_ns, 1000);
        assert_eq!(cues[0].end_ns, 5000);
    }

    #[test]
    fn doubled_control_code_acts_once() {
        let mut dec = Cea608::new();
        // EOC sent twice (doubled transmission) must flip only once.
        dec.push_pair(parity(b'A'), parity(b'B'), 100);
        // Without RCL we are in default pop-on; load into back buffer first.
        dec.push_pair(parity(0x14), parity(0x2F), 200); // EOC
        dec.push_pair(parity(0x14), parity(0x2F), 200); // doubled EOC -> ignored
        dec.push_pair(parity(0x14), parity(0x2C), 400); // EDM
        let cues = dec.take_cues();
        // One caption ("AB"), shown at 200, erased at 400.
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text, "AB");
        assert_eq!((cues[0].start_ns, cues[0].end_ns), (200, 400));
    }

    #[test]
    fn basic_charset_substitutions() {
        assert_eq!(basic_char(0x2A), 'á');
        assert_eq!(basic_char(0x7E), 'ñ');
        assert_eq!(basic_char(b'A'), 'A');
    }

    #[test]
    fn pac_row_mapping() {
        assert_eq!(pac_row(0x11, 0x40), Some(1));
        assert_eq!(pac_row(0x11, 0x60), Some(2));
        assert_eq!(pac_row(0x14, 0x40), Some(14));
        assert_eq!(pac_row(0x14, 0x60), Some(15));
        assert_eq!(pac_row(0x10, 0x40), Some(11));
        assert_eq!(pac_row(0x13, 0x60), Some(13));
    }

    #[test]
    fn decodes_roll_up_scroll() {
        let mut dec = Cea608::new();
        dec.push_pair(parity(0x14), parity(0x25), 0); // RU2: roll-up, 2 rows
        dec.push_pair(parity(b'A'), parity(b'B'), 100); // type "AB" on the base row
        dec.push_pair(parity(0x14), parity(0x2D), 200); // CR: emit + scroll
        dec.push_pair(parity(b'C'), parity(b'D'), 300); // type "CD" on the new base row
        dec.flush(500);
        let cues = dec.take_cues();
        // First the lone base row, then the scrolled two-row window.
        assert_eq!(cues.len(), 2);
        assert_eq!(cues[0].text, "AB");
        assert_eq!((cues[0].start_ns, cues[0].end_ns), (100, 200));
        assert_eq!(cues[1].text, "AB\nCD");
        assert_eq!((cues[1].start_ns, cues[1].end_ns), (200, 500));
    }

    #[test]
    fn selects_only_the_requested_channel() {
        let mut dec = Cea608::for_channel(Cea608Channel::Cc2);
        // Channel-1 control + text (base byte 0x14) must be ignored by a CC2 decoder.
        dec.push_pair(parity(0x14), parity(0x20), 0); // RCL on channel 1
        dec.push_pair(parity(b'Z'), parity(b'Z'), 0); // channel-1 text
        // Channel-2 control + text (base byte 0x1C = 0x14 | 0x08) is rendered.
        dec.push_pair(parity(0x1C), parity(0x20), 0); // RCL on channel 2
        dec.push_pair(parity(b'X'), parity(b'Y'), 0); // channel-2 text
        dec.push_pair(parity(0x1C), parity(0x2F), 100); // EOC channel 2
        dec.push_pair(parity(0x1C), parity(0x2C), 300); // EDM channel 2
        let cues = dec.take_cues();
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text, "XY");
    }

    #[test]
    fn decodes_special_and_extended_characters() {
        let mut dec = Cea608::new();
        dec.push_pair(parity(0x14), parity(0x20), 0); // RCL
        dec.push_pair(parity(0x11), parity(0x37), 0); // special char: music note
        dec.push_pair(parity(b'E'), 0, 0); // fallback glyph for the extended char
        dec.push_pair(parity(0x12), parity(0x33), 0); // extended set 1 0x33 -> 'È' (overwrites)
        dec.push_pair(parity(0x14), parity(0x2F), 100); // EOC
        dec.push_pair(parity(0x14), parity(0x2C), 300); // EDM
        let cues = dec.take_cues();
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text, "♪È");
    }

    #[test]
    fn pac_indent_drives_position() {
        let mut dec = Cea608::new();
        dec.push_pair(parity(0x14), parity(0x20), 0); // RCL
        // PAC row 3, indent form, column group 2 (= 8 columns): 0x40|0x10|0x04.
        dec.push_pair(parity(0x12), parity(0x54), 0);
        dec.push_pair(parity(b'H'), parity(b'I'), 0);
        dec.push_pair(parity(0x14), parity(0x2F), 100); // EOC
        dec.flush(500);
        let cues = dec.take_cues();
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text, "HI");
        // Row 3 of 15 -> ~14% down; 8 of 32 columns -> 25% across; left-set.
        assert_eq!(cues[0].settings.line, Some(14));
        assert_eq!(cues[0].settings.position, Some(25));
        assert_eq!(cues[0].settings.align, TextAlign::Start);
    }

    #[test]
    fn extracts_field_two_triples_for_h265() {
        // An H.265 prefix-SEI (NAL type 39, 2-byte header) carrying CC3/CC4 data.
        let au = {
            let base = h264_cc_sei(&[(1, 0x20, 0x21)]);
            // Replace the H.264 NAL header (1 byte, type 6) with an H.265 prefix-SEI
            // header (2 bytes: type 39 in bits 1..=6 of the first byte).
            let mut v = vec![0x00, 0x00, 0x00, 0x01, 39 << 1, 0x01];
            v.extend_from_slice(&base[5..]);
            v
        };
        let triples = extract_cc_data(&au, VideoCodec::H265);
        assert_eq!(triples, vec![CcTriple { cc_type: 1, b0: 0x20, b1: 0x21 }]);
    }

    /// Wrap service-block command bytes for `service` in a DTVCC service-block
    /// header.
    fn service_block(service: u8, data: &[u8]) -> Vec<u8> {
        let mut v = vec![((service & 0x07) << 5) | (data.len() as u8 & 0x1F)];
        v.extend_from_slice(data);
        v
    }

    /// Wrap concatenated service blocks in a DTVCC packet (header byte +
    /// odd-length-padded data) ready to split into caption triples.
    fn dtvcc_packet(blocks: &[u8]) -> Vec<u8> {
        let mut data = blocks.to_vec();
        if data.len() % 2 == 0 {
            data.push(0x00); // pad so data_size is odd (= size_code * 2 - 1)
        }
        let size_code = ((data.len() + 1) / 2) as u8;
        let mut pkt = vec![size_code & 0x3F]; // sequence 0
        pkt.extend_from_slice(&data);
        pkt
    }

    /// Feed a DTVCC packet to the decoder as caption triples (first pair `cc_type`
    /// 3, the rest `cc_type` 2), all stamped `pts`.
    fn feed(dec: &mut Cea708, pkt: &[u8], pts: u64) {
        let mut i = 0;
        let mut first = true;
        while i < pkt.len() {
            let b0 = pkt[i];
            let b1 = if i + 1 < pkt.len() { pkt[i + 1] } else { 0 };
            dec.push_triple(if first { 3 } else { 2 }, b0, b1, pts);
            first = false;
            i += 2;
        }
    }

    /// DefineWindow params: hidden, absolute anchor `(v, h)`, `rows` x `cols`.
    fn define_params(visible: bool, v: u8, h: u8, rows: u8, cols: u8) -> [u8; 6] {
        [
            if visible { 0x20 } else { 0x00 },
            v & 0x7F,
            h,
            (rows - 1) & 0x0F,
            (cols - 1) & 0x3F,
            0x00,
        ]
    }

    #[test]
    fn decodes_a_708_window_caption() {
        let mut dec = Cea708::new();
        // Packet 1: DefineWindow 0 (hidden, anchor v=72), write "HI", DisplayWindows.
        let p = define_params(false, 72, 0, 3, 32);
        let mut cmds = vec![0x98]; // DF0
        cmds.extend_from_slice(&p);
        cmds.extend_from_slice(b"HI");
        cmds.extend_from_slice(&[0x89, 0x01]); // DSW window 0
        feed(&mut dec, &dtvcc_packet(&service_block(1, &cmds)), 1000);
        // Packet 2: HideWindows 0 -> ends the caption.
        feed(&mut dec, &dtvcc_packet(&service_block(1, &[0x8A, 0x01])), 5000);
        let cues = dec.take_cues();
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text, "HI");
        assert_eq!((cues[0].start_ns, cues[0].end_ns), (1000, 5000));
        // Anchor v=72 of 74 -> ~97% down the safe area.
        assert_eq!(cues[0].settings.line, Some(97));
    }

    #[test]
    fn set_pen_location_lays_out_rows() {
        let mut dec = Cea708::new();
        let p = define_params(true, 60, 0, 2, 32);
        let mut cmds = vec![0x98]; // DF0 (visible)
        cmds.extend_from_slice(&p);
        cmds.extend_from_slice(b"AB");
        cmds.extend_from_slice(&[0x92, 0x01, 0x00]); // SPL row 1, col 0
        cmds.extend_from_slice(b"CD");
        feed(&mut dec, &dtvcc_packet(&service_block(1, &cmds)), 100);
        dec.flush(900);
        let cues = dec.take_cues();
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text, "AB\nCD");
    }

    #[test]
    fn ignores_other_services() {
        // A service-1 decoder must not render a service-2 block.
        let mut dec = Cea708::new();
        let p = define_params(true, 70, 0, 1, 32);
        let mut cmds = vec![0x98];
        cmds.extend_from_slice(&p);
        cmds.extend_from_slice(b"NO");
        feed(&mut dec, &dtvcc_packet(&service_block(2, &cmds)), 100);
        dec.flush(900);
        assert!(dec.take_cues().is_empty());
    }

    #[test]
    fn reassembles_packet_across_triples() {
        // A packet split over several cc_type-2 continuation triples decodes whole.
        let mut dec = Cea708::new();
        let p = define_params(true, 70, 0, 1, 32);
        let mut cmds = vec![0x98];
        cmds.extend_from_slice(&p);
        cmds.extend_from_slice(b"LONGER CAPTION TEXT");
        let pkt = dtvcc_packet(&service_block(1, &cmds));
        assert!(pkt.len() > 6, "packet should span multiple triples");
        feed(&mut dec, &pkt, 100);
        dec.flush(900);
        let cues = dec.take_cues();
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text, "LONGER CAPTION TEXT");
    }
}
