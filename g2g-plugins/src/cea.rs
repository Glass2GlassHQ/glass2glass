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
use crate::subparse::{Cue, CueSettings};

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

/// Caption presentation mode. M426 decodes pop-on; roll-up and paint-on are
/// recognised enough to switch mode but their scrolling is layered on in M427.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    PopOn,
    RollUp,
    PaintOn,
}

/// A caption currently on screen: its rows, the running time it appeared, and its
/// placement. Held until the next display flips or an erase clears it, at which
/// point it is finalized into a [`Cue`] with the end time.
#[derive(Debug, Clone)]
struct Displayed {
    rows: [String; ROWS],
    start_ns: u64,
    settings: CueSettings,
}

/// A CEA-608 line-21 caption decoder (M426: field 1, channel CC1, pop-on mode,
/// basic North-American character set). Feed it the `(cc_data_1, cc_data_2)` byte
/// pairs of `cc_type` 0 in presentation order via [`Cea608::push_pair`]; finished
/// cues accumulate and are drained with [`Cea608::take_cues`].
#[derive(Debug)]
pub struct Cea608 {
    mode: Mode,
    /// The non-displayed (back) buffer pop-on captions are loaded into.
    back: [String; ROWS],
    /// The caption currently on screen, awaiting an end time.
    front: Option<Displayed>,
    /// Last control pair, to drop the immediate doubled retransmission.
    last_ctrl: Option<(u8, u8)>,
    /// Current write row (1..=15), set by a PAC.
    row: usize,
    out: Vec<Cue>,
}

impl Default for Cea608 {
    fn default() -> Self {
        Self::new()
    }
}

impl Cea608 {
    /// A fresh decoder in pop-on mode with empty buffers.
    pub fn new() -> Self {
        Self {
            mode: Mode::PopOn,
            back: core::array::from_fn(|_| String::new()),
            front: None,
            last_ctrl: None,
            row: 15,
            out: Vec::new(),
        }
    }

    /// Take the cues finished so far, leaving the decoder ready for more pairs.
    pub fn take_cues(&mut self) -> Vec<Cue> {
        core::mem::take(&mut self.out)
    }

    /// Finalize any on-screen caption at running time `end_ns` (call at EOS).
    pub fn flush(&mut self, end_ns: u64) {
        self.finalize_front(end_ns);
    }

    /// Decode one field-1 `(cc_data_1, cc_data_2)` pair seen at running time
    /// `pts_ns`. Parity bits are stripped; control codes are interpreted and
    /// printable bytes written to the back buffer at the current row.
    pub fn push_pair(&mut self, raw0: u8, raw1: u8, pts_ns: u64) {
        let b0 = raw0 & 0x7F;
        let b1 = raw1 & 0x7F;
        // A null pair (both 0 after parity strip) is padding.
        if b0 == 0 {
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
            self.handle_control(b0, b1, pts_ns);
        } else {
            self.last_ctrl = None;
            self.write_char(b0);
            if b1 >= 0x20 {
                self.write_char(b1);
            }
        }
    }

    /// Interpret a control code pair. Recognises channel-1 PACs (row select) and
    /// the misc-control commands that drive pop-on display (RCL / EOC / EDM /
    /// ENM); mode-switch commands set the mode for later milestones.
    fn handle_control(&mut self, b0: u8, b1: u8, pts_ns: u64) {
        // PAC: base byte in the channel-1 set with the second byte in 0x40..=0x7F.
        if (0x40..=0x7F).contains(&b1) {
            if let Some(row) = pac_row(b0, b1) {
                self.row = row as usize;
            }
            return;
        }
        // Misc control: channel-1 base byte 0x14 (or its 0x15 alias) with the
        // second byte in 0x20..=0x2F.
        if matches!(b0, 0x14 | 0x15) && (0x20..=0x2F).contains(&b1) {
            match b1 {
                0x20 => self.mode = Mode::PopOn,                       // RCL
                0x25..=0x27 => self.mode = Mode::RollUp,               // RU2/RU3/RU4
                0x29 => self.mode = Mode::PaintOn,                     // RDC
                0x2C => self.finalize_front(pts_ns),                  // EDM (erase displayed)
                0x2E => self.clear_back(),                            // ENM (erase non-displayed)
                0x2F => self.flip(pts_ns),                            // EOC (end of caption)
                _ => {}
            }
        }
    }

    /// End-of-caption: show the loaded back buffer. The caption already on screen
    /// ends now; the back buffer becomes the new displayed caption starting now.
    fn flip(&mut self, pts_ns: u64) {
        self.finalize_front(pts_ns);
        let rows = core::mem::replace(&mut self.back, core::array::from_fn(|_| String::new()));
        if rows.iter().any(|r| !r.is_empty()) {
            self.front = Some(Displayed { rows, start_ns: pts_ns, settings: CueSettings::default() });
        }
    }

    /// Finalize the on-screen caption into a cue ending at `end_ns`, if any.
    fn finalize_front(&mut self, end_ns: u64) {
        let Some(front) = self.front.take() else { return };
        if end_ns <= front.start_ns {
            return;
        }
        let text = join_rows(&front.rows);
        if text.is_empty() {
            return;
        }
        self.out.push(Cue {
            start_ns: front.start_ns,
            end_ns,
            text,
            settings: front.settings,
        });
    }

    fn clear_back(&mut self) {
        for row in &mut self.back {
            row.clear();
        }
    }

    /// Append a basic-character-set glyph to the current back-buffer row.
    fn write_char(&mut self, c: u8) {
        let row = self.row.clamp(1, ROWS);
        self.back[row - 1].push(basic_char(c));
    }
}

/// Join the non-empty rows of a caption grid top-to-bottom with newlines, trimming
/// trailing spaces from each row.
fn join_rows(rows: &[String; ROWS]) -> String {
    let mut out = String::new();
    for row in rows {
        let trimmed = row.trim_end();
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
}
