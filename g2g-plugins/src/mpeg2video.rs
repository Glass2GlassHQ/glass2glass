//! MPEG-1 / MPEG-2 video bitstream headers and the timestamp synthesis they
//! drive, shared by the program-stream and transport-stream demuxers.
//!
//! Both containers stamp MPEG video sparsely (a DVD roughly once per GOP, a
//! transport stream at least every 700 ms), so most pictures arrive with no
//! timestamp of their own. The picture header carries the one field that fixes
//! that: `temporal_reference`, the picture's display index within its GOP.
//!
//! Every offset here comes off the wire, so the scans are bounds-checked and a
//! truncated or malformed header yields `None` rather than panicking.

/// The video geometry an MPEG sequence header declares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceHeader {
    pub width: u32,
    pub height: u32,
    /// Frame rate as a Q16 fixed-point value, matching [`Rate::Fixed`].
    pub framerate_q16: u32,
    /// `progressive_sequence` from the MPEG-2 sequence extension: false means the
    /// stream is interlaced and wants deinterlacing before display. True for
    /// MPEG-1 (which has no such signalling) and whenever the extension is
    /// missing or malformed, so an unreadable stream plays untouched rather than
    /// being filtered on a guess.
    pub progressive: bool,
}

/// The MPEG frame_rate_code table (ISO 13818-2 Table 6-4), as exact
/// numerator / denominator pairs. Index 0 and 9..=15 are forbidden / reserved.
const FRAME_RATES: [(u32, u32); 9] = [
    (0, 1),
    (24000, 1001),
    (24, 1),
    (25, 1),
    (30000, 1001),
    (30, 1),
    (50, 1),
    (60000, 1001),
    (60, 1),
];

/// Parse the first MPEG sequence header (`00 00 01 B3`) in an access unit: the
/// 12-bit horizontal and vertical sizes and the 4-bit frame_rate_code that
/// follow it, plus `progressive_sequence` from the MPEG-2 sequence extension
/// after it. Returns `None` when the unit carries no sequence header, when it
/// is truncated, or when a field is out of range (a zero dimension, a reserved
/// frame rate), so a malformed header leaves the placeholder caps standing
/// rather than fixating on nonsense.
pub fn parse_sequence_header(au: &[u8]) -> Option<SequenceHeader> {
    // The start code prefix is unique in an MPEG video bitstream, so a plain
    // scan for it needs no emulation-prevention handling.
    let found = au.windows(4).position(|w| w == [0x00, 0x00, 0x01, 0xB3])?;
    let body = found.checked_add(4)?;
    let h = au.get(body..body.checked_add(4)?)?;
    let width = ((h[0] as u32) << 4) | ((h[1] as u32) >> 4);
    let height = (((h[1] & 0x0F) as u32) << 8) | h[2] as u32;
    let (num, den) = *FRAME_RATES.get((h[3] & 0x0F) as usize)?;
    if width == 0 || height == 0 || num == 0 {
        return None;
    }
    // num is at most 60000, so the shift stays well inside u64 and the quotient
    // inside u32 (60 << 16 at the top of the table).
    let framerate_q16 = (((num as u64) << 16) / den as u64) as u32;
    Some(SequenceHeader {
        width,
        height,
        framerate_q16,
        progressive: parse_progressive_sequence(au, body),
    })
}

/// `progressive_sequence` from the MPEG-2 sequence extension (ISO 13818-2
/// 6.2.2.3), the extension that must directly follow a sequence header. `body` is
/// the first byte after the sequence header's start code.
///
/// The header's own length is variable (either quantiser matrix may be present,
/// and the second one is not byte aligned), so the extension is found by scanning
/// for the next start code past the 8-byte fixed part instead of computing an
/// offset. No start code can be emulated inside the header: its marker bit
/// forbids the pattern in the fixed part, and quantiser matrix values are never
/// zero. Anything unexpected (truncation, a different extension, MPEG-1's absent
/// one) reads as progressive, so the stream plays unfiltered.
fn parse_progressive_sequence(au: &[u8], body: usize) -> bool {
    let Some(from) = body.checked_add(8) else {
        return true;
    };
    let Some(rest) = au.get(from..) else {
        return true;
    };
    let Some(at) = rest.windows(4).position(|w| w[..3] == [0x00, 0x00, 0x01]) else {
        return true;
    };
    if rest[at + 3] != 0xB5 {
        return true;
    }
    // extension_start_code_identifier (4 bits) then profile_and_level_indication
    // (8 bits), so progressive_sequence is bit 3 of the second payload byte.
    let Some(id) = rest.get(at + 4) else {
        return true;
    };
    if id >> 4 != 0b0001 {
        return true;
    }
    match rest.get(at + 5) {
        Some(b) => b & 0x08 != 0,
        None => true,
    }
}

/// The picture header's `temporal_reference` (the picture's display index
/// within its GOP, ISO 13818-2 6.3.9) and whether a GOP header precedes it in
/// this access unit. Headers come before slice data in an access unit, so the
/// scan ends at the first picture start code; `None` for a unit with no whole
/// picture header (truncation).
pub(crate) fn picture_temporal_reference(au: &[u8]) -> Option<(bool, u16)> {
    let mut has_gop = false;
    let mut i = 0;
    while i + 4 <= au.len() {
        if au[i..i + 3] == [0x00, 0x00, 0x01] {
            match au[i + 3] {
                0xB8 => has_gop = true,
                0x00 => {
                    let hi = u16::from(*au.get(i + 4)?);
                    let lo = u16::from(*au.get(i + 5)?);
                    return Some((has_gop, (hi << 2) | (lo >> 6)));
                }
                _ => {}
            }
            i += 4;
        } else {
            i += 1;
        }
    }
    None
}

/// Per-frame video timestamp synthesis between container stamps (M934). Most
/// pictures arrive unstamped; left as duplicates of the last stamp, a pacing
/// sink plays each GOP as a burst then a freeze. The picture header's `temporal_reference` is the picture's display
/// index within its GOP, so with the sequence header's frame period every
/// picture has an exact display time: `pts = base + temporal_reference *
/// period`. A real PES PTS re-anchors `base` exactly (self-correcting, drift
/// never outlives a GOP); an unstamped GOP header advances it by the span of
/// the GOP just closed. Display-order indexing keeps B-frame reordering exact,
/// and degenerates to last-stamp-plus-period for I/P-only streams.
#[derive(Debug, Default)]
pub(crate) struct Mpeg2TimestampSynth {
    /// Display-time base (90 kHz) of the current GOP.
    base_90: Option<u64>,
    /// Pictures seen in the current GOP so far (max temporal_reference + 1).
    span: u64,
    /// Last decode timestamp (90 kHz), synthesized or real: DTS advances one
    /// frame period per picture in coded order.
    dts_90: Option<u64>,
}

impl Mpeg2TimestampSynth {
    /// Stamp one video access unit in place, updating the anchor state.
    /// `period_90` is the frame period in 90 kHz units from the sequence header
    /// in effect. `au` must hold exactly one coded picture.
    pub(crate) fn stamp(
        &mut self,
        au: &[u8],
        pts_90khz: &mut Option<u64>,
        dts_90khz: &mut Option<u64>,
        period_90: u64,
    ) {
        let Some((has_gop, tref)) = picture_temporal_reference(au) else {
            return;
        };
        let tref = u64::from(tref);
        if has_gop {
            // A new GOP with no stamp of its own: its base is the previous
            // GOP's base advanced past every picture that GOP displayed.
            if pts_90khz.is_none() {
                if let Some(b) = self.base_90 {
                    self.base_90 = Some(b + self.span.max(1) * period_90);
                }
            }
            self.span = 0;
        }
        match *pts_90khz {
            Some(pts) => self.base_90 = Some(pts.saturating_sub(tref * period_90)),
            None => {
                if let Some(b) = self.base_90 {
                    *pts_90khz = Some(b + tref * period_90);
                }
            }
        }
        self.span = self.span.max(tref + 1);
        match *dts_90khz {
            Some(d) => self.dts_90 = Some(d),
            None => {
                if let Some(prev) = self.dts_90 {
                    let d = prev + period_90;
                    *dts_90khz = Some(d);
                    self.dts_90 = Some(d);
                }
            }
        }
    }
}
