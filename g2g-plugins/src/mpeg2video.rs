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

use crate::annexb::BitReader;
use crate::startcodeparse::reduce_ratio;

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
    /// One sample's `(width, height)`, reduced. `Caps` has no field for it, so
    /// it reaches a pipeline through `mpegvideoparse`'s read-only
    /// `pixel-aspect-ratio` property. `None` when `aspect_ratio_information` is
    /// forbidden or reserved.
    pub pixel_aspect: Option<(u32, u32)>,
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

/// MPEG-1 `pel_aspect_ratio` (ISO/IEC 11172-2 Table 2-16), one pel's height
/// divided by its width, in ten-thousandths (the precision the table itself is
/// written to). Code 0 is forbidden and 15 reserved, both left at 0.
const MPEG1_PEL_ASPECT_TEN_THOUSANDTHS: [u32; 16] = [
    0, 10000, 6735, 7031, 7615, 8055, 8437, 8935, 9157, 9815, 10255, 10695, 10950, 11575, 12015, 0,
];

/// The denominator [`MPEG1_PEL_ASPECT_TEN_THOUSANDTHS`] is expressed against.
const PEL_ASPECT_SCALE: u32 = 10000;

/// MPEG-2 `aspect_ratio_information` (ISO/IEC 13818-2 Table 6-3) as the coded
/// frame's display aspect ratio. Code 1 means square samples and is handled
/// without the table; 0 and 5..=15 are forbidden or reserved.
const MPEG2_DISPLAY_ASPECT_BY_CODE: [(u32, u32); 16] = [
    (0, 0),
    (0, 0),
    (4, 3),
    (16, 9),
    (221, 100),
    (0, 0),
    (0, 0),
    (0, 0),
    (0, 0),
    (0, 0),
    (0, 0),
    (0, 0),
    (0, 0),
    (0, 0),
    (0, 0),
    (0, 0),
];

/// `aspect_ratio_information` for square samples, which needs no table.
const ASPECT_RATIO_SQUARE: u32 = 1;
/// Bytes of fixed-length fields at the head of a sequence header, before the
/// optional quantiser matrices.
const SEQUENCE_HEADER_FIXED_BYTES: usize = 8;
/// `extension_start_code`.
const EXTENSION_START_CODE: u8 = 0xB5;
/// `extension_start_code_identifier` of the sequence extension.
const SEQUENCE_EXTENSION_ID: u32 = 1;

/// Parse the first MPEG sequence header (`00 00 01 B3`) in an access unit: the
/// 12-bit horizontal and vertical sizes, the 4-bit aspect_ratio_information and
/// the 4-bit frame_rate_code that follow it, plus the sequence extension after
/// it (which widens the sizes, scales the frame rate, and carries
/// `progressive_sequence`). Returns `None` when the unit carries no sequence
/// header, when it is truncated, or when a field is out of range (a zero
/// dimension, a reserved frame rate), so a malformed header leaves the
/// placeholder caps standing rather than fixating on nonsense.
pub fn parse_sequence_header(au: &[u8]) -> Option<SequenceHeader> {
    // The start code prefix is unique in an MPEG video bitstream, so a plain
    // scan for it needs no emulation-prevention handling.
    let found = au.windows(4).position(|w| w == [0x00, 0x00, 0x01, 0xB3])?;
    let body = found.checked_add(4)?;
    let h = au.get(body..body.checked_add(4)?)?;
    let horizontal_size_value = ((h[0] as u32) << 4) | ((h[1] as u32) >> 4);
    let vertical_size_value = (((h[1] & 0x0F) as u32) << 8) | h[2] as u32;
    let aspect_ratio_information = (h[3] >> 4) as u32;
    let (num, den) = *FRAME_RATES.get((h[3] & 0x0F) as usize)?;

    let extension = parse_sequence_extension(au, body);
    let width = horizontal_size_value | extension.map_or(0, |e| e.horizontal_size_extension << 12);
    let height = vertical_size_value | extension.map_or(0, |e| e.vertical_size_extension << 12);
    if width == 0 || height == 0 || num == 0 {
        return None;
    }
    // The extension's scale terms are at most 4 and 32, and num at most 60000,
    // so the product stays well inside u64 and the quotient inside u32.
    let (scale_n, scale_d) = extension.map_or((1, 1), |e| {
        (
            e.frame_rate_extension_n.saturating_add(1),
            e.frame_rate_extension_d.saturating_add(1),
        )
    });
    let framerate_q16 =
        ((((num as u64) << 16) * scale_n as u64) / ((den as u64) * scale_d as u64)) as u32;
    // MPEG-1's table gives the pel's own aspect; MPEG-2's gives the frame's
    // display aspect, from which the sample's follows.
    let pixel_aspect = match extension {
        Some(_) => mpeg2_pixel_aspect(aspect_ratio_information, width, height),
        None => mpeg1_pixel_aspect(aspect_ratio_information),
    };
    Some(SequenceHeader {
        width,
        height,
        framerate_q16,
        progressive: extension.is_none_or(|e| e.progressive),
        pixel_aspect,
    })
}

/// One sample's `(width, height)` for an MPEG-1 `aspect_ratio_information`: the
/// reciprocal of the table's pel aspect ratio, which is coded height over width.
fn mpeg1_pixel_aspect(aspect_ratio_information: u32) -> Option<(u32, u32)> {
    let pel = *MPEG1_PEL_ASPECT_TEN_THOUSANDTHS.get(aspect_ratio_information as usize)?;
    reduce_ratio(PEL_ASPECT_SCALE, pel)
}

/// One sample's `(width, height)` for an MPEG-2 `aspect_ratio_information`,
/// whose table gives the frame's display aspect ratio: the sample aspect is the
/// display aspect divided by the coded aspect.
fn mpeg2_pixel_aspect(
    aspect_ratio_information: u32,
    width: u32,
    height: u32,
) -> Option<(u32, u32)> {
    if aspect_ratio_information == ASPECT_RATIO_SQUARE {
        return Some((1, 1));
    }
    let (display_n, display_d) =
        *MPEG2_DISPLAY_ASPECT_BY_CODE.get(aspect_ratio_information as usize)?;
    // The table terms are under 256 and the sizes under 2^14, so the products
    // stay far inside u32; saturate anyway rather than trust the header.
    reduce_ratio(
        display_n.saturating_mul(height),
        display_d.saturating_mul(width),
    )
}

/// The fields of the MPEG-2 sequence extension (ISO 13818-2 6.2.2.3) that reach
/// the caps.
#[derive(Clone, Copy, Debug)]
struct SequenceExtension {
    progressive: bool,
    horizontal_size_extension: u32,
    vertical_size_extension: u32,
    frame_rate_extension_n: u32,
    frame_rate_extension_d: u32,
}

/// The sequence extension that must directly follow the sequence header whose
/// body starts at `body`. `None` for MPEG-1 (no extension), a truncation before
/// one, or a different extension.
///
/// The header's own length is variable (either quantiser matrix may be present,
/// and the second one is not byte aligned), so the extension is found by scanning
/// for the next start code past the fixed part instead of computing an offset.
/// No start code can be emulated inside the header: its marker bit forbids the
/// pattern in the fixed part, and quantiser matrix values are never zero.
///
/// A field the extension is cut short of keeps a default that changes nothing:
/// no size or frame-rate scaling, and progressive, so an unreadable stream plays
/// unfiltered.
fn parse_sequence_extension(au: &[u8], body: usize) -> Option<SequenceExtension> {
    let from = body.checked_add(SEQUENCE_HEADER_FIXED_BYTES)?;
    let rest = au.get(from..)?;
    let at = rest.windows(4).position(|w| w[..3] == START_CODE_PREFIX)?;
    if rest[at + 3] != EXTENSION_START_CODE {
        return None;
    }
    let mut bits = BitReader::new(rest.get(at + 4..)?);
    if bits.read_bits(4)? != SEQUENCE_EXTENSION_ID {
        return None;
    }
    let mut extension = SequenceExtension {
        progressive: true,
        horizontal_size_extension: 0,
        vertical_size_extension: 0,
        frame_rate_extension_n: 0,
        frame_rate_extension_d: 0,
    };
    let mut read = || -> Option<()> {
        bits.skip_bits(8)?; // profile_and_level_indication
        extension.progressive = bits.read_bit()? == 1;
        bits.skip_bits(2)?; // chroma_format
        extension.horizontal_size_extension = bits.read_bits(2)?;
        extension.vertical_size_extension = bits.read_bits(2)?;
        bits.skip_bits(12)?; // bit_rate_extension
        bits.skip_bits(1)?; // marker_bit
        bits.skip_bits(8)?; // vbv_buffer_size_extension
        bits.skip_bits(1)?; // low_delay
        extension.frame_rate_extension_n = bits.read_bits(2)?;
        extension.frame_rate_extension_d = bits.read_bits(5)?;
        Some(())
    };
    let _ = read();
    Some(extension)
}

/// The start-code prefix every MPEG header is introduced by.
const START_CODE_PREFIX: [u8; 3] = [0x00, 0x00, 0x01];
/// `user_data_start_code` (ISO 13818-2 6.2.2.2), the block ATSC A/53 carries
/// closed captions in.
const USER_DATA_START_CODE: u8 = 0xB2;

/// Call `f` with the payload of every `user_data` block (`00 00 01 B2`) in one
/// MPEG-1 / MPEG-2 access unit, in bitstream order. The block carries no length:
/// its payload ends at the next start code, or at the end of the unit. The
/// standard forbids the start-code pattern inside user data, so a payload that
/// emulates one is cut there rather than read past.
pub(crate) fn for_each_user_data(au: &[u8], mut f: impl FnMut(&[u8])) {
    let mut i = 0usize;
    while i + 4 <= au.len() {
        if au[i..i + 3] != START_CODE_PREFIX {
            i += 1;
            continue;
        }
        if au[i + 3] != USER_DATA_START_CODE {
            i += 4;
            continue;
        }
        let start = i + 4;
        let end = au[start..]
            .windows(3)
            .position(|w| w == START_CODE_PREFIX)
            .map_or(au.len(), |at| start + at);
        f(&au[start..end]);
        // end >= start > i, so the scan always advances.
        i = end;
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
