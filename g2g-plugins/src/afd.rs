// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! Active Format Description and bar data from SDI and coded video.

#[cfg(feature = "metadata")]
use alloc::{boxed::Box, vec::Vec};

#[cfg(feature = "metadata")]
use g2g_core::meta::{CropRectangle, FrameMeta, Propagation, Transform};
#[cfg(feature = "metadata")]
use g2g_core::{Interlace, PixelAspect};

use crate::st291::{word_with_parity, AncPacket};

/// The SMPTE ST 2016-3 ancillary packet identifiers.
pub const ST2016_DID: u8 = 0x41;
pub const ST2016_SDID: u8 = 0x05;

/// The last sample of the leading bar and first sample of the trailing bar.
/// Vertical positions use broadcast raster line numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BarData {
    TopBottom { top_end: u16, bottom_start: u16 },
    LeftRight { left_end: u16, right_start: u16 },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AfdBar {
    /// VANC line carrying this entry, when received over SDI.
    pub line_number: Option<u16>,
    /// Four-bit active-format code, when signalled.
    pub active_format: Option<u8>,
    /// ST 2016 coded-frame aspect flag. True is 16:9 and false is 4:3.
    pub widescreen: Option<bool>,
    pub bars: Option<BarData>,
}

impl AfdBar {
    /// Decode an ST 2016-3 ANC packet. Bad bar data leaves the AFD available.
    pub fn from_st2016_anc(packet: &AncPacket) -> Option<Self> {
        if (packet.did, packet.sdid) != (ST2016_DID, ST2016_SDID) {
            return None;
        }
        let words: &[u16; 8] = packet.user_data.as_slice().try_into().ok()?;
        if words
            .iter()
            .any(|&word| word != word_with_parity(word as u8))
        {
            return None;
        }
        let [afd, reserved_1, reserved_2, flags, bar @ ..] = words.map(|word| word as u8);
        if afd & 0x83 != 0 || reserved_1 != 0 || reserved_2 != 0 {
            return None;
        }
        Some(Self {
            line_number: Some(packet.line_number),
            active_format: Some((afd >> 3) & 0x0f),
            widescreen: Some(afd & 0x04 != 0),
            bars: parse_bars(flags, 0, bar),
        })
    }

    /// Decode a registered T.35 payload from an AVC or HEVC SEI message.
    /// AFD (`DTG1`) and bars (`GA94` type 6) arrive in separate messages.
    pub fn from_t35(payload: &[u8]) -> Option<Self> {
        let user_data = payload.strip_prefix(&crate::sei::ATSC_T35_HEADER)?;
        if let Some([0x41, code, ..]) = user_data.strip_prefix(b"DTG1") {
            if code & 0xf0 == 0xf0 {
                return Some(Self {
                    active_format: Some(code & 0x0f),
                    ..Self::default()
                });
            }
        }
        let [6, flags, a, b, c, d, ..] =
            user_data.strip_prefix(&crate::cea::USER_IDENTIFIER_GA94)?
        else {
            return None;
        };
        Some(Self {
            bars: Some(parse_bars(*flags, 0x0f, [*a, *b, *c, *d])?),
            ..Self::default()
        })
    }
}

fn parse_bars(flags: u8, reserved: u8, values: [u8; 4]) -> Option<BarData> {
    // ST 2016 uses zero for the reserved bits and T.35 uses ones.
    const MARKER: u16 = 0xc000;
    let [a, b, c, d] = values;
    let first = u16::from_be_bytes([a, b]);
    let second = u16::from_be_bytes([c, d]);
    if flags & 0x0f != reserved || first & MARKER != MARKER || second & MARKER != MARKER {
        return None;
    }
    let (first, second) = (first & !MARKER, second & !MARKER);
    match flags & 0xf0 {
        0xc0 => Some(BarData::TopBottom {
            top_end: first,
            bottom_start: second,
        }),
        0x30 => Some(BarData::LeftRight {
            left_end: first,
            right_start: second,
        }),
        _ => None,
    }
}

/// Active-picture information for one video frame. SDI can carry one entry
/// per field, while the video parsers combine both kinds of SEI into one entry.
#[cfg(feature = "metadata")]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AfdBarMeta {
    pub entries: Vec<AfdBar>,
}

#[cfg(feature = "metadata")]
impl FrameMeta for AfdBarMeta {
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn core::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn FrameMeta> {
        Box::new(self.clone())
    }

    fn propagate(&self, transform: Transform) -> Propagation {
        match transform {
            Transform::Copy => Propagation::Keep,
            _ => Propagation::Drop,
        }
    }
}

#[cfg(feature = "metadata")]
impl AfdBarMeta {
    /// Resolve the signalled active picture against the uncropped raster.
    pub fn active_rectangle(
        &self,
        width: u32,
        height: u32,
        pixel_aspect: PixelAspect,
        interlace: Interlace,
    ) -> Option<CropRectangle> {
        let entry = self.entries.iter().min_by_key(|entry| entry.line_number)?;
        if entry.active_format.is_none_or(|code| matches!(code, 0 | 4)) {
            return bar_rectangle(entry.bars?, width, height, interlace);
        }
        // The protect area in codes 13 through 15 is a framing hint. Use the
        // full active aspect for an active-picture crop.
        let (numerator, denominator, top_aligned) = match entry.active_format? {
            2 => (16, 9, true),
            3 => (14, 9, true),
            9 | 13 => (4, 3, false),
            10 | 14 | 15 => (16, 9, false),
            11 => (14, 9, false),
            _ => return None,
        };
        let (pn, pd) = crate::display_aspect::pixel_aspect_fraction(pixel_aspect);
        let (width, height) = (u128::from(width), u128::from(height));
        let round = |n: u128, d| (n + d / 2) / d;
        let w = round(height * numerator * pd, denominator * pn).min(width);
        let h = round(width * denominator * pn, numerator * pd).min(height);
        Some(CropRectangle {
            left: ((width - w) / 2) as u32,
            top: if top_aligned {
                0
            } else {
                ((height - h) / 2) as u32
            },
            width: w as u32,
            height: h as u32,
        })
    }
}

#[cfg(feature = "metadata")]
fn bar_rectangle(
    bars: BarData,
    width: u32,
    height: u32,
    interlace: Interlace,
) -> Option<CropRectangle> {
    let mut rect = CropRectangle {
        left: 0,
        top: 0,
        width,
        height,
    };
    match bars {
        BarData::LeftRight {
            left_end,
            right_start,
        } => {
            rect.left = u32::from(left_end) + 1;
            let end = u32::from(right_start);
            rect.width = end.checked_sub(rect.left)?;
            (rect.width > 0 && end <= width).then_some(rect)
        }
        BarData::TopBottom {
            top_end,
            bottom_start,
        } => {
            rect.top = u32::try_from(bar_row(top_end, width, height, interlace)? + 1).ok()?;
            let end = u32::try_from(bar_row(bottom_start, width, height, interlace)?).ok()?;
            rect.height = end.checked_sub(rect.top)?;
            (rect.height > 0 && end <= height).then_some(rect)
        }
    }
}

#[cfg(feature = "metadata")]
fn bar_row(line: u16, width: u32, height: u32, interlace: Interlace) -> Option<i32> {
    // SMPTE ST 2016-1 Table 2 numbers the fields separately. A line
    // immediately outside the coded raster can signal a zero-height bar.
    let line = i32::from(line);
    let (first, second) = match (width, height, interlace.is_interleaved()) {
        (704 | 720, 480 | 576, false) => (45, None),
        (1280, 720, false) => (26, None),
        (1920, 1080, false) => (42, None),
        (704 | 720, 480, true) => (23, Some(286)),
        (704 | 720, 576, true) => (23, Some(336)),
        (1920, 1080, true) => (21, Some(584)),
        _ => return None,
    };
    match second {
        Some(second) if line >= second - 1 => {
            let row = (line - second) * 2 + 1;
            (row < height as i32).then_some(row)
        }
        Some(_) => {
            let row = (line - first) * 2;
            (0..=height as i32).contains(&row).then_some(row)
        }
        None => Some(line - first),
    }
}
