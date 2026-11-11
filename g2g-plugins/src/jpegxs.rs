// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::vec::Vec;

use g2g_core::caps::framerate_q16_from_fraction;
use g2g_core::{
    BitstreamAlignment, Caps, ColorPrimaries, ColorRange, Colorimetry, Dim, G2gError,
    MatrixCoefficients, PixelAspect, Rate, TransferCharacteristics, VideoCodec,
};

const SOC_MARKER: u16 = 0xFF10;
const CAP_MARKER: u16 = 0xFF50;
const PIH_MARKER: u16 = 0xFF12;
const CDT_MARKER: u16 = 0xFF13;
const SLH_MARKER: u16 = 0xFF20;
const PIH_LENGTH: usize = 26;

// ISO/IEC 13818-1 extension_descriptor, and the extension tag of its JXS_video_descriptor
const EXTENSION_DESCRIPTOR_TAG: u8 = 0x3F;
const JXS_VIDEO_EXTENSION_TAG: u8 = 0x14;
const JXS_VIDEO_DESCRIPTOR_VERSION: u8 = 0;
const JXS_VIDEO_DESCRIPTOR_BODY_LEN: usize = 29;

const JXES_BOX_LEN: usize = 30;
const JXES_BOX_TYPE: &[u8; 4] = b"jxes";

// frat: interlace mode in bits 30-31, framerate denominator code in bits 24-29, numerator in the low 16
const FRAT_TOP_FIELD_FIRST: u32 = 1 << 30;
const FRAT_INTERLACE_SHIFT: u32 = 30;
const FRAT_DENOMINATOR_SHIFT: u32 = 24;
const FRAT_DENOMINATOR_MASK: u32 = 0x3F;
const FRAT_NUMERATOR_MASK: u32 = 0xFFFF;
const FRAT_WHOLE_RATE: u32 = 1;
const FRAT_NTSC_RATE: u32 = 2;
const NTSC_NUMERATOR_SCALE: u32 = 1000;
const NTSC_DENOMINATOR: u32 = 1001;

// schar: valid flag in bit 15, bit depth minus one in bits 4-7, sampling in the low 4 bits
const SCHAR_VALID: u16 = 1 << 15;
const SCHAR_DEPTH_SHIFT: u16 = 4;
const SCHAR_NIBBLE: u16 = 0xF;
const SAMPLING_422: u16 = 0;
const SAMPLING_444: u16 = 1;

// gst's mpegtsmux values: H.222.0 buffer model 2 and a buffer of brat / 160
const BUFFER_MODEL_TYPE: u8 = 2;
const MAX_BUFFER_SIZE_BRAT_DIVISOR: u32 = 160;
const BITS_PER_MEGABIT_IN_BYTES: u128 = 125_000;
const FULL_RANGE_FLAG: u8 = 0x80;
const UNSPECIFIED_CICP: u8 = 2;

const Q16_SHIFT: u32 = 16;
const Q16_FRACTION_MASK: u32 = (1 << Q16_SHIFT) - 1;

const NANOS_PER_SECOND: u64 = 1_000_000_000;
const SECONDS_PER_MINUTE: u64 = 60;
const MINUTES_PER_HOUR: u64 = 60;
const HOURS_PER_DAY: u64 = 24;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CodestreamHeader {
    ppih: u16,
    plev: u16,
    width: u16,
    height: u16,
    schar: u16,
}

fn be16(data: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_be_bytes(data.get(at..at + 2)?.try_into().ok()?))
}

fn be32(data: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes(data.get(at..at + 4)?.try_into().ok()?))
}

fn read_codestream_header(codestream: &[u8]) -> Option<CodestreamHeader> {
    if be16(codestream, 0)? != SOC_MARKER || be16(codestream, 2)? != CAP_MARKER {
        return None;
    }
    let pih_at = 4 + usize::from(be16(codestream, 4)?);
    if be16(codestream, pih_at)? != PIH_MARKER
        || usize::from(be16(codestream, pih_at + 2)?) != PIH_LENGTH
    {
        return None;
    }
    // Lpih, then Lcod, Ppih, Plev, Wf, Hf, Cw, Hsl and Nc
    let ppih = be16(codestream, pih_at + 8)?;
    let plev = be16(codestream, pih_at + 10)?;
    let width = be16(codestream, pih_at + 12)?;
    let height = be16(codestream, pih_at + 14)?;
    let components = usize::from(*codestream.get(pih_at + 20)?);
    let mut at = pih_at + 2 + PIH_LENGTH;
    loop {
        let marker = be16(codestream, at)?;
        if marker == SLH_MARKER {
            return None;
        }
        let length = usize::from(be16(codestream, at + 2)?);
        if marker != CDT_MARKER {
            at = at.checked_add(2 + length)?;
            continue;
        }
        if length != 2 + 2 * components {
            return None;
        }
        let table = codestream.get(at + 4..at + 2 + length)?;
        let depth = u16::from(table[0]);
        let subsampling = |component: usize| table.get(2 * component + 1).copied();
        // 0x21 is Sx 2 by Sy 1, 0x11 is no subsampling, both on the two chroma components
        let sampling = match (components, subsampling(1)?, subsampling(2)?) {
            (3, 0x21, 0x21) => SAMPLING_422,
            (3, 0x11, 0x11) => SAMPLING_444,
            _ => return None,
        };
        let depth_bits = (depth.checked_sub(1)? & SCHAR_NIBBLE) << SCHAR_DEPTH_SHIFT;
        return Some(CodestreamHeader {
            ppih,
            plev,
            width,
            height,
            schar: SCHAR_VALID | depth_bits | sampling,
        });
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JxsVideoDescriptor {
    pub horizontal_size: u16,
    pub vertical_size: u16,
    pub brat: u32,
    pub frat: u32,
    pub schar: u16,
    pub ppih: u16,
    pub plev: u16,
    pub max_buffer_size: u32,
    pub colour_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
    pub full_range: bool,
}

fn frat_for(framerate: &Rate, fields_per_frame: u32) -> Option<u32> {
    let interlace = if fields_per_frame == 2 {
        FRAT_TOP_FIELD_FIRST
    } else {
        0
    };
    let (denominator_code, numerator) = match framerate {
        Rate::Fixed(q16) if q16 & Q16_FRACTION_MASK == 0 => (FRAT_WHOLE_RATE, q16 >> Q16_SHIFT),
        Rate::Fixed(q16) => {
            let whole = (q16 >> Q16_SHIFT) + 1;
            let ntsc = framerate_q16_from_fraction(whole * NTSC_NUMERATOR_SCALE, NTSC_DENOMINATOR);
            if ntsc != Some(*q16) {
                return None;
            }
            (FRAT_NTSC_RATE, whole)
        }
        // gst's mpegtsmux writes an unknown rate as 0/1
        Rate::Any | Rate::Range { .. } => (FRAT_WHOLE_RATE, 0),
    };
    if numerator > FRAT_NUMERATOR_MASK {
        return None;
    }
    Some(interlace | (denominator_code << FRAT_DENOMINATOR_SHIFT) | numerator)
}

impl JxsVideoDescriptor {
    pub(crate) fn for_frame(caps: &Caps, frame: &[u8]) -> Result<Self, G2gError> {
        let Caps::CompressedVideo {
            codec: VideoCodec::JpegXs,
            framerate,
            colorimetry,
            alignment,
            ..
        } = caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        let fields_per_frame = if *alignment == BitstreamAlignment::FieldPair {
            2
        } else {
            1
        };
        let header = read_codestream_header(frame).ok_or(G2gError::CapsMismatch)?;
        let frat = frat_for(framerate, fields_per_frame).ok_or(G2gError::CapsMismatch)?;
        let brat = match framerate {
            Rate::Fixed(q16) => {
                let megabits = (u128::from(*q16) * frame.len() as u128)
                    .div_ceil((1u128 << Q16_SHIFT) * BITS_PER_MEGABIT_IN_BYTES);
                u32::try_from(megabits).unwrap_or(u32::MAX)
            }
            Rate::Any | Rate::Range { .. } => u32::MAX,
        };
        Ok(Self {
            horizontal_size: header.width,
            vertical_size: header.height,
            brat,
            frat,
            schar: header.schar,
            ppih: header.ppih,
            plev: header.plev,
            max_buffer_size: brat / MAX_BUFFER_SIZE_BRAT_DIVISOR,
            colour_primaries: colorimetry.primaries.to_cicp(),
            transfer_characteristics: colorimetry.transfer.to_cicp(),
            matrix_coefficients: colorimetry.matrix.to_cicp(),
            full_range: colorimetry.range == ColorRange::Full,
        })
    }

    // the payload of an ES-info descriptor whose tag is the extension tag
    pub(crate) fn parse(payload: &[u8]) -> Option<Self> {
        let (&extension_tag, body) = payload.split_first()?;
        if extension_tag != JXS_VIDEO_EXTENSION_TAG
            || body.len() < JXS_VIDEO_DESCRIPTOR_BODY_LEN
            || body[0] != JXS_VIDEO_DESCRIPTOR_VERSION
        {
            return None;
        }
        Some(Self {
            horizontal_size: be16(body, 1)?,
            vertical_size: be16(body, 3)?,
            brat: be32(body, 5)?,
            frat: be32(body, 9)?,
            schar: be16(body, 13)?,
            ppih: be16(body, 15)?,
            plev: be16(body, 17)?,
            max_buffer_size: be32(body, 19)?,
            colour_primaries: body[24],
            transfer_characteristics: body[25],
            matrix_coefficients: body[26],
            full_range: body[27] & FULL_RANGE_FLAG != 0,
        })
    }

    pub(crate) fn is_extension_tag(tag: u8) -> bool {
        tag == EXTENSION_DESCRIPTOR_TAG
    }

    pub(crate) fn to_descriptor(self) -> Vec<u8> {
        let mut out = Vec::with_capacity(3 + JXS_VIDEO_DESCRIPTOR_BODY_LEN);
        out.push(EXTENSION_DESCRIPTOR_TAG);
        out.push(1 + JXS_VIDEO_DESCRIPTOR_BODY_LEN as u8);
        out.push(JXS_VIDEO_EXTENSION_TAG);
        out.push(JXS_VIDEO_DESCRIPTOR_VERSION);
        out.extend_from_slice(&self.horizontal_size.to_be_bytes());
        out.extend_from_slice(&self.vertical_size.to_be_bytes());
        out.extend_from_slice(&self.brat.to_be_bytes());
        out.extend_from_slice(&self.frat.to_be_bytes());
        out.extend_from_slice(&self.schar.to_be_bytes());
        out.extend_from_slice(&self.ppih.to_be_bytes());
        out.extend_from_slice(&self.plev.to_be_bytes());
        out.extend_from_slice(&self.max_buffer_size.to_be_bytes());
        out.push(BUFFER_MODEL_TYPE);
        out.push(self.colour_primaries);
        out.push(self.transfer_characteristics);
        out.push(self.matrix_coefficients);
        out.push(if self.full_range { FULL_RANGE_FLAG } else { 0 });
        // still_mode and mdm flags, both off
        out.push(0);
        out
    }

    fn fields_per_frame(&self) -> Option<u32> {
        match self.frat >> FRAT_INTERLACE_SHIFT {
            0 => Some(1),
            1 | 2 => Some(2),
            _ => None,
        }
    }

    fn framerate(&self) -> Option<Rate> {
        let numerator = self.frat & FRAT_NUMERATOR_MASK;
        if self.frat == 0 || numerator == 0 {
            return Some(Rate::Any);
        }
        let q16 = match (self.frat >> FRAT_DENOMINATOR_SHIFT) & FRAT_DENOMINATOR_MASK {
            FRAT_WHOLE_RATE => framerate_q16_from_fraction(numerator, 1),
            FRAT_NTSC_RATE => {
                framerate_q16_from_fraction(numerator * NTSC_NUMERATOR_SCALE, NTSC_DENOMINATOR)
            }
            _ => None,
        }?;
        Some(Rate::Fixed(q16))
    }

    pub(crate) fn caps(&self) -> Option<Caps> {
        let fields_per_frame = self.fields_per_frame()?;
        let codes = [
            self.colour_primaries,
            self.transfer_characteristics,
            self.matrix_coefficients,
        ];
        let range = match (self.full_range, codes == [UNSPECIFIED_CICP; 3]) {
            (true, _) => ColorRange::Full,
            (false, true) => ColorRange::Unknown,
            (false, false) => ColorRange::Limited,
        };
        Some(Caps::CompressedVideo {
            codec: VideoCodec::JpegXs,
            width: Dim::Fixed(u32::from(self.horizontal_size)),
            height: Dim::Fixed(u32::from(self.vertical_size) * fields_per_frame),
            framerate: self.framerate()?,
            colorimetry: Colorimetry {
                range,
                matrix: MatrixCoefficients::from_cicp(self.matrix_coefficients),
                transfer: TransferCharacteristics::from_cicp(self.transfer_characteristics),
                primaries: ColorPrimaries::from_cicp(self.colour_primaries),
            },
            pixel_aspect: PixelAspect::Any,
            alignment: if fields_per_frame == 2 {
                BitstreamAlignment::FieldPair
            } else {
                BitstreamAlignment::Unspecified
            },
        })
    }

    // the elementary stream header box every JPEG XS PES opens with, its timecode taken from the pts
    pub(crate) fn jxes_box(&self, pts_ns: u64) -> [u8; JXES_BOX_LEN] {
        let seconds = pts_ns / NANOS_PER_SECOND;
        let minutes = seconds / SECONDS_PER_MINUTE;
        let hours = minutes / MINUTES_PER_HOUR;
        let mut out = [0u8; JXES_BOX_LEN];
        out[0..4].copy_from_slice(&(JXES_BOX_LEN as u32).to_be_bytes());
        out[4..8].copy_from_slice(JXES_BOX_TYPE);
        out[8..12].copy_from_slice(&self.brat.to_be_bytes());
        out[12..16].copy_from_slice(&self.frat.to_be_bytes());
        out[16..18].copy_from_slice(&self.schar.to_be_bytes());
        out[18..20].copy_from_slice(&self.ppih.to_be_bytes());
        out[20..22].copy_from_slice(&self.plev.to_be_bytes());
        out[22] = self.colour_primaries;
        out[23] = self.transfer_characteristics;
        out[24] = self.matrix_coefficients;
        out[25] = if self.full_range { FULL_RANGE_FLAG } else { 0 };
        out[26] = (hours % HOURS_PER_DAY) as u8;
        out[27] = (minutes % MINUTES_PER_HOUR) as u8;
        out[28] = (seconds % SECONDS_PER_MINUTE) as u8;
        out
    }
}

pub(crate) fn strip_jxes_box(pes_payload: &[u8]) -> Option<&[u8]> {
    let length = be32(pes_payload, 0)? as usize;
    if length != JXES_BOX_LEN || pes_payload.get(4..8)? != JXES_BOX_TYPE {
        return None;
    }
    pes_payload.get(JXES_BOX_LEN..)
}

#[cfg(test)]
mod tests {
    use super::*;

    // tools/jpegxs-fixtures.sh: SvtJpegxsEncApp, 96x48 8-bit 4:2:2, three frames
    const FIXTURE: &[u8] = include_bytes!("../tests/fixtures/jpegxs_96x48_i422_3frames.jxs");
    const WIDTH: u16 = 96;
    const HEIGHT: u16 = 48;
    const DEPTH: u16 = 8;

    #[test]
    fn the_codestream_header_gives_size_depth_and_sampling() {
        let header = read_codestream_header(FIXTURE).expect("an SVT codestream");
        assert_eq!((header.width, header.height), (WIDTH, HEIGHT));
        assert_eq!(
            header.schar,
            SCHAR_VALID | ((DEPTH - 1) << SCHAR_DEPTH_SHIFT) | SAMPLING_422
        );
        assert_eq!(read_codestream_header(&FIXTURE[..40]), None, "cut short");
    }

    #[test]
    fn frat_holds_whole_and_ntsc_rates_and_the_field_order() {
        let ntsc = framerate_q16_from_fraction(30_000, 1_001).unwrap();
        assert_eq!(frat_for(&Rate::Fixed(25 << 16), 1), Some(0x0100_0019));
        assert_eq!(frat_for(&Rate::Fixed(ntsc), 1), Some(0x0200_001E));
        assert_eq!(frat_for(&Rate::Fixed(25 << 16), 2), Some(0x4100_0019));
        assert_eq!(frat_for(&Rate::Any, 1), Some(0x0100_0000));
        assert_eq!(frat_for(&Rate::Fixed(23 << 16 | 1 << 15), 1), None);
    }

    #[test]
    fn the_descriptor_parses_back_to_the_caps_it_was_built_from() {
        let caps = Caps::CompressedVideo {
            codec: VideoCodec::JpegXs,
            width: Dim::Fixed(u32::from(WIDTH)),
            height: Dim::Fixed(u32::from(HEIGHT)),
            framerate: Rate::Fixed(25 << 16),
            colorimetry: Colorimetry::UNKNOWN,
            pixel_aspect: PixelAspect::Any,
            alignment: BitstreamAlignment::Unspecified,
        };
        let descriptor = JxsVideoDescriptor::for_frame(&caps, FIXTURE).unwrap();
        let written = descriptor.to_descriptor();
        assert!(JxsVideoDescriptor::is_extension_tag(written[0]));
        assert_eq!(usize::from(written[1]), written.len() - 2);
        let parsed = JxsVideoDescriptor::parse(&written[2..]).unwrap();
        assert_eq!(parsed, descriptor);
        assert_eq!(parsed.caps(), Some(caps));

        let jxes = descriptor.jxes_box(0);
        let pes = [&jxes[..], FIXTURE].concat();
        assert_eq!(strip_jxes_box(&pes), Some(FIXTURE));
        assert_eq!(strip_jxes_box(FIXTURE), None);
    }
}
