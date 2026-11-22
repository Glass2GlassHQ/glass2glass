// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::vec::Vec;

use g2g_core::{
    BayerFormat, BayerPattern, BayerSampleFormat, Caps, Dim, G2gError, LosslessCompression,
    RawVideoFormat,
};

use crate::mp4box::{be32, find_entry_child, full_box, mp4_box};
use crate::uncompressedvideo::RawVideoStorage;

// ISO/IEC 23001-17 component types
const MONOCHROME: u16 = 0;
const LUMA: u16 = 1;
const CB: u16 = 2;
const CR: u16 = 3;
const RED: u16 = 4;
const GREEN: u16 = 5;
const BLUE: u16 = 6;
const ALPHA: u16 = 7;
const FILTER_ARRAY: u16 = 11;

const SAMPLING_NONE: u8 = 0;
const SAMPLING_422: u8 = 1;
const SAMPLING_420: u8 = 2;

const INTERLEAVE_COMPONENT: u8 = 0;
const INTERLEAVE_PIXEL: u8 = 1;
const INTERLEAVE_MIXED: u8 = 2;
const INTERLEAVE_ROW: u8 = 3;
const INTERLEAVE_MULTI_Y: u8 = 5;

const COMPONENTS_LITTLE_ENDIAN: u8 = 0x80;

const NO_PROFILE: [u8; 4] = [0; 4];

#[derive(Debug)]
struct UncvLayout {
    format: RawVideoFormat,
    profile: [u8; 4],
    components: &'static [u16],
    bit_depth: u8,
    sampling: u8,
    interleave: u8,
    block_size: u8,
    pixel_size: u32,
}

const fn layout(
    format: RawVideoFormat,
    profile: [u8; 4],
    components: &'static [u16],
    sampling: u8,
    interleave: u8,
    pixel_size: u32,
) -> UncvLayout {
    UncvLayout {
        format,
        profile,
        components,
        bit_depth: 8,
        sampling,
        interleave,
        block_size: 0,
        pixel_size,
    }
}

// the layouts GStreamer's isomp4mux writes for these formats
const UNCV_LAYOUTS: &[UncvLayout] = &[
    layout(
        RawVideoFormat::Rgb8,
        *b"rgb3",
        &[RED, GREEN, BLUE],
        SAMPLING_NONE,
        INTERLEAVE_PIXEL,
        3,
    ),
    layout(
        RawVideoFormat::Bgr8,
        NO_PROFILE,
        &[BLUE, GREEN, RED],
        SAMPLING_NONE,
        INTERLEAVE_PIXEL,
        3,
    ),
    layout(
        RawVideoFormat::Rgba8,
        *b"rgba",
        &[RED, GREEN, BLUE, ALPHA],
        SAMPLING_NONE,
        INTERLEAVE_PIXEL,
        4,
    ),
    layout(
        RawVideoFormat::Argb8,
        NO_PROFILE,
        &[ALPHA, RED, GREEN, BLUE],
        SAMPLING_NONE,
        INTERLEAVE_PIXEL,
        4,
    ),
    layout(
        RawVideoFormat::Bgra8,
        NO_PROFILE,
        &[BLUE, GREEN, RED, ALPHA],
        SAMPLING_NONE,
        INTERLEAVE_PIXEL,
        4,
    ),
    layout(
        RawVideoFormat::Bgrx8,
        NO_PROFILE,
        &[BLUE, GREEN, RED],
        SAMPLING_NONE,
        INTERLEAVE_PIXEL,
        4,
    ),
    layout(
        RawVideoFormat::Gray8,
        NO_PROFILE,
        &[MONOCHROME],
        SAMPLING_NONE,
        INTERLEAVE_COMPONENT,
        0,
    ),
    layout(
        RawVideoFormat::I420,
        *b"i420",
        &[LUMA, CB, CR],
        SAMPLING_420,
        INTERLEAVE_COMPONENT,
        0,
    ),
    layout(
        RawVideoFormat::I422,
        *b"yu22",
        &[LUMA, CB, CR],
        SAMPLING_422,
        INTERLEAVE_COMPONENT,
        0,
    ),
    layout(
        RawVideoFormat::I444,
        NO_PROFILE,
        &[LUMA, CB, CR],
        SAMPLING_NONE,
        INTERLEAVE_COMPONENT,
        0,
    ),
    layout(
        RawVideoFormat::Nv12,
        *b"nv12",
        &[LUMA, CB, CR],
        SAMPLING_420,
        INTERLEAVE_MIXED,
        0,
    ),
    layout(
        RawVideoFormat::Yuyv,
        *b"yuv2",
        &[LUMA, CB, LUMA, CR],
        SAMPLING_422,
        INTERLEAVE_MULTI_Y,
        4,
    ),
    layout(
        RawVideoFormat::Uyvy,
        *b"2vuy",
        &[CB, LUMA, CR, LUMA],
        SAMPLING_422,
        INTERLEAVE_MULTI_Y,
        4,
    ),
    layout(
        RawVideoFormat::Rgbp8,
        NO_PROFILE,
        &[RED, GREEN, BLUE],
        SAMPLING_NONE,
        INTERLEAVE_COMPONENT,
        0,
    ),
    layout(
        RawVideoFormat::Bgrp8,
        NO_PROFILE,
        &[BLUE, GREEN, RED],
        SAMPLING_NONE,
        INTERLEAVE_COMPONENT,
        0,
    ),
    // r210 packs three 10-bit components into each big-endian 32-bit word
    UncvLayout {
        format: RawVideoFormat::R210,
        profile: NO_PROFILE,
        components: &[RED, GREEN, BLUE],
        bit_depth: 10,
        sampling: SAMPLING_NONE,
        interleave: INTERLEAVE_PIXEL,
        block_size: 4,
        pixel_size: 4,
    },
];

fn bayer_sites(pattern: BayerPattern) -> [u16; 4] {
    // GStreamer's cpat lists the top-left, top-right, bottom-left and bottom-right sites by component type
    match pattern {
        BayerPattern::Bggr => [BLUE, GREEN, GREEN, RED],
        BayerPattern::Gbrg => [GREEN, BLUE, RED, GREEN],
        BayerPattern::Grbg => [GREEN, RED, BLUE, GREEN],
        BayerPattern::Rggb => [RED, GREEN, GREEN, BLUE],
    }
}

const BAYER_PATTERNS: [BayerPattern; 4] = [
    BayerPattern::Bggr,
    BayerPattern::Gbrg,
    BayerPattern::Grbg,
    BayerPattern::Rggb,
];

// samples above 8 bits sit in 2-byte blocks
const BAYER_WIDE_BLOCK_SIZE: u8 = 2;

fn bayer_bits(sample: BayerSampleFormat) -> (u8, u8) {
    match sample {
        BayerSampleFormat::Bits8 => (0, 0),
        wide if wide.is_big_endian() => (BAYER_WIDE_BLOCK_SIZE, 0),
        _ => (BAYER_WIDE_BLOCK_SIZE, COMPONENTS_LITTLE_ENDIAN),
    }
}

const BAYER_SAMPLE_FORMATS: [BayerSampleFormat; 9] = [
    BayerSampleFormat::Bits8,
    BayerSampleFormat::Bits10Le,
    BayerSampleFormat::Bits10Be,
    BayerSampleFormat::Bits12Le,
    BayerSampleFormat::Bits12Be,
    BayerSampleFormat::Bits14Le,
    BayerSampleFormat::Bits14Be,
    BayerSampleFormat::Bits16Le,
    BayerSampleFormat::Bits16Be,
];

#[derive(Debug)]
struct FrameConfig {
    profile: [u8; 4],
    component_count: usize,
    bit_depth: u8,
    sampling: u8,
    interleave: u8,
    block_size: u8,
    flags: u8,
    pixel_size: u32,
    row_align: u32,
}

fn uncc_box(config: &FrameConfig) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&config.profile);
    payload.extend_from_slice(&(config.component_count as u32).to_be_bytes());
    for index in 0..config.component_count {
        payload.extend_from_slice(&(index as u16).to_be_bytes());
        // bit depth minus one, then unsigned integer components with no alignment
        payload.extend_from_slice(&[config.bit_depth - 1, 0, 0]);
    }
    payload.extend_from_slice(&[
        config.sampling,
        config.interleave,
        config.block_size,
        config.flags,
    ]);
    payload.extend_from_slice(&config.pixel_size.to_be_bytes());
    payload.extend_from_slice(&config.row_align.to_be_bytes());
    // tile_align_size, then one tile across and down
    payload.extend_from_slice(&[0; 12]);
    full_box(b"uncC", 0, 0, &payload)
}

fn cmpd_box(components: &[u16]) -> Vec<u8> {
    let mut payload = (components.len() as u32).to_be_bytes().to_vec();
    for component in components {
        payload.extend_from_slice(&component.to_be_bytes());
    }
    mp4_box(b"cmpd", &payload)
}

const BAYER_PATTERN_SIZE: u16 = 2;
const UNIT_GAIN: f32 = 1.0;

fn cpat_box(sites: [u32; 4]) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&BAYER_PATTERN_SIZE.to_be_bytes());
    payload.extend_from_slice(&BAYER_PATTERN_SIZE.to_be_bytes());
    for site in sites {
        payload.extend_from_slice(&site.to_be_bytes());
        payload.extend_from_slice(&UNIT_GAIN.to_be_bytes());
    }
    full_box(b"cpat", 0, 0, &payload)
}

fn subsampling_fits(sampling: u8, width: u32, height: u32) -> bool {
    match sampling {
        SAMPLING_422 => width.is_multiple_of(2),
        SAMPLING_420 => width.is_multiple_of(2) && height.is_multiple_of(2),
        _ => true,
    }
}

fn frame_boxes(format: RawVideoFormat, width: u32, height: u32) -> Option<Vec<u8>> {
    if let RawVideoFormat::Bayer(BayerFormat { pattern, sample }) = format {
        let (block_size, flags) = bayer_bits(sample);
        let sample_bytes = u32::from(sample.bit_depth().div_ceil(8));
        let stride = format.plane_stride(0, width)?;
        let row_align = match stride == width.checked_mul(sample_bytes)? {
            true => 0,
            false => stride,
        };
        let config = FrameConfig {
            profile: NO_PROFILE,
            component_count: 1,
            bit_depth: sample.bit_depth(),
            sampling: SAMPLING_NONE,
            interleave: INTERLEAVE_COMPONENT,
            block_size,
            flags,
            pixel_size: 0,
            row_align,
        };
        return Some(
            [
                cpat_box(bayer_sites(pattern).map(u32::from)),
                cmpd_box(&[FILTER_ARRAY]),
                uncc_box(&config),
            ]
            .concat(),
        );
    }
    let layout = UNCV_LAYOUTS.iter().find(|layout| layout.format == format)?;
    if !subsampling_fits(layout.sampling, width, height) {
        return None;
    }
    // GStreamer's qtdemux reads a zero row_align_size as rows of the components alone, missing BGRx's fourth byte
    let row_align = match layout.interleave {
        INTERLEAVE_PIXEL => format.plane_stride(0, width)?,
        _ => 0,
    };
    let config = FrameConfig {
        profile: layout.profile,
        component_count: layout.components.len(),
        bit_depth: layout.bit_depth,
        sampling: layout.sampling,
        interleave: layout.interleave,
        block_size: layout.block_size,
        flags: 0,
        pixel_size: layout.pixel_size,
        row_align,
    };
    Some([cmpd_box(layout.components), uncc_box(&config)].concat())
}

fn compression_fourcc(compression: LosslessCompression) -> [u8; 4] {
    match compression {
        LosslessCompression::Zlib => *b"zlib",
        LosslessCompression::Deflate => *b"defl",
        LosslessCompression::Brotli => *b"brot",
    }
}

// ISO/IEC 23001-17 generic compression of whole samples
const GCMP_SCHEME_VERSION: u32 = 1;
const COMPRESSED_UNIT_WHOLE_SAMPLE: u8 = 0;

fn rinf_box(compression: LosslessCompression) -> Vec<u8> {
    let scheme = [*b"gcmp", GCMP_SCHEME_VERSION.to_be_bytes()].concat();
    let compression_config = [
        compression_fourcc(compression).as_slice(),
        &[COMPRESSED_UNIT_WHOLE_SAMPLE],
    ]
    .concat();
    let schi = mp4_box(b"schi", &full_box(b"cmpC", 0, 0, &compression_config));
    let children = [
        mp4_box(b"frma", b"uncv"),
        full_box(b"schm", 0, 0, &scheme),
        schi,
    ]
    .concat();
    mp4_box(b"rinf", &children)
}

pub(crate) fn formats() -> impl Iterator<Item = RawVideoFormat> {
    written_formats().chain(PACKED_PLANAR_FORMATS.iter().map(|&(format, _)| format))
}

fn written_formats() -> impl Iterator<Item = RawVideoFormat> {
    UNCV_LAYOUTS.iter().map(|layout| layout.format).chain(
        g2g_core::caps::ALL_BAYER_FORMATS
            .into_iter()
            .map(RawVideoFormat::Bayer),
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct UncvSampleEntry {
    pub(crate) fourcc: [u8; 4],
    pub(crate) boxes: Vec<u8>,
}

impl UncvSampleEntry {
    pub(crate) fn is_compressed(&self) -> bool {
        &self.fourcc == b"resv"
    }
}

fn fixed_raw_video(caps: &Caps) -> Option<(RawVideoFormat, u32, u32)> {
    match caps {
        Caps::RawVideo {
            format,
            width: Dim::Fixed(width),
            height: Dim::Fixed(height),
            ..
        } if u16::try_from(*width).is_ok() && u16::try_from(*height).is_ok() => {
            Some((*format, *width, *height))
        }
        _ => None,
    }
}

pub(crate) fn sample_entry(caps: &Caps) -> Option<UncvSampleEntry> {
    match caps {
        Caps::LosslesslyCompressed {
            compression,
            original: Some(original),
        } => {
            let (format, width, height) = fixed_raw_video(original)?;
            Some(UncvSampleEntry {
                fourcc: *b"resv",
                boxes: [rinf_box(*compression), frame_boxes(format, width, height)?].concat(),
            })
        }
        caps => {
            let (format, width, height) = fixed_raw_video(caps)?;
            Some(UncvSampleEntry {
                fourcc: *b"uncv",
                boxes: frame_boxes(format, width, height)?,
            })
        }
    }
}

pub(crate) fn original_caps(caps: &Caps) -> &Caps {
    match caps {
        Caps::LosslesslyCompressed {
            original: Some(original),
            ..
        } => original,
        caps => caps,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct UncvFrames {
    pub(crate) format: RawVideoFormat,
    pub(crate) storage: RawVideoStorage,
    pub(crate) compression: Option<LosslessCompression>,
}

struct Reader<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn bytes(&mut self, count: usize) -> Result<&'a [u8], G2gError> {
        let end = self.at.checked_add(count).ok_or(G2gError::CapsMismatch)?;
        let bytes = self.data.get(self.at..end).ok_or(G2gError::CapsMismatch)?;
        self.at = end;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8, G2gError> {
        Ok(self.bytes(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, G2gError> {
        let bytes = self.bytes(2)?;
        Ok(u16::from_be_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> Result<u32, G2gError> {
        let value = be32(self.data, self.at)?;
        self.at += 4;
        Ok(value)
    }
}

const FULL_BOX_HEADER: usize = 4;
// cmpd lists at most this many components, which bounds the allocation a crafted count asks for
const MAX_COMPONENTS: u32 = 64;
const FIRST_CUSTOM_COMPONENT_TYPE: u16 = 0x8000;

fn parse_cmpd(cmpd: &[u8]) -> Result<Vec<u16>, G2gError> {
    let mut reader = Reader { data: cmpd, at: 0 };
    let count = reader.u32()?;
    if count == 0 || count > MAX_COMPONENTS {
        return Err(G2gError::CapsMismatch);
    }
    (0..count)
        .map(|_| {
            let component = reader.u16()?;
            if component >= FIRST_CUSTOM_COMPONENT_TYPE {
                let uri = &cmpd[reader.at..];
                let end = memchr::memchr(0, uri).ok_or(G2gError::CapsMismatch)?;
                reader.bytes(end + 1)?;
            }
            Ok(component)
        })
        .collect()
}

fn bayer_pattern(cpat: &[u8], types: &[u16]) -> Result<Option<BayerPattern>, G2gError> {
    let mut reader = Reader { data: cpat, at: 0 };
    let version = reader.bytes(FULL_BOX_HEADER)?[0];
    if version != 0 || reader.u16()? != BAYER_PATTERN_SIZE || reader.u16()? != BAYER_PATTERN_SIZE {
        return Ok(None);
    }
    // GStreamer writes the colour types directly and declares only the filter array in cmpd.
    let legacy_types = types == [FILTER_ARRAY];
    let mut sites = [0u16; 4];
    for site in &mut sites {
        let index = reader.u32()?;
        *site = if legacy_types {
            let Ok(component) = u16::try_from(index) else {
                return Ok(None);
            };
            component
        } else {
            let index = usize::try_from(index).map_err(|_| G2gError::CapsMismatch)?;
            *types.get(index).ok_or(G2gError::CapsMismatch)?
        };
        reader.u32()?;
    }
    Ok(BAYER_PATTERNS
        .into_iter()
        .find(|pattern| bayer_sites(*pattern) == sites))
}

fn bayer_sample(bit_depth: u8, block_size: u8, flags: u8) -> Option<BayerSampleFormat> {
    // A 16-bit component occupies two bytes with or without an explicit block.
    let block_size = match (bit_depth, block_size) {
        (16, 0) => BAYER_WIDE_BLOCK_SIZE,
        _ => block_size,
    };
    BAYER_SAMPLE_FORMATS.into_iter().find(|sample| {
        sample.bit_depth() == bit_depth && bayer_bits(*sample) == (block_size, flags)
    })
}

fn profile_format(profile: [u8; 4]) -> Option<RawVideoFormat> {
    match &profile {
        b"rgb3" => Some(RawVideoFormat::Rgb8),
        b"rgba" => Some(RawVideoFormat::Rgba8),
        _ => None,
    }
}

fn parse_frames(
    children: &[u8],
    compression: Option<LosslessCompression>,
) -> Result<Option<UncvFrames>, G2gError> {
    let uncc = find_entry_child(children, b"uncC").ok_or(G2gError::CapsMismatch)?;
    let mut reader = Reader { data: uncc, at: 0 };
    let version = reader.u8()?;
    reader.bytes(3)?;
    let profile: [u8; 4] = reader.bytes(4)?.try_into().expect("4 bytes");
    if version == 1 {
        return Ok(profile_format(profile).map(|format| UncvFrames {
            format,
            storage: RawVideoStorage::Tight,
            compression,
        }));
    }
    if version != 0 {
        return Ok(None);
    }
    let cmpd = find_entry_child(children, b"cmpd").ok_or(G2gError::CapsMismatch)?;
    let types = parse_cmpd(cmpd)?;
    let count = reader.u32()?;
    if count == 0 || count > MAX_COMPONENTS {
        return Err(G2gError::CapsMismatch);
    }
    let mut components = Vec::new();
    let mut bit_depths = Vec::new();
    let mut plain = true;
    for _ in 0..count {
        let index = usize::from(reader.u16()?);
        let bit_depth = reader.u8()?.checked_add(1).ok_or(G2gError::CapsMismatch)?;
        let (component_format, component_align) = (reader.u8()?, reader.u8()?);
        components.push(*types.get(index).ok_or(G2gError::CapsMismatch)?);
        bit_depths.push(bit_depth);
        plain &= component_format == 0 && component_align == 0;
    }
    let (sampling, interleave, block_size, flags) =
        (reader.u8()?, reader.u8()?, reader.u8()?, reader.u8()?);
    let pixel_size = reader.u32()?;
    let row_align = reader.u32()?;
    let tile_align = reader.u32()?;
    let (tile_columns, tile_rows) = (reader.u32()?, reader.u32()?);
    let bit_depth = bit_depths[0];
    if !plain || tile_columns != 0 || tile_rows != 0 || bit_depths.iter().any(|d| *d != bit_depth) {
        return Ok(None);
    }
    let row_interleaved = interleave == INTERLEAVE_ROW;
    if row_interleaved && (sampling != SAMPLING_NONE || tile_align != 0) {
        return Ok(None);
    }
    // Both modes use the same per-plane row layout.
    let interleave = if row_interleaved {
        INTERLEAVE_COMPONENT
    } else {
        interleave
    };
    let describe = |format: RawVideoFormat, plane_order, bit_packed| UncvFrames {
        format,
        storage: RawVideoStorage::AlignedRows {
            row_align,
            plane_order,
            bit_packed,
            row_interleaved: row_interleaved && format.plane_count() > 1,
        },
        compression,
    };
    // Endianness has no effect on single-byte components.
    let flags = if bit_depth == 8 {
        flags & !COMPONENTS_LITTLE_ENDIAN
    } else {
        flags
    };
    let bit_packed = interleave == INTERLEAVE_COMPONENT
        && block_size == 0
        && flags == 0
        && pixel_size == 0
        && tile_align == 0
        && matches!(bit_depth, 10 | 12 | 14);
    if components == [FILTER_ARRAY] {
        let Some(cpat) = find_entry_child(children, b"cpat") else {
            return Ok(None);
        };
        let plain_bayer =
            sampling == SAMPLING_NONE && interleave == INTERLEAVE_COMPONENT && pixel_size == 0;
        let sample = if bit_packed {
            bayer_sample(bit_depth, BAYER_WIDE_BLOCK_SIZE, COMPONENTS_LITTLE_ENDIAN)
        } else {
            bayer_sample(bit_depth, block_size, flags)
        };
        let format = match (bayer_pattern(cpat, &types)?, sample) {
            (Some(pattern), Some(sample)) if plain_bayer => {
                RawVideoFormat::Bayer(BayerFormat { pattern, sample })
            }
            _ => return Ok(None),
        };
        return Ok(Some(describe(format, None, bit_packed)));
    }
    if bit_packed {
        // ISO/IEC 23001-17 requires even row alignment for subsampled planar YUV.
        if sampling != SAMPLING_NONE && !row_align.is_multiple_of(2) {
            return Ok(None);
        }
        return Ok(packed_planar_format(&components, bit_depth, sampling)
            .map(|(format, order)| describe(format, Some(order), true)));
    }
    // Without a pixel size, byte-aligned pixel interleave stores the listed components contiguously.
    let pixel_size = if interleave == INTERLEAVE_PIXEL && bit_depth == 8 && pixel_size == 0 {
        count
    } else {
        pixel_size
    };
    let matching_layouts = || {
        UNCV_LAYOUTS.iter().filter(move |layout| {
            layout.bit_depth == bit_depth
                && layout.sampling == sampling
                && layout.interleave == interleave
                && layout.block_size == block_size
                && flags == 0
                && layout.pixel_size == pixel_size
        })
    };
    if let Some(layout) = matching_layouts().find(|layout| layout.components == components) {
        return Ok(Some(describe(layout.format, None, false)));
    }
    if interleave == INTERLEAVE_MIXED
        && block_size == 0
        && flags == 0
        && pixel_size == 0
        && tile_align == 0
    {
        return Ok(
            mixed_chroma_layout(&components, sampling, bit_depth, row_align).map(
                |(format, storage)| UncvFrames {
                    format,
                    storage,
                    compression,
                },
            ),
        );
    }
    if interleave != INTERLEAVE_COMPONENT || compression.is_some() {
        return Ok(None);
    }
    Ok(matching_layouts().find_map(|layout| {
        let order = planar_component_order(&components, layout.components)?;
        Some(describe(layout.format, Some(order), false))
    }))
}

fn mixed_chroma_layout(
    components: &[u16],
    sampling: u8,
    bit_depth: u8,
    row_align: u32,
) -> Option<(RawVideoFormat, RawVideoStorage)> {
    let format = match (sampling, bit_depth) {
        (SAMPLING_420, 8) => RawVideoFormat::Nv12,
        (SAMPLING_422, 8) => RawVideoFormat::I422,
        (SAMPLING_420 | SAMPLING_422, 10 | 12) => {
            PACKED_PLANAR_FORMATS.iter().find_map(|&(format, mode)| {
                (mode == sampling && format.bit_depth() == bit_depth).then_some(format)
            })?
        }
        _ => return None,
    };
    let (chroma_first, vu) = match components {
        [LUMA, CB, CR] => (false, false),
        [LUMA, CR, CB] => (false, true),
        [CB, CR, LUMA] => (true, false),
        [CR, CB, LUMA] => (true, true),
        _ => return None,
    };
    Some((
        format,
        RawVideoStorage::MixedChromaRows {
            row_align,
            chroma_first,
            vu,
        },
    ))
}

// Dense layouts with an existing graph format that stores each sample as a LE word.
const PACKED_PLANAR_FORMATS: &[(RawVideoFormat, u8)] = &[
    (RawVideoFormat::I420p10, SAMPLING_420),
    (RawVideoFormat::I422p10, SAMPLING_422),
    (RawVideoFormat::I444p10, SAMPLING_NONE),
    (RawVideoFormat::I420p12, SAMPLING_420),
    (RawVideoFormat::I422p12, SAMPLING_422),
    (RawVideoFormat::I444p12, SAMPLING_NONE),
    (RawVideoFormat::Gbr10Le, SAMPLING_NONE),
];

fn packed_planar_format(
    components: &[u16],
    bit_depth: u8,
    sampling: u8,
) -> Option<(RawVideoFormat, [u8; 3])> {
    PACKED_PLANAR_FORMATS
        .iter()
        .filter(|(format, subsampling)| *subsampling == sampling && format.bit_depth() == bit_depth)
        .find_map(|&(format, _)| {
            let output = match format {
                RawVideoFormat::Gbr10Le => [GREEN, BLUE, RED],
                _ => [LUMA, CB, CR],
            };
            planar_component_order(components, &output).map(|order| (format, order))
        })
}

// Match a three-plane permutation. Each component must appear exactly once.
fn planar_component_order(stored: &[u16], output: &[u16]) -> Option<[u8; 3]> {
    let &[first, second, third] = stored else {
        return None;
    };
    if output.len() != 3 || first == second || first == third || second == third {
        return None;
    }
    let position = |component| output.iter().position(|&candidate| candidate == component);
    Some([
        position(first)? as u8,
        position(second)? as u8,
        position(third)? as u8,
    ])
}

fn generic_compression(rinf: &[u8]) -> Result<Option<LosslessCompression>, G2gError> {
    let frma = find_entry_child(rinf, b"frma").ok_or(G2gError::CapsMismatch)?;
    let schm = find_entry_child(rinf, b"schm").ok_or(G2gError::CapsMismatch)?;
    let scheme = schm
        .get(FULL_BOX_HEADER..FULL_BOX_HEADER + 4)
        .ok_or(G2gError::CapsMismatch)?;
    if frma.get(..4) != Some(b"uncv") || scheme != b"gcmp" {
        return Ok(None);
    }
    let schi = find_entry_child(rinf, b"schi").ok_or(G2gError::CapsMismatch)?;
    let cmpc = find_entry_child(schi, b"cmpC").ok_or(G2gError::CapsMismatch)?;
    let mut reader = Reader {
        data: cmpc,
        at: FULL_BOX_HEADER,
    };
    let fourcc: [u8; 4] = reader.bytes(4)?.try_into().expect("4 bytes");
    if reader.u8()? != COMPRESSED_UNIT_WHOLE_SAMPLE {
        return Ok(None);
    }
    Ok(LosslessCompression::ALL
        .into_iter()
        .find(|compression| compression_fourcc(*compression) == fourcc))
}

pub(crate) fn parse_sample_entry(
    fourcc: &[u8; 4],
    children: &[u8],
) -> Result<Option<UncvFrames>, G2gError> {
    let compression = match fourcc {
        b"uncv" => None,
        b"resv" => {
            let rinf = find_entry_child(children, b"rinf").ok_or(G2gError::CapsMismatch)?;
            match generic_compression(rinf)? {
                Some(compression) => Some(compression),
                None => return Ok(None),
            }
        }
        _ => return Ok(None),
    };
    parse_frames(children, compression)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(format: RawVideoFormat, width: u32, height: u32) -> Caps {
        crate::uncompressedvideo::raw_video_caps(
            format,
            Dim::Fixed(width),
            Dim::Fixed(height),
            g2g_core::Rate::Any,
            g2g_core::PixelAspect::Any,
        )
    }

    #[test]
    fn every_written_layout_parses_back_to_its_format() {
        for format in written_formats() {
            for compression in [None, Some(LosslessCompression::Brotli)] {
                let caps = match compression {
                    None => raw(format, 18, 4),
                    Some(compression) => Caps::LosslesslyCompressed {
                        compression,
                        original: Some(alloc::boxed::Box::new(raw(format, 18, 4))),
                    },
                };
                let entry = sample_entry(&caps).unwrap_or_else(|| panic!("{format:?} writes"));
                let parsed = parse_sample_entry(&entry.fourcc, &entry.boxes)
                    .expect("well formed")
                    .expect("a graph format");
                assert_eq!((parsed.format, parsed.compression), (format, compression));
            }
        }
    }

    #[test]
    fn cpat_resolves_colour_indices_in_cmpd() {
        let mut cmpd = 5u32.to_be_bytes().to_vec();
        for component in [FILTER_ARRAY, FIRST_CUSTOM_COMPONENT_TYPE, BLUE, RED, GREEN] {
            cmpd.extend_from_slice(&component.to_be_bytes());
            if component == FIRST_CUSTOM_COMPONENT_TYPE {
                cmpd.extend_from_slice(b"urn:example:unused-component\0");
            }
        }
        let cmpd = mp4_box(b"cmpd", &cmpd);
        let uncc = uncc_box(&FrameConfig {
            profile: NO_PROFILE,
            component_count: 1,
            bit_depth: 16,
            sampling: SAMPLING_NONE,
            interleave: INTERLEAVE_COMPONENT,
            block_size: 0,
            flags: COMPONENTS_LITTLE_ENDIAN,
            pixel_size: 0,
            row_align: 0,
        });
        let parse = |sites| {
            parse_sample_entry(
                b"uncv",
                &[cmpd.as_slice(), &uncc, &cpat_box(sites)].concat(),
            )
        };
        let entry = parse([3, 4, 4, 2]).expect("valid indices").unwrap();
        assert_eq!(
            entry.format,
            RawVideoFormat::Bayer(BayerFormat {
                pattern: BayerPattern::Rggb,
                sample: BayerSampleFormat::Bits16Le,
            })
        );
        // 6 is the legacy blue type, but a wrong index into this file's table.
        assert!(parse([3, 4, 4, 6]).is_err());
    }

    #[test]
    fn byte_aligned_layouts_accept_implicit_pixel_sizes() {
        for format in [
            RawVideoFormat::Rgb8,
            RawVideoFormat::Rgba8,
            RawVideoFormat::Bgrp8,
        ] {
            let layout = UNCV_LAYOUTS
                .iter()
                .find(|layout| layout.format == format)
                .unwrap();
            let boxes = [
                cmpd_box(layout.components),
                uncc_box(&FrameConfig {
                    profile: layout.profile,
                    component_count: layout.components.len(),
                    bit_depth: layout.bit_depth,
                    sampling: layout.sampling,
                    interleave: layout.interleave,
                    block_size: 0,
                    flags: COMPONENTS_LITTLE_ENDIAN,
                    pixel_size: 0,
                    row_align: 0,
                }),
            ];
            let entry = parse_sample_entry(b"uncv", &boxes.concat())
                .expect("valid uncv")
                .unwrap();
            assert_eq!(entry.format, format);
        }
    }

    #[test]
    fn subsampled_formats_refuse_odd_sizes() {
        assert!(sample_entry(&raw(RawVideoFormat::I420, 18, 3)).is_none());
        assert!(sample_entry(&raw(RawVideoFormat::Yuyv, 17, 4)).is_none());
        assert!(sample_entry(&raw(RawVideoFormat::I444, 17, 3)).is_some());
    }
}
