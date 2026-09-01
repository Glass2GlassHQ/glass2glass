//! Netpbm still-image codecs (`pnmenc` / `pnmdec`): packed RGB(A) raw video
//! to a PBM / PGM / PPM file and back. CPU-only `no_std` baseline.
//!
//! `pnmenc ascii=` writes ASCII P3 instead of binary P6; `pnmdec` has no
//! knobs. Geometry is the file's word, so it is
//! bounded before any buffer is sized (see [`crate::stillimage`]). Output is
//! packed RGB8: PGM / PBM expand to grey / black-white RGB because g2g has no
//! GRAY8 raw format. System memory.

use core::fmt::Write as _;
use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec, Rate, RawVideoFormat, VideoCodec,
};

use crate::stillframe::{ImageAssembler, MAX_ENCODED_BYTES};
use crate::stillimage::{packed_byte_size, StillImageOutput};

/// Netpbm caps ASCII raster lines at 70 characters.
const MAX_ASCII_LINE: usize = 70;

/// Netpbm magic `P1`..`P6`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PnmKind {
    /// ASCII bitmap.
    P1,
    /// ASCII graymap.
    P2,
    /// ASCII pixmap.
    P3,
    /// Binary bitmap.
    P4,
    /// Binary graymap.
    P5,
    /// Binary pixmap.
    P6,
}

impl PnmKind {
    fn from_magic(b: u8) -> Option<Self> {
        match b {
            b'1' => Some(Self::P1),
            b'2' => Some(Self::P2),
            b'3' => Some(Self::P3),
            b'4' => Some(Self::P4),
            b'5' => Some(Self::P5),
            b'6' => Some(Self::P6),
            _ => None,
        }
    }

    fn is_ascii(self) -> bool {
        matches!(self, Self::P1 | Self::P2 | Self::P3)
    }

    fn is_bitmap(self) -> bool {
        matches!(self, Self::P1 | Self::P4)
    }

    fn channels(self) -> usize {
        match self {
            Self::P3 | Self::P6 => 3,
            _ => 1,
        }
    }
}

/// Header recovered from a PNM prefix: kind, geometry, sample max, and the
/// byte index where the raster starts.
#[derive(Clone, Copy, Debug)]
struct PnmHeader {
    kind: PnmKind,
    width: u32,
    height: u32,
    maxval: u32,
    raster: usize,
}

/// Length of the complete image at the start of `data`, or `None` when more
/// bytes are needed. The header is attacker-controlled, so every count is
/// checked before it sizes a body.
pub(crate) fn pnm_frame_length(data: &[u8]) -> Result<Option<usize>, G2gError> {
    match parse_header(data)? {
        None => Ok(None),
        Some(header) => match raster_end(data, header)? {
            None => Ok(None),
            Some(end) => Ok(Some(end)),
        },
    }
}

fn parse_header(data: &[u8]) -> Result<Option<PnmHeader>, G2gError> {
    if data.is_empty() {
        return Ok(None);
    }
    if data[0] != b'P' {
        return Err(G2gError::CapsMismatch);
    }
    if data.len() < 2 {
        return Ok(None);
    }
    let kind = PnmKind::from_magic(data[1]).ok_or(G2gError::CapsMismatch)?;
    let mut cur = Cursor { data, pos: 2 };
    let Some(width) = cur.token_u32()? else {
        return Ok(None);
    };
    let Some(height) = cur.token_u32()? else {
        return Ok(None);
    };
    packed_byte_size(width, height, 3)?;
    let maxval = if kind.is_bitmap() {
        1
    } else {
        let Some(v) = cur.token_u32()? else {
            return Ok(None);
        };
        if v == 0 || v > 65535 {
            return Err(G2gError::CapsMismatch);
        }
        v
    };
    // One whitespace byte separates the last header token from the raster.
    if cur.pos >= data.len() {
        return Ok(None);
    }
    if !is_space(data[cur.pos]) {
        return Err(G2gError::CapsMismatch);
    }
    cur.pos += 1;
    Ok(Some(PnmHeader {
        kind,
        width,
        height,
        maxval,
        raster: cur.pos,
    }))
}

fn raster_end(data: &[u8], header: PnmHeader) -> Result<Option<usize>, G2gError> {
    let pixels = (header.width as usize)
        .checked_mul(header.height as usize)
        .ok_or(G2gError::CapsMismatch)?;
    if header.kind.is_ascii() {
        return ascii_raster_end(data, header.raster, pixels, header.kind);
    }
    let body = binary_body_bytes(header)?;
    let total = header
        .raster
        .checked_add(body)
        .filter(|t| *t <= MAX_ENCODED_BYTES)
        .ok_or(G2gError::CapsMismatch)?;
    Ok((data.len() >= total).then_some(total))
}

fn binary_body_bytes(header: PnmHeader) -> Result<usize, G2gError> {
    let w = header.width as usize;
    let h = header.height as usize;
    match header.kind {
        PnmKind::P4 => {
            let row = w.div_ceil(8);
            row.checked_mul(h).ok_or(G2gError::CapsMismatch)
        }
        PnmKind::P5 | PnmKind::P6 => {
            let sample_bytes = if header.maxval < 256 { 1 } else { 2 };
            w.checked_mul(h)
                .and_then(|n| n.checked_mul(header.kind.channels()))
                .and_then(|n| n.checked_mul(sample_bytes))
                .ok_or(G2gError::CapsMismatch)
        }
        _ => Err(G2gError::CapsMismatch),
    }
}

fn ascii_raster_end(
    data: &[u8],
    start: usize,
    pixels: usize,
    kind: PnmKind,
) -> Result<Option<usize>, G2gError> {
    let needed = pixels
        .checked_mul(kind.channels())
        .ok_or(G2gError::CapsMismatch)?;
    let mut cur = Cursor { data, pos: start };
    for _ in 0..needed {
        let token = if kind.is_bitmap() {
            cur.token_bit()?
        } else {
            cur.token_u32()?
        };
        if token.is_none() {
            return Ok(None);
        }
    }
    Ok(Some(cur.pos.min(data.len())))
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Cursor<'_> {
    fn skip_ws_and_comments(&mut self) -> Result<bool, G2gError> {
        loop {
            if self.pos >= self.data.len() {
                return Ok(false);
            }
            match self.data[self.pos] {
                b'#' => {
                    self.pos += 1;
                    while self.pos < self.data.len() && self.data[self.pos] != b'\n' {
                        self.pos += 1;
                    }
                }
                b if is_space(b) => self.pos += 1,
                _ => return Ok(true),
            }
            if self.pos > MAX_ENCODED_BYTES {
                return Err(G2gError::CapsMismatch);
            }
        }
    }

    fn token_u32(&mut self) -> Result<Option<u32>, G2gError> {
        if !self.skip_ws_and_comments()? {
            return Ok(None);
        }
        let start = self.pos;
        while self.pos < self.data.len() && self.data[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        if self.pos == start {
            return Err(G2gError::CapsMismatch);
        }
        let text = core::str::from_utf8(&self.data[start..self.pos])
            .map_err(|_| G2gError::CapsMismatch)?;
        text.parse::<u32>()
            .map(Some)
            .map_err(|_| G2gError::CapsMismatch)
    }

    /// One P1 pixel: a single `0` / `1`, which the format lets run together.
    fn token_bit(&mut self) -> Result<Option<u32>, G2gError> {
        if !self.skip_ws_and_comments()? {
            return Ok(None);
        }
        let bit = match self.data[self.pos] {
            b'0' => 0,
            b'1' => 1,
            _ => return Err(G2gError::CapsMismatch),
        };
        self.pos += 1;
        Ok(Some(bit))
    }
}

fn is_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\r' | b'\n')
}

/// Decode one PNM image to packed RGB8. Malformed or over-budget input fails
/// with `CapsMismatch` rather than allocating on the header's word.
pub(crate) fn decode_pnm(data: &[u8]) -> Result<(Vec<u8>, u32, u32), G2gError> {
    let header = parse_header(data)?.ok_or(G2gError::CapsMismatch)?;
    let needed = packed_byte_size(header.width, header.height, 3)?;
    let mut rgb = vec![0u8; needed];
    match header.kind {
        PnmKind::P6 => decode_pixmap(data, header, &mut rgb)?,
        PnmKind::P5 => decode_graymap(data, header, &mut rgb)?,
        PnmKind::P4 => decode_bitmap_bin(data, header, &mut rgb)?,
        PnmKind::P3 => decode_ascii_pixmap(data, header, &mut rgb)?,
        PnmKind::P2 => decode_ascii_gray(data, header, &mut rgb)?,
        PnmKind::P1 => decode_ascii_bitmap(data, header, &mut rgb)?,
    }
    Ok((rgb, header.width, header.height))
}

fn scale_sample(v: u32, maxval: u32) -> u8 {
    if maxval == 0 {
        return 0;
    }
    if maxval == 255 {
        return v.min(255) as u8;
    }
    ((v as u64 * 255) / maxval as u64).min(255) as u8
}

fn decode_pixmap(data: &[u8], header: PnmHeader, rgb: &mut [u8]) -> Result<(), G2gError> {
    let sample_bytes = if header.maxval < 256 { 1 } else { 2 };
    let samples = rgb.len();
    let body = data.get(header.raster..).ok_or(G2gError::CapsMismatch)?;
    let need = samples
        .checked_mul(sample_bytes)
        .ok_or(G2gError::CapsMismatch)?;
    if body.len() < need {
        return Err(G2gError::CapsMismatch);
    }
    if sample_bytes == 1 && header.maxval == 255 {
        rgb.copy_from_slice(&body[..samples]);
        return Ok(());
    }
    for (i, px) in rgb.iter_mut().enumerate() {
        let v = if sample_bytes == 1 {
            body[i] as u32
        } else {
            let o = i * 2;
            u16::from_be_bytes([body[o], body[o + 1]]) as u32
        };
        *px = scale_sample(v, header.maxval);
    }
    Ok(())
}

fn decode_graymap(data: &[u8], header: PnmHeader, rgb: &mut [u8]) -> Result<(), G2gError> {
    let sample_bytes = if header.maxval < 256 { 1 } else { 2 };
    let pixels = (header.width as usize)
        .checked_mul(header.height as usize)
        .ok_or(G2gError::CapsMismatch)?;
    let body = data.get(header.raster..).ok_or(G2gError::CapsMismatch)?;
    let need = pixels
        .checked_mul(sample_bytes)
        .ok_or(G2gError::CapsMismatch)?;
    if body.len() < need {
        return Err(G2gError::CapsMismatch);
    }
    for i in 0..pixels {
        let g = if sample_bytes == 1 {
            scale_sample(body[i] as u32, header.maxval)
        } else {
            let o = i * 2;
            scale_sample(
                u16::from_be_bytes([body[o], body[o + 1]]) as u32,
                header.maxval,
            )
        };
        let o = i * 3;
        rgb[o] = g;
        rgb[o + 1] = g;
        rgb[o + 2] = g;
    }
    Ok(())
}

fn decode_bitmap_bin(data: &[u8], header: PnmHeader, rgb: &mut [u8]) -> Result<(), G2gError> {
    let w = header.width as usize;
    let h = header.height as usize;
    let row = w.div_ceil(8);
    let body = data.get(header.raster..).ok_or(G2gError::CapsMismatch)?;
    let need = row.checked_mul(h).ok_or(G2gError::CapsMismatch)?;
    if body.len() < need {
        return Err(G2gError::CapsMismatch);
    }
    for y in 0..h {
        for x in 0..w {
            let bit = body[y * row + x / 8] & (0x80 >> (x % 8));
            // PBM: 1 is black, 0 is white.
            let v = if bit == 0 { 255 } else { 0 };
            let o = (y * w + x) * 3;
            rgb[o] = v;
            rgb[o + 1] = v;
            rgb[o + 2] = v;
        }
    }
    Ok(())
}

fn decode_ascii_pixmap(data: &[u8], header: PnmHeader, rgb: &mut [u8]) -> Result<(), G2gError> {
    let mut cur = Cursor {
        data,
        pos: header.raster,
    };
    for px in rgb.iter_mut() {
        let v = cur.token_u32()?.ok_or(G2gError::CapsMismatch)?;
        *px = scale_sample(v, header.maxval);
    }
    Ok(())
}

fn decode_ascii_gray(data: &[u8], header: PnmHeader, rgb: &mut [u8]) -> Result<(), G2gError> {
    let pixels = rgb.len() / 3;
    let mut cur = Cursor {
        data,
        pos: header.raster,
    };
    for i in 0..pixels {
        let g = scale_sample(
            cur.token_u32()?.ok_or(G2gError::CapsMismatch)?,
            header.maxval,
        );
        let o = i * 3;
        rgb[o] = g;
        rgb[o + 1] = g;
        rgb[o + 2] = g;
    }
    Ok(())
}

fn decode_ascii_bitmap(data: &[u8], header: PnmHeader, rgb: &mut [u8]) -> Result<(), G2gError> {
    let pixels = rgb.len() / 3;
    let mut cur = Cursor {
        data,
        pos: header.raster,
    };
    for i in 0..pixels {
        let bit = cur.token_bit()?.ok_or(G2gError::CapsMismatch)?;
        let v = if bit == 0 { 255 } else { 0 };
        let o = i * 3;
        rgb[o] = v;
        rgb[o + 1] = v;
        rgb[o + 2] = v;
    }
    Ok(())
}

/// Encode packed RGB or RGBA to a P6 (binary) or P3 (ASCII) pixmap.
pub(crate) fn encode_pnm(
    pixels: &[u8],
    format: RawVideoFormat,
    width: u32,
    height: u32,
    ascii: bool,
) -> Result<Vec<u8>, G2gError> {
    let samples = match format {
        RawVideoFormat::Rgb8 => 3,
        RawVideoFormat::Rgba8 => 4,
        _ => return Err(G2gError::CapsMismatch),
    };
    let needed = packed_byte_size(width, height, samples)?;
    if pixels.len() < needed {
        return Err(G2gError::CapsMismatch);
    }
    if ascii {
        encode_ascii(&pixels[..needed], samples, width, height)
    } else {
        encode_binary(&pixels[..needed], samples, width, height)
    }
}

fn encode_binary(
    pixels: &[u8],
    samples: usize,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, G2gError> {
    let rgb_len = packed_byte_size(width, height, 3)?;
    let mut out = String::new();
    let _ = write!(out, "P6\n{width} {height}\n255\n");
    let mut buf = out.into_bytes();
    buf.reserve(rgb_len);
    if samples == 3 {
        buf.extend_from_slice(pixels);
    } else {
        for px in pixels.as_chunks::<4>().0 {
            buf.extend_from_slice(&px[..3]);
        }
    }
    Ok(buf)
}

fn encode_ascii(
    pixels: &[u8],
    samples: usize,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, G2gError> {
    let mut out = String::new();
    let _ = write!(out, "P3\n{width} {height}\n255\n");
    let mut col = 0usize;
    for px in pixels.chunks_exact(samples) {
        for c in &px[..3] {
            let digits = if *c >= 100 {
                3
            } else if *c >= 10 {
                2
            } else {
                1
            };
            if col > 0 && col + 1 + digits > MAX_ASCII_LINE {
                out.push('\n');
                col = 0;
            } else if col > 0 {
                out.push(' ');
                col += 1;
            }
            let _ = write!(out, "{c}");
            col += digits;
        }
    }
    out.push('\n');
    Ok(out.into_bytes())
}

/// Encodes packed RGB / RGBA raw video into a Netpbm pixmap.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::pnm::PnmEnc;
///
/// let enc = PnmEnc::new().with_ascii(true);
/// ```
#[derive(Debug)]
pub struct PnmEnc {
    ascii: bool,
    format: RawVideoFormat,
    width: u32,
    height: u32,
    framerate: Rate,
    sequence: u64,
    caps_sent: bool,
    configured: bool,
}

impl Default for PnmEnc {
    fn default() -> Self {
        Self::new()
    }
}

impl PnmEnc {
    pub fn new() -> Self {
        Self {
            ascii: false,
            format: RawVideoFormat::Rgb8,
            width: 0,
            height: 0,
            framerate: Rate::Any,
            sequence: 0,
            caps_sent: false,
            configured: false,
        }
    }

    /// Write ASCII P3 instead of binary P6 (`ascii=`).
    pub fn with_ascii(mut self, ascii: bool) -> Self {
        self.ascii = ascii;
        self
    }

    fn input_alternatives() -> Vec<Caps> {
        let raw = |format| Caps::RawVideo {
            format,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        Vec::from([raw(RawVideoFormat::Rgb8), raw(RawVideoFormat::Rgba8)])
    }

    fn output_caps(&self) -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::Pnm,
            width: Dim::Fixed(self.width),
            height: Dim::Fixed(self.height),
            framerate: self.framerate.clone(),
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        }
    }
}

impl AsyncElement for PnmEnc {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    #[cfg(feature = "metadata")]
    fn meta_transform(&self) -> Option<g2g_core::meta::Transform> {
        Some(g2g_core::meta::Transform::Encode)
    }

    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        for alt in Self::input_alternatives() {
            if let Ok(c) = upstream_caps.intersect(&alt) {
                return Ok(c);
            }
        }
        Err(G2gError::CapsMismatch)
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::RawVideo {
                format: RawVideoFormat::Rgba8 | RawVideoFormat::Rgb8,
                width,
                height,
                framerate,
                ..
            } => CapsSet::one(Caps::CompressedVideo {
                codec: VideoCodec::Pnm,
                width: width.clone(),
                height: height.clone(),
                framerate: framerate.clone(),
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            }),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let Caps::RawVideo {
            format,
            width,
            height,
            framerate,
            ..
        } = absolute_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if !matches!(format, RawVideoFormat::Rgba8 | RawVideoFormat::Rgb8) {
            return Err(G2gError::CapsMismatch);
        }
        let (Dim::Fixed(w), Dim::Fixed(h)) = (width, height) else {
            return Err(G2gError::CapsMismatch);
        };
        packed_byte_size(*w, *h, 3)?;
        self.format = *format;
        self.width = *w;
        self.height = *h;
        self.framerate = framerate.clone();
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "PNM image encoder",
            "Codec/Encoder/Image",
            "Encodes raw RGB / RGBA video to a Netpbm pixmap",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        PNMENC_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "ascii" => self.ascii = value.as_bool().ok_or(PropError::Type)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "ascii" => Some(PropValue::Bool(self.ascii)),
            _ => None,
        }
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let slice = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    let pnm = encode_pnm(slice, self.format, self.width, self.height, self.ascii)?;
                    if !self.caps_sent {
                        out.push(PipelinePacket::CapsChanged(self.output_caps()))
                            .await?;
                        self.caps_sent = true;
                    }
                    let encoded = Frame::new(
                        MemoryDomain::System(SystemSlice::from_boxed(pnm.into_boxed_slice())),
                        frame.timing,
                        self.sequence,
                    );
                    self.sequence += 1;
                    out.push(PipelinePacket::DataFrame(encoded)).await?;
                }
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

static PNMENC_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "ascii",
    PropKind::Bool,
    "Encoding in ASCII rather than binary",
)
.with_default("false")];

impl PadTemplates for PnmEnc {
    fn pad_templates() -> Vec<PadTemplate> {
        let out = Caps::CompressedVideo {
            codec: VideoCodec::Pnm,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        Vec::from([
            PadTemplate::sink(CapsSet::from_alternatives(Self::input_alternatives())),
            PadTemplate::source(CapsSet::one(out)),
        ])
    }
}

/// Decodes Netpbm stills into packed RGB video.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::pnm::PnmDec;
///
/// let dec = PnmDec::new();
/// ```
#[derive(Debug)]
pub struct PnmDec {
    framerate: Rate,
    assembler: ImageAssembler,
    output: StillImageOutput,
    configured: bool,
}

impl Default for PnmDec {
    fn default() -> Self {
        Self::new()
    }
}

impl PnmDec {
    pub fn new() -> Self {
        Self {
            framerate: Rate::Any,
            assembler: ImageAssembler::default(),
            output: StillImageOutput::default(),
            configured: false,
        }
    }

    fn input_template() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::Pnm,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        }
    }
}

impl AsyncElement for PnmDec {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&Self::input_template())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::CompressedVideo {
                codec: VideoCodec::Pnm,
                width,
                height,
                framerate,
                ..
            } => CapsSet::one(Caps::RawVideo {
                format: RawVideoFormat::Rgb8,
                width: width.clone(),
                height: height.clone(),
                framerate: framerate.clone(),
                interlace: g2g_core::Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            }),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let Caps::CompressedVideo {
            codec: VideoCodec::Pnm,
            framerate,
            ..
        } = absolute_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        self.framerate = framerate.clone();
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "PNM image decoder",
            "Codec/Decoder/Image",
            "Decodes Netpbm stills to packed RGB",
            "g2g",
        )
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let slice = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    for image in self.assembler.push(slice, pnm_frame_length)? {
                        let (pixels, w, h) = decode_pnm(&image)?;
                        self.output
                            .push(
                                out,
                                pixels,
                                RawVideoFormat::Rgb8,
                                (w, h),
                                &self.framerate,
                                frame.timing,
                            )
                            .await?;
                    }
                }
                PipelinePacket::CapsChanged(_) => {}
                PipelinePacket::Flush => {
                    self.assembler.reset();
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Eos => self.assembler.finish()?,
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for PnmDec {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::one(Self::input_template())),
            PadTemplate::source(CapsSet::one(Caps::RawVideo {
                format: RawVideoFormat::Rgb8,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
                interlace: g2g_core::Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            })),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb2() -> Vec<u8> {
        vec![10, 20, 30, 40, 50, 60]
    }

    #[test]
    fn binary_roundtrip() {
        let encoded = encode_pnm(&rgb2(), RawVideoFormat::Rgb8, 2, 1, false).unwrap();
        assert!(encoded.starts_with(b"P6\n2 1\n255\n"));
        let (out, w, h) = decode_pnm(&encoded).unwrap();
        assert_eq!((w, h), (2, 1));
        assert_eq!(out, rgb2());
    }

    #[test]
    fn ascii_roundtrip() {
        let encoded = encode_pnm(&rgb2(), RawVideoFormat::Rgb8, 2, 1, true).unwrap();
        assert!(encoded.starts_with(b"P3\n"));
        let (out, w, h) = decode_pnm(&encoded).unwrap();
        assert_eq!((w, h), (2, 1));
        assert_eq!(out, rgb2());
    }

    #[test]
    fn graymap_expands_to_rgb() {
        let pgm = b"P5\n2 1\n255\n\x11\x22";
        let (out, w, h) = decode_pnm(pgm).unwrap();
        assert_eq!((w, h), (2, 1));
        assert_eq!(out, vec![0x11, 0x11, 0x11, 0x22, 0x22, 0x22]);
    }

    #[test]
    fn bitmap_expands_black_white() {
        // Two pixels, one byte 0b1000_0000: first black, second white.
        let pbm = b"P4\n2 1\n\x80";
        let (out, w, h) = decode_pnm(pbm).unwrap();
        assert_eq!((w, h), (2, 1));
        assert_eq!(out, vec![0, 0, 0, 255, 255, 255]);
    }

    #[test]
    fn ascii_bitmap_bits_may_run_together() {
        let (out, w, h) = decode_pnm(b"P1\n4 1\n0110\n").unwrap();
        assert_eq!((w, h), (4, 1));
        assert_eq!(out, vec![255, 255, 255, 0, 0, 0, 0, 0, 0, 255, 255, 255]);
    }

    #[test]
    fn small_maxval_scales_binary_samples() {
        let (out, _, _) = decode_pnm(b"P6\n1 1\n15\n\x0f\x00\x03").unwrap();
        assert_eq!(out, vec![255, 0, 51]);
    }

    #[test]
    fn ascii_lines_stay_within_70_chars() {
        let pixels = vec![200u8; 40 * 3];
        let encoded = encode_pnm(&pixels, RawVideoFormat::Rgb8, 40, 1, true).unwrap();
        for line in encoded.split(|b| *b == b'\n') {
            assert!(line.len() <= MAX_ASCII_LINE, "{}", line.len());
        }
    }

    #[test]
    fn frame_length_waits_for_body() {
        let prefix = b"P6\n2 1\n255\n";
        assert_eq!(pnm_frame_length(prefix).unwrap(), None);
        let mut full = prefix.to_vec();
        full.extend_from_slice(&rgb2());
        assert_eq!(pnm_frame_length(&full).unwrap(), Some(full.len()));
    }

    #[test]
    fn rejects_bogus_magic_and_zero_size() {
        assert!(pnm_frame_length(b"PX\n").is_err());
        assert!(decode_pnm(b"P6\n0 1\n255\n").is_err());
    }

    #[test]
    fn rgba_drops_alpha_on_encode() {
        let rgba = vec![1, 2, 3, 255, 4, 5, 6, 128];
        let encoded = encode_pnm(&rgba, RawVideoFormat::Rgba8, 2, 1, false).unwrap();
        let (out, _, _) = decode_pnm(&encoded).unwrap();
        assert_eq!(out, vec![1, 2, 3, 4, 5, 6]);
    }
}
