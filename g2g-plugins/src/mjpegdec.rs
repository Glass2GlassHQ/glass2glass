//! Motion-JPEG decode element (MjpegDec, `mjpeg` feature): `CompressedVideo{Mjpeg}`
//! in, `RawVideo{Rgba8}` out, via the pure-Rust `zune-jpeg` decoder.
//!
//! Each MJPEG access unit is an independent baseline JPEG, so decode is
//! stateless: one frame in, one RGBA frame out. Geometry is recovered from the
//! JPEG headers per frame, so the real output `Caps` may be narrower than what
//! negotiation pinned (a webcam can advertise `Mjpeg` with `Any` dims). A
//! `CapsChanged` carries the decoded geometry before the first frame and on any
//! mid-stream size change.
//!
//! Output is RGBA8 by default; `with_output_format(RawVideoFormat::I420)` instead
//! emits planar 4:2:0 (BT.601 limited range, matching `VideoConvert` / the other
//! decoders) so a downstream video encoder needs no intervening `VideoConvert`.
//! I420 requires even dimensions. System memory.
//!
//! A JPEG carries no colour signalling, so the output caps say what JFIF
//! (ITU-T T.871) defines the samples to be: sRGB out of the RGBA route, and the
//! same over a BT.601 matrix at limited range out of the I420 one.
//!
//! I420 out of a YCbCr JPEG skips the RGBA intermediate: the decoder is asked for
//! the stream's own colorspace (an identity copy in its color-convert step) and
//! the interleaved YCbCr is subsampled straight to planar, with only a full ->
//! limited range scale per sample. Grayscale / CMYK / YCCK JPEGs have no such
//! identity path and still decode through RGBA.
//!
//! An access unit holding two JPEG images, each half the frame height upstream
//! declares, is an interlaced frame stored one image per field (`jpegparse`
//! pairs them): both decode and their rows are woven into one frame, the first
//! field on the even rows, and the caps say `Interlace::Interleaved`.
//!
//! With the `mozjpeg` feature the `decoder=mozjpeg` property swaps zune-jpeg for
//! libjpeg-turbo's SIMD decoder; the default stays `zune` (pure Rust, no C).

use core::future::Future;
use core::pin::Pin;

use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    Interlace, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec, Rate, RawVideoFormat, Segment, VideoCodec,
};

use zune_jpeg::zune_core::bytestream::ZCursor;
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;
use zune_jpeg::JpegDecoder;

use crate::stillframe::{
    jpeg_frame_length, jpeg_geometry, jpeg_header_segments, JPEG_HUFFMAN_TABLE, JPEG_MARKER_PREFIX,
    JPEG_SEGMENT_LENGTH, JPEG_SIGNATURE, JPEG_START_OF_SCAN,
};

#[derive(Debug)]
struct HuffmanTable {
    class_and_id: u8,
    code_counts: [u8; 16],
    symbols: &'static [u8],
}

// ITU-T T.81 Annex K.3 tables K.3 to K.6, which motion JPEG in AVI leaves out of every frame
const DEFAULT_HUFFMAN_TABLES: [HuffmanTable; 4] = [
    HuffmanTable {
        class_and_id: 0x00,
        code_counts: [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        symbols: &[
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
        ],
    },
    HuffmanTable {
        class_and_id: 0x01,
        code_counts: [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        symbols: &[
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
        ],
    },
    HuffmanTable {
        class_and_id: 0x10,
        code_counts: [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7D],
        symbols: &[
            0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51,
            0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1,
            0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18,
            0x19, 0x1A, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
            0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57,
            0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
            0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x92,
            0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
            0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
            0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8,
            0xD9, 0xDA, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2,
            0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA,
        ],
    },
    HuffmanTable {
        class_and_id: 0x11,
        code_counts: [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77],
        symbols: &[
            0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07,
            0x61, 0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xA1, 0xB1, 0xC1, 0x09,
            0x23, 0x33, 0x52, 0xF0, 0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34, 0xE1, 0x25,
            0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38,
            0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56,
            0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74,
            0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
            0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
            0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA,
            0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6,
            0xD7, 0xD8, 0xD9, 0xDA, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF2,
            0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA,
        ],
    },
];

fn default_huffman_segment() -> Vec<u8> {
    let tables_len: usize = DEFAULT_HUFFMAN_TABLES
        .iter()
        .map(|table| 1 + table.code_counts.len() + table.symbols.len())
        .sum();
    let mut segment = Vec::from([JPEG_MARKER_PREFIX, JPEG_HUFFMAN_TABLE]);
    segment.extend_from_slice(&((JPEG_SEGMENT_LENGTH + tables_len) as u16).to_be_bytes());
    for table in &DEFAULT_HUFFMAN_TABLES {
        segment.push(table.class_and_id);
        segment.extend_from_slice(&table.code_counts);
        segment.extend_from_slice(table.symbols);
    }
    segment
}

// the frame with the default tables put ahead of its first scan, when no DHT comes before it
fn with_default_huffman_tables(jpeg: &[u8]) -> Cow<'_, [u8]> {
    let scan_without_tables = jpeg_header_segments(jpeg)
        .take_while(|segment| segment.marker != JPEG_HUFFMAN_TABLE)
        .find(|segment| segment.marker == JPEG_START_OF_SCAN);
    let Some(scan) = scan_without_tables else {
        return Cow::Borrowed(jpeg);
    };
    let scan_marker_at = scan.code_at - 1;
    Cow::Owned(
        [
            &jpeg[..scan_marker_at],
            &default_huffman_segment(),
            &jpeg[scan_marker_at..],
        ]
        .concat(),
    )
}

#[derive(Debug)]
struct DecodedPicture {
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    interlace: Interlace,
}

// the rows of the two fields alternate in every plane, the first field's on top
fn weave_fields(
    format: RawVideoFormat,
    first_field: &[u8],
    second_field: &[u8],
    width: u32,
    field_height: u32,
) -> Result<Vec<u8>, G2gError> {
    let mut frame = Vec::with_capacity(first_field.len() + second_field.len());
    let mut plane_start = 0;
    for plane in 0..format.plane_count() {
        let stride = format
            .plane_stride(plane, width)
            .ok_or(G2gError::CapsMismatch)? as usize;
        let plane_bytes = format
            .plane_bytes(plane, width, field_height)
            .ok_or(G2gError::CapsMismatch)? as usize;
        let plane_end = plane_start + plane_bytes;
        let first_rows = first_field[plane_start..plane_end].chunks_exact(stride);
        let second_rows = second_field[plane_start..plane_end].chunks_exact(stride);
        for (first_row, second_row) in first_rows.zip(second_rows) {
            frame.extend_from_slice(first_row);
            frame.extend_from_slice(second_row);
        }
        plane_start = plane_end;
    }
    Ok(frame)
}

/// Which JPEG decoder does the work (`mozjpeg` feature).
#[cfg(feature = "mozjpeg")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum JpegDecodeBackend {
    /// Pure-Rust `zune-jpeg`, the default.
    #[default]
    Zune,
    /// libjpeg-turbo / mozjpeg through FFI.
    Mozjpeg,
}

/// Decodes a Motion-JPEG stream into raw video (RGBA8 or I420).
///
/// # Example
///
/// ```no_run
/// use g2g_core::RawVideoFormat;
/// use g2g_plugins::mjpegdec::MjpegDec;
///
/// let element = MjpegDec::new().with_output_format(RawVideoFormat::I420);
/// ```
#[derive(Debug)]
pub struct MjpegDec {
    out_format: RawVideoFormat,
    framerate: Rate,
    #[cfg(feature = "mozjpeg")]
    backend: JpegDecodeBackend,
    /// Last emitted caps, so `CapsChanged` is sent only on change.
    declared_caps: Option<Caps>,
    // the frame height upstream declares, twice a field image's when access units hold fields
    upstream_height: Option<u32>,
    sequence: u64,
    configured: bool,
    segment: Segment,
}

impl Default for MjpegDec {
    fn default() -> Self {
        Self::new()
    }
}

impl MjpegDec {
    pub fn new() -> Self {
        Self {
            out_format: RawVideoFormat::Rgba8,
            framerate: Rate::Any,
            #[cfg(feature = "mozjpeg")]
            backend: JpegDecodeBackend::Zune,
            declared_caps: None,
            upstream_height: None,
            sequence: 0,
            configured: false,
            segment: Segment::new(),
        }
    }

    /// Choose the decoded pixel format: `Rgba8` (default) or `I420`. Other
    /// formats are rejected at configure.
    pub fn with_output_format(mut self, format: RawVideoFormat) -> Self {
        self.out_format = format;
        self
    }

    /// Choose the decoder implementation (`mozjpeg` feature).
    #[cfg(feature = "mozjpeg")]
    pub fn with_backend(mut self, backend: JpegDecodeBackend) -> Self {
        self.backend = backend;
        self
    }

    fn input_template() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::Mjpeg,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        }
    }

    fn output_caps(&self, picture: &DecodedPicture) -> Caps {
        Caps::RawVideo {
            format: self.out_format,
            width: Dim::Fixed(picture.width),
            height: Dim::Fixed(picture.height),
            framerate: self.framerate.clone(),
            interlace: picture.interlace,
            colorimetry: decoded_colorimetry(self.out_format),
        }
    }

    fn decode_access_unit(&self, access_unit: &[u8]) -> Result<DecodedPicture, G2gError> {
        let holds_fields = jpeg_geometry(access_unit).is_some_and(|(_, image_height)| {
            self.upstream_height
                .is_some_and(|frame_height| image_height.checked_mul(2) == Some(frame_height))
        });
        if !holds_fields {
            let (pixels, width, height) = self.decode(access_unit)?;
            return Ok(DecodedPicture {
                pixels,
                width,
                height,
                interlace: Interlace::Any,
            });
        }
        let first_length = jpeg_frame_length(access_unit)?.ok_or(G2gError::CapsMismatch)?;
        let after_first = &access_unit[first_length..];
        let second_at = after_first
            .windows(JPEG_SIGNATURE.len())
            .position(|window| window == JPEG_SIGNATURE)
            .ok_or(G2gError::CapsMismatch)?;
        let (first_field, width, field_height) = self.decode(&access_unit[..first_length])?;
        let (second_field, second_width, second_height) = self.decode(&after_first[second_at..])?;
        if (second_width, second_height) != (width, field_height) {
            return Err(G2gError::CapsMismatch);
        }
        Ok(DecodedPicture {
            pixels: weave_fields(
                self.out_format,
                &first_field,
                &second_field,
                width,
                field_height,
            )?,
            width,
            height: field_height * 2,
            interlace: Interlace::Interleaved,
        })
    }

    /// Decode one JPEG access unit, returning `(pixels, width, height)` in the
    /// configured output format. I420 requires even dims: a YCbCr JPEG takes the
    /// direct planar path, anything else decodes via RGBA then the shared BT.601
    /// conversion (so either way the range matches `VideoConvert`).
    pub(crate) fn decode(&self, jpeg: &[u8]) -> Result<(Vec<u8>, u32, u32), G2gError> {
        let jpeg = with_default_huffman_tables(jpeg);
        #[cfg(feature = "mozjpeg")]
        if self.backend == JpegDecodeBackend::Mozjpeg {
            return self.decode_mozjpeg(&jpeg);
        }
        self.decode_zune(&jpeg)
    }

    fn decode_zune(&self, jpeg: &[u8]) -> Result<(Vec<u8>, u32, u32), G2gError> {
        if self.out_format == RawVideoFormat::I420 {
            // Ask for the stream's own colorspace: zune copies the planes out
            // without a color matrix when input and output match on 3 components.
            let opts = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::YCbCr);
            let mut dec = JpegDecoder::new_with_options(ZCursor::new(jpeg), opts);
            if dec.decode_headers().is_ok() && dec.input_colorspace() == Some(ColorSpace::YCbCr) {
                let info = dec.info().ok_or(G2gError::CapsMismatch)?;
                let (w, h) = (info.width as u32, info.height as u32);
                if w % 2 != 0 || h % 2 != 0 {
                    return Err(G2gError::CapsMismatch);
                }
                let ycbcr = dec.decode().map_err(|_| G2gError::CapsMismatch)?;
                return Ok((ycbcr_to_i420(&ycbcr, w as usize, h as usize)?, w, h));
            }
        }
        let opts = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGBA);
        let mut dec = JpegDecoder::new_with_options(ZCursor::new(jpeg), opts);
        let rgba = dec.decode().map_err(|_| G2gError::CapsMismatch)?;
        let info = dec.info().ok_or(G2gError::CapsMismatch)?;
        let (w, h) = (info.width as u32, info.height as u32);
        Ok((self.rgba_to_out(rgba, w, h)?, w, h))
    }

    /// Finish a decode that produced RGBA: hand it through, or run the shared
    /// conversion for an I420 output.
    fn rgba_to_out(&self, rgba: Vec<u8>, w: u32, h: u32) -> Result<Vec<u8>, G2gError> {
        match self.out_format {
            RawVideoFormat::Rgba8 => Ok(rgba),
            RawVideoFormat::I420 => {
                if !w.is_multiple_of(2) || !h.is_multiple_of(2) {
                    return Err(G2gError::CapsMismatch);
                }
                Ok(crate::videoconvert::convert(
                    &rgba,
                    RawVideoFormat::Rgba8,
                    RawVideoFormat::I420,
                    w as usize,
                    h as usize,
                    decoded_colorimetry(RawVideoFormat::I420),
                )
                .into_vec())
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    /// libjpeg-turbo / mozjpeg decode. Its error path unwinds out of the C call
    /// instead of returning, so a malformed frame is caught here and reported as
    /// a decode failure rather than taking the pipeline down.
    #[cfg(feature = "mozjpeg")]
    fn decode_mozjpeg(&self, jpeg: &[u8]) -> Result<(Vec<u8>, u32, u32), G2gError> {
        match std::panic::catch_unwind(|| self.decode_mozjpeg_inner(jpeg)) {
            Ok(result) => result,
            Err(_) => Err(G2gError::CapsMismatch),
        }
    }

    #[cfg(feature = "mozjpeg")]
    fn decode_mozjpeg_inner(&self, jpeg: &[u8]) -> Result<(Vec<u8>, u32, u32), G2gError> {
        use mozjpeg::{ColorSpace as MozColorSpace, Decompress};

        let dec = Decompress::new_mem(jpeg).map_err(|_| G2gError::CapsMismatch)?;
        let (w, h) = dec.size();
        let (dw, dh) = (w as u32, h as u32);
        // libjpeg's null color conversion, the mozjpeg equivalent of the zune
        // identity path: only available when the JPEG itself is YCbCr.
        if self.out_format == RawVideoFormat::I420 && dec.color_space() == MozColorSpace::JCS_YCbCr
        {
            if w % 2 != 0 || h % 2 != 0 {
                return Err(G2gError::CapsMismatch);
            }
            let mut started = dec
                .to_colorspace(MozColorSpace::JCS_YCbCr)
                .map_err(|_| G2gError::CapsMismatch)?;
            let ycbcr = started
                .read_scanlines::<u8>()
                .map_err(|_| G2gError::CapsMismatch)?;
            started.finish().map_err(|_| G2gError::CapsMismatch)?;
            return Ok((ycbcr_to_i420(&ycbcr, w, h)?, dw, dh));
        }
        let mut started = dec.rgba().map_err(|_| G2gError::CapsMismatch)?;
        let rgba = started
            .read_scanlines::<u8>()
            .map_err(|_| G2gError::CapsMismatch)?;
        started.finish().map_err(|_| G2gError::CapsMismatch)?;
        Ok((self.rgba_to_out(rgba, dw, dh)?, dw, dh))
    }
}

/// How the decoded samples map to colour. A JFIF file is full-range BT.601 over
/// sRGB ([`Colorimetry::JPEG`](g2g_core::Colorimetry::JPEG)); the RGBA route ends
/// in sRGB's identity "matrix", and the I420 route deliberately scales to the
/// limited range the rest of the pipeline uses.
fn decoded_colorimetry(out_format: RawVideoFormat) -> g2g_core::Colorimetry {
    match out_format {
        RawVideoFormat::I420 => g2g_core::Colorimetry {
            range: g2g_core::ColorRange::Limited,
            ..g2g_core::Colorimetry::JPEG
        },
        _ => g2g_core::Colorimetry::SRGB,
    }
}

/// Interleaved JPEG YCbCr (full-range BT.601) -> planar I420 in the limited range
/// the rest of the pipeline uses. Chroma is the average of each 2x2 block, taken
/// before the range scale, which is what the RGBA route does in RGB. On saturated
/// colours the two paths diverge by up to ~20 per sample (the RGBA route clips at
/// the RGB gamut boundary, this one never leaves YCbCr). Even dims only.
fn ycbcr_to_i420(src: &[u8], w: usize, h: usize) -> Result<Vec<u8>, G2gError> {
    let luma = w * h;
    if src.len() < luma * 3 {
        return Err(G2gError::CapsMismatch);
    }
    let mut dst = vec![0u8; luma + luma / 2];
    for (i, out) in dst[..luma].iter_mut().enumerate() {
        *out = limit_luma(src[i * 3] as i32);
    }
    let (cw, ch) = (w / 2, h / 2);
    for cy in 0..ch {
        for cx in 0..cw {
            let (mut cb, mut cr) = (0i32, 0i32);
            for dy in 0..2 {
                for dx in 0..2 {
                    let p = ((cy * 2 + dy) * w + cx * 2 + dx) * 3;
                    cb += src[p + 1] as i32;
                    cr += src[p + 2] as i32;
                }
            }
            let ci = cy * cw + cx;
            dst[luma + ci] = limit_chroma(cb / 4);
            dst[luma + luma / 4 + ci] = limit_chroma(cr / 4);
        }
    }
    Ok(dst)
}

/// Full-range luma (0..=255) -> limited range (16..=235).
fn limit_luma(y: i32) -> u8 {
    (16 + (y * 219 + 127) / 255).clamp(0, 255) as u8
}

/// Full-range chroma (0..=255) -> limited range (16..=240), around 128.
fn limit_chroma(c: i32) -> u8 {
    let n = (c - 128) * 224;
    // round to nearest, away from zero (integer division truncates toward zero).
    let scaled = (if n >= 0 { n + 127 } else { n - 127 }) / 255;
    (128 + scaled).clamp(0, 255) as u8
}

impl AsyncElement for MjpegDec {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Reads host memory, so it takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&Self::input_template())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let out_format = self.out_format;
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match input {
            Caps::CompressedVideo {
                codec: VideoCodec::Mjpeg,
                width,
                height,
                framerate,
                ..
            } => CapsSet::one(Caps::RawVideo {
                format: out_format,
                width: width.clone(),
                height: height.clone(),
                framerate: framerate.clone(),
                interlace: g2g_core::Interlace::Any,
                colorimetry: decoded_colorimetry(out_format),
            }),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let Caps::CompressedVideo {
            codec: VideoCodec::Mjpeg,
            height,
            framerate,
            ..
        } = absolute_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if !matches!(
            self.out_format,
            RawVideoFormat::Rgba8 | RawVideoFormat::I420
        ) {
            return Err(G2gError::CapsMismatch);
        }
        self.framerate = framerate.clone();
        self.upstream_height = match height {
            Dim::Fixed(height) => Some(*height),
            _ => None,
        };
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "JPEG / MJPEG decoder",
            "Codec/Decoder/Video",
            "Decodes JPEG / Motion-JPEG to RGBA or I420 via zune-jpeg",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const OUTPUT_FORMAT: PropertySpec = PropertySpec::new(
            "output-format",
            PropKind::Str,
            "decoded pixel format: rgba | i420",
        )
        .with_default("rgba");
        #[cfg(feature = "mozjpeg")]
        const DECODER: PropertySpec =
            PropertySpec::new("decoder", PropKind::Str, "JPEG decoder implementation")
                .with_enum_values("zune | mozjpeg")
                .with_default("zune");
        #[cfg(not(feature = "mozjpeg"))]
        const PROPS: &[PropertySpec] = &[OUTPUT_FORMAT];
        #[cfg(feature = "mozjpeg")]
        const PROPS: &[PropertySpec] = &[OUTPUT_FORMAT, DECODER];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "output-format" => {
                self.out_format = match value.as_str().ok_or(PropError::Type)? {
                    "rgba" | "RGBA" | "rgba8" => RawVideoFormat::Rgba8,
                    "i420" | "I420" => RawVideoFormat::I420,
                    _ => return Err(PropError::Value),
                };
                Ok(())
            }
            #[cfg(feature = "mozjpeg")]
            "decoder" => {
                self.backend = match value.as_str().ok_or(PropError::Type)? {
                    "zune" => JpegDecodeBackend::Zune,
                    "mozjpeg" => JpegDecodeBackend::Mozjpeg,
                    _ => return Err(PropError::Value),
                };
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "output-format" => Some(PropValue::Str(
                match self.out_format {
                    RawVideoFormat::I420 => "i420",
                    _ => "rgba",
                }
                .into(),
            )),
            #[cfg(feature = "mozjpeg")]
            "decoder" => Some(PropValue::Str(
                match self.backend {
                    JpegDecodeBackend::Zune => "zune",
                    JpegDecodeBackend::Mozjpeg => "mozjpeg",
                }
                .into(),
            )),
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
                    if !self.segment.presents(frame.timing.pts_ns) {
                        return Ok(());
                    }
                    let slice = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    let picture = self.decode_access_unit(slice)?;
                    let caps = self.output_caps(&picture);
                    if self.declared_caps.as_ref() != Some(&caps) {
                        out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                        self.declared_caps = Some(caps);
                    }
                    let mut decoded = Frame::new(
                        MemoryDomain::System(SystemSlice::from_boxed(
                            picture.pixels.into_boxed_slice(),
                        )),
                        frame.timing,
                        self.sequence,
                    );
                    decoded.meta = frame.meta;
                    self.sequence += 1;
                    out.push(PipelinePacket::DataFrame(decoded)).await?;
                }
                PipelinePacket::CapsChanged(_) => {}
                // the runner forwards Eos after process(Eos) returns; re-emitting
                // it here races the sink's exit on the first one.
                PipelinePacket::Eos => {}
                PipelinePacket::Segment(segment) => {
                    self.segment = segment;
                    out.push(PipelinePacket::Segment(segment)).await?;
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for MjpegDec {
    fn pad_templates() -> Vec<PadTemplate> {
        let out = |format| Caps::RawVideo {
            format,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: decoded_colorimetry(format),
        };
        Vec::from([
            PadTemplate::sink(CapsSet::one(Self::input_template())),
            PadTemplate::source(CapsSet::from_alternatives(Vec::from([
                out(RawVideoFormat::Rgba8),
                out(RawVideoFormat::I420),
            ]))),
        ])
    }
}
