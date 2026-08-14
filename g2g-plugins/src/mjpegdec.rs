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
//! I420 out of a YCbCr JPEG skips the RGBA intermediate: the decoder is asked for
//! the stream's own colorspace (an identity copy in its color-convert step) and
//! the interleaved YCbCr is subsampled straight to planar, with only a full ->
//! limited range scale per sample. Grayscale / CMYK / YCCK JPEGs have no such
//! identity path and still decode through RGBA.
//!
//! With the `mozjpeg` feature the `decoder=mozjpeg` property swaps zune-jpeg for
//! libjpeg-turbo's SIMD decoder; the default stays `zune` (pure Rust, no C).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec, Rate, RawVideoFormat, VideoCodec,
};

use zune_jpeg::zune_core::bytestream::ZCursor;
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;
use zune_jpeg::JpegDecoder;

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
    /// Last emitted geometry, so `CapsChanged` is sent only on change.
    out_dims: Option<(u32, u32)>,
    sequence: u64,
    configured: bool,
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
            out_dims: None,
            sequence: 0,
            configured: false,
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
        }
    }

    fn output_caps(&self, w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: self.out_format,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: self.framerate.clone(),
            interlace: g2g_core::Interlace::Any,
        }
    }

    /// Decode one JPEG access unit, returning `(pixels, width, height)` in the
    /// configured output format. I420 requires even dims: a YCbCr JPEG takes the
    /// direct planar path, anything else decodes via RGBA then the shared BT.601
    /// conversion (so either way the range matches `VideoConvert`).
    pub(crate) fn decode(&self, jpeg: &[u8]) -> Result<(Vec<u8>, u32, u32), G2gError> {
        #[cfg(feature = "mozjpeg")]
        if self.backend == JpegDecodeBackend::Mozjpeg {
            return self.decode_mozjpeg(jpeg);
        }
        self.decode_zune(jpeg)
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

/// Interleaved JPEG YCbCr (full-range BT.601) -> planar I420 in the limited range
/// the rest of the pipeline uses. Chroma is the average of each 2x2 block, taken
/// before the range scale, which is what the RGBA route does in RGB: both matrices
/// are BT.601, so the two paths agree to a rounding step. Even dims only.
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
            } => CapsSet::one(Caps::RawVideo {
                format: out_format,
                width: width.clone(),
                height: height.clone(),
                framerate: framerate.clone(),
                interlace: g2g_core::Interlace::Any,
            }),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let Caps::CompressedVideo {
            codec: VideoCodec::Mjpeg,
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
                    let slice = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    let (pixels, w, h) = self.decode(slice)?;
                    if self.out_dims != Some((w, h)) {
                        out.push(PipelinePacket::CapsChanged(self.output_caps(w, h)))
                            .await?;
                        self.out_dims = Some((w, h));
                    }
                    let decoded = Frame::new(
                        MemoryDomain::System(SystemSlice::from_boxed(pixels.into_boxed_slice())),
                        frame.timing,
                        self.sequence,
                    );
                    self.sequence += 1;
                    out.push(PipelinePacket::DataFrame(decoded)).await?;
                }
                PipelinePacket::CapsChanged(_) => {}
                // the runner forwards Eos after process(Eos) returns; re-emitting
                // it here races the sink's exit on the first one.
                PipelinePacket::Eos => {}
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
