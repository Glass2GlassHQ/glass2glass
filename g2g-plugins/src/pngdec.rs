//! PNG decode element (PngDec, `png` feature): `CompressedVideo{Png}` in,
//! `RawVideo{Rgba8}` out, via the pure-Rust `png` crate. The GStreamer `pngdec`
//! analog.
//!
//! A PNG is one still image, so decode is stateless the way MJPEG's is: one
//! buffer in, one RGBA frame out, geometry recovered from the file's own `IHDR`
//! rather than from negotiation. A `CapsChanged` carries that geometry before
//! the first frame and on any change, so `multifilesrc` over a directory of
//! differently sized stills stays correct.
//!
//! Everything is normalized to 8-bit RGBA: palette and sub-8-bit grayscale are
//! expanded, grayscale and RGB gain an opaque alpha, and 16-bit samples are
//! narrowed to their high byte (`Transformations::STRIP_16`), so a 16-bit PNG
//! decodes with a loss of precision rather than being rejected. An APNG decodes
//! to its first frame.
//!
//! The header is attacker-controlled, so geometry is bounded before any buffer
//! is sized (see [`crate::stillimage`]) and the `png` decoder runs under its own
//! allocation limit. System memory.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, Rate, RawVideoFormat, VideoCodec,
};

use png::{ColorType, Decoder, Limits, Transformations};

use crate::stillframe::{png_frame_length, ImageAssembler};
use crate::stillimage::{rgba_byte_size, StillImageOutput, MAX_IMAGE_BYTES};

/// Decodes PNG stills into raw RGBA video.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::pngdec::PngDec;
///
/// let element = PngDec::new();
/// ```
#[derive(Debug)]
pub struct PngDec {
    framerate: Rate,
    assembler: ImageAssembler,
    output: StillImageOutput,
    configured: bool,
}

impl Default for PngDec {
    fn default() -> Self {
        Self::new()
    }
}

impl PngDec {
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
            codec: VideoCodec::Png,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }
    }

    /// Decode one PNG file, returning `(rgba, width, height)`. Any malformed or
    /// out-of-budget input fails with `CapsMismatch` rather than allocating on
    /// the header's word.
    pub(crate) fn decode(png: &[u8]) -> Result<(Vec<u8>, u32, u32), G2gError> {
        let mut decoder = Decoder::new_with_limits(
            std::io::Cursor::new(png),
            Limits {
                bytes: MAX_IMAGE_BYTES,
            },
        );
        // EXPAND | STRIP_16 (palette / sub-byte gray to 8-bit samples, 16-bit
        // down to 8), plus ALPHA so a paletted image with a tRNS chunk keeps its
        // transparency instead of dropping it on the way to RGB.
        decoder
            .set_transformations(Transformations::normalize_to_color8() | Transformations::ALPHA);

        let header = decoder
            .read_header_info()
            .map_err(|_| G2gError::CapsMismatch)?;
        let (width, height) = (header.width, header.height);
        rgba_byte_size(width, height)?;

        let mut reader = decoder.read_info().map_err(|_| G2gError::CapsMismatch)?;
        let buffer_size = reader
            .output_buffer_size()
            .filter(|size| *size <= MAX_IMAGE_BYTES)
            .ok_or(G2gError::CapsMismatch)?;
        let mut raw = vec![0u8; buffer_size];
        let info = reader
            .next_frame(&mut raw)
            .map_err(|_| G2gError::CapsMismatch)?;
        if info.bit_depth != png::BitDepth::Eight {
            return Err(G2gError::CapsMismatch);
        }
        // An APNG's first frame may be smaller than the image header's canvas;
        // the geometry we announce has to be the frame actually written.
        rgba_byte_size(info.width, info.height)?;
        raw.truncate(info.buffer_size());
        let rgba = to_rgba8(&raw, info.color_type, info.width, info.height)?;
        Ok((rgba, info.width, info.height))
    }
}

/// Widen a decoded 8-bit PNG frame to packed RGBA. `Indexed` cannot reach here
/// (the `EXPAND` transformation resolves the palette), so it is a decode failure
/// rather than an unhandled case.
fn to_rgba8(src: &[u8], color: ColorType, width: u32, height: u32) -> Result<Vec<u8>, G2gError> {
    let pixels = (width as usize) * (height as usize);
    let samples = color.samples();
    if src.len() < pixels * samples {
        return Err(G2gError::CapsMismatch);
    }
    match color {
        ColorType::Rgba => Ok(src[..pixels * 4].to_vec()),
        ColorType::Rgb => Ok(crate::videoconvert::convert(
            src,
            RawVideoFormat::Rgb8,
            RawVideoFormat::Rgba8,
            width as usize,
            height as usize,
        )
        .into_vec()),
        ColorType::Grayscale | ColorType::GrayscaleAlpha => {
            let mut rgba = vec![0u8; pixels * 4];
            for (out, gray) in rgba
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(src.chunks_exact(samples))
            {
                out[0] = gray[0];
                out[1] = gray[0];
                out[2] = gray[0];
                out[3] = if samples == 2 { gray[1] } else { 0xff };
            }
            Ok(rgba)
        }
        ColorType::Indexed => Err(G2gError::CapsMismatch),
    }
}

impl AsyncElement for PngDec {
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
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::CompressedVideo {
                codec: VideoCodec::Png,
                width,
                height,
                framerate,
            } => CapsSet::one(Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
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
            codec: VideoCodec::Png,
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
            "PNG decoder",
            "Codec/Decoder/Image",
            "Decodes PNG stills to RGBA via the pure-Rust png crate",
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
                    for image in self.assembler.push(slice, png_frame_length)? {
                        let (pixels, w, h) = Self::decode(&image)?;
                        self.output
                            .push_rgba(out, pixels, w, h, &self.framerate, frame.timing)
                            .await?;
                    }
                }
                PipelinePacket::CapsChanged(_) => {}
                // A flushing seek restarts the byte stream, so a partly received
                // image is stale.
                PipelinePacket::Flush => {
                    self.assembler.reset();
                    out.push(PipelinePacket::Flush).await?;
                }
                // the runner forwards Eos after process(Eos) returns; re-emitting
                // it here races the sink's exit on the first one.
                PipelinePacket::Eos => self.assembler.finish()?,
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for PngDec {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::one(Self::input_template())),
            PadTemplate::source(CapsSet::one(Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
                interlace: g2g_core::Interlace::Any,
            })),
        ])
    }
}
