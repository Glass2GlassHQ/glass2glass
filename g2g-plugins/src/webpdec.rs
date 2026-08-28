//! WebP decode element (WebPDec, `webp` feature): `CompressedVideo{WebP}` in,
//! `RawVideo{Rgba8}` out, via the pure-Rust `image-webp` crate. The GStreamer
//! `webpdec` analog.
//!
//! Both WebP bitstreams decode: lossy (VP8) and lossless (VP8L), simple or
//! extended container. A WebP is one still image, so decode is stateless the way
//! PNG's is: one buffer in, one RGBA frame out, geometry read from the file's own
//! header. An animated WebP decodes to its first frame only. Files without an
//! alpha channel are widened to opaque RGBA so the output format never varies.
//!
//! The header is attacker-controlled, so geometry is bounded before any buffer is
//! sized (see [`crate::stillimage`]). The crate's own memory limit is set too,
//! but it documents that some allocations still bypass it, so the geometry check
//! is what actually holds. System memory.
//!
//! There is no encoder here: the only pure-Rust WebP encoder (`image-webp`) does
//! VP8L lossless alone, with none of `webpenc`'s quality / speed / preset knobs.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec, Rate, RawVideoFormat, VideoCodec,
};

use image_webp::{UpsamplingMethod, WebPDecodeOptions, WebPDecoder};

use crate::stillframe::{ImageAssembler, MAX_ENCODED_BYTES};
use crate::stillimage::{rgba_byte_size, StillImageOutput, MAX_IMAGE_BYTES};
use crate::typefind::{riff_form, RIFF_FORM_OFFSET, RIFF_HEADER_LEN, RIFF_MAGIC, WEBP_MAGIC};

/// A WebP's length is its RIFF size field plus the header bytes that field
/// excludes (it counts everything after itself), so a byte stream that splits or
/// joins files is framed back into whole images. The size is the file's own word,
/// so it is held under `MAX_ENCODED_BYTES`.
fn webp_frame_length(data: &[u8]) -> Result<Option<usize>, G2gError> {
    if data.len() < RIFF_HEADER_LEN {
        // Not yet enough to tell a WebP from anything else.
        let seen = data.len().min(RIFF_MAGIC.len());
        if data[..seen] != RIFF_MAGIC[..seen] {
            return Err(G2gError::CapsMismatch);
        }
        return Ok(None);
    }
    if riff_form(data) != Some(WEBP_MAGIC) {
        return Err(G2gError::CapsMismatch);
    }
    let size: [u8; 4] = data[RIFF_MAGIC.len()..RIFF_FORM_OFFSET]
        .try_into()
        .map_err(|_| G2gError::CapsMismatch)?;
    let total = (u32::from_le_bytes(size) as usize)
        .checked_add(RIFF_FORM_OFFSET)
        .filter(|total| *total <= MAX_ENCODED_BYTES)
        .ok_or(G2gError::CapsMismatch)?;
    Ok((data.len() >= total).then_some(total))
}

/// Decodes WebP stills into raw RGBA video.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::webpdec::WebPDec;
///
/// let element = WebPDec::new();
/// ```
#[derive(Debug)]
pub struct WebPDec {
    /// `true` picks the nearest chroma sample instead of interpolating, the
    /// `dwebp -nofancy` trade: faster, slightly jagged edges. Lossy input only.
    no_fancy_upsampling: bool,
    framerate: Rate,
    assembler: ImageAssembler,
    output: StillImageOutput,
    configured: bool,
}

impl Default for WebPDec {
    fn default() -> Self {
        Self::new()
    }
}

impl WebPDec {
    pub fn new() -> Self {
        Self {
            no_fancy_upsampling: false,
            framerate: Rate::Any,
            assembler: ImageAssembler::default(),
            output: StillImageOutput::default(),
            configured: false,
        }
    }

    /// Use simple (nearest) chroma upsampling on lossy input instead of the
    /// bilinear default.
    pub fn with_no_fancy_upsampling(mut self, simple: bool) -> Self {
        self.no_fancy_upsampling = simple;
        self
    }

    fn input_template() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::WebP,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }
    }

    /// Decode one WebP file, returning `(rgba, width, height)`. Any malformed or
    /// out-of-budget input fails with `CapsMismatch` rather than allocating on
    /// the header's word.
    pub(crate) fn decode(&self, webp: &[u8]) -> Result<(Vec<u8>, u32, u32), G2gError> {
        let mut options = WebPDecodeOptions::default();
        options.lossy_upsampling = if self.no_fancy_upsampling {
            UpsamplingMethod::Simple
        } else {
            UpsamplingMethod::Bilinear
        };
        let mut decoder = WebPDecoder::new_with_options(std::io::Cursor::new(webp), options)
            .map_err(|_| G2gError::CapsMismatch)?;
        decoder.set_memory_limit(MAX_IMAGE_BYTES);

        let (width, height) = decoder.dimensions();
        rgba_byte_size(width, height)?;

        // The crate sizes its output at 4 bytes/pixel with alpha, 3 without, and
        // rejects any other buffer length outright.
        let buffer_size = decoder
            .output_buffer_size()
            .filter(|size| *size <= MAX_IMAGE_BYTES)
            .ok_or(G2gError::CapsMismatch)?;
        let mut raw = vec![0u8; buffer_size];
        decoder
            .read_image(&mut raw)
            .map_err(|_| G2gError::CapsMismatch)?;

        if decoder.has_alpha() {
            return Ok((raw, width, height));
        }
        let rgba = crate::videoconvert::convert(
            &raw,
            RawVideoFormat::Rgb8,
            RawVideoFormat::Rgba8,
            width as usize,
            height as usize,
        );
        Ok((rgba.into_vec(), width, height))
    }
}

impl AsyncElement for WebPDec {
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
                codec: VideoCodec::WebP,
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
            codec: VideoCodec::WebP,
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
            "WebP decoder",
            "Codec/Decoder/Image",
            "Decodes WebP stills to RGBA via the pure-Rust image-webp crate",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const NO_FANCY_UPSAMPLING: PropertySpec = PropertySpec::new(
            "no-fancy-upsampling",
            PropKind::Bool,
            "use simple (nearest) chroma upsampling on lossy input",
        )
        .with_default("false");
        &[NO_FANCY_UPSAMPLING]
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "no-fancy-upsampling" => {
                self.no_fancy_upsampling = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "no-fancy-upsampling" => Some(PropValue::Bool(self.no_fancy_upsampling)),
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
                    for image in self.assembler.push(slice, webp_frame_length)? {
                        let (pixels, w, h) = self.decode(&image)?;
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

impl PadTemplates for WebPDec {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A RIFF/WEBP shaped file whose size field covers the form type plus
    /// `payload` bytes.
    fn webp_bytes(payload: usize) -> Vec<u8> {
        let mut file = Vec::from(RIFF_MAGIC);
        let size = payload + WEBP_MAGIC.len();
        file.extend_from_slice(&(size as u32).to_le_bytes());
        file.extend_from_slice(&WEBP_MAGIC);
        file.extend_from_slice(&vec![0u8; payload]);
        file
    }

    #[test]
    fn frame_length_comes_from_the_riff_size() {
        let file = webp_bytes(58);
        assert_eq!(webp_frame_length(&file), Ok(Some(file.len())));
        for cut in 0..file.len() {
            assert_eq!(webp_frame_length(&file[..cut]), Ok(None), "prefix of {cut}");
        }
        // Trailing bytes do not extend the image.
        let mut two = file.clone();
        two.extend_from_slice(&file);
        assert_eq!(webp_frame_length(&two), Ok(Some(file.len())));
    }

    #[test]
    fn frame_length_refuses_a_non_webp_and_an_absurd_size() {
        assert_eq!(
            webp_frame_length(b"RIFF\0\0\0\0AVI "),
            Err(G2gError::CapsMismatch)
        );
        assert_eq!(webp_frame_length(b"nope"), Err(G2gError::CapsMismatch));
        let mut huge = Vec::from(RIFF_MAGIC);
        huge.extend_from_slice(&u32::MAX.to_le_bytes());
        huge.extend_from_slice(&WEBP_MAGIC);
        assert_eq!(webp_frame_length(&huge), Err(G2gError::CapsMismatch));
    }
}
