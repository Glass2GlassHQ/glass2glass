//! PNG encode element (PngEnc, `png` feature): `RawVideo{Rgba8|Rgb8}` in,
//! `CompressedVideo{Png}` out, via the pure-Rust `png` crate. The GStreamer
//! `pngenc` analog.
//!
//! Each frame encodes to an independent lossless PNG, so this is the snapshot /
//! screenshot / lossless-still encoder and the inverse of
//! [`crate::pngdec::PngDec`]. RGBA and RGB map straight onto PNG's own colour
//! types, so neither is converted on the way in; anything else needs a
//! `videoconvert` ahead of it. Geometry is fixed at configure. System memory.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec, Rate, RawVideoFormat, VideoCodec,
};

use png::{ColorType, DeflateCompression, Encoder};

use crate::stillimage::rgba_byte_size;

/// zlib's compression range, which is what GStreamer's `pngenc` takes on its
/// `compression-level` property (`Z_NO_COMPRESSION` .. `Z_BEST_COMPRESSION`).
const MAX_COMPRESSION_LEVEL: u8 = 9;
/// GStreamer `pngenc`'s default compression level.
const DEFAULT_COMPRESSION_LEVEL: u8 = 6;

/// Encodes packed RGBA / RGB raw video into PNG stills.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::pngenc::PngEnc;
///
/// let enc = PngEnc::new().with_compression_level(9);
/// ```
#[derive(Debug)]
pub struct PngEnc {
    compression_level: u8,
    format: RawVideoFormat,
    width: u32,
    height: u32,
    framerate: Rate,
    sequence: u64,
    caps_sent: bool,
    configured: bool,
}

impl Default for PngEnc {
    fn default() -> Self {
        Self::new()
    }
}

impl PngEnc {
    pub fn new() -> Self {
        Self {
            compression_level: DEFAULT_COMPRESSION_LEVEL,
            format: RawVideoFormat::Rgba8,
            width: 0,
            height: 0,
            framerate: Rate::Any,
            sequence: 0,
            caps_sent: false,
            configured: false,
        }
    }

    /// Set the deflate compression level (0 = store, 9 = smallest).
    pub fn with_compression_level(mut self, level: u8) -> Self {
        self.compression_level = level.min(MAX_COMPRESSION_LEVEL);
        self
    }

    fn input_alternatives() -> Vec<Caps> {
        let raw = |format| Caps::RawVideo {
            format,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
        };
        Vec::from([raw(RawVideoFormat::Rgba8), raw(RawVideoFormat::Rgb8)])
    }

    fn output_caps(&self) -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::Png,
            width: Dim::Fixed(self.width),
            height: Dim::Fixed(self.height),
            framerate: self.framerate.clone(),
        }
    }

    fn encode(&self, pixels: &[u8]) -> Result<Vec<u8>, G2gError> {
        let (color, samples) = match self.format {
            RawVideoFormat::Rgb8 => (ColorType::Rgb, 3),
            RawVideoFormat::Rgba8 => (ColorType::Rgba, 4),
            // configure_pipeline accepts no other format.
            _ => return Err(G2gError::CapsMismatch),
        };
        let needed = (self.width as usize)
            .checked_mul(self.height as usize)
            .and_then(|pixels| pixels.checked_mul(samples))
            .ok_or(G2gError::CapsMismatch)?;
        if pixels.len() < needed {
            return Err(G2gError::CapsMismatch);
        }

        let mut out = Vec::new();
        let mut encoder = Encoder::new(&mut out, self.width, self.height);
        encoder.set_color(color);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_deflate_compression(deflate_compression(self.compression_level));
        let mut writer = encoder.write_header().map_err(|_| G2gError::CapsMismatch)?;
        writer
            .write_image_data(&pixels[..needed])
            .map_err(|_| G2gError::CapsMismatch)?;
        writer.finish().map_err(|_| G2gError::CapsMismatch)?;
        Ok(out)
    }
}

/// GStreamer's zlib `compression-level` onto the `png` crate's deflate setting.
/// Level 0 is zlib's "store", which the crate spells as its own variant.
fn deflate_compression(level: u8) -> DeflateCompression {
    match level {
        0 => DeflateCompression::NoCompression,
        n => DeflateCompression::Level(n.min(MAX_COMPRESSION_LEVEL)),
    }
}

impl AsyncElement for PngEnc {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    // A re-encode to compressed PNG drops pixel-derived meta (AnalyticsMeta);
    // opaque side-data (BlobMeta) rides through.
    #[cfg(feature = "metadata")]
    fn meta_transform(&self) -> Option<g2g_core::meta::Transform> {
        Some(g2g_core::meta::Transform::Encode)
    }

    /// Reads host memory, so it takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
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
                interlace: _,
            } => CapsSet::one(Caps::CompressedVideo {
                codec: VideoCodec::Png,
                width: width.clone(),
                height: height.clone(),
                framerate: framerate.clone(),
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
            interlace: _,
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
        rgba_byte_size(*w, *h)?;
        self.format = *format;
        self.width = *w;
        self.height = *h;
        self.framerate = framerate.clone();
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "PNG encoder",
            "Codec/Encoder/Image",
            "Encodes raw RGBA / RGB video to PNG via the pure-Rust png crate",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const COMPRESSION_LEVEL: PropertySpec = PropertySpec::new(
            "compression-level",
            PropKind::Uint,
            "PNG compression level, 0 (store) ..= 9 (smallest)",
        )
        .with_range("0", "9")
        .with_default("6");
        &[COMPRESSION_LEVEL]
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "compression-level" => {
                let level = value.as_uint().ok_or(PropError::Type)?;
                if level > MAX_COMPRESSION_LEVEL as u64 {
                    return Err(PropError::Value);
                }
                self.compression_level = level as u8;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "compression-level" => Some(PropValue::Uint(self.compression_level as u64)),
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
                    let png = self.encode(slice)?;
                    if !self.caps_sent {
                        out.push(PipelinePacket::CapsChanged(self.output_caps()))
                            .await?;
                        self.caps_sent = true;
                    }
                    let encoded = Frame::new(
                        MemoryDomain::System(SystemSlice::from_boxed(png.into_boxed_slice())),
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

impl PadTemplates for PngEnc {
    fn pad_templates() -> Vec<PadTemplate> {
        let out = Caps::CompressedVideo {
            codec: VideoCodec::Png,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        Vec::from([
            PadTemplate::sink(CapsSet::from_alternatives(Self::input_alternatives())),
            PadTemplate::source(CapsSet::one(out)),
        ])
    }
}
