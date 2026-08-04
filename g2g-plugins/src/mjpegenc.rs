//! Motion-JPEG encode element (MjpegEnc, `mjpeg-encode` feature):
//! `RawVideo{Rgba8|Bgra8|I420}` in, `CompressedVideo{Mjpeg}` out, via the
//! pure-Rust `jpeg-encoder` crate. The GStreamer `jpegenc` analog.
//!
//! Each frame encodes to an independent baseline JPEG (intra-only), so this is
//! the snapshot / thumbnail / low-latency-capture encoder and the inverse of
//! [`crate::mjpegdec::MjpegDec`]. Quality is builder-configurable; geometry is
//! fixed at configure. Packed RGBA/BGRA encodes directly; planar I420 (even dims,
//! BT.601 limited range) converts to RGBA first via the shared `VideoConvert`
//! path, so a decoder can feed it without an intervening `VideoConvert`.
//!
//! With the `mozjpeg` feature the `encoder=mozjpeg` property swaps `jpeg-encoder`
//! for libjpeg-turbo / mozjpeg; the default stays `jpeg-encoder` (pure Rust, no C).

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

use jpeg_encoder::{ColorType, Encoder};

/// Default JPEG quality (0..=100); 85 is a good size/quality default.
const DEFAULT_QUALITY: u8 = 85;

/// Which JPEG encoder does the work (`mozjpeg` feature).
#[cfg(feature = "mozjpeg")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum JpegEncodeBackend {
    /// Pure-Rust `jpeg-encoder`, the default.
    #[default]
    JpegEncoder,
    /// libjpeg-turbo / mozjpeg through FFI.
    Mozjpeg,
}

/// Encodes packed RGBA/BGRA raw video into a Motion-JPEG stream.
#[derive(Debug)]
pub struct MjpegEnc {
    quality: u8,
    #[cfg(feature = "mozjpeg")]
    backend: JpegEncodeBackend,
    format: RawVideoFormat,
    width: u32,
    height: u32,
    framerate: Rate,
    sequence: u64,
    caps_sent: bool,
    configured: bool,
}

impl Default for MjpegEnc {
    fn default() -> Self {
        Self::new()
    }
}

impl MjpegEnc {
    pub fn new() -> Self {
        Self {
            quality: DEFAULT_QUALITY,
            #[cfg(feature = "mozjpeg")]
            backend: JpegEncodeBackend::JpegEncoder,
            format: RawVideoFormat::Rgba8,
            width: 0,
            height: 0,
            framerate: Rate::Any,
            sequence: 0,
            caps_sent: false,
            configured: false,
        }
    }

    /// Set the JPEG quality (0..=100).
    pub fn with_quality(mut self, quality: u8) -> Self {
        self.quality = quality.min(100);
        self
    }

    /// Choose the encoder implementation (`mozjpeg` feature).
    #[cfg(feature = "mozjpeg")]
    pub fn with_backend(mut self, backend: JpegEncodeBackend) -> Self {
        self.backend = backend;
        self
    }

    fn input_alternatives() -> Vec<Caps> {
        let raw = |format| Caps::RawVideo {
            format,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        Vec::from([
            raw(RawVideoFormat::Rgba8),
            raw(RawVideoFormat::Bgra8),
            raw(RawVideoFormat::I420),
        ])
    }

    fn output_caps(&self) -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::Mjpeg,
            width: Dim::Fixed(self.width),
            height: Dim::Fixed(self.height),
            framerate: self.framerate.clone(),
        }
    }

    fn encode(&self, pixels: &[u8]) -> Result<Vec<u8>, G2gError> {
        let (data, color) = self.packed_input(pixels)?;
        #[cfg(feature = "mozjpeg")]
        if self.backend == JpegEncodeBackend::Mozjpeg {
            return self.encode_mozjpeg(&data, color);
        }
        let mut out = Vec::new();
        let encoder = Encoder::new(&mut out, self.quality);
        encoder
            .encode(&data, self.width as u16, self.height as u16, color)
            .map_err(|_| G2gError::CapsMismatch)?;
        Ok(out)
    }

    /// Validate the input frame and hand back packed 4-byte pixels plus their
    /// channel order, the form both encoders take.
    fn packed_input<'p>(
        &self,
        pixels: &'p [u8],
    ) -> Result<(alloc::borrow::Cow<'p, [u8]>, ColorType), G2gError> {
        let (w, h) = (self.width as usize, self.height as usize);
        // I420 converts to packed RGBA first (shared BT.601 path); packed inputs
        // map straight to a jpeg-encoder ColorType.
        let (data, color): (alloc::borrow::Cow<[u8]>, ColorType) = match self.format {
            RawVideoFormat::I420 => {
                if pixels.len() < w * h * 3 / 2 {
                    return Err(G2gError::CapsMismatch);
                }
                let rgba = crate::videoconvert::convert(
                    pixels,
                    RawVideoFormat::I420,
                    RawVideoFormat::Rgba8,
                    w,
                    h,
                );
                (alloc::borrow::Cow::Owned(rgba.into_vec()), ColorType::Rgba)
            }
            RawVideoFormat::Bgra8 => {
                if pixels.len() < w * h * 4 {
                    return Err(G2gError::CapsMismatch);
                }
                (alloc::borrow::Cow::Borrowed(pixels), ColorType::Bgra)
            }
            _ => {
                if pixels.len() < w * h * 4 {
                    return Err(G2gError::CapsMismatch);
                }
                (alloc::borrow::Cow::Borrowed(pixels), ColorType::Rgba)
            }
        };
        Ok((data, color))
    }

    /// libjpeg-turbo / mozjpeg encode. Like its decode side, libjpeg signals
    /// fatal errors by unwinding out of the C call, so that is caught here.
    #[cfg(feature = "mozjpeg")]
    fn encode_mozjpeg(&self, pixels: &[u8], color: ColorType) -> Result<Vec<u8>, G2gError> {
        use mozjpeg::{ColorSpace, Compress};

        let in_color = match color {
            ColorType::Bgra => ColorSpace::JCS_EXT_BGRA,
            _ => ColorSpace::JCS_EXT_RGBA,
        };
        let compress = || -> Result<Vec<u8>, G2gError> {
            let mut comp = Compress::new(in_color);
            comp.set_size(self.width as usize, self.height as usize);
            comp.set_quality(self.quality as f32);
            let mut started = comp
                .start_compress(Vec::new())
                .map_err(|_| G2gError::CapsMismatch)?;
            started
                .write_scanlines(pixels)
                .map_err(|_| G2gError::CapsMismatch)?;
            started.finish().map_err(|_| G2gError::CapsMismatch)
        };
        match std::panic::catch_unwind(compress) {
            Ok(result) => result,
            Err(_) => Err(G2gError::CapsMismatch),
        }
    }
}

impl AsyncElement for MjpegEnc {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    // M759: a re-encode to compressed JPEG drops pixel-derived meta
    // (AnalyticsMeta); opaque side-data (BlobMeta) rides through.
    #[cfg(feature = "metadata")]
    fn meta_transform(&self) -> Option<g2g_core::meta::Transform> {
        Some(g2g_core::meta::Transform::Encode)
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
                format: RawVideoFormat::Rgba8 | RawVideoFormat::Bgra8 | RawVideoFormat::I420,
                width,
                height,
                framerate,
            } => CapsSet::one(Caps::CompressedVideo {
                codec: VideoCodec::Mjpeg,
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
        } = absolute_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if !matches!(
            format,
            RawVideoFormat::Rgba8 | RawVideoFormat::Bgra8 | RawVideoFormat::I420
        ) {
            return Err(G2gError::CapsMismatch);
        }
        let (Dim::Fixed(w), Dim::Fixed(h)) = (width, height) else {
            return Err(G2gError::CapsMismatch);
        };
        // I420 chroma is 2x2 subsampled; the shared conversion needs even dims.
        if *format == RawVideoFormat::I420 && (w % 2 != 0 || h % 2 != 0) {
            return Err(G2gError::CapsMismatch);
        }
        self.format = *format;
        self.width = *w;
        self.height = *h;
        self.framerate = framerate.clone();
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "JPEG / MJPEG encoder",
            "Codec/Encoder/Video",
            "Encodes raw video to JPEG / Motion-JPEG via jpeg-encoder",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const QUALITY: PropertySpec =
            PropertySpec::new("quality", PropKind::Uint, "JPEG quality, 0..=100")
                .with_range("0", "100")
                .with_default("85");
        #[cfg(feature = "mozjpeg")]
        const ENCODER: PropertySpec =
            PropertySpec::new("encoder", PropKind::Str, "JPEG encoder implementation")
                .with_enum_values("jpeg-encoder | mozjpeg")
                .with_default("jpeg-encoder");
        #[cfg(not(feature = "mozjpeg"))]
        const PROPS: &[PropertySpec] = &[QUALITY];
        #[cfg(feature = "mozjpeg")]
        const PROPS: &[PropertySpec] = &[QUALITY, ENCODER];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "quality" => {
                self.quality = (value.as_uint().ok_or(PropError::Type)? as u8).min(100);
                Ok(())
            }
            #[cfg(feature = "mozjpeg")]
            "encoder" => {
                self.backend = match value.as_str().ok_or(PropError::Type)? {
                    "jpeg-encoder" => JpegEncodeBackend::JpegEncoder,
                    "mozjpeg" => JpegEncodeBackend::Mozjpeg,
                    _ => return Err(PropError::Value),
                };
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "quality" => Some(PropValue::Uint(self.quality as u64)),
            #[cfg(feature = "mozjpeg")]
            "encoder" => Some(PropValue::Str(
                match self.backend {
                    JpegEncodeBackend::JpegEncoder => "jpeg-encoder",
                    JpegEncodeBackend::Mozjpeg => "mozjpeg",
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
                    let Some(slice) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let jpeg = self.encode(slice)?;
                    if !self.caps_sent {
                        out.push(PipelinePacket::CapsChanged(self.output_caps()))
                            .await?;
                        self.caps_sent = true;
                    }
                    let encoded = Frame::new(
                        MemoryDomain::System(SystemSlice::from_boxed(jpeg.into_boxed_slice())),
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

impl PadTemplates for MjpegEnc {
    fn pad_templates() -> Vec<PadTemplate> {
        let out = Caps::CompressedVideo {
            codec: VideoCodec::Mjpeg,
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
