//! Motion-JPEG encode element (MjpegEnc, `mjpeg-encode` feature): packed
//! `RawVideo{Rgba8|Bgra8}` in, `CompressedVideo{Mjpeg}` out, via the pure-Rust
//! `jpeg-encoder` crate. The GStreamer `jpegenc` analog.
//!
//! Each frame encodes to an independent baseline JPEG (intra-only), so this is
//! the snapshot / thumbnail / low-latency-capture encoder and the inverse of
//! [`crate::mjpegdec::MjpegDec`]. Quality is builder-configurable; geometry is
//! fixed at configure. Input is packed 4-byte RGBA/BGRA (run a `VideoConvert`
//! ahead of it for planar sources).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, MemoryDomain,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, RawVideoFormat, Rate, VideoCodec,
};

use jpeg_encoder::{ColorType, Encoder};

/// Default JPEG quality (0..=100); 85 is a good size/quality default.
const DEFAULT_QUALITY: u8 = 85;

/// Encodes packed RGBA/BGRA raw video into a Motion-JPEG stream.
#[derive(Debug)]
pub struct MjpegEnc {
    quality: u8,
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

    fn input_alternatives() -> Vec<Caps> {
        let raw = |format| Caps::RawVideo {
            format,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        Vec::from([raw(RawVideoFormat::Rgba8), raw(RawVideoFormat::Bgra8)])
    }

    fn output_caps(&self) -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::Mjpeg,
            width: Dim::Fixed(self.width),
            height: Dim::Fixed(self.height),
            framerate: self.framerate.clone(),
        }
    }

    fn color_type(&self) -> ColorType {
        match self.format {
            RawVideoFormat::Bgra8 => ColorType::Bgra,
            _ => ColorType::Rgba,
        }
    }

    fn encode(&self, pixels: &[u8]) -> Result<Vec<u8>, G2gError> {
        if pixels.len() < self.width as usize * self.height as usize * 4 {
            return Err(G2gError::CapsMismatch);
        }
        let mut out = Vec::new();
        let encoder = Encoder::new(&mut out, self.quality);
        encoder
            .encode(pixels, self.width as u16, self.height as u16, self.color_type())
            .map_err(|_| G2gError::CapsMismatch)?;
        Ok(out)
    }
}

impl AsyncElement for MjpegEnc {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

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
                format: RawVideoFormat::Rgba8 | RawVideoFormat::Bgra8,
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
        let Caps::RawVideo { format, width, height, framerate } = absolute_caps else {
            return Err(G2gError::CapsMismatch);
        };
        if !matches!(format, RawVideoFormat::Rgba8 | RawVideoFormat::Bgra8) {
            return Err(G2gError::CapsMismatch);
        }
        let (Dim::Fixed(w), Dim::Fixed(h)) = (width, height) else {
            return Err(G2gError::CapsMismatch);
        };
        self.format = *format;
        self.width = *w;
        self.height = *h;
        self.framerate = framerate.clone();
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
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
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let jpeg = self.encode(slice.as_slice())?;
                    if !self.caps_sent {
                        out.push(PipelinePacket::CapsChanged(self.output_caps())).await?;
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
