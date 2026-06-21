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

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, RawVideoFormat, Rate,
    VideoCodec,
};

use zune_jpeg::zune_core::bytestream::ZCursor;
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;
use zune_jpeg::JpegDecoder;

/// Decodes a Motion-JPEG stream into raw video (RGBA8 or I420).
#[derive(Debug)]
pub struct MjpegDec {
    out_format: RawVideoFormat,
    framerate: Rate,
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
        }
    }

    /// Decode one JPEG access unit, returning `(pixels, width, height)` in the
    /// configured output format. I420 decodes via RGBA then the shared BT.601
    /// conversion (so its range matches `VideoConvert`); it requires even dims.
    fn decode(&self, jpeg: &[u8]) -> Result<(Vec<u8>, u32, u32), G2gError> {
        let opts = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGBA);
        let mut dec = JpegDecoder::new_with_options(ZCursor::new(jpeg), opts);
        let rgba = dec.decode().map_err(|_| G2gError::CapsMismatch)?;
        let info = dec.info().ok_or(G2gError::CapsMismatch)?;
        let (w, h) = (info.width as u32, info.height as u32);
        let pixels = match self.out_format {
            RawVideoFormat::Rgba8 => rgba,
            RawVideoFormat::I420 => {
                if w % 2 != 0 || h % 2 != 0 {
                    return Err(G2gError::CapsMismatch);
                }
                crate::videoconvert::convert(
                    &rgba,
                    RawVideoFormat::Rgba8,
                    RawVideoFormat::I420,
                    w as usize,
                    h as usize,
                )
                .into_vec()
            }
            _ => return Err(G2gError::CapsMismatch),
        };
        Ok((pixels, w, h))
    }
}

impl AsyncElement for MjpegDec {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&Self::input_template())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let out_format = self.out_format;
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match input {
            Caps::CompressedVideo { codec: VideoCodec::Mjpeg, width, height, framerate } => {
                CapsSet::one(Caps::RawVideo {
                    format: out_format,
                    width: width.clone(),
                    height: height.clone(),
                    framerate: framerate.clone(),
                })
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let Caps::CompressedVideo { codec: VideoCodec::Mjpeg, framerate, .. } = absolute_caps else {
            return Err(G2gError::CapsMismatch);
        };
        if !matches!(self.out_format, RawVideoFormat::Rgba8 | RawVideoFormat::I420) {
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
                    let (pixels, w, h) = self.decode(slice.as_slice())?;
                    if self.out_dims != Some((w, h)) {
                        out.push(PipelinePacket::CapsChanged(self.output_caps(w, h))).await?;
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
