//! AV1 decode element (Dav1dDec, `dav1d` feature): `CompressedVideo{Av1}` in,
//! `RawVideo{I420}` out, via the `dav1d` crate's safe bindings to libdav1d (the C
//! AV1 decoder with hand-written assembly, the speed reference).
//!
//! The decoder is stateful (frame threading / reordering), so the send/drain
//! protocol is: hand each AV1 temporal unit to `send_data`, and on a `Try again`
//! drain the ready pictures via `get_picture` and push the pending data with
//! `send_pending_data`. Decoded geometry is recovered per picture, so a
//! `CapsChanged` carries it before the first frame and on any mid-stream change
//! (the source may negotiate `Any` dims). The decoded picture is packed into the
//! matching fully-planar [`RawVideoFormat`]: 4:2:0 / 4:2:2 / 4:4:4 at 8 / 10 /
//! 12-bit (`I420` / `I422` / `I444` and their `p10` / `p12` variants), the format
//! recovered per picture from dav1d's layout + bit depth and carried in the
//! `CapsChanged`. 10/12-bit samples pass through as the native little-endian
//! 2-byte words. Monochrome (I400) is rejected (no planar-YUV format for it).
//! System memory. NOT pure Rust (links libdav1d); for a pure-Rust AV1 decoder see
//! `rav1ddec.rs` (`Rav1dDec`, the `re_rav1d` port) behind the `rav1d` feature.

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

use dav1d::{Decoder, PixelLayout, PlanarImageComponent};

/// Decoded pictures from one fed unit: each `(format, packed pixels, (width, height))`.
type DecodedFrames = Vec<(RawVideoFormat, Vec<u8>, (u32, u32))>;

/// Decodes an AV1 stream into a fully-planar YUV format via libdav1d.
pub struct Dav1dDec {
    decoder: Option<Decoder>,
    framerate: Rate,
    /// Last emitted (format, width, height), so `CapsChanged` is sent only on change.
    out: Option<(RawVideoFormat, u32, u32)>,
    sequence: u64,
    configured: bool,
}

impl core::fmt::Debug for Dav1dDec {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // `dav1d::Decoder` is not Debug; report only the element's own state.
        f.debug_struct("Dav1dDec")
            .field("out", &self.out)
            .field("sequence", &self.sequence)
            .field("configured", &self.configured)
            .finish_non_exhaustive()
    }
}

impl Default for Dav1dDec {
    fn default() -> Self {
        Self::new()
    }
}

impl Dav1dDec {
    pub fn new() -> Self {
        Self { decoder: None, framerate: Rate::Any, out: None, sequence: 0, configured: false }
    }

    fn input_template() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::Av1,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }
    }

    fn output_caps(&self, format: RawVideoFormat, w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: self.framerate.clone(),
        }
    }

    /// Feed one AV1 temporal unit and collect every picture now decodable, each as
    /// `(format, packed pixels, (width, height))`. Drives the send/drain protocol
    /// so a frame-threading / reordering delay does not strand input (the `Try
    /// again` -> drain -> `send_pending_data` cycle).
    fn feed(decoder: &mut Decoder, unit: Vec<u8>) -> Result<DecodedFrames, G2gError> {
        let mut frames = Vec::new();
        let mut send = decoder.send_data(unit, None, None, None);
        loop {
            // Drain all pictures ready right now.
            loop {
                match decoder.get_picture() {
                    Ok(pic) => {
                        let format = pic_format(&pic)?;
                        frames.push((format, pack_planar(&pic, format)?, (pic.width(), pic.height())));
                    }
                    Err(e) if e.is_again() => break,
                    Err(_) => return Err(G2gError::CapsMismatch),
                }
            }
            match send {
                Ok(()) => break, // input fully consumed
                Err(e) if e.is_again() => send = decoder.send_pending_data(),
                Err(_) => return Err(G2gError::CapsMismatch),
            }
        }
        Ok(frames)
    }
}

/// The fully-planar [`RawVideoFormat`] matching a decoded picture's chroma layout
/// and bit depth. Rejects monochrome (I400), which has no planar-YUV format.
fn pic_format(pic: &dav1d::Picture) -> Result<RawVideoFormat, G2gError> {
    use RawVideoFormat as F;
    Ok(match (pic.pixel_layout(), pic.bit_depth()) {
        (PixelLayout::I420, 8) => F::I420,
        (PixelLayout::I420, 10) => F::I420p10,
        (PixelLayout::I420, 12) => F::I420p12,
        (PixelLayout::I422, 8) => F::I422,
        (PixelLayout::I422, 10) => F::I422p10,
        (PixelLayout::I422, 12) => F::I422p12,
        (PixelLayout::I444, 8) => F::I444,
        (PixelLayout::I444, 10) => F::I444p10,
        (PixelLayout::I444, 12) => F::I444p12,
        _ => return Err(G2gError::CapsMismatch),
    })
}

/// Pack a decoded picture into the tight planar layout of `format` (Y then U then
/// V), copying each plane row honoring its stride. The chroma plane dimensions and
/// per-sample byte size come from the format itself, so 4:2:0 / 4:2:2 / 4:4:4 at
/// 8 / 10 / 12-bit share one path; 10/12-bit samples are the native LE 2-byte words.
fn pack_planar(pic: &dav1d::Picture, format: RawVideoFormat) -> Result<Vec<u8>, G2gError> {
    let bps = format.bytes_per_sample();
    // `chroma_shift` is always `Some` here: `pic_format` only yields planar formats.
    let (hs, vs) = format.chroma_shift().ok_or(G2gError::CapsMismatch)?;
    let (w, h) = (pic.width(), pic.height());
    let (cw, ch) = (w.div_ceil(1 << hs), h.div_ceil(1 << vs));
    let mut out = Vec::with_capacity(((w * h + 2 * cw * ch) as usize) * bps);
    for (comp, pw, ph) in [
        (PlanarImageComponent::Y, w, h),
        (PlanarImageComponent::U, cw, ch),
        (PlanarImageComponent::V, cw, ch),
    ] {
        let plane = pic.plane(comp);
        let (stride, _) = pic.plane_data_geometry(comp);
        let bytes: &[u8] = &plane;
        let row_len = pw as usize * bps;
        for row in 0..ph {
            let start = (row * stride) as usize;
            let end = start + row_len;
            // The plane buffer is stride*height; a row never overruns it.
            out.extend_from_slice(bytes.get(start..end).ok_or(G2gError::CapsMismatch)?);
        }
    }
    Ok(out)
}

impl AsyncElement for Dav1dDec {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&Self::input_template())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::CompressedVideo { codec: VideoCodec::Av1, width, height, framerate } => {
                CapsSet::one(Caps::RawVideo {
                    format: RawVideoFormat::I420,
                    width: width.clone(),
                    height: height.clone(),
                    framerate: framerate.clone(),
                })
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let Caps::CompressedVideo { codec: VideoCodec::Av1, framerate, .. } = absolute_caps else {
            return Err(G2gError::CapsMismatch);
        };
        self.framerate = framerate.clone();
        self.decoder = Some(Decoder::new().map_err(|_| G2gError::CapsMismatch)?);
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "AV1 decoder (dav1d)",
            "Codec/Decoder/Video",
            "Decodes AV1 to I420 via libdav1d",
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
                    let decoder = self.decoder.as_mut().ok_or(G2gError::NotConfigured)?;
                    let unit = slice.as_slice().to_vec();
                    let frames = Self::feed(decoder, unit)?;
                    for (format, pixels, (w, h)) in frames {
                        if self.out != Some((format, w, h)) {
                            out.push(PipelinePacket::CapsChanged(self.output_caps(format, w, h)))
                                .await?;
                            self.out = Some((format, w, h));
                        }
                        let decoded = Frame::new(
                            MemoryDomain::System(SystemSlice::from_boxed(pixels.into_boxed_slice())),
                            frame.timing,
                            self.sequence,
                        );
                        self.sequence += 1;
                        out.push(PipelinePacket::DataFrame(decoded)).await?;
                    }
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

impl PadTemplates for Dav1dDec {
    fn pad_templates() -> Vec<PadTemplate> {
        let av1 = Self::input_template();
        let raw = Caps::RawVideo {
            format: RawVideoFormat::I420,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        Vec::from([PadTemplate::sink(CapsSet::one(av1)), PadTemplate::source(CapsSet::one(raw))])
    }
}
