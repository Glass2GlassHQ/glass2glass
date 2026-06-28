//! Pure-Rust AV1 decode element (Rav1dDec, `rav1d` feature): `CompressedVideo{Av1}`
//! in, `RawVideo{I420}` out, via `re_rav1d`, the Rust port of dav1d. Same codec, no
//! C: `re_rav1d` is a line-for-line safe-Rust reimplementation of the dav1d decoder
//! and re-exports dav1d-rs's safe API, so this element is `Dav1dDec` with the backend
//! swapped from libdav1d (FFI) to `re_rav1d` (Rust). It builds with no system deps
//! and no NASM (`default-features = false`), so it reaches the pure-Rust / wasm
//! targets that the libdav1d path cannot.
//!
//! The decoder is stateful (frame threading / reordering), so the send/drain
//! protocol is: hand each AV1 temporal unit to `send_data`, and on a `Try again`
//! drain the ready pictures via `get_picture` and push the pending data with
//! `send_pending_data`. Decoded geometry is recovered per picture, so a
//! `CapsChanged` carries it before the first frame and on any mid-stream change
//! (the source may negotiate `Any` dims). 8-bit 4:2:0 (the dominant AV1 profile)
//! is packed into planar I420 (matching the other decoders / `VideoConvert`);
//! 10/12-bit and 4:2:2 / 4:4:4 are rejected at decode (a follow-up). System
//! memory. Pure Rust, but it pays a speed cost versus libdav1d's hand-written asm.

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

use re_rav1d::{Decoder, Picture, PixelLayout, PlanarImageComponent};

/// Decoded pictures from one fed unit: each `(packed I420 pixels, (width, height))`.
type DecodedFrames = Vec<(Vec<u8>, (u32, u32))>;

/// Decodes an AV1 stream into planar I420 via the pure-Rust `re_rav1d` decoder.
pub struct Rav1dDec {
    decoder: Option<Decoder>,
    framerate: Rate,
    /// Last emitted geometry, so `CapsChanged` is sent only on change.
    out_dims: Option<(u32, u32)>,
    sequence: u64,
    configured: bool,
}

impl core::fmt::Debug for Rav1dDec {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // `re_rav1d::Decoder` is not Debug; report only the element's own state.
        f.debug_struct("Rav1dDec")
            .field("out_dims", &self.out_dims)
            .field("sequence", &self.sequence)
            .field("configured", &self.configured)
            .finish_non_exhaustive()
    }
}

impl Default for Rav1dDec {
    fn default() -> Self {
        Self::new()
    }
}

impl Rav1dDec {
    pub fn new() -> Self {
        Self { decoder: None, framerate: Rate::Any, out_dims: None, sequence: 0, configured: false }
    }

    fn input_template() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::Av1,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }
    }

    fn output_caps(&self, w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::I420,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: self.framerate.clone(),
        }
    }

    /// Feed one AV1 temporal unit and collect every picture now decodable, each as
    /// `(packed I420 pixels, (width, height))`. Drives the send/drain protocol so a
    /// frame-threading / reordering delay does not strand input (the `Try again`
    /// -> drain -> `send_pending_data` cycle).
    fn feed(decoder: &mut Decoder, unit: Vec<u8>) -> Result<DecodedFrames, G2gError> {
        let mut frames = Vec::new();
        let mut send = decoder.send_data(unit, None, None, None);
        loop {
            // Drain all pictures ready right now.
            loop {
                match decoder.get_picture() {
                    Ok(pic) => frames.push((Self::pack_i420(&pic)?, Self::last_dims(&pic))),
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

    /// Pack a decoded 8-bit 4:2:0 picture into tight planar I420 (Y then U then
    /// V), copying each plane row honoring its stride. Returns the packed pixels;
    /// geometry comes from [`last_dims`]. Rejects non-8-bit / non-4:2:0 pictures.
    fn pack_i420(pic: &Picture) -> Result<Vec<u8>, G2gError> {
        if pic.pixel_layout() != PixelLayout::I420 || pic.bit_depth() != 8 {
            return Err(G2gError::CapsMismatch); // only 8-bit 4:2:0 for now
        }
        let (w, h) = (pic.width(), pic.height());
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let mut out = Vec::with_capacity((w * h + 2 * cw * ch) as usize);
        for (comp, pw, ph) in [
            (PlanarImageComponent::Y, w, h),
            (PlanarImageComponent::U, cw, ch),
            (PlanarImageComponent::V, cw, ch),
        ] {
            let plane = pic.plane(comp);
            let (stride, _) = pic.plane_data_geometry(comp);
            let bytes: &[u8] = &plane;
            for row in 0..ph {
                let start = (row * stride) as usize;
                let end = start + pw as usize;
                // The plane buffer is stride*height; a row never overruns it.
                out.extend_from_slice(bytes.get(start..end).ok_or(G2gError::CapsMismatch)?);
            }
        }
        Ok(out)
    }

    /// The decoded geometry of `pic` (its width and height), for the `CapsChanged`.
    fn last_dims(pic: &Picture) -> (u32, u32) {
        (pic.width(), pic.height())
    }
}

impl AsyncElement for Rav1dDec {
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
            "AV1 decoder (rav1d)",
            "Codec/Decoder/Video",
            "Decodes AV1 to I420 via the pure-Rust re_rav1d",
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
                    for (pixels, (w, h)) in frames {
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

impl PadTemplates for Rav1dDec {
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
