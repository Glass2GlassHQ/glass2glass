//! Shared geometry bounds and packed-RGB(A) output for the still-image codec
//! elements (`PngDec`, `PngEnc`, `WebPDec`, `PnmDec`). The byte-stream framing
//! they also share is in [`crate::stillframe`], which the still-image parsers
//! use without a decoder.
//!
//! A PNG `IHDR` or a WebP `VP8X` header carries dimensions up to 2^31 / 2^24,
//! and both decoder crates size their output buffer from those numbers. The
//! header is attacker-controlled, so the geometry is checked against a fixed
//! budget here before anything allocates: a bogus 100000x100000 header fails the
//! parse instead of asking for 40 GB.
//!

use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    Caps, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket, Rate,
    RawVideoFormat,
};

/// Widest / tallest still image decoded. Above 8K in both axes, so no real
/// photo or screen capture hits it, and low enough that a single side cannot
/// carry the allocation on its own.
pub(crate) const MAX_IMAGE_DIMENSION: u32 = 16384;

/// Byte budget for one decoded image, and for the intermediate a decoder crate
/// allocates while producing it. Caps the width x height product that the
/// per-side bound alone would let through (16384 x 16384 RGBA is 1 GiB).
pub(crate) const MAX_IMAGE_BYTES: usize = 256 * 1024 * 1024;

/// Validate a decoded image's geometry and return its packed byte size at
/// `samples` bytes per pixel (4 for RGBA, 3 for RGB).
///
/// Rejects a zero side (nothing to decode, and the WebP encoder underflows on
/// one), a side past [`MAX_IMAGE_DIMENSION`], and a pixel count whose packed
/// size passes [`MAX_IMAGE_BYTES`]. The product is folded with checked
/// arithmetic so a header that would overflow `usize` fails here rather than
/// wrapping to a small allocation.
pub(crate) fn packed_byte_size(width: u32, height: u32, samples: usize) -> Result<usize, G2gError> {
    if width == 0
        || height == 0
        || samples == 0
        || width > MAX_IMAGE_DIMENSION
        || height > MAX_IMAGE_DIMENSION
    {
        return Err(G2gError::CapsMismatch);
    }
    let bytes = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(samples))
        .ok_or(G2gError::CapsMismatch)?;
    if bytes > MAX_IMAGE_BYTES {
        return Err(G2gError::CapsMismatch);
    }
    Ok(bytes)
}

/// Packed RGBA8 byte size of a decoded image, the common still-decoder output.
#[cfg(any(feature = "png", feature = "webp", test))]
pub(crate) fn rgba_byte_size(width: u32, height: u32) -> Result<usize, G2gError> {
    packed_byte_size(width, height, 4)
}

/// The downstream half every still-image decoder shares: each decoded image is
/// a packed RGB(A) system frame, preceded by a `CapsChanged` whenever the
/// geometry or format differs from the last one emitted (a still's real size
/// comes from the file, not from negotiation, and a sequence of files can
/// change size mid-stream).
#[derive(Debug, Default)]
pub(crate) struct StillImageOutput {
    out_dims: Option<(RawVideoFormat, u32, u32)>,
    sequence: u64,
}

impl StillImageOutput {
    #[cfg(any(feature = "png", feature = "webp"))]
    pub(crate) async fn push_rgba(
        &mut self,
        out: &mut dyn OutputSink,
        pixels: Vec<u8>,
        width: u32,
        height: u32,
        framerate: &Rate,
        timing: FrameTiming,
    ) -> Result<(), G2gError> {
        self.push(
            out,
            pixels,
            RawVideoFormat::Rgba8,
            (width, height),
            framerate,
            timing,
        )
        .await
    }

    pub(crate) async fn push(
        &mut self,
        out: &mut dyn OutputSink,
        pixels: Vec<u8>,
        format: RawVideoFormat,
        dims: (u32, u32),
        framerate: &Rate,
        timing: FrameTiming,
    ) -> Result<(), G2gError> {
        let (width, height) = dims;
        if self.out_dims != Some((format, width, height)) {
            out.push(PipelinePacket::CapsChanged(Caps::RawVideo {
                format,
                width: Dim::Fixed(width),
                height: Dim::Fixed(height),
                framerate: framerate.clone(),
                interlace: g2g_core::Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            }))
            .await?;
            self.out_dims = Some((format, width, height));
        }
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(pixels.into_boxed_slice())),
            timing,
            self.sequence,
        );
        self.sequence += 1;
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_ordinary_geometry_and_rejects_absurd() {
        assert_eq!(rgba_byte_size(640, 480), Ok(640 * 480 * 4));
        assert_eq!(rgba_byte_size(0, 480), Err(G2gError::CapsMismatch));
        assert_eq!(rgba_byte_size(640, 0), Err(G2gError::CapsMismatch));
        assert_eq!(
            rgba_byte_size(MAX_IMAGE_DIMENSION + 1, 1),
            Err(G2gError::CapsMismatch)
        );
        // Both sides in range, product over the byte budget.
        assert_eq!(
            rgba_byte_size(MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION),
            Err(G2gError::CapsMismatch)
        );
        // A header at the u32 ceiling cannot wrap to a small allocation.
        assert_eq!(
            rgba_byte_size(u32::MAX, u32::MAX),
            Err(G2gError::CapsMismatch)
        );
    }
}
