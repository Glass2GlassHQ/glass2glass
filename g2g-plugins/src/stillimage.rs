//! Shared geometry bounds and byte-stream framing for the still-image codec
//! elements (`PngDec`, `PngEnc`, `WebPDec`).
//!
//! A PNG `IHDR` or a WebP `VP8X` header carries dimensions up to 2^31 / 2^24,
//! and both decoder crates size their output buffer from those numbers. The
//! header is attacker-controlled, so the geometry is checked against a fixed
//! budget here before anything allocates: a bogus 100000x100000 header fails the
//! parse instead of asking for 40 GB.
//!
//! The framing side exists because a byte source hands over whatever a read
//! returned, not whole files: `filesrc` emits 64 KiB chunks, so an image larger
//! than that arrives in pieces, while `multifilesrc` hands over one complete
//! image per buffer. [`ImageAssembler`] covers both; the length function it walks
//! with is each format's own, and lives with that format's decoder.

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

/// Validate a decoded image's geometry and return its packed RGBA8 byte size.
///
/// Rejects a zero side (nothing to decode, and the WebP encoder underflows on
/// one), a side past [`MAX_IMAGE_DIMENSION`], and a pixel count whose RGBA size
/// passes [`MAX_IMAGE_BYTES`]. The product is folded with checked arithmetic so
/// a header that would overflow `usize` fails here rather than wrapping to a
/// small allocation.
pub(crate) fn rgba_byte_size(width: u32, height: u32) -> Result<usize, G2gError> {
    if width == 0 || height == 0 || width > MAX_IMAGE_DIMENSION || height > MAX_IMAGE_DIMENSION {
        return Err(G2gError::CapsMismatch);
    }
    let bytes = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(G2gError::CapsMismatch)?;
    if bytes > MAX_IMAGE_BYTES {
        return Err(G2gError::CapsMismatch);
    }
    Ok(bytes)
}

/// Ceiling on the encoded bytes held while waiting for the rest of an image.
/// Bounds a byte stream that opens with a plausible signature and then never
/// completes, which would otherwise grow the buffer for as long as it flows.
pub(crate) const MAX_ENCODED_BYTES: usize = 64 * 1024 * 1024;

/// Length of the complete image at the start of `data`, or `None` when more
/// bytes are needed. `Err` when the bytes cannot be an image of this format at
/// all, or when the length it claims is past [`MAX_ENCODED_BYTES`].
pub(crate) type FrameLength = fn(&[u8]) -> Result<Option<usize>, G2gError>;

/// Reassembles whole encoded images from a byte stream.
///
/// A buffer that already holds exactly one image passes through with a single
/// copy; anything else accumulates until the format's own length says the image
/// is complete, so a source's read size never decides whether a file decodes.
#[derive(Debug, Default)]
pub(crate) struct ImageAssembler {
    pending: Vec<u8>,
}

impl ImageAssembler {
    /// Take one buffer of the stream and hand back every image it completed.
    pub(crate) fn push(
        &mut self,
        bytes: &[u8],
        frame_length: FrameLength,
    ) -> Result<Vec<Vec<u8>>, G2gError> {
        if self.pending.len().saturating_add(bytes.len()) > MAX_ENCODED_BYTES {
            return Err(G2gError::CapsMismatch);
        }
        self.pending.extend_from_slice(bytes);

        let mut images = Vec::new();
        while let Some(length) = frame_length(&self.pending)? {
            let rest = self.pending.split_off(length);
            images.push(core::mem::replace(&mut self.pending, rest));
        }
        Ok(images)
    }

    /// End of stream: bytes still held are an image that never completed.
    pub(crate) fn finish(&mut self) -> Result<(), G2gError> {
        if self.pending.is_empty() {
            Ok(())
        } else {
            self.pending = Vec::new();
            Err(G2gError::CapsMismatch)
        }
    }

    /// Drop what a flushing seek made stale.
    pub(crate) fn reset(&mut self) {
        self.pending = Vec::new();
    }
}

/// The downstream half every still-image decoder shares: each decoded image is
/// an RGBA system frame, preceded by a `CapsChanged` whenever the geometry
/// differs from the last one emitted (a still's real size comes from the file,
/// not from negotiation, and a sequence of files can change size mid-stream).
#[derive(Debug, Default)]
pub(crate) struct StillImageOutput {
    out_dims: Option<(u32, u32)>,
    sequence: u64,
}

impl StillImageOutput {
    pub(crate) async fn push_rgba(
        &mut self,
        out: &mut dyn OutputSink,
        pixels: Vec<u8>,
        width: u32,
        height: u32,
        framerate: &Rate,
        timing: FrameTiming,
    ) -> Result<(), G2gError> {
        if self.out_dims != Some((width, height)) {
            out.push(PipelinePacket::CapsChanged(Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Fixed(width),
                height: Dim::Fixed(height),
                framerate: framerate.clone(),
                interlace: g2g_core::Interlace::Any,
            }))
            .await?;
            self.out_dims = Some((width, height));
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
    use alloc::vec;

    /// A stand-in image format for the assembler's own behaviour, which does not
    /// depend on PNG or WebP: a 4-byte big-endian length, then that many bytes.
    const LENGTH_PREFIX: usize = 4;

    fn prefixed_frame_length(data: &[u8]) -> Result<Option<usize>, G2gError> {
        let Some(header) = data.get(..LENGTH_PREFIX) else {
            return Ok(None);
        };
        let payload = u32::from_be_bytes([header[0], header[1], header[2], header[3]]) as usize;
        let total = payload
            .checked_add(LENGTH_PREFIX)
            .filter(|total| *total <= MAX_ENCODED_BYTES)
            .ok_or(G2gError::CapsMismatch)?;
        Ok((data.len() >= total).then_some(total))
    }

    fn prefixed_bytes(payload: usize) -> Vec<u8> {
        let mut file = Vec::from((payload as u32).to_be_bytes());
        file.extend_from_slice(&vec![0u8; payload]);
        file
    }

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

    #[test]
    fn assembler_rejoins_a_split_image_and_splits_a_joined_one() {
        let file = prefixed_bytes(40);
        // Split across three buffers, the `filesrc` case: nothing comes out
        // until the last piece lands, and then it is the whole file.
        let mut assembler = ImageAssembler::default();
        let (a, rest) = file.split_at(2);
        let (b, c) = rest.split_at(20);
        assert_eq!(assembler.push(a, prefixed_frame_length), Ok(Vec::new()));
        assert_eq!(assembler.push(b, prefixed_frame_length), Ok(Vec::new()));
        assert_eq!(
            assembler.push(c, prefixed_frame_length),
            Ok(vec![file.clone()])
        );
        assert_eq!(assembler.finish(), Ok(()));

        // Two whole images in one buffer come out as two.
        let mut joined = file.clone();
        joined.extend_from_slice(&file);
        let mut assembler = ImageAssembler::default();
        assert_eq!(
            assembler.push(&joined, prefixed_frame_length),
            Ok(vec![file.clone(), file.clone()])
        );
    }

    #[test]
    fn assembler_reports_a_stream_that_ends_mid_image() {
        let file = prefixed_bytes(40);
        let mut assembler = ImageAssembler::default();
        assert_eq!(
            assembler.push(&file[..12], prefixed_frame_length),
            Ok(Vec::new())
        );
        assert_eq!(
            assembler.finish(),
            Err(G2gError::CapsMismatch),
            "a half-received image is not silently dropped"
        );
        // The failed image is cleared, so the next one starts clean.
        assert_eq!(assembler.push(&file, prefixed_frame_length), Ok(vec![file]));
    }

    #[test]
    fn assembler_bounds_a_stream_that_never_completes() {
        // A length claiming almost the whole ceiling, so the length itself is
        // allowed and only the held bytes can stop the stream. Feeding it
        // forever must hit the ceiling rather than growing with the stream.
        const CHUNK_BYTES: usize = 8 * 1024 * 1024;
        let header = (MAX_ENCODED_BYTES as u32 - 32).to_be_bytes();

        let mut assembler = ImageAssembler::default();
        assert_eq!(
            assembler.push(&header, prefixed_frame_length),
            Ok(Vec::new())
        );
        let chunk = vec![0u8; CHUNK_BYTES];
        let mut pushes = 0;
        loop {
            match assembler.push(&chunk, prefixed_frame_length) {
                Ok(images) => {
                    assert!(images.is_empty());
                    pushes += 1;
                    assert!(pushes <= MAX_ENCODED_BYTES / CHUNK_BYTES, "must stop");
                }
                Err(e) => {
                    assert_eq!(e, G2gError::CapsMismatch);
                    break;
                }
            }
        }
    }
}
