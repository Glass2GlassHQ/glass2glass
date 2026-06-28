//! Shared pixel-format helpers for the packed-RGBA element family.

use g2g_core::RawVideoFormat;

/// Byte offsets of the red and blue channels in a packed 4-byte pixel (green is
/// always index 1, alpha index 3). RGBA is `[R, G, B, A]`, BGRA `[B, G, R, A]`.
/// Only the two packed formats are admitted by the callers' negotiation.
pub(crate) fn rgba_rb_offsets(format: RawVideoFormat) -> (usize, usize) {
    match format {
        RawVideoFormat::Rgba8 => (0, 2),
        RawVideoFormat::Bgra8 => (2, 0),
        _ => unreachable!("packed RGBA / BGRA only"),
    }
}

/// True for the planar 4:2:0 formats whose chroma is half-width and
/// half-height, so callers must validate even dimensions before slicing planes.
pub(crate) fn is_yuv420(format: RawVideoFormat) -> bool {
    matches!(format, RawVideoFormat::Nv12 | RawVideoFormat::I420)
}

/// Tightly-packed byte size of one `w x h` frame in `format` (no row padding).
pub(crate) fn frame_byte_size(format: RawVideoFormat, w: u32, h: u32) -> usize {
    // Fully-planar YUV (I420/I422/I444 at 8/10/12-bit): Y plus two chroma planes,
    // each chroma plane shrunk per the format's subsampling, all at this depth's
    // sample size. Derives from the format's own layout so a new variant needs no
    // edit here.
    if let Some((hs, vs)) = format.chroma_shift() {
        let (w, h) = (w as usize, h as usize);
        let (cw, ch) = (w.div_ceil(1 << hs), h.div_ceil(1 << vs));
        return (w * h + 2 * cw * ch) * format.bytes_per_sample();
    }
    let (w, h) = (w as usize, h as usize);
    match format {
        RawVideoFormat::Rgba8 | RawVideoFormat::Bgra8 => w * h * 4,
        RawVideoFormat::Nv12 => w * h * 3 / 2,
        // Packed 4:2:2: two bytes per pixel (Y0 U Y1 V over each pixel pair).
        RawVideoFormat::Yuyv => w * h * 2,
        // The fully-planar formats are handled above via `chroma_shift`.
        RawVideoFormat::I420
        | RawVideoFormat::I420p10
        | RawVideoFormat::I420p12
        | RawVideoFormat::I422
        | RawVideoFormat::I422p10
        | RawVideoFormat::I422p12
        | RawVideoFormat::I444
        | RawVideoFormat::I444p10
        | RawVideoFormat::I444p12 => unreachable!("planar YUV handled by chroma_shift"),
    }
}
