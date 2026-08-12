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

/// Whether a format's chroma subsampling forces an even (width, height): a
/// horizontally-subsampled format needs even width, a vertically-subsampled one
/// needs even height, so a crop / scale stays on chroma-sample boundaries. NV12
/// and YUYV are handled explicitly (NV12 is 4:2:0, YUYV packed 4:2:2); the fully
/// planar family follows its [`RawVideoFormat::chroma_shift`]; RGBA needs neither.
pub(crate) fn even_dims_required(format: RawVideoFormat) -> (bool, bool) {
    match format {
        RawVideoFormat::Nv12 | RawVideoFormat::P010 => (true, true),
        RawVideoFormat::Yuyv => (true, false),
        _ => match format.chroma_shift() {
            Some((hs, vs)) => (hs > 0, vs > 0),
            None => (false, false),
        },
    }
}

/// Byte layout of a fully-planar YUV `format` at `w x h`: `(byte offset, plane
/// width in samples, plane height)` for the Y, U, and V planes in turn. Chroma
/// plane dimensions follow the format's subsampling; the sample byte width is
/// [`RawVideoFormat::bytes_per_sample`]. Panics if `format` is not fully planar.
pub(crate) fn planar_planes(
    format: RawVideoFormat,
    w: usize,
    h: usize,
) -> [(usize, usize, usize); 3] {
    let (hs, vs) = format.chroma_shift().expect("fully-planar format");
    let bps = format.bytes_per_sample();
    let (cw, ch) = (w.div_ceil(1 << hs), h.div_ceil(1 << vs));
    let luma = w * h * bps;
    let chroma = cw * ch * bps;
    [(0, w, h), (luma, cw, ch), (luma + chroma, cw, ch)]
}

/// Per-plane `(row bytes, rows)` of one `w x h` frame in `format`, in plane
/// order: the shape a [`PlaneLayout`](g2g_core::meta::PlaneLayout) puts offsets
/// and strides on. A tightly-packed frame is exactly these rows back to back;
/// a padded one differs only in where each row starts.
#[cfg(feature = "metadata")]
pub(crate) fn plane_shapes(
    format: RawVideoFormat,
    w: usize,
    h: usize,
) -> alloc::vec::Vec<(usize, usize)> {
    let bps = format.bytes_per_sample();
    if let Some((hs, vs)) = format.chroma_shift() {
        let (cw, ch) = (w.div_ceil(1 << hs), h.div_ceil(1 << vs));
        return alloc::vec![(w * bps, h), (cw * bps, ch), (cw * bps, ch)];
    }
    match format {
        // Semi-planar: luma, then one interleaved Cb,Cr plane at half height and
        // the same byte width (half the samples, two of them per position).
        RawVideoFormat::Nv12 => alloc::vec![(w, h), (w, h.div_ceil(2))],
        RawVideoFormat::P010 => alloc::vec![(w * 2, h), (w * 2, h.div_ceil(2))],
        // Everything else is one packed plane.
        _ => alloc::vec![(row_bytes(format, w), h)],
    }
}

/// Byte width of one row of `format`'s **first** plane at `w` pixels: the row
/// pitch of a tightly-packed frame.
pub(crate) fn row_bytes(format: RawVideoFormat, w: usize) -> usize {
    match format {
        RawVideoFormat::Rgba8 | RawVideoFormat::Bgra8 => w * 4,
        RawVideoFormat::Rgb8 => w * 3,
        RawVideoFormat::Yuyv => w * 2,
        _ => w * format.bytes_per_sample(),
    }
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
        // Packed RGB: three bytes per pixel, no alpha.
        RawVideoFormat::Rgb8 => w * h * 3,
        RawVideoFormat::Nv12 => w * h * 3 / 2,
        // Semi-planar 10-bit: NV12's sample counts at 2 bytes each.
        RawVideoFormat::P010 => w * h * 3,
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
        // A packed format not modeled here (or one added since): fail loud
        // rather than mis-size a buffer.
        _ => unreachable!("unmodeled packed RawVideoFormat: {format:?}"),
    }
}
