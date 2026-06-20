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
