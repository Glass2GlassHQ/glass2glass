//! Shared software-blend primitive for the CPU overlays and compositor.
//! Integer source-over, no float intrinsics, `no_std` baseline.

/// Source-over blend of one RGBA `src` pixel onto `canvas` at byte offset `d`,
/// modulating the source alpha by `galpha` (0..=255). Integer math; keeps an
/// opaque canvas opaque. Callers clip `d` into the canvas first (no bounds
/// check here, to stay branch-free on the compositor's scaling hot path); pass
/// `galpha == 255` for an unmodulated overlay paint.
#[inline]
pub(crate) fn blend_px(canvas: &mut [u8], d: usize, src: [u8; 4], galpha: u8) {
    // Effective source alpha = src_a * galpha (0..=255).
    let a = (src[3] as u32 * galpha as u32 + 127) / 255;
    let inv = 255 - a;
    for c in 0..3 {
        canvas[d + c] = ((src[c] as u32 * a + canvas[d + c] as u32 * inv + 127) / 255) as u8;
    }
    canvas[d + 3] = (a + canvas[d + 3] as u32 * inv / 255) as u8;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opaque_source_overwrites() {
        let mut buf = [10u8, 20, 30, 255];
        blend_px(&mut buf, 0, [200, 100, 50, 255], 255);
        assert_eq!(buf, [200, 100, 50, 255]);
    }

    #[test]
    fn zero_alpha_leaves_canvas_untouched() {
        let mut buf = [10u8, 20, 30, 255];
        blend_px(&mut buf, 0, [200, 100, 50, 0], 255);
        assert_eq!(buf, [10, 20, 30, 255]);
    }

    #[test]
    fn galpha_modulates_source_alpha() {
        // galpha 0 must paint nothing even for an opaque source.
        let mut buf = [10u8, 20, 30, 255];
        blend_px(&mut buf, 0, [200, 100, 50, 255], 0);
        assert_eq!(buf, [10, 20, 30, 255]);

        // galpha 128 on an opaque source is the same as a ~50%-alpha source.
        let mut a = [0u8, 0, 0, 0];
        let mut b = [0u8, 0, 0, 0];
        blend_px(&mut a, 0, [255, 255, 255, 255], 128);
        blend_px(&mut b, 0, [255, 255, 255, 128], 255);
        assert_eq!(a, b);
    }
}
