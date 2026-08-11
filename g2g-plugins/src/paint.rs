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

/// Source-over onto a canvas that may itself be transparent, leaving the result
/// un-premultiplied. [`blend_px`] weights the destination by `255 - a` alone,
/// which only holds where the destination is opaque: against a cleared canvas it
/// scales the source colour by its own alpha, so a half transparent subtitle
/// pixel comes out half dark instead of faint. Identical to `blend_px` once the
/// destination is opaque.
pub(crate) fn over_px(canvas: &mut [u8], d: usize, src: [u8; 4]) {
    let sa = src[3] as u32;
    let keep = canvas[d + 3] as u32 * (255 - sa) / 255;
    let out_a = sa + keep;
    if out_a == 0 {
        canvas[d..d + 4].copy_from_slice(&[0; 4]);
        return;
    }
    for c in 0..3 {
        canvas[d + c] =
            ((src[c] as u32 * sa + canvas[d + c] as u32 * keep + out_a / 2) / out_a) as u8;
    }
    canvas[d + 3] = out_a as u8;
}

/// An RGBA8 pixel buffer plus the geometry needed to clip against it, for the
/// shape and glyph painting the CPU overlays share.
pub(crate) struct Canvas<'a> {
    pub pixels: &'a mut [u8],
    pub width: i32,
    pub height: i32,
}

impl Canvas<'_> {
    /// Source-over blend a filled rectangle, clipped to the canvas.
    pub(crate) fn fill_rect(&mut self, x: i32, y: i32, rw: i32, rh: i32, color: [u8; 4]) {
        for py in y..y + rh {
            if py < 0 || py >= self.height {
                continue;
            }
            for px in x..x + rw {
                if px < 0 || px >= self.width {
                    continue;
                }
                blend_px(
                    self.pixels,
                    ((py * self.width + px) * 4) as usize,
                    color,
                    255,
                );
            }
        }
    }

    /// Blit one 8x8 [`bitmapfont`](crate::bitmapfont) glyph at `(gx, gy)`, each
    /// set bit a `scale` x `scale` block of `color`.
    pub(crate) fn blit_glyph(
        &mut self,
        gx: i32,
        gy: i32,
        scale: i32,
        rows: [u8; 8],
        color: [u8; 4],
    ) {
        for (ry, bits) in rows.iter().enumerate() {
            if *bits == 0 {
                continue;
            }
            for col in 0..8i32 {
                if bits & (0x80 >> col) != 0 {
                    self.fill_rect(
                        gx + col * scale,
                        gy + ry as i32 * scale,
                        scale,
                        scale,
                        color,
                    );
                }
            }
        }
    }
}

/// Which matrix a limited-range Y'CrCb palette is converted through. The bitmap
/// subtitle formats carry no colorimetry, so the decoder picks one: DVB is
/// always BT.601, PGS switches on the video height.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum YcbcrMatrix {
    Bt601,
    Bt709,
}

/// Limited-range Y'CrCb to RGB, in the fixed-point form a reference subtitle
/// decoder uses, so the rendered colours are bit-identical to one. The
/// coefficients are `round(k * 2^10)` of the full-range gains, i.e. the studio
/// 219 / 224 excursions scaled back out to 0..255.
#[inline]
pub(crate) fn ycbcr_to_rgb(y: u8, cr: u8, cb: u8, matrix: YcbcrMatrix) -> (u8, u8, u8) {
    const SCALE: i32 = 10;
    const HALF: i32 = 1 << (SCALE - 1);
    const Y_GAIN: i32 = 1192; // 255/219

    // (cr->r, cb->g, cr->g, cb->b)
    let (cr_r, cb_g, cr_g, cb_b) = match matrix {
        // 1.40200, 0.34414, 0.71414, 1.77200
        YcbcrMatrix::Bt601 => (1634, 401, 832, 2066),
        // 1.5747, 0.1873, 0.4682, 1.8556
        YcbcrMatrix::Bt709 => (1836, 218, 546, 2163),
    };
    let cb = cb as i32 - 128;
    let cr = cr as i32 - 128;
    let yy = (y as i32 - 16) * Y_GAIN;
    let clip = |v: i32| v.clamp(0, 255) as u8;
    (
        clip((yy + cr_r * cr + HALF) >> SCALE),
        clip((yy - cb_g * cb - cr_g * cr + HALF) >> SCALE),
        clip((yy + cb_b * cb + HALF) >> SCALE),
    )
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
