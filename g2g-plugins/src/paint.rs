// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! Shared software-blend primitive for the CPU overlays and compositor.
//! Integer source-over, no float intrinsics, `no_std` baseline.

#[cfg(feature = "metadata")]
use alloc::vec::Vec;

#[cfg(feature = "metadata")]
use g2g_core::{meta::PlaneLayout, G2gError};

#[cfg(feature = "metadata")]
use crate::{pixel::chroma_420_size, yuvmatrix::YuvRgbMatrix};

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

/// Blend one source sample over a destination sample with scalar alpha (0..=255).
#[inline]
pub(crate) fn blend_u8(dst: u8, src: u8, alpha: u8) -> u8 {
    let a = alpha as u32;
    ((src as u32 * a + dst as u32 * (255 - a) + 127) / 255) as u8
}

// gst's compositor `source` operator: the colour is copied and the alpha scaled by `galpha`
#[inline]
pub(crate) fn source_px(canvas: &mut [u8], d: usize, src: [u8; 4], galpha: u8) {
    canvas[d..d + 3].copy_from_slice(&src[..3]);
    canvas[d + 3] = ((src[3] as u32 * galpha as u32 + 127) / 255) as u8;
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

/// Blend a premultiplied source onto a straight-alpha destination.
#[cfg(feature = "metadata")]
pub(crate) fn over_premultiplied(dst: &mut [u8; 4], src: [u8; 4]) {
    let alpha = u32::from(src[3]);
    if alpha == 0 {
        return;
    }
    let keep = u32::from(dst[3]) * (255 - alpha);
    let out_alpha = alpha * 255 + keep;
    for c in 0..3 {
        let color = u32::from(src[c]) * 255 * 255 + u32::from(dst[c]) * keep;
        dst[c] = ((color + out_alpha / 2) / out_alpha).min(255) as u8;
    }
    dst[3] = ((out_alpha + 127) / 255) as u8;
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
        self.fill_rect_alpha(x, y, rw, rh, color, u8::MAX);
    }

    /// [`fill_rect`](Self::fill_rect) with the colour's alpha scaled by
    /// `galpha` (0..=255).
    pub(crate) fn fill_rect_alpha(
        &mut self,
        x: i32,
        y: i32,
        rw: i32,
        rh: i32,
        color: [u8; 4],
        galpha: u8,
    ) {
        let columns = x.max(0)..(x + rw).min(self.width);
        for py in y.max(0)..(y + rh).min(self.height) {
            for px in columns.clone() {
                blend_px(
                    self.pixels,
                    ((py * self.width + px) * 4) as usize,
                    color,
                    galpha,
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
    fn a_rect_hanging_off_the_corner_paints_only_inside() {
        const SIDE: i32 = 3;
        const INK: [u8; 4] = [9, 9, 9, 255];
        let mut pixels = [0u8; (SIDE * SIDE * 4) as usize];
        let mut canvas = Canvas {
            pixels: &mut pixels,
            width: SIDE,
            height: SIDE,
        };
        canvas.fill_rect(-1, -1, 2, 2, INK);
        assert_eq!(pixels[..4], INK);
        assert!(pixels[4..].iter().all(|&byte| byte == 0));
    }

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

// rgba pixels keep alpha in their fourth byte
#[cfg(feature = "metadata")]
const ALPHA_OF_TWO_PIXELS: u64 = u64::from_ne_bytes([0, 0, 0, 0xFF, 0, 0, 0, 0xFF]);

#[cfg(feature = "metadata")]
fn transparent_pair(pixels: &[u8; 8]) -> bool {
    u64::from_ne_bytes(*pixels) & ALPHA_OF_TWO_PIXELS == 0
}

#[cfg(feature = "metadata")]
fn unpremultiplied(sum: u32, alpha: u32) -> i32 {
    (sum * 255 / alpha) as i32
}

#[cfg(feature = "metadata")]
fn blend_luma(luma: &mut u8, pixel: &[u8; 4], matrix: &YuvRgbMatrix) {
    let alpha = pixel[3];
    if alpha == 0 {
        return;
    }
    let [r, g, b] = [0, 1, 2].map(|c| unpremultiplied(pixel[c] as u32, alpha as u32));
    *luma = blend_u8(*luma, matrix.rgb_to_luma(r, g, b) as u8, alpha);
}

#[cfg(feature = "metadata")]
pub(crate) fn blend_layer_into_nv12(
    layer: &[u8],
    frame: &mut [u8],
    layout: &PlaneLayout,
    (width, height): (usize, usize),
    matrix: &YuvRgbMatrix,
) -> Result<(), G2gError> {
    let layer_rows: Vec<&[u8]> = layer.chunks_exact(width * 4).take(height).collect();
    for (y, layer_row) in layer_rows.iter().enumerate() {
        let range = layout
            .row_range(0, y, width)
            .ok_or(G2gError::CapsMismatch)?;
        let luma_row = frame.get_mut(range).ok_or(G2gError::CapsMismatch)?;
        let (luma_pairs, luma_last) = luma_row.as_chunks_mut::<2>();
        let (layer_pairs, layer_last) = layer_row.as_chunks::<8>();
        for (luma_pair, layer_pair) in luma_pairs.iter_mut().zip(layer_pairs) {
            if transparent_pair(layer_pair) {
                continue;
            }
            for (luma, pixel) in luma_pair.iter_mut().zip(layer_pair.as_chunks::<4>().0) {
                blend_luma(luma, pixel, matrix);
            }
        }
        for (luma, pixel) in luma_last.iter_mut().zip(layer_last.as_chunks::<4>().0) {
            blend_luma(luma, pixel, matrix);
        }
    }
    let (chroma_w, chroma_h) = chroma_420_size(width, height);
    for chroma_y in 0..chroma_h {
        let range = layout
            .row_range(1, chroma_y, 2 * chroma_w)
            .ok_or(G2gError::CapsMismatch)?;
        let chroma_row = frame.get_mut(range).ok_or(G2gError::CapsMismatch)?;
        let rows = &layer_rows[2 * chroma_y..(2 * chroma_y + 2).min(height)];
        for chroma_x in 0..chroma_w {
            let columns = 2 * chroma_x..(2 * chroma_x + 2).min(width);
            let drawn = rows
                .iter()
                .any(|row| match row.as_chunks::<8>().0.get(chroma_x) {
                    Some(pair) => !transparent_pair(pair),
                    None => columns.clone().any(|x| row[x * 4 + 3] != 0),
                });
            if !drawn {
                continue;
            }
            let mut sums = [0u32; 4];
            for row in rows {
                for x in columns.clone() {
                    for (sum, value) in sums.iter_mut().zip(&row[x * 4..][..4]) {
                        *sum += *value as u32;
                    }
                }
            }
            let samples = (rows.len() * columns.len()) as u32;
            let alpha = (sums[3] / samples) as u8;
            let [r, g, b] = [0, 1, 2].map(|c| unpremultiplied(sums[c], sums[3]));
            let (_, u, v) = matrix.rgb_to_yuv(r, g, b);
            let uv = &mut chroma_row[2 * chroma_x..][..2];
            uv[0] = blend_u8(uv[0], u as u8, alpha);
            uv[1] = blend_u8(uv[1], v as u8, alpha);
        }
    }
    Ok(())
}
