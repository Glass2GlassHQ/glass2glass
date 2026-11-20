// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::borrow::Cow;

use g2g_core::{G2gError, TensorLayout};

const F32_BYTES: usize = core::mem::size_of::<f32>();
pub const RGBA_BYTES_PER_PIXEL: usize = 4;
pub const RGB_CHANNELS: usize = 3;
const U8_FULL_SCALE: f32 = 255.0;

pub fn f32_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * F32_BYTES);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

// a model that diverged to NaN or infinity yields meaningless results, so it fails here
pub fn finite_f32_from_le_bytes(bytes: &[u8]) -> Result<Vec<f32>, G2gError> {
    let (words, rest) = bytes.as_chunks::<F32_BYTES>();
    if !rest.is_empty() {
        return Err(G2gError::CapsMismatch);
    }
    let values: Vec<f32> = words.iter().map(|word| f32::from_le_bytes(*word)).collect();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(G2gError::CapsMismatch);
    }
    Ok(values)
}

// RGBA8 to `value / 255` f32 in plane order: the R plane, then G, then B
pub fn rgba_to_nchw(rgba: &[u8], width: u32, height: u32) -> Result<Vec<f32>, G2gError> {
    let plane = (width as usize)
        .checked_mul(height as usize)
        .ok_or(G2gError::CapsMismatch)?;
    let needed = plane
        .checked_mul(RGBA_BYTES_PER_PIXEL)
        .ok_or(G2gError::CapsMismatch)?;
    if rgba.len() < needed {
        return Err(G2gError::CapsMismatch);
    }
    let mut nchw = vec![0f32; plane * RGB_CHANNELS];
    let (red, rest) = nchw.split_at_mut(plane);
    let (green, blue) = rest.split_at_mut(plane);
    let pixels = rgba[..needed].as_chunks::<RGBA_BYTES_PER_PIXEL>().0;
    for (index, pixel) in pixels.iter().enumerate() {
        red[index] = f32::from(pixel[0]) / U8_FULL_SCALE;
        green[index] = f32::from(pixel[1]) / U8_FULL_SCALE;
        blue[index] = f32::from(pixel[2]) / U8_FULL_SCALE;
    }
    Ok(nchw)
}

// RGB or RGBA video as the RGB bytes of a u8 model, interleaved for NHWC or one plane per channel for NCHW
pub fn rgb_pixels_in_layout(
    pixels: &[u8],
    bytes_per_pixel: usize,
    width: u32,
    height: u32,
    layout: TensorLayout,
) -> Result<Cow<'_, [u8]>, G2gError> {
    let plane = (width as usize)
        .checked_mul(height as usize)
        .ok_or(G2gError::CapsMismatch)?;
    let needed = plane
        .checked_mul(bytes_per_pixel)
        .ok_or(G2gError::CapsMismatch)?;
    let pixels = pixels.get(..needed).ok_or(G2gError::CapsMismatch)?;
    let rgb_of_each_pixel = pixels
        .chunks_exact(bytes_per_pixel)
        .map(|pixel| &pixel[..RGB_CHANNELS]);
    match layout {
        TensorLayout::Nhwc if bytes_per_pixel == RGB_CHANNELS => Ok(Cow::Borrowed(pixels)),
        TensorLayout::Nhwc => Ok(Cow::Owned(rgb_of_each_pixel.flatten().copied().collect())),
        TensorLayout::Nchw => {
            let mut planes = vec![0u8; plane * RGB_CHANNELS];
            for (index, rgb) in rgb_of_each_pixel.enumerate() {
                for (channel, value) in rgb.iter().enumerate() {
                    planes[channel * plane + index] = *value;
                }
            }
            Ok(Cow::Owned(planes))
        }
        _ => Err(G2gError::CapsMismatch),
    }
}

// little-endian f32 NCHW `[1, 3, height, width]` bytes to values, the tensor-input path's decode
pub fn tensor_bytes_to_chw(bytes: &[u8], width: u32, height: u32) -> Result<Vec<f32>, G2gError> {
    let plane = (width as usize)
        .checked_mul(height as usize)
        .ok_or(G2gError::CapsMismatch)?;
    let n = plane
        .checked_mul(RGB_CHANNELS)
        .ok_or(G2gError::CapsMismatch)?;
    let nbytes = n.checked_mul(F32_BYTES).ok_or(G2gError::CapsMismatch)?;
    if bytes.len() < nbytes {
        return Err(G2gError::CapsMismatch);
    }
    Ok(bytes[..nbytes]
        .as_chunks::<F32_BYTES>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    // two pixels, each red, green, blue then alpha
    const TWO_RGBA_PIXELS: [u8; 8] = [1, 2, 3, 255, 4, 5, 6, 255];

    #[test]
    fn rgba_pixels_drop_alpha_for_an_nhwc_model() {
        let bytes = rgb_pixels_in_layout(
            &TWO_RGBA_PIXELS,
            RGBA_BYTES_PER_PIXEL,
            2,
            1,
            TensorLayout::Nhwc,
        )
        .unwrap();
        assert_eq!(&*bytes, &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn rgba_pixels_split_into_planes_for_an_nchw_model() {
        let bytes = rgb_pixels_in_layout(
            &TWO_RGBA_PIXELS,
            RGBA_BYTES_PER_PIXEL,
            2,
            1,
            TensorLayout::Nchw,
        )
        .unwrap();
        assert_eq!(&*bytes, &[1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn rgb_pixels_pass_through_to_an_nhwc_model_uncopied() {
        let rgb = [1, 2, 3, 4, 5, 6];
        let bytes = rgb_pixels_in_layout(&rgb, RGB_CHANNELS, 2, 1, TensorLayout::Nhwc).unwrap();
        assert!(matches!(bytes, Cow::Borrowed(borrowed) if borrowed.as_ptr() == rgb.as_ptr()));
    }

    #[test]
    fn short_frames_are_rejected() {
        assert_eq!(
            rgb_pixels_in_layout(
                &TWO_RGBA_PIXELS,
                RGBA_BYTES_PER_PIXEL,
                3,
                1,
                TensorLayout::Nhwc
            ),
            Err(G2gError::CapsMismatch)
        );
    }

    #[test]
    fn a_short_frame_fails_instead_of_panicking() {
        let width = 2;
        let height = 2;
        let one_pixel_short = vec![0u8; (width * height - 1) * RGBA_BYTES_PER_PIXEL];
        assert!(rgba_to_nchw(&one_pixel_short, width as u32, height as u32).is_err());
    }

    #[test]
    fn rgba_splits_into_normalized_planes() {
        let rgba = [255, 0, 51, 9, 0, 255, 102, 9];
        let nchw = rgba_to_nchw(&rgba, 2, 1).unwrap();
        assert_eq!(nchw, [1.0, 0.0, 0.0, 1.0, 0.2, 0.4]);
    }

    #[test]
    fn a_non_finite_value_fails() {
        let bytes = f32_le_bytes(&[1.0, f32::NAN]);
        assert!(finite_f32_from_le_bytes(&bytes).is_err());
        assert_eq!(
            finite_f32_from_le_bytes(&f32_le_bytes(&[1.5, -2.0])).unwrap(),
            [1.5, -2.0]
        );
    }
}
