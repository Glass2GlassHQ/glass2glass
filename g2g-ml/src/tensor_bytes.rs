// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::borrow::Cow;

use g2g_core::{G2gError, TensorDType, TensorLayout};

use crate::safetensors::f16_to_f32;

pub const F32_BYTES: usize = core::mem::size_of::<f32>();
pub const F16_BYTES: usize = core::mem::size_of::<u16>();
pub const RGBA_BYTES_PER_PIXEL: usize = 4;
pub const RGB_CHANNELS: usize = 3;
const U8_FULL_SCALE: f32 = 255.0;

pub fn f32_le_bytes(values: impl ExactSizeIterator<Item = f32>) -> Vec<u8> {
    let mut bytes = vec![0u8; values.len() * F32_BYTES];
    for (slot, value) in bytes.as_chunks_mut::<F32_BYTES>().0.iter_mut().zip(values) {
        *slot = value.to_le_bytes();
    }
    bytes
}

// a model that diverged to NaN or infinity yields meaningless results, so it fails here
pub fn finite_f32_words(bytes: &[u8]) -> Result<&[[u8; F32_BYTES]], G2gError> {
    let (words, rest) = bytes.as_chunks::<F32_BYTES>();
    let finite = words.iter().fold(true, |finite, word| {
        finite & f32::from_le_bytes(*word).is_finite()
    });
    if !rest.is_empty() || !finite {
        return Err(G2gError::CapsMismatch);
    }
    Ok(words)
}

// half's exponent bits all set is infinity or NaN
const F16_EXPONENT: u16 = 0x7c00;
const F16_SIGN: u16 = 0x8000;

pub fn finite_f16_words(bytes: &[u8]) -> Result<&[[u8; F16_BYTES]], G2gError> {
    let (words, rest) = bytes.as_chunks::<F16_BYTES>();
    let finite = words.iter().fold(true, |finite, word| {
        finite & (u16::from_le_bytes(*word) & F16_EXPONENT != F16_EXPONENT)
    });
    if !rest.is_empty() || !finite {
        return Err(G2gError::CapsMismatch);
    }
    Ok(words)
}

pub fn f16_le_to_f32_le_bytes(bytes: &[u8]) -> Vec<u8> {
    f32_le_bytes(
        bytes
            .as_chunks::<F16_BYTES>()
            .0
            .iter()
            .map(|word| f16_to_f32(u16::from_le_bytes(*word))),
    )
}

// finite halves compare as these keys do
fn f16_order_key(bits: u16) -> u16 {
    match bits & F16_SIGN {
        0 => bits | F16_SIGN,
        _ => !bits,
    }
}

/// A tensor's finite little-endian values, read where they lie.
#[derive(Debug, Clone, Copy)]
pub enum TensorWords<'a> {
    F32(&'a [[u8; F32_BYTES]]),
    F16(&'a [[u8; F16_BYTES]]),
}

impl<'a> TensorWords<'a> {
    pub fn new(dtype: TensorDType, bytes: &'a [u8]) -> Result<Self, G2gError> {
        match dtype {
            TensorDType::F32 => finite_f32_words(bytes).map(Self::F32),
            TensorDType::F16 => finite_f16_words(bytes).map(Self::F16),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::F32(words) => words.len(),
            Self::F16(words) => words.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn value(&self, index: usize) -> f32 {
        match self {
            Self::F32(words) => f32::from_le_bytes(words[index]),
            Self::F16(words) => f16_to_f32(u16::from_le_bytes(words[index])),
        }
    }

    pub fn to_f32_vec(&self) -> Vec<f32> {
        (0..self.len()).map(|index| self.value(index)).collect()
    }

    /// The largest value in `range` and the first index holding it, as numpy's
    /// `argmax` picks, or `None` when that value is not above `threshold`.
    /// Halves are compared by their bits, so only the winner is converted.
    pub fn argmax_above(
        &self,
        range: core::ops::Range<usize>,
        threshold: f32,
    ) -> Option<(usize, f32)> {
        let start = range.start;
        match self {
            Self::F32(words) => {
                let row = words.get(range)?;
                let values = || row.iter().map(|word| f32::from_le_bytes(*word));
                let max = values().fold(f32::MIN, f32::max);
                if max <= threshold {
                    return None;
                }
                let index = values().position(|value| value == max)?;
                Some((start + index, max))
            }
            Self::F16(words) => {
                let row = words.get(range)?;
                let keys = || {
                    row.iter()
                        .map(|word| f16_order_key(u16::from_le_bytes(*word)))
                };
                let max_key = keys().max()?;
                let index = keys().position(|key| key == max_key)?;
                let max = f16_to_f32(u16::from_le_bytes(row[index]));
                if max <= threshold {
                    return None;
                }
                Some((start + index, max))
            }
        }
    }
}

pub fn finite_f32_from_le_bytes(bytes: &[u8]) -> Result<Vec<f32>, G2gError> {
    Ok(finite_f32_words(bytes)?
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect())
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
    use g2g_plugins::tensorconvert::f32_to_f16_bits;

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
        let bytes = f32_le_bytes([1.0, f32::NAN].into_iter());
        assert!(finite_f32_from_le_bytes(&bytes).is_err());
        assert_eq!(
            finite_f32_from_le_bytes(&f32_le_bytes([1.5, -2.0].into_iter())).unwrap(),
            [1.5, -2.0]
        );
    }

    // the f16 argmax compares raw bits, which must order like the values even below zero
    #[test]
    fn half_argmax_picks_what_the_f32_argmax_picks() {
        let rows: [&[f32]; 3] = [
            &[-2.0, -0.5, 1.5, 1.5, -3.0],
            &[-4.0, -0.25, -1.0],
            &[0.0, 0.125, 0.0625, 0.125],
        ];
        for row in rows {
            let f32_bytes = f32_le_bytes(row.iter().copied());
            let f16_bytes: Vec<u8> = row
                .iter()
                .flat_map(|&value| f32_to_f16_bits(value).to_le_bytes())
                .collect();
            let f32_words = TensorWords::new(TensorDType::F32, &f32_bytes).unwrap();
            let f16_words = TensorWords::new(TensorDType::F16, &f16_bytes).unwrap();
            for threshold in [f32::MIN, -1.0, 0.125] {
                assert_eq!(
                    f16_words.argmax_above(0..row.len(), threshold),
                    f32_words.argmax_above(0..row.len(), threshold),
                    "{row:?} above {threshold}"
                );
            }
        }
    }

    #[test]
    fn a_half_infinity_fails_validation() {
        let infinity = f32_to_f16_bits(f32::INFINITY).to_le_bytes();
        assert!(TensorWords::new(TensorDType::F16, &infinity).is_err());
    }
}
