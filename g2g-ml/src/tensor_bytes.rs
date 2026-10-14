use g2g_core::G2gError;

const F32_BYTES: usize = core::mem::size_of::<f32>();
const RGBA_BYTES_PER_PIXEL: usize = 4;
const RGB_CHANNELS: usize = 3;
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

#[cfg(test)]
mod tests {
    use super::*;

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
