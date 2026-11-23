// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::os::fd::{AsFd, BorrowedFd};
use std::path::Path;

use drm::control::Device as ControlDevice;
use drm::Device;

use g2g_core::{G2gError, HardwareError};
#[cfg(feature = "metadata")]
use {
    alloc::vec::Vec,
    drm::buffer::Buffer as _,
    drm::control::dumbbuffer::DumbBuffer,
    g2g_core::{OverlayCompositionMeta, RawVideoFormat},
    std::sync::Arc,
};

#[cfg(feature = "metadata")]
use crate::{overlaycomposition::composition_layer, pixel::rgba_rb_offsets};

pub(crate) const DEFAULT_CARD: &str = "/dev/dri/card0";

#[derive(Debug)]
pub(crate) struct Card(std::fs::File);

impl AsFd for Card {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}
impl Device for Card {}
impl ControlDevice for Card {}

impl Card {
    pub(crate) fn open<P: AsRef<Path>>(path: P) -> Result<Self, G2gError> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        Ok(Card(file))
    }
}

// frames reuse the composition's bitmaps by reference until the overlay changes
#[cfg(feature = "metadata")]
pub(crate) fn same_composition(
    first: &OverlayCompositionMeta,
    second: &OverlayCompositionMeta,
) -> bool {
    first.rectangles().len() == second.rectangles().len()
        && first
            .rectangles()
            .iter()
            .zip(second.rectangles())
            .all(|(first, second)| {
                first.placement() == second.placement()
                    && first.pixel_size() == second.pixel_size()
                    && Arc::ptr_eq(first.pixels(), second.pixels())
            })
}

// Draws `composition` over a cleared `width` x `height` picture into an ARGB8888 dumb buffer.
#[cfg(feature = "metadata")]
pub(crate) fn draw_composition(
    card: &Card,
    dumb: &mut DumbBuffer,
    composition: &OverlayCompositionMeta,
    (width, height): (u32, u32),
    layer: &mut Vec<u8>,
    row: &mut Vec<u8>,
) -> Result<(), G2gError> {
    let layer = composition_layer(composition, (width, height), layer)?;
    let pitch = dumb.pitch() as usize;
    let mut mapping = card
        .map_dumb_buffer(dumb)
        .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
    copy_as_argb8888(layer, width as usize, pitch, &mut mapping, row);
    Ok(())
}

#[cfg(feature = "metadata")]
fn copy_as_argb8888(
    layer: &[u8],
    width: usize,
    pitch: usize,
    mapped: &mut [u8],
    row: &mut Vec<u8>,
) {
    // drm ARGB8888 is BGRA in memory
    let (red, blue) = rgba_rb_offsets(RawVideoFormat::Bgra8);
    row.resize(width * 4, 0);
    for (layer_row, mapped_row) in layer.chunks_exact(width * 4).zip(mapped.chunks_mut(pitch)) {
        for (dst, src) in row
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(layer_row.as_chunks::<4>().0)
        {
            *dst = [src[red], src[1], src[blue], src[3]];
        }
        mapped_row[..row.len()].copy_from_slice(row);
    }
}

#[cfg(all(test, feature = "metadata"))]
mod tests {
    use super::*;

    #[test]
    fn overlay_rows_land_as_argb8888_at_the_buffer_pitch() {
        const WIDTH: usize = 2;
        const PITCH: usize = 12;
        const PADDING: u8 = 0xEE;
        let red = [200, 0, 0, 200];
        let half_blue = [0, 0, 64, 128];
        let layer = [red, half_blue, half_blue, red].concat();
        let mut mapped = [PADDING; 2 * PITCH];
        copy_as_argb8888(&layer, WIDTH, PITCH, &mut mapped, &mut Vec::new());
        // ARGB8888 is [blue, green, red, alpha] in memory
        let argb_red = [0, 0, 200, 200];
        let argb_half_blue = [64, 0, 0, 128];
        assert_eq!(mapped[..8], [argb_red, argb_half_blue].concat());
        assert_eq!(
            mapped[PITCH..PITCH + 8],
            [argb_half_blue, argb_red].concat()
        );
        assert_eq!(
            mapped[8..PITCH],
            [PADDING; PITCH - 8],
            "row padding untouched"
        );
    }
}
