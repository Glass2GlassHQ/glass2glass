// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::vec::Vec;

pub const CURSOR_BYTES_PER_PIXEL: usize = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CursorShape {
    pub width: u32,
    pub height: u32,
    pub hotspot_x: i32,
    pub hotspot_y: i32,
    // straight alpha, `width * CURSOR_BYTES_PER_PIXEL` bytes per row
    pub rgba: Vec<u8>,
}

// wl_shm, xcursor and xfixes cursor pixels are little-endian argb with premultiplied alpha
pub fn straight_rgba_from_premultiplied_bgra(pixel: [u8; 4]) -> [u8; 4] {
    let [blue, green, red, alpha] = pixel;
    if alpha == 0 {
        return [0; 4];
    }
    let straight = |channel: u8| (u32::from(channel) * 255 / u32::from(alpha)).min(255) as u8;
    [straight(red), straight(green), straight(blue), alpha]
}
