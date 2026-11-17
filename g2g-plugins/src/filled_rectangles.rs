// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::vec::Vec;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FilledRectangle {
    // target pixels from the top-left corner
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    // straight alpha
    pub rgba: [u8; 4],
}

pub(crate) const FLOATS_PER_VERTEX: usize = 6;
const VERTICES_PER_RECTANGLE: usize = 6;

pub(crate) fn rectangle_vertices(
    rectangles: &[FilledRectangle],
    (target_width, target_height): (u32, u32),
) -> Vec<f32> {
    // two triangles per rectangle, each vertex x and y in clip space, then r, g, b and a from 0 to 1
    let clip_x = |x: f32| 2.0 * x / target_width as f32 - 1.0;
    let clip_y = |y: f32| 1.0 - 2.0 * y / target_height as f32;
    let mut vertices =
        Vec::with_capacity(rectangles.len() * VERTICES_PER_RECTANGLE * FLOATS_PER_VERTEX);
    for rectangle in rectangles {
        let [r, g, b, a] = rectangle.rgba.map(|channel| f32::from(channel) / 255.0);
        let (left, right) = (clip_x(rectangle.x), clip_x(rectangle.x + rectangle.width));
        let (top, bottom) = (clip_y(rectangle.y), clip_y(rectangle.y + rectangle.height));
        for (x, y) in [
            (left, top),
            (right, top),
            (right, bottom),
            (left, top),
            (right, bottom),
            (left, bottom),
        ] {
            vertices.extend_from_slice(&[x, y, r, g, b, a]);
        }
    }
    vertices
}
