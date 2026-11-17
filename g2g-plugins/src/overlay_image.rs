// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::sync::Arc;

use crate::ViewportRectangle;

#[derive(Debug, Clone, PartialEq)]
pub struct OverlayImage {
    // target pixels the image is stretched over
    pub placement: ViewportRectangle,
    pub width: u32,
    pub height: u32,
    // premultiplied alpha, top row first
    pub rgba: Arc<[u8]>,
    // a renderer uploads the pixels again only when this changes
    pub generation: u64,
}
