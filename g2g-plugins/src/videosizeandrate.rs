// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::vec;
use alloc::vec::Vec;

use g2g_core::{Caps, CapsSet, Dim, G2gError, PropError, PropValue, Rate};

pub const MIN_REQUESTED_DIMENSION: u32 = 1;
pub const MAX_REQUESTED_DIMENSION: u32 = 16_384;
pub const MIN_REQUESTED_DIMENSION_TEXT: &str = "1";
pub const MAX_REQUESTED_DIMENSION_TEXT: &str = "16384";
pub const MAX_REQUESTED_FPS: u32 = 1_000;
pub const REQUESTED_DIMENSION: Dim = Dim::Range {
    min: MIN_REQUESTED_DIMENSION,
    max: MAX_REQUESTED_DIMENSION,
};

// the property takes the same sizes a caps filter can ask for
pub fn requested_dimension(value: &PropValue) -> Result<u32, PropError> {
    let dimension = value.as_uint().ok_or(PropError::Type)?;
    u32::try_from(dimension)
        .ok()
        .filter(|d| (MIN_REQUESTED_DIMENSION..=MAX_REQUESTED_DIMENSION).contains(d))
        .ok_or(PropError::Value)
}

#[derive(Debug, Clone, Copy)]
pub struct VideoSizeAndRate {
    width: u32,
    height: u32,
    framerate_q16: u32,
    width_pinned: bool,
    height_pinned: bool,
    framerate_pinned: bool,
}

impl VideoSizeAndRate {
    pub const fn new(width: u32, height: u32, framerate_q16: u32) -> Self {
        Self {
            width,
            height,
            framerate_q16,
            width_pinned: false,
            height_pinned: false,
            framerate_pinned: false,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn framerate_q16(&self) -> u32 {
        self.framerate_q16
    }

    pub fn pin_width(&mut self, width: u32) {
        self.width = width;
        self.width_pinned = true;
    }

    pub fn pin_height(&mut self, height: u32) {
        self.height = height;
        self.height_pinned = true;
    }

    pub fn pin_framerate_q16(&mut self, framerate_q16: u32) {
        self.framerate_q16 = framerate_q16;
        self.framerate_pinned = true;
    }

    pub fn produced_caps<F: Copy>(
        &self,
        formats: &[F],
        requested_framerate: Rate,
        caps_for: impl Fn(F, Dim, Dim, Rate) -> Option<Caps>,
    ) -> CapsSet {
        let mut widths = vec![(0, Dim::Fixed(self.width))];
        if !self.width_pinned {
            widths.push((1, REQUESTED_DIMENSION));
        }
        let mut heights = vec![(0, Dim::Fixed(self.height))];
        if !self.height_pinned {
            heights.push((1, REQUESTED_DIMENSION));
        }
        let mut framerates = vec![(0, Rate::Fixed(self.framerate_q16))];
        if !self.framerate_pinned {
            framerates.push((1, requested_framerate));
        }
        let mut alternatives = Vec::new();
        for format in formats {
            for (width_open, width) in &widths {
                for (height_open, height) in &heights {
                    for (framerate_open, framerate) in &framerates {
                        let open_fields = width_open + height_open + framerate_open;
                        if let Some(caps) =
                            caps_for(*format, width.clone(), height.clone(), framerate.clone())
                        {
                            alternatives.push((open_fields, caps));
                        }
                    }
                }
            }
        }
        // fewest open fields first: fixation keeps each property a filter leaves unset
        alternatives.sort_by_key(|(open_fields, _)| *open_fields);
        CapsSet::from_alternatives(alternatives.into_iter().map(|(_, caps)| caps).collect())
    }

    pub fn configure(&mut self, absolute_caps: &Caps) -> Result<(), G2gError> {
        let Caps::RawVideo {
            width: Dim::Fixed(width),
            height: Dim::Fixed(height),
            framerate: Rate::Fixed(framerate_q16),
            ..
        } = *absolute_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        self.width = width;
        self.height = height;
        self.framerate_q16 = framerate_q16;
        Ok(())
    }
}
