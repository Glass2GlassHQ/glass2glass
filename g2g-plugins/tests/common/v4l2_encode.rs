// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use g2g_core::element::AsyncElement;
use g2g_core::frame::PipelinePacket;
use g2g_core::runtime::block_on;
use g2g_core::{Caps, Colorimetry, Dim, Interlace, PixelAspect, Rate, RawVideoFormat};

use crate::h264_hardware_decode::{system_frame, Collect};

pub(crate) const WIDTH: u32 = 640;
pub(crate) const HEIGHT: u32 = 480;
const FRAMES_PER_SECOND_Q16: u32 = 30 << 16;

// a diagonal luma ramp that moves two samples a frame, over flat chroma
pub(crate) fn nv12_frame(index: u64) -> Vec<u8> {
    let luma = (0..HEIGHT)
        .flat_map(|y| (0..WIDTH).map(move |x| ((u64::from(x + y) + 2 * index) % 220 + 16) as u8));
    let chroma = core::iter::repeat_n(128, (WIDTH * HEIGHT / 2) as usize);
    luma.chain(chroma).collect()
}

pub(crate) fn raw_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Fixed(FRAMES_PER_SECOND_Q16),
        interlace: Interlace::Progressive,
        colorimetry: Colorimetry::UNKNOWN,
        pixel_aspect: PixelAspect::SQUARE,
    }
}

// the coded frames `encoder` emits for `frames`
pub(crate) fn encode(mut encoder: impl AsyncElement, frames: &[Vec<u8>]) -> Vec<Vec<u8>> {
    encoder
        .configure_pipeline(&raw_caps())
        .expect("configure takes NV12");
    let mut collect = Collect::default();
    for frame in frames {
        block_on(encoder.process(PipelinePacket::DataFrame(system_frame(frame)), &mut collect))
            .expect("encode a frame");
    }
    collect
        .into_frames()
        .iter()
        .map(|frame| {
            frame
                .domain
                .as_system_slice()
                .expect("system coded frame")
                .to_vec()
        })
        .collect()
}
