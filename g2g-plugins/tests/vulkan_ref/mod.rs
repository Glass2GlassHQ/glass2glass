// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![allow(dead_code)] // no one test file uses every helper here

#[path = "../decode_ref/mod.rs"]
pub(crate) mod decode_ref;
use decode_ref::two_plane_to_planar;
use g2g_plugins::vulkanvideo::{ColorMatrix, Nv12Frame};

pub(crate) fn planar_frames(frames: Vec<Nv12Frame>, size: (u32, u32)) -> Vec<Vec<u8>> {
    frames
        .into_iter()
        .map(|frame| {
            assert_eq!((frame.width, frame.height), size);
            two_plane_to_planar(
                &[frame.luma(), frame.chroma()].concat(),
                frame.bit_depth,
                size,
            )
        })
        .collect()
}

// the `(Kr, Kb)` luma weights of a matrix, as the converter's private `VideoColorSpace::luma_weights`
pub(crate) fn luma_weights(m: ColorMatrix) -> (f32, f32) {
    match m {
        ColorMatrix::Bt601 => (0.299, 0.114),
        ColorMatrix::Bt709 => (0.2126, 0.0722),
        ColorMatrix::Bt2020Ncl => (0.2627, 0.0593),
    }
}

// IEEE half bits as f32
fn half_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let bits = match exp {
        0 if mant == 0 => (sign as u32) << 31,
        0 => {
            // Subnormal: normalize.
            let mut e = -14i32;
            let mut m = mant as u32;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            ((sign as u32) << 31) | (((e + 127) as u32) << 23) | (m << 13)
        }
        0x1f => ((sign as u32) << 31) | (0xff << 23) | ((mant as u32) << 13),
        _ => {
            ((sign as u32) << 31) | (((exp as i32 - 15 + 127) as u32) << 23) | ((mant as u32) << 13)
        }
    };
    f32::from_bits(bits)
}

// an `Rgba16Float` readback of a `width` x `height` texture, four halves a texel, as RGB
pub(crate) fn rgba16f_to_rgb(bytes: &[u8], (width, height): (usize, usize)) -> Vec<[f32; 3]> {
    assert_eq!(
        bytes.len(),
        width * height * 8,
        "Rgba16Float readback is 8 bytes/pixel"
    );
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|texel| {
            let channel = |index: usize| {
                half_to_f32(u16::from_le_bytes([texel[2 * index], texel[2 * index + 1]]))
            };
            [channel(0), channel(1), channel(2)]
        })
        .collect()
}
