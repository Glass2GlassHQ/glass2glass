// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;

use ::hqdn3d::{PlaneState, Strength};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
    RawVideoFormat,
};

use crate::pixel::planar_planes;
use crate::videofx::{self, FilterState, PixelFilter};

const FORMATS: [RawVideoFormat; 3] = [
    RawVideoFormat::I420,
    RawVideoFormat::I422,
    RawVideoFormat::I444,
];

#[derive(Debug)]
pub struct Hqdn3d {
    strength: Strength,
    filters: ::hqdn3d::Hqdn3d,
    // Y, U and V
    planes: [PlaneState; 3],
    state: FilterState,
}

impl Default for Hqdn3d {
    fn default() -> Self {
        Self::new()
    }
}

impl Hqdn3d {
    pub fn new() -> Self {
        let strength = Strength::default();
        Self {
            strength,
            filters: ::hqdn3d::Hqdn3d::new(strength).expect("hqdn3d's default strengths are valid"),
            planes: Default::default(),
            state: FilterState::new(),
        }
    }

    fn set_strength(&mut self, strength: Strength) -> Result<(), PropError> {
        self.filters = ::hqdn3d::Hqdn3d::new(strength).map_err(|_| PropError::Value)?;
        self.strength = strength;
        Ok(())
    }
}

impl PixelFilter for Hqdn3d {
    const FORMATS: &'static [RawVideoFormat] = &FORMATS;

    fn state(&self) -> &FilterState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut FilterState {
        &mut self.state
    }

    fn apply(&mut self, format: RawVideoFormat, w: u32, h: u32, pixels: &mut [u8]) {
        let source = pixels.to_vec();
        let planes = planar_planes(format, w as usize, h as usize);
        for (index, (offset, width, height)) in planes.into_iter().enumerate() {
            let filter = match index {
                0 => self.filters.luma(),
                _ => self.filters.chroma(),
            };
            let bytes = offset..offset + width * height;
            filter
                .denoise_packed(
                    &mut self.planes[index],
                    &source[bytes.clone()],
                    &mut pixels[bytes],
                    width,
                    height,
                )
                .expect("planar_planes lays out a frame of the configured size");
        }
    }

    fn reset(&mut self) {
        self.planes.iter_mut().for_each(PlaneState::reset);
    }
}

impl AsyncElement for Hqdn3d {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        videofx::intercept_caps::<Self>(upstream_caps)
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        videofx::same_caps_constraint::<Self>()
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        videofx::configure(self, absolute_caps)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        videofx::drive(self, packet, out)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        HQDN3D_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "hqdn3d",
            "Filter/Effect/Video",
            "Denoises video with the hqdn3d spatial and temporal low-pass",
            crate::AARON_BOXER,
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        let value = value.as_double().ok_or(PropError::Type)?;
        let current = self.strength;
        let strength = match name {
            "luma-spatial" => Strength::new(
                value,
                current.chroma_spatial(),
                current.luma_temporal(),
                current.chroma_temporal(),
            ),
            "chroma-spatial" => Strength::new(
                current.luma_spatial(),
                value,
                current.luma_temporal(),
                current.chroma_temporal(),
            ),
            "luma-temporal" => Strength::new(
                current.luma_spatial(),
                current.chroma_spatial(),
                value,
                current.chroma_temporal(),
            ),
            "chroma-temporal" => Strength::new(
                current.luma_spatial(),
                current.chroma_spatial(),
                current.luma_temporal(),
                value,
            ),
            _ => return Err(PropError::Unknown),
        };
        self.set_strength(strength.map_err(|_| PropError::Value)?)
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        let current = self.strength;
        let value = match name {
            "luma-spatial" => current.luma_spatial(),
            "chroma-spatial" => current.chroma_spatial(),
            "luma-temporal" => current.luma_temporal(),
            "chroma-temporal" => current.chroma_temporal(),
            _ => return None,
        };
        Some(PropValue::Double(value))
    }
}

// hqdn3d strengths are distances on the 8-bit sample scale, zero turns that pass off
static HQDN3D_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "luma-spatial",
        PropKind::Double,
        "within-frame luma strength",
    )
    .with_default("4"),
    PropertySpec::new(
        "chroma-spatial",
        PropKind::Double,
        "within-frame chroma strength",
    )
    .with_default("3"),
    PropertySpec::new(
        "luma-temporal",
        PropKind::Double,
        "frame-to-frame luma strength",
    )
    .with_default("6"),
    PropertySpec::new(
        "chroma-temporal",
        PropKind::Double,
        "frame-to-frame chroma strength",
    )
    .with_default("4.5"),
];

impl PadTemplates for Hqdn3d {
    fn pad_templates() -> alloc::vec::Vec<PadTemplate> {
        videofx::pad_templates::<Self>()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::videofx::applied;

    const WIDTH: u32 = 16;
    const HEIGHT: u32 = 8;

    const MID_GREY: u8 = 128;
    // noise inside the default strengths, which hqdn3d smooths instead of keeping as edges
    const NOISE_LEVELS: u32 = 8;

    fn noisy_i420_frame(seed: u32) -> Vec<u8> {
        let size = crate::pixel::frame_byte_size(RawVideoFormat::I420, WIDTH, HEIGHT);
        let mut state = seed;
        (0..size)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                MID_GREY + ((state >> 24) % NOISE_LEVELS) as u8
            })
            .collect()
    }

    #[test]
    fn frames_match_the_crate_denoising_the_whole_i420_frame() {
        let mut element = Hqdn3d::new();
        let reference = ::hqdn3d::Hqdn3d::new(Strength::default()).unwrap();
        let mut reference_planes: [PlaneState; 3] = Default::default();
        let [y, u, v] = &mut reference_planes;
        for seed in [1, 2, 3] {
            let frame = noisy_i420_frame(seed);
            let mut expected = frame.clone();
            reference
                .denoise_yuv420(
                    y,
                    u,
                    v,
                    &frame,
                    &mut expected,
                    WIDTH as usize,
                    HEIGHT as usize,
                )
                .unwrap();
            let actual = applied(&mut element, RawVideoFormat::I420, WIDTH, HEIGHT, &frame);
            assert_eq!(actual, expected, "frame {seed}");
        }
    }

    #[test]
    fn a_reset_drops_the_previous_frame() {
        let (first, second) = (noisy_i420_frame(1), noisy_i420_frame(2));
        let alone = applied(
            &mut Hqdn3d::new(),
            RawVideoFormat::I420,
            WIDTH,
            HEIGHT,
            &second,
        );
        let mut element = Hqdn3d::new();
        applied(&mut element, RawVideoFormat::I420, WIDTH, HEIGHT, &first);
        let after_first = applied(&mut element, RawVideoFormat::I420, WIDTH, HEIGHT, &second);
        assert_ne!(after_first, alone);
        element.reset();
        let after_reset = applied(&mut element, RawVideoFormat::I420, WIDTH, HEIGHT, &second);
        assert_eq!(after_reset, alone);
    }

    #[test]
    fn strength_properties_reach_the_filter() {
        let mut element = Hqdn3d::new();
        element
            .set_property("luma-spatial", PropValue::Double(0.0))
            .unwrap();
        element
            .set_property("luma-temporal", PropValue::Double(0.0))
            .unwrap();
        assert_eq!(
            element.get_property("luma-spatial"),
            Some(PropValue::Double(0.0))
        );
        let frame = noisy_i420_frame(1);
        let luma_bytes = (WIDTH * HEIGHT) as usize;
        let output = applied(&mut element, RawVideoFormat::I420, WIDTH, HEIGHT, &frame);
        assert_eq!(output[..luma_bytes], frame[..luma_bytes]);
        assert_ne!(output[luma_bytes..], frame[luma_bytes..]);
        assert_eq!(
            element.set_property("chroma-spatial", PropValue::Double(-1.0)),
            Err(PropError::Value)
        );
    }
}
