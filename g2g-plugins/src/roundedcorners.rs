//! `roundedcorners`: writes alpha so the picture has rounded corners. Packed
//! RGBA / BGRA. CPU-only `no_std`.
//!
//! `border-radius-px` sets the radius (0 = passthrough). Pixels in a corner
//! square farther than the radius from the inscribed circle's centre become
//! transparent.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
    RawVideoFormat,
};

use crate::videofx::{self, FilterState, PixelFilter};

const FORMATS: [RawVideoFormat; 2] = [RawVideoFormat::Rgba8, RawVideoFormat::Bgra8];

/// # Example
///
/// ```no_run
/// use g2g_plugins::roundedcorners::RoundedCorners;
///
/// let rc = RoundedCorners::new().with_border_radius_px(16);
/// ```
#[derive(Debug)]
pub struct RoundedCorners {
    border_radius_px: u32,
    state: FilterState,
}

impl Default for RoundedCorners {
    fn default() -> Self {
        Self::new()
    }
}

impl RoundedCorners {
    pub fn new() -> Self {
        Self {
            border_radius_px: 0,
            state: FilterState::new(),
        }
    }

    pub fn with_border_radius_px(mut self, px: u32) -> Self {
        self.border_radius_px = px;
        self
    }

    const PROPS: &'static [PropertySpec] = &[PropertySpec::new(
        "border-radius-px",
        PropKind::Uint,
        "Draw rounded corners with given border radius",
    )
    .with_default("0")];
}

impl PixelFilter for RoundedCorners {
    const FORMATS: &'static [RawVideoFormat] = &FORMATS;

    fn state(&self) -> &FilterState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut FilterState {
        &mut self.state
    }

    fn apply(&mut self, _format: RawVideoFormat, w: u32, h: u32, src: &[u8]) -> Box<[u8]> {
        let bytes = (w as usize) * (h as usize) * 4;
        let mut dst = vec![0u8; bytes].into_boxed_slice();
        dst.copy_from_slice(&src[..bytes]);
        let r = self.border_radius_px.min(w / 2).min(h / 2);
        if r == 0 {
            return dst;
        }
        let r2 = (r as u64) * (r as u64);
        for y in 0..h {
            for x in 0..w {
                let (cx, cy) = if x < r && y < r {
                    (r, r)
                } else if x >= w - r && y < r {
                    (w - r, r)
                } else if x < r && y >= h - r {
                    (r, h - r)
                } else if x >= w - r && y >= h - r {
                    (w - r, h - r)
                } else {
                    continue;
                };
                let dx = x as i64 - cx as i64;
                let dy = y as i64 - cy as i64;
                let dist2 = (dx * dx + dy * dy) as u64;
                if dist2 > r2 {
                    dst[(y as usize * w as usize + x as usize) * 4 + 3] = 0;
                }
            }
        }
        dst
    }
}

impl AsyncElement for RoundedCorners {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Rounded Corners",
            "Filter/Effect/Converter/Video",
            "Adds rounded corners to video",
            "g2g",
        )
    }

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
        Self::PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "border-radius-px" => {
                self.border_radius_px = value.as_uint().ok_or(PropError::Type)? as u32;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "border-radius-px" => Some(PropValue::Uint(self.border_radius_px as u64)),
            _ => None,
        }
    }
}

impl PadTemplates for RoundedCorners {
    fn pad_templates() -> Vec<PadTemplate> {
        videofx::pad_templates::<Self>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radius_zero_is_passthrough() {
        let mut rc = RoundedCorners::new();
        let src = [10u8, 20, 30, 255];
        let out = rc.apply(RawVideoFormat::Rgba8, 1, 1, &src);
        assert_eq!(&*out, &src);
    }

    #[test]
    fn corner_pixel_is_cleared() {
        let mut rc = RoundedCorners::new().with_border_radius_px(4);
        let src = vec![255u8; 8 * 8 * 4];
        let out = rc.apply(RawVideoFormat::Rgba8, 8, 8, &src);
        assert_eq!(out[3], 0, "top-left corner is outside the arc");
        let mid = (4 * 8 + 4) * 4 + 3;
        assert_eq!(out[mid], 255, "centre stays opaque");
    }
}
