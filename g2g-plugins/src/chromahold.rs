//! Chroma hold (`chromahold`). Keeps the colour of every pixel whose hue is
//! within `tolerance` degrees of the target and turns the rest grey, on a packed
//! RGBA / BGRA frame. Format and geometry are preserved. CPU-only `no_std`
//! baseline.
//!
//! Hue is the HSV hue of the pixel, in whole degrees, computed the way
//! GStreamer's element does (integer, from the min / max / chroma of the three
//! components). The distance is measured around the 360 degree circle, so a
//! target near red matches hues on both sides of the wrap. A pixel with no
//! chroma at all (R = G = B) has no hue; so does a grey target, and a grey
//! target greys the whole frame.
//!
//! The grey level is the pixel's BT.709 luma, so a dropped colour keeps its
//! brightness.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
    RawVideoFormat,
};

use crate::pixel::{bt709_luma, rgba_rb_offsets};
use crate::videofx::{self, FilterState, PixelFilter};

const FORMATS: [RawVideoFormat; 2] = [RawVideoFormat::Rgba8, RawVideoFormat::Bgra8];

const DEFAULT_TARGET_R: u8 = 255;
const DEFAULT_TARGET_G: u8 = 0;
const DEFAULT_TARGET_B: u8 = 0;
/// Degrees of hue either side of the target that are kept.
const DEFAULT_TOLERANCE: u32 = 30;
/// Half the hue circle: past that the wrapped distance shrinks again, so a
/// larger tolerance could not mean anything more.
const MAX_TOLERANCE: u32 = 180;

const DEGREES_PER_CIRCLE: i32 = 360;
/// A colourless pixel (R = G = B) has no hue. GStreamer signals that with an
/// out-of-range value and then feeds it to the distance test unchanged, so the
/// sentinel has to take part in the arithmetic rather than short-circuit it.
const NO_HUE: i32 = -1;

/// # Example
///
/// ```no_run
/// use g2g_plugins::chromahold::ChromaHold;
///
/// // chromahold target-r=0 target-g=255 target-b=0 tolerance=20
/// let hold = ChromaHold::new().with_target(0, 255, 0).with_tolerance(20);
/// ```
#[derive(Debug)]
pub struct ChromaHold {
    target: [u8; 3],
    tolerance: u32,
    state: FilterState,
}

impl Default for ChromaHold {
    fn default() -> Self {
        Self::new()
    }
}

impl ChromaHold {
    /// Holds red within 30 degrees, GStreamer's defaults.
    pub fn new() -> Self {
        Self {
            target: [DEFAULT_TARGET_R, DEFAULT_TARGET_G, DEFAULT_TARGET_B],
            tolerance: DEFAULT_TOLERANCE,
            state: FilterState::new(),
        }
    }

    pub fn with_target(mut self, r: u8, g: u8, b: u8) -> Self {
        self.target = [r, g, b];
        self
    }

    /// Degrees of hue either side of the target that survive; clamped to 180.
    pub fn with_tolerance(mut self, tolerance: u32) -> Self {
        self.tolerance = tolerance.min(MAX_TOLERANCE);
        self
    }

    /// The target hue in degrees, or [`NO_HUE`] when the target is grey.
    fn target_hue(&self) -> i32 {
        rgb_to_hue(
            self.target[0] as i32,
            self.target[1] as i32,
            self.target[2] as i32,
        )
    }
}

/// HSV hue of an 8-bit RGB triple in whole degrees, or [`NO_HUE`] when the
/// triple has no chroma. Integer throughout: the sixth-of-a-circle sector is
/// picked from which component is largest, and the position within it is scaled
/// by 256 before the divide so the rounding survives.
fn rgb_to_hue(r: i32, g: i32, b: i32) -> i32 {
    const SECTOR_DEGREES: i32 = 60;
    const FIXED_ONE: i32 = 256;

    let min = r.min(g).min(b);
    let max = r.max(g).max(b);
    let chroma = max - min;
    if chroma == 0 {
        return NO_HUE;
    }
    let half = chroma >> 1;
    let sector_offset = |numerator: i32| (FIXED_ONE * SECTOR_DEGREES * numerator + half) / chroma;
    let scaled = if max == r {
        sector_offset(g - b)
    } else if max == g {
        sector_offset(b - r) + 2 * SECTOR_DEGREES * FIXED_ONE
    } else {
        sector_offset(r - g) + 4 * SECTOR_DEGREES * FIXED_ONE
    };
    let hue = scaled >> 8;

    if hue >= DEGREES_PER_CIRCLE {
        hue - DEGREES_PER_CIRCLE
    } else if hue < 0 {
        hue + DEGREES_PER_CIRCLE
    } else {
        hue
    }
}

/// Shortest distance between two hues around the circle, in degrees.
fn hue_distance(a: i32, b: i32) -> i32 {
    let forward = a - b;
    let backward = b - a;
    let wrap = |d: i32| if d < 0 { d + DEGREES_PER_CIRCLE } else { d };
    wrap(forward).min(wrap(backward))
}

impl PixelFilter for ChromaHold {
    const FORMATS: &'static [RawVideoFormat] = &FORMATS;

    fn state(&self) -> &FilterState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut FilterState {
        &mut self.state
    }

    fn apply(&mut self, format: RawVideoFormat, w: u32, h: u32, src: &[u8]) -> Box<[u8]> {
        let bytes = (w as usize) * (h as usize) * 4;
        let mut dst = vec![0u8; bytes].into_boxed_slice();
        dst.copy_from_slice(&src[..bytes]);

        let target_hue = self.target_hue();
        let tolerance = self.tolerance as i32;
        let (r_idx, b_idx) = rgba_rb_offsets(format);
        for px in dst.as_chunks_mut::<4>().0 {
            let (r, g, b) = (px[r_idx], px[1], px[b_idx]);
            let hue = rgb_to_hue(r as i32, g as i32, b as i32);
            let held = target_hue != NO_HUE && hue_distance(target_hue, hue) <= tolerance;
            if !held {
                let grey = bt709_luma(r, g, b);
                px[r_idx] = grey;
                px[1] = grey;
                px[b_idx] = grey;
            }
        }
        dst
    }
}

impl AsyncElement for ChromaHold {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Chroma hold filter",
            "Filter/Effect/Video",
            "Removes all colour information except for one colour",
            "g2g",
        )
    }

    /// Reads host memory, so it takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
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
        CHROMAHOLD_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        let v = value.as_uint().ok_or(PropError::Type)?;
        let channel = |v: u64| u8::try_from(v).map_err(|_| PropError::Value);
        match name {
            "target-r" => self.target[0] = channel(v)?,
            "target-g" => self.target[1] = channel(v)?,
            "target-b" => self.target[2] = channel(v)?,
            "tolerance" => {
                if v > MAX_TOLERANCE as u64 {
                    return Err(PropError::Value);
                }
                self.tolerance = v as u32;
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        let v = match name {
            "target-r" => self.target[0] as u64,
            "target-g" => self.target[1] as u64,
            "target-b" => self.target[2] as u64,
            "tolerance" => self.tolerance as u64,
            _ => return None,
        };
        Some(PropValue::Uint(v))
    }
}

/// `ChromaHold`'s settable properties, named as GStreamer's (M1084).
static CHROMAHOLD_PROPS: &[PropertySpec] = &[
    PropertySpec::new("target-r", PropKind::Uint, "the red target")
        .with_range("0", "255")
        .with_default("255"),
    PropertySpec::new("target-g", PropKind::Uint, "the green target")
        .with_range("0", "255")
        .with_default("0"),
    PropertySpec::new("target-b", PropKind::Uint, "the blue target")
        .with_range("0", "255")
        .with_default("0"),
    PropertySpec::new(
        "tolerance",
        PropKind::Uint,
        "degrees of hue either side of the target colour that are kept",
    )
    .with_range("0", "180")
    .with_default("30"),
];

impl PadTemplates for ChromaHold {
    fn pad_templates() -> alloc::vec::Vec<PadTemplate> {
        videofx::pad_templates::<Self>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hue_of_the_primaries() {
        // red 0, green 120, blue 240 degrees.
        assert_eq!(rgb_to_hue(255, 0, 0), 0);
        assert_eq!(rgb_to_hue(0, 255, 0), 120);
        assert_eq!(rgb_to_hue(0, 0, 255), 240);
        // brightness does not change hue.
        assert_eq!(rgb_to_hue(64, 0, 0), 0);
        // grey has none.
        assert_eq!(rgb_to_hue(80, 80, 80), NO_HUE);
    }

    #[test]
    fn hue_distance_wraps_around_the_circle() {
        assert_eq!(hue_distance(10, 350), 20);
        assert_eq!(hue_distance(350, 10), 20);
        assert_eq!(hue_distance(0, 180), 180);
        assert_eq!(hue_distance(45, 45), 0);
    }
}
