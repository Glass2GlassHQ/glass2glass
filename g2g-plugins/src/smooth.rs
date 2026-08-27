//! Tolerance-limited smoothing (`smooth`). Averages each I420 sample with the
//! neighbours whose level is within `tolerance` of it, so flat areas lose noise
//! while anything the tolerance cannot bridge, an edge above all, stays put.
//! Format and geometry are preserved. CPU-only `no_std` baseline.
//!
//! `filter-size` sets the reach of the neighbourhood: that many samples either
//! side horizontally, and vertically the same below but one row more above,
//! which is the lopsided window GStreamer's element builds. The sample itself
//! is always counted, so a neighbourhood with no qualifying neighbour leaves
//! the sample alone. `luma-only` (the default) leaves the chroma planes
//! untouched; `active=false` turns the whole filter into a pass-through.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
    RawVideoFormat,
};

use crate::pixel::{frame_byte_size, planar_planes};
use crate::videofx::{self, FilterState, PixelFilter};

const FORMATS: [RawVideoFormat; 1] = [RawVideoFormat::I420];

const DEFAULT_ACTIVE: bool = true;
const DEFAULT_TOLERANCE: i32 = 8;
const DEFAULT_FILTER_SIZE: i32 = 3;
const DEFAULT_LUMA_ONLY: bool = true;

/// # Example
///
/// ```no_run
/// use g2g_plugins::smooth::Smooth;
///
/// // smooth tolerance=16 filter-size=2
/// let smooth = Smooth::new().with_tolerance(16).with_filter_size(2);
/// ```
#[derive(Debug)]
pub struct Smooth {
    active: bool,
    tolerance: i32,
    filter_size: i32,
    luma_only: bool,
    state: FilterState,
}

impl Default for Smooth {
    fn default() -> Self {
        Self::new()
    }
}

impl Smooth {
    /// GStreamer's defaults: active, tolerance 8, filter size 3, luma only.
    pub fn new() -> Self {
        Self {
            active: DEFAULT_ACTIVE,
            tolerance: DEFAULT_TOLERANCE,
            filter_size: DEFAULT_FILTER_SIZE,
            luma_only: DEFAULT_LUMA_ONLY,
            state: FilterState::new(),
        }
    }

    pub fn with_active(mut self, active: bool) -> Self {
        self.active = active;
        self
    }

    /// Level difference a neighbour must stay within to be averaged in.
    pub fn with_tolerance(mut self, tolerance: i32) -> Self {
        self.tolerance = tolerance;
        self
    }

    pub fn with_filter_size(mut self, filter_size: i32) -> Self {
        self.filter_size = filter_size;
        self
    }

    pub fn with_luma_only(mut self, luma_only: bool) -> Self {
        self.luma_only = luma_only;
        self
    }
}

/// Average each sample with the neighbours within `tolerance` of it.
fn smooth_plane(dst: &mut [u8], src: &[u8], w: usize, h: usize, tolerance: i32, filter_size: i32) {
    let reach = filter_size.max(0) as usize;
    for y in 0..h {
        let rows = y.saturating_sub(reach + 1)..h.min(y + reach + 1);
        for x in 0..w {
            let columns = x.saturating_sub(reach)..w.min(x + reach + 1);
            let reference = src[y * w + x] as i32;
            // The sample itself opens the average and is counted again as a
            // member of its own neighbourhood, so it carries double weight.
            let mut sum = reference;
            let mut counted = 1;
            for row in rows.clone() {
                for column in columns.clone() {
                    let neighbour = src[row * w + column] as i32;
                    if (neighbour - reference).abs() < tolerance {
                        sum += neighbour;
                        counted += 1;
                    }
                }
            }
            dst[y * w + x] = (sum / counted) as u8;
        }
    }
}

impl PixelFilter for Smooth {
    const FORMATS: &'static [RawVideoFormat] = &FORMATS;

    fn state(&self) -> &FilterState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut FilterState {
        &mut self.state
    }

    fn apply(&mut self, format: RawVideoFormat, w: u32, h: u32, src: &[u8]) -> Box<[u8]> {
        let bytes = frame_byte_size(format, w, h);
        let mut dst = vec![0u8; bytes].into_boxed_slice();
        dst.copy_from_slice(&src[..bytes]);
        if !self.active {
            return dst;
        }

        let planes = planar_planes(format, w as usize, h as usize);
        let filtered = if self.luma_only { 1 } else { planes.len() };
        for &(offset, plane_w, plane_h) in &planes[..filtered] {
            let plane = offset..offset + plane_w * plane_h;
            smooth_plane(
                &mut dst[plane.clone()],
                &src[plane],
                plane_w,
                plane_h,
                self.tolerance,
                self.filter_size,
            );
        }
        dst
    }
}

impl AsyncElement for Smooth {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Smooth effect",
            "Filter/Effect/Video",
            "Smooths raw video without crossing edges",
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
        SMOOTH_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "active" => self.active = value.as_bool().ok_or(PropError::Type)?,
            "luma-only" => self.luma_only = value.as_bool().ok_or(PropError::Type)?,
            "tolerance" => {
                let v = value.as_int().ok_or(PropError::Type)?;
                self.tolerance = i32::try_from(v).map_err(|_| PropError::Value)?;
            }
            "filter-size" => {
                let v = value.as_int().ok_or(PropError::Type)?;
                self.filter_size = i32::try_from(v).map_err(|_| PropError::Value)?;
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "active" => Some(PropValue::Bool(self.active)),
            "luma-only" => Some(PropValue::Bool(self.luma_only)),
            "tolerance" => Some(PropValue::Int(self.tolerance as i64)),
            "filter-size" => Some(PropValue::Int(self.filter_size as i64)),
            _ => None,
        }
    }
}

/// `Smooth`'s settable properties, named as GStreamer's (M1084).
static SMOOTH_PROPS: &[PropertySpec] = &[
    PropertySpec::new("active", PropKind::Bool, "process video").with_default("true"),
    PropertySpec::new(
        "tolerance",
        PropKind::Int,
        "level difference a neighbour must stay within to be averaged in",
    )
    .with_default("8"),
    PropertySpec::new(
        "filter-size",
        PropKind::Int,
        "samples the neighbourhood reaches either side",
    )
    .with_default("3"),
    PropertySpec::new(
        "luma-only",
        PropKind::Bool,
        "filter the luma plane only, leaving chroma untouched",
    )
    .with_default("true"),
];

impl PadTemplates for Smooth {
    fn pad_templates() -> alloc::vec::Vec<PadTemplate> {
        videofx::pad_templates::<Self>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_flat_plane_is_untouched() {
        const LEVEL: u8 = 77;
        let (w, h) = (8, 8);
        let src = vec![LEVEL; w * h];
        let mut dst = vec![0u8; w * h];
        smooth_plane(&mut dst, &src, w, h, DEFAULT_TOLERANCE, DEFAULT_FILTER_SIZE);
        assert_eq!(dst, src);
    }

    #[test]
    fn a_step_taller_than_the_tolerance_survives() {
        // No neighbour across the step qualifies, so each side averages only
        // its own level and comes out unchanged.
        const DARK: u8 = 30;
        const LIGHT: u8 = 200;
        let (w, h) = (8, 8);
        let src: alloc::vec::Vec<u8> = (0..w * h)
            .map(|i| if i % w < w / 2 { DARK } else { LIGHT })
            .collect();
        let mut dst = vec![0u8; w * h];
        smooth_plane(&mut dst, &src, w, h, DEFAULT_TOLERANCE, DEFAULT_FILTER_SIZE);
        assert_eq!(dst, src);
    }

    #[test]
    fn a_step_inside_the_tolerance_is_bridged() {
        // With a tolerance wider than the step, samples near the step average
        // across it, so the two levels move toward each other.
        const DARK: u8 = 100;
        const LIGHT: u8 = 104;
        const TOLERANCE: i32 = 8;
        let (w, h) = (8, 8);
        let src: alloc::vec::Vec<u8> = (0..w * h)
            .map(|i| if i % w < w / 2 { DARK } else { LIGHT })
            .collect();
        let mut dst = vec![0u8; w * h];
        smooth_plane(&mut dst, &src, w, h, TOLERANCE, DEFAULT_FILTER_SIZE);
        let at = |x: usize, y: usize| dst[y * w + x];
        assert!(at(w / 2 - 1, h / 2) > DARK, "dark side lifted");
        assert!(at(w / 2, h / 2) < LIGHT, "light side pulled down");
    }
}
