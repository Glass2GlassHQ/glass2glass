//! Gamma correction (`gamma`). Applies `out = 255 * (in/255)^(1/gamma)` to the
//! R/G/B channels of a packed RGBA / BGRA frame via a 256-entry lookup table,
//! leaving alpha and geometry unchanged. `gamma > 1` brightens mid-tones,
//! `gamma < 1` darkens them, `gamma == 1` is the identity. CPU-only `no_std`.
//!
//! The `pow` comes from the crate's `libm`-free approximation
//! (`mathf::powf`), evaluated 256 times per gamma change (the LUT is
//! rebuilt only when `gamma` changes), not per pixel.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
    RawVideoFormat,
};

use crate::videofx::{self, FilterState, PixelFilter};

const FORMATS: [RawVideoFormat; 2] = [RawVideoFormat::Rgba8, RawVideoFormat::Bgra8];

/// Build the gamma LUT: `lut[v] = round(255 * (v/255)^(1/gamma))`. `gamma <= 0`
/// or `gamma == 1` yields the identity table.
fn build_lut(gamma: f64) -> [u8; 256] {
    let mut lut = [0u8; 256];
    if gamma <= 0.0 || gamma == 1.0 {
        for (v, e) in lut.iter_mut().enumerate() {
            *e = v as u8;
        }
        return lut;
    }
    let inv = 1.0 / gamma;
    for (v, e) in lut.iter_mut().enumerate() {
        let normalized = v as f64 / 255.0;
        let corrected = crate::mathf::powf(normalized, inv) * 255.0;
        *e = (corrected + 0.5).clamp(0.0, 255.0) as u8;
    }
    lut
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::gamma::Gamma;
///
/// // gst-launch equivalent: gamma gamma=2.2
/// let gamma = Gamma::new().with_gamma(2.2);
/// ```
#[derive(Debug)]
pub struct Gamma {
    gamma: f64,
    lut: [u8; 256],
    state: FilterState,
}

impl Default for Gamma {
    fn default() -> Self {
        Self::new()
    }
}

impl Gamma {
    /// Identity gamma (1.0); use the builder or the `gamma` property to adjust.
    pub fn new() -> Self {
        Self {
            gamma: 1.0,
            lut: build_lut(1.0),
            state: FilterState::new(),
        }
    }

    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self.lut = build_lut(gamma);
        self
    }
}

impl PixelFilter for Gamma {
    const FORMATS: &'static [RawVideoFormat] = &FORMATS;

    fn state(&self) -> &FilterState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut FilterState {
        &mut self.state
    }

    fn apply(&mut self, _format: RawVideoFormat, w: u32, h: u32, src: &[u8]) -> Box<[u8]> {
        let mut pixels: Box<[u8]> = src[..(w as usize) * (h as usize) * 4].into();
        apply_gamma(&mut pixels, &self.lut);
        pixels
    }
}

impl AsyncElement for Gamma {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

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
        GAMMA_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Gamma",
            "Filter/Effect/Video",
            "Performs gamma correction",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "gamma" => {
                self.gamma = value.as_double().ok_or(PropError::Type)?;
                self.lut = build_lut(self.gamma);
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "gamma" => Some(PropValue::Double(self.gamma)),
            _ => None,
        }
    }
}

static GAMMA_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "gamma",
    PropKind::Double,
    "gamma value (>1 brightens, 1 = none)",
)];

impl PadTemplates for Gamma {
    fn pad_templates() -> alloc::vec::Vec<PadTemplate> {
        videofx::pad_templates::<Self>()
    }
}

/// Apply the LUT to R/G/B of every packed pixel, leaving alpha (byte 3) as is.
/// The LUT is symmetric in R/G/B, so RGBA and BGRA both map channels 0/1/2.
fn apply_gamma(pixels: &mut [u8], lut: &[u8; 256]) {
    for px in pixels.as_chunks_mut::<4>().0 {
        px[0] = lut[px[0] as usize];
        px[1] = lut[px[1] as usize];
        px[2] = lut[px[2] as usize];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_gamma_is_exact() {
        let lut = build_lut(1.0);
        for v in 0..=255u8 {
            assert_eq!(lut[v as usize], v);
        }
    }

    #[test]
    fn endpoints_are_fixed() {
        let lut = build_lut(2.2);
        assert_eq!(lut[0], 0);
        assert_eq!(lut[255], 255);
    }

    #[test]
    fn lut_is_monotonic() {
        let lut = build_lut(2.2);
        for v in 1..=255usize {
            assert!(lut[v] >= lut[v - 1], "non-monotonic at {v}");
        }
    }

    #[test]
    fn gamma_above_one_brightens_midtones() {
        let lut = build_lut(2.0);
        // mid-grey rises: (128/255)^(1/2) * 255 ~ 181.
        assert!(
            lut[128] > 128,
            "gamma 2.0 should brighten 128, got {}",
            lut[128]
        );
        assert!((lut[128] as i32 - 181).abs() <= 1);
    }

    #[test]
    fn alpha_is_preserved() {
        let lut = build_lut(2.0);
        let mut pixel = [10u8, 20, 30, 200];
        apply_gamma(&mut pixel, &lut);
        assert_eq!(pixel[3], 200);
    }
}
