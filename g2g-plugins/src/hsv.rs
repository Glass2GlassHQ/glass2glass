//! `hsvfilter` and `hsvdetector`. Packed RGBA / BGRA, geometry preserved.
//! CPU-only `no_std` baseline.
//!
//! HSV conversion is the Wikipedia HSL/HSV formula. `hsvfilter` shifts hue and
//! scales/offsets saturation and value; `hsvdetector` writes alpha 255 when a
//! pixel's HSV is inside the configured box, 0 otherwise.

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

use crate::pixel::rgba_rb_offsets;
use crate::videofx::{self, FilterState, PixelFilter};

const FORMATS: [RawVideoFormat; 2] = [RawVideoFormat::Rgba8, RawVideoFormat::Bgra8];
const EPSILON: f32 = 0.00001;

fn rgb_to_hsv(r: u8, g: u8, b: u8) -> [f32; 3] {
    let rf = r as f32 / 255.0;
    let gf = g as f32 / 255.0;
    let bf = b as f32 / 255.0;
    let value = rf.max(gf).max(bf);
    let min = rf.min(gf).min(bf);
    let chroma = value - min;
    let mut hue = if chroma == 0.0 {
        0.0
    } else if (value - rf).abs() < EPSILON {
        60.0 * ((gf - bf) / chroma)
    } else if (value - gf).abs() < EPSILON {
        60.0 * (2.0 + ((bf - rf) / chroma))
    } else if (value - bf).abs() < EPSILON {
        60.0 * (4.0 + ((rf - gf) / chroma))
    } else {
        0.0
    };
    if hue < 0.0 {
        hue += 360.0;
    }
    let saturation = if value == 0.0 { 0.0 } else { chroma / value };
    [
        hue % 360.0,
        saturation.clamp(0.0, 1.0),
        value.clamp(0.0, 1.0),
    ]
}

fn hsv_to_rgb(hsv: [f32; 3]) -> (u8, u8, u8) {
    let c = hsv[2] * hsv[1];
    let hue_prime = hsv[0] / 60.0;
    let x = c * (1.0 - ((hue_prime % 2.0) - 1.0).abs());
    let rgb_prime = if hue_prime < 0.0 {
        [0.0, 0.0, 0.0]
    } else if hue_prime <= 1.0 {
        [c, x, 0.0]
    } else if hue_prime <= 2.0 {
        [x, c, 0.0]
    } else if hue_prime <= 3.0 {
        [0.0, c, x]
    } else if hue_prime <= 4.0 {
        [0.0, x, c]
    } else if hue_prime <= 5.0 {
        [x, 0.0, c]
    } else if hue_prime <= 6.0 {
        [c, 0.0, x]
    } else {
        [0.0, 0.0, 0.0]
    };
    let m = hsv[2] - c;
    (
        ((rgb_prime[0] + m) * 255.0).clamp(0.0, 255.0) as u8,
        ((rgb_prime[1] + m) * 255.0).clamp(0.0, 255.0) as u8,
        ((rgb_prime[2] + m) * 255.0).clamp(0.0, 255.0) as u8,
    )
}

fn wrap_hue(mut h: f32) -> f32 {
    h %= 360.0;
    if h < 0.0 {
        h += 360.0;
    }
    h
}

/// `hsvfilter`: hue / saturation / value transform.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::hsv::HsvFilter;
///
/// let filter = HsvFilter::new().with_hue_shift(180.0);
/// ```
#[derive(Debug)]
pub struct HsvFilter {
    hue_shift: f32,
    saturation_mul: f32,
    saturation_off: f32,
    value_mul: f32,
    value_off: f32,
    state: FilterState,
}

impl Default for HsvFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl HsvFilter {
    pub fn new() -> Self {
        Self {
            hue_shift: 0.0,
            saturation_mul: 1.0,
            saturation_off: 0.0,
            value_mul: 1.0,
            value_off: 0.0,
            state: FilterState::new(),
        }
    }

    pub fn with_hue_shift(mut self, v: f32) -> Self {
        self.hue_shift = v;
        self
    }

    const PROPS: &'static [PropertySpec] = &[
        PropertySpec::new("hue-shift", PropKind::Double, "Hue shifting in degrees")
            .with_default("0"),
        PropertySpec::new(
            "saturation-mul",
            PropKind::Double,
            "Saturation multiplier to apply to the saturation value (before offset)",
        )
        .with_default("1"),
        PropertySpec::new(
            "saturation-off",
            PropKind::Double,
            "Saturation offset to add to the saturation value (after multiplier)",
        )
        .with_default("0"),
        PropertySpec::new(
            "value-mul",
            PropKind::Double,
            "Value multiplier to apply to the value (before offset)",
        )
        .with_default("1"),
        PropertySpec::new(
            "value-off",
            PropKind::Double,
            "Value offset to add to the value (after multiplier)",
        )
        .with_default("0"),
    ];

    fn set_prop(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        let v = value.as_double().ok_or(PropError::Type)? as f32;
        match name {
            "hue-shift" => self.hue_shift = v,
            "saturation-mul" => self.saturation_mul = v,
            "saturation-off" => self.saturation_off = v,
            "value-mul" => self.value_mul = v,
            "value-off" => self.value_off = v,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_prop(&self, name: &str) -> Option<PropValue> {
        let v = match name {
            "hue-shift" => self.hue_shift,
            "saturation-mul" => self.saturation_mul,
            "saturation-off" => self.saturation_off,
            "value-mul" => self.value_mul,
            "value-off" => self.value_off,
            _ => return None,
        };
        Some(PropValue::Double(v as f64))
    }
}

impl PixelFilter for HsvFilter {
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
        let (r_idx, b_idx) = rgba_rb_offsets(format);
        let hue_shift = self.hue_shift;
        let sat_mul = self.saturation_mul;
        let sat_off = self.saturation_off;
        let val_mul = self.value_mul;
        let val_off = self.value_off;
        for px in dst.as_chunks_mut::<4>().0 {
            let mut hsv = rgb_to_hsv(px[r_idx], px[1], px[b_idx]);
            hsv[0] = wrap_hue(hsv[0] + hue_shift);
            hsv[1] = (sat_mul * hsv[1] + sat_off).clamp(0.0, 1.0);
            hsv[2] = (val_mul * hsv[2] + val_off).clamp(0.0, 1.0);
            let (r, g, b) = hsv_to_rgb(hsv);
            px[r_idx] = r;
            px[1] = g;
            px[b_idx] = b;
        }
        dst
    }
}

macro_rules! hsv_element {
    ($ty:ty, $name:literal, $klass:literal, $desc:literal) => {
        impl AsyncElement for $ty {
            type ProcessFuture<'a>
                = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
            where
                Self: 'a;

            fn metadata(&self) -> ElementMetadata {
                ElementMetadata::new($name, $klass, $desc, "g2g")
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

            fn configure_pipeline(
                &mut self,
                absolute_caps: &Caps,
            ) -> Result<ConfigureOutcome, G2gError> {
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
                self.set_prop(name, value)
            }

            fn get_property(&self, name: &str) -> Option<PropValue> {
                self.get_prop(name)
            }
        }

        impl PadTemplates for $ty {
            fn pad_templates() -> Vec<PadTemplate> {
                videofx::pad_templates::<Self>()
            }
        }
    };
}

hsv_element!(
    HsvFilter,
    "HSV filter",
    "Filter/Effect/Converter/Video",
    "Works within the HSV colorspace to apply transformations to incoming frames"
);

/// `hsvdetector`: mark pixels near a configured HSV colour by writing alpha.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::hsv::HsvDetector;
///
/// let det = HsvDetector::new().with_hue_ref(0.0);
/// ```
#[derive(Debug)]
pub struct HsvDetector {
    hue_ref: f32,
    hue_var: f32,
    saturation_ref: f32,
    saturation_var: f32,
    value_ref: f32,
    value_var: f32,
    state: FilterState,
}

impl Default for HsvDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl HsvDetector {
    pub fn new() -> Self {
        Self {
            hue_ref: 0.0,
            hue_var: 10.0,
            saturation_ref: 0.0,
            saturation_var: 0.15,
            value_ref: 0.0,
            value_var: 0.3,
            state: FilterState::new(),
        }
    }

    pub fn with_hue_ref(mut self, v: f32) -> Self {
        self.hue_ref = v;
        self
    }

    const PROPS: &'static [PropertySpec] = &[
        PropertySpec::new("hue-ref", PropKind::Double, "Hue reference in degrees")
            .with_default("0"),
        PropertySpec::new(
            "hue-var",
            PropKind::Double,
            "Allowed hue variation from the reference hue angle, in degrees",
        )
        .with_range("0", "180")
        .with_default("10"),
        PropertySpec::new(
            "saturation-ref",
            PropKind::Double,
            "Reference saturation value",
        )
        .with_range("0", "1")
        .with_default("0"),
        PropertySpec::new(
            "saturation-var",
            PropKind::Double,
            "Allowed saturation variation from the reference value",
        )
        .with_range("0", "1")
        .with_default("0.15"),
        PropertySpec::new("value-ref", PropKind::Double, "Reference value value")
            .with_range("0", "1")
            .with_default("0"),
        PropertySpec::new(
            "value-var",
            PropKind::Double,
            "Allowed value variation from the reference value",
        )
        .with_range("0", "1")
        .with_default("0.3"),
    ];

    fn set_prop(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        let v = value.as_double().ok_or(PropError::Type)? as f32;
        match name {
            "hue-ref" => self.hue_ref = v,
            "hue-var" => {
                if !(0.0..=180.0).contains(&v) {
                    return Err(PropError::Value);
                }
                self.hue_var = v;
            }
            "saturation-ref" => {
                if !(0.0..=1.0).contains(&v) {
                    return Err(PropError::Value);
                }
                self.saturation_ref = v;
            }
            "saturation-var" => {
                if !(0.0..=1.0).contains(&v) {
                    return Err(PropError::Value);
                }
                self.saturation_var = v;
            }
            "value-ref" => {
                if !(0.0..=1.0).contains(&v) {
                    return Err(PropError::Value);
                }
                self.value_ref = v;
            }
            "value-var" => {
                if !(0.0..=1.0).contains(&v) {
                    return Err(PropError::Value);
                }
                self.value_var = v;
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_prop(&self, name: &str) -> Option<PropValue> {
        let v = match name {
            "hue-ref" => self.hue_ref,
            "hue-var" => self.hue_var,
            "saturation-ref" => self.saturation_ref,
            "saturation-var" => self.saturation_var,
            "value-ref" => self.value_ref,
            "value-var" => self.value_var,
            _ => return None,
        };
        Some(PropValue::Double(v as f64))
    }

    fn matches(&self, hsv: [f32; 3]) -> bool {
        let ref_hue_offset = 180.0 - self.hue_ref;
        let mut shifted = hsv[0] + ref_hue_offset;
        if shifted < 0.0 {
            shifted += 360.0;
        }
        shifted %= 360.0;
        (shifted - 180.0).abs() <= self.hue_var
            && (hsv[1] - self.saturation_ref).abs() <= self.saturation_var
            && (hsv[2] - self.value_ref).abs() <= self.value_var
    }
}

impl PixelFilter for HsvDetector {
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
        let (r_idx, b_idx) = rgba_rb_offsets(format);
        for px in dst.as_chunks_mut::<4>().0 {
            let hsv = rgb_to_hsv(px[r_idx], px[1], px[b_idx]);
            px[3] = if self.matches(hsv) { 255 } else { 0 };
        }
        dst
    }
}

hsv_element!(
    HsvDetector,
    "HSV detector",
    "Filter/Effect/Converter/Video",
    "Works within the HSV colorspace to mark positive pixels"
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb_roundtrip_primaries() {
        assert_eq!(hsv_to_rgb(rgb_to_hsv(255, 0, 0)), (255, 0, 0));
        assert_eq!(hsv_to_rgb(rgb_to_hsv(0, 255, 0)), (0, 255, 0));
        assert_eq!(hsv_to_rgb(rgb_to_hsv(0, 0, 255)), (0, 0, 255));
        assert_eq!(hsv_to_rgb(rgb_to_hsv(255, 255, 255)), (255, 255, 255));
        assert_eq!(hsv_to_rgb(rgb_to_hsv(0, 0, 0)), (0, 0, 0));
    }

    #[test]
    fn hue_shift_180_turns_red_cyan() {
        let hsv = rgb_to_hsv(255, 0, 0);
        let shifted = [wrap_hue(hsv[0] + 180.0), hsv[1], hsv[2]];
        assert_eq!(hsv_to_rgb(shifted), (0, 255, 255));
    }

    #[test]
    fn hsvfilter_apply_shifts_red() {
        let mut f = HsvFilter::new().with_hue_shift(180.0);
        let src = [255u8, 0, 0, 255];
        let out = f.apply(RawVideoFormat::Rgba8, 1, 1, &src);
        assert_eq!(&out[..3], &[0, 255, 255]);
        assert_eq!(out[3], 255);
    }

    #[test]
    fn hsvdetector_keeps_red_drops_blue() {
        let mut d = HsvDetector::new().with_hue_ref(0.0);
        d.set_prop("saturation-ref", PropValue::Double(1.0))
            .unwrap();
        d.set_prop("saturation-var", PropValue::Double(1.0))
            .unwrap();
        d.set_prop("value-ref", PropValue::Double(1.0)).unwrap();
        d.set_prop("value-var", PropValue::Double(1.0)).unwrap();
        let red = [255u8, 0, 0, 128];
        let blue = [0u8, 0, 255, 128];
        assert_eq!(d.apply(RawVideoFormat::Rgba8, 1, 1, &red)[3], 255);
        assert_eq!(d.apply(RawVideoFormat::Rgba8, 1, 1, &blue)[3], 0);
    }
}
