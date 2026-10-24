//! Alpha control / chroma key (`alpha`). Rewrites the alpha channel of a packed
//! RGBA / BGRA frame, leaving the colour channels untouched. CPU-only `no_std`.
//!
//! `set` replaces alpha with a constant (`alpha` 0..1); `green` / `blue` are a
//! simple chroma key that makes a pixel transparent when the key channel
//! dominates the other two by `KEY_MARGIN`, opaque otherwise. The key is a
//! dominance test, not the full YUV-distance keyer GStreamer's `alpha` ships, so
//! it stays integer-only and libm-free.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
    RawVideoFormat,
};

use crate::pixel::rgba_rb_offsets;
use crate::videofx::{self, FilterState, PixelFilter};

const FORMATS: [RawVideoFormat; 2] = [RawVideoFormat::Rgba8, RawVideoFormat::Bgra8];

/// How far (0..255) the key channel must exceed the other two colour channels
/// for a chroma-key pixel to be treated as background and made transparent.
const KEY_MARGIN: i16 = 40;

/// What the element does to the alpha channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphaMethod {
    /// Replace alpha with the constant `alpha` value for every pixel.
    Set,
    /// Chroma key: green pixels become transparent, the rest opaque.
    Green,
    /// Chroma key: blue pixels become transparent, the rest opaque.
    Blue,
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::alpha::{Alpha, AlphaMethod};
///
/// let element = Alpha::new()
///     .with_method(AlphaMethod::Green)
///     .with_alpha(0.5);
/// ```
#[derive(Debug)]
pub struct Alpha {
    method: AlphaMethod,
    alpha: f64,
    state: FilterState,
}

impl Default for Alpha {
    fn default() -> Self {
        Self::new()
    }
}

impl Alpha {
    /// A `set` element at full opacity (alpha 1.0); use the builders or the
    /// properties to pick a method or constant.
    pub fn new() -> Self {
        Self {
            method: AlphaMethod::Set,
            alpha: 1.0,
            state: FilterState::new(),
        }
    }

    pub fn with_method(mut self, method: AlphaMethod) -> Self {
        self.method = method;
        self
    }

    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }
}

impl PixelFilter for Alpha {
    const FORMATS: &'static [RawVideoFormat] = &FORMATS;

    fn state(&self) -> &FilterState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut FilterState {
        &mut self.state
    }

    fn apply(&mut self, format: RawVideoFormat, _w: u32, _h: u32, pixels: &mut [u8]) {
        apply_alpha(format, pixels, self.method, alpha_u8(self.alpha));
    }
}

impl AsyncElement for Alpha {
    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Alpha filter",
            "Filter/Effect/Video",
            "Adds or keys an alpha channel on raw video",
            "g2g",
        )
    }
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
        ALPHA_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "method" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.method = method_from_str(s).ok_or(PropError::Value)?;
            }
            "alpha" => self.alpha = value.as_double().ok_or(PropError::Type)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "method" => Some(PropValue::Str(method_to_str(self.method).into())),
            "alpha" => Some(PropValue::Double(self.alpha)),
            _ => None,
        }
    }
}

/// `Alpha`'s settable properties.
static ALPHA_PROPS: &[PropertySpec] = &[
    PropertySpec::new("method", PropKind::Str, "alpha op: set | green | blue")
        .with_enum_values("set | green | blue"),
    PropertySpec::new("alpha", PropKind::Double, "constant alpha for 'set', 0..1"),
];

fn method_from_str(s: &str) -> Option<AlphaMethod> {
    match s {
        "set" => Some(AlphaMethod::Set),
        "green" => Some(AlphaMethod::Green),
        "blue" => Some(AlphaMethod::Blue),
        _ => None,
    }
}

fn method_to_str(m: AlphaMethod) -> &'static str {
    match m {
        AlphaMethod::Set => "set",
        AlphaMethod::Green => "green",
        AlphaMethod::Blue => "blue",
    }
}

fn alpha_u8(alpha: f64) -> u8 {
    // round to nearest without libm; the value is non-negative after clamping.
    (alpha.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

impl PadTemplates for Alpha {
    fn pad_templates() -> alloc::vec::Vec<PadTemplate> {
        videofx::pad_templates::<Self>()
    }
}

/// A chroma key's alpha: 0 (transparent) when `key` exceeds both other colour
/// channels by `KEY_MARGIN`, else 255 (opaque).
fn key_alpha(key: u8, other: u8, another: u8) -> u8 {
    let (key, other, another) = (key as i16, other as i16, another as i16);
    if key - other > KEY_MARGIN && key - another > KEY_MARGIN {
        0
    } else {
        u8::MAX
    }
}

/// Rewrite the alpha channel of every pixel in place, colour untouched.
fn apply_alpha(format: RawVideoFormat, pixels: &mut [u8], method: AlphaMethod, set_alpha: u8) {
    let (r_idx, b_idx) = rgba_rb_offsets(format);
    let pixels = pixels.as_chunks_mut::<4>().0;
    match method {
        AlphaMethod::Set => pixels.iter_mut().for_each(|px| px[3] = set_alpha),
        AlphaMethod::Green => pixels
            .iter_mut()
            .for_each(|px| px[3] = key_alpha(px[1], px[r_idx], px[b_idx])),
        AlphaMethod::Blue => pixels
            .iter_mut()
            .for_each(|px| px[3] = key_alpha(px[b_idx], px[r_idx], px[1])),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keyed_alpha(format: RawVideoFormat, px: [u8; 4], method: AlphaMethod) -> u8 {
        let mut pixel = px;
        apply_alpha(format, &mut pixel, method, u8::MAX);
        pixel[3]
    }

    #[test]
    fn set_replaces_alpha_and_keeps_colour() {
        // alpha 0.5 -> round(127.5) = 128; RGB untouched.
        let mut pixel = [10u8, 20, 30, 255];
        apply_alpha(
            RawVideoFormat::Rgba8,
            &mut pixel,
            AlphaMethod::Set,
            alpha_u8(0.5),
        );
        assert_eq!(pixel, [10, 20, 30, 128]);
        assert_eq!(alpha_u8(1.0), 255);
        assert_eq!(alpha_u8(0.0), 0);
    }

    #[test]
    fn green_key_makes_green_transparent_only() {
        let key = |px: [u8; 4]| keyed_alpha(RawVideoFormat::Rgba8, px, AlphaMethod::Green);
        assert_eq!(key([0, 255, 0, 255]), 0, "green keyed out");
        assert_eq!(key([255, 0, 0, 255]), 255, "red opaque");
        assert_eq!(key([128, 128, 128, 255]), 255, "grey opaque (no dominance)");
    }

    #[test]
    fn blue_key_respects_bgra_channel_order() {
        // BGRA blue pixel is [255, 0, 0, A]; the key must read blue at index 0,
        // not the byte that would be red in RGBA.
        let bgra_blue = [255u8, 0, 0, 255];
        assert_eq!(
            keyed_alpha(RawVideoFormat::Bgra8, bgra_blue, AlphaMethod::Blue),
            0
        );
        // The same bytes read as RGBA are a red pixel, untouched by a blue key.
        assert_eq!(
            keyed_alpha(RawVideoFormat::Rgba8, bgra_blue, AlphaMethod::Blue),
            255
        );
    }
}
