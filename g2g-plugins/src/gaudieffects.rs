//! GStreamer `gaudieffects`: per-pixel colour transforms over packed RGBA /
//! BGRA. Format and geometry are preserved. CPU-only `no_std` baseline.
//!
//! The arithmetic matches GStreamer's elements (Pete Warden's FreeFrame ports)
//! so the same knobs produce the same picture: `solarize` is a triangular
//! invert between `start` / `threshold` / `end`, `chromium` a cosine warp of
//! each channel, `exclusion` / `burn` a per-channel curve, `dodge` a
//! saturating divide, `dilate` a 4-neighbour luminance max (or min when
//! `erode=true`).

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

fn map_rgb(
    format: RawVideoFormat,
    w: u32,
    h: u32,
    src: &[u8],
    mut f: impl FnMut(u8, u8, u8) -> (u8, u8, u8),
) -> Box<[u8]> {
    let bytes = (w as usize) * (h as usize) * 4;
    let mut dst = vec![0u8; bytes].into_boxed_slice();
    dst.copy_from_slice(&src[..bytes]);
    let (r_idx, b_idx) = rgba_rb_offsets(format);
    for px in dst.as_chunks_mut::<4>().0 {
        let (r, g, b) = f(px[r_idx], px[1], px[b_idx]);
        px[r_idx] = r;
        px[1] = g;
        px[b_idx] = b;
    }
    dst
}

fn clamp_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Per-channel RGB effect. `PixelFilter` is implemented once for every type
/// that maps each component through [`RgbEffect::map_channel`].
trait RgbEffect {
    fn state(&self) -> &FilterState;
    fn state_mut(&mut self) -> &mut FilterState;
    fn map_channel(&self, v: u8) -> u8;
}

impl<T: RgbEffect> PixelFilter for T {
    const FORMATS: &'static [RawVideoFormat] = &FORMATS;

    fn state(&self) -> &FilterState {
        RgbEffect::state(self)
    }

    fn state_mut(&mut self) -> &mut FilterState {
        RgbEffect::state_mut(self)
    }

    fn apply(&mut self, format: RawVideoFormat, w: u32, h: u32, src: &[u8]) -> Box<[u8]> {
        map_rgb(format, w, h, src, |r, g, b| {
            (
                self.map_channel(r),
                self.map_channel(g),
                self.map_channel(b),
            )
        })
    }
}

fn set_at_most_256(slot: &mut u32, v: u64) -> Result<(), PropError> {
    if v > 256 {
        return Err(PropError::Value);
    }
    *slot = v as u32;
    Ok(())
}

macro_rules! pixel_element {
    ($ty:ty, $name:literal, $desc:literal) => {
        impl AsyncElement for $ty {
            type ProcessFuture<'a>
                = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
            where
                Self: 'a;

            fn metadata(&self) -> ElementMetadata {
                ElementMetadata::new($name, "Filter/Effect/Video", $desc, "g2g")
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

/// Solarize: a triangular invert of each channel between `start`, `threshold`
/// and `end`.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::gaudieffects::Solarize;
///
/// let solarize = Solarize::new().with_threshold(100);
/// ```
#[derive(Debug)]
pub struct Solarize {
    threshold: u32,
    start: u32,
    end: u32,
    state: FilterState,
}

impl Default for Solarize {
    fn default() -> Self {
        Self::new()
    }
}

impl Solarize {
    pub fn new() -> Self {
        Self {
            threshold: 127,
            start: 50,
            end: 185,
            state: FilterState::new(),
        }
    }

    pub fn with_threshold(mut self, v: u32) -> Self {
        self.threshold = v.min(256);
        self
    }

    pub fn with_start(mut self, v: u32) -> Self {
        self.start = v.min(256);
        self
    }

    pub fn with_end(mut self, v: u32) -> Self {
        self.end = v.min(256);
        self
    }

    const PROPS: &'static [PropertySpec] = &[
        PropertySpec::new("threshold", PropKind::Uint, "Threshold parameter")
            .with_range("0", "256")
            .with_default("127"),
        PropertySpec::new("start", PropKind::Uint, "Start parameter")
            .with_range("0", "256")
            .with_default("50"),
        PropertySpec::new("end", PropKind::Uint, "End parameter")
            .with_range("0", "256")
            .with_default("185"),
    ];

    fn set_prop(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        let v = value.as_uint().ok_or(PropError::Type)?;
        match name {
            "threshold" => set_at_most_256(&mut self.threshold, v),
            "start" => set_at_most_256(&mut self.start, v),
            "end" => set_at_most_256(&mut self.end, v),
            _ => Err(PropError::Unknown),
        }
    }

    fn get_prop(&self, name: &str) -> Option<PropValue> {
        let v = match name {
            "threshold" => self.threshold,
            "start" => self.start,
            "end" => self.end,
            _ => return None,
        };
        Some(PropValue::Uint(v as u64))
    }
}

fn solarize_channel(value: u8, threshold: i32, start: i32, end: i32) -> u8 {
    let period = {
        let p = end - start;
        if p == 0 {
            1
        } else {
            p.abs()
        }
    };
    let up_length = if threshold != start {
        threshold - start
    } else {
        1
    };
    let down_length = if threshold != end { end - threshold } else { 1 };
    if up_length == 0 || down_length == 0 {
        return value;
    }
    let mut param = i32::from(value) + 256 - start;
    param %= period;
    if param < 0 {
        param += period.abs();
    }
    let out = if param < up_length {
        param * 255 / up_length
    } else {
        (down_length - (param - up_length)) * 255 / down_length
    };
    clamp_u8(out)
}

impl RgbEffect for Solarize {
    fn state(&self) -> &FilterState {
        &self.state
    }
    fn state_mut(&mut self) -> &mut FilterState {
        &mut self.state
    }
    fn map_channel(&self, v: u8) -> u8 {
        solarize_channel(v, self.threshold as i32, self.start as i32, self.end as i32)
    }
}

pixel_element!(
    Solarize,
    "Solarize",
    "Solarize tunable inverse in the video signal"
);

/// Chromium: a cosine warp of each channel, controlled by `edge-a` / `edge-b`.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::gaudieffects::Chromium;
///
/// let chromium = Chromium::new().with_edge_a(180);
/// ```
#[derive(Debug)]
pub struct Chromium {
    edge_a: u32,
    edge_b: u32,
    cosine: [u8; COSINE_TABLE_LEN],
    state: FilterState,
}

impl Default for Chromium {
    fn default() -> Self {
        Self::new()
    }
}

impl Chromium {
    pub fn new() -> Self {
        Self {
            edge_a: 200,
            edge_b: 1,
            cosine: cosine_table(),
            state: FilterState::new(),
        }
    }

    pub fn with_edge_a(mut self, v: u32) -> Self {
        self.edge_a = v.min(256);
        self
    }

    pub fn with_edge_b(mut self, v: u32) -> Self {
        self.edge_b = v.min(256);
        self
    }

    const PROPS: &'static [PropertySpec] = &[
        PropertySpec::new("edge-a", PropKind::Uint, "First edge parameter")
            .with_range("0", "256")
            .with_default("200"),
        PropertySpec::new("edge-b", PropKind::Uint, "Second edge parameter")
            .with_range("0", "256")
            .with_default("1"),
    ];

    fn set_prop(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        let v = value.as_uint().ok_or(PropError::Type)?;
        match name {
            "edge-a" => set_at_most_256(&mut self.edge_a, v),
            "edge-b" => set_at_most_256(&mut self.edge_b, v),
            _ => Err(PropError::Unknown),
        }
    }

    fn get_prop(&self, name: &str) -> Option<PropValue> {
        let v = match name {
            "edge-a" => self.edge_a,
            "edge-b" => self.edge_b,
            _ => return None,
        };
        Some(PropValue::Uint(v as u64))
    }
}

/// An angle wraps into the table, as GStreamer's chromium does.
const COSINE_TABLE_MASK: usize = 1023;
const COSINE_TABLE_LEN: usize = COSINE_TABLE_MASK + 1;
/// Half a turn of the table, and the amplitude the cosine is scaled by.
const COSINE_TABLE_HALF_TURN: f64 = 512.0;
/// The pi GStreamer's chromium built its table with.
const GSTREAMER_PI: f64 = 3.141582;

fn cosine_table() -> [u8; COSINE_TABLE_LEN] {
    let mut table = [0u8; COSINE_TABLE_LEN];
    for (angle, slot) in table.iter_mut().enumerate() {
        let radians = angle as f64 * GSTREAMER_PI / COSINE_TABLE_HALF_TURN;
        let cos = crate::mathf::cos(radians) * COSINE_TABLE_HALF_TURN;
        *slot = clamp_u8(cos.abs() as i32);
    }
    table
}

impl RgbEffect for Chromium {
    fn state(&self) -> &FilterState {
        &self.state
    }
    fn state_mut(&mut self) -> &mut FilterState {
        &mut self.state
    }
    fn map_channel(&self, v: u8) -> u8 {
        let value = usize::from(v);
        let angle =
            (value + self.edge_a as usize + (value * self.edge_b as usize) / 2) & COSINE_TABLE_MASK;
        self.cosine[angle]
    }
}

pixel_element!(
    Chromium,
    "Chromium",
    "Chromium breaks the colors of the video signal"
);

/// Exclusion: GStreamer's RGB curve against `factor`.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::gaudieffects::Exclusion;
///
/// let exclusion = Exclusion::new().with_factor(100);
/// ```
#[derive(Debug)]
pub struct Exclusion {
    factor: u32,
    state: FilterState,
}

impl Default for Exclusion {
    fn default() -> Self {
        Self::new()
    }
}

impl Exclusion {
    pub fn new() -> Self {
        Self {
            factor: 175,
            state: FilterState::new(),
        }
    }

    pub fn with_factor(mut self, v: u32) -> Self {
        self.factor = v.clamp(1, 175);
        self
    }

    const PROPS: &'static [PropertySpec] =
        &[
            PropertySpec::new("factor", PropKind::Uint, "Exclusion factor parameter")
                .with_range("1", "175")
                .with_default("175"),
        ];

    fn set_prop(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "factor" => {
                let v = value.as_uint().ok_or(PropError::Type)?;
                if !(1..=175).contains(&v) {
                    return Err(PropError::Value);
                }
                self.factor = v as u32;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_prop(&self, name: &str) -> Option<PropValue> {
        match name {
            "factor" => Some(PropValue::Uint(self.factor as u64)),
            _ => None,
        }
    }
}

fn exclusion_channel(value: u8, cross: u8, factor: i32) -> u8 {
    let v = i32::from(value);
    let cross = i32::from(cross);
    let out = factor - (((factor - v) * (factor - v) / factor) + ((cross * v) / factor));
    clamp_u8(out)
}

impl PixelFilter for Exclusion {
    const FORMATS: &'static [RawVideoFormat] = &FORMATS;

    fn state(&self) -> &FilterState {
        &self.state
    }
    fn state_mut(&mut self) -> &mut FilterState {
        &mut self.state
    }
    fn apply(&mut self, format: RawVideoFormat, w: u32, h: u32, src: &[u8]) -> Box<[u8]> {
        let factor = self.factor as i32;
        map_rgb(format, w, h, src, |red, green, blue| {
            (
                exclusion_channel(red, green, factor),
                exclusion_channel(green, green, factor),
                exclusion_channel(blue, blue, factor),
            )
        })
    }
}

pixel_element!(
    Exclusion,
    "Exclusion",
    "Exclusion exclodes the colors in the video signal"
);

/// Dodge: saturating `256 * c / (256 - c)` per channel. No properties.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::gaudieffects::Dodge;
///
/// let dodge = Dodge::new();
/// ```
#[derive(Debug)]
pub struct Dodge {
    state: FilterState,
}

impl Default for Dodge {
    fn default() -> Self {
        Self::new()
    }
}

impl Dodge {
    pub fn new() -> Self {
        Self {
            state: FilterState::new(),
        }
    }

    const PROPS: &'static [PropertySpec] = &[];

    fn set_prop(&mut self, _name: &str, _value: PropValue) -> Result<(), PropError> {
        Err(PropError::Unknown)
    }

    fn get_prop(&self, _name: &str) -> Option<PropValue> {
        None
    }
}

fn dodge_channel(value: u8) -> u8 {
    let v = u32::from(value);
    let out = (256 * v) / (256 - v);
    out.min(255) as u8
}

impl RgbEffect for Dodge {
    fn state(&self) -> &FilterState {
        &self.state
    }
    fn state_mut(&mut self) -> &mut FilterState {
        &mut self.state
    }
    fn map_channel(&self, v: u8) -> u8 {
        dodge_channel(v)
    }
}

pixel_element!(
    Dodge,
    "Dodge",
    "Dodge saturates the colors in the video signal"
);

/// Burn: `255 - ((255 - c) * 128) / ((c + adjustment) / 2)` per channel.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::gaudieffects::Burn;
///
/// let burn = Burn::new().with_adjustment(100);
/// ```
#[derive(Debug)]
pub struct Burn {
    adjustment: u32,
    state: FilterState,
}

impl Default for Burn {
    fn default() -> Self {
        Self::new()
    }
}

impl Burn {
    pub fn new() -> Self {
        Self {
            adjustment: 175,
            state: FilterState::new(),
        }
    }

    pub fn with_adjustment(mut self, v: u32) -> Self {
        self.adjustment = v.min(256);
        self
    }

    const PROPS: &'static [PropertySpec] =
        &[
            PropertySpec::new("adjustment", PropKind::Uint, "Adjustment parameter")
                .with_range("0", "256")
                .with_default("175"),
        ];

    fn set_prop(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "adjustment" => set_at_most_256(
                &mut self.adjustment,
                value.as_uint().ok_or(PropError::Type)?,
            ),
            _ => Err(PropError::Unknown),
        }
    }

    fn get_prop(&self, name: &str) -> Option<PropValue> {
        match name {
            "adjustment" => Some(PropValue::Uint(self.adjustment as u64)),
            _ => None,
        }
    }
}

fn burn_channel(value: u8, adjustment: u32) -> u8 {
    let a = (u32::from(value) + adjustment) / 2;
    if a == 0 {
        return 0;
    }
    let tmp = ((255 - u32::from(value)) * 128) / a;
    255u32.saturating_sub(tmp).min(255) as u8
}

impl RgbEffect for Burn {
    fn state(&self) -> &FilterState {
        &self.state
    }
    fn state_mut(&mut self) -> &mut FilterState {
        &mut self.state
    }
    fn map_channel(&self, v: u8) -> u8 {
        burn_channel(v, self.adjustment)
    }
}

pixel_element!(Burn, "Burn", "Burn adjusts the colors in the video signal");

/// Dilate: copy the brightest 4-neighbour (or darkest when `erode=true`).
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::gaudieffects::Dilate;
///
/// let dilate = Dilate::new().with_erode(true);
/// ```
#[derive(Debug)]
pub struct Dilate {
    erode: bool,
    state: FilterState,
}

impl Default for Dilate {
    fn default() -> Self {
        Self::new()
    }
}

impl Dilate {
    pub fn new() -> Self {
        Self {
            erode: false,
            state: FilterState::new(),
        }
    }

    pub fn with_erode(mut self, erode: bool) -> Self {
        self.erode = erode;
        self
    }

    const PROPS: &'static [PropertySpec] =
        &[PropertySpec::new("erode", PropKind::Bool, "Erode parameter").with_default("false")];

    fn set_prop(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "erode" => {
                self.erode = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_prop(&self, name: &str) -> Option<PropValue> {
        match name {
            "erode" => Some(PropValue::Bool(self.erode)),
            _ => None,
        }
    }
}

/// Integer luminance weights GStreamer's dilate compares neighbours with.
const LUMA_RED: u32 = 90;
const LUMA_GREEN: u32 = 115;
const LUMA_BLUE: u32 = 51;

fn luminance(r: u8, g: u8, b: u8) -> u32 {
    LUMA_RED * u32::from(r) + LUMA_GREEN * u32::from(g) + LUMA_BLUE * u32::from(b)
}

impl PixelFilter for Dilate {
    const FORMATS: &'static [RawVideoFormat] = &FORMATS;
    fn state(&self) -> &FilterState {
        &self.state
    }
    fn state_mut(&mut self) -> &mut FilterState {
        &mut self.state
    }
    fn apply(&mut self, format: RawVideoFormat, w: u32, h: u32, src: &[u8]) -> Box<[u8]> {
        let width = w as usize;
        let height = h as usize;
        let bytes = width * height * 4;
        let mut dst = vec![0u8; bytes].into_boxed_slice();
        dst.copy_from_slice(&src[..bytes]);
        let (r_idx, b_idx) = rgba_rb_offsets(format);
        let erode = self.erode;
        let pix = |buf: &[u8], x: usize, y: usize| {
            let i = (y * width + x) * 4;
            (buf[i + r_idx], buf[i + 1], buf[i + b_idx])
        };
        for y in 0..height {
            for x in 0..width {
                let mut best = pix(src, x, y);
                let mut best_l = luminance(best.0, best.1, best.2);
                let consider = |nx: usize, ny: usize, best: &mut (u8, u8, u8), best_l: &mut u32| {
                    let n = pix(src, nx, ny);
                    let l = luminance(n.0, n.1, n.2);
                    if (erode && l < *best_l) || (!erode && l > *best_l) {
                        *best = n;
                        *best_l = l;
                    }
                };
                if y + 1 < height {
                    consider(x, y + 1, &mut best, &mut best_l);
                }
                if x + 1 < width {
                    consider(x + 1, y, &mut best, &mut best_l);
                }
                if y > 0 {
                    consider(x, y - 1, &mut best, &mut best_l);
                }
                if x > 0 {
                    consider(x - 1, y, &mut best, &mut best_l);
                }
                let i = (y * width + x) * 4;
                dst[i + r_idx] = best.0;
                dst[i + 1] = best.1;
                dst[i + b_idx] = best.2;
            }
        }
        dst
    }
}

pixel_element!(Dilate, "Dilate", "Dilate copies the brightest pixel around");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::videofx::PixelFilter;

    fn apply<E: PixelFilter>(e: &mut E, src: &[u8], w: u32, h: u32) -> Box<[u8]> {
        e.apply(RawVideoFormat::Rgba8, w, h, src)
    }

    fn pixel<E: PixelFilter>(e: &mut E, px: [u8; 4]) -> [u8; 4] {
        apply(e, &px, 1, 1)[..4].try_into().unwrap()
    }

    fn alpha_and_size<E: PixelFilter>(e: &mut E) {
        let alpha = 0xAB;
        let src = [10u8, 20, 30, alpha, 40, 50, 60, alpha];
        let out = apply(e, &src, 2, 1);
        assert_eq!(out.len(), src.len());
        for px in out.as_chunks::<4>().0 {
            assert_eq!(px[3], alpha);
        }
    }

    #[test]
    fn rgb_effects_keep_alpha_and_geometry() {
        alpha_and_size(&mut Solarize::new());
        alpha_and_size(&mut Chromium::new());
        alpha_and_size(&mut Exclusion::new());
        alpha_and_size(&mut Dodge::new());
        alpha_and_size(&mut Burn::new());
        alpha_and_size(&mut Dilate::new());
    }

    #[test]
    fn dodge_is_the_identity_at_zero_and_never_darkens() {
        // 0 / (256 - 0) = 0: black is a fixed point of colour dodge.
        let black = [0, 0, 0, 255];
        assert_eq!(pixel(&mut Dodge::new(), black), black);
        for v in 0..=u8::MAX {
            let out = dodge_channel(v);
            assert!(out >= v, "{v} -> {out}");
        }
    }

    #[test]
    fn exclusion_at_unit_factor_does_not_divide_by_zero() {
        let _ = pixel(&mut Exclusion::new().with_factor(1), [0, 1, 2, 255]);
    }

    #[test]
    fn exclusion_red_uses_green_like_gstreamer() {
        assert_eq!(
            pixel(&mut Exclusion::new(), [30, 20, 10, 255]),
            [52, 36, 20, 255]
        );
    }

    #[test]
    fn chromium_matches_gstreamer_cosine_table_values() {
        let chromium = Chromium::new().with_edge_a(200).with_edge_b(1);
        for (input, expected) in [(0, 172), (10, 127), (30, 34), (255, 255)] {
            assert_eq!(
                RgbEffect::map_channel(&chromium, input),
                expected,
                "{input}"
            );
        }
    }

    #[test]
    fn burn_black_with_zero_adjustment_stays_black() {
        assert_eq!(
            pixel(&mut Burn::new().with_adjustment(0), [0, 0, 0, 255]),
            [0, 0, 0, 255]
        );
    }

    #[test]
    fn dilate_copies_the_neighbour_that_wins() {
        let hot = [12u8, 34, 56, 99];
        let cold = [0u8, 0, 0, 99];
        let mut src = alloc::vec::Vec::new();
        src.extend_from_slice(&hot);
        src.extend_from_slice(&cold);
        src.extend_from_slice(&cold);
        src.extend_from_slice(&cold);
        let out = apply(&mut Dilate::new(), &src, 2, 2);
        // 4-neighbour of (0,0): right and below become hot; the diagonal stays cold.
        assert_eq!(&out[0..4], &hot);
        assert_eq!(&out[4..8], &hot);
        assert_eq!(&out[8..12], &hot);
        assert_eq!(&out[12..16], &cold);

        let eroded = apply(&mut Dilate::new().with_erode(true), &src, 2, 2);
        assert_eq!(&eroded[0..4], &cold);
    }
}
