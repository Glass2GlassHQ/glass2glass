//! Software colour balance (`videobalance`). Adjusts brightness, contrast, and
//! saturation of a packed RGBA / BGRA frame per pixel, preserving format and
//! geometry. CPU-only `no_std` baseline.
//!
//! `brightness` (-1..1) adds an offset, `contrast` (0..2) scales around mid-grey
//! (128), `saturation` (0..2) lerps each channel toward the pixel's Rec.601 luma
//! (0 = greyscale, 1 = unchanged, >1 = boosted), and `hue` (-1..1 = -180..180deg)
//! rotates the per-pixel chroma vector `(r-luma, g-luma, b-luma)` about the
//! Rec.601 luma axis (so luma is preserved, grey stays grey, and hue=0 is exactly
//! the pre-hue behaviour). The
//! `sin`/`cos` of the hue angle come from the crate's `libm`-free approximation
//! (`mathf`). Per frame the settings fold into a contrast table and one integer
//! 3x3 colour matrix, so each pixel is a table lookup and nine multiplies.

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

/// # Example
///
/// ```no_run
/// use g2g_plugins::videobalance::VideoBalance;
///
/// let balance = VideoBalance::new()
///     .with_brightness(0.1)
///     .with_saturation(1.2);
/// ```
#[derive(Debug)]
pub struct VideoBalance {
    brightness: f64,
    contrast: f64,
    saturation: f64,
    hue: f64,
    state: FilterState,
}

impl Default for VideoBalance {
    fn default() -> Self {
        Self::new()
    }
}

impl VideoBalance {
    /// An identity balance (brightness 0, contrast 1, saturation 1, hue 0); use
    /// the `with_*` builders or the properties to adjust.
    pub fn new() -> Self {
        Self {
            brightness: 0.0,
            contrast: 1.0,
            saturation: 1.0,
            hue: 0.0,
            state: FilterState::new(),
        }
    }

    pub fn with_brightness(mut self, brightness: f64) -> Self {
        self.brightness = brightness;
        self
    }

    pub fn with_contrast(mut self, contrast: f64) -> Self {
        self.contrast = contrast;
        self
    }

    pub fn with_saturation(mut self, saturation: f64) -> Self {
        self.saturation = saturation;
        self
    }

    pub fn with_hue(mut self, hue: f64) -> Self {
        self.hue = hue;
        self
    }
}

impl PixelFilter for VideoBalance {
    const FORMATS: &'static [RawVideoFormat] = &FORMATS;

    fn state(&self) -> &FilterState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut FilterState {
        &mut self.state
    }

    fn apply(&mut self, format: RawVideoFormat, _w: u32, _h: u32, pixels: &mut [u8]) {
        Balance::new(self.brightness, self.contrast, self.saturation, self.hue)
            .apply(format, pixels);
    }
}

impl AsyncElement for VideoBalance {
    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Video balance",
            "Filter/Effect/Video",
            "Adjusts brightness, contrast, hue and saturation",
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
        VIDEOBALANCE_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        let v = value.as_double().ok_or(PropError::Type)?;
        match name {
            "brightness" => self.brightness = v,
            "contrast" => self.contrast = v,
            "saturation" => self.saturation = v,
            "hue" => self.hue = v,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "brightness" => Some(PropValue::Double(self.brightness)),
            "contrast" => Some(PropValue::Double(self.contrast)),
            "saturation" => Some(PropValue::Double(self.saturation)),
            "hue" => Some(PropValue::Double(self.hue)),
            _ => None,
        }
    }
}

/// `VideoBalance`'s settable properties.
static VIDEOBALANCE_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "brightness",
        PropKind::Double,
        "additive brightness, -1..1 (0 = none)",
    ),
    PropertySpec::new(
        "contrast",
        PropKind::Double,
        "contrast about mid-grey, 0..2 (1 = none)",
    ),
    PropertySpec::new(
        "saturation",
        PropKind::Double,
        "saturation, 0..2 (0 = grey, 1 = none)",
    ),
    PropertySpec::new(
        "hue",
        PropKind::Double,
        "hue rotation, -1..1 (-180..180 deg, 0 = none)",
    ),
];

impl PadTemplates for VideoBalance {
    fn pad_templates() -> alloc::vec::Vec<PadTemplate> {
        videofx::pad_templates::<Self>()
    }
}

/// Fractional bits of a [`Balance`] contrast-table entry.
const LEVEL_SHIFT: u32 = 8;
/// Fractional bits of a [`Balance`] matrix coefficient.
const MATRIX_SHIFT: u32 = 16;
const LEVELS: usize = 256;
const MID_GREY: f64 = 128.0;
const MAX_LEVEL: f64 = 255.0;

/// Rec.601 luma weights for R, G, B.
const LUMA_WEIGHTS: [f64; 3] = [0.299, 0.587, 0.114];

/// The Rec.601 luma weights normalized to a unit rotation axis. Rotating the
/// chroma vector about this axis preserves luma exactly: the chroma `d` already
/// satisfies `w.d = 0` (the weights sum to one), so an in-plane rotation leaves
/// `w.d` at zero and the output luma unchanged.
const LUMA_AXIS: [f64; 3] = [0.447_179_5, 0.877_952_5, 0.170_502_5];

/// Frame-constant balance: brightness and contrast as a per-level table, and
/// saturation and hue folded into one colour matrix over (R, G, B).
#[derive(Debug, Clone)]
struct Balance {
    levels: [i32; LEVELS],
    matrix: [[i64; 3]; 3],
}

impl Balance {
    /// Contrast scales about 128 and brightness adds `b*255`. The chroma vector
    /// `(r-luma, g-luma, b-luma)` is rotated about the Rec.601 luma axis by the
    /// hue angle (Rodrigues, luma-preserving) and scaled by saturation before
    /// luma is added back, which is the matrix `W + sat * rot * (I - W)` with
    /// every row of `W` the luma weights.
    fn new(bright: f64, contrast: f64, sat: f64, hue: f64) -> Self {
        let level_scale = (1u32 << LEVEL_SHIFT) as f64;
        let mut levels = [0i32; LEVELS];
        for (level, entry) in levels.iter_mut().enumerate() {
            let adjusted = (level as f64 - MID_GREY) * contrast + MID_GREY + bright * MAX_LEVEL;
            *entry = (adjusted * level_scale) as i32;
        }

        // hue -1..1 -> angle -pi..pi, i.e. turns t = hue / 2.
        let turns = (hue * 0.5) as f32;
        let cos_h = crate::mathf::cos_turns(turns) as f64;
        let sin_h = crate::mathf::sin_turns(turns) as f64;
        // because k is parallel to the weights, k.d = 0, so Rodrigues collapses
        // to d' = d*cos + (k x d)*sin
        let [kr, kg, kb] = LUMA_AXIS;
        let cross = [[0.0, -kb, kg], [kb, 0.0, -kr], [-kg, kr, 0.0]];
        let identity = |i: usize, j: usize| if i == j { 1.0 } else { 0.0 };
        let matrix_scale = (1u64 << MATRIX_SHIFT) as f64;
        let mut matrix = [[0i64; 3]; 3];
        for (i, row) in matrix.iter_mut().enumerate() {
            for (j, coefficient) in row.iter_mut().enumerate() {
                let rotated_chroma: f64 = (0..3)
                    .map(|k| {
                        let rotation = cos_h * identity(i, k) + sin_h * cross[i][k];
                        rotation * (identity(k, j) - LUMA_WEIGHTS[j])
                    })
                    .sum();
                let value = LUMA_WEIGHTS[j] + sat * rotated_chroma;
                *coefficient = crate::mathf::round_to_i64(value * matrix_scale);
            }
        }
        Self { levels, matrix }
    }

    /// Balance every pixel of a packed 4-channel frame in place, alpha untouched.
    fn apply(&self, format: RawVideoFormat, pixels: &mut [u8]) {
        let (r_idx, b_idx) = rgba_rb_offsets(format);
        let shift = LEVEL_SHIFT + MATRIX_SHIFT;
        for px in pixels.as_chunks_mut::<4>().0 {
            let rgb = [
                self.levels[px[r_idx] as usize] as i64,
                self.levels[px[1] as usize] as i64,
                self.levels[px[b_idx] as usize] as i64,
            ];
            let channel = |row: &[i64; 3]| {
                let sum = row[0] * rgb[0] + row[1] * rgb[1] + row[2] * rgb[2];
                (sum >> shift).clamp(0, MAX_LEVEL as i64) as u8
            };
            px[r_idx] = channel(&self.matrix[0]);
            px[1] = channel(&self.matrix[1]);
            px[b_idx] = channel(&self.matrix[2]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bal(bright: f64, contrast: f64, sat: f64, hue: f64) -> Balance {
        Balance::new(bright, contrast, sat, hue)
    }

    fn balance_pixel(format: RawVideoFormat, px: &[u8; 4], balance: &Balance) -> [u8; 4] {
        let mut out = *px;
        balance.apply(format, &mut out);
        out
    }

    #[test]
    fn identity_is_exact() {
        // brightness 0, contrast 1, saturation 1, hue 0 reproduces the input byte
        // for byte (integer values round-trip through f64 exactly; hue=0 gives an
        // exact cos=1/sin=0 so the rotation is the identity).
        let src: alloc::vec::Vec<u8> = (0..=u8::MAX).collect();
        let mut dst = src.clone();
        bal(0.0, 1.0, 1.0, 0.0).apply(RawVideoFormat::Rgba8, &mut dst);
        assert_eq!(dst, src);
    }

    #[test]
    fn brightness_adds_a_scaled_offset() {
        // +0.2 brightness adds 0.2*255 = 51 to each channel; alpha is untouched.
        let px = balance_pixel(
            RawVideoFormat::Rgba8,
            &[100, 100, 100, 200],
            &bal(0.2, 1.0, 1.0, 0.0),
        );
        assert_eq!(px, [151, 151, 151, 200]);
    }

    #[test]
    fn contrast_pivots_around_mid_grey() {
        // contrast 2 doubles the distance from 128: 100 -> 72, 128 stays put.
        assert_eq!(
            balance_pixel(
                RawVideoFormat::Rgba8,
                &[100, 100, 100, 255],
                &bal(0.0, 2.0, 1.0, 0.0)
            )[0],
            72
        );
        assert_eq!(
            balance_pixel(
                RawVideoFormat::Rgba8,
                &[128, 128, 128, 255],
                &bal(0.0, 2.0, 1.0, 0.0)
            )[0],
            128
        );
    }

    #[test]
    fn saturation_zero_is_greyscale() {
        // saturation 0 collapses every channel to the pixel's luma, so R=G=B.
        let px = balance_pixel(
            RawVideoFormat::Rgba8,
            &[200, 100, 50, 255],
            &bal(0.0, 1.0, 0.0, 0.0),
        );
        assert_eq!(px[0], px[1]);
        assert_eq!(px[1], px[2]);
        // Rec.601 luma of (200,100,50) = 59.8 + 58.7 + 5.7 = 124.2 -> 124.
        assert_eq!(px[0], 124);
    }

    #[test]
    fn bgra_weights_luma_by_true_colour() {
        // Same colour as above but stored BGRA: bytes [50,100,200,255]. Luma must
        // match the RGBA case (channel roles respected), not be computed on the
        // raw byte order.
        let px = balance_pixel(
            RawVideoFormat::Bgra8,
            &[50, 100, 200, 255],
            &bal(0.0, 1.0, 0.0, 0.0),
        );
        assert_eq!(px[0], px[2]);
        assert_eq!(px[0], 124, "luma uses true R/G/B, not byte order");
    }

    #[test]
    fn hue_leaves_grey_untouched() {
        // A grey pixel has zero chroma, so any hue rotation is a no-op (to within
        // one LSB of u8 quantization; the luma of grey is not exactly integral in
        // f64, so a rotated near-zero chroma can truncate a step).
        for hue in [-0.5, 0.25, 0.75, 1.0] {
            let px = balance_pixel(
                RawVideoFormat::Rgba8,
                &[128, 128, 128, 255],
                &bal(0.0, 1.0, 1.0, hue),
            );
            for (ch, &v) in px[..3].iter().enumerate() {
                assert!(
                    (v as i32 - 128).abs() <= 1,
                    "hue {hue} shifted grey channel {ch} to {v}"
                );
            }
            assert_eq!(px[3], 255);
        }
    }

    #[test]
    fn hue_rotation_cycles_red_toward_green() {
        // +120 deg (hue 2/3) about the grey axis maps red chroma toward green:
        // the green channel rises well above red, and luma stays close.
        let src = [200u8, 40, 40, 255];
        let px = balance_pixel(RawVideoFormat::Rgba8, &src, &bal(0.0, 1.0, 1.0, 2.0 / 3.0));
        assert!(px[1] > px[0], "green {} should exceed red {}", px[1], px[0]);
        assert!(
            px[1] > px[2],
            "green {} should exceed blue {}",
            px[1],
            px[2]
        );
        // the rotation is luma-preserving by construction; only u8 rounding drifts.
        let luma_in = 0.299 * 200.0 + 0.587 * 40.0 + 0.114 * 40.0;
        let luma_out = 0.299 * px[0] as f64 + 0.587 * px[1] as f64 + 0.114 * px[2] as f64;
        assert!(
            (luma_out - luma_in).abs() < 2.0,
            "luma drifted: {luma_in} -> {luma_out}"
        );
    }

    #[test]
    fn hue_sign_is_opposite_for_opposite_angles() {
        // +h and -h push a colour in opposite chroma directions.
        let src = [200u8, 40, 40, 255];
        let plus = balance_pixel(RawVideoFormat::Rgba8, &src, &bal(0.0, 1.0, 1.0, 0.25));
        let minus = balance_pixel(RawVideoFormat::Rgba8, &src, &bal(0.0, 1.0, 1.0, -0.25));
        assert_ne!(plus, minus);
        // green rises under one sign and blue under the other.
        assert!(
            (plus[1] as i32 - minus[1] as i32).signum()
                != (plus[2] as i32 - minus[2] as i32).signum()
        );
    }
}
