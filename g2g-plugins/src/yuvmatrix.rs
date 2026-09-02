//! YUV <-> RGB coefficients for one stream's colorimetry.
//!
//! Every stage in this crate that crosses between YUV and RGB (the CPU
//! converter, the Wayland sink's NV12 blit, the compositor background, the GL
//! and WGSL shaders) takes its numbers from here, so two stages of a pipeline
//! cannot disagree about the matrix or the range. The weights are derived from
//! the stream's `(Kr, Kb)` in `g2g-core`; no coefficient is written out
//! anywhere else.

use g2g_core::{Colorimetry, YuvConversion};

/// The integer coefficients are `round(coefficient * 256)`, so each dot product
/// ends in `>> FIXED_POINT_SHIFT` with half added first for round-to-nearest.
const FIXED_POINT_ONE: f32 = 256.0;
const FIXED_POINT_SHIFT: u32 = 8;
const FIXED_POINT_HALF: i32 = 1 << (FIXED_POINT_SHIFT - 1);

/// The 8-bit code a chroma sample carrying no colour takes, what a converter
/// subtracts before applying [`YuvToRgbWeights`].
pub const CHROMA_NEUTRAL: i32 = 128;
/// What an 8-bit sample spans, the units [`YuvToRgbWeights::luma_floor`]
/// divides by.
pub const SAMPLE_SPAN: f32 = 255.0;
/// Limited-range 8-bit bounds: luma starts at 16 and spans 219, chroma spans
/// 224; full range spans all 255.
pub(crate) const SAMPLE_MAX: i32 = 255;
const LIMITED_LUMA_FLOOR: i32 = 16;
const LIMITED_LUMA_SPAN: f32 = 219.0;
const LIMITED_CHROMA_SPAN: f32 = 224.0;

/// Round to nearest, away from zero. `f32::round` is `std`-only, and this crate
/// is `no_std`.
fn round_to_i32(v: f32) -> i32 {
    if v >= 0.0 {
        (v + 0.5) as i32
    } else {
        (v - 0.5) as i32
    }
}

/// How far the luma and chroma samples of a conversion swing, and where its
/// black point sits.
struct SampleSwing {
    luma_span: f32,
    chroma_span: f32,
    luma_floor: i32,
}

fn sample_swing(conversion: YuvConversion) -> SampleSwing {
    match conversion.full_range {
        true => SampleSwing {
            luma_span: SAMPLE_SPAN,
            chroma_span: SAMPLE_SPAN,
            luma_floor: 0,
        },
        false => SampleSwing {
            luma_span: LIMITED_LUMA_SPAN,
            chroma_span: LIMITED_CHROMA_SPAN,
            luma_floor: LIMITED_LUMA_FLOOR,
        },
    }
}

/// YUV -> RGB weights in normalized (0..1 sample) units, what a fragment shader
/// multiplies its sampled luma and chroma by.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct YuvToRgbWeights {
    /// Luma is `luma_gain * (y - luma_floor)`.
    pub luma_gain: f32,
    pub luma_floor: f32,
    pub red_from_cr: f32,
    pub green_from_cb: f32,
    pub green_from_cr: f32,
    pub blue_from_cb: f32,
}

impl YuvToRgbWeights {
    pub fn new(colorimetry: Colorimetry) -> Self {
        let conversion = colorimetry.yuv_conversion();
        let swing = sample_swing(conversion);
        let (kr, kb) = (conversion.luma.kr, conversion.luma.kb);
        let kg = conversion.luma.kg();
        let chroma_gain = SAMPLE_SPAN / swing.chroma_span;
        let red_from_cr = chroma_gain * 2.0 * (1.0 - kr);
        let blue_from_cb = chroma_gain * 2.0 * (1.0 - kb);
        Self {
            luma_gain: SAMPLE_SPAN / swing.luma_span,
            luma_floor: swing.luma_floor as f32 / SAMPLE_SPAN,
            red_from_cr,
            green_from_cb: -blue_from_cb * kb / kg,
            green_from_cr: -red_from_cr * kr / kg,
            blue_from_cb,
        }
    }
}

/// 8-bit fixed-point YUV <-> RGB coefficients, the form the CPU paths convert
/// with. Built from the caps colorimetry, so an untagged stream gets the BT.601
/// limited-range numbers those paths always used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct YuvRgbMatrix {
    luma_from_rgb: [i32; 3],
    cb_from_rgb: [i32; 3],
    cr_from_rgb: [i32; 3],
    /// The luma a black pixel takes: 16 limited, 0 full.
    luma_floor: i32,
    luma_gain: i32,
    red_from_cr: i32,
    green_from_cb: i32,
    green_from_cr: i32,
    blue_from_cb: i32,
}

impl YuvRgbMatrix {
    pub(crate) fn new(colorimetry: Colorimetry) -> Self {
        let conversion = colorimetry.yuv_conversion();
        let swing = sample_swing(conversion);
        let (kr, kb) = (conversion.luma.kr, conversion.luma.kb);
        let kg = conversion.luma.kg();

        let luma_scale = swing.luma_span / SAMPLE_SPAN * FIXED_POINT_ONE;
        let chroma_scale = swing.chroma_span / SAMPLE_SPAN * FIXED_POINT_ONE;
        // Cb is (B - luma) / (2 * (1 - Kb)) and Cr is (R - luma) / (2 * (1 - Kr)),
        // each scaled to the chroma swing.
        let cb_scale = chroma_scale / (2.0 * (1.0 - kb));
        let cr_scale = chroma_scale / (2.0 * (1.0 - kr));

        let weights = YuvToRgbWeights::new(colorimetry);
        Self {
            luma_from_rgb: [
                round_to_i32(luma_scale * kr),
                round_to_i32(luma_scale * kg),
                round_to_i32(luma_scale * kb),
            ],
            cb_from_rgb: [
                round_to_i32(-cb_scale * kr),
                round_to_i32(-cb_scale * kg),
                round_to_i32(chroma_scale * 0.5),
            ],
            cr_from_rgb: [
                round_to_i32(chroma_scale * 0.5),
                round_to_i32(-cr_scale * kg),
                round_to_i32(-cr_scale * kb),
            ],
            luma_floor: swing.luma_floor,
            luma_gain: round_to_i32(weights.luma_gain * FIXED_POINT_ONE),
            red_from_cr: round_to_i32(weights.red_from_cr * FIXED_POINT_ONE),
            green_from_cb: round_to_i32(weights.green_from_cb * FIXED_POINT_ONE),
            green_from_cr: round_to_i32(weights.green_from_cr * FIXED_POINT_ONE),
            blue_from_cb: round_to_i32(weights.blue_from_cb * FIXED_POINT_ONE),
        }
    }

    /// 8-bit RGB -> 8-bit YUV, clamped to the sample range.
    pub(crate) fn rgb_to_yuv(&self, r: i32, g: i32, b: i32) -> (i32, i32, i32) {
        let dot =
            |c: [i32; 3]| (c[0] * r + c[1] * g + c[2] * b + FIXED_POINT_HALF) >> FIXED_POINT_SHIFT;
        (
            (dot(self.luma_from_rgb) + self.luma_floor).clamp(0, SAMPLE_MAX),
            (dot(self.cb_from_rgb) + CHROMA_NEUTRAL).clamp(0, SAMPLE_MAX),
            (dot(self.cr_from_rgb) + CHROMA_NEUTRAL).clamp(0, SAMPLE_MAX),
        )
    }

    /// 8-bit YUV -> 8-bit RGB, clamped to the sample range.
    pub(crate) fn yuv_to_rgb(&self, y: i32, u: i32, v: i32) -> (i32, i32, i32) {
        let luma = self.luma_gain * (y - self.luma_floor);
        let (cb, cr) = (u - CHROMA_NEUTRAL, v - CHROMA_NEUTRAL);
        let round = |v: i32| (v + FIXED_POINT_HALF) >> FIXED_POINT_SHIFT;
        (
            round(luma + self.red_from_cr * cr).clamp(0, SAMPLE_MAX),
            round(luma + self.green_from_cb * cb + self.green_from_cr * cr).clamp(0, SAMPLE_MAX),
            round(luma + self.blue_from_cb * cb).clamp(0, SAMPLE_MAX),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::{ColorRange, MatrixCoefficients};

    fn colorimetry(matrix: MatrixCoefficients, range: ColorRange) -> Colorimetry {
        Colorimetry {
            matrix,
            range,
            ..Colorimetry::UNKNOWN
        }
    }

    /// The derived BT.601 limited-range table must be the one the CPU paths
    /// carried by hand before the caps drove them, or every existing frame
    /// changes colour by a bit.
    #[test]
    fn bt601_limited_reproduces_the_hand_written_table() {
        let m = YuvRgbMatrix::new(Colorimetry::UNKNOWN);
        assert_eq!(m.luma_from_rgb, [66, 129, 25]);
        assert_eq!(m.cb_from_rgb, [-38, -74, 112]);
        assert_eq!(m.cr_from_rgb, [112, -94, -18]);
        assert_eq!(m.luma_floor, 16);
        assert_eq!(m.luma_gain, 298);
        assert_eq!(
            (
                m.red_from_cr,
                m.green_from_cb,
                m.green_from_cr,
                m.blue_from_cb
            ),
            (409, -100, -208, 516)
        );
        // An untagged stream and an explicit bt601 tag resolve the same way.
        assert_eq!(YuvRgbMatrix::new(Colorimetry::BT601), m);
    }

    /// Full range 8-bit BT.601 is the JFIF table.
    #[test]
    fn bt601_full_range_is_the_jfif_table() {
        let m = YuvRgbMatrix::new(colorimetry(MatrixCoefficients::Bt601, ColorRange::Full));
        assert_eq!(m.luma_from_rgb, [77, 150, 29]);
        assert_eq!(m.cb_from_rgb, [-43, -85, 128]);
        assert_eq!(m.cr_from_rgb, [128, -107, -21]);
        assert_eq!(m.luma_floor, 0);
        // Full range needs no luma stretch on the way back.
        assert_eq!(m.luma_gain, 256);
    }

    /// A saturated colour lands on different YUV under BT.709 than BT.601: the
    /// proof that the matrix, not just the range, reaches the arithmetic.
    #[test]
    fn bt709_differs_from_bt601_on_a_saturated_colour() {
        let red = (255, 0, 0);
        let bt601 = YuvRgbMatrix::new(Colorimetry::BT601).rgb_to_yuv(red.0, red.1, red.2);
        let bt709 = YuvRgbMatrix::new(Colorimetry::BT709).rgb_to_yuv(red.0, red.1, red.2);
        assert_ne!(bt601, bt709);
        // BT.709 weights red lower, so pure red is darker and further off neutral.
        assert!(bt709.0 < bt601.0, "{bt709:?} vs {bt601:?}");
    }

    /// Round-tripping through the same matrix returns the colour, within the
    /// rounding of two 8-bit fixed-point steps.
    #[test]
    fn round_trip_recovers_the_colour() {
        for colorimetry in [
            Colorimetry::BT601,
            Colorimetry::BT709,
            Colorimetry::BT2020,
            colorimetry(MatrixCoefficients::Bt709, ColorRange::Full),
        ] {
            let m = YuvRgbMatrix::new(colorimetry);
            for &(r, g, b) in &[(255, 255, 255), (0, 0, 0), (255, 0, 0), (30, 200, 90)] {
                let (y, u, v) = m.rgb_to_yuv(r, g, b);
                let back = m.yuv_to_rgb(y, u, v);
                let off = (back.0 - r)
                    .abs()
                    .max((back.1 - g).abs())
                    .max((back.2 - b).abs());
                assert!(off <= 2, "{colorimetry:?}: ({r},{g},{b}) -> {back:?}");
            }
        }
    }

    /// The shader weights are the integer table's numbers before rounding, so a
    /// GPU convert and a CPU convert of the same stream agree.
    #[test]
    fn shader_weights_match_the_integer_table() {
        for colorimetry in [Colorimetry::BT601, Colorimetry::BT709, Colorimetry::BT2020] {
            let w = YuvToRgbWeights::new(colorimetry);
            let m = YuvRgbMatrix::new(colorimetry);
            assert_eq!(round_to_i32(w.luma_gain * FIXED_POINT_ONE), m.luma_gain);
            assert_eq!(round_to_i32(w.red_from_cr * FIXED_POINT_ONE), m.red_from_cr);
            assert_eq!(
                round_to_i32(w.blue_from_cb * FIXED_POINT_ONE),
                m.blue_from_cb
            );
            assert_eq!(w.luma_floor, m.luma_floor as f32 / SAMPLE_SPAN);
        }
    }
}
