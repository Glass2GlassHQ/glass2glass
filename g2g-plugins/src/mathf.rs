//! Small `no_std` float helpers: `core` has no trig intrinsics, so a `libm`-free
//! sine (Bhaskara I's approximation, max error ~0.0016) covers the elements that
//! need a little trig (test-signal synthesis, hue rotation) without pulling a
//! math dep into the baseline.
//!
//! The f64 group below ([`sin`], [`exp`], [`asinh`] and friends) is accurate to
//! within a few ULP instead: filter coefficients need that, since an error of a
//! part in a thousand in a kernel tap lifts a stop band to -60 dB.

/// sin(2*pi*t) for any real `t` (reduced modulo one turn), via Bhaskara I.
pub(crate) fn sin_turns(t: f32) -> f32 {
    // reduce to [0, 1): manual floor (core has no f32::floor)
    let trunc = t as i64 as f32;
    let floor = if t < 0.0 && trunc != t {
        trunc - 1.0
    } else {
        trunc
    };
    let t = t - floor;

    // map to a half-turn x in [0, 1) with the sign of the half
    let (x, sign) = if t < 0.5 {
        (t * 2.0, 1.0f32)
    } else {
        ((t - 0.5) * 2.0, -1.0f32)
    };
    const PI: f32 = core::f32::consts::PI;
    let xr = x * PI;
    sign * (16.0 * xr * (PI - xr)) / (5.0 * PI * PI - 4.0 * xr * (PI - xr))
}

/// cos(2*pi*t): the sine a quarter turn ahead.
pub(crate) fn cos_turns(t: f32) -> f32 {
    sin_turns(t + 0.25)
}

/// sqrt(x) for x >= 0 (returns 0 for x <= 0), Newton-Raphson. `core` has no
/// `f64::sqrt` intrinsic in `no_std`, and the baseline pulls no math dep.
pub(crate) fn sqrt(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut y = x;
    for _ in 0..64 {
        let next = 0.5 * (y + x / y);
        if (next - y).abs() <= f64::EPSILON * next {
            return next;
        }
        y = next;
    }
    y
}

const LN2: f64 = core::f64::consts::LN_2;
const LOG2_E: f64 = core::f64::consts::LOG2_E;

/// log2(x) for x > 0 (returns 0 for x <= 0). Splits `x` into its binary exponent
/// and mantissa via the IEEE-754 bits, then evaluates log2 of the mantissa in
/// [1, 2) with the fast-converging atanh series for ln.
pub(crate) fn log2(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let bits = x.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i64 - 1023;
    // mantissa in [1, 2): force the exponent field to the bias.
    let m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | (1023u64 << 52));
    // ln(m) = 2*(s + s^3/3 + s^5/5 + ...), s = (m-1)/(m+1), |s| <= 1/3 here.
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let mut term = s;
    let mut sum = 0.0;
    let mut k = 1.0;
    for _ in 0..12 {
        sum += term / k;
        term *= s2;
        k += 2.0;
    }
    exp as f64 + 2.0 * sum * LOG2_E
}

/// 2^y for any real y (underflows to 0, overflows saturate to `f64::MAX`).
/// Splits `y = k + f`, builds `2^k` from the exponent field and `2^f` from the
/// Taylor series of exp.
pub(crate) fn exp2(y: f64) -> f64 {
    let k = {
        let t = y as i64;
        if y < 0.0 && (t as f64) != y {
            t - 1
        } else {
            t
        }
    };
    let f = y - k as f64;
    // 2^f = exp(f*ln2), f in [0,1) so the argument is in [0, ln2).
    let z = f * LN2;
    let mut term = 1.0;
    let mut two_f = 0.0;
    let mut n = 0.0;
    for _ in 0..18 {
        two_f += term;
        n += 1.0;
        term *= z / n;
    }
    let exp_field = k + 1023;
    if exp_field <= 0 {
        return 0.0;
    }
    if exp_field >= 0x7ff {
        return f64::MAX;
    }
    let two_k = f64::from_bits((exp_field as u64) << 52);
    two_k * two_f
}

/// x^y for x >= 0 (returns 0 for x <= 0), via `exp2(y * log2(x))`. Accurate to a
/// few ULP over the 8-bit LUT ranges the video elements build.
pub(crate) fn powf(x: f64, y: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if y == 0.0 {
        return 1.0;
    }
    exp2(y * log2(x))
}

/// e^y, via `exp2(y * log2(e))`.
pub(crate) fn exp(y: f64) -> f64 {
    exp2(y * LOG2_E)
}

/// ln(x) for x > 0 (returns 0 for x <= 0).
pub(crate) fn ln(x: f64) -> f64 {
    log2(x) * LN2
}

/// Round to the nearest integer, halves away from zero.
fn round_to_i64(v: f64) -> i64 {
    if v >= 0.0 {
        (v + 0.5) as i64
    } else {
        (v - 0.5) as i64
    }
}

/// sin(x) in radians. The angle is first reduced by whole turns, which costs the
/// low bits of very large arguments, then evaluated by its Taylor series.
pub(crate) fn sin(x: f64) -> f64 {
    let turns = round_to_i64(x / core::f64::consts::TAU);
    let r = x - turns as f64 * core::f64::consts::TAU;
    let r2 = r * r;
    let mut term = r;
    let mut sum = 0.0;
    let mut n = 1.0;
    for _ in 0..16 {
        sum += term;
        term *= -r2 / ((n + 1.0) * (n + 2.0));
        n += 2.0;
    }
    sum
}

/// cos(x) in radians.
pub(crate) fn cos(x: f64) -> f64 {
    sin(x + core::f64::consts::FRAC_PI_2)
}

/// tan(x) in radians (infinite at the poles).
pub(crate) fn tan(x: f64) -> f64 {
    sin(x) / cos(x)
}

pub(crate) fn sinh(x: f64) -> f64 {
    0.5 * (exp(x) - exp(-x))
}

pub(crate) fn cosh(x: f64) -> f64 {
    0.5 * (exp(x) + exp(-x))
}

/// asinh(x) = ln(x + sqrt(x^2 + 1)), evaluated on |x| so a negative argument
/// does not cancel the two terms against each other.
pub(crate) fn asinh(x: f64) -> f64 {
    let a = x.abs();
    let r = ln(a + sqrt(a * a + 1.0));
    if x < 0.0 {
        -r
    } else {
        r
    }
}

/// Largest integer not above `x` (core has no `f64::floor`). The `i64` cast
/// saturates, so a magnitude past `i64::MAX` comes back clamped.
pub(crate) fn floor(x: f64) -> f64 {
    let truncated = x as i64 as f64;
    if x < truncated {
        truncated - 1.0
    } else {
        truncated
    }
}

/// Smallest integer not below `x`.
pub(crate) fn ceil(x: f64) -> f64 {
    let truncated = x as i64 as f64;
    if x > truncated {
        truncated + 1.0
    } else {
        truncated
    }
}

#[cfg(test)]
mod transcendental_tests {
    use super::{asinh, cos, cosh, exp, ln, sin, sinh, tan};

    #[test]
    fn sin_cos_match_known_angles() {
        let pi = core::f64::consts::PI;
        assert!(sin(0.0).abs() < 1e-12);
        assert!((sin(pi / 6.0) - 0.5).abs() < 1e-12);
        assert!((sin(pi / 2.0) - 1.0).abs() < 1e-12);
        assert!(sin(pi).abs() < 1e-12);
        assert!((cos(0.0) - 1.0).abs() < 1e-12);
        assert!((cos(pi) + 1.0).abs() < 1e-12);
        // an argument many turns out still reduces.
        assert!((sin(100.0 * pi + pi / 6.0) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn sin_squared_plus_cos_squared_is_one() {
        for step in 0..64 {
            let x = -8.0 + step as f64 * 0.25;
            let s = sin(x);
            let c = cos(x);
            assert!((s * s + c * c - 1.0).abs() < 1e-12, "at {x}");
        }
    }

    #[test]
    fn tan_matches_the_ratio() {
        let x = 0.5;
        assert!((tan(x) - sin(x) / cos(x)).abs() < 1e-15);
        // tan(pi/4) == 1.
        assert!((tan(core::f64::consts::FRAC_PI_4) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn exp_and_ln_round_trip() {
        assert!((exp(0.0) - 1.0).abs() < 1e-12);
        assert!((exp(1.0) - core::f64::consts::E).abs() < 1e-12);
        assert!((ln(core::f64::consts::E) - 1.0).abs() < 1e-12);
        for &x in &[0.1f64, 1.0, 7.5, 1000.0] {
            assert!((exp(ln(x)) - x).abs() < 1e-9 * x, "at {x}");
        }
    }

    #[test]
    fn hyperbolics_satisfy_their_identities() {
        for &x in &[0.0f64, 0.3, 1.7, -2.5] {
            let s = sinh(x);
            let c = cosh(x);
            assert!((c * c - s * s - 1.0).abs() < 1e-9, "at {x}");
            assert!((asinh(s) - x).abs() < 1e-9, "at {x}");
        }
    }
}

#[cfg(test)]
mod sqrt_tests {
    use super::{exp2, log2, powf, sqrt};

    #[test]
    fn sqrt_known_values() {
        assert!((sqrt(4.0) - 2.0).abs() < 1e-9);
        assert!((sqrt(2.0) - core::f64::consts::SQRT_2).abs() < 1e-9);
        assert!((sqrt(1e-10) - 1e-5).abs() < 1e-12);
        assert!((sqrt(1e10) - 1e5).abs() < 1e-3);
        assert_eq!(sqrt(0.0), 0.0);
        assert_eq!(sqrt(-1.0), 0.0);
    }

    #[test]
    fn log2_and_exp2_round_trip() {
        assert!((log2(1.0)).abs() < 1e-9);
        assert!((log2(2.0) - 1.0).abs() < 1e-9);
        assert!((log2(8.0) - 3.0).abs() < 1e-9);
        assert!((log2(0.25) + 2.0).abs() < 1e-9);
        assert!((exp2(0.0) - 1.0).abs() < 1e-9);
        assert!((exp2(10.0) - 1024.0).abs() < 1e-6);
        assert!((exp2(-2.0) - 0.25).abs() < 1e-9);
    }

    #[test]
    fn powf_known_values() {
        assert!((powf(0.5, 2.0) - 0.25).abs() < 1e-9);
        assert!((powf(2.0, 10.0) - 1024.0).abs() < 1e-5);
        assert!((powf(0.25, 0.5) - 0.5).abs() < 1e-6);
        assert!((powf(0.7, 1.0) - 0.7).abs() < 1e-9);
        assert_eq!(powf(0.0, 2.0), 0.0);
        assert_eq!(powf(5.0, 0.0), 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cardinal_points() {
        assert!((sin_turns(0.0) - 0.0).abs() < 1e-6);
        assert!((sin_turns(0.25) - 1.0).abs() < 1e-3);
        assert!((sin_turns(0.5) - 0.0).abs() < 1e-3);
        assert!((sin_turns(0.75) + 1.0).abs() < 1e-3);
        assert!((cos_turns(0.0) - 1.0).abs() < 1e-6);
        assert!((cos_turns(0.25) - 0.0).abs() < 1e-3);
    }

    #[test]
    fn negative_and_wrapping_reduce() {
        // sin is periodic and odd: sin_turns(-t) == sin_turns(1-t) == -sin_turns(t).
        for &t in &[0.1f32, 0.3, 0.42] {
            assert!((sin_turns(-t) - sin_turns(1.0 - t)).abs() < 1e-4);
            assert!((sin_turns(-t) + sin_turns(t)).abs() < 1e-4);
        }
        // a multi-turn argument wraps to its fractional part.
        assert!((sin_turns(3.25) - sin_turns(0.25)).abs() < 1e-4);
    }

    #[test]
    fn floor_and_ceil_bracket_a_fraction() {
        assert_eq!(floor(2.5), 2.0);
        assert_eq!(ceil(2.5), 3.0);
        assert_eq!(floor(-0.45), -1.0);
        assert_eq!(ceil(-0.45), 0.0);
        // an exact integer is its own floor and ceiling.
        assert_eq!(floor(-3.0), -3.0);
        assert_eq!(ceil(-3.0), -3.0);
    }
}
