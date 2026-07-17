//! Small `no_std` float helpers: `core` has no trig intrinsics, so a `libm`-free
//! sine (Bhaskara I's approximation, max error ~0.0016) covers the elements that
//! need a little trig (test-signal synthesis, hue rotation) without pulling a
//! math dep into the baseline.

/// sin(2*pi*t) for any real `t` (reduced modulo one turn), via Bhaskara I.
pub(crate) fn sin_turns(t: f32) -> f32 {
    // reduce to [0, 1): manual floor (core has no f32::floor)
    let trunc = t as i64 as f32;
    let floor = if t < 0.0 && trunc != t { trunc - 1.0 } else { trunc };
    let t = t - floor;

    // map to a half-turn x in [0, 1) with the sign of the half
    let (x, sign) = if t < 0.5 { (t * 2.0, 1.0f32) } else { ((t - 0.5) * 2.0, -1.0f32) };
    const PI: f32 = core::f32::consts::PI;
    let xr = x * PI;
    sign * (16.0 * xr * (PI - xr)) / (5.0 * PI * PI - 4.0 * xr * (PI - xr))
}

/// cos(2*pi*t): the sine a quarter turn ahead.
pub(crate) fn cos_turns(t: f32) -> f32 {
    sin_turns(t + 0.25)
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
}
