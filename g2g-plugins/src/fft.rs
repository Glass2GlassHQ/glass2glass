use alloc::vec::Vec;
use core::f64::consts::TAU;

use crate::mathf;

// radix-2 complex FFT with its twiddles and bit-reversed order computed once per size
#[derive(Debug, Clone)]
pub(crate) struct FftPlan {
    twiddle_re: Vec<f64>,
    twiddle_im: Vec<f64>,
    bit_reversed: Vec<usize>,
}

impl FftPlan {
    // size must be a power of two
    pub(crate) fn new(size: usize) -> Self {
        assert!(
            size.is_power_of_two(),
            "FFT size {size} is not a power of two"
        );
        let half = size / 2;
        let angle = |k: usize| -TAU * k as f64 / size as f64;
        let index_bits = size.trailing_zeros();
        let bit_reversed = (0..size)
            .map(|i| {
                i.reverse_bits()
                    .checked_shr(usize::BITS - index_bits)
                    .unwrap_or(0)
            })
            .collect();
        Self {
            twiddle_re: (0..half).map(|k| mathf::cos(angle(k))).collect(),
            twiddle_im: (0..half).map(|k| mathf::sin(angle(k))).collect(),
            bit_reversed,
        }
    }

    pub(crate) fn size(&self) -> usize {
        self.bit_reversed.len()
    }

    pub(crate) fn forward(&self, re: &mut [f64], im: &mut [f64]) {
        self.transform(re, im, 1.0);
    }

    // scaled by 1 / size, so forward then inverse is the identity
    pub(crate) fn inverse(&self, re: &mut [f64], im: &mut [f64]) {
        self.transform(re, im, -1.0);
        let scale = 1.0 / self.size() as f64;
        re.iter_mut().for_each(|value| *value *= scale);
        im.iter_mut().for_each(|value| *value *= scale);
    }

    fn transform(&self, re: &mut [f64], im: &mut [f64], direction: f64) {
        let size = self.size();
        for (i, &j) in self.bit_reversed.iter().enumerate() {
            if i < j {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= size {
            let half = len / 2;
            let stride = size / len;
            for start in (0..size).step_by(len) {
                for k in 0..half {
                    let wr = self.twiddle_re[k * stride];
                    let wi = direction * self.twiddle_im[k * stride];
                    let (pr, pi) = (re[start + k], im[start + k]);
                    let (qr, qi) = (re[start + k + half], im[start + k + half]);
                    let vr = qr * wr - qi * wi;
                    let vi = qr * wi + qi * wr;
                    re[start + k] = pr + vr;
                    im[start + k] = pi + vi;
                    re[start + k + half] = pr - vr;
                    im[start + k + half] = pi - vi;
                }
            }
            len <<= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn matches_a_direct_dft_and_round_trips() {
        let size = 16;
        let plan = FftPlan::new(size);
        let input: Vec<f64> = (0..size).map(|i| ((i * 7) % 5) as f64 - 2.0).collect();
        let mut re = input.clone();
        let mut im = vec![0.0; size];
        plan.forward(&mut re, &mut im);
        for k in 0..size {
            let (mut expected_re, mut expected_im) = (0.0, 0.0);
            for (n, x) in input.iter().enumerate() {
                let angle = -TAU * (k * n) as f64 / size as f64;
                expected_re += x * mathf::cos(angle);
                expected_im += x * mathf::sin(angle);
            }
            assert!((re[k] - expected_re).abs() < 1e-9, "bin {k} real");
            assert!((im[k] - expected_im).abs() < 1e-9, "bin {k} imaginary");
        }
        plan.inverse(&mut re, &mut im);
        for (got, want) in re.iter().zip(&input) {
            assert!((got - want).abs() < 1e-9);
        }
    }
}
