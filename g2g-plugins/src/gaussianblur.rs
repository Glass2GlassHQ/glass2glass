//! Gaussian blur / sharpen (`gaussianblur`). Convolves a packed RGBA / BGRA
//! frame with a separable gaussian of standard deviation `sigma`, preserving
//! format and geometry. CPU-only `no_std` baseline.
//!
//! The kernel runs out to 2.5 sigma either side of the centre tap and is
//! normalised to unit gain. At the picture edges only the taps that land inside
//! the frame contribute and the gain is renormalised over them, so a flat frame
//! stays flat right up to the border instead of darkening.
//!
//! A negative `sigma` sharpens: the same gaussian is subtracted from an impulse
//! of twice its weight, which leaves a centre tap above one and negative
//! surround. All four components are filtered, alpha included, as GStreamer's
//! element does.
//!
//! `exp` and `sqrt` come from the crate's `mathf`, so the
//! baseline needs no libm.

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

use crate::mathf::{exp, sqrt};
use crate::videofx::{self, FilterState, PixelFilter};

const FORMATS: [RawVideoFormat; 2] = [RawVideoFormat::Rgba8, RawVideoFormat::Bgra8];

const DEFAULT_SIGMA: f64 = 1.2;
const SIGMA_LIMIT: f64 = 20.0;

/// How many standard deviations the kernel reaches either side of its centre.
const KERNEL_RADIUS_SIGMAS: f64 = 2.5;

/// Components per pixel of the packed formats. All four are filtered.
const CHANNELS: usize = 4;

/// # Example
///
/// ```no_run
/// use g2g_plugins::gaussianblur::GaussianBlur;
///
/// // gaussianblur sigma=3.0
/// let blur = GaussianBlur::new().with_sigma(3.0);
/// ```
#[derive(Debug)]
pub struct GaussianBlur {
    sigma: f64,
    /// The normalised taps and their running sums, rebuilt when `sigma` moves.
    kernel: Option<Kernel>,
    /// The horizontally filtered frame and one output row's sums, reused
    /// across frames.
    scratch: BlurScratch,
    state: FilterState,
}

impl Default for GaussianBlur {
    fn default() -> Self {
        Self::new()
    }
}

impl GaussianBlur {
    /// A mild blur (sigma 1.2), GStreamer's default.
    pub fn new() -> Self {
        Self {
            sigma: DEFAULT_SIGMA,
            kernel: None,
            scratch: BlurScratch::default(),
            state: FilterState::new(),
        }
    }

    /// Standard deviation of the gaussian; negative sharpens. Clamped to
    /// GStreamer's -20..20.
    pub fn with_sigma(mut self, sigma: f64) -> Self {
        self.set_sigma(sigma.clamp(-SIGMA_LIMIT, SIGMA_LIMIT));
        self
    }

    fn set_sigma(&mut self, sigma: f64) {
        if sigma != self.sigma {
            self.kernel = None;
        }
        self.sigma = sigma;
    }
}

/// Fractional bits of a fixed-point tap.
const TAP_BITS: u32 = 16;
/// The gain of a window that uses every tap.
const FULL_GAIN: i64 = 1 << TAP_BITS;
/// Fractional bits the horizontal pass keeps for the vertical one.
const INTERMEDIATE_BITS: u32 = 8;

/// One separable filter: `taps` are normalised to sum to one, `fixed_taps`
/// are the same taps in `TAP_BITS` fixed point summing to exactly
/// [`FULL_GAIN`], and `running` holds their prefix sums so an edge window's
/// gain is one subtraction away.
#[derive(Debug)]
struct Kernel {
    taps: Vec<f64>,
    fixed_taps: Vec<i64>,
    running: Vec<i64>,
}

/// Working buffers of [`blur`].
#[derive(Debug, Default)]
struct BlurScratch {
    /// The horizontally filtered frame, `INTERMEDIATE_BITS` fixed point.
    horizontal: Vec<i64>,
    /// One output row's vertical sums.
    row: Vec<i64>,
}

impl Kernel {
    /// The gaussian of `sigma`, sampled at whole-pixel offsets out to
    /// [`KERNEL_RADIUS_SIGMAS`] and normalised. A negative `sigma` yields the
    /// sharpening kernel described in the module docs.
    fn new(sigma: f64) -> Self {
        let radius = ceil_to_usize(KERNEL_RADIUS_SIGMAS * sigma.abs());
        let width = 1 + 2 * radius;
        let mut taps = vec![0.0f64; width];
        if radius == 0 {
            taps[0] = 1.0;
            return Self::from_taps(taps);
        }

        // The gaussian's own scale factor; it cancels in the normalisation, but
        // its sign is what turns a negative sigma into a sharpen.
        let peak = 1.0 / (sigma * sqrt(2.0 * core::f64::consts::PI));
        let falloff = -0.5 / (sigma * sigma);
        taps[radius] = peak;
        let mut sum = peak;
        for offset in 1..=radius {
            let tap = peak * exp(falloff * (offset * offset) as f64);
            taps[radius - offset] = tap;
            taps[radius + offset] = tap;
            sum += 2.0 * tap;
        }
        if sigma < 0.0 {
            sum = -sum;
            taps[radius] += 2.0 * sum;
        }
        for tap in taps.iter_mut() {
            *tap /= sum;
        }
        Self::from_taps(taps)
    }

    /// The fixed-point copies of normalised `taps` and their prefix sums.
    fn from_taps(taps: Vec<f64>) -> Self {
        let mut fixed_taps: Vec<i64> = taps
            .iter()
            .map(|&tap| crate::mathf::round_to_i64(tap * FULL_GAIN as f64))
            .collect();
        // rounding each tap leaves the sum a few units off one
        let centre = fixed_taps.len() / 2;
        fixed_taps[centre] += FULL_GAIN - fixed_taps.iter().sum::<i64>();
        let running = fixed_taps
            .iter()
            .scan(0, |accumulated, &tap| {
                *accumulated += tap;
                Some(*accumulated)
            })
            .collect();
        Self {
            taps,
            fixed_taps,
            running,
        }
    }

    fn width(&self) -> usize {
        self.taps.len()
    }

    fn radius(&self) -> usize {
        self.taps.len() / 2
    }

    /// Fixed-point gain of taps `first..last`, the normalisation an edge
    /// window needs.
    fn gain(&self, first: usize, last: usize) -> i64 {
        self.running[last - 1]
            - if first > 0 {
                self.running[first - 1]
            } else {
                0
            }
    }

    /// The taps that land inside a line of `extent` samples when centred on
    /// `position`, as `(first tap, one past the last tap, first sample read)`.
    fn window(&self, position: usize, extent: usize) -> (usize, usize, usize) {
        let first_tap = self.radius().saturating_sub(position);
        let first_sample = position + first_tap - self.radius();
        let last_tap = self.width().min(first_tap + extent - first_sample);
        (first_tap, last_tap, first_sample)
    }
}

/// Smallest integer at or above `v` (`v >= 0`); `core` has no `f64::ceil`.
fn ceil_to_usize(v: f64) -> usize {
    let truncated = v as usize;
    if (truncated as f64) < v {
        truncated + 1
    } else {
        truncated
    }
}

/// `sum / gain` rounded, where `sum` carries `sum_bits` more fractional bits
/// than the result. An interior window's gain is exactly [`FULL_GAIN`], so its
/// division is a shift.
fn normalise(sum: i64, gain: i64, sum_bits: u32) -> i64 {
    if gain == FULL_GAIN {
        let shift = TAP_BITS + sum_bits;
        return (sum + (1 << (shift - 1))) >> shift;
    }
    let divisor = gain << sum_bits;
    (sum + divisor / 2).div_euclid(divisor)
}

/// Convolve `pixels` with `kernel` along both axes in place, in fixed point.
/// Runs horizontally into a scratch frame, then vertically back to bytes one
/// output row at a time, each source row added in whole.
fn blur(kernel: &Kernel, pixels: &mut [u8], w: usize, h: usize, scratch: &mut BlurScratch) {
    let row_len = w * CHANNELS;
    let horizontal = &mut scratch.horizontal;
    horizontal.clear();
    horizontal.resize(row_len * h, 0);
    for (source_row, filtered_row) in pixels
        .chunks_exact(row_len)
        .zip(horizontal.chunks_exact_mut(row_len))
    {
        let source_pixels = source_row.as_chunks::<CHANNELS>().0;
        let filtered_pixels = filtered_row.as_chunks_mut::<CHANNELS>().0;
        for (column, filtered) in filtered_pixels.iter_mut().enumerate() {
            let (first_tap, last_tap, first_sample) = kernel.window(column, w);
            let gain = kernel.gain(first_tap, last_tap);
            let mut dot = [0i64; CHANNELS];
            let window = &source_pixels[first_sample..first_sample + last_tap - first_tap];
            for (&tap, sample) in kernel.fixed_taps[first_tap..last_tap].iter().zip(window) {
                for (sum, &value) in dot.iter_mut().zip(sample) {
                    *sum += value as i64 * tap;
                }
            }
            for (out, sum) in filtered.iter_mut().zip(dot) {
                // keep `INTERMEDIATE_BITS` of fraction for the vertical pass
                *out = normalise(sum << INTERMEDIATE_BITS, gain, 0);
            }
        }
    }

    let sums = &mut scratch.row;
    for (row, out_row) in pixels.chunks_exact_mut(row_len).enumerate() {
        let (first_tap, last_tap, first_sample) = kernel.window(row, h);
        let gain = kernel.gain(first_tap, last_tap);
        sums.clear();
        sums.resize(row_len, 0);
        for (offset, &tap) in kernel.fixed_taps[first_tap..last_tap].iter().enumerate() {
            let source_row = first_sample + offset;
            let source = &horizontal[source_row * row_len..(source_row + 1) * row_len];
            for (sum, &value) in sums.iter_mut().zip(source) {
                *sum += value * tap;
            }
        }
        for (out, &sum) in out_row.iter_mut().zip(sums.iter()) {
            *out = normalise(sum, gain, INTERMEDIATE_BITS).clamp(0, u8::MAX as i64) as u8;
        }
    }
}

impl PixelFilter for GaussianBlur {
    const FORMATS: &'static [RawVideoFormat] = &FORMATS;

    fn state(&self) -> &FilterState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut FilterState {
        &mut self.state
    }

    fn apply(&mut self, _format: RawVideoFormat, w: u32, h: u32, pixels: &mut [u8]) {
        if self.sigma == 0.0 {
            return;
        }
        let sigma = self.sigma;
        let kernel = self.kernel.get_or_insert_with(|| Kernel::new(sigma));
        blur(kernel, pixels, w as usize, h as usize, &mut self.scratch);
    }
}

impl AsyncElement for GaussianBlur {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Gaussian blur",
            "Filter/Effect/Video",
            "Performs a gaussian blur or sharpen on video",
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
        GAUSSIANBLUR_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "sigma" => {
                let v = value.as_double().ok_or(PropError::Type)?;
                if !(-SIGMA_LIMIT..=SIGMA_LIMIT).contains(&v) {
                    return Err(PropError::Value);
                }
                self.set_sigma(v);
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "sigma" => Some(PropValue::Double(self.sigma)),
            _ => None,
        }
    }
}

/// `GaussianBlur`'s settable properties, named as GStreamer's (M1084).
static GAUSSIANBLUR_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "sigma",
    PropKind::Double,
    "standard deviation of the gaussian, negative to sharpen",
)
.with_range("-20", "20")
.with_default("1.2")];

impl PadTemplates for GaussianBlur {
    fn pad_templates() -> Vec<PadTemplate> {
        videofx::pad_templates::<Self>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_is_normalised_and_symmetric() {
        let kernel = Kernel::new(DEFAULT_SIGMA);
        // 2.5 sigma either side of the centre tap.
        assert_eq!(
            kernel.width(),
            1 + 2 * ceil_to_usize(KERNEL_RADIUS_SIGMAS * DEFAULT_SIGMA)
        );
        let sum: f64 = kernel.taps.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "taps sum to {sum}");
        for offset in 1..=kernel.radius() {
            let left = kernel.taps[kernel.radius() - offset];
            let right = kernel.taps[kernel.radius() + offset];
            assert!((left - right).abs() < 1e-15);
            assert!(left < kernel.taps[kernel.radius()]);
        }
    }

    #[test]
    fn negative_sigma_sharpens() {
        let kernel = Kernel::new(-DEFAULT_SIGMA);
        let sum: f64 = kernel.taps.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "taps sum to {sum}");
        assert!(kernel.taps[kernel.radius()] > 1.0, "centre tap is boosted");
        assert!(kernel.taps[0] < 0.0, "surround is negative");
    }

    #[test]
    fn a_flat_frame_stays_flat_to_the_border() {
        const LEVEL: u8 = 173;
        const SIDE: usize = 9;
        for sigma in [DEFAULT_SIGMA, SIGMA_LIMIT, -DEFAULT_SIGMA] {
            let mut pixels = vec![LEVEL; SIDE * SIDE * CHANNELS];
            blur(
                &Kernel::new(sigma),
                &mut pixels,
                SIDE,
                SIDE,
                &mut BlurScratch::default(),
            );
            assert!(pixels.iter().all(|&v| v == LEVEL), "sigma {sigma}");
        }
    }

    #[test]
    fn edge_window_clips_and_renormalises() {
        let kernel = Kernel::new(DEFAULT_SIGMA);
        // Centred well inside a wide line, every tap is used and the gain is one.
        let (first, last, start) = kernel.window(kernel.radius(), 100);
        assert_eq!((first, last, start), (0, kernel.width(), 0));
        assert_eq!(kernel.gain(first, last), FULL_GAIN);
        // At the very first sample, the taps left of centre are dropped.
        let (first, last, start) = kernel.window(0, 100);
        assert_eq!((first, last, start), (kernel.radius(), kernel.width(), 0));
        assert!(kernel.gain(first, last) < FULL_GAIN);
    }
}
