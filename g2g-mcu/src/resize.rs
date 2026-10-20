use g2g_core::error::G2gError;
use g2g_core::frame::Frame;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::{StaticTransform, TensorDType, TensorLayout, TensorShape};

use crate::inference::TensorFormat;
use crate::lend::lend_converted;

const FRACTION_BITS: u32 = 11;
const FRACTION_ONE: u32 = 1 << FRACTION_BITS;
const COORDINATE_BITS: u32 = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageSize {
    pub width: u16,
    pub height: u16,
}

impl ImageSize {
    fn pixels(self) -> usize {
        usize::from(self.width) * usize::from(self.height)
    }
}

// The two source samples an output coordinate interpolates between, and the
// weight of the second in 1/2048 steps.
#[derive(Clone, Copy, Debug)]
struct Taps {
    first: usize,
    second: usize,
    weight: u32,
}

// Half-pixel centres, as TFLite's and OpenCV's bilinear resize place them:
// output index i samples source position (i + 0.5) * input / output - 0.5.
fn taps(index: u16, input: u16, output: u16) -> Taps {
    let numerator = ((2 * u64::from(index) + 1) * u64::from(input)) << COORDINATE_BITS;
    let position = (numerator / (2 * u64::from(output))).saturating_sub(1 << (COORDINATE_BITS - 1));
    let first = (position >> COORDINATE_BITS) as usize;
    let last = usize::from(input) - 1;
    let fraction = (position >> (COORDINATE_BITS - FRACTION_BITS)) as u32 & (FRACTION_ONE - 1);
    Taps {
        first: first.min(last),
        second: (first + 1).min(last),
        weight: if first >= last { 0 } else { fraction },
    }
}

pub struct ImageResize<'r, const N: usize, const BYTES: usize> {
    channels: u8,
    input: ImageSize,
    output: ImageSize,
    ring: &'r StaticLendRing<N, BYTES>,
}

impl<const N: usize, const BYTES: usize> ImageResize<'static, N, BYTES> {
    pub fn new(
        channels: u8,
        input: ImageSize,
        output: ImageSize,
        ring: &'static StaticLendRing<N, BYTES>,
    ) -> Result<Self, G2gError> {
        // SAFETY: `'static` trivially satisfies with_ring's outlives contract.
        unsafe { Self::with_ring(channels, input, output, ring) }
    }
}

impl<'r, const N: usize, const BYTES: usize> ImageResize<'r, N, BYTES> {
    /// # Safety
    /// The ring must outlive every frame this element publishes.
    pub unsafe fn with_ring(
        channels: u8,
        input: ImageSize,
        output: ImageSize,
        ring: &'r StaticLendRing<N, BYTES>,
    ) -> Result<Self, G2gError> {
        let valid = (1..=4).contains(&channels)
            && [input.width, input.height, output.width, output.height]
                .iter()
                .all(|&side| side > 0);
        if !valid {
            return Err(G2gError::CapsMismatch);
        }
        Ok(Self {
            channels,
            input,
            output,
            ring,
        })
    }

    pub fn output_format(&self) -> TensorFormat {
        TensorFormat {
            dtype: TensorDType::U8,
            shape: TensorShape::new([
                1,
                u32::from(self.output.height),
                u32::from(self.output.width),
                u32::from(self.channels),
            ]),
            layout: TensorLayout::Nhwc,
        }
    }
}

impl<const N: usize, const BYTES: usize> StaticTransform for ImageResize<'_, N, BYTES> {
    async fn process(&mut self, input: Frame) -> Result<Option<Frame>, G2gError> {
        let Some(image) = input.domain.as_system_slice() else {
            return Err(G2gError::UnsupportedDomain);
        };
        let channels = usize::from(self.channels);
        if image.len() != self.input.pixels() * channels {
            return Err(G2gError::CapsMismatch);
        }
        let (source, target) = (self.input, self.output);
        let row_bytes = usize::from(source.width) * channels;
        // SAFETY: the constructor established the ring-outlives-frames
        // contract (`new`: 'static, `with_ring`: the caller's contract).
        unsafe {
            lend_converted(
                self.ring,
                &input,
                target.pixels() * channels,
                |image, out| {
                    let output_row_bytes = usize::from(target.width) * channels;
                    for (y, out_row) in
                        (0..target.height).zip(out.chunks_exact_mut(output_row_bytes))
                    {
                        let rows = taps(y, source.height, target.height);
                        let (Some(top), Some(bottom)) = (
                            image.get(rows.first * row_bytes..(rows.first + 1) * row_bytes),
                            image.get(rows.second * row_bytes..(rows.second + 1) * row_bytes),
                        ) else {
                            continue;
                        };
                        for (x, out_pixel) in
                            (0..target.width).zip(out_row.chunks_exact_mut(channels))
                        {
                            let columns = taps(x, source.width, target.width);
                            let (first, second) =
                                (columns.first * channels, columns.second * channels);
                            let (
                                Some(top_first),
                                Some(top_second),
                                Some(bottom_first),
                                Some(bottom_second),
                            ) = (
                                top.get(first..first + channels),
                                top.get(second..second + channels),
                                bottom.get(first..first + channels),
                                bottom.get(second..second + channels),
                            )
                            else {
                                continue;
                            };
                            let samples = top_first
                                .iter()
                                .zip(top_second)
                                .zip(bottom_first.iter().zip(bottom_second));
                            for (value, ((&a, &b), (&c, &d))) in out_pixel.iter_mut().zip(samples) {
                                let blend = |left: u8, right: u8| {
                                    u32::from(left) * (FRACTION_ONE - columns.weight)
                                        + u32::from(right) * columns.weight
                                };
                                let sum = blend(a, b) * (FRACTION_ONE - rows.weight)
                                    + blend(c, d) * rows.weight;
                                *value = ((sum + (1 << (2 * FRACTION_BITS - 1)))
                                    >> (2 * FRACTION_BITS))
                                    as u8;
                            }
                        }
                    }
                },
            )
        }
        .map(Some)
    }
}

impl<const N: usize, const BYTES: usize> core::fmt::Debug for ImageResize<'_, N, BYTES> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ImageResize")
            .field("channels", &self.channels)
            .field("input", &self.input)
            .field("output", &self.output)
            .finish_non_exhaustive()
    }
}
