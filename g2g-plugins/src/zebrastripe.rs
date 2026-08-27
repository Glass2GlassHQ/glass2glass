//! Zebra stripe overlay (`zebrastripe`). Marks the overexposed parts of a frame
//! with diagonal black stripes, the exposure aid a camera viewfinder draws.
//! Format and geometry are preserved. CPU-only `no_std` baseline.
//!
//! A pixel is overexposed when its luma reaches the level `threshold` percent
//! names on the video-range scale, 0 % = 16 and 100 % = 235. Every fourth
//! diagonal of those pixels is driven to black, so the marked area shows as a
//! moving barber pole: the diagonals shift by one pixel per frame.
//!
//! On I420 the test and the stripe are on the luma plane and the chroma planes
//! pass through, so a striped area keeps its colour. On packed RGBA / BGRA the
//! pixel's BT.709 luma is tested and a striped pixel goes to neutral black,
//! alpha untouched.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
    RawVideoFormat,
};

use crate::pixel::{bt709_luma, frame_byte_size, rgba_rb_offsets};
use crate::videofx::{self, FilterState, PixelFilter};

const FORMATS: [RawVideoFormat; 3] = [
    RawVideoFormat::I420,
    RawVideoFormat::Rgba8,
    RawVideoFormat::Bgra8,
];

const DEFAULT_THRESHOLD: u32 = 90;
const MAX_THRESHOLD: u32 = 100;

/// Luma of video-range black, the level `threshold=0` names and the level a
/// striped pixel is driven to.
pub const STRIPE_LUMA: u8 = 16;
/// Luma span from video-range black to white (235 - 16), what the threshold
/// percentage is a fraction of.
const VIDEO_RANGE_SPAN: u32 = 219;

/// Width of one stripe, in pixels along the diagonal. The pattern repeats twice
/// as often as it is wide: a bit test picks alternating runs of four.
const STRIPE_WIDTH: u32 = 4;

/// # Example
///
/// ```no_run
/// use g2g_plugins::zebrastripe::ZebraStripe;
///
/// // zebrastripe threshold=95
/// let zebra = ZebraStripe::new().with_threshold(95);
/// ```
#[derive(Debug)]
pub struct ZebraStripe {
    threshold: u32,
    /// Frame counter: it shifts the stripe pattern by a pixel per frame.
    frame: u32,
    state: FilterState,
}

impl Default for ZebraStripe {
    fn default() -> Self {
        Self::new()
    }
}

impl ZebraStripe {
    /// Stripes above 90 % exposure, GStreamer's default.
    pub fn new() -> Self {
        Self {
            threshold: DEFAULT_THRESHOLD,
            frame: 0,
            state: FilterState::new(),
        }
    }

    /// Exposure percentage above which pixels are striped; clamped to 100.
    pub fn with_threshold(mut self, threshold: u32) -> Self {
        self.threshold = threshold.min(MAX_THRESHOLD);
        self
    }

    /// The luma level the percentage names: `threshold` percent of the way from
    /// video-range black to video-range white, rounded to nearest.
    pub fn luma_threshold(&self) -> u8 {
        const HALF: u32 = MAX_THRESHOLD / 2;
        let above_black = (VIDEO_RANGE_SPAN * self.threshold + HALF) / MAX_THRESHOLD;
        (STRIPE_LUMA as u32 + above_black).min(u8::MAX as u32) as u8
    }
}

/// Whether the pixel at `(x, y)` of frame `frame` falls on a stripe.
fn on_stripe(x: u32, y: u32, frame: u32) -> bool {
    (x.wrapping_add(y).wrapping_add(frame) & STRIPE_WIDTH) != 0
}

impl PixelFilter for ZebraStripe {
    const FORMATS: &'static [RawVideoFormat] = &FORMATS;

    fn state(&self) -> &FilterState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut FilterState {
        &mut self.state
    }

    fn apply(&mut self, format: RawVideoFormat, w: u32, h: u32, src: &[u8]) -> Box<[u8]> {
        let bytes = frame_byte_size(format, w, h);
        let mut dst = vec![0u8; bytes].into_boxed_slice();
        dst.copy_from_slice(&src[..bytes]);

        let threshold = self.luma_threshold();
        let frame = self.frame;
        self.frame = self.frame.wrapping_add(1);

        match format {
            RawVideoFormat::I420 => {
                let luma_plane = &mut dst[..(w as usize) * (h as usize)];
                for (i, sample) in luma_plane.iter_mut().enumerate() {
                    let (x, y) = ((i as u32) % w, (i as u32) / w);
                    if *sample >= threshold && on_stripe(x, y, frame) {
                        *sample = STRIPE_LUMA;
                    }
                }
            }
            packed => {
                let (r_idx, b_idx) = rgba_rb_offsets(packed);
                for (i, px) in dst.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    let (x, y) = ((i as u32) % w, (i as u32) / w);
                    if bt709_luma(px[r_idx], px[1], px[b_idx]) >= threshold
                        && on_stripe(x, y, frame)
                    {
                        px[r_idx] = STRIPE_LUMA;
                        px[1] = STRIPE_LUMA;
                        px[b_idx] = STRIPE_LUMA;
                    }
                }
            }
        }
        dst
    }

    fn reset(&mut self) {
        self.frame = 0;
    }
}

impl AsyncElement for ZebraStripe {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Zebra stripe overlay",
            "Filter/Analyzer/Video",
            "Overlays zebra striping on overexposed areas of video",
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
        ZEBRASTRIPE_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "threshold" => {
                let v = value.as_int().ok_or(PropError::Type)?;
                if !(0..=MAX_THRESHOLD as i64).contains(&v) {
                    return Err(PropError::Value);
                }
                self.threshold = v as u32;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "threshold" => Some(PropValue::Int(self.threshold as i64)),
            _ => None,
        }
    }
}

/// `ZebraStripe`'s settable properties, named as GStreamer's (M1084).
static ZEBRASTRIPE_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "threshold",
    PropKind::Int,
    "exposure percentage above which the video is striped",
)
.with_range("0", "100")
.with_default("90")];

impl PadTemplates for ZebraStripe {
    fn pad_templates() -> alloc::vec::Vec<PadTemplate> {
        videofx::pad_templates::<Self>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_percentage_maps_onto_the_video_range() {
        assert_eq!(ZebraStripe::new().with_threshold(0).luma_threshold(), 16);
        assert_eq!(ZebraStripe::new().with_threshold(100).luma_threshold(), 235);
        // the default 90 % sits 219 * 0.9 = 197.1 above black.
        assert_eq!(ZebraStripe::new().luma_threshold(), 16 + 197);
    }

    #[test]
    fn stripes_alternate_in_runs_of_four() {
        // along a row, four striped pixels then four clear, repeating.
        let marked: alloc::vec::Vec<bool> = (0..16).map(|x| on_stripe(x, 0, 0)).collect();
        assert_eq!(
            marked,
            [
                false, false, false, false, true, true, true, true, false, false, false, false,
                true, true, true, true
            ]
        );
        // the pattern shifts by one pixel per frame.
        assert!(on_stripe(3, 0, 1));
        assert!(!on_stripe(3, 0, 0));
    }
}
