//! Frame difference marker (`videodiff`). Passes the picture through but marks
//! every pixel whose brightness moved by more than `threshold` since the
//! previous frame with a diagonal black-and-white pattern, so motion shows up as
//! hatching over the still picture. Format and geometry are preserved. CPU-only
//! `no_std` baseline.
//!
//! The first frame after a start or a flush has nothing to compare against and
//! passes through untouched.
//!
//! On I420 the comparison and the marks are on the luma plane and the chroma
//! planes pass through, so a marked area keeps its colour. On packed RGBA /
//! BGRA the pixel's BT.709 luma is compared and a marked pixel goes to neutral
//! black or white, alpha untouched.

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

/// GStreamer's element fixes the difference that counts as motion at this
/// level and offers no way to change it; here it is the `threshold` property.
const DEFAULT_THRESHOLD: i32 = 10;

/// The two luma levels the mark alternates between, video-range black and
/// white.
pub const MARK_DARK_LUMA: u8 = 16;
pub const MARK_LIGHT_LUMA: u8 = 240;

/// Width of one run of the mark, in pixels along the diagonal.
const MARK_WIDTH: u32 = 4;

/// # Example
///
/// ```no_run
/// use g2g_plugins::videodiff::VideoDiff;
///
/// // videodiff threshold=20
/// let diff = VideoDiff::new().with_threshold(20);
/// ```
#[derive(Debug)]
pub struct VideoDiff {
    threshold: i32,
    /// The previous frame, kept whole so the next one can be compared to it.
    previous: Option<Box<[u8]>>,
    state: FilterState,
}

impl Default for VideoDiff {
    fn default() -> Self {
        Self::new()
    }
}

impl VideoDiff {
    /// Marks pixels that moved by more than 10 levels, GStreamer's fixed value.
    pub fn new() -> Self {
        Self {
            threshold: DEFAULT_THRESHOLD,
            previous: None,
            state: FilterState::new(),
        }
    }

    /// Brightness change, in 8-bit levels, above which a pixel is marked.
    pub fn with_threshold(mut self, threshold: i32) -> Self {
        self.threshold = threshold.clamp(0, u8::MAX as i32);
        self
    }
}

/// The mark level at `(x, y)`: the two levels alternate in runs of
/// [`MARK_WIDTH`] along the diagonal.
fn mark_luma(x: u32, y: u32) -> u8 {
    if (x.wrapping_add(y) & MARK_WIDTH) != 0 {
        MARK_DARK_LUMA
    } else {
        MARK_LIGHT_LUMA
    }
}

fn moved(previous: u8, current: u8, threshold: i32) -> bool {
    (current as i32 - previous as i32).abs() > threshold
}

impl PixelFilter for VideoDiff {
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

        // A previous frame of a different size is one from before a caps change
        // and has nothing to say about this one.
        let previous = self.previous.take().filter(|p| p.len() == bytes);
        if let Some(previous) = &previous {
            match format {
                RawVideoFormat::I420 => {
                    let luma_samples = (w as usize) * (h as usize);
                    for (i, sample) in dst[..luma_samples].iter_mut().enumerate() {
                        if moved(previous[i], *sample, self.threshold) {
                            *sample = mark_luma((i as u32) % w, (i as u32) / w);
                        }
                    }
                }
                packed => {
                    let (r_idx, b_idx) = rgba_rb_offsets(packed);
                    let was = previous.as_chunks::<4>().0;
                    for (i, px) in dst.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                        let before = bt709_luma(was[i][r_idx], was[i][1], was[i][b_idx]);
                        let now = bt709_luma(px[r_idx], px[1], px[b_idx]);
                        if moved(before, now, self.threshold) {
                            let mark = mark_luma((i as u32) % w, (i as u32) / w);
                            px[r_idx] = mark;
                            px[1] = mark;
                            px[b_idx] = mark;
                        }
                    }
                }
            }
        }
        self.previous = Some(Box::from(&src[..bytes]));
        dst
    }

    fn reset(&mut self) {
        self.previous = None;
    }
}

impl AsyncElement for VideoDiff {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Video difference",
            "Filter/Analyzer/Video",
            "Marks the pixels that changed between adjacent video frames",
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
        VIDEODIFF_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "threshold" => {
                let v = value.as_int().ok_or(PropError::Type)?;
                if !(0..=u8::MAX as i64).contains(&v) {
                    return Err(PropError::Value);
                }
                self.threshold = v as i32;
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

/// `VideoDiff`'s settable properties (M1084). GStreamer's element has none:
/// `threshold` is its internal constant made settable.
static VIDEODIFF_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "threshold",
    PropKind::Int,
    "brightness change, in levels, above which a pixel is marked",
)
.with_range("0", "255")
.with_default("10")];

impl PadTemplates for VideoDiff {
    fn pad_templates() -> alloc::vec::Vec<PadTemplate> {
        videofx::pad_templates::<Self>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_change_beyond_the_threshold_counts() {
        let threshold = DEFAULT_THRESHOLD;
        assert!(!moved(100, 100 + threshold as u8, threshold));
        assert!(moved(100, 100 + threshold as u8 + 1, threshold));
        assert!(moved(100, 100 - threshold as u8 - 1, threshold));
    }

    #[test]
    fn the_mark_alternates_in_runs_of_four() {
        let levels: alloc::vec::Vec<u8> = (0..8).map(|x| mark_luma(x, 0)).collect();
        assert_eq!(
            levels,
            [
                MARK_LIGHT_LUMA,
                MARK_LIGHT_LUMA,
                MARK_LIGHT_LUMA,
                MARK_LIGHT_LUMA,
                MARK_DARK_LUMA,
                MARK_DARK_LUMA,
                MARK_DARK_LUMA,
                MARK_DARK_LUMA
            ]
        );
    }
}
