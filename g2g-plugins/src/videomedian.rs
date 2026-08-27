//! Median filter (`videomedian`). Replaces each I420 sample with the median of
//! its neighbourhood, which removes isolated speckles without softening edges
//! the way an average would. Format and geometry are preserved. CPU-only
//! `no_std` baseline.
//!
//! `filtersize=5` takes the median of the sample and its four edge neighbours,
//! `filtersize=9` of the full 3x3 block. The one-pixel border has no complete
//! neighbourhood and is copied through. `lum-only` (the default) leaves the
//! chroma planes alone; with it off they are filtered at their own half
//! resolution.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
    RawVideoFormat,
};

use crate::pixel::{frame_byte_size, planar_planes};
use crate::videofx::{self, FilterState, PixelFilter};

const FORMATS: [RawVideoFormat; 1] = [RawVideoFormat::I420];

const DEFAULT_LUM_ONLY: bool = true;

/// The two neighbourhoods GStreamer offers, named by how many samples they take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MedianSize {
    /// The sample and its four edge neighbours.
    Five,
    /// The full 3x3 block.
    Nine,
}

impl MedianSize {
    const DEFAULT: Self = Self::Five;

    fn from_samples(samples: i64) -> Option<Self> {
        match samples {
            5 => Some(Self::Five),
            9 => Some(Self::Nine),
            _ => None,
        }
    }

    fn samples(self) -> i64 {
        match self {
            Self::Five => 5,
            Self::Nine => 9,
        }
    }
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::videomedian::{MedianSize, VideoMedian};
///
/// // videomedian filtersize=9 lum-only=false
/// let median = VideoMedian::new()
///     .with_size(MedianSize::Nine)
///     .with_luma_only(false);
/// ```
#[derive(Debug)]
pub struct VideoMedian {
    size: MedianSize,
    luma_only: bool,
    state: FilterState,
}

impl Default for VideoMedian {
    fn default() -> Self {
        Self::new()
    }
}

impl VideoMedian {
    /// The five-sample median over luma only, GStreamer's defaults.
    pub fn new() -> Self {
        Self {
            size: MedianSize::DEFAULT,
            luma_only: DEFAULT_LUM_ONLY,
            state: FilterState::new(),
        }
    }

    pub fn with_size(mut self, size: MedianSize) -> Self {
        self.size = size;
        self
    }

    pub fn with_luma_only(mut self, luma_only: bool) -> Self {
        self.luma_only = luma_only;
        self
    }
}

/// Median-filter one plane of `w x h` samples. The one-sample border is copied
/// through, since its neighbourhood runs off the plane.
fn median_plane(dst: &mut [u8], src: &[u8], w: usize, h: usize, size: MedianSize) {
    dst.copy_from_slice(&src[..w * h]);
    if w < 3 || h < 3 {
        return;
    }
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let at = |dx: isize, dy: isize| {
                src[(y as isize + dy) as usize * w + (x as isize + dx) as usize]
            };
            dst[y * w + x] = match size {
                MedianSize::Five => {
                    let mut window = [at(0, -1), at(-1, 0), at(0, 0), at(1, 0), at(0, 1)];
                    window.sort_unstable();
                    window[window.len() / 2]
                }
                MedianSize::Nine => {
                    let mut window = [
                        at(-1, -1),
                        at(0, -1),
                        at(1, -1),
                        at(-1, 0),
                        at(0, 0),
                        at(1, 0),
                        at(-1, 1),
                        at(0, 1),
                        at(1, 1),
                    ];
                    window.sort_unstable();
                    window[window.len() / 2]
                }
            };
        }
    }
}

impl PixelFilter for VideoMedian {
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

        let planes = planar_planes(format, w as usize, h as usize);
        let filtered = if self.luma_only { 1 } else { planes.len() };
        for &(offset, plane_w, plane_h) in &planes[..filtered] {
            let plane = offset..offset + plane_w * plane_h;
            median_plane(
                &mut dst[plane.clone()],
                &src[plane],
                plane_w,
                plane_h,
                self.size,
            );
        }
        dst
    }
}

impl AsyncElement for VideoMedian {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Median effect",
            "Filter/Effect/Video",
            "Applies a median filter to raw video",
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
        VIDEOMEDIAN_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "filtersize" => {
                let v = value.as_int().ok_or(PropError::Type)?;
                self.size = MedianSize::from_samples(v).ok_or(PropError::Value)?;
            }
            "lum-only" => self.luma_only = value.as_bool().ok_or(PropError::Type)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "filtersize" => Some(PropValue::Int(self.size.samples())),
            "lum-only" => Some(PropValue::Bool(self.luma_only)),
            _ => None,
        }
    }
}

/// `VideoMedian`'s settable properties, named as GStreamer's (M1084).
/// `filtersize` is GStreamer's two-value enum, whose nicks are the sample counts
/// themselves, so a launch line reads the same either way.
static VIDEOMEDIAN_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "filtersize",
        PropKind::Int,
        "samples in the neighbourhood: 5 (edge neighbours) or 9 (3x3)",
    )
    .with_enum_values("5 | 9")
    .with_default("5"),
    PropertySpec::new(
        "lum-only",
        PropKind::Bool,
        "filter the luma plane only, leaving chroma untouched",
    )
    .with_default("true"),
];

impl PadTemplates for VideoMedian {
    fn pad_templates() -> alloc::vec::Vec<PadTemplate> {
        videofx::pad_templates::<Self>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 5x5 plane of `background` with one `spike` sample in the middle.
    fn speckled(w: usize, h: usize, background: u8, spike: u8) -> alloc::vec::Vec<u8> {
        let mut plane = vec![background; w * h];
        plane[(h / 2) * w + w / 2] = spike;
        plane
    }

    #[test]
    fn a_lone_speckle_is_removed() {
        const BACKGROUND: u8 = 40;
        const SPIKE: u8 = 200;
        let (w, h) = (5, 5);
        let src = speckled(w, h, BACKGROUND, SPIKE);
        for size in [MedianSize::Five, MedianSize::Nine] {
            let mut dst = vec![0u8; w * h];
            median_plane(&mut dst, &src, w, h, size);
            assert_eq!(dst[(h / 2) * w + w / 2], BACKGROUND, "{size:?}");
        }
    }

    #[test]
    fn an_edge_survives() {
        // A vertical step: the median of a neighbourhood wholly on one side of
        // it is that side's level, so the step keeps its position and height.
        const DARK: u8 = 30;
        const LIGHT: u8 = 220;
        let (w, h) = (6, 5);
        let src: alloc::vec::Vec<u8> = (0..w * h)
            .map(|i| if i % w < w / 2 { DARK } else { LIGHT })
            .collect();
        let mut dst = vec![0u8; w * h];
        median_plane(&mut dst, &src, w, h, MedianSize::Nine);
        assert_eq!(dst, src);
    }

    #[test]
    fn the_border_is_copied_through() {
        const BACKGROUND: u8 = 40;
        const SPIKE: u8 = 200;
        let (w, h) = (5, 5);
        // a speckle on the top row has no full neighbourhood, so it stays.
        let mut src = vec![BACKGROUND; w * h];
        src[2] = SPIKE;
        let mut dst = vec![0u8; w * h];
        median_plane(&mut dst, &src, w, h, MedianSize::Five);
        assert_eq!(dst[2], SPIKE);
    }
}
