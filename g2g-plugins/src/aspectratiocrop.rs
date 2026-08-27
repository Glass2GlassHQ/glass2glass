//! Aspect-ratio crop (`aspectratiocrop`). Trims equal amounts off two opposite
//! edges so the picture comes out at the `aspect-ratio` the property names,
//! keeping the centre. The pixel format is preserved, the geometry is not.
//! CPU-only `no_std` baseline; the cropping itself is
//! [`videocrop`](crate::videocrop)'s.
//!
//! Whichever axis is too long is the one that loses pixels: a 4:3 picture asked
//! for 16:9 loses rows, a 16:9 picture asked for 4:3 loses columns. A crop that
//! would take half the axis or more is refused and the frame passes through
//! whole, as is a `0/1` ratio, which is the default and means "leave it alone".
//!
//! On a chroma-subsampled format the inset is rounded down to an even number of
//! pixels, so the crop lands on a chroma sample and the output ratio is off by
//! at most one pixel.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec, RawVideoFormat,
};

use crate::pixel::even_dims_required;
use crate::videocrop::{crop, FORMATS};
use crate::videofx::{self, FilterState, PixelFilter};

/// GStreamer's default `0/1`: no target ratio, so no cropping.
const DEFAULT_ASPECT_NUMERATOR: i32 = 0;
const DEFAULT_ASPECT_DENOMINATOR: i32 = 1;

/// # Example
///
/// ```no_run
/// use g2g_plugins::aspectratiocrop::AspectRatioCrop;
///
/// // aspectratiocrop aspect-ratio=16/9
/// let crop = AspectRatioCrop::new().with_aspect_ratio(16, 9);
/// ```
#[derive(Debug)]
pub struct AspectRatioCrop {
    aspect: (i32, i32),
    state: FilterState,
}

impl Default for AspectRatioCrop {
    fn default() -> Self {
        Self::new()
    }
}

impl AspectRatioCrop {
    /// A pass-through; name a ratio to make it crop.
    pub fn new() -> Self {
        Self {
            aspect: (DEFAULT_ASPECT_NUMERATOR, DEFAULT_ASPECT_DENOMINATOR),
            state: FilterState::new(),
        }
    }

    /// Target ratio as `numerator/denominator`, e.g. `16/9`. A numerator below
    /// one disables cropping.
    pub fn with_aspect_ratio(mut self, numerator: i32, denominator: i32) -> Self {
        self.aspect = (numerator, denominator);
        self
    }

    /// Pixels to trim off each edge as `(left and right, top and bottom)`; at
    /// most one of the two is non-zero.
    fn insets(&self, format: RawVideoFormat, w: u32, h: u32) -> (u32, u32) {
        insets(self.aspect, format, w, h)
    }
}

/// Pixels to trim off each edge to reach `aspect`, as `(left and right, top and
/// bottom)`.
fn insets(aspect: (i32, i32), format: RawVideoFormat, w: u32, h: u32) -> (u32, u32) {
    let (numerator, denominator) = aspect;
    if numerator < 1 || denominator < 1 {
        return (0, 0);
    }
    let (width, height) = (w as f64, h as f64);
    let requested = numerator as f64 / denominator as f64;
    let incoming = width / height;
    let (even_w, even_h) = even_dims_required(format);

    // A wider target than the picture has means the picture is too tall, so
    // rows go; otherwise columns do. The inset is half the excess, truncated
    // toward zero the way GStreamer's element truncates it.
    let (excess, extent, must_be_even) = if requested > incoming {
        (
            height - denominator as f64 / numerator as f64 * width,
            h,
            even_h,
        )
    } else if requested < incoming {
        (
            width - numerator as f64 / denominator as f64 * height,
            w,
            even_w,
        )
    } else {
        return (0, 0);
    };
    let mut inset = (excess / 2.0).max(0.0) as u32;
    if must_be_even {
        inset -= inset % 2;
    }
    // Refuse a crop that would take half the axis or more.
    if inset >= extent / 2 {
        return (0, 0);
    }
    if requested > incoming {
        (0, inset)
    } else {
        (inset, 0)
    }
}

/// Geometry left after the insets for `aspect` come off a `w x h` frame.
fn cropped_dims(aspect: (i32, i32), format: RawVideoFormat, w: u32, h: u32) -> (u32, u32) {
    let (horizontal, vertical) = insets(aspect, format, w, h);
    (w - 2 * horizontal, h - 2 * vertical)
}

impl PixelFilter for AspectRatioCrop {
    const FORMATS: &'static [RawVideoFormat] = &FORMATS;

    fn state(&self) -> &FilterState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut FilterState {
        &mut self.state
    }

    fn output_dims(&self, format: RawVideoFormat, w: u32, h: u32) -> (u32, u32) {
        cropped_dims(self.aspect, format, w, h)
    }

    fn apply(&mut self, format: RawVideoFormat, w: u32, h: u32, src: &[u8]) -> Box<[u8]> {
        let (horizontal, vertical) = self.insets(format, w, h);
        let (out_w, out_h) = self.output_dims(format, w, h);
        crop(
            src,
            format,
            (w as usize, h as usize),
            (
                horizontal as usize,
                vertical as usize,
                out_w as usize,
                out_h as usize,
            ),
        )
    }
}

impl AsyncElement for AspectRatioCrop {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    // M759: a spatial crop; meta propagates per its own Crop policy.
    #[cfg(feature = "metadata")]
    fn meta_transform(&self) -> Option<g2g_core::meta::Transform> {
        Some(g2g_core::meta::Transform::Crop)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Aspect ratio crop",
            "Filter/Effect/Video",
            "Crops video to a specified aspect ratio",
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

    /// Native `DerivedOutput`: the format and framerate survive, the geometry
    /// loses the insets the target ratio calls for. Only a fixed input geometry
    /// determines them, which is what the solver hands over.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let aspect = self.aspect;
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match input {
            Caps::RawVideo {
                format,
                width: Dim::Fixed(w),
                height: Dim::Fixed(h),
                framerate,
                interlace: _,
            } if FORMATS.contains(format) => {
                let (out_w, out_h) = cropped_dims(aspect, *format, *w, *h);
                CapsSet::one(Caps::RawVideo {
                    format: *format,
                    width: Dim::Fixed(out_w),
                    height: Dim::Fixed(out_h),
                    framerate: framerate.clone(),
                    interlace: g2g_core::Interlace::Any,
                })
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
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

    /// The crop rectangle is chosen in the buffer's stored coordinates, so it
    /// means something different once the picture is turned. The sink's
    /// `AbsorbOrientation` stops here rather than reaching a `videoflip`
    /// upstream, which would leave the crop applied to the un-turned picture.
    fn handles_orientation(&self) -> bool {
        true
    }

    fn properties(&self) -> &'static [PropertySpec] {
        ASPECTRATIOCROP_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "aspect-ratio" => {
                let (numerator, denominator) = value.as_fraction().ok_or(PropError::Type)?;
                if numerator < 0 || denominator < 1 {
                    return Err(PropError::Value);
                }
                self.aspect = (numerator, denominator);
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "aspect-ratio" => Some(PropValue::Fraction(self.aspect.0, self.aspect.1)),
            _ => None,
        }
    }
}

/// `AspectRatioCrop`'s settable properties, named as GStreamer's (M1084).
static ASPECTRATIOCROP_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "aspect-ratio",
    PropKind::Fraction,
    "target aspect ratio of the video; 0/1 leaves it alone",
)
.with_default("0/1")];

impl PadTemplates for AspectRatioCrop {
    fn pad_templates() -> Vec<PadTemplate> {
        videofx::pad_templates::<Self>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_taller_picture_loses_rows() {
        // 640x480 (4:3) to 16:9 keeps 640 * 9 / 16 = 360 rows, so 60 go off
        // each of the top and bottom.
        let crop = AspectRatioCrop::new().with_aspect_ratio(16, 9);
        assert_eq!(crop.insets(RawVideoFormat::Rgba8, 640, 480), (0, 60));
        assert_eq!(
            crop.output_dims(RawVideoFormat::Rgba8, 640, 480),
            (640, 360)
        );
    }

    #[test]
    fn a_wider_picture_loses_columns() {
        // 640x360 (16:9) to 4:3 keeps 360 * 4 / 3 = 480 columns, so 80 go off
        // each side.
        let crop = AspectRatioCrop::new().with_aspect_ratio(4, 3);
        assert_eq!(crop.insets(RawVideoFormat::Rgba8, 640, 360), (80, 0));
        assert_eq!(
            crop.output_dims(RawVideoFormat::Rgba8, 640, 360),
            (480, 360)
        );
    }

    #[test]
    fn a_matching_ratio_and_the_default_crop_nothing() {
        let matching = AspectRatioCrop::new().with_aspect_ratio(4, 3);
        assert_eq!(matching.insets(RawVideoFormat::Rgba8, 640, 480), (0, 0));
        let default = AspectRatioCrop::new();
        assert_eq!(default.insets(RawVideoFormat::Rgba8, 640, 480), (0, 0));
    }

    #[test]
    fn a_subsampled_format_keeps_the_inset_even() {
        // 640x482 to 16:9 wants (482 - 360) / 2 = 61 rows off each edge; 4:2:0
        // rounds that down to 60.
        let crop = AspectRatioCrop::new().with_aspect_ratio(16, 9);
        assert_eq!(crop.insets(RawVideoFormat::Rgba8, 640, 482), (0, 61));
        assert_eq!(crop.insets(RawVideoFormat::I420, 640, 482), (0, 60));
    }

    #[test]
    fn an_extreme_ratio_never_empties_the_frame() {
        // The half-the-axis refusal is what keeps a wild ratio from cropping
        // the picture away entirely.
        for (numerator, denominator) in [(1000, 1), (1, 1000), (10000, 3), (3, 10000)] {
            let crop = AspectRatioCrop::new().with_aspect_ratio(numerator, denominator);
            let (w, h) = crop.output_dims(RawVideoFormat::Rgba8, 640, 480);
            assert!(w > 0 && h > 0, "{numerator}/{denominator} left {w}x{h}");
        }
    }
}
