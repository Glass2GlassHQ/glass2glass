use g2g_core::{AspectRatio, PixelAspect};
#[cfg(any(feature = "wgpu-sink", feature = "wayland-sink"))]
use g2g_core::{PropKind, PropertySpec};

#[cfg(any(feature = "wgpu-sink", feature = "wayland-sink"))]
pub(crate) const FORCE_ASPECT_RATIO_NAME: &str = "force-aspect-ratio";

#[cfg(any(feature = "wgpu-sink", feature = "wayland-sink"))]
pub(crate) const FORCE_ASPECT_RATIO_PROPERTY: PropertySpec = PropertySpec::new(
    FORCE_ASPECT_RATIO_NAME,
    PropKind::Bool,
    "keep the picture at its display aspect, letterboxed in a window of another shape, instead of stretching it",
)
.with_default("true");

// gst reads an absent pixel-aspect-ratio as square pixels
fn pixel_aspect_fraction(pixel_aspect: PixelAspect) -> (u128, u128) {
    match pixel_aspect {
        PixelAspect::Any => (1, 1),
        PixelAspect::Exact(ratio) => (
            u128::from(ratio.numerator()),
            u128::from(ratio.denominator()),
        ),
    }
}

fn reduce(numerator: u128, denominator: u128) -> Option<(u128, u128)> {
    if numerator == 0 || denominator == 0 {
        return None;
    }
    let (mut a, mut b) = (numerator, denominator);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    Some((numerator / a, denominator / a))
}

// `None` for a zero width or height
pub(crate) fn display_aspect(
    width: u32,
    height: u32,
    pixel_aspect: PixelAspect,
) -> Option<(u128, u128)> {
    let (pixel_numerator, pixel_denominator) = pixel_aspect_fraction(pixel_aspect);
    reduce(
        u128::from(width) * pixel_numerator,
        u128::from(height) * pixel_denominator,
    )
}

#[cfg(any(feature = "wgpu-sink", feature = "wayland-sink"))]
fn saturate(value: u128) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

// gst's video sinks keep the stored height when the display aspect divides it, else the stored width
#[cfg(any(feature = "wgpu-present", feature = "wayland-sink"))]
pub(crate) fn display_size(width: u32, height: u32, pixel_aspect: PixelAspect) -> (u32, u32) {
    let Some((numerator, denominator)) = display_aspect(width, height, pixel_aspect) else {
        return (width, height);
    };
    let (wide, tall) = (u128::from(width), u128::from(height));
    if tall % denominator == 0 || wide % numerator != 0 {
        (saturate(tall * numerator / denominator), height)
    } else {
        (width, saturate(wide * denominator / numerator))
    }
}

// the pixel aspect that shows a resized picture at the display aspect of the original
pub(crate) fn scaled_pixel_aspect(
    width: u32,
    height: u32,
    pixel_aspect: PixelAspect,
    scaled_width: u32,
    scaled_height: u32,
) -> PixelAspect {
    let Some((display_numerator, display_denominator)) =
        display_aspect(width, height, pixel_aspect)
    else {
        return PixelAspect::Any;
    };
    let numerator = display_numerator * u128::from(scaled_height);
    let denominator = display_denominator * u128::from(scaled_width);
    let ratio = reduce(numerator, denominator).and_then(|(numerator, denominator)| {
        AspectRatio::new(
            u32::try_from(numerator).ok()?,
            u32::try_from(denominator).ok()?,
        )
    });
    match ratio.map(PixelAspect::Exact) {
        Some(PixelAspect::SQUARE) if pixel_aspect == PixelAspect::Any => PixelAspect::Any,
        Some(scaled) => scaled,
        None => PixelAspect::Any,
    }
}

#[cfg(feature = "wgpu-sink")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ViewportRectangle {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[cfg(feature = "wgpu-sink")]
impl ViewportRectangle {
    pub(crate) fn whole(width: u32, height: u32) -> Self {
        Self {
            x: 0,
            y: 0,
            width,
            height,
        }
    }
}

// the largest rectangle with the picture's display aspect, centred in the target
#[cfg(feature = "wgpu-sink")]
pub(crate) fn letterbox(
    width: u32,
    height: u32,
    pixel_aspect: PixelAspect,
    target_width: u32,
    target_height: u32,
) -> ViewportRectangle {
    let Some((numerator, denominator)) = display_aspect(width, height, pixel_aspect) else {
        return ViewportRectangle::whole(target_width, target_height);
    };
    let (target_wide, target_tall) = (u128::from(target_width), u128::from(target_height));
    let (fitted_width, fitted_height) = if target_wide * denominator <= target_tall * numerator {
        (
            target_width,
            saturate(target_wide * denominator / numerator),
        )
    } else {
        (
            saturate(target_tall * numerator / denominator),
            target_height,
        )
    };
    ViewportRectangle {
        x: (target_width - fitted_width) / 2,
        y: (target_height - fitted_height) / 2,
        width: fitted_width,
        height: fitted_height,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exact(numerator: u32, denominator: u32) -> PixelAspect {
        PixelAspect::Exact(AspectRatio::new(numerator, denominator).expect("nonzero"))
    }

    #[cfg(any(feature = "wgpu-present", feature = "wayland-sink"))]
    #[test]
    fn display_size_widens_or_heightens_like_gst() {
        // PAL 4:3 has 16:15 pixels, so 720x576 shows as 768x576
        assert_eq!(display_size(720, 576, exact(16, 15)), (768, 576));
        // NTSC 10:11 is 15:11 overall, which divides the width but not 480 rows
        assert_eq!(display_size(720, 480, exact(10, 11)), (720, 528));
        assert_eq!(display_size(1920, 1080, PixelAspect::Any), (1920, 1080));
        assert_eq!(display_size(1920, 1080, PixelAspect::SQUARE), (1920, 1080));
    }

    #[test]
    fn scaling_keeps_the_display_aspect() {
        // 16:9 squeezed to a square needs pixels 16:9 wide
        assert_eq!(
            scaled_pixel_aspect(1920, 1080, PixelAspect::Any, 640, 640),
            exact(16, 9)
        );
        // PAL 4:3 resized to 768x576 has square pixels again
        assert_eq!(
            scaled_pixel_aspect(720, 576, exact(16, 15), 768, 576),
            PixelAspect::SQUARE
        );
        assert_eq!(
            scaled_pixel_aspect(1920, 1080, PixelAspect::Any, 1280, 720),
            PixelAspect::Any,
            "a uniform scale of unknown pixels stays unknown"
        );
    }

    #[cfg(feature = "wgpu-sink")]
    #[test]
    fn letterbox_centres_the_display_aspect() {
        let pal_4_3 = letterbox(720, 576, exact(16, 15), 1920, 1080);
        assert_eq!(
            pal_4_3,
            ViewportRectangle {
                x: 240,
                y: 0,
                width: 1440,
                height: 1080
            },
            "pillarboxed in a 16:9 target"
        );
        let wide = letterbox(1920, 1080, PixelAspect::Any, 1000, 1000);
        assert_eq!(
            wide,
            ViewportRectangle {
                x: 0,
                y: 219,
                width: 1000,
                height: 562
            },
            "letterboxed in a square target"
        );
        assert_eq!(
            letterbox(1280, 720, PixelAspect::Any, 1280, 720),
            ViewportRectangle::whole(1280, 720)
        );
    }
}
