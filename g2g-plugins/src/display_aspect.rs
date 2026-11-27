// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use g2g_core::{AspectRatio, Caps, Dim, PixelAspect};
#[cfg(any(
    feature = "wgpu-sink",
    feature = "wayland-sink",
    all(windows, feature = "d3d11-sink"),
    all(target_os = "macos", feature = "metal-sink"),
    all(target_os = "linux", any(feature = "gl-sink", feature = "cuda-gl"))
))]
use g2g_core::{PropKind, PropertySpec};

#[cfg(any(
    feature = "wgpu-sink",
    feature = "wayland-sink",
    all(windows, feature = "d3d11-sink"),
    all(target_os = "macos", feature = "metal-sink"),
    all(target_os = "linux", any(feature = "gl-sink", feature = "cuda-gl"))
))]
pub(crate) const FORCE_ASPECT_RATIO_NAME: &str = "force-aspect-ratio";

#[cfg(any(
    feature = "wgpu-sink",
    feature = "wayland-sink",
    all(windows, feature = "d3d11-sink"),
    all(target_os = "macos", feature = "metal-sink"),
    all(target_os = "linux", any(feature = "gl-sink", feature = "cuda-gl"))
))]
pub(crate) const FORCE_ASPECT_RATIO_PROPERTY: PropertySpec = PropertySpec::new(
    FORCE_ASPECT_RATIO_NAME,
    PropKind::Bool,
    "keep the picture at its display aspect, letterboxed in a window of another shape, instead of stretching it",
)
.with_default("true");

// gst reads an absent pixel-aspect-ratio as square pixels
pub(crate) fn pixel_aspect_fraction(pixel_aspect: PixelAspect) -> (u128, u128) {
    let ratio = pixel_aspect.or_square();
    (
        u128::from(ratio.numerator()),
        u128::from(ratio.denominator()),
    )
}

#[cfg(any(
    all(windows, feature = "d3d11-sink"),
    all(target_os = "linux", feature = "hip")
))]
pub(crate) fn transpose_pixel_aspect(pixel_aspect: PixelAspect) -> PixelAspect {
    match pixel_aspect {
        PixelAspect::Any => PixelAspect::Any,
        PixelAspect::Exact(ratio) => PixelAspect::Exact(
            AspectRatio::new(ratio.denominator(), ratio.numerator())
                .expect("aspect ratio is nonzero"),
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

#[cfg(any(
    feature = "wgpu-sink",
    feature = "wayland-sink",
    feature = "gl-render",
    all(windows, feature = "d3d11-sink"),
    all(target_os = "macos", feature = "metal-sink"),
    feature = "kms-sink"
))]
fn saturate(value: u128) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

// gst's video sinks keep the stored height when the display aspect divides it, else the stored width
#[cfg(any(
    feature = "wgpu-present",
    feature = "wayland-sink",
    feature = "kms-sink",
    all(windows, feature = "d3d11-sink"),
    all(target_os = "macos", feature = "metal-sink"),
    all(target_os = "linux", any(feature = "gl-sink", feature = "cuda-gl"))
))]
pub(crate) fn display_size(width: u32, height: u32, pixel_aspect: PixelAspect) -> (u32, u32) {
    let Some((numerator, denominator)) = display_aspect(width, height, pixel_aspect) else {
        return (width, height);
    };
    let (wide, tall) = (u128::from(width), u128::from(height));
    if tall % denominator == 0 || wide % numerator != 0 {
        (saturate(tall * numerator / denominator).max(1), height)
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

pub(crate) fn pixel_aspect_at(
    input: &Caps,
    width: u32,
    height: u32,
    requested: PixelAspect,
) -> PixelAspect {
    match (requested, input) {
        (
            PixelAspect::Any,
            Caps::RawVideo {
                width: Dim::Fixed(in_w),
                height: Dim::Fixed(in_h),
                pixel_aspect,
                ..
            },
        ) => scaled_pixel_aspect(*in_w, *in_h, *pixel_aspect, width, height),
        _ => requested,
    }
}

#[cfg(all(any(target_os = "linux", windows), feature = "cuda"))]
pub(crate) fn scaled_video_caps(input: &Caps, output: &Caps) -> Caps {
    let mut caps = output.clone();
    if let Caps::RawVideo {
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        pixel_aspect,
        ..
    } = &mut caps
    {
        *pixel_aspect = pixel_aspect_at(input, *width, *height, *pixel_aspect);
    }
    caps
}

#[cfg(any(
    feature = "wgpu-sink",
    feature = "gl-render",
    feature = "wayland-sink",
    all(windows, feature = "d3d11-sink"),
    all(target_os = "macos", feature = "metal-sink"),
    feature = "kms-sink"
))]
#[cfg_attr(
    not(any(
        feature = "wgpu-sink",
        feature = "gl-render",
        feature = "wayland-sink",
        feature = "kms-sink"
    )),
    allow(
        unreachable_pub,
        reason = "the native sinks use the rectangle internally"
    )
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ViewportRectangle {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[cfg(any(
    feature = "wgpu-sink",
    feature = "gl-render",
    feature = "wayland-sink",
    all(windows, feature = "d3d11-sink"),
    all(target_os = "macos", feature = "metal-sink"),
    feature = "kms-sink"
))]
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

// the picture pixel drawn at a point of the target, or none in the bars around the picture
#[cfg(any(
    all(
        target_os = "linux",
        any(feature = "wgpu-present", feature = "cuda-gl", feature = "gl-sink")
    ),
    all(target_os = "windows", feature = "wgpu-present")
))]
pub(crate) fn picture_point_in(
    viewport: ViewportRectangle,
    picture_width: u32,
    picture_height: u32,
    target_x: f64,
    target_y: f64,
) -> Option<(u32, u32)> {
    let column =
        (target_x - f64::from(viewport.x)) * f64::from(picture_width) / f64::from(viewport.width);
    let row =
        (target_y - f64::from(viewport.y)) * f64::from(picture_height) / f64::from(viewport.height);
    let inside = |value: f64, extent: u32| (0.0..f64::from(extent)).contains(&value);
    (inside(column, picture_width) && inside(row, picture_height))
        .then_some((column as u32, row as u32))
}

// the largest rectangle with the picture's display aspect, centred in the target
#[cfg(any(
    feature = "wgpu-sink",
    feature = "gl-render",
    feature = "wayland-sink",
    all(windows, feature = "d3d11-sink"),
    all(target_os = "macos", feature = "metal-sink"),
    feature = "kms-sink"
))]
pub(crate) fn letterbox(
    width: u32,
    height: u32,
    pixel_aspect: PixelAspect,
    target_width: u32,
    target_height: u32,
) -> ViewportRectangle {
    if target_width == 0 || target_height == 0 {
        return ViewportRectangle::whole(target_width, target_height);
    }
    let Some((numerator, denominator)) = display_aspect(width, height, pixel_aspect) else {
        return ViewportRectangle::whole(target_width, target_height);
    };
    let (target_wide, target_tall) = (u128::from(target_width), u128::from(target_height));
    let (fitted_width, fitted_height) = if target_wide * denominator <= target_tall * numerator {
        (
            target_width,
            saturate(target_wide * denominator / numerator).max(1),
        )
    } else {
        (
            saturate(target_tall * numerator / denominator).max(1),
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

    #[cfg(any(
        all(any(target_os = "linux", windows), feature = "cuda"),
        all(windows, feature = "d3d11-convert")
    ))]
    async fn check_scaler_caps(
        mut scaler: impl g2g_core::AsyncElement,
        format: g2g_core::RawVideoFormat,
        output_format: g2g_core::RawVideoFormat,
    ) {
        use crate::testutil::CollectSink;
        use g2g_core::{Colorimetry, Interlace, PipelinePacket, Rate};

        let raw = |format, width, height, pixel_aspect| Caps::RawVideo {
            format,
            width: Dim::Fixed(width),
            height: Dim::Fixed(height),
            framerate: Rate::Fixed(30 << 16),
            interlace: Interlace::Progressive,
            colorimetry: Colorimetry::UNKNOWN,
            pixel_aspect,
        };
        let mut out = CollectSink::default();
        for (input_ratio, requested_ratio, expected) in [
            (exact(16, 15), PixelAspect::Any, exact(4, 3)),
            (PixelAspect::SQUARE, PixelAspect::Any, exact(5, 4)),
            (
                PixelAspect::SQUARE,
                PixelAspect::SQUARE,
                PixelAspect::SQUARE,
            ),
        ] {
            let input = raw(format, 720, 576, input_ratio);
            let output = raw(output_format, 576, 576, requested_ratio);
            scaler.configure_pipeline(&input).unwrap();
            scaler.configure_output(&output).unwrap();
            out.packets.clear();
            // The runner sends its fixed output caps after configuring both pads.
            for _ in 0..2 {
                scaler
                    .process(PipelinePacket::CapsChanged(output.clone()), &mut out)
                    .await
                    .unwrap();
            }
            assert!(
                matches!(
                    out.packets.as_slice(),
                    [PipelinePacket::CapsChanged(Caps::RawVideo {
                        pixel_aspect, width: Dim::Fixed(576), height: Dim::Fixed(576), ..
                    })] if *pixel_aspect == expected
                ),
                "{:?}",
                out.packets
            );
        }
    }

    #[cfg(all(any(target_os = "linux", windows), feature = "cuda"))]
    #[tokio::test]
    async fn cuda_scaler_emits_the_resized_pixel_aspect() {
        use g2g_core::RawVideoFormat::Rgba8;
        check_scaler_caps(crate::cudascale::CudaScale::new(), Rgba8, Rgba8).await;
    }

    #[cfg(all(target_os = "linux", feature = "hip"))]
    #[tokio::test]
    async fn hip_scaler_emits_the_resized_pixel_aspect() {
        use g2g_core::RawVideoFormat::Rgba8;
        check_scaler_caps(crate::hipconvertscale::HipConvertScale::new(), Rgba8, Rgba8).await;
    }

    #[cfg(all(windows, feature = "d3d11-convert"))]
    #[tokio::test]
    async fn d3d11_converter_emits_the_resized_pixel_aspect() {
        use g2g_core::RawVideoFormat::{Bgra8, Nv12};
        check_scaler_caps(crate::d3d11convert::D3D11Convert::new(), Bgra8, Nv12).await;
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

    #[cfg(all(target_os = "linux", feature = "wgpu-present"))]
    #[test]
    fn a_point_in_the_bars_is_off_the_picture_and_one_inside_scales_to_it() {
        let picture_width = 1920;
        let picture_height = 1080;
        let viewport = letterbox(picture_width, picture_height, PixelAspect::Any, 1000, 1000);
        let top_bar = f64::from(viewport.y) - 1.0;
        assert_eq!(
            picture_point_in(viewport, picture_width, picture_height, 500.0, top_bar),
            None
        );
        let centre_x = f64::from(viewport.x) + f64::from(viewport.width) / 2.0;
        let centre_y = f64::from(viewport.y) + f64::from(viewport.height) / 2.0;
        assert_eq!(
            picture_point_in(viewport, picture_width, picture_height, centre_x, centre_y),
            Some((picture_width / 2, picture_height / 2))
        );
        let bottom_right = picture_point_in(
            viewport,
            picture_width,
            picture_height,
            f64::from(viewport.x + viewport.width) - 0.5,
            f64::from(viewport.y + viewport.height) - 0.5,
        );
        assert_eq!(bottom_right, Some((picture_width - 1, picture_height - 1)));
    }
}
