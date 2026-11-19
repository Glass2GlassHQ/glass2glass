// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::fmt;

use g2g_core::{Caps, Dim};

use crate::VideoSize;

pub(crate) const VIDEO_PARAMS_PROBE: &str = "playervideoparams";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScaleFlags {
    Bilinear,
    Lanczos,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PadFill {
    Black,
    Transparent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Orientation {
    Clockwise,
    Counterclockwise,
    // ffmpeg's cclock_flip, a transpose across the top-left to bottom-right diagonal
    UpperLeftDiagonal,
    // ffmpeg's clock_flip
    UpperRightDiagonal,
    HorizontalFlip,
    VerticalFlip,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PictureFilter {
    Deinterlace,
    Denoise,
    Crop {
        width: u32,
        height: u32,
        x: u32,
        y: u32,
    },
    Orient(Orientation),
    Scale {
        width: u32,
        height: u32,
        flags: ScaleFlags,
    },
    Pad {
        width: u32,
        height: u32,
        x: u32,
        y: u32,
        fill: PadFill,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VideoFilterError {
    // the filter or one of its options is outside the set the player translates
    Unsupported(String),
    Malformed(String),
    // the chain does not fit the decoded picture, as ffmpeg refuses it
    DoesNotFit(String),
}

impl fmt::Display for VideoFilterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VideoFilterError::Unsupported(filter) => {
                write!(f, "unsupported video filter: {filter}")
            }
            VideoFilterError::Malformed(filter) => write!(f, "malformed video filter: {filter}"),
            VideoFilterError::DoesNotFit(reason) => {
                write!(f, "video filter does not fit: {reason}")
            }
        }
    }
}

impl std::error::Error for VideoFilterError {}

struct FilterOptions<'a> {
    filter: &'a str,
    positional: Vec<&'a str>,
    named: Vec<(&'a str, &'a str)>,
}

impl<'a> FilterOptions<'a> {
    fn parse(filter: &'a str, options: &'a str) -> Self {
        // ffmpeg separates options with ':', positional values first, then key=value pairs
        let mut positional = Vec::new();
        let mut named = Vec::new();
        for option in options.split(':').filter(|option| !option.is_empty()) {
            match option.split_once('=') {
                Some(pair) => named.push(pair),
                None => positional.push(option),
            }
        }
        FilterOptions {
            filter,
            positional,
            named,
        }
    }

    fn named(&self, keys: &[&str]) -> Option<&'a str> {
        self.named
            .iter()
            .find(|(key, _)| keys.contains(key))
            .map(|(_, value)| *value)
    }

    fn value(&self, keys: &[&str], position: usize) -> Option<&'a str> {
        self.named(keys)
            .or_else(|| self.positional.get(position).copied())
    }

    fn pixels(&self, keys: &[&str], position: usize) -> Result<u32, VideoFilterError> {
        let value = self
            .value(keys, position)
            .ok_or_else(|| VideoFilterError::Malformed(self.filter.to_string()))?;
        value
            .parse()
            .map_err(|_| VideoFilterError::Unsupported(self.filter.to_string()))
    }

    fn reject_unknown(
        &self,
        known: &[&str],
        positional_count: usize,
    ) -> Result<(), VideoFilterError> {
        let unknown_key = self.named.iter().any(|(key, _)| !known.contains(key));
        if unknown_key || self.positional.len() > positional_count {
            return Err(VideoFilterError::Unsupported(self.filter.to_string()));
        }
        Ok(())
    }
}

fn parse_filter(filter: &str) -> Result<PictureFilter, VideoFilterError> {
    let (name, options) = filter.split_once('=').unwrap_or((filter, ""));
    let options = FilterOptions::parse(filter, options);
    let unsupported = || VideoFilterError::Unsupported(filter.to_string());
    match name {
        "yadif" if options.positional.is_empty() && options.named.is_empty() => {
            Ok(PictureFilter::Deinterlace)
        }
        "hqdn3d" if options.positional.is_empty() && options.named.is_empty() => {
            Ok(PictureFilter::Denoise)
        }
        "hflip" if options.positional.is_empty() && options.named.is_empty() => {
            Ok(PictureFilter::Orient(Orientation::HorizontalFlip))
        }
        "vflip" if options.positional.is_empty() && options.named.is_empty() => {
            Ok(PictureFilter::Orient(Orientation::VerticalFlip))
        }
        "transpose" => {
            options.reject_unknown(&["dir"], 1)?;
            let orientation = match options.value(&["dir"], 0) {
                Some("clock" | "1") => Orientation::Clockwise,
                Some("cclock" | "2") => Orientation::Counterclockwise,
                Some("cclock_flip" | "0") | None => Orientation::UpperLeftDiagonal,
                Some("clock_flip" | "3") => Orientation::UpperRightDiagonal,
                Some(_) => return Err(unsupported()),
            };
            Ok(PictureFilter::Orient(orientation))
        }
        "crop" => {
            options.reject_unknown(&["w", "out_w", "h", "out_h", "x", "y"], 4)?;
            Ok(PictureFilter::Crop {
                width: options.pixels(&["w", "out_w"], 0)?,
                height: options.pixels(&["h", "out_h"], 1)?,
                x: options.pixels(&["x"], 2)?,
                y: options.pixels(&["y"], 3)?,
            })
        }
        "scale" => {
            options.reject_unknown(&["w", "width", "h", "height", "flags"], 2)?;
            let flags = match options.named(&["flags"]) {
                Some("lanczos") => ScaleFlags::Lanczos,
                Some("bilinear") => ScaleFlags::Bilinear,
                // ffmpeg scales bicubic without flags, which the player has no element for
                _ => return Err(unsupported()),
            };
            Ok(PictureFilter::Scale {
                width: options.pixels(&["w", "width"], 0)?,
                height: options.pixels(&["h", "height"], 1)?,
                flags,
            })
        }
        "pad" => {
            options.reject_unknown(&["w", "width", "h", "height", "x", "y", "color"], 5)?;
            let fill = match options.value(&["color"], 4) {
                Some("black") => PadFill::Black,
                Some("black@0") => PadFill::Transparent,
                _ => return Err(unsupported()),
            };
            Ok(PictureFilter::Pad {
                width: options.pixels(&["w", "width"], 0)?,
                height: options.pixels(&["h", "height"], 1)?,
                x: options.pixels(&["x"], 2)?,
                y: options.pixels(&["y"], 3)?,
                fill,
            })
        }
        _ => Err(unsupported()),
    }
}

pub(crate) fn parse_filters(filters: &[&str]) -> Result<Vec<PictureFilter>, VideoFilterError> {
    filters.iter().map(|filter| parse_filter(filter)).collect()
}

fn does_not_fit(filter: &PictureFilter, size: VideoSize) -> VideoFilterError {
    VideoFilterError::DoesNotFit(format!(
        "{filter:?} on a {}x{} picture",
        size.width, size.height
    ))
}

fn launch_element_and_output_size(
    filter: &PictureFilter,
    input: VideoSize,
) -> Result<(String, VideoSize), VideoFilterError> {
    match *filter {
        PictureFilter::Deinterlace => Ok((
            "deinterlace method=yadif mode=interlaced fields=auto".to_string(),
            input,
        )),
        // hqdn3d takes 8-bit planar YUV only
        PictureFilter::Denoise => Ok(("videoconvert format=I420 ! hqdn3d".to_string(), input)),
        PictureFilter::Orient(orientation) => {
            let (method, swaps_axes) = match orientation {
                Orientation::Clockwise => ("clockwise", true),
                Orientation::Counterclockwise => ("counterclockwise", true),
                Orientation::UpperLeftDiagonal => ("upper-left-diagonal", true),
                Orientation::UpperRightDiagonal => ("upper-right-diagonal", true),
                Orientation::HorizontalFlip => ("horizontal-flip", false),
                Orientation::VerticalFlip => ("vertical-flip", false),
            };
            let output = match swaps_axes {
                true => VideoSize {
                    width: input.height,
                    height: input.width,
                },
                false => input,
            };
            Ok((format!("videoflip method={method}"), output))
        }
        PictureFilter::Crop {
            width,
            height,
            x,
            y,
        } => {
            let right = input
                .width
                .checked_sub(x)
                .and_then(|rest| rest.checked_sub(width));
            let bottom = input
                .height
                .checked_sub(y)
                .and_then(|rest| rest.checked_sub(height));
            let (Some(right), Some(bottom)) = (right, bottom) else {
                return Err(does_not_fit(filter, input));
            };
            Ok((
                format!("videocrop left={x} top={y} right={right} bottom={bottom}"),
                VideoSize { width, height },
            ))
        }
        PictureFilter::Scale {
            width,
            height,
            flags,
        } => {
            let method = match flags {
                ScaleFlags::Bilinear => "bilinear",
                ScaleFlags::Lanczos => "lanczos",
            };
            Ok((
                format!("videoscale width={width} height={height} method={method}"),
                VideoSize { width, height },
            ))
        }
        PictureFilter::Pad {
            width,
            height,
            x,
            y,
            fill,
        } => {
            let right = width
                .checked_sub(x)
                .and_then(|rest| rest.checked_sub(input.width));
            let bottom = height
                .checked_sub(y)
                .and_then(|rest| rest.checked_sub(input.height));
            let (Some(right), Some(bottom)) = (right, bottom) else {
                return Err(does_not_fit(filter, input));
            };
            let fill = match fill {
                PadFill::Black => "black",
                PadFill::Transparent => "transparent",
            };
            // videobox takes RGBA only and borders an edge given a negative value
            Ok((
                format!(
                    "videoconvert format=RGBA ! \
                     videobox left=-{x} top=-{y} right=-{right} bottom=-{bottom} fill={fill}"
                ),
                VideoSize { width, height },
            ))
        }
    }
}

pub(crate) fn launch_chain(
    filters: &[PictureFilter],
    decoded: VideoSize,
    source: Option<VideoSize>,
) -> Result<String, VideoFilterError> {
    let mut size = decoded;
    let mut elements = Vec::with_capacity(filters.len());
    if let Some(source) = source.filter(|source| *source != decoded) {
        // The preview chain's geometry is in full-resolution source pixels.
        let scale = PictureFilter::Scale {
            width: source.width,
            height: source.height,
            flags: ScaleFlags::Bilinear,
        };
        let (element, output) = launch_element_and_output_size(&scale, size)?;
        elements.push(element);
        size = output;
    }
    for filter in filters {
        let (element, output) = launch_element_and_output_size(filter, size)?;
        elements.push(element);
        size = output;
    }
    Ok(elements.join(" ! "))
}

pub(crate) fn video_size_of(caps: &Caps) -> Option<VideoSize> {
    match caps {
        Caps::RawVideo {
            width: Dim::Fixed(width),
            height: Dim::Fixed(height),
            ..
        }
        | Caps::CompressedVideo {
            width: Dim::Fixed(width),
            height: Dim::Fixed(height),
            ..
        } => Some(VideoSize {
            width: *width,
            height: *height,
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DECODED: VideoSize = VideoSize {
        width: 1920,
        height: 1080,
    };

    fn chain(filters: &[&str]) -> Result<String, VideoFilterError> {
        launch_chain(&parse_filters(filters)?, DECODED, None)
    }

    #[test]
    fn dcpwizard_scope_placement_becomes_a_scale_and_a_pad() {
        // dcpwizard's placement of a 1920x1080 source in a 2k scope container
        assert_eq!(
            chain(&[
                "scale=w=1524:h=858:flags=lanczos",
                "pad=w=2048:h=858:x=262:y=0:color=black"
            ]),
            Ok(
                "videoscale width=1524 height=858 method=lanczos ! videoconvert format=RGBA ! \
                videobox left=-262 top=-0 right=-262 bottom=-0 fill=black"
                    .to_string()
            )
        );
    }

    #[test]
    fn dcpwizard_deinterlace_and_denoise_becomes_two_elements() {
        assert_eq!(
            chain(&["yadif", "hqdn3d"]),
            Ok("deinterlace method=yadif mode=interlaced fields=auto ! \
                videoconvert format=I420 ! hqdn3d"
                .to_string())
        );
    }

    #[test]
    fn a_crop_after_a_quarter_turn_counts_its_edges_on_the_turned_picture() {
        assert_eq!(
            chain(&["transpose=clock", "crop=1000:1800:40:60"]),
            Ok(
                "videoflip method=clockwise ! videocrop left=40 top=60 right=40 bottom=60"
                    .to_string()
            )
        );
    }

    #[test]
    fn a_filter_outside_the_translated_set_is_refused() {
        for filter in [
            "eq=brightness=0.1",
            "hqdn3d=8",
            "scale=w=10:h=10",
            "crop=in_w/2:in_h:0:0",
        ] {
            assert_eq!(
                parse_filters(&[filter]),
                Err(VideoFilterError::Unsupported(filter.to_string())),
                "{filter}"
            );
        }
    }

    #[test]
    fn a_crop_larger_than_the_picture_does_not_fit() {
        assert!(matches!(
            chain(&["crop=1920:1080:1:0"]),
            Err(VideoFilterError::DoesNotFit(_))
        ));
    }
}
