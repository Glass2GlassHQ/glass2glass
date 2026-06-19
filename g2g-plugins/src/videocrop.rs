//! Software rectangular crop (Tier-1 A1). Extracts a sub-rectangle of a raw
//! video frame, preserving the pixel format, for ROI-driven flows (a detector
//! emits boxes, the cropper extracts patches a classifier sees). No
//! resampling, a per-plane row copy.
//!
//! 4:2:0 (`Nv12`, `I420`) needs an even crop origin and size, since chroma is
//! subsampled 2x2; odd coords fail negotiation/configure loud. Packed formats
//! (`Rgba8`, `Bgra8`) crop at any coords. CPU-only `no_std` baseline.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, MemoryDomain,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, Rate, RawVideoFormat,
};

const FORMATS: [RawVideoFormat; 4] = [
    RawVideoFormat::Rgba8,
    RawVideoFormat::Bgra8,
    RawVideoFormat::Nv12,
    RawVideoFormat::I420,
];

#[derive(Debug)]
pub struct VideoCrop {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    /// Format, dims, and framerate of the configured input stream, updated by
    /// a mid-stream `CapsChanged`.
    input: Option<(RawVideoFormat, u32, u32, Rate)>,
    configured: bool,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl VideoCrop {
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            w: width,
            h: height,
            input: None,
            configured: false,
            last_caps: None,
            emitted: 0,
        }
    }

    pub fn rect(&self) -> (u32, u32, u32, u32) {
        (self.x, self.y, self.w, self.h)
    }

    fn accept_input(&self, caps: &Caps) -> Result<(RawVideoFormat, u32, u32, Rate), G2gError> {
        let Caps::RawVideo {
            format,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate,
        } = caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if !FORMATS.contains(format) || *w == 0 || *h == 0 {
            return Err(G2gError::CapsMismatch);
        }
        if is_yuv420(*format) && (*w % 2 != 0 || *h % 2 != 0) {
            return Err(G2gError::CapsMismatch);
        }
        Ok((*format, *w, *h, framerate.clone()))
    }

    /// The crop rect must be non-empty, lie inside the input frame, and be even
    /// on all four coords when the format is 4:2:0.
    fn validate_rect(&self, format: RawVideoFormat, in_w: u32, in_h: u32) -> Result<(), G2gError> {
        if self.w == 0 || self.h == 0 {
            return Err(G2gError::CapsMismatch);
        }
        if (self.x as u64 + self.w as u64) > in_w as u64
            || (self.y as u64 + self.h as u64) > in_h as u64
        {
            return Err(G2gError::CapsMismatch);
        }
        if is_yuv420(format) && (self.x % 2 != 0 || self.y % 2 != 0 || self.w % 2 != 0 || self.h % 2 != 0) {
            return Err(G2gError::CapsMismatch);
        }
        Ok(())
    }

    fn even_rect_ok(&self, format: RawVideoFormat) -> bool {
        !is_yuv420(format) || (self.x % 2 == 0 && self.y % 2 == 0 && self.w % 2 == 0 && self.h % 2 == 0)
    }
}

impl AsyncElement for VideoCrop {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        for format in FORMATS {
            let candidate = Caps::RawVideo {
                format,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
            };
            if let Ok(narrowed) = upstream_caps.intersect(&candidate) {
                return Ok(narrowed);
            }
        }
        Err(G2gError::CapsMismatch)
    }

    /// Native `DerivedOutput`: any supported raw input maps to the same format
    /// at the crop rect's dims, framerate preserved. A 4:2:0 format with an odd
    /// rect collapses to the empty set so the solve fails loud; the rect-fits-
    /// the-frame check is deferred to configure where input dims are absolute.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let (w, h) = (self.w, self.h);
        let even_rect_ok = |format| self.even_rect_ok(format);
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match input {
            Caps::RawVideo { format, framerate, .. } if FORMATS.contains(format) && even_rect_ok(*format) => {
                CapsSet::one(Caps::RawVideo {
                    format: *format,
                    width: Dim::Fixed(w),
                    height: Dim::Fixed(h),
                    framerate: framerate.clone(),
                })
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (format, w, h, rate) = self.accept_input(absolute_caps)?;
        self.validate_rect(format, w, h)?;
        self.input = Some((format, w, h, rate));
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let (format, in_w, in_h, rate) = match &self.input {
                        Some((f, w, h, r)) => (*f, *w, *h, r.clone()),
                        None => return Err(G2gError::NotConfigured),
                    };
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let src = slice.as_slice();
                    if src.len() < frame_byte_size(format, in_w, in_h) {
                        return Err(G2gError::CapsMismatch);
                    }
                    let cropped = crop(
                        src,
                        format,
                        (in_w as usize, in_h as usize),
                        (self.x as usize, self.y as usize, self.w as usize, self.h as usize),
                    );

                    let new_caps = Caps::RawVideo {
                        format,
                        width: Dim::Fixed(self.w),
                        height: Dim::Fixed(self.h),
                        framerate: rate,
                    };
                    if self.last_caps.as_ref() != Some(&new_caps) {
                        out.push(PipelinePacket::CapsChanged(new_caps.clone())).await?;
                        self.last_caps = Some(new_caps);
                    }
                    let out_frame = Frame {
                        domain: MemoryDomain::System(SystemSlice::from_boxed(cropped)),
                        timing: frame.timing,
                        sequence: self.emitted,
                    };
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(out_frame)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    let (format, w, h, rate) = self.accept_input(&c)?;
                    self.validate_rect(format, w, h)?;
                    self.input = Some((format, w, h, rate));
                }
                PipelinePacket::Flush => {
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                // Segment is control: forward unchanged.
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                PipelinePacket::Eos => {}
            }
            Ok(())
        })
    }
}

impl PadTemplates for VideoCrop {
    fn pad_templates() -> Vec<PadTemplate> {
        let any_geometry = |format| Caps::RawVideo {
            format,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        let set = CapsSet::from_alternatives(FORMATS.map(any_geometry).to_vec());
        Vec::from([PadTemplate::sink(set.clone()), PadTemplate::source(set)])
    }
}

fn is_yuv420(format: RawVideoFormat) -> bool {
    matches!(format, RawVideoFormat::Nv12 | RawVideoFormat::I420)
}

fn frame_byte_size(format: RawVideoFormat, w: u32, h: u32) -> usize {
    let (w, h) = (w as usize, h as usize);
    match format {
        RawVideoFormat::Rgba8 | RawVideoFormat::Bgra8 => w * h * 4,
        RawVideoFormat::Nv12 | RawVideoFormat::I420 => w * h * 3 / 2,
    }
}

/// Copy a `w x h` sub-rectangle at `(x, y)` out of one `channels`-interleaved
/// plane of `src_w x src_h`. NV12's UV plane uses `channels = 2`.
fn crop_plane(
    src: &[u8],
    src_w: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    channels: usize,
) -> Vec<u8> {
    let row_bytes = w * channels;
    let mut dst = vec![0u8; h * row_bytes];
    for row in 0..h {
        let src_off = ((y + row) * src_w + x) * channels;
        let dst_off = row * row_bytes;
        dst[dst_off..dst_off + row_bytes].copy_from_slice(&src[src_off..src_off + row_bytes]);
    }
    dst
}

/// Crop one frame to the `w x h` rect at `(x, y)`, preserving `format`. `src`
/// is validated to hold the input frame; all coords are even when the format
/// is 4:2:0.
fn crop(
    src: &[u8],
    format: RawVideoFormat,
    dims: (usize, usize),
    rect: (usize, usize, usize, usize),
) -> Box<[u8]> {
    let (in_w, in_h) = dims;
    let (x, y, w, h) = rect;
    match format {
        RawVideoFormat::Rgba8 | RawVideoFormat::Bgra8 => {
            crop_plane(src, in_w, x, y, w, h, 4).into_boxed_slice()
        }
        RawVideoFormat::Nv12 => {
            let luma_in = in_w * in_h;
            let chroma_in = (in_w / 2) * (in_h / 2) * 2;
            let mut out = crop_plane(src, in_w, x, y, w, h, 1);
            let chroma = crop_plane(
                &src[luma_in..luma_in + chroma_in],
                in_w / 2,
                x / 2,
                y / 2,
                w / 2,
                h / 2,
                2,
            );
            out.extend_from_slice(&chroma);
            out.into_boxed_slice()
        }
        RawVideoFormat::I420 => {
            let luma_in = in_w * in_h;
            let chroma_in = (in_w / 2) * (in_h / 2);
            let mut out = crop_plane(src, in_w, x, y, w, h, 1);
            let u = crop_plane(
                &src[luma_in..luma_in + chroma_in],
                in_w / 2,
                x / 2,
                y / 2,
                w / 2,
                h / 2,
                1,
            );
            let v = crop_plane(
                &src[luma_in + chroma_in..luma_in + 2 * chroma_in],
                in_w / 2,
                x / 2,
                y / 2,
                w / 2,
                h / 2,
                1,
            );
            out.extend_from_slice(&u);
            out.extend_from_slice(&v);
            out.into_boxed_slice()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgba_caps(w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Fixed(30 << 16),
        }
    }

    #[test]
    fn crop_plane_copies_subrect() {
        // 4x2 single-channel plane; crop a 2x2 at (1,0).
        let src: Vec<u8> = (0..8).collect();
        let out = crop_plane(&src, 4, 1, 0, 2, 2, 1);
        assert_eq!(&out[..], &[1, 2, 5, 6]);
    }

    #[test]
    fn crop_rgba_extracts_pixels() {
        // 4x4 RGBA where pixel p = [4p, 4p+1, 4p+2, 4p+3]; crop 2x2 at (1,1).
        let src: Vec<u8> = (0..(4 * 4 * 4) as u8).collect();
        let out = crop(&src, RawVideoFormat::Rgba8, (4, 4), (1, 1, 2, 2));
        // top-left of the crop is pixel (1,1) = index 5 -> byte 20.
        assert_eq!(&out[0..4], &[20, 21, 22, 23]);
        // next pixel (2,1) = index 6 -> byte 24.
        assert_eq!(&out[4..8], &[24, 25, 26, 27]);
        assert_eq!(out.len(), 2 * 2 * 4);
    }

    #[test]
    fn crop_nv12_keeps_plane_sizes() {
        // 4x4 NV12: 16 luma + 8 chroma. Crop 2x2 at (2,2) -> 4 luma + 2 chroma.
        let mut src = vec![0u8; 4 * 4 * 3 / 2];
        for (i, b) in src.iter_mut().enumerate() {
            *b = i as u8;
        }
        let out = crop(&src, RawVideoFormat::Nv12, (4, 4), (2, 2, 2, 2));
        assert_eq!(out.len(), 2 * 2 + (1 * 1 * 2));
        // luma row 0 of the crop starts at src[(2*4)+2] = src[10].
        assert_eq!(&out[0..2], &[10, 11]);
    }

    #[test]
    fn derived_output_maps_to_rect_dims() {
        let crop = VideoCrop::new(1, 1, 2, 2);
        let CapsConstraint::DerivedOutput(f) = crop.caps_constraint_as_transform() else {
            panic!("expected DerivedOutput");
        };
        let out = f(&rgba_caps(8, 8));
        assert_eq!(
            out.alternatives(),
            &[Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Fixed(2),
                height: Dim::Fixed(2),
                framerate: Rate::Fixed(30 << 16),
            }]
        );
        let h264 = Caps::CompressedVideo {
            codec: g2g_core::VideoCodec::H264,
            width: Dim::Fixed(8),
            height: Dim::Fixed(8),
            framerate: Rate::Any,
        };
        assert!(f(&h264).is_empty());
    }

    #[test]
    fn derived_output_rejects_odd_rect_for_yuv420() {
        let crop = VideoCrop::new(1, 0, 2, 2);
        let CapsConstraint::DerivedOutput(f) = crop.caps_constraint_as_transform() else {
            panic!("expected DerivedOutput");
        };
        let nv12 = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(8),
            height: Dim::Fixed(8),
            framerate: Rate::Any,
        };
        assert!(f(&nv12).is_empty(), "odd x is invalid for 4:2:0");
        assert!(!f(&rgba_caps(8, 8)).is_empty(), "packed formats allow odd coords");
    }

    #[test]
    fn configure_validates_fit_and_evenness() {
        // rect outside the frame fails.
        let mut c = VideoCrop::new(6, 0, 4, 4);
        assert_eq!(
            c.configure_pipeline(&rgba_caps(8, 8)).expect_err("rect overruns width"),
            G2gError::CapsMismatch
        );
        // odd rect into a 4:2:0 stream fails.
        let nv12 = |w, h| Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Any,
        };
        let mut c = VideoCrop::new(1, 2, 2, 2);
        assert_eq!(
            c.configure_pipeline(&nv12(8, 8)).expect_err("odd x for 4:2:0"),
            G2gError::CapsMismatch
        );
        // valid even rect inside the frame is accepted.
        let mut c = VideoCrop::new(2, 2, 4, 4);
        assert!(c.configure_pipeline(&nv12(8, 8)).is_ok());
        // packed format with an odd rect inside the frame is fine.
        let mut c = VideoCrop::new(1, 1, 3, 3);
        assert!(c.configure_pipeline(&rgba_caps(8, 8)).is_ok());
    }
}
