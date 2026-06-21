//! Software rectangular crop (Tier-1 A1). Removes pixels from each edge of a raw
//! video frame, preserving the pixel format, for ROI-driven flows (a detector
//! emits boxes, the cropper extracts patches a classifier sees). No resampling,
//! a per-plane row copy.
//!
//! Properties follow GStreamer's `videocrop`: `top` / `bottom` / `left` /
//! `right` are the pixels to crop off each edge (M183). Output geometry is the
//! input minus the edge insets. GStreamer's `-1` auto-crop sentinel is not
//! supported; negative values are rejected.
//!
//! 4:2:0 (`Nv12`, `I420`) needs even insets on all four edges, since chroma is
//! subsampled 2x2; odd values fail negotiation/configure loud. Packed formats
//! (`Rgba8`, `Bgra8`) crop at any inset. CPU-only `no_std` baseline.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec, Rate, RawVideoFormat,
};

const FORMATS: [RawVideoFormat; 4] = [
    RawVideoFormat::Rgba8,
    RawVideoFormat::Bgra8,
    RawVideoFormat::Nv12,
    RawVideoFormat::I420,
];

#[derive(Debug)]
pub struct VideoCrop {
    /// Pixels cropped from each edge (GStreamer `videocrop` model). Output
    /// geometry is the input minus `left + right` by `top + bottom`.
    top: u32,
    bottom: u32,
    left: u32,
    right: u32,
    /// Format, dims, and framerate of the configured input stream, updated by
    /// a mid-stream `CapsChanged`.
    input: Option<(RawVideoFormat, u32, u32, Rate)>,
    configured: bool,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl VideoCrop {
    /// Crop `top` / `bottom` / `left` / `right` pixels off the respective edges
    /// (GStreamer `videocrop` order). All-zero is an identity pass-through.
    pub fn new(top: u32, bottom: u32, left: u32, right: u32) -> Self {
        Self {
            top,
            bottom,
            left,
            right,
            input: None,
            configured: false,
            last_caps: None,
            emitted: 0,
        }
    }

    /// The configured edge insets `(top, bottom, left, right)`.
    pub fn insets(&self) -> (u32, u32, u32, u32) {
        (self.top, self.bottom, self.left, self.right)
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

    /// The insets must leave a non-empty frame (`left + right < in_w`,
    /// `top + bottom < in_h`) and be even on all four edges when the format is
    /// 4:2:0 (so the derived crop origin and size stay even).
    fn validate_insets(&self, format: RawVideoFormat, in_w: u32, in_h: u32) -> Result<(), G2gError> {
        if self.left + self.right >= in_w || self.top + self.bottom >= in_h {
            return Err(G2gError::CapsMismatch);
        }
        if is_yuv420(format) && !self.even_insets() {
            return Err(G2gError::CapsMismatch);
        }
        Ok(())
    }

    fn even_insets(&self) -> bool {
        self.top % 2 == 0 && self.bottom % 2 == 0 && self.left % 2 == 0 && self.right % 2 == 0
    }

    fn even_insets_ok(&self, format: RawVideoFormat) -> bool {
        !is_yuv420(format) || self.even_insets()
    }

    /// The crop rectangle `(x, y, w, h)` the insets describe on an `in_w x in_h`
    /// frame. Callers ensure the insets fit (via [`validate_insets`]).
    fn rect(&self, in_w: u32, in_h: u32) -> (u32, u32, u32, u32) {
        (
            self.left,
            self.top,
            in_w - self.left - self.right,
            in_h - self.top - self.bottom,
        )
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
    /// at the input geometry minus the edge insets, framerate preserved. A
    /// 4:2:0 format with an odd inset, or insets that consume the whole frame,
    /// collapses to the empty set so the solve fails loud.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let (lr, tb) = (self.left + self.right, self.top + self.bottom);
        let even_insets_ok = |format| self.even_insets_ok(format);
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match input {
            Caps::RawVideo { format, width, height, framerate }
                if FORMATS.contains(format) && even_insets_ok(*format) =>
            {
                match (shrink(width, lr), shrink(height, tb)) {
                    (Some(w), Some(h)) => CapsSet::one(Caps::RawVideo {
                        format: *format,
                        width: w,
                        height: h,
                        framerate: framerate.clone(),
                    }),
                    _ => CapsSet::from_alternatives(Vec::new()),
                }
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (format, w, h, rate) = self.accept_input(absolute_caps)?;
        self.validate_insets(format, w, h)?;
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
                    let (x, y, w, h) = self.rect(in_w, in_h);
                    let cropped = crop(
                        src,
                        format,
                        (in_w as usize, in_h as usize),
                        (x as usize, y as usize, w as usize, h as usize),
                    );

                    let new_caps = Caps::RawVideo {
                        format,
                        width: Dim::Fixed(w),
                        height: Dim::Fixed(h),
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
                        meta: Default::default(),
                    };
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(out_frame)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    let (format, w, h, rate) = self.accept_input(&c)?;
                    self.validate_insets(format, w, h)?;
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

    fn properties(&self) -> &'static [PropertySpec] {
        VIDEOCROP_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Video cropper",
            "Filter/Effect/Video",
            "Crops pixels from the edges of raw video",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        // GStreamer `videocrop` props are gint; `-1` means auto-crop, which we
        // don't implement, so negatives are rejected.
        let v = value.as_int().ok_or(PropError::Type)?;
        if v < 0 {
            return Err(PropError::Value);
        }
        let v = v as u32;
        match name {
            "top" => self.top = v,
            "bottom" => self.bottom = v,
            "left" => self.left = v,
            "right" => self.right = v,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        let v = match name {
            "top" => self.top,
            "bottom" => self.bottom,
            "left" => self.left,
            "right" => self.right,
            _ => return None,
        };
        Some(PropValue::Int(v as i64))
    }
}

/// `VideoCrop`'s settable properties (GStreamer `videocrop` model, M183): the
/// pixels to crop off each edge.
static VIDEOCROP_PROPS: &[PropertySpec] = &[
    PropertySpec::new("top", PropKind::Int, "pixels to crop at the top").with_default("0"),
    PropertySpec::new("bottom", PropKind::Int, "pixels to crop at the bottom").with_default("0"),
    PropertySpec::new("left", PropKind::Int, "pixels to crop at the left").with_default("0"),
    PropertySpec::new("right", PropKind::Int, "pixels to crop at the right").with_default("0"),
];

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

/// Shrink a dimension by `by`, collapsing to `None` (an empty caps) when the
/// crop would consume the whole extent.
fn shrink(d: &Dim, by: u32) -> Option<Dim> {
    match d {
        Dim::Fixed(v) => v.checked_sub(by).filter(|&r| r > 0).map(Dim::Fixed),
        Dim::Range { min, max } => {
            let max = max.checked_sub(by).filter(|&r| r > 0)?;
            let min = min.saturating_sub(by).max(1).min(max);
            Some(Dim::Range { min, max })
        }
        Dim::Any => Some(Dim::Any),
    }
}

fn frame_byte_size(format: RawVideoFormat, w: u32, h: u32) -> usize {
    let (w, h) = (w as usize, h as usize);
    match format {
        RawVideoFormat::Rgba8 | RawVideoFormat::Bgra8 => w * h * 4,
        RawVideoFormat::Nv12 | RawVideoFormat::I420 => w * h * 3 / 2,
        RawVideoFormat::Yuyv => w * h * 2,
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
        // YUYV is absent from SUPPORTED, so negotiation never admits it here;
        // convert to a planar format upstream before cropping.
        RawVideoFormat::Yuyv => unreachable!("videocrop: YUYV is not negotiated"),
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

    fn nv12_caps(w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Any,
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
    fn insets_derive_the_crop_rect() {
        // 8x8, crop top=2 bottom=2 left=2 right=2 -> rect (2,2) sized 4x4.
        let c = VideoCrop::new(2, 2, 2, 2);
        assert_eq!(c.rect(8, 8), (2, 2, 4, 4));
        // Asymmetric: top=0 bottom=4 left=0 right=4 -> rect (0,0) sized 4x4.
        let c = VideoCrop::new(0, 4, 0, 4);
        assert_eq!(c.rect(8, 8), (0, 0, 4, 4));
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
    fn derived_output_maps_to_inset_dims() {
        // 8x8 with 2px insets all round -> 4x4.
        let crop = VideoCrop::new(2, 2, 2, 2);
        let CapsConstraint::DerivedOutput(f) = crop.caps_constraint_as_transform() else {
            panic!("expected DerivedOutput");
        };
        let out = f(&rgba_caps(8, 8));
        assert_eq!(
            out.alternatives(),
            &[Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Fixed(4),
                height: Dim::Fixed(4),
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
    fn derived_output_rejects_odd_inset_for_yuv420() {
        // left=1 is odd: invalid for 4:2:0, fine for packed.
        let crop = VideoCrop::new(0, 0, 1, 1);
        let CapsConstraint::DerivedOutput(f) = crop.caps_constraint_as_transform() else {
            panic!("expected DerivedOutput");
        };
        assert!(f(&nv12_caps(8, 8)).is_empty(), "odd inset is invalid for 4:2:0");
        assert!(!f(&rgba_caps(8, 8)).is_empty(), "packed formats allow odd insets");
    }

    #[test]
    fn derived_output_empty_when_insets_consume_frame() {
        // left+right == width leaves nothing.
        let crop = VideoCrop::new(0, 0, 4, 4);
        let CapsConstraint::DerivedOutput(f) = crop.caps_constraint_as_transform() else {
            panic!("expected DerivedOutput");
        };
        assert!(f(&rgba_caps(8, 8)).is_empty(), "insets consume the whole width");
    }

    #[test]
    fn configure_validates_fit_and_evenness() {
        // insets wider than the frame fail.
        let mut c = VideoCrop::new(0, 0, 6, 4);
        assert_eq!(
            c.configure_pipeline(&rgba_caps(8, 8)).expect_err("insets overrun width"),
            G2gError::CapsMismatch
        );
        // odd inset into a 4:2:0 stream fails.
        let mut c = VideoCrop::new(0, 0, 1, 0);
        assert_eq!(
            c.configure_pipeline(&nv12_caps(8, 8)).expect_err("odd left for 4:2:0"),
            G2gError::CapsMismatch
        );
        // valid even insets are accepted.
        let mut c = VideoCrop::new(2, 2, 2, 2);
        assert!(c.configure_pipeline(&nv12_caps(8, 8)).is_ok());
        // packed format with odd insets inside the frame is fine.
        let mut c = VideoCrop::new(1, 1, 1, 1);
        assert!(c.configure_pipeline(&rgba_caps(8, 8)).is_ok());
        // all-zero insets are an identity pass-through.
        let mut c = VideoCrop::new(0, 0, 0, 0);
        assert!(c.configure_pipeline(&rgba_caps(8, 8)).is_ok());
    }
}
