//! Software flip / rotate (Tier-1 A). Mirrors or rotates a raw video frame by
//! a fixed `FlipMethod`, preserving the pixel format, for portrait-mode mobile
//! sources fed to a landscape pipeline. No resampling, a per-plane coordinate
//! remap.
//!
//! `Rotate90Cw` / `Rotate90Ccw` swap width and height; the mirrors and
//! `Rotate180` keep the geometry. 4:2:0 (`Nv12`, `I420`) needs even input dims
//! since chroma is subsampled 2x2; odd dims fail negotiation/configure loud.
//! Packed formats (`Rgba8`, `Bgra8`) take any dims. CPU-only `no_std` baseline.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::{SystemSlice, SystemView};
use g2g_core::tensor::TensorView;
use g2g_core::log::{short_type_name, LogSource};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec, Rate, RawVideoFormat,
};
use g2g_core::{g2g_info, g2g_trace};
use alloc::string::String;

const FORMATS: [RawVideoFormat; 4] = [
    RawVideoFormat::Rgba8,
    RawVideoFormat::Bgra8,
    RawVideoFormat::Nv12,
    RawVideoFormat::I420,
];

/// The geometric operation `VideoFlip` applies. The two 90-degree rotations
/// transpose the frame and so swap width and height; the rest preserve it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlipMethod {
    HorizontalMirror,
    VerticalMirror,
    Rotate90Cw,
    Rotate180,
    Rotate90Ccw,
}

impl FlipMethod {
    fn swaps_dims(self) -> bool {
        matches!(self, FlipMethod::Rotate90Cw | FlipMethod::Rotate90Ccw)
    }
}

#[derive(Debug)]
pub struct VideoFlip {
    method: FlipMethod,
    /// Format, dims, and framerate of the configured input stream, updated by
    /// a mid-stream `CapsChanged`.
    input: Option<(RawVideoFormat, u32, u32, Rate)>,
    configured: bool,
    last_caps: Option<Caps>,
    emitted: u64,
    /// Instance name assigned by the runner (M179), for this element's log lines.
    instance_name: Option<String>,
}

impl VideoFlip {
    pub fn new(method: FlipMethod) -> Self {
        Self {
            method,
            input: None,
            configured: false,
            last_caps: None,
            emitted: 0,
            instance_name: None,
        }
    }

    pub fn method(&self) -> FlipMethod {
        self.method
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

    /// Output geometry for the configured method: the 90-degree rotations
    /// transpose, the mirrors and 180 preserve.
    fn output_dims(&self, w: u32, h: u32) -> (u32, u32) {
        if self.method.swaps_dims() {
            (h, w)
        } else {
            (w, h)
        }
    }
}

impl AsyncElement for VideoFlip {
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
    /// and framerate, with width and height swapped for the 90-degree
    /// rotations and preserved otherwise. The 4:2:0 even-dim check is deferred
    /// to configure where input dims are absolute.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let swaps = self.method.swaps_dims();
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match input {
            Caps::RawVideo { format, width, height, framerate } if FORMATS.contains(format) => {
                let (out_w, out_h) = if swaps {
                    (height.clone(), width.clone())
                } else {
                    (width.clone(), height.clone())
                };
                CapsSet::one(Caps::RawVideo {
                    format: *format,
                    width: out_w,
                    height: out_h,
                    framerate: framerate.clone(),
                })
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (format, w, h, rate) = self.accept_input(absolute_caps)?;
        self.input = Some((format, w, h, rate));
        self.configured = true;
        g2g_info!(self, "configured {:?} {}x{} {:?}", format, w, h, self.method);
        Ok(ConfigureOutcome::Accepted)
    }

    fn set_instance_name(&mut self, name: String) {
        self.instance_name = Some(name);
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

                    let (out_w, out_h) = self.output_dims(in_w, in_h);
                    let new_caps = Caps::RawVideo {
                        format,
                        width: Dim::Fixed(out_w),
                        height: Dim::Fixed(out_h),
                        framerate: rate,
                    };
                    if self.last_caps.as_ref() != Some(&new_caps) {
                        out.push(PipelinePacket::CapsChanged(new_caps.clone())).await?;
                        self.last_caps = Some(new_caps);
                    }

                    // Packed RGBA/BGRA already in shared CPU memory is the
                    // zero-copy case: a flip is a pure coordinate remap, so we
                    // compose strides on the *same* `Arc` backing and copy
                    // nothing. Planar (4:2:0) is excluded because its subsampled
                    // planes aren't one strided tensor (see tensor.rs), and an
                    // owned `System` buffer has no shared backing to alias, so
                    // both fall through to the copy path below.
                    let packed =
                        matches!(format, RawVideoFormat::Rgba8 | RawVideoFormat::Bgra8);
                    let out_frame = match &frame.domain {
                        MemoryDomain::SystemView(sv) if packed => {
                            let out_view = flip_view(*sv.view(), self.method);
                            g2g_trace!(self, "zero-copy flip frame #{} {}x{} -> {}x{}", self.emitted, in_w, in_h, out_w, out_h);
                            Frame {
                                domain: MemoryDomain::SystemView(SystemView::new(
                                    sv.backing().clone(),
                                    out_view,
                                )),
                                timing: frame.timing,
                                sequence: self.emitted,
                                meta: Default::default(),
                            }
                        }
                        _ => {
                            // Copy path: owned `System` bytes, or a non-packed
                            // `SystemView` materialized to contiguous first.
                            let flipped = match &frame.domain {
                                MemoryDomain::System(slice) => {
                                    let src = slice.as_slice();
                                    if src.len() < frame_byte_size(format, in_w, in_h) {
                                        return Err(G2gError::CapsMismatch);
                                    }
                                    flip(src, format, (in_w as usize, in_h as usize), self.method)
                                }
                                MemoryDomain::SystemView(sv) => {
                                    let src = sv.materialize();
                                    if src.len() < frame_byte_size(format, in_w, in_h) {
                                        return Err(G2gError::CapsMismatch);
                                    }
                                    flip(&src, format, (in_w as usize, in_h as usize), self.method)
                                }
                                _ => return Err(G2gError::UnsupportedDomain),
                            };
                            g2g_trace!(self, "flip frame #{} {}x{} -> {}x{}", self.emitted, in_w, in_h, out_w, out_h);
                            Frame {
                                domain: MemoryDomain::System(SystemSlice::from_boxed(flipped)),
                                timing: frame.timing,
                                sequence: self.emitted,
                                meta: Default::default(),
                            }
                        }
                    };
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(out_frame)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    let (format, w, h, rate) = self.accept_input(&c)?;
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
        VIDEOFLIP_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Video flip / rotate",
            "Filter/Effect/Video",
            "Flips or rotates raw video (mirror, 90/180/270 degree rotations)",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "method" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.method = flip_method_from_str(s).ok_or(PropError::Value)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "method" => Some(PropValue::Str(flip_method_to_str(self.method).into())),
            _ => None,
        }
    }
}

/// `VideoFlip`'s settable properties (M104).
static VIDEOFLIP_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "method",
    PropKind::Str,
    "flip / rotate method",
)
.with_enum_values(
    "horizontal-mirror | vertical-mirror | rotate-90cw | rotate-180 | rotate-90ccw",
)
.with_default("horizontal-mirror")];

/// Parse a `method` property string to a [`FlipMethod`].
fn flip_method_from_str(s: &str) -> Option<FlipMethod> {
    match s {
        "horizontal-mirror" => Some(FlipMethod::HorizontalMirror),
        "vertical-mirror" => Some(FlipMethod::VerticalMirror),
        "rotate-90cw" => Some(FlipMethod::Rotate90Cw),
        "rotate-180" => Some(FlipMethod::Rotate180),
        "rotate-90ccw" => Some(FlipMethod::Rotate90Ccw),
        _ => None,
    }
}

/// The `method` property string for a [`FlipMethod`].
fn flip_method_to_str(m: FlipMethod) -> &'static str {
    match m {
        FlipMethod::HorizontalMirror => "horizontal-mirror",
        FlipMethod::VerticalMirror => "vertical-mirror",
        FlipMethod::Rotate90Cw => "rotate-90cw",
        FlipMethod::Rotate180 => "rotate-180",
        FlipMethod::Rotate90Ccw => "rotate-90ccw",
    }
}

impl PadTemplates for VideoFlip {
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

/// M179: log identity. Category is the short type name (matching what the runner
/// derives, so `G2G_DEBUG=VideoFlip:debug` filters both); instance is the
/// runner-assigned name.
impl LogSource for VideoFlip {
    fn log_category(&self) -> &'static str {
        short_type_name::<Self>()
    }
    fn log_instance(&self) -> Option<&str> {
        self.instance_name.as_deref()
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
        RawVideoFormat::Yuyv => w * h * 2,
    }
}

/// The zero-copy analog of [`flip`] for a packed `[H, W, C]` view: express the
/// method as stride manipulations over the same bytes (M180). A mirror reverses
/// one spatial axis; a 90-degree rotation transposes the H/W axes then reverses
/// one. The channel axis (2) is never touched, so each pixel's bytes stay
/// intact. Matches [`src_coord`]'s mapping exactly, verified by the m180 test.
fn flip_view(view: TensorView, method: FlipMethod) -> TensorView {
    match method {
        FlipMethod::HorizontalMirror => view.reversed_axis(1),
        FlipMethod::VerticalMirror => view.reversed_axis(0),
        FlipMethod::Rotate180 => view.reversed_axis(0).reversed_axis(1),
        FlipMethod::Rotate90Cw => view.transposed(0, 1).reversed_axis(1),
        FlipMethod::Rotate90Ccw => view.transposed(0, 1).reversed_axis(0),
    }
}

/// Source coordinate that feeds output `(ox, oy)` for one plane of input dims
/// `(pw, ph)`. The 90-degree rotations read from a transposed position; the
/// mirrors and 180 reflect within the same dims.
fn src_coord(method: FlipMethod, ox: usize, oy: usize, pw: usize, ph: usize) -> (usize, usize) {
    match method {
        FlipMethod::HorizontalMirror => (pw - 1 - ox, oy),
        FlipMethod::VerticalMirror => (ox, ph - 1 - oy),
        FlipMethod::Rotate180 => (pw - 1 - ox, ph - 1 - oy),
        FlipMethod::Rotate90Cw => (oy, ph - 1 - ox),
        FlipMethod::Rotate90Ccw => (pw - 1 - oy, ox),
    }
}

/// Remap one `channels`-interleaved plane of `pw x ph` by `method`. NV12's UV
/// plane uses `channels = 2` so each chroma pair moves as a unit.
fn transform_plane(
    src: &[u8],
    pw: usize,
    ph: usize,
    channels: usize,
    method: FlipMethod,
) -> Vec<u8> {
    let (ow, oh) = if method.swaps_dims() { (ph, pw) } else { (pw, ph) };
    let mut dst = vec![0u8; ow * oh * channels];
    for oy in 0..oh {
        for ox in 0..ow {
            let (ix, iy) = src_coord(method, ox, oy, pw, ph);
            let s = (iy * pw + ix) * channels;
            let d = (oy * ow + ox) * channels;
            dst[d..d + channels].copy_from_slice(&src[s..s + channels]);
        }
    }
    dst
}

/// Flip one frame by `method`, preserving `format`. `src` is validated to hold
/// the input frame; dims are even when the format is 4:2:0.
fn flip(
    src: &[u8],
    format: RawVideoFormat,
    dims: (usize, usize),
    method: FlipMethod,
) -> Box<[u8]> {
    let (in_w, in_h) = dims;
    match format {
        RawVideoFormat::Rgba8 | RawVideoFormat::Bgra8 => {
            transform_plane(src, in_w, in_h, 4, method).into_boxed_slice()
        }
        RawVideoFormat::Nv12 => {
            let luma_in = in_w * in_h;
            let chroma_in = (in_w / 2) * (in_h / 2) * 2;
            let mut out = transform_plane(src, in_w, in_h, 1, method);
            let chroma = transform_plane(
                &src[luma_in..luma_in + chroma_in],
                in_w / 2,
                in_h / 2,
                2,
                method,
            );
            out.extend_from_slice(&chroma);
            out.into_boxed_slice()
        }
        RawVideoFormat::I420 => {
            let luma_in = in_w * in_h;
            let chroma_in = (in_w / 2) * (in_h / 2);
            let mut out = transform_plane(src, in_w, in_h, 1, method);
            let u = transform_plane(
                &src[luma_in..luma_in + chroma_in],
                in_w / 2,
                in_h / 2,
                1,
                method,
            );
            let v = transform_plane(
                &src[luma_in + chroma_in..luma_in + 2 * chroma_in],
                in_w / 2,
                in_h / 2,
                1,
                method,
            );
            out.extend_from_slice(&u);
            out.extend_from_slice(&v);
            out.into_boxed_slice()
        }
        // YUYV is absent from SUPPORTED, so negotiation never admits it here;
        // convert to a planar format upstream before flipping.
        RawVideoFormat::Yuyv => unreachable!("videoflip: YUYV is not negotiated"),
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
    fn transform_plane_square_methods() {
        // 2x2 single-channel plane: row0 [0,1], row1 [2,3].
        let src = vec![0u8, 1, 2, 3];
        assert_eq!(transform_plane(&src, 2, 2, 1, FlipMethod::HorizontalMirror), vec![1, 0, 3, 2]);
        assert_eq!(transform_plane(&src, 2, 2, 1, FlipMethod::VerticalMirror), vec![2, 3, 0, 1]);
        assert_eq!(transform_plane(&src, 2, 2, 1, FlipMethod::Rotate180), vec![3, 2, 1, 0]);
        assert_eq!(transform_plane(&src, 2, 2, 1, FlipMethod::Rotate90Cw), vec![2, 0, 3, 1]);
        assert_eq!(transform_plane(&src, 2, 2, 1, FlipMethod::Rotate90Ccw), vec![1, 3, 0, 2]);
    }

    #[test]
    fn transform_plane_rotate90_swaps_dims() {
        // 3x2 plane: row0 [0,1,2], row1 [3,4,5]. Rotate 90 CW -> 2x3.
        let src: Vec<u8> = (0..6).collect();
        let out = transform_plane(&src, 3, 2, 1, FlipMethod::Rotate90Cw);
        assert_eq!(out, vec![3, 0, 4, 1, 5, 2]);
    }

    #[test]
    fn flip_rgba_mirrors_pixels() {
        // 2x2 RGBA where pixel p = [4p, 4p+1, 4p+2, 4p+3].
        let src: Vec<u8> = (0..(2 * 2 * 4) as u8).collect();
        let out = flip(&src, RawVideoFormat::Rgba8, (2, 2), FlipMethod::HorizontalMirror);
        // row 0 swaps pixel 0 and 1: [4,5,6,7, 0,1,2,3].
        assert_eq!(&out[0..4], &[4, 5, 6, 7]);
        assert_eq!(&out[4..8], &[0, 1, 2, 3]);
        assert_eq!(out.len(), 2 * 2 * 4);
    }

    #[test]
    fn flip_nv12_rotate90_swaps_geometry() {
        // 4x2 NV12: 8 luma + 4 chroma. Rotate 90 CW -> 2x4, byte total preserved.
        let mut src = vec![0u8; 4 * 2 * 3 / 2];
        for (i, b) in src.iter_mut().enumerate() {
            *b = i as u8;
        }
        let out = flip(&src, RawVideoFormat::Nv12, (4, 2), FlipMethod::Rotate90Cw);
        assert_eq!(out.len(), 2 * 4 * 3 / 2);
        // luma plane is the 4x2 [0..8] rotated to 2x4: first output row is the
        // input's left column bottom-to-top -> src[4], src[0].
        assert_eq!(&out[0..2], &[4, 0]);
    }

    #[test]
    fn derived_output_swaps_dims_for_rotation() {
        let flip = VideoFlip::new(FlipMethod::Rotate90Cw);
        let CapsConstraint::DerivedOutput(f) = flip.caps_constraint_as_transform() else {
            panic!("expected DerivedOutput");
        };
        let out = f(&Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(320),
            height: Dim::Fixed(240),
            framerate: Rate::Fixed(30 << 16),
        });
        assert_eq!(
            out.alternatives(),
            &[Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Fixed(240),
                height: Dim::Fixed(320),
                framerate: Rate::Fixed(30 << 16),
            }]
        );
    }

    #[test]
    fn derived_output_preserves_dims_for_mirror_and_rejects_compressed() {
        let flip = VideoFlip::new(FlipMethod::HorizontalMirror);
        let CapsConstraint::DerivedOutput(f) = flip.caps_constraint_as_transform() else {
            panic!("expected DerivedOutput");
        };
        let out = f(&rgba_caps(320, 240));
        assert_eq!(
            out.alternatives(),
            &[Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Fixed(320),
                height: Dim::Fixed(240),
                framerate: Rate::Fixed(30 << 16),
            }]
        );
        let h264 = Caps::CompressedVideo {
            codec: g2g_core::VideoCodec::H264,
            width: Dim::Fixed(320),
            height: Dim::Fixed(240),
            framerate: Rate::Any,
        };
        assert!(f(&h264).is_empty());
    }

    #[test]
    fn configure_rejects_odd_yuv420_and_compressed() {
        // odd-width 4:2:0 fails.
        let mut f = VideoFlip::new(FlipMethod::Rotate90Cw);
        assert_eq!(
            f.configure_pipeline(&nv12_caps(5, 4)).expect_err("odd width for 4:2:0"),
            G2gError::CapsMismatch
        );
        // even 4:2:0 is accepted.
        let mut f = VideoFlip::new(FlipMethod::Rotate90Cw);
        assert!(f.configure_pipeline(&nv12_caps(4, 4)).is_ok());
        // packed RGBA at any dims is accepted.
        let mut f = VideoFlip::new(FlipMethod::Rotate180);
        assert!(f.configure_pipeline(&rgba_caps(5, 3)).is_ok());
    }
}
