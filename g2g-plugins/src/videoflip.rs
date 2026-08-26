//! Software flip / rotate (Tier-1 A). Mirrors or rotates a raw video frame by
//! a fixed `Orientation`, preserving the pixel format, for portrait-mode mobile
//! sources fed to a landscape pipeline. No resampling, a per-plane coordinate
//! remap.
//!
//! The quarter rotations and the two diagonal mirrors swap width and height;
//! `Rotate180` and the axis mirrors keep the geometry. 4:2:0 (`Nv12`, `I420`)
//! needs even input dims since chroma is subsampled 2x2; odd dims fail
//! negotiation/configure loud. Packed formats (`Rgba8`, `Bgra8`) take any dims.
//! CPU-only `no_std` baseline.
//!
//! When the sink downstream answers the first push with
//! `Reconfigure::AbsorbOrientation` (M1058), the frame goes through untouched
//! with an `OrientationMeta` on it instead and the output caps stay the input
//! caps: the sink turns the picture at present time and the pixel work
//! disappears.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use crate::pixel::{even_dims_required, frame_byte_size, planar_planes};
use alloc::string::String;
use g2g_core::frame::Frame;
use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::memory::{SystemSlice, SystemView};
pub use g2g_core::meta::Orientation;
use g2g_core::tensor::TensorView;
use g2g_core::{g2g_info, g2g_trace};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec, PushOutcome, Rate, RawVideoFormat, Reconfigure,
};

const FORMATS: [RawVideoFormat; 12] = [
    RawVideoFormat::Rgba8,
    RawVideoFormat::Bgra8,
    RawVideoFormat::Nv12,
    RawVideoFormat::I420,
    RawVideoFormat::I420p10,
    RawVideoFormat::I420p12,
    RawVideoFormat::I422,
    RawVideoFormat::I422p10,
    RawVideoFormat::I422p12,
    RawVideoFormat::I444,
    RawVideoFormat::I444p10,
    RawVideoFormat::I444p12,
];

/// # Example
///
/// ```no_run
/// use g2g_plugins::videoflip::{Orientation, VideoFlip};
///
/// let flip = VideoFlip::new(Orientation::Rotate90Cw);
/// ```
#[derive(Debug)]
pub struct VideoFlip {
    method: Orientation,
    /// Format, dims, and framerate of the configured input stream, updated by
    /// a mid-stream `CapsChanged`.
    input: Option<(RawVideoFormat, u32, u32, Rate)>,
    configured: bool,
    last_caps: Option<Caps>,
    emitted: u64,
    /// Set once downstream answered a push with `Reconfigure::AbsorbOrientation`
    /// (M1058): frames then pass through with an `OrientationMeta` attached and
    /// the output caps are the input caps.
    descriptor_mode: bool,
    /// Instance name assigned by the runner (M179), for this element's log lines.
    log_name: LogName,
}

impl VideoFlip {
    pub fn new(method: Orientation) -> Self {
        Self {
            method,
            input: None,
            configured: false,
            last_caps: None,
            emitted: 0,
            descriptor_mode: false,
            log_name: LogName::new(),
        }
    }

    pub fn method(&self) -> Orientation {
        self.method
    }

    fn accept_input(&self, caps: &Caps) -> Result<(RawVideoFormat, u32, u32, Rate), G2gError> {
        let Caps::RawVideo {
            format,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate,
            interlace: _,
        } = caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if !FORMATS.contains(format) || *w == 0 || *h == 0 {
            return Err(G2gError::CapsMismatch);
        }
        let (ew, eh) = even_dims_required(*format);
        if (ew && *w % 2 != 0) || (eh && *h % 2 != 0) {
            return Err(G2gError::CapsMismatch);
        }
        // A 90-degree rotation transposes the frame, swapping the horizontal and
        // vertical chroma subsampling. That is fine for symmetric subsampling
        // (4:2:0, 4:4:4) but a transposed 4:2:2 is not a representable I422 layout,
        // so reject an asymmetric format under a dimension-swapping method.
        if self.method.swaps_dims() {
            if let Some((hs, vs)) = format.chroma_shift() {
                if hs != vs {
                    return Err(G2gError::CapsMismatch);
                }
            }
        }
        Ok((*format, *w, *h, framerate.clone()))
    }

    /// Output geometry for the configured method: the quarter rotations and the
    /// diagonal mirrors transpose, the axis mirrors and 180 preserve. In
    /// descriptor mode nothing is remapped, so the geometry is the input's.
    fn output_dims(&self, w: u32, h: u32) -> (u32, u32) {
        if self.method.swaps_dims() && !self.descriptor_mode {
            (h, w)
        } else {
            (w, h)
        }
    }

    /// The caps this element emits for an input stream, under the mode it is in.
    fn output_caps(&self, format: RawVideoFormat, w: u32, h: u32, rate: &Rate) -> Caps {
        let (out_w, out_h) = self.output_dims(w, h);
        Caps::RawVideo {
            format,
            width: Dim::Fixed(out_w),
            height: Dim::Fixed(out_h),
            framerate: rate.clone(),
            interlace: g2g_core::Interlace::Any,
        }
    }

    /// The descriptor-mode output for `frame`: the very same buffer, in the
    /// same memory domain, with this element's turn recorded on it. An
    /// orientation already on the input composes with ours, so two flips in a
    /// row reach the sink as the one turn they add up to.
    fn describe(&self, frame: Frame) -> Frame {
        #[cfg_attr(not(feature = "metadata"), allow(unused_mut))]
        let mut meta = frame.meta;
        #[cfg(feature = "metadata")]
        if self.method != Orientation::Identity {
            use g2g_core::meta::OrientationMeta;
            let carried = meta
                .get::<OrientationMeta>()
                .map_or(Orientation::Identity, |m| m.orientation);
            meta.attach(OrientationMeta {
                orientation: carried.compose(self.method),
            });
        }
        Frame {
            domain: frame.domain,
            timing: frame.timing,
            sequence: self.emitted,
            meta,
        }
    }

    /// The eager output for `frame`: the pixels remapped by `method`.
    ///
    /// Packed RGBA/BGRA already in shared CPU memory is the zero-copy case: a
    /// flip is a pure coordinate remap, so we compose strides on the *same*
    /// `Arc` backing and copy nothing. Planar (4:2:0) is excluded because its
    /// subsampled planes aren't one strided tensor (see tensor.rs), and an owned
    /// `System` buffer has no shared backing to alias, so both fall through to
    /// the copy path.
    fn flip_frame(
        &self,
        frame: &Frame,
        format: RawVideoFormat,
        in_w: u32,
        in_h: u32,
    ) -> Result<Frame, G2gError> {
        let (out_w, out_h) = self.output_dims(in_w, in_h);
        let packed = matches!(format, RawVideoFormat::Rgba8 | RawVideoFormat::Bgra8);
        let out_frame = match &frame.domain {
            MemoryDomain::SystemView(sv) if packed => {
                let out_view = flip_view(*sv.view(), self.method);
                g2g_trace!(
                    self,
                    "zero-copy flip frame #{} {}x{} -> {}x{}",
                    self.emitted,
                    in_w,
                    in_h,
                    out_w,
                    out_h
                );
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
                g2g_trace!(
                    self,
                    "flip frame #{} {}x{} -> {}x{}",
                    self.emitted,
                    in_w,
                    in_h,
                    out_w,
                    out_h
                );
                Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(flipped)),
                    timing: frame.timing,
                    sequence: self.emitted,
                    meta: Default::default(),
                }
            }
        };
        Ok(out_frame)
    }

    /// Answer a downstream `AbsorbOrientation` seen on `outcome`: stop remapping
    /// pixels, let the turn ride along as metadata instead, and re-announce the
    /// output shape, which is now the input's. Returns false when `outcome` was
    /// anything else.
    ///
    /// The pre-send check holds the packet that carried the advertisement back
    /// rather than enqueuing it, so the caller has to send that packet again.
    async fn absorb(
        &mut self,
        out: &mut dyn OutputSink,
        outcome: PushOutcome,
    ) -> Result<bool, G2gError> {
        if !matches!(
            outcome,
            PushOutcome::Reconfigure(Reconfigure::AbsorbOrientation)
        ) {
            return Ok(false);
        }
        if !self.descriptor_mode {
            self.descriptor_mode = true;
            g2g_info!(
                self,
                "downstream absorbs orientation: attaching {:?} instead of rotating",
                self.method
            );
        }
        // The shape announced before the switch was the rotated one, which no
        // frame will now carry: a display sink configured on it would open a
        // window at the wrong size and rebuild it on the first frame.
        if let Some((format, w, h, rate)) = self.input.clone() {
            let caps = self.output_caps(format, w, h, &rate);
            if self.last_caps.as_ref() != Some(&caps) {
                out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
            }
            self.last_caps = Some(caps);
        }
        Ok(true)
    }
}

impl AsyncElement for VideoFlip {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        for format in FORMATS {
            let candidate = Caps::RawVideo {
                format,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
                interlace: g2g_core::Interlace::Any,
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
            Caps::RawVideo {
                format,
                width,
                height,
                framerate,
                interlace: _,
            } if FORMATS.contains(format) => {
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
                    interlace: g2g_core::Interlace::Any,
                })
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (format, w, h, rate) = self.accept_input(absolute_caps)?;
        self.input = Some((format, w, h, rate));
        self.configured = true;
        g2g_info!(
            self,
            "configured {:?} {}x{} {:?}",
            format,
            w,
            h,
            self.method
        );
        Ok(ConfigureOutcome::Accepted)
    }

    fn set_instance_name(&mut self, name: String) {
        self.log_name.set_instance(name);
    }

    fn set_log_category(&mut self, category: String) {
        self.log_name.set_category(category);
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

                    // Announce the shape for the mode we are in. A sink's
                    // `AbsorbOrientation` lands on the first push this element
                    // makes, and the pre-send check holds that packet back, so
                    // the announcement is re-made under the mode it switched us
                    // to and no frame is ever rotated first.
                    let new_caps = self.output_caps(format, in_w, in_h, &rate);
                    if self.last_caps.as_ref() != Some(&new_caps) {
                        let outcome = out
                            .push(PipelinePacket::CapsChanged(new_caps.clone()))
                            .await?;
                        if !self.absorb(out, outcome).await? {
                            self.last_caps = Some(new_caps);
                        }
                    }

                    if !self.descriptor_mode {
                        let out_frame = self.flip_frame(&frame, format, in_w, in_h)?;
                        let outcome = out.push(PipelinePacket::DataFrame(out_frame)).await?;
                        if !self.absorb(out, outcome).await? {
                            self.emitted += 1;
                            return Ok(());
                        }
                        // The remapped frame was held back, so `frame` is still
                        // the one downstream is owed: send it as a descriptor.
                    }
                    let out_frame = self.describe(frame);
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(out_frame)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    // `c` is the runner arm's forward *output* caps (it already
                    // called configure_pipeline for our input). Forward it and
                    // record last_caps to suppress the data path's duplicate
                    // emit; do NOT accept_input, which would clobber the input
                    // with our own (rotated) output and corrupt the next frame.
                    // A sink that absorbs the orientation answers here, and
                    // `absorb` re-announces the corrected shape in place of the
                    // packet the pre-send check held back.
                    let outcome = out.push(PipelinePacket::CapsChanged(c.clone())).await?;
                    if !self.absorb(out, outcome).await? {
                        self.last_caps = Some(c);
                    }
                }
                PipelinePacket::Flush => {
                    self.last_caps = None;
                    let outcome = out.push(PipelinePacket::Flush).await?;
                    if self.absorb(out, outcome).await? {
                        out.push(PipelinePacket::Flush).await?;
                    }
                }
                // Segment is control: forward unchanged.
                PipelinePacket::Segment(seg) => {
                    let outcome = out.push(PipelinePacket::Segment(seg)).await?;
                    if self.absorb(out, outcome).await? {
                        out.push(PipelinePacket::Segment(seg)).await?;
                    }
                }
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    /// This is the element a sink's `AbsorbOrientation` is aimed at, so the
    /// advertisement stops here rather than travelling on toward the source.
    /// Without the `metadata` feature there is no meta set to record the turn
    /// on, so the advertisement passes through and the flip stays eager.
    fn handles_orientation(&self) -> bool {
        cfg!(feature = "metadata")
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
    "none | identity | clockwise | rotate-90cw | rotate-180 | counterclockwise | rotate-90ccw \
     | horizontal-flip | horizontal-mirror | vertical-flip | vertical-mirror \
     | upper-left-diagonal | transpose | upper-right-diagonal | transverse",
)
.with_default("none")];

/// Parse a `method` property string to an [`Orientation`]. Canonical names are
/// GStreamer's `videoflip` nicknames; the historical g2g spellings are accepted
/// as aliases so both port. GStreamer's `automatic` (take the turn from the
/// image's own tag) has no g2g equivalent and returns `None` (a loud
/// "unsupported method" rather than silent wrong behavior).
fn flip_method_from_str(s: &str) -> Option<Orientation> {
    match s {
        "none" | "identity" => Some(Orientation::Identity),
        "horizontal-flip" | "horizontal-mirror" => Some(Orientation::HorizontalMirror),
        "vertical-flip" | "vertical-mirror" => Some(Orientation::VerticalMirror),
        "clockwise" | "rotate-90cw" => Some(Orientation::Rotate90Cw),
        "rotate-180" => Some(Orientation::Rotate180),
        "counterclockwise" | "rotate-90ccw" => Some(Orientation::Rotate90Ccw),
        "upper-left-diagonal" | "transpose" => Some(Orientation::Transpose),
        "upper-right-diagonal" | "transverse" => Some(Orientation::Transverse),
        _ => None,
    }
}

/// The canonical (GStreamer) `method` property string for an [`Orientation`].
fn flip_method_to_str(m: Orientation) -> &'static str {
    match m {
        Orientation::Identity => "none",
        Orientation::HorizontalMirror => "horizontal-flip",
        Orientation::VerticalMirror => "vertical-flip",
        Orientation::Rotate90Cw => "clockwise",
        Orientation::Rotate180 => "rotate-180",
        Orientation::Rotate90Ccw => "counterclockwise",
        Orientation::Transpose => "upper-left-diagonal",
        Orientation::Transverse => "upper-right-diagonal",
    }
}

impl PadTemplates for VideoFlip {
    fn pad_templates() -> Vec<PadTemplate> {
        let any_geometry = |format| Caps::RawVideo {
            format,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
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
        self.log_name.instance()
    }
    fn log_category_override(&self) -> Option<&str> {
        self.log_name.category()
    }
}

/// The zero-copy analog of [`flip`] for a packed `[H, W, C]` view: express the
/// method as stride manipulations over the same bytes (M180). A mirror reverses
/// one spatial axis; a 90-degree rotation transposes the H/W axes then reverses
/// one. The channel axis (2) is never touched, so each pixel's bytes stay
/// intact. Matches [`src_coord`]'s mapping exactly, verified by the m180 test.
fn flip_view(view: TensorView, method: Orientation) -> TensorView {
    match method {
        Orientation::Identity => view,
        Orientation::HorizontalMirror => view.reversed_axis(1),
        Orientation::VerticalMirror => view.reversed_axis(0),
        Orientation::Rotate180 => view.reversed_axis(0).reversed_axis(1),
        Orientation::Rotate90Cw => view.transposed(0, 1).reversed_axis(1),
        Orientation::Rotate90Ccw => view.transposed(0, 1).reversed_axis(0),
        Orientation::Transpose => view.transposed(0, 1),
        Orientation::Transverse => view.transposed(0, 1).reversed_axis(0).reversed_axis(1),
    }
}

/// Source coordinate that feeds output `(ox, oy)` for one plane of input dims
/// `(pw, ph)`. The 90-degree rotations read from a transposed position; the
/// mirrors and 180 reflect within the same dims.
fn src_coord(method: Orientation, ox: usize, oy: usize, pw: usize, ph: usize) -> (usize, usize) {
    match method {
        Orientation::Identity => (ox, oy),
        Orientation::HorizontalMirror => (pw - 1 - ox, oy),
        Orientation::VerticalMirror => (ox, ph - 1 - oy),
        Orientation::Rotate180 => (pw - 1 - ox, ph - 1 - oy),
        Orientation::Rotate90Cw => (oy, ph - 1 - ox),
        Orientation::Rotate90Ccw => (pw - 1 - oy, ox),
        Orientation::Transpose => (oy, ox),
        Orientation::Transverse => (pw - 1 - oy, ph - 1 - ox),
    }
}

/// Remap one `channels`-interleaved plane of `pw x ph` by `method`. NV12's UV
/// plane uses `channels = 2` so each chroma pair moves as a unit.
fn transform_plane(
    src: &[u8],
    pw: usize,
    ph: usize,
    channels: usize,
    method: Orientation,
) -> Vec<u8> {
    let (ow, oh) = if method.swaps_dims() {
        (ph, pw)
    } else {
        (pw, ph)
    };
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
    method: Orientation,
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
        // YUYV is input-only / not produced here; negotiation never admits it.
        RawVideoFormat::Yuyv => unreachable!("videoflip: YUYV is not negotiated"),
        // The fully-planar family (I420 / I422 / I444 at 8 / 10 / 12-bit): flip each
        // of the three planes, treating a sample as an opaque `bytes_per_sample`-wide
        // unit (a flip / rotation moves samples, never blends, so it is depth
        // agnostic). A dimension-swapping method on an asymmetric (4:2:2) format was
        // rejected at negotiation, so each plane transposes to a valid output plane.
        f => {
            let bps = f.bytes_per_sample();
            let planes = planar_planes(f, in_w, in_h);
            let mut out = transform_plane(src, in_w, in_h, bps, method);
            for (off, pw, ph) in [planes[1], planes[2]] {
                let plane = transform_plane(&src[off..off + pw * ph * bps], pw, ph, bps, method);
                out.extend_from_slice(&plane);
            }
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
            interlace: g2g_core::Interlace::Any,
        }
    }

    fn nv12_caps(w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
        }
    }

    #[test]
    fn transform_plane_square_methods() {
        // 2x2 single-channel plane: row0 [0,1], row1 [2,3].
        let src = vec![0u8, 1, 2, 3];
        assert_eq!(
            transform_plane(&src, 2, 2, 1, Orientation::HorizontalMirror),
            vec![1, 0, 3, 2]
        );
        assert_eq!(
            transform_plane(&src, 2, 2, 1, Orientation::VerticalMirror),
            vec![2, 3, 0, 1]
        );
        assert_eq!(
            transform_plane(&src, 2, 2, 1, Orientation::Rotate180),
            vec![3, 2, 1, 0]
        );
        assert_eq!(
            transform_plane(&src, 2, 2, 1, Orientation::Rotate90Cw),
            vec![2, 0, 3, 1]
        );
        assert_eq!(
            transform_plane(&src, 2, 2, 1, Orientation::Rotate90Ccw),
            vec![1, 3, 0, 2]
        );
    }

    #[test]
    fn transform_plane_rotate90_swaps_dims() {
        // 3x2 plane: row0 [0,1,2], row1 [3,4,5]. Rotate 90 CW -> 2x3.
        let src: Vec<u8> = (0..6).collect();
        let out = transform_plane(&src, 3, 2, 1, Orientation::Rotate90Cw);
        assert_eq!(out, vec![3, 0, 4, 1, 5, 2]);
    }

    #[test]
    fn flip_rgba_mirrors_pixels() {
        // 2x2 RGBA where pixel p = [4p, 4p+1, 4p+2, 4p+3].
        let src: Vec<u8> = (0..(2 * 2 * 4) as u8).collect();
        let out = flip(
            &src,
            RawVideoFormat::Rgba8,
            (2, 2),
            Orientation::HorizontalMirror,
        );
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
        let out = flip(&src, RawVideoFormat::Nv12, (4, 2), Orientation::Rotate90Cw);
        assert_eq!(out.len(), 2 * 4 * 3 / 2);
        // luma plane is the 4x2 [0..8] rotated to 2x4: first output row is the
        // input's left column bottom-to-top -> src[4], src[0].
        assert_eq!(&out[0..2], &[4, 0]);
    }

    #[test]
    fn derived_output_swaps_dims_for_rotation() {
        let flip = VideoFlip::new(Orientation::Rotate90Cw);
        let CapsConstraint::DerivedOutput(f) = flip.caps_constraint_as_transform() else {
            panic!("expected DerivedOutput");
        };
        let out = f(&Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(320),
            height: Dim::Fixed(240),
            framerate: Rate::Fixed(30 << 16),
            interlace: g2g_core::Interlace::Any,
        });
        assert_eq!(
            out.alternatives(),
            &[Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Fixed(240),
                height: Dim::Fixed(320),
                framerate: Rate::Fixed(30 << 16),
                interlace: g2g_core::Interlace::Any,
            }]
        );
    }

    #[test]
    fn derived_output_preserves_dims_for_mirror_and_rejects_compressed() {
        let flip = VideoFlip::new(Orientation::HorizontalMirror);
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
                interlace: g2g_core::Interlace::Any,
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
        let mut f = VideoFlip::new(Orientation::Rotate90Cw);
        assert_eq!(
            f.configure_pipeline(&nv12_caps(5, 4))
                .expect_err("odd width for 4:2:0"),
            G2gError::CapsMismatch
        );
        // even 4:2:0 is accepted.
        let mut f = VideoFlip::new(Orientation::Rotate90Cw);
        assert!(f.configure_pipeline(&nv12_caps(4, 4)).is_ok());
        // packed RGBA at any dims is accepted.
        let mut f = VideoFlip::new(Orientation::Rotate180);
        assert!(f.configure_pipeline(&rgba_caps(5, 3)).is_ok());
    }

    fn planar_caps(format: RawVideoFormat, w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
        }
    }

    #[test]
    fn flips_high_bit_depth_4_4_4() {
        // 2x2 I444p10: three full-res LE-u16 planes. Horizontal mirror swaps columns.
        let mk =
            |base: u16| -> Vec<u8> { (0..4u16).flat_map(|s| (base + s).to_le_bytes()).collect() };
        let mut src = mk(0);
        src.extend(mk(100));
        src.extend(mk(200));
        let out = flip(
            &src,
            RawVideoFormat::I444p10,
            (2, 2),
            Orientation::HorizontalMirror,
        );
        assert_eq!(out.len(), 3 * 2 * 2 * 2);
        let rd = |o: usize| u16::from_le_bytes([out[o], out[o + 1]]);
        // Y row0 was [0,1] -> mirrored [1,0]; U row0 [100,101] -> [101,100].
        assert_eq!((rd(0), rd(2)), (1, 0));
        assert_eq!((rd(8), rd(10)), (101, 100));
    }

    #[test]
    fn rotate90_rejects_4_2_2_but_accepts_4_4_4() {
        // A 90-degree rotation transposes chroma subsampling: invalid for 4:2:2
        // (not a representable I422), fine for symmetric 4:4:4 / 4:2:0.
        let mut f = VideoFlip::new(Orientation::Rotate90Cw);
        assert_eq!(
            f.configure_pipeline(&planar_caps(RawVideoFormat::I422, 4, 4))
                .expect_err("4:2:2 transpose"),
            G2gError::CapsMismatch
        );
        let mut f = VideoFlip::new(Orientation::Rotate90Cw);
        assert!(f
            .configure_pipeline(&planar_caps(RawVideoFormat::I444p10, 4, 4))
            .is_ok());
        // A non-swapping method accepts 4:2:2 fine.
        let mut f = VideoFlip::new(Orientation::HorizontalMirror);
        assert!(f
            .configure_pipeline(&planar_caps(RawVideoFormat::I422, 4, 4))
            .is_ok());
    }
}
