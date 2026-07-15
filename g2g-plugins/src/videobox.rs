//! Box a packed RGBA / BGRA frame (`videobox`). Follows GStreamer's `videobox`
//! model: `top` / `bottom` / `left` / `right` are signed pixel counts, a
//! positive value crops that edge, a negative value adds that many border
//! pixels of `fill` (letterbox / pillarbox). Output geometry is the input minus
//! the four edge values. CPU-only `no_std`; packed formats only.

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

use crate::pixel::rgba_rb_offsets;

const FORMATS: [RawVideoFormat; 2] = [RawVideoFormat::Rgba8, RawVideoFormat::Bgra8];

/// A named border colour. Matches GStreamer's `GstVideoBoxFill`
/// (black/green/blue/red/yellow/white); `transparent` is a g2g extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fill {
    Black,
    Green,
    Blue,
    Red,
    Yellow,
    White,
    Transparent,
}

impl Fill {
    /// The (R, G, B, A) tuple for this colour.
    fn rgba(self) -> (u8, u8, u8, u8) {
        match self {
            Fill::Black => (0, 0, 0, 255),
            Fill::Green => (0, 255, 0, 255),
            Fill::Blue => (0, 0, 255, 255),
            Fill::Red => (255, 0, 0, 255),
            Fill::Yellow => (255, 255, 0, 255),
            Fill::White => (255, 255, 255, 255),
            Fill::Transparent => (0, 0, 0, 0),
        }
    }
}

#[derive(Debug)]
pub struct VideoBox {
    /// Signed edge counts (GStreamer `videobox` model): positive crops that
    /// edge, negative adds a border of `fill`. Output is the input minus
    /// `left + right` by `top + bottom`.
    top: i32,
    bottom: i32,
    left: i32,
    right: i32,
    fill: Fill,
    input: Option<(RawVideoFormat, u32, u32, Rate)>,
    configured: bool,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl Default for VideoBox {
    fn default() -> Self {
        Self::new()
    }
}

impl VideoBox {
    /// Identity geometry; use the builder or properties to crop or border.
    pub fn new() -> Self {
        Self {
            top: 0,
            bottom: 0,
            left: 0,
            right: 0,
            fill: Fill::Black,
            input: None,
            configured: false,
            last_caps: None,
            emitted: 0,
        }
    }

    /// Add a border of the given pixel widths on each edge (a convenience over
    /// the signed model: border widths are negative `videobox` edge values).
    pub fn with_borders(mut self, top: u32, bottom: u32, left: u32, right: u32, fill: Fill) -> Self {
        self.top = -(top as i32);
        self.bottom = -(bottom as i32);
        self.left = -(left as i32);
        self.right = -(right as i32);
        self.fill = fill;
        self
    }

    fn accept_input(&self, caps: &Caps) -> Result<(RawVideoFormat, u32, u32, Rate), G2gError> {
        let Caps::RawVideo { format, width: Dim::Fixed(w), height: Dim::Fixed(h), framerate } = caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if !FORMATS.contains(format) || *w == 0 || *h == 0 {
            return Err(G2gError::CapsMismatch);
        }
        Ok((*format, *w, *h, framerate.clone()))
    }

    /// Output geometry on an `in_w x in_h` frame, or `None` if the crop would
    /// consume the whole frame (a positive edge sum `>=` the extent).
    fn out_dims(&self, in_w: u32, in_h: u32) -> Option<(u32, u32)> {
        let w = in_w as i64 - self.left as i64 - self.right as i64;
        let h = in_h as i64 - self.top as i64 - self.bottom as i64;
        if w > 0 && h > 0 {
            Some((w as u32, h as u32))
        } else {
            None
        }
    }
}

impl AsyncElement for VideoBox {
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
            };
            if let Ok(narrowed) = upstream_caps.intersect(&candidate) {
                return Ok(narrowed);
            }
        }
        Err(G2gError::CapsMismatch)
    }

    /// Native `DerivedOutput`: same format and framerate, geometry adjusted by
    /// the signed edge counts (shrunk by a crop, grown by a border). A crop that
    /// consumes the whole frame collapses to the empty set.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let (lr, tb) = (self.left + self.right, self.top + self.bottom);
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match input {
            Caps::RawVideo { format, width, height, framerate } if FORMATS.contains(format) => {
                match (adjust(width, lr), adjust(height, tb)) {
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
        // Reject a crop that leaves nothing.
        self.out_dims(w, h).ok_or(G2gError::CapsMismatch)?;
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
                    let bytes = (in_w as usize) * (in_h as usize) * 4;
                    if src.len() < bytes {
                        return Err(G2gError::CapsMismatch);
                    }
                    let (out_w, out_h) = self.out_dims(in_w, in_h).ok_or(G2gError::CapsMismatch)?;
                    let boxed = build_boxed(
                        format,
                        &src[..bytes],
                        in_w,
                        in_h,
                        self.left,
                        self.top,
                        out_w,
                        out_h,
                        self.fill,
                    );

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
                    let out_frame = Frame {
                        domain: MemoryDomain::System(SystemSlice::from_boxed(boxed)),
                        timing: frame.timing,
                        sequence: self.emitted,
                        meta: Default::default(),
                    };
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(out_frame)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    // `c` is the runner arm's forward *output* caps (it already
                    // called configure_pipeline for our input). Forward it and
                    // record last_caps to suppress the data path's duplicate
                    // emit; do NOT accept_input, which would clobber the input
                    // with our own (boxed) output and corrupt the next frame.
                    out.push(PipelinePacket::CapsChanged(c.clone())).await?;
                    self.last_caps = Some(c);
                }
                PipelinePacket::Flush => {
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        VIDEOBOX_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Video box",
            "Filter/Effect/Video",
            "Crops or borders raw video (positive edge crops, negative adds a border)",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            // Canonical GStreamer `videobox` props: signed (>0 crop, <0 border).
            "top" => self.top = value.as_int().ok_or(PropError::Type)? as i32,
            "bottom" => self.bottom = value.as_int().ok_or(PropError::Type)? as i32,
            "left" => self.left = value.as_int().ok_or(PropError::Type)? as i32,
            "right" => self.right = value.as_int().ok_or(PropError::Type)? as i32,
            "fill" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.fill = fill_from_str(s).ok_or(PropError::Value)?;
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "top" => Some(PropValue::Int(self.top as i64)),
            "bottom" => Some(PropValue::Int(self.bottom as i64)),
            "left" => Some(PropValue::Int(self.left as i64)),
            "right" => Some(PropValue::Int(self.right as i64)),
            "fill" => Some(PropValue::Str(fill_to_str(self.fill).into())),
            _ => None,
        }
    }
}

/// `VideoBox`'s settable properties (GStreamer `videobox` model, M183).
static VIDEOBOX_PROPS: &[PropertySpec] = &[
    PropertySpec::new("top", PropKind::Int, "top edge: >0 crops, <0 borders").with_default("0"),
    PropertySpec::new("bottom", PropKind::Int, "bottom edge: >0 crops, <0 borders").with_default("0"),
    PropertySpec::new("left", PropKind::Int, "left edge: >0 crops, <0 borders").with_default("0"),
    PropertySpec::new("right", PropKind::Int, "right edge: >0 crops, <0 borders").with_default("0"),
    PropertySpec::new("fill", PropKind::Str, "border colour: black|green|blue|red|yellow|white|transparent"),
];

fn fill_from_str(s: &str) -> Option<Fill> {
    match s {
        "black" => Some(Fill::Black),
        "green" => Some(Fill::Green),
        "blue" => Some(Fill::Blue),
        "red" => Some(Fill::Red),
        "yellow" => Some(Fill::Yellow),
        "white" => Some(Fill::White),
        "transparent" => Some(Fill::Transparent),
        _ => None,
    }
}

fn fill_to_str(f: Fill) -> &'static str {
    match f {
        Fill::Black => "black",
        Fill::Green => "green",
        Fill::Blue => "blue",
        Fill::Red => "red",
        Fill::Yellow => "yellow",
        Fill::White => "white",
        Fill::Transparent => "transparent",
    }
}

/// Adjust a dimension by subtracting the signed edge sum `delta` (a positive
/// crop shrinks, a negative border grows). `None` if a crop would leave nothing.
fn adjust(d: &Dim, delta: i32) -> Option<Dim> {
    match d {
        Dim::Fixed(v) => {
            let r = *v as i64 - delta as i64;
            (r > 0).then_some(Dim::Fixed(r as u32))
        }
        Dim::Range { min, max } => {
            let max = *max as i64 - delta as i64;
            if max <= 0 {
                return None;
            }
            let min = (*min as i64 - delta as i64).max(1).min(max);
            Some(Dim::Range { min: min as u32, max: max as u32 })
        }
        Dim::Any => Some(Dim::Any),
    }
}

/// The fill colour as bytes in `format`'s channel order.
fn fill_bytes(format: RawVideoFormat, fill: Fill) -> [u8; 4] {
    let (r, g, b, a) = fill.rgba();
    let (r_idx, b_idx) = rgba_rb_offsets(format);
    let mut px = [0u8; 4];
    px[r_idx] = r;
    px[1] = g;
    px[b_idx] = b;
    px[3] = a;
    px
}

/// Build the boxed frame of size `out_w x out_h`: a fill-coloured canvas onto
/// which the input is mapped by `out(ox, oy) <- src(ox + left, oy + top)`, with
/// out-of-bounds output pixels left as the border fill. A single mapping handles
/// both crop (positive `left`/`top` skip input pixels) and border (negative
/// `left`/`top` leave a fill margin).
#[allow(clippy::too_many_arguments)]
fn build_boxed(
    format: RawVideoFormat,
    src: &[u8],
    in_w: u32,
    in_h: u32,
    left: i32,
    top: i32,
    out_w: u32,
    out_h: u32,
    fill: Fill,
) -> Box<[u8]> {
    let (in_w, in_h, out_w, out_h) = (in_w as i64, in_h as i64, out_w as usize, out_h as usize);
    let (left, top) = (left as i64, top as i64);
    let fc = fill_bytes(format, fill);
    let mut dst = vec![0u8; out_w * out_h * 4].into_boxed_slice();
    for px in dst.chunks_exact_mut(4) {
        px.copy_from_slice(&fc);
    }
    for oy in 0..out_h {
        let sy = oy as i64 + top;
        if sy < 0 || sy >= in_h {
            continue;
        }
        // Output columns whose source x lands inside [0, in_w): copy that span.
        let ox0 = (-left).max(0);
        let ox1 = (in_w - left).min(out_w as i64).max(ox0);
        if ox1 <= ox0 {
            continue;
        }
        let span = (ox1 - ox0) as usize;
        let sx0 = (ox0 + left) as usize;
        let src_off = ((sy as usize) * (in_w as usize) + sx0) * 4;
        let dst_off = (oy * out_w + ox0 as usize) * 4;
        dst[dst_off..dst_off + span * 4].copy_from_slice(&src[src_off..src_off + span * 4]);
    }
    dst
}

impl PadTemplates for VideoBox {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    // Pixel offsets below are written as (row * width + col) * channels; keep
    // the unit factors (e.g. row 1 * width) for legibility.
    #[allow(clippy::identity_op)]
    fn one_pixel_border_grows_and_centres() {
        // 1x1 white pixel, 1px border all sides -> 3x3 with white centre.
        // Border = negative edges; out = 1 - (-1) - (-1) = 3.
        let src = [255u8, 255, 255, 255];
        let out = build_boxed(RawVideoFormat::Rgba8, &src, 1, 1, -1, -1, 3, 3, Fill::Black);
        assert_eq!(out.len(), 3 * 3 * 4);
        // Centre pixel (1,1) is the source pixel.
        let centre = (1 * 3 + 1) * 4;
        assert_eq!(&out[centre..centre + 4], &[255, 255, 255, 255]);
        // Top-left corner is the black border.
        assert_eq!(&out[0..4], &[0, 0, 0, 255]);
    }

    #[test]
    fn positive_edges_crop() {
        // 4x4 RGBA, pixel p = [4p..]; crop left=1 top=1 -> out 3x3, out(0,0)=src(1,1).
        let src: Vec<u8> = (0..(4 * 4 * 4) as u8).collect();
        let out = build_boxed(RawVideoFormat::Rgba8, &src, 4, 4, 1, 1, 3, 3, Fill::Black);
        assert_eq!(out.len(), 3 * 3 * 4);
        // src pixel (1,1) = index 5 -> byte 20.
        assert_eq!(&out[0..4], &[20, 21, 22, 23]);
    }

    #[test]
    fn asymmetric_border_grows_each_axis() {
        // 2x2 input, border left=1 top=2 (others 0) -> out 3x4.
        let src = vec![0u8; 2 * 2 * 4];
        let vb = VideoBox::new().with_borders(2, 0, 1, 0, Fill::Black);
        let (ow, oh) = vb.out_dims(2, 2).unwrap();
        assert_eq!((ow, oh), (3, 4), "width 2+1, height 2+2");
        let out = build_boxed(RawVideoFormat::Rgba8, &src, 2, 2, vb.left, vb.top, ow, oh, Fill::Black);
        assert_eq!(out.len(), 3 * 4 * 4);
    }

    #[test]
    fn fill_colour_respects_channel_order() {
        // Red in RGBA is [255,0,0,A]; in BGRA the red byte is index 2.
        assert_eq!(fill_bytes(RawVideoFormat::Rgba8, Fill::Red), [255, 0, 0, 255]);
        assert_eq!(fill_bytes(RawVideoFormat::Bgra8, Fill::Red), [0, 0, 255, 255]);
        assert_eq!(fill_bytes(RawVideoFormat::Rgba8, Fill::Yellow), [255, 255, 0, 255]);
        assert_eq!(fill_bytes(RawVideoFormat::Rgba8, Fill::Transparent), [0, 0, 0, 0]);
    }

    #[test]
    fn signed_edges_round_trip() {
        let mut vb = VideoBox::new();
        // Positive crops, negative borders (gst videobox semantics).
        vb.set_property("left", PropValue::Int(8)).unwrap();
        assert_eq!(vb.get_property("left"), Some(PropValue::Int(8)));
        vb.set_property("top", PropValue::Int(-8)).unwrap();
        assert_eq!(vb.get_property("top"), Some(PropValue::Int(-8)));
    }

    #[test]
    fn derived_output_grows_on_border_and_shrinks_on_crop() {
        // Border 2/2/4/4 grows 320x240 -> 328x244.
        let vb = VideoBox::new().with_borders(2, 2, 4, 4, Fill::Black);
        let CapsConstraint::DerivedOutput(f) = vb.caps_constraint_as_transform() else {
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
                width: Dim::Fixed(328),  // 320 + 4 + 4
                height: Dim::Fixed(244), // 240 + 2 + 2
                framerate: Rate::Fixed(30 << 16),
            }]
        );

        // Crop left=right=160 on 320 wide leaves nothing -> empty.
        let mut vb = VideoBox::new();
        vb.set_property("left", PropValue::Int(160)).unwrap();
        vb.set_property("right", PropValue::Int(160)).unwrap();
        let CapsConstraint::DerivedOutput(f) = vb.caps_constraint_as_transform() else {
            panic!("expected DerivedOutput");
        };
        assert!(
            f(&Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Fixed(320),
                height: Dim::Fixed(240),
                framerate: Rate::Any,
            })
            .is_empty(),
            "a crop consuming the whole width yields no caps"
        );
    }
}
