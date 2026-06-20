//! Add borders (`videobox`). Surrounds a packed RGBA / BGRA frame with a solid
//! colour border on any side, enlarging the geometry (letterbox / pillarbox).
//! CPU-only `no_std`. Borders only; cropping is `videocrop`.
//!
//! `border-top` / `-bottom` / `-left` / `-right` are widths in pixels; `fill`
//! names the border colour. Output dims are the input plus the borders.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, MemoryDomain,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec, Rate, RawVideoFormat,
};

use crate::pixel::rgba_rb_offsets;

const FORMATS: [RawVideoFormat; 2] = [RawVideoFormat::Rgba8, RawVideoFormat::Bgra8];

/// A named border colour. Kept a small palette so it reads cleanly from text; a
/// packed-ARGB property is a follow-up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fill {
    Black,
    White,
    Red,
    Green,
    Blue,
    Transparent,
}

impl Fill {
    /// The (R, G, B, A) tuple for this colour.
    fn rgba(self) -> (u8, u8, u8, u8) {
        match self {
            Fill::Black => (0, 0, 0, 255),
            Fill::White => (255, 255, 255, 255),
            Fill::Red => (255, 0, 0, 255),
            Fill::Green => (0, 255, 0, 255),
            Fill::Blue => (0, 0, 255, 255),
            Fill::Transparent => (0, 0, 0, 0),
        }
    }
}

#[derive(Debug)]
pub struct VideoBox {
    top: u32,
    bottom: u32,
    left: u32,
    right: u32,
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
    /// No border (identity geometry); use the builder or properties to add one.
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

    /// Set all four border widths and the fill colour.
    pub fn with_borders(mut self, top: u32, bottom: u32, left: u32, right: u32, fill: Fill) -> Self {
        self.top = top;
        self.bottom = bottom;
        self.left = left;
        self.right = right;
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

    fn out_dims(&self, w: u32, h: u32) -> (u32, u32) {
        (w + self.left + self.right, h + self.top + self.bottom)
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

    /// Native `DerivedOutput`: the same format and framerate, geometry grown by
    /// the border widths on each axis.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let (lr, tb) = (self.left + self.right, self.top + self.bottom);
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match input {
            Caps::RawVideo { format, width, height, framerate } if FORMATS.contains(format) => {
                CapsSet::one(Caps::RawVideo {
                    format: *format,
                    width: grow(width, lr),
                    height: grow(height, tb),
                    framerate: framerate.clone(),
                })
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.input = Some(self.accept_input(absolute_caps)?);
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
                    let boxed = build_boxed(
                        format,
                        &src[..bytes],
                        in_w,
                        in_h,
                        self.left,
                        self.right,
                        self.top,
                        self.bottom,
                        self.fill,
                    );

                    let (out_w, out_h) = self.out_dims(in_w, in_h);
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
                    self.input = Some(self.accept_input(&c)?);
                }
                PipelinePacket::Flush => {
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                PipelinePacket::Eos => {}
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        VIDEOBOX_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "border-top" => self.top = value.as_uint().ok_or(PropError::Type)? as u32,
            "border-bottom" => self.bottom = value.as_uint().ok_or(PropError::Type)? as u32,
            "border-left" => self.left = value.as_uint().ok_or(PropError::Type)? as u32,
            "border-right" => self.right = value.as_uint().ok_or(PropError::Type)? as u32,
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
            "border-top" => Some(PropValue::Uint(self.top as u64)),
            "border-bottom" => Some(PropValue::Uint(self.bottom as u64)),
            "border-left" => Some(PropValue::Uint(self.left as u64)),
            "border-right" => Some(PropValue::Uint(self.right as u64)),
            "fill" => Some(PropValue::Str(fill_to_str(self.fill).into())),
            _ => None,
        }
    }
}

/// `VideoBox`'s settable properties (M104).
static VIDEOBOX_PROPS: &[PropertySpec] = &[
    PropertySpec::new("border-top", PropKind::Uint, "top border width in pixels"),
    PropertySpec::new("border-bottom", PropKind::Uint, "bottom border width in pixels"),
    PropertySpec::new("border-left", PropKind::Uint, "left border width in pixels"),
    PropertySpec::new("border-right", PropKind::Uint, "right border width in pixels"),
    PropertySpec::new("fill", PropKind::Str, "border colour: black|white|red|green|blue|transparent"),
];

fn fill_from_str(s: &str) -> Option<Fill> {
    match s {
        "black" => Some(Fill::Black),
        "white" => Some(Fill::White),
        "red" => Some(Fill::Red),
        "green" => Some(Fill::Green),
        "blue" => Some(Fill::Blue),
        "transparent" => Some(Fill::Transparent),
        _ => None,
    }
}

fn fill_to_str(f: Fill) -> &'static str {
    match f {
        Fill::Black => "black",
        Fill::White => "white",
        Fill::Red => "red",
        Fill::Green => "green",
        Fill::Blue => "blue",
        Fill::Transparent => "transparent",
    }
}

/// Grow a dimension by `n`, leaving an unfixed dimension open.
fn grow(d: &Dim, n: u32) -> Dim {
    match d {
        Dim::Fixed(v) => Dim::Fixed(v + n),
        Dim::Range { min, max } => Dim::Range { min: min + n, max: max + n },
        Dim::Any => Dim::Any,
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

/// Build the bordered frame: a fill-coloured canvas of the grown size with the
/// input blitted into the interior at `(left, top)`.
#[allow(clippy::too_many_arguments)]
fn build_boxed(
    format: RawVideoFormat,
    src: &[u8],
    in_w: u32,
    in_h: u32,
    left: u32,
    right: u32,
    top: u32,
    bottom: u32,
    fill: Fill,
) -> Box<[u8]> {
    let out_w = (in_w + left + right) as usize;
    let out_h = (in_h + top + bottom) as usize;
    let fc = fill_bytes(format, fill);
    let mut dst = vec![0u8; out_w * out_h * 4].into_boxed_slice();
    for px in dst.chunks_exact_mut(4) {
        px.copy_from_slice(&fc);
    }
    let iw = in_w as usize;
    let row_bytes = iw * 4;
    for y in 0..in_h as usize {
        let dst_off = ((top as usize + y) * out_w + left as usize) * 4;
        let src_off = y * row_bytes;
        dst[dst_off..dst_off + row_bytes].copy_from_slice(&src[src_off..src_off + row_bytes]);
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
    fn one_pixel_border_grows_and_centres() {
        // 1x1 white pixel, 1px border all sides -> 3x3 with white centre.
        let src = [255u8, 255, 255, 255];
        let out = build_boxed(RawVideoFormat::Rgba8, &src, 1, 1, 1, 1, 1, 1, Fill::Black);
        assert_eq!(out.len(), 3 * 3 * 4);
        // Centre pixel (1,1).
        let centre = (1 * 3 + 1) * 4;
        assert_eq!(&out[centre..centre + 4], &[255, 255, 255, 255]);
        // Top-left corner is the black border.
        assert_eq!(&out[0..4], &[0, 0, 0, 255]);
    }

    #[test]
    fn asymmetric_borders_grow_each_axis() {
        // 2x2 input, left=1 right=0 top=2 bottom=0 -> 3x4.
        let src = vec![0u8; 2 * 2 * 4];
        let out = build_boxed(RawVideoFormat::Rgba8, &src, 2, 2, 1, 0, 2, 0, Fill::Black);
        assert_eq!(out.len(), 3 * 4 * 4, "width 2+1, height 2+2");
    }

    #[test]
    fn fill_colour_respects_channel_order() {
        // Red in RGBA is [255,0,0,A]; in BGRA the red byte is index 2.
        assert_eq!(fill_bytes(RawVideoFormat::Rgba8, Fill::Red), [255, 0, 0, 255]);
        assert_eq!(fill_bytes(RawVideoFormat::Bgra8, Fill::Red), [0, 0, 255, 255]);
        assert_eq!(fill_bytes(RawVideoFormat::Rgba8, Fill::Transparent), [0, 0, 0, 0]);
    }

    #[test]
    fn derived_output_grows_fixed_dims() {
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
    }
}
