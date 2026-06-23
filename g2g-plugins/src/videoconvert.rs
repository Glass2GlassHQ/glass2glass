//! Software raw-video format converter (M23). Converts between the packed
//! raw formats (`Rgba8`, `Bgra8`) and the 4:2:0 planar/semi-planar formats
//! (`Nv12`, `I420`) at the same geometry, so chains compose across format
//! boundaries: `VideoTestSrc (RGBA) -> VideoConvert -> MfEncode (NV12)`, or
//! `decoder (NV12) -> VideoConvert -> OrtInference (RGBA)`.
//!
//! Color math is integer-only BT.601 limited range (the convention the
//! display sinks use). Same-family conversions skip it: RGBA<->BGRA is a
//! channel swizzle and NV12<->I420 a chroma-plane repack, both lossless.
//! 4:2:0 formats require even dims (chroma is subsampled 2x2); odd dims
//! fail negotiation loud. CPU-only and `no_std`: this element lives in the
//! crate baseline.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    MemoryDomain, OutputSink, PadTemplate, PadTemplates, PassthroughFields, PipelinePacket,
    PropError, PropKind, PropValue, PropertySpec, Rate, RawVideoFormat,
};

/// Formats this element can both consume and produce. The convert `target`
/// is always one of these.
const FORMATS: [RawVideoFormat; 4] = [
    RawVideoFormat::Rgba8,
    RawVideoFormat::Bgra8,
    RawVideoFormat::Nv12,
    RawVideoFormat::I420,
];

/// Formats accepted as **input**. Superset of [`FORMATS`]: `Yuyv` (packed
/// 4:2:2, the usual webcam output) is unpacked to a planar / RGB target but is
/// never produced, so it is input-only.
const INPUT_FORMATS: [RawVideoFormat; 5] = [
    RawVideoFormat::Rgba8,
    RawVideoFormat::Bgra8,
    RawVideoFormat::Nv12,
    RawVideoFormat::I420,
    RawVideoFormat::Yuyv,
];

#[derive(Debug)]
pub struct VideoConvert {
    /// Target output format from the `format` property. `None` means "auto":
    /// take the output format from the negotiated caps (a downstream
    /// capsfilter), the gst caps-driven idiom (M186).
    target: Option<RawVideoFormat>,
    /// Format, geometry, and framerate of the configured input stream, updated
    /// by a mid-stream `CapsChanged`. The framerate is carried through to the
    /// output caps unchanged (a convert does not retime), so downstream sees a
    /// fixed rate rather than `Rate::Any` (which a fixating peer, e.g. a
    /// compositor input, would reject).
    input: Option<(RawVideoFormat, u32, u32, Rate)>,
    /// Output format resolved from the negotiated output caps (M186), set by
    /// `configure_output`. Used in auto mode; `None` until then so `process`
    /// falls back to the property and runners that don't deliver output caps
    /// keep the property-driven behavior.
    resolved: Option<RawVideoFormat>,
    configured: bool,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl VideoConvert {
    /// Convert to a fixed `target` format (property-driven).
    pub fn new(target: RawVideoFormat) -> Self {
        Self {
            target: Some(target),
            input: None,
            resolved: None,
            configured: false,
            last_caps: None,
            emitted: 0,
        }
    }

    /// Caps-driven (M186): take the output format from the negotiated caps (a
    /// downstream capsfilter). With no downstream constraint it defaults to
    /// passthrough (no conversion).
    pub fn auto() -> Self {
        Self {
            target: None,
            input: None,
            resolved: None,
            configured: false,
            last_caps: None,
            emitted: 0,
        }
    }

    /// The configured target format, or `None` in auto mode.
    pub fn target(&self) -> Option<RawVideoFormat> {
        self.target
    }

    /// The effective output format: the property when set, else the
    /// caps-resolved format (auto).
    fn out_format(&self) -> Option<RawVideoFormat> {
        self.target.or(self.resolved)
    }

    /// Validate a raw-video caps as a convertible input and return its
    /// format and dims. 4:2:0 endpoints need even dims on either side.
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
        if !INPUT_FORMATS.contains(format) || *w == 0 || *h == 0 {
            return Err(G2gError::CapsMismatch);
        }
        if (is_yuv420(*format) || self.target.is_some_and(is_yuv420)) && (*w % 2 != 0 || *h % 2 != 0)
        {
            return Err(G2gError::CapsMismatch);
        }
        // YUYV pairs two horizontal pixels per macropixel, so the width must be
        // even regardless of the target.
        if *format == RawVideoFormat::Yuyv && *w % 2 != 0 {
            return Err(G2gError::CapsMismatch);
        }
        Ok((*format, *w, *h, framerate.clone()))
    }
}

impl AsyncElement for VideoConvert {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // any supported raw format at any geometry; per-format alternatives
        // intersected in declaration order.
        for format in INPUT_FORMATS {
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

    /// Native `DerivedOutput`: any supported raw input maps to the target
    /// format at the same dims/framerate.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let target = self.target;
        // Passthrough geometry + framerate (retarget format only), so a downstream
        // geometry pin couples back through this format-only converter (M188's
        // scale_then_convert case).
        let passthrough =
            PassthroughFields::NONE.with_width().with_height().with_framerate();
        let derive = Box::new(move |input: &Caps| match input {
            Caps::RawVideo { format, width, height, framerate }
                if INPUT_FORMATS.contains(format) =>
            {
                let mk = |f: RawVideoFormat| Caps::RawVideo {
                    format: f,
                    width: width.clone(),
                    height: height.clone(),
                    framerate: framerate.clone(),
                };
                match target {
                    // Property-driven: the fixed target format.
                    Some(t) => CapsSet::one(mk(t)),
                    // Caps-driven (auto): any producible format at this geometry,
                    // preferring passthrough (the input format, no conversion)
                    // when it is itself producible. Yuyv is input-only, so a
                    // Yuyv input must convert and lists the producible set.
                    None => {
                        let prefer_passthrough = FORMATS.contains(format);
                        let mut alts = Vec::new();
                        if prefer_passthrough {
                            alts.push(mk(*format));
                        }
                        for f in FORMATS {
                            if !(prefer_passthrough && f == *format) {
                                alts.push(mk(f));
                            }
                        }
                        CapsSet::from_alternatives(alts)
                    }
                }
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        });
        CapsConstraint::DerivedCoupled { derive, passthrough }
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (format, w, h, framerate) = self.accept_input(absolute_caps)?;
        self.input = Some((format, w, h, framerate));
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    /// M186: take the output format from the negotiated output caps when the
    /// `format` property is unset (caps-driven). Validates the format is
    /// producible and, if 4:2:0, that the input dims are even.
    fn configure_output(&mut self, output_caps: &Caps) -> Result<(), G2gError> {
        let Caps::RawVideo { format, .. } = output_caps else {
            return Err(G2gError::CapsMismatch);
        };
        if !FORMATS.contains(format) {
            return Err(G2gError::CapsMismatch);
        }
        if is_yuv420(*format) {
            if let Some((_, w, h, _)) = self.input {
                if w % 2 != 0 || h % 2 != 0 {
                    return Err(G2gError::CapsMismatch);
                }
            }
        }
        self.resolved = Some(*format);
        Ok(())
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
                    let (format, w, h, framerate) =
                        self.input.clone().ok_or(G2gError::NotConfigured)?;
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let src = slice.as_slice();
                    if src.len() < frame_byte_size(format, w, h) {
                        return Err(G2gError::CapsMismatch);
                    }
                    // Effective output format: property, or caps-resolved (auto).
                    // Auto without a delivered output caps (a runner that doesn't
                    // call configure_output) is unfixed.
                    let out_fmt = self.out_format().ok_or(G2gError::NotConfigured)?;
                    let converted = convert(src, format, out_fmt, w as usize, h as usize);

                    // A convert changes format/geometry but not rate: carry the
                    // input framerate so a fixating downstream peer (e.g. a
                    // compositor input) does not reject a `Rate::Any`.
                    let new_caps = Caps::RawVideo {
                        format: out_fmt,
                        width: Dim::Fixed(w),
                        height: Dim::Fixed(h),
                        framerate,
                    };
                    if self.last_caps.as_ref() != Some(&new_caps) {
                        out.push(PipelinePacket::CapsChanged(new_caps.clone())).await?;
                        self.last_caps = Some(new_caps);
                    }
                    let out_frame = Frame {
                        domain: MemoryDomain::System(SystemSlice::from_boxed(converted)),
                        timing: frame.timing,
                        sequence: self.emitted,
                        meta: Default::default(),
                    };
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(out_frame)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    let (format, w, h, framerate) = self.accept_input(&c)?;
                    self.input = Some((format, w, h, framerate));
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
        VIDEOCONVERT_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Video colorspace converter",
            "Filter/Converter/Video",
            "Converts between raw video pixel formats (RGBA, BGRA, NV12, I420)",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "format" => {
                self.target = Some(raw_format_from_str(value.as_str().ok_or(PropError::Type)?)
                    .ok_or(PropError::Value)?);
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            // The effective output format: the property, or the caps-resolved
            // one once negotiated. `None` for an unconfigured auto instance.
            "format" => self.out_format().map(|f| PropValue::Str(raw_format_to_str(f).into())),
            _ => None,
        }
    }
}

/// `VideoConvert`'s settable properties (M107): the output pixel format.
static VIDEOCONVERT_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "format",
    PropKind::Str,
    "output pixel format: NV12 | I420 | RGBA | BGRA | YUY2",
)];

/// Parse a pixel-format property string to a [`RawVideoFormat`]. Shared name set
/// for the `gst-launch` DSL. GStreamer caps name formats uppercase (NV12, RGBA,
/// YUY2); accept any case and the historical lowercase spellings as aliases so
/// both port.
pub(crate) fn raw_format_from_str(s: &str) -> Option<RawVideoFormat> {
    match s.to_ascii_lowercase().as_str() {
        "nv12" => Some(RawVideoFormat::Nv12),
        "i420" => Some(RawVideoFormat::I420),
        "rgba" => Some(RawVideoFormat::Rgba8),
        "bgra" => Some(RawVideoFormat::Bgra8),
        // `yuyv` is GStreamer's `YUY2` fourcc; accept both names.
        "yuy2" | "yuyv" => Some(RawVideoFormat::Yuyv),
        _ => None,
    }
}

/// The canonical (GStreamer) property string for a [`RawVideoFormat`].
pub(crate) fn raw_format_to_str(f: RawVideoFormat) -> &'static str {
    match f {
        RawVideoFormat::Nv12 => "NV12",
        RawVideoFormat::I420 => "I420",
        RawVideoFormat::Rgba8 => "RGBA",
        RawVideoFormat::Bgra8 => "BGRA",
        RawVideoFormat::Yuyv => "YUY2",
    }
}

impl PadTemplates for VideoConvert {
    /// Static superset: any supported raw format in, any out (an instance
    /// narrows the source pad to its configured target).
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
        // Packed 4:2:2: two bytes per pixel (Y0 U Y1 V over each pixel pair).
        RawVideoFormat::Yuyv => w * h * 2,
    }
}

/// Dispatch one frame conversion. `src` is validated to hold at least the
/// input frame; dims are even whenever a 4:2:0 format is involved.
pub(crate) fn convert(
    src: &[u8],
    from: RawVideoFormat,
    to: RawVideoFormat,
    w: usize,
    h: usize,
) -> Box<[u8]> {
    use RawVideoFormat::*;
    match (from, to) {
        (a, b) if a == b => src[..frame_byte_size(a, w as u32, h as u32)].into(),
        (Rgba8, Bgra8) | (Bgra8, Rgba8) => swizzle_rb(src, w, h),
        (Nv12, I420) => nv12_to_i420(src, w, h),
        (I420, Nv12) => i420_to_nv12(src, w, h),
        (Rgba8, Nv12) => rgb_to_yuv420(src, w, h, 0, 2, true),
        (Rgba8, I420) => rgb_to_yuv420(src, w, h, 0, 2, false),
        (Bgra8, Nv12) => rgb_to_yuv420(src, w, h, 2, 0, true),
        (Bgra8, I420) => rgb_to_yuv420(src, w, h, 2, 0, false),
        (Nv12, Rgba8) => yuv420_to_rgb(src, w, h, true, 0, 2),
        (I420, Rgba8) => yuv420_to_rgb(src, w, h, false, 0, 2),
        (Nv12, Bgra8) => yuv420_to_rgb(src, w, h, true, 2, 0),
        (I420, Bgra8) => yuv420_to_rgb(src, w, h, false, 2, 0),
        // YUYV (packed 4:2:2) is input-only: unpack to the planar / RGB target.
        (Yuyv, I420) => yuyv_to_yuv420(src, w, h, false),
        (Yuyv, Nv12) => yuyv_to_yuv420(src, w, h, true),
        (Yuyv, Rgba8) => yuyv_to_rgb(src, w, h, 0, 2),
        (Yuyv, Bgra8) => yuyv_to_rgb(src, w, h, 2, 0),
        // every reachable pair is enumerated; identical pairs hit the guard,
        // and no path converts *to* Yuyv (it is never a `target`).
        _ => unreachable!("unhandled raw-video conversion {from:?} -> {to:?}"),
    }
}

/// Packed YUYV (4:2:2, byte order Y0 U Y1 V) -> 4:2:0 YUV. The luma plane is a
/// direct deinterleave; chroma drops to half vertical resolution by averaging
/// the two source rows that share each output chroma sample. `interleaved`
/// selects NV12 (true) vs I420 (false) chroma layout. Width is even (checked at
/// negotiation); height is even whenever the 4:2:0 target is involved.
fn yuyv_to_yuv420(src: &[u8], w: usize, h: usize, interleaved: bool) -> Box<[u8]> {
    let luma = w * h;
    let mut dst = vec![0u8; luma + luma / 2];
    // Luma: every macropixel (4 src bytes) yields two Y samples.
    for y in 0..h {
        for x in 0..w {
            dst[y * w + x] = src[(y * w + x) * 2];
        }
    }
    let (cw, ch) = (w / 2, h / 2);
    for cy in 0..ch {
        for cx in 0..cw {
            // The macropixel at column `cx` carries one U and one V per row;
            // average the two rows (cy*2, cy*2+1) for the 4:2:0 sample.
            let row0 = ((cy * 2) * w + cx * 2) * 2;
            let row1 = ((cy * 2 + 1) * w + cx * 2) * 2;
            let u = (src[row0 + 1] as u32 + src[row1 + 1] as u32).div_ceil(2);
            let v = (src[row0 + 3] as u32 + src[row1 + 3] as u32).div_ceil(2);
            let ci = cy * cw + cx;
            if interleaved {
                dst[luma + 2 * ci] = u as u8;
                dst[luma + 2 * ci + 1] = v as u8;
            } else {
                dst[luma + ci] = u as u8;
                dst[luma + luma / 4 + ci] = v as u8;
            }
        }
    }
    dst.into_boxed_slice()
}

/// Packed YUYV (4:2:2) -> packed 4-byte RGB(A), BT.601 limited range, integer
/// math (same coefficients as [`yuv420_to_rgb`]). Each macropixel's two Y
/// samples share its U/V; alpha is opaque. `r_off`/`b_off` pick the channel
/// order (RGBA: 0/2, BGRA: 2/0).
fn yuyv_to_rgb(src: &[u8], w: usize, h: usize, r_off: usize, b_off: usize) -> Box<[u8]> {
    let mut dst = vec![0u8; w * h * 4];
    for y in 0..h {
        for mx in 0..(w / 2) {
            let s = (y * w + mx * 2) * 2;
            let (y0, u, y1, v) = (
                src[s] as i32,
                src[s + 1] as i32,
                src[s + 2] as i32,
                src[s + 3] as i32,
            );
            let d = u - 128;
            let e = v - 128;
            for (yi, luma) in [y0, y1].into_iter().enumerate() {
                let c = luma - 16;
                let p = (y * w + mx * 2 + yi) * 4;
                dst[p + r_off] = ((298 * c + 409 * e + 128) >> 8).clamp(0, 255) as u8;
                dst[p + 1] = ((298 * c - 100 * d - 208 * e + 128) >> 8).clamp(0, 255) as u8;
                dst[p + b_off] = ((298 * c + 516 * d + 128) >> 8).clamp(0, 255) as u8;
                dst[p + 3] = 255;
            }
        }
    }
    dst.into_boxed_slice()
}

/// RGBA<->BGRA: swap the R and B channels.
fn swizzle_rb(src: &[u8], w: usize, h: usize) -> Box<[u8]> {
    let mut dst = src[..w * h * 4].to_vec();
    for px in dst.chunks_exact_mut(4) {
        px.swap(0, 2);
    }
    dst.into_boxed_slice()
}

/// NV12 -> I420: split the interleaved UV plane into separate U and V.
fn nv12_to_i420(src: &[u8], w: usize, h: usize) -> Box<[u8]> {
    let luma = w * h;
    let chroma = luma / 4;
    let mut dst = vec![0u8; luma + 2 * chroma];
    dst[..luma].copy_from_slice(&src[..luma]);
    for i in 0..chroma {
        dst[luma + i] = src[luma + 2 * i];
        dst[luma + chroma + i] = src[luma + 2 * i + 1];
    }
    dst.into_boxed_slice()
}

/// I420 -> NV12: interleave the U and V planes.
fn i420_to_nv12(src: &[u8], w: usize, h: usize) -> Box<[u8]> {
    let luma = w * h;
    let chroma = luma / 4;
    let mut dst = vec![0u8; luma + 2 * chroma];
    dst[..luma].copy_from_slice(&src[..luma]);
    for i in 0..chroma {
        dst[luma + 2 * i] = src[luma + i];
        dst[luma + 2 * i + 1] = src[luma + chroma + i];
    }
    dst.into_boxed_slice()
}

/// Packed 4-byte RGB(A) -> 4:2:0 YUV, BT.601 limited range, integer math.
/// `r_off`/`b_off` select the source channel order (RGBA: 0/2, BGRA: 2/0);
/// `interleaved` picks NV12 (true) vs I420 (false) chroma layout. Chroma is
/// the average of each 2x2 block.
fn rgb_to_yuv420(
    src: &[u8],
    w: usize,
    h: usize,
    r_off: usize,
    b_off: usize,
    interleaved: bool,
) -> Box<[u8]> {
    let luma = w * h;
    let mut dst = vec![0u8; luma + luma / 2];
    for y in 0..h {
        for x in 0..w {
            let p = (y * w + x) * 4;
            let (r, g, b) = (
                src[p + r_off] as i32,
                src[p + 1] as i32,
                src[p + b_off] as i32,
            );
            dst[y * w + x] = (((66 * r + 129 * g + 25 * b + 128) >> 8) + 16).clamp(0, 255) as u8;
        }
    }
    let (cw, ch) = (w / 2, h / 2);
    for cy in 0..ch {
        for cx in 0..cw {
            // average the 2x2 block's RGB before the chroma transform.
            let (mut r, mut g, mut b) = (0i32, 0i32, 0i32);
            for dy in 0..2 {
                for dx in 0..2 {
                    let p = ((cy * 2 + dy) * w + cx * 2 + dx) * 4;
                    r += src[p + r_off] as i32;
                    g += src[p + 1] as i32;
                    b += src[p + b_off] as i32;
                }
            }
            let (r, g, b) = (r / 4, g / 4, b / 4);
            let u = (((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128).clamp(0, 255) as u8;
            let v = (((112 * r - 94 * g - 18 * b + 128) >> 8) + 128).clamp(0, 255) as u8;
            let ci = cy * cw + cx;
            if interleaved {
                dst[luma + 2 * ci] = u;
                dst[luma + 2 * ci + 1] = v;
            } else {
                dst[luma + ci] = u;
                dst[luma + luma / 4 + ci] = v;
            }
        }
    }
    dst.into_boxed_slice()
}

/// 4:2:0 YUV -> packed 4-byte RGB(A), BT.601 limited range, integer math.
/// Alpha is set opaque. `interleaved` selects NV12 vs I420 chroma layout;
/// `r_off`/`b_off` the destination channel order.
fn yuv420_to_rgb(
    src: &[u8],
    w: usize,
    h: usize,
    interleaved: bool,
    r_off: usize,
    b_off: usize,
) -> Box<[u8]> {
    let luma = w * h;
    let cw = w / 2;
    let mut dst = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let ci = (y / 2) * cw + x / 2;
            let (u, v) = if interleaved {
                (src[luma + 2 * ci] as i32, src[luma + 2 * ci + 1] as i32)
            } else {
                (
                    src[luma + ci] as i32,
                    src[luma + luma / 4 + ci] as i32,
                )
            };
            let c = src[y * w + x] as i32 - 16;
            let d = u - 128;
            let e = v - 128;
            let p = (y * w + x) * 4;
            dst[p + r_off] = ((298 * c + 409 * e + 128) >> 8).clamp(0, 255) as u8;
            dst[p + 1] = ((298 * c - 100 * d - 208 * e + 128) >> 8).clamp(0, 255) as u8;
            dst[p + b_off] = ((298 * c + 516 * d + 128) >> 8).clamp(0, 255) as u8;
            dst[p + 3] = 255;
        }
    }
    dst.into_boxed_slice()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgba_caps(w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Any,
        }
    }

    #[test]
    fn derived_output_maps_any_supported_raw_to_target() {
        let conv = VideoConvert::new(RawVideoFormat::Nv12);
        let CapsConstraint::DerivedCoupled { derive: f, passthrough } =
            conv.caps_constraint_as_transform()
        else {
            panic!("expected DerivedCoupled");
        };
        assert_eq!(
            passthrough,
            PassthroughFields::NONE.with_width().with_height().with_framerate()
        );
        let out = f(&rgba_caps(64, 48));
        assert_eq!(
            out.alternatives(),
            &[Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(64),
                height: Dim::Fixed(48),
                framerate: Rate::Any,
            }]
        );
        // compressed input is not convertible
        let h264 = Caps::CompressedVideo {
            codec: g2g_core::VideoCodec::H264,
            width: Dim::Fixed(64),
            height: Dim::Fixed(48),
            framerate: Rate::Any,
        };
        assert!(f(&h264).is_empty());
    }

    #[test]
    fn yuv420_targets_reject_odd_dims() {
        let mut conv = VideoConvert::new(RawVideoFormat::Nv12);
        let err = conv
            .configure_pipeline(&rgba_caps(3, 2))
            .expect_err("odd dims into a 4:2:0 target must fail");
        assert_eq!(err, G2gError::CapsMismatch);
        // packed -> packed has no subsampling, odd dims are fine
        let mut swz = VideoConvert::new(RawVideoFormat::Bgra8);
        assert!(swz.configure_pipeline(&rgba_caps(3, 3)).is_ok());
    }

    #[test]
    fn swizzle_swaps_r_and_b() {
        let src = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let out = swizzle_rb(&src, 2, 1);
        assert_eq!(&out[..], &[3, 2, 1, 4, 7, 6, 5, 8]);
    }

    #[test]
    fn nv12_i420_repack_round_trips_losslessly() {
        // 2x2: 4 luma bytes + 1 U + 1 V
        let nv12 = [10u8, 20, 30, 40, 100, 200];
        let i420 = nv12_to_i420(&nv12, 2, 2);
        assert_eq!(&i420[..], &[10, 20, 30, 40, 100, 200]);
        let back = i420_to_nv12(&i420, 2, 2);
        assert_eq!(&back[..], &nv12[..]);
        // a chroma layout where U != V proves the deinterleave
        let nv12b = [0u8, 0, 0, 0, 7, 9];
        let i420b = nv12_to_i420(&nv12b, 2, 2);
        assert_eq!(&i420b[4..], &[7, 9], "U plane then V plane");
    }

    #[test]
    fn bt601_primaries_round_trip_within_tolerance() {
        // uniform 2x2 blocks survive 4:2:0 subsampling, so RGBA -> NV12 ->
        // RGBA must come back close (BT.601 integer rounding only).
        for &(r, g, b) in &[
            (255u8, 255u8, 255u8),
            (0, 0, 0),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (128, 64, 32),
        ] {
            let src: Vec<u8> = (0..4).flat_map(|_| [r, g, b, 255]).collect();
            let nv12 = rgb_to_yuv420(&src, 2, 2, 0, 2, true);
            let rgba = yuv420_to_rgb(&nv12, 2, 2, true, 0, 2);
            for px in rgba.chunks_exact(4) {
                assert!(
                    (px[0] as i32 - r as i32).abs() <= 4
                        && (px[1] as i32 - g as i32).abs() <= 4
                        && (px[2] as i32 - b as i32).abs() <= 4,
                    "({r},{g},{b}) round-tripped to ({},{},{})",
                    px[0],
                    px[1],
                    px[2]
                );
                assert_eq!(px[3], 255, "alpha is opaque");
            }
        }
    }

    #[test]
    fn grey_maps_to_neutral_chroma() {
        // pure grey has no chroma: U = V = 128 exactly in BT.601.
        let src: Vec<u8> = (0..4).flat_map(|_| [128u8, 128, 128, 255]).collect();
        let nv12 = rgb_to_yuv420(&src, 2, 2, 0, 2, true);
        assert_eq!(&nv12[4..], &[128, 128], "neutral chroma for grey");
    }

    #[test]
    fn yuyv_unpacks_luma_and_averages_chroma_to_i420() {
        // 2x2 YUYV: one macropixel per row, [Y0, U, Y1, V].
        // row 0: Y=10,20 U=100 V=200 ; row 1: Y=30,40 U=110 V=210
        let yuyv = [10u8, 100, 20, 200, 30, 110, 40, 210];
        let i420 = yuyv_to_yuv420(&yuyv, 2, 2, false);
        // luma is a straight deinterleave; chroma is the vertical average
        // (4:2:2 -> 4:2:0), U plane then V plane.
        assert_eq!(&i420[..4], &[10, 20, 30, 40], "luma deinterleave");
        assert_eq!(i420[4], 105, "U = avg(100,110)");
        assert_eq!(i420[5], 205, "V = avg(200,210)");
        assert_eq!(i420.len(), 2 * 2 * 3 / 2);
    }

    #[test]
    fn yuyv_to_rgb_is_grey_for_neutral_chroma() {
        // Y0 == Y1, U = V = 128: both unpacked pixels are the same neutral grey
        // and alpha is opaque.
        let yuyv = [128u8, 128, 128, 128];
        let rgba = yuyv_to_rgb(&yuyv, 2, 1, 0, 2);
        assert_eq!(rgba.len(), 2 * 1 * 4);
        let (p0, p1) = (&rgba[0..4], &rgba[4..8]);
        assert_eq!(p0, p1, "equal luma -> identical pixels");
        assert_eq!(p0[0], p0[1], "grey: R == G");
        assert_eq!(p0[1], p0[2], "grey: G == B");
        assert_eq!(p0[3], 255, "alpha opaque");
    }
}
