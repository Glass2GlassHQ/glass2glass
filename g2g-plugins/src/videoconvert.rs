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

use crate::pixel::{even_dims_required, frame_byte_size, planar_planes};
use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    MemoryDomain, OutputSink, PadTemplate, PadTemplates, PassthroughFields, PipelinePacket,
    PropError, PropKind, PropValue, PropertySpec, Rate, RawVideoFormat,
};

/// Formats this element can both consume and produce. The convert `target`
/// is always one of these.
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

/// Formats accepted as **input**. Superset of [`FORMATS`]: `Yuyv` (packed
/// 4:2:2, the usual webcam output) is unpacked to a planar / RGB target but is
/// never produced, so it is input-only.
const INPUT_FORMATS: [RawVideoFormat; 13] = [
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
        // The dims must be even on every axis either the input or the (known)
        // target format subsamples, so chroma planes divide cleanly. YUYV folds in
        // as a horizontally-subsampled (even-width) format.
        let (mut ew, mut eh) = even_dims_required(*format);
        if let Some(target) = self.target {
            let (tw, th) = even_dims_required(target);
            ew |= tw;
            eh |= th;
        }
        if (ew && *w % 2 != 0) || (eh && *h % 2 != 0) {
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
        let (ew, eh) = even_dims_required(*format);
        if let Some((_, w, h, _)) = self.input {
            if (ew && w % 2 != 0) || (eh && h % 2 != 0) {
                return Err(G2gError::CapsMismatch);
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
                    // The runner's transform arm always calls `configure_pipeline`
                    // (input) then `configure_output` (output) immediately before
                    // pushing this packet, whose caps `c` is the arm's pre-fixed
                    // forward *output* (`forward_caps`), not a new input. Forward
                    // it so a strict downstream sees the converted format before
                    // the first frame, and record `last_caps` to suppress the
                    // duplicate emit from the data path. Do NOT call
                    // `accept_input` here: `c` is our output (e.g. NV12), and
                    // adopting it as the input would make the next RGBA frame a
                    // bogus NV12->NV12 passthrough (the stacked-convert bug). The
                    // real input is already set by `configure_pipeline`. Both our
                    // input and output are `Caps::RawVideo`, so unlike a decoder
                    // we cannot disambiguate the two by variant; we rely on the
                    // arm's ordering instead.
                    out.push(PipelinePacket::CapsChanged(c.clone())).await?;
                    self.last_caps = Some(c);
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
                other => {
                    out.push(other).await?;
                }
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
        RawVideoFormat::I420p10 => "I420_10LE",
        RawVideoFormat::I420p12 => "I420_12LE",
        RawVideoFormat::I422 => "Y42B",
        RawVideoFormat::I422p10 => "I422_10LE",
        RawVideoFormat::I422p12 => "I422_12LE",
        RawVideoFormat::I444 => "Y444",
        RawVideoFormat::I444p10 => "Y444_10LE",
        RawVideoFormat::I444p12 => "Y444_12LE",
        // A format added since: no canonical string here, fail loud.
        _ => unreachable!("unnamed RawVideoFormat: {f:?}"),
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

/// Dispatch one frame conversion. `src` is validated to hold at least the
/// input frame; dims are even whenever a 4:2:0 format is involved. Public so the
/// `convert` benchmark (M284) can exercise this hot path directly.
pub fn convert(
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
        // Any conversion involving a high-bit-depth / 4:2:2 / 4:4:4 format (the
        // pairs above are the fast 8-bit paths) goes through the general hub:
        // unpack to 4:4:4 YUV at a working depth, then repack to the target.
        _ => convert_via_hub(src, from, to, w, h),
    }
}

/// General conversion for any format pair, used for everything the fast 8-bit
/// paths above do not special-case (i.e. anything touching a high-bit-depth or
/// 4:2:2 / 4:4:4 format). The frame is unpacked to a canonical full-resolution
/// (4:4:4) planar YUV intermediate at a single working depth, then repacked to the
/// target: this turns the N x N matrix into N unpackers + N packers. Chroma is
/// upsampled by replication and downsampled by box-averaging; bit depth is scaled
/// by the full-range ratio; the YUV <-> RGB color step (BT.601 limited range) runs
/// at 8-bit, the precision an `Rgba8` endpoint carries anyway.
fn convert_via_hub(src: &[u8], from: RawVideoFormat, to: RawVideoFormat, w: usize, h: usize) -> Box<[u8]> {
    let (y, u, v, wd) = to_hub(src, from, w, h);
    from_hub(&y, &u, &v, wd, to, w, h)
}

/// Scale one sample from `from_d`-bit to `to_d`-bit full range, rounded to nearest.
fn scale_depth(v: i32, from_d: u8, to_d: u8) -> i32 {
    if from_d == to_d {
        return v;
    }
    let (fm, tm) = (((1i64 << from_d) - 1).max(1), (1i64 << to_d) - 1);
    (((v as i64 * tm + fm / 2) / fm) as i32).clamp(0, tm as i32)
}

/// RGB -> YUV (BT.601 limited range, 8-bit), the per-pixel form of `rgb_to_yuv420`.
fn rgb_to_yuv(r: i32, g: i32, b: i32) -> (i32, i32, i32) {
    let y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
    let u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
    let v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
    (y.clamp(0, 255), u.clamp(0, 255), v.clamp(0, 255))
}

/// YUV -> RGB (BT.601 limited range, 8-bit), the per-pixel form of `yuv420_to_rgb`.
fn yuv_to_rgb(y: i32, u: i32, v: i32) -> (i32, i32, i32) {
    let c = y - 16;
    let (d, e) = (u - 128, v - 128);
    let r = (298 * c + 409 * e + 128) >> 8;
    let g = (298 * c - 100 * d - 208 * e + 128) >> 8;
    let b = (298 * c + 516 * d + 128) >> 8;
    (r.clamp(0, 255), g.clamp(0, 255), b.clamp(0, 255))
}

/// Unpack any format to full-resolution (4:4:4) planar Y, U, V plus the working
/// bit depth: RGB is color-converted to 8-bit YUV; YUYV / NV12 / the planar family
/// are read at their own depth with chroma replicated up to full resolution.
fn to_hub(src: &[u8], from: RawVideoFormat, w: usize, h: usize) -> (Vec<i32>, Vec<i32>, Vec<i32>, u8) {
    let n = w * h;
    let mut y = vec![0i32; n];
    let mut u = vec![0i32; n];
    let mut v = vec![0i32; n];
    match from {
        RawVideoFormat::Rgba8 | RawVideoFormat::Bgra8 => {
            let (r_off, b_off) = crate::pixel::rgba_rb_offsets(from);
            for i in 0..n {
                let p = i * 4;
                let (yy, uu, vv) =
                    rgb_to_yuv(src[p + r_off] as i32, src[p + 1] as i32, src[p + b_off] as i32);
                (y[i], u[i], v[i]) = (yy, uu, vv);
            }
            (y, u, v, 8)
        }
        RawVideoFormat::Yuyv => {
            // Packed Y0 U Y1 V: each macropixel is two luma, one shared chroma pair,
            // replicated to both columns for 4:4:4.
            for row in 0..h {
                for col2 in 0..w / 2 {
                    let s = (row * (w / 2) + col2) * 4;
                    let (y0, cu, y1, cv) =
                        (src[s] as i32, src[s + 1] as i32, src[s + 2] as i32, src[s + 3] as i32);
                    let i = row * w + col2 * 2;
                    (y[i], u[i], v[i]) = (y0, cu, cv);
                    (y[i + 1], u[i + 1], v[i + 1]) = (y1, cu, cv);
                }
            }
            (y, u, v, 8)
        }
        RawVideoFormat::Nv12 => {
            let cw = w / 2;
            for row in 0..h {
                for col in 0..w {
                    let ci = (row / 2) * cw + col / 2;
                    let i = row * w + col;
                    y[i] = src[i] as i32;
                    u[i] = src[n + 2 * ci] as i32;
                    v[i] = src[n + 2 * ci + 1] as i32;
                }
            }
            (y, u, v, 8)
        }
        // The fully-planar family: read each plane at the format's depth, replicate
        // chroma up to 4:4:4 per the subsampling.
        f => {
            let d = f.bit_depth();
            let bps = f.bytes_per_sample();
            let (hs, vs) = f.chroma_shift().expect("planar format");
            let planes = planar_planes(f, w, h);
            let rd = |off: usize, idx: usize| -> i32 {
                if bps == 2 {
                    u16::from_le_bytes([src[off + idx * 2], src[off + idx * 2 + 1]]) as i32
                } else {
                    src[off + idx] as i32
                }
            };
            let (_, ucw, _) = planes[1];
            for row in 0..h {
                for col in 0..w {
                    let i = row * w + col;
                    y[i] = rd(planes[0].0, i);
                    let ci = (row >> vs) * ucw + (col >> hs);
                    u[i] = rd(planes[1].0, ci);
                    v[i] = rd(planes[2].0, ci);
                }
            }
            (y, u, v, d)
        }
    }
}

/// Repack full-resolution (4:4:4) planar Y, U, V at working depth `wd` into `to`:
/// RGB is color-converted from 8-bit YUV; the YUV targets downsample chroma by
/// box-averaging and scale the depth.
fn from_hub(
    y: &[i32],
    u: &[i32],
    v: &[i32],
    wd: u8,
    to: RawVideoFormat,
    w: usize,
    h: usize,
) -> Box<[u8]> {
    let n = w * h;
    match to {
        RawVideoFormat::Rgba8 | RawVideoFormat::Bgra8 => {
            let (r_off, b_off) = crate::pixel::rgba_rb_offsets(to);
            let mut dst = vec![0u8; n * 4];
            for i in 0..n {
                let (yy, uu, vv) =
                    (scale_depth(y[i], wd, 8), scale_depth(u[i], wd, 8), scale_depth(v[i], wd, 8));
                let (r, g, b) = yuv_to_rgb(yy, uu, vv);
                let p = i * 4;
                dst[p + r_off] = r as u8;
                dst[p + 1] = g as u8;
                dst[p + b_off] = b as u8;
                dst[p + 3] = 255;
            }
            dst.into_boxed_slice()
        }
        RawVideoFormat::Nv12 => {
            let (cw, ch) = (w / 2, h / 2);
            let mut dst = vec![0u8; n + 2 * cw * ch];
            for i in 0..n {
                dst[i] = scale_depth(y[i], wd, 8) as u8;
            }
            for cy in 0..ch {
                for cx in 0..cw {
                    let (su, sv) = avg_chroma(u, v, w, cx, cy, 1, 1);
                    let ci = cy * cw + cx;
                    dst[n + 2 * ci] = scale_depth(su, wd, 8) as u8;
                    dst[n + 2 * ci + 1] = scale_depth(sv, wd, 8) as u8;
                }
            }
            dst.into_boxed_slice()
        }
        // The fully-planar family: downsample chroma to the target subsampling
        // (averaging each block), scale to the target depth, write LE u16 if > 8.
        f => {
            let d = f.bit_depth();
            let bps = f.bytes_per_sample();
            let (hs, vs) = f.chroma_shift().expect("planar format");
            let planes = planar_planes(f, w, h);
            let total = planes[2].0 + planes[2].1 * planes[2].2 * bps;
            let mut dst = vec![0u8; total];
            let mut wr = |off: usize, idx: usize, val: i32| {
                let val = scale_depth(val, wd, d);
                if bps == 2 {
                    let b = (val as u16).to_le_bytes();
                    dst[off + idx * 2] = b[0];
                    dst[off + idx * 2 + 1] = b[1];
                } else {
                    dst[off + idx] = val as u8;
                }
            };
            for (i, &val) in y.iter().enumerate() {
                wr(planes[0].0, i, val);
            }
            let (_, cw, chh) = planes[1];
            for cy in 0..chh {
                for cx in 0..cw {
                    let (su, sv) = avg_chroma(u, v, w, cx, cy, hs, vs);
                    let ci = cy * cw + cx;
                    wr(planes[1].0, ci, su);
                    wr(planes[2].0, ci, sv);
                }
            }
            dst.into_boxed_slice()
        }
    }
}

/// Average the `2^hs x 2^vs` block of full-resolution chroma at chroma cell
/// `(cx, cy)` (the box filter for chroma downsampling). For 4:4:4 (`hs = vs = 0`)
/// this is the single co-located sample.
fn avg_chroma(u: &[i32], v: &[i32], w: usize, cx: usize, cy: usize, hs: u32, vs: u32) -> (i32, i32) {
    let (bw, bh) = (1usize << hs, 1usize << vs);
    let (mut su, mut sv) = (0i32, 0i32);
    for dy in 0..bh {
        for dx in 0..bw {
            let i = (cy * bh + dy) * w + cx * bw + dx;
            su += u[i];
            sv += v[i];
        }
    }
    let count = (bw * bh) as i32;
    ((su + count / 2) / count, (sv + count / 2) / count)
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
    // The expected lengths below are written as width * height * channels; keep
    // the unit factors (e.g. the height of 1) for legibility.
    #[allow(clippy::identity_op)]
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

    #[test]
    fn hub_scales_depth_8_to_10() {
        // 2x2 I420 (8-bit): luma all 255, chroma 128. Converting to I420p10 scales
        // each sample to 10-bit full range (luma 255 -> 1023) as little-endian u16.
        let src = vec![255u8, 255, 255, 255, 128, 128];
        let out = convert(&src, RawVideoFormat::I420, RawVideoFormat::I420p10, 2, 2);
        assert_eq!(out.len(), (4 + 1 + 1) * 2);
        let rd = |o: usize| u16::from_le_bytes([out[o], out[o + 1]]);
        assert_eq!(rd(0), 1023, "luma 255 -> 10-bit 1023");
    }

    #[test]
    fn hub_high_bit_depth_yuv_to_rgba_is_grey() {
        // 2x2 I444p10 with neutral chroma (U = V = 512, the 10-bit center) and a
        // mid luma converts to an opaque grey RGBA (R == G == B).
        let plane = |val: u16| (0..4).flat_map(move |_| val.to_le_bytes());
        let src: Vec<u8> = plane(512).chain(plane(512)).chain(plane(512)).collect();
        let out = convert(&src, RawVideoFormat::I444p10, RawVideoFormat::Rgba8, 2, 2);
        assert_eq!(out.len(), 2 * 2 * 4);
        for px in out.chunks_exact(4) {
            assert_eq!(px[0], px[1], "grey: R == G");
            assert_eq!(px[1], px[2], "grey: G == B");
            assert_eq!(px[3], 255, "alpha opaque");
            assert!((120..=140).contains(&px[0]), "neutral grey near mid");
        }
    }

    #[test]
    fn hub_chroma_resample_round_trips_a_flat_field() {
        // 4x4 I444 (8-bit) flat field: down to 4:2:0 (averaging equal samples) and
        // back up to 4:4:4 (replicating) recovers the original flat Y / U / V.
        let n = 16;
        let mut src = vec![100u8; n];
        src.extend(vec![120u8; n]);
        src.extend(vec![140u8; n]);
        let i420 = convert(&src, RawVideoFormat::I444, RawVideoFormat::I420, 4, 4);
        let back = convert(&i420, RawVideoFormat::I420, RawVideoFormat::I444, 4, 4);
        assert_eq!(back.len(), 3 * n);
        assert_eq!((back[0], back[n], back[2 * n]), (100, 120, 140), "flat Y / U / V preserved");
    }
}
