//! Format conversion and spatial resampling in one element, what the pair
//! `videoconvert ! videoscale` does in two.
//!
//! The gain is one pass instead of two: for the pairs a vision pipeline runs
//! (4:2:0 or packed RGB in, packed RGB out) each output pixel is sampled where
//! it lands in the source and converted right there, so a
//! `1080p NV12 -> 640x640 RGB` chain never builds a converted 1080p frame nor a
//! resampled intermediate. The sampling weights and the color math are
//! `videoscale`'s and `videoconvert`'s own, so the result agrees with theirs.
//!
//! Any other pair falls back to a [`VideoScale`] and a [`VideoConvert`] this
//! element owns and drives in sequence, which keeps the long tail of formats
//! working without a second copy of their code.
//!
//! The `format` / `width` / `height` properties pin the output; whichever is
//! left unset comes from a downstream capsfilter, exactly as for the two
//! elements alone.

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::pixel::frame_byte_size;
use crate::videoconvert::{yuv_to_rgb, VideoConvert};
use crate::videoscale::{bilerp, map_axis, VideoScale};
use alloc::vec;
use g2g_core::frame::Frame;
use g2g_core::memory::{DomainSet, SystemSlice};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsTransform, ConfigureOutcome, Dim, ElementMetadata,
    G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec, PushOutcome, RawVideoFormat, RawVideoShape,
};

/// Resample and convert in one pass over the output: each output pixel is
/// sampled where it lands in the source and converted right there, so neither a
/// full-size converted frame nor a full-size resampled one is ever built.
///
/// Covers the pairs a vision pipeline runs (a decoder's 4:2:0 or a packed RGB
/// input, to packed RGB out). `None` for anything else, which falls back to the
/// two elements in sequence. The sampling weights and the color math are the
/// ones `videoscale` and `videoconvert` use, so a fused frame equals the
/// two-step one.
fn fused_resample_convert(
    src: &[u8],
    from: RawVideoFormat,
    to: RawVideoFormat,
    in_w: usize,
    in_h: usize,
    out_w: usize,
    out_h: usize,
) -> Option<Box<[u8]>> {
    let out_channels = packed_channels(to)?;
    let (r_off, b_off) = channel_order(to)?;
    let columns: Vec<(usize, usize, u32)> = (0..out_w).map(|x| map_axis(x, out_w, in_w)).collect();
    let mut dst = vec![0u8; out_w * out_h * out_channels];

    if let Some(in_channels) = packed_channels(from) {
        let (src_r, src_b) = channel_order(from)?;
        for y in 0..out_h {
            let (y0, y1, fy) = map_axis(y, out_h, in_h);
            let (row0, row1) = (y0 * in_w, y1 * in_w);
            for (x, &(x0, x1, fx)) in columns.iter().enumerate() {
                let sample = |channel: usize| {
                    bilerp(
                        src[(row0 + x0) * in_channels + channel],
                        src[(row0 + x1) * in_channels + channel],
                        src[(row1 + x0) * in_channels + channel],
                        src[(row1 + x1) * in_channels + channel],
                        fx,
                        fy,
                    )
                };
                let out = (y * out_w + x) * out_channels;
                dst[out + r_off] = sample(src_r);
                dst[out + 1] = sample(1);
                dst[out + b_off] = sample(src_b);
                if out_channels == 4 {
                    dst[out + 3] = 255;
                }
            }
        }
        return Some(dst.into_boxed_slice());
    }

    // 4:2:0, chroma at half resolution: luma samples in its own plane, chroma in
    // the half-size one, so both are interpolated rather than point-sampled.
    let interleaved = match from {
        RawVideoFormat::Nv12 => true,
        RawVideoFormat::I420 => false,
        _ => return None,
    };
    let luma = in_w * in_h;
    let (cw, ch) = (in_w / 2, in_h / 2);
    let chroma_columns: Vec<(usize, usize, u32)> =
        (0..out_w).map(|x| map_axis(x, out_w, cw)).collect();
    let chroma_at = |index: usize| {
        if interleaved {
            (src[luma + 2 * index], src[luma + 2 * index + 1])
        } else {
            (src[luma + index], src[luma + cw * ch + index])
        }
    };
    for y in 0..out_h {
        let (y0, y1, fy) = map_axis(y, out_h, in_h);
        let (row0, row1) = (y0 * in_w, y1 * in_w);
        let (cy0, cy1, cfy) = map_axis(y, out_h, ch);
        let (crow0, crow1) = (cy0 * cw, cy1 * cw);
        for (x, &(x0, x1, fx)) in columns.iter().enumerate() {
            let luma_sample = bilerp(
                src[row0 + x0],
                src[row0 + x1],
                src[row1 + x0],
                src[row1 + x1],
                fx,
                fy,
            );
            let (cx0, cx1, cfx) = chroma_columns[x];
            let (u00, v00) = chroma_at(crow0 + cx0);
            let (u10, v10) = chroma_at(crow0 + cx1);
            let (u01, v01) = chroma_at(crow1 + cx0);
            let (u11, v11) = chroma_at(crow1 + cx1);
            let (r, g, b) = yuv_to_rgb(
                luma_sample as i32,
                bilerp(u00, u10, u01, u11, cfx, cfy) as i32,
                bilerp(v00, v10, v01, v11, cfx, cfy) as i32,
            );
            let out = (y * out_w + x) * out_channels;
            dst[out + r_off] = r as u8;
            dst[out + 1] = g as u8;
            dst[out + b_off] = b as u8;
            if out_channels == 4 {
                dst[out + 3] = 255;
            }
        }
    }
    Some(dst.into_boxed_slice())
}

/// Bytes per pixel of a packed RGB-family format, `None` for anything else.
fn packed_channels(format: RawVideoFormat) -> Option<usize> {
    match format {
        RawVideoFormat::Rgba8 | RawVideoFormat::Bgra8 => Some(4),
        RawVideoFormat::Rgb8 => Some(3),
        _ => None,
    }
}

/// Byte offsets of red and blue within a packed pixel; green is always 1.
fn channel_order(format: RawVideoFormat) -> Option<(usize, usize)> {
    match format {
        RawVideoFormat::Rgba8 | RawVideoFormat::Rgb8 => Some((0, 2)),
        RawVideoFormat::Bgra8 => Some((2, 0)),
        _ => None,
    }
}

/// Catches one element's output so it can be fed to the next.
#[derive(Default, Debug)]
struct Relay {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for Relay {
    fn poll_push(
        &mut self,
        _cx: &mut Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> Poll<Result<PushOutcome, G2gError>> {
        self.packets
            .push(packet_slot.take().expect("poll_push without a packet"));
        Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

/// Run `packet` through `first`, then every packet it produced through
/// `second`. Both are already configured (see `configure_output`).
async fn drive<A: AsyncElement, B: AsyncElement>(
    first: &mut A,
    second: &mut B,
    packet: PipelinePacket,
    out: &mut dyn OutputSink,
) -> Result<(), G2gError> {
    let mut relay = Relay::default();
    first.process(packet, &mut relay).await?;
    for produced in relay.packets {
        // The halfway caps stay inside: a transform reads an incoming
        // `CapsChanged` as its own already fixed output, so forwarding it would
        // announce the half-converted format downstream. `second` announces the
        // real output caps from its own data path.
        if matches!(produced, PipelinePacket::CapsChanged(_)) {
            continue;
        }
        second.process(produced, out).await?;
    }
    Ok(())
}

/// # Example
///
/// ```no_run
/// use g2g_core::RawVideoFormat;
/// use g2g_plugins::videoconvertscale::VideoConvertScale;
///
/// // gst-launch equivalent:
/// //   videoconvertscale ! video/x-raw,format=RGB,width=640,height=640
/// let caps_driven = VideoConvertScale::auto();
/// let pinned = VideoConvertScale::new(RawVideoFormat::Rgb8, 640, 640);
/// ```
#[derive(Debug)]
pub struct VideoConvertScale {
    scale: VideoScale,
    convert: VideoConvert,
    /// True when the input format resamples, so scaling runs first and the
    /// conversion sees the smaller frame. Decided at `configure_pipeline`.
    scale_first: bool,
    /// The negotiated output caps, held until the first element announces the
    /// intermediate ones and the second can be configured.
    output_caps: Option<Caps>,
    input_caps: Option<Caps>,
    second_ready: bool,
    configured: bool,
    announced: bool,
    emitted: u64,
}

/// `caps` with its format and geometry replaced, keeping framerate and
/// interlacing. Anything but a raw-video caps comes back unchanged.
fn retarget(caps: &Caps, format: RawVideoFormat, width: u32, height: u32) -> Caps {
    match caps {
        Caps::RawVideo {
            framerate,
            interlace,
            ..
        } => Caps::RawVideo {
            format,
            width: Dim::Fixed(width),
            height: Dim::Fixed(height),
            framerate: framerate.clone(),
            interlace: *interlace,
        },
        other => other.clone(),
    }
}

/// A fully fixed raw-video caps as `(format, width, height)`.
fn fixed_video(caps: &Caps) -> Option<(RawVideoFormat, usize, usize)> {
    match caps {
        Caps::RawVideo {
            format,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            ..
        } => Some((*format, *w as usize, *h as usize)),
        _ => None,
    }
}

impl VideoConvertScale {
    /// Convert and scale to a fixed format and geometry (property-driven).
    pub fn new(format: RawVideoFormat, width: u32, height: u32) -> Self {
        Self {
            scale: VideoScale::new(width, height),
            convert: VideoConvert::new(format),
            scale_first: true,
            output_caps: None,
            input_caps: None,
            second_ready: false,
            configured: false,
            announced: false,
            emitted: 0,
        }
    }

    /// Caps-driven: take format and geometry from the negotiated caps. With no
    /// downstream constraint this is a passthrough.
    pub fn auto() -> Self {
        Self {
            scale: VideoScale::new(0, 0),
            convert: VideoConvert::auto(),
            scale_first: true,
            output_caps: None,
            input_caps: None,
            second_ready: false,
            configured: false,
            announced: false,
            emitted: 0,
        }
    }
}

impl VideoConvertScale {
    /// The halfway caps when the properties alone pin this element's output, so
    /// the pair can be configured before any caps arrive from downstream.
    fn intermediate_from_properties(&self, input: &Caps) -> Option<Caps> {
        let (in_format, in_w, in_h) = fixed_video(input)?;
        if self.scale_first {
            let (w, h) = (self.scale.target_dims().0, self.scale.target_dims().1);
            (w > 0 && h > 0).then(|| retarget(input, in_format, w, h))
        } else {
            let format = self.convert.target()?;
            Some(retarget(input, format, in_w as u32, in_h as u32))
        }
    }

    /// Hand the halfway caps to whichever element runs second, and the final
    /// caps to whichever produces them.
    fn configure_second(
        &mut self,
        intermediate: &Caps,
        output_caps: Option<&Caps>,
    ) -> Result<(), G2gError> {
        if self.scale_first {
            self.scale.configure_output(intermediate)?;
            self.convert.configure_pipeline(intermediate)?;
            if let Some(caps) = output_caps {
                self.convert.configure_output(caps)?;
            }
        } else {
            self.convert.configure_output(intermediate)?;
            self.scale.configure_pipeline(intermediate)?;
            if let Some(caps) = output_caps {
                self.scale.configure_output(caps)?;
            }
        }
        self.second_ready = true;
        Ok(())
    }

    /// The fused output for this frame, or `None` when the negotiated pair has
    /// no fused path and the two elements have to run in sequence.
    fn fuse(&self, frame: &Frame) -> Result<Option<Box<[u8]>>, G2gError> {
        let (Some(input), Some(output)) = (&self.input_caps, &self.output_caps) else {
            return Ok(None);
        };
        let (Some((from, in_w, in_h)), Some((to, out_w, out_h))) =
            (fixed_video(input), fixed_video(output))
        else {
            return Ok(None);
        };
        let src = frame
            .domain
            .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
        if src.len() < frame_byte_size(from, in_w as u32, in_h as u32) {
            return Err(G2gError::CapsMismatch);
        }
        Ok(fused_resample_convert(
            src, from, to, in_w, in_h, out_w, out_h,
        ))
    }
}

impl AsyncElement for VideoConvertScale {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_domains(&self) -> DomainSet {
        self.convert.input_domains()
    }

    #[cfg(feature = "metadata")]
    fn meta_transform(&self) -> Option<g2g_core::meta::Transform> {
        self.scale.meta_transform()
    }

    #[cfg(feature = "metadata")]
    fn meta_requests(&self) -> g2g_core::meta::MetaRequests {
        self.convert.meta_requests()
    }

    /// The input side is the converter's: it accepts the wider set, the
    /// input-only packed 4:2:2 included.
    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        self.convert.intercept_caps(upstream_caps)
    }

    /// The two elements' own declarations, merged: each format the converter can
    /// produce, at each geometry the scaler can produce. Neither restates the
    /// other's rules, so a change to either lands here.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let convert_constraint = self.convert.caps_constraint_as_transform();
        let scale_constraint = self.scale.caps_constraint_as_transform();
        let (
            CapsConstraint::DerivedFields(CapsTransform::RawVideo {
                accept,
                produce,
                shapes: format_shapes,
            }),
            CapsConstraint::DerivedFields(CapsTransform::RawVideo {
                shapes: geometry_shapes,
                ..
            }),
        ) = (convert_constraint, scale_constraint)
        else {
            // Both elements declare `DerivedFields(RawVideo)`; anything else
            // means one changed shape and this merge no longer describes it.
            return self.convert.caps_constraint_as_transform();
        };
        let mut shapes: Vec<RawVideoShape> = Vec::new();
        for format in &format_shapes {
            for geometry in &geometry_shapes {
                let merged = RawVideoShape {
                    format: format.format.clone(),
                    width: geometry.width.clone(),
                    height: geometry.height.clone(),
                    framerate: format.framerate.clone(),
                };
                if !shapes.contains(&merged) {
                    shapes.push(merged);
                }
            }
        }
        CapsConstraint::DerivedFields(CapsTransform::RawVideo {
            accept,
            produce,
            shapes,
        })
    }

    /// Scaling goes first when the input format resamples; an input-only format
    /// is converted first instead. Only the first element is configured here:
    /// the second takes the intermediate caps the first announces.
    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.scale_first = self.scale.configure_pipeline(absolute_caps).is_ok();
        if !self.scale_first {
            self.convert.configure_pipeline(absolute_caps)?;
        }
        self.input_caps = Some(absolute_caps.clone());
        self.second_ready = false;
        self.announced = false;
        self.configured = true;
        // Properties that pin the output make the halfway caps knowable now. A
        // caps-driven output does not, and waits for `configure_output`.
        if let Some(intermediate) = self.intermediate_from_properties(absolute_caps) {
            self.configure_second(&intermediate, None)?;
        }
        Ok(ConfigureOutcome::Accepted)
    }

    /// The negotiated output caps carry the target format and geometry, so the
    /// element running first takes them now for the half it performs, and the
    /// second takes them once the intermediate caps have configured it.
    /// Both elements are configured here, from the caps on either side and the
    /// halfway caps between them. Deriving that rather than waiting for the
    /// first element to announce it matters: the runner hands a transform its
    /// output caps up front, `VideoScale` records those and then has nothing new
    /// to announce on the first frame, so anything waiting on that announcement
    /// would never be configured at all.
    fn configure_output(&mut self, output_caps: &Caps) -> Result<(), G2gError> {
        let input = self.input_caps.clone().ok_or(G2gError::NotConfigured)?;
        let (in_format, ..) = fixed_video(&input).ok_or(G2gError::CapsMismatch)?;
        let (out_format, out_w, out_h) = fixed_video(output_caps).ok_or(G2gError::CapsMismatch)?;

        // Scaling first changes only the geometry, converting first only the
        // format, so the halfway caps takes one field from each side.
        let intermediate = if self.scale_first {
            retarget(&input, in_format, out_w as u32, out_h as u32)
        } else {
            let (_, in_w, in_h) = fixed_video(&input).ok_or(G2gError::CapsMismatch)?;
            retarget(&input, out_format, in_w as u32, in_h as u32)
        };

        self.configure_second(&intermediate, Some(output_caps))?;
        self.output_caps = Some(output_caps.clone());
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
            // One pass over the output when the format pair allows it, which is
            // the whole point of fusing the two. Anything else, and every packet
            // that is not a frame, goes through them in sequence.
            if let PipelinePacket::DataFrame(frame) = &packet {
                if let Some(pixels) = self.fuse(frame)? {
                    let caps = self.output_caps.clone().ok_or(G2gError::NotConfigured)?;
                    if !self.announced {
                        out.push(PipelinePacket::CapsChanged(caps)).await?;
                        self.announced = true;
                    }
                    let out_frame = Frame {
                        domain: MemoryDomain::System(SystemSlice::from_boxed(pixels)),
                        timing: frame.timing,
                        sequence: self.emitted,
                        meta: Default::default(),
                    };
                    self.emitted += 1;
                    return out
                        .push(PipelinePacket::DataFrame(out_frame))
                        .await
                        .map(|_| ());
                }
            }
            if self.scale_first {
                drive(&mut self.scale, &mut self.convert, packet, out).await
            } else {
                drive(&mut self.convert, &mut self.scale, packet, out).await
            }
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        VIDEOCONVERTSCALE_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Video converter and scaler",
            "Filter/Converter/Video/Scaler",
            "Converts the pixel format and resizes raw video frames in one element",
            "g2g",
        )
    }

    /// Each property belongs to one of the two elements, which parses and
    /// validates it.
    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "format" => self.convert.set_property(name, value),
            "width" | "height" => self.scale.set_property(name, value),
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "format" => self.convert.get_property(name),
            "width" | "height" => self.scale.get_property(name),
            _ => None,
        }
    }
}

impl PadTemplates for VideoConvertScale {
    /// The converter's: what it accepts in, what it produces out. Geometry is
    /// unconstrained on both, which is what the scaler advertises anyway.
    fn pad_templates() -> Vec<PadTemplate> {
        VideoConvert::pad_templates()
    }
}

static VIDEOCONVERTSCALE_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "format",
        PropKind::Str,
        "output pixel format: RGBA | BGRA | RGB | NV12 | I420 | ...",
    ),
    PropertySpec::new("width", PropKind::Uint, "output width in pixels"),
    PropertySpec::new("height", PropKind::Uint, "output height in pixels"),
];
