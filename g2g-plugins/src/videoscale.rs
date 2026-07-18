//! Software spatial resampler (P1.1). Scales raw video to a configured
//! output geometry, preserving the pixel format: `1080p -> 720p`,
//! thumbnails, or fitting a stream to an ML model's fixed input size. Pairs
//! with `VideoRate` (temporal) and feeds `WgpuPreprocess` / `OrtInference`
//! at the geometry they expect.
//!
//! Bilinear interpolation, integer-only (Q16 fixed-point weights, half-
//! pixel-centred source mapping) so the element is deterministic and stays
//! in the `no_std` crate baseline. Packed formats (`Rgba8`, `Bgra8`)
//! resample as one 4-channel plane; the 4:2:0 formats (`Nv12`, `I420`)
//! resample luma and chroma independently at their own resolutions, so
//! chroma keeps its half-resolution sampling. 4:2:0 needs even input and
//! output dims (chroma is subsampled 2x2); odd dims fail negotiation loud.
//!
//! Bilinear is the baseline-correctness choice, not peak quality; a wgpu
//! variant for GPU-resident input lands later. True separable bilinear is
//! output-identical to the single-pass form here, so the cache-friendly
//! two-pass split is a later optimisation, not a behaviour change.

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

/// True when `(w, h)` violates the even-width / even-height a format's chroma
/// subsampling requires (so a scale stays on chroma-sample boundaries).
fn bad_even_dims(format: RawVideoFormat, w: u32, h: u32) -> bool {
    let (ew, eh) = even_dims_required(format);
    (ew && w % 2 != 0) || (eh && h % 2 != 0)
}

/// Upper bound on the scalable output range advertised in caps-driven (auto)
/// mode (M185). Covers up to 8K with headroom; a downstream capsfilter pins a
/// concrete dim within it.
const MAX_DIM: u32 = 32768;

#[derive(Debug)]
pub struct VideoScale {
    /// Target geometry from the `width`/`height` properties. Zero on either axis
    /// means "auto": take the output geometry from the negotiated caps instead
    /// (a downstream capsfilter), the gst caps-driven idiom.
    target_w: u32,
    target_h: u32,
    /// Format and dims of the configured input stream, updated by a
    /// mid-stream `CapsChanged`. Carries the framerate so the output caps
    /// preserve it (scaling is spatial only).
    input: Option<(RawVideoFormat, u32, u32, Rate)>,
    /// Output dims resolved from the negotiated output caps (M185), set by
    /// `configure_output`. Used when the properties are unset (auto); `None`
    /// until then, so `process` falls back to the properties and runners that
    /// don't deliver output caps keep the property-driven behavior.
    resolved: Option<(u32, u32)>,
    configured: bool,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl VideoScale {
    pub fn new(target_width: u32, target_height: u32) -> Self {
        Self {
            target_w: target_width,
            target_h: target_height,
            input: None,
            resolved: None,
            configured: false,
            last_caps: None,
            emitted: 0,
        }
    }

    pub fn target_dims(&self) -> (u32, u32) {
        (self.target_w, self.target_h)
    }

    /// Auto / caps-driven mode: neither property pins the geometry.
    fn is_auto(&self) -> bool {
        self.target_w == 0 || self.target_h == 0
    }

    /// The effective output geometry: caps-resolved when available (auto),
    /// otherwise the configured target properties.
    fn out_dims(&self) -> (u32, u32) {
        self.resolved.unwrap_or((self.target_w, self.target_h))
    }

    /// Validate a raw-video caps as a scalable input and return its format,
    /// dims, and framerate. 4:2:0 inputs need even dims.
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
        if bad_even_dims(*format, *w, *h) {
            return Err(G2gError::CapsMismatch);
        }
        Ok((*format, *w, *h, framerate.clone()))
    }

    /// The configured target geometry must be non-zero, and even when the
    /// negotiated format is 4:2:0.
    fn validate_target(&self, format: RawVideoFormat) -> Result<(), G2gError> {
        if self.target_w == 0 || self.target_h == 0 {
            return Err(G2gError::CapsMismatch);
        }
        if bad_even_dims(format, self.target_w, self.target_h) {
            return Err(G2gError::CapsMismatch);
        }
        Ok(())
    }
}

impl AsyncElement for VideoScale {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // input side only: any supported raw format at the upstream
        // geometry. The output geometry is the configured target, declared
        // through `caps_constraint_as_transform`.
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

    /// Native `DerivedOutput`: any supported raw input maps to the same
    /// format at the configured target dims, framerate preserved. A 4:2:0
    /// format with an odd target collapses to the empty set so the solve
    /// fails loud rather than fixating impossible caps.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let (tw, th) = (self.target_w, self.target_h);
        // Passthrough format + framerate (retarget width/height), so a downstream
        // geometry pin behind a format-only transform couples back to the scaler.
        let passthrough = PassthroughFields::NONE.with_format().with_framerate();
        let derive = Box::new(move |input: &Caps| match input {
            Caps::RawVideo {
                format,
                width,
                height,
                framerate,
            } if FORMATS.contains(format) => {
                if tw > 0 && th > 0 {
                    // Property-driven: fixed target geometry.
                    if bad_even_dims(*format, tw, th) {
                        return CapsSet::from_alternatives(Vec::new());
                    }
                    CapsSet::one(Caps::RawVideo {
                        format: *format,
                        width: Dim::Fixed(tw),
                        height: Dim::Fixed(th),
                        framerate: framerate.clone(),
                    })
                } else {
                    // Caps-driven (auto): default to passthrough (the input
                    // geometry), but advertise we can scale to any geometry so a
                    // downstream capsfilter pins the target. Passthrough is the
                    // preferred (first) alternative, so with no downstream
                    // constraint the output is the input size (identity scale).
                    CapsSet::from_alternatives(vec![
                        Caps::RawVideo {
                            format: *format,
                            width: width.clone(),
                            height: height.clone(),
                            framerate: framerate.clone(),
                        },
                        Caps::RawVideo {
                            format: *format,
                            width: Dim::Range {
                                min: 1,
                                max: MAX_DIM,
                            },
                            height: Dim::Range {
                                min: 1,
                                max: MAX_DIM,
                            },
                            framerate: framerate.clone(),
                        },
                    ])
                }
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        });
        CapsConstraint::DerivedCoupled {
            derive,
            passthrough,
        }
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (format, w, h, rate) = self.accept_input(absolute_caps)?;
        // In auto mode the output geometry comes from `configure_output`, not the
        // properties, so only validate a property-pinned target here.
        if !self.is_auto() {
            self.validate_target(format)?;
        }
        self.input = Some((format, w, h, rate));
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    /// M185: take the output geometry from the negotiated output caps when the
    /// `width`/`height` properties are unset (caps-driven). When they are set,
    /// the solve already fixated the output to them, so this just records the
    /// same dims. Validates the resolved geometry (non-zero, even for 4:2:0).
    fn configure_output(&mut self, output_caps: &Caps) -> Result<(), G2gError> {
        let Caps::RawVideo {
            format,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            ..
        } = output_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if *w == 0 || *h == 0 {
            return Err(G2gError::CapsMismatch);
        }
        if bad_even_dims(*format, *w, *h) {
            return Err(G2gError::CapsMismatch);
        }
        self.resolved = Some((*w, *h));
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
                    // Effective output geometry: caps-resolved (auto) or the
                    // configured target. Auto without a delivered output caps
                    // (a runner that doesn't call configure_output) is unfixed.
                    let (out_w, out_h) = self.out_dims();
                    if out_w == 0 || out_h == 0 {
                        return Err(G2gError::NotConfigured);
                    }
                    let scaled = scale(
                        src,
                        format,
                        in_w as usize,
                        in_h as usize,
                        out_w as usize,
                        out_h as usize,
                    );

                    let new_caps = Caps::RawVideo {
                        format,
                        width: Dim::Fixed(out_w),
                        height: Dim::Fixed(out_h),
                        framerate: rate,
                    };
                    if self.last_caps.as_ref() != Some(&new_caps) {
                        out.push(PipelinePacket::CapsChanged(new_caps.clone()))
                            .await?;
                        self.last_caps = Some(new_caps);
                    }
                    let out_frame = Frame {
                        domain: MemoryDomain::System(SystemSlice::from_boxed(scaled)),
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
                    // it and record `last_caps` to suppress the duplicate emit
                    // from the data path. Do NOT call `accept_input`: `c` carries
                    // our output geometry, and adopting it as the input would make
                    // the next frame scale from the wrong source dims (the stacked
                    // auto-transform bug). The real input is already set by
                    // `configure_pipeline`.
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
        VIDEOSCALE_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Video scaler",
            "Filter/Converter/Video",
            "Resizes raw video frames",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "width" => {
                self.target_w = value.as_uint().ok_or(PropError::Type)? as u32;
                Ok(())
            }
            "height" => {
                self.target_h = value.as_uint().ok_or(PropError::Type)? as u32;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "width" => Some(PropValue::Uint(self.target_w as u64)),
            "height" => Some(PropValue::Uint(self.target_h as u64)),
            _ => None,
        }
    }
}

/// `VideoScale`'s settable properties (M107).
static VIDEOSCALE_PROPS: &[PropertySpec] = &[
    PropertySpec::new("width", PropKind::Uint, "output width in pixels"),
    PropertySpec::new("height", PropKind::Uint, "output height in pixels"),
];

impl PadTemplates for VideoScale {
    /// Static superset: any supported raw format in at any geometry; the
    /// same formats out, narrowed to the configured target dims.
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

/// Map an output index to its source sampling position, returning the two
/// neighbouring source indices and the Q16 weight between them. Uses
/// half-pixel-centred coordinates (`src = (out + 0.5) * src_n / dst_n -
/// 0.5`) so up- and downscale stay unbiased; clamped to the source extent
/// at the edges. `dst_n` is non-zero (target dims validated) and a
/// single-sample axis (`src_n == 1`) collapses to weight 0.
fn map_axis(out: usize, dst_n: usize, src_n: usize) -> (usize, usize, u32) {
    let pos = ((2 * out as i64 + 1) * src_n as i64 * 32768) / dst_n as i64 - 32768;
    let max = ((src_n - 1) as i64) << 16;
    let pos = pos.clamp(0, max);
    let i0 = (pos >> 16) as usize;
    let i1 = (i0 + 1).min(src_n - 1);
    let frac = (pos & 0xFFFF) as u32;
    (i0, i1, frac)
}

/// Bilinear blend of the four neighbours with Q16 weights, rounded to
/// nearest. The result is a convex combination of `[0, 255]` samples so it
/// needs no clamping.
fn bilerp(p00: u8, p10: u8, p01: u8, p11: u8, fx: u32, fy: u32) -> u8 {
    let (fx, fy) = (fx as i64, fy as i64);
    let one = 1i64 << 16;
    let top = p00 as i64 * (one - fx) + p10 as i64 * fx;
    let bot = p01 as i64 * (one - fx) + p11 as i64 * fx;
    let val = top * (one - fy) + bot * fy;
    ((val + (1i64 << 31)) >> 32) as u8
}

/// Bilinear-resample one `channels`-interleaved plane from `src_w x src_h`
/// to `dst_w x dst_h`. NV12's UV plane uses `channels = 2` so U and V
/// resample together under one set of weights; every other plane is
/// single-channel.
fn resample_plane(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    channels: usize,
) -> Vec<u8> {
    let mut dst = vec![0u8; dst_w * dst_h * channels];
    let cols: Vec<(usize, usize, u32)> = (0..dst_w).map(|ox| map_axis(ox, dst_w, src_w)).collect();
    for oy in 0..dst_h {
        let (y0, y1, fy) = map_axis(oy, dst_h, src_h);
        let (row0, row1) = (y0 * src_w, y1 * src_w);
        for (ox, &(x0, x1, fx)) in cols.iter().enumerate() {
            let dbase = (oy * dst_w + ox) * channels;
            for ch in 0..channels {
                let p00 = src[(row0 + x0) * channels + ch];
                let p10 = src[(row0 + x1) * channels + ch];
                let p01 = src[(row1 + x0) * channels + ch];
                let p11 = src[(row1 + x1) * channels + ch];
                dst[dbase + ch] = bilerp(p00, p10, p01, p11, fx, fy);
            }
        }
    }
    dst
}

/// Resample one frame from `in_w x in_h` to `out_w x out_h`, preserving
/// `format`. `src` is validated to hold the input frame; all dims are even
/// when the format is 4:2:0. Equal in/out dims short-circuit to a copy so
/// an identity scale is exact.
fn scale(
    src: &[u8],
    format: RawVideoFormat,
    in_w: usize,
    in_h: usize,
    out_w: usize,
    out_h: usize,
) -> Box<[u8]> {
    if in_w == out_w && in_h == out_h {
        return src[..frame_byte_size(format, in_w as u32, in_h as u32)].into();
    }
    match format {
        RawVideoFormat::Rgba8 | RawVideoFormat::Bgra8 => {
            resample_plane(src, in_w, in_h, out_w, out_h, 4).into_boxed_slice()
        }
        RawVideoFormat::Nv12 => {
            let luma_in = in_w * in_h;
            let chroma_in = (in_w / 2) * (in_h / 2) * 2;
            let mut out = resample_plane(&src[..luma_in], in_w, in_h, out_w, out_h, 1);
            let chroma = resample_plane(
                &src[luma_in..luma_in + chroma_in],
                in_w / 2,
                in_h / 2,
                out_w / 2,
                out_h / 2,
                2,
            );
            out.extend_from_slice(&chroma);
            out.into_boxed_slice()
        }
        // YUYV is input-only / not produced here; negotiation never admits it.
        RawVideoFormat::Yuyv => unreachable!("videoscale: YUYV is not negotiated"),
        // The fully-planar family (I420 / I422 / I444 at 8 / 10 / 12-bit): resample
        // each plane at its subsampled geometry. 10/12-bit samples are interpolated
        // as little-endian `u16` (byte-wise bilerp would blend the high and low byte
        // of a sample independently and corrupt it).
        f => {
            let (hs, vs) = f.chroma_shift().expect("planar format");
            let bps = f.bytes_per_sample();
            let planes = planar_planes(f, in_w, in_h);
            let resample = |plane: &[u8], sw, sh, dw, dh| {
                if bps == 2 {
                    resample_plane16(plane, sw, sh, dw, dh)
                } else {
                    resample_plane(plane, sw, sh, dw, dh, 1)
                }
            };
            let (ocw, och) = (out_w.div_ceil(1 << hs), out_h.div_ceil(1 << vs));
            let mut out = resample(&src[..planes[1].0], in_w, in_h, out_w, out_h);
            for (off, pw, ph) in [planes[1], planes[2]] {
                let plane = resample(&src[off..off + pw * ph * bps], pw, ph, ocw, och);
                out.extend_from_slice(&plane);
            }
            out.into_boxed_slice()
        }
    }
}

/// Bilinear blend of four `u16` neighbours with Q16 weights, rounded to nearest;
/// the high-bit-depth analog of [`bilerp`]. The result is a convex combination of
/// the inputs so it stays in range.
fn bilerp16(p00: u32, p10: u32, p01: u32, p11: u32, fx: u32, fy: u32) -> u16 {
    let (fx, fy) = (fx as i64, fy as i64);
    let one = 1i64 << 16;
    let top = p00 as i64 * (one - fx) + p10 as i64 * fx;
    let bot = p01 as i64 * (one - fx) + p11 as i64 * fx;
    let val = top * (one - fy) + bot * fy;
    ((val + (1i64 << 31)) >> 32) as u16
}

/// Bilinear-resample one single-channel plane of little-endian `u16` samples
/// (a 10/12-bit luma or chroma plane) from `src_w x src_h` to `dst_w x dst_h`.
fn resample_plane16(src: &[u8], src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Vec<u8> {
    let rd = |i: usize| u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]) as u32;
    let cols: Vec<(usize, usize, u32)> = (0..dst_w).map(|ox| map_axis(ox, dst_w, src_w)).collect();
    let mut dst = vec![0u8; dst_w * dst_h * 2];
    for oy in 0..dst_h {
        let (y0, y1, fy) = map_axis(oy, dst_h, src_h);
        let (row0, row1) = (y0 * src_w, y1 * src_w);
        for (ox, &(x0, x1, fx)) in cols.iter().enumerate() {
            let v = bilerp16(
                rd(row0 + x0),
                rd(row0 + x1),
                rd(row1 + x0),
                rd(row1 + x1),
                fx,
                fy,
            );
            let o = (oy * dst_w + ox) * 2;
            dst[o..o + 2].copy_from_slice(&v.to_le_bytes());
        }
    }
    dst
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
    fn bilinear_upscale_interpolates_endpoints() {
        // one channel, 2px -> 4px: half-pixel mapping samples the two
        // endpoints and the 1/4, 3/4 interior points.
        let src = [0u8, 100];
        let out = resample_plane(&src, 2, 1, 4, 1, 1);
        assert_eq!(&out[..], &[0, 25, 75, 100]);
    }

    #[test]
    fn resample16_interpolates_u16_endpoints() {
        // Same 2px -> 4px mapping as the u8 case, but on LE-u16 samples [0, 1000]:
        // the interior points land at 1/4 and 3/4 (250, 750), proving samples are
        // interpolated as whole u16 values, not byte-wise.
        let src: Vec<u8> = [0u16, 1000].iter().flat_map(|s| s.to_le_bytes()).collect();
        let out = resample_plane16(&src, 2, 1, 4, 1);
        let got: Vec<u16> = out
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(got, vec![0, 250, 750, 1000]);
    }

    #[test]
    fn scales_high_bit_depth_4_2_0() {
        // 4x4 I420p10 (LE u16): luma 16 samples, two 2x2 chroma planes. Downscale to
        // 2x2 and check the buffer is a tight 2x2 I420p10 (4 + 2*1 samples, 2 bytes).
        let (w, h) = (4usize, 4usize);
        let n = w * h + 2 * (w / 2) * (h / 2);
        let src: Vec<u8> = (0..n as u16).flat_map(|s| s.to_le_bytes()).collect();
        let out = scale(&src, RawVideoFormat::I420p10, w, h, 2, 2);
        // tight 2x2 I420 = 4 luma + 2x(1x1) chroma = 6 samples, 2 bytes each.
        assert_eq!(out.len(), 6 * 2);
    }

    #[test]
    fn identity_scale_is_exact_copy() {
        let src: Vec<u8> = (0..64u32 * 48 * 4).map(|i| (i & 0xFF) as u8).collect();
        let out = scale(&src, RawVideoFormat::Rgba8, 64, 48, 64, 48);
        assert_eq!(&out[..], &src[..]);
    }

    #[test]
    fn output_byte_sizes_match_format() {
        let rgba: Vec<u8> = vec![0; 8 * 8 * 4];
        assert_eq!(
            scale(&rgba, RawVideoFormat::Rgba8, 8, 8, 4, 4).len(),
            4 * 4 * 4
        );
        let nv12: Vec<u8> = vec![0; 8 * 8 * 3 / 2];
        assert_eq!(
            scale(&nv12, RawVideoFormat::Nv12, 8, 8, 4, 4).len(),
            4 * 4 * 3 / 2
        );
        let i420: Vec<u8> = vec![0; 8 * 8 * 3 / 2];
        assert_eq!(
            scale(&i420, RawVideoFormat::I420, 8, 8, 4, 4).len(),
            4 * 4 * 3 / 2
        );
    }

    #[test]
    fn nv12_chroma_resamples_per_plane() {
        // 2x2 NV12: 4 luma + one UV pair. Upscale to 4x4: luma fills 16
        // bytes, chroma becomes 2x2 with the single (u,v) replicated.
        let nv12 = [10u8, 20, 30, 40, 70, 200];
        let out = scale(&nv12, RawVideoFormat::Nv12, 2, 2, 4, 4);
        assert_eq!(out.len(), 4 * 4 + 2 * 2 * 2);
        // the lone chroma sample replicates across the 2x2 upscaled chroma.
        for pair in out[16..].chunks_exact(2) {
            assert_eq!(pair, &[70, 200]);
        }
    }

    #[test]
    fn derived_output_maps_to_target_dims() {
        let scaler = VideoScale::new(64, 32);
        let CapsConstraint::DerivedCoupled {
            derive: f,
            passthrough,
        } = scaler.caps_constraint_as_transform()
        else {
            panic!("expected DerivedCoupled");
        };
        assert_eq!(
            passthrough,
            PassthroughFields::NONE.with_format().with_framerate()
        );
        let out = f(&rgba_caps(320, 240));
        assert_eq!(
            out.alternatives(),
            &[Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Fixed(64),
                height: Dim::Fixed(32),
                framerate: Rate::Fixed(30 << 16),
            }]
        );
        // compressed input is not scalable
        let h264 = Caps::CompressedVideo {
            codec: g2g_core::VideoCodec::H264,
            width: Dim::Fixed(320),
            height: Dim::Fixed(240),
            framerate: Rate::Any,
        };
        assert!(f(&h264).is_empty());
    }

    #[test]
    fn derived_output_rejects_odd_target_for_yuv420() {
        let scaler = VideoScale::new(63, 32);
        let CapsConstraint::DerivedCoupled {
            derive: f,
            passthrough,
        } = scaler.caps_constraint_as_transform()
        else {
            panic!("expected DerivedCoupled");
        };
        assert_eq!(
            passthrough,
            PassthroughFields::NONE.with_format().with_framerate()
        );
        let nv12_in = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(320),
            height: Dim::Fixed(240),
            framerate: Rate::Any,
        };
        assert!(
            f(&nv12_in).is_empty(),
            "odd target width is invalid for 4:2:0"
        );
        // a packed format with the same odd target is fine
        assert!(!f(&rgba_caps(320, 240)).is_empty());
    }

    #[test]
    fn configure_rejects_odd_dims_for_yuv420() {
        let nv12 = |w, h| Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Any,
        };
        // odd target into a 4:2:0 stream fails
        let mut s = VideoScale::new(63, 32);
        assert_eq!(
            s.configure_pipeline(&nv12(320, 240))
                .expect_err("odd target"),
            G2gError::CapsMismatch
        );
        // odd input dims fail too
        let mut s = VideoScale::new(64, 32);
        assert_eq!(
            s.configure_pipeline(&nv12(321, 240))
                .expect_err("odd input"),
            G2gError::CapsMismatch
        );
        // even in / even out is accepted
        let mut s = VideoScale::new(64, 32);
        assert!(s.configure_pipeline(&nv12(320, 240)).is_ok());
        // packed formats accept odd dims
        let mut s = VideoScale::new(63, 31);
        assert!(s.configure_pipeline(&rgba_caps(321, 241)).is_ok());
    }

    #[test]
    fn smooth_gradient_round_trips_above_30db() {
        // downscale then upscale a smooth gradient and check the mean
        // squared error stays under 65, which is PSNR > 30 dB
        // (10*log10(255^2/65) > 30). Smooth content survives bilinear.
        let (w, h) = (256usize, 192usize);
        let mut src = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let p = (y * w + x) * 4;
                src[p] = (x * 255 / (w - 1)) as u8;
                src[p + 1] = (y * 255 / (h - 1)) as u8;
                src[p + 2] = ((x + y) * 255 / (w + h - 2)) as u8;
                src[p + 3] = 255;
            }
        }
        let down = scale(&src, RawVideoFormat::Rgba8, w, h, w / 2, h / 2);
        let up = scale(&down, RawVideoFormat::Rgba8, w / 2, h / 2, w, h);
        let mse: f64 = src
            .iter()
            .zip(up.iter())
            .map(|(&a, &b)| {
                let d = a as f64 - b as f64;
                d * d
            })
            .sum::<f64>()
            / (src.len() as f64);
        assert!(mse < 65.0, "round-trip MSE {mse} too high (PSNR < 30 dB)");
    }
}
