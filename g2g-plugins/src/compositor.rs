//! Software video compositor (M93): overlays N raw RGBA8 input streams onto one
//! output canvas at configurable positions, z-order, and per-pad alpha, with
//! alpha blending. The `videomixer` / `compositor` analog (picture-in-picture,
//! multi-camera grids, sub-window UIs). Our `mux` is a fan-in *multiplexer*
//! (interleaving encoded tracks); this is a fan-in *mixer* (combining raw
//! pixels into one frame).
//!
//! CPU, `no_std` baseline like the other raw-video transforms
//! (videoconvert/videoscale/...); a wgpu GPU companion is a later follow-up.
//! The output format is chosen at construction ([`Compositor::with_format`]):
//! RGBA8 (default, packed source-over with per-pixel alpha) or 8-bit
//! NV12 / I420 / I422 / I444 (mixed plane-by-plane with the scalar per-pad alpha,
//! no RGBA round-trip). Every input must match that format (put a `VideoConvert`
//! upstream otherwise). Geometry per input is whatever each negotiates; the
//! output canvas size and framerate are fixed at construction. For a subsampled
//! YUV format, overlay positions and sizes are aligned down to even so the
//! chroma planes stay on sample boundaries.
//!
//! **Cadence:** input 0 is the timing driver (the background / main stream).
//! One composited output frame is emitted per input-0 frame, overlaying the
//! latest frame cached from every other input. Each overlay updates
//! independently as new frames land, so a live overlay animates at its own rate.
//! To emit at a different constant output rate, put a `VideoRate` downstream
//! (`compositor ! videorate`); the compositor stamps each output frame with
//! input 0's PTS, so videorate resamples the cadence without any compositor-side
//! frame-rate conversion.
//!
//! **Startup:** inputs start asynchronously and an overlay branch (camera warm-up,
//! extra transforms) can lag the background, in the extreme starting only after a
//! short background has fully drained. So at startup the compositor buffers
//! input-0 frames (bounded by [`PENDING_CAP`]) until every overlay has delivered
//! its first frame, then flushes them composited with the overlays and runs live.
//! Two failure modes are avoided: it must not block the background forever on a
//! slow overlay (so on buffer overflow the oldest input-0 frame is emitted
//! *overlay-less* rather than held or dropped, keeping output flowing and losing
//! no frames), and once primed it must not keep reusing a single stale overlay
//! frame (so live frames composite the latest overlay, not a frozen one).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use crate::paint::blend_px;
use crate::pixel::{frame_byte_size, planar_planes};
use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError, MemoryDomain,
    MultiInputElement, OutputSink, PipelinePacket, Rate, RawVideoFormat,
};

/// Placement of one input stream on the output canvas.
#[derive(Debug, Clone, Copy)]
pub struct CompositorPad {
    /// Left edge on the canvas, in pixels. May be negative (clipped at the left).
    pub xpos: i32,
    /// Top edge on the canvas, in pixels. May be negative (clipped at the top).
    pub ypos: i32,
    /// Paint order: lower z-order is painted first (further back). Ties break by
    /// input index, so input 0 is the backmost among equal z-orders.
    pub zorder: u32,
    /// Per-pad alpha 0..=255, multiplied with each pixel's source alpha. 255 is
    /// fully opaque (modulo the source's own alpha channel).
    pub alpha: u8,
    /// On-canvas size `(width, height)` to scale this input to as it composites.
    /// `None` draws the input at its native geometry; `Some` resamples it
    /// (bilinear), so a downscaled camera needs no upstream `VideoScale`.
    pub size: Option<(u32, u32)>,
}

impl CompositorPad {
    /// An opaque pad at `(xpos, ypos)`, z-order 0, drawn at native size.
    pub fn at(xpos: i32, ypos: i32) -> Self {
        Self {
            xpos,
            ypos,
            zorder: 0,
            alpha: 255,
            size: None,
        }
    }

    /// Set the paint order (lower is painted first / further back).
    pub fn with_zorder(mut self, zorder: u32) -> Self {
        self.zorder = zorder;
        self
    }

    /// Set the per-pad alpha (0 transparent, 255 opaque).
    pub fn with_alpha(mut self, alpha: u8) -> Self {
        self.alpha = alpha;
        self
    }

    /// Scale this input to `width` x `height` on the canvas (bilinear), instead
    /// of compositing it at its native geometry.
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.size = Some((width, height));
        self
    }
}

#[derive(Debug)]
pub struct Compositor {
    out_w: u32,
    out_h: u32,
    /// Output (and required input) pixel format. RGBA8 blends packed with a
    /// per-pixel source alpha; the YUV formats (NV12 / I420 / I422 / I444, 8-bit)
    /// blend per-plane with the scalar per-pad alpha and no RGBA round-trip.
    format: RawVideoFormat,
    framerate_q16: u32,
    /// Per-input placement; `pads.len()` is the input count.
    pads: Vec<CompositorPad>,
    /// Per-input configured geometry `(width, height)`, set at negotiation.
    inputs: Vec<Option<(u32, u32)>>,
    /// Per-overlay (input != 0) latest RGBA8 frame, overwritten as frames
    /// arrive. Index 0 is unused: input 0 composites from the in-flight frame.
    latest: Vec<Option<Box<[u8]>>>,
    /// True once every overlay input has delivered at least one frame (or there
    /// are no overlays). Until then the compositor is in startup, buffering
    /// input-0 frames in [`pending`](Self::pending) so a late-starting overlay
    /// still appears.
    primed: bool,
    /// Startup buffer of input-0 frames awaiting the first overlay, bounded to
    /// [`PENDING_CAP`]. On overflow the oldest is emitted overlay-less (output
    /// keeps flowing, no frame is dropped); on prime the rest flush composited
    /// with the now-available overlays. Empty once primed.
    pending: alloc::collections::VecDeque<(FrameTiming, Box<[u8]>)>,
    /// The canvas fill behind all inputs (RGBA8), default opaque black.
    background: [u8; 4],
    emitted: u64,
}

/// Max input-0 frames buffered during startup before output begins flowing
/// overlay-less (bounds startup memory and latency).
const PENDING_CAP: usize = 8;

impl Compositor {
    /// A compositor producing an `out_w` x `out_h` RGBA8 canvas at 30 fps, with
    /// one `CompositorPad` per input (input 0 is the timing driver). Panics if
    /// `pads` is empty.
    pub fn new(out_w: u32, out_h: u32, pads: Vec<CompositorPad>) -> Self {
        assert!(!pads.is_empty(), "Compositor needs at least one input");
        let n = pads.len();
        Self {
            out_w,
            out_h,
            format: RawVideoFormat::Rgba8,
            framerate_q16: 30 << 16,
            pads,
            inputs: vec![None; n],
            latest: vec![None; n],
            // No overlays (single input) means nothing to wait for: start live.
            primed: n == 1,
            pending: alloc::collections::VecDeque::new(),
            background: [0, 0, 0, 255],
            emitted: 0,
        }
    }

    /// Set the output framerate in nominal fps (stored Q16). This labels the
    /// output caps; the emit cadence follows input 0's frames. To resample the
    /// output to a different constant rate, put a `VideoRate` downstream
    /// (`compositor ! videorate`), which repeats / drops to the target rate off
    /// the per-frame PTS the compositor stamps.
    pub fn with_framerate(mut self, fps: u32) -> Self {
        self.framerate_q16 = fps << 16;
        self
    }

    /// Set the output (and required input) pixel format. Every input must arrive
    /// in this format (put a `VideoConvert` upstream otherwise), so a YUV mix
    /// composites planar without an RGBA round-trip. Supports RGBA8 (the default)
    /// and 8-bit NV12 / I420 / I422 / I444; panics on any other format.
    pub fn with_format(mut self, format: RawVideoFormat) -> Self {
        let ok = matches!(format, RawVideoFormat::Rgba8)
            || (format.bytes_per_sample() == 1
                && (matches!(format, RawVideoFormat::Nv12) || format.is_planar_yuv()));
        assert!(
            ok,
            "Compositor supports RGBA8 and 8-bit NV12/I420/I422/I444, got {format:?}"
        );
        self.format = format;
        self
    }

    /// Set the RGBA8 background the inputs composite over (default opaque black).
    /// Shows wherever no input covers the canvas.
    pub fn with_background(mut self, rgba: [u8; 4]) -> Self {
        self.background = rgba;
        self
    }

    /// Number of composited frames emitted so far (one per input-0 frame).
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    fn output(&self) -> Caps {
        Caps::RawVideo {
            format: self.format,
            width: Dim::Fixed(self.out_w),
            height: Dim::Fixed(self.out_h),
            framerate: Rate::Fixed(self.framerate_q16),
        }
    }

    /// The format the compositor accepts on every input, at any geometry.
    fn accepted(&self) -> Caps {
        Caps::RawVideo {
            format: self.format,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }
    }

    /// Composite onto a fresh background-filled canvas in z-order. Input 0 uses
    /// `base0` (the frame currently driving output); every other input uses its
    /// latest cached frame. Dispatches to the packed (RGBA) or planar (YUV) path
    /// by the output format.
    fn compose(&self, base0: &[u8]) -> Box<[u8]> {
        match self.format {
            RawVideoFormat::Rgba8 | RawVideoFormat::Bgra8 => self.compose_packed(base0),
            _ => self.compose_planar(base0),
        }
    }

    /// Packed RGBA compositing: source-over with per-pixel source alpha.
    fn compose_packed(&self, base0: &[u8]) -> Box<[u8]> {
        let (cw, ch) = (self.out_w as usize, self.out_h as usize);
        let mut canvas = vec![0u8; cw * ch * 4];
        for px in canvas.chunks_exact_mut(4) {
            px.copy_from_slice(&self.background);
        }
        // Paint order: z-order ascending, ties by input index (input 0 backmost).
        let mut order: Vec<usize> = (0..self.pads.len()).collect();
        order.sort_by_key(|&i| (self.pads[i].zorder, i));
        for i in order {
            let Some((w, h)) = self.inputs[i] else {
                continue;
            };
            let src: &[u8] = if i == 0 {
                base0
            } else {
                match self.latest[i].as_deref() {
                    Some(s) => s,
                    None => continue,
                }
            };
            let pad = self.pads[i];
            let (sw, sh) = (w as usize, h as usize);
            let (dw, dh) = pad
                .size
                .map(|(dw, dh)| (dw as usize, dh as usize))
                .unwrap_or((sw, sh));
            if (dw, dh) == (sw, sh) {
                blend_over(
                    &mut canvas,
                    cw,
                    ch,
                    src,
                    sw,
                    sh,
                    pad.xpos,
                    pad.ypos,
                    pad.alpha,
                );
            } else {
                blend_over_scaled(
                    &mut canvas,
                    cw,
                    ch,
                    src,
                    sw,
                    sh,
                    pad.xpos,
                    pad.ypos,
                    dw,
                    dh,
                    pad.alpha,
                );
            }
        }
        canvas.into_boxed_slice()
    }

    /// Planar / semi-planar YUV compositing: each plane (Y, then the two chroma
    /// planes at the format's subsampling) is blended independently with the
    /// scalar per-pad alpha, so a YUV mix needs no RGBA round-trip. Overlay
    /// positions and sizes are aligned to even for the subsampled chroma.
    fn compose_planar(&self, base0: &[u8]) -> Box<[u8]> {
        let (w, h) = (self.out_w as usize, self.out_h as usize);
        let mut canvas = vec![0u8; frame_byte_size(self.format, self.out_w, self.out_h)];

        let dst_chans = channels(self.format, w, h);
        let bg = rgba_to_yuv(self.background);
        for (chan, &val) in dst_chans.iter().zip(bg.iter()) {
            fill_channel(&mut canvas, chan, val);
        }

        // Paint order: z-order ascending, ties by input index (input 0 backmost).
        let mut order: Vec<usize> = (0..self.pads.len()).collect();
        order.sort_by_key(|&i| (self.pads[i].zorder, i));
        for i in order {
            let Some((sw, sh)) = self.inputs[i] else {
                continue;
            };
            let src: &[u8] = if i == 0 {
                base0
            } else {
                match self.latest[i].as_deref() {
                    Some(s) => s,
                    None => continue,
                }
            };
            let pad = self.pads[i];
            let src_chans = channels(self.format, sw as usize, sh as usize);
            // even-align so a subsampled plane's placement stays on a chroma
            // sample (harmless for the full-res luma plane).
            let (x0, y0) = (pad.xpos & !1, pad.ypos & !1);
            let scaled = pad
                .size
                .map(|(dw, dh)| ((dw & !1) as usize, (dh & !1) as usize));
            for (dc, sc) in dst_chans.iter().zip(src_chans.iter()) {
                let (cx, cy) = (x0 >> dc.hs, y0 >> dc.vs);
                match scaled {
                    None => blend_channel(&mut canvas, dc, src, sc, cx, cy, pad.alpha),
                    Some((dw, dh)) => blend_channel_scaled(
                        &mut canvas,
                        dc,
                        src,
                        sc,
                        cx,
                        cy,
                        dw >> dc.hs,
                        dh >> dc.vs,
                        pad.alpha,
                    ),
                }
            }
        }
        canvas.into_boxed_slice()
    }

    /// Wrap composited `canvas` bytes as the next output frame, advancing the
    /// output sequence counter.
    fn output_frame(&mut self, canvas: Box<[u8]>, timing: FrameTiming) -> Frame {
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(canvas)),
            timing,
            sequence: self.emitted,
            meta: Default::default(),
        };
        self.emitted += 1;
        frame
    }
}

/// Alpha-blend a `sw` x `sh` RGBA8 source onto a `cw` x `ch` RGBA8 canvas with
/// its top-left at `(x0, y0)` (may be negative), modulating source alpha by
/// `galpha`. Straight "source-over" compositing, integer math; pixels outside
/// the canvas are clipped. The arguments are the canvas + source geometry and
/// the placement: a flat parameter list keeps this inner loop allocation-free.
#[allow(clippy::too_many_arguments)]
fn blend_over(
    canvas: &mut [u8],
    cw: usize,
    ch: usize,
    src: &[u8],
    sw: usize,
    sh: usize,
    x0: i32,
    y0: i32,
    galpha: u8,
) {
    for sy in 0..sh {
        let dy = y0 + sy as i32;
        if dy < 0 || dy as usize >= ch {
            continue;
        }
        for sx in 0..sw {
            let dx = x0 + sx as i32;
            if dx < 0 || dx as usize >= cw {
                continue;
            }
            let s = (sy * sw + sx) * 4;
            let d = (dy as usize * cw + dx as usize) * 4;
            let px = [src[s], src[s + 1], src[s + 2], src[s + 3]];
            blend_px(canvas, d, px, galpha);
        }
    }
}

/// Alpha-blend a `sw` x `sh` RGBA8 source onto the canvas, resampled (bilinear)
/// to a `dw` x `dh` rectangle with its top-left at `(x0, y0)`. Same source-over
/// math as [`blend_over`], with integer fixed-point sampling (no float intrinsics
/// for the `no_std` baseline). Pixels outside the canvas are clipped.
#[allow(clippy::too_many_arguments)]
fn blend_over_scaled(
    canvas: &mut [u8],
    cw: usize,
    ch: usize,
    src: &[u8],
    sw: usize,
    sh: usize,
    x0: i32,
    y0: i32,
    dw: usize,
    dh: usize,
    galpha: u8,
) {
    if sw == 0 || sh == 0 || dw == 0 || dh == 0 {
        return;
    }
    // Center-aligned source coordinate for a destination index, in Q16 fixed
    // point: ((d + 0.5) * s / dst - 0.5). Clamped into the source extent.
    let map = |d: usize, s: usize, dst: usize, max: i64| -> i64 {
        let q = ((2 * d as i64 + 1) * s as i64 * 32768) / dst as i64 - 32768;
        q.clamp(0, max)
    };
    let max_x = ((sw - 1) as i64) << 16;
    let max_y = ((sh - 1) as i64) << 16;
    for ddy in 0..dh {
        let dy = y0 + ddy as i32;
        if dy < 0 || dy as usize >= ch {
            continue;
        }
        let fy = map(ddy, sh, dh, max_y);
        let y0i = (fy >> 16) as usize;
        let y1i = (y0i + 1).min(sh - 1);
        let ty = ((fy >> 8) & 0xFF) as u32;
        for ddx in 0..dw {
            let dx = x0 + ddx as i32;
            if dx < 0 || dx as usize >= cw {
                continue;
            }
            let fx = map(ddx, sw, dw, max_x);
            let x0i = (fx >> 16) as usize;
            let x1i = (x0i + 1).min(sw - 1);
            let tx = ((fx >> 8) & 0xFF) as u32;
            // Bilinear: interpolate the 2x2 source neighbourhood per channel.
            let i00 = (y0i * sw + x0i) * 4;
            let i01 = (y0i * sw + x1i) * 4;
            let i10 = (y1i * sw + x0i) * 4;
            let i11 = (y1i * sw + x1i) * 4;
            let mut px = [0u8; 4];
            for c in 0..4 {
                let top = src[i00 + c] as u32 * (256 - tx) + src[i01 + c] as u32 * tx;
                let bot = src[i10 + c] as u32 * (256 - tx) + src[i11 + c] as u32 * tx;
                px[c] = ((top * (256 - ty) + bot * ty) >> 16) as u8;
            }
            let d = (dy as usize * cw + dx as usize) * 4;
            blend_px(canvas, d, px, galpha);
        }
    }
}

/// One output/source plane of a YUV frame, addressed uniformly across NV12
/// (interleaved chroma, `stride == 2`) and the fully-planar family (`stride == 1`).
/// `hs` / `vs` are this plane's subsampling shift relative to luma.
#[derive(Debug, Clone, Copy)]
struct Chan {
    /// Byte offset of the plane's first sample.
    base: usize,
    /// Plane dimensions in samples.
    w: usize,
    h: usize,
    /// Bytes between successive samples in a row (2 for NV12 chroma, else 1).
    stride: usize,
    /// Byte offset of this channel within an interleaved sample (V in NV12 = 1).
    off: usize,
    /// Subsampling shift relative to luma (1 = half-resolution, 0 = full).
    hs: u32,
    vs: u32,
}

/// The Y, U, V plane layout of `format` at `w x h`. Handles NV12's interleaved
/// chroma and the fully-planar formats through the shared `pixel` helpers.
fn channels(format: RawVideoFormat, w: usize, h: usize) -> [Chan; 3] {
    match format {
        RawVideoFormat::Nv12 => {
            let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
            [
                Chan {
                    base: 0,
                    w,
                    h,
                    stride: 1,
                    off: 0,
                    hs: 0,
                    vs: 0,
                },
                Chan {
                    base: w * h,
                    w: cw,
                    h: ch,
                    stride: 2,
                    off: 0,
                    hs: 1,
                    vs: 1,
                },
                Chan {
                    base: w * h,
                    w: cw,
                    h: ch,
                    stride: 2,
                    off: 1,
                    hs: 1,
                    vs: 1,
                },
            ]
        }
        _ => {
            let p = planar_planes(format, w, h);
            let (hs, vs) = format.chroma_shift().expect("planar YUV");
            [
                Chan {
                    base: p[0].0,
                    w: p[0].1,
                    h: p[0].2,
                    stride: 1,
                    off: 0,
                    hs: 0,
                    vs: 0,
                },
                Chan {
                    base: p[1].0,
                    w: p[1].1,
                    h: p[1].2,
                    stride: 1,
                    off: 0,
                    hs,
                    vs,
                },
                Chan {
                    base: p[2].0,
                    w: p[2].1,
                    h: p[2].2,
                    stride: 1,
                    off: 0,
                    hs,
                    vs,
                },
            ]
        }
    }
}

/// Rec.601 full-range (JFIF) conversion of the RGBA background to `[Y, U, V]`.
fn rgba_to_yuv(rgba: [u8; 4]) -> [u8; 3] {
    let (r, g, b) = (rgba[0] as f32, rgba[1] as f32, rgba[2] as f32);
    let clamp = |v: f32| v.clamp(0.0, 255.0) as u8;
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let u = -0.168_736 * r - 0.331_264 * g + 0.5 * b + 128.0;
    let v = 0.5 * r - 0.418_688 * g - 0.081_312 * b + 128.0;
    [clamp(y), clamp(u), clamp(v)]
}

/// Fill one plane with a constant sample value.
fn fill_channel(buf: &mut [u8], c: &Chan, val: u8) {
    for y in 0..c.h {
        for x in 0..c.w {
            buf[c.base + (y * c.w + x) * c.stride + c.off] = val;
        }
    }
}

/// Blend one source sample over a destination sample with scalar alpha (0..=255).
#[inline]
fn blend_u8(dst: u8, src: u8, alpha: u8) -> u8 {
    let a = alpha as u32;
    ((src as u32 * a + dst as u32 * (255 - a) + 127) / 255) as u8
}

/// Alpha-blend a source plane onto a destination plane at channel-space
/// `(x0, y0)`, with the scalar per-pad `alpha`. Both planes are the same channel
/// of the same format; samples outside the destination are clipped.
#[allow(clippy::too_many_arguments)]
fn blend_channel(dst: &mut [u8], dc: &Chan, src: &[u8], sc: &Chan, x0: i32, y0: i32, alpha: u8) {
    for sy in 0..sc.h {
        let dy = y0 + sy as i32;
        if dy < 0 || dy as usize >= dc.h {
            continue;
        }
        for sx in 0..sc.w {
            let dx = x0 + sx as i32;
            if dx < 0 || dx as usize >= dc.w {
                continue;
            }
            let s = sc.base + (sy * sc.w + sx) * sc.stride + sc.off;
            let d = dc.base + (dy as usize * dc.w + dx as usize) * dc.stride + dc.off;
            dst[d] = blend_u8(dst[d], src[s], alpha);
        }
    }
}

/// Alpha-blend a source plane onto a destination plane, bilinearly resampled to
/// `dw` x `dh` (channel-space) at `(x0, y0)`. Integer fixed-point sampling, same
/// mapping as the packed [`blend_over_scaled`]; clipped to the destination.
#[allow(clippy::too_many_arguments)]
fn blend_channel_scaled(
    dst: &mut [u8],
    dc: &Chan,
    src: &[u8],
    sc: &Chan,
    x0: i32,
    y0: i32,
    dw: usize,
    dh: usize,
    alpha: u8,
) {
    if sc.w == 0 || sc.h == 0 || dw == 0 || dh == 0 {
        return;
    }
    let map = |d: usize, s: usize, dst_dim: usize, max: i64| -> i64 {
        let q = ((2 * d as i64 + 1) * s as i64 * 32768) / dst_dim as i64 - 32768;
        q.clamp(0, max)
    };
    let max_x = ((sc.w - 1) as i64) << 16;
    let max_y = ((sc.h - 1) as i64) << 16;
    let sample = |xi: usize, yi: usize| src[sc.base + (yi * sc.w + xi) * sc.stride + sc.off] as u32;
    for ddy in 0..dh {
        let dy = y0 + ddy as i32;
        if dy < 0 || dy as usize >= dc.h {
            continue;
        }
        let fy = map(ddy, sc.h, dh, max_y);
        let y0i = (fy >> 16) as usize;
        let y1i = (y0i + 1).min(sc.h - 1);
        let ty = ((fy >> 8) & 0xFF) as u32;
        for ddx in 0..dw {
            let dx = x0 + ddx as i32;
            if dx < 0 || dx as usize >= dc.w {
                continue;
            }
            let fx = map(ddx, sc.w, dw, max_x);
            let x0i = (fx >> 16) as usize;
            let x1i = (x0i + 1).min(sc.w - 1);
            let tx = ((fx >> 8) & 0xFF) as u32;
            let top = sample(x0i, y0i) * (256 - tx) + sample(x1i, y0i) * tx;
            let bot = sample(x0i, y1i) * (256 - tx) + sample(x1i, y1i) * tx;
            let v = ((top * (256 - ty) + bot * ty) >> 16) as u8;
            let d = dc.base + (dy as usize * dc.w + dx as usize) * dc.stride + dc.off;
            dst[d] = blend_u8(dst[d], v, alpha);
        }
    }
}

impl MultiInputElement for Compositor {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        self.pads.len()
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&self.accepted())
    }

    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(self.accepted()))
    }

    fn caps_constraint_for_output(&self) -> Result<CapsConstraint<'_>, G2gError> {
        Ok(CapsConstraint::Produces(CapsSet::one(self.output())))
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        let Caps::RawVideo {
            format,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            ..
        } = absolute_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if *format != self.format {
            return Err(G2gError::CapsMismatch);
        }
        self.inputs[input] = Some((*w, *h));
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(self.output())
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let (w, h) = self.inputs[input].ok_or(G2gError::NotConfigured)?;
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let src = slice.as_slice();
                    let need = frame_byte_size(self.format, w, h);
                    if src.len() < need {
                        return Err(G2gError::CapsMismatch);
                    }
                    let bytes: Box<[u8]> = src[..need].into();

                    if input == 0 {
                        if self.primed {
                            // Live: composite this frame with the latest overlays.
                            let canvas = self.compose(&bytes);
                            let frame = self.output_frame(canvas, frame.timing);
                            out.push(PipelinePacket::DataFrame(frame)).await?;
                        } else {
                            // Startup: buffer until an overlay primes. If the
                            // buffer is full, emit the oldest overlay-less rather
                            // than drop it, so output keeps flowing and no input-0
                            // frame is lost while a slow overlay starts up.
                            if self.pending.len() == PENDING_CAP {
                                let (timing, base) = self.pending.pop_front().expect("non-empty");
                                let canvas = self.compose(&base);
                                let frame = self.output_frame(canvas, timing);
                                out.push(PipelinePacket::DataFrame(frame)).await?;
                            }
                            self.pending.push_back((frame.timing, bytes));
                        }
                    } else {
                        // Overlay: cache the latest frame; it is picked up by the
                        // next input-0 frame and updates live as more arrive.
                        self.latest[input] = Some(bytes);
                    }

                    // Priming completes when every overlay has delivered a frame.
                    // Flush the buffered input-0 frames composited against the
                    // now-available overlays, in arrival order, then go live.
                    if !self.primed && self.latest.iter().skip(1).all(|l| l.is_some()) {
                        self.primed = true;
                        let pending = core::mem::take(&mut self.pending);
                        for (timing, base) in pending {
                            let canvas = self.compose(&base);
                            let frame = self.output_frame(canvas, timing);
                            out.push(PipelinePacket::DataFrame(frame)).await?;
                        }
                    }
                }
                // A per-input caps refinement updates that input's geometry; the
                // output caps are fixed, so nothing is forwarded.
                PipelinePacket::CapsChanged(Caps::RawVideo {
                    format,
                    width: Dim::Fixed(w),
                    height: Dim::Fixed(h),
                    ..
                }) if format == self.format => {
                    // A geometry change invalidates this input's cached frame:
                    // compose() would otherwise read the old (smaller) bytes
                    // at the new dims and panic out of bounds. The fresh frame
                    // at the new size repopulates the cache.
                    if self.inputs[input] != Some((w, h)) {
                        self.latest[input] = None;
                    }
                    self.inputs[input] = Some((w, h));
                }
                // A flush on an overlay input drops its cached frame so a stale
                // overlay never lingers across a discontinuity, and re-arms
                // startup so that overlay is waited for again. A flush on input 0
                // clears any buffered startup frames (nothing else is cached).
                PipelinePacket::Flush => {
                    self.latest[input] = None;
                    if input == 0 {
                        self.pending.clear();
                    } else if self.pads.len() > 1 {
                        self.primed = false;
                    }
                }
                // Per-input Eos is informational; the runner aggregates input
                // ends and emits the single merged Eos. Segment is per-input
                // control the compositor does not remap.
                PipelinePacket::Eos | PipelinePacket::Segment(_) => {}
                // future PipelinePacket variants: no-op.
                _ => {}
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: usize, h: usize, rgba: [u8; 4]) -> Vec<u8> {
        let mut v = Vec::with_capacity(w * h * 4);
        for _ in 0..w * h {
            v.extend_from_slice(&rgba);
        }
        v
    }

    fn px(buf: &[u8], cw: usize, x: usize, y: usize) -> [u8; 4] {
        let i = (y * cw + x) * 4;
        [buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]
    }

    #[test]
    fn opaque_overlay_replaces_destination() {
        // 4x4 red canvas-fill, then a 2x2 opaque blue square at (1,1).
        let mut canvas = solid(4, 4, [255, 0, 0, 255]);
        let blue = solid(2, 2, [0, 0, 255, 255]);
        blend_over(&mut canvas, 4, 4, &blue, 2, 2, 1, 1, 255);
        assert_eq!(
            px(&canvas, 4, 0, 0),
            [255, 0, 0, 255],
            "outside the square stays red"
        );
        assert_eq!(
            px(&canvas, 4, 1, 1),
            [0, 0, 255, 255],
            "square is fully blue"
        );
        assert_eq!(px(&canvas, 4, 2, 2), [0, 0, 255, 255], "square corner blue");
        assert_eq!(
            px(&canvas, 4, 3, 3),
            [255, 0, 0, 255],
            "beyond the square stays red"
        );
    }

    #[test]
    fn half_alpha_blends_halfway() {
        // Blue over red at 50% alpha -> roughly (128, 0, 128).
        let mut canvas = solid(2, 2, [255, 0, 0, 255]);
        let blue = solid(2, 2, [0, 0, 255, 255]);
        blend_over(&mut canvas, 2, 2, &blue, 2, 2, 0, 0, 128);
        let p = px(&canvas, 2, 0, 0);
        assert!((p[0] as i32 - 127).abs() <= 2, "red ~half: {}", p[0]);
        assert_eq!(p[1], 0);
        assert!((p[2] as i32 - 128).abs() <= 2, "blue ~half: {}", p[2]);
        assert_eq!(p[3], 255, "canvas stays opaque");
    }

    #[test]
    fn negative_offset_clips_to_canvas() {
        // A 4x4 green source placed at (-2,-2): only its bottom-right 2x2 lands.
        let mut canvas = solid(4, 4, [0, 0, 0, 255]);
        let green = solid(4, 4, [0, 255, 0, 255]);
        blend_over(&mut canvas, 4, 4, &green, 4, 4, -2, -2, 255);
        assert_eq!(px(&canvas, 4, 0, 0), [0, 255, 0, 255], "top-left now green");
        assert_eq!(
            px(&canvas, 4, 1, 1),
            [0, 255, 0, 255],
            "still in the clipped region"
        );
        assert_eq!(
            px(&canvas, 4, 2, 2),
            [0, 0, 0, 255],
            "beyond the source stays black"
        );
    }

    #[test]
    fn scaled_blend_upsamples_a_solid_source() {
        // A 2x2 blue source scaled into a 4x4 region at (1,1) on a 6x6 red
        // canvas: the whole region is blue (uniform bilinear is exact), the
        // border stays red.
        let mut canvas = solid(6, 6, [255, 0, 0, 255]);
        let blue = solid(2, 2, [0, 0, 255, 255]);
        blend_over_scaled(&mut canvas, 6, 6, &blue, 2, 2, 1, 1, 4, 4, 255);
        assert_eq!(px(&canvas, 6, 0, 0), [255, 0, 0, 255], "border stays red");
        assert_eq!(
            px(&canvas, 6, 1, 1),
            [0, 0, 255, 255],
            "region top-left blue"
        );
        assert_eq!(
            px(&canvas, 6, 4, 4),
            [0, 0, 255, 255],
            "region bottom-right blue"
        );
        assert_eq!(
            px(&canvas, 6, 5, 5),
            [255, 0, 0, 255],
            "beyond the region red"
        );
    }

    #[test]
    fn pad_with_size_downscales_overlay_into_the_inset() {
        // Background 8x8 red; a native 4x4 green overlay scaled down to a 2x2
        // inset at (2,2). The inset is green, everything else red.
        let mut comp = Compositor::new(
            8,
            8,
            Vec::from([
                CompositorPad::at(0, 0),
                CompositorPad::at(2, 2).with_zorder(1).with_size(2, 2),
            ]),
        );
        comp.inputs[0] = Some((8, 8));
        comp.inputs[1] = Some((4, 4)); // native overlay geometry
        let red = solid(8, 8, [255, 0, 0, 255]);
        comp.latest[1] = Some(solid(4, 4, [0, 255, 0, 255]).into());
        let out = comp.compose(&red);
        assert_eq!(px(&out, 8, 0, 0), [255, 0, 0, 255], "background red");
        assert_eq!(px(&out, 8, 2, 2), [0, 255, 0, 255], "inset top-left green");
        assert_eq!(
            px(&out, 8, 3, 3),
            [0, 255, 0, 255],
            "inset bottom-right green"
        );
        assert_eq!(
            px(&out, 8, 4, 4),
            [255, 0, 0, 255],
            "beyond the 2x2 inset red"
        );
    }

    #[test]
    fn background_shows_where_no_input_covers() {
        // A 4x4 canvas with a blue background; input 0 is a 2x2 green frame at
        // (0,0), so only the top-left quarter is green, the rest the background.
        let mut comp = Compositor::new(4, 4, Vec::from([CompositorPad::at(0, 0)]))
            .with_background([0, 0, 255, 255]);
        comp.inputs[0] = Some((2, 2));
        let out = comp.compose(&solid(2, 2, [0, 255, 0, 255]));
        assert_eq!(
            px(&out, 4, 0, 0),
            [0, 255, 0, 255],
            "input 0 paints its 2x2"
        );
        assert_eq!(
            px(&out, 4, 3, 3),
            [0, 0, 255, 255],
            "uncovered area is the background"
        );
        // The default background stays opaque black.
        let mut def = Compositor::new(4, 4, Vec::from([CompositorPad::at(0, 0)]));
        def.inputs[0] = Some((2, 2));
        let out = def.compose(&solid(2, 2, [0, 255, 0, 255]));
        assert_eq!(
            px(&out, 4, 3, 3),
            [0, 0, 0, 255],
            "default background opaque black"
        );
    }

    #[test]
    fn zorder_paints_higher_last() {
        // Two full-canvas pads at the same position; the higher z-order wins.
        let mut comp = Compositor::new(
            2,
            2,
            Vec::from([
                CompositorPad::at(0, 0).with_zorder(1),
                CompositorPad::at(0, 0).with_zorder(5),
            ]),
        );
        comp.inputs[0] = Some((2, 2));
        comp.inputs[1] = Some((2, 2));
        let red = solid(2, 2, [255, 0, 0, 255]);
        comp.latest[1] = Some(solid(2, 2, [0, 0, 255, 255]).into());
        // input 0 (red) is passed as the base; input 1 (blue) has higher z-order.
        let out = comp.compose(&red);
        assert_eq!(
            px(&out, 2, 0, 0),
            [0, 0, 255, 255],
            "z=5 (blue) painted over z=1 (red)"
        );
    }

    /// A solid YUV frame of the given format at `w x h`, every plane filled flat.
    fn solid_yuv(format: RawVideoFormat, w: usize, h: usize, yuv: [u8; 3]) -> Vec<u8> {
        let mut buf = vec![0u8; frame_byte_size(format, w as u32, h as u32)];
        let chans = channels(format, w, h);
        for (c, &v) in chans.iter().zip(yuv.iter()) {
            fill_channel(&mut buf, c, v);
        }
        buf
    }

    /// Read the Y (c=0), U (c=1), or V (c=2) sample under luma pixel `(x, y)`.
    fn yuv_at(
        buf: &[u8],
        format: RawVideoFormat,
        w: usize,
        h: usize,
        c: usize,
        x: usize,
        y: usize,
    ) -> u8 {
        let ch = channels(format, w, h)[c];
        let (cx, cy) = (x >> ch.hs, y >> ch.vs);
        buf[ch.base + (cy * ch.w + cx) * ch.stride + ch.off]
    }

    #[test]
    fn nv12_opaque_overlay_replaces_all_planes() {
        // 8x8 NV12 base (Y50/U60/V70); a 4x4 opaque overlay (Y200/U100/V150) at
        // (2,2). Y and both chroma planes take the overlay inside, base outside.
        let mut comp = Compositor::new(
            8,
            8,
            Vec::from([
                CompositorPad::at(0, 0),
                CompositorPad::at(2, 2).with_zorder(1),
            ]),
        )
        .with_format(RawVideoFormat::Nv12);
        comp.inputs[0] = Some((8, 8));
        comp.inputs[1] = Some((4, 4));
        comp.latest[1] = Some(solid_yuv(RawVideoFormat::Nv12, 4, 4, [200, 100, 150]).into());
        let out = comp.compose(&solid_yuv(RawVideoFormat::Nv12, 8, 8, [50, 60, 70]));

        let f = RawVideoFormat::Nv12;
        assert_eq!(
            yuv_at(&out, f, 8, 8, 0, 0, 0),
            50,
            "base luma outside overlay"
        );
        assert_eq!(yuv_at(&out, f, 8, 8, 0, 2, 2), 200, "overlay luma inside");
        assert_eq!(yuv_at(&out, f, 8, 8, 1, 2, 2), 100, "overlay U inside");
        assert_eq!(yuv_at(&out, f, 8, 8, 2, 2, 2), 150, "overlay V inside");
        assert_eq!(yuv_at(&out, f, 8, 8, 1, 0, 0), 60, "base U outside");
        assert_eq!(yuv_at(&out, f, 8, 8, 2, 6, 6), 70, "base V beyond overlay");
    }

    #[test]
    fn i420_opaque_overlay_replaces_all_planes() {
        // Same as the NV12 case for the fully-planar I420 layout (separate U/V).
        let mut comp = Compositor::new(
            8,
            8,
            Vec::from([
                CompositorPad::at(0, 0),
                CompositorPad::at(2, 2).with_zorder(1),
            ]),
        )
        .with_format(RawVideoFormat::I420);
        comp.inputs[0] = Some((8, 8));
        comp.inputs[1] = Some((4, 4));
        comp.latest[1] = Some(solid_yuv(RawVideoFormat::I420, 4, 4, [200, 100, 150]).into());
        let out = comp.compose(&solid_yuv(RawVideoFormat::I420, 8, 8, [50, 60, 70]));

        let f = RawVideoFormat::I420;
        assert_eq!(yuv_at(&out, f, 8, 8, 0, 2, 2), 200, "overlay luma");
        assert_eq!(yuv_at(&out, f, 8, 8, 1, 2, 2), 100, "overlay U");
        assert_eq!(yuv_at(&out, f, 8, 8, 2, 2, 2), 150, "overlay V");
        assert_eq!(yuv_at(&out, f, 8, 8, 0, 0, 0), 50, "base luma outside");
        assert_eq!(yuv_at(&out, f, 8, 8, 2, 6, 6), 70, "base V beyond overlay");
    }

    #[test]
    fn nv12_half_alpha_blends_luma_halfway() {
        // Overlay Y200 over base Y50 at alpha 128 -> ~125.
        let mut comp = Compositor::new(
            4,
            4,
            Vec::from([
                CompositorPad::at(0, 0),
                CompositorPad::at(0, 0).with_zorder(1).with_alpha(128),
            ]),
        )
        .with_format(RawVideoFormat::Nv12);
        comp.inputs[0] = Some((4, 4));
        comp.inputs[1] = Some((4, 4));
        comp.latest[1] = Some(solid_yuv(RawVideoFormat::Nv12, 4, 4, [200, 128, 128]).into());
        let out = comp.compose(&solid_yuv(RawVideoFormat::Nv12, 4, 4, [50, 128, 128]));
        let y = yuv_at(&out, RawVideoFormat::Nv12, 4, 4, 0, 0, 0);
        assert!(
            (y as i32 - 125).abs() <= 2,
            "half-blended luma ~125, got {y}"
        );
    }

    #[test]
    fn planar_background_fills_uncovered_area() {
        // A 4x4 NV12 canvas with a 2x2 input at (0,0): the uncovered area is the
        // default opaque-black background (Y0, U128, V128 full-range).
        let mut comp = Compositor::new(4, 4, Vec::from([CompositorPad::at(0, 0)]))
            .with_format(RawVideoFormat::Nv12);
        comp.inputs[0] = Some((2, 2));
        let out = comp.compose(&solid_yuv(RawVideoFormat::Nv12, 2, 2, [90, 40, 200]));
        let f = RawVideoFormat::Nv12;
        assert_eq!(yuv_at(&out, f, 4, 4, 0, 0, 0), 90, "input paints its luma");
        assert_eq!(yuv_at(&out, f, 4, 4, 0, 3, 3), 0, "uncovered luma is black");
        assert_eq!(yuv_at(&out, f, 4, 4, 1, 3, 3), 128, "uncovered U neutral");
        assert_eq!(yuv_at(&out, f, 4, 4, 2, 3, 3), 128, "uncovered V neutral");
    }

    #[test]
    fn planar_scaled_overlay_downsamples_into_inset() {
        // A native 4x4 solid overlay scaled to a 2x2 inset at (2,2) on an 8x8
        // NV12 base. Uniform source: the inset is exactly the overlay colour.
        let mut comp = Compositor::new(
            8,
            8,
            Vec::from([
                CompositorPad::at(0, 0),
                CompositorPad::at(2, 2).with_zorder(1).with_size(2, 2),
            ]),
        )
        .with_format(RawVideoFormat::Nv12);
        comp.inputs[0] = Some((8, 8));
        comp.inputs[1] = Some((4, 4));
        comp.latest[1] = Some(solid_yuv(RawVideoFormat::Nv12, 4, 4, [200, 100, 150]).into());
        let out = comp.compose(&solid_yuv(RawVideoFormat::Nv12, 8, 8, [50, 60, 70]));
        let f = RawVideoFormat::Nv12;
        assert_eq!(
            yuv_at(&out, f, 8, 8, 0, 2, 2),
            200,
            "inset luma is the overlay"
        );
        assert_eq!(
            yuv_at(&out, f, 8, 8, 1, 2, 2),
            100,
            "inset U is the overlay"
        );
        assert_eq!(
            yuv_at(&out, f, 8, 8, 0, 5, 5),
            50,
            "beyond the 2x2 inset is base"
        );
    }

    #[test]
    fn with_format_negotiates_the_chosen_yuv_format() {
        let mut comp = Compositor::new(640, 480, Vec::from([CompositorPad::at(0, 0)]))
            .with_format(RawVideoFormat::Nv12);
        let nv12 = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Fixed(30 << 16),
        };
        assert!(
            comp.configure_pipeline(0, &nv12).is_ok(),
            "matching NV12 input accepted"
        );
        // The output caps carry the chosen format.
        assert!(matches!(
            comp.output(),
            Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                ..
            }
        ));
        // A mismatched (RGBA) input is rejected.
        let rgba = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Fixed(30 << 16),
        };
        assert!(matches!(
            comp.configure_pipeline(0, &rgba),
            Err(G2gError::CapsMismatch)
        ));
    }

    #[test]
    fn negotiation_narrows_to_rgba_and_fixes_output() {
        let comp =
            Compositor::new(1920, 1080, Vec::from([CompositorPad::at(0, 0)])).with_framerate(60);
        assert_eq!(comp.input_count(), 1);
        // Output is the fixed canvas at the construction framerate.
        let CapsConstraint::Produces(set) = comp.caps_constraint_for_output().unwrap() else {
            panic!("expected Produces");
        };
        assert_eq!(
            set.alternatives(),
            &[Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Fixed(1920),
                height: Dim::Fixed(1080),
                framerate: Rate::Fixed(60 << 16),
            }]
        );
        // A non-RGBA input is rejected at configure.
        let nv12 = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Fixed(30 << 16),
        };
        let mut comp = comp;
        assert!(matches!(
            comp.configure_pipeline(0, &nv12),
            Err(G2gError::CapsMismatch)
        ));
    }
}
