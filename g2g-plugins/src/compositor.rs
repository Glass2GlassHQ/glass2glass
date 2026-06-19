//! Software video compositor (M93): overlays N raw RGBA8 input streams onto one
//! output canvas at configurable positions, z-order, and per-pad alpha, with
//! alpha blending. The `videomixer` / `compositor` analog (picture-in-picture,
//! multi-camera grids, sub-window UIs). Our `mux` is a fan-in *multiplexer*
//! (interleaving encoded tracks); this is a fan-in *mixer* (combining raw
//! pixels into one frame).
//!
//! CPU, `no_std` baseline like the other raw-video transforms
//! (videoconvert/videoscale/...); a wgpu GPU companion is a later follow-up.
//! All inputs and the output are RGBA8 (put a `VideoConvert` upstream of a
//! non-RGBA source). Geometry per input is whatever each negotiates; the output
//! canvas size and framerate are fixed at construction.
//!
//! **Cadence:** input 0 is the timing driver (the background / main stream).
//! One composited output frame is emitted per input-0 frame, overlaying the
//! latest frame cached from every other input; inputs that have not produced
//! yet are simply absent. This keeps output timing deterministic and matches
//! the common "background video + overlays" shape.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

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
}

impl CompositorPad {
    /// An opaque pad at `(xpos, ypos)`, z-order 0.
    pub fn at(xpos: i32, ypos: i32) -> Self {
        Self { xpos, ypos, zorder: 0, alpha: 255 }
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
}

#[derive(Debug)]
pub struct Compositor {
    out_w: u32,
    out_h: u32,
    framerate_q16: u32,
    /// Per-input placement; `pads.len()` is the input count.
    pads: Vec<CompositorPad>,
    /// Per-input configured geometry `(width, height)`, set at negotiation.
    inputs: Vec<Option<(u32, u32)>>,
    /// Per-input latest RGBA8 frame, overwritten as frames arrive.
    latest: Vec<Option<Box<[u8]>>>,
    emitted: u64,
}

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
            framerate_q16: 30 << 16,
            pads,
            inputs: vec![None; n],
            latest: vec![None; n],
            emitted: 0,
        }
    }

    /// Set the output framerate in nominal fps (stored Q16). The output cadence
    /// still follows input 0's frames; this only labels the output caps.
    pub fn with_framerate(mut self, fps: u32) -> Self {
        self.framerate_q16 = fps << 16;
        self
    }

    /// Number of composited frames emitted so far (one per input-0 frame).
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    fn output(&self) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(self.out_w),
            height: Dim::Fixed(self.out_h),
            framerate: Rate::Fixed(self.framerate_q16),
        }
    }

    /// Composite every cached input onto a fresh opaque-black canvas in z-order
    /// and return the RGBA8 bytes.
    fn compose(&self) -> Box<[u8]> {
        let (cw, ch) = (self.out_w as usize, self.out_h as usize);
        let mut canvas = vec![0u8; cw * ch * 4];
        // Opaque black background.
        for px in canvas.chunks_exact_mut(4) {
            px[3] = 255;
        }
        // Paint order: z-order ascending, ties by input index (input 0 backmost).
        let mut order: Vec<usize> = (0..self.pads.len()).collect();
        order.sort_by_key(|&i| (self.pads[i].zorder, i));
        for i in order {
            let (Some((w, h)), Some(src)) = (self.inputs[i], self.latest[i].as_deref()) else {
                continue;
            };
            let pad = self.pads[i];
            blend_over(&mut canvas, cw, ch, src, w as usize, h as usize, pad.xpos, pad.ypos, pad.alpha);
        }
        canvas.into_boxed_slice()
    }
}

/// Alpha-blend a `sw` x `sh` RGBA8 source onto a `cw` x `ch` RGBA8 canvas with
/// its top-left at `(x0, y0)` (may be negative), modulating source alpha by
/// `galpha`. Straight "source-over" compositing, integer math; pixels outside
/// the canvas are clipped.
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
            // Effective source alpha = src_a * galpha (0..=255).
            let a = (src[s + 3] as u32 * galpha as u32 + 127) / 255;
            let inv = 255 - a;
            for c in 0..3 {
                canvas[d + c] =
                    ((src[s + c] as u32 * a + canvas[d + c] as u32 * inv + 127) / 255) as u8;
            }
            // Composite the alpha channel too (keeps an opaque canvas opaque).
            canvas[d + 3] = (a + canvas[d + 3] as u32 * inv / 255) as u8;
        }
    }
}

/// RGBA8 at any geometry: the only input/output format the compositor mixes.
fn rgba_any() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
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
        upstream_caps.intersect(&rgba_any())
    }

    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(rgba_any()))
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
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            ..
        } = absolute_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
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
                    let need = (w as usize) * (h as usize) * 4;
                    if src.len() < need {
                        return Err(G2gError::CapsMismatch);
                    }
                    // Cache this input's latest frame (trimmed to its geometry).
                    self.latest[input] = Some(src[..need].into());

                    // Input 0 drives output cadence: composite and emit.
                    if input == 0 {
                        let canvas = self.compose();
                        let timing = FrameTiming {
                            duration_ns: frame.timing.duration_ns,
                            ..frame.timing
                        };
                        let composed = Frame {
                            domain: MemoryDomain::System(SystemSlice::from_boxed(canvas)),
                            timing,
                            sequence: self.emitted,
                            meta: Default::default(),
                        };
                        self.emitted += 1;
                        out.push(PipelinePacket::DataFrame(composed)).await?;
                    }
                }
                // A per-input caps refinement updates that input's geometry; the
                // output caps are fixed, so nothing is forwarded.
                PipelinePacket::CapsChanged(caps) => {
                    if let Caps::RawVideo {
                        format: RawVideoFormat::Rgba8,
                        width: Dim::Fixed(w),
                        height: Dim::Fixed(h),
                        ..
                    } = caps
                    {
                        self.inputs[input] = Some((w, h));
                    }
                }
                // A flush on an input drops its cached frame so a stale overlay
                // never lingers across a discontinuity.
                PipelinePacket::Flush => {
                    self.latest[input] = None;
                }
                // Per-input Eos is informational; the runner aggregates input
                // ends and emits the single merged Eos. Segment is per-input
                // control the compositor does not remap.
                PipelinePacket::Eos | PipelinePacket::Segment(_) => {}
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
        assert_eq!(px(&canvas, 4, 0, 0), [255, 0, 0, 255], "outside the square stays red");
        assert_eq!(px(&canvas, 4, 1, 1), [0, 0, 255, 255], "square is fully blue");
        assert_eq!(px(&canvas, 4, 2, 2), [0, 0, 255, 255], "square corner blue");
        assert_eq!(px(&canvas, 4, 3, 3), [255, 0, 0, 255], "beyond the square stays red");
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
        assert_eq!(px(&canvas, 4, 1, 1), [0, 255, 0, 255], "still in the clipped region");
        assert_eq!(px(&canvas, 4, 2, 2), [0, 0, 0, 255], "beyond the source stays black");
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
        comp.latest[0] = Some(solid(2, 2, [255, 0, 0, 255]).into());
        comp.latest[1] = Some(solid(2, 2, [0, 0, 255, 255]).into());
        let out = comp.compose();
        assert_eq!(px(&out, 2, 0, 0), [0, 0, 255, 255], "z=5 (blue) painted over z=1 (red)");
    }

    #[test]
    fn negotiation_narrows_to_rgba_and_fixes_output() {
        let comp = Compositor::new(1920, 1080, Vec::from([CompositorPad::at(0, 0)])).with_framerate(60);
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
        assert!(matches!(comp.configure_pipeline(0, &nv12), Err(G2gError::CapsMismatch)));
    }
}
