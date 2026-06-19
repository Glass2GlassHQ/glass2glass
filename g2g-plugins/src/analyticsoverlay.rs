//! Detection overlay (M101): draws the [`AnalyticsMeta`] detection boxes carried
//! on a frame onto its raw RGBA8 pixels, the visible end of a detector -> overlay
//! pipeline. The `cairooverlay` / `ovrenderhud` analog for ML analytics.
//!
//! Pairs with the M100 metadata-through-fan-out path: a `decode -> tee ->
//! {detect, video} -> overlay -> display` diamond runs the detector on one branch
//! and carries its `AnalyticsMeta` (shared by Arc) onto the video branch, where
//! this element renders the boxes onto the picture that actually reaches the sink.
//!
//! CPU, `no_std` baseline like the other raw-video transforms. Input and output
//! are both RGBA8 at the negotiated geometry (put a `VideoConvert` upstream of a
//! non-RGBA source); the element is an identity transform on the pixels apart from
//! the boxes it paints. Boxes are normalized `[0,1]` in the metadata, so this
//! works at any frame size without an upstream coordinate rewrite. A frame with no
//! `AnalyticsMeta` passes through untouched. The Vello GPU backend is the separate
//! `vello-overlay` feature (M102).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::{
    AnalyticsMeta, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError,
    MemoryDomain, ObjectDetection, OutputSink, PipelinePacket, RawVideoFormat,
};

/// Draws detection bounding boxes from an attached [`AnalyticsMeta`] onto an
/// RGBA8 frame. Box outline thickness is configurable; the colour is chosen per
/// class label from a fixed palette so different classes are distinguishable.
#[derive(Debug)]
pub struct AnalyticsOverlay {
    width: u32,
    height: u32,
    /// Outline thickness in pixels (>= 1).
    thickness: u32,
    configured: bool,
    drawn: u64,
}

impl Default for AnalyticsOverlay {
    fn default() -> Self {
        Self::new()
    }
}

impl AnalyticsOverlay {
    /// A new overlay with a 2px outline. Geometry is set at negotiation.
    pub fn new() -> Self {
        Self { width: 0, height: 0, thickness: 2, configured: false, drawn: 0 }
    }

    /// Set the box outline thickness in pixels (clamped to at least 1).
    pub fn with_thickness(mut self, px: u32) -> Self {
        self.thickness = px.max(1);
        self
    }

    /// Count of frames processed (whether or not they carried detections).
    pub fn drawn_count(&self) -> u64 {
        self.drawn
    }

    /// RGBA8 at fixed geometry, the only format this element draws on.
    fn dims(caps: &Caps) -> Option<(u32, u32)> {
        if let Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            ..
        } = caps
        {
            Some((*w, *h))
        } else {
            None
        }
    }

    /// Whether `caps` is RGBA8 (geometry may still be unfixed at negotiation).
    fn accepts(caps: &Caps) -> bool {
        matches!(caps, Caps::RawVideo { format: RawVideoFormat::Rgba8, .. })
    }

    /// Paint every detection box onto the RGBA8 `buf` of `self.width` x
    /// `self.height`. Normalized boxes are denormalized to pixel coordinates here.
    fn render(&self, buf: &mut [u8], detections: &[ObjectDetection]) {
        let w = self.width as i32;
        let h = self.height as i32;
        let t = self.thickness as i32;
        for d in detections {
            let color = class_color(d.label);
            // Denormalize; bbox fields are in [0, 1], so +0.5 rounds without the
            // std-only f32::round (the no_std baseline has no float intrinsics).
            let x0 = (d.bbox.x * w as f32 + 0.5) as i32;
            let y0 = (d.bbox.y * h as f32 + 0.5) as i32;
            let x1 = ((d.bbox.x + d.bbox.w) * w as f32 + 0.5) as i32 - 1;
            let y1 = ((d.bbox.y + d.bbox.h) * h as f32 + 0.5) as i32 - 1;
            if x1 < x0 || y1 < y0 {
                continue;
            }
            // Four outline bands, each `t` pixels thick, clipped to the canvas.
            for dy in 0..t {
                hspan(buf, w, h, x0, x1, y0 + dy, color);
                hspan(buf, w, h, x0, x1, y1 - dy, color);
            }
            for dx in 0..t {
                vspan(buf, w, h, y0, y1, x0 + dx, color);
                vspan(buf, w, h, y0, y1, x1 - dx, color);
            }
        }
    }
}

/// Source-over blend of one RGBA pixel `color` onto `buf` at byte offset `d`,
/// integer math (the same blend the compositor uses). An opaque colour fully
/// overwrites; a partial alpha tints. Keeps an opaque canvas opaque.
#[inline]
fn blend_px(buf: &mut [u8], d: usize, color: [u8; 4]) {
    let a = color[3] as u32;
    let inv = 255 - a;
    for c in 0..3 {
        buf[d + c] = ((color[c] as u32 * a + buf[d + c] as u32 * inv + 127) / 255) as u8;
    }
    buf[d + 3] = (a + buf[d + 3] as u32 * inv / 255) as u8;
}

/// Blend a horizontal run `x0..=x1` at row `y`, clipped to the canvas.
fn hspan(buf: &mut [u8], w: i32, h: i32, x0: i32, x1: i32, y: i32, color: [u8; 4]) {
    if y < 0 || y >= h {
        return;
    }
    let xs = x0.max(0);
    let xe = x1.min(w - 1);
    for x in xs..=xe {
        blend_px(buf, ((y * w + x) * 4) as usize, color);
    }
}

/// Blend a vertical run `y0..=y1` at column `x`, clipped to the canvas.
fn vspan(buf: &mut [u8], w: i32, h: i32, y0: i32, y1: i32, x: i32, color: [u8; 4]) {
    if x < 0 || x >= w {
        return;
    }
    let ys = y0.max(0);
    let ye = y1.min(h - 1);
    for y in ys..=ye {
        blend_px(buf, ((y * w + x) * 4) as usize, color);
    }
}

/// A fixed, opaque per-class colour palette so adjacent classes are visually
/// distinct. Cycles for labels beyond the palette length.
fn class_color(label: u32) -> [u8; 4] {
    const PALETTE: [[u8; 3]; 8] = [
        [0xFF, 0x3B, 0x30], // red
        [0x34, 0xC7, 0x59], // green
        [0x00, 0x7A, 0xFF], // blue
        [0xFF, 0xCC, 0x00], // yellow
        [0xAF, 0x52, 0xDE], // purple
        [0xFF, 0x95, 0x00], // orange
        [0x5A, 0xC8, 0xFA], // cyan
        [0xFF, 0x2D, 0x95], // magenta
    ];
    let c = PALETTE[(label as usize) % PALETTE.len()];
    [c[0], c[1], c[2], 0xFF]
}

impl AsyncElement for AnalyticsOverlay {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        if Self::accepts(upstream_caps) {
            Ok(upstream_caps.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        // Identity: pixels and geometry pass through; only boxes are painted.
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| {
            if Self::accepts(input) {
                CapsSet::one(input.clone())
            } else {
                CapsSet::from_alternatives(Vec::new())
            }
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (w, h) = Self::dims(absolute_caps).ok_or(G2gError::CapsMismatch)?;
        self.width = w;
        self.height = h;
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
                PipelinePacket::DataFrame(mut frame) => {
                    // Copy out the detections so the immutable meta borrow ends
                    // before the mutable pixel borrow below.
                    let detections: Vec<ObjectDetection> = frame
                        .meta
                        .get::<AnalyticsMeta>()
                        .map(|a| a.detections().copied().collect())
                        .unwrap_or_default();
                    if !detections.is_empty() {
                        let MemoryDomain::System(slice) = &mut frame.domain else {
                            return Err(G2gError::UnsupportedDomain);
                        };
                        let need = self.width as usize * self.height as usize * 4;
                        let buf = slice.as_mut_slice();
                        if buf.len() < need {
                            return Err(G2gError::CapsMismatch);
                        }
                        self.render(&mut buf[..need], &detections);
                    }
                    self.drawn += 1;
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                PipelinePacket::CapsChanged(caps) => {
                    if let Some((w, h)) = Self::dims(&caps) {
                        self.width = w;
                        self.height = h;
                    }
                    out.push(PipelinePacket::CapsChanged(caps)).await?;
                }
                // The runner's transform arm forwards EOS; don't double it.
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::frame::Frame;
    use g2g_core::memory::SystemSlice;
    use g2g_core::{BBox, FrameTiming, PushOutcome, Rate};

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

    fn rgba_caps(w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Any,
        }
    }

    fn det(x: f32, y: f32, w: f32, h: f32, label: u32) -> ObjectDetection {
        ObjectDetection { bbox: BBox { x, y, w, h }, label, confidence: 0.9 }
    }

    #[test]
    fn render_paints_box_border_and_leaves_interior() {
        // 8x8 black canvas; a normalized box covering [0.25,0.75] -> pixels (2,2)
        // to (5,5). A 1px red (class 0) outline; the interior stays black.
        let ov = AnalyticsOverlay { width: 8, height: 8, thickness: 1, configured: true, drawn: 0 };
        let mut buf = solid(8, 8, [0, 0, 0, 255]);
        ov.render(&mut buf, &[det(0.25, 0.25, 0.5, 0.5, 0)]);
        let red = class_color(0);
        assert_eq!(px(&buf, 8, 2, 2), red, "top-left corner on the border");
        assert_eq!(px(&buf, 8, 5, 5), red, "bottom-right corner on the border");
        assert_eq!(px(&buf, 8, 5, 2), red, "top-right corner on the border");
        assert_eq!(px(&buf, 8, 3, 3), [0, 0, 0, 255], "interior untouched");
        assert_eq!(px(&buf, 8, 0, 0), [0, 0, 0, 255], "outside the box untouched");
    }

    #[test]
    fn render_clips_box_to_canvas_bounds() {
        // A box running off the right/bottom edge must not panic or write OOB.
        let ov = AnalyticsOverlay { width: 4, height: 4, thickness: 2, configured: true, drawn: 0 };
        let mut buf = solid(4, 4, [0, 0, 0, 255]);
        ov.render(&mut buf, &[det(0.5, 0.5, 1.0, 1.0, 1)]);
        // The far corner is on the clipped border, painted the class-1 colour.
        assert_eq!(px(&buf, 4, 3, 3), class_color(1), "clipped corner painted");
    }

    /// Capturing sink that keeps the last forwarded frame's pixels.
    #[derive(Default)]
    struct PixelSink {
        last: Option<Vec<u8>>,
    }
    impl OutputSink for PixelSink {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async move {
                if let PipelinePacket::DataFrame(frame) = packet {
                    if let MemoryDomain::System(slice) = &frame.domain {
                        self.last = Some(slice.as_slice().to_vec());
                    }
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }

    fn rgba_frame_with_meta(w: u32, h: u32, dets: &[ObjectDetection]) -> Frame {
        let bytes = solid(w as usize, h as usize, [0, 0, 0, 255]);
        let mut frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        let mut a = AnalyticsMeta::new();
        for d in dets {
            a.add_detection(*d);
        }
        frame.meta.attach(a);
        frame
    }

    #[tokio::test]
    async fn process_draws_attached_detections_onto_the_frame() {
        let mut ov = AnalyticsOverlay::new().with_thickness(1);
        ov.configure_pipeline(&rgba_caps(8, 8)).unwrap();
        let frame = rgba_frame_with_meta(8, 8, &[det(0.25, 0.25, 0.5, 0.5, 0)]);

        let mut sink = PixelSink::default();
        ov.process(PipelinePacket::DataFrame(frame), &mut sink).await.unwrap();

        let out = sink.last.expect("frame forwarded");
        assert_eq!(px(&out, 8, 2, 2), class_color(0), "box border drawn");
        assert_eq!(px(&out, 8, 3, 3), [0, 0, 0, 255], "interior untouched");
        assert_eq!(ov.drawn_count(), 1);
    }

    #[tokio::test]
    async fn process_passes_through_a_frame_without_meta() {
        let mut ov = AnalyticsOverlay::new();
        ov.configure_pipeline(&rgba_caps(4, 4)).unwrap();
        let bytes = solid(4, 4, [10, 20, 30, 255]);
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.clone().into_boxed_slice())),
            FrameTiming::default(),
            0,
        );

        let mut sink = PixelSink::default();
        ov.process(PipelinePacket::DataFrame(frame), &mut sink).await.unwrap();
        assert_eq!(sink.last.expect("forwarded"), bytes, "pixels unchanged without meta");
    }

    #[test]
    fn intercept_rejects_non_rgba() {
        let ov = AnalyticsOverlay::new();
        let nv12 = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(8),
            height: Dim::Fixed(8),
            framerate: Rate::Any,
        };
        assert!(ov.intercept_caps(&nv12).is_err(), "only RGBA8 accepted");
        assert!(ov.intercept_caps(&rgba_caps(8, 8)).is_ok());
    }
}
