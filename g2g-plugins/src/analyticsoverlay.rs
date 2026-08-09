//! Analytics overlay (M101, masks M994): draws the [`AnalyticsMeta`] carried on a
//! frame onto its raw RGBA8 pixels, the visible end of a detector -> overlay
//! pipeline. The `cairooverlay` / `ovrenderhud` analog for ML analytics.
//!
//! Three shapes: a detection box as a solid outline in its class colour, an
//! instance segmentation as a translucent fill of its mask, and a region of
//! interest as a dashed outline in the colour of the mask that contains it.
//!
//! Pairs with the M100 metadata-through-fan-out path: a `decode -> tee ->
//! {detect, video} -> overlay -> display` diamond runs the detector on one branch
//! and carries its `AnalyticsMeta` (shared by Arc) onto the video branch, where
//! this element renders them onto the picture that actually reaches the sink.
//!
//! CPU, `no_std` baseline like the other raw-video transforms. Input and output
//! are both RGBA8 at the negotiated geometry (put a `VideoConvert` upstream of a
//! non-RGBA source); the element is an identity transform on the pixels apart from
//! the shapes it paints. Boxes are normalized `[0,1]` in the metadata and a mask
//! spans exactly its instance's box, so this works at any frame size without an
//! upstream coordinate rewrite. A frame with no `AnalyticsMeta` passes through
//! untouched. The Vello GPU backend is the separate `vello-overlay` feature (M102).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::{
    AnalyticsMeta, AnalyticsNode, AsyncElement, BBox, Caps, CapsConstraint, CapsSet,
    ConfigureOutcome, Dim, ElementMetadata, G2gError, MemoryDomain, ObjectDetection, OutputSink,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec, RawVideoFormat, RelationKind,
    Roi, Segmentation,
};

use crate::paint::blend_px;

/// Default mask fill alpha: faint enough that the picture reads through the fill.
pub(crate) const MASK_ALPHA_DEFAULT: u8 = 96;

/// Dash period of an ROI outline, in canvas pixels: `ROI_DASH_PX` on, then the
/// same off. Measured in absolute canvas coordinates, so the four sides of one
/// rectangle keep a common phase.
pub(crate) const ROI_DASH_PX: i32 = 6;

/// Draws the detection boxes, segmentation masks and regions of interest of an
/// attached [`AnalyticsMeta`] onto an RGBA8 frame. Outline thickness and mask fill
/// alpha are configurable; colours come from a fixed palette, indexed by class
/// label for a detection box and by instance for a mask and its ROI.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::analyticsoverlay::AnalyticsOverlay;
///
/// let overlay = AnalyticsOverlay::new().with_thickness(3).with_mask_alpha(120);
/// ```
#[derive(Debug)]
pub struct AnalyticsOverlay {
    width: u32,
    height: u32,
    /// Outline thickness in pixels (>= 1).
    thickness: u32,
    /// Alpha the mask fill is blended at (0 = invisible, 255 = opaque).
    mask_alpha: u8,
    configured: bool,
    drawn: u64,
}

/// A segmentation mask to fill, with the palette slot of the instance it belongs
/// to.
#[derive(Debug)]
pub(crate) struct PaintedMask {
    pub segmentation: Segmentation,
    pub palette_index: u32,
}

/// A region of interest to outline, in the palette slot of the mask that contains
/// it (or of its own `id` when it stands alone), so an instance's mask and ROI are
/// visually paired.
#[derive(Debug)]
pub(crate) struct PaintedRoi {
    pub roi: Roi,
    pub palette_index: u32,
}

/// Everything an overlay draws for one frame, copied out of the frame's
/// [`AnalyticsMeta`] so the immutable meta borrow ends before the pixels (or the
/// scene) are written. Shared by the CPU and Vello backends, so both resolve the
/// same palette slots and the same containment pairing.
#[derive(Debug, Default)]
pub(crate) struct AnalyticsShapes {
    pub detections: Vec<ObjectDetection>,
    pub masks: Vec<PaintedMask>,
    pub rois: Vec<PaintedRoi>,
}

impl AnalyticsShapes {
    /// Read the drawable nodes out of `meta`, numbering the segmentations for the
    /// palette and giving every ROI the slot of the segmentation that contains it.
    pub(crate) fn collect(meta: &AnalyticsMeta) -> Self {
        let mut shapes = Self::default();
        let mut slot_of_node: Vec<Option<u32>> = alloc::vec![None; meta.nodes.len()];
        for (index, node) in meta.nodes.iter().enumerate() {
            match node {
                AnalyticsNode::Detection(detection) => shapes.detections.push(*detection),
                AnalyticsNode::Segmentation(segmentation) => {
                    let palette_index = shapes.masks.len() as u32;
                    slot_of_node[index] = Some(palette_index);
                    shapes.masks.push(PaintedMask {
                        segmentation: segmentation.clone(),
                        palette_index,
                    });
                }
                _ => {}
            }
        }
        // A second pass, because a containment relation may name a segmentation
        // that appears after its ROI in the node list.
        for (index, node) in meta.nodes.iter().enumerate() {
            let AnalyticsNode::Roi(roi) = node else {
                continue;
            };
            let container = meta
                .relations
                .iter()
                .find(|r| r.to == index && r.kind == RelationKind::Contains)
                .and_then(|r| slot_of_node.get(r.from).copied().flatten());
            shapes.rois.push(PaintedRoi {
                roi: *roi,
                palette_index: container.unwrap_or(roi.id),
            });
        }
        shapes
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.detections.is_empty() && self.masks.is_empty() && self.rois.is_empty()
    }
}

impl Default for AnalyticsOverlay {
    fn default() -> Self {
        Self::new()
    }
}

impl AnalyticsOverlay {
    /// A new overlay with a 2px outline. Geometry is set at negotiation.
    pub fn new() -> Self {
        Self {
            width: 0,
            height: 0,
            thickness: 2,
            mask_alpha: MASK_ALPHA_DEFAULT,
            configured: false,
            drawn: 0,
        }
    }

    /// Set the box outline thickness in pixels (clamped to at least 1).
    pub fn with_thickness(mut self, px: u32) -> Self {
        self.thickness = px.max(1);
        self
    }

    /// Set the alpha the segmentation mask fill is blended at (0..=255).
    pub fn with_mask_alpha(mut self, alpha: u8) -> Self {
        self.mask_alpha = alpha;
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
        matches!(
            caps,
            Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                ..
            }
        )
    }

    /// Paint every shape onto the RGBA8 `buf` of `self.width` x `self.height`.
    /// Mask fills go down first, so a box or ROI edge stays readable over one.
    fn render(&self, buf: &mut [u8], shapes: &AnalyticsShapes) {
        for mask in &shapes.masks {
            self.fill_mask(buf, mask);
        }
        for detection in &shapes.detections {
            self.outline(buf, detection.bbox, palette_color(detection.label), false);
        }
        for roi in &shapes.rois {
            self.outline(buf, roi.roi.bbox, palette_color(roi.palette_index), true);
        }
    }

    /// Stroke the outline of a normalized box, `dashed` for an ROI so it reads
    /// apart from a detection box.
    fn outline(&self, buf: &mut [u8], bbox: BBox, color: [u8; 4], dashed: bool) {
        let w = self.width as i32;
        let h = self.height as i32;
        let t = self.thickness as i32;
        let Some((x0, y0, x1, y1)) = pixel_rect(bbox, w, h) else {
            return;
        };
        let paint = SpanPaint { color, dashed };
        // Four outline bands, each `t` pixels thick, clipped to the canvas.
        for dy in 0..t {
            hspan(buf, w, h, x0, x1, y0 + dy, paint);
            hspan(buf, w, h, x0, x1, y1 - dy, paint);
        }
        for dx in 0..t {
            vspan(buf, w, h, y0, y1, x0 + dx, paint);
            vspan(buf, w, h, y0, y1, x1 - dx, paint);
        }
    }

    /// Blend an instance's mask over its box as a translucent fill. The mask spans
    /// exactly the box (its grid is the model's, not the frame's), so a pixel reads
    /// the sample covering its share of the box.
    fn fill_mask(&self, buf: &mut [u8], painted: &PaintedMask) {
        let w = self.width as i32;
        let h = self.height as i32;
        let mask = &painted.segmentation.mask;
        let Some((x0, y0, x1, y1)) = pixel_rect(painted.segmentation.bbox, w, h) else {
            return;
        };
        let (span_w, span_h) = ((x1 - x0 + 1) as u32, (y1 - y0 + 1) as u32);
        let color = palette_color(painted.palette_index);
        for y in y0.max(0)..=y1.min(h - 1) {
            let j = (y - y0) as u32 * mask.height() / span_h;
            for x in x0.max(0)..=x1.min(w - 1) {
                let i = (x - x0) as u32 * mask.width() / span_w;
                let coverage = mask.sample(i, j).unwrap_or(0) as u32;
                if coverage == 0 {
                    continue;
                }
                let alpha = (coverage * self.mask_alpha as u32 / 255) as u8;
                blend_px(buf, ((y * w + x) * 4) as usize, color, alpha);
            }
        }
    }
}

/// The pixel rectangle (inclusive) a normalized box covers on a `w` x `h` canvas,
/// or `None` when it collapses to nothing. `+ 0.5` rounds without the std-only
/// `f32::round` (the `no_std` baseline has no float intrinsics).
fn pixel_rect(bbox: BBox, w: i32, h: i32) -> Option<(i32, i32, i32, i32)> {
    let x0 = (bbox.x * w as f32 + 0.5) as i32;
    let y0 = (bbox.y * h as f32 + 0.5) as i32;
    let x1 = ((bbox.x + bbox.w) * w as f32 + 0.5) as i32 - 1;
    let y1 = ((bbox.y + bbox.h) * h as f32 + 0.5) as i32 - 1;
    if x1 < x0 || y1 < y0 {
        return None;
    }
    Some((x0, y0, x1, y1))
}

/// How one outline band paints: its colour, and whether it dashes.
#[derive(Debug, Clone, Copy)]
struct SpanPaint {
    color: [u8; 4],
    dashed: bool,
}

impl SpanPaint {
    /// Whether the band paints at this canvas coordinate.
    fn paints_at(&self, coordinate: i32) -> bool {
        !self.dashed || (coordinate / ROI_DASH_PX) % 2 == 0
    }
}

/// Blend a horizontal run `x0..=x1` at row `y`, clipped to the canvas.
fn hspan(buf: &mut [u8], w: i32, h: i32, x0: i32, x1: i32, y: i32, paint: SpanPaint) {
    if y < 0 || y >= h {
        return;
    }
    let xs = x0.max(0);
    let xe = x1.min(w - 1);
    for x in xs..=xe {
        if paint.paints_at(x) {
            blend_px(buf, ((y * w + x) * 4) as usize, paint.color, 255);
        }
    }
}

/// Blend a vertical run `y0..=y1` at column `x`, clipped to the canvas.
fn vspan(buf: &mut [u8], w: i32, h: i32, y0: i32, y1: i32, x: i32, paint: SpanPaint) {
    if x < 0 || x >= w {
        return;
    }
    let ys = y0.max(0);
    let ye = y1.min(h - 1);
    for y in ys..=ye {
        if paint.paints_at(y) {
            blend_px(buf, ((y * w + x) * 4) as usize, paint.color, 255);
        }
    }
}

/// A fixed, opaque RGB palette so adjacent slots are visually distinct: a
/// detection box indexes it by class label, a mask and its ROI by instance.
/// Cycles for indices beyond the palette length. Shared with the Vello overlay
/// backend (`vello-overlay`) so both draw a slot the same colour.
pub(crate) fn palette_rgb(index: u32) -> [u8; 3] {
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
    PALETTE[(index as usize) % PALETTE.len()]
}

/// The opaque RGBA outline colour of a palette slot.
fn palette_color(index: u32) -> [u8; 4] {
    let c = palette_rgb(index);
    [c[0], c[1], c[2], 0xFF]
}

impl AsyncElement for AnalyticsOverlay {
    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Analytics overlay",
            "Filter/Editing/Video",
            "Draws detection boxes, segmentation masks and regions of interest onto raw video",
            "g2g",
        )
    }
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

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new(
                "thickness",
                PropKind::Uint,
                "box outline thickness in pixels (>= 1)",
            )
            .with_range("1", "65535")
            .with_default("2"),
            PropertySpec::new(
                "mask-alpha",
                PropKind::Uint,
                "alpha the segmentation mask fill is blended at (0..255)",
            )
            .with_range("0", "255")
            .with_default("96"),
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "thickness" => {
                self.thickness = (value.as_uint().ok_or(PropError::Type)? as u32).max(1);
                Ok(())
            }
            "mask-alpha" => {
                self.mask_alpha = value.as_uint().ok_or(PropError::Type)?.min(255) as u8;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "thickness" => Some(PropValue::Uint(self.thickness as u64)),
            "mask-alpha" => Some(PropValue::Uint(self.mask_alpha as u64)),
            _ => None,
        }
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
                    // Copy out the shapes so the immutable meta borrow ends before
                    // the mutable pixel borrow below.
                    let shapes = frame
                        .meta
                        .get::<AnalyticsMeta>()
                        .map(AnalyticsShapes::collect)
                        .unwrap_or_default();
                    if !shapes.is_empty() {
                        let MemoryDomain::System(slice) = &mut frame.domain else {
                            return Err(G2gError::UnsupportedDomain);
                        };
                        let need = self.width as usize * self.height as usize * 4;
                        let buf = slice.as_mut_slice();
                        if buf.len() < need {
                            return Err(G2gError::CapsMismatch);
                        }
                        self.render(&mut buf[..need], &shapes);
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
    use g2g_core::{FrameTiming, Mask, PushOutcome, Rate};

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
            interlace: g2g_core::Interlace::Any,
        }
    }

    fn det(x: f32, y: f32, w: f32, h: f32, label: u32) -> ObjectDetection {
        ObjectDetection {
            bbox: BBox { x, y, w, h },
            label,
            confidence: 0.9,
        }
    }

    fn overlay(width: u32, height: u32, thickness: u32) -> AnalyticsOverlay {
        AnalyticsOverlay {
            width,
            height,
            thickness,
            mask_alpha: MASK_ALPHA_DEFAULT,
            configured: true,
            drawn: 0,
        }
    }

    /// The shapes of a meta holding just these detections.
    fn detection_shapes(detections: &[ObjectDetection]) -> AnalyticsShapes {
        let mut meta = AnalyticsMeta::new();
        for d in detections {
            meta.add_detection(*d);
        }
        AnalyticsShapes::collect(&meta)
    }

    /// A segmentation over `bbox` whose `mask_w` x `mask_h` mask is covered where
    /// `covered(i, j)` holds.
    fn seg(
        bbox: BBox,
        label: u32,
        mask_w: u32,
        mask_h: u32,
        covered: impl Fn(u32, u32) -> bool,
    ) -> Segmentation {
        let mut data = Vec::with_capacity((mask_w * mask_h) as usize);
        for j in 0..mask_h {
            for i in 0..mask_w {
                data.push(if covered(i, j) { u8::MAX } else { 0 });
            }
        }
        Segmentation {
            bbox,
            label,
            confidence: 0.9,
            mask: Mask::new(mask_w, mask_h, mask_w, data).expect("mask geometry"),
        }
    }

    #[test]
    fn render_paints_box_border_and_leaves_interior() {
        // 8x8 black canvas; a normalized box covering [0.25,0.75] -> pixels (2,2)
        // to (5,5). A 1px red (class 0) outline; the interior stays black.
        let ov = overlay(8, 8, 1);
        let mut buf = solid(8, 8, [0, 0, 0, 255]);
        ov.render(&mut buf, &detection_shapes(&[det(0.25, 0.25, 0.5, 0.5, 0)]));
        let red = palette_color(0);
        assert_eq!(px(&buf, 8, 2, 2), red, "top-left corner on the border");
        assert_eq!(px(&buf, 8, 5, 5), red, "bottom-right corner on the border");
        assert_eq!(px(&buf, 8, 5, 2), red, "top-right corner on the border");
        assert_eq!(px(&buf, 8, 3, 3), [0, 0, 0, 255], "interior untouched");
        assert_eq!(
            px(&buf, 8, 0, 0),
            [0, 0, 0, 255],
            "outside the box untouched"
        );
    }

    #[test]
    fn render_clips_box_to_canvas_bounds() {
        // A box running off the right/bottom edge must not panic or write OOB.
        let ov = overlay(4, 4, 2);
        let mut buf = solid(4, 4, [0, 0, 0, 255]);
        ov.render(&mut buf, &detection_shapes(&[det(0.5, 0.5, 1.0, 1.0, 1)]));
        // The far corner is on the clipped border, painted the class-1 colour.
        assert_eq!(
            px(&buf, 4, 3, 3),
            palette_color(1),
            "clipped corner painted"
        );
    }

    #[test]
    fn render_blends_a_mask_over_the_covered_half_of_its_box() {
        // 8x8 black canvas; a box over pixels (2,2)..(5,5) with a 2x2 mask whose
        // left column is covered, i.e. columns 2 and 3 of the box.
        let ov = overlay(8, 8, 1);
        let mut buf = solid(8, 8, [0, 0, 0, 255]);
        let mut meta = AnalyticsMeta::new();
        meta.push(AnalyticsNode::Segmentation(seg(
            BBox {
                x: 0.25,
                y: 0.25,
                w: 0.5,
                h: 0.5,
            },
            0,
            2,
            2,
            |i, _| i == 0,
        )));
        ov.render(&mut buf, &AnalyticsShapes::collect(&meta));

        // Palette slot 0 (0xFF3B30) at alpha 96 over opaque black: each channel is
        // (c * 96 + 127) / 255, the alpha stays opaque.
        let fill = [96, 22, 18, 255];
        assert_eq!(px(&buf, 8, 2, 2), fill, "covered sample blended");
        assert_eq!(px(&buf, 8, 3, 5), fill, "covered sample blended");
        assert_eq!(
            px(&buf, 8, 4, 3),
            [0, 0, 0, 255],
            "uncovered half of the box untouched"
        );
        assert_eq!(
            px(&buf, 8, 1, 2),
            [0, 0, 0, 255],
            "outside the box untouched"
        );
    }

    #[test]
    fn render_dashes_an_roi_outline() {
        // 16x16; an ROI over pixels (4,4)..(11,11) with a 1px outline. The dash
        // phase is absolute, so along the top edge x=4,5 paint and x=6..11 do not.
        let ov = overlay(16, 16, 1);
        let mut buf = solid(16, 16, [0, 0, 0, 255]);
        let mut meta = AnalyticsMeta::new();
        meta.push(AnalyticsNode::Roi(Roi {
            bbox: BBox {
                x: 0.25,
                y: 0.25,
                w: 0.5,
                h: 0.5,
            },
            id: 1,
            label: 3,
        }));
        ov.render(&mut buf, &AnalyticsShapes::collect(&meta));

        // A lone ROI takes its palette slot from its own id, not its label.
        let green = palette_color(1);
        assert_eq!(px(&buf, 16, 4, 4), green, "dash on at x=4");
        assert_eq!(px(&buf, 16, 5, 4), green, "dash on at x=5");
        assert_eq!(px(&buf, 16, 8, 4), [0, 0, 0, 255], "dash off at x=8");
        assert_eq!(
            px(&buf, 16, 4, 8),
            [0, 0, 0, 255],
            "dash off down the left edge at y=8"
        );
    }

    #[test]
    fn collect_gives_an_roi_the_palette_slot_of_the_mask_containing_it() {
        // Two instances, each with a contained ROI whose id disagrees with the
        // instance order, so the pairing has to come from the relation.
        let mut meta = AnalyticsMeta::new();
        let bbox = BBox {
            x: 0.0,
            y: 0.0,
            w: 0.5,
            h: 0.5,
        };
        let first = meta.push(AnalyticsNode::Segmentation(seg(bbox, 7, 1, 1, |_, _| true)));
        let second = meta.push(AnalyticsNode::Segmentation(seg(bbox, 7, 1, 1, |_, _| true)));
        let first_roi = meta.push(AnalyticsNode::Roi(Roi {
            bbox,
            id: 40,
            label: 7,
        }));
        let second_roi = meta.push(AnalyticsNode::Roi(Roi {
            bbox,
            id: 41,
            label: 7,
        }));
        meta.relate(second, second_roi, RelationKind::Contains);
        meta.relate(first, first_roi, RelationKind::Contains);

        let shapes = AnalyticsShapes::collect(&meta);
        assert_eq!(shapes.masks.len(), 2);
        assert_eq!(
            (shapes.masks[0].palette_index, shapes.masks[1].palette_index),
            (0, 1),
            "masks are numbered in instance order"
        );
        assert_eq!(
            (shapes.rois[0].palette_index, shapes.rois[1].palette_index),
            (0, 1),
            "each ROI paints in its container's slot, not its own id"
        );
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
                    if let Some(slice) = frame.domain.as_system_slice() {
                        self.last = Some(slice.to_vec());
                    }
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }

    fn rgba_frame_with_meta(w: u32, h: u32, meta: AnalyticsMeta) -> Frame {
        let bytes = solid(w as usize, h as usize, [0, 0, 0, 255]);
        let mut frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        frame.meta.attach(meta);
        frame
    }

    #[tokio::test]
    async fn process_draws_attached_detections_onto_the_frame() {
        let mut ov = AnalyticsOverlay::new().with_thickness(1);
        ov.configure_pipeline(&rgba_caps(8, 8)).unwrap();
        let mut meta = AnalyticsMeta::new();
        meta.add_detection(det(0.25, 0.25, 0.5, 0.5, 0));
        let frame = rgba_frame_with_meta(8, 8, meta);

        let mut sink = PixelSink::default();
        ov.process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();

        let out = sink.last.expect("frame forwarded");
        assert_eq!(px(&out, 8, 2, 2), palette_color(0), "box border drawn");
        assert_eq!(px(&out, 8, 3, 3), [0, 0, 0, 255], "interior untouched");
        assert_eq!(ov.drawn_count(), 1);
    }

    #[tokio::test]
    async fn process_draws_an_attached_segmentation_and_its_roi() {
        // The M992 shape: a Segmentation plus the mask-tight Roi it contains, drawn
        // through the real process path at an explicit mask alpha.
        let mut ov = AnalyticsOverlay::new()
            .with_thickness(1)
            .with_mask_alpha(255);
        ov.configure_pipeline(&rgba_caps(16, 16)).unwrap();
        let bbox = BBox {
            x: 0.25,
            y: 0.25,
            w: 0.5,
            h: 0.5,
        };
        let mut meta = AnalyticsMeta::new();
        // Box pixels 4..11 with a 2x2 mask covered only at (0,0), so the fill is
        // its top-left quarter, pixels 4..7. The ROI is that quarter, mask-tight.
        let mask = meta.push(AnalyticsNode::Segmentation(seg(bbox, 0, 2, 2, |i, j| {
            i == 0 && j == 0
        })));
        let roi = meta.push(AnalyticsNode::Roi(Roi {
            bbox: BBox {
                x: 0.25,
                y: 0.25,
                w: 0.25,
                h: 0.25,
            },
            id: 9,
            label: 0,
        }));
        meta.relate(mask, roi, RelationKind::Contains);
        let frame = rgba_frame_with_meta(16, 16, meta);

        let mut sink = PixelSink::default();
        ov.process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();

        let out = sink.last.expect("frame forwarded");
        // At alpha 255 the fill is the palette colour itself; the ROI outline is the
        // same colour, so the fill is read inside it and the dashes on its edge.
        let fill = palette_color(0);
        assert_eq!(px(&out, 16, 5, 5), fill, "mask fill inside the ROI");
        assert_eq!(px(&out, 16, 6, 6), fill, "mask fill inside the ROI");
        assert_eq!(px(&out, 16, 4, 4), fill, "ROI dash on its top-left corner");
        assert_eq!(
            px(&out, 16, 9, 9),
            [0, 0, 0, 255],
            "uncovered mask samples left alone"
        );
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
        ov.process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();
        assert_eq!(
            sink.last.expect("forwarded"),
            bytes,
            "pixels unchanged without meta"
        );
    }

    #[test]
    fn intercept_rejects_non_rgba() {
        let ov = AnalyticsOverlay::new();
        let nv12 = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(8),
            height: Dim::Fixed(8),
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
        };
        assert!(ov.intercept_caps(&nv12).is_err(), "only RGBA8 accepted");
        assert!(ov.intercept_caps(&rgba_caps(8, 8)).is_ok());
    }
}
