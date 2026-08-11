//! Analytics overlay (M101, masks M994): draws the [`AnalyticsMeta`] carried on a
//! frame onto its raw RGBA8 pixels, the visible end of a detector -> overlay
//! pipeline. The `cairooverlay` / `ovrenderhud` analog for ML analytics.
//!
//! Three shapes: a detection box as a solid outline in its class colour, an
//! instance segmentation as a translucent fill of its mask, and a region of
//! interest as a dashed outline in the colour of the mask that contains it.
//!
//! `show-label` / `show-track` / `show-score` add a caption bar above each
//! detection box with its class name, tracking id and confidence. A node stores
//! a `u32` label id, so the name comes from the meta's shared `class_names`
//! table; a producer that publishes none leaves `show-label` with nothing to
//! draw.
//!
//! `show-trail` draws where each tracked object has been, as a polyline through
//! the bottom edge of its recent boxes that fades out towards the oldest point.
//! It is the one thing here that remembers anything between frames, so a trail
//! outlives a few missed detections and is dropped once its track stops coming.
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
use alloc::collections::{BTreeMap, VecDeque};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::{
    AnalyticsMeta, AnalyticsNode, AsyncElement, BBox, Caps, CapsConstraint, CapsSet,
    ConfigureOutcome, Dim, ElementMetadata, G2gError, MemoryDomain, ObjectDetection, OutputSink,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec, RawVideoFormat, RelationKind,
    Roi, Segmentation,
};

use crate::bitmapfont::{glyph, GLYPH_ADVANCE, GLYPH_HEIGHT};
use crate::paint::{blend_px, Canvas};

/// Default mask fill alpha: faint enough that the picture reads through the fill.
pub(crate) const MASK_ALPHA_DEFAULT: u8 = 96;

/// Dash period of an ROI outline, in canvas pixels: `ROI_DASH_PX` on, then the
/// same off. Measured in absolute canvas coordinates, so the four sides of one
/// rectangle keep a common phase.
pub(crate) const ROI_DASH_PX: i32 = 6;

/// How many past positions a trail keeps by default, i.e. about a second of
/// movement at 30 fps.
const TRAIL_LENGTH_DEFAULT: usize = 30;

/// A trail is dropped once its track has gone this many frames without a
/// detection, so an object that leaves the scene stops being drawn.
const TRAIL_TTL_FRAMES: u64 = 30;

/// The faintest a trail segment is blended at, so its oldest end stays visible
/// rather than fading to nothing.
const TRAIL_MIN_ALPHA: u32 = 40;

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
/// let overlay = AnalyticsOverlay::new()
///     .with_thickness(3)
///     .with_mask_alpha(120)
///     .with_score(true);
/// ```
#[derive(Debug)]
pub struct AnalyticsOverlay {
    width: u32,
    height: u32,
    /// Outline thickness in pixels (>= 1).
    thickness: u32,
    /// Alpha the mask fill is blended at (0 = invisible, 255 = opaque).
    mask_alpha: u8,
    /// Draw each detection's class name in its caption, where the producer
    /// published a name table.
    show_label: bool,
    /// Draw each detection's confidence in its caption.
    show_score: bool,
    /// Draw each detection's tracking id in its caption, where it has one.
    show_track: bool,
    /// Draw the path each tracked object took.
    show_trail: bool,
    /// How many past positions a trail keeps.
    trail_length: usize,
    /// The path of each tracked object, keyed by tracking id.
    trails: BTreeMap<u64, Trail>,
    configured: bool,
    drawn: u64,
}

/// The recent path of one tracked object: normalized bottom-centre points of its
/// boxes, oldest first, and the frame the newest one arrived on.
#[derive(Debug, Default)]
struct Trail {
    points: VecDeque<(f32, f32)>,
    last_seen: u64,
}

/// A detection box to outline, with the id of the tracking node related to it
/// when the producer wired one (`Detection -Tracks-> Tracking`).
#[derive(Debug)]
pub(crate) struct PaintedDetection {
    pub detection: ObjectDetection,
    pub track_id: Option<u64>,
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
    pub detections: Vec<PaintedDetection>,
    pub masks: Vec<PaintedMask>,
    pub rois: Vec<PaintedRoi>,
    /// The meta's class-name table, carried so a caption can name a label
    /// without the shapes borrowing the meta.
    pub class_names: Option<alloc::sync::Arc<[alloc::boxed::Box<str>]>>,
}

impl AnalyticsShapes {
    /// Read the drawable nodes out of `meta`, numbering the segmentations for the
    /// palette and giving every ROI the slot of the segmentation that contains it.
    pub(crate) fn collect(meta: &AnalyticsMeta) -> Self {
        let mut shapes = Self::default();
        shapes.class_names = meta.class_names.clone();
        let mut slot_of_node: Vec<Option<u32>> = alloc::vec![None; meta.nodes.len()];
        for (index, node) in meta.nodes.iter().enumerate() {
            match node {
                AnalyticsNode::Detection(detection) => shapes.detections.push(PaintedDetection {
                    detection: *detection,
                    track_id: track_id_of(meta, index),
                }),
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

/// The `object_id` of the tracking node the detection at `index` tracks as, if
/// the producer wired one.
fn track_id_of(meta: &AnalyticsMeta, index: usize) -> Option<u64> {
    meta.relations
        .iter()
        .find(|r| r.from == index && r.kind == RelationKind::Tracks)
        .and_then(|r| match meta.nodes.get(r.to) {
            Some(AnalyticsNode::Tracking(tracking)) => Some(tracking.object_id),
            _ => None,
        })
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
            show_label: false,
            show_score: false,
            show_track: false,
            show_trail: false,
            trail_length: TRAIL_LENGTH_DEFAULT,
            trails: BTreeMap::new(),
            configured: false,
            drawn: 0,
        }
    }

    /// Draw each detection's class name above its box.
    pub fn with_label(mut self, show: bool) -> Self {
        self.show_label = show;
        self
    }

    /// Draw each detection's confidence above its box.
    pub fn with_score(mut self, show: bool) -> Self {
        self.show_score = show;
        self
    }

    /// Draw each detection's tracking id above its box.
    pub fn with_track(mut self, show: bool) -> Self {
        self.show_track = show;
        self
    }

    /// Draw the path each tracked object took.
    pub fn with_trail(mut self, show: bool) -> Self {
        self.show_trail = show;
        self
    }

    /// Set how many past positions a trail keeps (clamped to at least 2, since a
    /// segment needs both ends).
    pub fn with_trail_length(mut self, points: usize) -> Self {
        self.trail_length = points.max(2);
        self
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
        if self.show_trail {
            self.draw_trails(buf);
        }
        for mask in &shapes.masks {
            self.fill_mask(buf, mask);
        }
        for painted in &shapes.detections {
            let color = palette_color(painted.detection.label);
            self.outline(buf, painted.detection.bbox, color, false);
            let name = shapes
                .class_names
                .as_ref()
                .and_then(|names| names.get(painted.detection.label as usize))
                .map(|name| &**name);
            if let Some(text) = self.caption(painted, name) {
                self.draw_caption(buf, painted.detection.bbox, &text, color);
            }
        }
        for roi in &shapes.rois {
            self.outline(buf, roi.roi.bbox, palette_color(roi.palette_index), true);
        }
    }

    /// Extend each tracked object's path with this frame's position, and forget
    /// the tracks that have stopped arriving. `drawn` counts every frame, so it
    /// is what a trail's age is measured against.
    fn record_trails(&mut self, shapes: &AnalyticsShapes) {
        for painted in &shapes.detections {
            let Some(id) = painted.track_id else {
                continue;
            };
            let bbox = painted.detection.bbox;
            let trail = self.trails.entry(id).or_default();
            trail
                .points
                .push_back((bbox.x + bbox.w / 2.0, bbox.y + bbox.h));
            while trail.points.len() > self.trail_length {
                trail.points.pop_front();
            }
            trail.last_seen = self.drawn;
        }
        let cutoff = self.drawn.saturating_sub(TRAIL_TTL_FRAMES);
        self.trails.retain(|_, trail| trail.last_seen >= cutoff);
    }

    /// Stroke every trail, each in the palette slot of its tracking id so two
    /// objects of the same class still read apart, fading towards its old end.
    fn draw_trails(&self, buf: &mut [u8]) {
        for (id, trail) in &self.trails {
            let color = palette_color(*id as u32);
            let count = trail.points.len() as u32;
            let mut previous = None;
            for (index, point) in trail.points.iter().enumerate() {
                let x = (point.0 * self.width as f32 + 0.5) as i32;
                let y = (point.1 * self.height as f32 + 0.5) as i32;
                if let Some(from) = previous {
                    let ramp = (255 - TRAIL_MIN_ALPHA) * (index as u32 + 1) / count;
                    self.trail_segment(buf, from, (x, y), color, (TRAIL_MIN_ALPHA + ramp) as u8);
                }
                previous = Some((x, y));
            }
        }
    }

    /// Blend a straight run between two canvas points (Bresenham, integer only
    /// for the `no_std` baseline).
    fn trail_segment(
        &self,
        buf: &mut [u8],
        from: (i32, i32),
        to: (i32, i32),
        color: [u8; 4],
        alpha: u8,
    ) {
        let (mut x, mut y) = from;
        let dx = (to.0 - x).abs();
        let dy = -(to.1 - y).abs();
        let sx = if x < to.0 { 1 } else { -1 };
        let sy = if y < to.1 { 1 } else { -1 };
        let mut err = dx + dy;
        loop {
            self.trail_dot(buf, x, y, color, alpha);
            if x == to.0 && y == to.1 {
                return;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
        }
    }

    /// Blend one square of the stroke, `thickness` wide, clipped to the canvas.
    fn trail_dot(&self, buf: &mut [u8], x: i32, y: i32, color: [u8; 4], alpha: u8) {
        let w = self.width as i32;
        let h = self.height as i32;
        let half = self.thickness as i32 / 2;
        for py in y - half..=y + half {
            if py < 0 || py >= h {
                continue;
            }
            for px in x - half..=x + half {
                if px < 0 || px >= w {
                    continue;
                }
                blend_px(buf, ((py * w + px) * 4) as usize, color, alpha);
            }
        }
    }

    /// The caption for one detection, or `None` when neither part is enabled (or
    /// a tracking id was asked for and the producer wired none).
    fn caption(&self, painted: &PaintedDetection, name: Option<&str>) -> Option<String> {
        let mut caption = String::new();
        let mut add = |part: &str| {
            if !caption.is_empty() {
                caption.push(' ');
            }
            caption.push_str(part);
        };
        if let Some(name) = name.filter(|_| self.show_label) {
            add(name);
        }
        if let Some(id) = painted.track_id.filter(|_| self.show_track) {
            add(&format!("ID:{id}"));
        }
        if self.show_score {
            add(&score_text(&painted.detection));
        }
        (!caption.is_empty()).then_some(caption)
    }

    /// One source font pixel per this many output pixels, from the frame height
    /// so a caption stays readable across resolutions.
    fn text_scale(&self) -> i32 {
        (self.height / 240).max(1) as i32
    }

    /// Draw `text` in a filled bar the colour of its box, sitting on the box's
    /// top edge, or just inside it when the box starts at the top of the frame.
    fn draw_caption(&self, buf: &mut [u8], bbox: BBox, text: &str, color: [u8; 4]) {
        let w = self.width as i32;
        let h = self.height as i32;
        let Some((x0, y0, _, _)) = pixel_rect(bbox, w, h) else {
            return;
        };
        let scale = self.text_scale();
        let pad = scale;
        let cell_w = GLYPH_ADVANCE as i32 * scale;
        let bar_w = text.chars().count() as i32 * cell_w + 2 * pad;
        let bar_h = GLYPH_HEIGHT as i32 * scale + 2 * pad;
        let bar_x = x0.min((w - bar_w).max(0)).max(0);
        let bar_y = if y0 - bar_h >= 0 { y0 - bar_h } else { y0 };
        let mut canvas = Canvas {
            pixels: buf,
            width: w,
            height: h,
        };
        canvas.fill_rect(bar_x, bar_y, bar_w, bar_h, color);
        let ink = contrast_ink(color);
        let mut gx = bar_x + pad;
        for c in text.chars() {
            canvas.blit_glyph(gx, bar_y + pad, scale, glyph(c), ink);
            gx += cell_w;
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

/// A detection's confidence as a fixed two-decimal string (`0.92`). Built from
/// integer parts: the `no_std` baseline has no float formatting to lean on, and
/// `+ 0.5` rounds without `f32::round`.
fn score_text(detection: &ObjectDetection) -> String {
    let hundredths = (detection.confidence.clamp(0.0, 1.0) * 100.0 + 0.5) as u32;
    format!("{}.{:02}", hundredths / 100, hundredths % 100)
}

/// Black or white, whichever reads on `background`. Rec. 601 luma, since the
/// palette is sRGB and the threshold only has to separate light from dark.
fn contrast_ink(background: [u8; 4]) -> [u8; 4] {
    let luma =
        (background[0] as u32 * 299 + background[1] as u32 * 587 + background[2] as u32 * 114)
            / 1000;
    if luma > 140 {
        [0x00, 0x00, 0x00, 0xFF]
    } else {
        [0xFF, 0xFF, 0xFF, 0xFF]
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
            PropertySpec::new(
                "show-label",
                PropKind::Bool,
                "draw each detection's class name above its box",
            )
            .with_default("false"),
            PropertySpec::new(
                "show-score",
                PropKind::Bool,
                "draw each detection's confidence above its box",
            )
            .with_default("false"),
            PropertySpec::new(
                "show-track",
                PropKind::Bool,
                "draw each detection's tracking id above its box",
            )
            .with_default("false"),
            PropertySpec::new(
                "show-trail",
                PropKind::Bool,
                "draw the path each tracked object took",
            )
            .with_default("false"),
            PropertySpec::new(
                "trail-length",
                PropKind::Uint,
                "how many past positions a trail keeps (>= 2)",
            )
            .with_range("2", "65535")
            .with_default("30"),
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
            "show-label" => {
                self.show_label = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "show-score" => {
                self.show_score = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "show-track" => {
                self.show_track = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "show-trail" => {
                self.show_trail = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "trail-length" => {
                self.trail_length = (value.as_uint().ok_or(PropError::Type)? as usize).max(2);
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "thickness" => Some(PropValue::Uint(self.thickness as u64)),
            "mask-alpha" => Some(PropValue::Uint(self.mask_alpha as u64)),
            "show-label" => Some(PropValue::Bool(self.show_label)),
            "show-score" => Some(PropValue::Bool(self.show_score)),
            "show-track" => Some(PropValue::Bool(self.show_track)),
            "show-trail" => Some(PropValue::Bool(self.show_trail)),
            "trail-length" => Some(PropValue::Uint(self.trail_length as u64)),
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
                    self.drawn += 1;
                    if self.show_trail {
                        self.record_trails(&shapes);
                    }
                    // A live trail outlives the detections that fed it, so this
                    // frame may have something to paint with no shapes of its own.
                    let has_trail = self.show_trail && !self.trails.is_empty();
                    if !shapes.is_empty() || has_trail {
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
    use g2g_core::{FrameTiming, Mask, PushOutcome, Rate, Tracking};

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
            show_label: false,
            show_score: false,
            show_track: false,
            show_trail: false,
            trail_length: TRAIL_LENGTH_DEFAULT,
            trails: BTreeMap::new(),
            configured: true,
            drawn: 0,
        }
    }

    /// A meta holding one detection wired to the tracking id `track`.
    fn tracked_meta(detection: ObjectDetection, track: u64) -> AnalyticsMeta {
        let mut meta = AnalyticsMeta::new();
        let node = meta.add_detection(detection);
        let tracking = meta.push(AnalyticsNode::Tracking(Tracking { object_id: track }));
        meta.relate(node, tracking, RelationKind::Tracks);
        meta
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
    fn score_text_is_two_decimals() {
        let text = |confidence| {
            score_text(&ObjectDetection {
                bbox: BBox {
                    x: 0.0,
                    y: 0.0,
                    w: 1.0,
                    h: 1.0,
                },
                label: 0,
                confidence,
            })
        };
        assert_eq!(text(0.923), "0.92");
        assert_eq!(text(0.5), "0.50");
        assert_eq!(text(1.0), "1.00");
        assert_eq!(text(0.0), "0.00");
    }

    #[test]
    fn collect_pairs_a_detection_with_the_tracking_it_tracks_as() {
        let mut meta = AnalyticsMeta::new();
        let tracked = meta.add_detection(det(0.1, 0.1, 0.2, 0.2, 0));
        let untracked = meta.add_detection(det(0.5, 0.5, 0.2, 0.2, 1));
        let tracking = meta.push(AnalyticsNode::Tracking(Tracking { object_id: 77 }));
        meta.relate(tracked, tracking, RelationKind::Tracks);

        let shapes = AnalyticsShapes::collect(&meta);
        assert_eq!(shapes.detections[0].track_id, Some(77));
        assert_eq!(
            shapes.detections[1].track_id, None,
            "detection {untracked} has no Tracks relation"
        );
    }

    #[test]
    fn captions_are_off_until_asked_for() {
        let shapes = detection_shapes(&[det(0.25, 0.25, 0.5, 0.5, 0)]);
        let painted = &shapes.detections[0];
        assert_eq!(overlay(64, 64, 1).caption(painted, None), None);
        assert_eq!(
            overlay(64, 64, 1).with_score(true).caption(painted, None),
            Some(String::from("0.90"))
        );
    }

    #[test]
    fn caption_shows_the_tracking_id_when_one_is_wired() {
        let mut meta = AnalyticsMeta::new();
        let detection = meta.add_detection(det(0.25, 0.25, 0.5, 0.5, 0));
        let tracking = meta.push(AnalyticsNode::Tracking(Tracking { object_id: 4 }));
        meta.relate(detection, tracking, RelationKind::Tracks);
        let shapes = AnalyticsShapes::collect(&meta);
        let painted = &shapes.detections[0];

        assert_eq!(
            overlay(64, 64, 1).with_track(true).caption(painted, None),
            Some(String::from("ID:4"))
        );
        assert_eq!(
            overlay(64, 64, 1)
                .with_track(true)
                .with_score(true)
                .caption(painted, None),
            Some(String::from("ID:4 0.90"))
        );
    }

    #[test]
    fn caption_paints_a_bar_above_the_box_and_leaves_the_box_alone() {
        // A 64x64 canvas so the glyph scale is 1 and the box has room above it.
        let ov = overlay(64, 64, 1).with_score(true);
        let mut buf = solid(64, 64, [0, 0, 0, 255]);
        // Box pixels (16,16)..(47,47); the caption bar sits directly above row 16.
        ov.render(&mut buf, &detection_shapes(&[det(0.25, 0.25, 0.5, 0.5, 0)]));
        let bar_row = 16 - (GLYPH_HEIGHT as usize + 2);
        assert_ne!(
            px(&buf, 64, 16, bar_row),
            [0, 0, 0, 255],
            "caption bar painted above the box"
        );
        assert_eq!(
            px(&buf, 64, 20, 20),
            [0, 0, 0, 255],
            "box interior still untouched"
        );
        assert_eq!(
            px(&buf, 64, 16, bar_row - 1),
            [0, 0, 0, 255],
            "nothing painted above the bar"
        );
    }

    #[test]
    fn caption_on_a_box_at_the_top_edge_stays_on_canvas() {
        // No room above, so the bar drops inside the box rather than off-canvas.
        let ov = overlay(32, 32, 1).with_score(true);
        let mut buf = solid(32, 32, [0, 0, 0, 255]);
        ov.render(&mut buf, &detection_shapes(&[det(0.0, 0.0, 1.0, 1.0, 0)]));
        assert_ne!(
            px(&buf, 32, 1, 1),
            [0, 0, 0, 255],
            "caption drawn inside the top edge"
        );
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
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");
            core::task::Poll::Ready({
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

    #[tokio::test]
    async fn trail_joins_a_track_across_frames_and_leaves_it_alone_untracked() {
        // Two frames of one tracked box moving down the right half of a 32x32
        // canvas. The trail runs through the bottom edge of each box, so the
        // midpoint between the two bottom edges must be painted on frame two.
        let mut ov = AnalyticsOverlay::new().with_thickness(1).with_trail(true);
        ov.configure_pipeline(&rgba_caps(32, 32)).unwrap();
        let mut sink = PixelSink::default();
        for y in [0.25_f32, 0.5] {
            let meta = tracked_meta(det(0.5, y, 0.25, 0.25, 0), 3);
            let frame = rgba_frame_with_meta(32, 32, meta);
            ov.process(PipelinePacket::DataFrame(frame), &mut sink)
                .await
                .unwrap();
        }
        let out = sink.last.expect("frame forwarded");
        // Bottom-centre of the boxes: (20, 16) then (20, 24); the segment
        // between them covers (20, 20).
        assert_ne!(
            px(&out, 32, 20, 20),
            [0, 0, 0, 255],
            "trail drawn between the two positions"
        );

        // The same movement without a Tracks relation records nothing to draw.
        let mut untracked = AnalyticsOverlay::new().with_thickness(1).with_trail(true);
        untracked.configure_pipeline(&rgba_caps(32, 32)).unwrap();
        let mut sink = PixelSink::default();
        for y in [0.25_f32, 0.5] {
            let mut meta = AnalyticsMeta::new();
            meta.add_detection(det(0.5, y, 0.25, 0.25, 0));
            let frame = rgba_frame_with_meta(32, 32, meta);
            untracked
                .process(PipelinePacket::DataFrame(frame), &mut sink)
                .await
                .unwrap();
        }
        assert!(untracked.trails.is_empty(), "no track, no trail");
        assert_eq!(
            px(&sink.last.expect("forwarded"), 32, 20, 20),
            [0, 0, 0, 255],
            "nothing painted between the boxes"
        );
    }

    #[tokio::test]
    async fn trail_is_capped_at_its_length_and_expires_with_its_track() {
        let mut ov = AnalyticsOverlay::new()
            .with_trail(true)
            .with_trail_length(4);
        ov.configure_pipeline(&rgba_caps(32, 32)).unwrap();
        let mut sink = PixelSink::default();
        for _ in 0..10 {
            let frame = rgba_frame_with_meta(32, 32, tracked_meta(det(0.4, 0.4, 0.2, 0.2, 0), 8));
            ov.process(PipelinePacket::DataFrame(frame), &mut sink)
                .await
                .unwrap();
        }
        assert_eq!(
            ov.trails[&8].points.len(),
            4,
            "only the last trail-length points are kept"
        );

        // Frames with no detections age the trail out; it survives the first few.
        for _ in 0..TRAIL_TTL_FRAMES {
            let frame = rgba_frame_with_meta(32, 32, AnalyticsMeta::new());
            ov.process(PipelinePacket::DataFrame(frame), &mut sink)
                .await
                .unwrap();
        }
        assert!(
            ov.trails.contains_key(&8),
            "trail outlives a gap in the track"
        );
        let frame = rgba_frame_with_meta(32, 32, AnalyticsMeta::new());
        ov.process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();
        assert!(
            ov.trails.is_empty(),
            "trail dropped once its track is stale"
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
