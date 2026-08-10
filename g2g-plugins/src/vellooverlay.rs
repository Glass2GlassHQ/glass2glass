//! Vello GPU analytics overlay (M102, masks M994): the GPU companion to the CPU
//! [`AnalyticsOverlay`](crate::analyticsoverlay), rendering the `AnalyticsMeta`
//! with the Vello GPU 2D renderer (wgpu) instead of the CPU blend loop. The HD /
//! many-box path: stroking dozens of antialiased boxes per frame is a GPU job, and
//! the result stays on the GPU.
//!
//! Same three shapes as the CPU backend, in the same palette: a solid box per
//! detection, a translucent image fill per segmentation mask, and a dashed
//! rectangle per region of interest.
//!
//! `Caps::RawVideo{Rgba8}` in (system memory), [`MemoryDomain::WgpuTexture`] out:
//! the input picture is drawn into a Vello scene as a full-frame image, the
//! analytics are drawn on top, and the scene is rendered into a
//! `wgpu::Texture` that the output frame carries by keep-alive. Nothing is read
//! back to the CPU, so a downstream GPU sink presents it directly (the keep-on-GPU
//! contract the decode-side CUDA / D3D11 domains already use). The pixel format
//! and geometry are unchanged, so the negotiated caps are identity; only the
//! memory domain changes.
//!
//! `vello-overlay` feature (implies `std` + `analytics`). The CPU overlay remains
//! the `no_std` baseline; this element is never on the RTOS path.
//!
//! [`VelloTextOverlay`] (the `vello-text-overlay` feature) is the same trick for
//! subtitle cues: it shares the renderer and the frame-image background with the
//! analytics overlay, and draws the cue's glyph outlines as Vello glyph runs.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::OwnedWgpuTexture;
use g2g_core::{
    AnalyticsMeta, AsyncElement, BBox, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim,
    G2gError, MemoryDomain, OutputSink, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec, RawVideoFormat,
};

use crate::analyticsoverlay::{
    palette_rgb, AnalyticsShapes, PaintedMask, MASK_ALPHA_DEFAULT, ROI_DASH_PX,
};
use crate::gpu::{gpu_err, GpuContext, WgpuTextureKeepAlive};
use vello::kurbo::{Affine, Rect, Stroke};
use vello::peniko::{Blob, Color, ImageAlphaType, ImageData, ImageFormat};
use vello::wgpu;
use vello::{AaConfig, AaSupport, RenderParams, Renderer, RendererOptions, Scene};

#[cfg(feature = "vello-text-overlay")]
use g2g_core::ElementMetadata;
#[cfg(feature = "vello-text-overlay")]
use vello::peniko::{Fill, FontData};
#[cfg(feature = "vello-text-overlay")]
use vello::Glyph;

#[cfg(feature = "vello-text-overlay")]
use crate::subparse::Cue;
#[cfg(feature = "vello-text-overlay")]
use crate::textoverlay::TextOverlay;
#[cfg(feature = "vello-text-overlay")]
use crate::textshape::FontId;

/// Renders the detection boxes, segmentation masks and regions of interest of an
/// attached [`AnalyticsMeta`] onto an RGBA8 frame with Vello, emitting a
/// GPU-resident [`MemoryDomain::WgpuTexture`].
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::vellooverlay::VelloAnalyticsOverlay;
///
/// let element = VelloAnalyticsOverlay::new().with_thickness(2.0);
/// ```
pub struct VelloAnalyticsOverlay {
    width: u32,
    height: u32,
    /// Outline stroke width in pixels.
    thickness: f64,
    /// Alpha the mask fill is drawn at (0 = invisible, 255 = opaque).
    mask_alpha: u8,
    configured: bool,
    drawn: u64,
    /// A shared device to render on, set via [`with_context`](Self::with_context)
    /// (eg the same context the downstream `WgpuSink` presents on, so the texture
    /// handoff is copy-free). When unset, [`ensure_gpu`] opens its own device on
    /// the first frame.
    ctx: Option<GpuContext>,
    gpu: Option<Gpu>,
}

/// Lazily-built GPU resources: the wgpu device/queue and the Vello renderer.
struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: Renderer,
}

impl Gpu {
    /// Render `scene` into a fresh `w` x `h` RGBA8 texture, returned for an
    /// output frame to own.
    fn render_scene(&mut self, scene: &Scene, w: u32, h: u32) -> Result<wgpu::Texture, G2gError> {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vello-overlay-target"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            // STORAGE_BINDING: Vello's fine stage writes the image as a storage
            // texture. COPY_SRC: lets a sink (or a test) read it back.
            // TEXTURE_BINDING: lets a GPU sink sample it for presentation.
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            // The pixels are sRGB-encoded video; an embedder sampling the frame
            // in a lit/tonemapped scene needs an sRGB view for correct gamma.
            view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer
            .render_to_texture(
                &self.device,
                &self.queue,
                scene,
                &view,
                &RenderParams {
                    // Transparent base: the image fill covers the frame, so the
                    // clear colour is only visible where the image does not draw.
                    base_color: Color::from_rgba8(0, 0, 0, 0),
                    width: w,
                    height: h,
                    antialiasing_method: AaConfig::Area,
                },
            )
            .map_err(gpu_err)?;
        Ok(texture)
    }
}

/// Build the wgpu device/queue and Vello renderer on the first frame, on the
/// shared `ctx` if one was given, else on a private headless device. Maps a
/// missing adapter / device to a structured hardware error so a host without a
/// GPU fails cleanly (and tests skip).
async fn ensure_gpu(gpu: &mut Option<Gpu>, ctx: &Option<GpuContext>) -> Result<(), G2gError> {
    if gpu.is_some() {
        return Ok(());
    }
    let ctx = match ctx.clone() {
        Some(ctx) => ctx,
        None => GpuContext::headless().await?,
    };
    let device = ctx.device;
    let queue = ctx.queue;
    let renderer = Renderer::new(
        &device,
        RendererOptions {
            use_cpu: false,
            // Area AA only: we never request MSAA, so do not compile those
            // pipeline permutations.
            antialiasing_support: AaSupport {
                area: true,
                msaa8: false,
                msaa16: false,
            },
            num_init_threads: None,
            pipeline_cache: None,
        },
    )
    .map_err(gpu_err)?;
    *gpu = Some(Gpu {
        device,
        queue,
        renderer,
    });
    Ok(())
}

/// Draw `rgba` (a full-frame `w` x `h` picture, consumed) as the scene's
/// background, so what is drawn after it composites over the actual frame on the
/// GPU (Vello clears the target first).
fn draw_frame_image(scene: &mut Scene, rgba: Vec<u8>, w: u32, h: u32) {
    let image = ImageData {
        data: Blob::from(rgba),
        format: ImageFormat::Rgba8,
        alpha_type: ImageAlphaType::Alpha,
        width: w,
        height: h,
    };
    scene.draw_image(&image, Affine::IDENTITY);
}

/// An opaque-or-translucent RGBA colour as a Vello brush colour.
#[cfg(feature = "vello-text-overlay")]
fn brush_color(rgba: [u8; 4]) -> Color {
    Color::from_rgba8(rgba[0], rgba[1], rgba[2], rgba[3])
}

impl core::fmt::Debug for VelloAnalyticsOverlay {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("VelloAnalyticsOverlay")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("thickness", &self.thickness)
            .field("mask_alpha", &self.mask_alpha)
            .field("configured", &self.configured)
            .field("drawn", &self.drawn)
            .field("gpu_ready", &self.gpu.is_some())
            .finish()
    }
}

impl Default for VelloAnalyticsOverlay {
    fn default() -> Self {
        Self::new()
    }
}

impl VelloAnalyticsOverlay {
    /// A new overlay with a 3px stroke. Geometry and GPU are set lazily.
    pub fn new() -> Self {
        Self {
            width: 0,
            height: 0,
            thickness: 3.0,
            mask_alpha: MASK_ALPHA_DEFAULT,
            configured: false,
            drawn: 0,
            ctx: None,
            gpu: None,
        }
    }

    /// Render on a shared [`GpuContext`] instead of opening a private device.
    /// Pass the same context the downstream [`WgpuSink`](crate::wgpusink) uses so
    /// the produced texture lives on the sink's device and presents with no copy.
    pub fn with_context(mut self, ctx: GpuContext) -> Self {
        self.ctx = Some(ctx);
        self
    }

    /// Set the box outline stroke width in pixels.
    pub fn with_thickness(mut self, px: f64) -> Self {
        self.thickness = px.max(0.5);
        self
    }

    /// Set the alpha the segmentation mask fill is drawn at (0..=255).
    pub fn with_mask_alpha(mut self, alpha: u8) -> Self {
        self.mask_alpha = alpha;
        self
    }

    /// Count of frames rendered.
    pub fn drawn_count(&self) -> u64 {
        self.drawn
    }

    /// RGBA8 at fixed geometry, the only input this element renders.
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

    fn accepts(caps: &Caps) -> bool {
        matches!(
            caps,
            Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                ..
            }
        )
    }

    /// Render `rgba` (full-frame image, consumed) with `shapes` drawn over it into
    /// a fresh `wgpu::Texture`, returned for the output frame to own.
    fn render(
        &mut self,
        rgba: Vec<u8>,
        shapes: &AnalyticsShapes,
    ) -> Result<wgpu::Texture, G2gError> {
        let (w, h) = (self.width, self.height);
        let thickness = self.thickness;
        let mask_alpha = self.mask_alpha;
        let gpu = self.gpu.as_mut().ok_or(G2gError::NotConfigured)?;

        let mut scene = Scene::new();
        // The caller already owns this buffer, so move it into the blob.
        draw_frame_image(&mut scene, rgba, w, h);

        // Mask fills first, so a box or ROI stroke stays readable over one.
        for mask in &shapes.masks {
            draw_mask(&mut scene, mask, w, h, mask_alpha);
        }
        let stroke = Stroke::new(thickness);
        for detection in &shapes.detections {
            if let Some(rect) = pixel_rect(detection.bbox, w, h) {
                scene.stroke(
                    &stroke,
                    Affine::IDENTITY,
                    palette_color(detection.label),
                    None,
                    &rect,
                );
            }
        }
        let dashed = Stroke::new(thickness).with_dashes(0.0, [ROI_DASH_PX as f64; 2]);
        for roi in &shapes.rois {
            if let Some(rect) = pixel_rect(roi.roi.bbox, w, h) {
                scene.stroke(
                    &dashed,
                    Affine::IDENTITY,
                    palette_color(roi.palette_index),
                    None,
                    &rect,
                );
            }
        }

        gpu.render_scene(&scene, w, h)
    }
}

/// The pixel rectangle a normalized box covers on a `w` x `h` canvas, or `None`
/// when it collapses to nothing.
fn pixel_rect(bbox: BBox, w: u32, h: u32) -> Option<Rect> {
    let x0 = (bbox.x as f64) * w as f64;
    let y0 = (bbox.y as f64) * h as f64;
    let x1 = ((bbox.x + bbox.w) as f64) * w as f64;
    let y1 = ((bbox.y + bbox.h) as f64) * h as f64;
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some(Rect::new(x0, y0, x1, y1))
}

/// Draw an instance's mask as a translucent image fill scaled onto its box: the
/// mask spans exactly the box (its grid is the model's, not the frame's), so
/// scaling the mask's own rect over the box is the whole placement.
fn draw_mask(scene: &mut Scene, painted: &PaintedMask, w: u32, h: u32, mask_alpha: u8) {
    let mask = &painted.segmentation.mask;
    let (mask_w, mask_h) = (mask.width(), mask.height());
    let Some(rect) = pixel_rect(painted.segmentation.bbox, w, h) else {
        return;
    };
    if mask_w == 0 || mask_h == 0 {
        return;
    }
    let rgb = palette_rgb(painted.palette_index);
    let mut data = Vec::with_capacity((mask_w as usize) * (mask_h as usize) * 4);
    for j in 0..mask_h {
        for i in 0..mask_w {
            let coverage = mask.sample(i, j).unwrap_or(0) as u32;
            // Every sample carries the fill colour, covered or not, so the
            // sampler cannot blend a covered edge sample toward black.
            let alpha = (coverage * mask_alpha as u32 / 255) as u8;
            data.extend_from_slice(&[rgb[0], rgb[1], rgb[2], alpha]);
        }
    }
    let image = ImageData {
        data: Blob::from(data),
        format: ImageFormat::Rgba8,
        alpha_type: ImageAlphaType::Alpha,
        width: mask_w,
        height: mask_h,
    };
    let transform = Affine::translate((rect.x0, rect.y0))
        * Affine::scale_non_uniform(rect.width() / mask_w as f64, rect.height() / mask_h as f64);
    scene.draw_image(&image, transform);
}

/// The opaque stroke colour of a palette slot, from the shared CPU-overlay palette
/// so the two backends draw the same slot the same colour.
fn palette_color(index: u32) -> Color {
    let c = palette_rgb(index);
    Color::from_rgba8(c[0], c[1], c[2], 0xFF)
}

impl AsyncElement for VelloAnalyticsOverlay {
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
        // Identity on caps: same RGBA8 format and geometry; only the memory
        // domain changes (System -> WgpuTexture), which caps do not describe.
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
                PipelinePacket::DataFrame(frame) => {
                    let shapes = frame
                        .meta
                        .get::<AnalyticsMeta>()
                        .map(AnalyticsShapes::collect)
                        .unwrap_or_default();
                    let Some(slice) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let need = self.width as usize * self.height as usize * 4;
                    if slice.len() < need {
                        return Err(G2gError::CapsMismatch);
                    }
                    let rgba = slice[..need].to_vec();

                    ensure_gpu(&mut self.gpu, &self.ctx).await?;
                    let texture = self.render(rgba, &shapes)?;

                    let domain = MemoryDomain::WgpuTexture(OwnedWgpuTexture::new(
                        self.width,
                        self.height,
                        alloc::sync::Arc::new(WgpuTextureKeepAlive(texture)),
                    ));
                    let mut out_frame = Frame::new(domain, frame.timing, frame.sequence);
                    // Carry the analytics forward so a downstream stage still sees
                    // the detections on the GPU frame.
                    out_frame.meta = frame.meta;
                    self.drawn += 1;
                    out.push(PipelinePacket::DataFrame(out_frame)).await?;
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

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new(
                "thickness",
                PropKind::Double,
                "box outline stroke width in pixels",
            )
            .with_range("0.5", "65535")
            .with_default("3"),
            PropertySpec::new(
                "mask-alpha",
                PropKind::Uint,
                "alpha the segmentation mask fill is drawn at (0..255)",
            )
            .with_range("0", "255")
            .with_default("96"),
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "thickness" => {
                self.thickness = value.as_double().ok_or(PropError::Type)?.max(0.5);
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
            "thickness" => Some(PropValue::Double(self.thickness)),
            "mask-alpha" => Some(PropValue::Uint(self.mask_alpha as u64)),
            _ => None,
        }
    }
}

/// Renders the subtitle cues active at the frame's PTS with Vello, emitting a
/// GPU-resident [`MemoryDomain::WgpuTexture`]: the GPU companion to the CPU
/// [`TextOverlay`](crate::textoverlay), for a pipeline that keeps frames on the
/// GPU (decode -> overlay -> present) and must not round-trip text through
/// system memory.
///
/// Cues, fonts, colours, cue selection and placement are the CPU overlay's
/// (`location=` / `font=` / `color=` / `font-size=` / `font-variations=` behave
/// the same, and cosmic-text shapes and picks fallback faces the same way, so a
/// Latin cue with CJK in it uses the same faces here). Only the drawing differs:
/// the glyph outlines of the face the shaper chose go to Vello as glyph runs,
/// and the backing box is a filled rect, both composited over the frame image on
/// the GPU.
///
/// Two limits against the CPU element. `vertical:rl` / `lr` cues draw nothing
/// here: vertical writing is a shaping limit (cosmic-text is horizontal-only)
/// and the CPU element covers it with its own column renderer, which has no
/// glyph runs to hand over. And there is no bitmap-font fallback, so a host
/// where neither `font=` nor font discovery yields a usable face renders no
/// text rather than the 8x8 ASCII baseline.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::subparse::parse_srt;
/// use g2g_plugins::vellooverlay::VelloTextOverlay;
///
/// let element = VelloTextOverlay::new()
///     .with_cues(parse_srt("1\n00:00:00,000 --> 00:00:02,000\nhello\n"));
/// assert_eq!(element.cue_count(), 1);
/// ```
#[cfg(feature = "vello-text-overlay")]
pub struct VelloTextOverlay {
    width: u32,
    height: u32,
    configured: bool,
    /// The cue list, fonts, colours and shaper. Holding the CPU element rather
    /// than a second copy of that state is what keeps the two backends drawing
    /// the same cue in the same place.
    text: TextOverlay,
    /// Faces already handed to Vello, keyed by the shaper's face id. A face's
    /// bytes are copied once here, never per frame.
    fonts: Vec<(FontId, FontData)>,
    ctx: Option<GpuContext>,
    gpu: Option<Gpu>,
    drawn: u64,
}

#[cfg(feature = "vello-text-overlay")]
impl core::fmt::Debug for VelloTextOverlay {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("VelloTextOverlay")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("configured", &self.configured)
            .field("cues", &self.text.cue_count())
            .field("drawn", &self.drawn)
            .field("gpu_ready", &self.gpu.is_some())
            .finish()
    }
}

#[cfg(feature = "vello-text-overlay")]
impl Default for VelloTextOverlay {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "vello-text-overlay")]
impl VelloTextOverlay {
    /// An overlay with no cues. Geometry and GPU are set lazily.
    pub fn new() -> Self {
        Self {
            width: 0,
            height: 0,
            configured: false,
            text: TextOverlay::new(),
            fonts: Vec::new(),
            ctx: None,
            gpu: None,
            drawn: 0,
        }
    }

    /// Render on a shared [`GpuContext`] instead of opening a private device, so
    /// the texture presents on the downstream sink's device with no copy.
    pub fn with_context(mut self, ctx: GpuContext) -> Self {
        self.ctx = Some(ctx);
        self
    }

    /// Use a preparsed cue list (`subparse::parse_srt` and friends).
    pub fn with_cues(mut self, cues: Vec<Cue>) -> Self {
        self.text = self.text.with_cues(cues);
        self
    }

    /// Append a font from a `.ttf` / `.otf` / `.ttc` file to the fallback chain.
    /// Without one the shaper renders from the discovered system fonts.
    pub fn with_font(mut self, path: impl AsRef<str>) -> Result<Self, G2gError> {
        self.text = self.text.with_font(path)?;
        Ok(self)
    }

    /// Append a font from in-memory face bytes to the fallback chain;
    /// `collection_index` selects a face in a `.ttc`.
    pub fn with_font_bytes(
        mut self,
        bytes: &[u8],
        collection_index: u32,
    ) -> Result<Self, G2gError> {
        self.text = self.text.with_font_bytes(bytes, collection_index)?;
        Ok(self)
    }

    /// Set the text height in pixels; 0 derives it from the frame height.
    pub fn with_font_size(mut self, px: u32) -> Self {
        self.text = self.text.with_font_size(px);
        self
    }

    /// Set the opaque text colour.
    pub fn with_text_color(mut self, rgb: [u8; 3]) -> Self {
        self.text = self.text.with_text_color(rgb);
        self
    }

    /// Number of loaded cues.
    pub fn cue_count(&self) -> usize {
        self.text.cue_count()
    }

    /// Count of frames rendered (whether or not a cue was active).
    pub fn drawn_count(&self) -> u64 {
        self.drawn
    }

    /// Render `rgba` (full-frame image, consumed) with the cues active at `t_ns`
    /// drawn over it, into a fresh `wgpu::Texture` for the output frame to own.
    fn render(&mut self, rgba: Vec<u8>, t_ns: u64) -> Result<wgpu::Texture, G2gError> {
        let (w, h) = (self.width, self.height);
        let mut scene = Scene::new();
        draw_frame_image(&mut scene, rgba, w, h);
        // Only touch the shaper when something is on screen: building it scans
        // the system font directories.
        if self.text.has_cue_at(t_ns) {
            self.draw_cues(&mut scene, t_ns);
        }
        let gpu = self.gpu.as_mut().ok_or(G2gError::NotConfigured)?;
        gpu.render_scene(&scene, w, h)
    }

    /// Draw each active cue's backing box, span fills, shadows and glyphs,
    /// batching the glyphs into runs of one face, colour and size (a
    /// `::cue(.class)` span or a fallback face starts a new run). Every shadow
    /// is drawn before any glyph, so a neighbour's shadow never lands on top of
    /// a glyph.
    fn draw_cues(&mut self, scene: &mut Scene, t_ns: u64) {
        let placed = self.text.place_shaped_cues(t_ns);
        for cue in placed {
            let (x, y, box_w, box_h) = cue.background;
            if box_w > 0 && box_h > 0 {
                fill_rect(scene, (x, y, box_w, box_h), cue.background_color);
            }
            for (rect, color) in &cue.span_backgrounds {
                fill_rect(scene, *rect, *color);
            }
            // Shadows first, then the glyphs over them.
            for drawing_shadows in [true, false] {
                let mut start = 0;
                while start < cue.glyphs.len() {
                    let head = &cue.glyphs[start];
                    let (font_id, size, shadow) = (head.key.font_id, head.font_size, head.shadow);
                    let color = head.color;
                    let end = cue.glyphs[start..]
                        .iter()
                        .position(|g| {
                            g.key.font_id != font_id
                                || g.color != color
                                || g.font_size != size
                                || g.shadow != shadow
                        })
                        .map_or(cue.glyphs.len(), |n| start + n);
                    let batch = &cue.glyphs[start..end];
                    start = end;
                    let (offset, brush) = match (drawing_shadows, shadow) {
                        (true, Some(shadow)) => (
                            (shadow.offset_x as f32, shadow.offset_y as f32),
                            shadow.color,
                        ),
                        (true, None) => continue,
                        (false, _) => ((0.0, 0.0), color),
                    };
                    let run = batch.iter().map(|g| Glyph {
                        id: g.key.glyph_id as u32,
                        x: g.x as f32 + offset.0,
                        y: g.y as f32 + offset.1,
                    });
                    if let Some(font) = font_handle(&mut self.fonts, &mut self.text, font_id) {
                        scene
                            .draw_glyphs(font)
                            .font_size(size)
                            .brush(brush_color(brush))
                            .draw(Fill::NonZero, run);
                    }
                }
            }
        }
    }
}

/// Fill one `(x, y, width, height)` rectangle in frame pixels.
#[cfg(feature = "vello-text-overlay")]
fn fill_rect(scene: &mut Scene, (x, y, w, h): (i32, i32, i32, i32), color: [u8; 4]) {
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        brush_color(color),
        None,
        &Rect::new(x as f64, y as f64, (x + w) as f64, (y + h) as f64),
    );
}

/// The Vello handle for the shaper face `id`, copying the face bytes into the
/// cache on first use. A free function so the cache and the shaper are borrowed
/// separately from the element.
#[cfg(feature = "vello-text-overlay")]
fn font_handle<'a>(
    cache: &'a mut Vec<(FontId, FontData)>,
    text: &mut TextOverlay,
    id: FontId,
) -> Option<&'a FontData> {
    if let Some(pos) = cache.iter().position(|(cached, _)| *cached == id) {
        return Some(&cache[pos].1);
    }
    let (bytes, index) = text.face_data(id)?;
    cache.push((id, FontData::new(Blob::from(bytes), index)));
    cache.last().map(|(_, font)| font)
}

#[cfg(feature = "vello-text-overlay")]
impl AsyncElement for VelloTextOverlay {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        if TextOverlay::accepts(upstream_caps) {
            Ok(upstream_caps.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        // Identity on caps: same RGBA8 format and geometry; only the memory
        // domain changes (System -> WgpuTexture), which caps do not describe.
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| {
            if TextOverlay::accepts(input) {
                CapsSet::one(input.clone())
            } else {
                CapsSet::from_alternatives(Vec::new())
            }
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (w, h) = TextOverlay::dims(absolute_caps).ok_or(G2gError::CapsMismatch)?;
        // The cue placement is the CPU element's, so it needs the geometry too.
        self.text.configure_pipeline(absolute_caps)?;
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
                PipelinePacket::DataFrame(frame) => {
                    let Some(slice) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let need = self.width as usize * self.height as usize * 4;
                    if slice.len() < need {
                        return Err(G2gError::CapsMismatch);
                    }
                    let rgba = slice[..need].to_vec();

                    ensure_gpu(&mut self.gpu, &self.ctx).await?;
                    let texture = self.render(rgba, frame.timing.pts_ns)?;

                    let domain = MemoryDomain::WgpuTexture(OwnedWgpuTexture::new(
                        self.width,
                        self.height,
                        alloc::sync::Arc::new(WgpuTextureKeepAlive(texture)),
                    ));
                    let mut out_frame = Frame::new(domain, frame.timing, frame.sequence);
                    out_frame.meta = frame.meta;
                    self.drawn += 1;
                    out.push(PipelinePacket::DataFrame(out_frame)).await?;
                }
                PipelinePacket::CapsChanged(caps) => {
                    if let Some((w, h)) = TextOverlay::dims(&caps) {
                        self.text.configure_pipeline(&caps)?;
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

    fn properties(&self) -> &'static [PropertySpec] {
        crate::textoverlay::TEXTOVERLAY_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Vello text overlay",
            "Filter/Editor/Video",
            "Renders subtitle cues over video on the GPU, output as a wgpu texture",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        self.text.set_property(name, value)
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        self.text.get_property(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::shared_ctx;
    use g2g_core::memory::SystemSlice;
    use g2g_core::{
        AnalyticsNode, FrameTiming, Mask, ObjectDetection, PushOutcome, Rate, RelationKind, Roi,
        Segmentation,
    };

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

    /// Read an Rgba8 texture back to a tightly-packed CPU buffer (un-padding the
    /// 256-byte row alignment wgpu requires for the copy).
    fn read_back(gpu: &Gpu, texture: &wgpu::Texture, w: u32, h: u32) -> Vec<u8> {
        let unpadded = (w * 4) as usize;
        let padded = unpadded.next_multiple_of(256);
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (padded * h as usize) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded as u32),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit([enc.finish()]);

        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        gpu.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
        rx.recv().unwrap().unwrap();

        let mapped = slice.get_mapped_range();
        let mut out = Vec::with_capacity(unpadded * h as usize);
        for row in 0..h as usize {
            let start = row * padded;
            out.extend_from_slice(&mapped[start..start + unpadded]);
        }
        drop(mapped);
        buffer.unmap();
        out
    }

    /// Capturing sink that keeps the last forwarded frame.
    #[derive(Default)]
    struct FrameSink {
        last: Option<Frame>,
    }
    impl OutputSink for FrameSink {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async move {
                if let PipelinePacket::DataFrame(frame) = packet {
                    self.last = Some(frame);
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }

    #[test]
    fn intercept_rejects_non_rgba() {
        let ov = VelloAnalyticsOverlay::new();
        let nv12 = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(8),
            height: Dim::Fixed(8),
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
        };
        assert!(ov.intercept_caps(&nv12).is_err());
        assert!(ov.intercept_caps(&rgba_caps(8, 8)).is_ok());
    }

    #[tokio::test]
    async fn renders_box_onto_gpu_texture() {
        let Some(ctx) = shared_ctx().await else {
            std::eprintln!("no wgpu adapter; skipping Vello GPU render test");
            return;
        };
        let (w, h) = (64u32, 64u32);
        let mut ov = VelloAnalyticsOverlay::new()
            .with_thickness(4.0)
            .with_context(ctx);
        ov.configure_pipeline(&rgba_caps(w, h)).unwrap();

        // Dark-grey input frame; a class-0 (red) box covering the centre.
        let mut bytes = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..w * h {
            bytes.extend_from_slice(&[20, 20, 20, 255]);
        }
        let mut frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        let mut a = AnalyticsMeta::new();
        a.add_detection(det(0.25, 0.25, 0.5, 0.5, 0)); // box spans pixels 16..48
        frame.meta.attach(a);

        let mut sink = FrameSink::default();
        ov.process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();

        let out = sink.last.expect("frame forwarded");
        let MemoryDomain::WgpuTexture(owned) = &out.domain else {
            panic!("output is a GPU texture domain");
        };
        assert_eq!((owned.width, owned.height), (w, h));
        let tex = crate::gpu::texture_of(owned).expect("texture keep-alive");

        let pixels = read_back(ov.gpu.as_ref().unwrap(), tex, w, h);
        let px = |x: u32, y: u32| {
            let i = ((y * w + x) * 4) as usize;
            [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
        };
        // A pixel on the top edge of the box (~row 16) is reddish (class 0).
        let edge = px(32, 16);
        assert!(
            edge[0] > 120 && edge[0] > edge[1] + 40 && edge[0] > edge[2] + 40,
            "box edge is red: {edge:?}"
        );
        // The box interior shows the dark-grey input, not the stroke colour.
        let interior = px(32, 32);
        assert!(
            interior[0] < 70 && interior[1] < 70 && interior[2] < 70,
            "interior is the dark input frame: {interior:?}"
        );
        assert_eq!(ov.drawn_count(), 1);
    }

    #[tokio::test]
    async fn renders_mask_fill_and_dashed_roi_onto_gpu_texture() {
        let Some(ctx) = shared_ctx().await else {
            std::eprintln!("no wgpu adapter; skipping Vello GPU mask render test");
            return;
        };
        let (w, h) = (64u32, 64u32);
        let mut ov = VelloAnalyticsOverlay::new()
            .with_thickness(4.0)
            .with_context(ctx);
        ov.configure_pipeline(&rgba_caps(w, h)).unwrap();

        let mut bytes = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..w * h {
            bytes.extend_from_slice(&[20, 20, 20, 255]);
        }
        let mut frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        // An instance over pixels 16..48 whose 2x2 mask covers only its left
        // column, so the fill is pixels 16..32; the ROI is that mask-tight half.
        let bbox = BBox {
            x: 0.25,
            y: 0.25,
            w: 0.5,
            h: 0.5,
        };
        let mask = Mask::new(2, 2, 2, alloc::vec![255, 0, 255, 0]).expect("mask geometry");
        let mut analytics = AnalyticsMeta::new();
        let instance = analytics.push(AnalyticsNode::Segmentation(Segmentation {
            bbox,
            label: 0,
            confidence: 0.9,
            mask,
        }));
        let roi = analytics.push(AnalyticsNode::Roi(Roi {
            bbox: BBox {
                x: 0.25,
                y: 0.25,
                w: 0.25,
                h: 0.5,
            },
            id: 5,
            label: 0,
        }));
        analytics.relate(instance, roi, RelationKind::Contains);
        frame.meta.attach(analytics);

        let mut sink = FrameSink::default();
        ov.process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();

        let out = sink.last.expect("frame forwarded");
        let MemoryDomain::WgpuTexture(owned) = &out.domain else {
            panic!("output is a GPU texture domain");
        };
        let tex = crate::gpu::texture_of(owned).expect("texture keep-alive");
        let pixels = read_back(ov.gpu.as_ref().unwrap(), tex, w, h);
        let px = |x: u32, y: u32| {
            let i = ((y * w + x) * 4) as usize;
            [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
        };

        // Inside the covered half, clear of the ROI stroke: reddish, but faint
        // enough that the dark input still shows through.
        let filled = px(24, 32);
        assert!(
            filled[0] > 80 && filled[0] < 200 && filled[0] > filled[1] + 40,
            "mask fill is translucent red: {filled:?}"
        );
        // The uncovered half of the same box keeps the input frame.
        let uncovered = px(40, 32);
        assert!(
            uncovered.iter().take(3).all(|c| *c < 60),
            "uncovered mask samples untouched: {uncovered:?}"
        );
        // The ROI outline dashes: along its top edge some pixels carry the opaque
        // stroke and some only the fill underneath it.
        let top_edge: Vec<[u8; 4]> = (17..31).map(|x| px(x, 16)).collect();
        assert!(
            top_edge.iter().any(|p| p[0] > 200),
            "a dash paints on the ROI edge: {top_edge:?}"
        );
        assert!(
            top_edge.iter().any(|p| p[0] < 150),
            "a gap leaves the ROI edge unstroked: {top_edge:?}"
        );
        assert_eq!(ov.drawn_count(), 1);
    }
}
