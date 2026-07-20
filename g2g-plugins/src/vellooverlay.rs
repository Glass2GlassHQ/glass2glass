//! Vello GPU analytics overlay (M102): the GPU companion to the CPU
//! [`AnalyticsOverlay`](crate::analyticsoverlay), rendering the `AnalyticsMeta`
//! detection boxes with the Vello GPU 2D renderer (wgpu) instead of the CPU
//! blend loop. The HD / many-box path: stroking dozens of antialiased boxes per
//! frame is a GPU job, and the result stays on the GPU.
//!
//! `Caps::RawVideo{Rgba8}` in (system memory), [`MemoryDomain::WgpuTexture`] out:
//! the input picture is drawn into a Vello scene as a full-frame image, the
//! detection boxes are stroked on top, and the scene is rendered into a
//! `wgpu::Texture` that the output frame carries by keep-alive. Nothing is read
//! back to the CPU, so a downstream GPU sink presents it directly (the keep-on-GPU
//! contract the decode-side CUDA / D3D11 domains already use). The pixel format
//! and geometry are unchanged, so the negotiated caps are identity; only the
//! memory domain changes.
//!
//! `vello-overlay` feature (implies `std` + `analytics`). The CPU overlay remains
//! the `no_std` baseline; this element is never on the RTOS path.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::OwnedWgpuTexture;
use g2g_core::{
    AnalyticsMeta, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError,
    MemoryDomain, ObjectDetection, OutputSink, PipelinePacket, RawVideoFormat,
};

use crate::gpu::{gpu_err, GpuContext, WgpuTextureKeepAlive};
use vello::kurbo::{Affine, Rect, Stroke};
use vello::peniko::{Blob, Color, ImageAlphaType, ImageData, ImageFormat};
use vello::wgpu;
use vello::{AaConfig, AaSupport, RenderParams, Renderer, RendererOptions, Scene};

/// Renders detection bounding boxes from an attached [`AnalyticsMeta`] onto an
/// RGBA8 frame with Vello, emitting a GPU-resident [`MemoryDomain::WgpuTexture`].
pub struct VelloAnalyticsOverlay {
    width: u32,
    height: u32,
    /// Outline stroke width in pixels.
    thickness: f64,
    configured: bool,
    drawn: u64,
    /// A shared device to render on, set via [`with_context`](Self::with_context)
    /// (eg the same context the downstream `WgpuSink` presents on, so the texture
    /// handoff is copy-free). When unset, [`ensure_gpu`](Self::ensure_gpu) opens
    /// its own device on the first frame.
    ctx: Option<GpuContext>,
    gpu: Option<Gpu>,
}

/// Lazily-built GPU resources: the wgpu device/queue and the Vello renderer.
struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: Renderer,
}

impl core::fmt::Debug for VelloAnalyticsOverlay {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("VelloAnalyticsOverlay")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("thickness", &self.thickness)
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

    /// Build the wgpu device/queue and Vello renderer on the first frame. Maps a
    /// missing adapter / device to a structured hardware error so a host without
    /// a GPU fails cleanly (and tests skip).
    async fn ensure_gpu(&mut self) -> Result<(), G2gError> {
        if self.gpu.is_some() {
            return Ok(());
        }
        // Use the shared context if one was provided, else open a private device.
        let ctx = match self.ctx.clone() {
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
        self.gpu = Some(Gpu {
            device,
            queue,
            renderer,
        });
        Ok(())
    }

    /// Render `rgba` (full-frame image, consumed) with `detections` stroked over
    /// it into a fresh `wgpu::Texture`, returned for the output frame to own.
    fn render(
        &mut self,
        rgba: Vec<u8>,
        detections: &[ObjectDetection],
    ) -> Result<wgpu::Texture, G2gError> {
        let (w, h) = (self.width, self.height);
        let thickness = self.thickness;
        let gpu = self.gpu.as_mut().ok_or(G2gError::NotConfigured)?;

        let mut scene = Scene::new();
        // The input picture as a full-frame image fill, so the boxes composite
        // over the actual frame on the GPU (Vello clears the target first). The
        // caller already owns this buffer, so move it into the blob.
        let image = ImageData {
            data: Blob::from(rgba),
            format: ImageFormat::Rgba8,
            alpha_type: ImageAlphaType::Alpha,
            width: w,
            height: h,
        };
        scene.draw_image(&image, Affine::IDENTITY);

        let stroke = Stroke::new(thickness);
        for d in detections {
            // Denormalize the [0,1] box to pixel coordinates.
            let x0 = (d.bbox.x as f64) * w as f64;
            let y0 = (d.bbox.y as f64) * h as f64;
            let x1 = ((d.bbox.x + d.bbox.w) as f64) * w as f64;
            let y1 = ((d.bbox.y + d.bbox.h) as f64) * h as f64;
            if x1 <= x0 || y1 <= y0 {
                continue;
            }
            let rect = Rect::new(x0, y0, x1, y1);
            scene.stroke(&stroke, Affine::IDENTITY, class_color(d.label), None, &rect);
        }

        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
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
        gpu.renderer
            .render_to_texture(
                &gpu.device,
                &gpu.queue,
                &scene,
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

/// The opaque stroke colour for a class label, from the shared CPU-overlay
/// palette so the two backends draw the same classes the same colour.
fn class_color(label: u32) -> Color {
    let c = crate::analyticsoverlay::class_rgb(label);
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
                    let detections: Vec<ObjectDetection> = frame
                        .meta
                        .get::<AnalyticsMeta>()
                        .map(|a| a.detections().copied().collect())
                        .unwrap_or_default();
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let need = self.width as usize * self.height as usize * 4;
                    if slice.as_slice().len() < need {
                        return Err(G2gError::CapsMismatch);
                    }
                    let rgba = slice.as_slice()[..need].to_vec();

                    self.ensure_gpu().await?;
                    let texture = self.render(rgba, &detections)?;

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::memory::SystemSlice;
    use g2g_core::{BBox, FrameTiming, PushOutcome, Rate};

    /// Whether a wgpu adapter is available; tests skip gracefully without a GPU.
    async fn gpu_available() -> bool {
        wgpu::Instance::default()
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .is_ok()
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
        };
        assert!(ov.intercept_caps(&nv12).is_err());
        assert!(ov.intercept_caps(&rgba_caps(8, 8)).is_ok());
    }

    #[tokio::test]
    async fn renders_box_onto_gpu_texture() {
        if !gpu_available().await {
            std::eprintln!("no wgpu adapter; skipping Vello GPU render test");
            return;
        }
        let (w, h) = (64u32, 64u32);
        let mut ov = VelloAnalyticsOverlay::new().with_thickness(4.0);
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
}
