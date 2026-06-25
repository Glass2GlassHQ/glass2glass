//! GPU presentation sink (M103): presents [`MemoryDomain::WgpuTexture`] frames
//! (eg from [`VelloAnalyticsOverlay`](crate::vellooverlay)) by blitting them onto
//! a target on the **same** wgpu device, with no GPU->CPU readback. The consuming
//! end of the keep-on-GPU overlay path.
//!
//! A `wgpu::Texture` is bound to the device that created it, so this sink shares
//! the producer's device through a [`GpuContext`]: build one context, clone it
//! into both the overlay and the sink. The incoming texture is then sampled
//! directly in a small fullscreen-triangle blit pass that writes the target,
//! handling any format / size difference between the source (`Rgba8Unorm` from
//! Vello) and the destination (eg a surface's `Bgra8UnormSrgb`).
//!
//! Two targets:
//! - [`WgpuSink::offscreen`]: an internal texture the sink owns and exposes via
//!   [`read_target`](WgpuSink::read_target). A render-to-texture / screenshot
//!   sink, and the headlessly-testable path.
//! - [`WgpuSink::with_surface`]: a caller-built, already-configured
//!   `wgpu::Surface`. The on-screen path. Window + event-loop ownership belongs
//!   to the application (wgpu surfaces are created from a window handle and must
//!   integrate with the app's event loop), so the app creates the surface and
//!   hands it in; the sink only presents to it.
//!
//! `wgpu-sink` feature (implies `std`).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, G2gError, MemoryDomain, OutputSink,
    PipelinePacket,
};

use crate::gpu::{gpu_err, texture_of, GpuContext};

/// Fullscreen-triangle blit: sample the source texture and write the target. The
/// UV flips Y so a top-left-origin source (Vello / video) lands top-left on the
/// target.
const BLIT_SHADER: &str = r#"
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VsOut {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let xy = corners[vid];
    var out: VsOut;
    out.pos = vec4<f32>(xy, 0.0, 1.0);
    out.uv = vec2<f32>((xy.x + 1.0) * 0.5, 1.0 - (xy.y + 1.0) * 0.5);
    return out;
}

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_smp: sampler;

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(src_tex, src_smp, in.uv);
}
"#;

/// Where a [`WgpuSink`] presents.
enum Target {
    /// An internal texture the sink owns; readable via [`WgpuSink::read_target`].
    Offscreen { texture: wgpu::Texture, width: u32, height: u32 },
    /// A caller-built, configured surface (an on-screen window).
    Surface { surface: wgpu::Surface<'static>, config: wgpu::SurfaceConfiguration },
}

/// Presents `MemoryDomain::WgpuTexture` frames to a target by GPU blit.
pub struct WgpuSink {
    ctx: GpuContext,
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    target: Target,
    configured: bool,
    presented: u64,
}

impl core::fmt::Debug for WgpuSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let kind = match &self.target {
            Target::Offscreen { width, height, .. } => ("offscreen", *width, *height),
            Target::Surface { config, .. } => ("surface", config.width, config.height),
        };
        f.debug_struct("WgpuSink")
            .field("target", &kind.0)
            .field("width", &kind.1)
            .field("height", &kind.2)
            .field("configured", &self.configured)
            .field("presented", &self.presented)
            .finish()
    }
}

impl WgpuSink {
    /// Format the offscreen target is allocated in (and read back as).
    pub const OFFSCREEN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

    /// A sink that presents into an internal `width` x `height` texture (read it
    /// back with [`read_target`](Self::read_target)). The render-to-texture path.
    pub fn offscreen(ctx: GpuContext, width: u32, height: u32) -> Self {
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("wgpu-sink-offscreen"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::OFFSCREEN_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        Self::build(ctx, Self::OFFSCREEN_FORMAT, Target::Offscreen { texture, width, height })
    }

    /// A sink that presents to a caller-built, already-`configure`d surface (an
    /// on-screen window). The application owns the window + event loop and the
    /// surface's lifetime.
    pub fn with_surface(
        ctx: GpuContext,
        surface: wgpu::Surface<'static>,
        config: wgpu::SurfaceConfiguration,
    ) -> Self {
        let format = config.format;
        Self::build(ctx, format, Target::Surface { surface, config })
    }

    fn build(ctx: GpuContext, target_format: wgpu::TextureFormat, target: Target) -> Self {
        let device = &ctx.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wgpu-sink-blit"),
            source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("wgpu-sink-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("wgpu-sink-layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("wgpu-sink-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("wgpu-sink-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            ctx,
            pipeline,
            bind_group_layout,
            sampler,
            target,
            configured: false,
            presented: 0,
        }
    }

    /// Count of frames presented.
    pub fn presented_count(&self) -> u64 {
        self.presented
    }

    /// Blit `src` onto the target. For a surface target, acquires and presents
    /// the swapchain image; for offscreen, renders into the owned texture.
    fn present(&mut self, src: &wgpu::Texture) -> Result<(), G2gError> {
        let src_view = src.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu-sink-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        // Acquire the destination view. For a surface, hold the SurfaceTexture
        // until after submit so it can be presented.
        let surface_frame = match &self.target {
            Target::Offscreen { .. } => None,
            Target::Surface { surface, .. } => match surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(t)
                | wgpu::CurrentSurfaceTexture::Suboptimal(t) => Some(t),
                // Transient acquisition states: skip this frame rather than fail
                // the pipeline (the next frame re-acquires).
                _ => return Ok(()),
            },
        };
        let dst_view = match (&self.target, &surface_frame) {
            (Target::Offscreen { texture, .. }, _) => {
                texture.create_view(&wgpu::TextureViewDescriptor::default())
            }
            (Target::Surface { .. }, Some(frame)) => {
                frame.texture.create_view(&wgpu::TextureViewDescriptor::default())
            }
            (Target::Surface { .. }, None) => unreachable!("returned above on no frame"),
        };

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("wgpu-sink") });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("wgpu-sink-blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dst_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.ctx.queue.submit([encoder.finish()]);
        if let Some(frame) = surface_frame {
            frame.present();
        }
        self.presented += 1;
        Ok(())
    }

    /// Read the offscreen target back to a tightly-packed RGBA8 buffer (panics if
    /// this sink targets a surface). For screenshots / tests.
    pub fn read_target(&self) -> Result<Vec<u8>, G2gError> {
        let Target::Offscreen { texture, width, height } = &self.target else {
            return Err(G2gError::UnsupportedDomain);
        };
        let (w, h) = (*width, *height);
        let unpadded = (w * 4) as usize;
        let padded = unpadded.next_multiple_of(256);
        let buffer = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-sink-readback"),
            size: (padded * h as usize) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_buffer(
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
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.ctx.queue.submit([encoder.finish()]);

        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.ctx
            .device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .map_err(gpu_err)?;
        rx.recv().map_err(gpu_err)?.map_err(gpu_err)?;

        let mapped = slice.get_mapped_range();
        let mut out = Vec::with_capacity(unpadded * h as usize);
        for row in 0..h as usize {
            let start = row * padded;
            out.extend_from_slice(&mapped[start..start + unpadded]);
        }
        drop(mapped);
        buffer.unmap();
        Ok(out)
    }
}

impl AsyncElement for WgpuSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        // The pixel caps are whatever the upstream GPU element produced (RGBA);
        // what this sink really requires is the WgpuTexture memory domain, which
        // caps do not describe, so it is checked at process() time.
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            if let PipelinePacket::DataFrame(frame) = packet {
                let MemoryDomain::WgpuTexture(owned) = &frame.domain else {
                    return Err(G2gError::UnsupportedDomain);
                };
                // A frame from a different GPU producer (foreign keep-alive type)
                // is not presentable by this sink.
                let texture = texture_of(owned).ok_or(G2gError::UnsupportedDomain)?;
                self.present(texture)?;
            }
            // Terminal sink: control packets are consumed, nothing is forwarded.
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::frame::Frame;
    use g2g_core::memory::OwnedWgpuTexture;
    use g2g_core::{FrameTiming, PushOutcome};

    async fn gpu_available() -> bool {
        wgpu::Instance::default()
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .is_ok()
    }

    /// A source texture filled with `pixels` (RGBA8, top-left origin), usable as a
    /// blit source (sampled) on `ctx`'s device.
    fn source_texture(ctx: &GpuContext, w: u32, h: u32, pixels: &[u8]) -> wgpu::Texture {
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test-source"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        texture
    }

    fn wgpu_frame(ctx: &GpuContext, w: u32, h: u32, texture: wgpu::Texture) -> Frame {
        use crate::gpu::WgpuTextureKeepAlive;
        let _ = ctx;
        Frame::new(
            MemoryDomain::WgpuTexture(OwnedWgpuTexture::new(
                w,
                h,
                alloc::sync::Arc::new(WgpuTextureKeepAlive(texture)),
            )),
            FrameTiming::default(),
            0,
        )
    }

    struct NullSink;
    impl OutputSink for NullSink {
        fn push<'a>(
            &'a mut self,
            _packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async { Ok(PushOutcome::Accepted) })
        }
    }

    #[tokio::test]
    async fn offscreen_blit_reproduces_source_orientation() {
        if !gpu_available().await {
            std::eprintln!("no wgpu adapter; skipping WgpuSink blit test");
            return;
        }
        let ctx = GpuContext::headless().await.unwrap();
        let (w, h) = (4u32, 4u32);
        // Top two rows red, bottom two rows blue.
        let mut pixels = Vec::new();
        for y in 0..h {
            for _ in 0..w {
                if y < h / 2 {
                    pixels.extend_from_slice(&[255, 0, 0, 255]);
                } else {
                    pixels.extend_from_slice(&[0, 0, 255, 255]);
                }
            }
        }
        let src = source_texture(&ctx, w, h, &pixels);

        let mut sink = WgpuSink::offscreen(ctx.clone(), w, h);
        sink.configure_pipeline(&g2g_core::Caps::RawVideo {
            format: g2g_core::RawVideoFormat::Rgba8,
            width: g2g_core::Dim::Fixed(w),
            height: g2g_core::Dim::Fixed(h),
            framerate: g2g_core::Rate::Any,
        })
        .unwrap();
        let frame = wgpu_frame(&ctx, w, h, src);
        sink.process(PipelinePacket::DataFrame(frame), &mut NullSink).await.unwrap();

        let out = sink.read_target().unwrap();
        let px = |x: u32, y: u32| {
            let i = ((y * w + x) * 4) as usize;
            [out[i], out[i + 1], out[i + 2], out[i + 3]]
        };
        // Orientation preserved: top stays red, bottom stays blue.
        assert!(px(0, 0)[0] > 200 && px(0, 0)[2] < 50, "top row red: {:?}", px(0, 0));
        assert!(px(0, 3)[2] > 200 && px(0, 3)[0] < 50, "bottom row blue: {:?}", px(0, 3));
        assert_eq!(sink.presented_count(), 1);
    }

    #[cfg(feature = "vello-overlay")]
    #[tokio::test]
    async fn overlay_to_sink_presents_boxes_on_shared_device() {
        use crate::vellooverlay::VelloAnalyticsOverlay;
        use g2g_core::memory::SystemSlice;
        use g2g_core::{AnalyticsMeta, BBox, Dim, ObjectDetection, RawVideoFormat, Rate};

        if !gpu_available().await {
            std::eprintln!("no wgpu adapter; skipping overlay->sink test");
            return;
        }
        let ctx = GpuContext::headless().await.unwrap();
        let (w, h) = (64u32, 64u32);

        // Overlay and sink share ONE device: the overlay's texture is presentable
        // by the sink with no copy.
        let mut overlay = VelloAnalyticsOverlay::new().with_context(ctx.clone()).with_thickness(4.0);
        let rgba_caps = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Any,
        };
        overlay.configure_pipeline(&rgba_caps).unwrap();
        let mut sink = WgpuSink::offscreen(ctx.clone(), w, h);
        sink.configure_pipeline(&rgba_caps).unwrap();

        // Dark input frame + a class-0 (red) detection over the centre.
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
        a.add_detection(ObjectDetection {
            bbox: BBox { x: 0.25, y: 0.25, w: 0.5, h: 0.5 },
            label: 0,
            confidence: 0.9,
        });
        frame.meta.attach(a);

        // overlay -> (WgpuTexture) -> sink, all on the shared device.
        let mut relay = CaptureSink { frame: None };
        overlay.process(PipelinePacket::DataFrame(frame), &mut relay).await.unwrap();
        let gpu_frame = relay.frame.expect("overlay produced a GPU frame");
        assert!(matches!(gpu_frame.domain, MemoryDomain::WgpuTexture(_)), "kept on GPU");
        sink.process(PipelinePacket::DataFrame(gpu_frame), &mut NullSink).await.unwrap();

        let out = sink.read_target().unwrap();
        let px = |x: u32, y: u32| {
            let i = ((y * w + x) * 4) as usize;
            [out[i], out[i + 1], out[i + 2], out[i + 3]]
        };
        let edge = px(32, 16);
        assert!(
            edge[0] > 120 && edge[0] > edge[1] + 40 && edge[0] > edge[2] + 40,
            "presented box edge is red: {edge:?}"
        );
        let interior = px(32, 32);
        assert!(
            interior[0] < 70 && interior[1] < 70 && interior[2] < 70,
            "presented interior is the dark input: {interior:?}"
        );
    }

    /// Captures a single forwarded frame (to relay the overlay output to the sink).
    #[cfg(feature = "vello-overlay")]
    struct CaptureSink {
        frame: Option<Frame>,
    }
    #[cfg(feature = "vello-overlay")]
    impl OutputSink for CaptureSink {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async move {
                if let PipelinePacket::DataFrame(f) = packet {
                    self.frame = Some(f);
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }
}
