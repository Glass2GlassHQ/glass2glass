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
//!   hands it in; the sink presents to it and, on the app's resize event,
//!   reconfigures it via [`WgpuSink::resize`].
//!
//! `wgpu-sink` feature (implies `std`).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::element::QosMessage;
use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::{
    AsyncElement, BusHandle, Caps, CapsConstraint, CapsSet, ClockSync, ConfigureOutcome, Dim,
    G2gError, Interlace, MemoryDomain, OutputSink, PipelinePacket, PresentationPacer, PropError,
    PropValue, PropertySpec, Rate, RawVideoFormat, PACING_PROPERTIES,
};

use crate::clock::wait_to_present;
use crate::gpu::{gpu_err, texture_layout, texture_of, GpuContext, WgpuTextureLayout};

/// Fullscreen-triangle vertex stage, shared by both fragment stages below. The
/// UV flips Y so a top-left-origin source (Vello / video) lands top-left on the
/// target.
const VERTEX_STAGE: &str = r#"
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
"#;

/// Blit an already-colour-converted texture: one filtered sample per pixel.
const FRAGMENT_RGBA: &str = r#"
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_smp: sampler;

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(src_tex, src_smp, in.uv);
}
"#;

/// Blit a packed-NV12 R8Uint plane (`width x height*3/2`: Y rows, then
/// interleaved CbCr), converting BT.601 limited-range YCbCr -> RGB with the same
/// coefficients the GL sink's shader uses. A uint texture is not filterable, so
/// this fetches texels instead of sampling; the picture height comes from the
/// texture (the packed plane is exactly 3/2 of it).
const FRAGMENT_NV12: &str = r#"
@group(0) @binding(0) var nv12_tex: texture_2d<u32>;

fn plane_value(x: u32, y: u32) -> f32 {
    return f32(textureLoad(nv12_tex, vec2<u32>(x, y), 0).r) / 255.0;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let packed = textureDimensions(nv12_tex);
    let luma_height = packed.y * 2u / 3u;
    let x = min(u32(in.uv.x * f32(packed.x)), packed.x - 1u);
    let y = min(u32(in.uv.y * f32(luma_height)), luma_height - 1u);

    let luma = 1.1643 * (plane_value(x, y) - 0.0625);
    let chroma_x = (x / 2u) * 2u;
    let chroma_y = luma_height + y / 2u;
    let cb = plane_value(chroma_x, chroma_y) - 0.5;
    let cr = plane_value(chroma_x + 1u, chroma_y) - 0.5;

    return vec4<f32>(
        luma + 1.5958 * cr,
        luma - 0.3917 * cb - 0.8129 * cr,
        luma + 2.0170 * cb,
        1.0,
    );
}
"#;

/// The offscreen target texture: a render attachment the blit writes, readable
/// (`COPY_SRC`) and re-samplable.
fn offscreen_texture(ctx: &GpuContext, width: u32, height: u32) -> wgpu::Texture {
    ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("wgpu-sink-offscreen"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: WgpuSink::OFFSCREEN_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

/// A blit pipeline plus the bind group layout its fragment stage expects.
#[derive(Debug)]
struct BlitPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl BlitPipeline {
    /// Build the pipeline for one source layout: `fragment_stage` is appended to
    /// the shared vertex stage, and the bind group layout matches the bindings
    /// that stage declares (a filtered colour texture + sampler for RGBA, a
    /// non-filterable uint texture alone for packed NV12).
    fn build(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        layout: WgpuTextureLayout,
        fragment_stage: &str,
    ) -> Self {
        let source = alloc::string::String::from(VERTEX_STAGE) + fragment_stage;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wgpu-sink-blit"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let texture_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: match layout {
                    WgpuTextureLayout::Rgba => wgpu::TextureSampleType::Float { filterable: true },
                    WgpuTextureLayout::PackedNv12 => wgpu::TextureSampleType::Uint,
                },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let sampler_entry = wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let entries: &[wgpu::BindGroupLayoutEntry] = match layout {
            WgpuTextureLayout::Rgba => &[texture_entry, sampler_entry],
            WgpuTextureLayout::PackedNv12 => &[texture_entry],
        };
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("wgpu-sink-bgl"),
            entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("wgpu-sink-layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("wgpu-sink-pipeline"),
            layout: Some(&pipeline_layout),
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
        Self {
            pipeline,
            bind_group_layout,
        }
    }
}

/// Pixel format + geometry of the frames the sink was configured for.
#[derive(Debug, Clone, Copy)]
struct SourceLayout {
    layout: WgpuTextureLayout,
    width: u32,
    height: u32,
}

impl SourceLayout {
    /// Texture format + size a system-memory frame of this layout uploads into,
    /// and its packed row stride in bytes.
    fn upload_geometry(&self) -> (wgpu::TextureFormat, u32, u32, u32) {
        match self.layout {
            WgpuTextureLayout::Rgba => (
                wgpu::TextureFormat::Rgba8Unorm,
                self.width,
                self.height,
                self.width * 4,
            ),
            WgpuTextureLayout::PackedNv12 => (
                wgpu::TextureFormat::R8Uint,
                self.width,
                packed_nv12_height(self.height),
                self.width,
            ),
        }
    }
}

/// Rows an NV12 frame of `height` occupies once packed into a single plane: the
/// luma rows plus the half-height interleaved chroma rows.
fn packed_nv12_height(height: u32) -> u32 {
    height + height / 2
}

/// The sink's accepted layouts: NV12 (what the decoders produce) and RGBA (what
/// the GPU overlay / decode elements produce), at any geometry. Shared with the
/// windowed present sink, which drives this renderer.
pub(crate) fn accepted_caps() -> CapsSet {
    CapsSet::from_alternatives(Vec::from([
        any_geometry(RawVideoFormat::Nv12),
        any_geometry(RawVideoFormat::Rgba8),
    ]))
}

fn any_geometry(format: RawVideoFormat) -> Caps {
    Caps::RawVideo {
        format,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        interlace: Interlace::Any,
    }
}

/// The source layout the negotiated caps settle on, rejecting anything the blit
/// pipelines cannot read.
fn source_layout(absolute_caps: &Caps) -> Result<SourceLayout, G2gError> {
    let Caps::RawVideo {
        format,
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        ..
    } = absolute_caps
    else {
        return Err(G2gError::CapsMismatch);
    };
    let layout = match format {
        RawVideoFormat::Nv12 => WgpuTextureLayout::PackedNv12,
        RawVideoFormat::Rgba8 => WgpuTextureLayout::Rgba,
        _ => return Err(G2gError::CapsMismatch),
    };
    // NV12 chroma is subsampled: odd geometry has no well-defined plane.
    if layout == WgpuTextureLayout::PackedNv12 && (width % 2 != 0 || height % 2 != 0) {
        return Err(G2gError::CapsMismatch);
    }
    if *width == 0 || *height == 0 {
        return Err(G2gError::CapsMismatch);
    }
    Ok(SourceLayout {
        layout,
        width: *width,
        height: *height,
    })
}

/// The picture size these caps settle on, rejecting anything the blit pipelines
/// cannot read. The windowed present sink checks this before opening a window.
#[cfg(all(target_os = "linux", feature = "wgpu-present"))]
pub(crate) fn source_geometry(absolute_caps: &Caps) -> Result<(u32, u32), G2gError> {
    let source = source_layout(absolute_caps)?;
    Ok((source.width, source.height))
}

/// Where a [`WgpuSink`] presents.
enum Target {
    /// An internal texture the sink owns; readable via [`WgpuSink::read_target`].
    Offscreen {
        texture: wgpu::Texture,
        width: u32,
        height: u32,
    },
    /// A caller-built, configured surface (an on-screen window).
    Surface {
        surface: wgpu::Surface<'static>,
        config: wgpu::SurfaceConfiguration,
    },
}

/// Presents `MemoryDomain::WgpuTexture` frames to a target by GPU blit.
///
/// # Example
///
/// ```no_run
/// use g2g_core::G2gError;
/// use g2g_plugins::gpu::GpuContext;
/// use g2g_plugins::wgpusink::WgpuSink;
///
/// async fn build() -> Result<WgpuSink, G2gError> {
///     let ctx = GpuContext::headless().await?;
///     Ok(WgpuSink::offscreen(ctx, 1920, 1080).with_max_lateness_ns(20_000_000))
/// }
/// ```
pub struct WgpuSink {
    ctx: GpuContext,
    /// One blit pipeline per source layout; which one runs is decided per frame
    /// from the texture, so a sink fed RGBA and NV12 in turn needs no rebuild.
    rgba_blit: BlitPipeline,
    nv12_blit: BlitPipeline,
    sampler: wgpu::Sampler,
    target: Target,
    /// Pixel format + geometry negotiated for the incoming frames, which a
    /// system-memory frame needs to be uploaded (a GPU frame carries its own).
    source: Option<SourceLayout>,
    /// Texture the system-memory upload path writes into, allocated on the first
    /// such frame and reused.
    upload: Option<wgpu::Texture>,
    configured: bool,
    presented: u64,
    /// PTS pacing + QoS late-drop: idle until the runner hands over a clock, and
    /// the default lateness bound never drops.
    pacer: PresentationPacer,
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
        let texture = offscreen_texture(&ctx, width, height);
        Self::build(
            ctx,
            Self::OFFSCREEN_FORMAT,
            Target::Offscreen {
                texture,
                width,
                height,
            },
        )
    }

    /// A sink that presents to a caller-built, already-`configure`d surface (an
    /// on-screen window). The application owns the window + event loop and the
    /// surface's lifetime, and forwards window resizes through
    /// [`resize`](Self::resize).
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
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("wgpu-sink-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            rgba_blit: BlitPipeline::build(
                device,
                target_format,
                WgpuTextureLayout::Rgba,
                FRAGMENT_RGBA,
            ),
            nv12_blit: BlitPipeline::build(
                device,
                target_format,
                WgpuTextureLayout::PackedNv12,
                FRAGMENT_NV12,
            ),
            ctx,
            sampler,
            target,
            source: None,
            upload: None,
            configured: false,
            presented: 0,
            pacer: PresentationPacer::new(),
        }
    }

    /// Count of frames presented.
    pub fn presented_count(&self) -> u64 {
        self.presented
    }

    /// Current target size: the surface's swapchain size, or the offscreen
    /// texture's.
    pub fn target_size(&self) -> (u32, u32) {
        match &self.target {
            Target::Offscreen { width, height, .. } => (*width, *height),
            Target::Surface { config, .. } => (config.width, config.height),
        }
    }

    /// Follow the window to a new size: reconfigure the surface's swapchain (or
    /// reallocate the offscreen texture) to `width` x `height`. The application
    /// owns the window, so it calls this from its resize event.
    ///
    /// The incoming frame keeps its negotiated geometry: the blit scales it to
    /// fill the target, at the new size as it did at the old one.
    ///
    /// A size that already matches, or a zero dimension (a minimised or
    /// not-yet-mapped window), is ignored, so calling this on every resize event
    /// is fine.
    pub fn resize(&mut self, width: u32, height: u32) {
        if (width, height) == self.target_size() || width == 0 || height == 0 {
            return;
        }
        match &mut self.target {
            Target::Offscreen {
                texture,
                width: target_width,
                height: target_height,
            } => {
                *texture = offscreen_texture(&self.ctx, width, height);
                *target_width = width;
                *target_height = height;
            }
            Target::Surface { surface, config } => {
                config.width = width;
                config.height = height;
                surface.configure(&self.ctx.device, config);
            }
        }
    }

    /// QoS late-drop bound: once PTS pacing is engaged, a frame past its
    /// deadline by more than `ns` is dropped instead of presented late, so the
    /// sink catches up. The default (`u64::MAX`) never drops.
    pub fn with_max_lateness_ns(mut self, ns: u64) -> Self {
        self.pacer.set_max_lateness_ns(ns);
        self
    }

    /// Post a running-stats `Qos` report every `ns` of clock time while frames
    /// present, on top of the per-drop reports. `0` (the default) reports only
    /// drops.
    pub fn with_qos_interval_ns(mut self, ns: u64) -> Self {
        self.pacer.set_report_interval_ns(ns);
        self
    }

    /// Attach the pipeline bus so QoS reports reach the application.
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.pacer.set_bus(bus);
        self
    }

    /// Frames dropped by QoS late-drop (past their deadline beyond the bound).
    pub fn late_dropped(&self) -> u64 {
        self.pacer.late_dropped()
    }

    /// Present one frame's pixels: a `WgpuTexture`-domain frame is blitted from
    /// the producer's own texture, a system-memory frame is uploaded to the
    /// sink's own texture first. The path [`process`](AsyncElement::process) and
    /// the windowed present sink share, past pacing.
    pub fn present_frame(&mut self, domain: &MemoryDomain) -> Result<(), G2gError> {
        match domain {
            MemoryDomain::WgpuTexture(owned) => {
                // A frame from a different GPU producer (foreign keep-alive type,
                // or a format no blit here samples) is not presentable.
                let texture = texture_of(owned).ok_or(G2gError::UnsupportedDomain)?;
                let layout = texture_layout(texture).ok_or(G2gError::UnsupportedDomain)?;
                // The texture belongs to the frame; the blit only reads it, but
                // `present` needs `&mut self`, so clone the handle (cheap: wgpu
                // handles are reference-counted).
                let texture = texture.clone();
                self.present(&texture, layout)
            }
            _ => {
                let slice = domain
                    .as_system_slice()
                    .ok_or(G2gError::UnsupportedDomain)?;
                self.upload_system(slice)
            }
        }
    }

    /// Upload a packed system-memory frame into the sink's own texture in the
    /// negotiated layout, then blit it.
    fn upload_system(&mut self, bytes: &[u8]) -> Result<(), G2gError> {
        let source = self.source.ok_or(G2gError::NotConfigured)?;
        let (format, width, rows, bytes_per_row) = source.upload_geometry();
        if bytes.len() < (bytes_per_row as usize) * (rows as usize) {
            return Err(G2gError::CapsMismatch);
        }
        let size = wgpu::Extent3d {
            width,
            height: rows,
            depth_or_array_layers: 1,
        };
        let texture = match &self.upload {
            Some(texture) => texture.clone(),
            None => {
                let texture = self.ctx.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("wgpu-sink-upload"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                self.upload = Some(texture.clone());
                texture
            }
        };
        self.ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(rows),
            },
            size,
        );
        self.present(&texture, source.layout)
    }

    /// Blit `src` onto the target with the pipeline for its `layout`. For a
    /// surface target, acquires and presents the swapchain image; for offscreen,
    /// renders into the owned texture.
    fn present(&mut self, src: &wgpu::Texture, layout: WgpuTextureLayout) -> Result<(), G2gError> {
        let blit = match layout {
            WgpuTextureLayout::Rgba => &self.rgba_blit,
            WgpuTextureLayout::PackedNv12 => &self.nv12_blit,
        };
        let src_view = src.create_view(&wgpu::TextureViewDescriptor::default());
        let texture_binding = wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&src_view),
        };
        let sampler_binding = wgpu::BindGroupEntry {
            binding: 1,
            resource: wgpu::BindingResource::Sampler(&self.sampler),
        };
        let entries: &[wgpu::BindGroupEntry] = match layout {
            WgpuTextureLayout::Rgba => &[texture_binding, sampler_binding],
            WgpuTextureLayout::PackedNv12 => &[texture_binding],
        };
        let bind_group = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wgpu-sink-bg"),
                layout: &blit.bind_group_layout,
                entries,
            });

        // Acquire the destination view. For a surface, hold the SurfaceTexture
        // until after submit so it can be presented.
        let surface_frame = match &self.target {
            Target::Offscreen { .. } => None,
            Target::Surface { surface, config } => match surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(t)
                | wgpu::CurrentSurfaceTexture::Suboptimal(t) => Some(t),
                // Stale / lost surface (e.g. a window resize): reconfigure and
                // skip this frame, so the next acquire targets the refreshed
                // surface instead of freezing on a permanently-Outdated one.
                wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                    surface.configure(&self.ctx.device, config);
                    return Ok(());
                }
                // Other transient states (Timeout / Occluded / Validation): skip
                // and re-acquire next frame.
                _ => return Ok(()),
            },
        };
        let dst_view = match (&self.target, &surface_frame) {
            (Target::Offscreen { texture, .. }, _) => {
                texture.create_view(&wgpu::TextureViewDescriptor::default())
            }
            (Target::Surface { .. }, Some(frame)) => frame
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default()),
            (Target::Surface { .. }, None) => unreachable!("returned above on no frame"),
        };

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu-sink"),
            });
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
            pass.set_pipeline(&blit.pipeline);
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
        let Target::Offscreen {
            texture,
            width,
            height,
        } = &self.target
        else {
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
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.ctx.queue.submit([encoder.finish()]);

        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.ctx
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
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

    /// RGBA or NV12, geometry open (the producer fixates it), whichever memory
    /// domain the frames arrive in.
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(accepted_caps())
    }

    /// A GPU texture is blitted where it lies; system memory is uploaded. Any
    /// other domain (a CUDA frame, say) needs a converter spliced ahead, and
    /// declaring this is what makes the M354 auto-plug do it.
    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::WgpuTexture).with(MemoryDomainKind::System)
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let source = source_layout(absolute_caps)?;
        // A geometry or format change invalidates the upload texture allocated
        // for the old one.
        if self.source.map(|s| (s.layout, s.width, s.height))
            != Some((source.layout, source.width, source.height))
        {
            self.upload = None;
        }
        self.source = Some(source);
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    /// Adopt the elected clock + base time so textures are blitted at their PTS
    /// deadline rather than as fast as the producer pushes them.
    fn set_clock_sync(&mut self, sync: ClockSync) {
        self.pacer.set_clock_sync(sync);
    }

    /// Relay a late drop upstream (M174): the runner forwards it onto the
    /// incoming link, where the producer can shed load.
    fn take_qos(&mut self) -> Option<QosMessage> {
        self.pacer.take_qos()
    }

    fn properties(&self) -> &'static [PropertySpec] {
        PACING_PROPERTIES
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        self.pacer
            .set_property(name, &value)
            .unwrap_or(Err(PropError::Unknown))
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        self.pacer.get_property(name)
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
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    // PTS pacing: hold the texture until its deadline on the
                    // elected clock, or drop it when it is already too late (the
                    // QoS bound) or outside the segment. Unpaced without a clock:
                    // blit as fast as the producer pushes.
                    let paced = self.pacer.judge(frame.timing.pts_ns, self.presented);
                    if !wait_to_present(paced).await {
                        return Ok(());
                    }
                    self.present_frame(&frame.domain)?;
                }
                // Track the playback segment so PTS maps to running time (correct
                // across a seek), and re-anchor after a seek flush.
                PipelinePacket::Segment(seg) => self.pacer.set_segment(seg),
                PipelinePacket::Flush => self.pacer.flush(),
                // Terminal sink: other control packets are consumed, nothing is
                // forwarded.
                _ => {}
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::shared_ctx;
    use g2g_core::frame::Frame;
    use g2g_core::memory::OwnedWgpuTexture;
    use g2g_core::{FrameTiming, PushOutcome};

    /// A source texture filled with `pixels` (RGBA8, top-left origin), usable as a
    /// blit source (sampled) on `ctx`'s device.
    fn source_texture(ctx: &GpuContext, w: u32, h: u32, pixels: &[u8]) -> wgpu::Texture {
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test-source"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
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
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        texture
    }

    fn wgpu_frame(ctx: &GpuContext, w: u32, h: u32, texture: wgpu::Texture, pts_ns: u64) -> Frame {
        use crate::gpu::WgpuTextureKeepAlive;
        let _ = ctx;
        Frame::new(
            MemoryDomain::WgpuTexture(OwnedWgpuTexture::new(
                w,
                h,
                alloc::sync::Arc::new(WgpuTextureKeepAlive(texture)),
            )),
            FrameTiming {
                pts_ns,
                ..FrameTiming::default()
            },
            0,
        )
    }

    struct NullSink;
    impl OutputSink for NullSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            packet_slot.take();
            core::task::Poll::Ready(Ok(PushOutcome::Accepted))
        }
    }

    #[tokio::test]
    async fn offscreen_blit_reproduces_source_orientation() {
        let Some(ctx) = shared_ctx().await else {
            std::eprintln!("no wgpu adapter; skipping WgpuSink blit test");
            return;
        };
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
            interlace: g2g_core::Interlace::Any,
        })
        .unwrap();
        let frame = wgpu_frame(&ctx, w, h, src, 0);
        sink.process(PipelinePacket::DataFrame(frame), &mut NullSink)
            .await
            .unwrap();

        let out = sink.read_target().unwrap();
        let px = |x: u32, y: u32| {
            let i = ((y * w + x) * 4) as usize;
            [out[i], out[i + 1], out[i + 2], out[i + 3]]
        };
        // Orientation preserved: top stays red, bottom stays blue.
        assert!(
            px(0, 0)[0] > 200 && px(0, 0)[2] < 50,
            "top row red: {:?}",
            px(0, 0)
        );
        assert!(
            px(0, 3)[2] > 200 && px(0, 3)[0] < 50,
            "bottom row blue: {:?}",
            px(0, 3)
        );
        assert_eq!(sink.presented_count(), 1);
    }

    /// Resize follows the window on the headlessly-testable target: the target
    /// grows, the next blit fills it at the new size, and a zero or unchanged
    /// size leaves it alone.
    #[tokio::test]
    async fn resize_retargets_the_sink_and_the_blit_fills_the_new_size() {
        let Some(ctx) = shared_ctx().await else {
            std::eprintln!("no wgpu adapter; skipping WgpuSink resize test");
            return;
        };
        let (w, h) = (4u32, 4u32);
        let caps = Caps::RawVideo {
            format: g2g_core::RawVideoFormat::Rgba8,
            width: g2g_core::Dim::Fixed(w),
            height: g2g_core::Dim::Fixed(h),
            framerate: g2g_core::Rate::Any,
            interlace: g2g_core::Interlace::Any,
        };
        let mut sink = WgpuSink::offscreen(ctx.clone(), w, h);
        sink.configure_pipeline(&caps).unwrap();
        assert_eq!(sink.target_size(), (w, h));

        // Ignored: already this size, and a minimised window.
        sink.resize(w, h);
        sink.resize(0, 200);
        sink.resize(200, 0);
        assert_eq!(sink.target_size(), (w, h));

        // The frame geometry stays 4x4; only the target follows the "window".
        let (new_w, new_h) = (16u32, 10u32);
        sink.resize(new_w, new_h);
        assert_eq!(sink.target_size(), (new_w, new_h));

        let pixels = alloc::vec![0u8, 255, 0, 255].repeat((w * h) as usize);
        let frame = wgpu_frame(&ctx, w, h, source_texture(&ctx, w, h, &pixels), 0);
        sink.process(PipelinePacket::DataFrame(frame), &mut NullSink)
            .await
            .unwrap();

        let out = sink.read_target().unwrap();
        assert_eq!(
            out.len(),
            (new_w * new_h * 4) as usize,
            "readback is the resized target"
        );
        // Scaled to fill: every pixel of the larger target is the source green.
        for (i, px) in out.chunks_exact(4).enumerate() {
            assert!(
                px[1] > 200 && px[0] < 50,
                "pixel {i} of the resized target is the blitted source: {px:?}"
            );
        }
        assert_eq!(sink.presented_count(), 1);
    }

    /// Caps for a `w` x `h` frame in `format`, the shape negotiation hands the
    /// sink.
    fn caps(format: g2g_core::RawVideoFormat, w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format,
            width: g2g_core::Dim::Fixed(w),
            height: g2g_core::Dim::Fixed(h),
            framerate: g2g_core::Rate::Any,
            interlace: g2g_core::Interlace::Any,
        }
    }

    /// A `w` x `h` NV12 frame, BT.601 limited-range: top half red, bottom half
    /// blue. Packed as the sink expects (Y rows, then interleaved CbCr rows).
    fn nv12_red_over_blue(w: u32, h: u32) -> Vec<u8> {
        const RED: (u8, u8, u8) = (81, 90, 240);
        const BLUE: (u8, u8, u8) = (41, 240, 110);
        let mut bytes = Vec::new();
        for y in 0..h {
            let luma = if y < h / 2 { RED.0 } else { BLUE.0 };
            bytes.extend(core::iter::repeat_n(luma, w as usize));
        }
        for y in 0..h / 2 {
            let (_, cb, cr) = if y < h / 4 { RED } else { BLUE };
            for _ in 0..w / 2 {
                bytes.extend_from_slice(&[cb, cr]);
            }
        }
        bytes
    }

    /// A system-memory frame carrying `bytes`.
    fn system_frame(bytes: Vec<u8>) -> Frame {
        Frame::new(
            MemoryDomain::System(g2g_core::memory::SystemSlice::from_boxed(
                bytes.into_boxed_slice(),
            )),
            FrameTiming::default(),
            0,
        )
    }

    /// Assert `out` (packed RGBA8, `w` wide) is red on top and blue at the
    /// bottom, the pattern both NV12 fixtures encode.
    fn assert_red_over_blue(out: &[u8], w: u32, h: u32) {
        let px = |x: u32, y: u32| {
            let i = ((y * w + x) * 4) as usize;
            [out[i], out[i + 1], out[i + 2]]
        };
        let top = px(0, 0);
        assert!(
            top[0] > 230 && top[1] < 25 && top[2] < 25,
            "top rows convert to red: {top:?}"
        );
        let bottom = px(0, h - 1);
        assert!(
            bottom[2] > 230 && bottom[0] < 25 && bottom[1] < 25,
            "bottom rows convert to blue: {bottom:?}"
        );
    }

    /// System-memory RGBA in: uploaded to the sink's own texture and blitted,
    /// orientation preserved.
    #[tokio::test]
    async fn system_rgba_frame_is_uploaded_and_presented() {
        let Some(ctx) = shared_ctx().await else {
            std::eprintln!("no wgpu adapter; skipping WgpuSink system RGBA test");
            return;
        };
        let (w, h) = (4u32, 4u32);
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
        let mut sink = WgpuSink::offscreen(ctx.clone(), w, h);
        sink.configure_pipeline(&caps(g2g_core::RawVideoFormat::Rgba8, w, h))
            .unwrap();
        sink.process(
            PipelinePacket::DataFrame(system_frame(pixels)),
            &mut NullSink,
        )
        .await
        .unwrap();

        assert_red_over_blue(&sink.read_target().unwrap(), w, h);
        assert_eq!(sink.presented_count(), 1);
    }

    /// System-memory NV12 in: uploaded as the packed plane and converted to RGB
    /// by the sink's shader (a CPU `videoconvert` would be the alternative).
    #[tokio::test]
    async fn system_nv12_frame_is_converted_on_the_gpu() {
        let Some(ctx) = shared_ctx().await else {
            std::eprintln!("no wgpu adapter; skipping WgpuSink system NV12 test");
            return;
        };
        let (w, h) = (8u32, 8u32);
        let mut sink = WgpuSink::offscreen(ctx.clone(), w, h);
        sink.configure_pipeline(&caps(g2g_core::RawVideoFormat::Nv12, w, h))
            .unwrap();
        sink.process(
            PipelinePacket::DataFrame(system_frame(nv12_red_over_blue(w, h))),
            &mut NullSink,
        )
        .await
        .unwrap();

        assert_red_over_blue(&sink.read_target().unwrap(), w, h);
        assert_eq!(sink.presented_count(), 1);
    }

    /// The zero-copy input the CUDA / GPU-decode bridges emit: an R8Uint packed
    /// NV12 texture already on the device. It is sampled where it lies, and comes
    /// out pixel-identical to the same bytes uploaded from system memory.
    #[tokio::test]
    async fn packed_nv12_texture_presents_without_an_upload() {
        use crate::gpu::WgpuNv12Texture;

        let Some(ctx) = shared_ctx().await else {
            std::eprintln!("no wgpu adapter; skipping WgpuSink NV12 texture test");
            return;
        };
        let (w, h) = (8u32, 8u32);
        let bytes = nv12_red_over_blue(w, h);
        let nv12_caps = caps(g2g_core::RawVideoFormat::Nv12, w, h);

        // The GPU-resident frame: one R8Uint plane of w x (h * 3/2).
        let packed_rows = h + h / 2;
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test-nv12"),
            size: wgpu::Extent3d {
                width: w,
                height: packed_rows,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Uint,
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
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w),
                rows_per_image: Some(packed_rows),
            },
            wgpu::Extent3d {
                width: w,
                height: packed_rows,
                depth_or_array_layers: 1,
            },
        );
        let frame = Frame::new(
            MemoryDomain::WgpuTexture(OwnedWgpuTexture::new(
                w,
                h,
                alloc::sync::Arc::new(WgpuNv12Texture::new(
                    ctx.device.clone(),
                    ctx.queue.clone(),
                    texture,
                )),
            )),
            FrameTiming::default(),
            0,
        );

        let mut sink = WgpuSink::offscreen(ctx.clone(), w, h);
        sink.configure_pipeline(&nv12_caps).unwrap();
        sink.process(PipelinePacket::DataFrame(frame), &mut NullSink)
            .await
            .unwrap();
        let from_gpu = sink.read_target().unwrap();
        assert_red_over_blue(&from_gpu, w, h);

        let mut cpu_sink = WgpuSink::offscreen(ctx.clone(), w, h);
        cpu_sink.configure_pipeline(&nv12_caps).unwrap();
        cpu_sink
            .process(
                PipelinePacket::DataFrame(system_frame(bytes)),
                &mut NullSink,
            )
            .await
            .unwrap();
        assert_eq!(
            from_gpu,
            cpu_sink.read_target().unwrap(),
            "the zero-copy texture path renders the same pixels as the upload path"
        );
    }

    /// A clock whose `now_ns` the test drives by hand.
    #[derive(Debug)]
    struct ManualClock(alloc::sync::Arc<core::sync::atomic::AtomicU64>);
    impl g2g_core::PipelineClock for ManualClock {
        fn now_ns(&self) -> u64 {
            self.0.load(core::sync::atomic::Ordering::Relaxed)
        }
    }

    /// PTS pacing on a real GPU: an on-time frame is blitted, one held until its
    /// deadline is blitted after waiting that long, and a frame past the lateness
    /// bound is dropped, reported on the bus, and offered upstream via `take_qos`.
    #[tokio::test]
    async fn pts_pacing_presents_on_time_and_drops_late_frames() {
        use core::sync::atomic::{AtomicU64, Ordering};
        use g2g_core::clock::PlayAnchor;
        use std::time::Instant;

        let Some(ctx) = shared_ctx().await else {
            std::eprintln!("no wgpu adapter; skipping WgpuSink pacing test");
            return;
        };
        let (w, h) = (4u32, 4u32);
        let pixels = alloc::vec![255u8; (w * h * 4) as usize];
        let rgba = Caps::RawVideo {
            format: g2g_core::RawVideoFormat::Rgba8,
            width: g2g_core::Dim::Fixed(w),
            height: g2g_core::Dim::Fixed(h),
            framerate: g2g_core::Rate::Any,
            interlace: g2g_core::Interlace::Any,
        };

        // The play anchor stamped at clock 0 makes each frame's deadline its PTS.
        let (bus, handle) = g2g_core::Bus::new(4);
        let clock = alloc::sync::Arc::new(AtomicU64::new(0));
        let anchor = PlayAnchor::new();
        anchor.stamp(0);
        let mut sink = WgpuSink::offscreen(ctx.clone(), w, h)
            .with_max_lateness_ns(0)
            .with_bus(handle);
        sink.configure_pipeline(&rgba).unwrap();
        AsyncElement::set_clock_sync(
            &mut sink,
            g2g_core::ClockSync::with_play_anchor(
                alloc::sync::Arc::new(ManualClock(clock.clone())),
                0,
                anchor,
            ),
        );

        // Due now: presented, nothing reported.
        let f = wgpu_frame(&ctx, w, h, source_texture(&ctx, w, h, &pixels), 0);
        sink.process(PipelinePacket::DataFrame(f), &mut NullSink)
            .await
            .unwrap();
        assert_eq!(sink.presented_count(), 1, "on-time frame presented");
        assert!(sink.late_dropped() == 0 && AsyncElement::take_qos(&mut sink).is_none());
        assert_eq!(bus.try_recv(), None, "no report for an on-time frame");

        // Due in 5 ms: held that long, then presented.
        let f = wgpu_frame(&ctx, w, h, source_texture(&ctx, w, h, &pixels), 5_000_000);
        let started = Instant::now();
        sink.process(PipelinePacket::DataFrame(f), &mut NullSink)
            .await
            .unwrap();
        assert!(
            started.elapsed() >= std::time::Duration::from_millis(5),
            "held until its deadline, waited {:?}",
            started.elapsed()
        );
        assert_eq!(sink.presented_count(), 2);

        // Clock jumps to 100 ms: a frame due at 10 ms is 90 ms late.
        clock.store(100_000_000, Ordering::Relaxed);
        let f = wgpu_frame(&ctx, w, h, source_texture(&ctx, w, h, &pixels), 10_000_000);
        sink.process(PipelinePacket::DataFrame(f), &mut NullSink)
            .await
            .expect("a dropped frame is not an error");
        assert_eq!(sink.presented_count(), 2, "the late frame was not blitted");
        assert_eq!(sink.late_dropped(), 1);
        let upstream = AsyncElement::take_qos(&mut sink).expect("upstream QoS report");
        assert_eq!(upstream.jitter_ns, 90_000_000);
        assert_eq!(upstream.running_time_ns, 10_000_000);
        match bus.try_recv() {
            Some(g2g_core::BusMessage::Qos {
                processed, dropped, ..
            }) => assert_eq!((processed, dropped), (2, 1)),
            other => panic!("expected a Qos message, got {other:?}"),
        }
    }

    #[cfg(feature = "vello-overlay")]
    #[tokio::test]
    async fn overlay_to_sink_presents_boxes_on_shared_device() {
        use crate::vellooverlay::VelloAnalyticsOverlay;
        use g2g_core::memory::SystemSlice;
        use g2g_core::{AnalyticsMeta, BBox, Dim, ObjectDetection, Rate, RawVideoFormat};

        let Some(ctx) = shared_ctx().await else {
            std::eprintln!("no wgpu adapter; skipping overlay->sink test");
            return;
        };
        let (w, h) = (64u32, 64u32);

        // Overlay and sink share ONE device: the overlay's texture is presentable
        // by the sink with no copy.
        let mut overlay = VelloAnalyticsOverlay::new()
            .with_context(ctx.clone())
            .with_thickness(4.0);
        let rgba_caps = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
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
            bbox: BBox {
                x: 0.25,
                y: 0.25,
                w: 0.5,
                h: 0.5,
            },
            label: 0,
            confidence: 0.9,
        });
        frame.meta.attach(a);

        // overlay -> (WgpuTexture) -> sink, all on the shared device.
        let mut relay = CaptureSink { frame: None };
        overlay
            .process(PipelinePacket::DataFrame(frame), &mut relay)
            .await
            .unwrap();
        let gpu_frame = relay.frame.expect("overlay produced a GPU frame");
        assert!(
            matches!(gpu_frame.domain, MemoryDomain::WgpuTexture(_)),
            "kept on GPU"
        );
        sink.process(PipelinePacket::DataFrame(gpu_frame), &mut NullSink)
            .await
            .unwrap();

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
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");
            core::task::Poll::Ready({
                if let PipelinePacket::DataFrame(f) = packet {
                    self.frame = Some(f);
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }
}
