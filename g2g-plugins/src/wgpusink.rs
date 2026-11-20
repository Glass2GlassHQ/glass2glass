// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! GPU presentation sink: presents [`MemoryDomain::WgpuTexture`] frames
//! (eg from `VelloAnalyticsOverlay`) by blitting them onto
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

use g2g_core::element::{PresentationStats, QosMessage};
use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::{
    AsyncElement, BusHandle, Caps, CapsConstraint, CapsSet, ClockSync, Colorimetry,
    ConfigureOutcome, G2gError, MemoryDomain, OutputSink, PipelinePacket, PixelAspect,
    PresentationPacer, PropError, PropValue, PropertySpec, RawVideoFormat, MAX_LATENESS_PROPERTY,
    PROCESSING_DEADLINE_PROPERTY, QOS_INTERVAL_PROPERTY, SYNC_PROPERTY,
};

use crate::clock::wait_to_present;
#[cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "wgpu-present"
))]
use crate::display_aspect::picture_point_in;
use crate::display_aspect::{
    letterbox, ViewportRectangle, FORCE_ASPECT_RATIO_NAME, FORCE_ASPECT_RATIO_PROPERTY,
};
use crate::filled_rectangles::{rectangle_vertices, FilledRectangle};
use crate::gpu::{
    nv12_plane_views, packed_nv12_extent, read_texture_plane, texture_holds_picture,
    texture_layout, texture_of, GpuContext, WgpuTextureLayout,
};
use crate::overlay_image::OverlayImage;
use crate::videoconvert::display::{display_sink_caps, DisplayInput};
use crate::yuvmatrix::{dither_step, YuvToRgbWeights, DITHER_NOISE_SCALE, DITHER_NOISE_WEIGHTS};

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
    // no local array or struct var: naga gives those a layout spirv-val rejects
    let xy = vec2<f32>(
        select(-1.0, 3.0, vid == 1u),
        select(-1.0, 3.0, vid == 2u),
    );
    return VsOut(
        vec4<f32>(xy, 0.0, 1.0),
        vec2<f32>((xy.x + 1.0) * 0.5, 1.0 - (xy.y + 1.0) * 0.5),
    );
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

/// Fetch from packed NV12 or P010 ([`packed_nv12_extent`]: Y rows, then
/// interleaved CbCr). A uint texture is not filterable, so this
/// fetches texels instead of sampling. The picture size comes from the
/// negotiated caps: at an odd width the texture is one column wider than the
/// picture.
macro_rules! packed_420_shader {
    ($code_scale:literal) => {
        concat!(
            r#"
@group(0) @binding(0) var nv12_tex: texture_2d<u32>;
@group(0) @binding(1) var<uniform> picture_size: vec2<u32>;

fn plane_value(x: u32, y: u32) -> f32 {
    return f32(textureLoad(nv12_tex, vec2<u32>(x, y), 0).r) / "#,
            $code_scale,
            r#";
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let x = min(u32(in.uv.x * f32(picture_size.x)), picture_size.x - 1u);
    let y = min(u32(in.uv.y * f32(picture_size.y)), picture_size.y - 1u);
    let chroma_x = (x / 2u) * 2u;
    let chroma_y = picture_size.y + y / 2u;
    return ycbcr_to_rgb(
        in.pos.xy,
        plane_value(x, y),
        plane_value(chroma_x, chroma_y),
        plane_value(chroma_x + 1u, chroma_y),
    );
}
"#
        )
    };
}

pub(crate) const FRAGMENT_PACKED_NV12: &str = packed_420_shader!("255.0");
// P010 stores 10-bit codes in the high bits of each word.
const FRAGMENT_PACKED_P010: &str = packed_420_shader!("65280.0");

/// Fetch a pixel from a two-plane `TextureFormat::NV12` / `TextureFormat::P010`
/// texture through its luma and chroma views (`gpu::nv12_plane_views`): the
/// normalized plane samples make one shader fit either depth. Full-size luma,
/// half-size chroma with Cb in `r` and Cr in `g`. Fetching the nearest chroma
/// texel rather than filtering between pairs is what keeps one sink's picture the
/// same whichever
/// NV12 layout the producer hands it: a filtered sample lands a quarter of a
/// chroma texel off the left-sited 4:2:0 grid and shifts colour across edges.
/// The picture size comes from the negotiated caps: a decoder may pad the
/// planes past the picture.
pub(crate) const FRAGMENT_MULTIPLANAR_NV12: &str = r#"
@group(0) @binding(0) var luma_plane: texture_2d<f32>;
@group(0) @binding(1) var chroma_plane: texture_2d<f32>;
@group(0) @binding(2) var<uniform> picture_size: vec2<u32>;

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let x = min(u32(in.uv.x * f32(picture_size.x)), picture_size.x - 1u);
    let y = min(u32(in.uv.y * f32(picture_size.y)), picture_size.y - 1u);
    let chroma = textureLoad(chroma_plane, vec2<u32>(x / 2u, y / 2u), 0);
    return ycbcr_to_rgb(
        in.pos.xy,
        textureLoad(luma_plane, vec2<u32>(x, y), 0).r,
        chroma.r,
        chroma.g,
    );
}
"#;

fn fragment_stacked_planar(format: RawVideoFormat) -> alloc::string::String {
    // planar 10 and 12 bit frames upload as one R16Uint texture: the luma plane, then the U and V planes stacked under it
    let (horizontal_shift, vertical_shift) = format
        .chroma_shift()
        .expect("only planar YUV formats upload stacked");
    // the YCbCr weights take 8-bit codes over 255, and these formats keep each code in the low bits
    let code_scale = (255u32 << (format.bit_depth() - 8)) as f32;
    alloc::format!(
        r#"
@group(0) @binding(0) var planes_tex: texture_2d<u32>;
@group(0) @binding(1) var<uniform> picture_size: vec2<u32>;

fn plane_value(x: u32, y: u32) -> f32 {{
    return f32(textureLoad(planes_tex, vec2<u32>(x, y), 0).r) / {code_scale:?};
}}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {{
    let x = min(u32(in.uv.x * f32(picture_size.x)), picture_size.x - 1u);
    let y = min(u32(in.uv.y * f32(picture_size.y)), picture_size.y - 1u);
    let chroma_rows = (picture_size.y + {vertical_round}u) >> {vertical_shift}u;
    let chroma_x = x >> {horizontal_shift}u;
    let u_row = picture_size.y + (y >> {vertical_shift}u);
    return ycbcr_to_rgb(
        in.pos.xy,
        plane_value(x, y),
        plane_value(chroma_x, u_row),
        plane_value(chroma_x, u_row + chroma_rows),
    );
}}
"#,
        vertical_round = (1u32 << vertical_shift) - 1,
    )
}

/// One NV12 fragment stage: the YCbCr -> RGB step for `colorimetry` (the same
/// weights the GL sink's shader gets) followed by `plane_fetch`, the per-layout
/// stage above that reads the planes. All three sample values are normalized,
/// with chroma still centred on the neutral 128 / 255.
pub(crate) fn fragment_nv12(
    colorimetry: Colorimetry,
    plane_fetch: &str,
    dither_step: f32,
) -> alloc::string::String {
    let w = YuvToRgbWeights::new(colorimetry);
    alloc::format!(
        r#"
fn ycbcr_to_rgb(pixel: vec2<f32>, luma_value: f32, cb_value: f32, cr_value: f32) -> vec4<f32> {{
    let luma = {luma_gain:?} * (luma_value - {luma_floor:?});
    let cb = cb_value - {chroma_neutral:?};
    let cr = cr_value - {chroma_neutral:?};
    let rgb = vec3<f32>(
        luma + ({red_from_cr:?}) * cr,
        luma + ({green_from_cb:?}) * cb + ({green_from_cr:?}) * cr,
        luma + ({blue_from_cb:?}) * cb,
    );
    let noise = fract({noise_scale:?} * fract(dot(pixel, vec2<f32>({noise_x:?}, {noise_y:?}))));
    return vec4<f32>(rgb + (noise - 0.5) * {dither_step:?}, 1.0);
}}
{plane_fetch}
"#,
        luma_gain = w.luma_gain,
        luma_floor = w.luma_floor,
        chroma_neutral = w.chroma_neutral,
        red_from_cr = w.red_from_cr,
        green_from_cb = w.green_from_cb,
        green_from_cr = w.green_from_cr,
        blue_from_cb = w.blue_from_cb,
        noise_scale = DITHER_NOISE_SCALE,
        noise_x = DITHER_NOISE_WEIGHTS[0],
        noise_y = DITHER_NOISE_WEIGHTS[1],
    )
}

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

/// What a layout's fragment stage binds, in binding order: one texture per plane
/// it reads, then a sampler when it filters instead of fetching texels, then the
/// picture size uniform when it indexes texels by the negotiated geometry. The
/// pipeline's bind group layout and the per-frame bind group are both built from
/// this, so the two cannot fall out of step.
#[derive(Debug)]
struct BlitBindings {
    planes: u32,
    sample_type: wgpu::TextureSampleType,
    sampler: bool,
    picture_size: bool,
}

impl BlitBindings {
    fn sampler_binding(&self) -> u32 {
        self.planes
    }

    fn picture_size_binding(&self) -> u32 {
        self.planes + u32::from(self.sampler)
    }
}

const PICTURE_SIZE_BYTES: u64 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlitLayout {
    Gpu(WgpuTextureLayout),
    PackedYuv16,
}

impl From<WgpuTextureLayout> for BlitLayout {
    fn from(layout: WgpuTextureLayout) -> Self {
        BlitLayout::Gpu(layout)
    }
}

fn blit_bindings(layout: BlitLayout) -> BlitBindings {
    match layout {
        BlitLayout::Gpu(WgpuTextureLayout::Rgba) => BlitBindings {
            planes: 1,
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            sampler: true,
            picture_size: false,
        },
        BlitLayout::Gpu(WgpuTextureLayout::PackedNv12) | BlitLayout::PackedYuv16 => BlitBindings {
            planes: 1,
            sample_type: wgpu::TextureSampleType::Uint,
            sampler: false,
            picture_size: true,
        },
        // The plane views are float textures, but both NV12 stages fetch texels,
        // so neither binds a sampler.
        BlitLayout::Gpu(WgpuTextureLayout::MultiplanarNv12) => BlitBindings {
            planes: 2,
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            sampler: false,
            picture_size: true,
        },
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlitCompositing {
    // the rest of the view outside the viewport is cleared to black
    ReplaceTarget,
    // premultiplied alpha drawn over what the view holds
    OverTarget,
}

impl BlitCompositing {
    fn blend(self) -> Option<wgpu::BlendState> {
        match self {
            BlitCompositing::ReplaceTarget => None,
            BlitCompositing::OverTarget => Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
        }
    }

    fn load(self) -> wgpu::LoadOp<wgpu::Color> {
        match self {
            BlitCompositing::ReplaceTarget => wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            BlitCompositing::OverTarget => wgpu::LoadOp::Load,
        }
    }
}

/// A blit pipeline plus the bind group layout its fragment stage expects.
#[derive(Debug)]
pub(crate) struct BlitPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    compositing: BlitCompositing,
}

impl BlitPipeline {
    /// Build the pipeline for one source layout: `fragment_stage` is appended to
    /// the shared vertex stage, and the bind group layout matches the bindings
    /// that stage declares (see [`blit_bindings`]).
    pub(crate) fn build(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        layout: BlitLayout,
        fragment_stage: &str,
        compositing: BlitCompositing,
    ) -> Self {
        let source = alloc::string::String::from(VERTEX_STAGE) + fragment_stage;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wgpu-sink-blit"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let bindings = blit_bindings(layout);
        let mut entries = Vec::new();
        for binding in 0..bindings.planes {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: bindings.sample_type,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            });
        }
        if bindings.sampler {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: bindings.sampler_binding(),
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });
        }
        if bindings.picture_size {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: bindings.picture_size_binding(),
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(PICTURE_SIZE_BYTES),
                },
                count: None,
            });
        }
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("wgpu-sink-bgl"),
            entries: &entries,
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
                    blend: compositing.blend(),
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
            compositing,
        }
    }

    /// Record a pass that draws `src`, stored in `layout`, over `target`'s
    /// viewport.
    pub(crate) fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Texture,
        layout: BlitLayout,
        resources: &BlitResources,
        target: BlitTarget<'_>,
    ) {
        let BlitTarget { view, viewport } = target;
        let bindings = blit_bindings(layout);
        // A two-plane NV12 texture cannot be bound whole, only through its two
        // plane views.
        let views: Vec<wgpu::TextureView> = match layout {
            BlitLayout::Gpu(WgpuTextureLayout::MultiplanarNv12) => Vec::from(nv12_plane_views(src)),
            BlitLayout::Gpu(WgpuTextureLayout::Rgba | WgpuTextureLayout::PackedNv12)
            | BlitLayout::PackedYuv16 => {
                Vec::from([src.create_view(&wgpu::TextureViewDescriptor::default())])
            }
        };
        let mut entries: Vec<wgpu::BindGroupEntry> = views
            .iter()
            .enumerate()
            .map(|(plane, view)| wgpu::BindGroupEntry {
                binding: plane as u32,
                resource: wgpu::BindingResource::TextureView(view),
            })
            .collect();
        if bindings.sampler {
            entries.push(wgpu::BindGroupEntry {
                binding: bindings.sampler_binding(),
                resource: wgpu::BindingResource::Sampler(&resources.sampler),
            });
        }
        if bindings.picture_size {
            entries.push(wgpu::BindGroupEntry {
                binding: bindings.picture_size_binding(),
                resource: resources.picture_size.as_entire_binding(),
            });
        }
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu-sink-bg"),
            layout: &self.bind_group_layout,
            entries: &entries,
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("wgpu-sink-blit"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: self.compositing.load(),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_viewport(
            viewport.x as f32,
            viewport.y as f32,
            viewport.width as f32,
            viewport.height as f32,
            0.0,
            1.0,
        );
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct BlitTarget<'a> {
    pub view: &'a wgpu::TextureView,
    pub viewport: ViewportRectangle,
}

/// The sampler and picture size uniform a blit binds besides the source planes.
#[derive(Debug)]
pub(crate) struct BlitResources {
    sampler: wgpu::Sampler,
    picture_size: wgpu::Buffer,
}

impl BlitResources {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        Self {
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("wgpu-sink-sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            picture_size: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("wgpu-sink-picture-size"),
                size: PICTURE_SIZE_BYTES,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        }
    }

    pub(crate) fn write_picture_size(&self, queue: &wgpu::Queue, width: u32, height: u32) {
        queue.write_buffer(
            &self.picture_size,
            0,
            &[width.to_le_bytes(), height.to_le_bytes()].concat(),
        );
    }
}

/// Pixel format + geometry of the frames the sink was configured for.
#[derive(Debug, Clone, Copy)]
struct SourceLayout {
    input: DisplayInput,
    layout: BlitLayout,
    width: u32,
    height: u32,
}

impl SourceLayout {
    fn stacked_planar_rows(&self) -> Result<u32, G2gError> {
        let format = self.input.presented;
        let chroma_rows = format
            .plane_rows(1, self.height)
            .ok_or(G2gError::CapsMismatch)?;
        Ok(self.height + 2 * chroma_rows)
    }

    /// Texture format + size a system-memory frame of this layout uploads into.
    fn upload_texture(&self) -> Result<(wgpu::TextureFormat, wgpu::Extent3d), G2gError> {
        let (format, (width, height)) = match self.layout {
            BlitLayout::Gpu(WgpuTextureLayout::Rgba) => {
                (wgpu::TextureFormat::Rgba8Unorm, (self.width, self.height))
            }
            BlitLayout::Gpu(WgpuTextureLayout::PackedNv12) => (
                wgpu::TextureFormat::R8Uint,
                packed_nv12_extent(self.width, self.height),
            ),
            // NV12 caps negotiate the packed layout, so a two-plane frame only
            // ever arrives as a GPU texture, never as bytes to upload.
            BlitLayout::Gpu(WgpuTextureLayout::MultiplanarNv12) => {
                return Err(G2gError::UnsupportedDomain)
            }
            BlitLayout::PackedYuv16 => (
                wgpu::TextureFormat::R16Uint,
                if self.input.presented == RawVideoFormat::P010 {
                    packed_nv12_extent(self.width, self.height)
                } else {
                    (self.width, self.stacked_planar_rows()?)
                },
            ),
        };
        Ok((
            format,
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        ))
    }
}

/// The sink's accepted layouts: NV12 and RGBA as they are, and every other
/// format `videoconvert` reads, at any geometry. Shared with the windowed
/// present sink, which drives this renderer.
const WGPU_NATIVE_FORMATS: &[RawVideoFormat] = &[
    RawVideoFormat::Nv12,
    RawVideoFormat::Rgba8,
    RawVideoFormat::P010,
    RawVideoFormat::I420p10,
    RawVideoFormat::I422p10,
    RawVideoFormat::I444p10,
    RawVideoFormat::I420p12,
    RawVideoFormat::I422p12,
    RawVideoFormat::I444p12,
];

pub fn accepted_caps() -> CapsSet {
    display_sink_caps(WGPU_NATIVE_FORMATS)
}

/// The source layout the negotiated caps settle on, rejecting anything the blit
/// pipelines cannot read.
fn source_layout(absolute_caps: &Caps) -> Result<SourceLayout, G2gError> {
    let input = DisplayInput::from_caps(absolute_caps, WGPU_NATIVE_FORMATS)?;
    let layout = match input.presented {
        RawVideoFormat::Nv12 => WgpuTextureLayout::PackedNv12.into(),
        RawVideoFormat::Rgba8 => WgpuTextureLayout::Rgba.into(),
        _ => BlitLayout::PackedYuv16,
    };
    if input.width == 0 || input.height == 0 {
        return Err(G2gError::CapsMismatch);
    }
    Ok(SourceLayout {
        input,
        layout,
        width: input.width,
        height: input.height,
    })
}

/// The picture size these caps settle on, rejecting anything the blit pipelines
/// cannot read. The windowed present sink checks this before opening a window.
#[cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "wgpu-present"
))]
pub(crate) fn source_geometry(absolute_caps: &Caps) -> Result<(u32, u32), G2gError> {
    let source = source_layout(absolute_caps)?;
    Ok((source.width, source.height))
}

// frames arrive gamma-encoded, an sRGB surface would encode them a second time
fn presentable_format(
    default: wgpu::TextureFormat,
    supported: &[wgpu::TextureFormat],
) -> wgpu::TextureFormat {
    let without_srgb = default.remove_srgb_suffix();
    if supported.contains(&without_srgb) {
        without_srgb
    } else {
        default
    }
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
    renderer: VideoRenderer,
    target: Target,
    presented: u64,
    /// PTS pacing + QoS late-drop: idle until the runner hands over a clock, and
    /// the default lateness bound is 5 ms past the frame end.
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
            .field("configured", &self.renderer.source.is_some())
            .field("presented", &self.presented)
            .finish()
    }
}

pub struct VideoRenderer {
    ctx: GpuContext,
    /// One blit pipeline per source layout; which one runs is decided per frame
    /// from the texture, so a sink fed RGBA and NV12 in turn needs no rebuild.
    rgba_blit: BlitPipeline,
    nv12_blit: BlitPipeline,
    /// The two-plane NV12 blit, for a decoder-emitted `TextureFormat::NV12`
    /// frame. Same caps as `nv12_blit`, different bindings, so which one runs is
    /// decided per frame from the texture.
    nv12_planes_blit: BlitPipeline,
    packed_yuv16_blit: Option<PackedYuv16Blit>,
    /// Colorimetry both NV12 pipelines' shaders were built with, so a mid-stream
    /// refinement rebuilds them.
    nv12_colorimetry: Colorimetry,
    /// Format of whatever the blits render into, needed to rebuild a pipeline.
    target_format: wgpu::TextureFormat,
    dither_step: f32,
    resources: BlitResources,
    /// Pixel format + geometry negotiated for the incoming frames, which a
    /// system-memory frame needs to be uploaded (a GPU frame carries its own).
    source: Option<SourceLayout>,
    /// Texture the system-memory upload path writes into, allocated on the first
    /// such frame and reused.
    upload: Option<wgpu::Texture>,
    pixel_aspect: PixelAspect,
    force_aspect_ratio: bool,
    overlay_pipeline: wgpu::RenderPipeline,
    image_blit: BlitPipeline,
    image_upload: Option<UploadedImage>,
}

#[derive(Debug)]
struct PackedYuv16Blit {
    format: RawVideoFormat,
    colorimetry: Colorimetry,
    pipeline: BlitPipeline,
}

#[derive(Debug)]
struct UploadedImage {
    texture: wgpu::Texture,
    generation: u64,
}

const OVERLAY_SHADER: &str = "
struct Shaded {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs(@location(0) position: vec2<f32>, @location(1) color: vec4<f32>) -> Shaded {
    var out: Shaded;
    out.position = vec4<f32>(position, 0.0, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs(shaded: Shaded) -> @location(0) vec4<f32> {
    return shaded.color;
}
";

fn overlay_pipeline(
    device: &wgpu::Device,
    target_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("wgpu-sink-overlay"),
        source: wgpu::ShaderSource::Wgsl(OVERLAY_SHADER.into()),
    });
    let stride = (crate::filled_rectangles::FLOATS_PER_VERTEX * size_of::<f32>()) as u64;
    let attributes = wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4];
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("wgpu-sink-overlay"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: stride,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &attributes,
            })],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

impl core::fmt::Debug for VideoRenderer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("VideoRenderer")
            .field("target_format", &self.target_format)
            .field("configured", &self.source.is_some())
            .finish_non_exhaustive()
    }
}

impl VideoRenderer {
    // dither_bits is the target's depth to dither the YCbCr conversion to, mpv's dither-depth
    pub fn new(
        ctx: GpuContext,
        target_format: wgpu::TextureFormat,
        dither_bits: Option<u32>,
    ) -> Self {
        let dither_step = dither_step(dither_bits);
        let device = &ctx.device;
        let resources = BlitResources::new(device);
        let overlay_pipeline = overlay_pipeline(device, target_format);
        let image_blit = BlitPipeline::build(
            device,
            target_format,
            WgpuTextureLayout::Rgba.into(),
            FRAGMENT_RGBA,
            BlitCompositing::OverTarget,
        );
        Self {
            rgba_blit: BlitPipeline::build(
                device,
                target_format,
                WgpuTextureLayout::Rgba.into(),
                FRAGMENT_RGBA,
                BlitCompositing::ReplaceTarget,
            ),
            nv12_blit: BlitPipeline::build(
                device,
                target_format,
                WgpuTextureLayout::PackedNv12.into(),
                &fragment_nv12(Colorimetry::UNKNOWN, FRAGMENT_PACKED_NV12, dither_step),
                BlitCompositing::ReplaceTarget,
            ),
            nv12_planes_blit: BlitPipeline::build(
                device,
                target_format,
                WgpuTextureLayout::MultiplanarNv12.into(),
                &fragment_nv12(Colorimetry::UNKNOWN, FRAGMENT_MULTIPLANAR_NV12, dither_step),
                BlitCompositing::ReplaceTarget,
            ),
            packed_yuv16_blit: None,
            nv12_colorimetry: Colorimetry::UNKNOWN,
            target_format,
            dither_step,
            ctx,
            resources,
            source: None,
            upload: None,
            pixel_aspect: PixelAspect::Any,
            force_aspect_ratio: true,
            overlay_pipeline,
            image_blit,
            image_upload: None,
        }
    }

    pub fn target_format(&self) -> wgpu::TextureFormat {
        self.target_format
    }

    /// Rebuild both NV12 blits for a new colorimetry. The weights are compiled
    /// into the shader, so a mid-stream refinement means new pipelines.
    fn set_nv12_colorimetry(&mut self, colorimetry: Colorimetry) {
        if colorimetry == self.nv12_colorimetry {
            return;
        }
        self.nv12_blit = BlitPipeline::build(
            &self.ctx.device,
            self.target_format,
            WgpuTextureLayout::PackedNv12.into(),
            &fragment_nv12(colorimetry, FRAGMENT_PACKED_NV12, self.dither_step),
            BlitCompositing::ReplaceTarget,
        );
        self.nv12_planes_blit = BlitPipeline::build(
            &self.ctx.device,
            self.target_format,
            WgpuTextureLayout::MultiplanarNv12.into(),
            &fragment_nv12(colorimetry, FRAGMENT_MULTIPLANAR_NV12, self.dither_step),
            BlitCompositing::ReplaceTarget,
        );
        self.nv12_colorimetry = colorimetry;
    }

    fn build_packed_yuv16_blit(&mut self, format: RawVideoFormat, colorimetry: Colorimetry) {
        let built = self
            .packed_yuv16_blit
            .as_ref()
            .is_some_and(|blit| (blit.format, blit.colorimetry) == (format, colorimetry));
        if built {
            return;
        }
        let planar;
        let plane_fetch = if format == RawVideoFormat::P010 {
            FRAGMENT_PACKED_P010
        } else {
            planar = fragment_stacked_planar(format);
            &planar
        };
        let pipeline = BlitPipeline::build(
            &self.ctx.device,
            self.target_format,
            BlitLayout::PackedYuv16,
            &fragment_nv12(colorimetry, plane_fetch, self.dither_step),
            BlitCompositing::ReplaceTarget,
        );
        self.packed_yuv16_blit = Some(PackedYuv16Blit {
            format,
            colorimetry,
            pipeline,
        });
    }

    /// Stretch the picture to fill the target (`false`) instead of letterboxing
    /// it at its display aspect (`true`, the default).
    pub fn set_force_aspect_ratio(&mut self, force_aspect_ratio: bool) {
        self.force_aspect_ratio = force_aspect_ratio;
    }

    pub fn force_aspect_ratio(&self) -> bool {
        self.force_aspect_ratio
    }

    pub fn configure(&mut self, absolute_caps: &Caps) -> Result<(), G2gError> {
        let mut source = source_layout(absolute_caps)?;
        // wgpu downlevel devices allow 2D textures only 2048 texels tall
        if source.layout == BlitLayout::PackedYuv16
            && source.upload_texture()?.1.height > self.ctx.device.limits().max_texture_dimension_2d
        {
            source.input = source.input.presented_as(RawVideoFormat::Nv12);
            source.layout = WgpuTextureLayout::PackedNv12.into();
        }
        // A geometry or format change invalidates the upload texture allocated
        // for the old one.
        if self.source.map(|s| (s.input.presented, s.width, s.height))
            != Some((source.input.presented, source.width, source.height))
        {
            self.upload = None;
        }
        if let Caps::RawVideo {
            colorimetry,
            pixel_aspect,
            ..
        } = absolute_caps
        {
            self.set_nv12_colorimetry(*colorimetry);
            if source.layout == BlitLayout::PackedYuv16 {
                self.build_packed_yuv16_blit(source.input.presented, *colorimetry);
            }
            self.pixel_aspect = *pixel_aspect;
        }
        self.resources
            .write_picture_size(&self.ctx.queue, source.width, source.height);
        self.source = Some(source);
        Ok(())
    }

    #[cfg(all(
        any(target_os = "linux", target_os = "windows"),
        feature = "wgpu-present"
    ))]
    pub(crate) fn picture_point(
        &self,
        target_size: (u32, u32),
        target_x: f64,
        target_y: f64,
    ) -> Option<(u32, u32)> {
        let source = self.source?;
        picture_point_in(
            self.viewport(target_size),
            source.width,
            source.height,
            target_x,
            target_y,
        )
    }

    fn viewport(&self, (target_width, target_height): (u32, u32)) -> ViewportRectangle {
        match (self.force_aspect_ratio, self.source) {
            (true, Some(source)) => letterbox(
                source.width,
                source.height,
                self.pixel_aspect,
                target_width,
                target_height,
            ),
            _ => ViewportRectangle::whole(target_width, target_height),
        }
    }

    pub fn draw(
        &mut self,
        domain: &MemoryDomain,
        view: &wgpu::TextureView,
        target_size: (u32, u32),
    ) -> Result<ViewportRectangle, G2gError> {
        let (texture, layout) = match domain {
            MemoryDomain::WgpuTexture(owned) => {
                // A frame from a different GPU producer (foreign keep-alive type,
                // or a format no blit here samples) is not presentable.
                let texture = texture_of(owned).ok_or(G2gError::UnsupportedDomain)?;
                let layout = texture_layout(texture).ok_or(G2gError::UnsupportedDomain)?;
                if layout != WgpuTextureLayout::Rgba {
                    let source = self.source.ok_or(G2gError::NotConfigured)?;
                    if !texture_holds_picture(texture, layout, source.width, source.height) {
                        return Err(G2gError::CapsMismatch);
                    }
                }
                (texture.clone(), layout.into())
            }
            _ => {
                let slice = domain
                    .as_system_slice()
                    .ok_or(G2gError::UnsupportedDomain)?;
                self.upload_system(slice)?
            }
        };
        let blit = match layout {
            BlitLayout::Gpu(WgpuTextureLayout::Rgba) => &self.rgba_blit,
            BlitLayout::Gpu(WgpuTextureLayout::PackedNv12) => &self.nv12_blit,
            BlitLayout::Gpu(WgpuTextureLayout::MultiplanarNv12) => &self.nv12_planes_blit,
            BlitLayout::PackedYuv16 => {
                &self
                    .packed_yuv16_blit
                    .as_ref()
                    .ok_or(G2gError::NotConfigured)?
                    .pipeline
            }
        };
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu-sink"),
            });
        let viewport = self.viewport(target_size);
        blit.encode(
            &self.ctx.device,
            &mut encoder,
            &texture,
            layout,
            &self.resources,
            BlitTarget { view, viewport },
        );
        self.ctx.queue.submit([encoder.finish()]);
        Ok(viewport)
    }

    pub fn clear(&self, view: &wgpu::TextureView) {
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu-sink-clear"),
            });
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("wgpu-sink-clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: BlitCompositing::ReplaceTarget.load(),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        self.ctx.queue.submit([encoder.finish()]);
    }

    pub fn draw_rectangles(
        &self,
        rectangles: &[FilledRectangle],
        view: &wgpu::TextureView,
        target_size: (u32, u32),
    ) {
        let vertices = rectangle_vertices(rectangles, target_size);
        if vertices.is_empty() {
            return;
        }
        let device = &self.ctx.device;
        let bytes: Vec<u8> = vertices
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-sink-overlay"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.ctx.queue.write_buffer(&buffer, 0, &bytes);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu-sink-overlay"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("wgpu-sink-overlay"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.overlay_pipeline);
            pass.set_vertex_buffer(0, buffer.slice(..));
            let vertex_count =
                (vertices.len() / crate::filled_rectangles::FLOATS_PER_VERTEX) as u32;
            pass.draw(0..vertex_count, 0..1);
        }
        self.ctx.queue.submit([encoder.finish()]);
    }

    pub fn draw_image(
        &mut self,
        image: &OverlayImage,
        view: &wgpu::TextureView,
    ) -> Result<(), G2gError> {
        let row_bytes = image.width.checked_mul(4).ok_or(G2gError::CapsMismatch)?;
        if (image.rgba.len() as u64) < u64::from(row_bytes) * u64::from(image.height) {
            return Err(G2gError::CapsMismatch);
        }
        let size = wgpu::Extent3d {
            width: image.width,
            height: image.height,
            depth_or_array_layers: 1,
        };
        let current = self
            .image_upload
            .as_ref()
            .filter(|uploaded| uploaded.texture.size() == size);
        if current.is_none_or(|uploaded| uploaded.generation != image.generation) {
            let texture = match current {
                Some(uploaded) => uploaded.texture.clone(),
                None => self.ctx.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("wgpu-sink-image"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                }),
            };
            self.ctx.queue.write_texture(
                texture.as_image_copy(),
                &image.rgba,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row_bytes),
                    rows_per_image: Some(image.height),
                },
                size,
            );
            self.image_upload = Some(UploadedImage {
                texture,
                generation: image.generation,
            });
        }
        let texture = &self.image_upload.as_ref().expect("uploaded above").texture;
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu-sink-image"),
            });
        self.image_blit.encode(
            &self.ctx.device,
            &mut encoder,
            texture,
            WgpuTextureLayout::Rgba.into(),
            &self.resources,
            BlitTarget {
                view,
                viewport: image.placement,
            },
        );
        self.ctx.queue.submit([encoder.finish()]);
        Ok(())
    }

    /// Upload a tightly packed system-memory frame into the renderer's own
    /// texture in the negotiated layout. Each plane is written as its own region,
    /// stacked top to bottom, since an odd-width NV12 frame has luma and chroma
    /// rows of different lengths.
    fn upload_system(&mut self, bytes: &[u8]) -> Result<(wgpu::Texture, BlitLayout), G2gError> {
        let source = self.source.ok_or(G2gError::NotConfigured)?;
        let converted = source.input.convert_frame(bytes)?;
        let bytes = converted.as_deref().unwrap_or(bytes);
        let (texture_format, size) = source.upload_texture()?;
        let (width, height) = (source.width, source.height);
        let frame_bytes = source
            .input
            .presented
            .unpadded_frame_bytes(width, height)
            .ok_or(G2gError::CapsMismatch)?;
        if (bytes.len() as u64) < frame_bytes {
            return Err(G2gError::CapsMismatch);
        }
        let texture = match &self.upload {
            Some(texture) => texture.clone(),
            None => {
                let texture = self.ctx.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("wgpu-sink-upload"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: texture_format,
                    usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                self.upload = Some(texture.clone());
                texture
            }
        };
        let texel_bytes = texture_format
            .block_copy_size(None)
            .ok_or(G2gError::UnsupportedDomain)?;
        let mut plane_top = 0;
        for plane in 0..source.input.presented.plane_count() {
            let stride = source
                .input
                .presented
                .plane_stride(plane, width)
                .ok_or(G2gError::CapsMismatch)?;
            let rows = source
                .input
                .presented
                .plane_rows(plane, height)
                .ok_or(G2gError::CapsMismatch)?;
            let offset = source
                .input
                .presented
                .plane_offset(plane, width, height)
                .ok_or(G2gError::CapsMismatch)?;
            self.ctx.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: plane_top,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                bytes,
                wgpu::TexelCopyBufferLayout {
                    offset,
                    bytes_per_row: Some(stride),
                    rows_per_image: Some(rows),
                },
                wgpu::Extent3d {
                    width: stride / texel_bytes,
                    height: rows,
                    depth_or_array_layers: 1,
                },
            );
            plane_top += rows;
        }
        Ok((texture, source.layout))
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
        mut config: wgpu::SurfaceConfiguration,
    ) -> Self {
        let supported = surface.get_capabilities(&ctx.adapter).formats;
        let format = presentable_format(config.format, &supported);
        if format != config.format {
            config.format = format;
            surface.configure(&ctx.device, &config);
        }
        Self::build(ctx, format, Target::Surface { surface, config })
    }

    fn build(ctx: GpuContext, target_format: wgpu::TextureFormat, target: Target) -> Self {
        Self {
            renderer: VideoRenderer::new(ctx.clone(), target_format, None),
            ctx,
            target,
            presented: 0,
            pacer: PresentationPacer::for_video_sink(),
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

    /// Stretch the picture to fill the target (`false`) instead of letterboxing
    /// it at its display aspect (`true`, the default).
    pub fn with_force_aspect_ratio(mut self, force_aspect_ratio: bool) -> Self {
        self.renderer.set_force_aspect_ratio(force_aspect_ratio);
        self
    }

    #[cfg(all(
        any(target_os = "linux", target_os = "windows"),
        feature = "wgpu-present"
    ))]
    pub(crate) fn picture_point(&self, target_x: f64, target_y: f64) -> Option<(u32, u32)> {
        self.renderer
            .picture_point(self.target_size(), target_x, target_y)
    }

    /// QoS late-drop bound: once PTS pacing is engaged, a frame past its
    /// end by more than `ns` is dropped instead of presented late, so the
    /// sink catches up. The default is 5 ms, `u64::MAX` never drops.
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

    /// Present one frame's pixels onto the target. For a surface target,
    /// acquires and presents the swapchain image; for offscreen, renders into the
    /// owned texture. The path [`process`](AsyncElement::process) and the
    /// windowed present sink share, past pacing.
    pub fn present_frame(&mut self, domain: &MemoryDomain) -> Result<(), G2gError> {
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
        let target_size = self.target_size();
        self.renderer.draw(domain, &dst_view, target_size)?;
        if let Some(frame) = surface_frame {
            self.ctx.queue.present(frame);
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
        read_texture_plane(
            &self.ctx.device,
            &self.ctx.queue,
            texture,
            wgpu::TextureAspect::All,
            crate::gpu::whole_texture(*width, *height),
            RGBA_BYTES_PER_PIXEL,
        )
    }
}

/// Bytes per pixel of the RGBA8 offscreen target.
const RGBA_BYTES_PER_PIXEL: u32 = 4;

impl AsyncElement for WgpuSink {
    fn latency(&self) -> g2g_core::LatencyReport {
        self.pacer.latency()
    }

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
    /// declaring this is what makes the auto-plug do it.
    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::WgpuTexture).with(MemoryDomainKind::System)
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.renderer.configure(absolute_caps)?;
        Ok(ConfigureOutcome::Accepted)
    }

    /// Adopt the elected clock + base time so textures are blitted at their PTS
    /// deadline rather than as fast as the producer pushes them.
    fn set_clock_sync(&mut self, sync: ClockSync) {
        self.pacer.set_clock_sync(sync);
    }

    /// Relay a late drop upstream: the runner forwards it onto the
    /// incoming link, where the producer can shed load.
    fn take_qos(&mut self) -> Option<QosMessage> {
        self.pacer.take_qos()
    }

    fn presentation_stats(&self) -> Option<PresentationStats> {
        Some(PresentationStats {
            presented: self.presented_count(),
            dropped: 0,
            late_dropped: self.late_dropped(),
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        &[
            MAX_LATENESS_PROPERTY,
            PROCESSING_DEADLINE_PROPERTY,
            QOS_INTERVAL_PROPERTY,
            SYNC_PROPERTY,
            FORCE_ASPECT_RATIO_PROPERTY,
        ]
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            FORCE_ASPECT_RATIO_NAME => {
                self.renderer
                    .set_force_aspect_ratio(value.as_bool().ok_or(PropError::Type)?);
                Ok(())
            }
            _ => self
                .pacer
                .set_property(name, &value)
                .unwrap_or(Err(PropError::Unknown)),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            FORCE_ASPECT_RATIO_NAME => Some(PropValue::Bool(self.renderer.force_aspect_ratio())),
            _ => self.pacer.get_property(name),
        }
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if self.renderer.source.is_none() {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    // PTS pacing: hold the texture until its deadline on the
                    // elected clock, or drop it when it is already too late (the
                    // QoS bound) or outside the segment. Unpaced without a clock:
                    // blit as fast as the producer pushes.
                    let paced = self.pacer.judge(&frame.timing, self.presented);
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
        Frame::new(
            MemoryDomain::WgpuTexture(OwnedWgpuTexture::new(
                w,
                h,
                alloc::sync::Arc::new(WgpuTextureKeepAlive::new(
                    ctx.device.clone(),
                    ctx.queue.clone(),
                    texture,
                )),
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

    #[test]
    fn an_srgb_surface_presents_through_its_non_srgb_format() {
        use wgpu::TextureFormat::{Bgra8Unorm, Bgra8UnormSrgb};
        assert_eq!(
            presentable_format(Bgra8UnormSrgb, &[Bgra8UnormSrgb, Bgra8Unorm]),
            Bgra8Unorm
        );
        assert_eq!(
            presentable_format(Bgra8UnormSrgb, &[Bgra8UnormSrgb]),
            Bgra8UnormSrgb
        );
    }

    #[tokio::test]
    async fn offscreen_blit_reproduces_source_orientation() {
        let _gpu = crate::GPU_TEST_LOCK.lock().await;
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
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
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
        let _gpu = crate::GPU_TEST_LOCK.lock().await;
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
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
        };
        let mut sink = WgpuSink::offscreen(ctx.clone(), w, h).with_force_aspect_ratio(false);
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
        for (i, px) in out.as_chunks::<4>().0.iter().enumerate() {
            assert!(
                px[1] > 200 && px[0] < 50,
                "pixel {i} of the resized target is the blitted source: {px:?}"
            );
        }
        assert_eq!(sink.presented_count(), 1);
    }

    #[tokio::test]
    async fn the_picture_is_letterboxed_at_its_display_aspect() {
        let _gpu = crate::GPU_TEST_LOCK.lock().await;
        let Some(ctx) = shared_ctx().await else {
            std::eprintln!("no wgpu adapter; skipping WgpuSink letterbox test");
            return;
        };
        // 8x8 pixels twice as wide as tall show 2:1, so a 16x16 target keeps rows 4..12
        let (w, h) = (8u32, 8u32);
        let (target_w, target_h) = (16u32, 16u32);
        let wide_pixels =
            g2g_core::PixelAspect::Exact(g2g_core::AspectRatio::new(2, 1).expect("nonzero"));
        let caps = Caps::RawVideo {
            format: g2g_core::RawVideoFormat::Rgba8,
            width: g2g_core::Dim::Fixed(w),
            height: g2g_core::Dim::Fixed(h),
            framerate: g2g_core::Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: wide_pixels,
        };
        let mut sink = WgpuSink::offscreen(ctx.clone(), target_w, target_h);
        sink.configure_pipeline(&caps).unwrap();
        let pixels = alloc::vec![0u8, 255, 0, 255].repeat((w * h) as usize);
        let frame = wgpu_frame(&ctx, w, h, source_texture(&ctx, w, h, &pixels), 0);
        sink.process(PipelinePacket::DataFrame(frame), &mut NullSink)
            .await
            .unwrap();

        let out = sink.read_target().unwrap();
        let green_at = |y: u32| out[((y * target_w + target_w / 2) * 4 + 1) as usize] > 200;
        assert!(!green_at(0) && !green_at(3), "black bar above");
        assert!(green_at(4) && green_at(11), "picture in the middle rows");
        assert!(!green_at(12) && !green_at(15), "black bar below");
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
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
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

    /// A `w` x `h` NV12 frame with a luma triangle wave over one constant chroma
    /// pair. Flat chroma is deliberate: the shader fetches the chroma pair
    /// nearest each pixel, so a chroma edge would compare a texel-selection
    /// difference rather than the colour math.
    fn nv12_luma_ramp(w: u32, h: u32, cb: u8, cr: u8) -> Vec<u8> {
        let (w, h) = (w as usize, h as usize);
        let mut bytes = alloc::vec![0u8; w * h + w * h / 2];
        for row in 0..h {
            for col in 0..w {
                let t = (row * 7 + col * 3) % 510;
                bytes[row * w + col] = if t < 255 { t } else { 509 - t } as u8;
            }
        }
        for pair in 0..(w * h / 4) {
            bytes[w * h + pair * 2] = cb;
            bytes[w * h + pair * 2 + 1] = cr;
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
        let _gpu = crate::GPU_TEST_LOCK.lock().await;
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
        let _gpu = crate::GPU_TEST_LOCK.lock().await;
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

    /// The shader converts with the caps colorimetry: one NV12 frame rendered
    /// under the untagged default and under BT.709 must each match the CPU
    /// reference of that colorimetry (within a per-channel LSB or two, the
    /// shader working in floats where the reference is 8-bit fixed point), and
    /// the two renders must differ, which a shader ignoring the caps could not
    /// manage.
    #[tokio::test]
    async fn nv12_shader_converts_with_the_caps_colorimetry() {
        let _gpu = crate::GPU_TEST_LOCK.lock().await;
        let Some(ctx) = shared_ctx().await else {
            std::eprintln!("no wgpu adapter; skipping WgpuSink colorimetry test");
            return;
        };
        let (w, h) = (8u32, 8u32);
        let bytes = nv12_luma_ramp(w, h, 90, 200);
        let mut rendered = Vec::new();
        for colorimetry in [g2g_core::Colorimetry::UNKNOWN, g2g_core::Colorimetry::BT709] {
            let mut nv12_caps = caps(g2g_core::RawVideoFormat::Nv12, w, h);
            if let Caps::RawVideo { colorimetry: c, .. } = &mut nv12_caps {
                *c = colorimetry;
            }
            let mut sink = WgpuSink::offscreen(ctx.clone(), w, h);
            sink.configure_pipeline(&nv12_caps).unwrap();
            sink.process(
                PipelinePacket::DataFrame(system_frame(bytes.clone())),
                &mut NullSink,
            )
            .await
            .unwrap();
            let out = sink.read_target().unwrap();

            let matrix = crate::yuvmatrix::YuvRgbMatrix::new(colorimetry);
            let mut max_delta = 0i32;
            for y in 0..h as usize {
                for x in 0..w as usize {
                    let luma = bytes[y * w as usize + x] as i32;
                    let want = matrix.yuv_to_rgb(luma, 90, 200);
                    let px = (y * w as usize + x) * 4;
                    for (channel, reference) in [want.0, want.1, want.2].into_iter().enumerate() {
                        let got = out[px + channel] as i32;
                        max_delta = max_delta.max((got - reference).abs());
                    }
                }
            }
            assert!(
                max_delta <= 2,
                "GPU NV12 convert ({colorimetry:?}) is off the CPU reference by {max_delta}"
            );
            rendered.push(out);
        }
        assert_ne!(
            rendered[0], rendered[1],
            "BT.709 caps must render differently from the untagged default"
        );
    }

    #[tokio::test]
    async fn planar_deep_yuv_draws_like_its_eight_bit_codes() {
        let _gpu = crate::GPU_TEST_LOCK.lock().await;
        let Some(ctx) = shared_ctx().await else {
            std::eprintln!("no wgpu adapter; skipping the planar deep YUV test");
            return;
        };
        let (w, h) = (8u32, 8u32);
        let (cb, cr) = (90u8, 200u8);
        let luma = &nv12_luma_ramp(w, h, cb, cr)[..(w * h) as usize];
        let colorimetry = g2g_core::Colorimetry::BT709;
        let matrix = crate::yuvmatrix::YuvRgbMatrix::new(colorimetry);
        for format in [
            g2g_core::RawVideoFormat::I420p10,
            g2g_core::RawVideoFormat::I422p10,
            g2g_core::RawVideoFormat::I444p12,
        ] {
            let shift = format.bit_depth() - 8;
            let [_, (_, chroma_width, chroma_height), _] =
                crate::pixel::planar_planes(format, w as usize, h as usize);
            let chroma_samples = chroma_width * chroma_height;
            let bytes: Vec<u8> = luma
                .iter()
                .copied()
                .chain(core::iter::repeat_n(cb, chroma_samples))
                .chain(core::iter::repeat_n(cr, chroma_samples))
                .flat_map(|code| (u16::from(code) << shift).to_le_bytes())
                .collect();
            let mut planar_caps = caps(format, w, h);
            if let Caps::RawVideo { colorimetry: c, .. } = &mut planar_caps {
                *c = colorimetry;
            }
            let mut sink = WgpuSink::offscreen(ctx.clone(), w, h);
            sink.configure_pipeline(&planar_caps).unwrap();
            sink.process(
                PipelinePacket::DataFrame(system_frame(bytes)),
                &mut NullSink,
            )
            .await
            .unwrap();
            let out = sink.read_target().unwrap();

            let mut max_delta = 0i32;
            for (pixel, &code) in luma.iter().enumerate() {
                let want = matrix.yuv_to_rgb(code as i32, cb as i32, cr as i32);
                for (channel, reference) in [want.0, want.1, want.2].into_iter().enumerate() {
                    let got = out[pixel * 4 + channel] as i32;
                    max_delta = max_delta.max((got - reference).abs());
                }
            }
            assert!(
                max_delta <= 2,
                "GPU {format:?} convert is off the CPU reference by {max_delta}"
            );
        }
    }

    #[tokio::test]
    async fn packed_p010_matches_planar_ten_bit_yuv() {
        let _gpu = crate::GPU_TEST_LOCK.lock().await;
        let Some(ctx) = shared_ctx().await else {
            std::eprintln!("no wgpu adapter; skipping the P010 test");
            return;
        };
        let (w, h) = (33u32, 3u32);
        let chroma_samples = (w.div_ceil(2) * h.div_ceil(2)) as usize;
        let (cb, cr) = (440u16, 552u16);
        let luma = || (0..h).flat_map(|_| 400..400 + w as u16);
        let planar = luma()
            .chain(core::iter::repeat_n(cb, chroma_samples))
            .chain(core::iter::repeat_n(cr, chroma_samples))
            .flat_map(u16::to_le_bytes)
            .collect();
        let packed = luma()
            .chain(core::iter::repeat_n([cb, cr], chroma_samples).flatten())
            .flat_map(|value| (value << 6).to_le_bytes())
            .collect();

        let mut sink = WgpuSink::offscreen(ctx, w, h);
        let mut rendered = Vec::new();
        for (format, bytes) in [
            (RawVideoFormat::I420p10, planar),
            (RawVideoFormat::P010, packed),
        ] {
            sink.configure_pipeline(&caps(format, w, h)).unwrap();
            sink.process(
                PipelinePacket::DataFrame(system_frame(bytes)),
                &mut NullSink,
            )
            .await
            .unwrap();
            rendered.push(sink.read_target().unwrap());
        }
        assert_eq!(rendered[0], rendered[1]);
    }

    // converts to 97.81 at 8 bits, which rounding alone draws as 98 everywhere
    const LUMA_BETWEEN_CODES: u8 = 100;
    const NEUTRAL_CHROMA: u8 = 128;
    const DITHERED_SIDE: u32 = 64;
    // rounding alone misses the exact value by 0.19 codes here
    const DITHERED_MEAN_TOLERANCE: f64 = 0.1;
    const EIGHT_BITS: u32 = 8;

    #[tokio::test]
    async fn a_dithered_renderer_averages_to_the_exact_converted_value() {
        let _gpu = crate::GPU_TEST_LOCK.lock().await;
        let Some(ctx) = shared_ctx().await else {
            std::eprintln!("no wgpu adapter; skipping the dither test");
            return;
        };
        let side = DITHERED_SIDE;
        let mut renderer =
            VideoRenderer::new(ctx.clone(), WgpuSink::OFFSCREEN_FORMAT, Some(EIGHT_BITS));
        renderer
            .configure(&caps(g2g_core::RawVideoFormat::Nv12, side, side))
            .unwrap();
        let pixels = (side * side) as usize;
        let mut bytes = alloc::vec![LUMA_BETWEEN_CODES; pixels];
        bytes.extend(core::iter::repeat_n(NEUTRAL_CHROMA, pixels / 2));
        let frame = MemoryDomain::System(g2g_core::memory::SystemSlice::from_boxed(
            bytes.into_boxed_slice(),
        ));
        let texture = offscreen_texture(&ctx, side, side);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        renderer.draw(&frame, &view, (side, side)).unwrap();
        let out = read_texture_plane(
            &ctx.device,
            &ctx.queue,
            &texture,
            wgpu::TextureAspect::All,
            crate::gpu::whole_texture(side, side),
            RGBA_BYTES_PER_PIXEL,
        )
        .unwrap();

        let reds = out
            .chunks(RGBA_BYTES_PER_PIXEL as usize)
            .map(|pixel| f64::from(pixel[0]));
        let mean = reds.sum::<f64>() / pixels as f64;
        let weights = YuvToRgbWeights::new(Colorimetry::UNKNOWN);
        let code_max = f64::from(u8::MAX);
        let exact = f64::from(weights.luma_gain)
            * (f64::from(LUMA_BETWEEN_CODES) / code_max - f64::from(weights.luma_floor))
            * code_max;
        assert!(
            (mean - exact).abs() < DITHERED_MEAN_TOLERANCE,
            "mean red {mean:.3}, exact {exact:.3}"
        );
    }

    /// The zero-copy input the CUDA / GPU-decode bridges emit: an R8Uint packed
    /// NV12 texture already on the device. It is sampled where it lies, and comes
    /// out pixel-identical to the same bytes uploaded from system memory.
    #[tokio::test]
    async fn packed_nv12_texture_presents_without_an_upload() {
        let _gpu = crate::GPU_TEST_LOCK.lock().await;
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
        let _gpu = crate::GPU_TEST_LOCK.lock().await;
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
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
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
        assert_eq!(
            AsyncElement::presentation_stats(&sink),
            Some(PresentationStats {
                presented: 2,
                dropped: 0,
                late_dropped: 1,
            })
        );
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
        let _gpu = crate::GPU_TEST_LOCK.lock().await;
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
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
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
