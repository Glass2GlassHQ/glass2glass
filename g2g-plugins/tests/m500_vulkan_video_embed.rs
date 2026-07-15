//! M500: consume decoded frames in an *application-owned* wgpu render pipeline.
//!
//! The M498/M499 player hands out decoded frames as `wgpu::Texture`s, but so far
//! only g2g's own `WgpuSink` has sampled them. The Rerun / Bevy integration
//! surface is the opposite: a viewer brings its *own* renderer and samples the
//! decoded frame as a texture in its *own* render pass. This test stands in for
//! that consumer -- a tiny "engine" with its own shader, bind group and pipeline,
//! built on the shared decode device (`VulkanVideoPlayer::gpu_context`) -- and
//! renders each decoded frame through a deliberate transform (grayscale) into an
//! offscreen target.
//!
//! It proves the integration primitive: a `frame_at` texture is a first-class
//! sampled input to a foreign render pipeline (right usage flags, format and
//! state), zero-copy on the decode device. If the texture were not bindable, the
//! render would fail; if the engine's shader did not run on *our* frame, the
//! output would not be a grayscale of the decoded pixels.
//!
//! Runs on the RTX 3060; skips with no Vulkan H.264 adapter / no compute queue.
#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "vulkan-video"))]

use g2g_core::runtime::block_on;
use g2g_plugins::gpu::GpuContext;
use g2g_plugins::vulkanvideo::{open_h264_decode_device, VulkanVideoError, VulkanVideoPlayer};

const CLIP: &[u8] = include_bytes!("fixtures/h264_640x480.h264");

/// A consumer render pipeline that samples an input texture and writes its
/// grayscale (Rec.601 luma) -- an app-authored effect a plain blit would not do,
/// so the output proves the pipeline both bound our texture and ran its shader.
const ENGINE_WGSL: &str = r#"
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    // Fullscreen triangle.
    var xy = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    var out: VsOut;
    out.pos = vec4<f32>(xy[i], 0.0, 1.0);
    out.uv = vec2<f32>((xy[i].x + 1.0) * 0.5, (xy[i].y + 1.0) * 0.5);
    return out;
}

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let c = textureSample(tex, samp, in.uv);
    let g = dot(c.rgb, vec3<f32>(0.299, 0.587, 0.114));
    return vec4<f32>(g, g, g, 1.0);
}
"#;

/// The application's renderer: owns a pipeline that samples one texture.
struct EngineRenderer {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

impl EngineRenderer {
    fn new(ctx: &GpuContext) -> Self {
        let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine-shader"),
            source: wgpu::ShaderSource::Wgsl(ENGINE_WGSL.into()),
        });
        let layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });
        let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine-pl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = ctx.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("engine-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("engine-sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        Self { pipeline, layout, sampler }
    }

    /// Render `src` through the engine's shader into a fresh offscreen target.
    fn render(&self, ctx: &GpuContext, src: &wgpu::Texture, w: u32, h: u32) -> wgpu::Texture {
        let target = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("engine-target"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let src_view = src.create_view(&wgpu::TextureViewDescriptor::default());
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("engine-bg"),
            layout: &self.layout,
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
        let mut enc =
            ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("engine-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
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
            rp.set_pipeline(&self.pipeline);
            rp.set_bind_group(0, &bind, &[]);
            rp.draw(0..3, 0..1);
        }
        ctx.queue.submit([enc.finish()]);
        target
    }
}

/// Rec.601 luma of an RGBA buffer, one byte per pixel, matching the shader.
fn luma(rgba: &[u8]) -> Vec<u8> {
    rgba.chunks_exact(4)
        .map(|c| (0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32).round() as u8)
        .collect()
}

fn sad(a: &[u8], b: &[u8]) -> f64 {
    let sum: u64 = a.iter().zip(b).map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u64).sum();
    sum as f64 / a.len().max(1) as f64
}

/// Vertically flip a one-byte-per-pixel image (to compare orientation-tolerantly,
/// since a fullscreen-triangle's UV convention may flip vs the readback rows).
fn vflip(gray: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; gray.len()];
    for y in 0..h {
        let src = &gray[y * w..(y + 1) * w];
        out[(h - 1 - y) * w..(h - y) * w].copy_from_slice(src);
    }
    out
}

#[test]
fn m500_vulkan_video_embed() {
    let device = match block_on(open_h264_decode_device()) {
        Ok(d) => d,
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip m500: no Vulkan H.264 decode adapter");
            return;
        }
        Err(e) => panic!("open decode device: {e:?}"),
    };

    let mut player = match VulkanVideoPlayer::new(device, CLIP.to_vec(), 30) {
        Ok(p) => p,
        Err(VulkanVideoError::NoComputeQueue) => {
            eprintln!("skip m500: no distinct compute queue for the RGBA path");
            return;
        }
        Err(e) => panic!("build player: {e:?}"),
    };
    let (w, h) = player.dimensions();
    let n = player.frame_count();

    // The consumer builds its renderer on the shared decode device (zero copy).
    let ctx = player.gpu_context();
    let engine = EngineRenderer::new(&ctx);

    // Consume a few frames across the stream through the app's own pipeline.
    for p in [0usize, 3, 7].into_iter().filter(|&p| p < n) {
        let frame_tex = player.frame_at_index(p).expect("frame_at_index").clone();
        let frame_rgba = player.read_texture(&frame_tex);

        let out_tex = engine.render(&ctx, &frame_tex, w, h);
        let out_rgba = player.read_texture(&out_tex);

        // 1. The app's grayscale shader ran: every output pixel is R==G==B.
        assert!(
            out_rgba.chunks_exact(4).all(|c| c[0].abs_diff(c[1]) <= 1 && c[1].abs_diff(c[2]) <= 1),
            "frame {p}: engine output must be grayscale (R==G==B), proving its shader ran",
        );

        // 2. It sampled real content, not a blank/cleared target.
        let out_gray: Vec<u8> = out_rgba.chunks_exact(4).map(|c| c[0]).collect();
        let (min, max) = out_gray.iter().fold((255u8, 0u8), |(lo, hi), &g| (lo.min(g), hi.max(g)));
        assert!(max - min > 20, "frame {p}: engine output is uniform ({min}..{max}); nothing sampled");

        // 3. It is the grayscale of *our* decoded frame (orientation-tolerant, to
        //    absorb the fullscreen-triangle UV flip; <= 2 for GPU rounding).
        let expected = luma(&frame_rgba);
        let direct = sad(&out_gray, &expected);
        let flipped = sad(&out_gray, &vflip(&expected, w as usize, h as usize));
        assert!(
            direct.min(flipped) <= 2.0,
            "frame {p}: engine output must be the grayscale of the decoded frame \
             (SAD direct {direct:.3}, flipped {flipped:.3})",
        );
    }
}
