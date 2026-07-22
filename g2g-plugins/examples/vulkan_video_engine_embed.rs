//! M500: consume Vulkan-Video-decoded frames in an application-owned wgpu render
//! pipeline (the game-engine / viewer integration surface), and dump the result as PPMs.
//!
//! Unlike `vulkan_video_smoke` (which reads the decoder's NV12 back on the CPU)
//! and `vulkan_video_on_screen` (which lets g2g's own `WgpuSink` present the
//! frame), this stands in for a *third-party* renderer: a tiny "engine" with its
//! own shader / bind group / pipeline, built on the shared decode device
//! (`VulkanVideoPlayer::gpu_context`), samples each decoded frame as a texture in
//! its own render pass. To make the app-side processing unmistakable it renders a
//! split: **left half the decoded frame, right half its grayscale**. That the
//! decoded texture is a first-class sampled input to a foreign pipeline, zero-copy
//! on the decode device, is the whole point.
//!
//! The self-checking version is `m500_vulkan_video_embed`; this is the visual one.
//!
//! Run (needs a Vulkan H.264 decode GPU, e.g. the RTX 3060):
//!
//! ```sh
//! cargo run --release -p g2g-plugins --features vulkan-video \
//!     --example vulkan_video_engine_embed                    # bundled 640x480 clip
//! cargo run --release -p g2g-plugins --features vulkan-video \
//!     --example vulkan_video_engine_embed -- my.h264 /tmp/out
//! ```

use std::io::Write;
use std::path::PathBuf;

use g2g_core::runtime::block_on;
use g2g_plugins::gpu::GpuContext;
use g2g_plugins::vulkanvideo::{open_h264_decode_device, VulkanVideoError, VulkanVideoPlayer};

const BUNDLED_CLIP: &[u8] = include_bytes!("../tests/fixtures/h264_640x480.h264");

/// The app's shader: left half is the decoded frame, right half its grayscale.
const ENGINE_WGSL: &str = r#"
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    var xy = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    var out: VsOut;
    out.pos = vec4<f32>(xy[i], 0.0, 1.0);
    out.uv = vec2<f32>((xy[i].x + 1.0) * 0.5, (1.0 - xy[i].y) * 0.5);
    return out;
}

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    if (in.uv.x < 0.5) {
        return textureSample(tex, samp, vec2<f32>(in.uv.x * 2.0, in.uv.y));
    }
    let c = textureSample(tex, samp, vec2<f32>((in.uv.x - 0.5) * 2.0, in.uv.y));
    let g = dot(c.rgb, vec3<f32>(0.299, 0.587, 0.114));
    return vec4<f32>(g, g, g, 1.0);
}
"#;

fn main() {
    let mut args = std::env::args().skip(1);
    let clip_path = args.next();
    let out_dir = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "engine_embed_out".to_string()),
    );
    let clip: Vec<u8> = match &clip_path {
        Some(p) => std::fs::read(p).unwrap_or_else(|e| panic!("read {p}: {e}")),
        None => BUNDLED_CLIP.to_vec(),
    };

    let device = match block_on(open_h264_decode_device()) {
        Ok(d) => d,
        Err(VulkanVideoError::NoVulkanAdapter) => {
            eprintln!("no Vulkan adapter; this demo needs a GPU with Vulkan H.264 decode.");
            return;
        }
        Err(e) => {
            eprintln!("this GPU has no usable Vulkan H.264 decode: {e:?}");
            return;
        }
    };
    let mut player = match VulkanVideoPlayer::new(device, clip, 30) {
        Ok(p) => p,
        Err(VulkanVideoError::NoComputeQueue) => {
            eprintln!("this GPU exposes no separate compute queue for the NV12->RGBA pass.");
            return;
        }
        Err(e) => panic!("build player: {e:?}"),
    };
    let (w, h) = player.dimensions();
    let n = player.frame_count();
    let ctx = player.gpu_context();
    println!(
        "decoded {n} frames at {w}x{h} on {}; compositing through an app pipeline",
        ctx.adapter.get_info().name
    );

    let engine = Engine::new(&ctx, w, h);
    std::fs::create_dir_all(&out_dir).expect("create out dir");
    for p in 0..n {
        let frame = player.frame_at_index(p).expect("frame_at_index").clone();
        let composited = engine.render(&ctx, &frame);
        let rgba = player.read_texture(&composited);
        let path = out_dir.join(format!("engine_{p:03}.ppm"));
        write_ppm(&path, w, h, &rgba);
    }
    println!(
        "wrote {n} PPMs to {} (left: decoded, right: app grayscale)",
        out_dir.display()
    );
}

/// The application's renderer: its own pipeline sampling one texture.
struct Engine {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    width: u32,
    height: u32,
}

impl Engine {
    fn new(ctx: &GpuContext, width: u32, height: u32) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("engine-shader"),
                source: wgpu::ShaderSource::Wgsl(ENGINE_WGSL.into()),
            });
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("engine-bgl"),
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
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("engine-pl"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            pipeline,
            layout,
            sampler,
            width,
            height,
        }
    }

    fn render(&self, ctx: &GpuContext, src: &wgpu::Texture) -> wgpu::Texture {
        let target = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("engine-target"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
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
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
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

/// Write an RGBA buffer as a binary (P6) PPM, dropping the alpha channel.
fn write_ppm(path: &std::path::Path, w: u32, h: u32, rgba: &[u8]) {
    let mut f = std::fs::File::create(path).unwrap_or_else(|e| panic!("create {path:?}: {e}"));
    write!(f, "P6\n{w} {h}\n255\n").expect("ppm header");
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for px in rgba.chunks_exact(4) {
        rgb.extend_from_slice(&px[..3]);
    }
    f.write_all(&rgb).expect("ppm body");
}
