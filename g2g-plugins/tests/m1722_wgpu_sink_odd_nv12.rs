#![cfg(feature = "wgpu-sink")]

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex, MutexGuard};

use g2g_core::frame::Frame;
use g2g_core::memory::{OwnedWgpuTexture, SystemSlice};
use g2g_core::runtime::block_on;
use g2g_core::{
    AsyncElement, Caps, Colorimetry, Dim, FrameTiming, G2gError, Interlace, MemoryDomain,
    OutputSink, PipelinePacket, PushOutcome, Rate, RawVideoFormat,
};
use g2g_plugins::gpu::{compliant_adapter, primary_instance, GpuContext, WgpuNv12Texture};
use g2g_plugins::wgpusink::WgpuSink;
use g2g_plugins::yuvmatrix::YuvToRgbWeights;

// odd width and height, then each one alone
const ODD_SIZES: [(u32, u32); 3] = [(7, 5), (6, 5), (7, 6)];
// the render target quantizes the shader's float result, the reference rounds it
const MAX_CHANNEL_DELTA: i32 = 1;
const LUMA_PLANE: usize = 0;
const CHROMA_PLANE: usize = 1;
const RGBA_PIXEL_BYTES: usize = 4;
const CHROMA_TEXEL_BYTES: u32 = 2;

// libtest runs these in parallel and concurrent instance creation SIGSEGVs the NVIDIA loader
static GPU_LOCK: Mutex<()> = Mutex::new(());

fn gpu_lock() -> MutexGuard<'static, ()> {
    GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner())
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

fn open_context() -> Option<GpuContext> {
    block_on(async {
        let instance = primary_instance();
        let adapter = compliant_adapter(
            &instance,
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            },
        )
        .await?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("m1722"),
                required_features: adapter.features() & wgpu::Features::TEXTURE_FORMAT_NV12,
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("open a device on the adapter");
        Some(GpuContext::from_wgpu(instance, adapter, device, queue))
    })
}

fn nv12_caps(width: u32, height: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        framerate: Rate::Any,
        interlace: Interlace::Any,
        colorimetry: Colorimetry::UNKNOWN,
    }
}

fn plane_offset(plane: usize, width: u32, height: u32) -> usize {
    RawVideoFormat::Nv12
        .plane_offset(plane, width, height)
        .expect("nv12 plane offset") as usize
}

fn plane_stride(plane: usize, width: u32) -> usize {
    RawVideoFormat::Nv12
        .plane_stride(plane, width)
        .expect("nv12 plane stride") as usize
}

// tight NV12 with luma varying per pixel and chroma varying per 2x2 pair
fn tight_nv12(width: u32, height: u32) -> Vec<u8> {
    let frame_bytes = RawVideoFormat::Nv12
        .unpadded_frame_bytes(width, height)
        .expect("nv12 frame size") as usize;
    let mut bytes = vec![0u8; frame_bytes];
    let luma_stride = plane_stride(LUMA_PLANE, width);
    for y in 0..height as usize {
        for x in 0..width as usize {
            bytes[y * luma_stride + x] = ((16 + 23 * x + 41 * y) % 220 + 16) as u8;
        }
    }
    let chroma_offset = plane_offset(CHROMA_PLANE, width, height);
    let chroma_stride = plane_stride(CHROMA_PLANE, width);
    for chroma_y in 0..height.div_ceil(2) as usize {
        for chroma_x in 0..width.div_ceil(2) as usize {
            let at = chroma_offset + chroma_y * chroma_stride + chroma_x * 2;
            bytes[at] = ((40 + 37 * chroma_x + 71 * chroma_y) % 200 + 28) as u8;
            bytes[at + 1] = ((200 + 53 * chroma_x + 29 * chroma_y) % 200 + 28) as u8;
        }
    }
    bytes
}

// the sink's YCbCr -> RGB on the CPU, one RGB triple per picture pixel
fn reference_rgb(bytes: &[u8], width: u32, height: u32) -> Vec<[u8; 3]> {
    let weights = YuvToRgbWeights::new(Colorimetry::UNKNOWN);
    let luma_stride = plane_stride(LUMA_PLANE, width);
    let chroma_offset = plane_offset(CHROMA_PLANE, width, height);
    let chroma_stride = plane_stride(CHROMA_PLANE, width);
    let normalized = |value: u8| value as f32 / 255.0;
    let to_code = |value: f32| (value.clamp(0.0, 1.0) * 255.0).round() as u8;
    let mut rgb = Vec::new();
    for y in 0..height as usize {
        for x in 0..width as usize {
            let chroma_at = chroma_offset + (y / 2) * chroma_stride + (x / 2) * 2;
            let luma =
                weights.luma_gain * (normalized(bytes[y * luma_stride + x]) - weights.luma_floor);
            let cb = normalized(bytes[chroma_at]) - 0.5;
            let cr = normalized(bytes[chroma_at + 1]) - 0.5;
            rgb.push([
                to_code(luma + weights.red_from_cr * cr),
                to_code(luma + weights.green_from_cb * cb + weights.green_from_cr * cr),
                to_code(luma + weights.blue_from_cb * cb),
            ]);
        }
    }
    rgb
}

fn rendered_rgb(rgba: &[u8]) -> Vec<[u8; 3]> {
    rgba.as_chunks::<RGBA_PIXEL_BYTES>()
        .0
        .iter()
        .map(|[red, green, blue, _]| [*red, *green, *blue])
        .collect()
}

fn present(ctx: &GpuContext, width: u32, height: u32, domain: MemoryDomain) -> Vec<[u8; 3]> {
    let mut sink = WgpuSink::offscreen(ctx.clone(), width, height);
    sink.configure_pipeline(&nv12_caps(width, height))
        .expect("the sink accepts odd NV12 caps");
    let frame = Frame::new(domain, FrameTiming::default(), 0);
    block_on(sink.process(PipelinePacket::DataFrame(frame), &mut NullSink)).expect("present");
    rendered_rgb(&sink.read_target().expect("read offscreen target"))
}

fn system_domain(bytes: Vec<u8>) -> MemoryDomain {
    MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice()))
}

// a decoder-style two-plane texture, even-rounded past the picture
fn two_plane_domain(ctx: &GpuContext, bytes: &[u8], width: u32, height: u32) -> MemoryDomain {
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("m1722-two-plane"),
        size: wgpu::Extent3d {
            width: 2 * width.div_ceil(2),
            height: 2 * height.div_ceil(2),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::NV12,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let (chroma_width, chroma_height) = (width.div_ceil(2), height.div_ceil(2));
    for (aspect, plane, plane_width, plane_height, bytes_per_row) in [
        (
            wgpu::TextureAspect::Plane0,
            LUMA_PLANE,
            width,
            height,
            plane_stride(LUMA_PLANE, width) as u32,
        ),
        (
            wgpu::TextureAspect::Plane1,
            CHROMA_PLANE,
            chroma_width,
            chroma_height,
            chroma_width * CHROMA_TEXEL_BYTES,
        ),
    ] {
        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect,
            },
            &bytes[plane_offset(plane, width, height)..],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(plane_height),
            },
            wgpu::Extent3d {
                width: plane_width,
                height: plane_height,
                depth_or_array_layers: 1,
            },
        );
    }
    MemoryDomain::WgpuTexture(OwnedWgpuTexture::new(
        width,
        height,
        Arc::new(WgpuNv12Texture::new(
            ctx.device.clone(),
            ctx.queue.clone(),
            texture,
        )),
    ))
}

fn assert_within_one_code(got: &[[u8; 3]], want: &[[u8; 3]], width: u32, label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: pixel count");
    for (index, (got_pixel, want_pixel)) in got.iter().zip(want).enumerate() {
        let delta = (0..3)
            .map(|channel| (got_pixel[channel] as i32 - want_pixel[channel] as i32).abs())
            .fold(0, i32::max);
        assert!(
            delta <= MAX_CHANNEL_DELTA,
            "{label}: pixel ({}, {}) is {got_pixel:?}, want {want_pixel:?}",
            index as u32 % width,
            index as u32 / width,
        );
    }
}

// a blit that stretches over the texture or reads its padding repeats one value down the edge
fn assert_edges_vary(rgb: &[[u8; 3]], width: u32, height: u32, label: &str) {
    let (width, height) = (width as usize, height as usize);
    let last_column: BTreeSet<[u8; 3]> = (0..height).map(|y| rgb[y * width + width - 1]).collect();
    let last_row: BTreeSet<[u8; 3]> = rgb[(height - 1) * width..].iter().copied().collect();
    assert!(last_column.len() > 1, "{label}: last column is flat");
    assert!(last_row.len() > 1, "{label}: last row is flat");
}

#[test]
fn odd_packed_nv12_renders_only_the_picture() {
    let _gpu = gpu_lock();
    let Some(ctx) = open_context() else {
        eprintln!("skip: no wgpu adapter");
        return;
    };
    for (width, height) in ODD_SIZES {
        let label = format!("packed {width}x{height}");
        let bytes = tight_nv12(width, height);
        let rendered = present(&ctx, width, height, system_domain(bytes.clone()));
        assert_within_one_code(
            &rendered,
            &reference_rgb(&bytes, width, height),
            width,
            &label,
        );
        assert_edges_vary(&rendered, width, height, &label);
    }
}

#[test]
fn odd_two_plane_nv12_matches_the_packed_render() {
    let _gpu = gpu_lock();
    let Some(ctx) = open_context() else {
        eprintln!("skip: no wgpu adapter");
        return;
    };
    if !ctx
        .device
        .features()
        .contains(wgpu::Features::TEXTURE_FORMAT_NV12)
    {
        eprintln!("skip: wgpu device lacks TEXTURE_FORMAT_NV12");
        return;
    }
    // wgpu's dx12 backend has no copy format for an NV12 plane and panics in write_texture
    if ctx.adapter.get_info().backend == wgpu::Backend::Dx12 {
        eprintln!("skip: dx12 cannot upload into NV12 planes");
        return;
    }
    for (width, height) in ODD_SIZES {
        let label = format!("two-plane {width}x{height}");
        let bytes = tight_nv12(width, height);
        let packed = present(&ctx, width, height, system_domain(bytes.clone()));
        let two_plane = present(
            &ctx,
            width,
            height,
            two_plane_domain(&ctx, &bytes, width, height),
        );
        assert_within_one_code(&two_plane, &packed, width, &label);
    }
}
