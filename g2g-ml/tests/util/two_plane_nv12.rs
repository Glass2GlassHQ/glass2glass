// Shared two-plane NV12 texture producer for the WgpuPreprocess texture-import
// tests (included via `include!`, so plain comments only). Needs `wgpu`.

// bytes per texel of an NV12 chroma plane, Cb and Cr interleaved
const CHROMA_BYTES_PER_TEXEL: u32 = 2;

// Tight NV12 bytes in a `TextureFormat::NV12` texture, the layout a GPU decoder
// hands out: rounded up to an even size past an odd picture, as the Vulkan Video
// decoder does. `None` when the adapter cannot make or fill one.
async fn nv12_to_two_plane_texture(
    nv12: &[u8],
    width: u32,
    height: u32,
) -> Option<g2g_core::memory::MemoryDomain> {
    let format = g2g_core::RawVideoFormat::Nv12;
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .ok()?;
    if !adapter
        .features()
        .contains(wgpu::Features::TEXTURE_FORMAT_NV12)
    {
        return None;
    }
    // wgpu's dx12 backend has no copy format for an NV12 plane and panics in write_texture
    if adapter.get_info().backend == wgpu::Backend::Dx12 {
        return None;
    }
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("two-plane-nv12"),
            required_features: wgpu::Features::TEXTURE_FORMAT_NV12,
            required_limits: adapter.limits(),
            ..Default::default()
        })
        .await
        .ok()?;

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("two-plane-nv12"),
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
    // Each plane is written on its own aspect: a whole-texture write has no
    // single texel size to describe.
    let (chroma_width, chroma_height) = (width.div_ceil(2), height.div_ceil(2));
    let chroma_offset = format.plane_offset(1, width, height)? as usize;
    for (aspect, bytes, plane_width, plane_height, bytes_per_row) in [
        (wgpu::TextureAspect::Plane0, nv12, width, height, width),
        (
            wgpu::TextureAspect::Plane1,
            &nv12[chroma_offset..],
            chroma_width,
            chroma_height,
            chroma_width * CHROMA_BYTES_PER_TEXEL,
        ),
    ] {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect,
            },
            bytes,
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

    let owner = g2g_ml::wgpupreprocess::WgpuNv12Texture::new(device, queue, texture);
    Some(g2g_core::memory::MemoryDomain::WgpuTexture(
        g2g_core::OwnedWgpuTexture::new(width, height, std::sync::Arc::new(owner)),
    ))
}
