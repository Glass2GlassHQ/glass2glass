//! M1157: `VulkanVideoDec`'s two GPU-resident NV12 outputs. Pinning `NV12` on
//! the `WgpuTexture` path hands out a two-plane `TextureFormat::NV12` texture
//! instead of the ycbcr pass's RGBA, and the `VulkanTexture` domain hands out the
//! raw `VkImage` behind a keep-alive an `ash` consumer can downcast. Both must be
//! bit-exact with the system-memory NV12 decode of the same clip. Runs on the
//! RTX 3060; skips when the GPU lacks the H.264 decode profile, a compute queue,
//! or the wgpu NV12 texture feature.
#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use g2g_core::memory::MemoryDomainKind;
use g2g_core::MemoryDomain;
use g2g_plugins::gpu::WgpuNv12Texture;
use g2g_plugins::vulkanvideo::VulkanImageOwner;

mod vulkan_nv12_common;
use vulkan_nv12_common::{decode, nv12_caps, reference_nv12, skip_reason, CLIP_FRAMES, H, W};

/// Bytes per texel of the two NV12 planes: luma R8, interleaved chroma Rg8.
const LUMA_BYTES_PER_TEXEL: u32 = 1;
const CHROMA_BYTES_PER_TEXEL: u32 = 2;
/// wgpu's required `bytes_per_row` alignment for texture -> buffer copies.
const COPY_ROW_ALIGN: usize = 256;

/// Read one plane of an NV12 texture back as tightly packed bytes.
fn read_plane(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    aspect: wgpu::TextureAspect,
    width: u32,
    height: u32,
    bytes_per_texel: u32,
) -> Vec<u8> {
    let tight = (width * bytes_per_texel) as usize;
    let padded = tight.div_ceil(COPY_ROW_ALIGN) * COPY_ROW_ALIGN;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("plane-readback"),
        size: (padded * height as usize) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded as u32),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([enc.finish()]);
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("poll");
    rx.recv().expect("map callback").expect("map");
    let mapped = slice.get_mapped_range();
    let mut out = Vec::with_capacity(tight * height as usize);
    for row in 0..height as usize {
        out.extend_from_slice(&mapped[row * padded..row * padded + tight]);
    }
    drop(mapped);
    buffer.unmap();
    out
}

/// Both planes of an NV12 texture, concatenated in the system NV12 byte layout.
fn read_nv12(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<u8> {
    assert_eq!(texture.format(), wgpu::TextureFormat::NV12);
    let mut bytes = read_plane(
        device,
        queue,
        texture,
        wgpu::TextureAspect::Plane0,
        W,
        H,
        LUMA_BYTES_PER_TEXEL,
    );
    bytes.extend(read_plane(
        device,
        queue,
        texture,
        wgpu::TextureAspect::Plane1,
        W / 2,
        H / 2,
        CHROMA_BYTES_PER_TEXEL,
    ));
    bytes
}

#[test]
fn nv12_caps_on_wgpu_texture_hand_out_two_plane_textures() {
    if let Some(reason) = skip_reason() {
        eprintln!("skipping: {reason}");
        return;
    }
    let reference = reference_nv12();
    let (dec, frames) = decode(MemoryDomainKind::WgpuTexture, Some(nv12_caps()));
    assert_eq!(frames.len(), CLIP_FRAMES);
    let ctx = dec.gpu_context().expect("device open");
    for (i, frame) in frames.iter().enumerate() {
        let MemoryDomain::WgpuTexture(owned) = &frame.domain else {
            panic!(
                "frame {i}: expected a WgpuTexture frame, got {:?}",
                frame.domain
            );
        };
        let owner = owned
            .keep_alive()
            .as_any()
            .downcast_ref::<WgpuNv12Texture>()
            .expect("an NV12 texture frame is owned by WgpuNv12Texture");
        let bytes = read_nv12(owner.device(), owner.queue(), owner.texture());
        assert!(
            bytes == reference[i],
            "frame {i}: two-plane texture differs from the system NV12 decode"
        );
    }
    drop(ctx);
}

#[test]
fn vulkan_texture_domain_hands_out_the_raw_image() {
    if let Some(reason) = skip_reason() {
        eprintln!("skipping: {reason}");
        return;
    }
    let reference = reference_nv12();
    let (dec, frames) = decode(MemoryDomainKind::VulkanTexture, None);
    assert_eq!(frames.len(), CLIP_FRAMES);
    let ctx = dec.gpu_context().expect("device open");
    for (i, frame) in frames.iter().enumerate() {
        let MemoryDomain::VulkanTexture(owned) = &frame.domain else {
            panic!(
                "frame {i}: expected a VulkanTexture frame, got {:?}",
                frame.domain
            );
        };
        assert_eq!((owned.width, owned.height), (W, H));
        assert_eq!(
            owned.format,
            ash::vk::Format::G8_B8R8_2PLANE_420_UNORM.as_raw(),
            "frame {i}: the raw image is the decoder's two-plane NV12"
        );
        let owner = owned
            .keep_alive()
            .as_any()
            .downcast_ref::<VulkanImageOwner>()
            .expect("the keep-alive is the decoder's VulkanImageOwner");
        assert_eq!(ash::vk::Handle::as_raw(owner.image()), owned.handle);
        assert_ne!(owned.handle, 0, "frame {i}: null VkImage");
        let bytes = read_nv12(&ctx.device, &ctx.queue, owner.texture());
        assert!(
            bytes == reference[i],
            "frame {i}: raw image differs from the system NV12 decode"
        );
    }
}

#[test]
fn unpinned_wgpu_texture_still_converts_to_rgba() {
    if let Some(reason) = skip_reason() {
        eprintln!("skipping: {reason}");
        return;
    }
    let (_, frames) = decode(MemoryDomainKind::WgpuTexture, None);
    assert_eq!(frames.len(), CLIP_FRAMES);
    for frame in &frames {
        let MemoryDomain::WgpuTexture(owned) = &frame.domain else {
            panic!("expected a WgpuTexture frame");
        };
        assert!(
            owned
                .keep_alive()
                .as_any()
                .downcast_ref::<WgpuNv12Texture>()
                .is_none(),
            "without NV12 pinned the frame is the ycbcr pass's RGBA texture"
        );
    }
}
