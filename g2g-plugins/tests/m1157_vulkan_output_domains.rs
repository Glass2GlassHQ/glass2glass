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
use vulkan_nv12_common::{
    decode, nv12_caps, read_two_plane, reference_nv12, skip_reason, two_plane_frame, CLIP_FRAMES,
    H, W,
};

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
        let owner = two_plane_frame(frame);
        assert_eq!(owner.texture().format(), wgpu::TextureFormat::NV12);
        let bytes = read_two_plane(owner.device(), owner.queue(), owner.texture());
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
        assert_eq!(owner.texture().format(), wgpu::TextureFormat::NV12);
        let bytes = read_two_plane(&ctx.device, &ctx.queue, owner.texture());
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
