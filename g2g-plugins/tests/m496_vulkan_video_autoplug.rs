//! M496: `vulkanvideodec` is auto-plugged for a WgpuTexture consumer.
//!
//! The Vulkan Video decoder is registered as an auto-plug candidate tagged
//! `produces(WgpuTexture)` + hardware (the wgpu-texture analog of `NvDec`'s
//! `produces(Cuda)`). A domain-aware `decodebin`-style search that prefers
//! `WgpuTexture` therefore picks it for H.264 (the copy-free wedge into a wgpu
//! consumer), while a plain (System) search is unaffected -- a WgpuTexture
//! producer is a domain mismatch there. GPU-free: this exercises only the
//! registry's capability scoring, so it needs no adapter.
#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "vulkan-video"))]

use g2g_core::memory::MemoryDomainKind;
use g2g_core::runtime::SelectionContext;
use g2g_core::{Caps, Dim, Rate, VideoCodec};
use g2g_plugins::registry::default_registry;

fn compressed_in(codec: VideoCodec) -> Caps {
    Caps::CompressedVideo {
        codec,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn h264_in() -> Caps {
    compressed_in(VideoCodec::H264)
}

/// Target: any decoded raw video.
fn is_raw(caps: &Caps) -> bool {
    matches!(caps, Caps::RawVideo { .. })
}

#[test]
fn wgpu_texture_consumer_autoplugs_vulkanvideodec() {
    let reg = default_registry();

    // A WgpuTexture-preferring search picks the Vulkan Video decoder (domain match
    // dominates the score), routing H.264 straight to a GPU-resident texture.
    let ctx = SelectionContext { preferred_memory: MemoryDomainKind::WgpuTexture, prefer_hardware: false };
    let chain = reg
        .autoplug_names_with(&h264_in(), &is_raw, 4, ctx)
        .expect("a decode chain exists for H.264 -> WgpuTexture");
    assert!(
        chain.contains(&"vulkanvideodec"),
        "expected vulkanvideodec in the WgpuTexture chain, got {chain:?}"
    );
}

// M517 generalized `VulkanVideoDec` over a codec enum (its sink pad template now
// advertises H.264 / H.265 / AV1), so a WgpuTexture-preferring search auto-plugs
// it for H.265 and AV1 streams too, not just H.264. Without this the decodebin /
// playbin path would only reach the hardware decoder for H.264.
#[test]
fn wgpu_texture_consumer_autoplugs_vulkanvideodec_for_h265_and_av1() {
    let reg = default_registry();
    let ctx =
        SelectionContext { preferred_memory: MemoryDomainKind::WgpuTexture, prefer_hardware: false };
    for codec in [VideoCodec::H265, VideoCodec::Av1] {
        let chain = reg
            .autoplug_names_with(&compressed_in(codec), &is_raw, 4, ctx)
            .unwrap_or_else(|| panic!("a decode chain exists for {codec:?} -> WgpuTexture"));
        assert!(
            chain.contains(&"vulkanvideodec"),
            "expected vulkanvideodec in the {codec:?} WgpuTexture chain, got {chain:?}"
        );
    }
}

// With a competing system-memory decoder present (ffmpegdec), a plain (System)
// search prefers it over the WgpuTexture-producing Vulkan decoder -- the domain
// match dominates the score, exactly as NvDec (Cuda) is skipped for System.
// (`vulkanvideodec` is still a valid System *fallback* when it is the only H.264
// decoder available, since it is multi-domain and can emit NV12 -- so this
// preference test only holds when an alternative exists.)
#[cfg(all(target_os = "linux", feature = "ffmpeg"))]
#[test]
fn system_consumer_prefers_the_system_decoder() {
    let reg = default_registry();
    let ctx = SelectionContext { preferred_memory: MemoryDomainKind::System, prefer_hardware: false };
    let chain = reg
        .autoplug_names_with(&h264_in(), &is_raw, 4, ctx)
        .expect("a decode chain exists for H.264 -> System");
    assert!(
        !chain.contains(&"vulkanvideodec"),
        "System search should prefer the system decoder, not vulkanvideodec, got {chain:?}"
    );
}
