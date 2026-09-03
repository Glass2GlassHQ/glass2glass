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

use g2g_core::frame::Frame;
use g2g_core::memory::{DomainSet, MemoryDomainKind, SystemSlice};
use g2g_core::runtime::block_on;
use g2g_core::{
    AllocationParams, AsyncElement, Caps, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PipelinePacket, PushOutcome, Rate, RawVideoFormat, VideoCodec,
};
use g2g_plugins::gpu::WgpuNv12Texture;
use g2g_plugins::vulkanvideo::{
    open_h264_decode_device, VulkanImageOwner, VulkanVideoDec, VulkanVideoError,
};

const H264_CLIP: &[u8] = include_bytes!("fixtures/h264_640x480.h264");
const W: u32 = 640;
const H: u32 = 480;
const FRAMERATE_Q16: u32 = 30 << 16;
/// Frames in the fixture clip.
const CLIP_FRAMES: usize = 10;
/// Bytes per texel of the two NV12 planes: luma R8, interleaved chroma Rg8.
const LUMA_BYTES_PER_TEXEL: u32 = 1;
const CHROMA_BYTES_PER_TEXEL: u32 = 2;
/// wgpu's required `bytes_per_row` alignment for texture -> buffer copies.
const COPY_ROW_ALIGN: usize = 256;

#[derive(Default)]
struct Collect {
    frames: Vec<Frame>,
}

impl OutputSink for Collect {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        if let Some(PipelinePacket::DataFrame(f)) = packet_slot.take() {
            self.frames.push(f);
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn au_frame(bytes: &[u8]) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.to_vec().into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    }
}

fn in_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(W),
        height: Dim::Fixed(H),
        framerate: Rate::Fixed(FRAMERATE_Q16),
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn nv12_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(W),
        height: Dim::Fixed(H),
        framerate: Rate::Fixed(FRAMERATE_Q16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

/// Whether this host can run the texture paths at all; `None` with the reason
/// when it cannot.
fn skip_reason() -> Option<&'static str> {
    match block_on(open_h264_decode_device()) {
        Ok(dev) => {
            if !dev
                .wgpu_device
                .features()
                .contains(wgpu::Features::TEXTURE_FORMAT_NV12)
            {
                return Some("wgpu device lacks TEXTURE_FORMAT_NV12");
            }
            None
        }
        Err(VulkanVideoError::NoVulkanAdapter) => Some("no Vulkan adapter"),
        Err(VulkanVideoError::NoDecodeQueue) => Some("no H.264 decode queue"),
        Err(VulkanVideoError::ExtensionUnsupported) => Some("decode extensions unsupported"),
        Err(e) => panic!("unexpected device open failure: {e:?}"),
    }
}

/// Decode the clip with the output domain negotiated to `domain`, delivering
/// `pinned_caps` first as the runner does with the solved output caps.
fn decode(domain: MemoryDomainKind, pinned_caps: Option<Caps>) -> (VulkanVideoDec, Vec<Frame>) {
    let mut dec = VulkanVideoDec::new();
    dec.configure_allocation(&AllocationParams {
        size_bytes: 0,
        min_buffers: 1,
        align: 1,
        domain,
        accepts: DomainSet::only(domain),
        ..Default::default()
    });
    dec.configure_pipeline(&in_caps())
        .expect("configure opens the decode device");
    let mut collect = Collect::default();
    if let Some(caps) = pinned_caps {
        block_on(dec.process(PipelinePacket::CapsChanged(caps), &mut collect))
            .expect("pre-fixed output caps accepted");
    }
    block_on(dec.process(PipelinePacket::DataFrame(au_frame(H264_CLIP)), &mut collect))
        .expect("decode elementary stream");
    block_on(dec.process(PipelinePacket::Eos, &mut collect)).expect("eos drains the reorder");
    (dec, collect.frames)
}

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

/// The system-memory NV12 decode every texture path has to match.
fn reference_nv12() -> Vec<Vec<u8>> {
    let (_, frames) = decode(MemoryDomainKind::System, None);
    assert_eq!(frames.len(), CLIP_FRAMES);
    frames
        .into_iter()
        .map(|f| f.domain.as_system_slice().expect("system frame").to_vec())
        .collect()
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
