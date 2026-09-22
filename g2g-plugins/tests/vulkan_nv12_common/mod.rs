//! The `VulkanVideoDec` two-plane decode harness the M1157 and M1180 tests share: the fixture
//! clip, its caps, the host-capability skip, and a decode run pinned to one
//! output domain.
#![allow(dead_code)] // no one test file uses every helper here

use g2g_core::frame::Frame;
use g2g_core::memory::{DomainSet, MemoryDomainKind, SystemSlice};
use g2g_core::runtime::block_on;
use g2g_core::{
    AllocationParams, AsyncElement, Caps, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PipelinePacket, PushOutcome, Rate, RawVideoFormat, VideoCodec,
};
use g2g_plugins::gpu::WgpuNv12Texture;
use g2g_plugins::vulkanvideo::{open_h264_decode_device, VulkanVideoDec, VulkanVideoError};

pub(crate) const H264_CLIP: &[u8] = include_bytes!("../fixtures/h264_640x480.h264");
pub(crate) const W: u32 = 640;
pub(crate) const H: u32 = 480;
pub(crate) const FRAMERATE_Q16: u32 = 30 << 16;
/// Frames in the fixture clip.
pub(crate) const CLIP_FRAMES: usize = 10;

#[derive(Default)]
pub(crate) struct Collect {
    pub(crate) packets: Vec<PipelinePacket>,
}

impl OutputSink for Collect {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        self.packets
            .push(packet_slot.take().expect("poll_push without a packet"));
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

impl Collect {
    pub(crate) fn into_frames(self) -> Vec<Frame> {
        self.packets
            .into_iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some(f),
                _ => None,
            })
            .collect()
    }

    pub(crate) fn frames(&self) -> Vec<&Frame> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some(f),
                _ => None,
            })
            .collect()
    }

    pub(crate) fn caps_changes(&self) -> Vec<&Caps> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::CapsChanged(c) => Some(c),
                _ => None,
            })
            .collect()
    }
}

/// A system-memory frame carrying `bytes`, the domain both the encoded clip and
/// a decoded NV12 picture travel in.
pub(crate) fn system_frame(bytes: &[u8]) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.to_vec().into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    }
}

pub(crate) fn in_caps(codec: VideoCodec) -> Caps {
    Caps::CompressedVideo {
        codec,
        width: Dim::Fixed(W),
        height: Dim::Fixed(H),
        framerate: Rate::Fixed(FRAMERATE_Q16),
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

pub(crate) fn raw_caps(format: RawVideoFormat) -> Caps {
    Caps::RawVideo {
        format,
        width: Dim::Fixed(W),
        height: Dim::Fixed(H),
        framerate: Rate::Fixed(FRAMERATE_Q16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

pub(crate) fn nv12_caps() -> Caps {
    raw_caps(RawVideoFormat::Nv12)
}

/// Whether this host can run the texture paths at all; `None` with the reason
/// when it cannot.
pub(crate) fn skip_reason() -> Option<&'static str> {
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

/// Feed a whole elementary stream through the element with the output domain
/// negotiated to `domain`, delivering `pinned_caps` first as the runner does with
/// the solved output caps. Returns the element (for its `gpu_context`), everything
/// the sink saw, and the result of the `DataFrame` push so a caps mismatch can be
/// asserted.
pub(crate) fn decode_stream(
    codec: VideoCodec,
    clip: &[u8],
    domain: MemoryDomainKind,
    pinned_caps: Option<Caps>,
) -> (VulkanVideoDec, Collect, Result<(), G2gError>) {
    let mut dec = VulkanVideoDec::new();
    dec.configure_allocation(&AllocationParams {
        size_bytes: 0,
        min_buffers: 1,
        align: 1,
        domain,
        accepts: DomainSet::only(domain),
        ..Default::default()
    });
    dec.configure_pipeline(&in_caps(codec))
        .expect("configure opens the decode device");
    let mut collect = Collect::default();
    if let Some(caps) = pinned_caps {
        block_on(dec.process(PipelinePacket::CapsChanged(caps), &mut collect))
            .expect("pre-fixed output caps accepted");
    }
    let decoded =
        block_on(dec.process(PipelinePacket::DataFrame(system_frame(clip)), &mut collect));
    if decoded.is_ok() {
        block_on(dec.process(PipelinePacket::Eos, &mut collect)).expect("eos drains the reorder");
    }
    (dec, collect, decoded)
}

/// The H.264 fixture clip through [`decode_stream`], for the tests that only want
/// its frames.
pub(crate) fn decode(
    domain: MemoryDomainKind,
    pinned_caps: Option<Caps>,
) -> (VulkanVideoDec, Vec<Frame>) {
    let (dec, collect, decoded) = decode_stream(VideoCodec::H264, H264_CLIP, domain, pinned_caps);
    decoded.expect("decode elementary stream");
    (dec, collect.into_frames())
}

/// The system-memory NV12 decode every texture path has to match, one
/// tightly-packed frame per picture.
pub(crate) fn reference_nv12() -> Vec<Vec<u8>> {
    let (_, frames) = decode(MemoryDomainKind::System, None);
    assert_eq!(frames.len(), CLIP_FRAMES);
    frames
        .into_iter()
        .map(|f| f.domain.as_system_slice().expect("system frame").to_vec())
        .collect()
}

/// The system-memory frames of a successful decode, each tightly packed.
pub(crate) fn system_bytes(collect: &Collect) -> Vec<Vec<u8>> {
    collect
        .frames()
        .iter()
        .map(|f| f.domain.as_system_slice().expect("system frame").to_vec())
        .collect()
}

/// wgpu's required `bytes_per_row` alignment for texture -> buffer copies.
const COPY_ROW_ALIGN: usize = 256;

/// Read one plane of a two-plane texture back as tightly packed bytes, taking the
/// plane's texel size from the texture's own format (1 / 2 bytes for NV12, 2 / 4
/// for P010).
pub(crate) fn read_plane(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    aspect: wgpu::TextureAspect,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let bytes_per_texel = texture
        .format()
        .block_copy_size(Some(aspect))
        .expect("a two-plane format's plane aspect has a texel size");
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

/// Both planes of a two-plane texture, concatenated in the system byte layout the
/// element's system path emits (luma then interleaved chroma).
pub(crate) fn read_two_plane(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
) -> Vec<u8> {
    let mut bytes = read_plane(
        device,
        queue,
        texture,
        wgpu::TextureAspect::Plane0,
        texture.width(),
        texture.height(),
    );
    bytes.extend(read_plane(
        device,
        queue,
        texture,
        wgpu::TextureAspect::Plane1,
        texture.width() / 2,
        texture.height() / 2,
    ));
    bytes
}

/// The two-plane texture owner behind a `WgpuTexture` frame.
pub(crate) fn two_plane_frame(frame: &Frame) -> &WgpuNv12Texture {
    let MemoryDomain::WgpuTexture(owned) = &frame.domain else {
        panic!("expected a WgpuTexture frame, got {:?}", frame.domain);
    };
    owned
        .keep_alive()
        .as_any()
        .downcast_ref::<WgpuNv12Texture>()
        .expect("a two-plane texture frame is owned by WgpuNv12Texture")
}
