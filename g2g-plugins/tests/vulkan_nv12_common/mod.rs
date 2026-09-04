//! The `VulkanVideoDec` NV12 decode harness the M1157 tests share: the fixture
//! clip, its caps, the host-capability skip, and a decode run pinned to one
//! output domain.

use g2g_core::frame::Frame;
use g2g_core::memory::{DomainSet, MemoryDomainKind, SystemSlice};
use g2g_core::runtime::block_on;
use g2g_core::{
    AllocationParams, AsyncElement, Caps, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PipelinePacket, PushOutcome, Rate, RawVideoFormat, VideoCodec,
};
use g2g_plugins::vulkanvideo::{open_h264_decode_device, VulkanVideoDec, VulkanVideoError};

pub(crate) const H264_CLIP: &[u8] = include_bytes!("../fixtures/h264_640x480.h264");
pub(crate) const W: u32 = 640;
pub(crate) const H: u32 = 480;
pub(crate) const FRAMERATE_Q16: u32 = 30 << 16;
/// Frames in the fixture clip.
pub(crate) const CLIP_FRAMES: usize = 10;

#[derive(Default)]
pub(crate) struct Collect {
    pub(crate) frames: Vec<Frame>,
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

pub(crate) fn in_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(W),
        height: Dim::Fixed(H),
        framerate: Rate::Fixed(FRAMERATE_Q16),
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

pub(crate) fn nv12_caps() -> Caps {
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

/// Decode the clip with the output domain negotiated to `domain`, delivering
/// `pinned_caps` first as the runner does with the solved output caps.
pub(crate) fn decode(
    domain: MemoryDomainKind,
    pinned_caps: Option<Caps>,
) -> (VulkanVideoDec, Vec<Frame>) {
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
    block_on(dec.process(
        PipelinePacket::DataFrame(system_frame(H264_CLIP)),
        &mut collect,
    ))
    .expect("decode elementary stream");
    block_on(dec.process(PipelinePacket::Eos, &mut collect)).expect("eos drains the reorder");
    (dec, collect.frames)
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
