//! M535: the zero-copy wedge end to end for **all three codecs** --
//! `VulkanVideoDec` -> `WgpuSink` for H.264, H.265, and AV1.
//!
//! M495 proved the GPU-resident decode->present wedge (decode straight into an
//! RGBA `wgpu::Texture`, present with no GPU->CPU readback) for H.264 only. The
//! H.265 / AV1 GPU-texture paths (`create_h265/av1_dpb_decoder_gpu` +
//! `decode_all_to_textures`) exist since M517 but had never executed: the element
//! decode-to-NV12 path is covered (M504) and auto-plug selection is covered
//! (M496), but the wedge payload -- decoded pictures landing in a wgpu texture on
//! the decode device, presented copy-free -- was validated for H.264 alone.
//!
//! This drives each codec's whole elementary stream through `VulkanVideoDec` with
//! the output domain negotiated to `WgpuTexture` (as a `WgpuSink` downstream
//! would), asserts every emitted frame is GPU-resident, presents each through a
//! `WgpuSink` sharing the decoder's Vulkan device, and reads the offscreen target
//! back only to confirm real content landed. Runs on the RTX 3060; skips per
//! codec if the GPU lacks that decode profile or a compute queue.
#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video",
    feature = "wgpu-sink"
))]

use std::future::Future;
use std::pin::Pin;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::runtime::block_on;
use g2g_core::{
    AllocationParams, AsyncElement, Caps, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PipelinePacket, PushOutcome, Rate, RawVideoFormat, VideoCodec,
};
use g2g_plugins::vulkanvideo::{
    open_av1_decode_device, open_h264_decode_device, open_h265_decode_device, VulkanVideoDec,
    VulkanVideoError,
};
use g2g_plugins::wgpusink::WgpuSink;

const H264_CLIP: &[u8] = include_bytes!("fixtures/h264_640x480.h264");
const H265_CLIP: &[u8] = include_bytes!("fixtures/h265_640x480.h265");
const AV1_CLIP: &[u8] = include_bytes!("fixtures/av1_640x480.obu");

const W: u32 = 640;
const H: u32 = 480;

#[derive(Default)]
struct Collect {
    frames: Vec<PipelinePacket>,
}
impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            self.frames.push(packet);
            Ok(PushOutcome::Accepted)
        })
    }
}

struct NullSink;
impl OutputSink for NullSink {
    fn push<'a>(
        &'a mut self,
        _packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async { Ok(PushOutcome::Accepted) })
    }
}

fn au_frame(bytes: &[u8]) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.to_vec().into_boxed_slice())),
        timing: FrameTiming {
            pts_ns: 0,
            ..Default::default()
        },
        sequence: 0,
        meta: Default::default(),
    }
}

/// Decode the whole `clip` for `codec` straight into GPU textures and present each
/// through a `WgpuSink` on the same device with no readback; assert 10 real frames.
fn drive_wedge(codec: VideoCodec, clip: &[u8]) {
    let mut dec = VulkanVideoDec::new();
    // Negotiate the zero-copy WgpuTexture domain, as a WgpuSink downstream would.
    dec.configure_allocation(&AllocationParams {
        size_bytes: 0,
        min_buffers: 1,
        align: 1,
        domain: MemoryDomainKind::WgpuTexture,
        accepts: DomainSet::only(MemoryDomainKind::WgpuTexture),
    });
    let in_caps = Caps::CompressedVideo {
        codec,
        width: Dim::Fixed(W),
        height: Dim::Fixed(H),
        framerate: Rate::Fixed(30 << 16),
    };
    dec.configure_pipeline(&in_caps)
        .expect("configure opens the decode device");

    // A WgpuSink sharing the decoder's Vulkan device: the WgpuTexture handoff is
    // copy-free (same device, no import).
    let ctx = dec.gpu_context().expect("device open after configure");
    let mut sink = WgpuSink::offscreen(ctx, W, H);
    let rgba_caps = Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(W),
        height: Dim::Fixed(H),
        framerate: Rate::Fixed(30 << 16),
    };
    sink.configure_pipeline(&rgba_caps).expect("sink configure");

    // The element's decode_all splits the whole stream into pictures, so feed it
    // in one DataFrame (as M504 does), then present each emitted texture frame.
    let mut collect = Collect::default();
    block_on(dec.process(PipelinePacket::DataFrame(au_frame(clip)), &mut collect))
        .expect("decode elementary stream to textures");

    let mut presented = 0usize;
    for pkt in collect.frames {
        let is_frame = matches!(pkt, PipelinePacket::DataFrame(_));
        if let PipelinePacket::DataFrame(ref f) = pkt {
            assert!(
                matches!(f.domain, MemoryDomain::WgpuTexture(_)),
                "{codec:?}: decoder must emit a GPU-resident WgpuTexture frame"
            );
        }
        block_on(sink.process(pkt, &mut NullSink)).expect("sink present");
        if is_frame {
            let rgba = sink.read_target().expect("read offscreen target");
            let min = *rgba.iter().min().unwrap();
            let max = *rgba.iter().max().unwrap();
            assert!(
                min <= 20 && max >= 200,
                "{codec:?} frame {presented} target {min}..={max} not real"
            );
            presented += 1;
        }
    }
    assert_eq!(
        presented, 10,
        "{codec:?}: all 10 decoded textures presented by WgpuSink"
    );
    eprintln!("VulkanVideoDec ({codec:?}) -> WgpuSink: presented {presented} GPU-resident frames (no readback)");
}

/// Whether the decode device (per codec) opens on this host; skip cleanly if the
/// GPU lacks that profile / a compute queue, panic on an unexpected error.
fn device_available(open: Result<impl Sized, VulkanVideoError>, codec: &str) -> bool {
    match open {
        Ok(_) => true,
        Err(VulkanVideoError::NoVulkanAdapter) => {
            eprintln!("skipping {codec}: no Vulkan adapter");
            false
        }
        Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue)
        | Err(VulkanVideoError::NoComputeQueue) => {
            eprintln!("skipping {codec}: GPU has no Vulkan {codec} decode / compute support");
            false
        }
        Err(e) => panic!("{codec} probe failed: {e:?}"),
    }
}

#[test]
fn h264_decode_to_texture_present_zero_copy() {
    if device_available(block_on(open_h264_decode_device()), "H.264") {
        drive_wedge(VideoCodec::H264, H264_CLIP);
    }
}

#[test]
fn h265_decode_to_texture_present_zero_copy() {
    if device_available(block_on(open_h265_decode_device()), "H.265") {
        drive_wedge(VideoCodec::H265, H265_CLIP);
    }
}

#[test]
fn av1_decode_to_texture_present_zero_copy() {
    if device_available(block_on(open_av1_decode_device()), "AV1") {
        drive_wedge(VideoCodec::Av1, AV1_CLIP);
    }
}
