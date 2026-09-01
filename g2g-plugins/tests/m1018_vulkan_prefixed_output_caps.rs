//! M1018: `VulkanVideoDec` accepts the output caps the runner solved for it.
//!
//! An interior element is handed its own solved output caps as an incoming
//! `CapsChanged` ahead of the first frame, so the sink sees them before any data.
//! The decoder used to reject anything but its compressed input caps there, which
//! failed every `gst-launch` line reaching it (the geometry a launch line starts
//! from is a placeholder, so the real caps always arrive mid-stream). It must
//! forward them instead, and not repeat them when the first picture is emitted.
#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::block_on;
use g2g_core::{
    AsyncElement, Caps, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket,
    PushOutcome, Rate, RawVideoFormat, VideoCodec,
};
use g2g_plugins::vulkanvideo::{open_h264_decode_device, VulkanVideoDec, VulkanVideoError};

const CLIP: &[u8] = include_bytes!("fixtures/h264_640x480.h264");

/// The clip's geometry and rate, which the solved output caps carry.
const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;
const FRAMERATE: u32 = 30 << 16;

#[derive(Default)]
struct RecordingSink {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for RecordingSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        self.packets.push(packet);
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

impl RecordingSink {
    fn caps_changes(&self) -> Vec<&Caps> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::CapsChanged(c) => Some(c),
                _ => None,
            })
            .collect()
    }
}

/// The first access unit of the clip (SPS + PPS + IDR), enough to build the
/// decode session and emit a picture.
fn first_access_unit() -> Vec<u8> {
    let starts: Vec<usize> = (0..CLIP.len().saturating_sub(3))
        .filter(|&i| CLIP[i..i + 4] == [0, 0, 0, 1])
        .collect();
    let mut au = Vec::new();
    for (k, &begin) in starts.iter().enumerate() {
        let end = starts.get(k + 1).copied().unwrap_or(CLIP.len());
        au.extend_from_slice(&CLIP[begin..end]);
        let nal_type = CLIP[begin + 4] & 0x1F;
        if nal_type == 1 || nal_type == 5 {
            break;
        }
    }
    au
}

fn nv12_output_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Fixed(FRAMERATE),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

/// A decoder configured for the clip, or `None` on a host without Vulkan H.264
/// decode.
fn configured_decoder() -> Option<VulkanVideoDec> {
    match block_on(open_h264_decode_device()) {
        Ok(_) => {}
        Err(VulkanVideoError::NoVulkanAdapter) => {
            eprintln!("skipping: no Vulkan adapter");
            return None;
        }
        Err(VulkanVideoError::ExtensionUnsupported) | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skipping: GPU has no Vulkan H.264 decode support");
            return None;
        }
        Err(e) => panic!("probe failed: {e:?}"),
    }
    let mut dec = VulkanVideoDec::new();
    dec.configure_pipeline(&Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Fixed(FRAMERATE),
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    })
    .expect("configure opens the decode device");
    Some(dec)
}

// One test per file: two decode devices opened concurrently crash the driver on
// this host, and cargo runs test functions in parallel.
#[test]
fn the_solved_output_caps_are_forwarded_once() {
    let Some(mut dec) = configured_decoder() else {
        return;
    };
    let mut sink = RecordingSink::default();
    block_on(dec.process(PipelinePacket::CapsChanged(nv12_output_caps()), &mut sink))
        .expect("the decoder accepts the output caps the runner solved for it");
    assert_eq!(
        sink.caps_changes(),
        [&nv12_output_caps()],
        "the solved output caps reach the sink before any frame"
    );

    let frame = Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(
            first_access_unit().into_boxed_slice(),
        )),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    };
    block_on(dec.process(PipelinePacket::DataFrame(frame), &mut sink))
        .expect("decode the keyframe");
    block_on(dec.process(PipelinePacket::Eos, &mut sink)).expect("flush at eos");
    assert_eq!(
        sink.caps_changes().len(),
        1,
        "the decoded picture does not repeat caps the sink already has"
    );

    let rgb = Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Fixed(FRAMERATE),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    };
    // The system path emits NV12; RGBA belongs to the GPU-texture path this
    // decoder was not steered onto.
    assert!(
        block_on(dec.process(PipelinePacket::CapsChanged(rgb), &mut sink)).is_err(),
        "a format this decoder cannot emit is still rejected"
    );
}
