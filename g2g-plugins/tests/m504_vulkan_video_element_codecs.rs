//! M504: the `VulkanVideoDec` pipeline element decodes H.265 and AV1, not just
//! H.264.
//!
//! M493 proved the element for H.264; M517 folded all three decoders onto a
//! shared `DpbCore` and generalized the element over a codec enum (one element,
//! codec chosen from the sink caps at `configure_pipeline`). This drives the
//! element for H.265 and AV1 the way a pipeline does -- compressed access units
//! in, `RawVideo{Nv12}` system frames out -- and asserts one NV12 frame per
//! coded picture with real content plus the leading output `CapsChanged`. Runs
//! on the RTX 3060; skips per codec if the GPU lacks that decode profile.
#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use std::future::Future;
use std::pin::Pin;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::block_on;
use g2g_core::{
    AsyncElement, Caps, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket,
    PushOutcome, Rate, RawVideoFormat, VideoCodec,
};
use g2g_plugins::vulkanvideo::{
    open_av1_decode_device, open_h265_decode_device, VulkanVideoDec, VulkanVideoError,
};

const H265_CLIP: &[u8] = include_bytes!("fixtures/h265_640x480.h265");
const AV1_CLIP: &[u8] = include_bytes!("fixtures/av1_640x480.obu");

#[derive(Default)]
struct RecordingSink {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for RecordingSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            self.packets.push(packet);
            Ok(PushOutcome::Accepted)
        })
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

/// Configure the element for `codec`, feed the whole elementary stream through
/// one `process` call (the element's `decode_all` splits it into pictures), and
/// assert 10 NV12 640x480 frames with real content plus one leading CapsChanged.
fn drive_codec(codec: VideoCodec, clip: &[u8]) {
    let mut dec = VulkanVideoDec::new();
    let in_caps = Caps::CompressedVideo {
        codec,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30 << 16),
    };
    dec.configure_pipeline(&in_caps)
        .expect("configure opens the decode device");

    let mut sink = RecordingSink::default();
    block_on(dec.process(PipelinePacket::DataFrame(au_frame(clip)), &mut sink))
        .expect("decode elementary stream");
    // System path is pipelined (output lags submission); flush the in-flight tail.
    block_on(dec.process(PipelinePacket::Eos, &mut sink)).expect("flush at eos");

    let caps_changes: Vec<&Caps> = sink
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::CapsChanged(c) => Some(c),
            _ => None,
        })
        .collect();
    assert_eq!(caps_changes.len(), 1, "{codec:?}: one output CapsChanged");
    assert_eq!(
        caps_changes[0],
        &Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Fixed(30 << 16),
        },
        "{codec:?}: emits NV12 640x480 at the input framerate"
    );

    let frames: Vec<&Frame> = sink
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f),
            _ => None,
        })
        .collect();
    assert_eq!(
        frames.len(),
        10,
        "{codec:?}: one NV12 frame per coded picture"
    );

    let nv12_len = 640 * 480 * 3 / 2;
    for (i, f) in frames.iter().enumerate() {
        let MemoryDomain::System(slice) = &f.domain else {
            panic!("{codec:?} frame {i} is not system memory");
        };
        let bytes = slice.as_slice();
        assert_eq!(
            bytes.len(),
            nv12_len,
            "{codec:?} frame {i} is a full NV12 buffer"
        );
        let luma = &bytes[..640 * 480];
        let min = *luma.iter().min().unwrap();
        let max = *luma.iter().max().unwrap();
        assert!(
            max > min,
            "{codec:?} frame {i} luma is uniform ({min}=={max}); no real content"
        );
    }
    eprintln!(
        "VulkanVideoDec ({codec:?}) emitted {} NV12 frames (640x480)",
        frames.len()
    );
}

/// One test drives both codecs sequentially: creating Vulkan devices in parallel
/// SIGSEGVs on this driver (see `vulkan_thread_teardown`), and the codebase runs
/// one device-creating test per file, so H.265 and AV1 share this single test
/// rather than racing as two.
#[test]
fn element_decodes_h265_and_av1_to_nv12_frames() {
    match block_on(open_h265_decode_device()) {
        Ok(_) => drive_codec(VideoCodec::H265, H265_CLIP),
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skipping H.265: GPU has no Vulkan H.265 decode support");
        }
        Err(e) => panic!("H.265 probe failed: {e:?}"),
    }

    match block_on(open_av1_decode_device()) {
        Ok(_) => drive_codec(VideoCodec::Av1, AV1_CLIP),
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skipping AV1: GPU has no Vulkan AV1 decode support");
        }
        Err(e) => panic!("AV1 probe failed: {e:?}"),
    }
}
