//! M519: `VulkanVideoDec` reconfigures mid-stream on a resolution change.
//!
//! The decode session + DPB are built lazily from the first keyframe's parameter
//! sets. A real stream can change resolution mid-stream (adaptive streaming, a
//! camera reconfig), which arrives as a later keyframe carrying a new SPS with
//! different geometry. This drives the element with a stream that switches
//! 640x480 -> 320x240 at frame 6 and asserts it rebuilds the session/DPB for the
//! new resolution: the right dimensions per segment, a fresh `CapsChanged` at the
//! switch, and real content on both sides (a decoder that ignored the reconfig
//! would emit garbage or the wrong size for the second segment).
//!
//! Runs on the RTX 3060; skips with no Vulkan H.264 decode support.
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
use g2g_plugins::vulkanvideo::{open_h264_decode_device, VulkanVideoDec, VulkanVideoError};

// 6 frames at 640x480 (IDR + P...), then 6 at 320x240, concatenated Annex-B.
const CLIP: &[u8] = include_bytes!("fixtures/h264_reconfig_640x480_to_320x240.h264");

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

/// Byte offsets of each NAL payload (just past its start code).
fn start_code_offsets(data: &[u8]) -> Vec<usize> {
    let mut offs = Vec::new();
    let mut i = 0;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 {
            if data[i + 2] == 1 {
                offs.push(i + 3);
                i += 3;
                continue;
            }
            if i + 4 <= data.len() && data[i + 2] == 0 && data[i + 3] == 1 {
                offs.push(i + 4);
                i += 4;
                continue;
            }
        }
        i += 1;
    }
    offs
}

/// Split an H.264 Annex-B stream into per-picture access units: each VCL NAL
/// (type 1/5) closes an AU, carrying preceding SPS/PPS/SEI (single-slice fixture).
fn split_access_units(stream: &[u8]) -> Vec<Vec<u8>> {
    let mut units = Vec::new();
    let mut cur = Vec::new();
    let starts = start_code_offsets(stream);
    for (k, &begin) in starts.iter().enumerate() {
        let end = starts.get(k + 1).copied().unwrap_or(stream.len());
        let nal = &stream[begin..end];
        cur.extend_from_slice(&[0, 0, 0, 1]);
        cur.extend_from_slice(nal);
        let nal_type = nal.first().map(|b| b & 0x1F).unwrap_or(0);
        if nal_type == 1 || nal_type == 5 {
            units.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        if let Some(last) = units.last_mut() {
            last.extend_from_slice(&cur);
        }
    }
    units
}

fn au_frame(bytes: Vec<u8>, seq: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming {
            pts_ns: seq * 33_000_000,
            ..Default::default()
        },
        sequence: seq,
        meta: Default::default(),
    }
}

#[test]
fn element_reconfigures_on_resolution_change() {
    match block_on(open_h264_decode_device()) {
        Ok(_) => {}
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skipping: GPU has no Vulkan H.264 decode support");
            return;
        }
        Err(e) => panic!("probe failed: {e:?}"),
    }

    let mut dec = VulkanVideoDec::new();
    // Configure with the leading resolution; the mid-stream switch is discovered
    // in-band from the second segment's keyframe SPS.
    let in_caps = Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30 << 16),
    };
    dec.configure_pipeline(&in_caps)
        .expect("configure opens the decode device");

    let aus = split_access_units(CLIP);
    assert_eq!(aus.len(), 12, "6 frames at 640x480 + 6 at 320x240");

    let mut sink = RecordingSink::default();
    for (i, au) in aus.into_iter().enumerate() {
        block_on(dec.process(PipelinePacket::DataFrame(au_frame(au, i as u64)), &mut sink))
            .expect("decode access unit");
    }
    // Pipelined system path: flush the in-flight tail of the final segment at eos
    // (the reconfig already flushed the first segment's tail on the rebuild).
    block_on(dec.process(PipelinePacket::Eos, &mut sink)).expect("flush at eos");

    // Collect emitted raw-video CapsChanged (the resolution timeline) and frames.
    let caps: Vec<(u32, u32)> = sink
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::CapsChanged(Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(w),
                height: Dim::Fixed(h),
                ..
            }) => Some((*w, *h)),
            _ => None,
        })
        .collect();
    assert_eq!(
        caps,
        [(640, 480), (320, 240)],
        "one CapsChanged per resolution, in order, so the reconfig re-negotiated caps"
    );

    // Every decoded NV12 frame, in order, with its buffer length implying its size.
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
        12,
        "one NV12 frame per coded picture across both segments"
    );

    let nv12_len = |w: usize, h: usize| w * h * 3 / 2;
    for (i, f) in frames.iter().enumerate() {
        let MemoryDomain::System(slice) = &f.domain else {
            panic!("frame {i} is not system memory");
        };
        let bytes = slice.as_slice();
        let (w, h) = if i < 6 { (640, 480) } else { (320, 240) };
        assert_eq!(
            bytes.len(),
            nv12_len(w, h),
            "frame {i} must be a full NV12 buffer for {w}x{h} (reconfig sized the output)"
        );
        // Real, non-uniform luma (a failed / ignored reconfig would be flat/garbage).
        let luma = &bytes[..w * h];
        let min = *luma.iter().min().unwrap();
        let max = *luma.iter().max().unwrap();
        assert!(
            max > min,
            "frame {i} luma is uniform ({min}=={max}); no real content"
        );
    }
    eprintln!(
        "VulkanVideoDec reconfigured 640x480 -> 320x240: {} frames",
        frames.len()
    );
}
