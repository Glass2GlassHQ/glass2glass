//! M764: `VulkanVideoDec` rebuilds the session on a same-geometry parameter-set
//! content change.
//!
//! A mid-stream keyframe can carry new SPS/PPS at the *same* dimensions (a new
//! profile / entropy coding / ref config, e.g. an encoder settings change). The
//! geometry-keyed reconfig would keep the stale session and mis-decode. This
//! drives the element with 6 constrained-baseline (CAVLC) frames followed by 6
//! high-profile (CABAC) frames, both 640x480, and asserts the second segment
//! decodes bit-identically to a fresh element fed only that segment, with no
//! spurious `CapsChanged` (the output caps never change).
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

// 6 frames constrained-baseline + 6 frames high profile, both 640x480,
// concatenated Annex-B (each segment opens with an IDR carrying its SPS/PPS).
const CLIP: &[u8] = include_bytes!("fixtures/h264_reconfig_profile_640x480.h264");

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

/// Decode a sequence of AUs through a fresh element, returning the emitted NV12
/// frame buffers in order.
fn decode_frames(aus: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let mut dec = VulkanVideoDec::new();
    let in_caps = Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30 << 16),
    };
    dec.configure_pipeline(&in_caps)
        .expect("configure opens the decode device");
    let mut sink = RecordingSink::default();
    for (i, au) in aus.iter().enumerate() {
        block_on(dec.process(
            PipelinePacket::DataFrame(au_frame(au.clone(), i as u64)),
            &mut sink,
        ))
        .expect("decode access unit");
    }
    block_on(dec.process(PipelinePacket::Eos, &mut sink)).expect("flush at eos");
    sink.packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => f.domain.as_system_slice().map(|s| s.to_vec()),
            _ => None,
        })
        .collect()
}

#[test]
fn element_rebuilds_on_same_geometry_parameter_change() {
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

    let aus = split_access_units(CLIP);
    assert_eq!(aus.len(), 12, "6 baseline + 6 high-profile frames");

    // Continuous run across the parameter-set switch.
    let mut dec = VulkanVideoDec::new();
    let in_caps = Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30 << 16),
    };
    dec.configure_pipeline(&in_caps)
        .expect("configure opens the decode device");
    let mut sink = RecordingSink::default();
    for (i, au) in aus.iter().enumerate() {
        block_on(dec.process(
            PipelinePacket::DataFrame(au_frame(au.clone(), i as u64)),
            &mut sink,
        ))
        .expect("decode access unit");
    }
    block_on(dec.process(PipelinePacket::Eos, &mut sink)).expect("flush at eos");

    // The geometry never changes, so exactly one CapsChanged: the rebuild must
    // not re-negotiate identical caps.
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
    assert_eq!(caps, [(640, 480)], "one CapsChanged, constant geometry");

    let frames: Vec<Vec<u8>> = sink
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => f.domain.as_system_slice().map(|s| s.to_vec()),
            _ => None,
        })
        .collect();
    assert_eq!(frames.len(), 12, "one NV12 frame per coded picture");

    // The second segment must decode exactly as a fresh element decoding it
    // alone: a kept stale (CAVLC baseline) session would mis-decode the CABAC
    // high-profile slices.
    let fresh = decode_frames(&aus[6..]);
    assert_eq!(fresh.len(), 6, "second segment alone decodes 6 frames");
    for (i, (cont, alone)) in frames[6..].iter().zip(&fresh).enumerate() {
        assert_eq!(
            cont,
            alone,
            "frame {} after the parameter switch must be bit-exact vs a fresh decode",
            i + 6
        );
    }

    // Both segments carry real content.
    for (i, f) in frames.iter().enumerate() {
        let luma = &f[..640 * 480];
        let min = *luma.iter().min().unwrap();
        let max = *luma.iter().max().unwrap();
        assert!(max > min, "frame {i} luma is uniform; no real content");
    }
    eprintln!("VulkanVideoDec rebuilt on same-geometry SPS/PPS change: 12 frames, bit-exact tail");
}
