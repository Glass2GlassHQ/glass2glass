//! M495: the zero-copy wedge end to end -- `VulkanVideoDec` -> `WgpuSink`.
//!
//! Negotiates the decoder's output domain to `WgpuTexture`, so it decodes each
//! H.264 picture straight into an RGBA `wgpu::Texture` on its Vulkan device. A
//! `WgpuSink` built from that same device (via `VulkanVideoDec::gpu_context`)
//! then blits each frame onto an offscreen target with NO GPU->CPU readback --
//! the frame never leaves the GPU between decode and present. The sink's target
//! is read back only to verify real content landed. Runs on the RTX 3060; skips
//! with no adapter / no compute queue.
#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video",
    feature = "wgpu-sink"
))]

use std::future::Future;
use std::pin::Pin;

use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::runtime::block_on;
use g2g_core::{
    AllocationParams, AsyncElement, Caps, Dim, G2gError, MemoryDomain, OutputSink, PipelinePacket,
    PushOutcome, Rate, RawVideoFormat, VideoCodec,
};
use g2g_plugins::vulkanvideo::{open_h264_decode_device, VulkanVideoDec, VulkanVideoError};
use g2g_plugins::wgpusink::WgpuSink;

const CLIP: &[u8] = include_bytes!("fixtures/h264_640x480.h264");

/// Collects the frames the decoder pushes so they can be fed to the sink.
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

fn split_access_units(stream: &[u8]) -> Vec<Vec<u8>> {
    let mut units = Vec::new();
    let mut cur = Vec::new();
    let mut i = 0;
    while i + 3 <= stream.len() {
        let sc = if stream[i] == 0 && stream[i + 1] == 0 && stream[i + 2] == 1 {
            Some(3)
        } else if i + 4 <= stream.len() && stream[i..i + 4] == [0, 0, 0, 1] {
            Some(4)
        } else {
            None
        };
        let Some(sclen) = sc else {
            i += 1;
            continue;
        };
        let payload = i + sclen;
        // Find the next start code to bound this NAL.
        let mut j = payload;
        let end = loop {
            if j + 3 > stream.len() {
                break stream.len();
            }
            if stream[j] == 0 && stream[j + 1] == 0 && stream[j + 2] == 1 {
                break j;
            }
            j += 1;
        };
        let nal = &stream[payload..end];
        cur.extend_from_slice(&[0, 0, 0, 1]);
        cur.extend_from_slice(nal);
        let t = nal.first().map(|b| b & 0x1F).unwrap_or(0);
        if t == 1 || t == 5 {
            units.push(std::mem::take(&mut cur));
        }
        i = end;
    }
    if !cur.is_empty() {
        if let Some(last) = units.last_mut() {
            last.extend_from_slice(&cur);
        }
    }
    units
}

#[test]
fn decode_to_texture_then_present_zero_copy() {
    match block_on(open_h264_decode_device()) {
        Ok(_) => {}
        Err(VulkanVideoError::NoVulkanAdapter) => {
            eprintln!("skipping: no Vulkan adapter");
            return;
        }
        Err(VulkanVideoError::ExtensionUnsupported) | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skipping: GPU has no Vulkan H.264 decode support");
            return;
        }
        Err(e) => panic!("probe failed: {e:?}"),
    }

    let mut dec = VulkanVideoDec::new();
    // Negotiate the zero-copy WgpuTexture domain (as a WgpuSink downstream would).
    dec.configure_allocation(&AllocationParams {
        size_bytes: 0,
        min_buffers: 1,
        align: 1,
        domain: MemoryDomainKind::WgpuTexture,
        accepts: DomainSet::only(MemoryDomainKind::WgpuTexture),
    });
    let in_caps = Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30 << 16),
    };
    dec.configure_pipeline(&in_caps)
        .expect("configure opens the decode device");

    // A WgpuSink sharing the decoder's Vulkan device -> the WgpuTexture handoff
    // is copy-free.
    let ctx = dec.gpu_context().expect("device open after configure");
    let mut sink = WgpuSink::offscreen(ctx, 640, 480);
    let rgba_caps = Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30 << 16),
    };
    sink.configure_pipeline(&rgba_caps).expect("sink configure");

    let aus = split_access_units(CLIP);
    assert_eq!(aus.len(), 10, "clip splits into 10 access units");

    let mut presented = 0usize;
    for (i, au) in aus.into_iter().enumerate() {
        let mut collect = Collect::default();
        let frame = au_frame(au, i as u64);
        block_on(dec.process(PipelinePacket::DataFrame(frame), &mut collect)).expect("decode AU");

        for pkt in collect.frames {
            // Only DataFrames carry a WgpuTexture; forward everything to the sink.
            let is_frame = matches!(pkt, PipelinePacket::DataFrame(_));
            if let PipelinePacket::DataFrame(ref f) = pkt {
                assert!(
                    matches!(f.domain, MemoryDomain::WgpuTexture(_)),
                    "decoder must emit a GPU-resident WgpuTexture frame"
                );
            }
            block_on(sink.process(pkt, &mut NullSink)).expect("sink present");
            if is_frame {
                // Read the presented target back and confirm real content landed.
                let rgba = sink.read_target().expect("read offscreen target");
                let min = *rgba.iter().min().unwrap();
                let max = *rgba.iter().max().unwrap();
                assert!(
                    min <= 20 && max >= 200,
                    "frame {presented} target {min}..={max} not real"
                );
                presented += 1;
            }
        }
    }
    assert_eq!(
        presented, 10,
        "all 10 decoded textures were presented by WgpuSink"
    );
    eprintln!(
        "VulkanVideoDec -> WgpuSink: presented {presented} GPU-resident frames (no readback)"
    );
}

// Local Frame constructor (the crate's `Frame` type, system-memory input AU).
fn au_frame(bytes: Vec<u8>, seq: u64) -> g2g_core::frame::Frame {
    use g2g_core::memory::SystemSlice;
    use g2g_core::FrameTiming;
    g2g_core::frame::Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming {
            pts_ns: seq * 33_000_000,
            ..Default::default()
        },
        sequence: seq,
        meta: Default::default(),
    }
}
