//! M493: the `VulkanVideoDec` pipeline element on real hardware.
//!
//! Wraps the M492 `H264DpbDecoder` as an `AsyncElement`: H.264 Annex-B in,
//! `RawVideo{Nv12}` system-memory frames out, decoding on the same Vulkan device
//! wgpu runs. This drives the element the way a pipeline does -- one access unit
//! per `process` call, so the DPB reference state must carry across calls -- and
//! asserts it emits one NV12 frame per coded picture with real content, plus the
//! output `CapsChanged`. Runs on the RTX 3060; skips with no adapter / support.
//!
//! The output caps carry the stream's own colour description, read from the SPS
//! at the first keyframe. This fixture writes no VUI colour block, so its CICP
//! codepoints are all "unspecified" and `video_full_range_flag` takes its coded
//! default of 0: the caps come out `range: Limited` with the other three fields
//! `Unknown`. The colour never changes mid-stream here, so the ten pictures
//! still produce exactly one `CapsChanged`.
#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::block_on;
use g2g_core::{
    AsyncElement, Caps, ColorRange, Colorimetry, Dim, FrameTiming, G2gError, MatrixCoefficients,
    MemoryDomain, OutputSink, PipelinePacket, PushOutcome, Rate, RawVideoFormat, VideoCodec,
};
use g2g_plugins::vulkanvideo::{
    extract_h264_parameter_sets, open_h264_decode_device, VulkanVideoDec, VulkanVideoError,
};

const CLIP: &[u8] = include_bytes!("fixtures/h264_640x480.h264");

/// The CICP codepoint every colour field spells "unspecified".
const CICP_UNSPECIFIED: u8 = 2;

/// What the fixture's SPS resolves to: no colour block, so only the range is
/// concrete (`video_full_range_flag` defaults to 0).
const EXPECTED_COLORIMETRY: Colorimetry = Colorimetry {
    range: ColorRange::Limited,
    matrix: MatrixCoefficients::Unknown,
    transfer: g2g_core::TransferCharacteristics::Unknown,
    primaries: g2g_core::ColorPrimaries::Unknown,
};

/// Read the fixture's declared colour codepoints, so `EXPECTED_COLORIMETRY`
/// stays tied to the bitstream rather than to a remembered value.
fn assert_fixture_declares_no_colour_description() {
    let ps = extract_h264_parameter_sets(CLIP).expect("the fixture carries an SPS");
    assert_eq!(
        (
            ps.sps.color_primaries,
            ps.sps.transfer_characteristics,
            ps.sps.matrix_coefficients,
            ps.sps.video_full_range_flag,
        ),
        (CICP_UNSPECIFIED, CICP_UNSPECIFIED, CICP_UNSPECIFIED, false),
        "fixture colour description changed; update EXPECTED_COLORIMETRY"
    );
}

/// A sink that records every packet the element pushes.
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
        core::task::Poll::Ready({
            self.packets.push(packet);
            Ok(PushOutcome::Accepted)
        })
    }
}

/// Split an Annex-B elementary stream into per-picture access units: each VCL
/// NAL (type 1/5) closes an AU, carrying any preceding SPS/PPS/SEI with it (the
/// fixture is single-slice, so one VCL == one picture).
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
        // Trailing non-VCL bytes (none for this clip); attach to the last AU.
        if let Some(last) = units.last_mut() {
            last.extend_from_slice(&cur);
        }
    }
    units
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
fn element_decodes_stream_to_nv12_frames() {
    // Skip cleanly on a host with no Vulkan H.264 decode (the element opens its
    // own device internally; this just decides whether to run).
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
    assert_fixture_declares_no_colour_description();

    let mut dec = VulkanVideoDec::new();
    let in_caps = Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30 << 16),
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    };
    dec.configure_pipeline(&in_caps)
        .expect("configure opens the decode device");

    let aus = split_access_units(CLIP);
    assert_eq!(aus.len(), 10, "clip splits into 10 access units");

    let mut sink = RecordingSink::default();
    for (i, au) in aus.into_iter().enumerate() {
        block_on(dec.process(PipelinePacket::DataFrame(au_frame(au, i as u64)), &mut sink))
            .expect("decode access unit");
    }
    // The system path is pipelined (decode output lags submission), so end of
    // stream must flush the in-flight tail before the frame count is complete.
    block_on(dec.process(PipelinePacket::Eos, &mut sink)).expect("flush at eos");

    // One NV12 DataFrame per coded picture, plus exactly one leading CapsChanged:
    // the colour description is constant across the clip, so the nine pictures
    // after the first re-emit nothing.
    let caps_changes: Vec<&Caps> = sink
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::CapsChanged(c) => Some(c),
            _ => None,
        })
        .collect();
    assert_eq!(caps_changes.len(), 1, "one output CapsChanged");
    assert_eq!(
        caps_changes[0],
        &Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Fixed(30 << 16),
            interlace: g2g_core::Interlace::Any,
            colorimetry: EXPECTED_COLORIMETRY
        },
        "emits NV12 640x480 at the input framerate, tagged with the SPS colour"
    );

    let frames: Vec<&Frame> = sink
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f),
            _ => None,
        })
        .collect();
    assert_eq!(frames.len(), 10, "one NV12 frame per coded picture");

    let nv12_len = 640 * 480 * 3 / 2;
    for (i, f) in frames.iter().enumerate() {
        let Some(slice) = f.domain.as_system_slice() else {
            panic!("frame {i} is not system memory");
        };
        let bytes = slice;
        assert_eq!(bytes.len(), nv12_len, "frame {i} is a full NV12 buffer");
        // The luma plane (first 640*480 bytes) must be real, varied content.
        let luma = &bytes[..640 * 480];
        let min = *luma.iter().min().unwrap();
        let max = *luma.iter().max().unwrap();
        assert!(
            min <= 20 && max >= 200,
            "frame {i} luma range {min}..={max} not a real picture"
        );
    }
    eprintln!(
        "VulkanVideoDec emitted {} NV12 frames (640x480)",
        frames.len()
    );
}
