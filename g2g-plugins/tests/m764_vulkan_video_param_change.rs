//! M764: `VulkanVideoDec` rebuilds the session on a same-geometry parameter-set
//! content change.
//!
//! A mid-stream keyframe can carry new SPS/PPS at the *same* dimensions (a new
//! profile / entropy coding / ref config, e.g. an encoder settings change). The
//! geometry-keyed reconfig would keep the stale session and mis-decode. This
//! drives the element with 6 constrained-baseline (CAVLC) frames followed by 6
//! high-profile (CABAC) frames, both 640x480, and asserts the second segment
//! decodes bit-identically to a fresh element fed only that segment.
//!
//! The two SPSs also differ in colour: the baseline one writes no VUI colour
//! block (all CICP codepoints unspecified), the high-profile one declares
//! `matrix_coefficients` 5 (BT.470BG, which resolves to the BT.601 matrix).
//! Neither codes a full-range flag, so both come out `range: Limited`. The
//! element therefore emits exactly two `CapsChanged`, one per segment: the
//! geometry is constant, so the only thing that changes is the colorimetry, and
//! the five pictures after each switch re-emit nothing.
//!
//! Runs on the RTX 3060; skips with no Vulkan H.264 decode support.
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

// 6 frames constrained-baseline + 6 frames high profile, both 640x480,
// concatenated Annex-B (each segment opens with an IDR carrying its SPS/PPS).
const CLIP: &[u8] = include_bytes!("fixtures/h264_reconfig_profile_640x480.h264");

/// Access unit index each segment starts at.
const BASELINE_SEGMENT: usize = 0;
const HIGH_PROFILE_SEGMENT: usize = 6;

/// The CICP codepoint every colour field spells "unspecified".
const CICP_UNSPECIFIED: u8 = 2;
/// CICP `matrix_coefficients` for BT.470BG, what the high-profile SPS declares.
const CICP_MATRIX_BT470BG: u8 = 5;

/// What each segment's SPS resolves to. Only the range is concrete in the
/// baseline segment; the high-profile one adds the BT.601 matrix.
const BASELINE_COLORIMETRY: Colorimetry = Colorimetry {
    range: ColorRange::Limited,
    matrix: MatrixCoefficients::Unknown,
    transfer: g2g_core::TransferCharacteristics::Unknown,
    primaries: g2g_core::ColorPrimaries::Unknown,
};
const HIGH_PROFILE_COLORIMETRY: Colorimetry = Colorimetry {
    matrix: MatrixCoefficients::Bt601,
    ..BASELINE_COLORIMETRY
};

fn nv12_caps(colorimetry: Colorimetry) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry,
    }
}

/// Read the colour codepoints each segment's SPS declares, so the expected
/// colorimetry stays tied to the fixture rather than to a remembered value.
fn assert_fixture_colour_descriptions(aus: &[Vec<u8>]) {
    let colour = |au: &[u8]| {
        let ps = extract_h264_parameter_sets(au).expect("the segment opens with an SPS");
        (
            ps.sps.color_primaries,
            ps.sps.transfer_characteristics,
            ps.sps.matrix_coefficients,
            ps.sps.video_full_range_flag,
        )
    };
    assert_eq!(
        colour(&aus[BASELINE_SEGMENT]),
        (CICP_UNSPECIFIED, CICP_UNSPECIFIED, CICP_UNSPECIFIED, false),
        "baseline segment colour changed; update BASELINE_COLORIMETRY"
    );
    assert_eq!(
        colour(&aus[HIGH_PROFILE_SEGMENT]),
        (
            CICP_UNSPECIFIED,
            CICP_UNSPECIFIED,
            CICP_MATRIX_BT470BG,
            false
        ),
        "high-profile segment colour changed; update HIGH_PROFILE_COLORIMETRY"
    );
}

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
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
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
    assert_fixture_colour_descriptions(&aus);

    // Continuous run across the parameter-set switch.
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
    let mut sink = RecordingSink::default();
    for (i, au) in aus.iter().enumerate() {
        block_on(dec.process(
            PipelinePacket::DataFrame(au_frame(au.clone(), i as u64)),
            &mut sink,
        ))
        .expect("decode access unit");
    }
    block_on(dec.process(PipelinePacket::Eos, &mut sink)).expect("flush at eos");

    // One CapsChanged per segment and no more: the geometry never changes, so
    // the rebuild must re-negotiate nothing but the segment's own colour, and
    // the five pictures following each switch must re-emit nothing.
    let caps: Vec<&Caps> = sink
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::CapsChanged(c) => Some(c),
            _ => None,
        })
        .collect();
    assert_eq!(
        caps,
        [
            &nv12_caps(BASELINE_COLORIMETRY),
            &nv12_caps(HIGH_PROFILE_COLORIMETRY)
        ],
        "constant geometry, one CapsChanged per segment colour"
    );

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
