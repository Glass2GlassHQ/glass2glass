//! M1018: `VulkanVideoDec` accepts the output caps the runner solved for it.
//!
//! An interior element is handed its own solved output caps as an incoming
//! `CapsChanged` ahead of the first frame, so the sink sees them before any data.
//! The decoder used to reject anything but its compressed input caps there, which
//! failed every `gst-launch` line reaching it (the geometry a launch line starts
//! from is a placeholder, so the real caps always arrive mid-stream). It must
//! forward them instead.
//!
//! The solved caps a launch line starts from carry no colour information, while
//! the decoder reads the stream's own from the SPS at the first keyframe. This
//! fixture writes no VUI colour block, so its CICP codepoints are all
//! "unspecified" and `video_full_range_flag` takes its coded default of 0: the
//! first picture refines the caps to `range: Limited`. That is one refinement,
//! at the first picture only, so the second picture re-emits nothing.
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

/// The clip's geometry and rate, which the solved output caps carry.
const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;
const FRAMERATE: u32 = 30 << 16;

/// The CICP codepoint every colour field spells "unspecified".
const CICP_UNSPECIFIED: u8 = 2;

/// What the fixture's SPS resolves to: no colour block, so only the range is
/// concrete (`video_full_range_flag` defaults to 0).
const STREAM_COLORIMETRY: Colorimetry = Colorimetry {
    range: ColorRange::Limited,
    matrix: MatrixCoefficients::Unknown,
    transfer: g2g_core::TransferCharacteristics::Unknown,
    primaries: g2g_core::ColorPrimaries::Unknown,
};

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

    fn frame_count(&self) -> usize {
        self.packets
            .iter()
            .filter(|p| matches!(p, PipelinePacket::DataFrame(_)))
            .count()
    }
}

/// The clip's access units, each closed by its VCL NAL (type 1/5) and carrying
/// the SPS/PPS/SEI ahead of it. The first builds the decode session; the second
/// is what proves the refined caps are not re-emitted per picture.
fn access_units() -> Vec<Vec<u8>> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= CLIP.len() {
        if CLIP[i] == 0 && CLIP[i + 1] == 0 {
            if CLIP[i + 2] == 1 {
                starts.push(i + 3);
                i += 3;
                continue;
            }
            if i + 4 <= CLIP.len() && CLIP[i + 2] == 0 && CLIP[i + 3] == 1 {
                starts.push(i + 4);
                i += 4;
                continue;
            }
        }
        i += 1;
    }
    let mut units = Vec::new();
    let mut current = Vec::new();
    for (k, &begin) in starts.iter().enumerate() {
        let end = starts.get(k + 1).copied().unwrap_or(CLIP.len());
        current.extend_from_slice(&[0, 0, 0, 1]);
        current.extend_from_slice(&CLIP[begin..end]);
        if matches!(CLIP[begin] & 0x1F, 1 | 5) {
            units.push(core::mem::take(&mut current));
        }
    }
    units
}

/// The caps the runner solves for a launch line: the geometry it negotiated, no
/// colour information.
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

/// The same caps once the decoder has read the stream's colour description.
fn refined_output_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Fixed(FRAMERATE),
        interlace: g2g_core::Interlace::Any,
        colorimetry: STREAM_COLORIMETRY,
    }
}

/// Read the fixture's declared colour codepoints, so `STREAM_COLORIMETRY` stays
/// tied to the bitstream rather than to a remembered value.
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
        "fixture colour description changed; update STREAM_COLORIMETRY"
    );
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
fn the_solved_output_caps_are_forwarded_then_refined_once() {
    let Some(mut dec) = configured_decoder() else {
        return;
    };
    assert_fixture_declares_no_colour_description();

    let mut sink = RecordingSink::default();
    block_on(dec.process(PipelinePacket::CapsChanged(nv12_output_caps()), &mut sink))
        .expect("the decoder accepts the output caps the runner solved for it");
    assert_eq!(
        sink.caps_changes(),
        [&nv12_output_caps()],
        "the solved output caps reach the sink before any frame"
    );

    // Two pictures: the first refines the colour, the second must add nothing.
    let aus = access_units();
    assert!(aus.len() >= 2, "the fixture carries at least two pictures");
    for (i, au) in aus.iter().take(2).enumerate() {
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(au.clone().into_boxed_slice())),
            timing: FrameTiming::default(),
            sequence: i as u64,
            meta: Default::default(),
        };
        block_on(dec.process(PipelinePacket::DataFrame(frame), &mut sink))
            .expect("decode access unit");
    }
    // The system path is pipelined, so end of stream releases the tail.
    block_on(dec.process(PipelinePacket::Eos, &mut sink)).expect("flush at eos");
    assert_eq!(
        sink.caps_changes(),
        [&nv12_output_caps(), &refined_output_caps()],
        "the first picture refines the colour once; the second repeats nothing"
    );
    assert_eq!(
        sink.frame_count(),
        2,
        "both pictures decoded, so the second really did pass the caps check"
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
