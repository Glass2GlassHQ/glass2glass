//! M586: `VulkanVideoDec` reorders B-frames to display order on the streaming
//! (AU-by-AU) system path.
//!
//! Hardware decode retires pictures in coding order; a stream with B-frames has a
//! different display (POC) order. The whole-stream `decode_all` path already
//! reorders (M569), and the element's GPU-texture path streams too since M744.
//! The element's system (NV12) path is the true streaming shape:
//! one access unit per `process` call, decoded through the pipelined `decode_push`
//! ring, which stays in coding order. This drives that path with a real B-frame
//! clip fed one AU at a time and asserts the element emits frames in DISPLAY
//! order, byte-for-byte matching the M569-validated `decode_all` output.
//!
//! The oracle is `decode_all` on the same clip (its display-order output is
//! checked bit-exact vs the software decoder in M569); H.264 / H.265 decode is
//! deterministic, so a correctly reordering element must reproduce it frame for
//! frame. The fixture must genuinely reorder (POC non-monotonic in decode order),
//! else a no-op would pass; that is asserted first.
//!
//! Runs on the RTX 3060; skips with no adapter / no decode support.
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
    PushOutcome, Rate, VideoCodec,
};
use g2g_plugins::vulkanvideo::{
    extract_h264_parameter_sets, extract_h265_parameter_sets, open_h264_decode_device,
    open_h265_decode_device, to_std_h265_params, Nv12Frame, VulkanVideoDec, VulkanVideoError,
};

const H264: &[u8] = include_bytes!("fixtures/h264_640x480_bframes.h264");
const H265: &[u8] = include_bytes!("fixtures/h265_640x480_bframes.h265");

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

/// Split an Annex-B stream into per-picture access units. `is_vcl(nal)` decides
/// whether a NAL is a coded-slice NAL; each VCL NAL closes an AU (the fixtures are
/// single-slice per picture), carrying any preceding parameter-set / SEI NALs.
fn split_access_units(stream: &[u8], is_vcl: impl Fn(&[u8]) -> bool) -> Vec<Vec<u8>> {
    let mut units = Vec::new();
    let mut cur = Vec::new();
    let starts = start_code_offsets(stream);
    for (k, &begin) in starts.iter().enumerate() {
        let end = starts.get(k + 1).copied().unwrap_or(stream.len());
        let nal = &stream[begin..end];
        cur.extend_from_slice(&[0, 0, 0, 1]);
        cur.extend_from_slice(nal);
        if is_vcl(nal) {
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

/// The concatenated NV12 planes of a decoded frame (luma then interleaved CbCr),
/// exactly what the element emits as its system-memory buffer.
fn planes(f: &Nv12Frame) -> Vec<u8> {
    let mut v = Vec::with_capacity(f.luma.len() + f.chroma.len());
    v.extend_from_slice(&f.luma);
    v.extend_from_slice(&f.chroma);
    v
}

/// Feed `clip` through the element one AU at a time and collect the emitted NV12
/// system buffers, in emission order.
fn element_streaming_output(codec: VideoCodec, clip: &[u8], aus: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    let mut dec = VulkanVideoDec::new();
    let in_caps = Caps::CompressedVideo {
        codec,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30 << 16),
    };
    dec.configure_pipeline(&in_caps)
        .expect("configure opens the decode device");
    let _ = clip;

    let mut sink = RecordingSink::default();
    for (i, au) in aus.into_iter().enumerate() {
        block_on(dec.process(PipelinePacket::DataFrame(au_frame(au, i as u64)), &mut sink))
            .expect("decode access unit");
    }
    block_on(dec.process(PipelinePacket::Eos, &mut sink)).expect("flush at eos");

    sink.packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(Frame {
                domain: MemoryDomain::System(s),
                ..
            }) => Some(s.as_slice().to_vec()),
            _ => None,
        })
        .collect()
}

#[test]
fn h264_streaming_bframes_emit_in_display_order() {
    let device = match block_on(open_h264_decode_device()) {
        Ok(d) => d,
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip m586 h264: no Vulkan H.264 decode adapter");
            return;
        }
        Err(e) => panic!("open h264 device: {e:?}"),
    };
    let ps = extract_h264_parameter_sets(H264).expect("sps/pps");
    let session = device.create_h264_session(&ps, 640, 480).expect("session");
    let mut oracle_dec = device
        .create_h264_dpb_decoder(&session, &ps)
        .expect("decoder");

    // The fixture must actually reorder, else a non-reordering element would pass.
    let pocs: Vec<i32> = oracle_dec
        .index_pictures(H264)
        .expect("index")
        .iter()
        .map(|m| m.poc)
        .collect();
    assert!(
        pocs.windows(2).any(|w| w[1] < w[0]),
        "fixture has no B-frame reorder (POC monotonic)"
    );

    // Oracle: whole-stream decode_all, display order (M569-validated).
    let oracle: Vec<Vec<u8>> = oracle_dec
        .decode_all(H264)
        .expect("decode_all")
        .iter()
        .map(planes)
        .collect();

    let aus = split_access_units(H264, |nal| {
        matches!(nal.first().map(|b| b & 0x1F), Some(1..=5))
    });
    assert_eq!(aus.len(), oracle.len(), "one AU per coded picture");

    let got = element_streaming_output(VideoCodec::H264, H264, aus);
    assert_eq!(
        got.len(),
        oracle.len(),
        "element emits one frame per coded picture"
    );
    for (i, (g, o)) in got.iter().zip(&oracle).enumerate() {
        assert_eq!(
            g, o,
            "frame {i} differs from display-order oracle (element did not reorder)"
        );
    }
    eprintln!(
        "m586 h264: {} streamed AUs emitted in display order",
        got.len()
    );
}

#[test]
fn h265_streaming_bframes_emit_in_display_order() {
    let device = match block_on(open_h265_decode_device()) {
        Ok(d) => d,
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip m586 h265: no Vulkan H.265 decode adapter");
            return;
        }
        Err(e) => panic!("open h265 device: {e:?}"),
    };
    let ps = extract_h265_parameter_sets(H265).expect("vps/sps/pps");
    let std = to_std_h265_params(&ps);
    let session = device.create_h265_session(&std, 640, 480).expect("session");
    let mut oracle_dec = device
        .create_h265_dpb_decoder(&session, &ps)
        .expect("decoder");

    let pocs: Vec<i32> = oracle_dec
        .index_pictures(H265)
        .expect("index")
        .iter()
        .map(|m| m.poc)
        .collect();
    assert!(
        pocs.windows(2).any(|w| w[1] < w[0]),
        "fixture has no B-frame reorder (POC monotonic)"
    );

    let oracle: Vec<Vec<u8>> = oracle_dec
        .decode_all(H265)
        .expect("decode_all")
        .iter()
        .map(planes)
        .collect();

    // H.265 VCL NAL types are 0..=31 (nal_unit_type = (byte >> 1) & 0x3F).
    let aus = split_access_units(H265, |nal| {
        nal.first()
            .map(|b| ((b >> 1) & 0x3F) <= 31)
            .unwrap_or(false)
    });
    assert_eq!(aus.len(), oracle.len(), "one AU per coded picture");

    let got = element_streaming_output(VideoCodec::H265, H265, aus);
    assert_eq!(
        got.len(),
        oracle.len(),
        "element emits one frame per coded picture"
    );
    for (i, (g, o)) in got.iter().zip(&oracle).enumerate() {
        assert_eq!(
            g, o,
            "frame {i} differs from display-order oracle (element did not reorder)"
        );
    }
    eprintln!(
        "m586 h265: {} streamed AUs emitted in display order",
        got.len()
    );
}
