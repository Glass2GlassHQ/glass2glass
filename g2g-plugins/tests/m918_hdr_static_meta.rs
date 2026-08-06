//! M918: HDR10 static metadata mined from the bitstream. The H.264 / H.265
//! parser reads the `mastering_display_colour_volume` and
//! `content_light_level_info` SEI messages and attaches them as an
//! `HdrStaticMeta`, latched across the coded video sequence so every frame
//! carries the grade (the SEI itself is coded once per IRAP).
//!
//! Unit under test = `NalParse`'s SEI mine, driven through the real element.

#![cfg(all(feature = "std", feature = "metadata"))]

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::meta::HdrStaticMeta;
use g2g_core::{
    AsyncElement, Caps, Dim, G2gError, MemoryDomain, OutputSink, PipelinePacket, PushOutcome, Rate,
    VideoCodec,
};
use g2g_plugins::h264parse::H264Parse;
use g2g_plugins::sei::{build_sei_nal, PAYLOAD_CONTENT_LIGHT_LEVEL, PAYLOAD_MASTERING_DISPLAY};

/// One VCL IDR slice NAL, the picture the SEI describes.
const VCL: [u8; 8] = [0x00, 0x00, 0x00, 0x01, 0x65, 0x88, 0x84, 0x00];

#[derive(Default)]
struct RecordingSink {
    frames: Vec<Frame>,
}

impl OutputSink for RecordingSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                self.frames.push(f);
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

/// A `mastering_display_colour_volume` payload for BT.2020 primaries + D65 at
/// 1000 nits peak / 0.005 nits black. Coded in the SEI's G, B, R primary order.
fn mastering_payload() -> Vec<u8> {
    let mut p = Vec::new();
    for (x, y) in [
        (0.170, 0.797),
        (0.131, 0.046),
        (0.708, 0.292),
        (0.3127, 0.3290),
    ] {
        p.extend_from_slice(&((x * 50_000.0) as u16).to_be_bytes());
        p.extend_from_slice(&((y * 50_000.0) as u16).to_be_bytes());
    }
    p.extend_from_slice(&10_000_000u32.to_be_bytes()); // 1000 cd/m^2
    p.extend_from_slice(&50u32.to_be_bytes()); // 0.005 cd/m^2
    p
}

fn content_light_payload(cll: u16, fall: u16) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&cll.to_be_bytes());
    p.extend_from_slice(&fall.to_be_bytes());
    p
}

fn frame(bytes: Vec<u8>, seq: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming::default(),
        seq,
    ))
}

/// The IRAP access unit: both HDR SEI messages then the coded picture.
fn hdr_au() -> Vec<u8> {
    let mut au = build_sei_nal(
        PAYLOAD_MASTERING_DISPLAY,
        &mastering_payload(),
        VideoCodec::H264,
    );
    au.extend_from_slice(&build_sei_nal(
        PAYLOAD_CONTENT_LIGHT_LEVEL,
        &content_light_payload(1200, 300),
        VideoCodec::H264,
    ));
    au.extend_from_slice(&VCL);
    au
}

#[tokio::test]
async fn hdr_sei_becomes_frame_metadata_and_latches() {
    let mut el = H264Parse::new();
    el.configure_pipeline(&h264_caps()).unwrap();
    let mut sink = RecordingSink::default();

    el.process(frame(hdr_au(), 0), &mut sink).await.unwrap();
    // Every following picture in the sequence carries no SEI of its own.
    for seq in 1..4u64 {
        el.process(frame(VCL.to_vec(), seq), &mut sink)
            .await
            .unwrap();
    }
    assert_eq!(sink.frames.len(), 4);

    for (i, f) in sink.frames.iter().enumerate() {
        let hdr = f
            .meta
            .get::<HdrStaticMeta>()
            .unwrap_or_else(|| panic!("frame {i} carries the stream's HDR metadata"));
        let m = hdr.mastering.expect("mastering display");
        assert!((m.display_primaries[0].x - 0.708).abs() < 1e-4, "red x");
        assert!((m.white_point.y - 0.3290).abs() < 1e-4, "D65 white y");
        assert!((m.max_luminance - 1000.0).abs() < 1e-3);
        assert!((m.min_luminance - 0.005).abs() < 1e-6);
        assert_eq!(hdr.max_content_light_level, Some(1200));
        assert_eq!(hdr.max_frame_average_light_level, Some(300));
    }
}

#[tokio::test]
async fn an_sdr_stream_carries_no_hdr_metadata() {
    let mut el = H264Parse::new();
    el.configure_pipeline(&h264_caps()).unwrap();
    let mut sink = RecordingSink::default();
    el.process(frame(VCL.to_vec(), 0), &mut sink).await.unwrap();
    assert_eq!(sink.frames.len(), 1);
    assert!(sink.frames[0].meta.get::<HdrStaticMeta>().is_none());
}

#[tokio::test]
async fn a_malformed_hdr_sei_is_ignored_not_fatal() {
    // A truncated mastering-display payload must leave the frame without HDR
    // metadata rather than panicking or inventing values.
    let payload = mastering_payload();
    let mut au = build_sei_nal(PAYLOAD_MASTERING_DISPLAY, &payload[..9], VideoCodec::H264);
    au.extend_from_slice(&VCL);

    let mut el = H264Parse::new();
    el.configure_pipeline(&h264_caps()).unwrap();
    let mut sink = RecordingSink::default();
    el.process(frame(au, 0), &mut sink).await.unwrap();
    assert!(sink.frames[0].meta.get::<HdrStaticMeta>().is_none());
}
