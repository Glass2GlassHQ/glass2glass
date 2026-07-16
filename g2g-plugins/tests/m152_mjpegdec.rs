//! M152 Motion-JPEG decode: `MjpegDec` decodes a baseline JPEG access unit to
//! RGBA8, recovering geometry from the JPEG headers and emitting it as a
//! `CapsChanged` before the first frame. The fixture is a 16x16 solid red JPEG.

#![cfg(feature = "mjpeg")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket,
    PushOutcome, RawVideoFormat, Rate, VideoCodec,
};
use g2g_plugins::mjpegdec::MjpegDec;

const RED16: &[u8] = include_bytes!("data/red16.jpg");

#[derive(Default)]
struct CaptureSink {
    caps: Vec<Caps>,
    frames: Vec<Vec<u8>>,
}

impl OutputSink for CaptureSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::CapsChanged(c) => self.caps.push(c),
                PipelinePacket::DataFrame(f) => {
                    if let MemoryDomain::System(s) = &f.domain {
                        self.frames.push(s.as_slice().to_vec());
                    }
                }
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

fn mjpeg_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::Mjpeg,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Fixed(30 << 16),
    }
}

fn frame(seq: u64) -> Frame {
    Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(RED16.to_vec().into_boxed_slice())),
        FrameTiming { pts_ns: seq * 33_000_000, ..FrameTiming::default() },
        seq,
    )
}

#[tokio::test]
async fn decodes_mjpeg_to_rgba8_with_recovered_geometry() {
    let mut dec = MjpegDec::new();
    dec.configure_pipeline(&mjpeg_caps()).unwrap();
    let mut sink = CaptureSink::default();

    for i in 0..2u64 {
        dec.process(PipelinePacket::DataFrame(frame(i)), &mut sink).await.unwrap();
    }

    // Geometry recovered from the JPEG headers, emitted once (constant size).
    assert_eq!(
        sink.caps,
        vec![Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(16),
            height: Dim::Fixed(16),
            framerate: Rate::Fixed(30 << 16),
        }],
        "one CapsChanged with the decoded 16x16 RGBA geometry"
    );

    assert_eq!(sink.frames.len(), 2, "one RGBA frame per JPEG access unit");
    for f in &sink.frames {
        assert_eq!(f.len(), 16 * 16 * 4, "RGBA8 is 4 bytes per pixel");
    }

    // The source was solid red; JPEG is lossy but the dominant channel survives.
    let px = &sink.frames[0][0..4];
    assert!(px[0] > 150, "red channel dominant (got {})", px[0]);
    assert!(px[1] < 100 && px[2] < 100, "green/blue low (got {},{})", px[1], px[2]);
    assert_eq!(px[3], 255, "opaque alpha");
}

#[tokio::test]
async fn decodes_mjpeg_to_i420() {
    let mut dec = MjpegDec::new().with_output_format(RawVideoFormat::I420);
    dec.configure_pipeline(&mjpeg_caps()).unwrap();
    let mut sink = CaptureSink::default();

    dec.process(PipelinePacket::DataFrame(frame(0)), &mut sink).await.unwrap();

    assert_eq!(
        sink.caps,
        vec![Caps::RawVideo {
            format: RawVideoFormat::I420,
            width: Dim::Fixed(16),
            height: Dim::Fixed(16),
            framerate: Rate::Fixed(30 << 16),
        }],
        "CapsChanged announces the I420 output format"
    );
    assert_eq!(sink.frames.len(), 1);
    // I420 is 4:2:0 planar: w*h luma + 2 * (w/2 * h/2) chroma.
    assert_eq!(sink.frames[0].len(), 16 * 16 * 3 / 2, "planar 4:2:0 byte size");
    // Solid red -> low luma, and the V (red-difference) plane sits well above 128.
    let v_plane_start = 16 * 16 + (8 * 8);
    assert!(sink.frames[0][v_plane_start] > 150, "red pushes the V chroma plane high");
}
