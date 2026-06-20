//! M153 Motion-JPEG encode: `MjpegEnc` encodes packed RGBA to a baseline JPEG,
//! round-tripped back through `MjpegDec` (M152) to prove the output is a valid
//! JPEG carrying the source geometry and the dominant colour.

#![cfg(all(feature = "mjpeg-encode", feature = "mjpeg"))]

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket,
    PushOutcome, RawVideoFormat, Rate, VideoCodec,
};
use g2g_plugins::mjpegdec::MjpegDec;
use g2g_plugins::mjpegenc::MjpegEnc;

const W: u32 = 32;
const H: u32 = 16;

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

fn rgba_solid(r: u8, g: u8, b: u8) -> Vec<u8> {
    let mut v = Vec::with_capacity((W * H * 4) as usize);
    for _ in 0..(W * H) {
        v.extend_from_slice(&[r, g, b, 255]);
    }
    v
}

fn rgba_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(W),
        height: Dim::Fixed(H),
        framerate: Rate::Fixed(30 << 16),
    }
}

#[tokio::test]
async fn encodes_rgba_to_mjpeg_that_roundtrips_through_mjpegdec() {
    let mut enc = MjpegEnc::new().with_quality(90);
    enc.configure_pipeline(&rgba_caps()).unwrap();
    let mut esink = CaptureSink::default();

    let blue = rgba_solid(20, 40, 210);
    let frame = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(blue.into_boxed_slice())),
        FrameTiming { pts_ns: 0, ..FrameTiming::default() },
        0,
    );
    enc.process(PipelinePacket::DataFrame(frame), &mut esink).await.unwrap();

    assert_eq!(
        esink.caps,
        vec![Caps::CompressedVideo {
            codec: VideoCodec::Mjpeg,
            width: Dim::Fixed(W),
            height: Dim::Fixed(H),
            framerate: Rate::Fixed(30 << 16),
        }],
        "one CapsChanged announcing the Mjpeg output geometry"
    );
    assert_eq!(esink.frames.len(), 1, "one JPEG access unit");
    let jpeg = &esink.frames[0];
    assert_eq!(&jpeg[0..2], &[0xFF, 0xD8], "JPEG SOI marker");

    // Round-trip: MjpegDec decodes the encoded JPEG back to RGBA.
    let mut dec = MjpegDec::new();
    dec.configure_pipeline(&Caps::CompressedVideo {
        codec: VideoCodec::Mjpeg,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    })
    .unwrap();
    let mut dsink = CaptureSink::default();
    let jframe = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(jpeg.clone().into_boxed_slice())),
        FrameTiming::default(),
        0,
    );
    dec.process(PipelinePacket::DataFrame(jframe), &mut dsink).await.unwrap();

    let geometry = dsink.caps.iter().find_map(|c| match c {
        Caps::RawVideo { width: Dim::Fixed(w), height: Dim::Fixed(h), .. } => Some((*w, *h)),
        _ => None,
    });
    assert_eq!(geometry, Some((W, H)), "decoded geometry matches the source");
    assert_eq!(dsink.frames.len(), 1);
    let px = &dsink.frames[0][0..4];
    assert!(px[2] > 150, "blue channel dominant after round-trip (got {})", px[2]);
    assert!(px[0] < 100 && px[1] < 120, "red/green low (got {},{})", px[0], px[1]);
}
