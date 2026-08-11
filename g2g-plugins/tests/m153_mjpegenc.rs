//! M153 Motion-JPEG encode: `MjpegEnc` encodes packed RGBA to a baseline JPEG,
//! round-tripped back through `MjpegDec` (M152) to prove the output is a valid
//! JPEG carrying the source geometry and the dominant colour.
//!
//! M871 adds the `mozjpeg` backend: same round trip through `encoder=mozjpeg`,
//! plus evidence that its `quality` property is applied.

#![cfg(all(feature = "mjpeg-encode", feature = "mjpeg"))]

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket,
    PushOutcome, Rate, RawVideoFormat, VideoCodec,
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
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            match packet {
                PipelinePacket::CapsChanged(c) => self.caps.push(c),
                PipelinePacket::DataFrame(f) => {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.frames.push(s.to_vec());
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
        interlace: g2g_core::Interlace::Any,
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
        FrameTiming {
            pts_ns: 0,
            ..FrameTiming::default()
        },
        0,
    );
    enc.process(PipelinePacket::DataFrame(frame), &mut esink)
        .await
        .unwrap();

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
    dec.process(PipelinePacket::DataFrame(jframe), &mut dsink)
        .await
        .unwrap();

    let geometry = dsink.caps.iter().find_map(|c| match c {
        Caps::RawVideo {
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            ..
        } => Some((*w, *h)),
        _ => None,
    });
    assert_eq!(
        geometry,
        Some((W, H)),
        "decoded geometry matches the source"
    );
    assert_eq!(dsink.frames.len(), 1);
    let px = &dsink.frames[0][0..4];
    assert!(
        px[2] > 150,
        "blue channel dominant after round-trip (got {})",
        px[2]
    );
    assert!(
        px[0] < 100 && px[1] < 120,
        "red/green low (got {},{})",
        px[0],
        px[1]
    );
}

/// Solid-colour I420 (BT.601 limited range) for the given Y/U/V.
fn i420_solid(y: u8, u: u8, v: u8) -> Vec<u8> {
    let luma = (W * H) as usize;
    let chroma = luma / 4;
    let mut buf = vec![y; luma];
    buf.extend(core::iter::repeat_n(u, chroma));
    buf.extend(core::iter::repeat_n(v, chroma));
    buf
}

#[tokio::test]
async fn encodes_i420_to_mjpeg_that_roundtrips_to_blue() {
    let mut enc = MjpegEnc::new().with_quality(90);
    enc.configure_pipeline(&Caps::RawVideo {
        format: RawVideoFormat::I420,
        width: Dim::Fixed(W),
        height: Dim::Fixed(H),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
    })
    .unwrap();
    let mut esink = CaptureSink::default();

    // Blue (20,40,210) in BT.601 limited range is roughly Y=62, U=205, V=107.
    let blue = i420_solid(62, 205, 107);
    let frame = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(blue.into_boxed_slice())),
        FrameTiming::default(),
        0,
    );
    enc.process(PipelinePacket::DataFrame(frame), &mut esink)
        .await
        .unwrap();
    assert_eq!(esink.frames.len(), 1);
    assert_eq!(&esink.frames[0][0..2], &[0xFF, 0xD8], "JPEG SOI marker");

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
        MemoryDomain::System(SystemSlice::from_boxed(
            esink.frames[0].clone().into_boxed_slice(),
        )),
        FrameTiming::default(),
        0,
    );
    dec.process(PipelinePacket::DataFrame(jframe), &mut dsink)
        .await
        .unwrap();

    let px = &dsink.frames[0][0..4];
    assert!(
        px[2] > 150,
        "blue dominant after I420 -> jpeg -> rgba (got {})",
        px[2]
    );
    assert!(
        px[0] < 110 && px[1] < 130,
        "red/green low (got {},{})",
        px[0],
        px[1]
    );
}

/// Encode one `w` x `h` RGBA frame with `enc` and return the JPEG.
#[cfg(feature = "mozjpeg")]
async fn encode_rgba(mut enc: MjpegEnc, pixels: Vec<u8>, w: u32, h: u32) -> Vec<u8> {
    enc.configure_pipeline(&Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
    })
    .unwrap();
    let mut sink = CaptureSink::default();
    let frame = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(pixels.into_boxed_slice())),
        FrameTiming::default(),
        0,
    );
    enc.process(PipelinePacket::DataFrame(frame), &mut sink)
        .await
        .unwrap();
    sink.frames.remove(0)
}

/// M871: `encoder=mozjpeg` produces a JPEG the decoder reads back with the source
/// geometry and colour, and its `quality` property still moves the output size.
#[cfg(feature = "mozjpeg")]
#[tokio::test]
async fn mozjpeg_backend_encodes_a_decodable_jpeg_honouring_quality() {
    use g2g_core::PropValue;
    use g2g_plugins::mjpegenc::JpegEncodeBackend;

    let mut enc = MjpegEnc::new();
    enc.set_property("encoder", PropValue::Str("mozjpeg".into()))
        .unwrap();
    assert_eq!(
        enc.get_property("encoder")
            .and_then(|v| v.as_str().map(str::to_string)),
        Some("mozjpeg".into())
    );
    let jpeg = encode_rgba(enc, rgba_solid(20, 40, 210), W, H).await;
    assert_eq!(&jpeg[0..2], &[0xFF, 0xD8], "JPEG SOI marker");

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
    dec.process(PipelinePacket::DataFrame(jframe), &mut dsink)
        .await
        .unwrap();
    assert_eq!(
        dsink.caps,
        vec![Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(W),
            height: Dim::Fixed(H),
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
        }],
        "decoded geometry matches the mozjpeg-encoded source"
    );
    let px = &dsink.frames[0][0..4];
    assert!(
        px[2] > 150,
        "blue dominant after round-trip (got {})",
        px[2]
    );
    assert!(
        px[0] < 100 && px[1] < 120,
        "red/green low (got {},{})",
        px[0],
        px[1]
    );

    // Detailed content compresses to far more bytes at high quality: the
    // property is applied, not accepted and dropped.
    const N: u32 = 64;
    let mut noise = Vec::with_capacity((N * N * 4) as usize);
    let mut seed = 0x2545_f491u32;
    for _ in 0..(N * N) {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let b = seed.to_le_bytes();
        noise.extend_from_slice(&[b[0], b[1], b[2], 255]);
    }
    let low = encode_rgba(
        MjpegEnc::new()
            .with_backend(JpegEncodeBackend::Mozjpeg)
            .with_quality(20),
        noise.clone(),
        N,
        N,
    )
    .await;
    let high = encode_rgba(
        MjpegEnc::new()
            .with_backend(JpegEncodeBackend::Mozjpeg)
            .with_quality(95),
        noise,
        N,
        N,
    )
    .await;
    assert!(
        high.len() > low.len() * 2,
        "quality 95 ({} bytes) should dwarf quality 20 ({} bytes)",
        high.len(),
        low.len()
    );
}
