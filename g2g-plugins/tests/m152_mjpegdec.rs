//! M152 Motion-JPEG decode: `MjpegDec` decodes a baseline JPEG access unit to
//! RGBA8, recovering geometry from the JPEG headers and emitting it as a
//! `CapsChanged` before the first frame. The fixture is a 16x16 solid red JPEG.
//!
//! M871 adds the direct YCbCr -> I420 path: it must agree with decoding to RGBA
//! and running the shared conversion, and a grayscale JPEG (no YCbCr identity
//! path) must still decode through the RGBA fallback.

#![cfg(feature = "mjpeg")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket,
    PushOutcome, Rate, RawVideoFormat, VideoCodec,
};
use g2g_plugins::mjpegdec::MjpegDec;

const RED16: &[u8] = include_bytes!("data/red16.jpg");
/// A 16x16 single-component (grayscale) JPEG gradient, authored with `cjpeg
/// -grayscale`: it has no YCbCr identity path, so it exercises the RGBA fallback.
const GRAY16: &[u8] = include_bytes!("data/gray16.jpg");

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
        FrameTiming {
            pts_ns: seq * 33_000_000,
            ..FrameTiming::default()
        },
        seq,
    )
}

#[tokio::test]
async fn decodes_mjpeg_to_rgba8_with_recovered_geometry() {
    let mut dec = MjpegDec::new();
    dec.configure_pipeline(&mjpeg_caps()).unwrap();
    let mut sink = CaptureSink::default();

    for i in 0..2u64 {
        dec.process(PipelinePacket::DataFrame(frame(i)), &mut sink)
            .await
            .unwrap();
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
    assert!(
        px[1] < 100 && px[2] < 100,
        "green/blue low (got {},{})",
        px[1],
        px[2]
    );
    assert_eq!(px[3], 255, "opaque alpha");
}

#[tokio::test]
async fn decodes_mjpeg_to_i420() {
    let mut dec = MjpegDec::new().with_output_format(RawVideoFormat::I420);
    dec.configure_pipeline(&mjpeg_caps()).unwrap();
    let mut sink = CaptureSink::default();

    dec.process(PipelinePacket::DataFrame(frame(0)), &mut sink)
        .await
        .unwrap();

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
    assert_eq!(
        sink.frames[0].len(),
        16 * 16 * 3 / 2,
        "planar 4:2:0 byte size"
    );
    // Solid red -> low luma, and the V (red-difference) plane sits well above 128.
    let v_plane_start = 16 * 16 + (8 * 8);
    assert!(
        sink.frames[0][v_plane_start] > 150,
        "red pushes the V chroma plane high"
    );
}

/// Decode one JPEG through a fresh `MjpegDec` set to `format`.
async fn decode_once(jpeg: &[u8], format: RawVideoFormat) -> Vec<u8> {
    decode_with(MjpegDec::new().with_output_format(format), jpeg).await
}

/// Push one JPEG access unit through an already-built decoder.
async fn decode_with(mut dec: MjpegDec, jpeg: &[u8]) -> Vec<u8> {
    dec.configure_pipeline(&mjpeg_caps()).unwrap();
    let mut sink = CaptureSink::default();
    let f = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(jpeg.to_vec().into_boxed_slice())),
        FrameTiming::default(),
        0,
    );
    dec.process(PipelinePacket::DataFrame(f), &mut sink)
        .await
        .unwrap();
    sink.frames.remove(0)
}

#[tokio::test]
async fn grayscale_jpeg_decodes_through_the_rgba_fallback() {
    let rgba = decode_once(GRAY16, RawVideoFormat::Rgba8).await;
    assert_eq!(rgba.len(), 16 * 16 * 4);
    for px in rgba.chunks_exact(4) {
        assert!(
            px[0] == px[1] && px[1] == px[2],
            "grayscale replicates luma across R/G/B, got {px:?}"
        );
    }
    assert!(
        rgba.chunks_exact(4).any(|px| px[0] != rgba[0]),
        "the fixture is a gradient, not a flat grey"
    );

    let i420 = decode_once(GRAY16, RawVideoFormat::I420).await;
    let luma = 16 * 16;
    assert_eq!(i420.len(), luma + luma / 2, "planar 4:2:0 byte size");
    for (i, &c) in i420[luma..].iter().enumerate() {
        assert!(
            (c as i32 - 128).abs() <= 2,
            "grey means neutral chroma, sample {i} is {c}"
        );
    }
    assert!(
        i420[..luma].iter().any(|&y| y != i420[0]),
        "the luma gradient survives the I420 path"
    );
}

/// A real 4:2:0 JPEG of a smooth RGBA gradient, so the parity check runs on
/// content with chroma detail rather than a flat colour.
#[cfg(feature = "mjpeg-encode")]
async fn gradient_jpeg(w: u32, h: u32) -> Vec<u8> {
    use g2g_plugins::mjpegenc::MjpegEnc;

    let mut pixels = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            pixels.extend_from_slice(&[(x * 7) as u8, (y * 13) as u8, (x * 3 + y * 5) as u8, 255]);
        }
    }
    let mut enc = MjpegEnc::new().with_quality(92);
    enc.configure_pipeline(&Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    })
    .unwrap();
    let mut sink = CaptureSink::default();
    let f = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(pixels.into_boxed_slice())),
        FrameTiming::default(),
        0,
    );
    enc.process(PipelinePacket::DataFrame(f), &mut sink)
        .await
        .unwrap();
    sink.frames.remove(0)
}

/// M871: the direct YCbCr -> I420 decode must land where decoding to RGBA and
/// running the shared `VideoConvert` path lands, geometry exactly and samples to
/// within a rounding step (both are BT.601, the direct path just skips the RGB
/// round trip).
#[cfg(feature = "mjpeg-encode")]
#[tokio::test]
async fn direct_i420_matches_the_rgba_conversion_route() {
    let (w, h) = (32u32, 16u32);
    let jpeg = gradient_jpeg(w, h).await;

    let direct = decode_once(&jpeg, RawVideoFormat::I420).await;
    let rgba = decode_once(&jpeg, RawVideoFormat::Rgba8).await;
    let reference = g2g_plugins::videoconvert::convert(
        &rgba,
        RawVideoFormat::Rgba8,
        RawVideoFormat::I420,
        w as usize,
        h as usize,
    );

    let expect = (w * h * 3 / 2) as usize;
    assert_eq!(direct.len(), expect, "planar 4:2:0 byte size");
    assert_eq!(reference.len(), expect);
    let worst = direct
        .iter()
        .zip(reference.iter())
        .map(|(a, b)| (*a as i32 - *b as i32).abs())
        .max()
        .unwrap();
    assert!(
        worst <= 2,
        "direct YCbCr -> I420 diverges from the RGBA route by {worst}"
    );
}

/// M871: `decoder=mozjpeg` decodes the same picture as the default zune backend,
/// in both output formats. The two use different IDCT and chroma-upsampling
/// kernels, so samples differ by a few LSB rather than matching exactly.
#[cfg(feature = "mozjpeg")]
#[tokio::test]
async fn mozjpeg_backend_decodes_the_same_picture_as_zune() {
    use g2g_plugins::mjpegdec::JpegDecodeBackend;

    let (w, h) = (32u32, 16u32);
    let jpeg = gradient_jpeg(w, h).await;

    for format in [RawVideoFormat::Rgba8, RawVideoFormat::I420] {
        let zune = decode_once(&jpeg, format).await;
        let moz = decode_with(
            MjpegDec::new()
                .with_output_format(format)
                .with_backend(JpegDecodeBackend::Mozjpeg),
            &jpeg,
        )
        .await;
        assert_eq!(moz.len(), zune.len(), "{format:?}: same geometry");
        let worst = moz
            .iter()
            .zip(zune.iter())
            .map(|(a, b)| (*a as i32 - *b as i32).abs())
            .max()
            .unwrap();
        assert!(worst <= 8, "{format:?}: backends differ by {worst}");
    }
}

/// The `decoder` property is the launch-line half of the backend switch.
#[cfg(feature = "mozjpeg")]
#[test]
fn decoder_property_round_trips() {
    use g2g_core::PropValue;

    let mut dec = MjpegDec::new();
    assert_eq!(
        dec.get_property("decoder")
            .and_then(|v| v.as_str().map(str::to_string)),
        Some("zune".into()),
        "zune is the default backend"
    );
    dec.set_property("decoder", PropValue::Str("mozjpeg".into()))
        .unwrap();
    assert_eq!(
        dec.get_property("decoder")
            .and_then(|v| v.as_str().map(str::to_string)),
        Some("mozjpeg".into())
    );
    assert!(dec
        .set_property("decoder", PropValue::Str("nope".into()))
        .is_err());
}
