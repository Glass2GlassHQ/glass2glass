#![cfg(all(feature = "dav1d", feature = "av1-encode"))]
//! AV1 decode round-trip: encode I420 frames to AV1 with the pure-Rust `Av1Enc`
//! (rav1e), then decode them back with `Dav1dDec` (libdav1d) and check the
//! recovered frames are correctly-sized I420 of the encoded geometry whose luma
//! is the flat grey that went in (AV1 is lossy, so within a tolerance). Proves
//! the dav1d decode path end to end without an external fixture; libdav1d is a
//! system dependency, so this builds only with the `dav1d` feature.

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, Caps, Dim, G2gError, OutputSink, PipelinePacket, PushOutcome, Rate,
    RawVideoFormat,
};

use g2g_plugins::av1enc::Av1Enc;
use g2g_plugins::dav1ddec::Dav1dDec;

const W: u32 = 64;
const H: u32 = 64;
const GREY: u8 = 128;

fn i420_grey(w: u32, h: u32) -> Vec<u8> {
    let (w, h) = (w as usize, h as usize);
    let (cw, ch) = (w / 2, h / 2);
    let mut v = vec![GREY; w * h]; // luma
    v.extend(vec![GREY; cw * ch]); // U
    v.extend(vec![GREY; cw * ch]); // V
    v
}

fn i420_caps(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::I420,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Any,
    }
}

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

#[tokio::test]
async fn av1_encode_then_dav1d_decode_round_trips_i420() {
    // Encode 6 flat-grey I420 frames to AV1.
    let mut enc = Av1Enc::new().with_speed(10);
    enc.configure_pipeline(&i420_caps(W, H)).unwrap();
    let mut encoded = CaptureSink::default();
    for i in 0..6u64 {
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(i420_grey(W, H).into_boxed_slice())),
            FrameTiming {
                pts_ns: i * 33_000_000,
                ..FrameTiming::default()
            },
            i,
        );
        enc.process(PipelinePacket::DataFrame(frame), &mut encoded)
            .await
            .unwrap();
    }
    enc.process(PipelinePacket::Eos, &mut encoded)
        .await
        .unwrap();
    assert!(
        !encoded.frames.is_empty(),
        "the encoder produced AV1 packets"
    );

    // Decode the AV1 packets back with libdav1d.
    let mut dec = Dav1dDec::new();
    dec.configure_pipeline(&encoded.caps[0]).unwrap();
    let mut decoded = CaptureSink::default();
    for data in &encoded.frames {
        let f = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(data.clone().into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        dec.process(PipelinePacket::DataFrame(f), &mut decoded)
            .await
            .unwrap();
    }

    // The decoder announced the encoded geometry as I420.
    assert!(
        decoded.caps.contains(&Caps::RawVideo {
            format: RawVideoFormat::I420,
            width: Dim::Fixed(W),
            height: Dim::Fixed(H),
            framerate: Rate::Any,
        }),
        "dav1d announced the 64x64 I420 geometry, got {:?}",
        decoded.caps,
    );

    assert!(
        !decoded.frames.is_empty(),
        "dav1d decoded at least one frame"
    );
    let expected_len = (W * H + 2 * (W / 2) * (H / 2)) as usize; // tight I420
    for plane in &decoded.frames {
        assert_eq!(
            plane.len(),
            expected_len,
            "decoded frame is a tightly-packed I420 buffer"
        );
        // The flat-grey input survives the lossy round trip: the luma mean stays
        // near 128 (a corrupt decode / wrong stride packing would not).
        let y = &plane[..(W * H) as usize];
        let mean = y.iter().map(|&b| b as u64).sum::<u64>() / y.len() as u64;
        assert!(
            (120..=136).contains(&mean),
            "decoded luma mean {mean} is near the grey input"
        );
    }
}

/// Flat-grey planar frame of `format` (8-bit): a Y plane plus two chroma planes
/// sized by the format's subsampling.
fn planar_grey(format: RawVideoFormat, w: u32, h: u32) -> Vec<u8> {
    let (hs, vs) = format.chroma_shift().unwrap();
    let (w, h) = (w as usize, h as usize);
    let (cw, ch) = (w.div_ceil(1 << hs), h.div_ceil(1 << vs));
    let mut v = vec![GREY; w * h]; // luma
    v.extend(vec![GREY; cw * ch]); // U
    v.extend(vec![GREY; cw * ch]); // V
    v
}

fn raw_caps(format: RawVideoFormat, w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Any,
    }
}

/// Encode flat-grey `format` frames with rav1e, decode with `Dav1dDec`, and assert
/// the decoder announces that exact format/geometry and recovers a tight buffer
/// whose luma survives the lossy round trip. Proves the multi-plane / subsampling
/// packing path beyond 4:2:0.
async fn roundtrip_chroma(format: RawVideoFormat) {
    let (hs, vs) = format.chroma_shift().unwrap();
    let mut enc = Av1Enc::new().with_speed(10);
    enc.configure_pipeline(&raw_caps(format, W, H)).unwrap();
    let mut encoded = CaptureSink::default();
    for i in 0..6u64 {
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(
                planar_grey(format, W, H).into_boxed_slice(),
            )),
            FrameTiming {
                pts_ns: i * 33_000_000,
                ..FrameTiming::default()
            },
            i,
        );
        enc.process(PipelinePacket::DataFrame(frame), &mut encoded)
            .await
            .unwrap();
    }
    enc.process(PipelinePacket::Eos, &mut encoded)
        .await
        .unwrap();
    assert!(
        !encoded.frames.is_empty(),
        "the encoder produced AV1 packets for {format:?}"
    );

    let mut dec = Dav1dDec::new();
    dec.configure_pipeline(&encoded.caps[0]).unwrap();
    let mut decoded = CaptureSink::default();
    for data in &encoded.frames {
        let f = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(data.clone().into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        dec.process(PipelinePacket::DataFrame(f), &mut decoded)
            .await
            .unwrap();
    }

    assert!(
        decoded.caps.contains(&raw_caps(format, W, H)),
        "dav1d announced {format:?} 64x64, got {:?}",
        decoded.caps,
    );
    assert!(
        !decoded.frames.is_empty(),
        "dav1d decoded at least one {format:?} frame"
    );
    let (cw, ch) = (
        (W as usize).div_ceil(1 << hs),
        (H as usize).div_ceil(1 << vs),
    );
    let expected_len = (W * H) as usize + 2 * cw * ch;
    for plane in &decoded.frames {
        assert_eq!(plane.len(), expected_len, "tight {format:?} buffer");
        let y = &plane[..(W * H) as usize];
        let mean = y.iter().map(|&b| b as u64).sum::<u64>() / y.len() as u64;
        assert!(
            (120..=136).contains(&mean),
            "{format:?} luma mean {mean} near grey"
        );
    }
}

#[tokio::test]
async fn av1_4_2_2_round_trips_through_dav1d() {
    roundtrip_chroma(RawVideoFormat::I422).await;
}

#[tokio::test]
async fn av1_4_4_4_round_trips_through_dav1d() {
    roundtrip_chroma(RawVideoFormat::I444).await;
}

/// Flat mid-grey planar frame of a 10/12-bit `format`: samples are little-endian
/// `u16`, matching the encoder's and decoder's high-bit-depth wire layout.
fn planar_grey16(format: RawVideoFormat, w: u32, h: u32, grey: u16) -> Vec<u8> {
    let (hs, vs) = format.chroma_shift().unwrap();
    let (w, h) = (w as usize, h as usize);
    let (cw, ch) = (w.div_ceil(1 << hs), h.div_ceil(1 << vs));
    let n_samples = w * h + 2 * cw * ch;
    let mut v = Vec::with_capacity(n_samples * 2);
    for _ in 0..n_samples {
        v.extend_from_slice(&grey.to_le_bytes());
    }
    v
}

/// Encode flat mid-grey 10/12-bit frames with rav1e (`Context<u16>`) and decode
/// with `Dav1dDec`, asserting the decoder announces the high-bit-depth format and
/// recovers a tight LE-`u16` buffer whose luma mean stays near the input value.
async fn roundtrip_high_depth(format: RawVideoFormat) {
    let depth = format.bit_depth();
    let grey: u16 = 1 << (depth - 1); // mid-grey for this depth
    let (hs, vs) = format.chroma_shift().unwrap();
    let mut enc = Av1Enc::new().with_speed(10);
    enc.configure_pipeline(&raw_caps(format, W, H)).unwrap();
    let mut encoded = CaptureSink::default();
    for i in 0..6u64 {
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(
                planar_grey16(format, W, H, grey).into_boxed_slice(),
            )),
            FrameTiming {
                pts_ns: i * 33_000_000,
                ..FrameTiming::default()
            },
            i,
        );
        enc.process(PipelinePacket::DataFrame(frame), &mut encoded)
            .await
            .unwrap();
    }
    enc.process(PipelinePacket::Eos, &mut encoded)
        .await
        .unwrap();
    assert!(
        !encoded.frames.is_empty(),
        "encoder produced AV1 packets for {format:?}"
    );

    let mut dec = Dav1dDec::new();
    dec.configure_pipeline(&encoded.caps[0]).unwrap();
    let mut decoded = CaptureSink::default();
    for data in &encoded.frames {
        let f = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(data.clone().into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        dec.process(PipelinePacket::DataFrame(f), &mut decoded)
            .await
            .unwrap();
    }

    assert!(
        decoded.caps.contains(&raw_caps(format, W, H)),
        "dav1d announced {format:?} 64x64, got {:?}",
        decoded.caps,
    );
    assert!(
        !decoded.frames.is_empty(),
        "dav1d decoded at least one {format:?} frame"
    );
    let (cw, ch) = (
        (W as usize).div_ceil(1 << hs),
        (H as usize).div_ceil(1 << vs),
    );
    let expected_len = ((W * H) as usize + 2 * cw * ch) * 2; // tight, 2 bytes/sample
    for plane in &decoded.frames {
        assert_eq!(plane.len(), expected_len, "tight {format:?} LE-u16 buffer");
        let y = &plane[..(W * H) as usize * 2];
        let mean = y
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]) as u64)
            .sum::<u64>()
            / (W * H) as u64;
        // Flat field survives the lossy round trip: mean near the input grey.
        assert!(
            (grey as u64).abs_diff(mean) <= (grey >> 3) as u64,
            "{format:?} luma mean {mean} near grey {grey}",
        );
    }
}

#[tokio::test]
async fn av1_10bit_4_2_0_round_trips_through_dav1d() {
    roundtrip_high_depth(RawVideoFormat::I420p10).await;
}

#[tokio::test]
async fn av1_12bit_4_2_0_round_trips_through_dav1d() {
    roundtrip_high_depth(RawVideoFormat::I420p12).await;
}

#[tokio::test]
async fn av1_10bit_4_2_2_round_trips_through_dav1d() {
    roundtrip_high_depth(RawVideoFormat::I422p10).await;
}

#[tokio::test]
async fn av1_12bit_4_4_4_round_trips_through_dav1d() {
    roundtrip_high_depth(RawVideoFormat::I444p12).await;
}
