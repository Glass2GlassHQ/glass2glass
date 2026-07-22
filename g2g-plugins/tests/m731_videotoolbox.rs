//! M731: `VtDecode` / `VtEncode` runtime validation on a real Mac (the macOS CI
//! runner is the Mac this repo's Linux dev host lacks). Decodes the checked-in
//! reference fixtures (x264 / x265 encoded, so the decoder is validated against
//! a reference peer, not just its own encoder) and round-trips synthetic frames
//! through the encoder. Mirrors `m19_mfencode` (Windows) and
//! `m489_vulkan_video_decode` (assertion style + conformance evidence).
#![cfg(all(target_os = "macos", feature = "vtdecode"))]

use core::future::Future;
use core::pin::Pin;

use g2g_core::conformance::{ConformanceDimension, Evidence};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, G2gError, OutputSink, PushOutcome, Rate,
    RawVideoFormat, VideoCodec,
};
use g2g_plugins::vtdecode::VtDecode;

const H264_CLIP: &[u8] = include_bytes!("fixtures/h264_640x480.h264");
const H265_CLIP: &[u8] = include_bytes!("fixtures/h265_640x480.h265");

const FRAME_DURATION_NS: u64 = 33_333_333;

#[derive(Default)]
struct Collect {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for Collect {
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

impl Collect {
    fn caps_changes(&self) -> Vec<Caps> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::CapsChanged(c) => Some(c.clone()),
                _ => None,
            })
            .collect()
    }

    fn data_frames(&self) -> Vec<&Frame> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some(f),
                _ => None,
            })
            .collect()
    }

    fn into_data_frames(self) -> Vec<Frame> {
        self.packets
            .into_iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some(f),
                _ => None,
            })
            .collect()
    }
}

/// The platform tag for the persisted `Hardware` evidence rows.
fn platform_tag() -> String {
    format!("macOS {} VideoToolbox", std::env::consts::ARCH)
}

/// Split an Annex-B elementary stream into access units with the real
/// re-framing parser (`H264Parse` / `H265Parse`), the element a pipeline would
/// put in front of the decoder.
async fn access_units(codec: VideoCodec, es: &[u8]) -> Vec<Frame> {
    let caps = Caps::CompressedVideo {
        codec,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    };
    let mut sink = Collect::default();
    match codec {
        VideoCodec::H265 => {
            let mut parse = g2g_plugins::h265parse::H265Parse::reframing();
            parse.configure_pipeline(&caps).expect("configure parser");
            drive(&mut parse, es, &mut sink).await;
        }
        _ => {
            let mut parse = g2g_plugins::h264parse::H264Parse::reframing();
            parse.configure_pipeline(&caps).expect("configure parser");
            drive(&mut parse, es, &mut sink).await;
        }
    }
    let aus = sink.into_data_frames();
    assert!(aus.len() > 1, "parser split the fixture into access units");
    aus
}

async fn drive<E: AsyncElement>(el: &mut E, es: &[u8], sink: &mut Collect) {
    for (i, chunk) in es.chunks(4096).enumerate() {
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(chunk.to_vec().into_boxed_slice())),
            FrameTiming::default(),
            i as u64,
        );
        el.process(PipelinePacket::DataFrame(frame), sink)
            .await
            .expect("parse chunk");
    }
    el.process(PipelinePacket::Eos, sink).await.expect("Eos");
}

/// Decode `es` through `dec` and assert a full, plausible NV12 picture stream
/// at the expected geometry comes out.
async fn decode_fixture(mut dec: VtDecode, codec: VideoCodec, es: &[u8]) {
    let aus = access_units(codec, es).await;
    let fed = aus.len();

    let upstream = Caps::CompressedVideo {
        codec,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Any,
    };
    let narrowed = dec.intercept_caps(&upstream).expect("intercept codec");
    let outcome = dec
        .configure_pipeline(&narrowed)
        .expect("VideoToolbox session must be reachable");
    assert!(matches!(outcome, ConfigureOutcome::Accepted));

    let mut sink = Collect::default();
    for (i, mut au) in aus.into_iter().enumerate() {
        // Distinct, monotonic PTS per access unit (the parser leaves the
        // fixture's frames untimed).
        au.timing.pts_ns = i as u64 * FRAME_DURATION_NS;
        dec.process(PipelinePacket::DataFrame(au), &mut sink)
            .await
            .expect("decode access unit");
    }
    dec.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("drain at Eos");

    let caps_changes = sink.caps_changes();
    assert!(
        matches!(
            caps_changes.first(),
            Some(Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(640),
                height: Dim::Fixed(480),
                ..
            })
        ),
        "decoded caps must be NV12 640x480, got {caps_changes:?}"
    );

    let frames = sink.data_frames();
    assert_eq!(
        frames.len(),
        fed,
        "every access unit must decode to a frame"
    );
    let expected_len = 640 * 480 * 3 / 2;
    for f in &frames {
        let Some(slice) = f.domain.as_system_slice() else {
            panic!("decoder must emit System-domain frames");
        };
        assert_eq!(slice.len(), expected_len, "packed NV12 length");
    }

    // The fixtures are test cards with near-black (16) and near-white regions.
    // Uniform luma = no real decode; missing extremes = a desynced one (same
    // checks as the Vulkan Video decode test).
    let Some(first) = frames[0].domain.as_system_slice() else {
        unreachable!()
    };
    let luma = &first[..640 * 480];
    let min = *luma.iter().min().unwrap();
    let max = *luma.iter().max().unwrap();
    assert!(max > min, "decoded luma is uniform ({min}=={max})");
    assert!(min <= 20, "no near-black content (min {min})");
    assert!(max >= 200, "no bright content (max {max})");
    eprintln!("decoded {fed} frames; first luma range {min}..={max}");
}

#[tokio::test(flavor = "current_thread")]
async fn vtdecode_h264_fixture() {
    decode_fixture(VtDecode::h264(), VideoCodec::H264, H264_CLIP).await;

    g2g_plugins::conformance::persist::record_evidence(
        "vtdecode",
        &Evidence::new(ConformanceDimension::Hardware)
            .platform(platform_tag())
            .codec("h264")
            .detail("VideoToolbox decode of the reference fixture to NV12"),
    )
    .expect("record hardware evidence");
}

#[tokio::test(flavor = "current_thread")]
async fn vtdecode_h265_fixture() {
    decode_fixture(VtDecode::h265(), VideoCodec::H265, H265_CLIP).await;

    g2g_plugins::conformance::persist::record_evidence(
        "vtdecode",
        &Evidence::new(ConformanceDimension::Hardware)
            .platform(platform_tag())
            .codec("h265")
            .detail("VideoToolbox decode of the reference fixture to NV12"),
    )
    .expect("record hardware evidence");
}

#[cfg(feature = "vtencode")]
mod encode {
    use super::*;
    use g2g_plugins::vtencode::VtEncode;

    const WIDTH: u32 = 320;
    const HEIGHT: u32 = 240;
    const FPS_Q16: u32 = 30 << 16;
    const FRAMES: usize = 10;

    /// A synthetic packed NV12 picture: a per-frame luma gradient over a neutral
    /// chroma plane, so consecutive frames differ and the encoder has real work.
    fn nv12_frame(index: usize) -> Frame {
        let w = WIDTH as usize;
        let h = HEIGHT as usize;
        let mut data = vec![128u8; w * h * 3 / 2];
        for row in 0..h {
            for col in 0..w {
                data[row * w + col] = ((row + col + index * 8) % 256) as u8;
            }
        }
        Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
            timing: FrameTiming {
                pts_ns: index as u64 * FRAME_DURATION_NS,
                dts_ns: index as u64 * FRAME_DURATION_NS,
                duration_ns: FRAME_DURATION_NS,
                capture_ns: index as u64 * FRAME_DURATION_NS,
                ..FrameTiming::default()
            },
            sequence: index as u64,
            meta: Default::default(),
        }
    }

    fn input_caps() -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(WIDTH),
            height: Dim::Fixed(HEIGHT),
            framerate: Rate::Fixed(FPS_Q16),
        }
    }

    /// Walk the Annex-B start codes of one access unit and yield each NAL's
    /// first header byte.
    fn nal_header_bytes(au: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut i = 0;
        while i + 3 <= au.len() {
            if au[i..].starts_with(&[0, 0, 1]) {
                if let Some(&b) = au.get(i + 3) {
                    out.push(b);
                }
                i += 3;
            } else {
                i += 1;
            }
        }
        out
    }

    async fn encode_frames(mut enc: VtEncode) -> Collect {
        let narrowed = enc.intercept_caps(&input_caps()).expect("intercept NV12");
        let outcome = enc
            .configure_pipeline(&narrowed)
            .expect("VideoToolbox compression session must initialise");
        assert!(matches!(outcome, ConfigureOutcome::Accepted));

        let mut sink = Collect::default();
        for i in 0..FRAMES {
            enc.process(PipelinePacket::DataFrame(nv12_frame(i)), &mut sink)
                .await
                .expect("encode DataFrame");
        }
        enc.process(PipelinePacket::Eos, &mut sink)
            .await
            .expect("drain at Eos");
        sink
    }

    /// Round-trip: encode synthetic frames, assert the Annex-B stream shape,
    /// then decode it back and assert every picture is recovered.
    async fn round_trip(codec: VideoCodec) {
        let enc = match codec {
            VideoCodec::H265 => VtEncode::h265(),
            _ => VtEncode::h264(),
        }
        .with_bitrate(1_000_000);
        let encoded = encode_frames(enc).await;

        let caps = encoded.caps_changes();
        assert!(
            matches!(
                caps.first(),
                Some(Caps::CompressedVideo {
                    codec: c,
                    width: Dim::Fixed(WIDTH),
                    height: Dim::Fixed(HEIGHT),
                    ..
                }) if *c == codec
            ),
            "encoder caps must carry the input geometry, got {caps:?}"
        );

        let aus = encoded.data_frames();
        assert_eq!(aus.len(), FRAMES, "no-reorder mode: one output per input");
        for f in &aus {
            let Some(slice) = f.domain.as_system_slice() else {
                panic!("encoder must emit System-domain frames");
            };
            let data = slice;
            assert!(
                data.starts_with(&[0, 0, 0, 1]) || data.starts_with(&[0, 0, 1]),
                "encoded output must be Annex-B (got {:?})",
                &data[..data.len().min(8)]
            );
        }

        // The first access unit is the IDR/IRAP and must carry the in-band
        // parameter sets the element prepends from the format description.
        let Some(first) = aus[0].domain.as_system_slice() else {
            unreachable!()
        };
        let headers = nal_header_bytes(first);
        let has_params = match codec {
            VideoCodec::H265 => {
                let types: Vec<u8> = headers.iter().map(|b| (b >> 1) & 0x3f).collect();
                types.contains(&32) && types.contains(&33) && types.contains(&34)
            }
            _ => {
                let types: Vec<u8> = headers.iter().map(|b| b & 0x1f).collect();
                types.contains(&7) && types.contains(&8)
            }
        };
        assert!(
            has_params,
            "keyframe must carry parameter sets: {headers:x?}"
        );

        // Decode it back through VtDecode.
        let mut dec = match codec {
            VideoCodec::H265 => VtDecode::h265(),
            _ => VtDecode::h264(),
        };
        let h_caps = encoded.caps_changes().pop().expect("encoder caps");
        let narrowed = dec.intercept_caps(&h_caps).expect("intercept codec");
        dec.configure_pipeline(&narrowed).expect("decoder session");

        let mut sink = Collect::default();
        for f in encoded.into_data_frames() {
            dec.process(PipelinePacket::DataFrame(f), &mut sink)
                .await
                .expect("decode DataFrame");
        }
        dec.process(PipelinePacket::Eos, &mut sink)
            .await
            .expect("drain at Eos");

        let frames = sink.data_frames();
        assert_eq!(frames.len(), FRAMES, "every picture must decode back out");
        let expected_len = (WIDTH * HEIGHT * 3 / 2) as usize;
        for f in frames {
            let Some(slice) = f.domain.as_system_slice() else {
                panic!("decoder must emit System-domain frames");
            };
            assert_eq!(slice.len(), expected_len, "packed NV12 length");
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn vtencode_h264_round_trip() {
        round_trip(VideoCodec::H264).await;

        g2g_plugins::conformance::persist::record_evidence(
            "vtencode",
            &Evidence::new(ConformanceDimension::Hardware)
                .platform(platform_tag())
                .codec("h264")
                .detail("VideoToolbox encode to Annex-B and decode round trip"),
        )
        .expect("record hardware evidence");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn vtencode_h265_round_trip() {
        round_trip(VideoCodec::H265).await;

        g2g_plugins::conformance::persist::record_evidence(
            "vtencode",
            &Evidence::new(ConformanceDimension::Hardware)
                .platform(platform_tag())
                .codec("h265")
                .detail("VideoToolbox encode to Annex-B and decode round trip"),
        )
        .expect("record hardware evidence");
    }
}
