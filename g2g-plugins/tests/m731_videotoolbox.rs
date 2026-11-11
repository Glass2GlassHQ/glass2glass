// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! M731: `VtDecode` / `VtEncode` runtime validation on a real Mac (the macOS CI
//! runner is the Mac this repo's Linux dev host lacks). Decodes the checked-in
//! reference fixtures (x264 / x265 encoded, so the decoder is validated against
//! a reference peer, not just its own encoder) and round-trips synthetic frames
//! through the encoder. Mirrors `m19_mfencode` (Windows) and
//! `m489_vulkan_video_decode` (assertion style + conformance evidence).
#![cfg(all(target_os = "macos", feature = "vtdecode"))]

#[path = "common/collect_sink.rs"]
mod collect_sink;
use collect_sink::CollectSink;
#[cfg(feature = "vtencode")]
#[path = "common/keyframe_request_sink.rs"]
mod keyframe_request_sink;
#[cfg(feature = "vtencode")]
use keyframe_request_sink::KeyframeRequestSink;

use g2g_core::conformance::{ConformanceDimension, Evidence};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{AsyncElement, Caps, ConfigureOutcome, Dim, Rate, RawVideoFormat, VideoCodec};
use g2g_plugins::vtdecode::VtDecode;

const H264_CLIP: &[u8] = include_bytes!("fixtures/h264_640x480.h264");
const H265_CLIP: &[u8] = include_bytes!("fixtures/h265_640x480.h265");

const FRAME_DURATION_NS: u64 = 33_333_333;

impl CollectSink {
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
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        pixel_aspect: g2g_core::PixelAspect::Any,
        alignment: g2g_core::BitstreamAlignment::Unspecified,
    };
    let mut sink = CollectSink::default();
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

async fn drive<E: AsyncElement>(el: &mut E, es: &[u8], sink: &mut CollectSink) {
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
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        pixel_aspect: g2g_core::PixelAspect::Any,
        alignment: g2g_core::BitstreamAlignment::Unspecified,
    };
    let narrowed = dec.intercept_caps(&upstream).expect("intercept codec");
    let outcome = dec
        .configure_pipeline(&narrowed)
        .expect("VideoToolbox session must be reachable");
    assert!(matches!(outcome, ConfigureOutcome::Accepted));

    let mut sink = CollectSink::default();
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
                framerate: Rate::Any,
                ..
            })
        ),
        "decoded caps must be NV12 640x480 with the undeclared rate left unknown, got {caps_changes:?}"
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
async fn vtdecode_outputs_b_frames_in_presentation_order() {
    const B_FRAME_MP4: &[u8] = include_bytes!("fixtures/h264_64x48_25fps_gop10_bframes_offset.mp4");
    let mut demux = g2g_plugins::mp4demux::Mp4Demux::new();
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: g2g_core::ByteStreamEncoding::Mp4,
        })
        .expect("configure demuxer");
    let mut demuxed = CollectSink::default();
    drive(&mut demux, B_FRAME_MP4, &mut demuxed).await;
    let coded = demuxed
        .caps_changes()
        .into_iter()
        .find(|caps| matches!(caps, Caps::CompressedVideo { .. }))
        .expect("the demuxer reports the video track");
    let aus = demuxed.into_data_frames();
    let input_pts: Vec<u64> = aus.iter().map(|au| au.timing.pts_ns).collect();
    assert!(
        input_pts.windows(2).any(|pair| pair[1] < pair[0]),
        "the fixture codes frames out of presentation order"
    );

    let mut dec = VtDecode::h264();
    let narrowed = dec.intercept_caps(&coded).expect("intercept codec");
    dec.configure_pipeline(&narrowed)
        .expect("VideoToolbox session must be reachable");
    let mut sink = CollectSink::default();
    for au in aus {
        dec.process(PipelinePacket::DataFrame(au), &mut sink)
            .await
            .expect("decode access unit");
    }
    dec.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("drain at Eos");

    let output_pts: Vec<u64> = sink.data_frames().iter().map(|f| f.timing.pts_ns).collect();
    let mut presentation_order = input_pts;
    presentation_order.sort_unstable();
    assert_eq!(output_pts, presentation_order);
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
    use g2g_core::{Interlace, OutputSink};
    use g2g_plugins::vtencode::VtEncode;

    const WIDTH: u32 = 320;
    const HEIGHT: u32 = 240;
    const FPS_Q16: u32 = 30 << 16;
    const FRAMES: usize = 10;

    /// A synthetic packed NV12 picture: a per-frame luma gradient over a neutral
    /// chroma plane, so consecutive frames differ and the encoder has real work.
    fn nv12_picture(index: usize) -> Vec<u8> {
        let w = WIDTH as usize;
        let h = HEIGHT as usize;
        let mut data = vec![128u8; w * h * 3 / 2];
        for row in 0..h {
            for col in 0..w {
                data[row * w + col] = ((row + col + index * 8) % 256) as u8;
            }
        }
        data
    }

    // alpha ramps across each row, so the alpha plane is neither opaque nor flat
    fn bgra_picture(index: usize) -> Vec<u8> {
        let w = WIDTH as usize;
        let h = HEIGHT as usize;
        let mut data = vec![0u8; w * h * 4];
        for row in 0..h {
            for col in 0..w {
                let pixel = (row * w + col) * 4;
                let shade = ((row + col + index * 8) % 256) as u8;
                data[pixel..pixel + 4].copy_from_slice(&[
                    shade,
                    255 - shade,
                    128,
                    (col * 255 / w) as u8,
                ]);
            }
        }
        data
    }

    fn frame(format: RawVideoFormat, index: usize) -> Frame {
        let data = match format {
            RawVideoFormat::Nv12 => nv12_picture(index),
            RawVideoFormat::Bgra8 => bgra_picture(index),
            other => panic!("no test picture for {other:?}"),
        };
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

    fn input_caps(format: RawVideoFormat) -> Caps {
        Caps::RawVideo {
            format,
            width: Dim::Fixed(WIDTH),
            height: Dim::Fixed(HEIGHT),
            framerate: Rate::Fixed(FPS_Q16),
            interlace: Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
        }
    }

    /// Walk the Annex-B start codes of one access unit and yield each NAL's
    /// first two header bytes.
    fn nal_header_bytes(au: &[u8]) -> Vec<[u8; 2]> {
        let mut out = Vec::new();
        let mut i = 0;
        while i + 3 <= au.len() {
            if au[i..].starts_with(&[0, 0, 1]) {
                if let Some(&[first, second]) = au.get(i + 3..i + 5) {
                    out.push([first, second]);
                }
                i += 3;
            } else {
                i += 1;
            }
        }
        out
    }

    async fn encode_frames(enc: VtEncode, format: RawVideoFormat) -> CollectSink {
        let mut sink = CollectSink::default();
        encode_into(enc, format, &mut sink).await;
        sink
    }

    async fn encode_into(mut enc: VtEncode, format: RawVideoFormat, sink: &mut dyn OutputSink) {
        let narrowed = enc
            .intercept_caps(&input_caps(format))
            .expect("intercept raw input");
        let outcome = enc
            .configure_pipeline(&narrowed)
            .expect("VideoToolbox compression session must initialise");
        assert!(matches!(outcome, ConfigureOutcome::Accepted));

        for i in 0..FRAMES {
            enc.process(PipelinePacket::DataFrame(frame(format, i)), sink)
                .await
                .expect("encode DataFrame");
        }
        enc.process(PipelinePacket::Eos, sink)
            .await
            .expect("drain at Eos");
    }

    async fn forces_a_keyframe_on_request(enc: VtEncode) {
        const REQUEST_AT: usize = 3;
        let mut sink = KeyframeRequestSink::new(REQUEST_AT);
        encode_into(enc, RawVideoFormat::Nv12, &mut sink).await;
        assert_eq!(
            sink.keyframe_indices(),
            vec![0, REQUEST_AT + 1],
            "the access unit after the request is a keyframe, the interval puts no other one in {FRAMES} frames"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn vtencode_h264_forces_a_keyframe_on_request() {
        forces_a_keyframe_on_request(VtEncode::h264().with_bitrate(1_000_000)).await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn vtencode_h265_forces_a_keyframe_on_request() {
        forces_a_keyframe_on_request(VtEncode::h265().with_bitrate(1_000_000)).await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn vtencode_h265_with_alpha_codes_an_alpha_layer() {
        let enc = VtEncode::h265_with_alpha().with_bitrate(1_000_000);
        let encoded = encode_frames(enc, RawVideoFormat::Bgra8).await;

        let caps = encoded.caps_changes();
        assert!(
            matches!(
                caps.first(),
                Some(Caps::CompressedVideo {
                    codec: VideoCodec::H265,
                    width: Dim::Fixed(WIDTH),
                    height: Dim::Fixed(HEIGHT),
                    ..
                })
            ),
            "alpha encode is HEVC at the input geometry, got {caps:?}"
        );
        let aus = encoded.data_frames();
        assert_eq!(aus.len(), FRAMES, "no-reorder mode: one output per input");
        assert!(
            aus[0].timing.keyframe,
            "the first access unit is a keyframe"
        );
        for (index, au) in aus.iter().enumerate() {
            let au = au.domain.as_system_slice().expect("System-domain output");
            let headers = nal_header_bytes(au);
            // videotoolbox codes the alpha picture as slices outside base layer 0
            let alpha_slice = headers.iter().any(|h| {
                let layer_id = ((h[0] & 0x1) << 5) | (h[1] >> 3);
                let nal_type = (h[0] >> 1) & 0x3f;
                layer_id > 0 && nal_type < 32
            });
            assert!(
                alpha_slice,
                "access unit {index} has no alpha layer slice: {headers:x?}"
            );
        }
    }

    /// Round-trip: encode synthetic frames, assert the Annex-B stream shape,
    /// then decode it back and assert every picture is recovered.
    async fn round_trip(codec: VideoCodec) {
        let enc = match codec {
            VideoCodec::H265 => VtEncode::h265(),
            _ => VtEncode::h264(),
        }
        .with_bitrate(1_000_000);
        let encoded = encode_frames(enc, RawVideoFormat::Nv12).await;

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
                let types: Vec<u8> = headers.iter().map(|h| (h[0] >> 1) & 0x3f).collect();
                types.contains(&32) && types.contains(&33) && types.contains(&34)
            }
            _ => {
                let types: Vec<u8> = headers.iter().map(|h| h[0] & 0x1f).collect();
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

        let mut sink = CollectSink::default();
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
