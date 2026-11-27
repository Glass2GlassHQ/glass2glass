// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! M736: `MetalVideoSink` on the macOS CI runner. Presents decoded fixture
//! frames to a standalone `CAMetalLayer` (a real drawable swapchain, headless)
//! over both input domains: the M735 zero-copy `CvPixelBuffer` path (IOSurface
//! planes imported as Metal textures) and packed System NV12. Readback
//! verifies the YUV -> RGB shader against the test card. Skips without a Metal
//! device, like the wgpu/Vulkan suites.
#![cfg(all(target_os = "macos", feature = "metal-sink", feature = "vtdecode"))]

#[path = "common/collect_sink.rs"]
mod collect_sink;
use collect_sink::CollectSink;

use g2g_core::conformance::{ConformanceDimension, Evidence};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::MemoryDomain;
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{
    AspectRatio, AsyncElement, Caps, ConfigureOutcome, Dim, Interlace, PipelineClock, PixelAspect,
    PropValue, Rate, RawVideoFormat, VideoCodec,
};
use g2g_plugins::metalvideosink::MetalVideoSink;
use g2g_plugins::registry::default_registry;
use g2g_plugins::vtdecode::VtDecode;

const H264_CLIP: &[u8] = include_bytes!("fixtures/h264_640x480.h264");
const FRAME_DURATION_NS: u64 = 33_333_333;

impl CollectSink {
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

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Decode the fixture (cv-output or packed) into frames ready to present.
async fn decode_fixture(cv: bool) -> Vec<Frame> {
    let caps = Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        pixel_aspect: g2g_core::PixelAspect::Any,
        alignment: g2g_core::BitstreamAlignment::Unspecified,
    };
    let mut parse = g2g_plugins::h264parse::H264Parse::reframing();
    parse.configure_pipeline(&caps).expect("configure parser");
    let mut sink = CollectSink::default();
    for (i, chunk) in H264_CLIP.chunks(4096).enumerate() {
        let frame = Frame::from_system_bytes(chunk.to_vec(), FrameTiming::default(), i as u64);
        parse
            .process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .expect("parse chunk");
    }
    parse
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("Eos");
    let mut aus = sink.into_data_frames();
    for (i, au) in aus.iter_mut().enumerate() {
        au.timing.pts_ns = i as u64 * FRAME_DURATION_NS;
    }

    let mut dec = if cv {
        VtDecode::h264().with_cv_output()
    } else {
        VtDecode::h264()
    };
    let dec_caps = Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30 << 16),
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        pixel_aspect: g2g_core::PixelAspect::Any,
        alignment: g2g_core::BitstreamAlignment::Unspecified,
    };
    let narrowed = dec.intercept_caps(&dec_caps).expect("intercept H.264");
    dec.configure_pipeline(&narrowed).expect("decoder session");
    let mut sink = CollectSink::default();
    for au in aus {
        dec.process(PipelinePacket::DataFrame(au), &mut sink)
            .await
            .expect("decode");
    }
    dec.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("drain");
    sink.into_data_frames()
}

/// Present `frames` through a readback sink and check the rendered test card.
async fn present_and_check(frames: Vec<Frame>) -> u64 {
    let n = frames.len() as u64;
    let mut sink = MetalVideoSink::new().with_readback();
    let caps = Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30 << 16),
        interlace: Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        pixel_aspect: g2g_core::PixelAspect::Any,
    };
    let narrowed = sink.intercept_caps(&caps).expect("intercept NV12");
    assert!(matches!(
        sink.configure_pipeline(&narrowed).expect("metal state"),
        ConfigureOutcome::Accepted
    ));
    let mut out = CollectSink::default();
    for f in frames {
        sink.process(PipelinePacket::DataFrame(f), &mut out)
            .await
            .expect("present");
    }
    assert_eq!(sink.presented(), n, "every frame presented");

    // The rendered test card must contain bright and dark pixels: a black
    // output means the shader sampled nothing; uniform grey means a desync.
    let rgba = sink.last_rgba().expect("readback captured");
    assert_eq!(rgba.len(), 640 * 480 * 4, "full RGBA surface");
    let mut min = u8::MAX;
    let mut max = 0u8;
    for px in rgba.as_chunks::<4>().0 {
        // Luma-ish check on the green channel (present in both extremes).
        min = min.min(px[1]);
        max = max.max(px[1]);
    }
    assert!(min <= 40, "no dark content in render (min {min})");
    assert!(max >= 190, "no bright content in render (max {max})");
    eprintln!("presented {n} frames; rendered green range {min}..={max}");
    n
}

#[tokio::test(flavor = "current_thread")]
async fn presents_zero_copy_cvpixelbuffer_frames() {
    if !MetalVideoSink::device_available() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let frames = decode_fixture(true).await;
    assert!(
        frames
            .iter()
            .all(|f| matches!(f.domain, MemoryDomain::CvPixelBuffer(_))),
        "decoder emitted the zero-copy domain"
    );
    present_and_check(frames).await;

    g2g_plugins::conformance::persist::record_evidence(
        "metalvideosink",
        &Evidence::new(ConformanceDimension::Hardware)
            .platform(format!("macOS {} Metal", std::env::consts::ARCH))
            .codec("h264")
            .detail("zero-copy CVPixelBuffer decode presented via CAMetalLayer"),
    )
    .expect("record hardware evidence");
}

#[tokio::test(flavor = "current_thread")]
async fn presents_packed_system_frames() {
    if !MetalVideoSink::device_available() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let frames = decode_fixture(false).await;
    present_and_check(frames).await;
}

#[tokio::test(flavor = "current_thread")]
async fn letterboxes_a_non_square_picture_in_the_host_drawable() {
    if !MetalVideoSink::device_available() {
        eprintln!("skipping: no Metal device");
        return;
    }
    const SIDE: u32 = 64;
    let layer = objc2_quartz_core::CAMetalLayer::layer();
    layer.setDrawableSize(objc2_core_foundation::CGSize::new(
        f64::from(SIDE),
        f64::from(SIDE),
    ));
    // SAFETY: the layer lives through the test and has no view hierarchy.
    let mut sink = unsafe { MetalVideoSink::new().with_layer(core::ptr::NonNull::from(&*layer)) }
        .with_readback();
    let caps = Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(SIDE),
        height: Dim::Fixed(SIDE),
        framerate: Rate::Any,
        interlace: Interlace::Progressive,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        pixel_aspect: PixelAspect::Exact(AspectRatio::new(2, 1).unwrap()),
    };
    sink.configure_pipeline(&caps).unwrap();
    let luma_len = (SIDE * SIDE) as usize;
    let mut white = vec![235; luma_len];
    white.resize(luma_len * 3 / 2, 128);
    let mut out = CollectSink::default();
    for force in [true, false] {
        sink.set_property("force-aspect-ratio", PropValue::Bool(force))
            .unwrap();
        assert_eq!(
            sink.get_property("force-aspect-ratio"),
            Some(PropValue::Bool(force))
        );
        let frame = Frame::from_system_bytes(white.clone(), FrameTiming::default(), 0);
        sink.process(PipelinePacket::DataFrame(frame), &mut out)
            .await
            .unwrap();
        let rgba = sink.last_rgba().expect("readback");
        assert_eq!(rgba.len(), luma_len * 4, "the host owns the drawable size");
        for (row, bytes) in rgba.chunks_exact(SIDE as usize * 4).enumerate() {
            let bright = !force || (16..48).contains(&row);
            assert!(
                bytes.chunks_exact(4).all(|px| {
                    px[3] == 255
                        && if bright {
                            px[..3].iter().all(|&v| v >= 240)
                        } else {
                            px[..3] == [0, 0, 0]
                        }
                }),
                "canvas row {row}, force-aspect-ratio={force}"
            );
        }
    }
}

/// The launch path: decode -> present in a text pipeline, with the
/// `autovideosink` alias resolving to `metalvideosink` on this platform.
#[tokio::test(flavor = "current_thread")]
async fn presents_in_a_text_pipeline_via_autovideosink() {
    if !MetalVideoSink::device_available() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let path = std::env::temp_dir().join(format!("g2g-m736-{}.h264", std::process::id()));
    std::fs::write(&path, H264_CLIP).expect("write temp fixture");
    let line = format!(
        "filesrc location={} ! h264parse ! vtdec cv-output=true ! autovideosink",
        path.display()
    );
    let reg = default_registry();
    let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("`{line}` runs: {e:?}"));
    std::fs::remove_file(&path).ok();
    assert_eq!(stats.frames_consumed, 10, "every fixture frame presented");
}
