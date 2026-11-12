// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! M219: on-device smoke test for `MediaCodecDec` (Android NDK MediaCodec).
//!
//! Unlike the cross-compile `cargo check` CI runs, this actually decodes on a
//! real device: it feeds an embedded H.264 Annex-B clip through the NDK
//! MediaCodec H.264 decoder and asserts it emits NV12 frames of the right
//! geometry. The H.264 fixture is `include_bytes!`d (so the test binary is
//! self-contained: nothing extra to push to the device), and the decoder is
//! headless (configured with no Surface, ByteBuffer output), so this runs as a
//! plain native binary with no APK / Activity.
//!
//! It only builds for `aarch64-linux-android` (et al.) with the `mediacodec`
//! feature; on the dev host it compiles to nothing. Run it on a connected
//! device with `tools/android-mediacodec-smoke.sh` (cargo-ndk build + adb push +
//! run), or via cargo-dinghy: `cargo dinghy -d android test -p g2g-plugins
//! --features mediacodec`.

#![cfg(all(target_os = "android", feature = "mediacodec"))]

#[path = "common/collect_sink.rs"]
mod collect_sink;
use collect_sink::CollectSink;

#[path = "common/parsed_access_units.rs"]
mod parsed_access_units;
use parsed_access_units::parsed_access_units;

#[path = "common/android_binder.rs"]
mod android_binder;
use android_binder::start_binder_threadpool;

use g2g_core::element::AsyncElement;
use g2g_core::frame::PipelinePacket;
use g2g_core::{Caps, ConfigureOutcome, Dim, Rate, RawVideoFormat, VideoCodec};
use g2g_plugins::mediacodecdec::MediaCodecDec;

/// Short Annex-B clips (640x480, 10 frames: parameter sets + IDR + inter frames),
/// committed under `tests/fixtures/` and embedded so the test binary needs no
/// companion file on the device. 640x480 (not a tiny size) because hardware
/// decoders' graphic-buffer allocators reject sub-minimum dimensions.
const H264: &[u8] = include_bytes!("fixtures/h264_640x480.h264");
const H265: &[u8] = include_bytes!("fixtures/h265_640x480.h265");

/// Drive `dec` with the access units of `stream` and assert it emits NV12 frames
/// of the right geometry. Shared by the H.264 and H.265 tests.
async fn decode_to_nv12(mut dec: MediaCodecDec, stream: &[u8], codec: VideoCodec, w: u32, h: u32) {
    start_binder_threadpool(1);

    // Negotiation surrogate: upstream caps carry the fixture's geometry, as a
    // parser would emit from the SPS. MediaCodec's `configure()` requires
    // width/height (it returns EINVAL without them), unlike the codec-private
    // parameter sets which ride the stream.
    let upstream = Caps::CompressedVideo {
        codec,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        pixel_aspect: g2g_core::PixelAspect::Any,
        alignment: g2g_core::BitstreamAlignment::Unspecified,
    };
    let narrowed = dec.intercept_caps(&upstream).expect("intercept caps");
    let outcome = dec
        .configure_pipeline(&narrowed)
        .expect("configure decoder");
    assert!(matches!(outcome, ConfigureOutcome::Accepted));

    let mut sink = CollectSink::default();

    // M756: the graph runner hands each element its own solved OUTPUT caps as an
    // incoming `CapsChanged` before the first frame (transform_arm in
    // graph_runner.rs sends `process(CapsChanged(forward_caps))`, where
    // forward_caps is the derived NV12 output). The element must accept and
    // forward it, not reject it as an unrecognised input shape. This drives that
    // packet exactly as the runner would; a decoder that only accepts its
    // compressed input shape returns a bare `CapsMismatch` here.
    let prefixed_out = Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        pixel_aspect: g2g_core::PixelAspect::Any,
    };
    dec.process(PipelinePacket::CapsChanged(prefixed_out.clone()), &mut sink)
        .await
        .expect("decoder must accept the runner's pre-fixed output CapsChanged");
    assert!(
        matches!(sink.packets.first(), Some(PipelinePacket::CapsChanged(c)) if *c == prefixed_out),
        "the pre-fixed output caps must be forwarded downstream before any frame, got {:?}",
        sink.packets.first()
    );

    let access_units = parsed_access_units(codec, stream).await;
    let fed = access_units.len();
    for access_unit in access_units {
        dec.process(PipelinePacket::DataFrame(access_unit), &mut sink)
            .await
            .expect("process access unit");
    }
    // EOS drains any frames the codec is still holding.
    dec.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("process Eos drains the codec");

    let caps_changes: Vec<_> = sink
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::CapsChanged(c) => Some(c.clone()),
            _ => None,
        })
        .collect();
    let data_frames: Vec<_> = sink
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f),
            _ => None,
        })
        .collect();

    eprintln!(
        "{:?}: decoded {} frame(s); {} CapsChanged emitted",
        codec,
        data_frames.len(),
        caps_changes.len()
    );
    assert!(
        !caps_changes.is_empty(),
        "expected at least one NV12 CapsChanged"
    );
    assert_eq!(data_frames.len(), fed, "every access unit decodes");
    let latency = dec.latency();
    assert!(
        latency.min_ns > 0,
        "the pictures MediaCodec holds are reported: {latency:?}"
    );
    assert_eq!(latency.max_ns, Some(latency.min_ns));
    let decoded_caps = caps_changes.last().unwrap();
    assert!(
        matches!(
            decoded_caps,
            Caps::RawVideo {
                framerate: Rate::Any,
                ..
            }
        ),
        "the upstream declared no frame rate, so the decoder must not invent one, got {decoded_caps:?}"
    );

    // The first CapsChanged advertises the decoded NV12 geometry; the first
    // frame's buffer must match it (Y + interleaved UV = w*h*3/2).
    match caps_changes.first().unwrap() {
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(fw),
            height: Dim::Fixed(fh),
            ..
        } => {
            eprintln!("first NV12 caps: {}x{}", fw, fh);
            assert!(*fw > 0 && *fh > 0);
            let expected = (*fw as usize) * (*fh as usize) * 3 / 2;
            assert_eq!(
                data_frames
                    .first()
                    .unwrap()
                    .domain
                    .as_system_slice()
                    .expect("decoder must emit System-domain NV12 frames")
                    .len(),
                expected,
                "NV12 byte length mismatch"
            );
        }
        other => panic!("expected NV12 fixed caps, got {:?}", other),
    }
}

#[tokio::test]
async fn mediacodec_decodes_h264_to_nv12() {
    decode_to_nv12(MediaCodecDec::h264(), H264, VideoCodec::H264, 640, 480).await;
}

#[tokio::test]
async fn mediacodec_decodes_h265_to_nv12() {
    decode_to_nv12(MediaCodecDec::h265(), H265, VideoCodec::H265, 640, 480).await;
}
