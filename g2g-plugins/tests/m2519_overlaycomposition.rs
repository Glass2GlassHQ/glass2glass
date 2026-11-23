// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(feature = "std", feature = "metadata"))]

#[path = "common/collect_sink.rs"]
mod collect_sink;

use g2g_core::meta::PlaneLayout;
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{
    AllocationParams, AsyncElement, Caps, CapsSet, Frame, FrameTiming, MemoryDomain,
    OverlayCompositionMeta, OverlayPlacement, OverlayRectangle, PipelinePacket,
};
use g2g_plugins::overlaycomposition::OverlayComposition;
use g2g_plugins::registry::default_registry;

use collect_sink::CollectSink;

fn caps(format: &str) -> Caps {
    CapsSet::from_gst_string(&format!(
        "video/x-raw,format={format},width=2,height=2,framerate=25/1"
    ))
    .unwrap()
    .fixate()
    .unwrap()
}

fn frame(bytes: Vec<u8>) -> Frame {
    Frame::new(
        MemoryDomain::System(g2g_core::memory::SystemSlice::from_boxed(
            bytes.into_boxed_slice(),
        )),
        FrameTiming {
            pts_ns: 17,
            ..FrameTiming::default()
        },
        3,
    )
}

fn red_green() -> OverlayRectangle {
    OverlayRectangle::new(
        OverlayPlacement {
            x: -1,
            y: 0,
            width: 2,
            height: 1,
        },
        2,
        1,
        [255, 0, 0, 255, 0, 255, 0, 255].into(),
    )
    .unwrap()
}

fn half_blue() -> OverlayRectangle {
    OverlayRectangle::new(
        OverlayPlacement {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        },
        1,
        1,
        [0, 0, 128, 128].into(),
    )
    .unwrap()
}

async fn process(overlay: &mut OverlayComposition, frame: Frame) -> Frame {
    let mut sink = CollectSink::default();
    overlay
        .process(PipelinePacket::DataFrame(frame), &mut sink)
        .await
        .unwrap();
    let Some(PipelinePacket::DataFrame(frame)) = sink.packets.pop() else {
        panic!("expected output frame")
    };
    frame
}

#[tokio::test]
async fn cpu_blends_ordered_clipped_overlays_into_padded_bgra() {
    let mut overlay = OverlayComposition::new().with_draw(|frame, size| {
        assert_eq!(size, (2, 2));
        (frame.timing.pts() == Some(17)).then(|| OverlayCompositionMeta::new(vec![half_blue()]))
    });
    overlay.configure_pipeline(&caps("BGRA")).unwrap();
    let padded = [0, 0, 0, 255, 0, 0, 0, 255, 7, 9].repeat(2);
    let mut input = frame(padded.clone());
    let ptr = input.domain.as_system_slice().unwrap().as_ptr();
    let layout = PlaneLayout::single(10);
    input.meta.attach(layout);
    input
        .meta
        .attach(OverlayCompositionMeta::new(vec![red_green()]));
    let output = process(&mut overlay, input).await;
    let bytes = output.domain.as_system_slice().unwrap();
    assert_eq!(bytes.as_ptr(), ptr);
    assert_eq!(&bytes[..4], &[128, 127, 0, 255]);
    assert_eq!(&bytes[4..], &padded[4..]);
    assert_eq!(output.meta.get::<PlaneLayout>(), Some(&layout));
    assert!(output.meta.get::<OverlayCompositionMeta>().is_none());

    let mut next = frame(padded.clone());
    next.timing.pts_ns = 18;
    let next = process(&mut overlay, next).await;
    assert_eq!(next.domain.as_system_slice().unwrap(), padded);
}

#[tokio::test]
async fn cpu_blends_nv12_and_clips_large_placements_to_the_frame() {
    let mut overlay = OverlayComposition::new().with_draw(|_, _| {
        let white = OverlayRectangle::new(
            OverlayPlacement {
                x: i32::MIN,
                y: i32::MIN,
                width: u32::MAX,
                height: u32::MAX,
            },
            1,
            1,
            [255, 255, 255, 255].into(),
        )
        .unwrap();
        Some(OverlayCompositionMeta::new(vec![white]))
    });
    overlay.configure_pipeline(&caps("NV12")).unwrap();
    let output = process(&mut overlay, frame(vec![16, 16, 16, 16, 64, 192])).await;
    assert_eq!(
        output.domain.as_system_slice().unwrap(),
        [235, 235, 235, 235, 128, 128]
    );
    assert!(output.meta.get::<OverlayCompositionMeta>().is_none());
}

#[tokio::test]
async fn allocation_request_preserves_video_and_appends_rectangles() {
    let mut overlay = OverlayComposition::new()
        .with_draw(|_, _| Some(OverlayCompositionMeta::new(vec![half_blue()])));
    let requests = overlay.meta_requests();
    overlay.configure_pipeline(&caps("RGBA")).unwrap();
    overlay.configure_allocation(&AllocationParams::meta_demand(requests));
    let mut input = frame(vec![37; 16]);
    let ptr = input.domain.as_system_slice().unwrap().as_ptr();
    input
        .meta
        .attach(OverlayCompositionMeta::new(vec![red_green()]));
    let output = process(&mut overlay, input).await;
    let bytes = output.domain.as_system_slice().unwrap();
    assert_eq!(bytes.as_ptr(), ptr);
    assert_eq!(bytes, &[37; 16]);
    assert_eq!(output.timing.pts(), Some(17));
    assert_eq!(output.sequence, 3);
    assert_eq!(
        output
            .meta
            .get::<OverlayCompositionMeta>()
            .unwrap()
            .rectangles(),
        [red_green(), half_blue()]
    );
}

#[tokio::test]
async fn launch_registers_the_transform() {
    let graph = parse_launch(
        &default_registry(),
        "videotestsrc num-buffers=2 ! overlaycomposition ! fakesink",
    )
    .unwrap();
    let stats = run_graph(graph, &g2g_core::clock::MonotonicClock, 4)
        .await
        .unwrap();
    assert_eq!(stats.frames_consumed, 2);
}
