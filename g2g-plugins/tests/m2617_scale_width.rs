// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "std")]

use g2g_core::runtime::{parse_launch, run_graph_observed, Observer};
use g2g_core::PipelineClock;
use g2g_plugins::appsink::{register_appsink_pull, Pull};
use g2g_plugins::appsrc::register_appsrc;
use g2g_plugins::registry::default_registry;

struct ZeroClock;

impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

async fn check_width(id: &str, stages: &str, (width, height): (usize, usize)) {
    let feed = register_appsrc(id);
    let sink = register_appsink_pull(id);
    assert!(feed.push(&[128; 12 * 8 * 3 / 2], 0));
    assert!(feed.end_of_stream());
    let line = format!(
        "appsrc channel={id} caps=video/x-raw,format=NV12,width=12,height=8,framerate=30/1,pixel-aspect-ratio=1/1 \
         ! {stages} ! appsink channel={id}"
    );
    let graph = parse_launch(&default_registry(), &line).unwrap();
    let observer = Observer::new();
    let stats = run_graph_observed(graph, &ZeroClock, 4, &observer, None)
        .await
        .unwrap_or_else(|err| panic!("{line}: {err:?}"));
    assert_eq!(stats.frames_consumed, 1, "{line}");
    let Pull::Frame(frame) = sink.try_pull() else {
        panic!("expected a scaled frame from {line}");
    };
    assert_eq!(
        frame.domain.as_system_slice().unwrap().len(),
        width * height * 3 / 2,
        "{line}"
    );
    let wanted = format!("width={width},height={height}");
    let snapshot = observer.snapshot();
    assert!(
        snapshot.edges.iter().any(|edge| edge
            .observed_caps
            .as_deref()
            .is_some_and(|caps| { caps.contains(&wanted) })),
        "no caps with {wanted} in {line}: {:?}",
        snapshot.edges
    );
}

#[tokio::test]
async fn a_pinned_ratio_derives_the_open_width() {
    for (id, scaler) in [
        ("m2617_videoscale", "videoscale"),
        ("m2617_videoconvertscale", "videoconvertscale"),
        ("m2617_through_convert", "videoscale ! videoconvert"),
    ] {
        check_width(
            id,
            &format!("{scaler} ! video/x-raw,format=NV12,pixel-aspect-ratio=2/1"),
            (6, 8),
        )
        .await;
    }
}

#[tokio::test]
async fn height_and_width_pins_constrain_the_choice() {
    check_width(
        "m2617_fixed_height",
        "videoscale height=16 ! video/x-raw,format=NV12,pixel-aspect-ratio=2/1",
        (12, 16),
    )
    .await;
    check_width(
        "m2617_fixed_width",
        "videoconvertscale ! video/x-raw,format=NV12,width=10,pixel-aspect-ratio=2/1",
        (10, 8),
    )
    .await;
    check_width(
        "m2617_width_property",
        "videoscale width=10 ! video/x-raw,format=NV12,pixel-aspect-ratio=2/1",
        (10, 8),
    )
    .await;
    check_width(
        "m2617_range",
        "videoscale ! video/x-raw,format=NV12,width=[8,10],pixel-aspect-ratio=2/1",
        (8, 8),
    )
    .await;
}

#[tokio::test]
async fn an_input_ratio_change_refixates_width_through_a_converter() {
    check_width(
        "m2617_runtime_aspect",
        "capssetter replace=true caps=video/x-raw,format=NV12,width=12,height=8,framerate=30/1,pixel-aspect-ratio=4/3 \
         ! videoscale ! videoconvert ! video/x-raw,format=NV12,height=8,pixel-aspect-ratio=1/1",
        (16, 8),
    ).await;
}
