// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "analytics")]

use g2g_core::runtime::{parse_launch, run_graph, Registry};
use g2g_plugins::clock::WallClock;
use pyo3::prelude::*;

const FIXTURE_MODULE: &str = "framerate_element";
const FRAME_COUNT: usize = 3;
const LINK_CAPACITY: usize = 4;

fn registry() -> Registry {
    static SET: std::sync::Once = std::sync::Once::new();
    SET.call_once(|| {
        std::env::set_var(
            "PYTHONPATH",
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"),
        );
    });
    let mut registry = g2g_plugins::registry::default_registry();
    g2g_python::register(&mut registry);
    registry
}

fn framerates_seen(caps: &str) -> Vec<Option<(u32, u32)>> {
    let line = format!(
        "videotestsrc num-buffers={FRAME_COUNT} ! {caps} ! \
         pyelement module={FIXTURE_MODULE} class=FramerateRecorder ! fakesink"
    );
    let graph = parse_launch(&registry(), &line).expect("the launch line parses");
    let clock = WallClock::new();
    tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .unwrap()
        .block_on(run_graph(graph, &clock, LINK_CAPACITY))
        .expect("the run completes");
    Python::attach(|py| {
        let seen = PyModule::import(py, FIXTURE_MODULE)?.getattr("FRAMERATES_SEEN")?;
        let values: Vec<Option<(u32, u32)>> = seen.extract()?;
        seen.call_method0("clear")?;
        Ok::<_, PyErr>(values)
    })
    .expect("the fixture records what framerate said")
}

#[test]
fn meta_reports_the_negotiated_framerate() {
    assert_eq!(
        framerates_seen("video/x-raw,format=RGBA,width=64,height=48,framerate=50/1"),
        vec![Some((50, 1)); FRAME_COUNT]
    );
    assert_eq!(
        framerates_seen("video/x-raw,format=RGBA,width=64,height=48,framerate=25/2"),
        vec![Some((25, 2)); FRAME_COUNT]
    );
}
