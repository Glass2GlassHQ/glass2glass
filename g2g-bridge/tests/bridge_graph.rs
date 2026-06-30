//! `BridgeGraph`: an embedded g2g sub-graph driven from synchronous code, the
//! cross-thread push/pull path a GStreamer `chain` function uses (DESIGN.md §7).
//!
//! `default_registry` (and the bridge) are `std`-gated, so this file is too.
#![cfg(feature = "std")]

use g2g_bridge::{frame_bytes, BridgeError, BridgeGraph};

const CAPS: &str = "video/x-raw,format=RGBA,width=2,height=2,framerate=30/1";

/// The buffers an embedder pushes flow through the sub-graph and come back out,
/// with timestamps preserved, across the thread boundary: the graph runs on its
/// own OS thread while the test pushes and drains from this one.
#[test]
fn round_trips_buffers_across_the_thread_boundary() {
    let bridge = BridgeGraph::new("identity", CAPS).expect("appsrc ! identity ! appsink builds");

    // Push three distinct 2x2 RGBA buffers from this thread.
    for i in 0u8..3 {
        assert!(bridge.push(&[i; 16], u64::from(i) * 1_000), "feed accepted buffer {i}");
    }
    bridge.end_of_stream();

    // Drain them back on this thread; the graph produced them on its own.
    let mut out = Vec::new();
    while let Some(frame) = bridge.pull_blocking() {
        let bytes = frame_bytes(&frame).expect("system-memory frame").to_vec();
        out.push((bytes, frame.timing.pts_ns));
    }

    assert_eq!(out.len(), 3, "every pushed buffer came back");
    assert_eq!(out[0].0, vec![0u8; 16], "bytes round-tripped through the sub-graph");
    assert_eq!(out[1].1, 1_000, "presentation timestamp carried through");

    let stats = bridge.finish().expect("clean shutdown");
    assert_eq!(stats.frames_consumed, 3, "sink consumed every frame");
}

/// A real caps-driven transform (not just a pass-through) runs inside the
/// sub-graph and its output reaches the drain. This exercises the path where the
/// runner cascades caps a second time through the embedded graph (a format/size
/// transform), which must not strand the frame at the `appsink`.
#[test]
fn caps_driven_transform_delivers_output() {
    let bridge = BridgeGraph::new("videoconvert", CAPS).expect("appsrc ! videoconvert ! appsink");
    assert!(bridge.push(&[42u8; 16], 0));
    bridge.end_of_stream();

    let mut frames = 0;
    while let Some(frame) = bridge.pull_blocking() {
        assert!(frame_bytes(&frame).is_some(), "system-memory output");
        frames += 1;
    }
    assert_eq!(frames, 1, "the transformed frame reached the drain");
}

/// A rescaling fragment changes the buffer size: `with_output_caps` pins the
/// sub-graph's output, and the drained frame is the smaller output size, not the
/// input size. (The GStreamer shell relies on this to allocate output buffers.)
#[test]
fn rescaling_fragment_changes_output_size() {
    let in_caps = "video/x-raw,format=RGBA,width=8,height=8,framerate=30/1"; // 8*8*4 = 256
    let out_caps = "video/x-raw,format=RGBA,width=4,height=4,framerate=30/1"; // 4*4*4 = 64
    let bridge =
        BridgeGraph::with_output_caps("videoscale", in_caps, out_caps).expect("scale sub-graph");
    assert!(bridge.push(&[9u8; 256], 0));
    bridge.end_of_stream();

    let mut out_lens = Vec::new();
    while let Some(frame) = bridge.pull_blocking() {
        out_lens.push(frame_bytes(&frame).expect("system memory").len());
    }
    assert_eq!(out_lens, vec![64], "the downscaled frame is 4x4 RGBA, not the 8x8 input");
}

/// A fragment that names an element g2g lacks fails construction with a parse
/// error (carrying the launch diagnostics / porting hint), not a panic or a hung
/// thread. This is the feedback an app developer gets while porting.
#[test]
fn unknown_element_fails_to_build() {
    let err = BridgeGraph::new("x264enc", CAPS).expect_err("no SW H.264 encoder in g2g");
    assert!(matches!(err, BridgeError::Parse(_)), "surfaced as a parse error: {err}");
}

/// Dropping a `BridgeGraph` without draining must not deadlock: releasing the
/// pull handle lets the sink discard undeliverable frames so the run thread can
/// reach EOS and be joined. (If this regressed, the test would hang.)
#[test]
fn drop_without_draining_does_not_deadlock() {
    let bridge = BridgeGraph::new("identity", CAPS).expect("builds");
    for i in 0u8..3 {
        bridge.push(&[i; 16], 0);
    }
    bridge.end_of_stream();
    drop(bridge); // joins the run thread in Drop; must return.
}
