//! M485 per-edge queue depth: a `queue max-size-buffers=N` in a `gst-launch`
//! line sets the depth of the edge it contracts to (the gst per-queue buffer
//! bound), overriding the runner's graph-wide `link_capacity` for just that link.

use g2g_core::runtime::{parse_launch, LaunchFactory, Registry, SourceFactory};
use g2g_core::{Caps, Dim, Rate, RawVideoFormat};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::videotestsrc::VideoTestSrc;

fn rgba_any() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

fn registry() -> Registry {
    let mut reg = Registry::new();
    reg.register_source(SourceFactory::new("videotestsrc", rgba_any(), || {
        Box::new(VideoTestSrc::new(64, 48, 30, 0))
    }));
    reg.register_launch(LaunchFactory::new("fakesink", Vec::new(), || Box::new(FakeSink::new())));
    reg
}

#[test]
fn queue_max_size_buffers_sets_edge_depth() {
    let reg = registry();
    let g = parse_launch(&reg, "videotestsrc ! queue max-size-buffers=2 ! fakesink").unwrap();
    // The queue contracts out; the src->sink edge carries its depth.
    let caps: Vec<Option<usize>> = g.edges().iter().map(|e| e.capacity).collect();
    assert!(caps.contains(&Some(2)), "an edge takes the queue depth: {caps:?}");
}

#[test]
fn plain_queue_leaves_edge_depth_default() {
    let reg = registry();
    // No `max-size-buffers` -> capacity stays None (runner uses link_capacity).
    let g = parse_launch(&reg, "videotestsrc ! queue leaky=2 ! fakesink").unwrap();
    assert!(g.edges().iter().all(|e| e.capacity.is_none()), "no explicit depth");
    // `max-size-buffers=0` (gst "unbounded") is ignored, not a zero-depth channel.
    let g = parse_launch(&reg, "videotestsrc ! queue max-size-buffers=0 ! fakesink").unwrap();
    assert!(g.edges().iter().all(|e| e.capacity.is_none()), "unbounded (0) ignored");
}
