//! gst-launch <-> g2g worked equivalents.
//!
//! `cargo run -p g2g-plugins --features std --example gst_equivalents`
//!
//! For a GStreamer user, porting an app is mostly translating a `gst-launch`
//! line into the typed Rust graph API. Each recipe here runs the SAME pipeline
//! two ways and asserts they produce the same frame count:
//!
//!   1. the `gst-launch` text, through `parse_launch` (what `g2g-launch` does), and
//!   2. the equivalent graph built by hand with `Graph` + `GraphNode`.
//!
//! So the left column is the line you already know; the right column is the code
//! it maps to. All recipes use baseline elements, so this runs with just `std`.

#[cfg(not(feature = "std"))]
fn main() {
    eprintln!("run with --features std: cargo run -p g2g-plugins --features std --example gst_equivalents");
}

#[cfg(feature = "std")]
fn main() {
    use g2g_core::graph::Graph;
    use g2g_core::runtime::{parse_launch, run_graph, GraphNode};
    use g2g_core::{Caps, Dim, PipelineClock, Rate, RawVideoFormat};
    use g2g_plugins::capsfilter::CapsFilter;
    use g2g_plugins::fakesink::FakeSink;
    use g2g_plugins::registry::default_registry;
    use g2g_plugins::videoconvert::VideoConvert;
    use g2g_plugins::videotestsrc::VideoTestSrc;

    // File-to-fake work is non-live, so a zero clock (no pacing) is enough.
    struct ZeroClock;
    impl PipelineClock for ZeroClock {
        fn now_ns(&self) -> u64 {
            0
        }
    }

    let reg = default_registry();
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().expect("tokio rt");

    // Run a gst-launch line through the text parser (the `g2g-launch` path).
    let run_text = |line: &str| -> u64 {
        let g = parse_launch(&reg, line).expect("gst-launch line parses");
        rt.block_on(run_graph(g, &ZeroClock, 4)).expect("text pipeline runs").frames_consumed
    };
    // Run a hand-built typed graph.
    let run_typed = |g: Graph<GraphNode>| -> u64 {
        rt.block_on(run_graph(g, &ZeroClock, 4)).expect("typed pipeline runs").frames_consumed
    };

    // A recipe: report the two runs and assert they agree.
    let report = |title: &str, line: &str, text: u64, typed: u64| {
        println!("recipe: {title}");
        println!("  gst-launch-1.0 {line}");
        println!("  text : {text} frames");
        println!("  typed: {typed} frames");
        assert_eq!(text, typed, "text and typed builds must be equivalent: {title}");
        println!("  -> equivalent\n");
    };

    // 1. A basic transform chain. `videoconvert` is caps-driven (`::auto`), the
    //    same as a bare `videoconvert` in the text line.
    {
        let line = "videotestsrc num-buffers=3 ! videoconvert ! fakesink";
        let mut g: Graph<GraphNode> = Graph::new();
        let src = g.add_source(GraphNode::source(VideoTestSrc::new(320, 240, 30, 3)));
        let cvt = g.add_transform(GraphNode::element(VideoConvert::auto()));
        let sink = g.add_sink(GraphNode::element(FakeSink::new()));
        g.link(src, cvt).unwrap();
        g.link(cvt, sink).unwrap();
        report("basic transform chain", line, run_text(line), run_typed(g));
    }

    // 2. An inline caps filter. `video/x-raw,format=NV12` in the text becomes a
    //    `CapsFilter`, and `videoconvert` converts to NV12 driven by it.
    {
        let line = "videotestsrc num-buffers=3 ! videoconvert ! video/x-raw,format=NV12 ! fakesink";
        let nv12 = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        let mut g: Graph<GraphNode> = Graph::new();
        let src = g.add_source(GraphNode::source(VideoTestSrc::new(320, 240, 30, 3)));
        let cvt = g.add_transform(GraphNode::element(VideoConvert::auto()));
        let filt = g.add_transform(GraphNode::element(CapsFilter::new(nv12)));
        let sink = g.add_sink(GraphNode::element(FakeSink::new()));
        g.link(src, cvt).unwrap();
        g.link(cvt, filt).unwrap();
        g.link(filt, sink).unwrap();
        report("inline caps filter", line, run_text(line), run_typed(g));
    }

    // 3. A `tee` fan-out. The typed graph adds a tee with two output pads and
    //    links each to a sink; both branches see every frame (3 x 2 = 6). In g2g
    //    the text `tee` is optional (a bare fan-out auto-inserts one), but here we
    //    build it explicitly to mirror the canonical GStreamer line.
    {
        let line = "videotestsrc num-buffers=3 ! tee name=t ! fakesink t. ! fakesink";
        let mut g: Graph<GraphNode> = Graph::new();
        let src = g.add_source(GraphNode::source(VideoTestSrc::new(320, 240, 30, 3)));
        let tee = g.add_tee(2);
        let sink0 = g.add_sink(GraphNode::element(FakeSink::new()));
        let sink1 = g.add_sink(GraphNode::element(FakeSink::new()));
        g.link(src, tee.input()).unwrap();
        g.link(tee.out(0), sink0).unwrap();
        g.link(tee.out(1), sink1).unwrap();
        report("tee fan-out to two sinks", line, run_text(line), run_typed(g));
    }

    println!("all recipes: gst-launch text == typed graph");
}
