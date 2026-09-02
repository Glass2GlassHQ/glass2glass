//! Splicing a transform onto a live pipeline, on screen (M1120, M1133).
//!
//! `rtspsrc -> ffmpegdec -> waylandsink` plays an RTSP feed in a window. Every
//! few seconds a `videoflip` is spliced onto the decoded-video edge, held, and
//! lifted back off, while the stream keeps running: the picture turns over and
//! back with no gap and no restart. Each operation is printed as it happens.
//!
//! ```sh
//! cd examples/g2g-mutate-demo && cargo run --release
//! ```
//!
//! `cargo run --release -- tee` runs the structural variant (M1133): the
//! decoder feeds a tee with a window on each branch, and the flip is spliced
//! onto one branch by `insert_before` on that branch's sink, so one window
//! turns over while its sibling plays on untouched.
//!
//! `G2G_RTSP_URL` picks the feed (default `rtsp://localhost:8554/pattern`),
//! `G2G_DEMO_SECONDS` ends the run after N seconds instead of waiting for
//! ctrl-c.

use std::time::Duration;

use g2g_core::graph::Graph;
use g2g_core::runtime::{run_graph_mutable, GraphNode, LatencyProfile};
use g2g_core::PipelineClock;
use g2g_plugins::ffmpegdec::{Backend, FfmpegH264Dec, OutputFormat};
use g2g_plugins::rtspsrc::RtspSrc;
use g2g_plugins::videoflip::{Orientation, VideoFlip};
use g2g_plugins::waylandsink::WaylandSink;

/// The feed the mediamtx + ffmpeg recipe in README.md serves.
const DEFAULT_URL: &str = "rtsp://localhost:8554/pattern";
const WINDOW_TITLE: &str = "g2g mutate demo";
/// How long the picture runs untouched, and how long the spliced element stays
/// in before it is lifted out again.
const PHASE: Duration = Duration::from_secs(5);
/// The element the demo splices below: the decoder, so the new element lands on
/// the decoded-video edge feeding the display.
const SPLICE_AFTER: &str = "dec";

/// The display sink paces on the compositor's frame callbacks, so the demo
/// needs no clock of its own.
struct ZeroClock;

impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// The pipeline: an RTSP feed decoded to NV12 and shown in a window. Named
/// nodes, because a mutation addresses the position by element name.
fn build_graph(url: String) -> Graph<GraphNode> {
    let mut graph: Graph<GraphNode> = Graph::new();

    let source = graph.add_source(GraphNode::source(RtspSrc::new(url)));
    graph.set_node_name(source, "src".into());

    // NV12 because that is what `waylandsink` takes, so the decoded edge needs
    // no converter and the splice is caps-preserving.
    let decoder = graph.add_transform(GraphNode::element(
        FfmpegH264Dec::new()
            .with_output_format(OutputFormat::Nv12)
            .with_backend(Backend::Software),
    ));
    graph.set_node_name(decoder, SPLICE_AFTER.into());

    let display = graph.add_sink(GraphNode::element(
        WaylandSink::new().with_title(WINDOW_TITLE),
    ));
    graph.set_node_name(display, "sink".into());

    graph.link(source, decoder).expect("src -> dec");
    graph.link(decoder, display).expect("dec -> sink");
    graph
}

/// The name of the sink whose branch the tee variant splices, addressed with
/// `insert_before`: a tee has several edges below it, so the branch is named by
/// its consumer.
const SPLICE_BEFORE: &str = "flipped";

/// The structural variant: the decoder feeds a tee, each branch its own window.
/// The splice lands on the edge between the tee and the `SPLICE_BEFORE` sink,
/// so only that window turns over.
fn build_tee_graph(url: String) -> Graph<GraphNode> {
    let mut graph: Graph<GraphNode> = Graph::new();

    let source = graph.add_source(GraphNode::source(RtspSrc::new(url)));
    graph.set_node_name(source, "src".into());
    let decoder = graph.add_transform(GraphNode::element(
        FfmpegH264Dec::new()
            .with_output_format(OutputFormat::Nv12)
            .with_backend(Backend::Software),
    ));
    graph.set_node_name(decoder, SPLICE_AFTER.into());
    let tee = graph.add_tee(2);
    let steady = graph.add_sink(GraphNode::element(
        WaylandSink::new().with_title(format!("{WINDOW_TITLE}: steady")),
    ));
    graph.set_node_name(steady, "steady".into());
    let flipped = graph.add_sink(GraphNode::element(
        WaylandSink::new().with_title(format!("{WINDOW_TITLE}: splice target")),
    ));
    graph.set_node_name(flipped, SPLICE_BEFORE.into());

    graph.link(source, decoder).expect("src -> dec");
    graph.link(decoder, tee.input()).expect("dec -> tee");
    graph.link(tee.out(0), steady).expect("tee -> steady");
    graph.link(tee.out(1), flipped).expect("tee -> flipped");
    graph
}

/// Sleeps for `seconds`, or never when the demo is unbounded.
async fn time_limit(seconds: Option<u64>) {
    match seconds {
        Some(s) => tokio::time::sleep(Duration::from_secs(s)).await,
        None => std::future::pending().await,
    }
}

#[tokio::main]
async fn main() {
    g2g_core::log::init_from_env();

    if std::env::var_os("WAYLAND_DISPLAY").is_none() {
        eprintln!("no WAYLAND_DISPLAY: run this inside a Wayland session");
        return;
    }
    let url = std::env::var("G2G_RTSP_URL").unwrap_or_else(|_| DEFAULT_URL.to_string());
    let seconds = std::env::var("G2G_DEMO_SECONDS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok());
    let tee_mode = std::env::args().nth(1).as_deref() == Some("tee");
    if tee_mode {
        println!("playing {url} in two windows fed by one tee");
        println!(
            "splicing a 180-degree videoflip in and out of the {SPLICE_BEFORE:?} branch \
             every {PHASE:?}; the steady window never changes; ctrl-c to stop"
        );
    } else {
        println!("playing {url} in a window titled {WINDOW_TITLE:?}");
        println!("splicing a 180-degree videoflip in and out every {PHASE:?}; ctrl-c to stop");
    }

    let graph = if tee_mode {
        build_tee_graph(url)
    } else {
        build_graph(url)
    };
    let (mutator, run) = run_graph_mutable(graph, &ZeroClock, LatencyProfile::Live.link_capacity());

    // Turn the picture over and back, for as long as the stream lasts. Each
    // operation names the element it addresses and what came of it: a refusal
    // is a legitimate answer (the mutator leaves the graph running and
    // unchanged), so it ends the cycling rather than the run.
    let cycle = async {
        tokio::time::sleep(PHASE).await;
        loop {
            let flip = Box::new(VideoFlip::new(Orientation::Rotate180));
            let (answer, position) = if tee_mode {
                (mutator.insert_before(SPLICE_BEFORE, flip).await, "before")
            } else {
                (mutator.insert_after(SPLICE_AFTER, flip).await, "after")
            };
            let addressed = if tee_mode {
                SPLICE_BEFORE
            } else {
                SPLICE_AFTER
            };
            let spliced = match answer {
                Ok(name) => {
                    println!("insert {position} {addressed}: {name} is now turning the picture");
                    name
                }
                Err(e) => {
                    println!("insert {position} {addressed} refused: {e:?}");
                    break;
                }
            };
            tokio::time::sleep(PHASE).await;
            match mutator.remove(&spliced).await {
                // The element comes back live, still carrying the settings it
                // ran with, and is dropped here.
                Ok(element) => println!(
                    "remove {spliced}: handed back (method={:?}), back to the decoder's own picture",
                    element.get_property("method")
                ),
                Err(e) => {
                    println!("remove {spliced} refused: {e:?}");
                    break;
                }
            }
            tokio::time::sleep(PHASE).await;
        }
        std::future::pending::<()>().await;
    };

    tokio::select! {
        result = run => match result {
            Ok(stats) => println!(
                "stream ended: {} frames decoded, {} shown",
                stats.frames_emitted, stats.frames_consumed
            ),
            Err(e) => println!("pipeline failed: {e:?}"),
        },
        _ = cycle => {}
        _ = tokio::signal::ctrl_c() => println!("ctrl-c: closing the pipeline"),
        _ = time_limit(seconds) => println!("time limit reached: closing the pipeline"),
    }
    // Dropping the run future here ends every arm, which closes the RTSP
    // session and the window with it.
}
