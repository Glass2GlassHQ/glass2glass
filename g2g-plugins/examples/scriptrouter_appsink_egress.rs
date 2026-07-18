//! Live appsink egress: a Rhai `scriptrouter` splits one stream onto two
//! `appsink` pull channels, and the application pulls each channel *live* while
//! the pipeline runs, observing real per-frame delivery timing. This is the
//! "route buffers into my own code" seam, end to end.
//!
//!   cargo run -p g2g-plugins --features script-rhai --example scriptrouter_appsink_egress
//!
//! The graph is built programmatically; the `gst-launch` equivalent is:
//!
//!   videotestsrc num-buffers=8 ! scriptrouter name=r \
//!       script="fn route(f){ f.sequence % 2 }" \
//!     r. ! appsink channel=even   r. ! appsink channel=odd
//!
//! `route()` sends even-sequence frames to port 0 (channel "even") and odd to
//! port 1 (channel "odd"). Two consumer loops `pull()` from their channel as
//! frames arrive: each blocks until the next frame is delivered, so the printed
//! timestamps trace the pipeline's real cadence, not a post-hoc drain.
//!
//! The same pattern feeds any application (a Python `AppSinkPull` over the C ABI,
//! an inference loop, a file writer). Swap the `route()` body to fan out by
//! content (`f.keyframe`, `f[0]`), or return an array (`[0, 1]`) to *multicast*
//! one frame to several consumers.

#[cfg(feature = "script-rhai")]
fn main() {
    use std::time::Instant;

    use g2g_core::graph::Graph;
    use g2g_core::runtime::{run_graph, GraphNode};
    use g2g_core::PipelineClock;
    use g2g_plugins::appsink::{register_appsink_pull, AppSink};
    use g2g_plugins::script::ScriptRouter;
    use g2g_plugins::videotestsrc::VideoTestSrc;

    // Non-live authoring clock: no wall-clock pacing, the source runs flat out so
    // the timing we print is the pipeline's own throughput.
    struct Immediate;
    impl PipelineClock for Immediate {
        fn now_ns(&self) -> u64 {
            0
        }
    }

    const N: u64 = 8;

    // Register the two pull channels BEFORE launch; each `appsink` claims its
    // channel by name at configure time.
    let even = register_appsink_pull("even");
    let odd = register_appsink_pull("odd");

    // videotestsrc ! scriptrouter(parity) -> { appsink even, appsink odd }.
    let router = ScriptRouter::new(2).with_script("fn route(f) { f.sequence % 2 }");
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(VideoTestSrc::new(4, 4, 30, N)));
    let r = g.add_demux(GraphNode::demux(router), 2);
    let s0 = g.add_sink(GraphNode::element(AppSink::new().with_channel("even")));
    let s1 = g.add_sink(GraphNode::element(AppSink::new().with_channel("odd")));
    g.link(src, r.input())
        .expect("link videotestsrc -> scriptrouter");
    g.link(r.out(0), s0)
        .expect("link scriptrouter.0 -> appsink even");
    g.link(r.out(1), s1)
        .expect("link scriptrouter.1 -> appsink odd");

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    rt.block_on(async {
        let t0 = Instant::now();

        // A consumer loop per channel: block on the next frame, print when it
        // lands, stop when the channel ends (EOS). Returns the count pulled.
        async fn drain(name: &str, pull: g2g_plugins::appsink::AppSinkPull, t0: Instant) -> usize {
            let mut n = 0usize;
            while let Some(f) = pull.pull().await {
                n += 1;
                println!(
                    "[{name:>4}] frame #{n} seq={} pts={}ns   arrived +{:?}",
                    f.sequence,
                    f.timing.pts_ns,
                    t0.elapsed()
                );
            }
            println!("[{name:>4}] end of stream after {n} frames");
            n
        }

        // Run the pipeline and both consumers concurrently on this one task: the
        // cooperative runner pushes each routed frame into its channel, and the
        // matching consumer wakes to pull it. Interleaving them is what makes the
        // pull timing live rather than a batch read after the run.
        let (stats, ne, no) = tokio::join!(
            run_graph(g, &Immediate, 4),
            drain("even", even, t0),
            drain("odd", odd, t0),
        );
        let stats = stats.expect("pipeline runs to completion");

        println!(
            "\n{} frames consumed: even={ne} odd={no}",
            stats.frames_consumed
        );
        assert_eq!(
            ne + no,
            N as usize,
            "every frame reached exactly one consumer"
        );
        assert_eq!(ne, no, "parity split is even for {N} frames");
    });
}

#[cfg(not(feature = "script-rhai"))]
fn main() {
    eprintln!("rebuild with --features script-rhai to run the scriptrouter egress demo");
}
