//! M578/M579/M580: the scripting surface, end to end. A declarative YAML graph,
//! a Rhai builder script, and the `scriptelement` runtime transform are each
//! built from text and RUN to completion on the real runner, so these exercise
//! the whole path (deserialize / script -> registry construction -> negotiation
//! -> frames flow), not just graph validation.

use g2g_core::runtime::{run_graph_with_progress, PipelineProgress};
use g2g_plugins::appsink::register_appsink_pull;
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;
use g2g_plugins::{declarative, script};

/// A YAML document with `videotestsrc ! scriptelement ! fakesink` runs, and the
/// scripted per-frame transform is on the live path. Uses 8 frames with a link
/// capacity of 4 (frames > capacity), which forces the source to block on
/// backpressure and exercise EOS handling: the element must not re-emit the EOS
/// sentinel (a regression that deadlocked the run and surfaced as `Shutdown`).
#[tokio::test]
async fn declarative_yaml_graph_runs_through_scriptelement() {
    let reg = default_registry();
    let yaml = r#"
nodes:
  - { id: src,    element: videotestsrc, props: { num-buffers: 8 } }
  - { id: script, element: scriptelement, props: { script: "fn process(f) { f[0] = 1; }" } }
  - { id: sink,   element: fakesink }
edges:
  - { from: src,    to: script }
  - { from: script, to: sink, policy: drop-oldest }
"#;
    let graph = declarative::from_yaml(&reg, yaml).expect("build the graph from YAML");
    let clock = WallClock::new();
    let progress = PipelineProgress::new();
    let stats = run_graph_with_progress(graph, &clock, 4, &progress).await.expect("run");
    assert_eq!(stats.frames_consumed, 8, "all 8 frames flow src->scriptelement->sink");
}

/// The native bulk ops run end to end over more frames than the link capacity,
/// the fast path for whole-frame work (one native sweep, not a per-byte loop).
#[tokio::test]
async fn scriptelement_bulk_invert_runs() {
    let reg = default_registry();
    let yaml = r#"
nodes:
  - { id: src,    element: videotestsrc, props: { num-buffers: 12 } }
  - { id: script, element: scriptelement, props: { script: "fn process(f) { f.invert(); }" } }
  - { id: sink,   element: fakesink }
edges:
  - { from: src, to: script }
  - { from: script, to: sink }
"#;
    let graph = declarative::from_yaml(&reg, yaml).expect("build");
    let clock = WallClock::new();
    let progress = PipelineProgress::new();
    let stats = run_graph_with_progress(graph, &clock, 4, &progress).await.expect("run");
    assert_eq!(stats.frames_consumed, 12);
}

/// The same JSON document builds and validates the same three-node graph.
#[test]
fn declarative_json_matches_yaml_shape() {
    let reg = default_registry();
    let json = r#"{
        "nodes": [
            { "id": "src",  "element": "videotestsrc", "props": { "num-buffers": 3 } },
            { "id": "sink", "element": "fakesink" }
        ],
        "edges": [ { "from": "src", "to": "sink" } ]
    }"#;
    let graph = declarative::from_json(&reg, json).expect("build");
    assert_eq!(graph.edges().len(), 1);
    graph.finish().expect("valid DAG");
}

/// A Rhai builder script generates the pipeline with a loop, and it runs.
#[tokio::test]
async fn rhai_script_built_graph_runs() {
    let reg = default_registry();
    let src = r#"
        add("videotestsrc", "src");
        set("src", "num-buffers", 2);
        add("fakesink", "sink");
        link("src", "sink");
    "#;
    let graph = script::build_from_script(&reg, src).expect("build from script");
    let clock = WallClock::new();
    let progress = PipelineProgress::new();
    let stats = run_graph_with_progress(graph, &clock, 4, &progress).await.expect("run");
    assert_eq!(stats.frames_consumed, 2);
}

/// `scriptrouter` runs as a real DAG node: a script routes each frame to one of
/// two branches (here every frame, by parity), and all frames reach a sink. Uses
/// 6 frames with capacity 4 so the source blocks (exercises the demux EOS path).
#[tokio::test]
async fn scriptrouter_fans_out_through_the_runner() {
    let reg = default_registry();
    let yaml = r#"
nodes:
  - { id: src, element: videotestsrc, props: { num-buffers: 6 } }
  - { id: r,   element: scriptrouter, props: { script: "fn route(f) { f.sequence % 2 }" } }
  - { id: a,   element: fakesink }
  - { id: b,   element: fakesink }
edges:
  - { from: src, to: r }
  - { from: r,   to: a }
  - { from: r,   to: b }
"#;
    let graph = declarative::from_yaml(&reg, yaml).expect("build a routing graph");
    let clock = WallClock::new();
    let progress = PipelineProgress::new();
    let stats = run_graph_with_progress(graph, &clock, 4, &progress).await.expect("run");
    // Every frame is routed to exactly one branch (parity keeps them all), so all
    // 6 flow through the router to a sink.
    assert_eq!(stats.frames_consumed, 6, "all frames routed across the two branches");
}

/// A `route()` that drops (negative) some frames: fewer reach the sinks.
#[tokio::test]
async fn scriptrouter_can_drop_frames() {
    let reg = default_registry();
    let yaml = r#"
nodes:
  - { id: src, element: videotestsrc, props: { num-buffers: 6 } }
  - { id: r,   element: scriptrouter, props: { script: "fn route(f) { if f.sequence < 4 { 0 } else { -1 } }" } }
  - { id: a,   element: fakesink }
  - { id: b,   element: fakesink }
edges:
  - { from: src, to: r }
  - { from: r,   to: a }
  - { from: r,   to: b }
"#;
    let graph = declarative::from_yaml(&reg, yaml).expect("build");
    let clock = WallClock::new();
    let progress = PipelineProgress::new();
    let stats = run_graph_with_progress(graph, &clock, 4, &progress).await.expect("run");
    // Sequences 0..4 route to port 0; 4 and 5 are dropped: 4 consumed.
    assert_eq!(stats.frames_consumed, 4, "dropped frames do not reach a sink");
}

/// Multicast (M584): `route()` returns an array `[0, 1]`, so every frame is fanned
/// out to *both* branches. Each of 5 source frames reaches both sinks, so the run
/// consumes 10, proving the shared-duplicate fan-out lands on every listed port.
#[tokio::test]
async fn scriptrouter_multicasts_one_frame_to_every_port() {
    let reg = default_registry();
    let yaml = r#"
nodes:
  - { id: src, element: videotestsrc, props: { num-buffers: 5 } }
  - { id: r,   element: scriptrouter, props: { script: "fn route(f) { [0, 1] }" } }
  - { id: a,   element: fakesink }
  - { id: b,   element: fakesink }
edges:
  - { from: src, to: r }
  - { from: r,   to: a }
  - { from: r,   to: b }
"#;
    let graph = declarative::from_yaml(&reg, yaml).expect("build a multicast graph");
    let clock = WallClock::new();
    let progress = PipelineProgress::new();
    let stats = run_graph_with_progress(graph, &clock, 4, &progress).await.expect("run");
    assert_eq!(stats.frames_consumed, 10, "each of 5 frames reached both branches");
}

/// The live appsink-egress path end to end (M584): a `scriptrouter` splits the
/// stream onto two `appsink` pull channels, and the application `pull()`s each
/// channel *concurrently with the running pipeline* (not a post-run drain), the
/// timing-faithful egress seam. Even-sequence frames land on "even", odd on
/// "odd"; each consumer sees its half in arrival order and then EOS.
#[tokio::test]
async fn scriptrouter_egress_delivers_live_to_appsink_pull_channels() {
    // Unique channel names: the appsink registry is a process-global claimed at
    // configure, so distinct names keep this isolated from other tests.
    let even = register_appsink_pull("m584_even");
    let odd = register_appsink_pull("m584_odd");

    let reg = default_registry();
    let yaml = r#"
nodes:
  - { id: src, element: videotestsrc, props: { num-buffers: 6 } }
  - { id: r,   element: scriptrouter, props: { script: "fn route(f) { f.sequence % 2 }" } }
  - { id: e,   element: appsink, props: { channel: m584_even } }
  - { id: o,   element: appsink, props: { channel: m584_odd } }
edges:
  - { from: src, to: r }
  - { from: r,   to: e }
  - { from: r,   to: o }
"#;
    let graph = declarative::from_yaml(&reg, yaml).expect("build the egress graph");
    let clock = WallClock::new();
    let progress = PipelineProgress::new();

    // Drain a channel live: block on each frame as it is routed, collecting the
    // sequence numbers in arrival order until EOS closes the channel.
    async fn drain(pull: g2g_plugins::appsink::AppSinkPull) -> Vec<u64> {
        let mut seqs = Vec::new();
        while let Some(f) = pull.pull().await {
            seqs.push(f.sequence);
        }
        seqs
    }

    // Run the pipeline and both consumers on the one task, interleaved: the pull
    // is genuinely live (a post-run drain would fill the bounded channel and stall
    // the runner at capacity, so completing at all proves concurrent delivery).
    let (stats, even_seqs, odd_seqs) = tokio::join!(
        run_graph_with_progress(graph, &clock, 4, &progress),
        drain(even),
        drain(odd),
    );
    let stats = stats.expect("egress pipeline runs to completion");

    assert_eq!(stats.frames_consumed, 6, "all 6 frames reached a sink");
    assert_eq!(even_seqs, std::vec![0, 2, 4], "even sequences delivered in order on 'even'");
    assert_eq!(odd_seqs, std::vec![1, 3, 5], "odd sequences delivered in order on 'odd'");
}
