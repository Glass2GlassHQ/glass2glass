//! M849: the visual builder's YAML export loads through the real declarative
//! loader. The fixtures below are the verbatim output of `toYAML` in
//! `tools/builder/src/export.js` (asserted there by `src/import.test.mjs`), so a
//! drift in either spelling of the shared schema fails here.

#![cfg(feature = "declarative-yaml")]

use g2g_plugins::clock::WallClock;
use g2g_plugins::declarative;
use g2g_plugins::registry::default_registry;

use g2g_core::runtime::run_graph;

/// Exported from a three-node linear canvas with `num-buffers` set on the source.
const LINEAR: &str = r#"nodes:
  - id: "videotestsrc0"
    element: "videotestsrc"
    props:
      num-buffers: "3"
  - id: "videoconvert0"
    element: "videoconvert"
  - id: "fakesink0"
    element: "fakesink"
edges:
  - from: "videotestsrc0"
    to: "videoconvert0"
  - from: "videoconvert0"
    to: "fakesink0"
"#;

/// Exported from a fan-in canvas: two sources into a funnel, then a sink.
const FANIN: &str = r#"nodes:
  - id: "videotestsrc0"
    element: "videotestsrc"
    props:
      num-buffers: "2"
  - id: "videotestsrc1"
    element: "videotestsrc"
    props:
      num-buffers: "2"
  - id: "funnel0"
    element: "funnel"
  - id: "fakesink0"
    element: "fakesink"
edges:
  - from: "videotestsrc0"
    to: "funnel0"
  - from: "videotestsrc1"
    to: "funnel0"
  - from: "funnel0"
    to: "fakesink0"
"#;

/// The linear export builds and runs: the quoted `num-buffers: "3"` is typed by
/// the element's property spec, so exactly 3 frames flow.
#[tokio::test]
async fn linear_yaml_export_runs() {
    let reg = default_registry();
    let graph = declarative::from_yaml(&reg, LINEAR).expect("builder YAML loads");
    let clock = WallClock::new();
    let stats = run_graph(graph, &clock, 4).await.expect("run");
    assert_eq!(stats.frames_consumed, 3);
}

/// The fan-in export resolves the two-input node to a registered muxer and links
/// both arms onto its pads.
#[test]
fn fanin_yaml_export_builds_a_muxer_graph() {
    let reg = default_registry();
    let graph = declarative::from_yaml(&reg, FANIN).expect("builder YAML loads");
    assert_eq!(graph.edges().len(), 3, "two inputs + the sink link");
    graph.finish().expect("valid DAG");
}
