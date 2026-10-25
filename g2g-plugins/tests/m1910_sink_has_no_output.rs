// a sink has no output pad, so every way of building a graph refuses to link one to a downstream element

use g2g_core::log::{self, LogLevel, RingSink};
use g2g_core::runtime::{block_on, parse_launch, run_graph, GraphNode, ParseError};
use g2g_core::{G2gError, Graph};
use g2g_plugins::clock::WallClock;
use g2g_plugins::declarative::{self, SpecError};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::registry::default_registry;
use g2g_plugins::videotestsrc::VideoTestSrc;

const FRAMES: u64 = 4;
const LINK_CAPACITY: usize = 4;

#[test]
fn a_launch_line_linking_after_a_sink_fails_to_parse() {
    let registry = default_registry();
    let error = parse_launch(
        &registry,
        "videotestsrc num-buffers=4 ! fakesink ! videoconvert ! fakesink",
    )
    .expect_err("a sink cannot feed a downstream element");
    assert_eq!(
        error,
        ParseError::LinkFromSink {
            sink: "fakesink".to_string(),
            downstream: "videoconvert".to_string(),
        }
    );
}

#[test]
fn a_declarative_graph_linking_after_a_sink_fails_to_build() {
    let registry = default_registry();
    let json = r#"{
        "nodes": [
            { "id": "src",   "element": "videotestsrc", "props": { "num-buffers": 4 } },
            { "id": "first", "element": "fakesink" },
            { "id": "last",  "element": "fakesink" }
        ],
        "edges": [ { "from": "src", "to": "first" }, { "from": "first", "to": "last" } ]
    }"#;
    let error =
        declarative::from_json(&registry, json).expect_err("a sink cannot feed a downstream node");
    assert!(
        matches!(
            &error,
            SpecError::LinkFromSink { node, downstream } if node == "first" && downstream == "last"
        ),
        "the error names the sink and its consumer: {error}"
    );
}

#[test]
fn a_built_graph_linking_after_a_sink_fails_to_run_and_names_the_sink() {
    let ring = RingSink::new(64);
    log::set_sink(Box::new(ring.clone()));
    log::set_default_level(LogLevel::Error);

    let mut graph: Graph<GraphNode> = Graph::new();
    let source = graph.add_source(GraphNode::source(VideoTestSrc::new(64, 64, 30, FRAMES)));
    let first = graph.add_transform(GraphNode::element(FakeSink::new()));
    let last = graph.add_sink(GraphNode::element(FakeSink::new()));
    graph.set_node_name(first, "first".to_string());
    graph.link(source, first).expect("link");
    graph.link(first, last).expect("link");

    let result = block_on(run_graph(graph, &WallClock::new(), LINK_CAPACITY));
    let messages: Vec<String> = ring.drain().into_iter().map(|r| r.message).collect();
    log::reset();
    assert_eq!(result.err(), Some(G2gError::CapsMismatch));
    assert!(
        messages
            .iter()
            .any(|m| m.starts_with("first is a sink and has no output")),
        "the run names the sink: {messages:?}"
    );
}
