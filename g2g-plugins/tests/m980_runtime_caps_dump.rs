//! M980 runtime caps dump: `g2g-launch --run-json` runs the line to EOS and
//! prints the `--validate-json` shape with each edge's caps as observed while it
//! ran, so a stream whose geometry only arrives with the data reports what it
//! really carried instead of the solver's startup placeholder.
//!
//! Run with `cargo test -p g2g-plugins --features tooling-json --test
//! m980_runtime_caps_dump`.
#![cfg(feature = "tooling-json")]

use g2g_core::runtime::{run_graph_observed, Observer};
use g2g_core::PipelineClock;
use g2g_plugins::registry::default_registry;
use g2g_plugins::toolingjson::{observed_graph_json, validate_json};

/// An H.264 clip that switches resolution mid-stream, so the parser refines its
/// output caps twice while running.
const RECONFIG_CLIP: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/h264_reconfig_640x480_to_320x240.h264"
);

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn line() -> String {
    format!("filesrc location={RECONFIG_CLIP} ! h264parse ! fakesink")
}

/// The solve-time dump can only offer the placeholder geometry the parser
/// advertises before it has seen an SPS; the run dump replaces it with the caps
/// the frames crossed under, and says which edge is which.
#[tokio::test]
async fn run_dump_reports_refined_caps_where_negotiation_had_a_placeholder() {
    let reg = default_registry();
    let solved = validate_json(&reg, &line()).await;
    assert_eq!(solved["ok"], true, "{solved}");
    let solved_parse_out = solved["edges"][1]["caps"].as_str().unwrap().to_string();
    assert!(
        solved_parse_out.contains("width=16") && solved_parse_out.contains("height=16"),
        "negotiation has only the placeholder geometry, got {solved_parse_out}"
    );

    let ran = observed_graph_json(&reg, &line()).await;
    assert_eq!(ran["ok"], true, "{ran}");

    // Same shape as the solve-time dump: three nodes, two edges, same wiring.
    let nodes = ran["nodes"].as_array().unwrap();
    assert_eq!(nodes.len(), 3);
    for (index, node) in nodes.iter().enumerate() {
        assert_eq!(node["index"], index);
    }
    assert_eq!(nodes[1]["name"], "NalParse0");
    let edges = ran["edges"].as_array().unwrap();
    assert_eq!(
        edges
            .iter()
            .map(|e| (e["from"].as_u64().unwrap(), e["to"].as_u64().unwrap()))
            .collect::<Vec<_>>(),
        vec![(0, 1), (1, 2)],
    );

    // The parser's output link carried the clip's final geometry, not the
    // placeholder, and is marked as read from the run.
    let observed = edges[1]["caps"].as_str().unwrap();
    assert_eq!(edges[1]["caps_source"], "runtime");
    assert!(
        observed.contains("width=320") && observed.contains("height=240"),
        "the parser refined to the clip's second resolution, got {observed}"
    );

    // Nothing refines the byte stream feeding the parser, so that edge honestly
    // reports the negotiated caps rather than claiming a runtime reading.
    assert_eq!(edges[0]["caps_source"], "negotiated");
    assert_eq!(edges[0]["caps"].as_str().unwrap(), solved_parse_out);
}

/// The dump reads the observer, so the same refinement is visible on a live
/// snapshot: the edge keeps its negotiated caps *and* the last ones that
/// crossed, and they differ once the stream refines.
#[tokio::test]
async fn observer_keeps_negotiated_and_observed_caps_apart() {
    let reg = default_registry();
    let graph = g2g_core::runtime::parse_launch(&reg, &line()).expect("pipeline parses");
    let obs = Observer::new();
    let stats = run_graph_observed(graph, &ZeroClock, 4, &obs, None)
        .await
        .expect("observed run");
    assert!(stats.frames_consumed > 1, "the clip decoded some frames");

    let snap = obs.snapshot();
    let parse_out = &snap.edges[1];
    let negotiated = parse_out
        .caps
        .as_deref()
        .expect("edge carries the solution");
    let observed = parse_out
        .observed_caps
        .as_deref()
        .expect("a CapsChanged crossed the parser's output link");
    assert!(negotiated.contains("width=16"), "{negotiated}");
    assert!(observed.contains("width=320"), "{observed}");

    // The source's byte-stream link never sees a CapsChanged, so it has no
    // runtime reading to report.
    assert!(snap.edges[0].observed_caps.is_none());
}
