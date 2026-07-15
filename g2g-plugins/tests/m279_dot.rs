//! M279: the pipeline -> Graphviz DOT visualizer (`Graph::to_dot`), end to end
//! through the launch parser. A pipeline string is parsed against the standard
//! registry and rendered to DOT, the data behind `g2g-launch --dot`. Asserts the
//! topology, the per-element labels (each node's log category), the structural
//! fallbacks (a tee), and a leaky branch's policy annotation all reach the dump.

#![cfg(feature = "std")]

use g2g_core::runtime::parse_launch;
use g2g_core::DotAnnotations;
use g2g_plugins::registry::default_registry;

/// The closure `g2g-launch --dot` uses: label each node by its element's log
/// category, falling back to the node kind (a tee) for the structural nodes.
fn dump(pipeline: &str) -> String {
    let reg = default_registry();
    let graph = parse_launch(&reg, pipeline).expect("pipeline parses");
    graph.to_dot(
        "pipeline",
        |n| graph.element(n).map(|e| e.log_category().to_string()),
        &DotAnnotations::default(),
    )
}

#[test]
fn linear_pipeline_dumps_nodes_and_edges() {
    let dot = dump("videotestsrc num-buffers=1 ! videoconvert ! fakesink");

    assert!(dot.starts_with("digraph \"pipeline\" {"));
    assert!(dot.trim_end().ends_with('}'));
    // Three element nodes, each labelled by its log category (short type name).
    assert!(dot.contains("label=\"VideoTestSrc\""), "{dot}");
    assert!(dot.contains("label=\"VideoConvert\""), "{dot}");
    assert!(dot.contains("label=\"FakeSink\""), "{dot}");
    // Two links, in order.
    assert!(dot.contains("n0 -> n1"), "{dot}");
    assert!(dot.contains("n1 -> n2"), "{dot}");
    // The source is green, the sink red (the role-coded fills).
    assert!(dot.contains("fillcolor=\"#cde8cd\""), "{dot}");
    assert!(dot.contains("fillcolor=\"#f0cdcd\""), "{dot}");
}

#[test]
fn tee_branch_pads_and_leaky_policy_are_shown() {
    // A tee with a lossless recording branch and a leaky (drop-oldest) preview
    // branch: the queue maps to a per-edge LinkPolicy (there is no Queue
    // element), so the policy must surface on the edge, not as a node.
    let dot = dump(
        "videotestsrc num-buffers=1 ! tee name=t \
         ! fakesink \
         t. ! queue leaky=2 ! fakesink",
    );

    // The tee carries no element, so it falls back to its kind, drawn diamond.
    assert!(dot.contains("label=\"tee\""), "{dot}");
    assert!(dot.contains("shape=diamond"), "{dot}");
    // The second tee output pad is named.
    assert!(dot.contains("taillabel=\"1\""), "{dot}");
    // leaky=2 is drop-oldest; the policy annotates the leaky branch's edge.
    assert!(dot.contains("[DropOldest]"), "leaky branch policy shown: {dot}");
}
