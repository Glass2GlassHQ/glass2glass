//! M190: `queue` / `queue2` in a gst-launch line. g2g has no queue element by
//! design, per-edge `LinkPolicy` is the leaky-queue analog, so a `queue` node
//! collapses into the backpressure policy of the edge it sits on: `leaky=`
//! becomes `Block` / `DropOldest` / `DropNewest`, and the queue adds no element
//! hop. This keeps real gst command lines (which sprinkle `queue` liberally)
//! parsing and running.
#![cfg(feature = "std")]

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{LinkPolicy, PipelineClock};
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

async fn run_line(line: &str) -> u64 {
    let reg = default_registry();
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("{line:?} should parse: {e:?}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("{line:?} should run: {e:?}"))
        .frames_consumed
}

/// The single edge policy of a parsed (still linear) line, for asserting the
/// leaky mapping. Panics if the line did not parse to exactly one edge.
fn sole_edge_policy(line: &str) -> LinkPolicy {
    let reg = default_registry();
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("{line:?} should parse: {e:?}"));
    let edges = graph.edges();
    assert_eq!(
        edges.len(),
        1,
        "{line:?} should contract to one edge, got {}",
        edges.len()
    );
    edges[0].policy
}

#[tokio::test]
async fn queue_parses_and_runs() {
    // The queue collapses to a direct videotestsrc -> fakesink edge; frames flow.
    let line = "videotestsrc num-buffers=2 ! queue ! fakesink";
    assert_eq!(run_line(line).await, 2, "{line}");
}

#[tokio::test]
async fn leaky_maps_to_link_policy() {
    // gst leaky enum: 0/no = lossless (Block), 1/upstream = drop incoming
    // (DropNewest), 2/downstream = drop oldest queued (DropOldest). Nick and
    // numeric forms both parse.
    assert_eq!(
        sole_edge_policy("videotestsrc num-buffers=1 ! queue ! fakesink"),
        LinkPolicy::Block,
        "no leaky = lossless block",
    );
    assert_eq!(
        sole_edge_policy("videotestsrc num-buffers=1 ! queue leaky=downstream ! fakesink"),
        LinkPolicy::DropOldest,
    );
    assert_eq!(
        sole_edge_policy("videotestsrc num-buffers=1 ! queue leaky=2 ! fakesink"),
        LinkPolicy::DropOldest,
        "numeric leaky matches the nick",
    );
    assert_eq!(
        sole_edge_policy("videotestsrc num-buffers=1 ! queue leaky=upstream ! fakesink"),
        LinkPolicy::DropNewest,
    );
}

#[tokio::test]
async fn size_bounds_are_accepted_but_ignored() {
    // The max-size-* bounds have no per-edge analog (link_capacity is global);
    // they are accepted for paste compatibility and do not change the policy.
    let line = "videotestsrc num-buffers=2 ! queue max-size-buffers=4 max-size-time=0 ! fakesink";
    assert_eq!(run_line(line).await, 2, "{line}");
    assert_eq!(sole_edge_policy(line), LinkPolicy::Block);
}

#[tokio::test]
async fn stacked_queues_collapse_to_one_edge() {
    // A run of queues contracts to a single edge; the downstream-most leaky wins.
    let line = "videotestsrc num-buffers=2 ! queue ! queue leaky=downstream ! fakesink";
    assert_eq!(sole_edge_policy(line), LinkPolicy::DropOldest);
    assert_eq!(run_line(line).await, 2, "{line}");
}

#[tokio::test]
async fn queue_on_a_tee_branch_runs() {
    // queue after a tee branch head and before a sink: both branches deliver.
    let line = "videotestsrc num-buffers=2 ! tee name=t \
                ! queue ! fakesink   t. ! queue leaky=downstream ! fakesink";
    let reg = default_registry();
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("{line:?} should parse: {e:?}"));
    // tee has one input edge + two output edges = three edges, all queues gone.
    assert_eq!(graph.edges().len(), 3, "{line}");
    let consumed = run_graph(graph, &ZeroClock, 4)
        .await
        .expect("tee+queue runs")
        .frames_consumed;
    assert_eq!(consumed, 4, "both branches consume 2 frames each");
}

#[tokio::test]
async fn queue_as_source_or_sink_fails_loud() {
    let reg = default_registry();
    // A queue with no upstream (source position) and one with no downstream
    // (sink position) are both invalid: a queue is a 1-in/1-out edge policy.
    assert!(
        parse_launch(&reg, "queue ! fakesink").is_err(),
        "queue as a source must fail",
    );
    assert!(
        parse_launch(&reg, "videotestsrc num-buffers=1 ! queue").is_err(),
        "queue as a sink must fail",
    );
}
