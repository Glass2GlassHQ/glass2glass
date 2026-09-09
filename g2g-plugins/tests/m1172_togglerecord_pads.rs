//! M1172: `togglerecord name=t` takes gst's `sink_%u` / `src_%u` request pads.
//!
//! gst spells several streams that start and stop together as one element with a
//! main `sink`/`src` pair plus request-pad pairs. That line now parses here: the
//! inline keyword is stream 0, the main stream whose keyframes decide, and each
//! `t.sink_K` / `t.src_K` pair is a secondary stream. It expands at parse time
//! into one `ToggleRecord` per stream sharing one group, the way every other g2g
//! bin flattens, so the graph holds plain 1-in 1-out transforms and the runtime
//! behaviour is M1155's.
//!
//! What is checked: the expanded line builds the same shape and records the same
//! frames as the `group=` spelling it replaces, the group name is the keyword's
//! own `name=` unless the line set `group=`, the secondaries are not the main
//! stream, and an unpaired or unnamed request pad is a parse error rather than a
//! stream with one end dangling.
//!
//! `default_registry` is `std`-gated, so this file is too: run with
//! `cargo test -p g2g-plugins --features std`.
#![cfg(feature = "std")]

use g2g_core::runtime::{parse_launch, run_graph, ParseError};
use g2g_core::{NodeId, NodeKind, PipelineClock, PropValue};
use g2g_plugins::registry::default_registry;
use g2g_plugins::togglerecord::RecordGroup;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Frames each branch's source emits, and the link capacity the runs use.
const FRAMES: u32 = 4;

/// Two streams through one `togglerecord`, written with request pads: the first
/// chain carries the main stream inline, the second enters at `sink_1` and leaves
/// at `src_1`.
///
/// Every check names its own keyword, because a group is reachable by name for
/// the whole process and, across several parses of one name, `RecordGroup::named`
/// reaches the earliest of them (M1167). Two checks sharing a name would decide
/// each other's recording.
fn request_pad_line(keyword: &str) -> String {
    format!(
        "videotestsrc num-buffers={FRAMES} ! togglerecord name={keyword} ! fakesink \
         videotestsrc num-buffers={FRAMES} ! {keyword}.sink_1   {keyword}.src_1 ! fakesink"
    )
}

/// The same two streams written the way M1155 spells them, as two elements
/// joined by a `group=` name. The request-pad line has to build the same thing.
fn group_line(group: &str) -> String {
    format!(
        "videotestsrc num-buffers={FRAMES} ! togglerecord group={group} ! fakesink \
         videotestsrc num-buffers={FRAMES} ! togglerecord group={group} main=false ! fakesink"
    )
}

fn kinds(line: &str) -> Vec<NodeKind> {
    let reg = default_registry();
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("{line}: {e}"));
    let valid = graph.finish().expect("the built graph is valid");
    valid.topo().iter().map(|&n| valid.kind(n)).collect()
}

fn node_names(line: &str) -> Vec<String> {
    let reg = default_registry();
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("{line}: {e}"));
    (0..graph.node_count())
        .filter_map(|i| graph.node_name(NodeId(i as u32)).map(String::from))
        .collect()
}

/// The request-pad line builds the same graph shape as the `group=` line: two
/// sources, two transforms, two sinks, and no fan-in (the streams stay separate,
/// which is what tells this apart from a muxer).
#[test]
fn a_request_pad_line_builds_one_element_per_stream() {
    let expanded = kinds(&request_pad_line("m1172-shape-pads"));
    let spelled_out = kinds(&group_line("m1172-shape"));
    assert_eq!(
        expanded, spelled_out,
        "the request pads expand to what `group=` spells out"
    );
    assert_eq!(
        expanded
            .iter()
            .filter(|k| **k == NodeKind::Transform)
            .count(),
        2,
        "one togglerecord per stream: {expanded:?}"
    );
    assert!(
        !expanded
            .iter()
            .any(|k| matches!(k, NodeKind::Muxer(_) | NodeKind::FaninSink(_))),
        "the streams stay separate, so nothing joins them: {expanded:?}"
    );
}

/// The keyword keeps the name the line gave it and each secondary takes that
/// name plus its stream index, so every node is addressable and distinct.
#[test]
fn each_stream_takes_the_keywords_name_and_its_index() {
    const KEYWORD: &str = "m1172-names";
    let mut names = node_names(&request_pad_line(KEYWORD));
    assert!(
        names.iter().any(|n| n == KEYWORD),
        "the inline keyword keeps its name: {names:?}"
    );
    assert!(
        names.iter().any(|n| n == &format!("{KEYWORD}-1")),
        "the secondary takes the stream index: {names:?}"
    );
    let count = names.len();
    names.sort();
    names.dedup();
    assert_eq!(names.len(), count, "every name is distinct: {names:?}");
}

/// The group is the keyword's own `name=`, so an application reaches the line's
/// group by the name it wrote, and both streams record the frames it asked for.
#[tokio::test]
async fn the_group_is_the_keywords_name_and_both_streams_record() {
    const KEYWORD: &str = "m1172-by-name";
    let reg = default_registry();
    let line = request_pad_line(KEYWORD);
    let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("{line}: {e}"));

    // Held for the whole run: the name only reaches the line's group after the
    // parse, and dropping it would let a later lookup build a fresh one.
    let group = RecordGroup::named(KEYWORD);
    group.set_record(true);

    let stats = run_graph(graph, &ZeroClock, 4).await.expect("runs");
    assert_eq!(
        stats.frames_consumed,
        (FRAMES * 2) as u64,
        "both streams recorded every frame"
    );
}

/// An explicit `group=` on the keyword wins over its `name=`, so a line can put
/// two request-pad sets in one group, or name the group something else.
#[tokio::test]
async fn an_explicit_group_overrides_the_name() {
    // Its own keyword name, because a group is reachable by name across the
    // whole process and a name another check records through would decide this
    // one for it.
    const KEYWORD: &str = "m1172-named";
    const GROUP: &str = "m1172-explicit";
    let reg = default_registry();
    let line = format!(
        "videotestsrc num-buffers={FRAMES} ! togglerecord name={KEYWORD} group={GROUP} ! fakesink \
         videotestsrc num-buffers={FRAMES} ! {KEYWORD}.sink_1   {KEYWORD}.src_1 ! fakesink"
    );
    let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("{line}: {e}"));
    let group = RecordGroup::named(GROUP);
    group.set_record(true);

    // Recording is asked for on the explicit group alone, so frames only arrive
    // if that is the group both streams joined.
    let stats = run_graph(graph, &ZeroClock, 4).await.expect("runs");
    assert_eq!(stats.frames_consumed, (FRAMES * 2) as u64);
}

/// Exactly one member of a group may be the main stream. The inline keyword is
/// it, so a secondary the expansion built reports `main=false` and the line does
/// not fail the way two mains would.
#[test]
fn only_the_inline_keyword_is_the_main_stream() {
    let reg = default_registry();
    let element = reg
        .make_element("togglerecord")
        .expect("registered under the gst name");
    assert_eq!(
        element.get_property("main"),
        Some(PropValue::Bool(true)),
        "a bare togglerecord is the main stream of its own group"
    );
    // The expanded line configures every element, which is where a second main
    // would be refused, so a successful run is the assertion.
    let line = request_pad_line("m1172-one-main");
    parse_launch(&reg, &line).unwrap_or_else(|e| panic!("{line}: {e}"));
}

/// A request pad with no partner leaves a stream with one end dangling, so it is
/// refused at parse time and the message names the pad.
#[test]
fn an_unpaired_request_pad_is_refused() {
    let reg = default_registry();
    const KEYWORD: &str = "m1172-unpaired";
    for (line, pad) in [
        (
            format!(
                "videotestsrc num-buffers={FRAMES} ! togglerecord name={KEYWORD} ! fakesink \
                 videotestsrc num-buffers={FRAMES} ! {KEYWORD}.sink_1"
            ),
            "sink_1",
        ),
        (
            format!(
                "videotestsrc num-buffers={FRAMES} ! togglerecord name={KEYWORD} ! fakesink \
                 {KEYWORD}.src_1 ! fakesink"
            ),
            "src_1",
        ),
    ] {
        match parse_launch(&reg, &line) {
            Err(ParseError::UnpairedRequestPad { element, pad: p }) => {
                assert_eq!(element, KEYWORD);
                assert_eq!(p, pad);
            }
            other => panic!("{line}: expected an unpaired-pad error, got {other:?}"),
        }
    }
}

/// The pads are paired by index, so a reference has to name one: a bare `t.` or
/// a stream-0 pad has nothing to pair with and is refused.
#[test]
fn a_reference_that_names_no_request_pad_is_refused() {
    let reg = default_registry();
    const KEYWORD: &str = "m1172-bad-pad";
    for pad in ["", "sink_0", "src_0", "video_1"] {
        let line = format!(
            "videotestsrc num-buffers={FRAMES} ! togglerecord name={KEYWORD} ! fakesink \
             videotestsrc num-buffers={FRAMES} ! {KEYWORD}.{pad}   {KEYWORD}.{pad} ! fakesink"
        );
        match parse_launch(&reg, &line) {
            Err(ParseError::BadRequestPad { element, .. }) => assert_eq!(element, KEYWORD),
            other => panic!("{line}: expected a bad-pad error, got {other:?}"),
        }
    }
}

/// A `togglerecord` nobody references is untouched: a lone one is still the
/// single-stream element of M1155, group and all.
#[tokio::test]
async fn an_unreferenced_togglerecord_stays_single_stream() {
    let reg = default_registry();
    let line =
        format!("videotestsrc num-buffers={FRAMES} ! togglerecord name=m1172-lone ! fakesink");
    let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("{line}: {e}"));
    let stats = run_graph(graph, &ZeroClock, 4).await.expect("runs");
    assert_eq!(
        stats.frames_consumed, 0,
        "`record` defaults to false, so nothing is written"
    );
}
