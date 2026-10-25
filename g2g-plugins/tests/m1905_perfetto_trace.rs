#![cfg(all(feature = "std", feature = "tooling-json"))]

use std::collections::{BTreeMap, BTreeSet};

use g2g_core::runtime::{parse_launch, RunBuilder};
use g2g_plugins::clock::WallClock;
use g2g_plugins::perfetto_trace::PerfettoTrace;
use g2g_plugins::registry::default_registry;
use serde_json::Value;

const FRAMES: u64 = 20;
const LINK_CAPACITY: usize = 4;

#[derive(Debug, Default)]
struct ElementSpans {
    work_sequences: BTreeSet<u64>,
    work_spans: u64,
    wait_begins: u64,
    wait_ends: u64,
}

#[tokio::test]
async fn every_frame_gets_a_wait_and_a_work_span_per_element() {
    let path = std::env::temp_dir().join(format!("m1905-trace-{}.json", std::process::id()));
    let line = format!("videotestsrc num-buffers={FRAMES} ! videoconvert ! fakesink");
    let graph = parse_launch(&default_registry(), &line).expect("the line parses");
    let clock = WallClock::new();
    let trace = PerfettoTrace::create(&path).expect("the trace file opens");
    RunBuilder::new(graph, &clock, LINK_CAPACITY)
        .with_visit_recorder(trace.clone())
        .run()
        .await
        .expect("the pipeline runs to EOS");
    trace.finish().expect("the trace file closes");

    let text = std::fs::read_to_string(&path).expect("the trace file reads back");
    std::fs::remove_file(&path).expect("the trace file is removed");
    let document: Value = serde_json::from_str(&text).expect("the trace is valid JSON");
    let events = document["traceEvents"]
        .as_array()
        .expect("a traceEvents array");

    let mut element_names: BTreeMap<u64, String> = BTreeMap::new();
    let mut spans: BTreeMap<u64, ElementSpans> = BTreeMap::new();
    for event in events {
        let process = event["pid"].as_u64().expect("every event has a pid");
        match (event["ph"].as_str(), event["name"].as_str()) {
            (Some("M"), Some("process_name")) => {
                let name = event["args"]["name"].as_str().expect("a process name");
                element_names.insert(process, name.to_string());
            }
            (Some("X"), Some("work")) => {
                let element = spans.entry(process).or_default();
                element.work_spans += 1;
                element
                    .work_sequences
                    .insert(event["args"]["sequence"].as_u64().expect("a sequence id"));
            }
            (Some("b"), Some("wait")) => spans.entry(process).or_default().wait_begins += 1,
            (Some("e"), Some("wait")) => spans.entry(process).or_default().wait_ends += 1,
            _ => {}
        }
    }

    let names: BTreeSet<&str> = element_names.values().map(String::as_str).collect();
    assert_eq!(names, BTreeSet::from(["VideoConvert0", "FakeSink0"]));
    for (process, element) in &spans {
        let name = &element_names[process];
        assert_eq!(element.work_spans, FRAMES, "{name} work spans");
        assert_eq!(
            element.work_sequences,
            (0..FRAMES).collect::<BTreeSet<u64>>(),
            "{name} has one work span per frame sequence id"
        );
        assert_eq!(element.wait_begins, FRAMES, "{name} wait spans");
        assert_eq!(element.wait_ends, FRAMES, "{name} wait spans close");
    }
    assert_eq!(spans.len(), 2, "both elements recorded spans");
}
