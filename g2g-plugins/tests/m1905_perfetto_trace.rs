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

async fn spans_by_element(line: &str, trace_name: &str) -> BTreeMap<String, ElementSpans> {
    let path = std::env::temp_dir().join(format!("{trace_name}-{}.json", std::process::id()));
    let graph = parse_launch(&default_registry(), line).expect("the line parses");
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
    spans
        .into_iter()
        .map(|(process, element)| (element_names[&process].clone(), element))
        .collect()
}

fn every_sequence() -> BTreeSet<u64> {
    (0..FRAMES).collect()
}

#[tokio::test]
async fn every_frame_gets_a_wait_and_a_work_span_per_element() {
    let line = format!("videotestsrc num-buffers={FRAMES} ! videoconvert ! fakesink");
    let spans = spans_by_element(&line, "m1905-trace").await;

    let names: BTreeSet<&str> = spans.keys().map(String::as_str).collect();
    assert_eq!(
        names,
        BTreeSet::from(["VideoTestSrc0", "VideoConvert0", "FakeSink0"])
    );
    for name in ["VideoConvert0", "FakeSink0"] {
        let element = &spans[name];
        assert_eq!(element.work_spans, FRAMES, "{name} work spans");
        assert_eq!(
            element.work_sequences,
            every_sequence(),
            "{name} has one work span per frame sequence id"
        );
        assert_eq!(element.wait_begins, FRAMES, "{name} wait spans");
        assert_eq!(element.wait_ends, FRAMES, "{name} wait spans close");
    }
    let source = &spans["VideoTestSrc0"];
    assert_eq!(
        source.work_sequences,
        every_sequence(),
        "one span per frame pushed"
    );
    assert_eq!(source.wait_begins, 0, "a source waits on no input link");
}

#[tokio::test]
async fn a_demuxer_and_a_muxer_record_spans_per_input_frame() {
    let line = format!(
        "audiotestsrc name=src num-buffers={FRAMES} channels=2 ! deinterleave name=split \
         split.src_0 ! merge.sink_0 \
         split.src_1 ! merge.sink_1 \
         interleave name=merge ! fakesink name=out"
    );
    let spans = spans_by_element(&line, "m1918-trace").await;

    let names: BTreeSet<&str> = spans.keys().map(String::as_str).collect();
    assert_eq!(names, BTreeSet::from(["src", "split", "merge", "out"]));
    assert_eq!(spans["src"].work_sequences, every_sequence());
    assert_eq!(
        spans["split"].work_spans, FRAMES,
        "one span per stereo frame"
    );
    assert_eq!(
        spans["merge"].work_spans,
        2 * FRAMES,
        "one span per mono frame on either pad"
    );
    for name in ["split", "merge"] {
        let element = &spans[name];
        assert_eq!(
            element.wait_begins, element.work_spans,
            "{name} has a wait span for every frame"
        );
        assert_eq!(
            element.wait_ends, element.wait_begins,
            "{name} wait spans close"
        );
    }
}
