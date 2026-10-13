//! M1232: `pysink` hosts a gst-python-ml element as a terminal, so a launch line
//! can end in a Python class that reads every frame and sends nothing on.
#![cfg(feature = "analytics")]

use std::path::{Path, PathBuf};

use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{
    AsyncElement, Caps, Frame, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket,
    PropValue, PushOutcome, TextFormat,
};
use g2g_plugins::clock::WallClock;
use g2g_python::PySink;

const FIXTURE_MODULE: &str = "sink_element";
const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
const FRAME_COUNT: u32 = 5;
const LINK_CAPACITY: usize = 4;
const TEXT_CAPS: &str = "text/x-raw,format=utf8";

fn fixtures_on_path() {
    static SET: std::sync::Once = std::sync::Once::new();
    SET.call_once(|| std::env::set_var("PYTHONPATH", FIXTURES));
}

fn report_path(test: &str) -> PathBuf {
    std::env::temp_dir().join(format!("m1232-{test}-{}.txt", std::process::id()))
}

fn take_report(path: &Path) -> String {
    let report = std::fs::read_to_string(path).expect("the hosted sink wrote its report");
    std::fs::remove_file(path).expect("remove the report");
    report
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .unwrap()
}

struct NothingDownstream;

impl OutputSink for NothingDownstream {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        panic!("a sink pushed {:?}", packet_slot.take());
    }
}

fn text_frame(text: &str) -> Frame {
    Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(
            text.as_bytes().to_vec().into_boxed_slice(),
        )),
        FrameTiming::default(),
        0,
    )
}

fn text_sink(class: &str, location: &Path) -> PySink {
    fixtures_on_path();
    let mut sink = PySink::new(FIXTURE_MODULE, class);
    sink.set_property("input-caps", PropValue::Str(TEXT_CAPS.into()))
        .unwrap();
    sink.set_property(
        "location",
        PropValue::Str(location.to_str().unwrap().into()),
    )
    .unwrap();
    sink.configure_pipeline(&Caps::Text {
        format: TextFormat::Utf8,
    })
    .unwrap();
    sink
}

#[test]
fn a_launch_line_ends_in_a_hosted_sink() {
    fixtures_on_path();
    let location = report_path("counting");
    let mut registry = g2g_plugins::registry::default_registry();
    g2g_python::register(&mut registry);
    let line = format!(
        "videotestsrc num-buffers={FRAME_COUNT} ! pysink module={FIXTURE_MODULE} \
         class=CountingSink location={}",
        location.display()
    );
    let graph = parse_launch(&registry, &line).expect("pysink ends a launch line");
    let clock = WallClock::new();
    runtime()
        .block_on(run_graph(graph, &clock, LINK_CAPACITY))
        .expect("the run completes");
    assert_eq!(
        take_report(&location),
        format!("frames={FRAME_COUNT}\neos=True\n")
    );
}

#[test]
fn a_payload_reaches_the_payload_hook() {
    let location = report_path("payload");
    let mut sink = text_sink("CountingSink", &location);
    let rt = runtime();
    rt.block_on(sink.process(
        PipelinePacket::DataFrame(text_frame("hello")),
        &mut NothingDownstream,
    ))
    .expect("the sink reads the payload");
    rt.block_on(sink.process(PipelinePacket::Eos, &mut NothingDownstream))
        .expect("the sink ends");
    assert_eq!(
        take_report(&location),
        format!("frames=1\neos=True\npayload={TEXT_CAPS} hello\n")
    );
}

#[test]
fn a_sink_that_emits_a_buffer_fails_the_frame() {
    let location = report_path("emitting");
    let mut sink = text_sink("EmittingSink", &location);
    let result = runtime().block_on(sink.process(
        PipelinePacket::DataFrame(text_frame("hello")),
        &mut NothingDownstream,
    ));
    take_report(&location);
    assert!(result.is_err(), "an emitted buffer is refused");
}

#[cfg(feature = "launch")]
fn launcher(args: &[&str]) -> std::process::Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_g2g-launch-py"))
        .args(args)
        .env("PYTHONPATH", FIXTURES)
        .output()
        .expect("run g2g-launch-py")
}

#[cfg(feature = "launch")]
#[test]
fn a_sink_hook_that_returns_a_frame_fails_the_run() {
    let location = report_path("returning");
    let num_buffers = format!("num-buffers={FRAME_COUNT}");
    let module = format!("module={FIXTURE_MODULE}");
    let location_property = format!("location={}", location.display());
    let output = launcher(&[
        "-q",
        "videotestsrc",
        &num_buffers,
        "!",
        "pysink",
        &module,
        "class=ReturningSink",
        &location_property,
    ]);
    take_report(&location);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "the run fails: {stderr}");
    assert!(
        stderr.contains("ReturningSink is a sink and cannot produce"),
        "the error names the class: {stderr}"
    );
}

#[cfg(feature = "launch")]
#[test]
fn inspection_lists_the_hosted_sink_class_properties() {
    let module = format!("module={FIXTURE_MODULE}");
    let output = launcher(&["--inspect", "pysink", &module, "class=CountingSink"]);
    assert!(output.status.success());
    let dump = String::from_utf8_lossy(&output.stdout);
    assert!(dump.contains("  location:"), "{dump}");
}
