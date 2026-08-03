//! M842: instance names. A launch line's `name=` is the element's run-time
//! instance name (not just the handle `t.` references resolve against), unnamed
//! elements keep the auto `<category>N` numbering, and the bespoke linear runner
//! names its elements like the graph runner does.
//!
//! Every fixture records the name the runner hands it, and the lines go through
//! the real `parse_launch` + `run_graph`, so the naming under test is the
//! shipped one.
#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use std::sync::{Mutex, MutexGuard, OnceLock};

use g2g_core::runtime::{
    block_on, parse_launch, run_graph, run_simple_pipeline, LaunchFactory, ParseError, Registry,
    RunStats, SourceFactory, SourceLoop,
};
use g2g_core::{
    AsyncElement, Caps, CapsSet, ConfigureOutcome, Dim, G2gError, OutputSink, PadTemplate,
    PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(2),
        height: Dim::Fixed(2),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Every instance name the runner assigned, in assignment order. Process-wide
/// (the factories take a bare `fn()`, so a fixture cannot carry a handle), hence
/// the serializing guard in `run_line`.
fn assigned() -> &'static Mutex<Vec<String>> {
    static ASSIGNED: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
    ASSIGNED.get_or_init(|| Mutex::new(Vec::new()))
}

fn record(name: String) {
    assigned().lock().unwrap().push(name);
}

struct NameSrc;

impl SourceLoop for NameSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(caps()))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn set_instance_name(&mut self, name: String) {
        record(name);
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(PipelinePacket::Eos).await?;
            Ok(0)
        })
    }
}

/// One element type used for both the transform and the sink position, so the
/// per-category numbering is visible across several instances of one category.
struct NameElem;

impl AsyncElement for NameElem {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn set_instance_name(&mut self, name: String) {
        record(name);
    }
    fn process<'a>(
        &'a mut self,
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move { Ok(()) })
    }
}

fn registry() -> Registry {
    let mut reg = Registry::new();
    reg.register_source(SourceFactory::new("namesrc", caps(), || Box::new(NameSrc)));
    reg.register_launch(LaunchFactory::new(
        "nameelem",
        Vec::from([
            PadTemplate::sink(CapsSet::one(caps())),
            PadTemplate::source(CapsSet::one(caps())),
        ]),
        || Box::new(NameElem),
    ));
    reg
}

fn serialized() -> MutexGuard<'static, ()> {
    static SERIAL: Mutex<()> = Mutex::new(());
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// Parse and run `line`, returning the instance names the runner assigned (in
/// order) plus the run's stats (whose `per_element` rows are keyed by name).
fn run_line(line: &str) -> (Vec<String>, RunStats) {
    let _guard = serialized();
    assigned().lock().unwrap().clear();
    let graph = parse_launch(&registry(), line).expect("line parses");
    let stats = block_on(run_graph(graph, &ZeroClock, 2)).expect("graph runs");
    let names = assigned().lock().unwrap().clone();
    (names, stats)
}

#[test]
fn explicit_name_becomes_the_instance_name() {
    let (names, stats) = run_line("namesrc name=cam ! nameelem ! nameelem name=out");
    assert_eq!(names, ["cam", "NameElem0", "out"]);
    // The probe (and so the per-element telemetry row) is keyed by the same name.
    let rows: Vec<&str> = stats.per_element.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(rows, ["NameElem0", "out"]);
}

#[test]
fn unnamed_elements_keep_category_numbering() {
    let (names, _) = run_line("namesrc ! nameelem ! nameelem ! nameelem");
    assert_eq!(names, ["NameSrc0", "NameElem0", "NameElem1", "NameElem2"]);
}

#[test]
fn an_explicit_name_does_not_consume_a_number() {
    // gst-launch numbers only the auto-named instances, so the unnamed element
    // after a `name=` sibling is still `<category>0`.
    let (names, _) = run_line("namesrc ! nameelem name=first ! nameelem ! nameelem");
    assert_eq!(names, ["NameSrc0", "first", "NameElem0", "NameElem1"]);
}

#[test]
fn duplicate_explicit_names_are_rejected() {
    let _guard = serialized();
    let err = parse_launch(
        &registry(),
        "namesrc name=dup ! nameelem name=dup ! nameelem",
    )
    .expect_err("a repeated name= is an error, as in gst-launch");
    assert_eq!(err, ParseError::DuplicateName("dup".into()));
    assert!(err.to_string().contains("dup"), "{err}");
}

#[test]
fn the_linear_runner_names_its_elements_too() {
    let _guard = serialized();
    assigned().lock().unwrap().clear();
    let mut src = NameSrc;
    let mut sink = NameElem;
    let stats =
        block_on(run_simple_pipeline(&mut src, &mut sink, &ZeroClock, 2)).expect("pipeline runs");
    assert_eq!(*assigned().lock().unwrap(), ["NameSrc0", "NameElem0"]);
    assert_eq!(stats.per_element[0].name, "NameElem0");
}
