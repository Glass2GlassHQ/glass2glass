//! M842: instance names. A launch line's `name=` is the element's run-time
//! instance name (not just the handle `t.` references resolve against), unnamed
//! elements keep the auto `<category>N` numbering, and the bespoke linear runner
//! names its elements like the graph runner does.
//!
//! M847 adds the caller half of the per-instance log-category override: a launch
//! line's `log-category=` reaches the element (and keys its `G2G_DEBUG`
//! filtering) while naming stays type-based, and the fan-in / fan-out payloads
//! are named too.
//!
//! Every fixture records the name the runner hands it, and the lines go through
//! the real `parse_launch` + `run_graph`, so the naming under test is the
//! shipped one.
#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use g2g_core::log::{self, LogName, LogRecord, LogSink, LogSource};
use g2g_core::runtime::{
    block_on, parse_launch, run_graph, run_simple_pipeline, GraphNode, LaunchFactory, ParseError,
    Registry, RunStats, SourceFactory, SourceLoop,
};
use g2g_core::{
    g2g_info, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, Graph,
    MultiInputElement, MultiOutputElement, MultiOutputSink, OutputSink, PadTemplate, PipelineClock,
    PipelinePacket, Rate, RawVideoFormat,
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
        interlace: g2g_core::Interlace::Any,
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

/// Both fixtures log about themselves, so a capturing sink sees which category
/// their lines carry (the type name, or a `log-category=` override).
#[derive(Default)]
struct NameSrc {
    log_name: LogName,
}

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
        g2g_info!(self, "configured");
        Ok(ConfigureOutcome::Accepted)
    }
    fn set_instance_name(&mut self, name: String) {
        record(name.clone());
        self.log_name.set_instance(name);
    }
    fn set_log_category(&mut self, category: String) {
        self.log_name.set_category(category);
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(PipelinePacket::Eos).await?;
            Ok(0)
        })
    }
}

impl LogSource for NameSrc {
    fn log_category(&self) -> &'static str {
        "NameSrc"
    }
    fn log_instance(&self) -> Option<&str> {
        self.log_name.instance()
    }
    fn log_category_override(&self) -> Option<&str> {
        self.log_name.category()
    }
}

/// One element type used for both the transform and the sink position, so the
/// per-category numbering is visible across several instances of one category.
#[derive(Default)]
struct NameElem {
    log_name: LogName,
}

impl AsyncElement for NameElem {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        g2g_info!(self, "configured");
        Ok(ConfigureOutcome::Accepted)
    }
    fn set_instance_name(&mut self, name: String) {
        record(name.clone());
        self.log_name.set_instance(name);
    }
    fn set_log_category(&mut self, category: String) {
        self.log_name.set_category(category);
    }
    fn process<'a>(
        &'a mut self,
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move { Ok(()) })
    }
}

impl LogSource for NameElem {
    fn log_category(&self) -> &'static str {
        "NameElem"
    }
    fn log_instance(&self) -> Option<&str> {
        self.log_name.instance()
    }
    fn log_category_override(&self) -> Option<&str> {
        self.log_name.category()
    }
}

/// A 2-input muxer that records the name the runner hands it (M847): fan-in
/// payloads are named like transforms.
#[derive(Default)]
struct NameMux;

impl MultiInputElement for NameMux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        2
    }
    fn intercept_caps(&self, _input: usize, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }
    fn configure_pipeline(
        &mut self,
        _input: usize,
        _absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(caps())
    }
    fn set_instance_name(&mut self, name: String) {
        record(name);
    }
    fn process<'a>(
        &'a mut self,
        _input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            // the runner emits the merged Eos itself, so a muxer must not forward it
            match packet {
                PipelinePacket::Eos => Ok(()),
                p => out.push(p).await.map(|_| ()),
            }
        })
    }
}

/// A 1-output demux that records the name the runner hands it (M847).
#[derive(Default)]
struct NameDemux;

impl MultiOutputElement for NameDemux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }
    fn caps_constraint_as_input(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn set_instance_name(&mut self, name: String) {
        record(name);
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            // the arm closes each branch with its own Eos, so this only flushes
            match packet {
                PipelinePacket::Eos => Ok(()),
                p => out.push_to(0, p).await.map(|_| ()),
            }
        })
    }
}

fn registry() -> Registry {
    let mut reg = Registry::new();
    reg.register_source(SourceFactory::new("namesrc", caps(), || {
        Box::new(NameSrc::default())
    }));
    reg.register_launch(LaunchFactory::new(
        "nameelem",
        Vec::from([
            PadTemplate::sink(CapsSet::one(caps())),
            PadTemplate::source(CapsSet::one(caps())),
        ]),
        || Box::new(NameElem::default()),
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
    let mut src = NameSrc::default();
    let mut sink = NameElem::default();
    let stats =
        block_on(run_simple_pipeline(&mut src, &mut sink, &ZeroClock, 2)).expect("pipeline runs");
    assert_eq!(*assigned().lock().unwrap(), ["NameSrc0", "NameElem0"]);
    assert_eq!(stats.per_element[0].name, "NameElem0");
}

/// One captured record, flattened to owned data.
#[derive(Clone, Debug)]
struct Rec {
    category: String,
    instance: Option<String>,
    message: String,
}

struct CaptureSink(Arc<Mutex<Vec<Rec>>>);
impl LogSink for CaptureSink {
    fn emit(&self, r: &LogRecord<'_>) {
        self.0.lock().unwrap().push(Rec {
            category: r.category.to_string(),
            instance: r.instance.map(|s| s.to_string()),
            message: std::format!("{}", r.message),
        });
    }
}

#[test]
fn log_category_override_from_a_launch_line() {
    let _guard = serialized();
    assigned().lock().unwrap().clear();
    // Only the two overridden instances pass the filter: their un-overridden
    // siblings keep the type category, which is off.
    let captured = Arc::new(Mutex::new(Vec::new()));
    log::reset();
    log::set_sink(Box::new(CaptureSink(captured.clone())));
    log::configure("*:off,cam-cat:info,flip-cat:info");

    let graph = parse_launch(
        &registry(),
        "namesrc log-category=cam-cat ! nameelem log-category=flip-cat ! nameelem",
    )
    .expect("line parses");
    let names = block_on(async {
        run_graph(graph, &ZeroClock, 2).await.expect("graph runs");
        assigned().lock().unwrap().clone()
    });
    let recs = captured.lock().unwrap().clone();
    log::reset();

    // The override is the filter key, so exactly the two overridden instances
    // logged, under their new categories.
    let lines: Vec<(&str, Option<&str>)> = recs
        .iter()
        .filter(|r| r.message == "configured")
        .map(|r| (r.category.as_str(), r.instance.as_deref()))
        .collect();
    assert_eq!(
        lines,
        [
            ("cam-cat", Some("NameSrc0")),
            ("flip-cat", Some("NameElem0"))
        ],
        "got {recs:?}"
    );
    assert!(
        recs.iter().all(|r| r.category != "NameElem"),
        "the un-overridden nameelem stayed filtered out: {recs:?}"
    );
    // Naming still keys on the element type, not the override, so the numbering
    // is unchanged and `log-category=` never consumed a `name=`.
    assert_eq!(names, ["NameSrc0", "NameElem0", "NameElem1"]);
}

#[test]
fn fan_in_and_fan_out_payloads_are_named() {
    let _guard = serialized();

    // Two sources into a muxer: the muxer payload is named `mux0`.
    assigned().lock().unwrap().clear();
    let mut g: Graph<GraphNode> = Graph::new();
    let a = g.add_source(GraphNode::source(NameSrc::default()));
    let b = g.add_source(GraphNode::source(NameSrc::default()));
    let mux = g.add_muxer(GraphNode::muxer(NameMux), 2);
    let sink = g.add_sink(GraphNode::element(NameElem::default()));
    g.link(a, mux.input(0)).unwrap();
    g.link(b, mux.input(1)).unwrap();
    g.link(mux.output(), sink).unwrap();
    block_on(run_graph(g, &ZeroClock, 2)).expect("mux graph runs");
    assert_eq!(
        *assigned().lock().unwrap(),
        ["NameSrc0", "NameSrc1", "mux0", "NameElem0"]
    );

    // One source through a demux: the demux payload is named `demux0`.
    assigned().lock().unwrap().clear();
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(NameSrc::default()));
    let demux = g.add_demux(GraphNode::demux(NameDemux), 1);
    let sink = g.add_sink(GraphNode::element(NameElem::default()));
    g.link(src, demux.input()).unwrap();
    g.link(demux.out(0), sink).unwrap();
    block_on(run_graph(g, &ZeroClock, 2)).expect("demux graph runs");
    assert_eq!(
        *assigned().lock().unwrap(),
        ["NameSrc0", "demux0", "NameElem0"]
    );
}
