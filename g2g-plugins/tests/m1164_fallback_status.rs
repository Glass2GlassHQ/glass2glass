//! M1164: a restarting source reports itself on the bus. `RestartSrc` posts
//! `BusMessage::SourceRestart` as each life starts, as a rebuild is decided, and
//! when the stream ends, carrying the death's reason and the rebuild tally; the
//! `fallbacksrc` launch keyword names the two sources it builds off its own
//! `name=`, and tags each with its `FallbackSourceRole`.
//!
//! `default_registry` is `std`-gated, so this file is too: run with
//! `cargo test -p g2g-plugins --features std`.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::path::PathBuf;
use std::time::Duration;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{
    parse_launch, run_graph, run_graph_with_bus, DynSourceLoop, FallbackSourceRole, GraphNode,
    ParseError, RestartPolicy, SourceLoop, UriRebuild, FALLBACK_FALLBACK_SOURCE_SUFFIX,
    FALLBACK_MAIN_SOURCE_SUFFIX, FALLBACK_SWITCH_NAME,
};
use g2g_core::{
    Bus, BusHandle, BusMessage, Caps, ConfigureOutcome, G2gError, Graph, MemoryDomain, NodeId,
    OutputSink, PipelineClock, PipelinePacket, PushOutcome, SourceRestartReason,
    SourceRestartStatus,
};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::fallbacksrc::{RestartSrc, RETRY_DELAY};
use g2g_plugins::filesrc::FileSrc;
use g2g_plugins::registry::default_registry;
use g2g_plugins::videotestsrc::VideoTestSrc;

type DynSource = Box<dyn DynSourceLoop>;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Slack past the last event a timed run waits for.
const SETTLE: Duration = Duration::from_millis(500);

/// A stall timeout short enough that a stalled life is caught well inside one
/// retry delay.
const SHORT_STALL: Duration = Duration::from_millis(200);

/// Frames each generated life emits before ending.
const FRAMES_PER_LIFE: u64 = 3;

/// The generated test source's geometry and rate.
const WIDTH: u32 = 16;
const HEIGHT: u32 = 16;
const FRAMERATE: u32 = 30;

/// A stall check that never fires.
const NO_STALL_CHECK: u64 = 0;

/// A retry budget large enough that no test here spends it by accident.
const GENEROUS_RETRY_NS: u64 = 60_000_000_000;

/// The instance name the directly-driven wrapper is given, so the report has
/// something to label itself with.
const INSTANCE: &str = "restart-under-test";

/// Bus backlog: every test here posts a handful of messages and drains after.
const BUS_CAPACITY: usize = 64;

/// Bound on a `fallbacksrc` graph run, which need not end by itself.
const RUN_BOUND: Duration = Duration::from_secs(3);

/// One `SourceRestart` message, flattened for comparison.
type Restart = (
    String,
    Option<FallbackSourceRole>,
    SourceRestartStatus,
    u64,
    Option<SourceRestartReason>,
);

fn restarts(bus: &Bus) -> Vec<Restart> {
    let mut out = Vec::new();
    while let Some(message) = bus.try_recv() {
        if let BusMessage::SourceRestart {
            element,
            role,
            status,
            retries,
            reason,
        } = message
        {
            out.push((element, role, status, retries, reason));
        }
    }
    out
}

fn reasons(seen: &[Restart]) -> Vec<Option<SourceRestartReason>> {
    seen.iter().map(|r| r.4).collect()
}

fn temp_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g_m1164_{}_{tag}", std::process::id()))
}

/// Write a fixture by running `line` (which must end in a `filesink`) and return
/// its path, so the bytes come from the real encoder.
async fn fixture(tag: &str, line: &str) -> PathBuf {
    let path = temp_path(tag);
    let reg = default_registry();
    let line = format!("{line} ! filesink location={}", path.display());
    let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("{line}: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("{line}: {e:?}"));
    path
}

/// A PNM still `fallbacksrc` can open, decode, and play once.
async fn still(tag: &str, pattern: &str) -> PathBuf {
    fixture(
        tag,
        &format!("videotestsrc num-buffers=1 pattern={pattern} ! pnmenc"),
    )
    .await
}

fn generated_source() -> DynSource {
    Box::new(VideoTestSrc::new(WIDTH, HEIGHT, FRAMERATE, FRAMES_PER_LIFE))
}

/// The wrapper's own sources fail to open: a file that is not there.
fn missing_file_source(caps: Caps) -> DynSource {
    Box::new(FileSrc::new(temp_path("missing.bin"), caps))
}

fn rebuild_with(build: impl Fn() -> DynSource + Send + 'static) -> UriRebuild {
    Box::new(move || {
        let mut source = build();
        let caps = block_on(source.intercept_caps()).expect("a test source's caps");
        Ok((source, caps))
    })
}

/// The rebuild contract returns the source's declared caps alongside it; the
/// test sources answer `intercept_caps` synchronously, so polling once is enough.
fn block_on<F: Future>(future: F) -> F::Output {
    let waker = core::task::Waker::noop();
    let mut cx = core::task::Context::from_waker(waker);
    let mut future = core::pin::pin!(future);
    match future.as_mut().poll(&mut cx) {
        core::task::Poll::Ready(v) => v,
        core::task::Poll::Pending => panic!("a test source's intercept_caps must be immediate"),
    }
}

/// Negotiate and configure the wrapper the way the runner does, and hand it the
/// bus and the instance name the runner would.
async fn prepared(mut wrapper: RestartSrc, bus: &BusHandle) -> RestartSrc {
    let caps = SourceLoop::intercept_caps(&mut wrapper)
        .await
        .expect("the first life negotiates");
    SourceLoop::configure_pipeline(&mut wrapper, &caps).expect("the first life configures");
    SourceLoop::set_instance_name(&mut wrapper, INSTANCE.to_string());
    SourceLoop::set_bus(&mut wrapper, bus.clone());
    wrapper
}

/// Discards every packet the wrapper pushes.
#[derive(Default)]
struct Discard;

impl OutputSink for Discard {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

#[tokio::test]
async fn repeated_failure_reports_each_retry_then_stops() {
    let caps = block_on(generated_source().intercept_caps()).expect("test caps");
    let rebuild_caps = caps.clone();
    let rebuild = rebuild_with(move || missing_file_source(rebuild_caps.clone()));
    // Two rebuilds start inside the budget; the third would start past it.
    let retry_budget = RETRY_DELAY * 2 + RETRY_DELAY / 2;
    let policy = RestartPolicy {
        restart_on_eos: false,
        restart_timeout_ns: NO_STALL_CHECK,
        retry_timeout_ns: retry_budget.as_nanos() as u64,
    };
    let (bus, handle) = Bus::new(BUS_CAPACITY);
    let mut wrapper = prepared(
        RestartSrc::new(missing_file_source(caps), rebuild, policy, None),
        &handle,
    )
    .await;
    SourceLoop::run(&mut wrapper, &mut Discard)
        .await
        .expect("giving up ends the stream cleanly");

    let expected: Vec<Restart> = [
        (SourceRestartStatus::Running, 0, None),
        (
            SourceRestartStatus::Retrying,
            1,
            Some(SourceRestartReason::Error),
        ),
        (SourceRestartStatus::Running, 1, None),
        (
            SourceRestartStatus::Retrying,
            2,
            Some(SourceRestartReason::Error),
        ),
        (SourceRestartStatus::Running, 2, None),
        (
            SourceRestartStatus::Stopped,
            2,
            Some(SourceRestartReason::Error),
        ),
    ]
    .into_iter()
    .map(|(status, retries, reason)| (INSTANCE.to_string(), None, status, retries, reason))
    .collect();
    assert_eq!(restarts(&bus), expected);
}

#[tokio::test]
async fn clean_end_of_stream_stops_without_a_reason() {
    let policy = RestartPolicy {
        restart_on_eos: false,
        restart_timeout_ns: NO_STALL_CHECK,
        retry_timeout_ns: GENEROUS_RETRY_NS,
    };
    let (bus, handle) = Bus::new(BUS_CAPACITY);
    let mut wrapper = prepared(
        RestartSrc::new(
            generated_source(),
            rebuild_with(generated_source),
            policy,
            None,
        ),
        &handle,
    )
    .await;
    let frames = SourceLoop::run(&mut wrapper, &mut Discard)
        .await
        .expect("the single life ends the stream");

    assert_eq!(frames, FRAMES_PER_LIFE);
    assert_eq!(
        restarts(&bus),
        vec![
            (
                INSTANCE.to_string(),
                None,
                SourceRestartStatus::Running,
                0,
                None
            ),
            (
                INSTANCE.to_string(),
                None,
                SourceRestartStatus::Stopped,
                0,
                None
            ),
        ]
    );
}

/// Pushes one frame, then never returns: a peer that went silent mid-stream.
struct StallSrc;

impl SourceLoop for StallSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(block_on(generated_source().intercept_caps()))
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let bytes = vec![0u8; (WIDTH * HEIGHT * 4) as usize];
            let frame = Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
                timing: Default::default(),
                sequence: 0,
                meta: Default::default(),
            };
            out.push(PipelinePacket::DataFrame(frame)).await?;
            core::future::pending::<()>().await;
            Ok(1)
        })
    }
}

#[tokio::test]
async fn a_stall_reports_timeout_not_error() {
    let policy = RestartPolicy {
        restart_on_eos: false,
        restart_timeout_ns: SHORT_STALL.as_nanos() as u64,
        retry_timeout_ns: GENEROUS_RETRY_NS,
    };
    let (bus, handle) = Bus::new(BUS_CAPACITY);
    let mut wrapper = prepared(
        RestartSrc::new(
            Box::new(StallSrc),
            rebuild_with(|| Box::new(StallSrc) as DynSource),
            policy,
            None,
        ),
        &handle,
    )
    .await;
    // The first life stalls and its replacement is running when the run is cut.
    let _ = tokio::time::timeout(
        SHORT_STALL + RETRY_DELAY + SETTLE,
        SourceLoop::run(&mut wrapper, &mut Discard),
    )
    .await;

    let seen = restarts(&bus);
    assert_eq!(
        reasons(&seen)[..3],
        [None, Some(SourceRestartReason::Timeout), None],
        "the stalled life is retried for Timeout, not Error: {seen:?}"
    );
    assert_eq!(
        seen[1].2,
        SourceRestartStatus::Retrying,
        "the stall is reported as a retry: {seen:?}"
    );
    assert!(
        !reasons(&seen).contains(&Some(SourceRestartReason::Error)),
        "a stall is never reported as a run error: {seen:?}"
    );
}

#[tokio::test]
async fn the_graph_runner_hands_a_source_its_bus() {
    let policy = RestartPolicy {
        restart_on_eos: false,
        restart_timeout_ns: NO_STALL_CHECK,
        retry_timeout_ns: GENEROUS_RETRY_NS,
    };
    let wrapper = RestartSrc::new(
        generated_source(),
        rebuild_with(generated_source),
        policy,
        None,
    );
    let (bus, handle) = Bus::new(BUS_CAPACITY);
    {
        let mut graph: Graph<GraphNode> = Graph::new();
        let source = graph.add_source(GraphNode::source(wrapper));
        let sink = graph.add_sink(GraphNode::element(FakeSink::new()));
        graph.link(source, sink).expect("a two-node graph links");
        graph.set_node_name(source, INSTANCE.to_string());
        tokio::time::timeout(RUN_BOUND, run_graph_with_bus(graph, &ZeroClock, 4, &handle))
            .await
            .expect("the single life ends the run")
            .expect("the graph runs");
    }

    assert_eq!(
        restarts(&bus)
            .into_iter()
            .map(|r| (r.0, r.2))
            .collect::<Vec<_>>(),
        vec![
            (INSTANCE.to_string(), SourceRestartStatus::Running),
            (INSTANCE.to_string(), SourceRestartStatus::Stopped),
        ],
        "the runner handed the source both its name and the bus"
    );
}

fn node_names(graph: &Graph<GraphNode>) -> Vec<String> {
    (0..graph.node_count())
        .filter_map(|i| graph.node_name(NodeId(i as u32)).map(String::from))
        .collect()
}

#[tokio::test]
async fn fallbacksrc_names_the_sources_it_builds() {
    let main = still("named_main.pnm", "smpte").await;
    let fallback = still("named_fallback.pnm", "black").await;
    let reg = default_registry();
    let switch = "fb";
    let line = format!(
        "fallbacksrc name={switch} uri=file://{} fallback-uri=file://{} ! fakesink",
        main.display(),
        fallback.display()
    );
    let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("{line}: {e}"));
    let names = node_names(&graph);
    for expected in [
        format!("{switch}{FALLBACK_MAIN_SOURCE_SUFFIX}"),
        format!("{switch}{FALLBACK_FALLBACK_SOURCE_SUFFIX}"),
    ] {
        assert!(
            names.contains(&expected),
            "{expected} missing from {names:?}"
        );
    }

    // Unnamed, the sources hang off the generated switch name instead.
    let line = format!(
        "fallbacksrc uri=file://{} fallback-uri=file://{} ! fakesink",
        main.display(),
        fallback.display()
    );
    let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("{line}: {e}"));
    let names = node_names(&graph);
    let generated = format!("{FALLBACK_SWITCH_NAME}-0");
    for expected in [
        format!("{generated}{FALLBACK_MAIN_SOURCE_SUFFIX}"),
        format!("{generated}{FALLBACK_FALLBACK_SOURCE_SUFFIX}"),
    ] {
        assert!(
            names.contains(&expected),
            "{expected} missing from {names:?}"
        );
    }

    std::fs::remove_file(&main).ok();
    std::fs::remove_file(&fallback).ok();
}

#[tokio::test]
async fn a_line_colliding_with_a_generated_source_name_is_rejected() {
    let main = still("collide_main.pnm", "smpte").await;
    let reg = default_registry();
    let switch = "fb";
    let taken = format!("{switch}{FALLBACK_MAIN_SOURCE_SUFFIX}");
    let line = format!(
        "fallbacksrc name={switch} uri=file://{} ! identity name={taken} ! fakesink",
        main.display()
    );
    match parse_launch(&reg, &line) {
        Err(ParseError::DuplicateName(name)) => assert_eq!(name, taken),
        other => panic!("{line}: expected DuplicateName, got {other:?}"),
    }
    std::fs::remove_file(&main).ok();
}

#[tokio::test]
async fn the_two_expanded_sources_report_their_roles() {
    let main = still("roles_main.pnm", "smpte").await;
    let fallback = still("roles_fallback.pnm", "black").await;
    let reg = default_registry();
    let line = format!(
        "fallbacksrc uri=file://{} fallback-uri=file://{} ! fakesink",
        main.display(),
        fallback.display()
    );
    let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("{line}: {e}"));
    let (bus, handle) = Bus::new(BUS_CAPACITY);
    let _ =
        tokio::time::timeout(RUN_BOUND, run_graph_with_bus(graph, &ZeroClock, 4, &handle)).await;
    drop(handle);

    let mut roles: Vec<Option<FallbackSourceRole>> = restarts(&bus)
        .into_iter()
        .filter(|r| r.2 == SourceRestartStatus::Running)
        .map(|r| r.1)
        .collect();
    roles.sort_by_key(|role| matches!(role, Some(FallbackSourceRole::Fallback)));
    roles.dedup();
    assert_eq!(
        roles,
        vec![
            Some(FallbackSourceRole::Main),
            Some(FallbackSourceRole::Fallback)
        ],
        "both expanded sources report which one they are"
    );

    std::fs::remove_file(&main).ok();
    std::fs::remove_file(&fallback).ok();
}
