//! M1198: `fallbacksrc` runs an application-built source in place of `uri=`.
//! The application registers a `FallbacksrcSourceFactory` on the registry, and a
//! `fallbacksrc` line with no `uri=` builds its main branch from it. The factory
//! is called once at parse and again for every restart, so the existing restart
//! policy rebuilds the application's source the way it rebuilds a URI source.
//!
//! Frames are told apart by size: the application's source draws small frames,
//! the dummy fallback its own default geometry.
//!
//! `default_registry` is `std`-gated, so this file is too: run with
//! `cargo test -p g2g-plugins --features std`.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use g2g_core::error::HardwareError;
use g2g_core::runtime::{parse_launch, run_graph, FallbacksrcSourceFactory, Registry, SourceLoop};
use g2g_core::{Caps, ConfigureOutcome, G2gError, NodeKind, OutputSink, PipelineClock};
use g2g_plugins::appsink::register_appsink_pull;
use g2g_plugins::fallbacksrc::RETRY_DELAY;
use g2g_plugins::registry::default_registry;
use g2g_plugins::videotestsrc::{Pattern, VideoTestSrc};

type DynSource = Box<dyn g2g_core::runtime::DynSourceLoop>;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// The application's source geometry and rate. Small, so its frames cannot be
/// mistaken for the dummy's.
const WIDTH: u32 = 16;
const HEIGHT: u32 = 16;
const FRAMERATE: u32 = 30;
const BYTES_PER_PIXEL: usize = 4;

/// Frames each life of the application's source emits before ending.
const FRAMES_PER_LIFE: u64 = 3;

/// A switch `timeout` short enough that the main stream's end hands over to the
/// fallback quickly.
const SHORT_TIMEOUT_NS: u64 = 50_000_000;

/// Slack past the last expected event a timed run waits for.
const SETTLE: Duration = Duration::from_millis(500);

/// A retry budget large enough that no test here spends it by accident.
const GENEROUS_RETRY_NS: u64 = 60_000_000_000;

/// The errno a capture device reports once it is unplugged.
const DEVICE_GONE_ERRNO: i32 = 19;

fn application_source() -> DynSource {
    Box::new(
        VideoTestSrc::new(WIDTH, HEIGHT, FRAMERATE, FRAMES_PER_LIFE)
            .with_pattern(Pattern::SmpteBars),
    )
}

fn application_frame_bytes() -> usize {
    WIDTH as usize * HEIGHT as usize * BYTES_PER_PIXEL
}

/// The caps of a test source, which answers `intercept_caps` without waiting.
fn immediate_caps(mut source: DynSource) -> Caps {
    let waker = core::task::Waker::noop();
    let mut cx = core::task::Context::from_waker(waker);
    let mut future = source.intercept_caps();
    match future.as_mut().poll(&mut cx) {
        core::task::Poll::Ready(caps) => caps.expect("a test source's caps"),
        core::task::Poll::Pending => panic!("a test source's intercept_caps must be immediate"),
    }
}

/// The default registry with `build` registered as the `fallbacksrc` main
/// source, and a count of how often the factory ran.
fn registry_with(build: fn() -> DynSource) -> (Registry, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = calls.clone();
    let caps = immediate_caps(build());
    let mut registry = default_registry();
    registry.register_fallbacksrc_main_source(FallbacksrcSourceFactory::new(move || {
        seen.fetch_add(1, Ordering::SeqCst);
        (build(), caps.clone())
    }));
    (registry, calls)
}

/// Frames that reached the sink, split by whose geometry they have.
#[derive(Debug, Default)]
struct Delivered {
    application: u64,
    fallback: u64,
}

/// Run `line` (which must end in `appsink channel=<channel>`) for `duration`
/// and sort what the sink received. The fallback never ends, so the run is
/// cancelled rather than awaited.
async fn run_for(registry: &Registry, line: &str, channel: &str, duration: Duration) -> Delivered {
    let pull = register_appsink_pull(channel);
    let graph = parse_launch(registry, line).unwrap_or_else(|e| panic!("{line}: {e}"));
    let mut delivered = Delivered::default();
    let drain = async {
        while let Some(frame) = pull.pull().await {
            let bytes = frame
                .domain
                .as_system_slice()
                .expect("videotestsrc emits system memory")
                .len();
            if bytes == application_frame_bytes() {
                delivered.application += 1;
            } else {
                delivered.fallback += 1;
            }
        }
    };
    let _ = tokio::time::timeout(duration, async {
        tokio::join!(run_graph(graph, &ZeroClock, 4), drain)
    })
    .await;
    delivered
}

#[test]
fn the_dummy_frame_size_differs_from_the_application_frame_size() {
    let dummy = default_registry()
        .make_source("videotestsrc")
        .expect("videotestsrc is a baseline source");
    let dimension = |name| {
        dummy
            .get_property(name)
            .and_then(|v| v.as_uint())
            .unwrap_or_else(|| panic!("videotestsrc reports its {name}")) as usize
    };
    assert_ne!(
        dimension("width") * dimension("height") * BYTES_PER_PIXEL,
        application_frame_bytes(),
        "the tests here count frames by size"
    );
}

#[tokio::test]
async fn the_registered_source_plays_then_the_fallback_takes_over() {
    let (registry, calls) = registry_with(application_source);
    let channel = "m1198_plays_once";
    let line = format!("fallbacksrc timeout={SHORT_TIMEOUT_NS} ! appsink channel={channel}");
    let delivered = run_for(&registry, &line, channel, RETRY_DELAY + SETTLE).await;

    assert_eq!(
        delivered.application, FRAMES_PER_LIFE,
        "the application's source is the main branch: {delivered:?}"
    );
    assert!(
        delivered.fallback > 0,
        "the dummy took over once the application's source ended: {delivered:?}"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "built once at parse, and not rebuilt without restart-on-eos"
    );
}

#[test]
fn a_lone_fallbacksrc_over_the_registered_source_ends_on_a_sink() {
    let (registry, calls) = registry_with(application_source);
    let graph = parse_launch(&registry, "fallbacksrc").expect("a lone fallbacksrc parses");
    let valid = graph
        .finish()
        .expect("a lone fallbacksrc is a whole pipeline");
    let kinds: Vec<NodeKind> = valid.topo().iter().map(|&n| valid.kind(n)).collect();

    assert_eq!(
        kinds.iter().filter(|k| **k == NodeKind::Sink).count(),
        1,
        "the single-stream expansion appends its automatic sink: {kinds:?}"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "the fan-out probe declines without building the source"
    );
}

#[tokio::test]
async fn restart_on_eos_rebuilds_the_registered_source() {
    let (registry, calls) = registry_with(application_source);
    let channel = "m1198_restart_on_eos";
    let line = format!(
        "fallbacksrc timeout={SHORT_TIMEOUT_NS} restart-on-eos=true retry-timeout={GENEROUS_RETRY_NS} ! appsink channel={channel}"
    );
    let delivered = run_for(&registry, &line, channel, RETRY_DELAY * 2 + SETTLE).await;

    let builds = calls.load(Ordering::SeqCst);
    assert!(
        builds >= 3,
        "two retry delays fit two rebuilds after the parse-time build, got {builds} builds"
    );
    assert!(
        delivered.application > FRAMES_PER_LIFE,
        "a rebuilt life reached the sink: {delivered:?}"
    );
}

/// Fails on every run without delivering: a capture device that is gone.
struct FailingSrc;

impl SourceLoop for FailingSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(immediate_caps(application_source())))
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, _out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async { Err(G2gError::Hardware(HardwareError::V4l2(DEVICE_GONE_ERRNO))) })
    }
}

fn failing_source() -> DynSource {
    Box::new(FailingSrc)
}

#[tokio::test]
async fn a_failing_registered_source_is_rebuilt_until_the_retry_budget_is_spent() {
    let (registry, calls) = registry_with(failing_source);
    // Two rebuilds start inside the budget, the third would start past it.
    let retry_budget = RETRY_DELAY * 2 + RETRY_DELAY / 2;
    let channel = "m1198_retry_budget";
    let line = format!(
        "fallbacksrc timeout={SHORT_TIMEOUT_NS} retry-timeout={} ! appsink channel={channel}",
        retry_budget.as_nanos()
    );
    let delivered = run_for(&registry, &line, channel, retry_budget + SETTLE).await;

    assert_eq!(
        calls.load(Ordering::SeqCst),
        3,
        "the parse-time build plus the two rebuilds inside the budget"
    );
    assert_eq!(delivered.application, 0);
    assert!(
        delivered.fallback > 0,
        "the dummy holds the output while the application's source is down: {delivered:?}"
    );
}
