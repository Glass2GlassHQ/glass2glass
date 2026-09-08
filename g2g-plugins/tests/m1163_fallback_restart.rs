//! M1163: `fallbacksrc` rebuilds a dead source. The launch keyword wraps its
//! `uri=` and `fallback-uri=` sources in `RestartSrc`, which rebuilds the source
//! when it fails, stalls for `restart-timeout`, or ends under `restart-on-eos`,
//! and gives up once `retry-timeout` of failures has passed.
//!
//! The end-to-end checks use PNM stills: `file://` builds a fresh file source per
//! rebuild, so a restarted still replays from the start and the number of times
//! its pixels reach the sink is the number of lives it had. The wrapper's own
//! rules run against `RestartSrc` directly with a recording sink.
//!
//! `default_registry` is `std`-gated, so this file is too: run with
//! `cargo test -p g2g-plugins --features std`.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{
    parse_launch, run_graph, ParseError, RestartPolicy, SourceLoop, UriRebuild,
};
use g2g_core::{
    Caps, ConfigureOutcome, G2gError, MemoryDomain, OutputSink, PipelineClock, PipelinePacket,
    PushOutcome,
};
use g2g_plugins::fallbacksrc::{RestartSrc, RETRY_DELAY};
use g2g_plugins::filesrc::FileSrc;
use g2g_plugins::registry::default_registry;
use g2g_plugins::videotestsrc::VideoTestSrc;

type DynSource = Box<dyn g2g_core::runtime::DynSourceLoop>;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Slack past the last rebuild a timed run waits for, so the life it starts has
/// time to deliver.
const SETTLE: Duration = Duration::from_millis(500);

/// A stall timeout short enough that a stalled life is caught well inside one
/// retry delay.
const SHORT_STALL: Duration = Duration::from_millis(200);

/// Frames each generated life emits before ending.
const FRAMES_PER_LIFE: u64 = 3;

/// The generated test source's geometry and rate. Small, so a life is quick.
const WIDTH: u32 = 16;
const HEIGHT: u32 = 16;
const FRAMERATE: u32 = 30;

/// A stall check that never fires: the wrapper's zero-means-disabled reading.
const NO_STALL_CHECK: u64 = 0;

/// A retry budget large enough that no test here spends it by accident.
const GENEROUS_RETRY_NS: u64 = 60_000_000_000;

fn temp_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g_m1163_{}_{tag}", std::process::id()))
}

/// Tests in one binary share a process, so two of them writing and deleting the
/// same fixture path race; each write takes a path of its own.
fn unique_tag(tag: &str) -> String {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    format!("{}_{tag}", NEXT.fetch_add(1, Ordering::SeqCst))
}

/// Write a fixture by running `line` (which must end in a `filesink`) and return
/// its path, so the bytes come from the real encoder.
async fn fixture(tag: &str, line: &str) -> PathBuf {
    let path = temp_path(&unique_tag(tag));
    let reg = default_registry();
    let line = format!("{line} ! filesink location={}", path.display());
    let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("{line}: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("{line}: {e:?}"));
    path
}

/// RGB pixels that are not black in a `pnmdec` output dump.
fn lit_pixels(bytes: &[u8]) -> usize {
    bytes
        .as_chunks::<3>()
        .0
        .iter()
        .filter(|px| **px != [0, 0, 0])
        .count()
}

/// How many lit pixels one pass of the SMPTE still leaves at the sink, read off
/// the same encode + decode path the `fallbacksrc` line runs.
async fn smpte_pass_pixels() -> usize {
    let path = fixture(
        "smpte_pass.raw",
        "videotestsrc num-buffers=1 pattern=smpte ! pnmenc ! pnmdec",
    )
    .await;
    let bytes = std::fs::read(&path).expect("the reference pass wrote its frame");
    std::fs::remove_file(&path).ok();
    let lit = lit_pixels(&bytes);
    assert!(lit > 0, "an SMPTE frame has lit pixels");
    lit
}

/// Run a `fallbacksrc` line whose main is an SMPTE still and whose fallback is a
/// black still, for `duration`, and return how many lit pixels reached the sink.
async fn lit_pixels_after(restart_props: &str, duration: Duration) -> usize {
    let main = fixture(
        &format!("main_{}.pnm", restart_props.len()),
        "videotestsrc num-buffers=1 pattern=smpte ! pnmenc",
    )
    .await;
    let fallback = fixture(
        &format!("fallback_{}.pnm", restart_props.len()),
        "videotestsrc num-buffers=1 pattern=black ! pnmenc",
    )
    .await;
    let out = temp_path(&unique_tag("out.raw"));
    let line = format!(
        "fallbacksrc uri=file://{} fallback-uri=file://{} {restart_props} ! filesink location={}",
        main.display(),
        fallback.display(),
        out.display()
    );
    let reg = default_registry();
    let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("{line}: {e}"));
    // Cancelled, not awaited: the fallback still loops forever.
    let _ = tokio::time::timeout(duration, run_graph(graph, &ZeroClock, 4)).await;
    let bytes = std::fs::read(&out).expect("the sink wrote what the switch forwarded");
    std::fs::remove_file(&main).ok();
    std::fs::remove_file(&fallback).ok();
    std::fs::remove_file(&out).ok();
    lit_pixels(&bytes)
}

#[tokio::test]
async fn restart_on_eos_replays_the_main_uri() {
    let one_pass = smpte_pass_pixels().await;
    // Lives start at 0 and then one retry delay apart: three delays fit three
    // more.
    let lit = lit_pixels_after("restart-on-eos=true", RETRY_DELAY * 3 + SETTLE).await;
    assert!(
        lit > one_pass,
        "the SMPTE still should reach the sink more than once: {lit} lit pixels vs {one_pass} per pass"
    );
}

#[tokio::test]
async fn default_policy_plays_the_main_uri_once() {
    let one_pass = smpte_pass_pixels().await;
    let lit = lit_pixels_after("", RETRY_DELAY * 2 + SETTLE).await;
    assert_eq!(
        lit, one_pass,
        "without restart-on-eos the main still plays exactly once"
    );
}

#[tokio::test]
async fn restart_properties_reject_bad_values() {
    let reg = default_registry();
    for (key, value) in [
        ("retry-timeout", "soon"),
        ("restart-timeout", "-1"),
        ("restart-on-eos", "maybe"),
    ] {
        let line = format!("fallbacksrc uri=file:///nonexistent {key}={value} ! fakesink");
        match parse_launch(&reg, &line) {
            Err(ParseError::BadValue {
                key: k, value: v, ..
            }) => {
                assert_eq!((k.as_str(), v.as_str()), (key, value));
            }
            other => panic!("{line}: expected BadValue, got {other:?}"),
        }
    }
}

/// Records every packet the wrapper pushes.
#[derive(Default)]
struct Collect {
    packets: Vec<PipelinePacket>,
}

impl Collect {
    fn pts(&self) -> Vec<u64> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some(f.timing.pts_ns),
                _ => None,
            })
            .collect()
    }

    fn eos_count(&self) -> usize {
        self.packets
            .iter()
            .filter(|p| matches!(p, PipelinePacket::Eos))
            .count()
    }
}

impl OutputSink for Collect {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        self.packets.push(packet);
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn generated_source() -> DynSource {
    Box::new(VideoTestSrc::new(WIDTH, HEIGHT, FRAMERATE, FRAMES_PER_LIFE))
}

/// A rebuild closure over `build` that counts how often it was called.
fn counting_rebuild(
    build: impl Fn() -> DynSource + Send + 'static,
) -> (UriRebuild, Arc<AtomicUsize>) {
    let count = Arc::new(AtomicUsize::new(0));
    let seen = count.clone();
    let rebuild: UriRebuild = Box::new(move || {
        seen.fetch_add(1, Ordering::SeqCst);
        let mut source = build();
        let caps = futures_lite_block_on(source.intercept_caps()).expect("a test source's caps");
        Ok((source, caps))
    });
    (rebuild, count)
}

/// The rebuild contract returns the source's declared caps alongside it; the
/// test sources answer `intercept_caps` synchronously, so polling once is enough.
fn futures_lite_block_on<F: Future>(future: F) -> F::Output {
    let waker = core::task::Waker::noop();
    let mut cx = core::task::Context::from_waker(waker);
    let mut future = core::pin::pin!(future);
    match future.as_mut().poll(&mut cx) {
        core::task::Poll::Ready(v) => v,
        core::task::Poll::Pending => panic!("a test source's intercept_caps must be immediate"),
    }
}

/// Negotiate and configure the wrapper the way the runner does before `run`.
async fn configured(mut wrapper: RestartSrc) -> RestartSrc {
    let caps = SourceLoop::intercept_caps(&mut wrapper)
        .await
        .expect("the first life negotiates");
    SourceLoop::configure_pipeline(&mut wrapper, &caps).expect("the first life configures");
    wrapper
}

#[tokio::test]
async fn timeline_never_goes_backwards_across_restarts() {
    let (rebuild, rebuilds) = counting_rebuild(generated_source);
    let policy = RestartPolicy {
        restart_on_eos: true,
        restart_timeout_ns: NO_STALL_CHECK,
        retry_timeout_ns: GENEROUS_RETRY_NS,
    };
    let mut wrapper = configured(RestartSrc::new(generated_source(), rebuild, policy, None)).await;
    let mut sink = Collect::default();
    let _ = tokio::time::timeout(
        RETRY_DELAY * 2 + SETTLE,
        SourceLoop::run(&mut wrapper, &mut sink),
    )
    .await;

    let lives = rebuilds.load(Ordering::SeqCst) as u64 + 1;
    assert!(
        lives >= 3,
        "two retry delays fit two rebuilds, got {lives} lives"
    );
    let pts = sink.pts();
    assert_eq!(
        pts.len() as u64,
        lives * FRAMES_PER_LIFE,
        "every life's frames reached the sink: {pts:?}"
    );
    assert!(
        pts.windows(2).all(|w| w[0] < w[1]),
        "PTS must rise across the restart boundary: {pts:?}"
    );
    assert_eq!(sink.eos_count(), 0, "inner EOS packets are swallowed");
}

/// The wrapper's own sources fail to open: a file that is not there.
fn missing_file_source(caps: Caps) -> DynSource {
    Box::new(FileSrc::new(temp_path("missing.bin"), caps))
}

#[tokio::test]
async fn retry_budget_ends_the_stream_after_repeated_failure() {
    let caps = futures_lite_block_on(generated_source().intercept_caps()).expect("test caps");
    let rebuild_caps = caps.clone();
    let (rebuild, rebuilds) = counting_rebuild(move || missing_file_source(rebuild_caps.clone()));
    // Two rebuilds start inside the budget; the third would start past it.
    let retry_budget = RETRY_DELAY * 2 + RETRY_DELAY / 2;
    let policy = RestartPolicy {
        restart_on_eos: false,
        restart_timeout_ns: NO_STALL_CHECK,
        retry_timeout_ns: retry_budget.as_nanos() as u64,
    };
    let mut wrapper = configured(RestartSrc::new(
        missing_file_source(caps),
        rebuild,
        policy,
        None,
    ))
    .await;
    let mut sink = Collect::default();
    let started = Instant::now();
    let frames = SourceLoop::run(&mut wrapper, &mut sink)
        .await
        .expect("giving up ends the stream cleanly rather than failing the run");

    assert_eq!(frames, 0);
    assert_eq!(
        rebuilds.load(Ordering::SeqCst),
        2,
        "rebuilds inside the budget"
    );
    let elapsed = started.elapsed();
    assert!(
        elapsed >= RETRY_DELAY * 2 && elapsed < retry_budget,
        "each rebuild waited its delay and the third was not attempted: {elapsed:?}"
    );
    assert_eq!(sink.eos_count(), 1, "one terminal EOS: {:?}", sink.packets);
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
        core::future::ready(futures_lite_block_on(generated_source().intercept_caps()))
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
async fn stalled_source_is_rebuilt_after_restart_timeout() {
    let (rebuild, rebuilds) = counting_rebuild(|| Box::new(StallSrc) as DynSource);
    let policy = RestartPolicy {
        restart_on_eos: false,
        restart_timeout_ns: SHORT_STALL.as_nanos() as u64,
        retry_timeout_ns: GENEROUS_RETRY_NS,
    };
    let mut wrapper = configured(RestartSrc::new(Box::new(StallSrc), rebuild, policy, None)).await;
    let mut sink = Collect::default();
    // The first life stalls, waits out one retry delay, and its replacement
    // delivers before the run is cut.
    let _ = tokio::time::timeout(
        SHORT_STALL + RETRY_DELAY + SETTLE,
        SourceLoop::run(&mut wrapper, &mut sink),
    )
    .await;

    assert_eq!(
        rebuilds.load(Ordering::SeqCst),
        1,
        "the stalled life was replaced once"
    );
    assert_eq!(
        sink.pts().len(),
        2,
        "each life delivered its one frame: {:?}",
        sink.packets
    );
}

#[tokio::test]
async fn zero_restart_timeout_disables_the_stall_check() {
    let (rebuild, rebuilds) = counting_rebuild(|| Box::new(StallSrc) as DynSource);
    let policy = RestartPolicy {
        restart_on_eos: false,
        restart_timeout_ns: NO_STALL_CHECK,
        retry_timeout_ns: GENEROUS_RETRY_NS,
    };
    let mut wrapper = configured(RestartSrc::new(Box::new(StallSrc), rebuild, policy, None)).await;
    let mut sink = Collect::default();
    let _ = tokio::time::timeout(
        SHORT_STALL + RETRY_DELAY + SETTLE,
        SourceLoop::run(&mut wrapper, &mut sink),
    )
    .await;

    assert_eq!(
        rebuilds.load(Ordering::SeqCst),
        0,
        "a stalled life is left alone"
    );
    assert_eq!(sink.pts().len(), 1);
}
