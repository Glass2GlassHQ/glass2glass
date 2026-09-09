//! M1166: `fallbacksrc`'s `manual-unblock`. With an `UnblockHandle` set, every
//! life of a `RestartSrc` waits for the application to release it before it
//! delivers, and the wait re-arms on each restart, so a restarted source is held
//! again. The launch keyword takes `manual-unblock=true` only when the registry
//! carries the handle the line would be released through.
//!
//! `default_registry` is `std`-gated, so this file is too: run with
//! `cargo test -p g2g-plugins --features std`.
#![cfg(feature = "std")]

use core::future::Future;
use core::task::{Context, Poll};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use g2g_core::runtime::{
    parse_launch, run_graph, ParseError, RestartPolicy, SourceLoop, UnblockHandle, UriRebuild,
};
use g2g_core::{Bus, BusHandle, BusMessage, G2gError, PipelineClock, PipelinePacket, PushOutcome};
use g2g_plugins::fallbacksrc::{RestartSrc, RETRY_DELAY};
use g2g_plugins::registry::default_registry;
use g2g_plugins::videotestsrc::VideoTestSrc;

type DynSource = Box<dyn g2g_core::runtime::DynSourceLoop>;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Slack past the last event a timed run waits for.
const SETTLE: Duration = Duration::from_millis(500);

/// Frames each generated life emits before ending.
const FRAMES_PER_LIFE: u64 = 3;

/// The generated test source's geometry and rate. Small, so a life is quick.
const WIDTH: u32 = 16;
const HEIGHT: u32 = 16;
const FRAMERATE: u32 = 30;

/// A stall check that never fires.
const NO_STALL_CHECK: u64 = 0;

/// A retry budget large enough that no test here spends it by accident.
const GENEROUS_RETRY_NS: u64 = 60_000_000_000;

/// The instance name the directly-driven wrapper is given, so a bus report has
/// something to label itself with.
const INSTANCE: &str = "held-under-test";

/// Bus backlog: every test here posts a handful of messages and drains after.
const BUS_CAPACITY: usize = 64;

/// A sink the test can read while the running wrapper still holds its own
/// handle on it.
#[derive(Clone, Default)]
struct Recorder {
    frames: Arc<AtomicUsize>,
}

impl Recorder {
    fn frames(&self) -> u64 {
        self.frames.load(Ordering::SeqCst) as u64
    }
}

impl g2g_core::OutputSink for Recorder {
    fn poll_push(
        &mut self,
        _cx: &mut Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        if matches!(packet, PipelinePacket::DataFrame(_)) {
            self.frames.fetch_add(1, Ordering::SeqCst);
        }
        Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn generated_source() -> DynSource {
    Box::new(VideoTestSrc::new(WIDTH, HEIGHT, FRAMERATE, FRAMES_PER_LIFE))
}

fn rebuild() -> UriRebuild {
    Box::new(move || {
        let mut source = generated_source();
        let caps = block_on(source.intercept_caps()).expect("a test source's caps");
        Ok((source, caps))
    })
}

/// The rebuild contract returns the source's declared caps alongside it; the
/// test sources answer `intercept_caps` synchronously, so polling once is enough.
fn block_on<F: Future>(future: F) -> F::Output {
    let waker = core::task::Waker::noop();
    let mut cx = Context::from_waker(waker);
    let mut future = core::pin::pin!(future);
    match future.as_mut().poll(&mut cx) {
        Poll::Ready(v) => v,
        Poll::Pending => panic!("a test source's intercept_caps must be immediate"),
    }
}

fn policy(restart_on_eos: bool) -> RestartPolicy {
    RestartPolicy {
        restart_on_eos,
        restart_timeout_ns: NO_STALL_CHECK,
        retry_timeout_ns: GENEROUS_RETRY_NS,
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

fn restart_reports(bus: &Bus) -> usize {
    let mut seen = 0;
    while let Some(message) = bus.try_recv() {
        if matches!(message, BusMessage::SourceRestart { .. }) {
            seen += 1;
        }
    }
    seen
}

fn held(handle: &UnblockHandle, restart_on_eos: bool, bus: Option<&BusHandle>) -> RestartSrc {
    let mut wrapper = RestartSrc::new(generated_source(), rebuild(), policy(restart_on_eos), None)
        .with_unblock_handle(handle.clone());
    if let Some(bus) = bus {
        SourceLoop::set_instance_name(&mut wrapper, INSTANCE.to_string());
        SourceLoop::set_bus(&mut wrapper, bus.clone());
    }
    wrapper
}

#[tokio::test]
async fn a_held_life_delivers_nothing_until_it_is_unblocked() {
    let handle = UnblockHandle::new();
    let (bus, posts) = Bus::new(BUS_CAPACITY);
    let mut wrapper = configured(held(&handle, false, Some(&posts))).await;
    let mut sink = Recorder::default();
    let recorded = sink.clone();

    let mut run = core::pin::pin!(SourceLoop::run(&mut wrapper, &mut sink));
    let waker = core::task::Waker::noop();
    let mut cx = Context::from_waker(waker);
    assert!(
        run.as_mut().poll(&mut cx).is_pending(),
        "the first life parks on the handle"
    );
    assert_eq!(recorded.frames(), 0, "a held life delivers nothing");
    assert_eq!(
        restart_reports(&bus),
        0,
        "a held life does not report itself running"
    );

    handle.unblock();
    let total = tokio::time::timeout(SETTLE, run.as_mut())
        .await
        .expect("the released life ends the stream")
        .expect("the run succeeds");

    assert_eq!(total, FRAMES_PER_LIFE);
    assert_eq!(recorded.frames(), FRAMES_PER_LIFE, "the release delivered");
    assert_eq!(handle.released_count(), 1, "one life took one release");
    assert!(
        restart_reports(&bus) > 0,
        "the released life reports itself"
    );
}

#[tokio::test]
async fn every_restarted_life_waits_for_its_own_release() {
    let handle = UnblockHandle::new();
    let mut wrapper = configured(held(&handle, true, None)).await;
    let mut sink = Recorder::default();
    let recorded = sink.clone();
    // Long enough for a life to end, the retry delay to pass, and the next life
    // to be built and park on the handle.
    let one_life = RETRY_DELAY + SETTLE;

    let mut run = core::pin::pin!(SourceLoop::run(&mut wrapper, &mut sink));
    handle.unblock();
    let _ = tokio::time::timeout(one_life, run.as_mut()).await;
    assert_eq!(
        recorded.frames(),
        FRAMES_PER_LIFE,
        "life 1 played and life 2 is held"
    );
    assert_eq!(handle.released_count(), 1, "life 2 took no release");

    handle.unblock();
    let _ = tokio::time::timeout(one_life, run.as_mut()).await;
    assert_eq!(
        recorded.frames(),
        FRAMES_PER_LIFE * 2,
        "life 2 played on its own release and life 3 is held"
    );
    assert_eq!(handle.released_count(), 2, "one release per life");
}

#[tokio::test]
async fn without_a_handle_the_first_life_delivers_at_once() {
    let mut wrapper = configured(RestartSrc::new(
        generated_source(),
        rebuild(),
        policy(false),
        None,
    ))
    .await;
    let mut sink = Recorder::default();
    let recorded = sink.clone();
    let total = tokio::time::timeout(SETTLE, SourceLoop::run(&mut wrapper, &mut sink))
        .await
        .expect("an unheld life needs no release")
        .expect("the run succeeds");

    assert_eq!(total, FRAMES_PER_LIFE);
    assert_eq!(recorded.frames(), FRAMES_PER_LIFE);
}

fn temp_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g_m1166_{}_{tag}", std::process::id()))
}

/// A PNM still `fallbacksrc` can open, decode, and play once, written by running
/// the encoder so the bytes are real.
async fn still(tag: &str) -> PathBuf {
    let path = temp_path(tag);
    let reg = default_registry();
    let line = format!(
        "videotestsrc num-buffers=1 pattern=smpte ! pnmenc ! filesink location={}",
        path.display()
    );
    let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("{line}: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("{line}: {e:?}"));
    path
}

#[test]
fn manual_unblock_without_a_registered_handle_is_rejected() {
    let reg = default_registry();
    assert_eq!(
        parse_launch(
            &reg,
            "fallbacksrc uri=file:///x.pnm manual-unblock=true ! fakesink"
        )
        .unwrap_err(),
        ParseError::MissingUnblockHandle("fallbacksrc".to_string())
    );
}

#[tokio::test]
async fn manual_unblock_parses_against_a_registered_handle() {
    let main = still("parse_main.pnm").await;
    let handle = UnblockHandle::new();
    let mut reg = default_registry();
    reg.register_unblock_handle(&handle);
    let line = format!(
        "fallbacksrc uri=file://{} manual-unblock=true ! fakesink",
        main.display()
    );
    parse_launch(&reg, &line).unwrap_or_else(|e| panic!("{line}: {e}"));

    // The default off keeps a line without the property parsing on a registry
    // that has no handle at all.
    let plain = default_registry();
    let line = format!("fallbacksrc uri=file://{} ! fakesink", main.display());
    parse_launch(&plain, &line).unwrap_or_else(|e| panic!("{line}: {e}"));

    std::fs::remove_file(&main).ok();
}
