#![cfg(feature = "embassy-link")]
//! M264: a three-stage pipeline runs as three *independently spawned* tasks on a
//! real Embassy executor (`platform-std`), not the single `block_on`-joined
//! future of M43/M45/M260. A source, an `IdentityTransform`, and a consumer are
//! each their own `#[embassy_executor::task]`, wired by two static
//! `SharedPacketChannel`s; the executor's own scheduler interleaves them. Proves
//! the no_std runtime drives a genuine multi-task pipeline on the Embassy
//! executor primitive an RTOS app uses, with the inter-task links statically
//! allocated.
//!
//! `run_until` is the std platform's testable entry point: it polls the executor
//! and returns once the consumer raises `DONE`, so the host test terminates
//! instead of the diverging `run()` an embedded `fn main() -> !` would call.

use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use embassy_executor::{Executor, Spawner};

use g2g_core::runtime::SourceLoop;
use g2g_core::{AsyncElement, Caps, Dim, OutputSink, PipelinePacket, Rate, RawVideoFormat};
use g2g_plugins::embassylink::SharedPacketChannel;
use g2g_plugins::identity::IdentityTransform;
use g2g_plugins::videotestsrc::VideoTestSrc;

const FRAMES: u64 = 6;
/// Link depth < FRAMES, so the source blocks on backpressure and the executor
/// must interleave the three tasks rather than draining one to completion first.
const DEPTH: usize = 4;

// Inter-task links live in `static`s so the spawned 'static tasks reach them by
// reference. `SharedPacketChannel` (CriticalSectionRawMutex) is `Sync`;
// `SinglePacketChannel` (NoopRawMutex) is not, so it cannot be a `static`.
static SRC_TO_XFORM: SharedPacketChannel<DEPTH> = SharedPacketChannel::new();
static XFORM_TO_SINK: SharedPacketChannel<DEPTH> = SharedPacketChannel::new();

// Per-stage observations, read back after the executor returns. Single-threaded
// here (one executor, one thread), so the ordering is incidental; DONE uses
// Release/Acquire to pair cleanly with the `run_until` predicate regardless.
static PRODUCED: AtomicU32 = AtomicU32::new(0);
static FORWARDED: AtomicU32 = AtomicU32::new(0);
static CONSUMED: AtomicU32 = AtomicU32::new(0);
static DONE: AtomicBool = AtomicBool::new(false);

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(16),
        height: Dim::Fixed(8),
        framerate: Rate::Fixed(30 << 16),
    }
}

#[embassy_executor::task]
async fn source_task(out: &'static SharedPacketChannel<DEPTH>) {
    let mut src = VideoTestSrc::new(16, 8, 30, FRAMES);
    src.configure_pipeline(&caps()).expect("source configure");
    let mut sink = out.sink();
    let produced = src.run(&mut sink).await.expect("source run");
    PRODUCED.store(produced as u32, Ordering::Relaxed);
}

#[embassy_executor::task]
async fn transform_task(
    input: &'static SharedPacketChannel<DEPTH>,
    out: &'static SharedPacketChannel<DEPTH>,
) {
    let mut elem = IdentityTransform::new();
    elem.configure_pipeline(&caps()).expect("transform configure");
    let rx = input.receiver();
    let mut sink = out.sink();
    loop {
        let packet = rx.receive().await;
        let eos = matches!(packet, PipelinePacket::Eos);
        elem.process(packet, &mut sink).await.expect("transform process");
        if eos {
            // Identity does not emit Eos itself (the runner's contract); the
            // driving loop forwards it so the downstream sink terminates.
            sink.push(PipelinePacket::Eos).await.expect("forward eos");
            break;
        }
    }
    FORWARDED.store(elem.forwarded() as u32, Ordering::Relaxed);
}

#[embassy_executor::task]
async fn sink_task(input: &'static SharedPacketChannel<DEPTH>) {
    let rx = input.receiver();
    let mut frames = 0u32;
    loop {
        match rx.receive().await {
            PipelinePacket::DataFrame(_) => frames += 1,
            PipelinePacket::Eos => break,
            _ => {}
        }
    }
    CONSUMED.store(frames, Ordering::Relaxed);
    DONE.store(true, Ordering::Release);
}

#[test]
fn three_tasks_stream_a_pipeline_on_the_embassy_executor() {
    let mut executor = Executor::new();
    // SAFETY: `executor` is a local that outlives this `run_until` call, which
    // returns once `DONE` is set (before the local is dropped at end of scope).
    // This is the same lifetime upgrade embassy's `#[main]` macro performs for a
    // diverging `fn main() -> !`; here `run_until` bounds it, so the &'static mut
    // is dead before the executor drops, and nothing else aliases it.
    let executor: &'static mut Executor = unsafe { core::mem::transmute(&mut executor) };

    executor.run_until(
        |spawner: Spawner| {
            // The `#[task]` fn allocates from its static pool (fallible); the
            // returned `SpawnToken` is then handed to the infallible `spawn`.
            spawner.spawn(source_task(&SRC_TO_XFORM).expect("alloc source task"));
            spawner.spawn(
                transform_task(&SRC_TO_XFORM, &XFORM_TO_SINK).expect("alloc transform task"),
            );
            spawner.spawn(sink_task(&XFORM_TO_SINK).expect("alloc sink task"));
        },
        || DONE.load(Ordering::Acquire),
    );

    assert_eq!(PRODUCED.load(Ordering::Relaxed), FRAMES as u32, "source produced all frames");
    assert_eq!(
        FORWARDED.load(Ordering::Relaxed),
        FRAMES as u32,
        "transform task forwarded every frame"
    );
    assert_eq!(
        CONSUMED.load(Ordering::Relaxed),
        FRAMES as u32,
        "sink consumed every frame across both static links"
    );
}
