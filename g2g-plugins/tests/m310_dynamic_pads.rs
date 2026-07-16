//! M310: runtime request pads. A dynamic fan-out's branches are attached *while
//! the pipeline runs* via `DynamicFanoutHandle::add_branch` (the runtime
//! equivalent of GStreamer tee request pads), not declared at build time. The
//! router routes each frame round-robin across the attached branches and replays
//! the fan-out's sticky caps to every branch on attach, so a branch configures
//! correctly without having seen the original negotiation.
//!
//! Frames are not `Clone` (zero-copy design), so this routes rather than
//! broadcasts; a true broadcast tee (every frame to every branch) needs frame
//! sharing and is a follow-up. Fan-in request pads (a muxer input added at
//! runtime) are also a follow-up.

use core::future::Future;
use core::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use g2g_core::element::DynAsyncElement;
use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_source_router_dynamic, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, MemoryDomain,
    OutputSink, PipelinePacket, RawVideoFormat, Rate,
};

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(16),
        height: Dim::Fixed(16),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn make_frame(seq: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
        timing: FrameTiming::default(),
        sequence: seq,
        meta: Default::default(),
    }
}

/// Emits `n` `DataFrame`s then `Eos`. Caps are fixed (the runner reads them via
/// `intercept_caps` to seed the fan-out's sticky caps).
struct CountingSource {
    n: u64,
    configured: bool,
}

impl SourceLoop for CountingSource {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(caps()))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(caps()))))
    }

    fn configure_pipeline(&mut self, _absolute: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for seq in 0..self.n {
                out.push(PipelinePacket::DataFrame(make_frame(seq))).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.n)
        })
    }
}

/// Records how many data frames and caps changes it saw, into shared counters
/// the test reads after the run (the sink itself is owned by the branch).
struct RecordSink {
    frames: Arc<AtomicUsize>,
    caps_changes: Arc<AtomicUsize>,
}

impl AsyncElement for RecordSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, _absolute: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        match packet {
            PipelinePacket::DataFrame(_) => {
                self.frames.fetch_add(1, Ordering::Relaxed);
            }
            PipelinePacket::CapsChanged(_) => {
                self.caps_changes.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
        Box::pin(async { Ok(()) })
    }
}

#[tokio::test]
async fn runtime_branches_split_the_stream_and_get_sticky_caps() {
    const N: u64 = 8;
    let mut source = CountingSource { n: N, configured: false };

    let f0 = Arc::new(AtomicUsize::new(0));
    let c0 = Arc::new(AtomicUsize::new(0));
    let f1 = Arc::new(AtomicUsize::new(0));
    let c1 = Arc::new(AtomicUsize::new(0));

    let (handle, run) = run_source_router_dynamic(&mut source, 8);

    // Request two output pads at runtime, before the run is driven: they are
    // queued on the control channel and folded in on the first router poll.
    handle
        .add_branch(Box::new(RecordSink { frames: f0.clone(), caps_changes: c0.clone() })
            as Box<dyn DynAsyncElement>)
        .expect("add branch 0");
    handle
        .add_branch(Box::new(RecordSink { frames: f1.clone(), caps_changes: c1.clone() })
            as Box<dyn DynAsyncElement>)
        .expect("add branch 1");
    // Dropping the handle closes the control channel so the router stops
    // watching for new branches; the two already queued still attach.
    drop(handle);

    let stats = run.await.expect("dynamic fan-out run");

    let (got0, got1) = (f0.load(Ordering::Relaxed), f1.load(Ordering::Relaxed));
    eprintln!("M310: branch0={got0} frames, branch1={got1} frames; emitted={}", stats.frames_emitted);

    // The source emitted N; every frame reached exactly one branch (routed).
    assert_eq!(stats.frames_emitted, N, "source emitted all frames");
    assert_eq!(got0 + got1, N as usize, "every frame routed to exactly one branch");
    assert_eq!(stats.frames_consumed, N, "run stats account for all consumed frames");
    // Both runtime-attached branches actually participated and were configured
    // by the replayed sticky caps.
    assert!(got0 > 0 && got1 > 0, "both runtime branches received frames ({got0}, {got1})");
    assert_eq!(c0.load(Ordering::Relaxed), 1, "branch 0 saw the sticky caps once");
    assert_eq!(c1.load(Ordering::Relaxed), 1, "branch 1 saw the sticky caps once");
}

#[tokio::test]
async fn single_runtime_branch_receives_the_whole_stream() {
    const N: u64 = 5;
    let mut source = CountingSource { n: N, configured: false };
    let frames = Arc::new(AtomicUsize::new(0));
    let caps_changes = Arc::new(AtomicUsize::new(0));

    let (handle, run) = run_source_router_dynamic(&mut source, 8);
    handle
        .add_branch(Box::new(RecordSink { frames: frames.clone(), caps_changes: caps_changes.clone() })
            as Box<dyn DynAsyncElement>)
        .expect("add branch");
    drop(handle);

    let stats = run.await.expect("run");
    assert_eq!(frames.load(Ordering::Relaxed), N as usize, "single branch gets every frame");
    assert_eq!(stats.frames_consumed, N);
    assert_eq!(caps_changes.load(Ordering::Relaxed), 1, "sticky caps delivered once");
}
