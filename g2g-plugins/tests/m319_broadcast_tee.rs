//! M319: dynamic broadcast tee. Unlike the M310 dynamic *router*
//! (`run_source_router_dynamic`), which routes each frame round-robin to one
//! branch, `run_source_tee_dynamic` broadcasts every frame to *every* attached
//! branch via the M250 zero-copy frame-sharing path (`make_shareable` once, then
//! a refcount handle per branch). This is the runtime equivalent of GStreamer's
//! `tee` request pads: branches attach while the pipeline runs and each sees the
//! whole stream.

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use g2g_core::element::DynAsyncElement;
use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_source_tee_dynamic, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, MemoryDomain,
    OutputSink, PipelinePacket, Rate, RawVideoFormat,
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

/// Emits `n` `DataFrame`s then `Eos`.
struct CountingSource {
    n: u64,
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

/// Records how many data frames and caps changes it saw.
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
async fn tee_broadcasts_every_frame_to_every_branch() {
    const N: u64 = 6;
    let mut source = CountingSource { n: N };

    let f0 = Arc::new(AtomicUsize::new(0));
    let c0 = Arc::new(AtomicUsize::new(0));
    let f1 = Arc::new(AtomicUsize::new(0));
    let c1 = Arc::new(AtomicUsize::new(0));

    let (handle, run) = run_source_tee_dynamic(&mut source, 8);

    handle
        .add_branch(Box::new(RecordSink {
            frames: f0.clone(),
            caps_changes: c0.clone(),
        }) as Box<dyn DynAsyncElement>)
        .expect("add branch 0");
    handle
        .add_branch(Box::new(RecordSink {
            frames: f1.clone(),
            caps_changes: c1.clone(),
        }) as Box<dyn DynAsyncElement>)
        .expect("add branch 1");
    drop(handle);

    let stats = run.await.expect("dynamic tee run");

    let (got0, got1) = (f0.load(Ordering::Relaxed), f1.load(Ordering::Relaxed));
    // The defining difference from M310's router: BOTH branches see every frame.
    assert_eq!(
        got0, N as usize,
        "branch 0 received the whole stream (broadcast)"
    );
    assert_eq!(
        got1, N as usize,
        "branch 1 received the whole stream (broadcast)"
    );
    assert_eq!(stats.frames_emitted, N, "source emitted all frames once");
    assert_eq!(
        stats.frames_consumed,
        2 * N,
        "each branch consumed the full stream"
    );
    assert_eq!(
        c0.load(Ordering::Relaxed),
        1,
        "branch 0 saw the sticky caps once"
    );
    assert_eq!(
        c1.load(Ordering::Relaxed),
        1,
        "branch 1 saw the sticky caps once"
    );
}

#[tokio::test]
async fn single_tee_branch_still_gets_the_whole_stream() {
    const N: u64 = 4;
    let mut source = CountingSource { n: N };
    let frames = Arc::new(AtomicUsize::new(0));
    let caps_changes = Arc::new(AtomicUsize::new(0));

    let (handle, run) = run_source_tee_dynamic(&mut source, 8);
    handle
        .add_branch(Box::new(RecordSink {
            frames: frames.clone(),
            caps_changes: caps_changes.clone(),
        }) as Box<dyn DynAsyncElement>)
        .expect("add branch");
    drop(handle);

    let stats = run.await.expect("run");
    assert_eq!(
        frames.load(Ordering::Relaxed),
        N as usize,
        "the single branch gets every frame"
    );
    assert_eq!(stats.frames_consumed, N);
    assert_eq!(
        caps_changes.load(Ordering::Relaxed),
        1,
        "sticky caps delivered once"
    );
}
