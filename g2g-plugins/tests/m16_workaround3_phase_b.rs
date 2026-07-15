//! M16 workaround #3 Phase B — sink-side downstream subgraph re-solve.
//!
//! Builds on Phase A: decoders now stop swallowing input `CapsChanged`
//! (they validate + record). Phase B is the runner-side half: on every
//! forward `CapsChanged` at the boundary, the runner re-solves the
//! downstream subgraph (sink) against the new caps via `solve_linear`
//! before calling `configure_pipeline`. If the sink's declared
//! `caps_constraint_as_sink()` rejects, the runner drops the forward
//! `CapsChanged` and signals a reverse `Reconfigure::Renegotiate` into
//! the boundary instead of letting `configure_pipeline` see and accept
//! a shape the sink advertises it cannot handle.
//!
//! For the current 3-element runner the subgraph is one link, so the
//! observable change is structural — a sink with a restrictive
//! `Accepts(set)` constraint can reject a hostile mid-stream
//! `CapsChanged` via the solver even if its legacy
//! `configure_pipeline` would have silently accepted. Longer chains
//! (future) reconfigure every changed downstream link.

use core::future::Future;
use core::pin::Pin;
use std::boxed::Box;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError,
    MemoryDomain, OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn nv12_caps(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn rgba_caps(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Scripted source: emits the initial caps as its negotiated caps,
/// then drives a sequence of frames + one EOS through `run`. Mirrors
/// the M16 Phase A test source.
struct NvSource {
    initial: Caps,
    configured: bool,
    target_frames: u32,
}

impl SourceLoop for NvSource {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;

    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.initial.clone()))
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for i in 0..self.target_frames {
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(
                        vec![i as u8].into_boxed_slice(),
                    )),
                    timing: FrameTiming::default(),
                    sequence: i as u64,
                    meta: Default::default(),
                };
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.target_frames as u64)
        })
    }
}

/// Pass-through transform that, on a configurable input-frame count,
/// injects a hostile `CapsChanged` downstream — modeling a buggy
/// boundary whose declared `DerivedOutput` would never have produced
/// that shape. The Phase B runner check should reject this via the
/// sink's `caps_constraint_as_sink` and surface a reverse
/// `Renegotiate` rather than letting the sink's `configure_pipeline`
/// receive the bad caps.
struct HostileBoundary {
    inject_after_frames: u32,
    injected: bool,
    hostile_caps: Caps,
}

impl AsyncElement for HostileBoundary {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        // Accept anything; this transform's job is just to forward and
        // inject one bad CapsChanged.
        Ok(upstream.clone())
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        let hostile = self.hostile_caps.clone();
        let inject_after = self.inject_after_frames;
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(f) => {
                    let seq = f.sequence as u32;
                    out.push(PipelinePacket::DataFrame(f)).await?;
                    if !self.injected && seq + 1 >= inject_after {
                        out.push(PipelinePacket::CapsChanged(hostile)).await?;
                        self.injected = true;
                    }
                }
                PipelinePacket::Eos => {
                    // EOS forwarded by the runner; transforms are
                    // expected to no-op here. (Matching
                    // `IdentityTransform`'s convention.)
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

/// Sink that:
///   - Declares `Accepts(CapsSet::one(nv12_caps(...)))` for the
///     solver (rejects anything except matching NV12 at startup
///     negotiation).
///   - In `configure_pipeline` accepts EVERY video format silently —
///     modeling a legacy sink whose runtime check is laxer than its
///     declared constraint. Without Phase B, a hostile mid-stream
///     `CapsChanged` would pass `configure_pipeline` and be `process`'d
///     under wrong caps; with Phase B, the solver mediates first and
///     short-circuits.
///   - Records every `CapsChanged` it actually receives via `process`,
///     so the test can assert which ones leaked through.
#[derive(Default)]
struct PickySinkInner {
    received_caps_changes: VecDeque<Caps>,
    received_frames: u32,
    saw_eos: bool,
}

struct PickySink {
    accepts: Caps,
    inner: Arc<Mutex<PickySinkInner>>,
}

impl AsyncElement for PickySink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        // Declared constraint: only the configured NV12 shape. The
        // solver consults this on every mid-stream re-solve.
        CapsConstraint::Accepts(CapsSet::one(self.accepts.clone()))
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        // Deliberately lax: accept anything that lands here. Phase B's
        // solver gate is the only thing standing between the runner
        // and a hostile `CapsChanged`.
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        let inner = Arc::clone(&self.inner);
        Box::pin(async move {
            let mut g = inner.lock().unwrap();
            match packet {
                PipelinePacket::DataFrame(_) => {
                    g.received_frames += 1;
                }
                PipelinePacket::CapsChanged(c) => {
                    g.received_caps_changes.push_back(c);
                }
                PipelinePacket::Eos => {
                    g.saw_eos = true;
                }
                PipelinePacket::Flush | PipelinePacket::Segment(_) => {}
                _ => {}
            }
            Ok(())
        })
    }
}

/// Hostile mid-stream `CapsChanged` (RGBA from a "boundary" that
/// declares NV12 nowhere) is short-circuited by Phase B's solver gate
/// against the sink's declared `Accepts(NV12)` constraint — never
/// reaches `configure_pipeline` or `process`. The pipeline keeps
/// draining; the upstream Renegotiate signal is in-band but no element
/// reacts in this test.
#[tokio::test]
async fn hostile_mid_stream_capschanged_blocked_by_solver_gate() {
    let nv12 = nv12_caps(1280, 720);
    let rgba_hostile = rgba_caps(1280, 720);

    let mut src = NvSource {
        initial: nv12.clone(),
        configured: false,
        target_frames: 6,
    };
    let mut tx = HostileBoundary {
        inject_after_frames: 3,
        injected: false,
        hostile_caps: rgba_hostile.clone(),
    };
    let inner = Arc::new(Mutex::new(PickySinkInner::default()));
    let mut snk = PickySink {
        accepts: nv12.clone(),
        inner: Arc::clone(&inner),
    };
    let clock = ZeroClock;

    g2g_core::runtime::run_source_transform_sink(&mut src, &mut tx, &mut snk, &clock, 8)
        .await
        .expect("pipeline must complete despite hostile mid-stream CapsChanged");

    let g = inner.lock().unwrap();
    assert!(g.saw_eos, "EOS must reach the sink");
    assert_eq!(
        g.received_frames, 6,
        "all source frames must reach the sink"
    );
    // The hostile RGBA `CapsChanged` was rejected by the solver gate;
    // the sink's `process` never saw it. Pre-Phase B it would have
    // landed in the queue because `configure_pipeline` accepts anything.
    assert!(
        g.received_caps_changes.iter().all(|c| c != &rgba_hostile),
        "hostile RGBA CapsChanged must NOT reach the sink (got {:?})",
        g.received_caps_changes
    );
}

/// A mid-stream `CapsChanged` whose shape matches the sink's declared
/// `Accepts(set)` passes the solver gate and reaches `process` as
/// before. Regression guard: Phase B must not break the legitimate
/// path (e.g. a decoder emitting a real geometry-change `CapsChanged`).
#[tokio::test]
async fn matching_mid_stream_capschanged_still_propagates() {
    let nv12_small = nv12_caps(640, 480);
    let nv12_large = nv12_caps(1920, 1080);

    let mut src = NvSource {
        initial: nv12_small.clone(),
        configured: false,
        target_frames: 4,
    };
    // The "boundary" emits a NV12 geometry change — fully consistent
    // with the sink's Accepts(NV12-with-Any-dims) shape, which a sink
    // declaring Accepts(nv12_small) would NOT accept. So model the
    // declared constraint as Any-dims to admit the geometry change.
    let mut tx = HostileBoundary {
        inject_after_frames: 2,
        injected: false,
        hostile_caps: nv12_large.clone(),
    };
    let inner = Arc::new(Mutex::new(PickySinkInner::default()));
    // Declare an NV12-of-any-geometry sink so the larger geometry
    // passes the constraint check.
    let mut snk = PickySink {
        accepts: Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        },
        inner: Arc::clone(&inner),
    };
    let clock = ZeroClock;

    g2g_core::runtime::run_source_transform_sink(&mut src, &mut tx, &mut snk, &clock, 8)
        .await
        .expect("legitimate mid-stream CapsChanged must complete");

    let g = inner.lock().unwrap();
    assert!(g.saw_eos);
    assert_eq!(g.received_frames, 4);
    assert!(
        g.received_caps_changes.iter().any(|c| c == &nv12_large),
        "NV12 1920x1080 CapsChanged must reach the sink (got {:?})",
        g.received_caps_changes
    );
}
