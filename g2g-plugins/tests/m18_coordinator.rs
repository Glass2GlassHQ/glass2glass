//! M18 β (Session B) — coordinator control-channel topology.
//!
//! Validates the channel scaffolding `run_source_transform_sink` now
//! spawns: a single coordinator task drains an out-of-band control
//! channel alongside the data-plane arms (sink reports, coordinator
//! observes). No reconfiguration logic lives there yet, so this asserts
//! two things only:
//!   1. The coordinator observes one event per applied mid-stream
//!      `CapsChanged` at the sink boundary (`coordinator_events`).
//!   2. With no mid-stream caps change the count is `0` and the
//!      coordinator still terminates cleanly (channel closes when the
//!      sink arm drops its handle — no deadlock, no hang).
//!
//! Session E turns each observed `CapsChanged` event into a real
//! `Recascade`; this test pins the topology before that lands.

use core::future::Future;
use core::pin::Pin;
use std::boxed::Box;
use std::sync::{Arc, Mutex};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError,
    MemoryDomain, OutputSink, PipelineClock, PipelinePacket, Rate, VideoCodec, RawVideoFormat,
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

/// Scripted source: emits `initial` caps, then `target_frames` frames
/// followed by EOS.
struct NvSource {
    initial: Caps,
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
                };
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.target_frames as u64)
        })
    }
}

/// Pass-through transform that optionally injects one `CapsChanged`
/// downstream after a given input-frame count. `inject` of `None`
/// models a chain with no mid-stream caps change.
struct Boundary {
    inject: Option<(u32, Caps)>,
    injected: bool,
}

impl AsyncElement for Boundary {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
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
        let inject = self.inject.clone();
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(f) => {
                    let seq = f.sequence as u32;
                    out.push(PipelinePacket::DataFrame(f)).await?;
                    if let Some((after, caps)) = inject {
                        if !self.injected && seq + 1 >= after {
                            out.push(PipelinePacket::CapsChanged(caps)).await?;
                            self.injected = true;
                        }
                    }
                }
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

/// Recording sink declaring an NV12-of-any-geometry constraint, so a
/// mid-stream geometry change passes the runner's re-solve gate and is
/// applied (which is what makes the sink arm report to the coordinator).
#[derive(Default)]
struct SinkInner {
    received_frames: u32,
    saw_eos: bool,
}

struct RecordingSink {
    inner: Arc<Mutex<SinkInner>>,
}

impl AsyncElement for RecordingSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }))
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
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
                PipelinePacket::DataFrame(_) => g.received_frames += 1,
                PipelinePacket::Eos => g.saw_eos = true,
                _ => {}
            }
            Ok(())
        })
    }
}

/// One applied mid-stream `CapsChanged` at the sink boundary surfaces as
/// exactly one observed coordinator event.
#[tokio::test]
async fn applied_mid_stream_capschanged_reaches_coordinator() {
    let inner = Arc::new(Mutex::new(SinkInner::default()));
    let mut src = NvSource {
        initial: nv12_caps(640, 480),
        target_frames: 4,
    };
    let mut tx = Boundary {
        inject: Some((2, nv12_caps(1920, 1080))),
        injected: false,
    };
    let mut snk = RecordingSink {
        inner: Arc::clone(&inner),
    };
    let clock = ZeroClock;

    let stats =
        g2g_core::runtime::run_source_transform_sink(&mut src, &mut tx, &mut snk, &clock, 8)
            .await
            .expect("pipeline completes");

    let g = inner.lock().unwrap();
    assert!(g.saw_eos, "EOS must reach the sink");
    assert_eq!(g.received_frames, 4, "all frames reach the sink");
    assert_eq!(
        stats.coordinator_events, 1,
        "coordinator observes exactly the one applied mid-stream CapsChanged"
    );
}

/// No mid-stream caps change: zero events, and the coordinator still
/// terminates cleanly (the run returns rather than hanging).
#[tokio::test]
async fn no_caps_change_yields_zero_events_and_terminates() {
    let inner = Arc::new(Mutex::new(SinkInner::default()));
    let mut src = NvSource {
        initial: nv12_caps(1280, 720),
        target_frames: 5,
    };
    let mut tx = Boundary {
        inject: None,
        injected: false,
    };
    let mut snk = RecordingSink {
        inner: Arc::clone(&inner),
    };
    let clock = ZeroClock;

    let stats =
        g2g_core::runtime::run_source_transform_sink(&mut src, &mut tx, &mut snk, &clock, 8)
            .await
            .expect("pipeline completes without hanging");

    let g = inner.lock().unwrap();
    assert!(g.saw_eos);
    assert_eq!(g.received_frames, 5);
    assert_eq!(
        stats.coordinator_events, 0,
        "no caps change means no coordinator events"
    );
}
