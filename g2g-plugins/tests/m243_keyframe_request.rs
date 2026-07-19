//! Keyframe-request reverse channel: a sink that needs a fresh keyframe (a
//! WebRTC sink on a remote PLI) returns `Reconfigure::ForceKeyframe` from
//! `take_reconfigure`; the runner forwards it up the incoming link, and the
//! producing source observes it as `PushOutcome::Reconfigure(ForceKeyframe)`.
//! Mirrors the M174 upstream-QoS test, for the keyframe path.

#![cfg(feature = "std")]

use core::future::{ready, Future, Ready};
use core::pin::Pin;

use g2g_core::runtime::{run_simple_pipeline, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, G2gError, OutputSink, PipelineClock,
    PipelinePacket, PushOutcome, Rate, RawVideoFormat, Reconfigure,
};

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::I420,
        width: Dim::Fixed(16),
        height: Dim::Fixed(16),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Source that pushes `total` frames and counts how many `ForceKeyframe`
/// requests it observed coming back up the reverse channel.
struct CountingSource {
    total: u64,
    force_keyframes_seen: u64,
}

impl SourceLoop for CountingSource {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = Ready<Result<Caps, G2gError>>;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        ready(Ok(caps()))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(PipelinePacket::CapsChanged(caps())).await?;
            let frame_len = 16 * 16 * 3 / 2;
            for seq in 0..self.total {
                let frame = g2g_core::frame::Frame::new(
                    g2g_core::MemoryDomain::System(g2g_core::memory::SystemSlice::from_boxed(
                        alloc_vec(frame_len).into_boxed_slice(),
                    )),
                    g2g_core::FrameTiming {
                        pts_ns: seq * 33_000_000,
                        ..Default::default()
                    },
                    seq,
                );
                if let PushOutcome::Reconfigure(Reconfigure::ForceKeyframe) =
                    out.push(PipelinePacket::DataFrame(frame)).await?
                {
                    self.force_keyframes_seen += 1;
                }
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.total)
        })
    }
}

fn alloc_vec(n: usize) -> std::vec::Vec<u8> {
    std::vec![0u8; n]
}

/// Sink that requests a keyframe on every frame after the first (the way a
/// WebRTC sink would on a remote PLI), via `take_reconfigure`.
struct KeyframeRequestingSink {
    received: u64,
    pending: Option<Reconfigure>,
}

impl AsyncElement for KeyframeRequestingSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(_) = packet {
                self.received += 1;
                if self.received >= 2 {
                    self.pending = Some(Reconfigure::ForceKeyframe);
                }
            }
            Ok(())
        })
    }
    fn take_reconfigure(&mut self) -> Option<Reconfigure> {
        self.pending.take()
    }
}

#[tokio::test]
async fn force_keyframe_reaches_the_source() {
    let mut src = CountingSource {
        total: 6,
        force_keyframes_seen: 0,
    };
    let mut sink = KeyframeRequestingSink {
        received: 0,
        pending: None,
    };

    let stats = run_simple_pipeline(&mut src, &mut sink, &ZeroClock, 1)
        .await
        .expect("pipeline runs");

    assert_eq!(sink.received, 6, "sink consumed every frame");
    assert_eq!(stats.frames_emitted, 6);
    // The sink asked for a keyframe from frame 2 on; the source must have
    // observed at least one ForceKeyframe back up the reverse channel.
    assert!(
        src.force_keyframes_seen > 0,
        "source observed a ForceKeyframe request (seen={})",
        src.force_keyframes_seen
    );
}

#[tokio::test]
async fn no_request_means_no_force_keyframe() {
    // Control: a sink that never requests one leaves the source unbothered.
    struct QuietSink(u64);
    impl AsyncElement for QuietSink {
        type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;
        fn intercept_caps(&self, c: &Caps) -> Result<Caps, G2gError> {
            Ok(c.clone())
        }
        fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
            CapsConstraint::AcceptsAny
        }
        fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
            Ok(ConfigureOutcome::Accepted)
        }
        fn process<'a>(
            &'a mut self,
            packet: PipelinePacket,
            _out: &'a mut dyn OutputSink,
        ) -> Self::ProcessFuture<'a> {
            Box::pin(async move {
                if matches!(packet, PipelinePacket::DataFrame(_)) {
                    self.0 += 1;
                }
                Ok(())
            })
        }
    }

    let mut src = CountingSource {
        total: 5,
        force_keyframes_seen: 0,
    };
    let mut sink = QuietSink(0);
    run_simple_pipeline(&mut src, &mut sink, &ZeroClock, 1)
        .await
        .expect("runs");
    assert_eq!(src.force_keyframes_seen, 0, "no request, no ForceKeyframe");
}
