//! Bitrate (congestion-control) reverse channel: a sink that learns a target
//! send bitrate (a WebRTC sink relaying its BWE estimate) returns it from
//! `take_bitrate`; the runner forwards it up the incoming link, and the
//! producing encoder observes it as `PushOutcome::Bitrate`. Mirrors the M243
//! keyframe-request test for the bitrate path.

#![cfg(feature = "std")]

use core::future::{ready, Future, Ready};
use core::pin::Pin;

use g2g_core::runtime::{run_simple_pipeline, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, G2gError, OutputSink, PipelineClock,
    PipelinePacket, PushOutcome, RawVideoFormat, Rate,
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

/// Source that pushes `total` frames and records the bitrate targets it observed
/// coming back up the reverse channel.
struct BitrateObservingSource {
    total: u64,
    targets: std::vec::Vec<u32>,
}

impl SourceLoop for BitrateObservingSource {
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
                        std::vec![0u8; frame_len].into_boxed_slice(),
                    )),
                    g2g_core::FrameTiming { pts_ns: seq * 33_000_000, ..Default::default() },
                    seq,
                );
                if let PushOutcome::Bitrate(bps) =
                    out.push(PipelinePacket::DataFrame(frame)).await?
                {
                    self.targets.push(bps);
                }
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.total)
        })
    }
}

/// Sink that reports a target bitrate once it has received a couple of frames,
/// the way a WebRTC sink relays its first BWE estimate.
struct BweSink {
    received: u64,
    pending: Option<u32>,
}

impl AsyncElement for BweSink {
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
                if self.received == 2 {
                    self.pending = Some(750_000);
                }
            }
            Ok(())
        })
    }
    fn take_bitrate(&mut self) -> Option<u32> {
        self.pending.take()
    }
}

#[tokio::test]
async fn bitrate_target_reaches_the_source() {
    let mut src = BitrateObservingSource { total: 6, targets: std::vec::Vec::new() };
    let mut sink = BweSink { received: 0, pending: None };

    let stats = run_simple_pipeline(&mut src, &mut sink, &ZeroClock, 1)
        .await
        .expect("pipeline runs");

    assert_eq!(stats.frames_emitted, 6);
    assert_eq!(sink.received, 6);
    assert_eq!(src.targets, std::vec![750_000], "the source observed the BWE target once");
}
