//! Terminal fan-out runner (`run_fanout_session`): drives one terminal
//! `MultiOutputSource` (0 inputs -> N outputs, the shape a multi-track WHEP
//! session uses) into N sinks, with no upstream. Asserts each output's frames
//! reach the matching sink and both end on the session's per-output EOS.

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::runtime::run_fanout_session;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, G2gError, MultiOutputSink,
    MultiOutputSource, OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
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

fn frame(seq: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(g2g_core::frame::Frame::new(
        g2g_core::MemoryDomain::System(g2g_core::memory::SystemSlice::from_boxed(
            std::vec![0u8; 4].into_boxed_slice(),
        )),
        g2g_core::FrameTiming {
            pts_ns: seq,
            ..Default::default()
        },
        seq,
    ))
}

/// Source pushing `port0` frames to output 0 and `port1` to output 1, then EOS
/// to both. Interleaves so both ports are exercised concurrently.
struct TwoPortSource {
    port0: u64,
    port1: u64,
}

impl MultiOutputSource for TwoPortSource {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;

    fn output_count(&self) -> usize {
        2
    }
    fn output_caps(&self, _output: usize) -> Result<Caps, G2gError> {
        Ok(caps())
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn MultiOutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let mut pushed = 0u64;
            let max = self.port0.max(self.port1);
            for i in 0..max {
                if i < self.port0 {
                    out.push_to(0, frame(i)).await?;
                    pushed += 1;
                }
                if i < self.port1 {
                    out.push_to(1, frame(i)).await?;
                    pushed += 1;
                }
            }
            out.push_to(0, PipelinePacket::Eos).await?;
            out.push_to(1, PipelinePacket::Eos).await?;
            Ok(pushed)
        })
    }
}

/// Sink counting frames + EOS, recording into a shared cell.
struct CountingSink {
    frames: std::sync::Arc<std::sync::atomic::AtomicU64>,
    eos: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl AsyncElement for CountingSink {
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
            use std::sync::atomic::Ordering;
            match packet {
                PipelinePacket::DataFrame(_) => {
                    self.frames.fetch_add(1, Ordering::SeqCst);
                }
                PipelinePacket::Eos => {
                    self.eos.fetch_add(1, Ordering::SeqCst);
                }
                _ => {}
            }
            Ok(())
        })
    }
}

#[tokio::test]
async fn fanout_session_routes_each_output_and_ends_on_eos() {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    let (vf, ve) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
    let (af, ae) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
    let mut video = CountingSink {
        frames: vf.clone(),
        eos: ve.clone(),
    };
    let mut audio = CountingSink {
        frames: af.clone(),
        eos: ae.clone(),
    };
    let mut session = TwoPortSource { port0: 5, port1: 3 };

    let sinks: std::vec::Vec<&mut dyn g2g_core::element::DynAsyncElement> =
        std::vec![&mut video, &mut audio];
    let stats = run_fanout_session(&mut session, sinks, &ZeroClock, 4)
        .await
        .expect("session runs to completion");

    assert_eq!(vf.load(Ordering::SeqCst), 5, "video output frames");
    assert_eq!(af.load(Ordering::SeqCst), 3, "audio output frames");
    assert_eq!(ve.load(Ordering::SeqCst), 1, "video EOS once");
    assert_eq!(ae.load(Ordering::SeqCst), 1, "audio EOS once");
    assert_eq!(stats.frames_consumed, 8, "both sinks' frames summed");
}
