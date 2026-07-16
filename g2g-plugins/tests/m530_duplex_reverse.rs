//! T1-B: per-input reverse-signal routing through the terminal duplex runner
//! (`run_duplex_session`). A [`MultiDuplexSession`] exposes a per-send-input
//! [`ReverseChannel`]; the runner clones each before running and polls it after
//! every push from the matching source, surfacing a pending PLI / BWE to that
//! source as its [`PushOutcome`], exactly as the fan-in session runner does
//! (M523). This asserts a signal the session posts on input 0's channel reaches
//! send source 0 and never fires source 1's, the routing that lets a sendrecv
//! WebRTC session forward a remote per-mid keyframe request to the right encoder.

#![cfg(feature = "std")]

use core::future::{ready, Future, Ready};
use core::pin::Pin;

use g2g_core::element::DynAsyncElement;
use g2g_core::runtime::{run_duplex_session, DynSourceLoop, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, DuplexInbound, G2gError,
    MultiDuplexSession, MultiOutputSink, OutputSink, PipelineClock, PipelinePacket, PushOutcome,
    RawVideoFormat, Rate, Reconfigure, ReverseChannel,
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

/// Send source that records whether any of its pushes was answered with a
/// `ForceKeyframe` reconfigure (a reverse PLI routed back to it by the runner).
struct RecordingSource {
    n: u64,
    saw_keyframe: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl SourceLoop for RecordingSource {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = Ready<Result<Caps, G2gError>>;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        ready(Ok(caps()))
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            use std::sync::atomic::Ordering;
            // The first push observes the pre-armed signal (the runner polls the
            // reverse channel after every push, CapsChanged included).
            if let PushOutcome::Reconfigure(Reconfigure::ForceKeyframe) =
                out.push(PipelinePacket::CapsChanged(caps())).await?
            {
                self.saw_keyframe.store(true, Ordering::SeqCst);
            }
            for seq in 0..self.n {
                let frame = g2g_core::frame::Frame::new(
                    g2g_core::MemoryDomain::System(g2g_core::memory::SystemSlice::from_boxed(
                        std::vec![0u8; 4].into_boxed_slice(),
                    )),
                    g2g_core::FrameTiming { pts_ns: seq, ..Default::default() },
                    seq,
                );
                if let PushOutcome::Reconfigure(Reconfigure::ForceKeyframe) =
                    out.push(PipelinePacket::DataFrame(frame)).await?
                {
                    self.saw_keyframe.store(true, Ordering::SeqCst);
                }
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.n)
        })
    }
}

/// Recv sink that just drains (the recv direction is covered by M249).
struct DrainSink;

impl AsyncElement for DrainSink {
    type ProcessFuture<'a> = Ready<Result<(), G2gError>>;

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
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        ready(Ok(()))
    }
}

/// Duplex session that owns one reverse channel per send input and hands them to
/// the runner via `reverse_channel`. The test pre-arms input 0's channel (as a
/// remote PLI naming that track's m-line would), so the routing is deterministic.
struct SignalingDuplex {
    reverse: Vec<ReverseChannel>,
    outputs: usize,
}

impl SignalingDuplex {
    fn new(inputs: usize, outputs: usize) -> Self {
        Self { reverse: (0..inputs).map(|_| ReverseChannel::new()).collect(), outputs }
    }
    /// A clone of send input `i`'s reverse channel, for the test to post on.
    fn channel(&self, i: usize) -> ReverseChannel {
        self.reverse[i].clone()
    }
}

impl MultiDuplexSession for SignalingDuplex {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;

    fn input_count(&self) -> usize {
        self.reverse.len()
    }
    fn output_count(&self) -> usize {
        self.outputs
    }
    fn intercept_caps(&self, _i: usize, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn caps_constraint_as_input(&self, _i: usize) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn configure_input(&mut self, _i: usize, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn output_caps(&self, _o: usize) -> Result<Caps, G2gError> {
        Ok(caps())
    }
    fn reverse_channel(&self, input: usize) -> Option<ReverseChannel> {
        self.reverse.get(input).cloned()
    }
    fn run<'a>(
        &'a mut self,
        inbound: &'a mut dyn DuplexInbound,
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::RunFuture<'a> {
        let outputs = self.outputs;
        Box::pin(async move {
            let mut received = 0u64;
            while let Some((_idx, packet)) = inbound.recv().await {
                if let PipelinePacket::DataFrame(_) = packet {
                    received += 1;
                }
            }
            for o in 0..outputs {
                out.push_to(o, PipelinePacket::Eos).await?;
            }
            Ok(received)
        })
    }
}

#[tokio::test]
async fn duplex_runner_routes_reverse_signal_to_matching_send_source() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let mut session = SignalingDuplex::new(2, 2);
    // Pre-arm input 0's channel (a remote PLI on that track's m-line); input 1's
    // stays quiet, so a leak between channels would show as source 1 seeing it.
    session.channel(0).request_keyframe();

    let saw0 = Arc::new(AtomicBool::new(false));
    let saw1 = Arc::new(AtomicBool::new(false));
    let mut send0 = RecordingSource { n: 3, saw_keyframe: saw0.clone() };
    let mut send1 = RecordingSource { n: 3, saw_keyframe: saw1.clone() };
    let mut recv0 = DrainSink;
    let mut recv1 = DrainSink;

    let sources: std::vec::Vec<&mut dyn DynSourceLoop> = std::vec![&mut send0, &mut send1];
    let sinks: std::vec::Vec<&mut dyn DynAsyncElement> = std::vec![&mut recv0, &mut recv1];
    run_duplex_session(sources, &mut session, sinks, &ZeroClock, 4)
        .await
        .expect("duplex session runs to completion");

    assert!(saw0.load(Ordering::SeqCst), "send source 0 saw the reverse keyframe request");
    assert!(
        !saw1.load(Ordering::SeqCst),
        "send source 1 must NOT see input 0's reverse signal (per-input routing)"
    );
}
