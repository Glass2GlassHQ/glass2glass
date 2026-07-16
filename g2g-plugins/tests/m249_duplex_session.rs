//! Terminal duplex-session runner (`run_duplex_session`): drives N send-side
//! sources **and** M recv-side sinks through one terminal `MultiDuplexSession`
//! (the shape a bidirectional sendrecv WebRTC session uses, where the element is
//! at once a sink for its inputs and a source for its outputs). Asserts each
//! send input reaches the session and is routed to the matching recv output, and
//! that the run ends once every send source has reached EOS.

#![cfg(feature = "std")]

use core::future::{ready, Future, Ready};
use core::pin::Pin;

use g2g_core::element::DynAsyncElement;
use g2g_core::runtime::{run_duplex_session, DynSourceLoop, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, DuplexInbound, G2gError,
    MultiDuplexSession, MultiOutputSink, OutputSink, PipelineClock, PipelinePacket, RawVideoFormat,
    Rate,
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

/// Send source pushing `n` frames then EOS (its frames are the "outgoing" tracks).
struct CountedSource {
    n: u64,
}

impl SourceLoop for CountedSource {
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
            out.push(PipelinePacket::CapsChanged(caps())).await?;
            for seq in 0..self.n {
                let frame = g2g_core::frame::Frame::new(
                    g2g_core::MemoryDomain::System(g2g_core::memory::SystemSlice::from_boxed(
                        std::vec![0u8; 4].into_boxed_slice(),
                    )),
                    g2g_core::FrameTiming { pts_ns: seq, ..Default::default() },
                    seq,
                );
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.n)
        })
    }
}

/// Recv sink counting frames + EOS into shared cells.
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

/// Terminal duplex session: echoes each send input `i`'s `DataFrame`s to recv
/// output `i` (so send-side routing AND recv-side routing are both exercised
/// through the one element), and EOSes every output once the inbound channel
/// closes (all send sources ended). Stands in for a sendrecv PeerConnection
/// whose "received" tracks happen to be its own published tracks looped back.
struct EchoDuplex {
    inputs: usize,
    outputs: usize,
}

impl MultiDuplexSession for EchoDuplex {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;

    fn input_count(&self) -> usize {
        self.inputs
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
    fn run<'a>(
        &'a mut self,
        inbound: &'a mut dyn DuplexInbound,
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::RunFuture<'a> {
        let outputs = self.outputs;
        Box::pin(async move {
            let mut received = 0u64;
            // Drain the send side; echo each frame to the matching recv output.
            while let Some((idx, packet)) = inbound.recv().await {
                if let PipelinePacket::DataFrame(f) = packet {
                    out.push_to(idx % outputs, PipelinePacket::DataFrame(f)).await?;
                    received += 1;
                }
            }
            // Send side exhausted: EOS every recv output so no branch is stranded.
            for o in 0..outputs {
                out.push_to(o, PipelinePacket::Eos).await?;
            }
            Ok(received)
        })
    }
}

#[tokio::test]
async fn duplex_session_routes_send_and_recv_both_directions() {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    // Two send sources (5 + 3 frames), two recv sinks.
    let mut send_a = CountedSource { n: 5 };
    let mut send_b = CountedSource { n: 3 };
    let mut session = EchoDuplex { inputs: 2, outputs: 2 };

    let (f0, e0) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
    let (f1, e1) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
    let mut recv0 = CountingSink { frames: f0.clone(), eos: e0.clone() };
    let mut recv1 = CountingSink { frames: f1.clone(), eos: e1.clone() };

    let sources: std::vec::Vec<&mut dyn DynSourceLoop> = std::vec![&mut send_a, &mut send_b];
    let sinks: std::vec::Vec<&mut dyn DynAsyncElement> = std::vec![&mut recv0, &mut recv1];
    let stats = run_duplex_session(sources, &mut session, sinks, &ZeroClock, 4)
        .await
        .expect("duplex session runs to completion");

    // Send input 0 (5 frames) echoed to recv output 0; input 1 (3) to output 1.
    assert_eq!(f0.load(Ordering::SeqCst), 5, "recv output 0 frames");
    assert_eq!(f1.load(Ordering::SeqCst), 3, "recv output 1 frames");
    // Each recv output ends on exactly one EOS.
    assert_eq!(e0.load(Ordering::SeqCst), 1, "recv output 0 EOS once");
    assert_eq!(e1.load(Ordering::SeqCst), 1, "recv output 1 EOS once");
    // Send side counted (5 + 3); recv side counted (8 echoed through the sinks).
    assert_eq!(stats.frames_emitted, 8, "send-side frames summed");
    assert_eq!(stats.frames_consumed, 8, "recv-side frames summed");
}
