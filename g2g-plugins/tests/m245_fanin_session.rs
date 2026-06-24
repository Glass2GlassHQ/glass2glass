//! Terminal fan-in runner (`run_fanin_session`): drives N sources into one
//! terminal `MultiInputElement` with no downstream sink (the shape a multi-track
//! WebRTC session uses). Asserts each input's frames reach the session tagged
//! with the right pad index, and the run ends once every input has sent EOS.

#![cfg(feature = "std")]

use core::future::{ready, Future, Ready};
use core::pin::Pin;

use g2g_core::runtime::{run_fanin_session, SourceLoop};
use g2g_core::{
    Caps, CapsConstraint, ConfigureOutcome, Dim, G2gError, MultiInputElement, OutputSink,
    PipelineClock, PipelinePacket, RawVideoFormat, Rate,
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

/// Source pushing `n` frames then EOS.
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

/// Terminal multi-input element: records how many frames + EOS it saw per pad.
struct RecordingSession {
    inputs: usize,
    frames: std::vec::Vec<u64>,
    eos: std::vec::Vec<u64>,
}

impl RecordingSession {
    fn new(inputs: usize) -> Self {
        Self { inputs, frames: std::vec![0; inputs], eos: std::vec![0; inputs] }
    }
}

impl MultiInputElement for RecordingSession {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn input_count(&self) -> usize {
        self.inputs
    }
    fn intercept_caps(&self, _i: usize, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn configure_pipeline(&mut self, _i: usize, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(caps())
    }
    fn caps_constraint_as_input(&self, _i: usize) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(_) => self.frames[input] += 1,
                PipelinePacket::Eos => self.eos[input] += 1,
                _ => {}
            }
            Ok(())
        })
    }
}

#[tokio::test]
async fn fanin_session_routes_each_input_and_ends_on_all_eos() {
    let mut video = CountedSource { n: 5 };
    let mut audio = CountedSource { n: 3 };
    let mut session = RecordingSession::new(2);

    let sources: std::vec::Vec<&mut dyn g2g_core::runtime::DynSourceLoop> =
        std::vec![&mut video, &mut audio];
    let stats = run_fanin_session(sources, &mut session, &ZeroClock, 4)
        .await
        .expect("session runs to completion");

    // Each pad's frames landed on the right input index.
    assert_eq!(session.frames, std::vec![5, 3], "per-pad frame routing");
    // Every input delivered exactly one EOS to the session.
    assert_eq!(session.eos, std::vec![1, 1], "per-input EOS delivered");
    // The runner counted the union of frames consumed.
    assert_eq!(stats.frames_consumed, 8);
    assert_eq!(stats.frames_emitted, 8, "both sources' frames summed");
}
