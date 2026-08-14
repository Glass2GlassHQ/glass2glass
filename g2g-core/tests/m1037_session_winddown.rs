//! M1037: a session that finishes while one of its send sources is still
//! streaming ends the run cleanly. The session arm owns the only inbound
//! receiver, so the source's next push finds a closed channel and stops with
//! `Shutdown`; that wind-down is what a finished session looks like from
//! upstream, not a failed run.
#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::{ready, Future, Ready};
use core::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use g2g_core::element::DynAsyncElement;
use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{
    block_on, run_duplex_session, run_fanin_session, DynSourceLoop, SourceLoop,
};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, DuplexInbound, G2gError,
    MemoryDomain, MultiDuplexSession, MultiInputElement, MultiOutputSink, OutputSink,
    PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

/// Frames the duplex session takes before it returns, with its source still
/// producing.
const SESSION_TAKES: u64 = 3;
/// Frames the fan-in source sends before the `Eos` that ends the session's
/// input accounting; it keeps producing past it, as a live source starting a
/// new segment does.
const FRAMES_BEFORE_EOS: u64 = 2;
/// Small enough that the source blocks on the channel rather than racing far
/// ahead of the session.
const LINK_CAPACITY: usize = 2;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Yuyv,
        width: Dim::Fixed(2),
        height: Dim::Fixed(2),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
    }
}

fn frame(seq: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(std::vec![0u8; 4].into())),
        FrameTiming::default(),
        seq,
    ))
}

/// Source that never runs out: it pushes frames until downstream stops taking
/// them. `eos_at` optionally interleaves one `Eos` after that many frames and
/// keeps going, which is what ends a fan-in session's input accounting under a
/// still-live source.
struct EndlessSource {
    eos_at: Option<u64>,
    wound_down: Arc<AtomicBool>,
    pushed: Arc<AtomicU64>,
}

impl SourceLoop for EndlessSource {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = Ready<Result<Caps, G2gError>>;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        ready(Ok(caps()))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let mut sent = 0u64;
            let stopped: Result<(), G2gError> = loop {
                if self.eos_at == Some(sent) {
                    if let Err(e) = out.push(PipelinePacket::Eos).await {
                        break Err(e);
                    }
                }
                match out.push(frame(sent)).await {
                    Ok(_) => {
                        sent += 1;
                        self.pushed.store(sent, Ordering::SeqCst);
                    }
                    Err(e) => break Err(e),
                }
            };
            if matches!(stopped, Err(G2gError::Shutdown)) {
                self.wound_down.store(true, Ordering::SeqCst);
            }
            stopped.map(|()| sent)
        })
    }
}

/// Recv-side sink of the duplex run: accepts anything and counts nothing the
/// test does not read back through `RunStats`.
struct CountingSink;

impl AsyncElement for CountingSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move { Ok(()) })
    }
}

/// Duplex session that hangs up on its own schedule: it echoes `takes` frames
/// to its recv output, then ends the call while its send source is mid-stream.
struct HangsUpSession {
    takes: u64,
}

impl MultiDuplexSession for HangsUpSession {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;

    fn input_count(&self) -> usize {
        1
    }

    fn output_count(&self) -> usize {
        1
    }

    fn intercept_caps(&self, _input: usize, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_input(
        &mut self,
        _input: usize,
        _absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self, _output: usize) -> Result<Caps, G2gError> {
        Ok(caps())
    }

    fn run<'a>(
        &'a mut self,
        inbound: &'a mut dyn DuplexInbound,
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::RunFuture<'a> {
        let takes = self.takes;
        Box::pin(async move {
            let mut received = 0u64;
            while received < takes {
                let Some((_input, packet)) = inbound.recv().await else {
                    break;
                };
                if let PipelinePacket::DataFrame(f) = packet {
                    out.push_to(0, PipelinePacket::DataFrame(f)).await?;
                    received += 1;
                }
            }
            out.push_to(0, PipelinePacket::Eos).await?;
            Ok(received)
        })
    }
}

/// Terminal fan-in session that counts the frames it is handed.
struct CountingFaninSession {
    frames: u64,
}

impl MultiInputElement for CountingFaninSession {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn input_count(&self) -> usize {
        1
    }

    fn intercept_caps(&self, _input: usize, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(
        &mut self,
        _input: usize,
        _absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(caps())
    }

    fn process<'a>(
        &'a mut self,
        _input: usize,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if matches!(packet, PipelinePacket::DataFrame(_)) {
                self.frames += 1;
            }
            Ok(())
        })
    }
}

#[test]
fn a_duplex_session_that_ends_first_winds_its_send_source_down() {
    let wound_down = Arc::new(AtomicBool::new(false));
    let pushed = Arc::new(AtomicU64::new(0));
    let mut source = EndlessSource {
        eos_at: None,
        wound_down: wound_down.clone(),
        pushed: pushed.clone(),
    };
    let mut session = HangsUpSession {
        takes: SESSION_TAKES,
    };
    let mut sink = CountingSink;

    let sources: Vec<&mut dyn DynSourceLoop> = std::vec![&mut source];
    let sinks: Vec<&mut dyn DynAsyncElement> = std::vec![&mut sink];
    let stats = match block_on(run_duplex_session(
        sources,
        &mut session,
        sinks,
        &ZeroClock,
        LINK_CAPACITY,
    )) {
        Ok(stats) => stats,
        Err(e) => panic!("a completed session must not fail the run: {e:?}"),
    };

    assert!(
        wound_down.load(Ordering::SeqCst),
        "the source was still streaming when the session ended, so it hit the closed channel"
    );
    assert_eq!(
        stats.frames_consumed, SESSION_TAKES,
        "the recv sink took every frame the session echoed"
    );
    assert!(
        stats.frames_emitted >= SESSION_TAKES,
        "the wound-down source still reports what it delivered, got {}",
        stats.frames_emitted
    );
    assert_eq!(
        stats.frames_emitted,
        pushed.load(Ordering::SeqCst),
        "every frame the source got into the channel is counted"
    );
}

#[test]
fn a_fanin_session_that_ends_first_winds_its_source_down() {
    let wound_down = Arc::new(AtomicBool::new(false));
    let pushed = Arc::new(AtomicU64::new(0));
    let mut source = EndlessSource {
        eos_at: Some(FRAMES_BEFORE_EOS),
        wound_down: wound_down.clone(),
        pushed: pushed.clone(),
    };
    let mut session = CountingFaninSession { frames: 0 };

    let sources: Vec<&mut dyn DynSourceLoop> = std::vec![&mut source];
    let stats = match block_on(run_fanin_session(
        sources,
        &mut session,
        &ZeroClock,
        LINK_CAPACITY,
    )) {
        Ok(stats) => stats,
        Err(e) => panic!("a completed session must not fail the run: {e:?}"),
    };

    assert!(
        wound_down.load(Ordering::SeqCst),
        "the source kept producing past its EOS, so it hit the closed channel"
    );
    assert!(
        pushed.load(Ordering::SeqCst) > FRAMES_BEFORE_EOS,
        "the source really had more to send after the EOS that ended the session"
    );
    assert_eq!(
        session.frames, FRAMES_BEFORE_EOS,
        "the session saw the frames that preceded its EOS"
    );
    assert_eq!(
        stats.frames_emitted,
        pushed.load(Ordering::SeqCst),
        "the wound-down source still reports every frame it delivered"
    );
}
