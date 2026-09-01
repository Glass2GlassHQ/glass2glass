//! M955: the fan-in and duplex runners negotiate a source branch's whole
//! produce set, like the DAG runner does (M954). A branch source that offers
//! several formats is selected by what the pad it feeds accepts: the merged
//! sink for `run_fanin_sink`, the per-input pad for a fan-in / duplex session.
#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::{ready, Future, Ready};
use core::pin::Pin;
use std::sync::{Arc, Mutex};

use g2g_core::fanout::Merger;
use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{
    block_on, run_aggregator_dynamic, run_duplex_session, run_fanin_session, run_fanin_sink,
    DynSourceLoop, SourceLoop,
};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, DuplexInbound, G2gError,
    MemoryDomain, MultiDuplexSession, MultiInputElement, MultiOutputSink, OutputSink,
    PipelineClock, PipelinePacket, Rate, RawVideoFormat, VideoCodec,
};

const FRAMES: u64 = 3;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn raw_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Yuyv,
        width: Dim::Fixed(2),
        height: Dim::Fixed(2),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn compressed_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::Mjpeg,
        width: Dim::Fixed(2),
        height: Dim::Fixed(2),
        framerate: Rate::Fixed(30 << 16),
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

/// Where a source records the caps it was configured with, so a test can read
/// back which alternative negotiation picked.
type Chosen = Arc<Mutex<Option<Caps>>>;

/// Offers raw first, then compressed, the shape of a capture source advertising
/// its device's pixel formats.
struct TwoFormatSource {
    chosen: Chosen,
}

impl TwoFormatSource {
    fn new() -> (Self, Chosen) {
        let chosen: Chosen = Arc::new(Mutex::new(None));
        (
            Self {
                chosen: chosen.clone(),
            },
            chosen,
        )
    }
}

impl SourceLoop for TwoFormatSource {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = Ready<Result<Caps, G2gError>>;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        ready(Ok(raw_caps()))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        ready(Ok(CapsConstraint::Produces(CapsSet::from_alternatives(
            std::vec![raw_caps(), compressed_caps()],
        ))))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        *self.chosen.lock().unwrap() = Some(absolute_caps.clone());
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for seq in 0..FRAMES {
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(std::vec![0u8; 4].into())),
                    FrameTiming::default(),
                    seq,
                );
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(FRAMES)
        })
    }
}

/// Terminal sink accepting exactly one caps (`None` accepts anything), the
/// merged peer every `run_fanin_sink` branch is narrowed against.
struct PinnedSink {
    accepts: Option<Caps>,
}

impl AsyncElement for PinnedSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        match &self.accepts {
            Some(pin) => pin.intersect(upstream),
            None => Ok(upstream.clone()),
        }
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        match &self.accepts {
            Some(pin) => CapsConstraint::Accepts(CapsSet::one(pin.clone())),
            None => CapsConstraint::AcceptsAny,
        }
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

/// Terminal fan-in session whose input pads each accept one caps, so the two
/// branches of one run can be pinned differently.
struct PerPadSession {
    accepts: Vec<Caps>,
    frames: Vec<u64>,
}

impl MultiInputElement for PerPadSession {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn input_count(&self) -> usize {
        self.accepts.len()
    }

    fn intercept_caps(&self, input: usize, caps: &Caps) -> Result<Caps, G2gError> {
        self.accepts[input].intersect(caps)
    }

    fn caps_constraint_as_input(&self, input: usize) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(self.accepts[input].clone()))
    }

    fn configure_pipeline(
        &mut self,
        _input: usize,
        _caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(raw_caps())
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if matches!(packet, PipelinePacket::DataFrame(_)) {
                self.frames[input] += 1;
            }
            Ok(())
        })
    }
}

/// Duplex session with the same per-pad accept sets on its send inputs; its recv
/// outputs carry raw caps and are drained by `PinnedSink`s.
struct PerPadDuplex {
    accepts: Vec<Caps>,
    outputs: usize,
}

impl MultiDuplexSession for PerPadDuplex {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;

    fn input_count(&self) -> usize {
        self.accepts.len()
    }

    fn output_count(&self) -> usize {
        self.outputs
    }

    fn intercept_caps(&self, input: usize, caps: &Caps) -> Result<Caps, G2gError> {
        self.accepts[input].intersect(caps)
    }

    fn caps_constraint_as_input(&self, input: usize) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(self.accepts[input].clone()))
    }

    fn configure_input(
        &mut self,
        _input: usize,
        _caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self, _output: usize) -> Result<Caps, G2gError> {
        Ok(raw_caps())
    }

    fn run<'a>(
        &'a mut self,
        inbound: &'a mut dyn DuplexInbound,
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::RunFuture<'a> {
        let outputs = self.outputs;
        Box::pin(async move {
            let mut received = 0u64;
            while let Some((index, packet)) = inbound.recv().await {
                if let PipelinePacket::DataFrame(frame) = packet {
                    out.push_to(index % outputs, PipelinePacket::DataFrame(frame))
                        .await?;
                    received += 1;
                }
            }
            for output in 0..outputs {
                out.push_to(output, PipelinePacket::Eos).await?;
            }
            Ok(received)
        })
    }
}

/// The merged sink pins the branch format for every `run_fanin_sink` input, so
/// two multi-format sources both capture in the alternative it takes.
#[test]
fn fanin_branches_follow_the_merged_sinks_pin() {
    for (accepts, expected) in [
        (None, raw_caps()),
        (Some(compressed_caps()), compressed_caps()),
    ] {
        let (mut a, chosen_a) = TwoFormatSource::new();
        let (mut b, chosen_b) = TwoFormatSource::new();
        let mut merger = Merger::new(2);
        let mut sink = PinnedSink { accepts };
        let sources: Vec<&mut dyn DynSourceLoop> = std::vec![&mut a, &mut b];
        block_on(run_fanin_sink(
            sources,
            &mut merger,
            &mut sink,
            &ZeroClock,
            2,
        ))
        .expect("fan-in runs to completion");
        assert_eq!(chosen_a.lock().unwrap().clone(), Some(expected.clone()));
        assert_eq!(chosen_b.lock().unwrap().clone(), Some(expected));
    }
}

/// Each input pad of a fan-in session narrows its own branch, so one run can
/// carry raw on pad 0 and compressed on pad 1 out of identical sources.
#[test]
fn fanin_session_pads_select_their_branch_format() {
    let (mut a, chosen_a) = TwoFormatSource::new();
    let (mut b, chosen_b) = TwoFormatSource::new();
    let mut session = PerPadSession {
        accepts: std::vec![raw_caps(), compressed_caps()],
        frames: std::vec![0, 0],
    };
    let sources: Vec<&mut dyn DynSourceLoop> = std::vec![&mut a, &mut b];
    block_on(run_fanin_session(sources, &mut session, &ZeroClock, 2))
        .expect("session runs to completion");

    assert_eq!(chosen_a.lock().unwrap().clone(), Some(raw_caps()));
    assert_eq!(
        chosen_b.lock().unwrap().clone(),
        Some(compressed_caps()),
        "pad 1 accepts only the source's second alternative"
    );
    assert_eq!(session.frames, std::vec![FRAMES, FRAMES]);
}

/// An input attached at runtime is narrowed against the pad it lands on, so a
/// late branch selects its format like a static one.
#[test]
fn a_runtime_attached_input_selects_its_pads_format() {
    let mut aggregator = PerPadSession {
        accepts: std::vec![raw_caps(), compressed_caps()],
        frames: std::vec![0, 0],
    };
    let (first, chosen_first) = TwoFormatSource::new();
    let (second, chosen_second) = TwoFormatSource::new();
    let (handle, run) = run_aggregator_dynamic(&mut aggregator, 2);
    handle
        .add_input(Box::new(first) as Box<dyn DynSourceLoop>)
        .expect("attach pad 0");
    handle
        .add_input(Box::new(second) as Box<dyn DynSourceLoop>)
        .expect("attach pad 1");
    // Dropping the handle stops accepting inputs; the queued two still attach.
    drop(handle);
    block_on(run).expect("dynamic fan-in runs to completion");

    assert_eq!(chosen_first.lock().unwrap().clone(), Some(raw_caps()));
    assert_eq!(
        chosen_second.lock().unwrap().clone(),
        Some(compressed_caps()),
        "the late input follows the pad it landed on"
    );
    assert_eq!(aggregator.frames, std::vec![FRAMES, FRAMES]);
}

/// The duplex runner's send side negotiates its branches the same way.
#[test]
fn duplex_send_inputs_select_their_branch_format() {
    let (mut a, chosen_a) = TwoFormatSource::new();
    let (mut b, chosen_b) = TwoFormatSource::new();
    let mut session = PerPadDuplex {
        accepts: std::vec![compressed_caps(), raw_caps()],
        outputs: 1,
    };
    let mut recv = PinnedSink { accepts: None };
    let sources: Vec<&mut dyn DynSourceLoop> = std::vec![&mut a, &mut b];
    let sinks: Vec<&mut dyn g2g_core::element::DynAsyncElement> = std::vec![&mut recv];
    block_on(run_duplex_session(
        sources,
        &mut session,
        sinks,
        &ZeroClock,
        2,
    ))
    .expect("duplex session runs to completion");

    assert_eq!(
        chosen_a.lock().unwrap().clone(),
        Some(compressed_caps()),
        "send pad 0 accepts only the source's second alternative"
    );
    assert_eq!(chosen_b.lock().unwrap().clone(), Some(raw_caps()));
}
