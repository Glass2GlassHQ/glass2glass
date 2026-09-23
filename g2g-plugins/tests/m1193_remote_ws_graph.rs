//! M1193 - hosting a branching subgraph as the stage a remote transform offloads.
//!
//! `serve_ws_stage` takes a `Bin` with one ghost input and one ghost output and
//! runs it on the DAG runner, so the offloaded stage can fan out and rejoin. The
//! first test hosts a tee whose two processing branches rejoin into the reply
//! while a third branch ends at a local sink. The second hosts a subgraph that
//! answers every frame twice, which the host has to refuse rather than let the
//! client pair a frame with another frame's reply.
#![cfg(feature = "remote-ws")]

use core::future::Future;
use core::pin::Pin;
use std::sync::{Arc, Mutex};

use g2g_core::frame::Frame;
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{
    run_source_transform_sink, GraphNodeRef, LatencyProfile, RunStats, SourceLoop,
};
use g2g_core::{
    AsyncElement, Bin, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError,
    MultiInputElement, OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

use g2g_plugins::remotewsstage::serve_ws_stage_on;
use g2g_plugins::remotewstransform::RemoteWsTransform;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

const FRAMES: u8 = 5;
const FRAME_LEN: usize = 4 * 4 * 4;
const PTS_STEP_NS: u64 = 1_000_000;
const HOST_LINK_CAPACITY: usize = 4;
const RUN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
/// What each rejoined branch adds to every byte, so the reply names both.
const FIRST_BRANCH_ADDS: u8 = 3;
const SECOND_BRANCH_ADDS: u8 = 10;
const FIRST_BRANCH: u8 = 0;
const SECOND_BRANCH: u8 = 1;
const SIDE_BRANCH: u8 = 2;
const REJOINED_BRANCHES: u8 = 2;
/// The byte of the reply each rejoined branch fills.
const FIRST_BRANCH_BYTE: usize = 0;
const SECOND_BRANCH_BYTE: usize = 1;

fn test_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(4),
        height: Dim::Fixed(4),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

/// Emits `FRAMES` frames whose every byte is the frame index, then EOS.
struct CountSrc;

impl SourceLoop for CountSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(test_caps()))
    }
    fn caps_constraint(
        &mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'_>, G2gError>> + '_ {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(test_caps()))))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(PipelinePacket::CapsChanged(test_caps())).await?;
            for i in 0..FRAMES {
                out.push(PipelinePacket::DataFrame(Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(
                        vec![i; FRAME_LEN].into_boxed_slice(),
                    )),
                    FrameTiming {
                        pts_ns: u64::from(i) * PTS_STEP_NS,
                        ..FrameTiming::default()
                    },
                    u64::from(i),
                )))
                .await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(u64::from(FRAMES))
        })
    }
}

/// One hosted branch: adds `step` to every byte.
struct AddStage {
    step: u8,
}

impl AsyncElement for AddStage {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;
    fn intercept_caps(&self, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| CapsSet::one(input.clone())))
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(mut frame) = packet {
                if let MemoryDomain::System(slice) = &mut frame.domain {
                    for byte in slice.as_mut_slice() {
                        *byte = byte.wrapping_add(self.step);
                    }
                }
                out.push(PipelinePacket::DataFrame(frame)).await?;
            } else {
                out.push(packet).await?;
            }
            Ok(())
        })
    }
}

/// The two-pad caps plumbing every test muxer here shares: identity per pad,
/// output following the first pad.
macro_rules! identity_mux_caps {
    () => {
        fn input_count(&self) -> usize {
            usize::from(REJOINED_BRANCHES)
        }
        fn intercept_caps(&self, _input: usize, upstream: &Caps) -> Result<Caps, G2gError> {
            Ok(upstream.clone())
        }
        fn configure_pipeline(
            &mut self,
            _input: usize,
            _caps: &Caps,
        ) -> Result<ConfigureOutcome, G2gError> {
            Ok(ConfigureOutcome::Accepted)
        }
        fn output_caps(&self) -> Result<Caps, G2gError> {
            Ok(test_caps())
        }
        fn output_follows_input(&self) -> Option<usize> {
            Some(usize::from(FIRST_BRANCH))
        }
    };
}

/// Rejoins the two processing branches: once both copies of a frame are in, the
/// reply is the first branch's frame with its second byte taken from the second
/// branch's.
#[derive(Default)]
struct RejoinMux {
    pending: [Option<Frame>; REJOINED_BRANCHES as usize],
}

impl MultiInputElement for RejoinMux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    identity_mux_caps!();

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    self.pending[input] = Some(frame);
                    if self.pending.iter().all(Option::is_some) {
                        let [first, second] =
                            core::mem::take(&mut self.pending).map(|f| f.expect("both in"));
                        out.push(PipelinePacket::DataFrame(rejoin(first, &second)))
                            .await?;
                    }
                }
                PipelinePacket::CapsChanged(caps) if input == usize::from(FIRST_BRANCH) => {
                    out.push(PipelinePacket::CapsChanged(caps)).await?;
                }
                _ => {}
            }
            Ok(())
        })
    }
}

fn rejoin(mut first: Frame, second: &Frame) -> Frame {
    let second_byte = second.domain.as_system_slice().expect("system")[FIRST_BRANCH_BYTE];
    if let MemoryDomain::System(slice) = &mut first.domain {
        slice.as_mut_slice()[SECOND_BRANCH_BYTE] = second_byte;
    }
    first
}

/// Forwards every frame from either pad, so a tee in front of it answers each
/// frame twice.
struct FunnelMux;

impl MultiInputElement for FunnelMux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    identity_mux_caps!();

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(_) => {
                    out.push(packet).await?;
                }
                PipelinePacket::CapsChanged(_) if input == usize::from(FIRST_BRANCH) => {
                    out.push(packet).await?;
                }
                _ => {}
            }
            Ok(())
        })
    }
}

/// What a sink saw of one frame: its sequence, pts and first two bytes.
type Seen = (u64, u64, u8, u8);

/// Records each frame it takes, on the client or on a hosted side branch.
#[derive(Default, Clone)]
struct RecordSink {
    frames: Arc<Mutex<Vec<Seen>>>,
}

impl AsyncElement for RecordSink {
    type ProcessFuture<'a>
        = core::future::Ready<Result<(), G2gError>>
    where
        Self: 'a;
    fn intercept_caps(&self, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        if let PipelinePacket::DataFrame(frame) = packet {
            if let Some(slice) = frame.domain.as_system_slice() {
                self.frames.lock().unwrap().push((
                    frame.sequence,
                    frame.timing.pts_ns,
                    slice[FIRST_BRANCH_BYTE],
                    slice[SECOND_BRANCH_BYTE],
                ));
            }
        }
        core::future::ready(Ok(()))
    }
}

/// Runs `stage` behind a host on loopback against a real `RemoteWsTransform`
/// client, returning the client's result, the host's result and what the
/// client received.
async fn offload_through(
    stage: Bin<GraphNodeRef<'static>>,
) -> (
    Result<RunStats, G2gError>,
    Result<RunStats, G2gError>,
    Vec<Seen>,
) {
    // Bound up front: the transform's connect does not retry.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let host = serve_ws_stage_on(listener, stage, &ZeroClock, HOST_LINK_CAPACITY, false);

    let mut src = CountSrc;
    let mut xform = RemoteWsTransform::new(format!("ws://127.0.0.1:{port}"));
    let mut sink = RecordSink::default();
    let received = sink.frames.clone();
    let clock = ZeroClock;
    let run = tokio::time::timeout(
        RUN_TIMEOUT,
        run_source_transform_sink(
            &mut src,
            &mut xform,
            &mut sink,
            &clock,
            LatencyProfile::Live.link_capacity(),
        ),
    );
    // The runner's futures are not Send, so join! rather than spawn.
    let (run_res, host_res) = tokio::join!(run, host);
    let run_res = run_res.expect("client finishes within the timeout");
    let received = received.lock().unwrap().clone();
    (run_res, host_res, received)
}

#[tokio::test]
async fn a_hosted_branching_graph_replies_once_per_frame() {
    let side = RecordSink::default();
    let side_seen = side.frames.clone();

    let mut stage = Bin::new();
    let tee = stage.add_tee(REJOINED_BRANCHES + 1);
    let first = stage.add_transform(GraphNodeRef::element(AddStage {
        step: FIRST_BRANCH_ADDS,
    }));
    let second = stage.add_transform(GraphNodeRef::element(AddStage {
        step: SECOND_BRANCH_ADDS,
    }));
    let side_sink = stage.add_sink(GraphNodeRef::element(side));
    let mux = stage.add_muxer(GraphNodeRef::muxer(RejoinMux::default()), REJOINED_BRANCHES);
    stage.link(tee.out(FIRST_BRANCH), first).expect("link");
    stage.link(tee.out(SECOND_BRANCH), second).expect("link");
    stage.link(tee.out(SIDE_BRANCH), side_sink).expect("link");
    stage.link(first, mux.input(FIRST_BRANCH)).expect("link");
    stage.link(second, mux.input(SECOND_BRANCH)).expect("link");
    stage.ghost_input(tee.input()).expect("ghost input");
    stage.ghost_output(mux.output()).expect("ghost output");

    let (run_res, host_res, received) = offload_through(stage).await;
    run_res.expect("client pipeline ok");
    host_res.expect("host pipeline ok");

    assert_eq!(received.len(), usize::from(FRAMES), "one reply per frame");
    for (i, (sequence, pts_ns, first_byte, second_byte)) in received.into_iter().enumerate() {
        let original = i as u8;
        assert_eq!(sequence, i as u64, "order preserved");
        assert_eq!(pts_ns, i as u64 * PTS_STEP_NS, "timing crossed both ways");
        assert_eq!(
            first_byte,
            original.wrapping_add(FIRST_BRANCH_ADDS),
            "the first branch ran and fed the reply"
        );
        assert_eq!(
            second_byte,
            original.wrapping_add(SECOND_BRANCH_ADDS),
            "the second branch ran and rejoined the reply"
        );
    }

    let side_seen = side_seen.lock().unwrap();
    assert_eq!(side_seen.len(), usize::from(FRAMES), "the side branch ran");
    for (i, (sequence, _, first_byte, second_byte)) in side_seen.iter().enumerate() {
        assert_eq!(*sequence, i as u64);
        assert_eq!(
            (*first_byte, *second_byte),
            (i as u8, i as u8),
            "the side branch got its own untouched copy"
        );
    }
}

#[tokio::test]
async fn a_hosted_graph_that_answers_twice_fails_both_ends() {
    let mut stage = Bin::new();
    let tee = stage.add_tee(REJOINED_BRANCHES);
    let mux = stage.add_muxer(GraphNodeRef::muxer(FunnelMux), REJOINED_BRANCHES);
    stage
        .link(tee.out(FIRST_BRANCH), mux.input(FIRST_BRANCH))
        .expect("link");
    stage
        .link(tee.out(SECOND_BRANCH), mux.input(SECOND_BRANCH))
        .expect("link");
    stage.ghost_input(tee.input()).expect("ghost input");
    stage.ghost_output(mux.output()).expect("ghost output");

    let (run_res, host_res, received) = offload_through(stage).await;
    assert!(host_res.is_err(), "the host refuses the second reply");
    assert!(run_res.is_err(), "the client fails instead of waiting");
    assert!(received.len() < usize::from(FRAMES));
    for (i, (sequence, ..)) in received.into_iter().enumerate() {
        assert_eq!(sequence, i as u64, "no frame got another frame's reply");
    }
}
