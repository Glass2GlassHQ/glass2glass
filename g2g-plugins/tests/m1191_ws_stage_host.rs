//! M1191 - hosting a subgraph as the stage a remote transform offloads.
//!
//! `RemoteWsTransform` ships each frame to a peer and emits what comes back,
//! but that peer had to be hand-written. `serve_ws_stage` is the other half: it
//! accepts one client and runs a whole chain of transforms (a `Bin`'s interior,
//! flattened into its stages) against the arriving frames, replying one frame
//! per frame.
//!
//! The chain here is two stages, so the test says something a single stage
//! could not: both run, in order, and the client sees their combined result
//! with its own timing and sequence intact.
#![cfg(feature = "remote-ws")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::element::DynAsyncElement;
use g2g_core::frame::Frame;
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{run_source_transform_sink, LatencyProfile, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError,
    OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
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
/// What each hosted stage does to every byte, so the reply names both.
const FIRST_STAGE_ADDS: u8 = 3;
const SECOND_STAGE_ADDS: u8 = 10;

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
                        pts_ns: u64::from(i) * 1_000_000,
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

/// One hosted stage: adds `step` to every byte. Two of these in a row are the
/// subgraph the peer runs.
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
            match packet {
                PipelinePacket::DataFrame(mut frame) => {
                    if let MemoryDomain::System(slice) = &mut frame.domain {
                        for byte in slice.as_mut_slice() {
                            *byte = byte.wrapping_add(self.step);
                        }
                    }
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

/// Records what came back per frame.
#[derive(Default)]
struct CollectSink {
    frames: Vec<(u64, u64, u8)>,
}

impl AsyncElement for CollectSink {
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
                self.frames
                    .push((frame.sequence, frame.timing.pts_ns, slice[0]));
            }
        }
        core::future::ready(Ok(()))
    }
}

/// A listener the host adopts, plus the URL the client dials. Bound up front:
/// the transform's connect does not retry, so the port has to be listening
/// before the client's first frame.
fn bound_listener() -> (std::net::TcpListener, String) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    (listener, format!("ws://127.0.0.1:{port}"))
}

#[tokio::test]
async fn a_hosted_chain_processes_every_frame_the_client_offloads() {
    let (listener, url) = bound_listener();

    // The peer: two stages run as one offloaded subgraph. Driven on this task
    // (the runner's futures are not Send, so join! rather than spawn).
    let mut first = AddStage {
        step: FIRST_STAGE_ADDS,
    };
    let mut second = AddStage {
        step: SECOND_STAGE_ADDS,
    };
    let host = async {
        let stages: Vec<&mut dyn DynAsyncElement> = vec![&mut first, &mut second];
        serve_ws_stage_on(listener, stages, &ZeroClock, 4, false).await
    };

    // The client graph: offload the middle stage to that peer.
    let mut src = CountSrc;
    let mut xform = RemoteWsTransform::new(url.clone());
    let mut sink = CollectSink::default();
    let clock = ZeroClock;
    let run = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        run_source_transform_sink(
            &mut src,
            &mut xform,
            &mut sink,
            &clock,
            LatencyProfile::Live.link_capacity(),
        ),
    );

    let (run_res, host_res) = tokio::join!(run, host);
    run_res
        .expect("finishes within 10s")
        .expect("client pipeline ok");
    let hosted = host_res.expect("host pipeline ok");
    assert_eq!(
        hosted.frames_consumed, FRAMES as u64,
        "the host ran every offloaded frame"
    );

    assert_eq!(sink.frames.len(), FRAMES as usize);
    for (i, (sequence, pts_ns, byte)) in sink.frames.iter().enumerate() {
        assert_eq!(*sequence, i as u64, "order preserved");
        assert_eq!(
            *pts_ns,
            i as u64 * 1_000_000,
            "the frame's timing crossed both ways"
        );
        assert_eq!(
            *byte,
            (i as u8)
                .wrapping_add(FIRST_STAGE_ADDS)
                .wrapping_add(SECOND_STAGE_ADDS),
            "both hosted stages ran, in order"
        );
    }
}

/// The host's `meta_only` reply mode, the peer half of M1189: a stage that only
/// attaches metadata returns it with an empty payload, and the client emits the
/// frame it kept.
#[cfg(feature = "metadata")]
#[tokio::test]
async fn a_meta_only_host_returns_the_stage_s_metadata_alone() {
    use g2g_core::meta::{AnalyticsMeta, AnalyticsNode, BBox, ObjectDetection};

    /// The label the hosted stage attaches, so the sink's reading names it.
    const DETECTED_LABEL: u32 = 11;

    /// A hosted stage that attaches one detection and leaves the pixels alone.
    struct DetectStage;

    impl AsyncElement for DetectStage {
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
                match packet {
                    PipelinePacket::DataFrame(mut frame) => {
                        let mut meta = AnalyticsMeta::new();
                        meta.nodes.push(AnalyticsNode::Detection(ObjectDetection {
                            bbox: BBox {
                                x: 0.0,
                                y: 0.0,
                                w: 1.0,
                                h: 1.0,
                            },
                            label: DETECTED_LABEL,
                            confidence: 0.5,
                        }));
                        frame.meta.attach(meta);
                        out.push(PipelinePacket::DataFrame(frame)).await?;
                    }
                    other => {
                        out.push(other).await?;
                    }
                }
                Ok(())
            })
        }
    }

    /// Records each frame's first pixel and the label of its lone detection.
    #[derive(Default)]
    struct MetaSink {
        frames: Vec<(u8, Option<u32>)>,
    }

    impl AsyncElement for MetaSink {
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
                let pixel = frame.domain.as_system_slice().map(|s| s[0]).unwrap_or(0);
                let label = frame
                    .meta
                    .get::<AnalyticsMeta>()
                    .and_then(|m| m.detections().next().map(|d| d.label));
                self.frames.push((pixel, label));
            }
            core::future::ready(Ok(()))
        }
    }

    let (listener, url) = bound_listener();
    let mut stage = DetectStage;
    let host = async {
        let stages: Vec<&mut dyn DynAsyncElement> = vec![&mut stage];
        serve_ws_stage_on(listener, stages, &ZeroClock, 4, true).await
    };

    let mut src = CountSrc;
    let mut xform = RemoteWsTransform::new(url).with_meta_only(true);
    let mut sink = MetaSink::default();
    let clock = ZeroClock;
    let run = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        run_source_transform_sink(
            &mut src,
            &mut xform,
            &mut sink,
            &clock,
            LatencyProfile::Live.link_capacity(),
        ),
    );
    let (run_res, host_res) = tokio::join!(run, host);
    run_res
        .expect("finishes within 10s")
        .expect("client pipeline ok");
    host_res.expect("host pipeline ok");

    assert_eq!(sink.frames.len(), FRAMES as usize);
    for (i, (pixel, label)) in sink.frames.iter().enumerate() {
        assert_eq!(
            *pixel, i as u8,
            "the client kept its own frame, pixels untouched"
        );
        assert_eq!(
            *label,
            Some(DETECTED_LABEL),
            "the hosted stage's detection rode back"
        );
    }
}
