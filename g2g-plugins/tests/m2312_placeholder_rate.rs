// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::sync::{Arc, Mutex};

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_graph, GraphNode, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, G2gError, Graph, MemoryDomain,
    OutputSink, PipelineClock, PipelinePacket, Rate, VideoCodec,
};
use g2g_plugins::h264parse::H264Parse;
use g2g_plugins::typefind::elementary_video_caps;

const H264: &[u8] = include_bytes!("fixtures/h264_640x480.h264");

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

struct FixtureSrc {
    announces_placeholder: bool,
}

impl SourceLoop for FixtureSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(elementary_video_caps(VideoCodec::H264)))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if self.announces_placeholder {
                out.push(PipelinePacket::CapsChanged(elementary_video_caps(
                    VideoCodec::H264,
                )))
                .await?;
            }
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(H264.into())),
                FrameTiming {
                    keyframe: true,
                    ..Default::default()
                },
                0,
            );
            out.push(PipelinePacket::DataFrame(frame)).await?;
            out.push(PipelinePacket::Eos).await?;
            Ok(1)
        })
    }
}

struct CapsRecorder {
    caps: Arc<Mutex<Vec<Caps>>>,
}

impl AsyncElement for CapsRecorder {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;
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
            if let PipelinePacket::CapsChanged(c) = packet {
                self.caps.lock().unwrap().push(c);
            }
            Ok(())
        })
    }
}

async fn refined_framerate(announces_placeholder: bool) -> Rate {
    let caps = Arc::new(Mutex::new(Vec::new()));
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(FixtureSrc {
        announces_placeholder,
    }));
    let parse = g.add_transform(GraphNode::element(H264Parse::new()));
    let sink = g.add_sink(GraphNode::element(CapsRecorder { caps: caps.clone() }));
    g.link(src, parse).unwrap();
    g.link(parse, sink).unwrap();
    run_graph(g, &NullClock, 4).await.expect("graph runs");
    let last = caps.lock().unwrap().last().cloned();
    last.expect("the parser refines caps from the SPS")
        .framerate()
}

#[tokio::test]
async fn a_placeholder_announcement_leaves_the_sps_rate_in_place() {
    let from_sps = refined_framerate(false).await;
    assert!(from_sps.is_fixed(), "the fixture's SPS carries a rate");
    assert_eq!(refined_framerate(true).await, from_sps);
}
