// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use std::time::Duration;

use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::metrics::monotonic_ns;
use g2g_core::runtime::{block_on, GraphNode, RunBuilder, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, Frame, FrameTiming, G2gError, Graph, OutputSink,
    PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

const LINK_CAPACITY: usize = 2;
const FRAME_COUNT: u64 = 8;
const HOLD: Duration = Duration::from_millis(20);

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(16),
        height: Dim::Fixed(16),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        pixel_aspect: g2g_core::PixelAspect::Any,
    }
}

struct StampedSource;

impl SourceLoop for StampedSource {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(caps()))
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for sequence in 0..FRAME_COUNT {
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
                    FrameTiming {
                        arrival_ns: monotonic_ns(),
                        ..FrameTiming::default()
                    },
                    sequence,
                );
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(FRAME_COUNT)
        })
    }
}

struct HoldingTransform;

impl AsyncElement for HoldingTransform {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if matches!(packet, PipelinePacket::DataFrame(_)) {
                std::thread::sleep(HOLD);
            }
            out.push(packet).await?;
            Ok(())
        })
    }
}

struct DiscardSink;

impl AsyncElement for DiscardSink {
    type ProcessFuture<'a>
        = core::future::Ready<Result<(), G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        core::future::ready(Ok(()))
    }
}

#[test]
fn a_sink_reports_the_age_its_frames_gained_upstream() {
    let mut graph: Graph<GraphNode> = Graph::new();
    let source = graph.add_source(GraphNode::source(StampedSource));
    let hold = graph.add_transform(GraphNode::element(HoldingTransform));
    let sink = graph.add_sink(GraphNode::element(DiscardSink));
    graph.link(source, hold).unwrap();
    graph.link(hold, sink).unwrap();

    let stats = block_on(RunBuilder::new(graph, &ZeroClock, LINK_CAPACITY).run())
        .expect("the run reaches end of stream");
    let row = |name: &str| {
        stats
            .per_element
            .iter()
            .find(|row| row.name == name)
            .unwrap_or_else(|| panic!("{name} has a telemetry row"))
    };

    let sink_age = row("DiscardSink0").age_at_process;
    assert_eq!(
        sink_age.count, FRAME_COUNT,
        "every frame reached the sink with an age"
    );
    assert!(
        sink_age.p50_ns >= HOLD.as_nanos() as u64,
        "the sink sees the transform's hold in each frame's age, p50 {} ns",
        sink_age.p50_ns
    );
    assert!(
        row("DiscardSink0").report_line().contains("age-in"),
        "the run report prints the sink's age"
    );
}
