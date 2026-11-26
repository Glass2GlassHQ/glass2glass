// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use std::sync::{Arc, Mutex};

use g2g_core::frame::Frame;
use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::runtime::{block_on, run_graph, GraphNode, SourceLoop};
use g2g_core::{
    AllocationParams, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim,
    FrameTiming, G2gError, Graph, MultiInputElement, OutputSink, PipelineClock, PipelinePacket,
    Rate, RawVideoFormat,
};

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn nv12(frames_per_second: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(frames_per_second << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        pixel_aspect: g2g_core::PixelAspect::Any,
    }
}

fn frame(sequence: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::from_system_bytes(
        vec![sequence as u8],
        FrameTiming::default(),
        sequence,
    ))
}

type DomainLog = Arc<Mutex<Vec<MemoryDomainKind>>>;

struct FramerateSwitchSource {
    decoder_log: DomainLog,
}

const MAX_FRAMES: u64 = 256;

impl SourceLoop for FramerateSwitchSource {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(nv12(30)))
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(frame(0)).await?;
            out.push(PipelinePacket::CapsChanged(nv12(60))).await?;
            let mut sequence = 1;
            while sequence < MAX_FRAMES && self.decoder_log.lock().unwrap().len() < 2 {
                out.push(frame(sequence)).await?;
                sequence += 1;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(sequence)
        })
    }
}

struct DualDomainDecoder {
    log: DomainLog,
}

impl AsyncElement for DualDomainDecoder {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::DmaBuf).with(MemoryDomainKind::System)
    }

    fn configure_allocation(&mut self, params: &AllocationParams) {
        let resolved = params
            .clone()
            .resolve_for_producer(self.output_domains())
            .expect("the decoder shares a domain with every proposal");
        self.log.lock().unwrap().push(resolved.domain);
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move { out.push(packet).await.map(|_| ()) })
    }
}

struct SystemOnlyTransform;

impl AsyncElement for SystemOnlyTransform {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move { out.push(packet).await.map(|_| ()) })
    }
}

struct HoldingMux;

const VIDEO_PAD: usize = 0;
const HELD_NS: u64 = 100_000_000;

impl MultiInputElement for HoldingMux {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn input_count(&self) -> usize {
        2
    }

    fn intercept_caps(&self, _input: usize, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn caps_constraint_for_output(&self) -> Result<CapsConstraint<'_>, G2gError> {
        Ok(CapsConstraint::Produces(CapsSet::one(nv12(30))))
    }

    fn propose_allocation_for_input(&self, input: usize, caps: &Caps) -> Option<AllocationParams> {
        if input != VIDEO_PAD {
            return None;
        }
        let (_, _, Rate::Fixed(framerate_q16)) = caps.dims()? else {
            return None;
        };
        let period_ns = g2g_core::caps::framerate_q16_period_ns(*framerate_q16)?;
        Some(AllocationParams {
            min_buffers: usize::try_from(HELD_NS.div_ceil(period_ns)).ok()?,
            ..AllocationParams::meta_demand(Default::default())
        })
    }

    fn configure_pipeline(
        &mut self,
        _input: usize,
        _absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(nv12(30))
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let (VIDEO_PAD, PipelinePacket::DataFrame(f)) = (input, packet) {
                out.push(PipelinePacket::DataFrame(f)).await?;
            }
            Ok(())
        })
    }
}

struct AnySink;

impl AsyncElement for AnySink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

#[test]
fn a_muxer_pad_demand_keeps_the_other_branch_on_system_memory() {
    let decoder_log: DomainLog = Arc::default();
    let mut graph: Graph<GraphNode> = Graph::new();
    let source = graph.add_source(GraphNode::source(FramerateSwitchSource {
        decoder_log: Arc::clone(&decoder_log),
    }));
    let decoder = graph.add_transform(GraphNode::element(DualDomainDecoder {
        log: Arc::clone(&decoder_log),
    }));
    let tee = graph.add_tee(2);
    let system_only = graph.add_transform(GraphNode::element(SystemOnlyTransform));
    let mux = graph.add_muxer(GraphNode::muxer(HoldingMux), 2);
    let sink = graph.add_sink(GraphNode::element(AnySink));
    graph.link(source, decoder).unwrap();
    graph.link(decoder, tee.input()).unwrap();
    graph.link(tee.out(0), mux.input(0)).unwrap();
    graph.link(tee.out(1), system_only).unwrap();
    graph.link(system_only, mux.input(1)).unwrap();
    graph.link(mux.output(), sink).unwrap();

    block_on(run_graph(graph, &NullClock, 4)).expect("the graph runs");

    let settled = decoder_log.lock().unwrap().clone();
    assert!(
        settled.len() >= 2,
        "the framerate change re-cascades into the decoder: {settled:?}"
    );
    assert!(
        settled.iter().all(|&d| d == MemoryDomainKind::System),
        "both branches read the decoder's frames, so it stays on system memory: {settled:?}"
    );
}
