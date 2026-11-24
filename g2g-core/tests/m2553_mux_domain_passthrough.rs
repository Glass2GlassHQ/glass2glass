// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use std::sync::{Arc, Mutex};

use g2g_core::frame::Frame;
use g2g_core::memory::{DomainSet, MemoryDomainKind, SystemSlice};
use g2g_core::runtime::{block_on, negotiate_graph, run_graph, GraphNode, SourceLoop};
use g2g_core::{
    AllocationParams, AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, FrameTiming,
    G2gError, Graph, MemoryDomain, MultiInputElement, OutputSink, PipelineClock, PipelinePacket,
    Rate, RawVideoFormat,
};

const STARTING_FRAMES_PER_SECOND: u32 = 30;

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

// a hardware decoder that copies to System memory unless a proposal asks for its dma-bufs
struct DecoderStandIn {
    settled: MemoryDomainKind,
}

impl SourceLoop for DecoderStandIn {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(nv12(STARTING_FRAMES_PER_SECOND)))
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::DmaBuf)
    }

    fn output_memory(&self) -> MemoryDomainKind {
        self.settled
    }

    fn configure_allocation(&mut self, params: &AllocationParams) {
        if let Ok(resolved) = params.clone().resolve_for_producer(self.output_domains()) {
            self.settled = resolved.domain;
        }
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(PipelinePacket::Eos).await?;
            Ok(0)
        })
    }
}

struct DecoratingMux {
    domain_follows_input: Option<usize>,
    video_caps: Option<Caps>,
}

impl MultiInputElement for DecoratingMux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        1
    }

    fn output_follows_input(&self) -> Option<usize> {
        Some(0)
    }

    fn output_domain_follows_input(&self) -> Option<usize> {
        self.domain_follows_input
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
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        self.video_caps = Some(absolute_caps.clone());
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        self.video_caps.clone().ok_or(G2gError::NotConfigured)
    }

    fn process<'a>(
        &'a mut self,
        _input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move { out.push(packet).await.map(|_| ()) })
    }
}

// a display sink that imports dma-bufs
struct DmaBufSink;

impl AsyncElement for DmaBufSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::DmaBuf)
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
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

fn domain_into_mux(domain_follows_input: Option<usize>) -> MemoryDomainKind {
    let mut graph: Graph<GraphNode> = Graph::new();
    let src = graph.add_source(GraphNode::source(DecoderStandIn {
        settled: MemoryDomainKind::System,
    }));
    let mux = graph.add_muxer(
        GraphNode::muxer(DecoratingMux {
            domain_follows_input,
            video_caps: None,
        }),
        1,
    );
    let sink = graph.add_sink(GraphNode::element(DmaBufSink));
    graph.link(src, mux.input(0)).unwrap();
    graph.link(mux.output(), sink).unwrap();

    let (_vg, _caps, edge_memory) = block_on(negotiate_graph(graph)).expect("negotiation");
    edge_memory[0]
}

#[test]
fn a_mux_passing_frames_through_carries_the_sinks_domain_to_the_producer() {
    assert_eq!(domain_into_mux(Some(0)), MemoryDomainKind::DmaBuf);
}

#[test]
fn a_mux_declaring_no_pass_through_leaves_the_producer_on_its_default() {
    assert_eq!(domain_into_mux(None), MemoryDomainKind::System);
}

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn frame(sequence: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(
            vec![sequence as u8].into_boxed_slice(),
        )),
        timing: FrameTiming::default(),
        sequence,
        meta: Default::default(),
    })
}

type DomainLog = Arc<Mutex<Vec<MemoryDomainKind>>>;

const MAX_FRAMES: u64 = 256;
const SWITCHED_FRAMES_PER_SECOND: u32 = 60;

struct FramerateSwitchSource {
    decoder_log: DomainLog,
}

impl SourceLoop for FramerateSwitchSource {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(nv12(STARTING_FRAMES_PER_SECOND)))
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(frame(0)).await?;
            out.push(PipelinePacket::CapsChanged(nv12(
                SWITCHED_FRAMES_PER_SECOND,
            )))
            .await?;
            let mut sequence = 1;
            while sequence < MAX_FRAMES && self.decoder_log.lock().unwrap().is_empty() {
                out.push(frame(sequence)).await?;
                sequence += 1;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(sequence)
        })
    }
}

struct LoggingDecoder {
    log: DomainLog,
}

impl AsyncElement for LoggingDecoder {
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
        DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::DmaBuf)
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

struct SwitchingSink;

impl AsyncElement for SwitchingSink {
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

    fn propose_allocation(&self, caps: &Caps) -> Option<AllocationParams> {
        if caps.framerate() != Rate::Fixed(SWITCHED_FRAMES_PER_SECOND << 16) {
            return None;
        }
        Some(AllocationParams {
            domain: MemoryDomainKind::DmaBuf,
            accepts: DomainSet::only(MemoryDomainKind::DmaBuf),
            ..AllocationParams::default()
        })
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
fn a_mid_stream_sink_demand_crosses_a_pass_through_mux_to_the_producer() {
    let decoder_log: DomainLog = Arc::default();
    let mut graph: Graph<GraphNode> = Graph::new();
    let source = graph.add_source(GraphNode::source(FramerateSwitchSource {
        decoder_log: Arc::clone(&decoder_log),
    }));
    let decoder = graph.add_transform(GraphNode::element(LoggingDecoder {
        log: Arc::clone(&decoder_log),
    }));
    let mux = graph.add_muxer(
        GraphNode::muxer(DecoratingMux {
            domain_follows_input: Some(0),
            video_caps: None,
        }),
        1,
    );
    let sink = graph.add_sink(GraphNode::element(SwitchingSink));
    graph.link(source, decoder).unwrap();
    graph.link(decoder, mux.input(0)).unwrap();
    graph.link(mux.output(), sink).unwrap();

    block_on(run_graph(graph, &NullClock, 4)).expect("the graph runs");

    assert_eq!(
        decoder_log.lock().unwrap().as_slice(),
        [MemoryDomainKind::DmaBuf]
    );
}
