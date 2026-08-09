//! M989: `Registry::decodebin` takes its memory-domain preference from the
//! consumer already in the graph. Two H.264 decoders are registered, a CPU one
//! first and a `Cuda`-producing one second, and only the sink differs between
//! the cases: a sink that accepts `Cuda` gets the GPU decoder spliced in, one
//! that accepts `System` (or declares nothing) keeps the CPU decoder, and an
//! explicit `decodebin_preferring` overrides whatever the sink says.
#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;

use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::runtime::{
    is_raw_video, AutoplugParams, ElementFactory, GraphNode, GraphNodeRef, Registry, SourceLoop,
};
use g2g_core::{
    AsyncElement, Caps, CapsSet, ConfigureOutcome, Dim, G2gError, Graph, NodeId, OutputSink,
    PadTemplate, PipelinePacket, Rate, RawVideoFormat, VideoCodec,
};

fn h264() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

fn nv12() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        interlace: g2g_core::Interlace::Any,
    }
}

/// Depth bound: one decoder hop is all these registries need.
const MAX_DEPTH: usize = 4;

struct H264Source;

impl SourceLoop for H264Source {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(h264()))
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(PipelinePacket::Eos).await?;
            Ok(0)
        })
    }
}

/// H.264 -> NV12 decoder stand-in whose only distinguishing trait is the memory
/// domain it emits, so the spliced node reports which factory the search chose.
struct StubDecoder {
    output_memory: MemoryDomainKind,
}

impl AsyncElement for StubDecoder {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, _upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(nv12())
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_memory(&self) -> MemoryDomainKind {
        self.output_memory
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move { out.push(packet).await.map(|_| ()) })
    }
}

/// Raw-video sink that accepts exactly `accepted` memory.
struct DomainSink {
    accepted: DomainSet,
}

impl AsyncElement for DomainSink {
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

    fn input_domains(&self) -> DomainSet {
        self.accepted
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move { out.push(packet).await.map(|_| ()) })
    }
}

fn decoder_templates() -> Vec<PadTemplate> {
    Vec::from([
        PadTemplate::sink(CapsSet::one(h264())),
        PadTemplate::source(CapsSet::one(nv12())),
    ])
}

/// CPU decoder registered first, `Cuda` decoder second: registration order
/// alone picks the CPU one, so any GPU pick came from the preference.
fn two_decoders() -> Registry {
    let mut registry = Registry::new();
    registry
        .register(ElementFactory::new(
            "h264dec",
            decoder_templates(),
            |_out| {
                Box::new(StubDecoder {
                    output_memory: MemoryDomainKind::System,
                })
            },
        ))
        .register(
            ElementFactory::new("nvdec", decoder_templates(), |_out| {
                Box::new(StubDecoder {
                    output_memory: MemoryDomainKind::Cuda,
                })
            })
            .produces(MemoryDomainKind::Cuda),
        );
    registry
}

/// `source -> (gap) -> sink`, the sink accepting `accepted` memory.
fn graph_with_sink(accepted: DomainSet) -> (Graph<GraphNode>, NodeId, NodeId) {
    let mut graph: Graph<GraphNode> = Graph::new();
    let source = graph.add_source(GraphNodeRef::source(H264Source));
    let sink = graph.add_sink(GraphNodeRef::element(DomainSink { accepted }));
    (graph, source, sink)
}

/// The memory domain of the single spliced decoder.
fn spliced_domain(graph: &Graph<GraphNode>, inserted: &[NodeId]) -> MemoryDomainKind {
    assert_eq!(inserted.len(), 1, "one decoder hop");
    graph
        .element(inserted[0])
        .expect("the spliced node carries its element")
        .output_memory()
}

#[test]
fn a_cuda_consumer_derives_the_gpu_decoder() {
    let registry = two_decoders();
    let (mut graph, source, sink) = graph_with_sink(DomainSet::only(MemoryDomainKind::Cuda));
    let inserted = registry
        .decodebin(&mut graph, source, sink, &h264(), &is_raw_video, MAX_DEPTH)
        .expect("a decoder bridges H.264 to raw");
    assert_eq!(
        spliced_domain(&graph, &inserted),
        MemoryDomainKind::Cuda,
        "a Cuda-only sink derives the Cuda preference with no explicit domain"
    );
}

#[test]
fn a_system_consumer_derives_no_gpu_preference() {
    let registry = two_decoders();
    let (mut graph, source, sink) = graph_with_sink(DomainSet::only(MemoryDomainKind::System));
    let inserted = registry
        .decodebin(&mut graph, source, sink, &h264(), &is_raw_video, MAX_DEPTH)
        .expect("a decoder bridges H.264 to raw");
    assert_eq!(
        spliced_domain(&graph, &inserted),
        MemoryDomainKind::System,
        "a System sink keeps the CPU decoder"
    );
}

#[test]
fn a_consumer_declaring_nothing_derives_no_gpu_preference() {
    // The default `input_domains` (ALL) means "imposes no requirement", not
    // "wants the GPU": a plain graph's selection is unchanged.
    let registry = two_decoders();
    let (mut graph, source, sink) = graph_with_sink(DomainSet::ALL);
    let inserted = registry
        .decodebin(&mut graph, source, sink, &h264(), &is_raw_video, MAX_DEPTH)
        .expect("a decoder bridges H.264 to raw");
    assert_eq!(
        spliced_domain(&graph, &inserted),
        MemoryDomainKind::System,
        "an undeclared sink keeps the CPU decoder"
    );
}

#[test]
fn an_explicit_preference_overrides_the_derivation() {
    // The sink accepts only Cuda, yet the caller asks for System: the explicit
    // domain wins (the converter auto-plug uploads on the edge afterwards).
    let registry = two_decoders();
    let (mut graph, source, sink) = graph_with_sink(DomainSet::only(MemoryDomainKind::Cuda));
    let inserted = registry
        .decodebin_preferring(
            &mut graph,
            source,
            sink,
            &h264(),
            &is_raw_video,
            MAX_DEPTH,
            MemoryDomainKind::System,
        )
        .expect("a decoder bridges H.264 to raw");
    assert_eq!(
        spliced_domain(&graph, &inserted),
        MemoryDomainKind::System,
        "an explicit System preference beats the sink's Cuda"
    );
}

#[test]
fn the_params_path_derives_the_same_preference() {
    let registry = two_decoders();
    let (mut graph, source, sink) = graph_with_sink(DomainSet::only(MemoryDomainKind::Cuda));
    let inserted = registry
        .decodebin_with_params(
            &mut graph,
            source,
            sink,
            &h264(),
            &is_raw_video,
            MAX_DEPTH,
            &AutoplugParams::new(),
        )
        .expect("a decoder bridges H.264 to raw");
    assert_eq!(
        spliced_domain(&graph, &inserted),
        MemoryDomainKind::Cuda,
        "the property-applying decodebin derives the same domain"
    );
}

#[test]
fn the_derivation_reads_the_immediate_consumer() {
    // The helper itself, on a pad with no element behind it (a tee): no
    // requirement to read, so the plain System selection.
    let mut graph: Graph<GraphNode> = Graph::new();
    let tee = graph.add_tee(2);
    assert_eq!(
        Registry::derived_memory_preference(&graph, tee.input()),
        MemoryDomainKind::System,
        "a tee declares no domain requirement"
    );
}
