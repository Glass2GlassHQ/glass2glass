#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use std::sync::Mutex;

use g2g_core::memory::DomainSet;
use g2g_core::runtime::{auto_plug_domain_converters, GraphNode, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, G2gError, Graph, MemoryDomainKind,
    NodeId, OutputSink, PipelinePacket, Rate, RawVideoFormat,
};

fn rgba() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(8),
        height: Dim::Fixed(8),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

struct TextureSource;

impl SourceLoop for TextureSource {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(rgba()))
    }
    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn output_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::WgpuTexture)
    }
    fn run<'a>(&'a mut self, _out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async { Ok(0) })
    }
}

struct FakeConverter {
    emits: MemoryDomainKind,
}

impl AsyncElement for FakeConverter {
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
    fn output_memory(&self) -> MemoryDomainKind {
        self.emits
    }
    fn process<'a>(
        &'a mut self,
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

struct DomainSink {
    accepts: DomainSet,
}

impl AsyncElement for DomainSink {
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
    fn input_domains(&self) -> DomainSet {
        self.accepts
    }
    fn process<'a>(
        &'a mut self,
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

fn texture_to_sink(accepts: DomainSet) -> Graph<GraphNode> {
    let mut graph: Graph<GraphNode> = Graph::new();
    let source = graph.add_source(GraphNode::source(TextureSource));
    let sink = graph.add_sink(GraphNode::element(DomainSink { accepts }));
    graph.link(source, sink).unwrap();
    graph
}

fn dmabuf_or_system() -> DomainSet {
    DomainSet::only(MemoryDomainKind::DmaBuf).with(MemoryDomainKind::System)
}

fn emitted_by(graph: &Graph<GraphNode>, id: u32) -> Option<MemoryDomainKind> {
    graph.element(NodeId(id)).map(|node| node.output_memory())
}

#[test]
fn falls_back_to_the_consumers_second_domain() {
    let asked = Mutex::new(Vec::new());
    let factory = |from: MemoryDomainKind, to: MemoryDomainKind| {
        asked.lock().unwrap().push(to);
        match (from, to) {
            (MemoryDomainKind::WgpuTexture, MemoryDomainKind::System) => {
                Some(GraphNode::element(FakeConverter { emits: to }))
            }
            _ => None,
        }
    };

    let graph = auto_plug_domain_converters(texture_to_sink(dmabuf_or_system()), &factory);

    assert_eq!(graph.node_count(), 3, "a converter was spliced");
    assert_eq!(emitted_by(&graph, 2), Some(MemoryDomainKind::System));
    assert_eq!(
        *asked.lock().unwrap(),
        [MemoryDomainKind::DmaBuf, MemoryDomainKind::System],
        "the preferred domain is asked first, then the next one"
    );
}

#[test]
fn preferred_domain_converter_still_wins() {
    let asked = Mutex::new(Vec::new());
    let factory = |_from: MemoryDomainKind, to: MemoryDomainKind| {
        asked.lock().unwrap().push(to);
        Some(GraphNode::element(FakeConverter { emits: to }))
    };

    let graph = auto_plug_domain_converters(texture_to_sink(dmabuf_or_system()), &factory);

    assert_eq!(graph.node_count(), 3);
    assert_eq!(emitted_by(&graph, 2), Some(MemoryDomainKind::DmaBuf));
    assert_eq!(*asked.lock().unwrap(), [MemoryDomainKind::DmaBuf]);
}

#[test]
fn consumer_sharing_the_texture_domain_gets_no_converter() {
    let factory = |_from: MemoryDomainKind, to: MemoryDomainKind| {
        Some(GraphNode::element(FakeConverter { emits: to }))
    };
    let cuda_texture_or_system = DomainSet::only(MemoryDomainKind::Cuda)
        .with(MemoryDomainKind::WgpuTexture)
        .with(MemoryDomainKind::System);

    for accepts in [DomainSet::ALL, cuda_texture_or_system] {
        let graph = auto_plug_domain_converters(texture_to_sink(accepts), &factory);
        assert_eq!(graph.node_count(), 2, "{accepts:?} takes the texture as is");
    }
}
