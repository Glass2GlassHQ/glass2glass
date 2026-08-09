//! M985: a hosted element declares the memory domain frames actually leave in.
//!
//! `PyTransform` forwards the frame untouched, so both pads carry one domain:
//! System, or CUDA under `cuda-frames`. These are declarations plus the splice
//! decisions they drive, so no GPU and no interpreter are involved and the file
//! compiles without the `python` feature.

use core::future::Future;
use core::pin::Pin;
use std::sync::{Arc, Mutex};

use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::runtime::{auto_plug_domain_converters, GraphNode, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, Graph,
    OutputSink, PipelinePacket, PropValue, Rate, RawVideoFormat,
};
use g2g_python::PyTransform;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

fn nv12() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Fixed(30),
        interlace: g2g_core::Interlace::Any,
    }
}

fn nv12_any() -> CapsSet {
    CapsSet::one(Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        interlace: g2g_core::Interlace::Any,
    })
}

fn nv12_element() -> PyTransform {
    PyTransform::new("module", "Class").with_accept(nv12())
}

/// A source that declares which domain it emits, so the auto-plug has a producer
/// domain to reconcile against. It never runs.
struct DomainSource(MemoryDomainKind);

impl SourceLoop for DomainSource {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(nv12()))
    }
    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn output_memory(&self) -> MemoryDomainKind {
        self.0
    }
    fn run<'a>(&'a mut self, _out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async { Ok(0) })
    }
}

/// A sink that only accepts one domain. It never runs.
struct DomainSink(MemoryDomainKind);

impl AsyncElement for DomainSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(nv12_any())
    }
    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn input_domains(&self) -> DomainSet {
        DomainSet::only(self.0)
    }
    fn process<'a>(
        &'a mut self,
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

/// Every conversion the auto-plug asked for, as `(from, to)` pairs. The factory
/// returns `None`, so the graph is left alone and the request itself is the
/// observation.
fn requested_conversions(
    source_domain: MemoryDomainKind,
    hosted: PyTransform,
    sink_domain: MemoryDomainKind,
) -> Vec<(MemoryDomainKind, MemoryDomainKind)> {
    let asked = Arc::new(Mutex::new(Vec::new()));
    let mut graph: Graph<GraphNode> = Graph::new();
    let source = graph.add_source(GraphNode::source(DomainSource(source_domain)));
    let element = graph.add_transform(GraphNode::element(hosted));
    let sink = graph.add_sink(GraphNode::element(DomainSink(sink_domain)));
    graph.link(source, element).unwrap();
    graph.link(element, sink).unwrap();

    let recorder = Arc::clone(&asked);
    let factory = move |from, to| {
        recorder.lock().unwrap().push((from, to));
        None
    };
    let graph = auto_plug_domain_converters(graph, &factory);
    assert_eq!(graph.node_count(), 3, "the factory splices nothing");
    let recorded = asked.lock().unwrap().clone();
    recorded
}

#[test]
fn a_cpu_hosted_element_declares_system_on_both_pads() {
    let element = nv12_element();
    assert_eq!(
        element.input_domains(),
        DomainSet::only(MemoryDomainKind::System)
    );
    assert_eq!(element.output_memory(), MemoryDomainKind::System);
    assert_eq!(
        element.output_domains(),
        DomainSet::only(MemoryDomainKind::System)
    );
}

#[test]
fn a_cuda_hosted_element_declares_cuda_on_both_pads() {
    let element = nv12_element().with_cuda_frames(true);
    assert_eq!(
        element.input_domains(),
        DomainSet::only(MemoryDomainKind::Cuda),
        "the hosted code reads device memory, so System input needs converting"
    );
    assert_eq!(element.output_memory(), MemoryDomainKind::Cuda);
    assert_eq!(
        element.output_domains(),
        DomainSet::only(MemoryDomainKind::Cuda),
        "the frame leaves in the domain it arrived in"
    );
}

#[test]
fn a_gpu_frame_passes_through_with_no_converter_spliced() {
    let asked = requested_conversions(
        MemoryDomainKind::Cuda,
        nv12_element().with_cuda_frames(true),
        MemoryDomainKind::Cuda,
    );
    assert!(
        asked.is_empty(),
        "a GPU frame through a GPU element to a GPU sink needs no conversion, got {asked:?}"
    );
}

#[test]
fn a_cpu_element_downloads_ahead_of_itself_not_after() {
    let asked = requested_conversions(
        MemoryDomainKind::Cuda,
        nv12_element(),
        MemoryDomainKind::System,
    );
    assert_eq!(
        asked,
        vec![(MemoryDomainKind::Cuda, MemoryDomainKind::System)],
        "one download, on the edge into the element that cannot read device memory"
    );
}

#[test]
fn a_gpu_element_uploads_ahead_of_itself() {
    let asked = requested_conversions(
        MemoryDomainKind::System,
        nv12_element().with_cuda_frames(true),
        MemoryDomainKind::Cuda,
    );
    assert_eq!(
        asked,
        vec![(MemoryDomainKind::System, MemoryDomainKind::Cuda)],
        "one upload, ahead of the element, and nothing after it"
    );
}

#[test]
fn the_upstream_allocation_proposal_names_the_domain_the_hosted_code_reads() {
    let nv12_bytes = (WIDTH * HEIGHT * 3 / 2) as usize;

    let cpu = nv12_element().propose_allocation(&nv12()).unwrap();
    assert_eq!(cpu.domain, MemoryDomainKind::System);
    assert_eq!(cpu.size_bytes, nv12_bytes);
    assert_eq!((cpu.min_buffers, cpu.align), (1, 1), "no pool constraint");

    let gpu = nv12_element()
        .with_cuda_frames(true)
        .propose_allocation(&nv12())
        .unwrap();
    assert_eq!(
        gpu.domain,
        MemoryDomainKind::Cuda,
        "a decoder that can do either is asked to keep the frame on the device"
    );
    assert_eq!(gpu.size_bytes, nv12_bytes);

    // Unfixed caps carry no frame size, so there is nothing to propose.
    let unfixed = Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Any,
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Any,
        interlace: g2g_core::Interlace::Any,
    };
    assert!(nv12_element().propose_allocation(&unfixed).is_none());
}

#[test]
fn cuda_frames_and_format_are_runtime_properties() {
    let mut element = PyTransform::new("module", "Class");
    assert_eq!(
        element.get_property("cuda-frames"),
        Some(PropValue::Bool(false))
    );
    assert_eq!(
        element.get_property("format"),
        Some(PropValue::Str("RGBA".into()))
    );

    element
        .set_property("cuda-frames", PropValue::Bool(true))
        .unwrap();
    element
        .set_property("format", PropValue::Str("NV12".into()))
        .unwrap();
    assert_eq!(
        element.get_property("cuda-frames"),
        Some(PropValue::Bool(true))
    );
    assert_eq!(
        element.get_property("format"),
        Some(PropValue::Str("NV12".into()))
    );
    assert_eq!(element.output_memory(), MemoryDomainKind::Cuda);
    // The format property drives what negotiation accepts, so an NV12 decoder
    // now links to a launch-built `pyelement`.
    assert!(element.intercept_caps(&nv12()).is_ok());

    // A format g2g does not model is refused rather than silently kept.
    assert!(element
        .set_property("format", PropValue::Str("GRAY8".into()))
        .is_err());
}
