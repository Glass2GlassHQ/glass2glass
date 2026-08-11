#![cfg(all(
    target_os = "linux",
    feature = "wgpu-present",
    feature = "nvdec",
    feature = "cuda-wgpu"
))]
//! M1017: a CUDA-resident decoded frame reaches the wgpu display sink without a
//! PCIe round trip, and a text pipeline gets there without naming the bridge.
//!
//! `nvdec` can keep frames in CUDA device memory or download them; `wgpusink`
//! takes a wgpu texture or system memory. The only domain they share is system
//! memory, so the parse-time domain auto-plug splices the CUDA -> wgpu bridge
//! rather than letting the pair agree on a download.
//!
//! The window itself is not testable here (it needs a compositor session), so the
//! hardware test runs the same graph into an *offscreen* `WgpuSink` on the shared
//! interop device: everything the windowed sink does bar the surface present.
//!
//! Hardware test: needs an NVIDIA GPU with NVCUVID plus a wgpu adapter. Skips
//! gracefully when the decoder cannot initialise.

use core::future::Future;
use core::pin::Pin;
use std::boxed::Box;
use std::sync::{Arc, Mutex};

use g2g_core::graph::NodeId;
use g2g_core::memory::{DomainSet, MemoryDomain, MemoryDomainKind};
use g2g_core::runtime::{block_on, negotiate_graph, parse_launch, run_graph, GraphNode};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, Graph, Interlace,
    OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};
use g2g_plugins::cuda::auto_plug_cuda_converters;
use g2g_plugins::cudawgpu::shared_interop_device;
use g2g_plugins::filesrc::FileSrc;
use g2g_plugins::gpu::GpuContext;
use g2g_plugins::h264parse::H264Parse;
use g2g_plugins::nvdec::NvDec;
use g2g_plugins::registry::default_registry;
use g2g_plugins::wgpusink::WgpuSink;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/h264_640x480.h264"
);
const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;
/// The fixture's frame rate, 16.16 fixed point.
const FRAMERATE: u32 = 30 << 16;

/// The fixture's caps: a 640x480 30 fps H.264 elementary stream.
fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: g2g_core::VideoCodec::H264,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Fixed(FRAMERATE),
    }
}

/// The decoded frames' caps: NV12 at the fixture's geometry.
fn nv12_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Any,
        interlace: Interlace::Any,
    }
}

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// A machine without the NVIDIA decoder (or without a wgpu adapter) is a skip,
/// not a failure: this file is a hardware test.
fn skip_without_hardware(err: &G2gError) -> bool {
    let skip = matches!(err, G2gError::Hardware(_));
    if skip {
        eprintln!("skipping: no usable NVDEC / wgpu device here ({err:?})");
    }
    skip
}

/// `nvdec ! wgpusink` on a launch line: the bridge is spliced by the parser, so
/// the text pipeline never names it, and the sink is fed the GPU texture domain.
#[test]
fn a_launch_line_splices_the_cuda_bridge_into_the_wgpu_sink() {
    let registry = default_registry();
    let graph = parse_launch(
        &registry,
        &format!("filesrc location={FIXTURE} ! h264parse ! nvdec ! wgpusink"),
    )
    .expect("the pipeline parses");

    assert_eq!(
        graph.node_count(),
        5,
        "filesrc, h264parse, nvdec, the spliced bridge, wgpusink"
    );
    // The sink is the node nothing reads from; whatever feeds it must hand over
    // the GPU texture domain.
    let sink = (0..graph.node_count() as u32)
        .map(NodeId)
        .find(|node| !graph.edges().iter().any(|e| e.src.node == *node))
        .expect("a terminal sink node");
    let into_sink = graph
        .edges()
        .iter()
        .find(|e| e.dst.node == sink)
        .expect("an edge into the sink");
    assert_eq!(
        graph
            .element(into_sink.src.node)
            .expect("the node feeding the sink is an element")
            .output_domains(),
        DomainSet::only(MemoryDomainKind::WgpuTexture),
        "the sink is fed a GPU texture, not a system-memory download"
    );

    // The spliced graph still negotiates: NV12 all the way, with the link into
    // the sink carrying GPU memory.
    let (_, caps, domains) =
        block_on(negotiate_graph(graph)).expect("the spliced pipeline negotiates");
    assert!(
        domains.contains(&MemoryDomainKind::WgpuTexture),
        "a link carries the GPU texture domain: {domains:?}"
    );
    // Geometry is whatever the solver's placeholder settles on: a text pipeline
    // learns the real size from the SPS at runtime, and the sink follows.
    assert!(
        matches!(
            caps.last(),
            Some(Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                ..
            })
        ),
        "the sink's link carries decoded NV12: {caps:?}"
    );
}

/// A system-memory source needs no bridge: the sink takes system memory too.
#[test]
fn a_system_memory_source_reaches_the_wgpu_sink_unspliced() {
    let registry = default_registry();
    let graph = parse_launch(&registry, "videotestsrc num-buffers=1 ! wgpusink")
        .expect("the pipeline parses");
    assert_eq!(graph.node_count(), 2, "nothing spliced");

    // Negotiation settles on a layout the sink's renderer can upload, all in
    // system memory. (It stops short of `configure_pipeline`, so no window opens.)
    let (_, caps, domains) = block_on(negotiate_graph(graph)).expect("the pipeline negotiates");
    assert!(matches!(
        caps[0],
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(_),
            height: Dim::Fixed(_),
            ..
        }
    ));
    assert_eq!(domains[0], MemoryDomainKind::System);
}

/// The hardware path: NVDEC decodes into CUDA memory, the auto-plug splices the
/// bridge, and every frame reaching the sink is a wgpu texture (never a
/// system-memory download) that the sink blits. The sink is offscreen (no
/// compositor here) on the device the bridge produced the texture on: what the
/// windowed sink does bar the surface present.
#[tokio::test]
async fn nvdec_frames_reach_the_wgpu_sink_on_the_gpu() {
    let interop = match shared_interop_device().await {
        Ok(interop) => interop,
        Err(e) => {
            assert!(skip_without_hardware(&e));
            return;
        }
    };
    let ctx = GpuContext::from_wgpu(
        interop.instance.clone(),
        interop.adapter.clone(),
        interop.device.clone(),
        interop.queue.clone(),
    );
    let kinds = Arc::new(Mutex::new(Vec::new()));
    let pixels = Arc::new(Mutex::new(Vec::new()));

    let mut graph: Graph<GraphNode> = Graph::new();
    let src = graph.add_source(GraphNode::source(FileSrc::new(FIXTURE, h264_caps())));
    let parse = graph.add_transform(GraphNode::element(H264Parse::new()));
    let decode = graph.add_transform(GraphNode::element(NvDec::new()));
    let present = graph.add_sink(GraphNode::element(PresentOffscreen {
        sink: WgpuSink::offscreen(ctx, WIDTH, HEIGHT),
        kinds: Arc::clone(&kinds),
        pixels: Arc::clone(&pixels),
    }));
    graph.link(src, parse).unwrap();
    graph.link(parse, decode).unwrap();
    graph.link(decode, present).unwrap();

    let graph = auto_plug_cuda_converters(graph);
    assert_eq!(graph.node_count(), 5, "the CUDA -> wgpu bridge was spliced");
    match run_graph(graph, &NullClock, 2).await {
        Ok(_) => {}
        Err(e) if skip_without_hardware(&e) => return,
        Err(e) => panic!("unexpected error: {e:?}"),
    }

    let kinds = kinds.lock().unwrap();
    assert!(!kinds.is_empty(), "the decoder produced frames");
    assert!(
        kinds.iter().all(|k| *k == MemoryDomainKind::WgpuTexture),
        "every decoded frame arrived as a wgpu texture, none downloaded: {kinds:?}"
    );

    let pixels = pixels.lock().unwrap();
    assert_eq!(pixels.len(), (WIDTH * HEIGHT * 4) as usize);
    let lit = pixels
        .chunks_exact(4)
        .filter(|px| px[0] > 16 || px[1] > 16 || px[2] > 16)
        .count();
    assert!(
        lit > (WIDTH * HEIGHT / 10) as usize,
        "the presented frame carries picture, not a cleared target ({lit} lit pixels)"
    );
}

/// Stands in for the windowed sink: the same accepted caps and memory domains,
/// driving the same renderer into an offscreen target. Each frame is presented
/// and released inside `process`, because a bridged frame must not outlive the
/// decoder whose CUDA context its image was imported into.
struct PresentOffscreen {
    sink: WgpuSink,
    kinds: Arc<Mutex<Vec<MemoryDomainKind>>>,
    pixels: Arc<Mutex<Vec<u8>>>,
}

impl AsyncElement for PresentOffscreen {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: Interlace::Any,
        }))
    }
    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::WgpuTexture).with(MemoryDomainKind::System)
    }
    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.sink.configure_pipeline(absolute_caps)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    self.kinds.lock().unwrap().push(frame.domain.kind());
                    self.sink.present_frame(&frame.domain)?;
                    *self.pixels.lock().unwrap() = self.sink.read_target()?;
                }
                PipelinePacket::Eos => {}
                _ => {}
            }
            Ok(())
        })
    }
}

/// The keep-alive the bridge emits is one the sink recognises, whatever produced
/// it: a texture wrapped by any other type is refused rather than mis-sampled.
#[test]
fn a_foreign_gpu_frame_is_refused_by_the_sink() {
    use g2g_core::memory::OwnedWgpuTexture;

    #[derive(Debug)]
    struct ForeignTexture;
    impl g2g_core::WgpuKeepAlive for ForeignTexture {
        fn as_any(&self) -> &dyn core::any::Any {
            self
        }
    }

    let interop = match g2g_core::runtime::block_on(shared_interop_device()) {
        Ok(interop) => interop,
        Err(e) => {
            assert!(skip_without_hardware(&e));
            return;
        }
    };
    let ctx = GpuContext::from_wgpu(
        interop.instance.clone(),
        interop.adapter.clone(),
        interop.device.clone(),
        interop.queue.clone(),
    );
    let mut sink = WgpuSink::offscreen(ctx, WIDTH, HEIGHT);
    AsyncElement::configure_pipeline(&mut sink, &nv12_caps()).expect("NV12 configure");

    let domain = MemoryDomain::WgpuTexture(OwnedWgpuTexture::new(
        WIDTH,
        HEIGHT,
        Arc::new(ForeignTexture),
    ));
    assert!(matches!(
        sink.present_frame(&domain),
        Err(G2gError::UnsupportedDomain)
    ));
}
