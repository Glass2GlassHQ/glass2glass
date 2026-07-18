//! M352: prove the M351 two-sided allocation-domain negotiation end-to-end on a
//! real NVIDIA GPU. `NvDec` is now a multi-domain producer: it can keep decoded
//! NV12 surfaces resident in CUDA device memory (zero-copy) *or* download them to
//! System, advertising both via `output_domains` and settling the choice in
//! `configure_allocation` (reconciling the downstream proposal against its
//! capability with `AllocationParams::resolve_for_producer`).
//!
//! The negotiation, not a flag, decides: the *same* decoder driven through
//! `run_graph` keeps frames on the GPU for a CUDA-accepting consumer and
//! downloads for a System-only one. The diamond cases additionally exercise the
//! M351 tee `join` (set-intersection of the branches' accepted domains) with real
//! NVDEC frames. We verify the outcome by inspecting each frame's
//! `MemoryDomain::kind()` at the sink.
//!
//! Hardware test: needs an NVIDIA GPU with NVCUVID (the `nvdec` feature). Skips
//! gracefully (no panic) when the decoder cannot initialise, so it is a no-op on
//! a machine without the hardware.

#![cfg(feature = "nvdec")]

use core::future::Future;
use core::pin::Pin;
use std::boxed::Box;
use std::sync::{Arc, Mutex};

use g2g_core::memory::{DomainSet, SystemSlice};
use g2g_core::runtime::{run_graph, GraphNode, SourceLoop};
use g2g_core::{
    AllocationParams, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, Frame,
    FrameTiming, G2gError, Graph, MemoryDomain, MemoryDomainKind, OutputSink, PipelineClock,
    PipelinePacket, Rate, RawVideoFormat, VideoCodec,
};
use g2g_plugins::nvdec::NvDec;

const W: u32 = 640;
const H: u32 = 480;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(W),
        height: Dim::Fixed(H),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn nv12_any() -> CapsSet {
    CapsSet::one(Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    })
}

/// Annex-B access-unit splitter (a new AU starts at the first VCL slice once the
/// current AU already has one). Same helper as the cudagl / cuda-wgpu smokes.
fn split_access_units(bs: &[u8]) -> Vec<Vec<u8>> {
    let mut codes: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i + 3 <= bs.len() {
        if bs[i] == 0 && bs[i + 1] == 0 && bs[i + 2] == 1 {
            codes.push((i, i + 3));
            i += 3;
        } else if i + 4 <= bs.len()
            && bs[i] == 0
            && bs[i + 1] == 0
            && bs[i + 2] == 0
            && bs[i + 3] == 1
        {
            codes.push((i, i + 4));
            i += 4;
        } else {
            i += 1;
        }
    }
    let mut aus = Vec::new();
    let mut start: Option<usize> = None;
    let mut has_vcl = false;
    for &(sc, nal) in &codes {
        let is_vcl = (1..=5).contains(&(bs[nal] & 0x1f));
        if is_vcl && has_vcl {
            aus.push(bs[start.take().unwrap()..sc].to_vec());
            has_vcl = false;
        }
        if start.is_none() {
            start = Some(sc);
        }
        has_vcl |= is_vcl;
    }
    if let Some(s) = start {
        aus.push(bs[s..].to_vec());
    }
    aus
}

fn fixture_access_units() -> Vec<Vec<u8>> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/h264_640x480.h264"
    );
    let bs = std::fs::read(path).expect("read committed H.264 fixture");
    let aus = split_access_units(&bs);
    assert!(!aus.is_empty(), "no access units in fixture");
    aus
}

/// Source that replays the fixture's H.264 access units as System-memory
/// `CompressedVideo` frames, then EOS.
struct H264Replay {
    aus: Vec<Vec<u8>>,
}

impl SourceLoop for H264Replay {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(h264_caps()))
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let aus = core::mem::take(&mut self.aus);
            let mut seq = 0u64;
            for au in aus {
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(au.into_boxed_slice())),
                    timing: FrameTiming {
                        pts_ns: seq * 33_000_000,
                        ..FrameTiming::default()
                    },
                    sequence: seq,
                    meta: Default::default(),
                };
                out.push(PipelinePacket::DataFrame(frame)).await?;
                seq += 1;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(seq)
        })
    }
}

/// Sink that proposes `proposal` (its accepted domain set) and records the memory
/// domain of every frame it actually receives, so we can see whether NVDEC kept
/// the frame on the GPU or downloaded it.
struct CaptureSink {
    proposal: AllocationParams,
    seen: Arc<Mutex<Vec<MemoryDomainKind>>>,
}

impl AsyncElement for CaptureSink {
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

    fn propose_allocation(&self, _caps: &Caps) -> Option<AllocationParams> {
        Some(self.proposal)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                self.seen.lock().unwrap().push(f.domain.kind());
            }
            Ok(())
        })
    }
}

fn capture(proposal: AllocationParams) -> (CaptureSink, Arc<Mutex<Vec<MemoryDomainKind>>>) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    (
        CaptureSink {
            proposal,
            seen: Arc::clone(&seen),
        },
        seen,
    )
}

/// A proposal accepting exactly the given domain set, preferring `prefer`.
fn proposal(prefer: MemoryDomainKind, accepts: DomainSet) -> AllocationParams {
    AllocationParams {
        size_bytes: 64,
        min_buffers: 2,
        align: 256,
        domain: prefer,
        accepts,
    }
}

fn cuda_only() -> AllocationParams {
    proposal(
        MemoryDomainKind::Cuda,
        DomainSet::only(MemoryDomainKind::Cuda),
    )
}

fn system_only() -> AllocationParams {
    proposal(
        MemoryDomainKind::System,
        DomainSet::only(MemoryDomainKind::System),
    )
}

fn cuda_or_system() -> AllocationParams {
    proposal(
        MemoryDomainKind::Cuda,
        DomainSet::only(MemoryDomainKind::Cuda).with(MemoryDomainKind::System),
    )
}

/// Returns false (and prints a skip line) when the error is a hardware failure,
/// i.e. there is no usable NVDEC on this machine; panics on any other error.
fn skip_if_no_gpu(err: &G2gError) -> bool {
    if matches!(err, G2gError::Hardware(_)) {
        std::eprintln!("skipping m352: NVDEC unavailable ({err:?})");
        true
    } else {
        false
    }
}

/// NVDEC -> CUDA-accepting sink: the negotiation settles on Cuda, so every frame
/// stays device-resident (zero-copy, no PCIe download).
#[tokio::test]
async fn nvdec_keeps_frames_on_gpu_for_cuda_sink() {
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(H264Replay {
        aus: fixture_access_units(),
    }));
    let dec = g.add_transform(GraphNode::element(NvDec::new()));
    let (sink, seen) = capture(cuda_only());
    let snk = g.add_sink(GraphNode::element(sink));
    g.link(src, dec).unwrap();
    g.link(dec, snk).unwrap();

    match run_graph(g, &NullClock, 4).await {
        Ok(_) => {}
        Err(e) if skip_if_no_gpu(&e) => return,
        Err(e) => panic!("unexpected error: {e:?}"),
    }
    let domains = seen.lock().unwrap().clone();
    assert!(!domains.is_empty(), "decoder produced frames");
    assert!(
        domains.iter().all(|&d| d == MemoryDomainKind::Cuda),
        "a CUDA-accepting sink keeps NVDEC frames on the GPU: {domains:?}",
    );
}

/// NVDEC -> System-only sink: the negotiation settles on System, so the same
/// decoder downloads every frame device->host.
#[tokio::test]
async fn nvdec_downloads_for_system_sink() {
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(H264Replay {
        aus: fixture_access_units(),
    }));
    let dec = g.add_transform(GraphNode::element(NvDec::new()));
    let (sink, seen) = capture(system_only());
    let snk = g.add_sink(GraphNode::element(sink));
    g.link(src, dec).unwrap();
    g.link(dec, snk).unwrap();

    match run_graph(g, &NullClock, 4).await {
        Ok(_) => {}
        Err(e) if skip_if_no_gpu(&e) => return,
        Err(e) => panic!("unexpected error: {e:?}"),
    }
    let domains = seen.lock().unwrap().clone();
    assert!(!domains.is_empty(), "decoder produced frames");
    assert!(
        domains.iter().all(|&d| d == MemoryDomainKind::System),
        "a System-only sink makes NVDEC download every frame: {domains:?}",
    );
}

/// Diamond, GPU outcome: NVDEC -> tee -> {accepts both, accepts CUDA only}. The
/// tee join intersects the branches' accepted sets to {Cuda}, so NVDEC keeps the
/// shared surface on the GPU and both branches see Cuda (zero-copy fan-out).
#[tokio::test]
async fn nvdec_diamond_joins_to_gpu() {
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(H264Replay {
        aus: fixture_access_units(),
    }));
    let dec = g.add_transform(GraphNode::element(NvDec::new()));
    let tee = g.add_tee(2);
    let (sink_a, seen_a) = capture(cuda_or_system());
    let (sink_b, seen_b) = capture(cuda_only());
    let a = g.add_sink(GraphNode::element(sink_a));
    let b = g.add_sink(GraphNode::element(sink_b));
    g.link(src, dec).unwrap();
    g.link(dec, tee.input()).unwrap();
    g.link(tee.out(0), a).unwrap();
    g.link(tee.out(1), b).unwrap();

    match run_graph(g, &NullClock, 4).await {
        Ok(_) => {}
        Err(e) if skip_if_no_gpu(&e) => return,
        Err(e) => panic!("unexpected error: {e:?}"),
    }
    for (seen, label) in [(seen_a, "accepts-both"), (seen_b, "cuda-only")] {
        let domains = seen.lock().unwrap().clone();
        assert!(!domains.is_empty(), "{label} branch saw frames");
        assert!(
            domains.iter().all(|&d| d == MemoryDomainKind::Cuda),
            "{label} branch should see Cuda (tee join settled on GPU): {domains:?}",
        );
    }
}

/// Diamond, System outcome: NVDEC -> tee -> {accepts both, accepts System only}.
/// The join intersects to {System}, so the same decoder downloads, and both
/// branches see System.
#[tokio::test]
async fn nvdec_diamond_joins_to_system() {
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(H264Replay {
        aus: fixture_access_units(),
    }));
    let dec = g.add_transform(GraphNode::element(NvDec::new()));
    let tee = g.add_tee(2);
    let (sink_a, seen_a) = capture(cuda_or_system());
    let (sink_b, seen_b) = capture(system_only());
    let a = g.add_sink(GraphNode::element(sink_a));
    let b = g.add_sink(GraphNode::element(sink_b));
    g.link(src, dec).unwrap();
    g.link(dec, tee.input()).unwrap();
    g.link(tee.out(0), a).unwrap();
    g.link(tee.out(1), b).unwrap();

    match run_graph(g, &NullClock, 4).await {
        Ok(_) => {}
        Err(e) if skip_if_no_gpu(&e) => return,
        Err(e) => panic!("unexpected error: {e:?}"),
    }
    for (seen, label) in [(seen_a, "accepts-both"), (seen_b, "system-only")] {
        let domains = seen.lock().unwrap().clone();
        assert!(!domains.is_empty(), "{label} branch saw frames");
        assert!(
            domains.iter().all(|&d| d == MemoryDomainKind::System),
            "{label} branch should see System (tee join settled on download): {domains:?}",
        );
    }
}
