//! M354: domain-converter auto-plug. `auto_plug_domain_converters` splices a
//! memory-domain converter onto any edge whose producer and consumer cannot agree
//! on a domain, the structural complement to the M351/M352 in-band negotiation
//! (which settles a shared domain when one exists). The mechanism tests use fake
//! elements + a fake converter (domains are declarative, so no GPU is needed),
//! so they run in CI. The hardware test (`nvenc` feature) proves the real CUDA
//! path on the 3060.

use core::future::Future;
use core::pin::Pin;
use std::boxed::Box;
use std::sync::{Arc, Mutex};

use g2g_core::memory::{DomainSet, SystemSlice};
use g2g_core::runtime::{auto_plug_domain_converters, run_graph, GraphNode, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, Frame, FrameTiming,
    G2gError, Graph, MemoryDomain, MemoryDomainKind, OutputSink, PipelineClock, PipelinePacket,
    Rate, RawVideoFormat,
};

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn nv12() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(8),
        height: Dim::Fixed(8),
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

fn frame(seq: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(vec![seq as u8].into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: seq,
        meta: Default::default(),
    })
}

/// System-memory source (default `output_domains == {System}`).
struct SysSource {
    count: u32,
}

impl SourceLoop for SysSource {
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
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for seq in 0..self.count as u64 {
                out.push(frame(seq)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.count as u64)
        })
    }
}

/// Fake converter: a pass-through that *declares* it emits the GPU domain, so a
/// downstream domain-strict consumer is satisfied. The runtime frame is left as
/// it is (domains are declarative for the splice decision; the real CUDA copy is
/// the hardware test's job).
struct FakeConverter;

impl AsyncElement for FakeConverter {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Identity(nv12_any())
    }
    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn output_memory(&self) -> MemoryDomainKind {
        MemoryDomainKind::Cuda
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

/// Sink with a domain requirement; counts the frames it receives.
struct DomainSink {
    requires: DomainSet,
    seen: Arc<Mutex<u64>>,
}

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
        self.requires
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if matches!(packet, PipelinePacket::DataFrame(_)) {
                *self.seen.lock().unwrap() += 1;
            }
            Ok(())
        })
    }
}

/// Fake factory: a System->Cuda converter, nothing else.
fn fake_factory(from: MemoryDomainKind, to: MemoryDomainKind) -> Option<GraphNode> {
    match (from, to) {
        (MemoryDomainKind::System, MemoryDomainKind::Cuda) => {
            Some(GraphNode::element(FakeConverter))
        }
        _ => None,
    }
}

/// A System producer feeding a Cuda-only consumer gets a converter spliced, and
/// the spliced graph negotiates and runs.
#[tokio::test]
async fn splices_converter_on_linear_domain_conflict() {
    let seen = Arc::new(Mutex::new(0u64));
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(SysSource { count: 4 }));
    let snk = g.add_sink(GraphNode::element(DomainSink {
        requires: DomainSet::only(MemoryDomainKind::Cuda),
        seen: Arc::clone(&seen),
    }));
    g.link(src, snk).unwrap();

    let g = auto_plug_domain_converters(g, &fake_factory);
    assert_eq!(
        g.node_count(),
        3,
        "a converter was spliced between source and sink"
    );

    let stats = run_graph(g, &NullClock, 4)
        .await
        .expect("spliced graph runs");
    assert_eq!(stats.frames_consumed, 4);
    assert_eq!(
        *seen.lock().unwrap(),
        4,
        "all frames reached the Cuda-only sink via the converter"
    );
}

/// No conflict, no splice: a System sink leaves the graph untouched.
#[tokio::test]
async fn no_splice_when_domains_agree() {
    let seen = Arc::new(Mutex::new(0u64));
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(SysSource { count: 3 }));
    let snk = g.add_sink(GraphNode::element(DomainSink {
        requires: DomainSet::only(MemoryDomainKind::System),
        seen: Arc::clone(&seen),
    }));
    g.link(src, snk).unwrap();

    let g = auto_plug_domain_converters(g, &fake_factory);
    assert_eq!(g.node_count(), 2, "domains agree, nothing spliced");
    run_graph(g, &NullClock, 4).await.expect("runs");
    assert_eq!(*seen.lock().unwrap(), 3);
}

/// Tee fan-out: only the branch whose consumer conflicts with the (traced-through
/// the tee) producer domain gets a converter; the agreeing branch is untouched.
#[tokio::test]
async fn splices_only_the_conflicting_tee_branch() {
    let cuda_seen = Arc::new(Mutex::new(0u64));
    let sys_seen = Arc::new(Mutex::new(0u64));
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(SysSource { count: 5 }));
    let tee = g.add_tee(2);
    let cuda = g.add_sink(GraphNode::element(DomainSink {
        requires: DomainSet::only(MemoryDomainKind::Cuda),
        seen: Arc::clone(&cuda_seen),
    }));
    let sys = g.add_sink(GraphNode::element(DomainSink {
        requires: DomainSet::only(MemoryDomainKind::System),
        seen: Arc::clone(&sys_seen),
    }));
    g.link(src, tee.input()).unwrap();
    g.link(tee.out(0), cuda).unwrap();
    g.link(tee.out(1), sys).unwrap();

    let g = auto_plug_domain_converters(g, &fake_factory);
    assert_eq!(
        g.node_count(),
        5,
        "exactly one converter spliced (on the Cuda branch)"
    );

    let stats = run_graph(g, &NullClock, 4)
        .await
        .expect("spliced diamond runs");
    assert_eq!(
        stats.frames_consumed, 10,
        "5 frames to each of the two branches"
    );
    assert_eq!(
        *cuda_seen.lock().unwrap(),
        5,
        "Cuda branch fed via the converter"
    );
    assert_eq!(*sys_seen.lock().unwrap(), 5, "System branch untouched");
}

// --- Hardware: the real CUDA path on an NVIDIA GPU. ---

/// A CPU-side NV12 stream feeds `NvEnc` (CUDA-only input) with no hand-wiring:
/// `auto_plug_cuda_converters` splices a `CudaUpload`, so the System source
/// reaches the encoder and produces H.264. Skips gracefully without a GPU.
#[cfg(feature = "nvenc")]
#[tokio::test]
async fn auto_plugs_cuda_upload_before_nvenc() {
    use g2g_plugins::cuda::auto_plug_cuda_converters;
    use g2g_plugins::nvenc::NvEnc;

    const W: u32 = 320;
    const H: u32 = 240;

    fn nv12_hw() -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(W),
            height: Dim::Fixed(H),
            framerate: Rate::Fixed(30 << 16),
        }
    }

    /// System NV12 source at HW geometry.
    struct HwSysSource {
        count: u32,
    }
    impl SourceLoop for HwSysSource {
        type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
        type CapsFuture<'a>
            = core::future::Ready<Result<Caps, G2gError>>
        where
            Self: 'a;
        fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
            core::future::ready(Ok(nv12_hw()))
        }
        fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
            Ok(ConfigureOutcome::Accepted)
        }
        fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
            let n = self.count;
            Box::pin(async move {
                let total = (W * H + 2 * (W / 2) * (H / 2)) as usize;
                for seq in 0..n as u64 {
                    let mut buf = vec![128u8; total];
                    for (i, b) in buf[..(W * H) as usize].iter_mut().enumerate() {
                        *b = ((i as u64 + seq) & 0xff) as u8;
                    }
                    out.push(PipelinePacket::DataFrame(Frame {
                        domain: MemoryDomain::System(SystemSlice::from_boxed(
                            buf.into_boxed_slice(),
                        )),
                        timing: FrameTiming {
                            pts_ns: seq * 33_000_000,
                            ..FrameTiming::default()
                        },
                        sequence: seq,
                        meta: Default::default(),
                    }))
                    .await?;
                }
                out.push(PipelinePacket::Eos).await?;
                Ok(n as u64)
            })
        }
    }

    /// Records H.264 Annex-B access units.
    struct AuSink {
        aus: Arc<Mutex<Vec<Vec<u8>>>>,
    }
    impl AsyncElement for AuSink {
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
            packet: PipelinePacket,
            _out: &'a mut dyn OutputSink,
        ) -> Self::ProcessFuture<'a> {
            Box::pin(async move {
                if let PipelinePacket::DataFrame(f) = packet {
                    if let MemoryDomain::System(s) = &f.domain {
                        self.aus.lock().unwrap().push(s.as_slice().to_vec());
                    }
                }
                Ok(())
            })
        }
    }

    let aus = Arc::new(Mutex::new(Vec::new()));
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(HwSysSource { count: 8 }));
    let enc = g.add_transform(GraphNode::element(NvEnc::new()));
    let snk = g.add_sink(GraphNode::element(AuSink {
        aus: Arc::clone(&aus),
    }));
    g.link(src, enc).unwrap();
    g.link(enc, snk).unwrap();

    // The System source and CUDA-only NvEnc disagree on domain; auto-plug bridges.
    let g = auto_plug_cuda_converters(g);
    assert_eq!(
        g.node_count(),
        4,
        "a CudaUpload was spliced between source and NvEnc"
    );

    match run_graph(g, &NullClock, 4).await {
        Ok(_) => {}
        Err(G2gError::Hardware(_)) => {
            std::eprintln!("skipping auto_plugs_cuda_upload_before_nvenc: no NVIDIA GPU");
            return;
        }
        Err(e) => panic!("unexpected error: {e:?}"),
    }
    let aus = aus.lock().unwrap();
    assert!(
        !aus.is_empty(),
        "encoder produced H.264 via the auto-plugged CudaUpload"
    );
    assert!(
        aus.iter().any(|au| au.windows(3).any(|w| w == [0, 0, 1])),
        "access units carry Annex-B start codes",
    );
}
