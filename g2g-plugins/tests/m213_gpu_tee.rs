//! M213 - zero-copy GPU fan-out. A frame whose memory lives on the GPU (here a
//! CUDA device buffer) is broadcast through a `tee` to two branches. Before
//! M213 the tee deep-copied `System` frames and failed loud
//! (`UnsupportedDomain`) on any GPU domain; now it shares the backing allocation
//! by reference count (`MemoryDomain::share`), so the canonical
//! decode-on-GPU -> {inference, display} graph fans out with no device-to-host
//! copy. No real GPU: a mock keep-alive stands in for the decoder's `AVFrame`
//! and proves the allocation is shared once, not copied per branch.

use core::future::Future;
use core::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use g2g_core::frame::Frame;
use g2g_core::runtime::{run_graph, GraphNode, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, CudaKeepAlive, Dim, FrameTiming,
    G2gError, Graph, MemoryDomain, OutputSink, OwnedCudaBuffer, PipelineClock, PipelinePacket,
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
        width: Dim::Fixed(1920),
        height: Dim::Fixed(1080),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Stands in for a decoder's CUDA `AVFrame` owner; counts its own drops so the
/// test can prove the device allocation is released exactly once (shared by both
/// branches), not once per branch (which a copy would imply). Naturally
/// `Send + Sync` (it only holds an `Arc<Atomic>`), so no `unsafe` is needed.
#[derive(Debug)]
struct MockCudaFrame(Arc<AtomicUsize>);
impl Drop for MockCudaFrame {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
impl CudaKeepAlive for MockCudaFrame {}

/// Emits one CUDA-domain frame, then EOS.
struct CudaSrc {
    drops: Arc<AtomicUsize>,
    sent: bool,
}
impl SourceLoop for CudaSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(nv12()))
    }
    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(PipelinePacket::CapsChanged(nv12())).await?;
            let buf = OwnedCudaBuffer::new(
                0x1000,
                0x2000,
                2048,
                2048,
                1920,
                1080,
                0xC0FFEE,
                Arc::new(MockCudaFrame(self.drops.clone())),
            );
            let frame = Frame::new(MemoryDomain::Cuda(buf), FrameTiming::default(), 0);
            out.push(PipelinePacket::DataFrame(frame)).await?;
            self.sent = true;
            out.push(PipelinePacket::Eos).await?;
            Ok(1)
        })
    }
}

/// Asserts every frame it receives is still on the GPU (the share kept the
/// domain), and counts them via RunStats.
#[derive(Default)]
struct GpuSink;
impl AsyncElement for GpuSink {
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
                assert!(
                    matches!(f.domain, MemoryDomain::Cuda(_)),
                    "branch received the frame still on the GPU, not a host copy",
                );
            }
            Ok(())
        })
    }
}

#[tokio::test]
async fn gpu_frame_fans_out_through_a_tee_zero_copy() {
    let drops = Arc::new(AtomicUsize::new(0));
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(CudaSrc {
        drops: drops.clone(),
        sent: false,
    }));
    let tee = g.add_tee(2);
    let s0 = g.add_sink(GraphNode::element(GpuSink));
    let s1 = g.add_sink(GraphNode::element(GpuSink));
    g.link(src, tee.input()).unwrap();
    g.link(tee.out(0), s0).unwrap();
    g.link(tee.out(1), s1).unwrap();

    let stats = run_graph(g, &NullClock, 4)
        .await
        .expect("GPU tee pipeline runs");
    // One source frame reached BOTH branches: the tee fanned out a GPU frame
    // (previously UnsupportedDomain). Both branches asserted it stayed on the GPU.
    assert_eq!(stats.frames_emitted, 1);
    assert_eq!(
        stats.frames_consumed, 2,
        "the GPU frame reached both branches"
    );
    // The backing allocation was shared, not copied: the single mock AVFrame is
    // released exactly once after both branch frames drop.
    assert_eq!(
        drops.load(Ordering::SeqCst),
        1,
        "device allocation released once, not per branch"
    );
}
