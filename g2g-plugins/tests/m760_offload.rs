//! M760: element-level cooperative offload (`offload::run_blocking`).
//!
//! Two properties:
//!   1. Offload is transparent: the real `VideoConvert` element produces
//!      byte-identical output whether `run_blocking` takes its `spawn_blocking`
//!      path (a tokio runtime is active) or its inline fallback (none is).
//!   2. Offload actually pipelines: with a heavy synchronous element's compute
//!      on tokio's blocking pool, the cooperative `run_graph` keeps servicing
//!      sibling arms. A transform whose blocking closure spin-waits on a flag the
//!      downstream sink sets can only make progress if the sink's arm runs while
//!      the closure is on the pool. A `block_in_place`-style runner would deadlock
//!      here (the test times out and fails instead of hanging).
#![cfg(feature = "offload")]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use g2g_core::element::{AsyncElement, BoxFuture, ConfigureOutcome, OutputSink};
use g2g_core::graph::Graph;
use g2g_core::runtime::{run_graph, run_source_transform_sink, GraphNode};
use g2g_core::RawVideoFormat;
use g2g_core::{Caps, G2gError, HardwareError, MemoryDomain, PipelineClock, PipelinePacket};

use g2g_plugins::offload;
use g2g_plugins::videoconvert::VideoConvert;
use g2g_plugins::videotestsrc::VideoTestSrc;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

// --- Property 1: spawn_blocking path is byte-identical to the inline path. ---

/// Sink that copies every `DataFrame`'s system bytes into a shared vec.
struct CapturingSink {
    frames: Arc<Mutex<Vec<Vec<u8>>>>,
}

impl AsyncElement for CapturingSink {
    type ProcessFuture<'a> = BoxFuture<'a, Result<(), G2gError>>;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(frame) = packet {
                let Some(slice) = frame.domain.as_system_slice() else {
                    return Err(G2gError::UnsupportedDomain);
                };
                self.frames.lock().unwrap().push(slice.as_slice().to_vec());
            }
            Ok(())
        })
    }
}

/// Run RGBA -> NV12 through the real `VideoConvert` element and return each
/// converted frame's bytes. `use_tokio` picks the executor: a tokio runtime
/// (so `run_blocking` uses `spawn_blocking`) or `embassy_futures::block_on`
/// (no tokio handle, so `run_blocking` runs the convert inline).
fn convert_frames(use_tokio: bool) -> Vec<Vec<u8>> {
    let frames = Arc::new(Mutex::new(Vec::new()));
    let captured = frames.clone();
    let fut = async move {
        let mut src = VideoTestSrc::new(32, 16, 30, 3);
        let mut conv = VideoConvert::new(RawVideoFormat::Nv12);
        let mut sink = CapturingSink { frames: captured };
        run_source_transform_sink(&mut src, &mut conv, &mut sink, &ZeroClock, 4)
            .await
            .expect("RGBA -> NV12 chain runs");
    };
    if use_tokio {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(fut);
    } else {
        embassy_futures::block_on(fut);
    }
    Arc::try_unwrap(frames).unwrap().into_inner().unwrap()
}

#[test]
fn offload_convert_is_byte_identical_to_inline() {
    let via_pool = convert_frames(true);
    let inline = convert_frames(false);

    assert_eq!(via_pool.len(), 3, "three frames converted");
    assert!(!via_pool[0].is_empty(), "converted frame has bytes");
    assert_eq!(
        via_pool, inline,
        "spawn_blocking convert must match the inline convert byte for byte"
    );
}

// --- Property 2: the sink's arm runs while a transform's closure is on the pool. ---

/// Passthrough transform whose second (and later) frame runs an offloaded
/// blocking closure that spin-waits until `flag` is set. The first frame is
/// forwarded immediately so the sink can receive it and set the flag; that can
/// only happen while this transform's closure is parked on the blocking pool,
/// which is the pipelining the test proves. The spin has a deadline so a
/// non-pipelining runner surfaces as a returned error, not an infinite loop.
struct SpinTransform {
    seen: u64,
    flag: Arc<AtomicBool>,
}

impl AsyncElement for SpinTransform {
    type ProcessFuture<'a> = BoxFuture<'a, Result<(), G2gError>>;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(_) => {
                    let n = self.seen;
                    self.seen += 1;
                    if n >= 1 {
                        let flag = self.flag.clone();
                        let saw = offload::run_blocking(move || {
                            let deadline = Instant::now() + Duration::from_secs(5);
                            while !flag.load(Ordering::SeqCst) {
                                if Instant::now() >= deadline {
                                    return false;
                                }
                                core::hint::spin_loop();
                            }
                            true
                        })
                        .await;
                        if !saw {
                            // The sink never ran while we were on the pool: the
                            // runner did not pipeline. Fail the graph.
                            return Err(G2gError::Hardware(HardwareError::Other));
                        }
                    }
                    out.push(packet_frame(n)).await?;
                    Ok(())
                }
                other => {
                    out.push(other).await?;
                    Ok(())
                }
            }
        })
    }
}

/// Rebuild a trivial passthrough packet carrying a fresh frame index. The
/// transform consumed the original `DataFrame` (moved into the match); the sink
/// only needs a `DataFrame` to fire, so emit a 1-byte system frame.
fn packet_frame(seq: u64) -> PipelinePacket {
    use g2g_core::frame::Frame;
    use g2g_core::memory::SystemSlice;
    PipelinePacket::DataFrame(Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(vec![0u8].into_boxed_slice())),
        timing: Default::default(),
        sequence: seq,
        meta: Default::default(),
    })
}

/// Sink that sets the shared flag on the first `DataFrame` it receives.
struct FlagSink {
    flag: Arc<AtomicBool>,
    count: Arc<AtomicU64>,
}

impl AsyncElement for FlagSink {
    type ProcessFuture<'a> = BoxFuture<'a, Result<(), G2gError>>;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(_) = packet {
                self.flag.store(true, Ordering::SeqCst);
                self.count.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        })
    }
}

#[tokio::test]
async fn sink_arm_runs_while_transform_closure_is_on_the_pool() {
    let flag = Arc::new(AtomicBool::new(false));
    let count = Arc::new(AtomicU64::new(0));

    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(VideoTestSrc::new(16, 16, 30, 2)));
    let tx = g.add_transform(GraphNode::element(SpinTransform {
        seen: 0,
        flag: flag.clone(),
    }));
    let sink = g.add_sink(GraphNode::element(FlagSink {
        flag: flag.clone(),
        count: count.clone(),
    }));
    g.link(src, tx).unwrap();
    g.link(tx, sink).unwrap();

    // The blocking closure can only return if the sink's arm ran while it was on
    // the pool. Cap the wall clock so a non-pipelining runner fails instead of
    // hanging the suite.
    let res = tokio::time::timeout(Duration::from_secs(10), run_graph(g, &ZeroClock, 4))
        .await
        .expect("graph must not hang: sink must run while the transform closure is on the pool");
    res.expect("graph runs to completion");

    assert!(
        flag.load(Ordering::SeqCst),
        "sink set the flag, so its arm ran"
    );
    assert_eq!(count.load(Ordering::SeqCst), 2, "sink received both frames");
}
