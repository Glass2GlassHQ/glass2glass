//! M9 fan-out: a `Router` switched mid-stream sends frame 0 to branch A and
//! frames 1.. to branch B, through the `run_source_fanout` runner, with EOS
//! broadcast to both branches.
//!
//! Deterministic via the `m8_slot_swap` barrier pattern: branch A signals
//! when it receives frame 0, the driver then re-targets the router and
//! releases the parked source, so no frame is in flight during the switch.
//! (Routing/broadcast logic itself is unit-tested in `g2g-core::fanout`.)

use core::future::Future;
use core::pin::Pin;
use std::sync::Arc;

use tokio::sync::{oneshot, Notify};

use g2g_core::element::DynAsyncElement;
use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_source_fanout, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, G2gError, MemoryDomain, OutputSink, PipelineClock,
    PipelinePacket, Rate, Router, RawVideoFormat,
};

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(16),
        height: Dim::Fixed(16),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn make_frame(seq: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
        timing: FrameTiming::default(),
        sequence: seq,
    }
}

/// Pushes frame 0, parks on `release`, then pushes frames 1..4 and EOS. The
/// router is re-targeted by the driver while this source is parked.
struct BarrierSrc {
    release: Arc<Notify>,
    configured: bool,
}

impl SourceLoop for BarrierSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(caps()))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            assert!(self.configured, "runner must configure source before run");
            out.push(PipelinePacket::DataFrame(make_frame(0))).await?;
            self.release.notified().await;
            for seq in 1..4 {
                out.push(PipelinePacket::DataFrame(make_frame(seq))).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(4)
        })
    }
}

/// Counting sink that fires a oneshot on its first `DataFrame`.
struct SignalSink {
    received: u64,
    last_seq: Option<u64>,
    eos: bool,
    first_tx: Option<oneshot::Sender<()>>,
}

impl SignalSink {
    fn new(first_tx: Option<oneshot::Sender<()>>) -> Self {
        Self { received: 0, last_seq: None, eos: false, first_tx }
    }
}

impl AsyncElement for SignalSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

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
        match packet {
            PipelinePacket::DataFrame(f) => {
                self.received += 1;
                self.last_seq = Some(f.sequence);
                if let Some(tx) = self.first_tx.take() {
                    let _ = tx.send(());
                }
            }
            PipelinePacket::Eos => self.eos = true,
            PipelinePacket::CapsChanged(_) | PipelinePacket::Flush => {}
        }
        Box::pin(async { Ok(()) })
    }
}

#[tokio::test]
async fn router_switched_mid_stream_splits_across_branches() {
    let mut router = Router::new(2);
    let router_handle = router.handle();
    let release = Arc::new(Notify::new());
    let (first_tx, first_rx) = oneshot::channel();

    let mut src = BarrierSrc { release: release.clone(), configured: false };
    let mut sink_a = SignalSink::new(Some(first_tx));
    let mut sink_b = SignalSink::new(None);
    let clock = ZeroClock;

    let driver = async move {
        first_rx.await.expect("branch A must signal on frame 0");
        router_handle.select(1);
        release.notify_one();
    };

    let run = async {
        let sinks: Vec<&mut dyn DynAsyncElement> = vec![&mut sink_a, &mut sink_b];
        run_source_fanout(&mut src, &mut router, sinks, &clock, 1).await
    };

    let (res, ()) = tokio::join!(run, driver);
    let stats = res.expect("fan-out pipeline should complete");

    assert_eq!(stats.frames_emitted, 4);
    assert_eq!(stats.frames_consumed, 4, "every frame reaches exactly one branch");

    assert_eq!(sink_a.received, 1, "branch A got frame 0 before the switch");
    assert_eq!(sink_a.last_seq, Some(0));
    assert!(sink_a.eos, "EOS broadcast to branch A");

    assert_eq!(sink_b.received, 3, "branch B got frames 1,2,3 after the switch");
    assert_eq!(sink_b.last_seq, Some(3));
    assert!(sink_b.eos, "EOS broadcast to branch B");
}
