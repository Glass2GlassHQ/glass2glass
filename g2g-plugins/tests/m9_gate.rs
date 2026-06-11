//! M9 fan-out: a `Gate` closed mid-stream stops the data flow while EOS still
//! passes, driven through the existing `run_source_transform_sink` runner.
//!
//! `Gate` is a plain `AsyncElement`, so it needs no new runner. The toggle is
//! made deterministic with the barrier pattern from `m8_slot_swap`: the sink
//! signals when it has received frame 0, a driver future then closes the gate
//! and releases the source — which was parked — so no frame is in flight
//! during the toggle. (The per-frame drop logic itself is unit-tested in
//! `g2g-core::fanout`.)

use core::future::Future;
use core::pin::Pin;
use std::sync::Arc;

use tokio::sync::{oneshot, Notify};

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_source_transform_sink, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, Gate, G2gError, MemoryDomain, OutputSink,
    PipelineClock, PipelinePacket, Rate, RawVideoFormat,
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
/// gate is toggled by the driver while this source is parked, so the toggle
/// is causally ordered before frames 1.. reach the gate.
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

/// Counting sink that fires a oneshot on its first `DataFrame` so the driver
/// can act once frame 0 has flowed through.
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
async fn gate_closed_mid_stream_stops_data_but_not_eos() {
    let mut gate = Gate::new(true);
    let gate_handle = gate.handle();
    let release = Arc::new(Notify::new());
    let (first_tx, first_rx) = oneshot::channel();

    let mut src = BarrierSrc { release: release.clone(), configured: false };
    let mut snk = SignalSink::new(Some(first_tx));
    let clock = ZeroClock;

    let run = run_source_transform_sink(&mut src, &mut gate, &mut snk, &clock, 1);
    let driver = async move {
        first_rx.await.expect("sink must signal on frame 0");
        gate_handle.set_open(false);
        release.notify_one();
    };

    let (res, ()) = tokio::join!(run, driver);
    let stats = res.expect("gated pipeline should complete");

    assert_eq!(stats.frames_emitted, 4, "source emits all four frames");
    assert_eq!(stats.frames_consumed, 1, "only frame 0 passed before the gate closed");
    assert_eq!(snk.received, 1);
    assert_eq!(snk.last_seq, Some(0));
    assert!(snk.eos, "EOS passes the gate regardless of open state");
}
