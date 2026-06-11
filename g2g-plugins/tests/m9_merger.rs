//! M9 fan-in: a `Merger` forwards the selected input and drains the rest,
//! emitting one EOS only after every input has ended, driven through
//! `run_fanin_sink`.
//!
//! Both tests are deterministic. The fixed-selection case never switches, so
//! interleaving doesn't matter. The cut-over case parks input B until the
//! switch (the `m8_slot_swap` barrier pattern), removing the two-producers
//! race; a signaling sink tells the driver when input A's frames have landed.

use core::future::Future;
use core::pin::Pin;
use std::sync::Arc;

use tokio::sync::{oneshot, Notify};

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_fanin_sink, DynSourceLoop, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, G2gError, Merger, MemoryDomain, OutputSink,
    PipelineClock, PipelinePacket, Rate, VideoCodec, RawVideoFormat,
};
use g2g_plugins::fakesink::FakeSink;

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

/// Branch source: optionally parks on a `Notify`, then emits `count` frames
/// numbered from `start_seq`, then EOS.
struct BranchSrc {
    start_seq: u64,
    count: u64,
    park: Option<Arc<Notify>>,
    configured: bool,
}

impl SourceLoop for BranchSrc {
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
        let park = self.park.clone();
        let start = self.start_seq;
        let count = self.count;
        let configured = self.configured;
        Box::pin(async move {
            assert!(configured, "runner must configure source before run");
            if let Some(gate) = park {
                gate.notified().await;
            }
            for i in 0..count {
                out.push(PipelinePacket::DataFrame(make_frame(start + i))).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(count)
        })
    }
}

/// Counting sink that fires a oneshot once it has received `signal_at`
/// frames, so the driver can switch the merger after input A has landed.
struct SignalSink {
    received: u64,
    last_seq: Option<u64>,
    eos: bool,
    signal_at: u64,
    tx: Option<oneshot::Sender<()>>,
}

impl SignalSink {
    fn new(signal_at: u64, tx: oneshot::Sender<()>) -> Self {
        Self { received: 0, last_seq: None, eos: false, signal_at, tx: Some(tx) }
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
                if self.received == self.signal_at {
                    if let Some(tx) = self.tx.take() {
                        let _ = tx.send(());
                    }
                }
            }
            PipelinePacket::Eos => self.eos = true,
            PipelinePacket::CapsChanged(_) | PipelinePacket::Flush => {}
        }
        Box::pin(async { Ok(()) })
    }
}

#[tokio::test]
async fn merger_forwards_selected_input_and_discards_others() {
    let mut merger = Merger::new(2); // selected = 0
    let mut a = BranchSrc { start_seq: 0, count: 3, park: None, configured: false };
    let mut b = BranchSrc { start_seq: 100, count: 2, park: None, configured: false };
    let mut snk = FakeSink::new();
    let clock = ZeroClock;

    let stats = {
        let sources: Vec<&mut dyn DynSourceLoop> = vec![&mut a, &mut b];
        run_fanin_sink(sources, &mut merger, &mut snk, &clock, 4)
            .await
            .expect("fan-in should complete")
    };

    assert_eq!(stats.frames_emitted, 5, "both branches produced (3 + 2)");
    assert_eq!(stats.frames_consumed, 3, "only input 0 forwarded; input 1 discarded");
    assert_eq!(snk.received(), 3);
    assert_eq!(snk.last_sequence(), Some(2));
    assert!(snk.eos_seen(), "single merged EOS after both inputs ended");
}

#[tokio::test]
async fn merger_switched_mid_stream_cuts_over_inputs() {
    let mut merger = Merger::new(2);
    let merger_handle = merger.handle();
    let release = Arc::new(Notify::new());
    let (tx, rx) = oneshot::channel();

    // A runs free (input 0, frames 0,1); B parks until the switch (input 1,
    // frames 2,3).
    let mut a = BranchSrc { start_seq: 0, count: 2, park: None, configured: false };
    let mut b = BranchSrc { start_seq: 2, count: 2, park: Some(release.clone()), configured: false };
    let mut snk = SignalSink::new(2, tx);
    let clock = ZeroClock;

    let driver = async move {
        rx.await.expect("sink must signal after input 0's two frames");
        merger_handle.select(1);
        release.notify_one();
    };

    let run = async {
        let sources: Vec<&mut dyn DynSourceLoop> = vec![&mut a, &mut b];
        run_fanin_sink(sources, &mut merger, &mut snk, &clock, 4).await
    };

    let (res, ()) = tokio::join!(run, driver);
    let stats = res.expect("fan-in should complete");

    assert_eq!(stats.frames_emitted, 4);
    assert_eq!(stats.frames_consumed, 4, "input 0 then input 1 both forwarded");
    assert_eq!(snk.received, 4);
    assert_eq!(snk.last_seq, Some(3), "branch B's frames followed branch A's");
    assert!(snk.eos, "single merged EOS after both inputs ended");
}
