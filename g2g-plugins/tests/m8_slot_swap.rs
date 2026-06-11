//! M8 piece 6 end-to-end: an `ElementSlot` driven by the real runner, with
//! its element atomically swapped mid-stream through a detached `SwapHandle`.
//!
//! Proves three things together that the in-crate unit tests cannot:
//!   1. the `DynAsyncElement` blanket impl boxes a real `AsyncElement`,
//!   2. `SwapHandle` mutates the slot while the runner owns it by `&mut`,
//!   3. `ElementSlot` composes with `run_source_transform_sink` unchanged.
//!
//! The swap point is made deterministic: transform A signals a `oneshot` when
//! it handles the first frame; a swap-driver future joined into the run waits
//! for that signal, installs transform B, then releases a `Notify` gate the
//! source awaits before emitting any further frames. So frame 0 is guaranteed
//! to reach A and every later frame to reach B.

use core::future::Future;
use core::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tokio::sync::{oneshot, Notify};

use g2g_core::element::DynAsyncElement;
use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_source_transform_sink, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, ElementSlot, G2gError, MemoryDomain, OutputSink,
    PipelineClock, PipelinePacket, Rate, VideoCodec, RawVideoFormat,
};
use g2g_plugins::fakesink::FakeSink;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn fixed_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(64),
        height: Dim::Fixed(64),
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

/// Counting pass-through transform. Increments `counter` per `DataFrame`,
/// forwards every non-`Eos` packet downstream (the runner forwards `Eos`
/// itself), and fires `first_tx` exactly once on its first data frame.
struct Tap {
    counter: Arc<AtomicU64>,
    first_tx: Option<oneshot::Sender<()>>,
}

impl AsyncElement for Tap {
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
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        let is_data = matches!(packet, PipelinePacket::DataFrame(_));
        let is_eos = matches!(packet, PipelinePacket::Eos);
        Box::pin(async move {
            if is_data {
                self.counter.fetch_add(1, Ordering::SeqCst);
                if let Some(tx) = self.first_tx.take() {
                    let _ = tx.send(());
                }
            }
            if !is_eos {
                out.push(packet).await?;
            }
            Ok(())
        })
    }
}

/// Source that emits `total` frames but parks on `gate` after the first,
/// so the test can install the swap before frames 1.. are produced.
struct GatedSrc {
    caps: Caps,
    total: u64,
    gate: Arc<Notify>,
    configured: bool,
}

impl SourceLoop for GatedSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.caps.clone()))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            assert!(self.configured, "runner must configure source before run");

            out.push(PipelinePacket::DataFrame(make_frame(0))).await?;
            // Block until the test has swapped A -> B.
            self.gate.notified().await;
            for seq in 1..self.total {
                out.push(PipelinePacket::DataFrame(make_frame(seq))).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.total)
        })
    }
}

#[tokio::test]
async fn slot_swap_mid_stream_through_runner() {
    let total = 4;
    let counter_a = Arc::new(AtomicU64::new(0));
    let counter_b = Arc::new(AtomicU64::new(0));
    let caps = fixed_caps();

    let (first_tx, first_rx) = oneshot::channel();
    let gate = Arc::new(Notify::new());

    let transform_a = Tap { counter: counter_a.clone(), first_tx: Some(first_tx) };
    let mut slot = ElementSlot::new(Box::new(transform_a));
    let handle = slot.handle();

    let mut src = GatedSrc { caps: caps.clone(), total, gate: gate.clone(), configured: false };
    let mut snk = FakeSink::new();
    let clock = ZeroClock;

    // Link capacity 1 keeps source and transform lock-step so the swap lands
    // strictly between frame 0 and frame 1.
    let run = run_source_transform_sink(&mut src, &mut slot, &mut snk, &clock, 1);

    let counter_b_driver = counter_b.clone();
    let caps_for_b = caps.clone();
    let driver = async move {
        first_rx.await.expect("transform A must signal on its first frame");
        // Configure B against the live caps before installing it: the slot
        // does not re-run negotiation on swap (DESIGN.md §4.8.2).
        let mut transform_b: Box<dyn DynAsyncElement + Send> =
            Box::new(Tap { counter: counter_b_driver, first_tx: None });
        transform_b.configure_pipeline(&caps_for_b).expect("configure B");
        handle.swap(transform_b);
        gate.notify_one();
    };

    let (res, ()) = tokio::join!(run, driver);
    let stats = res.expect("pipeline must complete across the swap");

    assert_eq!(stats.frames_emitted, total);
    assert_eq!(stats.frames_consumed, total, "every frame must reach the sink");
    assert_eq!(counter_a.load(Ordering::SeqCst), 1, "A handles only frame 0");
    assert_eq!(counter_b.load(Ordering::SeqCst), total - 1, "B handles frames 1..");
    assert_eq!(snk.received(), total);
    assert_eq!(snk.last_sequence(), Some(total - 1));
    assert!(snk.eos_seen(), "sink must observe EOS");
}
