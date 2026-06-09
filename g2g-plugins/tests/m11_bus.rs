//! M11 bus: an element posts out-of-band messages to the application via a
//! `BusHandle` while forwarding the data stream transparently.
//!
//! `BusTap` posts `Custom(seq)` per frame and `Eos` at end; the app drains the
//! bus after the run. Posting during the run and draining after keeps the
//! assertion deterministic (FIFO order, no drops with ample capacity).

use core::future::Future;
use core::pin::Pin;

use g2g_core::runtime::run_source_transform_sink;
use g2g_core::{
    AsyncElement, Bus, BusHandle, BusMessage, Caps, ConfigureOutcome, G2gError, OutputSink,
    PipelineClock, PipelinePacket,
};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::videotestsrc::VideoTestSrc;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Transparent transform that mirrors each frame to the bus as a `Custom`
/// message and posts `Eos` at end of stream. Forwards `DataFrame`/`CapsChanged`
/// downstream; the runner forwards the `Eos` sentinel itself.
struct BusTap {
    bus: BusHandle,
}

impl AsyncElement for BusTap {
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
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(f) => {
                    self.bus.try_post(BusMessage::Custom(f.sequence));
                    out.push(PipelinePacket::DataFrame(f)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    out.push(PipelinePacket::CapsChanged(c)).await?;
                }
                PipelinePacket::Flush => {
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Eos => {
                    self.bus.try_post(BusMessage::Eos);
                }
            }
            Ok(())
        })
    }
}

#[tokio::test]
async fn element_posts_messages_app_drains_bus() {
    let (bus, handle) = Bus::new(16);
    let mut src = VideoTestSrc::new(16, 16, 30, 3);
    let mut tap = BusTap { bus: handle };
    let mut snk = FakeSink::new();
    let clock = ZeroClock;

    let stats = run_source_transform_sink(&mut src, &mut tap, &mut snk, &clock, 8)
        .await
        .expect("pipeline should complete");

    // Data path unaffected by the tap.
    assert_eq!(stats.frames_consumed, 3);
    assert_eq!(snk.received(), 3);
    assert!(snk.eos_seen());

    // Drain the bus: FIFO, one Custom per frame then Eos.
    let mut messages = Vec::new();
    while let Some(m) = bus.try_recv() {
        messages.push(m);
    }
    assert_eq!(
        messages,
        vec![
            BusMessage::Custom(0),
            BusMessage::Custom(1),
            BusMessage::Custom(2),
            BusMessage::Eos,
        ],
    );
}
