//! M174 upstream QoS propagation: a sink that runs behind the clock signals QoS
//! upstream; the source observes it as `PushOutcome::Qos` and skips frames to
//! shed load. Here a test sink requests QoS on every frame, and `VideoTestSrc`
//! reacts by skipping ahead, so it emits fewer frames than its target.

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::runtime::run_simple_pipeline;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, G2gError, OutputSink, PipelineClock,
    PipelinePacket, QosMessage,
};
use g2g_plugins::videotestsrc::VideoTestSrc;

const FPS: u32 = 30;
/// One frame period in ns at FPS.
const FRAME_NS: u64 = 1_000_000_000 / FPS as u64;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Sink that consumes frames and, optionally, reports a fixed QoS lateness on
/// each one so the source upstream sheds load.
struct QosSink {
    received: u64,
    /// Lateness to report per frame, or `None` to never signal QoS.
    jitter_ns: Option<i64>,
    pending: Option<QosMessage>,
}

impl QosSink {
    fn behind(jitter_ns: i64) -> Self {
        Self { received: 0, jitter_ns: Some(jitter_ns), pending: None }
    }
    fn on_time() -> Self {
        Self { received: 0, jitter_ns: None, pending: None }
    }
}

impl AsyncElement for QosSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                self.received += 1;
                if let Some(j) = self.jitter_ns {
                    self.pending = Some(QosMessage { jitter_ns: j, running_time_ns: f.timing.pts_ns });
                }
            }
            Ok(())
        })
    }
    fn take_qos(&mut self) -> Option<QosMessage> {
        self.pending.take()
    }
}

#[tokio::test]
async fn source_skips_frames_under_upstream_qos() {
    let target = 30u64;
    let mut src = VideoTestSrc::new(16, 16, FPS, target);
    // Report ~3 frame periods behind on every frame, so the source skips ~3
    // frames each time it observes the signal.
    let mut sink = QosSink::behind(3 * FRAME_NS as i64);

    let stats = run_simple_pipeline(&mut src, &mut sink, &ZeroClock, 1)
        .await
        .expect("pipeline runs");

    assert!(src.skipped() > 0, "source skipped frames under QoS (skipped={})", src.skipped());
    assert!(
        stats.frames_emitted < target,
        "source emitted fewer than target due to skips ({} < {target})",
        stats.frames_emitted
    );
    // The sink consumed exactly what the source pushed (this sink signals QoS but
    // drops nothing of its own).
    assert_eq!(stats.frames_consumed, stats.frames_emitted, "every pushed frame reached the sink");
    assert_eq!(sink.received, stats.frames_emitted);
}

#[tokio::test]
async fn source_emits_every_frame_without_qos() {
    // Control: a sink that never signals QoS leaves the source unchanged.
    let target = 12u64;
    let mut src = VideoTestSrc::new(16, 16, FPS, target);
    let mut sink = QosSink::on_time();

    let stats = run_simple_pipeline(&mut src, &mut sink, &ZeroClock, 4)
        .await
        .expect("pipeline runs");

    assert_eq!(src.skipped(), 0, "no QoS, no skips");
    assert_eq!(stats.frames_emitted, target, "all frames emitted");
    assert_eq!(stats.frames_consumed, target);
}
