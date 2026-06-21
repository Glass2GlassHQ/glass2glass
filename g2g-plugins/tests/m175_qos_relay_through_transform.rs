//! M175 QoS relay through a transform: a sink that runs behind the clock signals
//! QoS, and the report must reach the source *through* an intervening transform
//! so multi-element pipelines shed load at the source, not just one hop up. M174
//! relayed only across a single source->sink link; here an `IdentityTransform`
//! sits between `VideoTestSrc` and a QoS-signalling sink, and the source must
//! still skip frames. Covers both the bespoke `run_source_transform_sink` runner
//! and the `run_linear_chain` builder over the DAG runner (`run_graph`).

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::element::DynAsyncElement;
use g2g_core::runtime::{run_linear_chain, run_source_transform_sink};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, G2gError, OutputSink, PipelineClock,
    PipelinePacket, QosMessage,
};
use g2g_plugins::identity::IdentityTransform;
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

/// Sink that consumes frames and reports a fixed QoS lateness on each one so the
/// source upstream sheds load.
struct QosSink {
    received: u64,
    jitter_ns: i64,
    pending: Option<QosMessage>,
}

impl QosSink {
    fn behind(jitter_ns: i64) -> Self {
        Self { received: 0, jitter_ns, pending: None }
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
                self.pending =
                    Some(QosMessage { jitter_ns: self.jitter_ns, running_time_ns: f.timing.pts_ns });
            }
            Ok(())
        })
    }
    fn take_qos(&mut self) -> Option<QosMessage> {
        self.pending.take()
    }
}

#[tokio::test]
async fn source_skips_through_transform_bespoke_runner() {
    let target = 30u64;
    let mut src = VideoTestSrc::new(16, 16, FPS, target);
    let mut transform = IdentityTransform::new();
    // ~3 frame periods behind on every frame, so the source skips when it
    // observes the relayed signal.
    let mut sink = QosSink::behind(3 * FRAME_NS as i64);

    let stats = run_source_transform_sink(&mut src, &mut transform, &mut sink, &ZeroClock, 1)
        .await
        .expect("pipeline runs");

    assert!(
        src.skipped() > 0,
        "source skipped frames under QoS relayed through the transform (skipped={})",
        src.skipped()
    );
    assert!(
        stats.frames_emitted < target,
        "source emitted fewer than target due to relayed skips ({} < {target})",
        stats.frames_emitted
    );
    assert_eq!(sink.received, stats.frames_consumed, "every consumed frame reached the sink");
}

#[tokio::test]
async fn source_skips_through_transform_graph_runner() {
    let target = 30u64;
    let mut src = VideoTestSrc::new(16, 16, FPS, target);
    let mut transform = IdentityTransform::new();
    let mut sink = QosSink::behind(3 * FRAME_NS as i64);

    let transforms: Vec<&mut dyn DynAsyncElement> = vec![&mut transform];
    let stats = run_linear_chain(&mut src, transforms, &mut sink, &ZeroClock, 1)
        .await
        .expect("pipeline runs");

    assert!(
        src.skipped() > 0,
        "source skipped frames under QoS relayed through run_graph (skipped={})",
        src.skipped()
    );
    assert!(
        stats.frames_emitted < target,
        "source emitted fewer than target via the DAG runner relay ({} < {target})",
        stats.frames_emitted
    );
}
