use g2g_core::runtime::{run_simple_pipeline, run_source_transform_sink};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, G2gError, OutputSink, PipelineClock, PipelinePacket,
    VideoFormat,
};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::identity::IdentityTransform;
use g2g_plugins::videotestsrc::VideoTestSrc;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[tokio::test]
async fn videotestsrc_to_fakesink_30_frames_round_trip() {
    let mut src = VideoTestSrc::new(64, 64, 30, 30);
    let mut snk = FakeSink::new();
    let clock = ZeroClock;

    let stats = run_simple_pipeline(&mut src, &mut snk, &clock, 32)
        .await
        .expect("pipeline should complete");

    assert_eq!(stats.frames_emitted, 30, "source must report 30 emitted");
    assert_eq!(stats.frames_consumed, 30, "sink must report 30 consumed");
    assert_eq!(snk.received(), 30, "sink internal counter must match");
    assert_eq!(snk.last_sequence(), Some(29), "sequence must reach 29");
    assert!(snk.eos_seen(), "sink must observe EOS");
}

#[tokio::test]
async fn small_pipeline_with_eos_marker() {
    // Capacity must accommodate `target_frames + 1` (EOS occupies one slot).
    // M1 source uses sync `OutputSink::push`; backpressure-aware send lands in M2/M3.
    let mut src = VideoTestSrc::new(16, 16, 60, 8);
    let mut snk = FakeSink::new();
    let clock = ZeroClock;

    let stats = run_simple_pipeline(&mut src, &mut snk, &clock, 9)
        .await
        .expect("pipeline should complete");

    assert_eq!(stats.frames_consumed, 8);
    assert!(snk.eos_seen());
}

#[tokio::test]
async fn source_identity_sink_3_element_pipeline() {
    let mut src = VideoTestSrc::new(32, 32, 30, 30);
    let mut tx = IdentityTransform::new();
    let mut snk = FakeSink::new();
    let clock = ZeroClock;

    let stats = run_source_transform_sink(&mut src, &mut tx, &mut snk, &clock, 8)
        .await
        .expect("3-element pipeline should complete");

    assert_eq!(stats.frames_emitted, 30);
    assert_eq!(stats.frames_consumed, 30);
    assert_eq!(tx.forwarded(), 30, "transform must see every frame");
    assert_eq!(snk.last_sequence(), Some(29));
    assert!(snk.eos_seen());
}

#[tokio::test]
async fn pipeline_tight_link_uses_backpressure() {
    // M2: async OutputSink::push awaits capacity instead of failing fast.
    // A tight link (capacity 2 for 8 frames + EOS) still completes correctly.
    let mut src = VideoTestSrc::new(16, 16, 60, 8);
    let mut snk = FakeSink::new();
    let clock = ZeroClock;

    let stats = run_simple_pipeline(&mut src, &mut snk, &clock, 2)
        .await
        .expect("backpressure-aware push must drain the pipeline");

    assert_eq!(stats.frames_consumed, 8);
    assert_eq!(snk.last_sequence(), Some(7));
    assert!(snk.eos_seen());
}

/// M16 step 5d regression: a chain `source → format-changing transform
/// → AcceptsAny sink` must pass each element its *per-link* caps. The
/// transform is a boundary that intercepts the source's format and
/// proposes a different output format. Before 5d, the runner clobbered
/// all link slots with the final fixated caps and fed the transform
/// the sink's downstream caps — which a real decoder (FfmpegH264Dec)
/// rejects with `CapsMismatch`. This test exercises the same shape
/// without needing ffmpeg.
#[tokio::test]
async fn format_changing_transform_receives_input_side_caps() {
    use core::cell::Cell;
    use core::future::Future;
    use core::pin::Pin;
    use std::boxed::Box;

    struct FormatBoundary {
        configured_with: Cell<Option<VideoFormat>>,
    }
    impl AsyncElement for FormatBoundary {
        type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;
        fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
            Ok(upstream.clone())
        }
        fn configure_pipeline(&mut self, caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
            // Record whatever format we receive. The test asserts this
            // is the source's format (Rgba8), not the downstream-facing
            // post-decode format.
            if let Caps::Video { format, .. } = caps {
                self.configured_with.set(Some(*format));
            }
            Ok(ConfigureOutcome::Accepted)
        }
        fn process<'a>(
            &'a mut self,
            packet: PipelinePacket,
            out: &'a mut dyn OutputSink,
        ) -> Self::ProcessFuture<'a> {
            Box::pin(async move {
                let _ = out.push(packet).await?;
                Ok(())
            })
        }
        fn is_format_boundary(&self) -> bool {
            true
        }
        fn propose_output_caps(&self, input: &Caps) -> Caps {
            // Swap format: Rgba8 → Nv12, dims preserved.
            match input {
                Caps::Video { width, height, framerate, .. } => Caps::Video {
                    format: VideoFormat::Nv12,
                    width: width.clone(),
                    height: height.clone(),
                    framerate: framerate.clone(),
                },
                other => other.clone(),
            }
        }
        // Default `caps_constraint_as_transform` returns LegacyTransform,
        // which the solver's mixed cascade unpacks via intercept +
        // propose_output. The chain hits the mixed path because
        // FakeSink is AcceptsAny (native).
    }

    let mut src = VideoTestSrc::new(32, 32, 30, 5);
    let mut tx = FormatBoundary { configured_with: Cell::new(None) };
    let mut snk = FakeSink::new();
    let clock = ZeroClock;

    let stats = run_source_transform_sink(&mut src, &mut tx, &mut snk, &clock, 4)
        .await
        .expect("3-element pipeline with format boundary should complete");

    assert_eq!(stats.frames_consumed, 5);
    // The transform must receive the *source* format (Rgba8), not
    // the downstream format (Nv12).
    assert_eq!(
        tx.configured_with.get(),
        Some(VideoFormat::Rgba8),
        "transform should be configured with its input-side caps"
    );
}
