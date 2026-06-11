use g2g_core::runtime::{run_simple_pipeline, run_source_transform_sink};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, G2gError, OutputSink, PipelineClock, PipelinePacket,
    VideoCodec, RawVideoFormat,
};
use g2g_plugins::capsfilter::CapsFilter;
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

/// Glass-to-glass latency regression guard. `VideoTestSrc` stamps
/// `arrival_ns` at frame emission, `FakeSink` records
/// `monotonic_ns() - arrival_ns` per frame into a `LatencyHistogram`.
/// For an all-in-memory pipeline through the M16 solver the bound is
/// trivially small — single-digit ms on any machine that isn't
/// overloaded. A regression to >25ms would point to a real
/// degradation (lock contention, blocking I/O, runner serialization).
/// Gated on `std` because the wall-clock stamps live behind g2g-core's
/// `std` feature (`monotonic_ns`).
#[cfg(feature = "std")]
#[tokio::test]
async fn videotestsrc_to_fakesink_latency_under_25ms() {
    let mut src = VideoTestSrc::new(64, 64, 240, 30);
    let mut snk = FakeSink::new();
    let clock = ZeroClock;

    run_simple_pipeline(&mut src, &mut snk, &clock, 32)
        .await
        .expect("pipeline should complete");

    let snap = snk.latency_snapshot();
    assert_eq!(snap.count, 30, "every frame must be timed");
    // Histogram buckets are factor-of-2; pin to a coarse bound that
    // catches order-of-magnitude regressions while tolerating shared
    // CI hardware.
    assert!(
        snap.max_ns < 25_000_000,
        "max latency {}ns exceeded 25ms regression threshold (mean={}, p99={})",
        snap.max_ns,
        snap.mean_ns,
        snap.p99_ns,
    );
    assert!(
        snap.p99_ns < 25_000_000,
        "p99 latency {}ns exceeded 25ms regression threshold",
        snap.p99_ns
    );
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

/// M16 step 5g: chain `LegacySource(H264) → H264Parse (Identity native)
/// → FakeSink (AcceptsAny)` exercises the mixed cascade with a native
/// `Identity` transform in the middle. The parse element doesn't see
/// real H.264 bytes here (no NAL data), it just needs to negotiate and
/// pass through `Eos`.
#[tokio::test]
async fn h264parse_identity_negotiates_in_mixed_chain() {
    use core::future::Future;
    use core::pin::Pin;
    use g2g_core::runtime::SourceLoop;
    use g2g_core::{Dim, Rate};
    use g2g_plugins::h264parse::H264Parse;
    use std::boxed::Box;

    // Minimal H.264-advertising source that emits just an EOS — enough
    // to drive negotiation through the chain without crafting valid
    // SPS bytes. H264Parse's `process` for `Eos` is a pass-through.
    struct H264EosSource {
        configured: bool,
    }
    impl SourceLoop for H264EosSource {
        type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
        type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
        where
            Self: 'a;

        fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
            core::future::ready(Ok(Caps::CompressedVideo {
    codec: VideoCodec::H264,
    width: Dim::Fixed(1280),
    height: Dim::Fixed(720),
    framerate: Rate::Fixed(30 << 16),
}))
        }
        fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
            self.configured = true;
            Ok(ConfigureOutcome::Accepted)
        }
        fn run<'a>(
            &'a mut self,
            out: &'a mut dyn OutputSink,
        ) -> Self::RunFuture<'a> {
            Box::pin(async move {
                out.push(PipelinePacket::Eos).await?;
                Ok(0)
            })
        }
    }

    let mut src = H264EosSource { configured: false };
    let mut parse = H264Parse::new();
    let mut snk = FakeSink::new();
    let clock = ZeroClock;

    let stats = run_source_transform_sink(&mut src, &mut parse, &mut snk, &clock, 4)
        .await
        .expect("negotiation through H264Parse Identity must succeed");

    assert_eq!(stats.frames_consumed, 0);
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
        configured_with: Cell<Option<RawVideoFormat>>,
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
            if let Caps::RawVideo { format, .. } = caps {
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
                Caps::RawVideo { width, height, framerate, .. } => Caps::RawVideo {
                    format: RawVideoFormat::Nv12,
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
        Some(RawVideoFormat::Rgba8),
        "transform should be configured with its input-side caps"
    );
}

/// M16 step 6: `CapsFilter` (native `Identity`) in a real solver chain.
/// `VideoTestSrc` (Produces Rgba8) → `CapsFilter` (Identity, Rgba8/any
/// dims) → `FakeSink` (AcceptsAny) is all-native, so it runs the
/// arc-consistency path. The filter narrows the link to Rgba8 and every
/// frame flows through.
#[tokio::test]
async fn capsfilter_passes_matching_format_in_native_chain() {
    use g2g_core::{Caps, Dim, Rate};

    let mut src = VideoTestSrc::new(32, 32, 30, 12);
    let mut filter = CapsFilter::new(Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    });
    let mut snk = FakeSink::new();
    let clock = ZeroClock;

    let stats = run_source_transform_sink(&mut src, &mut filter, &mut snk, &clock, 8)
        .await
        .expect("native chain with matching CapsFilter must negotiate");

    assert_eq!(stats.frames_consumed, 12);
    assert_eq!(filter.forwarded(), 12, "filter must forward every frame");
    assert_eq!(snk.last_sequence(), Some(11));
    assert!(snk.eos_seen());
}

/// M16 step 6: a `CapsFilter` whose set is disjoint from the source's
/// produced format makes the link empty, so negotiation fails before any
/// frame flows.
#[tokio::test]
async fn capsfilter_rejects_incompatible_format() {
    use g2g_core::{Caps, Dim, Rate};

    let mut src = VideoTestSrc::new(32, 32, 30, 12);
    let mut filter = CapsFilter::new(Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    });
    let mut snk = FakeSink::new();
    let clock = ZeroClock;

    let result = run_source_transform_sink(&mut src, &mut filter, &mut snk, &clock, 8).await;

    assert!(
        result.is_err(),
        "CapsFilter disjoint from the source format must fail negotiation"
    );
    assert_eq!(filter.forwarded(), 0, "no frames should reach a rejected filter");
}
