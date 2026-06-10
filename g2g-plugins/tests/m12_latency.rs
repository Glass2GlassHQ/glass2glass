//! M12: pipeline latency query.
//!
//! Each element reports a `LatencyReport`; the linear runners fold the chain
//! (source → transform → sink) into `RunStats::latency` after negotiation,
//! the way GStreamer answers a `LATENCY` query from sink to source. These
//! tests drive the real runners with elements that override `latency()` and
//! assert the aggregate the runner computed.

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_simple_pipeline, run_source_transform_sink, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, G2gError, LatencyReport, MemoryDomain, OutputSink,
    PipelineClock, PipelinePacket, Rate, VideoFormat,
};

const MS: u64 = 1_000_000;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn caps() -> Caps {
    Caps::Video {
        format: VideoFormat::Rgba8,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn make_frame(seq: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
        caps: caps(),
        timing: FrameTiming::default(),
        sequence: seq,
    }
}

/// Source that emits `frames` DataFrames then Eos and reports a fixed latency.
struct LatencySrc {
    frames: u64,
    latency: LatencyReport,
}

impl SourceLoop for LatencySrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self) -> Result<Caps, G2gError> {
        Ok(caps())
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for seq in 0..self.frames {
                out.push(PipelinePacket::DataFrame(make_frame(seq))).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.frames)
        })
    }

    fn latency(&self) -> LatencyReport {
        self.latency
    }
}

/// Identity transform that reports a fixed latency contribution.
struct LatencyTransform {
    latency: LatencyReport,
}

impl AsyncElement for LatencyTransform {
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
                PipelinePacket::Eos => Ok(()),
                other => {
                    out.push(other).await?;
                    Ok(())
                }
            }
        })
    }

    fn latency(&self) -> LatencyReport {
        self.latency
    }
}

/// Terminal sink, zero latency (the measurement point), counts nothing.
struct PlainSink;

impl AsyncElement for PlainSink {
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
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

#[tokio::test]
async fn three_stage_path_sums_min_and_max() {
    let mut src = LatencySrc {
        frames: 3,
        latency: LatencyReport::live(10 * MS, Some(40 * MS)),
    };
    let mut tx = LatencyTransform {
        latency: LatencyReport::buffered(5 * MS, Some(15 * MS)),
    };
    let mut sink = PlainSink;
    let clock = ZeroClock;

    let stats = run_source_transform_sink(&mut src, &mut tx, &mut sink, &clock, 4)
        .await
        .expect("pipeline should complete");

    assert_eq!(stats.frames_consumed, 3);
    assert!(stats.latency.live, "a live source makes the path live");
    assert_eq!(stats.latency.min_ns, 15 * MS, "min latencies sum");
    assert_eq!(stats.latency.max_ns, Some(55 * MS), "max latencies sum");
    assert!(!stats.latency.is_unsatisfiable());
}

#[tokio::test]
async fn unbounded_transform_max_propagates() {
    // A transform with no max ceiling makes the whole path's max unbounded.
    let mut src = LatencySrc {
        frames: 1,
        latency: LatencyReport::live(8 * MS, Some(40 * MS)),
    };
    let mut tx = LatencyTransform {
        latency: LatencyReport::buffered(2 * MS, None),
    };
    let mut sink = PlainSink;
    let clock = ZeroClock;

    let stats = run_source_transform_sink(&mut src, &mut tx, &mut sink, &clock, 4)
        .await
        .expect("pipeline should complete");

    assert_eq!(stats.latency.min_ns, 10 * MS);
    assert_eq!(stats.latency.max_ns, None);
}

#[tokio::test]
async fn source_sink_path_reports_source_latency() {
    let mut src = LatencySrc {
        frames: 2,
        latency: LatencyReport::live(8 * MS, None),
    };
    let mut sink = PlainSink;
    let clock = ZeroClock;

    let stats = run_simple_pipeline(&mut src, &mut sink, &clock, 4)
        .await
        .expect("pipeline should complete");

    assert!(stats.latency.live);
    assert_eq!(stats.latency.min_ns, 8 * MS);
    assert_eq!(stats.latency.max_ns, None);
}

#[tokio::test]
async fn elements_without_override_report_zero_latency() {
    // A non-live source (default latency) and a default sink: ZERO path.
    let mut src = LatencySrc { frames: 1, latency: LatencyReport::ZERO };
    let mut sink = PlainSink;
    let clock = ZeroClock;

    let stats = run_simple_pipeline(&mut src, &mut sink, &clock, 4)
        .await
        .expect("pipeline should complete");

    assert_eq!(stats.latency, LatencyReport::ZERO);
}
