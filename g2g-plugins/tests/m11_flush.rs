//! M11 flush: a mid-stream `Flush` packet resets element position so the
//! stream resumes (here, with a restarted sequence) without terminating.
//!
//! `FakeSink` rejects a non-increasing sequence, so the second frame 0 would
//! error were it not for the flush resetting `last_sequence` — the test fails
//! if `Flush` handling regresses.

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_simple_pipeline, SourceLoop};
use g2g_core::{
    Caps, ConfigureOutcome, Dim, G2gError, MemoryDomain, OutputSink, PipelineClock, PipelinePacket,
    Rate, VideoFormat,
};
use g2g_plugins::fakesink::FakeSink;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn caps() -> Caps {
    Caps::Video {
        format: VideoFormat::Rgba8,
        width: Dim::Fixed(16),
        height: Dim::Fixed(16),
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

/// Emits frame 0, a `Flush`, then frame 0 again (a post-seek restart), then EOS.
struct FlushingSrc {
    configured: bool,
}

impl SourceLoop for FlushingSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self) -> Result<Caps, G2gError> {
        Ok(caps())
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            assert!(self.configured, "runner must configure source before run");
            out.push(PipelinePacket::DataFrame(make_frame(0))).await?;
            out.push(PipelinePacket::Flush).await?;
            // Restarted sequence: only valid because Flush reset the sink.
            out.push(PipelinePacket::DataFrame(make_frame(0))).await?;
            out.push(PipelinePacket::Eos).await?;
            Ok(2)
        })
    }
}

#[tokio::test]
async fn flush_resets_sink_position_allowing_sequence_restart() {
    let mut src = FlushingSrc { configured: false };
    let mut snk = FakeSink::new();
    let clock = ZeroClock;

    let stats = run_simple_pipeline(&mut src, &mut snk, &clock, 8)
        .await
        .expect("flush must reset the sink so the restarted frame is accepted");

    assert_eq!(stats.frames_consumed, 2);
    assert_eq!(snk.received(), 2);
    assert_eq!(snk.flushes(), 1, "sink observed one flush");
    assert_eq!(snk.last_sequence(), Some(0), "stream resumed at the restarted sequence");
    assert!(snk.eos_seen(), "flush is non-terminal; EOS still ends the stream");
}
