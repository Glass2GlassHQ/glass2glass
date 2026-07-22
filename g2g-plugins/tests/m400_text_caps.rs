//! M400: `Caps::Text` flows through the pipeline as a first-class media kind.
//!
//! Text generalizes "subtitles": a `Caps::Text` link carries any timed-or-untimed
//! text payload, "subtitle" being just timed `Text` (cue PTS + duration). This
//! drives a real `source(Text{Srt}) -> SubParse -> sink(Text{Utf8})` chain through
//! the linear runner and asserts the parser's decoder-style negotiation
//! (`Text{Srt}` in, `Text{Utf8}` derived out) resolves and timed cue frames reach
//! the sink. The unit under test is the caps algebra + `SubParse`, exercised end
//! to end (no mocking of the runner or the element).

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_source_transform_sink, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, G2gError, MemoryDomain, OutputSink, PipelineClock,
    PipelinePacket, TextFormat,
};
use g2g_plugins::subparse::SubParse;

const DOC: &str = "1\n00:00:01,000 --> 00:00:04,000\nHello world\n\n\
                   2\n00:01:02,500 --> 00:01:05,000\nSecond cue\n";

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Emits the SRT document as one `Text{Srt}` DataFrame, then Eos.
struct SrtSrc {
    sent: bool,
}

impl SourceLoop for SrtSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(Caps::Text {
            format: TextFormat::Srt,
        }))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            self.sent = true;
            let bytes = DOC.as_bytes().to_vec().into_boxed_slice();
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(bytes)),
                FrameTiming::default(),
                0,
            );
            out.push(PipelinePacket::DataFrame(frame)).await?;
            out.push(PipelinePacket::Eos).await?;
            Ok(1)
        })
    }
}

/// Sink accepting `Text{Utf8}`, recording each cue's PTS and text.
#[derive(Default)]
struct CueSink {
    pts: Vec<u64>,
    texts: Vec<String>,
}

impl AsyncElement for CueSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::Text {
                format: TextFormat::Utf8,
            } => Ok(upstream_caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::Text {
                format: TextFormat::Utf8,
            } => Ok(ConfigureOutcome::Accepted),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(frame) = packet {
                if let Some(slice) = frame.domain.as_system_slice() {
                    self.pts.push(frame.timing.pts_ns);
                    self.texts.push(String::from_utf8_lossy(slice).into_owned());
                }
            }
            Ok(())
        })
    }
}

#[tokio::test]
async fn srt_source_subparse_text_sink_runs_end_to_end() {
    let mut src = SrtSrc { sent: false };
    let mut sub = SubParse::new();
    let mut sink = CueSink::default();
    let clock = ZeroClock;

    let stats = run_source_transform_sink(&mut src, &mut sub, &mut sink, &clock, 4)
        .await
        .expect("Text{Srt} -> SubParse -> Text{Utf8} negotiates and runs");

    // The two cues reached the sink as timed Text{Utf8} frames.
    assert_eq!(stats.frames_consumed, 2, "two cue frames consumed");
    assert_eq!(sink.pts, [1_000_000_000, 62_500_000_000]);
    assert_eq!(sink.texts, ["Hello world", "Second cue"]);
}
