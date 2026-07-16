//! M403: `TextOverlayN` overlays a *streamed* subtitle track onto video.
//!
//! Drives the real fan-in graph `video -> textoverlayn.video` +
//! `srt -> subparse -> textoverlayn.text` -> sink, proving a `Caps::Text` stream
//! (parsed by `SubParse`) paints onto video through the multi-input runner: the
//! cue covers a PTS window, so frames inside it come out painted and frames
//! outside it untouched. The PTS-merge delivers each cue before the video frame
//! it covers. Unit under test = `TextOverlayN` + `SubParse`, end to end.

use core::future::Future;
use core::pin::Pin;
use std::sync::{Arc, Mutex};

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_graph, GraphNode, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, G2gError, Graph, MemoryDomain,
    OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat, TextFormat,
};
use g2g_plugins::subparse::SubParse;
use g2g_plugins::textoverlay::TextOverlayN;

const W: u32 = 64;
const H: u32 = 64;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn rgba(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Emits black RGBA8 frames at the given PTS values, then Eos.
struct BlackVideoSrc {
    pts: Vec<u64>,
}

impl SourceLoop for BlackVideoSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(rgba(W, H)))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let n = self.pts.len() as u64;
            for &pts in &self.pts {
                let buf = [0u8, 0, 0, 255].repeat((W * H) as usize).into_boxed_slice();
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(buf)),
                    FrameTiming { pts_ns: pts, ..FrameTiming::default() },
                    0,
                );
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(n)
        })
    }
}

/// Emits one SRT document as a `Text{Srt}` frame, then Eos.
struct SrtTextSrc;

impl SourceLoop for SrtTextSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(Caps::Text { format: TextFormat::Srt }))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            // One cue covering [1s, 3s).
            let doc = b"1\n00:00:01,000 --> 00:00:03,000\nHELLO\n";
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(doc.to_vec().into_boxed_slice())),
                FrameTiming::default(),
                0,
            );
            out.push(PipelinePacket::DataFrame(frame)).await?;
            out.push(PipelinePacket::Eos).await?;
            Ok(1)
        })
    }
}

/// Records each received frame's `(pts, painted?)` into shared state.
struct RecSink {
    log: Arc<Mutex<Vec<(u64, bool)>>>,
}

impl AsyncElement for RecSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
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
            if let PipelinePacket::DataFrame(frame) = packet {
                if let MemoryDomain::System(slice) = &frame.domain {
                    let buf = slice.as_slice();
                    let painted = (0..(W * H) as usize)
                        .any(|i| buf[i * 4] != 0 || buf[i * 4 + 1] != 0 || buf[i * 4 + 2] != 0);
                    self.log.lock().unwrap().push((frame.timing.pts_ns, painted));
                }
            }
            Ok(())
        })
    }
}

#[tokio::test]
async fn streamed_srt_track_paints_video_in_the_cue_window() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut g: Graph<GraphNode> = Graph::new();

    // Video frames straddling the cue window [1s, 3s): 0 and 4s outside, 1.5s and
    // 2.5s inside.
    let video = g.add_source(GraphNode::source(BlackVideoSrc {
        pts: vec![0, 1_500_000_000, 2_500_000_000, 4_000_000_000],
    }));
    let srt = g.add_source(GraphNode::source(SrtTextSrc));
    let subparse = g.add_transform(GraphNode::element(SubParse::new()));
    let mux = g.add_muxer(GraphNode::muxer(TextOverlayN::new()), 2);
    let sink = g.add_sink(GraphNode::element(RecSink { log: log.clone() }));

    g.link(video, mux.input(0)).unwrap();
    g.link(srt, subparse).unwrap();
    g.link(subparse, mux.input(1)).unwrap();
    g.link(mux.output(), sink).unwrap();

    let stats = run_graph(g, &NullClock, 4).await.expect("overlay graph runs");
    assert_eq!(stats.frames_consumed, 4, "every video frame reaches the sink");

    let log = log.lock().unwrap();
    assert_eq!(log.len(), 4);
    for &(pts, painted) in log.iter() {
        let in_window = (1_000_000_000..3_000_000_000).contains(&pts);
        assert_eq!(
            painted, in_window,
            "frame at {pts} ns: painted={painted}, expected in_window={in_window}"
        );
    }
}
