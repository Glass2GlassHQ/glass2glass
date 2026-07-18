//! M433: turnkey caption authoring from a subtitle file. Drives the real graph
//! `subtitlesrc -> subparse -> ccinsert.cue` + `h264 video -> ccinsert.video ->
//! sink`, then extracts + decodes the captioned output: a `.srt` file becomes
//! embedded CEA-608 captions in the video bitstream, no hand-built cues. Unit under
//! test = `SubtitleSrc` + `SubParse` + `CcInsert`, end to end through the
//! multi-input runner.

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::sync::{Arc, Mutex};

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_graph, GraphNode, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, G2gError, Graph, MemoryDomain,
    OutputSink, PipelineClock, PipelinePacket, Rate, VideoCodec,
};
use g2g_plugins::ccinsert::CcInsert;
use g2g_plugins::cea::{extract_cc_data, Cea608};
use g2g_plugins::subparse::SubParse;
use g2g_plugins::subtitlesrc::SubtitleSrc;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn h264() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(64),
        height: Dim::Fixed(64),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Emits `count` plain Annex-B IDR access units at 30 fps, then Eos. Each AU is a
/// single VCL slice NAL (type 5) with a little payload, the video the captions
/// ride on.
struct H264Src {
    count: u64,
}
impl SourceLoop for H264Src {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(h264()))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let frame_dur = 33_000_000u64;
            for i in 0..self.count {
                let au = [0x00, 0x00, 0x00, 0x01, 0x65, 0x88, 0x84, 0x00];
                let f = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(au.to_vec().into_boxed_slice())),
                    FrameTiming {
                        pts_ns: i * frame_dur,
                        keyframe: true,
                        ..Default::default()
                    },
                    i,
                );
                out.push(PipelinePacket::DataFrame(f)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.count)
        })
    }
}

/// Collects each emitted access unit's bytes.
struct RecSink {
    aus: Arc<Mutex<Vec<Vec<u8>>>>,
}
impl AsyncElement for RecSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;
    fn intercept_caps(&self, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let MemoryDomain::System(s) = &f.domain {
                    self.aus.lock().unwrap().push(s.as_slice().to_vec());
                }
            }
            Ok(())
        })
    }
}

#[tokio::test]
async fn srt_file_authors_embedded_cea608_captions() {
    // A SubRip file with one cue, [1s, 3s) "HELLO".
    let path = std::env::temp_dir().join(format!("g2g_m433_{}.srt", std::process::id()));
    std::fs::write(&path, "1\n00:00:01,000 --> 00:00:03,000\nHELLO\n").unwrap();

    let aus = Arc::new(Mutex::new(Vec::new()));
    let mut g: Graph<GraphNode> = Graph::new();
    let video = g.add_source(GraphNode::source(H264Src { count: 130 })); // ~4.3 s at 30 fps
    let subs = g.add_source(GraphNode::source(SubtitleSrc::from_location(&path)));
    let subparse = g.add_transform(GraphNode::element(SubParse::new()));
    let mux = g.add_muxer(GraphNode::muxer(CcInsert::new()), 2);
    let sink = g.add_sink(GraphNode::element(RecSink { aus: aus.clone() }));

    g.link(video, mux.input(0)).unwrap();
    g.link(subs, subparse).unwrap();
    g.link(subparse, mux.input(1)).unwrap();
    g.link(mux.output(), sink).unwrap();

    let stats = run_graph(g, &NullClock, 4)
        .await
        .expect("authoring graph runs");
    std::fs::remove_file(&path).ok();
    assert_eq!(
        stats.frames_consumed, 130,
        "every video access unit reaches the sink"
    );

    // Extract + decode the captioned output: the SRT cue must come back out.
    let aus = aus.lock().unwrap();
    let mut dec = Cea608::new();
    let frame_dur = 33_000_000u64;
    for (i, au) in aus.iter().enumerate() {
        for t in extract_cc_data(au, VideoCodec::H264) {
            if t.cc_type == 0 {
                dec.push_pair(t.b0, t.b1, i as u64 * frame_dur);
            }
        }
    }
    dec.flush(u64::MAX / 2);
    let cues = dec.take_cues();
    assert_eq!(
        cues.len(),
        1,
        "the authored caption is recovered from the bitstream"
    );
    assert_eq!(cues[0].text, "HELLO");
}
