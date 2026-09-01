//! M917: closed captions survive a transcode as frame metadata. `H264Parse`
//! attaches each access unit's `cc_data` as a `CaptionMeta`; the runner carries
//! that meta across an element declaring `Transform::Encode` (which throws the
//! bitstream, and with it the caption SEI, away); `CcInsert::from_meta` writes it
//! back into the new access units.
//!
//! Unit under test = `NalParse`'s caption producer + `CaptionMeta::propagate` +
//! `CcInsert`'s meta-sourced mode, end to end through the real runner. The mock
//! encoder emits a caption-free access unit with an empty meta set, so a green
//! run proves the metadata (not the bitstream) carried the captions.

#![cfg(all(feature = "std", feature = "metadata"))]

use core::future::Future;
use core::pin::Pin;
use std::sync::{Arc, Mutex};

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::meta::Transform;
use g2g_core::runtime::{run_graph, GraphNode, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, G2gError, Graph, MemoryDomain,
    OutputSink, PipelineClock, PipelinePacket, Rate, VideoCodec,
};
use g2g_plugins::ccinsert::CcInsert;
use g2g_plugins::cea::{build_cc_sei, extract_cc_data, Cc608Enc, CcTriple, Cea608};
use g2g_plugins::h264parse::H264Parse;
use g2g_plugins::subparse::Cue;

const FRAME_DUR: u64 = 33_000_000;
/// A minimal Annex-B access unit: one VCL IDR slice NAL (type 5), no captions.
const PLAIN_AU: [u8; 8] = [0x00, 0x00, 0x00, 0x01, 0x65, 0x88, 0x84, 0x00];

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
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

/// Emits `count` Annex-B IDR access units at 30 fps. With `captioned`, each one
/// carries a `GA94` caption SEI holding that frame's CEA-608 byte pair for the
/// cue "HELLO", the way a broadcast source would.
struct CaptionedSrc {
    count: u64,
    captioned: bool,
}

impl SourceLoop for CaptionedSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(h264()))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let mut enc = Cc608Enc::new();
            enc.push_cue(&Cue {
                start_ns: 0,
                end_ns: 40 * FRAME_DUR,
                text: String::from("HELLO"),
                settings: Default::default(),
            });
            for i in 0..self.count {
                let mut au = Vec::new();
                if self.captioned {
                    let (b0, b1) = enc.next_pair();
                    au.extend_from_slice(&build_cc_sei(
                        &[CcTriple { cc_type: 0, b0, b1 }],
                        VideoCodec::H264,
                    ));
                }
                au.extend_from_slice(&PLAIN_AU);
                let f = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(au.into_boxed_slice())),
                    FrameTiming {
                        pts_ns: i * FRAME_DUR,
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

/// Stands in for a decode + re-encode: emits a brand-new, caption-free access
/// unit with an empty meta set, declaring `Transform::Encode` so the runner
/// applies the metadata propagation contract. Nothing of the input bitstream
/// (and so nothing of its caption SEI) reaches the output.
struct MockEncode;

impl AsyncElement for MockEncode {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn meta_transform(&self) -> Option<Transform> {
        Some(Transform::Encode)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(input) => {
                    let fresh = Frame::new(
                        MemoryDomain::System(SystemSlice::from_boxed(Box::new(PLAIN_AU))),
                        input.timing,
                        input.sequence,
                    );
                    out.push(PipelinePacket::DataFrame(fresh)).await?;
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
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
                if let Some(s) = f.domain.as_system_slice() {
                    self.aus.lock().unwrap().push(s.to_vec());
                }
            }
            Ok(())
        })
    }
}

/// Run `src -> h264parse -> mock encode -> ccinsert(from meta) -> sink` and
/// return the access units the sink saw.
async fn run_transcode(count: u64, captioned: bool) -> Vec<Vec<u8>> {
    let aus = Arc::new(Mutex::new(Vec::new()));
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(CaptionedSrc { count, captioned }));
    let parse = g.add_transform(GraphNode::element(H264Parse::new()));
    let encode = g.add_transform(GraphNode::element(MockEncode));
    let insert = g.add_muxer(GraphNode::muxer(CcInsert::from_meta()), 1);
    let sink = g.add_sink(GraphNode::element(RecSink { aus: aus.clone() }));

    g.link(src, parse).unwrap();
    g.link(parse, encode).unwrap();
    g.link(encode, insert.input(0)).unwrap();
    g.link(insert.output(), sink).unwrap();

    run_graph(g, &NullClock, 4).await.expect("transcode runs");
    let out = aus.lock().unwrap().clone();
    out
}

#[tokio::test]
async fn captions_survive_a_re_encode_as_frame_metadata() {
    let aus = run_transcode(40, true).await;
    assert_eq!(aus.len(), 40, "every access unit reaches the sink");

    // The re-encoded bitstream carries the source's caption bytes again, and they
    // decode back to the original cue.
    let mut dec = Cea608::new();
    for (i, au) in aus.iter().enumerate() {
        for t in extract_cc_data(au, VideoCodec::H264) {
            if t.cc_type == 0 {
                dec.push_pair(t.b0, t.b1, i as u64 * FRAME_DUR);
            }
        }
    }
    dec.flush(u64::MAX / 2);
    let cues = dec.take_cues();
    assert_eq!(cues.len(), 1, "the caption survived the transcode");
    assert_eq!(cues[0].text, "HELLO");
}

#[tokio::test]
async fn a_caption_free_stream_is_left_untouched() {
    // No CaptionMeta on the frames means no SEI is written: the access units come
    // out exactly as the encoder produced them.
    let aus = run_transcode(5, false).await;
    assert_eq!(aus.len(), 5);
    for au in &aus {
        assert_eq!(au.as_slice(), &PLAIN_AU, "no caption SEI was inserted");
    }
}
