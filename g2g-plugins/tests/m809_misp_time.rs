//! M809: MISB ST 0604 MISP time stamps embedded in the video bitstream. Drives
//! the real graph `h264/h265 source -> misptimeinsert -> misptimeextract -> sink`
//! through the runner: an absolute microsecond time is written into each access
//! unit's SEI and recovered exactly on the other side, for both codecs. Also
//! pins that the stamped access unit is byte-identical apart from the added SEI,
//! that an existing caption SEI survives, and that a malformed SEI yields no
//! time. Unit under test = `MispTimeInsert` + `MispTimeExtract` + the ST 0604
//! codec.

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
use g2g_plugins::cea::{build_cc_sei, extract_cc_data, CcTriple};
use g2g_plugins::misptime::{
    build_misp_time_sei, extract_misp_time, MispTimeExtract, MispTimeInsert, MISP_STATUS_DEFAULT,
};

/// PTS 0 maps here: 2023-11-14 22:13:20 UTC.
const EPOCH_OFFSET_US: u64 = 1_700_000_000_000_000;
const FRAME_DUR_NS: u64 = 33_000_000;
const FRAMES: u64 = 5;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn caps(codec: VideoCodec) -> Caps {
    Caps::CompressedVideo {
        codec,
        width: Dim::Fixed(64),
        height: Dim::Fixed(64),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// A plain Annex-B access unit: one VCL IDR slice NAL (H.264 type 5 / H.265
/// type 19) with a little payload, no SEI.
fn plain_au(codec: VideoCodec) -> Vec<u8> {
    let mut au = vec![0x00, 0x00, 0x00, 0x01];
    match codec {
        VideoCodec::H265 => au.extend_from_slice(&[0x26, 0x01]), // IDR_W_RADL
        _ => au.push(0x65),                                      // IDR slice
    }
    au.extend_from_slice(&[0x88, 0x84, 0x00]);
    au
}

/// Emits `aus` one per 33 ms frame, then Eos.
struct AuSrc {
    codec: VideoCodec,
    aus: Vec<Vec<u8>>,
}

impl SourceLoop for AuSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(caps(self.codec)))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let n = self.aus.len() as u64;
            for (i, au) in self.aus.iter().enumerate() {
                let f = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(au.clone().into_boxed_slice())),
                    FrameTiming {
                        pts_ns: i as u64 * FRAME_DUR_NS,
                        keyframe: true,
                        ..Default::default()
                    },
                    i as u64,
                );
                out.push(PipelinePacket::DataFrame(f)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(n)
        })
    }
}

/// Collects each emitted frame's payload bytes.
struct RecSink {
    frames: Arc<Mutex<Vec<Vec<u8>>>>,
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
                    self.frames.lock().unwrap().push(s.to_vec());
                }
            }
            Ok(())
        })
    }
}

/// Run `source AUs -> misptimeinsert -> misptimeextract -> sink` and return the
/// text lines the extractor emitted.
async fn run_round_trip(codec: VideoCodec, aus: Vec<Vec<u8>>) -> Vec<String> {
    let lines = Arc::new(Mutex::new(Vec::new()));
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(AuSrc {
        codec,
        aus: aus.clone(),
    }));
    let insert = g.add_transform(GraphNode::element(
        MispTimeInsert::new().with_epoch_offset_us(EPOCH_OFFSET_US),
    ));
    let extract = g.add_transform(GraphNode::element(MispTimeExtract::new()));
    let sink = g.add_sink(GraphNode::element(RecSink {
        frames: lines.clone(),
    }));
    g.link(src, insert).unwrap();
    g.link(insert, extract).unwrap();
    g.link(extract, sink).unwrap();

    run_graph(g, &NullClock, 4).await.expect("graph runs");
    let out = lines.lock().unwrap();
    out.iter()
        .map(|b| String::from_utf8(b.clone()).expect("utf-8 cue"))
        .collect()
}

#[test]
fn both_elements_are_launch_registered() {
    let names = g2g_plugins::registry::default_registry().element_names();
    assert!(names.contains(&"misptimeinsert"));
    assert!(names.contains(&"misptimeextract"));
}

/// Run `source AUs -> misptimeinsert -> sink` and return the stamped bitstreams.
async fn run_insert(codec: VideoCodec, aus: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    let stamped = Arc::new(Mutex::new(Vec::new()));
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(AuSrc { codec, aus }));
    let insert = g.add_transform(GraphNode::element(
        MispTimeInsert::new().with_epoch_offset_us(EPOCH_OFFSET_US),
    ));
    let sink = g.add_sink(GraphNode::element(RecSink {
        frames: stamped.clone(),
    }));
    g.link(src, insert).unwrap();
    g.link(insert, sink).unwrap();

    run_graph(g, &NullClock, 4).await.expect("graph runs");
    let out = stamped.lock().unwrap();
    out.clone()
}

/// The expected MISP microsecond time for frame `i`.
fn expected_us(i: u64) -> u64 {
    i * FRAME_DUR_NS / 1_000 + EPOCH_OFFSET_US
}

#[tokio::test]
async fn h264_stamps_round_trip_through_the_runner() {
    let aus = (0..FRAMES).map(|_| plain_au(VideoCodec::H264)).collect();
    let lines = run_round_trip(VideoCodec::H264, aus).await;
    let want: Vec<String> = (0..FRAMES)
        .map(|i| format!("ts={}", expected_us(i)))
        .collect();
    assert_eq!(lines, want, "every access unit's stamp is recovered");
}

#[tokio::test]
async fn h265_stamps_round_trip_through_the_runner() {
    let aus = (0..FRAMES).map(|_| plain_au(VideoCodec::H265)).collect();
    let lines = run_round_trip(VideoCodec::H265, aus).await;
    let want: Vec<String> = (0..FRAMES)
        .map(|i| format!("ts={}", expected_us(i)))
        .collect();
    assert_eq!(lines, want, "every access unit's stamp is recovered");
}

#[tokio::test]
async fn the_stamped_access_unit_is_unchanged_apart_from_the_sei() {
    for codec in [VideoCodec::H264, VideoCodec::H265] {
        let au = plain_au(codec);
        let stamped = run_insert(codec, vec![au.clone()]).await;
        assert_eq!(stamped.len(), 1);
        let sei = build_misp_time_sei(expected_us(0), MISP_STATUS_DEFAULT, codec);
        let at = stamped[0]
            .windows(sei.len())
            .position(|w| w == sei)
            .expect("the MISP SEI NAL is present verbatim");
        // The SEI precedes the VCL slice, and removing it restores the input.
        assert_eq!(at, 0, "{codec:?}: the SEI leads the access unit");
        assert_eq!(&stamped[0][sei.len()..], &au[..], "{codec:?}");
    }
}

#[tokio::test]
async fn an_existing_caption_sei_survives_stamping() {
    // An access unit that already carries a CEA-608 caption SEI keeps it, and
    // gains the MISP stamp alongside.
    let triple = CcTriple {
        cc_type: 0,
        b0: b'H',
        b1: b'I',
    };
    let mut au = build_cc_sei(&[triple], VideoCodec::H264);
    au.extend_from_slice(&plain_au(VideoCodec::H264));

    let stamped = run_insert(VideoCodec::H264, vec![au.clone()]).await;
    assert_eq!(
        extract_cc_data(&stamped[0], VideoCodec::H264),
        vec![triple],
        "the caption SEI is untouched"
    );
    let t = extract_misp_time(&stamped[0], VideoCodec::H264).expect("stamped");
    assert_eq!(t.micros(), expected_us(0));

    // The only change is the inserted SEI NAL.
    let sei = build_misp_time_sei(expected_us(0), MISP_STATUS_DEFAULT, VideoCodec::H264);
    let at = stamped[0]
        .windows(sei.len())
        .position(|w| w == sei)
        .expect("the MISP SEI NAL is present verbatim");
    let mut stripped = stamped[0][..at].to_vec();
    stripped.extend_from_slice(&stamped[0][at + sei.len()..]);
    assert_eq!(stripped, au);
}

#[tokio::test]
async fn malformed_stamps_yield_no_time_and_no_cues() {
    // Access units carrying a truncated MISP SEI, a MISP SEI whose separator
    // bytes are wrong, and no SEI at all: the extractor emits nothing and the
    // graph completes.
    let good = build_misp_time_sei(EPOCH_OFFSET_US, MISP_STATUS_DEFAULT, VideoCodec::H264);

    let mut truncated = good[..good.len() - 6].to_vec();
    truncated.extend_from_slice(&plain_au(VideoCodec::H264));

    let mut corrupt = good.clone();
    let sep = corrupt.iter().position(|&b| b == 0xFF).expect("separator");
    corrupt[sep] = 0x00;
    corrupt.extend_from_slice(&plain_au(VideoCodec::H264));

    for au in [truncated, corrupt, plain_au(VideoCodec::H264)] {
        assert!(extract_misp_time(&au, VideoCodec::H264).is_none());
    }

    let lines = run_round_trip_extract_only(vec![
        good[..good.len() - 6].to_vec(),
        plain_au(VideoCodec::H264),
    ])
    .await;
    assert!(lines.is_empty(), "no cue from a malformed or absent stamp");
}

/// Run `source AUs -> misptimeextract -> sink` (no inserter) and return the text
/// lines, so a hand-built malformed bitstream reaches the extractor untouched.
async fn run_round_trip_extract_only(aus: Vec<Vec<u8>>) -> Vec<String> {
    let lines = Arc::new(Mutex::new(Vec::new()));
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(AuSrc {
        codec: VideoCodec::H264,
        aus,
    }));
    let extract = g.add_transform(GraphNode::element(MispTimeExtract::new()));
    let sink = g.add_sink(GraphNode::element(RecSink {
        frames: lines.clone(),
    }));
    g.link(src, extract).unwrap();
    g.link(extract, sink).unwrap();

    run_graph(g, &NullClock, 4).await.expect("graph runs");
    let out = lines.lock().unwrap();
    out.iter()
        .map(|b| String::from_utf8(b.clone()).expect("utf-8 cue"))
        .collect()
}
