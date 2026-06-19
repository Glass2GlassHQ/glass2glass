//! M83 (auto-plug, slice a): the runtime element [`Registry`] + decode-chain
//! search, exercised over real plugin elements and run end-to-end onto
//! [`run_graph`].
//!
//! Two angles:
//! - **Search over real metadata.** A registry of the real `H264Parse` plus a
//!   decoder descriptor finds the H.264 -> raw chain by name, and a
//!   parser-only registry correctly reports no route to raw.
//! - **Instantiate + run.** A registry containing `VideoConvert` auto-plugs an
//!   RGBA source's caps toward NV12, and the returned boxed element splices
//!   between a real source and sink and flows frames through `run_graph` (the
//!   "decodebin returns a sub-graph" payoff, for a converter chain).

use g2g_core::runtime::{
    is_raw_video, run_graph, ElementFactory, GraphNode, GraphNodeRef, Registry, RunStats,
};
use g2g_core::{
    Caps, CapsSet, Dim, Graph, PadTemplate, PipelineClock, Rate, RawVideoFormat, VideoCodec,
};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::h264parse::H264Parse;
use g2g_plugins::identity::IdentityTransform;
use g2g_plugins::videoconvert::VideoConvert;
use g2g_plugins::videotestsrc::VideoTestSrc;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn h264(width: Dim) -> Caps {
    Caps::CompressedVideo { codec: VideoCodec::H264, width, height: Dim::Any, framerate: Rate::Any }
}

/// A decoder descriptor: H.264 in, raw NV12 out. The body is an
/// `IdentityTransform` stand-in; this registry is used only for the by-name
/// search (it is never run), so the templates are what matter.
fn decoder_factory() -> ElementFactory {
    let templates = Vec::from([
        PadTemplate::sink(CapsSet::one(h264(Dim::Any))),
        PadTemplate::source(CapsSet::one(Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        })),
    ]);
    ElementFactory::new("h264dec", templates, |_caps| Box::new(IdentityTransform::new()))
}

/// A converter factory that derives its target format from the output caps the
/// search chose for this hop (the M83b caps-aware builder), rather than baking
/// a fixed format into the closure.
fn videoconvert_factory() -> ElementFactory {
    ElementFactory::of::<VideoConvert>("videoconvert", |out| match out {
        Caps::RawVideo { format, .. } => Box::new(VideoConvert::new(*format)),
        _ => unreachable!("autoplug only routes raw caps into videoconvert"),
    })
}

#[test]
fn registry_finds_decoder_for_h264_to_raw() {
    let mut reg = Registry::new();
    reg.register(ElementFactory::of::<H264Parse>("h264parse", |_caps| Box::new(H264Parse::new())))
        .register(decoder_factory());

    let chain = reg
        .autoplug_names(&h264(Dim::Fixed(1280)), &is_raw_video, 4)
        .expect("a decoder bridges H.264 to raw video");
    // Shortest route to raw is the decoder alone; the same-shape parser never
    // shortens it.
    assert_eq!(chain, Vec::from(["h264dec"]));
}

#[test]
fn registry_reports_no_route_when_only_a_parser_is_registered() {
    let mut reg = Registry::new();
    reg.register(ElementFactory::of::<H264Parse>("h264parse", |_caps| Box::new(H264Parse::new())));
    assert!(
        reg.autoplug_names(&h264(Dim::Any), &is_raw_video, 8).is_none(),
        "H.264 -> H.264 parsing alone never reaches raw video"
    );
}

fn rgba_any() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

fn is_nv12(c: &Caps) -> bool {
    matches!(c, Caps::RawVideo { format: RawVideoFormat::Nv12, .. })
}

#[tokio::test]
async fn autoplugged_converter_chain_runs_through_run_graph() {
    // A registry whose one element converts raw video. Auto-plug from the RGBA
    // caps a VideoTestSrc produces toward an NV12 target; the converter is built
    // from the caps the search chose, not a hard-coded format.
    let mut reg = Registry::new();
    reg.register(videoconvert_factory());

    let chain = reg.autoplug(&rgba_any(), &is_nv12, 4).expect("videoconvert reaches NV12");
    assert_eq!(chain.len(), 1, "one converter splices RGBA -> NV12");

    // Splice the instantiated chain between a real source and sink as a
    // sub-graph of transforms, then run it.
    let mut g: Graph<GraphNodeRef<'static>> = Graph::new();
    let src = g.add_source(GraphNodeRef::source(VideoTestSrc::new(8, 8, 30, 4)));
    let mut prev = src;
    for boxed in chain {
        let node = g.add_transform(GraphNodeRef::Element(boxed));
        g.link(prev, node).unwrap();
        prev = node;
    }
    let sink = g.add_sink(GraphNodeRef::element(FakeSink::new()));
    g.link(prev, sink).unwrap();

    let stats: RunStats = run_graph(g, &NullClock, 4).await.expect("autoplugged chain runs");
    assert_eq!(stats.frames_emitted, 4);
    assert_eq!(stats.frames_consumed, 4);
}

#[tokio::test]
async fn decodebin_splices_chain_between_source_and_sink() {
    // The decodebin convenience: the caller builds only the source and sink and
    // names the input caps + target; the registry fills the middle and links it.
    let mut reg = Registry::new();
    reg.register(videoconvert_factory());

    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(VideoTestSrc::new(8, 8, 30, 4)));
    let sink = g.add_sink(GraphNode::element(FakeSink::new()));
    let inserted = reg
        .decodebin(&mut g, src, sink, &rgba_any(), &is_nv12, 4)
        .expect("decodebin splices an RGBA -> NV12 converter");
    assert_eq!(inserted.len(), 1, "one converter node inserted between src and sink");

    let stats: RunStats = run_graph(g, &NullClock, 4).await.expect("spliced graph runs");
    assert_eq!(stats.frames_emitted, 4);
    assert_eq!(stats.frames_consumed, 4);
}

/// With the real `ffmpeg` decoder registered, the search routes H.264 to raw
/// through it. Compile + run only under the `ffmpeg` feature (it reads the
/// decoder's pad templates; no decode is performed, so no media is needed).
#[cfg(feature = "ffmpeg")]
#[test]
fn registry_finds_real_ffmpeg_decoder_for_h264() {
    use g2g_plugins::ffmpegdec::FfmpegH264Dec;
    let mut reg = Registry::new();
    reg.register(ElementFactory::of::<FfmpegH264Dec>("ffmpegdec", |_caps| {
        Box::new(FfmpegH264Dec::new())
    }));
    let chain = reg
        .autoplug_names(&h264(Dim::Any), &is_raw_video, 4)
        .expect("ffmpegdec bridges H.264 to raw video");
    assert_eq!(chain, Vec::from(["ffmpegdec"]));
}
