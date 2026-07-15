//! M419 - HLS subtitle-rendition playback fan-out: `build_hls_subtitle_overlay`
//! assembles the cross-source overlay graph. The video rides the variant's muxed
//! MPEG-TS segments (`HlsSrc -> TsDemuxN -> decode -> videoconvert(RGBA8) ->
//! TextOverlayN`), while the subtitle is a *separate* WebVTT rendition
//! (`HlsSrc(text) -> SubParse -> overlay.text`), the two sources joined at the
//! overlay. This is the network-free assembly half (the sources probe / fetch at
//! run, validated live); here we assert the topology has the cross-source fan-in.

#![cfg(all(feature = "std", feature = "hls"))]

use core::future::Future;
use core::pin::Pin;

use g2g_core::element::AsyncElement;
use g2g_core::runtime::{ElementFactory, GraphNode, LaunchFactory, Registry};
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, Graph, OutputSink, PadTemplate,
    PadTemplates, Rate, RawVideoFormat, StreamType, VideoCodec,
};
use g2g_plugins::hlssrc::HlsStreamInfo;

fn h264_any() -> Caps {
    Caps::CompressedVideo { codec: VideoCodec::H264, width: Dim::Any, height: Dim::Any, framerate: Rate::Any }
}
fn raw_video() -> Caps {
    Caps::RawVideo { format: RawVideoFormat::Nv12, width: Dim::Any, height: Dim::Any, framerate: Rate::Any }
}

#[derive(Default)]
struct NullSink;
impl PadTemplates for NullSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::new()
    }
}
impl AsyncElement for NullSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>> where Self: 'a;
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
        _packet: g2g_core::PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

/// A registry with a stub H.264 decoder and the auto sinks (no playbin hook needed:
/// the builder is called directly).
fn registry() -> Registry {
    let mut reg = Registry::new();
    reg.register(ElementFactory::new(
        "h264stub",
        Vec::from([PadTemplate::sink(CapsSet::one(h264_any())), PadTemplate::source(CapsSet::one(raw_video()))]),
        |_| Box::new(g2g_plugins::identity::IdentityTransform::new()),
    ));
    reg.register_launch(LaunchFactory::of::<NullSink>("autovideosink", || Box::new(NullSink)));
    reg.register_launch(LaunchFactory::of::<NullSink>("autoaudiosink", || Box::new(NullSink)));
    reg
}

fn muxed_video() -> HlsStreamInfo {
    HlsStreamInfo {
        stream_type: StreamType::Video,
        caps: h264_any(),
        video: true,
        uri: None, // muxed into the variant's own TS segments
        name: "video".into(),
        language: None,
    }
}

fn inbound_counts(graph: &Graph<GraphNode>) -> std::collections::HashMap<u32, usize> {
    let mut counts = std::collections::HashMap::new();
    for e in graph.edges() {
        *counts.entry(e.dst.node.0).or_insert(0) += 1;
    }
    counts
}

#[test]
fn builds_a_cross_source_subtitle_overlay() {
    let reg = registry();
    let streams = vec![muxed_video()];
    let graph = g2g_plugins::uridecodebin::build_hls_subtitle_overlay(
        &reg,
        "https://h/master.m3u8",
        &streams,
        "https://h/subs/en.m3u8",
    )
    .expect("builds")
    .expect("a routable muxed video yields a graph");

    // HlsSrc(video), TsDemuxN, h264 stub, videoconvert(RGBA8), TextOverlayN,
    // videoconvert(NV12), autovideosink, HlsSrc(subtitle), SubParse = 9 nodes.
    assert_eq!(graph.node_count(), 9, "video chain + overlay + sink + subtitle source + subparse");

    // Exactly one fan-in (the overlay), fed by the video convert and the subtitle
    // SubParse from a *different* source.
    let counts = inbound_counts(&graph);
    let fan_ins: Vec<_> = counts.values().filter(|&&n| n >= 2).collect();
    assert_eq!(fan_ins.len(), 1, "one fan-in node: the subtitle overlay");
    assert_eq!(*fan_ins[0], 2, "the overlay joins the video and the WebVTT text stream");

    // Two source nodes with no inbound edges: the video master + the separate
    // subtitle rendition (the cross-source shape).
    let has_inbound: std::collections::HashSet<u32> =
        graph.edges().iter().map(|e| e.dst.node.0).collect();
    let sources = (0..graph.node_count() as u32).filter(|n| !has_inbound.contains(n)).count();
    assert_eq!(sources, 2, "two independent HLS sources (video master + subtitle rendition)");
}

#[test]
fn declines_without_a_muxed_video() {
    let reg = registry();
    // An audio-only muxed set (no video): nothing to overlay onto.
    let audio = HlsStreamInfo {
        stream_type: StreamType::Audio,
        caps: Caps::Audio { format: g2g_core::AudioFormat::Aac, channels: 0, sample_rate: 0 },
        video: false,
        uri: None,
        name: "audio".into(),
        language: None,
    };
    let graph = g2g_plugins::uridecodebin::build_hls_subtitle_overlay(
        &reg,
        "https://h/master.m3u8",
        &[audio],
        "https://h/subs/en.m3u8",
    )
    .expect("ok");
    assert!(graph.is_none(), "no muxed video -> decline");
}
