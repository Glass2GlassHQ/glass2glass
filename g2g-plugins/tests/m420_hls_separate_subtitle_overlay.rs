//! M420 - HLS three-source subtitle overlay: `build_hls_separate_subtitle_overlay`
//! assembles a graph for a variant that pairs its (video-only) TS segments with
//! BOTH a separate audio rendition and a separate WebVTT subtitle rendition. Three
//! independent sources join in one graph: the video master (`HlsSrc -> TsDemuxN ->
//! decode -> videoconvert(RGBA8) -> TextOverlayN`), the audio rendition (`HlsSrc ->
//! TsDemuxN -> decode -> autoaudiosink`), and the subtitle rendition (`HlsSrc(text)
//! -> SubParse -> overlay.text`). Network-free assembly half (the sources fetch at
//! run, validated live); here we assert the cross-source topology.

#![cfg(all(feature = "std", feature = "hls"))]

use core::future::Future;
use core::pin::Pin;

use g2g_core::element::AsyncElement;
use g2g_core::runtime::{ElementFactory, GraphNode, LaunchFactory, Registry};
use g2g_core::{
    AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, Graph, OutputSink,
    PadTemplate, PadTemplates, Rate, RawVideoFormat, StreamType, VideoCodec,
};
use g2g_plugins::hlssrc::HlsStreamInfo;

fn h264_any() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}
fn raw_video() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}
fn aac_any() -> Caps {
    Caps::Audio {
        format: AudioFormat::Aac,
        channels: 0,
        sample_rate: 0,
    }
}
fn raw_audio() -> Caps {
    Caps::Audio {
        format: AudioFormat::PcmS16Le,
        channels: 0,
        sample_rate: 0,
    }
}

#[derive(Default)]
struct NullSink;
impl PadTemplates for NullSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::new()
    }
}
impl AsyncElement for NullSink {
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
        _packet: g2g_core::PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

/// A registry with stub H.264 / AAC decoders and the auto sinks; the builder is
/// called directly, so no playbin hook is needed.
fn registry() -> Registry {
    let mut reg = Registry::new();
    reg.register(ElementFactory::new(
        "h264stub",
        Vec::from([
            PadTemplate::sink(CapsSet::one(h264_any())),
            PadTemplate::source(CapsSet::one(raw_video())),
        ]),
        |_| Box::new(g2g_plugins::identity::IdentityTransform::new()),
    ));
    reg.register(ElementFactory::new(
        "aacstub",
        Vec::from([
            PadTemplate::sink(CapsSet::one(aac_any())),
            PadTemplate::source(CapsSet::one(raw_audio())),
        ]),
        |_| Box::new(g2g_plugins::identity::IdentityTransform::new()),
    ));
    reg.register_launch(LaunchFactory::of::<NullSink>("autovideosink", || {
        Box::new(NullSink)
    }));
    reg.register_launch(LaunchFactory::of::<NullSink>("autoaudiosink", || {
        Box::new(NullSink)
    }));
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
fn builds_a_three_source_overlay() {
    let reg = registry();
    let streams = vec![muxed_video()];
    let graph = g2g_plugins::uridecodebin::build_hls_separate_subtitle_overlay(
        &reg,
        "https://h/master.m3u8",
        &streams,
        "https://h/audio/en.m3u8",
        "https://h/subs/en.m3u8",
    )
    .expect("builds")
    .expect("a routable muxed video yields a graph");

    // Video chain: HlsSrc, TsDemuxN, h264 stub, videoconvert(RGBA8), TextOverlayN,
    // videoconvert(NV12), autovideosink = 7. Audio chain: HlsSrc, TsDemuxN, aac
    // stub, autoaudiosink = 4. Subtitle chain: HlsSrc(text), SubParse = 2. = 13.
    assert_eq!(
        graph.node_count(),
        13,
        "video+overlay+sink, audio chain, subtitle source+subparse"
    );

    // Exactly one fan-in (the overlay), fed by the video convert and the subtitle
    // SubParse, the two arriving from different sources.
    let counts = inbound_counts(&graph);
    let fan_ins: Vec<_> = counts.values().filter(|&&n| n >= 2).collect();
    assert_eq!(fan_ins.len(), 1, "one fan-in node: the subtitle overlay");
    assert_eq!(
        *fan_ins[0], 2,
        "the overlay joins the video and the WebVTT text stream"
    );

    // Three source nodes with no inbound edges: the video master, the separate audio
    // rendition, and the separate subtitle rendition.
    let has_inbound: std::collections::HashSet<u32> =
        graph.edges().iter().map(|e| e.dst.node.0).collect();
    let sources = (0..graph.node_count() as u32)
        .filter(|n| !has_inbound.contains(n))
        .count();
    assert_eq!(
        sources, 3,
        "three independent HLS sources (video master + audio + subtitle)"
    );
}

#[test]
fn declines_without_a_muxed_video() {
    let reg = registry();
    // An audio-only muxed set (no video in the variant's TS): nothing to overlay.
    let audio = HlsStreamInfo {
        stream_type: StreamType::Audio,
        caps: aac_any(),
        video: false,
        uri: None,
        name: "audio".into(),
        language: None,
    };
    let graph = g2g_plugins::uridecodebin::build_hls_separate_subtitle_overlay(
        &reg,
        "https://h/master.m3u8",
        &[audio],
        "https://h/audio/en.m3u8",
        "https://h/subs/en.m3u8",
    )
    .expect("ok");
    assert!(graph.is_none(), "no muxed video -> decline");
}
