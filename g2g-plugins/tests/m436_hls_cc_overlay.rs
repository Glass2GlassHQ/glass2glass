//! M436 - HLS closed-caption (CEA-608 / 708) auto-plug: `playbin
//! uri=hls://...#closed-captions=cc1` overlays the in-SEI captions onto the video,
//! the HLS analog of the file hooks' caption auto-plug (M430). Captions are not a
//! rendition (they ride the video SEI), so the URI fragment opts them in. The
//! network probe (`hls_playbin`) is validated live; this exercises the network-free
//! assembly: `build_hls_ts_cc_overlay` (muxed TS) and `build_hls_separate_cc_overlay`
//! (video TS + separate audio rendition) tee the compressed video so one copy
//! decodes for display and the other feeds a `CcExtract` into the overlay text pad.

#![cfg(all(feature = "std", feature = "hls"))]

use core::future::Future;
use core::pin::Pin;

use g2g_core::element::AsyncElement;
use g2g_core::runtime::{ElementFactory, GraphNode, LaunchFactory, Registry};
use g2g_core::{
    AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, Graph, OutputSink,
    PadTemplate, PadTemplates, Rate, RawVideoFormat, StreamType, VideoCodec,
};
use g2g_plugins::ccextract::CcSource;
use g2g_plugins::cea::Cea608Channel;
use g2g_plugins::hlssrc::HlsStreamInfo;
use g2g_plugins::uridecodebin::{build_hls_separate_cc_overlay, build_hls_ts_cc_overlay};

fn h264_any() -> Caps {
    Caps::CompressedVideo { codec: VideoCodec::H264, width: Dim::Any, height: Dim::Any, framerate: Rate::Any }
}
fn aac_any() -> Caps {
    Caps::Audio { format: AudioFormat::Aac, channels: 0, sample_rate: 0 }
}
fn raw_video() -> Caps {
    Caps::RawVideo { format: RawVideoFormat::Nv12, width: Dim::Any, height: Dim::Any, framerate: Rate::Any }
}
fn raw_audio() -> Caps {
    Caps::Audio { format: AudioFormat::PcmS16Le, channels: 0, sample_rate: 0 }
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

/// A registry with stub H.264 / AAC decoders and the auto sinks (the builders are
/// called directly, so no playbin hook is registered).
fn registry() -> Registry {
    let mut reg = Registry::new();
    reg.register(ElementFactory::new(
        "h264stub",
        Vec::from([PadTemplate::sink(CapsSet::one(h264_any())), PadTemplate::source(CapsSet::one(raw_video()))]),
        |_| Box::new(g2g_plugins::identity::IdentityTransform::new()),
    ));
    reg.register(ElementFactory::new(
        "aacstub",
        Vec::from([PadTemplate::sink(CapsSet::one(aac_any())), PadTemplate::source(CapsSet::one(raw_audio()))]),
        |_| Box::new(g2g_plugins::identity::IdentityTransform::new()),
    ));
    reg.register_launch(LaunchFactory::of::<NullSink>("autovideosink", || Box::new(NullSink)));
    reg.register_launch(LaunchFactory::of::<NullSink>("autoaudiosink", || Box::new(NullSink)));
    reg
}

fn muxed(stream_type: StreamType, caps: Caps, video: bool) -> HlsStreamInfo {
    HlsStreamInfo { stream_type, caps, video, uri: None, name: String::new(), language: None }
}

fn inbound_counts(graph: &Graph<GraphNode>) -> std::collections::HashMap<u32, usize> {
    let mut counts = std::collections::HashMap::new();
    for e in graph.edges() {
        *counts.entry(e.dst.node.0).or_insert(0) += 1;
    }
    counts
}

fn outbound_counts(graph: &Graph<GraphNode>) -> std::collections::HashMap<u32, usize> {
    let mut counts = std::collections::HashMap::new();
    for e in graph.edges() {
        *counts.entry(e.src.node.0).or_insert(0) += 1;
    }
    counts
}

fn source_count(graph: &Graph<GraphNode>) -> usize {
    let has_inbound: std::collections::HashSet<u32> =
        graph.edges().iter().map(|e| e.dst.node.0).collect();
    (0..graph.node_count() as u32).filter(|n| !has_inbound.contains(n)).count()
}

#[test]
fn muxed_ts_variant_overlays_in_sei_captions() {
    let reg = registry();
    // A muxed A/V variant (video + audio in the variant's own TS segments).
    let streams = vec![
        muxed(StreamType::Video, h264_any(), true),
        muxed(StreamType::Audio, aac_any(), false),
    ];
    let graph =
        build_hls_ts_cc_overlay(&reg, "https://h/master.m3u8", &streams, CcSource::Cea608(Cea608Channel::Cc1))
            .expect("builds")
            .expect("a routable muxed video yields a caption-overlay graph");

    // HlsSrc, TsDemuxN, tee, display decoder, videoconvert(RGBA8), TextOverlayN,
    // videoconvert(NV12), autovideosink, H264Parse(reframe), CcExtract, plus the
    // audio branch (decoder, audioconvert, audioresample, autoaudiosink) = 14.
    assert_eq!(graph.node_count(), 14, "video caption overlay + audio fan-out");

    // The video is teed (one copy decodes for display, one feeds the captions): a
    // node with two outbound edges that is not the single demux (the demux also
    // fans video + audio, so two nodes fan out by two).
    let out = outbound_counts(&graph);
    let fan_outs = out.values().filter(|&&n| n == 2).count();
    assert_eq!(fan_outs, 2, "the demux (video+audio) and the caption tee both fan out by two");

    // Exactly one fan-in: the overlay, fed by the display videoconvert and the
    // CcExtract caption branch.
    let counts = inbound_counts(&graph);
    let fan_ins: Vec<_> = counts.values().filter(|&&n| n >= 2).collect();
    assert_eq!(fan_ins.len(), 1, "one fan-in node: the caption overlay");
    assert_eq!(*fan_ins[0], 2, "the overlay joins the decoded video and the extracted captions");

    // A single muxed source.
    assert_eq!(source_count(&graph), 1, "one muxed HLS source");
}

#[test]
fn separate_audio_variant_overlays_captions_across_sources() {
    let reg = registry();
    // A video-only muxed variant; audio is a separate rendition (its own playlist).
    let streams = vec![muxed(StreamType::Video, h264_any(), true)];
    let graph = build_hls_separate_cc_overlay(
        &reg,
        "https://h/master.m3u8",
        &streams,
        "https://h/audio/en.m3u8",
        CcSource::Cea608(Cea608Channel::Cc1),
    )
    .expect("builds")
    .expect("a routable muxed video yields a caption-overlay graph");

    // Two independent HLS sources: the video master + the separate audio rendition.
    assert_eq!(source_count(&graph), 2, "video master + separate audio rendition");

    // One fan-in (the overlay): the audio rendition decodes straight to its own
    // sink, so the only >=2 inbound node is the caption overlay.
    let counts = inbound_counts(&graph);
    let fan_ins: Vec<_> = counts.values().filter(|&&n| n >= 2).collect();
    assert_eq!(fan_ins.len(), 1, "one fan-in node: the caption overlay");
    assert_eq!(*fan_ins[0], 2, "the overlay joins the decoded video and the extracted captions");
}

#[test]
fn declines_without_a_muxed_video() {
    let reg = registry();
    // An audio-only muxed set: no video SEI to mine captions from.
    let streams = vec![muxed(StreamType::Audio, aac_any(), false)];
    let graph =
        build_hls_ts_cc_overlay(&reg, "https://h/master.m3u8", &streams, CcSource::Cea608(Cea608Channel::Cc1))
            .expect("ok");
    assert!(graph.is_none(), "no muxed video -> decline");
}
