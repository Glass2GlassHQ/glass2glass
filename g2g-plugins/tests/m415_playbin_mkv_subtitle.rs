//! M415 - `playbin uri=*.mkv` routes an embedded subtitle track through a
//! `TextOverlayN` onto the video, the Matroska sibling of M412 (MP4). When a probed
//! MKV carries both a video track and an `S_TEXT/UTF8` subtitle track, `mkv_playbin`
//! builds `FileSrc -> MkvDemuxN -> { video: decode -> videoconvert(RGBA8) ->
//! overlay ; text: -> overlay } -> videoconvert(NV12) -> autovideosink`. An MKV with
//! no subtitle track keeps the plain per-stream A/V fan-out (M382), proving the
//! overlay is additive.

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::path::PathBuf;

use g2g_core::element::AsyncElement;
use g2g_core::runtime::{parse_launch, ElementFactory, GraphNode, LaunchFactory, Registry};
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, Graph, OutputSink, PadTemplate,
    PadTemplates, Rate, RawVideoFormat, VideoCodec,
};

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

/// A registry with the MKV playbin hook, a stub H.264 decoder, and the auto sinks.
fn registry() -> Registry {
    let mut reg = Registry::new();
    reg.register_playbin(g2g_plugins::uridecodebin::mkv_playbin);
    reg.register(ElementFactory::new(
        "h264stub",
        Vec::from([PadTemplate::sink(CapsSet::one(h264_any())), PadTemplate::source(CapsSet::one(raw_video()))]),
        |_| Box::new(g2g_plugins::identity::IdentityTransform::new()),
    ));
    reg.register_launch(LaunchFactory::of::<NullSink>("autovideosink", || Box::new(NullSink)));
    reg.register_launch(LaunchFactory::of::<NullSink>("autoaudiosink", || Box::new(NullSink)));
    reg
}

fn temp_uri(tag: &str, bytes: &[u8]) -> (PathBuf, String) {
    let path = std::env::temp_dir().join(format!("g2g_m415_{}_{}.mkv", std::process::id(), tag));
    std::fs::write(&path, bytes).expect("write fixture");
    let uri = format!("file://{}", path.display());
    (path, uri)
}

// --- synthetic Matroska builders (mirror the mkvdemux unit tests) ---
fn vint(value: u64) -> Vec<u8> {
    let mut len = 1usize;
    while len < 8 && value >= (1u64 << (7 * len)) - 1 {
        len += 1;
    }
    let mut out = vec![0u8; len];
    let mut v = value;
    for i in (0..len).rev() {
        out[i] = (v & 0xFF) as u8;
        v >>= 8;
    }
    out[0] |= 1 << (8 - len);
    out
}
fn elem(id: &[u8], body: &[u8]) -> Vec<u8> {
    let mut out = id.to_vec();
    out.extend_from_slice(&vint(body.len() as u64));
    out.extend_from_slice(body);
    out
}
fn uint_body(v: u64) -> Vec<u8> {
    if v == 0 {
        return vec![0];
    }
    let mut bytes = v.to_be_bytes().to_vec();
    while bytes.len() > 1 && bytes[0] == 0 {
        bytes.remove(0);
    }
    bytes
}
fn video_track(num: u64, codec: &[u8], w: u32, h: u32) -> Vec<u8> {
    let v = [elem(&[0xB0], &uint_body(w as u64)), elem(&[0xBA], &uint_body(h as u64))].concat();
    // TrackNumber, TrackType(video=1), CodecID, Video.
    let body =
        [elem(&[0xD7], &uint_body(num)), elem(&[0x83], &uint_body(1)), elem(&[0x86], codec), elem(&[0xE0], &v)]
            .concat();
    elem(&[0xAE], &body)
}
fn subtitle_track(num: u64, codec: &[u8]) -> Vec<u8> {
    // TrackNumber, TrackType(subtitle=0x11), CodecID. The codec ID pins the kind.
    let body =
        [elem(&[0xD7], &uint_body(num)), elem(&[0x83], &uint_body(0x11)), elem(&[0x86], codec)].concat();
    elem(&[0xAE], &body)
}

/// An MKV with a V_MPEG4/ISO/AVC (H.264) video track and, when `with_text`, an
/// `S_TEXT/UTF8` subtitle track. Only the `Tracks` element is needed: the playbin
/// graph build reads track info, not clusters.
fn mkv_with_optional_text(with_text: bool) -> Vec<u8> {
    let mut tracks_body = video_track(1, b"V_MPEG4/ISO/AVC", 320, 240);
    if with_text {
        tracks_body.extend_from_slice(&subtitle_track(2, b"S_TEXT/UTF8"));
    }
    let tracks = elem(&[0x16, 0x54, 0xAE, 0x6B], &tracks_body);
    let segment = elem(&[0x18, 0x53, 0x80, 0x67], &tracks);
    [elem(&[0x1A, 0x45, 0xDF, 0xA3], &[]), segment].concat()
}

/// Incoming-edge count per node id, to spot the fan-in (the overlay muxer).
fn inbound_counts(graph: &Graph<GraphNode>) -> std::collections::HashMap<u32, usize> {
    let mut counts = std::collections::HashMap::new();
    for e in graph.edges() {
        *counts.entry(e.dst.node.0).or_insert(0) += 1;
    }
    counts
}

#[test]
fn subtitle_track_routes_through_a_text_overlay() {
    let (path, uri) = temp_uri("vid_text", &mkv_with_optional_text(true));
    let reg = registry();
    let graph = parse_launch(&reg, &format!("playbin uri={uri}")).expect("mkv+subs playbin builds");
    std::fs::remove_file(&path).ok();

    // FileSrc, MkvDemuxN, h264 stub, videoconvert(RGBA8), TextOverlayN,
    // videoconvert(NV12), autovideosink.
    assert_eq!(graph.node_count(), 7, "source + demux + decode + 2 converts + overlay + sink");
    assert_eq!(graph.edges().len(), 7, "video decode path + text join + sink path");

    // Exactly one node is a fan-in (the overlay), fed by the video convert and the
    // demux's text port; everything else has a single input.
    let counts = inbound_counts(&graph);
    let fan_ins: Vec<_> = counts.values().filter(|&&n| n >= 2).collect();
    assert_eq!(fan_ins.len(), 1, "one fan-in node: the subtitle overlay");
    assert_eq!(*fan_ins[0], 2, "the overlay joins the video and the text streams");
}

#[test]
fn an_mkv_without_subtitles_keeps_the_plain_fanout() {
    let (path, uri) = temp_uri("vid_only", &mkv_with_optional_text(false));
    let reg = registry();
    let graph = parse_launch(&reg, &format!("playbin uri={uri}")).expect("video-only playbin builds");
    std::fs::remove_file(&path).ok();

    // FileSrc -> MkvDemuxN -> h264 stub -> autovideosink: no overlay inserted.
    assert_eq!(graph.node_count(), 4, "source + demux + decode + sink, no overlay");
    let counts = inbound_counts(&graph);
    assert!(counts.values().all(|&n| n <= 1), "no fan-in node without a subtitle track");
}
