//! M430 - `playbin uri=*.mkv#closed-captions=cc1` overlays the in-SEI closed
//! captions onto the video. Closed captions ride inside the compressed bitstream,
//! not a container track, so the video is teed: one copy decodes for display, the
//! other feeds a `CcExtract` whose timed text cues drive the same `TextOverlayN`
//! text pad a subtitle track would. The graph is `FileSrc -> MkvDemuxN -> Tee ->
//! { decode -> videoconvert(RGBA8) -> overlay ; h264parse -> ccextract -> overlay }
//! -> videoconvert(NV12) -> autovideosink`. Without the fragment the plain fan-out
//! (or a subtitle overlay) is kept, proving the caption path is opt-in; with both
//! a subtitle track and the fragment, the explicit caption request wins.

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

/// A registry with the MKV playbin hook, a stub H.264 decoder, and the auto sinks.
fn registry() -> Registry {
    let mut reg = Registry::new();
    reg.register_playbin(g2g_plugins::uridecodebin::mkv_playbin);
    reg.register(ElementFactory::new(
        "h264stub",
        Vec::from([
            PadTemplate::sink(CapsSet::one(h264_any())),
            PadTemplate::source(CapsSet::one(raw_video())),
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

fn temp_uri(tag: &str, bytes: &[u8]) -> (PathBuf, String) {
    let path = std::env::temp_dir().join(format!("g2g_m430_{}_{}.mkv", std::process::id(), tag));
    std::fs::write(&path, bytes).expect("write fixture");
    let uri = format!("file://{}", path.display());
    (path, uri)
}

// --- synthetic Matroska builders (mirror the m415 fixtures) ---
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
    let v = [
        elem(&[0xB0], &uint_body(w as u64)),
        elem(&[0xBA], &uint_body(h as u64)),
    ]
    .concat();
    let body = [
        elem(&[0xD7], &uint_body(num)),
        elem(&[0x83], &uint_body(1)),
        elem(&[0x86], codec),
        elem(&[0xE0], &v),
    ]
    .concat();
    elem(&[0xAE], &body)
}
fn subtitle_track(num: u64, codec: &[u8]) -> Vec<u8> {
    let body = [
        elem(&[0xD7], &uint_body(num)),
        elem(&[0x83], &uint_body(0x11)),
        elem(&[0x86], codec),
    ]
    .concat();
    elem(&[0xAE], &body)
}

/// An MKV with an H.264 video track and, when `with_text`, an `S_TEXT/UTF8`
/// subtitle track. Only `Tracks` is needed: the playbin build reads track info.
fn mkv_with_optional_text(with_text: bool) -> Vec<u8> {
    let mut tracks_body = video_track(1, b"V_MPEG4/ISO/AVC", 320, 240);
    if with_text {
        tracks_body.extend_from_slice(&subtitle_track(2, b"S_TEXT/UTF8"));
    }
    let tracks = elem(&[0x16, 0x54, 0xAE, 0x6B], &tracks_body);
    let segment = elem(&[0x18, 0x53, 0x80, 0x67], &tracks);
    [elem(&[0x1A, 0x45, 0xDF, 0xA3], &[]), segment].concat()
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

/// The closed-caption overlay topology: the video is teed (one fan-out node with
/// two outgoing edges) and the overlay joins the decoded video with the caption
/// text (one fan-in node with two incoming edges).
fn assert_cc_overlay(graph: &Graph<GraphNode>) {
    // FileSrc, MkvDemuxN, Tee, h264 stub, videoconvert(RGBA8), TextOverlayN,
    // videoconvert(NV12), autovideosink, h264parse (caption reframer), CcExtract.
    assert_eq!(
        graph.node_count(),
        10,
        "source+demux+tee+decode+2 converts+overlay+sink+parse+ccextract"
    );
    assert_eq!(
        graph.edges().len(),
        10,
        "src, tee in, 2 tee outs, decode chain, caption chain, overlay join, sink"
    );

    let fan_ins: Vec<_> = inbound_counts(graph)
        .into_values()
        .filter(|&n| n >= 2)
        .collect();
    assert_eq!(
        fan_ins,
        vec![2],
        "one fan-in (the overlay) joining the video and the caption text"
    );

    let fan_outs: Vec<_> = outbound_counts(graph)
        .into_values()
        .filter(|&n| n >= 2)
        .collect();
    assert_eq!(
        fan_outs,
        vec![2],
        "one fan-out (the video tee) feeding the decoder and the extractor"
    );
}

#[test]
fn closed_captions_fragment_routes_through_ccextract() {
    let (path, uri) = temp_uri("cc", &mkv_with_optional_text(false));
    let reg = registry();
    let graph = parse_launch(&reg, &format!("playbin uri={uri}#closed-captions=cc1"))
        .expect("mkv + #closed-captions playbin builds");
    std::fs::remove_file(&path).ok();
    assert_cc_overlay(&graph);
}

#[test]
fn cc_fragment_overrides_a_subtitle_track() {
    // The MKV carries an S_TEXT/UTF8 subtitle track, but the explicit caption
    // request wins (there is only one overlay text pad): the tee + CcExtract path
    // is built, not the subtitle-track overlay.
    let (path, uri) = temp_uri("cc_over_sub", &mkv_with_optional_text(true));
    let reg = registry();
    let graph = parse_launch(&reg, &format!("playbin uri={uri}#cc=cc1"))
        .expect("mkv + subtitle + #cc playbin builds");
    std::fs::remove_file(&path).ok();
    assert_cc_overlay(&graph);
}

#[test]
fn without_the_fragment_no_caption_path_is_added() {
    // The same video-only MKV with no fragment keeps the plain A/V fan-out: no
    // tee, no overlay, no extractor.
    let (path, uri) = temp_uri("plain", &mkv_with_optional_text(false));
    let reg = registry();
    let graph =
        parse_launch(&reg, &format!("playbin uri={uri}")).expect("video-only playbin builds");
    std::fs::remove_file(&path).ok();

    assert_eq!(
        graph.node_count(),
        4,
        "source + demux + decode + sink, no caption path"
    );
    assert!(
        inbound_counts(&graph).values().all(|&n| n <= 1),
        "no fan-in node"
    );
    assert!(
        outbound_counts(&graph).values().all(|&n| n <= 1),
        "no tee fan-out node"
    );
}
