//! M196: `uridecodebin` / `playbin` in a gst-launch line. `uridecodebin uri=X`
//! builds its source from the URI scheme handler and auto-plugs the decode chain
//! to raw, splicing both into the pipeline as pre-built nodes; `playbin uri=X` is
//! that plus an auto sink (a complete pipeline). The `file://` handler uses the
//! self-demuxing `Mp4Src` (emits H.264), so a decoder candidate is needed to
//! reach raw, here a stand-in stub (templates drive the search; we assert the
//! parsed graph structure, not a decode run, which would need a real file).
#![cfg(feature = "std")]

use g2g_core::runtime::{parse_launch, ElementFactory, ParseError, Registry};
use g2g_core::{Caps, CapsSet, Dim, PadTemplate, Rate, RawVideoFormat, VideoCodec};
use g2g_plugins::identity::IdentityTransform;
use g2g_plugins::registry::default_registry;

fn h264_any() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

/// `default_registry` plus a stub H.264 decoder candidate, so the `file://` ->
/// `Mp4Src` (H.264) source can auto-plug to raw.
fn registry_with_h264_stub() -> Registry {
    let mut reg = default_registry();
    let raw = Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    };
    reg.register(ElementFactory::new(
        "h264stub",
        Vec::from([
            PadTemplate::sink(CapsSet::one(h264_any())),
            PadTemplate::source(CapsSet::one(raw)),
        ]),
        |_| Box::new(IdentityTransform::new()),
    ));
    reg
}

#[test]
fn uridecodebin_expands_source_and_decode_chain() {
    let reg = registry_with_h264_stub();
    // Mp4Src (from file://) -> h264stub -> fakesink: the source + decoder are
    // spliced in as pre-built nodes, leaving two edges.
    let line = "uridecodebin uri=file:///clip.mp4 ! fakesink";
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("{line:?} parse: {e}"));
    assert_eq!(graph.edges().len(), 2, "source -> decoder -> sink");
}

#[test]
fn uridecodebin_feeds_a_custom_chain() {
    let reg = registry_with_h264_stub();
    // The decoded output flows into a normal parser-built chain.
    let line = "uridecodebin uri=file:///clip.mp4 ! videoconvert ! fakesink";
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("{line:?} parse: {e}"));
    // Mp4Src -> h264stub -> videoconvert -> fakesink = three edges.
    assert_eq!(graph.edges().len(), 3, "{line}");
}

#[test]
fn playbin_is_a_complete_pipeline() {
    let reg = registry_with_h264_stub();
    // playbin appends an auto sink (autovideosink -> fakesink here), so the line
    // is complete on its own: Mp4Src -> h264stub -> autovideosink.
    let line = "playbin uri=file:///clip.mp4";
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("{line:?} parse: {e}"));
    assert_eq!(graph.edges().len(), 2, "source -> decoder -> auto sink");
}

#[test]
fn missing_uri_fails_loud() {
    let reg = registry_with_h264_stub();
    let err = parse_launch(&reg, "uridecodebin ! fakesink").unwrap_err();
    assert!(matches!(err, ParseError::MissingUri(_)), "got {err:?}");
}

#[test]
fn unknown_scheme_fails_loud() {
    let reg = registry_with_h264_stub();
    let err = parse_launch(&reg, "uridecodebin uri=bogus://nowhere ! fakesink").unwrap_err();
    assert!(matches!(err, ParseError::Uri(_)), "got {err:?}");
}

#[test]
fn uri_source_not_at_head_fails_loud() {
    let reg = registry_with_h264_stub();
    // uridecodebin provides the source, so it cannot sit mid-pipeline.
    let line = "videotestsrc num-buffers=1 ! uridecodebin uri=file:///clip.mp4 ! fakesink";
    let err = parse_launch(&reg, line).unwrap_err();
    assert!(
        matches!(err, ParseError::UriSourceNotAtHead(_)),
        "got {err:?}"
    );
}

#[test]
fn no_decoder_feature_fails_loud() {
    // Without the stub (baseline: file:// -> Mp4Src emits H.264, no decoder), the
    // chain cannot reach raw and the parse fails rather than silently dropping.
    let reg = default_registry();
    let line = "uridecodebin uri=file:///clip.mp4 ! fakesink";
    match parse_launch(&reg, line) {
        // Baseline build: no decoder, must fail loud.
        #[cfg(not(any(
            feature = "ffmpeg",
            feature = "vaapi",
            feature = "nvdec",
            feature = "mediacodec",
            feature = "vulkan-video"
        )))]
        Ok(_) => panic!("expected NoDecodeChain without a decoder feature"),
        #[cfg(not(any(
            feature = "ffmpeg",
            feature = "vaapi",
            feature = "nvdec",
            feature = "mediacodec",
            feature = "vulkan-video"
        )))]
        Err(e) => assert!(matches!(e, ParseError::NoDecodeChain(_)), "got {e:?}"),
        // With a real decoder feature on, it resolves instead, also fine.
        #[cfg(any(
            feature = "ffmpeg",
            feature = "vaapi",
            feature = "nvdec",
            feature = "mediacodec",
            feature = "vulkan-video"
        ))]
        result => {
            let _ = result;
        }
    }
}
