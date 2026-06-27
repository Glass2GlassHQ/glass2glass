//! M193: `decodebin` in a gst-launch line. decodebin is not an element; the
//! parser expands it, at parse time, into the chain of decoders / parsers the
//! auto-plug search (M83) finds from its upstream caps down to raw video or
//! audio. An already-raw input (the source produces raw) expands to nothing, so
//! decodebin becomes a pass-through. This also exercises the newly-registered
//! auto-plug candidates (`default_registry` registers the parsers / decoders as
//! `ElementFactory`s the search composes).
#![cfg(feature = "std")]

use g2g_core::runtime::{
    is_raw_video, parse_launch, run_graph, ElementFactory, ParseError, SourceFactory,
};
use g2g_core::{Caps, CapsSet, Dim, PadTemplate, PipelineClock, Rate, RawVideoFormat, VideoCodec};
use g2g_plugins::identity::IdentityTransform;
use g2g_plugins::registry::default_registry;
use g2g_plugins::videotestsrc::VideoTestSrc;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

// Used by the baseline "no route" test (no decoder feature) and the ffmpeg
// "routes to raw" test; gate the helper to exactly those builds so it is not
// dead code under another decoder feature (vaapi / nvdec / mediacodec).
#[cfg(any(
    feature = "ffmpeg",
    not(any(feature = "ffmpeg", feature = "vaapi", feature = "nvdec", feature = "mediacodec"))
))]
fn h264_any() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

/// VP9 has no in-tree decoder, so a stub decoder for it never competes with the
/// real (H.264-only) `ffmpegdec` / `vaapidec` when those features are on, keeping
/// the expansion test deterministic across feature combinations.
fn vp9_any() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::Vp9,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

fn nv12_any() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

#[tokio::test]
async fn decodebin_passthrough_on_already_raw_runs() {
    // videotestsrc produces raw RGBA, which already satisfies the decode target,
    // so decodebin expands to an empty chain and drops out: the graph is just
    // videotestsrc -> fakesink, and it runs.
    let reg = default_registry();
    let line = "videotestsrc num-buffers=2 ! decodebin ! fakesink";
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("{line:?} parse: {e}"));
    assert_eq!(graph.edges().len(), 1, "decodebin drops out, leaving one edge");
    let consumed = run_graph(graph, &ZeroClock, 4).await.expect("runs").frames_consumed;
    assert_eq!(consumed, 2, "{line}");
}

#[tokio::test]
async fn decodebin_passthrough_with_downstream_runs() {
    let reg = default_registry();
    let line = "videotestsrc num-buffers=2 ! decodebin ! videoconvert ! fakesink";
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("{line:?} parse: {e}"));
    // videotestsrc -> videoconvert -> fakesink (decodebin gone).
    assert_eq!(graph.edges().len(), 2, "{line}");
    assert_eq!(run_graph(graph, &ZeroClock, 4).await.expect("runs").frames_consumed, 2);
}

#[tokio::test]
async fn decodebin_expands_a_decoder_chain() {
    // A source declaring compressed H.264 plus a decoder candidate (H.264 -> raw
    // NV12). decodebin auto-plugs the decoder between them, so the parsed graph
    // is vp9src -> decoder -> fakesink (two edges). The decoder body is an
    // IdentityTransform stand-in (templates drive the search; we assert the parse
    // structure, not a decode run).
    let mut reg = default_registry();
    reg.register_source(SourceFactory::new("vp9src", vp9_any(), || {
        Box::new(VideoTestSrc::new(8, 8, 30, 2))
    }));
    let templates = Vec::from([
        PadTemplate::sink(CapsSet::one(vp9_any())),
        PadTemplate::source(CapsSet::one(nv12_any())),
    ]);
    reg.register(ElementFactory::new("fakedec", templates, |_| Box::new(IdentityTransform::new())));
    // Buildable by name, so the parser can instantiate the expanded element.
    reg.register_launch(g2g_core::runtime::LaunchFactory::new(
        "fakedec",
        Vec::new(),
        || Box::new(IdentityTransform::new()),
    ));

    // Sanity: the search routes the compressed stream to raw in a single decoder
    // hop. (We assert the shape, not the decoder name: with a real decoder feature
    // on, its candidate may be chosen over the stub, but it is still one hop.)
    let chain = reg.autoplug_names(&vp9_any(), &is_raw_video, 4).expect("a decoder reaches raw");
    assert_eq!(chain.len(), 1, "one decoder hop to raw, got {chain:?}");

    let line = "vp9src ! decodebin ! fakesink";
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("{line:?} parse: {e}"));
    assert_eq!(graph.edges().len(), 2, "decodebin expanded to one decoder node");
}

#[tokio::test]
async fn decodebin_without_upstream_fails_loud() {
    let reg = default_registry();
    // decodebin as the first element has no upstream caps to decode.
    let err = parse_launch(&reg, "decodebin ! fakesink").unwrap_err();
    assert!(matches!(err, ParseError::DecodebinNoUpstream), "got {err:?}");
}

// Only meaningful without an H.264 decoder compiled in; any of `ffmpegdec`,
// `vaapidec`, `nvdec` (Linux) or `mediacodecdec` (Android) would (correctly)
// provide the route this asserts is absent.
#[cfg(not(any(
    feature = "ffmpeg",
    feature = "vaapi",
    feature = "nvdec",
    feature = "mediacodec"
)))]
#[test]
fn baseline_registry_has_no_route_from_h264_to_raw() {
    // The baseline build registers parsers as auto-plug candidates but no decoder
    // (decoders are feature-gated), so H.264 cannot reach raw: a decodebin on a
    // compressed input fails loud rather than silently dropping the stream.
    let reg = default_registry();
    assert!(
        reg.autoplug_names(&h264_any(), &is_raw_video, 6).is_none(),
        "parsers alone never reach raw video without a decoder feature",
    );
}

/// Under the `ffmpeg` feature the default registry's `ffmpegdec` candidate routes
/// H.264 to raw, so the search (and a `decodebin` on H.264) resolves. Reads
/// templates only; no decode is performed.
#[cfg(all(target_os = "linux", feature = "ffmpeg"))]
#[test]
fn ffmpeg_feature_routes_h264_to_raw() {
    let reg = default_registry();
    let chain = reg
        .autoplug_names(&h264_any(), &is_raw_video, 4)
        .expect("ffmpegdec bridges H.264 to raw");
    assert_eq!(chain, Vec::from(["ffmpegdec"]));
}
