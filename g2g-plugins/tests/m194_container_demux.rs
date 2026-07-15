//! M194: container-demux auto-plug. g2g demuxers are 1-in/1-out (a container
//! byte stream in, one selected elementary stream out), so the M83 chain search
//! composes them like any other element: `ByteStream{container}` -> demuxer ->
//! `CompressedVideo` -> decoder -> raw. Registering the demuxers as auto-plug
//! candidates lets `decodebin` (and `Registry::autoplug` / `build_playbin`) route
//! a container all the way to raw, not just an elementary stream.
#![cfg(feature = "std")]

use g2g_core::runtime::is_raw_video;
// Only the baseline "no decoder" test and the ffmpeg decode test drive a launch
// line; gate the import to match so it is not unused under another decoder feature.
#[cfg(any(
    all(target_os = "linux", feature = "ffmpeg"),
    not(any(
        feature = "ffmpeg",
        feature = "vaapi",
        feature = "nvdec",
        feature = "mediacodec",
        feature = "vulkan-video"
    ))
))]
use g2g_core::runtime::parse_launch;
use g2g_core::{ByteStreamEncoding, Caps, Dim, Rate};
use g2g_plugins::registry::default_registry;

fn container(encoding: ByteStreamEncoding) -> Caps {
    Caps::ByteStream { encoding }
}

fn is_compressed_video(c: &Caps) -> bool {
    matches!(c, Caps::CompressedVideo { .. })
}

/// Each video container's demuxer is found as the one-hop route from the
/// container byte stream to a compressed elementary stream. This works in the
/// baseline build (no decoder feature needed) because the target is the
/// compressed stream, not raw.
#[test]
fn demuxers_route_containers_to_an_elementary_stream() {
    let reg = default_registry();
    for (encoding, demux) in [
        (ByteStreamEncoding::MpegTs, "tsdemux"),
        (ByteStreamEncoding::Matroska, "matroskademux"),
        (ByteStreamEncoding::IsoBmff, "fmp4demux"),
        (ByteStreamEncoding::Flv, "flvdemux"),
    ] {
        let chain = reg
            .autoplug_names(&container(encoding), &is_compressed_video, 4)
            .unwrap_or_else(|| panic!("{demux} should route {encoding:?} to a compressed stream"));
        assert_eq!(chain, Vec::from([demux]), "shortest route from {encoding:?}");
    }
}

// Only meaningful without an H.264 decoder (the MPEG-TS default stream is H.264);
// any of `ffmpegdec`, `vaapidec`, `nvdec` (Linux), `mediacodecdec` (Android) or
// `vulkanvideodec` (its NV12 System fallback) would provide the route to raw this
// asserts is absent.
#[cfg(not(any(
    feature = "ffmpeg",
    feature = "vaapi",
    feature = "nvdec",
    feature = "mediacodec",
    feature = "vulkan-video"
)))]
#[test]
fn container_without_a_decoder_does_not_reach_raw() {
    // Baseline: the demuxer reaches a compressed stream, but with no decoder
    // feature compiled in there is no further hop to raw, so a decodebin on a
    // container fails loud rather than silently dropping it.
    let reg = default_registry();
    assert!(
        reg.autoplug_names(&container(ByteStreamEncoding::MpegTs), &is_raw_video, 6).is_none(),
        "MPEG-TS -> raw needs a decoder feature",
    );
    // The decodebin macro surfaces the same as a parse error (filesrc declares
    // MPEG-TS by default).
    let line = "filesrc ! decodebin ! fakesink";
    assert!(parse_launch(&reg, line).is_err(), "{line:?} must fail without a decoder");
}

/// Re-uses the same compressed-video target on a synthetic registry to prove the
/// demuxer composes with a following hop, independent of any decoder feature: a
/// demuxer to compressed, then (separately) a decoder to raw, is the two-hop
/// chain the search builds. Here we assert the two-hop shape via the public
/// `find_chain` over the registry's descriptors using a decoder candidate.
#[test]
fn demuxer_then_decoder_is_a_two_hop_chain() {
    use g2g_core::runtime::ElementFactory;
    use g2g_core::{CapsSet, PadTemplate, RawVideoFormat, VideoCodec};
    use g2g_plugins::identity::IdentityTransform;

    let mut reg = default_registry();
    // A stand-in H.264 decoder candidate (templates drive the search; the body is
    // never run here). With it present, MPEG-TS routes container -> tsdemux ->
    // (this) decoder -> raw.
    let h264 = Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    };
    let raw = Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    };
    reg.register(ElementFactory::new(
        "stubdec",
        Vec::from([PadTemplate::sink(CapsSet::one(h264)), PadTemplate::source(CapsSet::one(raw))]),
        |_| Box::new(IdentityTransform::new()),
    ));

    let chain = reg
        .autoplug_names(&container(ByteStreamEncoding::MpegTs), &is_raw_video, 6)
        .expect("MPEG-TS -> tsdemux -> decoder -> raw");
    // Two hops: demux to a compressed stream, then decode to raw. We assert the
    // shape and the demuxer (the first hop); the decoder hop may be the stub or a
    // real decoder candidate when its feature is on, but it is exactly one hop.
    assert_eq!(chain.len(), 2, "container demux then decode, got {chain:?}");
    assert_eq!(chain[0], "tsdemux", "the demuxer is the first hop");
}

/// With the real ffmpeg decoder compiled in, a container decodes to raw: the
/// search routes MPEG-TS -> tsdemux -> ffmpegdec, and `filesrc ! decodebin`
/// expands to that chain. Reads templates only (no media is decoded).
#[cfg(all(target_os = "linux", feature = "ffmpeg"))]
#[test]
fn ffmpeg_decodes_a_container_to_raw() {
    let reg = default_registry();
    let chain = reg
        .autoplug_names(&container(ByteStreamEncoding::MpegTs), &is_raw_video, 6)
        .expect("MPEG-TS decodes to raw under ffmpeg");
    assert_eq!(chain, Vec::from(["tsdemux", "ffmpegdec"]));

    // The decodebin macro builds the same chain inline: filesrc (declares
    // MPEG-TS) -> tsdemux -> ffmpegdec -> fakesink (three edges).
    let line = "filesrc ! decodebin ! fakesink";
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("{line:?} parse: {e}"));
    assert_eq!(graph.edges().len(), 3, "decodebin expanded demux + decode");
}
