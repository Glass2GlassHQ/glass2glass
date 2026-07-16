//! M195: reflect a source's `bytestream-format` into `decodebin`'s auto-plug
//! input. A `filesrc bytestream-format=matroska ! decodebin` now selects
//! `matroskademux`, not `filesrc`'s factory-default container (MPEG-TS). The
//! parser reads the configured container synchronously via
//! `SourceLoop::configured_output_caps`, overridden by `FileSrc`, so the chain
//! search starts from the right container caps.
#![cfg(feature = "std")]

use g2g_core::runtime::{parse_launch, ElementFactory, Registry, SourceLoop};
use g2g_core::{
    ByteStreamEncoding, Caps, CapsSet, Dim, PadTemplate, PropValue, Rate, RawVideoFormat,
};
use g2g_plugins::filesrc::FileSrc;
use g2g_plugins::identity::IdentityTransform;
use g2g_plugins::registry::default_registry;

fn container(encoding: ByteStreamEncoding) -> Caps {
    Caps::ByteStream { encoding }
}

fn is_compressed_video(c: &Caps) -> bool {
    matches!(c, Caps::CompressedVideo { .. })
}

fn filesrc_with_format(format: &str) -> FileSrc {
    let mut fs = FileSrc::new("clip", container(ByteStreamEncoding::MpegTs));
    SourceLoop::set_property(&mut fs, "bytestream-format", PropValue::Str(format.into()))
        .expect("bytestream-format is settable");
    fs
}

#[test]
fn filesrc_reflects_explicit_container() {
    // An explicit container nick is readable synchronously, the property having
    // re-typed the source's output caps.
    for (nick, encoding) in [
        ("matroska", ByteStreamEncoding::Matroska),
        ("mpegts", ByteStreamEncoding::MpegTs),
        // M479: `mp4` is the whole-file form (demuxed by `mp4demux`); the streaming
        // `isobmff` / `cmaf` / `fmp4` nicks name the fragmented form (`fmp4demux`).
        ("mp4", ByteStreamEncoding::Mp4),
        ("isobmff", ByteStreamEncoding::IsoBmff),
        ("flv", ByteStreamEncoding::Flv),
    ] {
        let fs = filesrc_with_format(nick);
        assert_eq!(
            SourceLoop::configured_output_caps(&fs),
            Some(container(encoding)),
            "bytestream-format={nick}",
        );
    }
}

#[test]
fn filesrc_auto_is_unknown_until_runtime() {
    // `auto` sniffs the header at run time, so the container is not known
    // synchronously: the parser falls back to the declared default.
    let fs = filesrc_with_format("auto");
    assert_eq!(SourceLoop::configured_output_caps(&fs), None, "auto needs a run-time sniff");
}

#[test]
fn reflected_container_routes_to_its_demuxer() {
    // The reflected caps drive the auto-plug search to the matching demuxer,
    // each container to its own (target the compressed stream so no decoder
    // feature is needed).
    let reg = default_registry();
    for (encoding, demux) in [
        (ByteStreamEncoding::Matroska, "matroskademux"),
        (ByteStreamEncoding::IsoBmff, "fmp4demux"),
        (ByteStreamEncoding::Flv, "flvdemux"),
    ] {
        let caps = container(encoding);
        let chain = reg
            .autoplug_names(&caps, &is_compressed_video, 4)
            .unwrap_or_else(|| panic!("{encoding:?} routes to a demuxer"));
        assert_eq!(chain, Vec::from([demux]), "{encoding:?}");
    }
}

/// Register a stub decoder for VP9 (Matroska's default elementary stream) so a
/// Matroska `decodebin` can reach raw. With it, the container selected by
/// `bytestream-format` decides whether the chain completes: Matroska routes
/// through `matroskademux` to the VP9 stub (matroskademux's parameterless default
/// stream is VP9), while the MPEG-TS default emits H.264 / H.265 / AAC, none of
/// which the VP9 stub decodes, so it fails, proving the property chose the
/// container.
fn registry_with_vp9_stub() -> Registry {
    let mut reg = default_registry();
    let vp9 = Caps::CompressedVideo {
        codec: g2g_core::VideoCodec::Vp9,
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
        "vp9stub",
        Vec::from([PadTemplate::sink(CapsSet::one(vp9)), PadTemplate::source(CapsSet::one(raw))]),
        |_| Box::new(IdentityTransform::new()),
    ));
    reg.register_launch(g2g_core::runtime::LaunchFactory::new(
        "vp9stub",
        Vec::new(),
        || Box::new(IdentityTransform::new()),
    ));
    reg
}

#[test]
fn decodebin_uses_the_reflected_container() {
    let reg = registry_with_vp9_stub();
    // Matroska: filesrc -> matroskademux -> vp9stub -> fakesink (three edges).
    // In the baseline build this alone proves reflection: without it, filesrc
    // would declare its MPEG-TS default, decodebin would pick tsdemux (H.264), and
    // with only a VP9 stub the chain would not reach raw, so the parse would fail.
    let line = "filesrc bytestream-format=matroska ! decodebin ! fakesink";
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("{line:?} parse: {e}"));
    assert_eq!(graph.edges().len(), 3, "matroska decode chain: demux + decoder");
}

// The contrast that nails it down: without the property, filesrc's MPEG-TS
// default emits H.264, which the VP9 stub cannot decode, so the same line fails.
// Only valid without a real H.264 decoder (`ffmpegdec` / `vaapidec` / `nvdec` /
// `mediacodecdec` / `vulkanvideodec` would decode the default and the contrast
// would not hold).
#[cfg(not(any(
    feature = "ffmpeg",
    feature = "vaapi",
    feature = "nvdec",
    feature = "mediacodec",
    feature = "vulkan-video"
)))]
#[test]
fn mpegts_default_without_a_matching_decoder_fails() {
    let reg = registry_with_vp9_stub();
    let default_line = "filesrc ! decodebin ! fakesink";
    assert!(
        parse_launch(&reg, default_line).is_err(),
        "MPEG-TS default has no VP9 stub route to raw",
    );
}
