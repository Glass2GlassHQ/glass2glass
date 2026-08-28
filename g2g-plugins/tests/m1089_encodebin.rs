//! M1089 `encodebin` / `transcodebin`: an encoding profile expands, at parse
//! time, into the encoders it names plus the muxer for its container.
//!
//! `encodebin` is a macro, so the test of it is equivalence: what a profile line
//! writes must be byte for byte what the hand-written chain writes. Each case
//! below runs both and compares the files, which also pins which element the
//! profile chose (a different encoder or muxer would write different bytes), and
//! that no converter was spliced where the encoder already took what arrived
//! (M1091 splices one only when it does not).
//!
//! A profile stream may also pin a geometry, a framerate, a sample rate, a
//! channel count or a bitrate (M1097). What a pin does is put a converter in the
//! graph (or a property on the encoder), so those cases assert on the built
//! graph as well as on the file.
//!
//! `default_registry` is `std`-gated, so this file is too: run with
//! `cargo test -p g2g-plugins --features std,mjpeg,mjpeg-encode`.
#![cfg(feature = "std")]

use std::path::PathBuf;

use g2g_core::element::DynAsyncElement;
use g2g_core::runtime::{parse_launch, run_graph, GraphNode, GraphNodeRef};
use g2g_core::{Graph, NodeId, PipelineClock, PropValue};
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn temp_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m1089-{tag}-{}.bin", std::process::id()))
}

/// Only the transcode leg reads a file, and it needs a decoder compiled in.
#[cfg(all(feature = "mjpeg", feature = "mjpeg-encode"))]
fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// Run a line whose `{}` is the output path, and return what it wrote.
async fn run_to_bytes(template: &str, tag: &str) -> Vec<u8> {
    let out = temp_path(tag);
    let _ = std::fs::remove_file(&out);
    let line = template.replace("{}", &out.display().to_string());
    let reg = default_registry();
    let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
    let bytes = std::fs::read(&out).expect("the sink wrote a file");
    std::fs::remove_file(&out).ok();
    bytes
}

/// The graph a line builds.
fn graph_of(line: &str) -> Graph<GraphNode> {
    let reg = default_registry();
    parse_launch(&reg, line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"))
}

/// Run a line with no file behind it, for a graph whose point is what it holds
/// rather than what it writes: the run is what proves the spliced elements
/// negotiate.
async fn run_line(line: &str) {
    run_graph(graph_of(line), &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
}

/// The one element in `graph` that the named factory built. An element a profile
/// spliced carries no `name=`, so it is found by the metadata that same factory
/// reports.
fn spliced<'g>(graph: &'g Graph<GraphNode>, factory: &str) -> &'g dyn DynAsyncElement {
    let reg = default_registry();
    let metadata = reg
        .make_element(factory)
        .unwrap_or_else(|| panic!("`{factory}` is a registered element"))
        .metadata();
    let found: Vec<&dyn DynAsyncElement> = (0..graph.node_count() as u32)
        .filter_map(|index| match graph.element(NodeId(index)) {
            Some(GraphNodeRef::Element(element)) if element.metadata() == metadata => {
                Some(&**element)
            }
            _ => None,
        })
        .collect();
    assert_eq!(found.len(), 1, "the graph carries one `{factory}`");
    found[0]
}

/// The parse error a line fails with.
fn parse_error(line: &str) -> String {
    let reg = default_registry();
    match parse_launch(&reg, line) {
        Ok(_) => panic!("`{line}` should not parse"),
        Err(e) => format!("{e}"),
    }
}

// ---------------------------------------------------------------------------
// the profile expands to the chain it names
// ---------------------------------------------------------------------------

/// An uncompressed stream profile stores what arrives: the container's muxer and
/// no encoder at all.
#[tokio::test]
async fn an_uncompressed_profile_is_the_muxer_alone() {
    const SOURCE: &str = "audiotestsrc num-buffers=5";
    let profile = run_to_bytes(
        &format!(
            "{SOURCE} ! encodebin profile=\"audio/x-wav:audio/x-raw\" ! filesink location={{}}"
        ),
        "wav-profile",
    )
    .await;
    let explicit = run_to_bytes(
        &format!("{SOURCE} ! wavenc ! filesink location={{}}"),
        "wav-explicit",
    )
    .await;
    assert_eq!(profile, explicit, "profile == `wavenc`");
    assert!(profile.starts_with(b"RIFF"), "a WAV file was written");
}

/// A container-less profile is the encoder alone, gst's one-stream form.
#[cfg(feature = "mjpeg-encode")]
#[tokio::test]
async fn a_container_less_profile_is_the_encoder_alone() {
    const SOURCE: &str = "videotestsrc num-buffers=3";
    let profile = run_to_bytes(
        &format!("{SOURCE} ! encodebin profile=\"image/jpeg\" ! filesink location={{}}"),
        "jpeg-profile",
    )
    .await;
    let explicit = run_to_bytes(
        &format!("{SOURCE} ! mjpegenc ! filesink location={{}}"),
        "jpeg-explicit",
    )
    .await;
    assert_eq!(profile, explicit, "profile == `mjpegenc`");
    assert!(profile.starts_with(&[0xFF, 0xD8]), "a JPEG was written");
}

/// Container plus stream: the encoder feeding the container's muxer.
#[cfg(feature = "mjpeg-encode")]
#[tokio::test]
async fn a_container_profile_is_the_encoder_and_the_muxer() {
    const SOURCE: &str = "videotestsrc num-buffers=3";
    let profile = run_to_bytes(
        &format!(
            "{SOURCE} ! encodebin profile=\"video/x-msvideo:image/jpeg\" ! filesink location={{}}"
        ),
        "avi-profile",
    )
    .await;
    let explicit = run_to_bytes(
        &format!("{SOURCE} ! mjpegenc ! avimux ! filesink location={{}}"),
        "avi-explicit",
    )
    .await;
    assert_eq!(profile, explicit, "profile == `mjpegenc ! avimux`");
    assert!(profile.starts_with(b"RIFF"), "an AVI file was written");
}

/// Two streams: the branch that reaches the bin by name gets the encoder for its
/// own kind of input, and both land in the one muxer.
#[cfg(feature = "mjpeg-encode")]
#[tokio::test]
async fn a_two_stream_profile_encodes_each_branch_for_its_own_input() {
    const VIDEO: &str = "videotestsrc num-buffers=3";
    const AUDIO: &str = "audiotestsrc num-buffers=10";
    let profile = run_to_bytes(
        &format!(
            "{VIDEO} ! encodebin profile=\"video/x-msvideo:image/jpeg:audio/x-raw\" name=e ! filesink location={{}}   {AUDIO} ! e."
        ),
        "av-profile",
    )
    .await;
    let explicit = run_to_bytes(
        &format!("{VIDEO} ! mjpegenc ! avimux name=m ! filesink location={{}}   {AUDIO} ! m."),
        "av-explicit",
    )
    .await;
    assert_eq!(
        profile, explicit,
        "profile == `mjpegenc ! avimux` with the audio branch unencoded"
    );
    let video_only = run_to_bytes(
        &format!(
            "{VIDEO} ! encodebin profile=\"video/x-msvideo:image/jpeg\" ! filesink location={{}}"
        ),
        "av-video-only",
    )
    .await;
    assert!(
        profile.len() > video_only.len(),
        "the audio stream is in the file too"
    );
}

/// `transcodebin` is `decodebin ! encodebin`, so it re-encodes whatever the file
/// holds.
#[cfg(all(feature = "mjpeg", feature = "mjpeg-encode"))]
#[tokio::test]
async fn transcodebin_decodes_then_encodes() {
    let source = format!(
        "filesrc location={}",
        fixture("multipart_64x48_jpeg.mjpg").display()
    );
    let transcoded = run_to_bytes(
        &format!("{source} ! transcodebin profile=\"image/jpeg\" ! filesink location={{}}"),
        "transcode",
    )
    .await;
    let explicit = run_to_bytes(
        &format!("{source} ! decodebin ! mjpegenc ! filesink location={{}}"),
        "transcode-explicit",
    )
    .await;
    assert_eq!(
        transcoded, explicit,
        "transcodebin == `decodebin ! encodebin`"
    );
    assert!(!transcoded.is_empty(), "the re-encode wrote frames");
}

/// The same transcode into a container: the decoded stream's size is known only
/// at runtime, which the muxer takes from its `CapsChanged` (M1089 fixed
/// `avimux` refusing it).
#[cfg(all(feature = "mjpeg", feature = "mjpeg-encode"))]
#[tokio::test]
async fn transcodebin_writes_a_container_from_a_runtime_geometry() {
    let source = format!(
        "filesrc location={}",
        fixture("multipart_64x48_jpeg.mjpg").display()
    );
    let avi = run_to_bytes(
        &format!(
            "{source} ! transcodebin profile=\"video/x-msvideo:image/jpeg\" ! filesink location={{}}"
        ),
        "transcode-avi",
    )
    .await;
    assert!(avi.starts_with(b"RIFF"), "an AVI file was written");
    // The fixture's own geometry, which only a runtime `CapsChanged` carried.
    assert!(
        avi.windows(4).any(|w| w == 64u32.to_le_bytes()),
        "the header carries the decoded width"
    );
}

/// M1091: `videotestsrc` produces RGBA and the H.264 encoder takes only planar
/// YUV, so the profile has to splice the converter itself. The proof is that the
/// H.264-in-MP4 profile runs at all, and that its output is what the chain with
/// the converter written out produces.
#[cfg(feature = "ffmpeg")]
#[tokio::test]
async fn a_profile_splices_the_converter_the_encoder_needs() {
    const SOURCE: &str = "videotestsrc num-buffers=3";
    let profile = run_to_bytes(
        &format!(
            "{SOURCE} ! encodebin profile=\"video/quicktime:video/x-h264\" ! filesink location={{}}"
        ),
        "h264-profile",
    )
    .await;
    let explicit = run_to_bytes(
        &format!("{SOURCE} ! videoconvert ! x264enc ! mp4mux ! filesink location={{}}"),
        "h264-explicit",
    )
    .await;
    assert_eq!(
        profile, explicit,
        "profile == `videoconvert ! x264enc ! mp4mux`"
    );
}

/// The audio side of the same thing: `audioconvert` and `audioresample` go in
/// ahead of an encoder that wants a sample format or rate the source does not
/// produce.
#[cfg(feature = "opus")]
#[tokio::test]
async fn an_audio_profile_splices_the_resampler_chain() {
    const SOURCE: &str = "audiotestsrc num-buffers=20";
    let profile = run_to_bytes(
        &format!(
            "{SOURCE} ! encodebin profile=\"video/x-matroska:audio/x-opus\" ! filesink location={{}}"
        ),
        "opus-profile",
    )
    .await;
    let explicit = run_to_bytes(
        &format!(
            "{SOURCE} ! audioconvert ! audioresample ! opusenc ! matroskamux ! filesink location={{}}"
        ),
        "opus-explicit",
    )
    .await;
    assert_eq!(
        profile, explicit,
        "profile == `audioconvert ! audioresample ! opusenc ! matroskamux`"
    );
}

// ---------------------------------------------------------------------------
// a stream part's pinned settings reach the elements that apply them (M1097)
// ---------------------------------------------------------------------------

/// A pinned geometry splices the scaler, even on an uncompressed stream, where
/// there is no encoder after it.
#[tokio::test]
async fn a_pinned_geometry_splices_the_scaler() {
    const WIDTH: u64 = 64;
    const HEIGHT: u64 = 48;
    let line = format!(
        "videotestsrc num-buffers=2 ! encodebin profile=\"video/x-raw,width={WIDTH},height={HEIGHT}\" ! fakesink"
    );
    let graph = graph_of(&line);
    let scaler = spliced(&graph, "videoscale");
    assert_eq!(scaler.get_property("width"), Some(PropValue::Uint(WIDTH)));
    assert_eq!(scaler.get_property("height"), Some(PropValue::Uint(HEIGHT)));
    run_line(&line).await;
}

/// The temporal half of the same thing.
#[tokio::test]
async fn a_pinned_framerate_splices_the_rate_converter() {
    const NUMERATOR: i32 = 15;
    const DENOMINATOR: i32 = 1;
    let line = format!(
        "videotestsrc num-buffers=2 ! encodebin profile=\"video/x-raw,framerate={NUMERATOR}/{DENOMINATOR}\" ! fakesink"
    );
    let graph = graph_of(&line);
    assert_eq!(
        spliced(&graph, "videorate").get_property("framerate"),
        Some(PropValue::Fraction(NUMERATOR, DENOMINATOR))
    );
    run_line(&line).await;
}

/// A pinned geometry ahead of a real encoder: the profile writes what the chain
/// with the scaler written out writes, at the pinned size rather than the
/// source's.
#[cfg(feature = "mjpeg-encode")]
#[tokio::test]
async fn a_pinned_geometry_reaches_the_encoder() {
    const SOURCE: &str = "videotestsrc num-buffers=3";
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 48;
    let profile = run_to_bytes(
        &format!(
            "{SOURCE} ! encodebin profile=\"video/x-msvideo:image/jpeg,width={WIDTH},height={HEIGHT}\" ! filesink location={{}}"
        ),
        "jpeg-scaled-profile",
    )
    .await;
    let explicit = run_to_bytes(
        &format!(
            "{SOURCE} ! videoscale width={WIDTH} height={HEIGHT} ! mjpegenc ! avimux ! filesink location={{}}"
        ),
        "jpeg-scaled-explicit",
    )
    .await;
    assert_eq!(
        profile, explicit,
        "profile == `videoscale ! mjpegenc ! avimux`"
    );
    assert!(
        profile.windows(4).any(|w| w == WIDTH.to_le_bytes()),
        "the AVI header carries the pinned width"
    );
    let unpinned = run_to_bytes(
        &format!(
            "{SOURCE} ! encodebin profile=\"video/x-msvideo:image/jpeg\" ! filesink location={{}}"
        ),
        "jpeg-unscaled-profile",
    )
    .await;
    assert!(
        profile.len() < unpinned.len(),
        "the pinned size is the smaller encode"
    );
}

/// A pinned framerate ahead of a real encoder: the rate converter drops the
/// frames the profile did not ask for, so fewer of them are encoded.
#[cfg(feature = "mjpeg-encode")]
#[tokio::test]
async fn a_pinned_framerate_reaches_the_encoder() {
    const SOURCE: &str = "videotestsrc num-buffers=6";
    const FRAMERATE: &str = "15/1";
    let profile = run_to_bytes(
        &format!(
            "{SOURCE} ! encodebin profile=\"video/x-msvideo:image/jpeg,framerate={FRAMERATE}\" ! filesink location={{}}"
        ),
        "jpeg-rated-profile",
    )
    .await;
    let explicit = run_to_bytes(
        &format!(
            "{SOURCE} ! videorate framerate={FRAMERATE} ! mjpegenc ! avimux ! filesink location={{}}"
        ),
        "jpeg-rated-explicit",
    )
    .await;
    assert_eq!(
        profile, explicit,
        "profile == `videorate ! mjpegenc ! avimux`"
    );
    let unpinned = run_to_bytes(
        &format!(
            "{SOURCE} ! encodebin profile=\"video/x-msvideo:image/jpeg\" ! filesink location={{}}"
        ),
        "jpeg-unrated-profile",
    )
    .await;
    assert!(
        profile.len() < unpinned.len(),
        "half the source rate is half the frames"
    );
}

/// The audio pins go on the `audioconvert` / `audioresample` pair the macro
/// already splices, and reach the file.
#[tokio::test]
async fn a_pinned_rate_and_channel_count_convert_ahead_of_the_encoder() {
    const SOURCE: &str = "audiotestsrc num-buffers=5";
    const RATE: u64 = 8_000;
    const CHANNELS: u64 = 1;
    const PINNED: &str = "audio/x-wav:audio/x-mulaw,rate=8000,channels=1";
    let graph = graph_of(&format!(
        "{SOURCE} ! encodebin profile=\"{PINNED}\" ! fakesink"
    ));
    assert_eq!(
        spliced(&graph, "audioconvert").get_property("channels"),
        Some(PropValue::Uint(CHANNELS))
    );
    assert_eq!(
        spliced(&graph, "audioresample").get_property("samplerate"),
        Some(PropValue::Uint(RATE))
    );
    let profile = run_to_bytes(
        &format!("{SOURCE} ! encodebin profile=\"{PINNED}\" ! filesink location={{}}"),
        "mulaw-pinned-profile",
    )
    .await;
    let explicit = run_to_bytes(
        &format!(
            "{SOURCE} ! audioconvert channels={CHANNELS} ! audioresample samplerate={RATE} ! mulawenc ! wavenc ! filesink location={{}}"
        ),
        "mulaw-pinned-explicit",
    )
    .await;
    assert_eq!(
        profile, explicit,
        "profile == `audioconvert ! audioresample ! mulawenc ! wavenc`"
    );
    let unpinned = run_to_bytes(
        &format!(
            "{SOURCE} ! encodebin profile=\"audio/x-wav:audio/x-mulaw\" ! filesink location={{}}"
        ),
        "mulaw-unpinned-profile",
    )
    .await;
    assert!(
        profile.len() < unpinned.len(),
        "one 8 kHz channel is less than two at 48 kHz"
    );
}

/// A pinned bitrate is the encoder's own `bitrate` property, in gst's unit.
#[cfg(feature = "opus")]
#[tokio::test]
async fn a_pinned_bitrate_reaches_the_encoder() {
    const SOURCE: &str = "audiotestsrc num-buffers=20";
    const BITRATE: u64 = 32_000;
    let profile = format!("video/x-matroska:audio/x-opus,bitrate={BITRATE}");
    let graph = graph_of(&format!(
        "{SOURCE} ! encodebin profile=\"{profile}\" ! fakesink"
    ));
    assert_eq!(
        spliced(&graph, "opusenc").get_property("bitrate"),
        Some(PropValue::Uint(BITRATE))
    );
    let written = run_to_bytes(
        &format!("{SOURCE} ! encodebin profile=\"{profile}\" ! filesink location={{}}"),
        "opus-bitrate-profile",
    )
    .await;
    let explicit = run_to_bytes(
        &format!(
            "{SOURCE} ! audioconvert ! audioresample ! opusenc bitrate={BITRATE} ! matroskamux ! filesink location={{}}"
        ),
        "opus-bitrate-explicit",
    )
    .await;
    assert_eq!(written, explicit, "profile == `opusenc bitrate=...`");
}

// ---------------------------------------------------------------------------
// a profile that cannot be honoured says why
// ---------------------------------------------------------------------------

#[test]
fn a_malformed_profile_is_refused_with_the_expected_form() {
    for (line, expected) in [
        (
            "audiotestsrc ! encodebin ! fakesink",
            "missing",
        ),
        (
            "audiotestsrc ! encodebin profile=\"nope/x-thing:audio/x-mulaw\" ! fakesink",
            "unknown media type `nope/x-thing`",
        ),
        (
            "audiotestsrc ! encodebin profile=\"audio/x-wav\" ! fakesink",
            "no stream",
        ),
        (
            "audiotestsrc ! encodebin profile=\"audio/x-wav:nope/x-thing\" ! fakesink",
            "unknown stream `nope/x-thing`",
        ),
        (
            "videotestsrc ! encodebin profile=\"video/x-matroska:video/x-h264,width=1280\" ! fakesink",
            "a scale needs both",
        ),
        (
            "videotestsrc ! encodebin profile=\"video/x-matroska:video/x-h264,profile=high\" ! fakesink",
            "pins `profile`, which a profile cannot apply",
        ),
        (
            "videotestsrc ! encodebin profile=\"video/x-matroska:video/x-h264,width=abc,height=720\" ! fakesink",
            "bad `width=abc`",
        ),
        // Nothing encodes an uncompressed stream, so there is nothing to set a
        // rate on.
        (
            "audiotestsrc ! encodebin profile=\"audio/x-wav:audio/x-raw,bitrate=64000\" ! fakesink",
            "no encoder to set it on",
        ),
        // G.711 is a fixed 8 bits per sample, so its encoder has no such knob.
        (
            "audiotestsrc ! encodebin profile=\"audio/x-wav:audio/x-mulaw,bitrate=64000\" ! fakesink",
            "which `mulawenc` has no property for",
        ),
        (
            "videotestsrc ! encodebin profile=\"video/x-matroska:image/jpeg\" bitrate=100 ! fakesink",
            "unknown property `bitrate`",
        ),
    ] {
        let err = parse_error(line);
        assert!(err.contains(expected), "`{line}`\n  expected {expected:?}, got {err:?}");
        assert!(
            err.contains("container-caps:stream-caps"),
            "the error states the form: {err:?}"
        );
    }
}

#[test]
fn a_profile_with_no_writer_or_encoder_here_says_which() {
    // IVF has no muxer at all, in any build.
    let err =
        parse_error("videotestsrc ! encodebin profile=\"video/x-ivf:video/x-vp9\" ! fakesink");
    assert!(err.contains("no muxer for video/x-ivf"), "{err:?}");
    // H.265 encoding needs hardware (NVENC / VideoToolbox / MediaCodec), so a
    // plain software build has no encoder for it.
    #[cfg(not(any(feature = "nvenc", feature = "vtencode", feature = "mediacodec")))]
    {
        let err = parse_error(
            "videotestsrc ! encodebin profile=\"video/x-matroska:video/x-h265\" ! fakesink",
        );
        assert!(err.contains("no encoder for video/x-h265"), "{err:?}");
    }
}

#[test]
fn a_profile_that_cannot_encode_the_branchs_input_says_so() {
    let err = parse_error(
        "audiotestsrc ! encodebin profile=\"video/x-msvideo:image/jpeg:video/x-h264\" ! fakesink",
    );
    assert!(
        err.contains("no stream that encodes audio/x-raw"),
        "the audio branch has no audio stream to use: {err:?}"
    );
}

#[test]
fn the_bin_names_answer_the_gst_compat_query() {
    use g2g_plugins::gst_compat::{gst_equivalent, GstEquivalent};
    let reg = default_registry();
    for name in ["encodebin", "encodebin2", "transcodebin"] {
        assert_eq!(
            gst_equivalent(&reg, name),
            GstEquivalent::Available,
            "{name} is a launch keyword here"
        );
    }
}
