//! M1168: a lone `fallbacksrc` carries audio and video at once. When the whole
//! pipeline is a single `fallbacksrc uri=X`, the URI is probed by the same
//! container hooks a lone `playbin` uses, and every stream kind the probe reports
//! gets its own decode chain, its own `fallbackswitch`, and its own automatic
//! sink off one byte source and one demuxer. An inline `fallbacksrc`, and a URI
//! no hook claims, stay single-stream.
//!
//! The structural checks build a registry of stub decoders and counting sinks, so
//! they need no decoder feature; the decode-and-run check does too, and the frames
//! it counts are the container's real access units.
//!
//! `default_registry` is `std`-gated, so this file is too: run with
//! `cargo test -p g2g-plugins --features std`.
#![cfg(feature = "std")]

use std::path::PathBuf;
use std::sync::atomic::Ordering;

use g2g_core::runtime::{
    run_graph, Registry, FALLBACK_AUDIO_SUFFIX, FALLBACK_FALLBACK_SOURCE_SUFFIX,
    FALLBACK_MAIN_SOURCE_SUFFIX, FALLBACK_SWITCH_NAME, FALLBACK_VIDEO_SUFFIX,
};
use g2g_core::{AudioFormat, ByteStreamEncoding, NodeKind, VideoCodec};
use g2g_plugins::registry::default_registry;

mod fallback_fanout_common;
use fallback_fanout_common::{
    byte_stream, categories, compressed_audio, compressed_video, kinds, node_names, parsed,
    short_type_name, ZeroClock, AUDIO_FRAMES, RUN_BOUND, VIDEO_FRAMES,
};

/// The A/V transport stream every fan-out check probes: one H.264 stream and one
/// AAC stream in one container.
const AV_FIXTURE: &str = "tests/fixtures/av_h264_aac44100.ts";

/// The `fallbacksrc name=` the named checks use, and the stem the unnamed ones
/// fall back to (the expansion numbers its generated switches from zero).
const NAMED: &str = "fb";
fn unnamed_stem() -> String {
    format!("{FALLBACK_SWITCH_NAME}-0")
}

fn av_uri() -> String {
    format!("file://{}/{AV_FIXTURE}", env!("CARGO_MANIFEST_DIR"))
}

fn registry() -> Registry {
    fallback_fanout_common::registry(fallback_fanout_common::StubContainer {
        fanout: g2g_plugins::uridecodebin::ts_uri_fanout,
        bytes: byte_stream(ByteStreamEncoding::MpegTs),
        video: compressed_video(VideoCodec::H264),
        audio: compressed_audio(AudioFormat::Aac),
    })
}

/// A PNM still no container fan-out hook claims, so a lone `fallbacksrc` over it
/// stays single-stream.
async fn still(tag: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("g2g_m1168_{}_{tag}", std::process::id()));
    let reg = default_registry();
    let line = format!(
        "videotestsrc num-buffers=1 pattern=smpte ! pnmenc ! filesink location={}",
        path.display()
    );
    let graph = parsed(&reg, &line);
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("{line}: {e:?}"));
    path
}

#[test]
fn a_lone_fallbacksrc_gives_each_kind_a_switch_and_a_sink() {
    let reg = registry();
    let line = format!("fallbacksrc name={NAMED} uri={}", av_uri());
    let graph = parsed(&reg, &line);
    let names = node_names(&graph);
    let categories = categories(&graph);
    let kinds = kinds(graph);

    // One switch and one sink per kind, off one byte source and one demuxer.
    assert_eq!(
        kinds.iter().filter(|k| **k == NodeKind::Muxer(2)).count(),
        2,
        "one 2-input fallbackswitch per kind: {kinds:?}"
    );
    assert_eq!(
        kinds.iter().filter(|k| **k == NodeKind::Sink).count(),
        2,
        "one automatic sink per kind: {kinds:?}"
    );
    assert_eq!(
        kinds
            .iter()
            .filter(|k| matches!(k, NodeKind::Tee(_)))
            .count(),
        1,
        "one demuxer feeding both kinds: {kinds:?}"
    );
    // The main byte source plus one dummy generator per kind.
    assert_eq!(
        kinds.iter().filter(|k| **k == NodeKind::Source).count(),
        3,
        "the byte source and both dummy generators: {kinds:?}"
    );
    for expected in [
        format!("{NAMED}{FALLBACK_VIDEO_SUFFIX}"),
        format!("{NAMED}{FALLBACK_AUDIO_SUFFIX}"),
    ] {
        assert!(
            names.contains(&expected),
            "{expected} missing from {names:?}"
        );
    }
    // Only the audio branch carries the converters that fix the sink's PCM shape.
    for tail in [
        short_type_name::<g2g_plugins::audioconvert::AudioConvert>(),
        short_type_name::<g2g_plugins::audioresample::AudioResample>(),
    ] {
        assert_eq!(
            categories.iter().filter(|c| **c == tail).count(),
            1,
            "{tail} sits between the audio switch and its sink: {categories:?}"
        );
    }
    // Each kind's dummy fallback is its own generator.
    for dummy in [
        short_type_name::<g2g_plugins::videotestsrc::VideoTestSrc>(),
        short_type_name::<g2g_plugins::audiotestsrc::AudioTestSrc>(),
    ] {
        assert!(
            categories.contains(&dummy),
            "{dummy} missing from {categories:?}"
        );
    }
}

#[test]
fn the_fanned_out_nodes_take_their_generated_names() {
    let reg = registry();
    let line = format!(
        "fallbacksrc name={NAMED} uri={uri} fallback-uri={uri}",
        uri = av_uri()
    );
    let names = node_names(&parsed(&reg, &line));
    for expected in [
        format!("{NAMED}{FALLBACK_MAIN_SOURCE_SUFFIX}"),
        format!("{NAMED}{FALLBACK_FALLBACK_SOURCE_SUFFIX}"),
        format!("{NAMED}{FALLBACK_VIDEO_SUFFIX}"),
        format!("{NAMED}{FALLBACK_AUDIO_SUFFIX}"),
    ] {
        assert!(
            names.contains(&expected),
            "{expected} missing from {names:?}"
        );
    }

    // Unnamed, every generated name hangs off the generated stem instead.
    let line = format!("fallbacksrc uri={}", av_uri());
    let names = node_names(&parsed(&reg, &line));
    let stem = unnamed_stem();
    for expected in [
        format!("{stem}{FALLBACK_MAIN_SOURCE_SUFFIX}"),
        format!("{stem}{FALLBACK_VIDEO_SUFFIX}"),
        format!("{stem}{FALLBACK_AUDIO_SUFFIX}"),
    ] {
        assert!(
            names.contains(&expected),
            "{expected} missing from {names:?}"
        );
    }
}

/// Every name the fan-out generates goes through the same duplicate check the
/// parser applies to a line's own `name=`, so two nodes can never share one.
#[test]
fn the_generated_names_are_all_distinct() {
    let reg = registry();
    let line = format!(
        "fallbacksrc name={NAMED} uri={uri} fallback-uri={uri}",
        uri = av_uri()
    );
    let mut names = node_names(&parsed(&reg, &line));
    let count = names.len();
    names.sort();
    names.dedup();
    assert_eq!(names.len(), count, "{names:?}");
}

/// The fallback URI's own fan-out feeds each switch's second input, so a kind's
/// fallback is that kind's stream rather than the dummy generator.
#[test]
fn a_fallback_uri_that_fans_out_replaces_both_dummies() {
    let reg = registry();
    let line = format!(
        "fallbacksrc name={NAMED} uri={uri} fallback-uri={uri}",
        uri = av_uri()
    );
    let graph = parsed(&reg, &line);
    let categories = categories(&graph);
    for dummy in [
        short_type_name::<g2g_plugins::videotestsrc::VideoTestSrc>(),
        short_type_name::<g2g_plugins::audiotestsrc::AudioTestSrc>(),
    ] {
        assert!(
            !categories.contains(&dummy),
            "{dummy} should not be built when the fallback URI carries that kind: {categories:?}"
        );
    }
    assert_eq!(
        kinds(graph)
            .iter()
            .filter(|k| matches!(k, NodeKind::Tee(_)))
            .count(),
        2,
        "the main and the fallback container each get a demuxer"
    );
}

/// A URI no fan-out hook claims still builds a runnable single-stream pipeline:
/// the expansion appends the kind's automatic sink so the lone keyword is a
/// complete line.
#[tokio::test]
async fn an_unclaimed_uri_stays_single_stream() {
    let path = still("unclaimed.pnm").await;
    let reg = default_registry();
    let line = format!("fallbacksrc uri=file://{}", path.display());
    let kinds = kinds(parsed(&reg, &line));
    std::fs::remove_file(&path).ok();

    assert_eq!(
        kinds.iter().filter(|k| **k == NodeKind::Muxer(2)).count(),
        1,
        "one switch, not one per kind: {kinds:?}"
    );
    assert_eq!(
        kinds.iter().filter(|k| **k == NodeKind::Sink).count(),
        1,
        "one automatic sink: {kinds:?}"
    );
}

/// An inline `fallbacksrc` heads a chain the line wrote itself, which has no way
/// to name a second output, so it keeps its single-stream shape even over a
/// container the fan-out hooks do claim.
#[test]
fn an_inline_fallbacksrc_stays_single_stream() {
    let reg = registry();
    let line = format!("fallbacksrc uri={} ! autovideosink", av_uri());
    let kinds = kinds(parsed(&reg, &line));
    assert_eq!(
        kinds.iter().filter(|k| **k == NodeKind::Muxer(2)).count(),
        1,
        "one switch: {kinds:?}"
    );
    assert_eq!(
        kinds.iter().filter(|k| **k == NodeKind::Sink).count(),
        1,
        "the line's own sink, and only it: {kinds:?}"
    );
}

/// The fanned-out graph runs: frames off the container's video stream and its
/// audio stream both reach their own sink within one bounded run. The fallback
/// URI is the same container, so no dummy generator is built and every frame
/// counted came out of a decode chain.
#[tokio::test]
async fn both_kinds_deliver_frames_to_their_own_sink() {
    VIDEO_FRAMES.store(0, Ordering::SeqCst);
    AUDIO_FRAMES.store(0, Ordering::SeqCst);
    let reg = registry();
    let line = format!("fallbacksrc uri={uri} fallback-uri={uri}", uri = av_uri());
    let graph = parsed(&reg, &line);
    tokio::time::timeout(RUN_BOUND, run_graph(graph, &ZeroClock, 4))
        .await
        .unwrap_or_else(|_| panic!("{line}: the run did not finish inside {RUN_BOUND:?}"))
        .unwrap_or_else(|e| panic!("{line}: {e:?}"));

    assert!(
        VIDEO_FRAMES.load(Ordering::SeqCst) > 0,
        "the video branch delivered nothing"
    );
    assert!(
        AUDIO_FRAMES.load(Ordering::SeqCst) > 0,
        "the audio branch delivered nothing"
    );
}
