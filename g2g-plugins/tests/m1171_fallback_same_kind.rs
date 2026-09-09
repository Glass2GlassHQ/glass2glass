//! M1171: a fanned-out `fallbacksrc` carries several streams of *one* kind. Until
//! now a repeated kind made the fan-out decline, because two ports of a kind
//! would need two switches named alike; each port of a kind now takes its
//! ordinal, so a grouped Ogg file's two audio bitstreams get two decode chains,
//! two `fallbackswitch`es and two audio sinks. The first port of a kind keeps the
//! bare suffix, so a single-track container's names are the ones M1168 generated.
//!
//! Ogg is the container that needs this: every Ogg mapping g2g reads is audio, so
//! an Ogg fan-out is all one kind by construction.
//!
//! The stubs stand in for the Vorbis decoder (see `fallback_fanout_common`), and
//! they take their caps from the probe rather than restating the fixture's
//! format, so the fixture and the test cannot drift.
//!
//! Run with `cargo test -p g2g-plugins --features std`.
#![cfg(feature = "std")]

use std::sync::atomic::Ordering;

use g2g_core::runtime::{
    run_graph, Registry, FALLBACK_AUDIO_SUFFIX, FALLBACK_MAIN_SOURCE_SUFFIX, FALLBACK_VIDEO_SUFFIX,
};
use g2g_core::stream::StreamType;
use g2g_core::{AudioFormat, ByteStreamEncoding, Caps, NodeKind, VideoCodec};
use g2g_plugins::uridecodebin::ogg_uri_fanout;

mod fallback_fanout_common;
use fallback_fanout_common::{
    byte_stream, compressed_video, kinds, node_names, parsed, StubContainer, ZeroClock,
    AUDIO_FRAMES, RUN_BOUND,
};

/// A grouped Ogg file: two concurrent Vorbis logical bitstreams, same format, so
/// one stub decoder serves both ports.
const OGG_FIXTURE: &str = "tests/fixtures/two_vorbis_48000.ogg";

/// A single-stream Ogg file, which reports one port and so does not fan out.
const ONE_STREAM_FIXTURE: &str = "tests/fixtures/vorbis_stereo_48k.ogg";

const NAMED: &str = "fb";

fn uri(fixture: &str) -> String {
    format!("file://{}/{fixture}", env!("CARGO_MANIFEST_DIR"))
}

/// The probe's ports for the grouped fixture, which is where the stub decoder's
/// input caps come from too.
fn probed_ports() -> Vec<g2g_core::runtime::UriFanoutPort> {
    ogg_uri_fanout(&Registry::new(), &uri(OGG_FIXTURE))
        .expect("probing a readable Ogg file is not an error")
        .expect("a grouped Ogg file fans out")
        .ports
}

fn registry() -> Registry {
    fallback_fanout_common::registry(StubContainer {
        fanout: ogg_uri_fanout,
        bytes: byte_stream(ByteStreamEncoding::Ogg),
        // No video port in an Ogg file; the video stub goes unused.
        video: compressed_video(VideoCodec::H264),
        audio: probed_ports()[0].caps.clone(),
    })
}

/// Both logical bitstreams are audio, and both become ports: the repeated kind no
/// longer makes the probe's answer unusable.
#[test]
fn a_grouped_ogg_file_reports_two_audio_ports() {
    let ports = probed_ports();
    assert_eq!(
        ports.iter().map(|p| p.stream_type).collect::<Vec<_>>(),
        vec![StreamType::Audio, StreamType::Audio]
    );
    for port in &ports {
        assert!(
            matches!(
                port.caps,
                Caps::Audio {
                    format: AudioFormat::Vorbis,
                    ..
                }
            ),
            "{:?}",
            port.caps
        );
    }
}

/// Two ports of one kind get two switches and two sinks off one byte source and
/// one demuxer, and no video branch is built for a container with no video.
#[test]
fn each_audio_port_gets_its_own_switch_and_sink() {
    let reg = registry();
    let line = format!("fallbacksrc name={NAMED} uri={}", uri(OGG_FIXTURE));
    let graph = parsed(&reg, &line);
    let names = node_names(&graph);
    let kinds = kinds(graph);

    assert_eq!(
        kinds.iter().filter(|k| **k == NodeKind::Muxer(2)).count(),
        2,
        "one fallbackswitch per audio port: {kinds:?}"
    );
    assert_eq!(
        kinds.iter().filter(|k| **k == NodeKind::Sink).count(),
        2,
        "one audio sink per port: {kinds:?}"
    );
    assert_eq!(
        kinds
            .iter()
            .filter(|k| matches!(k, NodeKind::Tee(_)))
            .count(),
        1,
        "one demuxer feeding both ports: {kinds:?}"
    );
    // The byte source plus one dummy generator per port.
    assert_eq!(
        kinds.iter().filter(|k| **k == NodeKind::Source).count(),
        3,
        "the byte source and both dummy generators: {kinds:?}"
    );
    assert!(
        !names.iter().any(|n| n.ends_with(FALLBACK_VIDEO_SUFFIX)),
        "no video branch for an all-audio container: {names:?}"
    );
}

/// The first port of a kind keeps the bare suffix and later ones take their
/// ordinal, so the names are distinct and a one-track container's names are
/// unchanged.
#[test]
fn the_second_port_of_a_kind_takes_its_ordinal() {
    let reg = registry();
    let line = format!("fallbacksrc name={NAMED} uri={}", uri(OGG_FIXTURE));
    let mut names = node_names(&parsed(&reg, &line));
    for expected in [
        format!("{NAMED}{FALLBACK_MAIN_SOURCE_SUFFIX}"),
        format!("{NAMED}{FALLBACK_AUDIO_SUFFIX}"),
        format!("{NAMED}{FALLBACK_AUDIO_SUFFIX}-1"),
    ] {
        assert!(
            names.contains(&expected),
            "{expected} missing from {names:?}"
        );
    }
    let count = names.len();
    names.sort();
    names.dedup();
    assert_eq!(names.len(), count, "every generated name is distinct");
}

/// One port is still not a fan-out: a single-stream Ogg file falls through to the
/// single-stream expansion plus its one automatic sink.
#[test]
fn a_single_stream_ogg_file_does_not_fan_out() {
    let reg = registry();
    let line = format!("fallbacksrc uri={}", uri(ONE_STREAM_FIXTURE));
    let kinds = kinds(parsed(&reg, &line));
    assert_eq!(
        kinds.iter().filter(|k| **k == NodeKind::Muxer(2)).count(),
        1,
        "one switch, not one per port: {kinds:?}"
    );
    assert_eq!(
        kinds.iter().filter(|k| **k == NodeKind::Sink).count(),
        1,
        "one automatic sink: {kinds:?}"
    );
}

/// The fanned-out grouped file runs: packets off both bitstreams reach a sink
/// inside one bounded run. The fallback URI is the same file, so no dummy
/// generator is built and every frame counted came out of a decode chain.
#[tokio::test]
async fn both_bitstreams_deliver_frames() {
    AUDIO_FRAMES.store(0, Ordering::SeqCst);
    let reg = registry();
    let line = format!(
        "fallbacksrc uri={uri} fallback-uri={uri}",
        uri = uri(OGG_FIXTURE)
    );
    let graph = parsed(&reg, &line);
    tokio::time::timeout(RUN_BOUND, run_graph(graph, &ZeroClock, 4))
        .await
        .unwrap_or_else(|_| panic!("{line}: the run did not finish inside {RUN_BOUND:?}"))
        .unwrap_or_else(|e| panic!("{line}: {e:?}"));

    assert!(
        AUDIO_FRAMES.load(Ordering::SeqCst) > 0,
        "neither audio branch delivered anything"
    );
}
