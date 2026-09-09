//! M1169: a lone `fallbacksrc uri=file://X.mpg` fans an MPEG program stream out,
//! the program stream sibling of M1168's MPEG-TS fan-out. The same probe
//! `playbin` uses reports the disc's MPEG-2 video and MP2 audio, and each gets
//! its own decode chain, `fallbackswitch`, and automatic sink off one byte source
//! and one `PsDemuxN`.
//!
//! The stubs stand in for the MPEG-2 and MP2 decoders (see
//! `fallback_fanout_common`), so nothing here needs a decoder feature; the frames
//! the run counts are the fixture's real access units.
//!
//! The HLS half of M1169 (`hls_ts_uri_fanout` / `hls_fmp4_uri_fanout`) is in
//! `m395_playbin_hls`, next to the `playbin` assembly it mirrors.
//!
//! Run with `cargo test -p g2g-plugins --features std`.
#![cfg(feature = "std")]

use std::sync::atomic::Ordering;

use g2g_core::runtime::{
    run_graph, Registry, FALLBACK_AUDIO_SUFFIX, FALLBACK_MAIN_SOURCE_SUFFIX, FALLBACK_VIDEO_SUFFIX,
};
use g2g_core::{AudioFormat, ByteStreamEncoding, Caps, NodeKind, StreamType, VideoCodec};

mod fallback_fanout_common;
use fallback_fanout_common::{
    byte_stream, categories, compressed_audio, compressed_video, kinds, node_names, parsed,
    short_type_name, StubContainer, ZeroClock, AUDIO_FRAMES, RUN_BOUND, VIDEO_FRAMES,
};

/// The A/V program stream every check probes: one MPEG-2 video stream and one
/// MPEG audio (Layer II) stream in one container, ffmpeg's own `vob` muxer.
const AV_FIXTURE: &str = "tests/fixtures/av_mpeg2_mp2.mpg";

const NAMED: &str = "fb";

fn av_uri() -> String {
    format!("file://{}/{AV_FIXTURE}", env!("CARGO_MANIFEST_DIR"))
}

fn registry() -> Registry {
    fallback_fanout_common::registry(StubContainer {
        fanout: g2g_plugins::uridecodebin::ps_uri_fanout,
        bytes: byte_stream(ByteStreamEncoding::MpegPs),
        video: compressed_video(VideoCodec::Mpeg2),
        audio: compressed_audio(AudioFormat::Mp2),
    })
}

#[test]
fn a_program_stream_gives_each_kind_a_switch_and_a_sink() {
    let reg = registry();
    let line = format!("fallbacksrc name={NAMED} uri={}", av_uri());
    let graph = parsed(&reg, &line);
    let names = node_names(&graph);
    let categories = categories(&graph);
    let kinds = kinds(graph);

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
        "one PsDemuxN feeding both kinds: {kinds:?}"
    );
    assert_eq!(
        kinds.iter().filter(|k| **k == NodeKind::Source).count(),
        3,
        "the byte source and both dummy generators: {kinds:?}"
    );
    for expected in [
        format!("{NAMED}{FALLBACK_MAIN_SOURCE_SUFFIX}"),
        format!("{NAMED}{FALLBACK_VIDEO_SUFFIX}"),
        format!("{NAMED}{FALLBACK_AUDIO_SUFFIX}"),
    ] {
        assert!(
            names.contains(&expected),
            "{expected} missing from {names:?}"
        );
    }
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

/// The probe's own answer, before any graph is built: one port per forwardable
/// stream, each tagged with the kind the caller picks its switch and sink by, and
/// carrying the elementary caps that port's decode chain is plugged from. A
/// subpicture track would not appear here (it is a bitmap cue with no switch of
/// its own), which is why the fixture's two A/V streams are the whole list.
#[test]
fn the_probe_reports_a_port_per_forwardable_stream() {
    let fanout = g2g_plugins::uridecodebin::ps_uri_fanout(&Registry::new(), &av_uri())
        .expect("probing a readable program stream is not an error")
        .expect("an A/V program stream fans out");
    assert_eq!(
        fanout
            .ports
            .iter()
            .map(|p| p.stream_type)
            .collect::<Vec<_>>(),
        vec![StreamType::Video, StreamType::Audio],
        "one port per forwardable stream, in demux port order"
    );
    // What the fixture actually carries, so a probe that reported the wrong
    // elementary stream would plug the wrong decoder here.
    assert!(
        matches!(
            fanout.ports[0].caps,
            Caps::CompressedVideo {
                codec: VideoCodec::Mpeg2,
                ..
            }
        ),
        "{:?}",
        fanout.ports[0].caps
    );
    assert!(
        matches!(
            fanout.ports[1].caps,
            Caps::Audio {
                format: AudioFormat::Mp2,
                ..
            }
        ),
        "{:?}",
        fanout.ports[1].caps
    );

    // A file no program stream packet turned up in: decline, so the single-stream
    // expansion builds the line instead.
    let not_ps = format!("file://{}/Cargo.toml", env!("CARGO_MANIFEST_DIR"));
    assert!(
        g2g_plugins::uridecodebin::ps_uri_fanout(&Registry::new(), &not_ps)
            .expect("declining is not an error")
            .is_none()
    );
}

/// The fanned-out program stream runs: access units off the disc's MPEG-2 stream
/// and its MP2 stream both reach their own sink inside one bounded run. The
/// fallback URI is the same file, so no dummy generator is built and every frame
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
