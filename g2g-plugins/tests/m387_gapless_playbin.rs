//! M387 - gapless playbin convenience builder. `gapless_playbin` assembles a
//! runnable `GaplessSrc -> decode -> sink` graph from a playlist of URIs and
//! returns the shared `GaplessController`, so an app gets gapless playback without
//! hand-wiring the source, the decode chain, and the controller (the M383-M386
//! pieces). The first URI's source plays immediately; the rest are pre-enqueued.

#![cfg(feature = "std")]

use g2g_core::runtime::{
    is_raw_video, ElementFactory, Registry, Uri, UriError, UriSourceFactory,
};
use g2g_core::{
    Caps, CapsSet, Dim, PadTemplate, RawVideoFormat, Rate, VideoCodec,
};

use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::gaplesssrc::{gapless_playbin, GaplessPlaybinError};

fn h264_any() -> Caps {
    Caps::CompressedVideo { codec: VideoCodec::H264, width: Dim::Any, height: Dim::Any, framerate: Rate::Any }
}
fn raw_video() -> Caps {
    Caps::RawVideo { format: RawVideoFormat::Nv12, width: Dim::Any, height: Dim::Any, framerate: Rate::Any }
}

/// A `mem://` URI source stand-in (its identity is irrelevant to graph assembly,
/// which never runs it): a placeholder source declaring H.264 output, so the
/// per-playlist-item `build_uri_source` succeeds and the decode chain plugs.
fn mem_uri_build(
    _uri: &Uri,
) -> Result<(Box<dyn g2g_core::runtime::DynSourceLoop>, Caps), UriError> {
    Ok((Box::new(g2g_plugins::videotestsrc::VideoTestSrc::new(8, 8, 30, 1)), h264_any()))
}

/// A registry with the `mem://` handler and a stub H.264 decoder (so the reused
/// decode chain reaches raw), mirroring m379's stubs.
fn registry_with_stubs() -> Registry {
    let mut reg = Registry::new();
    reg.register_uri(UriSourceFactory::new("mem", mem_uri_build));
    reg.register(ElementFactory::new(
        "h264stub",
        Vec::from([
            PadTemplate::sink(CapsSet::one(h264_any())),
            PadTemplate::source(CapsSet::one(raw_video())),
        ]),
        |_| Box::new(g2g_plugins::identity::IdentityTransform::new()),
    ));
    reg
}

#[test]
fn gapless_playbin_builds_and_preloads_the_playlist() {
    let reg = registry_with_stubs();
    let uris = ["mem://clip1", "mem://clip2", "mem://clip3"];

    let (graph, ctl) = gapless_playbin(&reg, &uris, FakeSink::new(), &is_raw_video, 6)
        .expect("gapless playbin builds");

    // GaplessSrc -> stub decoder -> sink (the decode chain is plugged once and
    // reused across items): 3 nodes, 2 edges.
    assert_eq!(graph.node_count(), 3, "gapless source, one decoder, sink");
    assert_eq!(graph.edges().len(), 2, "source->decode->sink");

    // The first URI plays from the GaplessSrc; the other two are pre-enqueued on
    // the returned controller, ready to play back-to-back.
    assert_eq!(ctl.queued(), 2, "the two successor clips are enqueued");

    // The app drives the playlist through the returned controller (enqueue more,
    // switch_now, finish). Finishing here ends it after the last clip.
    ctl.finish();
    assert!(ctl.is_finished());
}

#[test]
fn empty_playlist_is_rejected() {
    let reg = registry_with_stubs();
    let err = gapless_playbin(&reg, &[], FakeSink::new(), &is_raw_video, 6).unwrap_err();
    assert!(matches!(err, GaplessPlaybinError::EmptyPlaylist), "got {err:?}");
}

#[test]
fn unknown_scheme_is_rejected() {
    let reg = registry_with_stubs();
    let err = gapless_playbin(&reg, &["bogus://x"], FakeSink::new(), &is_raw_video, 6).unwrap_err();
    assert!(
        matches!(err, GaplessPlaybinError::Uri(UriError::UnknownScheme)),
        "got {err:?}"
    );
}

#[test]
fn a_later_playlist_item_with_a_bad_scheme_is_rejected() {
    // The first URI is fine but a later one is not: the eager per-item source
    // build surfaces the error rather than failing mid-playback.
    let reg = registry_with_stubs();
    let err = gapless_playbin(&reg, &["mem://ok", "bogus://x"], FakeSink::new(), &is_raw_video, 6)
        .unwrap_err();
    assert!(matches!(err, GaplessPlaybinError::Uri(UriError::UnknownScheme)), "got {err:?}");
}
