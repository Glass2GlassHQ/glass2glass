//! M192: gst-canonical-name aliases. Pasted `gst-launch` lines that use
//! GStreamer's element names (`autovideosink`, `avdec_h264`, ...) resolve to the
//! g2g equivalents. Auto sinks fall back through the available display / audio
//! sinks to `fakesink`, so a tutorial line runs headless here.
#![cfg(feature = "std")]

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[tokio::test]
async fn autovideosink_resolves_and_runs() {
    // No wayland/kms feature in the default test build, so autovideosink falls
    // back to fakesink and the pipeline runs to completion.
    let reg = default_registry();
    let line = "videotestsrc num-buffers=3 ! videoconvert ! autovideosink";
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("{line:?} parse: {e:?}"));
    let consumed = run_graph(graph, &ZeroClock, 4).await.expect("runs").frames_consumed;
    assert_eq!(consumed, 3, "{line}");
}

#[tokio::test]
async fn autoaudiosink_resolves_and_runs() {
    let reg = default_registry();
    let line = "audiotestsrc num-buffers=3 ! audioconvert ! autoaudiosink";
    // audiotestsrc may not be registered in this build; only assert the alias
    // resolves + runs when the source exists, otherwise the parse error is about
    // the source, not the alias.
    match parse_launch(&reg, line) {
        Ok(graph) => {
            let consumed =
                run_graph(graph, &ZeroClock, 4).await.expect("runs").frames_consumed;
            assert_eq!(consumed, 3, "{line}");
        }
        Err(e) => {
            // The failure must be the missing source, never an unknown
            // autoaudiosink (the alias must resolve to fakesink).
            let msg = format!("{e}");
            assert!(!msg.contains("autoaudiosink"), "autoaudiosink must resolve: {msg}");
        }
    }
}

#[tokio::test]
async fn aliases_do_not_shadow_canonical_names() {
    // The g2g-native names still work alongside the aliases.
    let reg = default_registry();
    let line = "videotestsrc num-buffers=2 ! videoconvert ! fakesink";
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("{line:?} parse: {e:?}"));
    assert_eq!(run_graph(graph, &ZeroClock, 4).await.expect("runs").frames_consumed, 2);
}

#[tokio::test]
async fn desktop_video_sink_names_alias_to_a_sink() {
    // xvimagesink / glimagesink / ximagesink all map onto the available display
    // sink (here fakesink), so legacy desktop lines parse and run.
    let reg = default_registry();
    for sink in ["xvimagesink", "ximagesink", "glimagesink"] {
        let line = format!("videotestsrc num-buffers=1 ! videoconvert ! {sink}");
        let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("{line:?} parse: {e:?}"));
        assert_eq!(run_graph(graph, &ZeroClock, 4).await.expect("runs").frames_consumed, 1, "{line}");
    }
}

/// When the `ffmpeg` feature is on, `avdec_h264` (gst's libav decoder name)
/// resolves to the g2g `ffmpegdec`. Gated so the alias target actually exists.
#[cfg(feature = "ffmpeg")]
#[tokio::test]
async fn avdec_h264_alias_resolves_to_ffmpegdec() {
    let reg = default_registry();
    // We only assert the element name resolves (constructs), not a full decode
    // run (that needs a real H.264 source). A bare make via the parser inside a
    // minimal chain is enough: the alias must not be an unknown element.
    let line = "videotestsrc num-buffers=1 ! avdec_h264 ! fakesink";
    // The decoder will reject raw video at negotiation, but parsing must succeed
    // (the alias resolved); a parse error naming avdec_h264 would be the bug.
    if let Err(e) = parse_launch(&reg, line) {
        let msg = format!("{e}");
        assert!(!msg.contains("avdec_h264"), "avdec_h264 must resolve to ffmpegdec: {msg}");
    }
}

/// The native WebRTC WHIP sink is launch-parsable under `webrtcsink`, with the
/// `location` property targeting the endpoint. We only assert the name resolves
/// (the live publish path needs a WHIP server + network); a parse error naming
/// `webrtcsink` would be the regression we guard against.
#[cfg(feature = "webrtc")]
#[test]
fn webrtcsink_resolves_from_launch() {
    let reg = default_registry();
    let line = "videotestsrc num-buffers=1 ! webrtcsink location=http://localhost:8889/s/whip";
    if let Err(e) = parse_launch(&reg, line) {
        let msg = format!("{e}");
        assert!(!msg.contains("webrtcsink"), "webrtcsink must resolve as a launch element: {msg}");
    }
}

/// The native WebRTC WHEP ingest source is registered as `webrtcsrc` and takes
/// its endpoint via the `location` property (the build-by-name + set path the
/// parser drives). Live subscribe needs a WHEP server + network.
#[cfg(feature = "webrtc")]
#[test]
fn webrtcsrc_resolves_and_takes_location() {
    use g2g_core::PropValue;
    let reg = default_registry();
    let mut src = reg.make_source("webrtcsrc").expect("webrtcsrc builds by name");
    src.set_property("location", PropValue::Str("http://localhost:8889/s/whep".into())).unwrap();
    assert_eq!(
        src.get_property("location"),
        Some(PropValue::Str("http://localhost:8889/s/whep".into()))
    );
}

/// M237: the ffmpeg VAAPI hwaccel backend is launch-parsable under its own name
/// and the gst VA-API names resolve to it (preferred over the cros-codecs
/// `vaapidec`, which is blocked on Mesa radeonsi). Like the avdec_h264 test we
/// only assert the names resolve (construct) — a real decode needs an H.264
/// source and a libva render node (see `ffmpeg_smoke::*_vaapi`).
#[cfg(feature = "ffmpeg")]
#[tokio::test]
async fn vaapi_hwaccel_names_resolve_to_ffmpegvaapidec() {
    let reg = default_registry();
    for name in ["ffmpegvaapidec", "vaapih264dec", "vah264dec"] {
        let line = format!("videotestsrc num-buffers=1 ! {name} ! fakesink");
        // The decoder rejects raw video at negotiation; parsing must still
        // succeed (the name resolved). A parse error naming the element is the
        // bug we're guarding against.
        if let Err(e) = parse_launch(&reg, &line) {
            let msg = format!("{e}");
            assert!(!msg.contains(name), "{name} must resolve to ffmpegvaapidec: {msg}");
        }
    }
}

/// M237: the `device` property pins the VAAPI render node from a `gst-launch`
/// line (`ffmpegvaapidec device=/dev/dri/renderD128`). Round-trips through the
/// registry's build-by-name + set/get, the path the parser drives.
#[cfg(feature = "ffmpeg")]
#[test]
fn ffmpegvaapidec_device_property_round_trips() {
    use g2g_core::PropValue;
    let reg = default_registry();
    let mut dec = reg.make_element("ffmpegvaapidec").expect("ffmpegvaapidec builds by name");
    // Unset reads back empty (libva default), then a pin round-trips.
    assert_eq!(dec.get_property("device"), Some(PropValue::Str(String::new())));
    dec.set_property("device", PropValue::Str("/dev/dri/renderD128".into())).unwrap();
    assert_eq!(dec.get_property("device"), Some(PropValue::Str("/dev/dri/renderD128".into())));
}
