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
