//! M982: the decoder's decode-loop caps must not replace the framerate the
//! parser recovered from the SPS with its own fallback default. The runner
//! steers the mid-stream refinement, so the decoder learns the real input caps
//! from `configure_pipeline`, not from the `CapsChanged` it is handed.
//!
//! Run with `cargo test -p g2g-plugins --features ffmpeg,tooling-json --test
//! m982_decoder_framerate_refinement`.
#![cfg(all(feature = "ffmpeg", feature = "tooling-json"))]

use g2g_plugins::registry::default_registry;
use g2g_plugins::toolingjson::observed_graph_json;

/// A 640x480 Annex-B H.264 clip whose SPS VUI declares 10 fps.
const CLIP: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/h264_640x480.h264"
);

#[tokio::test]
async fn decoder_output_carries_the_streams_framerate_not_the_fallback() {
    let reg = default_registry();
    let line = format!("filesrc location={CLIP} ! h264parse ! ffmpegdec ! fakesink");
    let ran = observed_graph_json(&reg, &line).await;
    assert_eq!(ran["ok"], true, "{ran}");

    let edges = ran["edges"].as_array().unwrap();
    let parser_out = edges[1]["caps"].as_str().unwrap();
    assert!(
        parser_out.contains("framerate=10/1"),
        "the parser recovers the clip's rate from the SPS, got {parser_out}"
    );

    // The decoder emits its own caps from the decoded pictures *after* the
    // refinement crossed, so the last caps on its output link are the ones that
    // must still carry 10/1 rather than the decoder's 30/1 default.
    assert_eq!(edges[2]["caps_source"], "runtime");
    let decoder_out = edges[2]["caps"].as_str().unwrap();
    assert!(
        decoder_out.contains("framerate=10/1"),
        "the decoder must not override the refined rate, got {decoder_out}"
    );
}
