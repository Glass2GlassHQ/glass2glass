// An http(s) URL naming an `.m3u8` playlist is an HLS URI, as gst's `uridecodebin` plays it,
// so `uridecodebin uri=http://host/master.m3u8 name=d` fans out the variant's muxed streams.
#![cfg(all(feature = "hls", feature = "ffmpeg", target_os = "linux"))]

use std::time::Duration;

use g2g_core::runtime::{parse_launch, ParseError};
use g2g_plugins::registry::default_registry;

#[path = "common/checksum_counts.rs"]
mod checksum_counts;
use checksum_counts::count_checksums;
#[path = "common/hls_vod_playlist.rs"]
mod hls_vod_playlist;
use hls_vod_playlist::serve_vod_playlist;

const SINKS: [&str; 2] = ["video", "audio"];
// the fixture is a fraction of a second long, so the run checks that each branch decodes at all
const FIRST_FRAME: usize = 1;
const DEADLINE: Duration = Duration::from_secs(20);

#[test]
fn an_http_playlist_uri_fans_out_to_video_and_audio() {
    let line = format!(
        "uridecodebin uri={} name=d \
         d. ! videoconvert ! checksumsink name=video \
         d. ! audioconvert ! checksumsink name=audio",
        serve_vod_playlist()
    );
    let graph =
        parse_launch(&default_registry(), &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let counts = count_checksums(graph, &SINKS, FIRST_FRAME, DEADLINE);
    assert!(
        counts.iter().all(|&count| count >= FIRST_FRAME),
        "decoded frames per branch {SINKS:?}: {counts:?}"
    );
}

// canary, MCP and g2g-launch parse lines from inside a tokio runtime
#[test]
fn an_http_playlist_uri_parses_inside_a_runtime() {
    let line = format!(
        "uridecodebin uri={} name=d d. ! fakesink name={} d. ! fakesink name={}",
        serve_vod_playlist(),
        SINKS[0],
        SINKS[1]
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build a runtime");
    let graph = runtime
        .block_on(async { parse_launch(&default_registry(), &line) })
        .unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    for sink in SINKS {
        assert!(
            graph.node_by_name(sink).is_some(),
            "no `{sink}` branch in `{line}`"
        );
    }
}

#[test]
fn an_http_url_that_names_no_playlist_is_refused() {
    let line = "uridecodebin uri=http://127.0.0.1/clip.mp4 ! fakesink";
    let error = parse_launch(&default_registry(), line).expect_err("not a playlist");
    assert!(
        matches!(&error, ParseError::Uri(message) if message.contains("UnknownContent")),
        "{error}"
    );
}
