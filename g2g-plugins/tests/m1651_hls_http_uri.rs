// An http(s) URL naming an `.m3u8` playlist is an HLS URI, as gst's `uridecodebin` plays it,
// so `uridecodebin uri=http://host/master.m3u8 name=d` fans out the variant's muxed streams.
#![cfg(all(feature = "hls", feature = "ffmpeg", target_os = "linux"))]

use std::time::Duration;

use g2g_core::runtime::{parse_launch, ParseError};
use g2g_core::stream::StreamType;
use g2g_plugins::registry::default_registry;

#[path = "common/checksum_counts.rs"]
mod checksum_counts;
use checksum_counts::count_checksums;
#[path = "common/hls_vod_playlist.rs"]
mod hls_vod_playlist;
use hls_vod_playlist::{serve_vod_playlist, serve_vod_playlist_of};
#[path = "common/multitrack_fixture.rs"]
mod multitrack_fixture;
use multitrack_fixture::{assert_decoded_stream, probed_of_kind, run_recording, RECORDING_SINK};

const SINKS: [&str; 2] = ["video", "audio"];
// ffmpeg's hls muxer over the multi-track TS fixture's first video and both audio tracks, stream copied
const TWO_AAC_SEGMENT: &str = "multitrack_h264_aac_hls_segment.ts";
const TWO_AAC_SOURCE_CONTAINER: &str = "ts";
// Constrained Baseline level 1 video and AAC LC audio
const TWO_AAC_VARIANT_CODECS: &str = "avc1.42c00a,mp4a.40.2";
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

#[tokio::test]
async fn each_aac_track_of_a_muxed_variant_reaches_its_own_port() {
    let probed = probed_of_kind(TWO_AAC_SOURCE_CONTAINER, StreamType::Audio);
    assert!(
        probed.len() > 1,
        "the source fixture has several AAC tracks"
    );
    let labels: Vec<String> = (0..probed.len())
        .map(|index| format!("hls-audio-{index}"))
        .collect();
    let branches: String = labels
        .iter()
        .enumerate()
        .map(|(index, label)| format!("  d.audio_{index} ! {RECORDING_SINK} label={label}"))
        .collect();
    let line = format!(
        "uridecodebin uri={} name=d{branches}",
        serve_vod_playlist_of(TWO_AAC_SEGMENT, TWO_AAC_VARIANT_CODECS)
    );
    let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
    let decoded = run_recording(&line, &label_refs).await;
    for (index, (caps, stream)) in decoded.iter().zip(&probed).enumerate() {
        assert_decoded_stream(caps, stream, &format!("`{line}` audio port {index}"));
    }
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
