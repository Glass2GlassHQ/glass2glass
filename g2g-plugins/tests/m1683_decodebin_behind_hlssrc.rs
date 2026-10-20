// `hlssrc location=X ! decodebin name=d  d. ! ...  d. ! ...` decodes every muxed stream of the
// playlist, as gst's `souphttpsrc ! hlsdemux ! decodebin name=d` does.
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
fn decodebin_fans_out_the_streams_of_an_hls_playlist() {
    let line = format!(
        "hlssrc location={} ! decodebin name=d \
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

#[test]
fn a_source_with_no_container_to_probe_is_named() {
    let upstream = "videotestsrc";
    let line = format!("{upstream} ! decodebin name=d  d. ! fakesink  d. ! fakesink");
    let error = parse_launch(&default_registry(), &line).expect_err("nothing to fan out");
    assert!(
        matches!(&error, ParseError::DecodebinFanoutUpstream(name) if name == upstream),
        "{error}"
    );
}
