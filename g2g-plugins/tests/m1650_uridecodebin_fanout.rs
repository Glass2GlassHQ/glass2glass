// `uridecodebin uri=X name=d  d. ! ...  d. ! ...` decodes every stream the URI carries, one
// per reference, as gst's does. A bare `d.` reads the next port, whatever kind it is.
#![cfg(all(feature = "std", feature = "ffmpeg", target_os = "linux"))]

use std::time::Duration;

use g2g_core::runtime::parse_launch;
use g2g_plugins::registry::default_registry;

#[path = "common/checksum_counts.rs"]
mod checksum_counts;
use checksum_counts::count_checksums;
#[cfg(feature = "rtsp")]
#[path = "common/ffmpeg_tool.rs"]
mod ffmpeg_tool;
#[cfg(feature = "rtsp")]
#[path = "common/rtsp_av_relay.rs"]
mod rtsp_av_relay;

const AV_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/av_h264_aac.mp4"
);
const SINKS: [&str; 2] = ["video", "audio"];
// each branch is built for one kind, so a port routed to the wrong branch fails negotiation
const BRANCHES: &str = "d. ! videoconvert ! checksumsink name=video \
                        d. ! audioconvert ! checksumsink name=audio";
const DEADLINE: Duration = Duration::from_secs(20);
// the fixture is a fraction of a second long, so a file run checks that each branch decodes at all
const FILE_FRAMES: usize = 1;
// a live run checks that both tracks keep flowing, not just a first frame each
#[cfg(feature = "rtsp")]
const LIVE_FRAMES: usize = 10;

fn assert_both_branches_decode(head: &str, enough: usize) {
    let line = format!("{head} name=d {BRANCHES}");
    let graph =
        parse_launch(&default_registry(), &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let counts = count_checksums(graph, &SINKS, enough, DEADLINE);
    for (sink, count) in SINKS.iter().zip(&counts) {
        assert!(
            *count >= enough,
            "the {sink} branch of `{line}` got {count} decoded frames"
        );
    }
}

#[test]
fn a_file_uri_fans_out_to_bare_references() {
    assert_both_branches_decode(
        &format!("uridecodebin uri=file://{AV_FIXTURE}"),
        FILE_FRAMES,
    );
}

#[test]
fn decodebin_decodes_an_audio_port_behind_a_bare_reference() {
    assert_both_branches_decode(
        &format!("filesrc location={AV_FIXTURE} ! decodebin"),
        FILE_FRAMES,
    );
}

#[test]
fn named_references_pick_their_port_in_any_order() {
    let line = format!(
        "uridecodebin uri=file://{AV_FIXTURE} name=d \
         d.audio_0 ! audioconvert ! checksumsink name=audio \
         d.video_0 ! videoconvert ! checksumsink name=video"
    );
    let graph =
        parse_launch(&default_registry(), &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let counts = count_checksums(graph, &SINKS, FILE_FRAMES, DEADLINE);
    assert!(
        counts.iter().all(|&n| n >= FILE_FRAMES),
        "counts {counts:?}"
    );
}

#[test]
fn a_pad_the_uri_does_not_carry_is_refused() {
    let line = format!(
        "uridecodebin uri=file://{AV_FIXTURE} name=d \
         d.video_0 ! fakesink  d.video_1 ! fakesink"
    );
    let error = parse_launch(&default_registry(), &line).expect_err("one video stream");
    assert!(
        matches!(
            error,
            g2g_core::runtime::ParseError::UnknownOutputPad { ref pad, .. } if pad == "video_1"
        ),
        "{error}"
    );
}

// the hook DESCRIBEs the stream at parse time
#[cfg(feature = "rtsp")]
#[test]
fn an_rtsp_uri_fans_out_to_video_and_audio() {
    if !ffmpeg_tool::have_ffmpeg() {
        eprintln!("ffmpeg not present; skipping the RTSP A/V fan-out");
        return;
    }
    let relay = rtsp_av_relay::serve_av_fixture(AV_FIXTURE);
    assert_both_branches_decode(&format!("uridecodebin uri={}", relay.url), LIVE_FRAMES);
}
