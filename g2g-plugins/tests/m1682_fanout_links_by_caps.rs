// A bare `d.` / `s.` branch of a fan-out takes the first free stream its first element
// accepts, as gst-launch links a delayed pad by caps, so an audio branch written before the
// video branch still gets the audio.
#![cfg(all(feature = "std", feature = "ffmpeg", target_os = "linux"))]

use std::time::Duration;

use g2g_core::runtime::{parse_launch, ParseError};
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
const SINKS: [&str; 2] = ["audio", "video"];
// each branch is built for one kind, so a stream routed to the wrong branch fails negotiation
const DECODED_BRANCHES: &str = "d. ! audioconvert ! checksumsink name=audio \
                                d. ! videoconvert ! checksumsink name=video";
const DEADLINE: Duration = Duration::from_secs(20);
// the fixture is a fraction of a second long, so a file run checks that each branch decodes at all
const FILE_FRAMES: usize = 1;
// a live run checks that both tracks keep flowing, not just a first frame each
#[cfg(feature = "rtsp")]
const LIVE_FRAMES: usize = 10;

fn assert_both_branches_decode(line: &str, enough: usize) {
    let graph =
        parse_launch(&default_registry(), line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let counts = count_checksums(graph, &SINKS, enough, DEADLINE);
    assert!(
        counts.iter().all(|&count| count >= enough),
        "decoded frames per branch {SINKS:?} of `{line}`: {counts:?}"
    );
}

#[test]
fn uridecodebin_gives_the_audio_branch_written_first_the_audio() {
    assert_both_branches_decode(
        &format!("uridecodebin uri=file://{AV_FIXTURE} name=d {DECODED_BRANCHES}"),
        FILE_FRAMES,
    );
}

#[test]
fn decodebin_gives_the_audio_branch_written_first_the_audio() {
    assert_both_branches_decode(
        &format!("filesrc location={AV_FIXTURE} ! decodebin name=d {DECODED_BRANCHES}"),
        FILE_FRAMES,
    );
}

#[test]
fn a_branch_with_no_stream_of_its_kind_left_is_refused() {
    let line = format!(
        "uridecodebin uri=file://{AV_FIXTURE} name=d \
         d. ! audioconvert ! fakesink  d. ! audioconvert ! fakesink"
    );
    let error = parse_launch(&default_registry(), &line).expect_err("one audio stream");
    assert!(
        matches!(
            &error,
            ParseError::NoFanoutStream { consumer, .. } if consumer == "audioconvert"
        ),
        "{error}"
    );
}

#[cfg(feature = "rtsp")]
mod live {
    use super::*;

    #[test]
    fn rtspsrc_gives_the_audio_branch_written_first_the_audio() {
        if !ffmpeg_tool::have_ffmpeg() {
            eprintln!("ffmpeg not present; skipping the RTSP pads by caps");
            return;
        }
        let relay = rtsp_av_relay::serve_av_fixture(AV_FIXTURE);
        assert_both_branches_decode(
            &format!(
                "rtspsrc name=s location={} \
                 s. ! aacparse ! avdec_aac ! checksumsink name=audio \
                 s. ! h264parse ! avdec_h264 ! checksumsink name=video",
                relay.url
            ),
            LIVE_FRAMES,
        );
    }

    // the lone branch takes no video, so it reads the audio pad and the video pad is dropped
    #[test]
    fn rtspsrc_plays_an_audio_branch_alone() {
        if !ffmpeg_tool::have_ffmpeg() {
            eprintln!("ffmpeg not present; skipping the RTSP audio branch alone");
            return;
        }
        let relay = rtsp_av_relay::serve_av_fixture(AV_FIXTURE);
        let line = format!(
            "rtspsrc name=s location={} s. ! aacparse ! avdec_aac ! checksumsink name=audio",
            relay.url
        );
        let graph = parse_launch(&default_registry(), &line)
            .unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
        let counts = count_checksums(graph, &SINKS[..1], LIVE_FRAMES, DEADLINE);
        assert!(counts[0] >= LIVE_FRAMES, "`{line}` decoded {counts:?}");
    }

    // the hook DESCRIBEs the stream at parse time
    #[test]
    fn an_rtsp_uri_gives_the_audio_branch_written_first_the_audio() {
        if !ffmpeg_tool::have_ffmpeg() {
            eprintln!("ffmpeg not present; skipping the RTSP fan-out by caps");
            return;
        }
        let relay = rtsp_av_relay::serve_av_fixture(AV_FIXTURE);
        assert_both_branches_decode(
            &format!("uridecodebin uri={} name=d {DECODED_BRANCHES}", relay.url),
            LIVE_FRAMES,
        );
    }
}
